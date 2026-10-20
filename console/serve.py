"""The console: one page on this laptop for both robots -- one at a time, or side by side,
with a STOP for each and STOP ALL always on screen.

    python console/serve.py            # the real robots, as console/robots.json lists them
    python console/serve.py --fake     # a pretend X3 (x3plus/tools/x3bridge.py --sim) and M3
                                       # (m3pro/arena/fake_m3.py) started alongside, in the
                                       # project venv: .venv/Scripts/python console/serve.py --fake
    -> http://localhost:8099/

The page (index.html, console.js, console.css) talks to each robot's own web API. JSON goes
through here as /robot/<id>/<path> -- the X3's panel sends no CORS headers, and one origin keeps
the page simple; camera frames and maps come straight from the robots, which images may. The
3D view gets the M3's exported scans from this laptop: /clouds.json lists the PLY files in
m3pro/out/arena/rtab (where `arena_map.py rtab-export` puts them), each with its floor print if
there is one, and the Gaussian splats there (.splat, train/splat/train_splat.py's, made on hekate),
and /clouds/<name> is one of those files (.ply, .splat or .png). Voice commands go to
/api/voice: a WAV in, what was said and the command in it out (voice.py; the speech model loads
the first time the page asks, GET /api/voice). An object scan's views are fused here, on this
laptop: POST /api/scan {"robot": id} fetches the robot's newest scan (GET /pick/scan.zip) into
m3pro/out/scan/ and runs m3pro/tools/m3_scan.py fuse on it, which puts object_STAMP.ply among
the 3D scans; GET /api/scan says how far it is. POST /api/splat {"name": THING or none} makes the
newest scan there a Gaussian splat (tools/splat_cycle.py: hekate fits it, ~2 min) among the 3D scans,
splat_[THING_]STAMP.splat; GET /api/splat says how far it is. The
console listens on localhost only: it is this laptop's (and a microphone needs localhost).
Standard library only (and sherpa-onnx for voice, when it is there; Open3D for the fuse, in
the fuse's own process).
"""

import argparse
import io
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import llm
import voice

HERE =pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PORT = 8099
CLOUDS = ROOT / "m3pro" / "out" / "arena" / "rtab"   # the 3D scans, from arena_map.py rtab-export
SCANS = ROOT / "m3pro" / "out" / "scan"              # object scans' views, as m3_scan.py fetch puts them
FAKES = [   # --fake starts these; robots.fake.json points at them
    [sys.executable, str(ROOT / "x3plus" / "tools" / "x3bridge.py"), "--sim", "--port", "8180", "--out", ""],
    [sys.executable, str(ROOT / "m3pro" / "arena" / "fake_m3.py"), "--port", "8195"],
]
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".json": "application/json"}
CLOUD_TYPES = {".ply": "application/octet-stream", ".splat": "application/octet-stream",
               ".png": "image/png"}   # scans, Gaussian splats, floor prints
GET_TIMEOUT, POST_TIMEOUT = 3, 30   # s; a POST may move an arm before it answers
FETCH_TIMEOUT, FUSE_TIMEOUT = 120, 900   # s: a scan's ~110 files (~15 MB); its fuse (~20 s for 36 views)
SPLAT_TIMEOUT = 1800   # s: a splat takes ~2 min; gsplat's first build on hekate ~10


def fuse_scan(stamp: str) -> str:
    """m3pro/tools/m3_scan.py fuse on out/scan/STAMP, in its own process (Open3D): what it said."""
    done = subprocess.run([sys.executable, str(ROOT / "m3pro" / "tools" / "m3_scan.py"), "fuse", stamp],
                          cwd=ROOT / "m3pro", capture_output=True, text=True, timeout=FUSE_TIMEOUT, check=False)
    said = [line for line in done.stdout.splitlines() if line.strip()]
    if done.returncode != 0:
        raise RuntimeError((done.stderr.strip().splitlines() or ["the fuse failed"])[-1])
    return "; ".join(said[:2])


class Scans:
    """One object scan at a time from a robot to the 3D view: its views fetched, then fused."""

    def __init__(self, folder=SCANS, fuse=fuse_scan):
        self.folder, self.fuse = folder, fuse
        self.lock = threading.Lock()
        self.state = {"state": "idle", "stamp": None, "cloud": None, "message": ""}

    def status(self) -> dict:
        with self.lock:
            return dict(self.state)

    def start(self, url: str) -> dict:
        with self.lock:
            if self.state["state"] in ("fetching", "fusing"):
                return {"error": f"a scan is {self.state['state']} already"}
            self.state = {"state": "fetching", "stamp": None, "cloud": None, "message": "fetching the views"}
        threading.Thread(target=self.run, args=(url,), daemon=True).start()
        return self.status()

    def set(self, **changes):
        with self.lock:
            self.state.update(changes)

    def run(self, url: str):
        try:
            with urllib.request.urlopen(f"{url}/pick/scan.zip", timeout=FETCH_TIMEOUT) as reply:
                data = reply.read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                names = z.namelist()
                stamp = names[0].split("/")[0] if names else ""
                if not stamp or any(pathlib.PurePosixPath(n).parts[:1] != (stamp,) or ".." in n for n in names):
                    raise RuntimeError("the robot's zip is not one scan folder")
                z.extractall(self.folder)
            self.set(state="fusing", stamp=stamp, message=f"fusing {len(names)} files")
            said = self.fuse(stamp)
            self.set(state="done", cloud=f"object_{stamp}.ply", message=said)
        except urllib.error.HTTPError as e:   # the robot's own reason: no scan yet, say
            try:
                reason = json.loads(e.read()).get("error")
            except ValueError:
                reason = None
            self.set(state="failed", message=reason or f"HTTP {e.code}")
        except Exception as e:   # noqa: BLE001 -- whatever it was, the page says it
            self.set(state="failed", message=str(e) or type(e).__name__)


def splat_cycle(args: list[str]) -> list[str]:
    """tools/splat_cycle.py with `args`, as a command (this venv's Python: the fuse needs Open3D)."""
    return [sys.executable, "-u", str(ROOT / "tools" / "splat_cycle.py"), *args]


class Splats:
    """One Gaussian splat at a time, of the newest scan on this laptop, into the 3D scans: hekate fits
    it (tools/splat_cycle.py); its lines as they come say how far it is."""

    def __init__(self, folder=SCANS, command=splat_cycle):
        self.folder, self.command = folder, command
        self.lock = threading.Lock()
        self.state = {"state": "idle", "stamp": None, "splat": None, "message": ""}

    def status(self) -> dict:
        with self.lock:
            return dict(self.state)

    def set(self, **changes):
        with self.lock:
            self.state.update(changes)

    def start(self, name: str | None) -> dict:
        """The newest scan's splat, named after the thing (`name`, made file-safe) if given."""
        stamps = sorted(f.parent.name for f in self.folder.glob("*/scan.json")) if self.folder.is_dir() else []
        if not stamps:
            return {"error": "no scan on this laptop yet: Scan, or Fuse last scan"}
        name = re.sub(r"[^A-Za-z0-9_-]+", "-", name or "").strip("-")
        with self.lock:
            if self.state["state"] == "running":
                return {"error": f"a splat of {self.state['stamp']} is running already"}
            self.state = {"state": "running", "stamp": stamps[-1], "splat": None, "message": "starting"}
        threading.Thread(target=self.run, args=(stamps[-1], name), daemon=True).start()
        return self.status()

    def run(self, stamp: str, name: str):
        try:
            proc = subprocess.Popen(self.command([stamp, *(["--name", name] if name else [])]), cwd=ROOT,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                                    env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"})
            watchdog = threading.Timer(SPLAT_TIMEOUT, proc.kill)
            watchdog.start()
            last = scored = ""
            for line in proc.stdout:
                if (line := line.strip()) and not line.startswith("** "):   # not ssh's warnings
                    last = line
                    scored = line if line.startswith("held-out views") else scored
                    self.set(message=line[:240])
            proc.wait()
            watchdog.cancel()
            if proc.returncode != 0:
                raise RuntimeError(last or f"splat_cycle.py ended with {proc.returncode}")
            self.set(state="done", splat=f"splat_{name + '_' if name else ''}{stamp}.splat",
                     message=(scored or last).split(" -> ")[0])   # not where it lies on hekate
        except Exception as e:   # noqa: BLE001 -- whatever it was, the page says it
            self.set(state="failed", message=str(e) or type(e).__name__)


def make_handler(robots, clouds=CLOUDS, recognizer=None, scans=None, splats=None, brain=None):
    urls = {r["id"]: r["url"].rstrip("/") for r in robots}
    recognizer = recognizer or voice.Recognizer()   # loads nothing until the page asks
    scans = scans or Scans()
    splats = splats or Splats()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"   # keep-alive: the page polls several times a second

        def log_message(self, *args):
            pass

        def send(self, code, body, kind):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def error(self, code, message):
            self.send(code, json.dumps({"error": message}).encode(), TYPES[".json"])

        def do_GET(self):
            path = self.path.split("?")[0]
            if path.startswith("/robot/"):
                return self.proxy("GET")
            if path == "/robots.json":
                return self.send(200, json.dumps(robots).encode(), TYPES[".json"])
            if path == "/clouds.json":
                return self.send(200, json.dumps(list_clouds(clouds)).encode(), TYPES[".json"])
            if path == "/api/scan":   # the object scan's fetch and fuse: how far
                return self.send(200, json.dumps(scans.status()).encode(), TYPES[".json"])
            if path == "/api/splat":   # the newest scan's Gaussian splat, on hekate: how far
                return self.send(200, json.dumps(splats.status()).encode(), TYPES[".json"])
            if path == "/api/voice":   # ready, loading, or why not; starts the model's load
                status = recognizer.status() | {"llm": brain.status() if brain else None}
                return self.send(200, json.dumps(status).encode(), TYPES[".json"])
            if path.startswith("/clouds/"):
                return self.file(clouds, urllib.parse.unquote(path[len("/clouds/"):]), CLOUD_TYPES)
            self.file(HERE, "index.html" if path == "/" else path.lstrip("/"), TYPES)

        def file(self, folder, name, types):   # from that folder only, and only of those types
            file = (folder / name).resolve()
            if file.parent != folder.resolve() or file.suffix not in types or not file.is_file():
                return self.error(404, "not found")
            self.send(200, file.read_bytes(), types[file.suffix])

        def do_POST(self):
            if self.path.startswith("/robot/"):
                return self.proxy("POST")
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))   # drained, too
            if self.path == "/api/voice":   # a WAV (or typed text) -> what was said, the command
                code, reply = voice.answer(recognizer, body, self.headers.get("Content-Type", ""), brain)
                return self.send(code, json.dumps(reply).encode(), TYPES[".json"])
            if self.path == "/api/scan":   # {"robot": id}: fetch its newest scan and fuse it
                try:
                    robot = json.loads(body or b"{}").get("robot")
                except (ValueError, AttributeError):
                    robot = None
                if robot not in urls:
                    return self.error(404, f"no robot {robot!r}")
                reply = scans.start(urls[robot])
                return self.send(409 if "error" in reply else 200, json.dumps(reply).encode(), TYPES[".json"])
            if self.path == "/api/splat":   # {"name": THING or none}: the newest scan's splat
                try:
                    name = json.loads(body or b"{}").get("name")
                except (ValueError, AttributeError):
                    name = None
                reply = splats.start(name if isinstance(name, str) else None)
                return self.send(409 if "error" in reply else 200, json.dumps(reply).encode(), TYPES[".json"])
            self.error(404, "not found")

        def proxy(self, method):
            robot, _slash, tail = self.path[len("/robot/"):].partition("/")
            if robot not in urls:
                return self.error(404, f"no robot {robot!r}")
            body = None
            if method == "POST":
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"
            request = urllib.request.Request(f"{urls[robot]}/{tail}", data=body, method=method,
                                             headers={"Content-Type": "application/json"})
            timeout = POST_TIMEOUT if method == "POST" else GET_TIMEOUT
            try:
                with urllib.request.urlopen(request, timeout=timeout) as reply:
                    self.send(reply.status, reply.read(),
                              reply.headers.get("Content-Type", TYPES[".json"]))
            except urllib.error.HTTPError as e:   # the robot answered, with an error
                self.send(e.code, e.read(), e.headers.get("Content-Type", TYPES[".json"]))
            except (urllib.error.URLError, OSError) as e:
                self.error(502, f"{robot} not reachable: {getattr(e, 'reason', e)}")

    return Handler


def list_clouds(folder):
    """The scans in the folder, newest first: the point clouds (PLY) and the Gaussian splats (.splat,
    train/splat/train_splat.py's): name, kind ("cloud" or "splat"), size, when each was written, and
    the floor print exported with it (NAME_arena.ply -> NAME_floor.png), or None."""
    files = sorted([*folder.glob("*.ply"), *folder.glob("*.splat")], key=lambda f: f.stat().st_mtime, reverse=True)
    prints = [f.with_name(f.name.removesuffix("_arena.ply").removesuffix(".ply") + "_floor.png")
              for f in files]
    return [{"name": f.name, "kind": "splat" if f.suffix == ".splat" else "cloud", "bytes": f.stat().st_size,
             "written": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime)),
             "floor": floor.name if floor.is_file() else None}
            for f, floor in zip(files, prints)]


class Server6(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fake", action="store_true", help="pretend robots, no hardware")
    parser.add_argument("--port", type=int, default=PORT, help=f"(default {PORT}) another, for a second console beside one open")
    parser.add_argument("--llm", default=llm.DEFAULT, choices=[*llm.MODELS, "off"],
                        help=f"the language model voice commands go to (default {llm.DEFAULT}; off: the grammar only)")
    args = parser.parse_args()
    robots = json.loads((HERE / ("robots.fake.json" if args.fake else "robots.json")).read_text())
    # not waited for: the page asks each robot every 0.5 s until it answers
    procs = [subprocess.Popen(cmd, cwd=ROOT) for cmd in FAKES] if args.fake else []
    brain = None if args.llm == "off" else llm.Brain(args.llm, names=taught(robots))
    if brain is not None:
        brain.start()   # loads in seconds, on the GPU; without its files the grammar does it all
        brain.warm()    # its first answer is slow: taken now, not on the first command
        why = brain.missing()
        print(f"voice commands: {args.llm}" if not why else f"voice commands: the grammar ({why[0]})", flush=True)
    handler = make_handler(robots, brain=brain)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    try:   # "localhost" is tried as ::1 first on Windows; without this each connection waits
        server6 = Server6(("::1", args.port), handler)
        threading.Thread(target=server6.serve_forever, daemon=True).start()
    except OSError:
        pass   # no IPv6 loopback here
    print(f"console: http://localhost:{args.port}/  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for proc in procs:
            proc.terminate()
        if brain is not None:
            brain.stop()


def taught(robots, keep_s=20):
    """What the M3 was taught to pick by name (its bridge's GET /pick/taught), for the language
    model: asked at most every `keep_s`, an empty list while it does not answer."""
    url = next((r["url"].rstrip("/") for r in robots if r["kind"] == "m3arm"), None)
    kept = {"t": 0.0, "names": []}

    def names():
        if url and time.time() - kept["t"] > keep_s:
            kept["t"] = time.time()
            try:
                with urllib.request.urlopen(f"{url}/pick/taught", timeout=1) as reply:
                    kept["names"] = json.loads(reply.read()).get("names", [])
            except (OSError, ValueError):
                pass   # offline: the names known last
        return kept["names"]
    return names


if __name__ == "__main__":
    main()
