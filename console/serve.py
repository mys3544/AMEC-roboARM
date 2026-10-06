"""The console: one page on this laptop for both robots -- one at a time, or side by side,
with a STOP for each and STOP ALL always on screen.

    python console/serve.py            # the real robots, as console/robots.json lists them
    python console/serve.py --fake     # a pretend X3 (x3plus/tools/webapp.py --sim) and M3
                                       # (m3pro/arena/fake_m3.py) started alongside, in the
                                       # project venv: .venv/Scripts/python console/serve.py --fake
    -> http://localhost:8099/

The page (index.html, console.js, console.css) talks to each robot's own web API. JSON goes
through here as /robot/<id>/<path> -- the X3's panel sends no CORS headers, and one origin keeps
the page simple; camera frames and maps come straight from the robots, which images may. The
3D view gets the M3's exported scans from this laptop: /clouds.json lists the PLY files in
m3pro/out/arena/rtab (where `arena_map.py rtab-export` puts them), each with its floor print if
there is one, and /clouds/<name> is one of those files (.ply or .png). Voice commands go to
/api/voice: a WAV in, what was said and the command in it out (voice.py; the speech model loads
the first time the page asks, GET /api/voice). The
console listens on localhost only: it is this laptop's (and a microphone needs localhost).
Standard library only (and sherpa-onnx for voice, when it is there).
"""

import argparse
import json
import pathlib
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import voice

HERE =pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PORT = 8099
CLOUDS = ROOT / "m3pro" / "out" / "arena" / "rtab"   # the 3D scans, from arena_map.py rtab-export
FAKES = [   # --fake starts these; robots.fake.json points at them
    [sys.executable, str(ROOT / "x3plus" / "tools" / "webapp.py"), "--sim", "--port", "8180", "--out", ""],
    [sys.executable, str(ROOT / "m3pro" / "arena" / "fake_m3.py"), "--port", "8195"],
]
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".json": "application/json"}
CLOUD_TYPES = {".ply": "application/octet-stream", ".png": "image/png"}   # scans, floor prints
GET_TIMEOUT, POST_TIMEOUT = 3, 30   # s; a POST may move an arm before it answers


def make_handler(robots, clouds=CLOUDS, recognizer=None):
    urls = {r["id"]: r["url"].rstrip("/") for r in robots}
    recognizer = recognizer or voice.Recognizer()   # loads nothing until the page asks

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
            if path == "/api/voice":   # ready, loading, or why not; starts the model's load
                return self.send(200, json.dumps(recognizer.status()).encode(), TYPES[".json"])
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
                code, reply = voice.answer(recognizer, body, self.headers.get("Content-Type", ""))
                return self.send(code, json.dumps(reply).encode(), TYPES[".json"])
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
    """The PLY files in the folder, newest first: name, size, when each was written, and the
    floor print exported with it (NAME_arena.ply -> NAME_floor.png), or None."""
    files = sorted(folder.glob("*.ply"), key=lambda f: f.stat().st_mtime, reverse=True)
    prints = [f.with_name(f.name.removesuffix("_arena.ply").removesuffix(".ply") + "_floor.png")
              for f in files]
    return [{"name": f.name, "bytes": f.stat().st_size,
             "written": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime)),
             "floor": floor.name if floor.is_file() else None}
            for f, floor in zip(files, prints)]


class Server6(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fake", action="store_true", help="pretend robots, no hardware")
    args = parser.parse_args()
    robots = json.loads((HERE / ("robots.fake.json" if args.fake else "robots.json")).read_text())
    # not waited for: the page asks each robot every 0.5 s until it answers
    procs = [subprocess.Popen(cmd, cwd=ROOT) for cmd in FAKES] if args.fake else []
    handler = make_handler(robots)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    try:   # "localhost" is tried as ::1 first on Windows; without this each connection waits
        server6 = Server6(("::1", PORT), handler)
        threading.Thread(target=server6.serve_forever, daemon=True).start()
    except OSError:
        pass   # no IPv6 loopback here
    print(f"console: http://localhost:{PORT}/  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for proc in procs:
            proc.terminate()


if __name__ == "__main__":
    main()
