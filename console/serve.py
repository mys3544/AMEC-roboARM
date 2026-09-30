"""The console: one page on this laptop for both robots -- one at a time, or side by side,
with a STOP for each and STOP ALL always on screen.

    python console/serve.py            # the real robots, as console/robots.json lists them
    python console/serve.py --fake     # a pretend X3 (tools/webapp.py --sim) and M3
                                       # (m3pro/arena/fake_m3.py) started alongside, in the
                                       # project venv: .venv/Scripts/python console/serve.py --fake
    -> http://localhost:8099/

The page (index.html, console.js, console.css) talks to each robot's own web API. JSON goes
through here as /robot/<id>/<path> -- the X3's panel sends no CORS headers, and one origin keeps
the page simple; camera frames and maps come straight from the robots, which images may. The
console listens on localhost only: it is this laptop's (and a microphone needs localhost).
Standard library only.
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
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PORT = 8099
FAKES = [   # --fake starts these; robots.fake.json points at them
    [sys.executable, str(ROOT / "tools" / "webapp.py"), "--sim", "--port", "8180", "--out", ""],
    [sys.executable, str(ROOT / "m3pro" / "arena" / "fake_m3.py"), "--port", "8195"],
]
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".json": "application/json"}
GET_TIMEOUT, POST_TIMEOUT = 3, 30   # s; a POST may move an arm before it answers


def make_handler(robots):
    urls = {r["id"]: r["url"].rstrip("/") for r in robots}

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
            file = (HERE / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
            if file.parent != HERE or file.suffix not in TYPES or not file.is_file():
                return self.error(404, "not found")
            self.send(200, file.read_bytes(), TYPES[file.suffix])

        def do_POST(self):
            if self.path.startswith("/robot/"):
                return self.proxy("POST")
            self.rfile.read(int(self.headers.get("Content-Length") or 0))   # keep-alive: drain
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


class Server(ThreadingHTTPServer):
    daemon_threads = True


class Server6(Server):
    address_family = socket.AF_INET6


def start_fakes(robots):
    """Start the pretend robots and wait (up to 60 s) until each answers."""
    procs = [subprocess.Popen(cmd, cwd=ROOT) for cmd in FAKES]
    for robot in robots:
        for _ in range(120):
            try:
                urllib.request.urlopen(robot["url"].rstrip("/") + "/api/state", timeout=1).close()
                break
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        else:
            print(f"note: the pretend {robot['id']} did not answer", file=sys.stderr)
    return procs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fake", action="store_true", help="pretend robots, no hardware")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    robots = json.loads((HERE / ("robots.fake.json" if args.fake else "robots.json")).read_text())
    procs = start_fakes(robots) if args.fake else []
    handler = make_handler(robots)
    server = Server(("127.0.0.1", args.port), handler)
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


if __name__ == "__main__":
    main()
