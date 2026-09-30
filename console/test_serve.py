"""console/serve.py against a pretend robot: JSON passes through both ways, an unreachable
robot is a 502 with a reason, and nothing outside console/ is served.

    python -m pytest console
"""

import json
import pathlib
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import serve


class Robot(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def answer(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.answer(200, {"mode": "idle", "path": self.path})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.answer(409 if body.get("busy") else 200, {"got": body, "path": self.path})


def start(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]


def fetch(url, body=None):
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as reply:
            return reply.status, reply.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_the_console_passes_json_through_and_serves_only_its_own_files():
    robot_port = start(ThreadingHTTPServer(("127.0.0.1", 0), Robot))
    robots = [{"id": "m3", "url": f"http://127.0.0.1:{robot_port}"},
              {"id": "gone", "url": "http://127.0.0.1:9"}]   # nothing listens on port 9
    port = start(serve.Server(("127.0.0.1", 0), serve.make_handler(robots)))
    base = f"http://127.0.0.1:{port}"

    status, body = fetch(f"{base}/robot/m3/api/state?x=1")
    assert status == 200 and json.loads(body) == {"mode": "idle", "path": "/api/state?x=1"}
    status, body = fetch(f"{base}/robot/m3/api/drive", {"vx": 0.5})
    assert status == 200 and json.loads(body)["got"] == {"vx": 0.5}
    status, body = fetch(f"{base}/robot/m3/api/auto", {"busy": True})   # the robot's own error
    assert status == 409
    status, body = fetch(f"{base}/robot/gone/api/state")
    assert status == 502 and "not reachable" in json.loads(body)["error"]
    assert fetch(f"{base}/robot/nobody/api/state")[0] == 404

    status, body = fetch(f"{base}/")
    assert status == 200 and b"RoboARM console" in body
    assert json.loads(fetch(f"{base}/robots.json")[1]) == robots
    assert fetch(f"{base}/serve.py")[0] == 404            # not a page type
    assert fetch(f"{base}/..%2FPLAN.md")[0] == 404         # nothing outside console/
    assert fetch(f"{base}/../README.md")[0] == 404
