"""console/serve.py against a pretend robot: JSON passes through both ways, an unreachable
robot is a 502 with a reason, and nothing outside console/ is served; of the 3D scans, only the
PLY files, splats and floor prints in their own folder; an object scan fetched from the robot and
fused; the newest scan made a Gaussian splat.

    python -m pytest console
"""

import io
import json
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
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
    port = start(ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(robots)))
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


def test_the_clouds_are_the_scans_and_floor_prints_in_their_folder_and_nothing_else(tmp_path):
    clouds = tmp_path / "rtab"
    (clouds / "sub").mkdir(parents=True)
    (clouds / "arena_arena.ply").write_bytes(b"ply " + bytes(range(256)) * 400)
    (clouds / "arena_floor.png").write_bytes(b"png")
    (clouds / "arena_camera_poses.txt").write_bytes(b"poses")
    (clouds / "old.ply").write_bytes(b"ply ")   # a scan without a floor print
    (clouds / "splat_can.splat").write_bytes(bytes(64))   # a Gaussian splat: two of 32 bytes
    (clouds / "sub" / "deeper.ply").write_bytes(b"ply ")
    (tmp_path / "outside.ply").write_bytes(b"ply ")
    port = start(ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler([], clouds)))
    base = f"http://127.0.0.1:{port}"

    listed = {c["name"]: c for c in json.loads(fetch(f"{base}/clouds.json")[1])}
    assert sorted(listed) == ["arena_arena.ply", "old.ply", "splat_can.splat"]
    assert listed["splat_can.splat"]["kind"] == "splat" and listed["old.ply"]["kind"] == "cloud"
    assert fetch(f"{base}/clouds/splat_can.splat") == (200, bytes(64))
    assert listed["arena_arena.ply"]["bytes"] == 4 + 256 * 400
    assert listed["arena_arena.ply"]["floor"] == "arena_floor.png" and listed["old.ply"]["floor"] is None
    status, body = fetch(f"{base}/clouds/arena_arena.ply")
    assert status == 200 and body == (clouds / "arena_arena.ply").read_bytes()
    assert fetch(f"{base}/clouds/arena_floor.png") == (200, b"png")
    assert fetch(f"{base}/clouds/arena_camera_poses.txt")[0] == 404   # scans and prints only
    assert fetch(f"{base}/clouds/sub/deeper.ply")[0] == 404         # that folder only
    assert fetch(f"{base}/clouds/..%2Foutside.ply")[0] == 404
    assert fetch(f"{base}/clouds/nothing.ply")[0] == 404

    gone = serve.make_handler([], tmp_path / "not there")   # no export yet: an empty list
    port = start(ThreadingHTTPServer(("127.0.0.1", 0), gone))
    assert json.loads(fetch(f"http://127.0.0.1:{port}/clouds.json")[1]) == []


def test_a_scan_is_fetched_from_the_robot_and_fused_into_the_3d_scans(tmp_path):
    stamp = "20261012_165541"
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as z:   # as the robot's /pick/scan.zip gives it
        z.writestr(f"{stamp}/00_0.json", "{}")
        z.writestr(f"{stamp}/scan.json", "{}")

    class ScanRobot(Robot):
        def do_GET(self):
            if self.path != "/pick/scan.zip":
                return self.answer(404, {"error": "no such endpoint"})
            if self.server.zipped is None:
                return self.answer(404, {"error": "no scan yet"})
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(len(self.server.zipped)))
            self.end_headers()
            self.wfile.write(self.server.zipped)

    with_scan = ThreadingHTTPServer(("127.0.0.1", 0), ScanRobot)
    with_scan.zipped = zipped.getvalue()
    without = ThreadingHTTPServer(("127.0.0.1", 0), ScanRobot)
    without.zipped = None
    robots = [{"id": "m3arm", "url": f"http://127.0.0.1:{start(with_scan)}"},
              {"id": "bare", "url": f"http://127.0.0.1:{start(without)}"}]
    fused = []
    scans = serve.Scans(tmp_path / "scan", fuse=lambda s: fused.append(s) or "the thing: 16.7 cm tall")
    port = start(ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(robots, tmp_path / "rtab", scans=scans)))
    base = f"http://127.0.0.1:{port}"

    def finished():
        for _ in range(100):
            state = json.loads(fetch(f"{base}/api/scan")[1])
            if state["state"] in ("done", "failed"):
                return state
            time.sleep(0.05)
        raise AssertionError(f"still {state}")

    assert json.loads(fetch(f"{base}/api/scan")[1])["state"] == "idle"
    status, _body = fetch(f"{base}/api/scan", {"robot": "m3arm"})
    assert status == 200
    assert finished() == {"state": "done", "stamp": stamp, "cloud": f"object_{stamp}.ply",
                          "message": "the thing: 16.7 cm tall"}
    assert (tmp_path / "scan" / stamp / "00_0.json").is_file() and fused == [stamp]

    fetch(f"{base}/api/scan", {"robot": "bare"})   # the robot's own reason
    assert finished() == {"state": "failed", "stamp": None, "cloud": None, "message": "no scan yet"}
    assert fetch(f"{base}/api/scan", {"robot": "nobody"})[0] == 404


def test_the_newest_scan_becomes_a_splat_among_the_3d_scans(tmp_path):
    scans = tmp_path / "scan"
    for stamp in ("20261012_165048", "20261012_172202", "20261013_090000"):
        (scans / stamp).mkdir(parents=True)
    for stamp in ("20261012_165048", "20261012_172202"):   # the third has no scan.json: not a scan
        (scans / stamp / "scan.json").write_text("{}")
    said = ["2. scan: its views", "4. fitting the splat on hekate", "** WARNING: ssh's key exchange",
            "held-out views: PSNR 25.52 dB", "-> m3pro/out/arena/rtab/splat_jetson-box_20261012_172202.splat"]
    asked = []

    def cycle(args):   # tools/splat_cycle.py, pretended: what it was asked, then its lines
        asked.append(args)
        return [sys.executable, "-c", f"import sys; print(*{said!r}, sep=chr(10)); sys.exit({len(asked) - 1})"]

    splats = serve.Splats(scans, command=cycle)
    port = start(ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler([], tmp_path / "rtab", splats=splats)))
    base = f"http://127.0.0.1:{port}"

    def finished():
        for _ in range(200):
            state = json.loads(fetch(f"{base}/api/splat")[1])
            if state["state"] in ("done", "failed"):
                return state
            time.sleep(0.05)
        raise AssertionError(f"still {state}")

    assert json.loads(fetch(f"{base}/api/splat")[1])["state"] == "idle"
    status, body = fetch(f"{base}/api/splat", {"name": "jetson box!"})
    assert status == 200 and json.loads(body)["stamp"] == "20261012_172202"
    assert finished() == {"state": "done", "stamp": "20261012_172202", "message": "held-out views: PSNR 25.52 dB",
                          "splat": "splat_jetson-box_20261012_172202.splat"}
    assert asked == [["20261012_172202", "--name", "jetson-box"]]

    fetch(f"{base}/api/splat", {})   # the cycle fails (exit 1): its last line says why
    state = finished()
    assert state["state"] == "failed" and state["message"].endswith("splat_jetson-box_20261012_172202.splat")
    assert asked[-1] == ["20261012_172202"]

    empty = serve.make_handler([], tmp_path / "rtab", splats=serve.Splats(tmp_path / "none", command=cycle))
    port = start(ThreadingHTTPServer(("127.0.0.1", 0), empty))
    status, body = fetch(f"http://127.0.0.1:{port}/api/splat", {})
    assert status == 409 and "no scan" in json.loads(body)["error"]
