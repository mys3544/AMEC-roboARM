"""tools/m3bridge.py's HTTP side on this laptop, no ROS: the X3's bridge contract in front of
roboarm.m3arm.M3Arm on a pretend board, and /rgbd from frames fed as the ROS node feeds them."""

import base64
import json
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import m3bridge

from roboarm import m3arm
from roboarm.web import bridge


def ask(url, body=None):
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_bridge_drives_the_m3_arm_and_serves_rgbd(tmp_path):
    sent = []
    frames, stream = m3bridge.Frames(), m3bridge.RosStream()
    fast = lambda s: None   # the pretend board needs no waiting
    the_bridge = bridge.Bridge(lambda: m3arm.M3Arm(lambda p, ms: sent.append((p, ms)), lambda: 11.8,
                                                  sleep=fast), lambda: stream, tmp_path / "none.json")
    server = m3bridge.Server(("127.0.0.1", 0), m3bridge.make_handler(the_bridge, frames))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        the_bridge.connect()
        status, health = ask(f"{base}/health")
        assert status == 200 and health["arm"]["connected"]

        status, reply = ask(f"{base}/rgbd")
        assert status == 503 and "no colour" in reply["error"]
        colour = np.full((480, 640, 3), (30, 90, 200), np.uint8)
        depth = np.full((480, 640), 612, np.uint16)
        with frames.lock:
            frames.color, frames.color_stamp = colour, 100.05
            frames.depth, frames.depth_stamp = depth, 100.0
            frames.K = [480.8, 0, 320, 0, 480.8, 240, 0, 0, 1]
        status, reply = ask(f"{base}/rgbd")
        assert status == 200 and reply["joints"] is None and reply["depth_lag_s"] == 0.05
        got = cv2.imdecode(np.frombuffer(base64.b64decode(reply["depth_png"]), np.uint8), cv2.IMREAD_UNCHANGED)
        assert got.dtype == np.uint16 and int(got[100, 100]) == 612
        assert reply["K"][0][0] == 480.8

        status, reply = ask(f"{base}/arm/call", {"method": "read"})
        assert status == 400 and "engage" in reply["error"]   # nothing known before engage
        status, reply = ask(f"{base}/arm/call", {"method": "engage"})
        assert status == 200 and sent[-1] == (dict(zip(m3arm.JOINTS, m3arm.POSES["home"])), m3arm.ENGAGE_MS)
        status, reply = ask(f"{base}/arm/call", {"method": "move_to", "args": [{"1": 100}]})
        assert status == 200 and reply["result"]["1"] == 100 and sent[-1][0][1] == 100
        status, reply = ask(f"{base}/arm/call", {"method": "close_gripper"})
        assert status == 400 and reply["type"] == "ArmError" and "width" in reply["error"]
        time.sleep(0.3)   # the last step's 125 ms: read() believes the arm on its way till then
        status, reply = ask(f"{base}/rgbd")
        assert reply["joints"]["1"] == 100
    finally:
        server.shutdown()
        the_bridge.close()
