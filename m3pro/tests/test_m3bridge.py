"""tools/m3bridge.py's HTTP side on this laptop, no ROS: the X3's bridge contract in front of
roboarm.m3arm.M3Arm on a pretend board, /rgbd from frames fed as the ROS node feeds them, and a
stop reaching a move under way."""

import base64
import json
import pathlib
import re
import sys
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import m3bridge

from roboarm import camera, m3arm
from roboarm.m3arm import ArmError
from roboarm.web import bridge

HOME = dict(zip(m3arm.JOINTS, m3arm.POSES["home"]))
SURVEY = dict(zip(m3arm.JOINTS, m3arm.POSES["survey"]))


def ask(url, body=None):
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_bridge_drives_the_m3_arm_and_serves_rgbd():
    sent = []
    frames, stream = m3bridge.Frames(), camera.Stream()
    fast = lambda s: None   # the pretend board needs no waiting
    the_bridge = bridge.Bridge(lambda: m3arm.M3Arm(lambda p, ms: sent.append((p, ms)), lambda: 11.8,
                                                  sleep=fast), lambda: stream)
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

        server.max_stream_frames = 1   # the console's live view: one part, then the stream ends
        stream.publish(colour)
        with urllib.request.urlopen(f"{base}/stream.mjpg", timeout=10) as stream_reply:
            part = stream_reply.read()
        jpeg = part[part.index(b"\r\n\r\n") + 4:part.rindex(b"\r\n")]
        assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (480, 640, 3)
        assert ask(f"{base}/health")[1]["camera"]["open"]
    finally:
        server.shutdown()
        the_bridge.close()


def test_a_stop_reaches_a_move_under_way_and_only_the_allowed_calls_pass():
    """/arm/interrupt, outside the calls' lock, holds the arm where a move got to, and a stale
    interrupt does not stop the next move (the X3's bridge tests, on the M3's arm)."""
    sent = []
    the_bridge = bridge.Bridge(lambda: m3arm.M3Arm(lambda p, ms: sent.append((p, ms)), lambda: 11.8),
                               camera.Stream)
    server = m3bridge.Server(("127.0.0.1", 0), m3bridge.make_handler(the_bridge, m3bridge.Frames()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, reply = ask(f"{base}/arm/call", {"method": "glide"})
        assert status == 400 and "bridge" in reply["error"]
        the_bridge.connect()
        the_bridge.arm._send(HOME, 0)   # as engage() leaves it, without its 3.3 s on the real clock
        replies = []
        move = threading.Thread(target=lambda: replies.append(ask(
            f"{base}/arm/call", {"method": "move_to", "args": [SURVEY], "kwargs": {"speed_dps": 30}})))
        move.start()   # J2 180 -> 120 at 30 deg/s: 2 s
        time.sleep(0.4)
        assert ask(f"{base}/arm/interrupt", {})[0] == 200
        move.join(timeout=5)
        status, reply = replies[0]
        assert status == 400 and "interrupted" in reply["error"]
        held, ms = sent[-1]
        assert ms == 0 and SURVEY[2] < held[2] < HOME[2]   # held on the way, not at the goal
        status, reply = ask(f"{base}/arm/call", {"method": "move_to", "args": [SURVEY],
                                                 "kwargs": {"speed_dps": 300, "settle": False}})
        assert status == 200 and reply["result"]["2"] == SURVEY[2]
    finally:
        server.shutdown()
        the_bridge.close()


def test_a_camera_that_has_stopped_sending_is_an_error_not_an_old_picture():
    frames = m3bridge.Frames()
    with frames.lock:
        frames.color, frames.depth = np.zeros((480, 640, 3), np.uint8), np.zeros((480, 640), np.uint16)
        frames.K = [480.8, 0, 320, 0, 480.8, 240, 0, 0, 1]
        frames.arrived = {"colour": time.monotonic(), "depth": time.monotonic()}
    assert frames.arrays()[0].shape == (480, 640, 3)   # fresh: the picture
    frames.arrived["depth"] -= m3bridge.STALE_S + 1    # the depth stopped coming
    with pytest.raises(ArmError, match="camera has stopped: its newest depth"):
        frames.arrays()


def test_a_driver_whose_frames_stop_is_restarted_but_not_while_it_starts():
    frames, now = m3bridge.Frames(), [100.0]
    driver = m3bridge.Driver(frames, clock=lambda: now[0])
    driver.started = 100.0
    now[0] += m3bridge.STARTUP_S - 1
    assert driver.stalled() is None                      # starting: no frame yet is fine
    now[0] += 2
    assert driver.stalled() == "no colour frame yet"
    frames.arrived = {"colour": now[0], "depth": now[0] - m3bridge.WATCHDOG_S + 1}
    assert driver.stalled() is None
    now[0] += 2                                           # the depth stopped, the colour came on
    frames.arrived["colour"] = now[0]
    assert driver.stalled() == "no depth frame for 6 s"

    said, done = [], []
    driver.stop, driver.start = lambda: done.append("stop"), lambda: done.append("start")
    watching = threading.Thread(target=driver.watch, args=(said.append,))
    watching.start()
    time.sleep(1.5)
    driver.stalled = lambda: None                         # the new driver's frames come
    time.sleep(1.2)
    driver.close()
    watching.join(timeout=3)
    assert not watching.is_alive()
    assert said[0] == "the camera's driver restarted: no depth frame for 6 s"
    assert done[:2] == ["stop", "start"] and done[-1] == "stop"   # restarted, then stopped by close()


def test_health_says_the_camera_has_stopped_without_a_job_looking(monkeypatch):
    """2026-10-15: the DCW2's driver had sent nothing for two days and /health still said 22 fps --
    only a job's arrays() noticed. Now /health gives the newest colour's and depth's age, an error
    once either is older than STALE_S, and the stream's fps falls to 0 when frames stop."""
    monkeypatch.setattr(camera, "FPS_WINDOW_S", 0.3)   # the fps window, short for the test
    frames, stream = m3bridge.Frames(), camera.Stream()
    the_bridge = bridge.Bridge(lambda: None, lambda: stream)
    the_bridge.stream()   # as the console's live view opens it
    server = m3bridge.Server(("127.0.0.1", 0), m3bridge.make_handler(the_bridge, frames))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        camera_now = lambda: ask(f"{base}/health")[1]["camera"]
        cam = camera_now()
        assert cam["colour_age_s"] is None and cam["depth_age_s"] is None
        assert cam["error"] == "no colour or depth from the camera yet"

        colour = np.zeros((480, 640, 3), np.uint8)
        for _ in range(6):   # the camera sending, as the ROS node feeds the frames and the stream
            with frames.lock:
                frames.arrived = {"colour": time.monotonic(), "depth": time.monotonic()}
            stream.publish(colour)
            time.sleep(0.02)
        status, health = ask(f"{base}/health")
        cam = health["camera"]
        assert status == 200 and health["ok"] and cam["open"] and cam["fps"] > 10 and cam["error"] is None
        assert cam["colour_age_s"] < 1 and cam["depth_age_s"] < 1

        time.sleep(0.35)   # the driver stops: nothing more comes, past the fps window...
        with frames.lock:   # ...nor for STALE_S and more
            frames.arrived = {what: when - m3bridge.STALE_S - 3 for what, when in frames.arrived.items()}
        cam = camera_now()
        assert cam["open"] and cam["fps"] == 0   # not the last rate, held
        assert cam["colour_age_s"] > m3bridge.STALE_S + 3 and cam["depth_age_s"] > m3bridge.STALE_S + 3
        assert re.match(r"the camera has stopped: its newest colour is \d+ s old \(the bridge restarts", cam["error"])
    finally:
        server.shutdown()
