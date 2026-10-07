"""tools/x3bridge.py over HTTP against the simulator: the console's arm card's calls -- the M3's
/pick/* -- answered by the X3, a job run through them, and the bridge's own arm calls beside."""

import importlib.util
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from roboarm import x3pick
from roboarm.web import bridge, sim

spec = importlib.util.spec_from_file_location("x3bridge", Path(__file__).resolve().parents[1] / "tools" / "x3bridge.py")
x3bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(x3bridge)


@pytest.fixture
def url():
    world = sim.World([sim.Block(0.160, 0.020, 0.040, (0, 120, 255))])
    the_bridge = bridge.Bridge(arm_factory=lambda: sim.SimArm(world, time_scale=0),
                               stream=lambda: sim.SimStream(the_bridge.arm, fps=40), calibration_path=Path("sim"))
    the_bridge.connect()
    box = []
    picker = x3pick.Picker(x3bridge.serve.arm_caller(the_bridge, lambda: box[0]), the_bridge.stream, sim.calibration,
                           settle_s=0.0, rng=np.random.default_rng(0), detector_url=None)
    box.append(picker)
    server = bridge.Server(("127.0.0.1", 0), x3bridge.make_handler(the_bridge, picker))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    the_bridge.close()


def get(url, path):
    with urllib.request.urlopen(url + path, timeout=10) as reply:
        return reply.status, reply.headers.get("Content-Type"), reply.read()


def post(url, path, body):
    request = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=10) as reply:
        return json.loads(reply.read())


def test_the_arm_cards_calls_are_answered_and_a_pick_runs_through_them(url):
    _s, _t, body = get(url, "/pick/state")
    state = json.loads(body)
    assert state["job"] is None and not state["busy"] and state["moved_cm"] == [0.0, 0.0] and state["box"] is None
    keys = {s["key"] for s in json.loads(get(url, "/pick/settings")[2])["settings"]}
    assert {"speed", "free_dps", "reach_offset_mm"} <= keys
    assert post(url, "/pick/settings", {"speed": 60})["settings"]
    assert post(url, "/pick/start", {"kind": None, "drive": False}) == {"ok": True, "job": "pick_once"}
    for _ in range(300):
        state = json.loads(get(url, "/pick/state")[2])
        if not state["busy"]:
            break
        time.sleep(0.05)
    assert state["error"] is None and state["picked"] == 1
    lines = json.loads(get(url, "/pick/log?since=0")[2])["lines"]
    assert any("holding something" in line["text"] for line in lines)
    status, kind, jpeg = get(url, "/pick/frame.jpg")
    assert status == 200 and kind == "image/jpeg" and jpeg[:2] == b"\xff\xd8"
    assert "runs" in json.loads(get(url, "/api/metrics?events=0")[2])


def test_the_bridges_own_calls_stay(url):
    assert json.loads(get(url, "/health")[2])["arm"]["connected"]
    pose = post(url, "/arm/call", {"method": "read", "args": [], "kwargs": {}})["result"]
    assert set(pose) >= {"1", "2", "3", "4", "5", "6"}
    fine = post(url, "/arm/call", {"method": "read_fine", "args": [], "kwargs": {}})["result"]
    assert set(fine) >= {"1", "2", "3", "4", "5"}
    for _ in range(50):   # 503 until the camera's first frame
        try:
            _status, kind, jpeg = get(url, "/snapshot.jpg")
            break
        except urllib.error.HTTPError:
            time.sleep(0.05)
    assert kind == "image/jpeg" and jpeg[:2] == b"\xff\xd8"
