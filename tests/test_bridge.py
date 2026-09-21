"""The hardware bridge and its remote clients, end to end over loopback.

A Bridge wrapping the simulator plays the robot; RemoteArm and RemoteStream on
this side are what tools/webapp.py --robot uses. The interesting part is that the
stop button still works with the arm on the far side of an HTTP call.
"""

import json
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from roboarm import config as cfg
from roboarm import detect
from roboarm import kinematics as kin
from roboarm.arm import ArmError
from roboarm.web import bridge, remote, session, sim


@pytest.fixture
def robot():
    """(bridge, base url) around a simulated arm, moves instant."""
    arm = sim.SimArm(time_scale=0)
    matrix, pose, name = sim.calibration()[0]
    path = Path(tempfile.mkdtemp()) / "table_homography.json"
    path.write_text(json.dumps({"homography": matrix.tolist(), "name": name,
                                "survey_pose": {str(j): a for j, a in pose.items()}}))
    hw = bridge.Bridge(arm_factory=lambda: arm,
                       stream=lambda: sim.SimStream(arm, fps=40),
                       calibration_path=path)
    srv = bridge.make_server(hw, "127.0.0.1", 0)
    srv.max_stream_frames = 5
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield hw, f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    hw.close()


def _wait(condition, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting")


def test_health_before_and_after_connecting(robot):
    _hw, url = robot
    info = remote.health(url)
    assert info["ok"] and info["arm"]["connected"] is False and info["calibration"]
    with remote.RemoteArm(url):
        assert remote.health(url)["arm"]["connected"] is True


def test_remote_arm_reads_and_moves_with_int_joint_keys(robot):
    hw, url = robot
    with remote.RemoteArm(url) as arm:
        pose = arm.read()
        assert pose == cfg.HOME_POSE and all(isinstance(j, int) for j in pose)
        landed = arm.move_to({1: 100, 2: 80}, speed_dps=30)
        assert landed[1] == 100 and landed[2] == 80
        assert hw.arm.pose[1] == 100, "the robot's own arm moved"
        assert arm.get_battery_voltage() == pytest.approx(12.3)
        arm.close_gripper()
        assert arm.read()[6] == cfg.GRIPPER_CLOSED and arm.grasped() is False


def test_arm_errors_cross_the_wire_as_arm_errors(robot):
    _hw, url = robot
    with remote.RemoteArm(url) as arm, pytest.raises(ArmError, match="safe range"):
        arm.move_to({2: 170})


def test_unknown_methods_are_refused(robot):
    _hw, url = robot
    reply = remote._request(url + "/arm/call", {"method": "bot", "args": [], "kwargs": {}})
    assert "error" in reply and "bridge" in reply["error"]


def test_the_stop_button_reaches_a_move_on_the_robot(robot):
    hw, url = robot
    stop = threading.Event()
    with remote.RemoteArm(url) as arm:
        # Slow enough that the watcher's 50 ms poll and one HTTP round trip land
        # inside the move, which then takes about a second.
        hw.arm.time_scale = 0.5
        arm.interrupt = stop.is_set
        result = {}

        def long_move():
            try:
                arm.move_to({1: 20}, speed_dps=30)
            except ArmError as exc:
                result["error"] = str(exc)
        thread = threading.Thread(target=long_move)
        thread.start()
        _wait(lambda: hw.arm.pose[1] < 88)
        stop.set()
        thread.join(timeout=5)
        assert "interrupted" in result["error"]
        assert 20 < hw.arm.pose[1] < 90
        # ...and a stale interrupt does not poison the next call.
        stop.clear()
        assert arm.hold()[1] == hw.arm.pose[1]
        hw.arm.time_scale = 0
        assert arm.move_to({1: 60})[1] == 60


def test_unreachable_bridge_is_an_arm_error():
    with pytest.raises(ArmError, match="cannot reach"), remote.RemoteArm("http://127.0.0.1:1"):
        pass


def test_remote_stream_delivers_decoded_frames(robot):
    _hw, url = robot
    stream = remote.RemoteStream(url).start()
    try:
        _wait(lambda: stream.latest()[0] is not None)
        frame, seq = stream.latest()
        assert frame.shape == (cfg.WRIST_CAM_SIZE[1], cfg.WRIST_CAM_SIZE[0], 3)
        assert seq >= 1
        assert remote.health(url)["camera"]["open"] is True
    finally:
        stream.stop()


def test_a_dead_bridge_is_reported_by_the_stream():
    stream = remote.RemoteStream("http://127.0.0.1:1").start()
    try:
        _wait(lambda: stream.error is not None)
        assert stream.latest()[0] is None
    finally:
        stream.stop()


def test_calibration_comes_across_intact(robot):
    _hw, url = robot
    looks = remote.calibration(url)
    assert len(looks) == 1
    matrix, pose, name = looks[0]
    assert name == "primary" and pose == sim.REAL_SURVEY
    assert np.allclose(matrix, sim.REAL_H)


def test_a_whole_session_runs_over_the_bridge(robot, monkeypatch):
    """The same sweep-and-pick the local session does, with the arm and camera
    behind HTTP. The block ends up under the drop pose on the 'robot'."""
    hw, url = robot
    monkeypatch.setattr(detect, "BACKGROUND_PATH",
                        Path(tempfile.mkdtemp()) / "table_background.png")
    sess = session.Session(
        arm_factory=lambda: remote.RemoteArm(url),
        stream=lambda: remote.RemoteStream(url),
        calibration=lambda: remote.calibration(url),
    )
    sess.settle_s = 0.05
    sess.start()
    try:
        assert sess.snapshot()["arm"]["connected"]
        sess.set_mode("auto")
        job = sess.start_job("pick", refine=False)
        _wait(lambda: job.status != "running", timeout=30)
        assert job.status == "done", job.message
        x, y, _z = kin.forward({**cfg.DROP_POSE, cfg.GRIPPER_ID: cfg.GRIPPER_OPEN})
        moved = [b for b in hw.arm.world.blocks
                 if abs(b.x - x) < 0.012 and abs(b.y - y) < 0.012]
        assert len(moved) == 1 and not moved[0].held
    finally:
        sess.close()
