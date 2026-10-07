"""roboarm/x3pick.py -- the X3's body for the shared jobs (roboshared.task) -- against the simulator
(roboarm/web/sim.py): the M3's flow, the X3's eyes (the calibrated look, its ring, the colour rung)
and hands (grasp.pick, the gripper's read-back)."""

import numpy as np
import pytest

from roboarm import config as cfg
from roboarm import x3pick
from roboarm.web import sim


@pytest.fixture
def rig():
    made = []

    def make(blocks):
        world = sim.World(blocks)
        arm = sim.SimArm(world, time_scale=0).__enter__()   # connected, as the bridge does
        stream = sim.SimStream(arm, fps=40).start()
        picker = x3pick.Picker(lambda m, *a, **k: getattr(arm, m)(*a, **k), lambda: stream, sim.calibration,
                               settle_s=0.0, rng=np.random.default_rng(0), detector_url=None)
        made.append(stream)
        return world, arm, picker
    yield make
    for stream in made:
        stream.stop()


def run(picker, job, **kwargs):
    picker.start(job, **kwargs)
    picker.wait(180)
    return picker.status()


def test_one_pick_takes_the_cube_to_the_line_on_the_right_and_goes_home(rig):
    world, arm, picker = rig([sim.Block(0.160, 0.020, 0.040, (0, 120, 255))])
    state = run(picker, "pick_once")
    assert state["error"] is None, state["error"]
    assert state["picked"] == 1 and state["missed"] == 0
    block = world.blocks[0]
    assert not block.held and block.y < -0.15, (block.x, block.y)   # let go out on the right
    assert picker.line_drops == 1 and state["box"] is None          # no marker in the simulator
    assert arm.read()[2] == cfg.HOME_POSE[2]
    last = state["last"]
    assert last["picked"] and last["kind"].endswith("mm cube") and last["yaw_from"] == "outline"
    assert any("holding something" in e["text"] for e in picker.log())   # grasp.py's narration


def test_a_miss_the_gripper_feels_is_not_dropped_off(rig):
    _world, arm, picker = rig([sim.Block(0.160, 0.020, 0.040, (0, 120, 255))])
    arm.grasped = lambda: False   # the fingers met nothing
    state = run(picker, "pick_once")
    assert state["missed"] == 1 and state["picked"] == 0
    assert picker.line_drops == 0
    assert any("closed on nothing: nothing to drop off" in e["text"] for e in picker.log())


def test_clean_up_takes_both_and_says_why_it_stopped(rig):
    world, _arm, picker = rig([sim.Block(0.150, 0.010, 0.040, (0, 120, 255)),
                               sim.Block(0.170, -0.055, 0.030, (255, 100, 0))])
    state = run(picker, "clean_up")
    assert state["error"] is None, state["error"]
    assert state["picked"] == 2, [e["text"] for e in picker.log()][-12:]
    assert all(b.y < -0.15 for b in world.blocks)
    assert any("clean-up done" in e["text"] for e in picker.log())


def test_the_found_dict_is_the_shared_flows():
    from roboarm import detect
    target = detect.Target(x=0.16, y=0.02, width_m=0.03, length_m=0.06, angle_deg=200.0, label="colour",
                           confidence=0.9, height_m=0.03)
    o = x3pick.found_of(target, reach=True)
    assert o["kind"] == "block" and o["period"] == 180 and o["yaw"] == 20.0
    assert abs(o["centre"][2] - (x3pick.TABLE_Z + 0.015)) < 1e-12 and o["whole"]
