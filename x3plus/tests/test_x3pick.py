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


def test_the_search_looks_past_a_spot_given_up(rig):
    _world, _arm, picker = rig([sim.Block(0.150, 0.010, 0.040, (0, 120, 255)),
                                sim.Block(0.050, 0.160, 0.030, (255, 100, 0))])
    picker.skip = [np.array([0.150, 0.010])]   # missed twice (clean_up)
    found = picker.wide_look(sides=True)
    assert any(abs(o["centre"][0] - 0.050) < 0.02 and abs(o["centre"][1] - 0.160) < 0.02 for o in found), \
        [np.round(o["centre"][:2], 3).tolist() for o in found]


def test_the_detectors_overlapping_on_one_cube_are_combined():
    from roboarm import detect
    region = detect.Target(x=0.200, y=-0.090, width_m=0.027, length_m=0.030, angle_deg=10.0, label="raised",
                           confidence=0.6, height_m=0.027)
    mask = detect.Target(x=0.190, y=-0.100, width_m=0.029, length_m=0.031, angle_deg=40.0, label="cube",
                         confidence=0.5)
    elsewhere = detect.Target(x=0.120, y=0.050, width_m=0.030, length_m=0.030, angle_deg=0.0, label="colour 1")
    (one,) = detect.combine([region], [mask, elsewhere])
    assert abs(one.x - 0.195) < 1e-9 and abs(one.y + 0.095) < 1e-9
    assert one.angle_deg == 10.0 and one.label == "raised+1" and abs(one.confidence - 0.8) < 1e-9


def test_kit_sizes_and_the_robots_own_wheels():
    from roboarm import detect
    read = detect.Target(x=0.17, y=-0.09, width_m=0.027, length_m=0.033, angle_deg=0.0, label="raised",
                         height_m=0.027)
    sized = x3pick.kit_sized(read)
    assert (sized.width_m, sized.length_m, sized.height_m) == (0.030, 0.030, 0.030) and sized.x == read.x
    wheel = detect.Target(x=0.003, y=-0.106, width_m=0.028, length_m=0.058, angle_deg=0.0, label="raised")
    assert x3pick.on_body(wheel) and not x3pick.on_body(read)

