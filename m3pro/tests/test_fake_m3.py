"""arena/fake_m3.py on the saved arena_v2 map: the lidar sees the fence where the fence fit puts
it, a drive command moves the pretend robot -- by hand right up to a wall (no distance limit),
where the simulation's chassis stops it -- the loop stops and goes, and the survey stands
still whenever the map records."""

import itertools
import math
import pathlib
import sys

import numpy as np

ARENA = pathlib.Path(__file__).resolve().parents[1] / "arena"
sys.path.insert(0, str(ARENA))
import drive_core
import fake_m3


def test_scan_sees_the_fence_and_by_hand_the_robot_drives_up_to_a_wall():
    robot = fake_m3.FakeM3(ARENA / "maps" / "arena_v2.yaml", seed=0)
    scan = robot.scan()
    ranges = np.hypot(scan[:, 0], scan[:, 1])
    assert len(scan) > 600 and ranges.max() < 2.6   # all round, nothing past the far corner

    x0, y0, _ = robot.pose()
    for _ in range(200):   # 20 s of full speed forward, the command repeated as the page does
        robot.core.drive({"vx": 1.0})
        robot.core.set_scan(robot.scan())
        robot.core.tick()
        robot.step(0.1)
    x, y, _yaw = robot.pose()
    moved = math.hypot(x - x0, y - y0)
    assert moved > 0.5                                    # it went
    ahead = robot.scan()
    along = ahead[np.abs(ahead[:, 1]) < drive_core.HALF_PATH][:, 0]
    nearest = along[along > 0].min()
    assert nearest < drive_core.STOP_AT - 0.05          # closer than the loop would go
    assert nearest > fake_m3.CHASSIS - 0.03             # up to the wall, not through it


def run(robot, seconds, dt=0.1):
    for _ in range(int(seconds / dt)):
        robot.core.set_scan(robot.scan())
        robot.core.tick()
        robot.step(dt)


def test_stop_and_go_pauses_every_step_until_told_to_go_on():
    robot = fake_m3.FakeM3(ARENA / "maps" / "arena_v2.yaml", seed=0)
    assert robot.core.resume() is not None                 # nothing to go on with yet
    assert robot.core.start_auto(step=0.5) is None
    start = np.array(robot.pose()[:2])
    run(robot, 12)                                         # 0.5 m at 8 cm/s: about 6 s
    state = robot.core.state()
    assert state["paused"] and state["mode"] == "paused" and robot.vel == (0.0, 0.0, 0.0)
    here = np.array(robot.pose()[:2])
    assert 0.45 < np.linalg.norm(here - start) < 0.62      # it drove its half metre, no more
    run(robot, 3)
    assert np.linalg.norm(np.array(robot.pose()[:2]) - here) < 0.01   # and waits
    assert robot.core.resume() is None
    run(robot, 3)
    assert not robot.core.state()["paused"] and np.linalg.norm(np.array(robot.pose()[:2]) - here) > 0.1


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def test_the_survey_stands_still_while_the_map_records():
    robot = fake_m3.FakeM3(ARENA / "maps" / "arena_v4.yaml", seed=0)
    clock = robot.core.clock = Clock()
    robot.core.set_scan(robot.scan())
    assert robot.core.start_auto(plan="survey") is None
    events = [(robot.core.recording, robot.pose())]   # each change of `recording`, and where
    assert events == [(False, robot.pose())] and robot.core.state()["plan"] == "survey"
    for k in range(int(60 / 0.05)):   # a minute: the first stations
        if k % 3 == 0:
            robot.core.set_scan(robot.scan())
        robot.core.tick()
        if robot.core.recording != events[-1][0]:
            events.append((robot.core.recording, robot.pose()))
        robot.step(0.05)
        clock.t += 0.05
    assert robot.core.auto is not None and robot.core.auto["i"] >= 6 and robot.core.auto["skips"] == 0
    for (on, p), (off, q) in itertools.pairwise(events[1:]):
        if on and not off:   # recording: the robot did not move
            assert math.hypot(q[0] - p[0], q[1] - p[1]) < 1e-6 and abs(q[2] - p[2]) < 1e-6
    assert sum(on for on, _p in events) >= 6
    robot.core.stop()
    robot.core.tick()
    assert robot.core.recording   # a survey stopped: the map records as usual again
