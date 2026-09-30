"""arena/fake_m3.py on the saved arena_v2 map: the lidar sees the fence where the fence fit puts
it, a drive command moves the pretend robot, and the page's limit stops it before a wall."""

import math
import pathlib
import sys

import numpy as np

ARENA = pathlib.Path(__file__).resolve().parents[1] / "arena"
sys.path.insert(0, str(ARENA))
import drive_core
import fake_m3


def test_scan_sees_the_fence_and_the_robot_drives_but_not_into_a_wall():
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
    assert along[along > 0].min() > drive_core.STOP_AT - 0.03   # and stopped short of the wall
    assert robot.core.nearest < drive_core.SLOW_FROM
