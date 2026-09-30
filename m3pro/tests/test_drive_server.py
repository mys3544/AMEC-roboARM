"""arena/drive_server.py's safety limit and its loop planner, without ROS (its modules stubbed)."""

import math
import pathlib
import sys
from unittest import mock

import numpy as np

for name in ("rclpy", "rclpy.node", "rclpy.qos", "rclpy.signals", "rclpy.time", "geometry_msgs",
             "geometry_msgs.msg", "nav_msgs", "nav_msgs.msg", "sensor_msgs", "sensor_msgs.msg",
             "std_msgs", "std_msgs.msg", "tf2_ros"):
    sys.modules.setdefault(name, mock.MagicMock())
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "arena"))
import itertools

import drive_server as ds


def wall_ahead(distance):
    """Scan points (base_link) of a wall across the robot's path, `distance` ahead."""
    return np.array([[distance, y] for y in np.linspace(-1, 1, 81)])


def test_stops_short_of_what_is_ahead_and_slows_before():
    vx, vy, _wz, nearest = ds.limit(0.15, 0.0, 0.0, wall_ahead(0.30))
    assert (vx, vy) == (0.0, 0.0) and math.isclose(nearest, 0.30)
    vx, _vy, _wz, _n = ds.limit(0.15, 0.0, 0.0, wall_ahead(0.475))
    assert math.isclose(vx, 0.075)   # half way between STOP_AT and SLOW_FROM: half speed
    vx, _vy, _wz, _n = ds.limit(0.15, 0.0, 0.0, wall_ahead(1.0))
    assert vx == 0.15


def test_what_is_behind_or_beside_does_not_stop_it():
    vx, _vy, _wz, _n = ds.limit(0.15, 0.0, 0.0, wall_ahead(-0.30))
    assert vx == 0.15
    beside = np.array([[x, 0.30] for x in np.linspace(-1, 1, 81)])   # a wall 30 cm to the left
    vx, _vy, _wz, _n = ds.limit(0.15, 0.0, 0.0, beside)
    assert vx == 0.15
    vx, vy, _wz, _n = ds.limit(0.0, 0.15, 0.0, beside)   # but not sideways into it
    assert (vx, vy) == (0.0, 0.0)


def test_no_turning_with_something_touching():
    assert ds.limit(0.0, 0.0, 0.6, wall_ahead(1.0))[2] == 0.6
    assert ds.limit(0.0, 0.0, 0.6, np.array([[0.2, 0.0]]))[2] == 0.0


def test_the_loop_keeps_clear_of_the_landmarks():
    length, width = 3.2, 2.8
    loop = [np.array(p) for p in ((2.75, 0.45), (2.75, 2.35), (0.45, 2.35), (0.45, 0.45),
                                  (2.75, 0.45), (0.45, 1.4), (2.75, 1.4))]
    # one beside a leg, one on a loop point, and the robot starting 33 cm from the first, with
    # the first corner past it (2026-09-30: the robot drove up to a landmark that way); the
    # arena's landmarks are 21-28 cm across
    blobs = [((2.37, 0.72), 0.12), ((0.5, 1.5), 0.12)]
    start = np.array((2.05, 0.62))
    route = ds.detour(loop, blobs, length, width, start)

    def gaps(p, q, c):
        return np.linalg.norm(p + np.linspace(0, 1, 200)[:, None] * (q - p) - np.array(c), axis=1)

    first = gaps(start, route[0], blobs[0][0])
    assert first.min() >= first[0] - 1e-9   # from inside the circle, only away from the blob
    for p, q in itertools.pairwise(route):
        for c, r in blobs:
            # a detour's corners cut the circle a little, but never into the robot's path band
            assert gaps(p, q, c).min() - r > ds.HALF_PATH
    for p in route:
        assert 0.3 - 1e-9 <= p[0] <= length - 0.3 + 1e-9 and 0.3 - 1e-9 <= p[1] <= width - 0.3 + 1e-9
