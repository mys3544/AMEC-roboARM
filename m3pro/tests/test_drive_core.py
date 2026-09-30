"""arena/drive_core.py: the drive page's safety limit, its loop planner, and the Driver's
dead-man and stop, with a pretend clock and base."""

import itertools
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "arena"))
import drive_core as ds


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


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def test_a_command_lapses_without_repeats_and_stop_stops_at_once():
    clock, sent = Clock(), []
    driver = ds.Driver(lambda: (0.0, 0.0, 0.0), lambda *v: sent.append(v), clock)
    driver.set_scan(wall_ahead(2.0))
    driver.drive({"vx": 1.0})
    driver.tick()
    assert driver.mode == "manual" and sent[-1] == (ds.MAX_V, 0.0, 0.0)
    clock.t += ds.DEADMAN + 0.01   # the page stopped repeating it (a closed tab)
    driver.set_scan(wall_ahead(2.0))
    driver.tick()
    assert driver.mode == "idle" and sent[-1] == (0.0, 0.0, 0.0)
    driver.drive({"vx": 1.0})
    driver.tick()
    driver.stop()
    driver.tick()
    assert driver.mode == "idle" and sent[-1] == (0.0, 0.0, 0.0)
    count = len(sent)
    driver.tick()
    assert len(sent) == count   # idle: nothing sent, the base is the gamepad's


def test_the_loop_keeps_its_distance_but_driving_by_hand_does_not():
    clock, sent = Clock(), []
    driver = ds.Driver(lambda: (0.0, 0.0, 0.0), lambda *v: sent.append(v), clock)
    driver.set_scan(wall_ahead(0.30))   # something 30 cm ahead
    driver.drive({"vx": 1.0})
    driver.tick()
    assert sent[-1] == (ds.MAX_V, 0.0, 0.0) and driver.nearest == 0.30   # by hand: shown only
    driver.stop()
    driver.tick()
    driver.auto = {"waypoints": [(2.0, 0.0)], "i": 0, "heading": 0.0, "blocked_since": None,
                   "skips": 0, "step": None, "travelled": 0.0, "from": (0.0, 0.0), "paused": False}
    driver.tick()
    assert driver.mode == "auto" and sent[-1][:2] == (0.0, 0.0)   # the loop stops short


def test_bench_mode_refuses_to_move():
    clock, sent = Clock(), []
    driver = ds.Driver(lambda: (0.0, 0.0, 0.0), lambda *v: sent.append(v), clock, motion=False)
    driver.set_scan(wall_ahead(2.0))
    assert driver.drive({"vx": 1.0}) == ds.BENCH
    assert driver.start_auto() == ds.BENCH
    driver.tick()
    assert sent == [] and driver.mode == "idle" and driver.state()["motion"] is False


def test_the_loop_needs_a_fresh_scan_driving_by_hand_does_not():
    clock, sent = Clock(), []
    driver = ds.Driver(lambda: (0.0, 0.0, 0.0), lambda *v: sent.append(v), clock)
    driver.set_scan(wall_ahead(2.0))
    clock.t += ds.SCAN_STALE + 0.01
    driver.auto = {"waypoints": [(2.0, 0.0)], "i": 0, "heading": 0.0, "blocked_since": None,
                   "skips": 0, "step": None, "travelled": 0.0, "from": (0.0, 0.0), "paused": False}
    driver.tick()
    assert sent[-1] == (0.0, 0.0, 0.0) and "no lidar" in driver.note
    driver.drive({"vx": 1.0})
    driver.tick()
    assert sent[-1] == (ds.MAX_V, 0.0, 0.0)


def test_the_survey_keeps_its_distance_and_sees_the_floor_and_the_landmarks():
    length, width = 3.2, 2.8   # the arena as arena_v4 has it: two landmarks 0.6-0.8 m off a wall
    blobs = [((2.39, 0.76), 0.11), ((2.39, 1.95), 0.15), ((0.82, 1.94), 0.11)]
    stations = ds.survey(blobs, length, width, (1.6, 1.4))
    assert 30 < len(stations) < 120
    route = np.array([s[:2] for s in stations])
    for p, q in itertools.pairwise(np.vstack([(1.6, 1.4), route])):   # every leg, not only stops
        for t in np.linspace(0, 1, 20):
            x, y = p + t * (q - p)
            assert min(x, length - x, y, width - y) >= ds.SURVEY_CLEAR - 0.03
            for c, r in blobs:
                assert math.hypot(x - c[0], y - c[1]) - r >= ds.SURVEY_CLEAR - 0.03
    step = 0.05
    A, B = np.meshgrid((np.arange(64) + 0.5) * step, (np.arange(56) + 0.5) * step, indexing="ij")
    floor = np.ones_like(A, bool)
    for c, r in blobs:
        floor &= np.hypot(A - c[0], B - c[1]) > r + 0.02
    quality, _ = ds.seen_by(stations, np.stack([A[floor], B[floor]], axis=1), blobs)
    assert (quality.max(axis=0) > 0).mean() > 0.97   # the floor, corners behind landmarks too
    for c, r in blobs:   # and most of each landmark's face, from its own side
        rim = np.array([(c[0] + (r + 0.02) * math.cos(t), c[1] + (r + 0.02) * math.sin(t))
                        for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)])
        quality, _ = ds.seen_by(stations, rim, blobs)
        cams = np.array([(a + ds.CAM_AHEAD * math.cos(h), b + ds.CAM_AHEAD * math.sin(h))
                         for a, b, h in stations])
        facing = np.einsum("vtk,tk->vt", cams[:, None, :] - rim[None], rim - np.array(c)) > 0
        assert ((quality * facing) > 0).any(axis=0).mean() >= 0.75
