"""roboarm/m3kinematics.py: J1 and J5 turn the measured ways, the guard keeps the arm off the chassis Yahboom's
model describes and off the floor, and reach() finds again the poses that put the grip in front."""

import math

import numpy as np

from roboarm import m3kinematics as k


def test_a_bigger_j1_turns_the_arm_to_the_right():
    left, _ = k.grip({1: 70, 2: 120, 3: 30, 4: 0, 5: 90})
    right, _ = k.grip({1: 110, 2: 120, 3: 30, 4: 0, 5: 90})
    assert left[1] > 0.02 > -0.02 > right[1]   # +y is the robot's left


def test_the_guard_follows_the_chassis_and_the_floor():
    assert k.problem({1: 90, 2: 120, 3: 30, 4: 0, 5: 90}) is None             # the survey pose
    assert "chassis" in k.problem({1: 90, 2: 60, 3: 10, 4: 10, 5: 90})         # onto the front deck
    assert "floor" in k.problem({1: 90, 2: 0, 3: 30, 4: 60, 5: 90})            # into the floor ahead
    assert k.problem({1: 90, 2: 40, 3: 30, 4: 20, 5: 90}) is None              # low, just past the bumper


def test_reach_finds_again_the_poses_that_grip_in_front():
    rng = np.random.default_rng(2)
    hits = tries = 0
    while tries < 80:
        pose = {1: int(rng.integers(50, 131)), 2: int(rng.integers(0, 181)), 3: int(rng.integers(0, 181)),
                4: int(rng.integers(0, 181)), 5: 90}
        lean = (90 - pose[2]) + (90 - pose[3]) + (90 - pose[4])   # the tool pointing forward
        if k.problem(pose) or not 0 < lean < 180:
            continue
        point, pitch = k.grip(pose)
        if point[0] < 0.12:
            continue
        tries += 1
        found = k.reach(point, pitch)
        if found is not None:
            hits += 1
            again, again_pitch = k.grip(found)
            assert np.linalg.norm(again - point) <= 0.005 and abs(again_pitch - pitch) <= 3
    assert hits >= 0.95 * tries


def test_a_cube_in_front_can_be_reached_from_above_at_a_slant():
    pose = k.reach((0.24, 0.0, k.FLOOR + 0.06), 70)
    assert pose is not None and k.problem(pose) is None
    point, pitch = k.grip(pose)
    assert abs(point[0] - 0.24) < 0.005 and abs(pitch - 70) <= 3


def test_j5_turns_the_way_it_was_seen_to():
    """2026-10-01, the gripper pointing down over the table: J5 90 -> 130 swung the finger on the
    robot's left (+y) forward (+x), away from the robot."""
    pose = {1: 90, 2: 56, 3: 21, 4: 27}
    left = k.frames({**pose, 5: 90})["arm5"][:3, 1]
    assert left[1] > 0.9                                   # at J5 90 the fingers open left-right
    turned = k.frames({**pose, 5: 130})["arm5"][:3, 1]
    assert turned[0] > 0.5 and math.degrees(math.atan2(turned[1], turned[0])) < 60
