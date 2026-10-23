"""tools/m3_sim.py's side of the simulator, without it: servo degrees become the URDF's radians as the
arm model turns them, the gripper's J6 the finger joint through the measured opening, and a plan's keys
are timed by the furthest joint at its speed."""

import itertools
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import m3_sim

from roboarm import m3kinematics


def test_servo_degrees_become_the_urdfs_radians():
    q = m3_sim.urdf_targets({1: 60, 2: 180, 3: 90, 4: 0, 5: 90, 6: 0})
    assert q["arm1_Joint"] == pytest.approx(math.radians(30))     # J1 mirrored: a bigger J1 turns right
    assert q["arm2_Joint"] == pytest.approx(math.radians(90))
    assert q["arm3_Joint"] == pytest.approx(0.0)
    assert q["arm4_Joiint"] == pytest.approx(math.radians(-90))
    assert q["rlink1_Joint"] == pytest.approx(0.6)                # the open stop


def test_the_finger_closes_as_the_measured_gripper_does():
    angles = [m3_sim.urdf_targets({1: 90, 2: 90, 3: 90, 4: 90, 5: 90, 6: j6})["rlink1_Joint"] for j6 in range(0, 181, 10)]
    assert all(b <= a for a, b in itertools.pairwise(angles))   # never opens as J6 grows
    assert m3_sim.urdf_targets({1: 90, 2: 90, 3: 90, 4: 90, 5: 90, 6: 90})["rlink1_Joint"] == pytest.approx(-0.022)


def test_a_plan_is_timed_by_its_furthest_joint_and_the_sag_left_as_it_was():
    plan = m3_sim.Plan({1: 90, 2: 180, 3: 5, 4: 0, 5: 90, 6: 0})
    plan.go({2: 120, 3: 35}, 60, "there", hold=0.5)
    assert [round(t, 3) for t, _ in plan.keys] == [0.0, 1.0, 1.5] and plan.marks == {"there": 1.5}
    sag = dict(m3kinematics.SAG)
    m3_sim.grip_plan(0.20, 0.0, 0.0, 0.03, 0.015, 0.6)
    assert m3kinematics.SAG == sag   # the simulated arm plans without it; the real one keeps it
