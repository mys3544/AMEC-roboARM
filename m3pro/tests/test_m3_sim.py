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


def test_the_expert_sends_what_puts_the_real_arm_where_the_plan_has_it():
    import numpy as np
    card = m3_sim.draw_card(np.random.default_rng(0))
    assert card["twin_sag"] == m3_sim.TWIN_SAG
    keys = [(0.0, {1: 90, 2: 40, 3: 30, 4: 40, 5: 90, 6: 30}), (1.0, {1: 90, 2: 30, 3: 30, 4: 40, 5: 90, 6: 30})]
    _, (_, sent) = m3_sim.compensated(keys, card)
    for j in (2, 3, 4):   # the arm with the card sits off its commands by its gain, zero and half its backlash
        way = 0.0 if j != 2 else -1.0
        sits = 90 + (sent[j] + m3kinematics.SAG.get(j, 0.0) - 90) * card["gain"][j] + card["offset_deg"][j]             - way * card["backlash_deg"][j] / 2
        assert sits == pytest.approx(keys[1][1][j] + m3kinematics.SAG.get(j, 0.0))
    assert m3_sim.true_sag()[2] == pytest.approx(m3kinematics.SAG.get(2, 0.0) + m3_sim.TWIN_SAG.get(2, 0.0))

