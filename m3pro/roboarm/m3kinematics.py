"""The M3 Pro arm's geometry, from Yahboom's URDF (yahboom_M3Pro_description/urdf/M3Pro.urdf on
the robot): where the arm's parts are for a joint pose, in base_link (x forward, z up; base_link is
2 cm above the floor). Joint angles are servo degrees, 90 = the URDF's zero, as Yahboom's own
kin_srv converts them.

Provisional: the real arm sits 3-5 degrees lower than this in every pose tried (camcheck,
2026-09-30) -- about 2.5 cm at full reach -- so the guard below keeps margins that cover it, and
the measured geometry (hand-eye calibration) replaces the URDF's numbers.

The camera (the DCW2) is fixed to arm4: joints 2-4 set its tilt; joint 5 only rolls the gripper.
Lowering joint 2 leans the arm forward and tilts the camera down.

SIGNS: J1 turns the other way than the URDF's axis -- a bigger J1 turns the arm to the right (-y).
Found by the hand-eye pictures (2026-09-30): through the URDF's sign the board seen from J1 76..104
moved 0.37 m with J1; mirrored, it stayed within 8 mm. Yahboom's kin_srv has the URDF's sign.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

# (origin xyz, origin rpy, joint axis) of arm1..arm5, each in its parent's frame
CHAIN = (
    ((0.033617, -1.2421e-05, 0.185), (6.3751e-05, 0, 0.028395), (0, 0, 1)),
    ((-0.0021522, 6.1185e-05, 0.033), (-6.3725e-05, 0, -0.028422), (0, -1, -6.3725e-05)),
    ((0, 0, 0.08285), (6.3702e-05, 0, 0.027102), (-0.027098, -0.99963, 0)),
    ((0, 0, 0.08285), (0, 0, -0.027075), (-2.6392e-05, -1, 0)),
    ((0.00057305, 0, 0.074815), (0, 0, 0), (0, 0, 1)),
)
CAMERA = ((-0.0483883323542816, 0.00245180097510241, 0.0671900000306381),
          (1.84287098025439, -1.57079631189374, -1.81447538896515))   # arm4 -> DCW2
GRIPPING = ((-0.00062499, 0.00023987, 0.093235), (-0.041648, -1.5708, 0.041619))   # arm5 ->
FINGERS = 0.03   # m the fingers reach past the Gripping point (the URDF's finger links)

# The chassis and what stands on it, as a box in base_link: the two lidars sit on its corners at
# z 0.125 in housings ~4 cm tall. Any part of the arm over this footprint stays above its top.
# Estimated from the URDF's wheels and lidars with ~3 cm to spare -- to measure: the box also
# keeps the arm from reaching down just in front of the robot, which floor picks will need.
CHASSIS_X, CHASSIS_Y, CHASSIS_TOP = (-0.20, 0.19), 0.16, 0.21   # m
FLOOR = -0.02     # m, base_link's floor
FLOOR_CLEAR = 0.015   # m above the floor the fingertips stay (the URDF's sag: ~2.5 cm at reach)


def _rot(axis, q):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * k + (1 - math.cos(q)) * k @ k


def _rpy(r, p, y):
    return _rot((0, 0, 1), y) @ _rot((0, 1, 0), p) @ _rot((1, 0, 0), r)


def _frame(xyz, r):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, xyz
    return m


def frames(pose: dict[int, float]) -> dict[str, np.ndarray]:
    """4 x 4 transforms base_link -> each part, for servo degrees {1..5: angle} (6 ignored)."""
    out, m = {}, np.eye(4)
    for joint, (xyz, rpy, axis) in enumerate(CHAIN, 1):
        angle = pose[joint] - 90 if joint != 1 else 90 - pose[joint]   # J1: see SIGNS
        m = m @ _frame(xyz, _rpy(*rpy)) @ _frame((0, 0, 0), _rot(axis, math.radians(angle)))
        out[f"arm{joint}"] = m
    out["camera"] = out["arm4"] @ _frame(CAMERA[0], _rpy(*CAMERA[1]))
    out["gripping"] = out["arm5"] @ _frame(GRIPPING[0], _rpy(*GRIPPING[1]))
    return out


def camera(pose: dict[int, float]) -> tuple[np.ndarray, float]:
    """Where the camera is (base_link, m) and how far it looks down (degrees, + = down): the
    DCW2 link looks along its x axis."""
    f = frames(pose)["camera"]
    return f[:3, 3], -math.degrees(math.asin(f[2, 0]))


def outline(pose: dict[int, float]) -> list[np.ndarray]:
    """Points along the arm, shoulder to fingertips, every ~2 cm: what the guard checks."""
    f = frames(pose)
    joints = [f[f"arm{j}"][:3, 3] for j in (2, 3, 4, 5)] + [f["gripping"][:3, 3]]
    along = joints[-1] - joints[-2]   # the fingers go on the way the wrist reaches the Gripping point
    joints.append(joints[-1] + FINGERS * along / np.linalg.norm(along))
    joints.append(f["camera"][:3, 3])
    points = []
    for a, b in itertools.pairwise(joints[:-1]):
        n = max(1, int(np.linalg.norm(b - a) / 0.02))
        points += [a + (b - a) * k / n for k in range(n + 1)]
    return points + [joints[-1]]


def problem(pose: dict[int, float]) -> str | None:
    """Why the pose would put the arm into the chassis, the lidars or the floor; None if it
    would not."""
    for p in outline(pose):
        over = CHASSIS_X[0] < p[0] < CHASSIS_X[1] and abs(p[1]) < CHASSIS_Y
        if over and p[2] < CHASSIS_TOP:
            return (f"the arm would come down to {100 * (p[2] - FLOOR):.0f} cm above the floor over"
                    f" the chassis (x {p[0]:+.2f} m): keep it above {100 * (CHASSIS_TOP - FLOOR):.0f} cm there")
        if p[2] < FLOOR + FLOOR_CLEAR:
            return (f"the arm would reach {100 * (p[2] - FLOOR):.1f} cm above the floor"
                    f" (x {p[0]:+.2f} m): the fingertips stay {100 * FLOOR_CLEAR:.1f} cm clear of it")
    return None
