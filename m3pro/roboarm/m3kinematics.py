"""The M3 Pro arm's geometry, from Yahboom's URDF (yahboom_M3Pro_description/urdf/M3Pro.urdf on
the robot): where the arm's parts are for a joint pose, in base_link (x forward, z up; base_link is
2 cm above the floor). Joint angles are servo degrees, 90 = the URDF's zero, as Yahboom's own
kin_srv converts them.

Provisional: the real arm sits 3-5 degrees lower than this in every pose tried (camcheck,
2026-09-30) -- about 2.5 cm at full reach -- so the guard below keeps margins that cover it, and
the measured geometry (hand-eye calibration) replaces the URDF's numbers.

The camera (the DCW2) is fixed to arm4: joints 2-4 set its tilt; joint 5 only rolls the gripper.
Lowering joint 2 leans the arm forward and tilts the camera down.

SAG: the arm sits lower than it is told, the more so leaning out: the touch test (tools/m3_touch.py,
2026-10-01) put the grip point 16-24 mm short and the look pose's camera 16 mm low and 4.8 deg
steeper than the URDF has them. A J2 4.0 deg and a J3 -8.5 deg off what was commanded explains the
camera to 2 mm and 1.3 deg and the three touches to 0-10 mm forward, 5 sideways, 8-13 up. Fitted to
ruler readings: a working correction, not a measurement (the arm also repeats only to ~1-2 deg).

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
# The Gripping point is Yahboom's tool point -- the camera sits 100 mm behind it and 48 mm above
# (their EndToCamMat), where the object sits between the fingertips; the pads reach a little
# past it. Provisional: the touch test measures it.
FINGERS = 0.015   # m the fingertips reach past the Gripping point

# The chassis, from Yahboom's own model of it (meshes/base_link.STL, 2026-09-30): x -0.153..+0.124,
# y +-0.112 (+-0.12 with the wheels); its top 0.196 over the arm's base, 0.133 on the front deck
# (x 0.08..0.124), where the front lidar's housing adds ~2.5 cm. As steps, each (x from, x to,
# the height the arm keeps above), with 1.5 cm to spare all round: the front step lets the arm
# reach down just in front of the robot, where it picks.
CHASSIS_Y = 0.135   # m either side
# The middle step is the arm's own base (its shoulder joint sits at 0.218): 0.9 cm over it, not 1.5,
# or the folded home pose -- the SAG overstates the droop of an upright, unloaded forearm -- reads 2 mm in.
CHASSIS = ((-0.17, 0.08, 0.205), (0.08, 0.14, 0.175))
# base_link is ON the floor: the URDF's wheel axles are 0.040 above it and the wheels' radius is
# 0.040 (meshes/lwheel1.STL); camcheck's 0.350 m for the camera at the mapping pose against the
# URDF's 0.355 agrees. (Yahboom's launch puts base_link 2 cm up, base_footprint -> base_link.)
FLOOR = 0.0     # m, the floor in base_link
FLOOR_CLEAR = 0.015   # m above the floor the fingertips stay, with the SAG's ~1 cm uncertainty
SAG = {2: 4.0, 3: -8.5}   # degrees the joints really sit off their command (see SAG above)


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
        angle = pose[joint] + SAG.get(joint, 0.0) - 90 if joint != 1 else 90 - pose[joint]   # SAG, SIGNS
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
        for x0, x1, top in CHASSIS:
            if x0 < p[0] < x1 and abs(p[1]) < CHASSIS_Y and p[2] < top:
                return (f"the arm would come down to {100 * (p[2] - FLOOR):.0f} cm above the floor over"
                        f" the chassis (x {p[0]:+.2f} m): keep it above {100 * (top - FLOOR):.0f} cm there")
        if p[2] < FLOOR + FLOOR_CLEAR:
            return (f"the arm would reach {100 * (p[2] - FLOOR):.1f} cm above the floor"
                    f" (x {p[0]:+.2f} m): the fingertips stay {100 * FLOOR_CLEAR:.1f} cm clear of it")
    return None


def grip(pose: dict[int, float]) -> tuple[np.ndarray, float]:
    """Where the gripper's grip point is (base_link, m) and how far its fingers point below
    level (degrees, + = down)."""
    f = frames(pose)
    point, wrist = f["gripping"][:3, 3], f["arm5"][:3, 3]
    along = (point - wrist) / np.linalg.norm(point - wrist)
    return point, -math.degrees(math.asin(float(np.clip(along[2], -1, 1))))


# The arm in its own vertical plane (J1 turns the plane): the shoulder (J2's axis) 2 mm behind
# J1's axis and 0.218 m up, two 82.85 mm links, then 168 mm from J4's axis to the grip point.
SHOULDER_R = CHAIN[1][0][0]
SHOULDER_Z = CHAIN[0][0][2] + CHAIN[1][0][2]
L2, L3 = CHAIN[2][0][2], CHAIN[3][0][2]
L4 = CHAIN[4][0][2] + GRIPPING[0][2]


def _planar(radius: float, z: float, pitch_down: float) -> list[tuple[float, float, float]]:
    """(J2, J3, J4) in servo degrees putting the grip point at `radius` from J1's axis and height
    `z`, pointing `pitch_down` below level -- both elbow branches; the URDF's small offsets left
    out. A link's lean phi is measured from upright towards the front: phi = sum of (90 - J)."""
    phi4 = math.radians(90 + pitch_down)
    wr, wz = radius - L4 * math.sin(phi4), z - L4 * math.cos(phi4)   # J4's axis
    dr, dz = wr - SHOULDER_R, wz - SHOULDER_Z
    d = math.hypot(dr, dz)
    cos_bend = (d * d - L2 * L2 - L3 * L3) / (2 * L2 * L3)
    if not -1 <= cos_bend <= 1:
        return []
    out = []
    for bend in (math.acos(cos_bend), -math.acos(cos_bend)):
        phi2 = math.atan2(dr, dz) - math.atan2(L3 * math.sin(bend), L2 + L3 * math.cos(bend))
        phi3 = phi2 + bend
        out.append(tuple(90 - math.degrees(a) for a in (phi2, phi3 - phi2, phi4 - phi3)))
    return out


def reach(target, pitch_down: float, j5: float = 90, tolerance: float = 0.005) -> dict[int, int] | None:
    """Servo degrees that put the grip point on `target` (base_link, m) with the fingers
    `pitch_down` degrees below level, or None when no pose the guard allows does it -- within
    `tolerance` m and 3 degrees after rounding to whole servo degrees (one degree of J2 moves the
    grip ~4 mm at reach; the arm itself repeats to ~1 cm). J1 turns the arm's plane to the target;
    J2-J4 in closed form in that plane, then a few Gauss-Newton steps on the full geometry."""
    target = np.asarray(target, float)
    base = np.asarray(CHAIN[0][0][:2])
    yaw = math.degrees(math.atan2(target[1] - base[1], target[0] - base[0]))
    j1 = 90 - yaw                                  # J1: see SIGNS
    if not 0 <= j1 <= 180:
        return None
    radius = float(np.hypot(*(target[:2] - base)))

    def error(q):
        point, pitch = grip({1: j1, 2: q[0], 3: q[1], 4: q[2], 5: j5})
        return np.concatenate([(point - target) / 0.01, [(pitch - pitch_down) / 2.0]])   # cm, 2 deg

    best = None
    for guess in _planar(radius, float(target[2]), pitch_down):
        q = np.array(guess)
        for _ in range(6):   # the URDF's small offsets: a numerical Jacobian, a few steps
            e = error(q)
            J = np.column_stack([(error(q + d) - e) / 0.01 for d in np.eye(3) * 0.01])
            q = q + np.linalg.lstsq(J, -e, rcond=None)[0]
        if not all(0 <= v <= 180 for v in q):
            continue
        pose = {1: round(j1), 2: round(q[0]), 3: round(q[1]), 4: round(q[2]), 5: round(j5)}
        point, pitch = grip(pose)
        miss = float(np.linalg.norm(point - target))
        good = miss <= tolerance and abs(pitch - pitch_down) <= 3 and problem(pose) is None
        if good and (best is None or miss < best[0]):
            best = (miss, pose)
    return None if best is None else best[1]
