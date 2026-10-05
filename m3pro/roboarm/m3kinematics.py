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
J5 turns the URDF's way, seen (2026-10-01): the gripper pointing down, J5 90 -> 130 swings the
finger on the robot's left forward. (A mirrored J5, read off a missed pick's pictures that
morning, put the fingers along a 60 mm block instead of across it: three misses.)
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import cv2
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
# Yahboom's table, object width (mm) -> the gripper angle (J6) that just holds it; 0 is widest open
GRIPPER_WIDTH = {0: 180, 20: 152, 30: 134, 40: 115, 50: 95, 60: 57}
FINGER_SIDE = 0.008   # m a finger's outer edge past the opening's edge (its thickness, and a margin)

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
FLOOR_CLEAR = 0.015   # m above the floor the fingertips stay, with the SAG's ~1 cm uncertainty --
                      # for moves planned on this model alone; a move planned on the floor the
                      # depth just measured (the last descent of a place or a pick) may ask less
MIN_FLOOR_CLEAR = 0.002   # m: never less, whoever asks
SAG = {2: 4.0, 3: -8.5}   # degrees the joints really sit off their command (see SAG above)
MIRRORED = (1,)           # joints that turn against the URDF's axis (see SIGNS above)

# The steps above keep a line through the joints out of the chassis; the arm is thicker than that:
# its parts reach up to 5 cm past the line, the DCW2 4.5 cm to either side of it. 2026-10-02: the
# arm had hit the front-right lidar (the user), and poses the steps allowed put the camera into it
# -- reaching front-right, the camera rides over that corner. So the arm's own shell (Yahboom's
# meshes, roboarm/m3body.npz from tools/m3_body_model.py) is kept off what sticks out into its
# space: the lidars by LIDAR_ROOM, the body in front of BODY_FROM_X by BODY_ROOM. Behind it the
# folded arm rests ~1 mm over the body by design (home): the steps alone guard that.
# The lidars, measured: no part of Yahboom's model. (name, x from-to, y from-to, top), base_link m;
# the front-right one reaches 14.5 cm from the arm's axis at its outer corner, 13.5 cm tall (the
# user, 2026-10-02); the rear-left one taken alike.
LIDARS = (("front-right lidar", (0.060, 0.124), (-0.112, -0.070), 0.135),
          ("rear-left lidar", (-0.140, -0.090), (0.090, 0.112), 0.135))
LIDAR_ROOM = 0.010   # m the arm keeps from a lidar, every way (the user: 2.5 cm after the hit; 1.5 cm
                     # from 2026-10-05, after watching the arm 2.0-2.4 cm off it down a pick's path;
                     # 1.0 the same day, with HAND_BACK as seen, for the 40 mm cube just by it)
HAND_BACK = 0.028    # m the hand's shell reaches along its own x -- the finger motor's housing: 4.2 cm by
                     # Yahboom's mesh, 1.4 less as the user saw it (2026-10-05: hovering over a 40 mm cube
                     # by the right lidar, the housing ~1 cm in front of the lidar's face, where the mesh
                     # had it 0.4 cm over it)
BODY_ROOM = 0.0      # m from the rest: the shell must not go in; the steps keep their own margins
BODY_FROM_X = 0.0    # m: the body in front of this, less the arm's own turntable (ARM_COLUMN round its axis)
ARM_COLUMN = 0.05
SHELL_REACH = 0.06   # m the shell can be from the outline's nearest point: 5.0 cm over 3000 poses


def _rot(axis, q):
    a = np.asarray(axis, float)
    return cv2.Rodrigues(a / np.linalg.norm(a) * q)[0]


def _rpy(r, p, y):
    return _rot((0, 0, 1), y) @ _rot((0, 1, 0), p) @ _rot((1, 0, 0), r)


def _frame(xyz, r):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, xyz
    return m


def _axis(axis) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return k, k @ k


# What does not change with the pose, worked out once: a grasp plan runs frames() thousands of times,
# and building these again each time was most of its time (2026-10-01: 0.5 ms a call, 2.7 s a plan
# on the robot).
_ORIGINS = [_frame(xyz, _rpy(*rpy)) for xyz, rpy, _ in CHAIN]
_AXES = [_axis(axis) for *_, axis in CHAIN]
_TO_CAMERA = _frame(CAMERA[0], _rpy(*CAMERA[1]))
_TO_GRIPPING = _frame(GRIPPING[0], _rpy(*GRIPPING[1]))

# The shell and what it keeps clear of (see LIDARS): each cell of the body's height map holds the
# height a shell point over it stays above -- its own top grown by the room, sideways and up.
_MODEL = np.load(Path(__file__).with_name("m3body.npz"))
_SHELL = tuple((part, frame, _MODEL[f"shell_{frame}"].astype(float)) for part, frame in (
    ("forearm", "arm3"), ("wrist", "arm4"), ("hand", "arm5"), ("camera", "camera"), ("gripper", "gripping")))
_SHELL[2][2][:, 0] = np.minimum(_SHELL[2][2][:, 0], HAND_BACK)   # the hand: its housing as seen
_BOUNDS = tuple(np.array(list(itertools.product(*zip(points.min(0), points.max(0), strict=True))))
                for _part, _frame, points in _SHELL)   # each part's box's 8 corners, in its frame
_CELL = float(_MODEL["body_cell"])
_X0, _Y0 = (float(v) for v in _MODEL["body_origin"])
_TOPS = np.where(_MODEL["body_mm"] >= 0, _MODEL["body_mm"] / 1000.0, -np.inf)


def _grow(h: np.ndarray, r: float, square: bool = False) -> np.ndarray:
    """Each cell: the highest within `r` of it (a disc; a square when `square`, cheaper and wider)."""
    n = math.ceil(r / _CELL - 1e-9)
    d = np.arange(-n, n + 1)
    disc = square | ((d[:, None] ** 2 + d[None, :] ** 2) * _CELL * _CELL <= r * r + 1e-12)
    return cv2.dilate(h, disc.astype(np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=-np.inf)


def _keep_out() -> tuple[list[tuple[str, float, np.ndarray]], np.ndarray, np.ndarray]:
    gx, gy = np.meshgrid(_X0 + (np.arange(_TOPS.shape[0]) + 0.5) * _CELL,
                         _Y0 + (np.arange(_TOPS.shape[1]) + 0.5) * _CELL, indexing="ij")
    parts, lidar = [], np.zeros(_TOPS.shape, bool)
    for name, (x0, x1), (y0, y1), top in LIDARS:
        inside = (gx > x0) & (gx < x1) & (gy > y0) & (gy < y1)
        lidar |= inside
        parts.append((name, LIDAR_ROOM, _grow(np.where(inside, np.maximum(_TOPS, top), -np.inf), LIDAR_ROOM) + LIDAR_ROOM))
    column = np.hypot(gx - CHAIN[0][0][0], gy - CHAIN[0][0][1]) < ARM_COLUMN
    front = np.where((gx >= BODY_FROM_X) & ~lidar & ~column, _TOPS, -np.inf)
    parts.append(("robot's front", BODY_ROOM, _grow(front, BODY_ROOM) + BODY_ROOM))
    limit = np.maximum.reduce([grid for *_, grid in parts])
    # where the outline must be for any shell point to be in the way: SHELL_REACH, and a cell more
    near = _grow(limit, SHELL_REACH + _CELL, square=True) + SHELL_REACH
    return parts, limit, near


_PARTS, _LIMIT, _NEAR = _keep_out()


def _clear_above(corners: np.ndarray) -> bool:
    """A part's box (its corners, base_link) wholly over the highest keep-out under it: all of it clear."""
    ix0, iy0 = np.floor((corners[:, :2].min(0) - (_X0, _Y0)) / _CELL).astype(int)
    ix1, iy1 = np.floor((corners[:, :2].max(0) - (_X0, _Y0)) / _CELL).astype(int)
    under = _LIMIT[max(ix0, 0):max(ix1 + 1, 0), max(iy0, 0):max(iy1 + 1, 0)]
    return under.size == 0 or corners[:, 2].min() >= under.max()


def _over(grid: np.ndarray, points: np.ndarray) -> np.ndarray:
    """The grid's value under each point (-inf off the map)."""
    ix = np.floor((points[:, 0] - _X0) / _CELL).astype(int)
    iy = np.floor((points[:, 1] - _Y0) / _CELL).astype(int)
    ok = (ix >= 0) & (ix < grid.shape[0]) & (iy >= 0) & (iy < grid.shape[1])
    out = np.full(len(points), -np.inf)
    out[ok] = grid[ix[ok], iy[ok]]
    return out


def frames(pose: dict[int, float]) -> dict[str, np.ndarray]:
    """4 x 4 transforms base_link -> each part, for servo degrees {1..5: angle} (6 ignored)."""
    out, m = {}, np.eye(4)
    turn = np.eye(4)
    for joint, (origin, (k, kk)) in enumerate(zip(_ORIGINS, _AXES, strict=True), 1):
        angle = math.radians(90 - pose[joint] if joint in MIRRORED else pose[joint] + SAG.get(joint, 0.0) - 90)
        turn[:3, :3] = math.sin(angle) * k + (1 - math.cos(angle)) * kk   # Rodrigues, less the identity
        turn[0, 0] += 1.0
        turn[1, 1] += 1.0
        turn[2, 2] += 1.0
        m = m @ origin @ turn
        out[f"arm{joint}"] = m
    out["camera"] = out["arm4"] @ _TO_CAMERA
    out["gripping"] = out["arm5"] @ _TO_GRIPPING
    return out


OPTICAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # the DCW2 link -> its optical frame
                                                                  # (x right, y down, z ahead)


def camera_optical() -> np.ndarray:
    """arm4 -> the camera's optical frame, as the URDF has it: the one in use (the hand-eye fit of
    the mount was degenerate, 2026-09-30; the depth's floor levels whatever it gets wrong)."""
    m = np.eye(4)
    m[:3, :3] = _rpy(*CAMERA[1]) @ OPTICAL
    m[:3, 3] = CAMERA[0]
    return m


def finger_span(angle: float) -> float:
    """m between the fingers at gripper angle `angle` (J6): Yahboom's table inverted, its last
    stretch carried on past 60 mm (GRIPPER_OPEN, 30, is ~67)."""
    table = sorted(GRIPPER_WIDTH.items())
    (w0, a0), (w1, a1) = table[-2], table[-1]
    if angle < a1:
        return max(0.0, (w1 + (angle - a1) * (w1 - w0) / (a1 - a0)) / 1000)
    return float(np.interp(angle, [a for _w, a in table[::-1]], [w for w, _a in table[::-1]])) / 1000


def outline(pose: dict[int, float], f: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Points along the arm, shoulder to fingertips, every ~2 cm, and the camera: what the guard
    checks (n x 3, in that order). With the gripper's angle (6) in the pose, both fingers too --
    either side of the hand by half the opening, along the line they close on: with the wrist
    turned and the hand not pointing straight down that line tilts, one finger lower than the
    middle by up to half the opening (2026-10-02: J5 45 at 60 deg, a finger 4 mm into the table
    with the middle 8.5 mm over it; it touched and pushed the robot). `f`: frames(pose), if had."""
    f = frames(pose) if f is None else f
    joints = [f[f"arm{j}"][:3, 3] for j in (2, 3, 4, 5)] + [f["gripping"][:3, 3]]
    along = joints[-1] - joints[-2]   # the fingers go on the way the wrist reaches the Gripping point
    joints.append(joints[-1] + FINGERS * along / np.linalg.norm(along))
    stretches = []
    for a, b in itertools.pairwise(joints):
        n = max(1, int(np.linalg.norm(b - a) / 0.02))
        stretches.append(a + (b - a) * (np.arange(n + 1)[:, None] / n))
    if pose.get(6) is not None:
        side = (finger_span(pose[6]) / 2 + FINGER_SIDE) * f["arm5"][:3, 1]   # the fingers close along arm5's y
        root, tip = joints[-2], joints[-1]
        stretches.append(np.array([root + side, tip + side, root - side, tip - side]))
    return np.vstack([*stretches, f["camera"][:3, 3][None]])


def problem(pose: dict[int, float], floor_clear: float = FLOOR_CLEAR) -> str | None:
    """Why the pose would put the arm into the chassis or the floor, or its shell near a lidar or
    into the robot's front; None if it would not. The first point along the arm that is in the
    way says why."""
    f = frames(pose)
    p = outline(pose, f)
    step = np.full(len(p), -1)
    for i, (x0, x1, top) in enumerate(CHASSIS):
        over = (p[:, 0] > x0) & (p[:, 0] < x1) & (np.abs(p[:, 1]) < CHASSIS_Y) & (p[:, 2] < top)
        step = np.where(over & (step < 0), i, step)
    low = p[:, 2] < FLOOR + max(floor_clear, MIN_FLOOR_CLEAR)
    bad = (step >= 0) | low
    if not bad.any():
        return too_near(f, p)
    first = int(np.argmax(bad))
    x, z = p[first, 0], p[first, 2]
    if step[first] >= 0:
        top = CHASSIS[step[first]][2]
        return (f"the arm would come down to {100 * (z - FLOOR):.0f} cm above the floor over"
                f" the chassis (x {x:+.2f} m): keep it above {100 * (top - FLOOR):.0f} cm there")
    return (f"the arm would reach {100 * (z - FLOOR):.1f} cm above the floor"
            f" (x {x:+.2f} m): the fingertips stay {100 * floor_clear:.1f} cm clear of it")


def too_near(f: dict[str, np.ndarray], line: np.ndarray) -> str | None:
    """Why the arm's shell (frames `f`) or its outline (`line`) would come nearer a lidar than
    LIDAR_ROOM, or into the robot's front; None if not. Most poses are nowhere near: the outline
    tells that, the shell's points are only looked at when it is within SHELL_REACH."""
    if np.all(line[:, 2] >= _over(_NEAR, line)):
        return None
    shells = ((part, points @ f[frame][:3, :3].T + f[frame][:3, 3]) for (part, frame, points), corners in zip(_SHELL, _BOUNDS, strict=True)
              if not _clear_above(corners @ f[frame][:3, :3].T + f[frame][:3, 3]))
    for part, points in itertools.chain(shells, [("arm", line)]):
        low = points[:, 2] < _over(_LIMIT, points)
        if not low.any():
            continue
        x, y, z = points[np.argmax(low)]
        what, room = next((name, room) for name, room, grid in _PARTS if z < _over(grid, np.array([[x, y, z]]))[0])
        near = f"within {100 * room:.1f} cm of" if room else "into"
        return f"the arm's {part} would come {near} the {what} (x {x:+.2f} m, y {y:+.2f} m, {100 * z:.0f} cm up)"
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


def reach(target, pitch_down: float, j5: float = 90, tolerance: float = 0.005,
          floor_clear: float = FLOOR_CLEAR) -> dict[int, int] | None:
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
        if not all(-15 <= v <= 195 for v in guess):   # the refinement moves a few degrees: no use
            continue
        q = np.array(guess)
        for _ in range(6):   # the URDF's small offsets: a numerical Jacobian, a few steps
            e = error(q)
            if np.abs(e).max() < 0.01:   # 0.1 mm and 0.02 deg off: there
                break
            J = np.column_stack([(error(q + d) - e) / 0.01 for d in np.eye(3) * 0.01])
            q = q + np.linalg.lstsq(J, -e, rcond=None)[0]
        if not all(0 <= v <= 180 for v in q):
            continue
        pose = {1: round(j1), 2: round(q[0]), 3: round(q[1]), 4: round(q[2]), 5: round(j5)}
        point, pitch = grip(pose)
        miss = float(np.linalg.norm(point - target))
        good = miss <= tolerance and abs(pitch - pitch_down) <= 3 and problem(pose, floor_clear) is None
        if good and (best is None or miss < best[0]):
            best = (miss, pose)
    return None if best is None else best[1]
