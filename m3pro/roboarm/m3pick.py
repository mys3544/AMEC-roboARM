"""The M3's look-again pick: a kit cube or block found by the depth on whatever floor, picked up and
taken to the drop-off -- one, or every one in reach (clean_up).

  1. the wide look (LOOK), the gripper opening on the way (OPEN_MM: a 30 mm cube fits turned any
     way, its diagonal is 42 mm): the depth's floor levels the camera, m3vision.objects finds the
     kit on it -- no board, no tags; the nearest one in reach (or the one nearest `at`) is the target;
  2. the pre-grasp pose: the fingertips BACK short along the approach line;
  3. the close look: the thing found again from there, the aim corrected (the arm repeats to
     ~1-2 deg) and its turn read -- the wrist (J5) turned to square the fingers with it;
  4. down the line to the grip point in one glide -- the thing's middle (or higher, up to its top,
     at the far edge), at least GRIP_Z_MIN up -- the fingers close to its width, back up the line;
  5. the drop-off, the X3 Plus's (its roboarm/config.py DROP_POSE): the base a full 90 deg to the right
     (J1 180), the fingertips DROP_OUT out and DROP_UP above the floor -- the thing falls the last
     bit -- and a random 0..DROP_SPREAD further out along that line so they do not land on each
     other; up over the chassis and turned there, never swung across it low with something in hand;
  6. the wide look again: nothing where it stood = picked -- and it is the next pick's wide look.

The flow -- the jobs, the drop-off box, driving back, what the console watches -- is both robots'
(roboshared.task.Task, roboshared.box, since 2026-10-07); Picker here is the M3's body for it: steps
1-4, its carries and its box looks.

Every move is one smooth command, or a glide through several (M3Arm.move_to / follow), quicker in
the open and slower near the floor -- the speeds and pauses are SETTINGS, changed from the console;
each step's time goes to roboshared.metrics. Where the base drove up to something out of reach, it
drives back to where the job began before each drop: the drop-off box stays there.

The arm picks off the floor 16-27 cm ahead of base_link (3.5-14 cm before the bumper; nearer, the
hand cannot point down steeply enough), less far aside: further out the shoulder (J2) would have to
lean past its end at 0 (2026-10-01).

ROS-free: the arm is whatever `call(method, *args, **kwargs)` reaches -- roboarm.web.bridge.Bridge
in the bridge itself (tools/m3bridge.py serves /pick/*) -- and the camera whatever `snapshot()`
returns: (colour BGR, depth mm aligned to it, K, the believed joint pose). Every step is timed
(roboshared.metrics: one run per job).
"""

from __future__ import annotations

import itertools
import json
import math
import time

import cv2
import numpy as np
from roboshared import metrics
from roboshared.box import (  # noqa: F401 -- m3pick's
    BOX_DROP_OVER,
    BOX_FRONT_MARGIN,
    BOX_INSIDE,
    BOX_LID_MARGIN,
    BOX_MARGIN,
    BOX_MARKER,
    BOX_WALL,  # own names for these
    box_from_marker,
    box_spots,
    in_box,
)
from roboshared.task import (  # noqa: F401
    DRIVE_GAP,
    MATCH_M,
    PART_M,
    NothingToPick,
    Refused,
    Stopped,
    Task,
    describe,
    footprint_gap,
    name,
    refusable,
)

from roboarm import m3aim, m3arm, m3kinematics, m3kit, m3policy, m3scan, m3teach, m3vision

LOOK = {1: 90, 2: 105, 3: 10, 4: 0, 5: 90}   # the wide look: the floor from the bumper (12 cm) to 45 cm
                                             # ahead, +-17 cm aside at the near edge; a 40 mm top there is
                                             # 22 cm off, the depth's near end ~19 (until 2026-10-01 J3 25:
                                             # from 20 cm, and the near cubes cut by the picture's edge)
FOLDED = {1: 90, 2: 180, 3: 5, 4: 0, 5: 90}   # home, without the gripper: it keeps what it holds
CARRIES = ((130, 40, 40), (150, 30, 30), (160, 10, 10), (180, 5, 0))   # (J2, J3, J4) to carry at, lowest
                                                                       # first: the last is FOLDED
BACK = 0.12                 # m the fingertips stop short along the approach: the camera ~24 cm off
BACKS = (BACK, 0.10, 0.08)  # and shorter when that pre-grasp's open fingers would come over the front deck
                            # (near the bumper, 2026-10-02); 0.08 still keeps the camera ~20 cm off
GUARD_SLACK = 0.0015        # m the arm's own guard lets the fingertips come lower than the plan has them:
                            # the wrist's turn after planning moves them up to ~1 mm (2026-10-02: a plan at
                            # 8.0 mm, J5 35, refused at 7.9)
FLOOR_CLEAR_PLACE = 0.008   # m the fingertips stay above the floor on the descent (the floor just measured):
                            # the "fingertips_mm" setting's default. 8 for a day (2026-10-02, they had touched
                            # the table) -- it was the turned wrist's lower finger the model did not have;
                            # with both fingers in it, back to 3 (the user's call): blocks turned far, further
                            # out, had no square grip at 8. 8 again from 2026-10-05 (the user's: they still
                            # touched; 74 of 112 cube grips planned at 3 mm and at 8 alike). 5 for an hour on
                            # 2026-10-21 (the user's, for more grips: two blocks had none at 8 or 6, both at 5)
                            # -- the fingers touched the table by small cubes and shoved the robot. So 8 first,
FLOOR_CLEAR_LEAST = 0.005   # and this only where nothing plans at the first: Picker.plan (the user's, that day)
OPEN_MM = 55        # the fingers apart at the wide look; at the pre-grasp opened_for() the target
SPARE_MM = 25       # the opening over the target's width (2026-10-01: a 40 mm cube -- 57 mm across
                    # its diagonal -- missed at 55; it opens to ~67 now, the gripper's widest)
# The careful grip, where no other plans (2026-10-05, the user's: a block lying across in front of the
# left wheel, "we have to pick things from such position"): opened only CAREFUL_SPARE_MM over the
# thing and aimed further out along the line the fingers close on, CAREFUL_SHIFTS tried in turn --
# the near finger 4.5 (2.5) mm off the thing and 1.0 (1.2) cm further from the robot; closing, it
# pushes the thing that much out. That block, 11 mm in front of the bumper: 3 mm planned turned one
# way, whole servo degrees left the descent 1 mm short the other; 5 mm planned both.
CAREFUL_SPARE_MM, CAREFUL_SHIFTS = 15, (0.003, 0.005)
RELEASE_SPARE_MM = 15   # let go opened this much over the thing, not wide: the fingers stay off the box's
                        # walls and lid (2026-10-05: the box hit at a drop by its lid)
FINGER_M = 0.008    # m a fingertip reaches past the opening's edge (its own thickness and a margin)
DROP_OUT, DROP_UP, DROP_SPREAD = 0.189, 0.110, 0.100   # m: the X3 Plus's drop-off (config.DROP_POSE)
LINE_OVER = 0.02   # m what is held is let go over the table on the line, its grip maybe slipped a little --
                   # where the arm reaches that low: 19-21 cm out; further, the lowest of LINE_RELEASES
LINE_RELEASES = (0.05, 0.06, 0.08, DROP_UP)   # m the grip point's height, tried in turn
GRIP_Z_MIN = 0.018   # m: the grip point's lowest; the fingertips reach ~1.4 cm below it at 70 deg
PITCHES = (70, 65, 75, 60, 80, 55, 50, 85, 90, 45)   # the approaches tried first, then every whole degree
FINE = tuple(p for p in range(90, 39, -1) if p not in PITCHES)   # between: at the far edge the window is a
                                                                # few degrees wide; near the bumper only
                                                                # straight down reaches (2026-10-01)
LIFTS = (0.0, 0.003, 0.006, 0.009, 0.012, 0.015)   # m the grip may go higher on the thing, up to its top,
                                                   # when its middle is out of reach: the fingertips
                                                   # reach ~1.2 cm past the grip point, so even at the
                                                   # top they hold its upper 1.2 cm
# What the console may change while the robot runs (POST /pick/settings), kept in out/pick/settings.json:
# key -> (what it is, unit, lowest, highest, step, default). Arm speeds are deg/s of the joint with the
# furthest to go, all times "speed" %. 2026-10-01: 120/80/40 deg/s shoved the robot ~2 cm on its mecanum
# rollers -- the swing to the drop-off with a cube in hand -- and the descent arrived swaying (two
# misses); 90/60/25 picked 2 of 2. A 0.35 s settle before the wide look after the swing back from the
# drop-off read everything ~3 cm off; 0.35 before the close look did no harm.
SETTINGS = {
    "speed": ("arm speed, every move", "%", 30, 150, 5, 100),
    "fingertips_mm": ("fingertips above the table at the grip, at least (the lower finger)", "mm", 2, 25, 1,
                      round(FLOOR_CLEAR_PLACE * 1000)),
    "fingertips_least_mm": ("... where no grip keeps that: lower, for that one only", "mm", 2, 25, 1,
                            round(FLOOR_CLEAR_LEAST * 1000)),
    "free_dps": ("moves in the open: the look, the drop-off", "deg/s", 20, 150, 5, 90),
    "near_dps": ("to the pre-grasp, over the things", "deg/s", 10, 120, 5, 60),
    "descend_dps": ("down the line to the grip", "deg/s", 5, 60, 1, 25),
    "lift_dps": ("back up it, holding", "deg/s", 10, 120, 5, 60),
    "gripper_dps": ("the gripper", "deg/s", 30, 300, 10, 150),
    "settle_wide_s": ("still before the wide look", "s", 0.0, 2.0, 0.05, 0.6),
    "settle_close_s": ("still before the close look", "s", 0.0, 2.0, 0.05, 0.35),
    "grip_s": ("held shut before lifting", "s", 0.0, 1.5, 0.05, 0.3),
    "release_s": ("open at the drop-off before going back", "s", 0.0, 1.0, 0.05, 0.2),
    "drive_cm_s": ("driving speed", "cm/s", 1.0, 10.0, 0.5, 5.0),
    "driven_s": ("still after driving, before the look", "s", 0.0, 2.0, 0.05, 0.4),
    "nudge_cm": ("driving per move, at most", "cm", 1, 15, 1, 10),
    "right_cm": ("driving right, at most (and 3 cm short of the drop-off box)", "cm", 0, 15, 1, 10),
    "carry_cm": ("carried round this high: the bottom of what is held over the floor", "cm", 6, 25, 1, 15),
    # the aim past where the camera saw a thing is taught (roboarm/m3aim.py); these trim it by hand
    "trim_out_mm": ("the aim trimmed: further out", "mm", -30, 30, 1, 0),
    "trim_left_mm": ("the aim trimmed: further left", "mm", -30, 30, 1, 0),
}
                 # (a kit thing's length): from the side looks a lying block cut short read as a cube
                 # 3-4 cm off its middle (2026-10-05, the tests)
APPROACH_X = 0.21    # m ahead of base_link a thing out of reach is driven to (and to y 0): mid-reach
BUMPER_X = 0.124     # m: the chassis' front edge (m3kinematics), and what the drive keeps clear of it
BUMPER_GAP = 0.04
# 2026-10-02: the arm keeps LIDAR_ROOM off the front-right lidar (m3kinematics), and nothing just in
# front of it is picked from where it is -- the base slides right till it is (the user's choice),
# tried in SIDESTEP steps; a message works out up to SIDESTEP_MOST. The base's own footprint
# (base_link: the body, the wheels, the lidar) keeps roboshared.task.DRIVE_GAP from the drop-off box
# -- its inside and BOX_WALL round it -- and from the line on the right it has dropped onto.
SIDESTEP, SIDESTEP_MOST = 0.02, 0.12
CLOSER_MOST = 0.10   # m: why_not says how much closer the robot would have to come, up to this
ROBOT_X, ROBOT_Y = (-0.16, 0.13), 0.12








# ------------------------------------------------------------------------------------- geometry --



def opened_for(size_m: float, spare_mm: float = SPARE_MM) -> int:
    """The gripper angle that opens the fingers `spare_mm` wider than a thing `size_m` across --
    beyond Yahboom's table (60 mm), its widest, GRIPPER_OPEN (~67 mm)."""
    width = size_m * 1000 + spare_mm
    return m3arm.gripper_angle(width) if width <= max(m3arm.GRIPPER_WIDTH) else m3arm.GRIPPER_OPEN


def finger_clearance(pose: dict, opening: float, others: list[dict]) -> float:
    """m from the two open fingertips, at `pose`, to the nearest of `others` (their footprints as
    circles round their corners); inf with nothing about."""
    f = m3kinematics.frames(pose)
    grip, closing = f["gripping"][:3, 3][:2], f["arm5"][:3, 1][:2]
    closing = closing / max(float(np.linalg.norm(closing)), 1e-9)
    tips = [grip + side * (opening / 2 + FINGER_M) * closing for side in (1, -1)]
    gaps = [float(np.linalg.norm(t - o["centre"][:2])) - 0.5 * float(np.hypot(*o["footprint"]))
            for t in tips for o in others]
    return min(gaps, default=float("inf"))


def wrist_squares(pose: dict, yaw: float, period: float = 90, opening: float = OPEN_MM / 1000,
                  others: list[dict] = ()) -> list[int]:
    """Every J5 that squares the fingers with the object (a cube's two, 90 deg apart; a lying
    block's one), best first: the fingers furthest from `others`, then the smaller turn. They close
    along the wrist's y axis (the finger links sit either side of it, URDF), at J5 90 square to the
    arm's own plane -- with the arm turned to reach a cube that sits square to the robot, 25 deg off
    its faces: they caught its corners and shoved it (2026-10-01). `yaw` is a line the fingers may
    close along, the same every `period` deg (90 a cube, 180 across a lying block); of the ways that
    square them, the one whose open fingers keep furthest from `others` first (2026-10-01: fingers
    opening left and right of a cube hit a 40 mm one 6.5 cm beside it)."""
    top = m3arm.SAFE_LIMITS[5][1]   # J5's end (m3arm): past 180 a square grip turned over is one more --
    # the finger motor's housing then on the hand's other side (2026-10-05, the user's: away from the lidar)
    # J5 turns the hand about its own z: the closing line for every J5 from the hand's x and y at J5 90 (one
    # frames() a grip, not one a degree -- 2026-10-05: 131k of them, 10 s, in a plan that failed)
    hand = m3kinematics.frames({**pose, 5: 90})["arm5"][:3, :2]
    turn = np.radians(np.arange(top + 1) - 90.0)
    closing = -np.sin(turn)[:, None] * hand[:, 0] + np.cos(turn)[:, None] * hand[:, 1]
    mismatch = np.abs((np.degrees(np.arctan2(closing[:, 1], closing[:, 0])) - yaw + period / 2) % period - period / 2)
    best = min(range(top + 1), key=lambda j5: (round(mismatch[j5]), abs(j5 - 90)))
    squared = [j5 for j5 in {best + 90 * n for n in range(-3, 4)}
               if 0 <= j5 <= top and round(mismatch[j5]) <= round(mismatch[best]) + 1]
    return sorted(squared, key=lambda j5: (-min(finger_clearance({**pose, 5: j5}, opening, others), 0.03)
                                           if others else 0.0, abs(j5 - 90)))


def off_square(pose: dict, yaw: float, period: float = 90) -> float:
    """deg the fingers at `pose` close off the line `yaw` (the same every `period` deg)."""
    y = m3kinematics.frames(pose)["arm5"][:3, 1]
    return abs((math.degrees(math.atan2(y[1], y[0])) - yaw + period / 2) % period - period / 2)


WRIST_SLACK = {90: 20, 180: 12}   # deg the fingers may close off square, by the thing's period: a 30 mm
                                  # cube 20 deg off is 40 mm across, a 30 x 60 block 12 deg off 42 --
                                  # both inside a 55 mm opening, and the closing fingers square them


def wrist_choices(pose: dict, yaw: float, period: float = 90, opening: float = OPEN_MM / 1000,
                  others: list[dict] = ()) -> tuple[list[int], list[int]]:
    """(J5s square to the object, best first -- both of a cube's (2026-10-02: the one kept clear of
    a neighbour tilted the fingers into the table at 26 cm, the other was never tried); then J5s a
    step at a time off square towards the wrist straight (J5 90, or 270), up to WRIST_SLACK -- a turned
    wrist tilts the open fingers, one lower, and with the hand not straight down that one can reach
    the table."""
    squares = wrist_squares(pose, yaw, period, opening, others)
    slack = WRIST_SLACK.get(int(period), 0)
    off = []
    for square in squares:
        straight = 90 if square <= 180 else 270   # across the arm's plane: J5 90, or 270 turned over
        step = 5 if square < straight else -5
        off += [square + step * k for k in range(1, slack // 5 + 1) if (square + step * k - straight) * step < 0]
    return squares, [j5 for j5 in off if j5 not in squares]


def turn_of(close: dict | None, wide: dict) -> tuple[dict, str]:
    """(the reading to grip by -- its yaw, period, kind, size -- and which look it came from): the
    first look that saw the thing whole and read its turn from the colour edges (refine_turns:
    1-5 deg where the depth's wandered 6-16), the close one first; else the first that saw it whole.
    A thing cut by the picture's edge (or, before 2026-10-01, by the depth's near limit) gives the
    cut's turn: 0.7 deg three times running for cubes turned up to 45; a lying block cut short reads
    as a cube."""
    looks = (("close", close), ("wide", wide))
    for look_name, o in looks:
        if o is not None and o["whole"] and o.get("turn_from") == "colour":
            return o, f"{look_name}, colour"
    for look_name, o in looks:
        if o is not None and o["whole"]:
            return o, look_name
    return (close or wide), ("close" if close else "wide") + ", seen in part"


TILTS = (5, -5, 10, -10, 15, -15)   # deg the hand may tilt off the approach's pitch, up the line
EXACT_M = 0.04   # the last bit of the line, at exactly the approach's pitch


def line(a: np.ndarray, b: np.ndarray, pitch: float, floor_clear: float, step: float = 0.02, ok=None):
    """IK poses along the straight line a -> b, every `step`, the hand at `pitch` over the last
    EXACT_M and up to 15 deg off it before that -- at the far edge the shoulder is at its end and the
    elbow runs into its own (J3 0) going back up the line at one pitch (2026-10-01); None if an end,
    or two points in a row, are out of reach. `ok(pose)`: the guard, as the pose will be driven (its
    wrist, its gripper) -- by default the arm's own at J5 90."""
    n = max(1, math.ceil(np.linalg.norm(b - a) / step))
    out, skipped = [], False
    for i in range(n, -1, -1):   # from the grip back up: each point's pitch near the one below it
        point = a + (b - a) * i / n
        last = pitch if not out else out[-1][1]
        tilts = (0,) if np.linalg.norm(b - point) <= EXACT_M + 1e-9 else (0, *TILTS)
        for tilt in tilts:
            if abs(last + tilt - pitch) > 15:
                continue
            p = m3kinematics.reach(point, last + tilt, floor_clear=floor_clear, guard=ok is None)
            if p is not None and (ok is None or ok(p)):
                out.append((p, last + tilt))
                skipped = False
                break
        else:
            # one point in the middle missed by whole servo degrees (2026-10-01: straight down by the
            # bumper, 2 cm up, while 4-10 cm up reached): the glide goes on to the next, checked too
            if i in (0, n) or skipped:
                return None
            skipped = True
    return [p for p, _ in reversed(out)]


def furthest(z: float) -> float:
    """m from J1's axis the grip point can be at height z with the shoulder no further forward than
    level (J2 0, its end): the forearm and the hand straight from the elbow down to it."""
    drop = m3kinematics.SHOULDER_Z - z
    span = m3kinematics.L3 + m3kinematics.L4
    return m3kinematics.SHOULDER_R + m3kinematics.L2 + math.sqrt(max(0.0, span * span - drop * drop))


def plan_grasp(centre: np.ndarray, top: float | None = None, prefer: tuple[float, float] | None = None,
               floor_clear: float = FLOOR_CLEAR_PLACE, wrist=None, gripper: int | None = None,
               held: int | None = None):
    """(pitch, the line's poses pre-grasp -> grip, lift) with the grip point in the object's middle
    (centre: base, m, at half its height) -- or higher, up to its `top`, where the middle is out of
    reach: the shoulder's end (J2 0) bounds the reach, and a grip higher up is a grip further out
    (2026-10-01: 24.7-26.3 cm out at 18-30 mm up). `prefer` (pitch, lift): tried first -- after the
    close look the aim moves a few mm, and a plan at another pitch would put the pre-grasp
    somewhere else, the way there maybe over the chassis (2026-10-01: refused at 17 cm over it).
    `wrist(grip pose) -> (J5s square, best first; J5s off square)` (wrist_choices) and `gripper`
    (J6): the plan as it will be driven -- the turned wrist tilts the open fingers, one lower
    (m3kinematics.outline): checked with them, set in the poses; `held` (J6 shut on the thing): the
    way back up checked with that (2026-10-02: up with the open fingers passed, with them shut on a
    40 mm cube the arm refused)."""
    base = np.asarray(m3kinematics.CHAIN[0][0][:2])
    yaw = math.atan2(centre[1] - base[1], centre[0] - base[0])
    radius = float(np.hypot(*(np.asarray(centre[:2]) - base)))
    tries = [(back, lift, pitch) for back in BACKS for lift in LIFTS for pitch in PITCHES + FINE]
    if prefer is not None:
        tries.insert(0, (BACK, prefer[1], prefer[0]))
    hand = {} if gripper is None else {6: gripper}
    reached, lines = {}, {}   # (lift, pitch) -> the grip pose; (back, lift, pitch, J5) -> the line: both passes
    # square first, everywhere -- every approach and grip height -- and only then off square: a
    # cube gripped 20 deg off turns in the fingers, and the grip was luck (2026-10-02)
    for squared in ((True, False) if wrist is not None else (True,)):
        for back, lift, pitch in tries:
            z = max(centre[2], GRIP_Z_MIN) + lift
            if (top is not None and z > top + 1e-6) or radius > furthest(z) + 0.005:   # no pitch reaches it
                continue
            grip = np.array([centre[0], centre[1], z])
            if (lift, pitch) not in reached:   # the grip first: cheap
                reached[lift, pitch] = m3kinematics.reach(grip, pitch, floor_clear=floor_clear, guard=False)
            bottom = reached[lift, pitch]
            if bottom is None:
                continue
            j5s = [bottom[5]] if wrist is None else wrist(bottom)[0 if squared else 1]
            j5s = [j5 for j5 in j5s   # at the grip a turned finger is lowest: the line only if one fits
                   if m3kinematics.problem({**bottom, 5: j5, **hand}, floor_clear) is None]
            if not j5s:
                continue
            r = math.radians(pitch)
            along = np.array([math.cos(yaw) * math.cos(r), math.sin(yaw) * math.cos(r), -math.sin(r)])
            for j5 in j5s:
                if (back, lift, pitch, j5) not in lines:   # the line as it will be driven: this wrist, this
                    # gripper -- a hand turned over clears what it at J5 90 does not (2026-10-05)
                    lines[back, lift, pitch, j5] = line(grip - back * along, grip, pitch, floor_clear, ok=lambda q, j5=j5:
                                                        m3kinematics.problem({**q, 5: j5, **hand}, floor_clear) is None)
                poses = lines[back, lift, pitch, j5]
                if not poses:
                    continue
                driven = [{**q, 5: j5, **hand} for q in poses]
                lifted = driven if held is None else [{**q, 5: j5, 6: held} for q in poses]
                if m3kinematics.problem(driven[0]) is None \
                        and m3arm.glide_problem(driven[0], driven[1:], floor_clear) is None \
                        and m3arm.glide_problem(lifted[-1], lifted[-2::-1], floor_clear) is None:
                    # the pre-grasp pose with the usual margin, and the glides down and back up as the
                    # arm checks them -- every 5 deg, corners cut -- not only at the line's points: in
                    # between the forearm came down to 17 cm over the arm's base, refused (2026-10-01)
                    return pitch, driven, lift
    return None


# The drop-off box (roboshared.box): where the M3 looks for it and how it carries to it.
                         # (2026-10-05: a drop 2.5 cm from it, and the box was hit and turned)
SIDE_LOOK = {1: 180, 2: 95, 3: 30, 4: 0, 5: 90}   # the look at the box: lid, marker and inside in view
LEVEL_OFF_M, LEVEL_OFF_DEG = 0.03, 5.0   # levelled by the floor the camera moves off the arm model by 12 mm
# and 1.7 deg (2026-10-15/16, 112 looks; p90 16 mm, 3.3 deg): more is another plane taken for the floor --
# 2026-10-16 a box look at J1 135, the room's floor and chairs past the table's edge in view, put the box
# 10 cm off and out of reach (the arm model's camera: where it stood) and the clean-up stopped
# The looks either side of LOOK (2026-10-05, the user's: every cube in reach across the front -- LOOK sees
# J1 ~40-140 deg at reach), in a shape that turns freely, J1 0-180, where LOOK's is allowed at J1 89-91
# only (its hand at the bumper's limit): the box look, the right one, the left one and LOOK are one turn
# of J1 and a settle at 90, no way up and down between them (the user saw a first try, in shapes of
# their own, go over a carry pose three times). Looking down at the band a pick reaches, 12-26 cm out:
# SIDE_LOOK's shape saw as much of it but out to 60 cm too -- the laptop, the box's insides, a cube 51 cm
# away (the user: "you cant pick things further then 26 cm"). With LOOK they see every spot a pick reaches.
WIDE_SIDE_SHAPE = {2: 50, 3: 55, 4: 5, 5: 90}
WIDE_SIDES = ({1: 160, **WIDE_SIDE_SHAPE}, {1: 20, **WIDE_SIDE_SHAPE})   # right first: from the box look at J1 180
SEARCH_FAR = 0.45   # m from J1's axis: what lies further, or behind the axis, is no job's -- not in reach, nor
                    # brought there by a drive forward (LOOK sees ahead to 45 cm)
# After the lift the box is looked at again from where it was found (SIDE_LOOK at box_dir: the user's,
# 2026-10-05, "semi home position ... so it sees more" -- over the spot the marker read close and
# askew, 153 deg one look and 170 the next, and the job looked for it all round, holding: the arm up and
# down). Turned there in SIDE_LOOK's shape what is held passes over the front deck's corner lower than
# the chassis' step allows (the wrist turned for a grip, the arm itself 19 cm over its own base): it
# turns with the hand VIEW_TURN deg up -- the grip point 4 cm higher and further out, what is held over
# the steps all the way round -- and tilts it down into the view at the end.
VIEW_TURN = 20
# (m the grip point up, m out from J1's axis): view_way's arc round, lowest first -- what is held clears the
# front deck's 17.5 cm (CHASSIS) from 20 cm up, its corner (16.2 cm out) from 17 cm out
VIEW_ARCS = tuple((high, radius) for high in (0.20, 0.22, 0.24, 0.26) for radius in (0.17, 0.19))
HELD_ROUND = 0.02   # m what is held reaches round the grip point (a 40 mm cube's half): a thing just
                    # picked in front of the bumper is not over the chassis yet
BOX_LOOKS = (180, 158, 135, 112, 90, 68, 45, 22, 0)   # J1s the box is looked for at, SIDE_LOOK's other joints:
                                    # 15-61 cm out from the arm's axis each way -- the box anywhere in the arm's
                                    # half circle, its marker facing the arm (the user's, 2026-10-02). Every 22.5
                                    # deg, not 45 (2026-10-21, the user: "it misses it even when it is in the
                                    # picture"): the box beside the robot's right, 25 cm out, had its marker cut by
                                    # the picture's right edge at J1 135 and by its left edge at 180 -- a 45 deg turn
                                    # moved it a whole picture (640 px, 67 deg), the marker a quarter of one
CARRY_OVER = 0.02   # m the carry swings round above the drop-off's height
ARC_STEP = 15       # deg of J1 between the carry's points


def drop_at(radius: float, height: float, j1: float) -> dict | None:
    """The grip point `radius` m from J1's axis, `height` m up, the arm turned to `j1` (the drop-off
    line's: DROP_OUT and more, DROP_UP, J1 180 -- 2026-10-02, the arm's shell kept off the
    front-right lidar, the first five pitches had none there)."""
    base = np.asarray(m3kinematics.CHAIN[0][0][:2])
    yaw = math.radians(90 - j1)   # J1 is mirrored: bigger turns right
    target = (base[0] + radius * math.cos(yaw), base[1] + radius * math.sin(yaw), height)
    for pitch in (70, 60, 50, 80, 40, 65, 55, 75, 45, 35, 85):   # at full reach whole degrees leave gaps
        pose = m3kinematics.reach(target, pitch)
        if pose:
            return pose
    return None


def carry_to(here: dict, radius: float, j1: float, release: float, high: float | None = None) -> list[dict] | None:
    """Holding something, from `here` to `radius` m from J1's axis with the arm turned to `j1`,
    let go `release` m up: up and out to that distance in the arm's own direction, to `high` (the
    grip point; CARRY_OVER over the release when not given), round at that height and distance --
    never over the chassis (its front corner is 16.2 cm from J1's axis; the drop-offs are further
    out) -- and straight down onto the spot (the user's, 2026-10-02: rather than folded high and
    down again; and 15 cm up, not just over the release -- it nearly hit the box's standing lid).
    None when the guard would not."""
    high = release + CARRY_OVER if high is None else max(high, release)
    start, end = float(here[1]), float(j1)
    count = max(1, math.ceil(abs(end - start) / ARC_STEP))
    way = []
    skipped = False
    for i in range(count + 1):
        pose = drop_at(radius, high, start + (end - start) * i / count)
        if pose is None:   # one point in the middle missed by whole degrees: on to the next, the glide checked
            if i in (0, count) or skipped:
                return None
            skipped = True
            continue
        skipped = False
        way.append({**here, **{j: pose[j] for j in (1, 2, 3, 4)}})
    down = drop_at(radius, release, end)
    if down is None:
        return None
    way.append({**here, **{j: down[j] for j in (1, 2, 3, 4)}})
    return way if m3arm.glide_problem(here, way) is None else None









def way_to(here: dict, goal: dict, direct: bool = True) -> list[dict]:
    """The quickest way the guard allows from `here` to `goal` (whole poses), for one glide: straight
    there (unless not `direct`: holding something, never swung across the chassis low); else turned
    to the goal's J1 while rising to a carry pose, then on; else up first, turned, then on.
    CARRIES lowest first, FOLDED -- always allowed -- last."""
    def at(carry, j1):
        return {**here, 1: j1, 2: carry[0], 3: carry[1], 4: carry[2]}
    ways = ([[goal]] if direct else []) + [[at(c, goal[1]), goal] for c in CARRIES] \
        + [[at(c, here[1]), at(c, goal[1]), goal] for c in CARRIES]
    for way in ways:
        if m3arm.glide_problem(here, way) is None:
            return way
    return ways[-1]


TURN_SHUT = 150   # J6 the fingers close to while the wrist turns over (turn_way): 0 widest, 180 shut


def turn_way(here: dict, goal: dict) -> list[list[dict]] | None:
    """The wrist turned where the hand is, as glides one after another (2026-10-21, the user's: from
    the close look, J5 90, to a line's top turned over, J5 270, the arm went up to a carry pose and back
    each time -- the open fingers, swung across, came over the front deck): the fingers closed to
    TURN_SHUT, the arm to `goal` with them closed, then opened as `goal` has them. None when `goal` is
    reached straight, without the gripper in both, or when the guard allows none of it (way_to's)."""
    if m3arm.GRIPPER not in here or m3arm.GRIPPER not in goal or m3arm.glide_problem(here, [goal]) is None:
        return None
    shut = max(TURN_SHUT, here[m3arm.GRIPPER])
    legs = [[{**here, m3arm.GRIPPER: shut}], [{**goal, m3arm.GRIPPER: shut}], [goal]]
    start = here
    for leg in legs:
        if m3arm.glide_problem(start, leg) is not None:
            return None
        start = leg[-1]
    return legs


def look_way(here: dict, goal: dict) -> list[list[dict]]:
    """The way between looks without going up and down, as glides one after another: straight there;
    else turned in the side looks' shape (it turns freely), then into the goal's -- each a glide of its
    own, the corners between them not cut (cut, LOOK -> the right look's came down over the chassis);
    else way_to's."""
    shape = {j: WIDE_SIDE_SHAPE[j] for j in (2, 3, 4)}
    for legs in ([[goal]], [[{**goal, **shape}], [goal]], [[{**here, **shape}], [{**goal, **shape}], [goal]]):
        start, ok = here, True
        for leg in legs:
            ok = ok and m3arm.glide_problem(start, leg) is None
            start = leg[-1]
        if ok:
            return [leg for leg in legs if leg[-1] != here]
    return [way_to(here, goal)]


def held_problem(start: dict, poses: list[dict], below: float) -> str | None:
    """Why what is held -- its bottom `below` m under the grip point, HELD_ROUND round it -- would come
    down onto the chassis' steps on the glides start -> poses (each a glide of its own, every
    STEP_DEG), or None: the arm's guard knows the arm, not what it holds."""
    last = start
    for goal in poses:
        n = max(1, math.ceil(max(abs(goal[j] - last[j]) for j in (1, 2, 3, 4)) / m3arm.STEP_DEG))
        for k in range(1, n + 1):
            p, _ = m3kinematics.grip({j: last[j] + (goal[j] - last[j]) * k / n for j in (1, 2, 3, 4, 5)})
            for x0, x1, top, half in m3kinematics.CHASSIS:   # the steps' tops keep their spare; `half`: how
                # far either side each is, the body's or with its wheels (m3kinematics)
                if x0 - HELD_ROUND < p[0] < x1 + HELD_ROUND and abs(p[1]) < half + HELD_ROUND and p[2] - below < top:
                    return (f"what is held would come down to {100 * (p[2] - below):.0f} cm above the floor"
                            f" over the chassis (x {p[0]:+.2f} m): keep it above {100 * top:.0f} cm there")
        last = goal
    return None


# The close look again when the first saw the thing cut by the picture's edge, or not at all (2026-10-05:
# a block by the bumper, its near half below the picture -- aimed by the wide look, a careful grip's
# finger came down on its edge and rolled it over): the camera turned onto it, J1 up to 15 deg and J2-J4
# up to RELOOK_DEG, the least move that has its top face whole in the picture, clear of the fingers at the
# bottom (to v LOOK_BOTTOM), LOOK_NEAR_M off at least (the depth answers from ~19 cm).
RELOOK_DEG, LOOK_BOTTOM, LOOK_NEAR_M = 30, 420, 0.19
_RELOOKS = sorted(itertools.product(range(-15, 16, 5), *[range(-RELOOK_DEG, RELOOK_DEG + 1, 5)] * 3),
                  key=lambda d: sum(map(abs, d)))[1:]


def in_picture(pose: dict, thing: dict, K, size: tuple[int, int] = (640, 480), margin: float = 0.005) -> bool:
    """`thing`'s top face (its size along the line the fingers close on, its length across it, `margin`
    round it) whole in the picture from `pose`, 20 px in from its edges and above LOOK_BOTTOM, the
    camera LOOK_NEAR_M off it at least -- and looking down at a slant: m3vision.floor_plane takes no
    floor seen straight from above (its normal must point against the picture's down)."""
    along = np.array([math.cos(math.radians(thing["yaw"])), math.sin(math.radians(thing["yaw"]))])
    across = np.array([-along[1], along[0]])
    w, length = thing["size"], thing.get("length", thing["size"])
    face = [np.r_[thing["centre"][:2] + along * a * (w / 2 + margin) + across * b * (length / 2 + margin), thing["height"]]
            for a in (-1, 1) for b in (-1, 1)]
    camera = m3kinematics.frames(pose)["arm4"] @ m3kinematics.camera_optical()
    if camera[2, 1] > -0.15:
        return False
    q = (np.linalg.inv(camera) @ np.c_[np.array(face), np.ones(4)].T)[:3]
    if q[2].min() < LOOK_NEAR_M:
        return False
    u, v = (np.asarray(K) @ (q / q[2]))[:2]
    return bool(u.min() >= 20 and u.max() <= size[0] - 20 and v.min() >= 20 and v.max() <= LOOK_BOTTOM)


def look_at(here: dict, pre: dict, thing: dict, K) -> dict | None:
    """A pose near `here` -- the least move of J1-J4 (_RELOOKS) -- from which `thing` is whole in the
    picture, that the guard allows and the arm glides to, and from it to `pre` (the pre-grasp); None."""
    for d in _RELOOKS:
        pose = {**here, **{j: here[j] + dj for j, dj in zip((1, 2, 3, 4), d)}}
        if all(0 <= pose[j] <= 180 for j in (1, 2, 3, 4)) and in_picture(pose, thing, K) \
                and m3kinematics.problem(pose) is None and m3arm.glide_problem(here, [pose]) is None \
                and m3arm.glide_problem(pose, [pre]) is None:
            return pose
    return None


def level_trusted(camera: np.ndarray, model: np.ndarray) -> bool:
    """Whether a camera levelled by the floor its depth found (m3vision.level) is near enough the arm
    model's to be the floor indeed: LEVEL_OFF_M in height, LEVEL_OFF_DEG turned."""
    turned = math.degrees(float(np.linalg.norm(cv2.Rodrigues(camera[:3, :3] @ model[:3, :3].T)[0])))
    return abs(float(camera[2, 3] - model[2, 3])) <= LEVEL_OFF_M and turned <= LEVEL_OFF_DEG


def view_way(here: dict, view: dict, below: float) -> list[list[dict]]:
    """Holding something, from the lift to the box `view` (a whole pose: SIDE_LOOK at box_dir) without
    going up and down, as glides one after another: risen in place into the view's shape and turned
    there; else risen into it with the hand VIEW_TURN deg up, turned, and the hand tilted down into
    the view; else up and out to VIEW_ARCS, round at that height and into the view (2026-10-21, the
    user's: with the box at J1 135 every pick went up to a carry pose -- in the view's shape what is
    held came over the front deck's corner, turning); else way_to's, swung high. `below`: what is held
    hangs that far under the grip point."""
    shape = {j: view[j] for j in (2, 3, 4, 5)}
    raised = {**shape, 4: view[4] + VIEW_TURN}
    for legs in ([[{**here, **shape}], [view]],
                 [[{**here, **raised}], [{**view, **raised}], [view]]):
        start, ok = here, True
        for leg in legs:
            ok = ok and m3arm.glide_problem(start, leg) is None
            start = leg[-1]
        if ok and held_problem(here, [leg[-1] for leg in legs], below) is None:
            return [leg for leg in legs if leg[-1] != here]
    for high, radius in VIEW_ARCS:   # what is held checked round the arc; into the view as way_to's
        # is, by the arm alone: at J1 45 or 135 the view holds a 40 mm cube's bottom ~1 cm under the
        # front deck's 1.5 cm spare whichever way it comes (2026-10-21)
        arc = carry_to(here, radius, view[1], high, high)
        if arc is not None and m3arm.glide_problem(arc[-1], [view]) is None and held_problem(here, arc, below) is None:
            return [arc, [view]]
    return [way_to(here, view, direct=False)]


def nudge(found: list[dict], kit: list[dict], at=None, most: float = 0.10, right: float = 0.03) -> tuple[float, float] | None:
    """(forward, left) m to drive so that the kit thing nearest the robot (or `at`) comes to
    APPROACH_X ahead, straight ahead -- forward and sideways only, at most `most` each way and
    `right` to the right (Picker.right_room: short of the drop-off box), never so far that anything
    in the robot's way ends up within BUMPER_GAP of the bumper; None when that is under a centimetre."""
    key = (lambda o: float(np.linalg.norm(o["centre"][:2] - at))) if at is not None else \
        (lambda o: float(np.hypot(*o["centre"][:2])))
    target = min(kit, key=key)
    goal = target["centre"]
    dx = float(np.clip(goal[0] - APPROACH_X, 0.0, most))
    axis = np.asarray(m3kinematics.CHAIN[0][0][:2])   # J1's: what the arm reaches round
    near = float(np.hypot(goal[0] - dx - axis[0], goal[1] - axis[1])) <= furthest(target["height"]) - 0.02
    dy = 0.0 if near else float(np.clip(goal[1], -right, most))   # sideways only if forward is not enough
    for o in found:   # anything the chassis would sweep into, kit or not, before or after a sideways move
        if min(abs(o["centre"][1]), abs(o["centre"][1] - dy)) < m3kinematics.CHASSIS_Y + 0.03:
            dx = min(dx, max(0.0, o["centre"][0] - BUMPER_X - BUMPER_GAP))
    if abs(dx) < 0.01 and abs(dy) < 0.01:
        return None
    return round(float(dx), 3), round(float(dy), 3)


def slide_for(thing: dict, found: list[dict], most: float, floor_clear: float, plan=None) -> float | None:
    """The least slide right (m, in SIDESTEP steps up to `most`) after which `thing` (one of
    `found`) has a plan: a plain one, cheap, then `plan(it, all)` with everything moved, if given."""
    index = next(i for i, o in enumerate(found) if o is thing)
    for slide in np.arange(SIDESTEP, most + 1e-9, SIDESTEP):
        moved = [{**o, "centre": o["centre"] + [0.0, slide, 0.0]} for o in found]
        it = moved[index]
        if plan_grasp(it["centre"], it["height"], floor_clear=floor_clear) is not None \
                and (plan is None or plan(it, moved) is not None):
            return float(slide)
    return None


def closer_for(thing: dict, floor_clear: float, most: float = CLOSER_MOST) -> float | None:
    """The least drive forward (m, a cm at a time up to `most`) after which `thing` has a plain plan:
    how much closer the robot would have to come; None when driving forward does not do it."""
    for dx in np.arange(0.01, most + 1e-9, 0.01):
        if plan_grasp(thing["centre"] - [dx, 0.0, 0.0], thing["height"], floor_clear=floor_clear) is not None:
            return round(float(dx), 2)
    return None





def aimed(found: list[dict], axis, out_mm: float = 0.0, left_mm: float = 0.0) -> None:
    """Each thing's centre moved to where the arm must aim to have it between its fingers: the taught
    offset there (m3aim), trimmed `out_mm` further from J1's `axis` and `left_mm` to the left as the
    arm sees it (the trim_* SETTINGS); where the camera put it is kept as "seen"."""
    axis = np.asarray(axis, float)
    for o in found:
        o["seen"] = o["centre"].copy()
        away = o["centre"][:2] - axis
        out = away / max(float(np.linalg.norm(away)), 1e-6)
        shift = m3aim.offset(*o["centre"][:2]) + out * out_mm / 1000 + np.array([-out[1], out[0]]) * left_mm / 1000
        o["centre"] = np.r_[o["centre"][:2] + shift, o["centre"][2]]


def mark(image: np.ndarray, K, camera: np.ndarray, found: list[dict], target: dict | None = None) -> np.ndarray:
    """The picture with what was found: kit green (the target thicker), the rest orange."""
    marked = image.copy()
    inv = np.linalg.inv(camera)
    for o in found:
        top = inv @ np.r_[o.get("seen", o["centre"])[:2], o["top"], 1.0]   # where the camera saw it
        if top[2] <= 0:
            continue
        u, v = (np.asarray(K) @ (top[:3] / top[2]))[:2]
        colour = (0, 255, 0) if o["kind"] else (0, 165, 255)
        size = 30 if o is target else 20
        cv2.drawMarker(marked, (int(np.rint(u)), int(np.rint(v))), colour, cv2.MARKER_CROSS, size, 3 if o is target else 2)
        label = name(o) if o["kind"] else f"? {o['top'] * 1000:.0f} mm tall"
        cv2.putText(marked, label, (int(u) + 10, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return marked


# --------------------------------------------------------------------------------------- picker --

# --------------------------------------------------------------------------------------- picker --
class Picker(Task):
    """The M3's body for the shared jobs (roboshared.task.Task): its depth camera's looks, its grasp
    planning (plan_grasp, the careful grip), its way down and back up, its carries, its box looks."""

    SETTINGS = SETTINGS
    AXIS = tuple(float(v) for v in m3kinematics.CHAIN[0][0][:2])
    FLOOR_Z = m3kinematics.FLOOR
    FOOTPRINT = (ROBOT_X, ROBOT_Y)
    DROP_OUT, DROP_SPREAD = DROP_OUT, DROP_SPREAD
    BOX_LOOKS = BOX_LOOKS

    def __init__(self, *args, odom=None, goto=None, colour=None, **kwargs):
        """Task's, and for the scan: `odom()` -> the wheel odometry's (x, y, yaw) of the base, and
        `goto(x, y, yaw)` driving it there (moving and turning at once); None: it cannot. For the
        policy: `colour()` -> the newest colour picture (BGR) and its age, s."""
        super().__init__(*args, **kwargs)
        self.odom, self.goto, self.colour = odom, goto, colour
        self.floor_m: float | None = None   # this pick's fingertips over the table (plan, grip); None: the first

    def jobs(self) -> dict:
        return {**super().jobs(), "scan": self.scan, "teach": self.teach, "teach_scan": self.teach_scan,
                "find": self.find, "pick_named": self.pick_named, "policy": self.policy}

    def policy(self, seconds: float = 30.0, max_dps: float = m3policy.MAX_DPS, dry: bool = False) -> None:
        """A trained policy drives the arm ten times a second for `seconds` (m3policy.run)."""
        if self.colour is None:
            raise RuntimeError("no camera here for a policy")
        m3policy.run(self, self.colour, seconds, max_dps, dry)

    def teach(self, name: str) -> None:
        """Learn the thing alone in front as `name`, from views round it (m3teach.teach)."""
        m3teach.teach(self, name)

    def find(self) -> None:
        """Name what is on the table in front by the taught things (m3teach.find)."""
        m3teach.find(self)

    def pick_named(self, name: str, drive: bool = False) -> None:
        """Pick the thing taught as `name` (m3teach.pick_named); `drive`: the base may drive up to it."""
        m3teach.pick_named(self, name, drive=drive)

    def taught(self) -> list[str]:
        """The names taught so far (the teach model's), or none when it does not answer."""
        try:
            return m3teach.ask("/health", timeout=2.0)["names"]
        except RuntimeError:
            return []

    def scan(self, radius: float = m3scan.STOP_R, stops: float = m3scan.STOPS, height: float | None = None,
             width: float | None = None, name: str | None = None) -> None:
        """Drive round the thing about `radius` m ahead, the camera on it, keeping every view (m3scan.run);
        its size as given (`height`, `width`, m) or as the first look reads it. Given a `name`, then
        learn the thing as that from the views (m3teach.teach_from_scan)."""
        if name is not None:
            m3teach.checked_name(name)   # before the drive, not after it
        folder = m3scan.run(self, radius, int(stops), height, width)
        if name is not None and folder is not None:
            m3teach.teach_from_scan(self, name, folder)

    def teach_scan(self, name: str, stamp: str | None = None) -> None:
        """Learn `name` from a scan already made (out/scan/STAMP, the newest by default)."""
        if self.out is None:
            raise RuntimeError("no pictures kept here: no scan to teach from")
        folders = sorted(f for f in (self.out.parent / "scan").glob("*_*") if f.is_dir() and stamp in (None, f.name))
        if not folders:
            raise RuntimeError(f"no scan {stamp or 'yet'} to teach from")
        m3teach.teach_from_scan(self, name, folders[-1])

    def wrist(self, thing: dict, around: list[dict], whole: bool | None = None, opened: int | None = None):
        """grip pose -> (J5s square, J5s off square) for `thing` (wrist_choices): the fingers, opened to
        `opened` (opened_for's by default), clear of `around` -- or straight (90) while its turn is
        not known (seen in part)."""
        if not (thing["whole"] if whole is None else whole):
            return lambda grip: ([90], [])
        others = [o for o in around if o is not thing]
        span = m3kinematics.finger_span(opened_for(thing["size"]) if opened is None else opened)
        return lambda grip: wrist_choices(grip, thing["yaw"], thing["period"], span, others)

    def floor_clear(self) -> float:
        """m the fingertips keep above the measured floor on the descent: grips rise to keep it. This
        pick's (plan), else the "fingertips_mm" setting's."""
        return self.cfg["fingertips_mm"] / 1000 if self.floor_m is None else self.floor_m

    def floors(self) -> list[float]:
        """The fingertips' heights a plan tries, in turn: "fingertips_mm", then "fingertips_least_mm"."""
        first, least = self.cfg["fingertips_mm"] / 1000, self.cfg["fingertips_least_mm"] / 1000
        return [first] if least >= first else [first, least]

    def arm_clear(self) -> float:
        """What the arm's guard is asked to keep on the descent: the plan's, less GUARD_SLACK."""
        return max(self.floor_clear() - GUARD_SLACK, m3kinematics.MIN_FLOOR_CLEAR)

    def move(self, pose: dict, floor_clear: float | None = None, speed: float = 20) -> None:
        """One smooth move (M3Arm.move_to: one command, every step of the way checked first)."""
        self.check()
        kwargs = {"speed_dps": speed, "settle": False}
        if floor_clear is not None:
            kwargs["floor_clear"] = floor_clear
        self.call("move_to", {int(j): v for j, v in pose.items()}, **kwargs)
        self.check()

    def go(self, goal: dict, speed: float = 20, holding: bool = False) -> None:
        """To `goal` the quickest way the guard allows (way_to): straight, or round over the chassis
        (2026-10-01: the look -> a straight-down pre-grasp, straight, passed 17 cm over the arm's base);
        `holding` something: never straight, swung across the chassis low."""
        here = self.here()
        self.through(way_to(here, {**here, **{int(j): v for j, v in goal.items()}}, direct=not holding), speed=speed)

    def through(self, poses: list[dict], floor_clear: float | None = None, speed: float = 20) -> None:
        """Through `poses` without stopping at each (M3Arm.follow)."""
        self.check()
        kwargs = {"speed_dps": speed} if floor_clear is None else {"speed_dps": speed, "floor_clear": floor_clear}
        self.call("follow", [{int(j): v for j, v in p.items()} for p in poses], **kwargs)
        self.check()

    # -------------------------------------------------------------------------- seeing --
    def look(self, tag: str, target_near: np.ndarray | None = None):
        """(levelled camera, what was found) from where the arm is; the marked picture kept."""
        time.sleep(self.settle_s if self.settle_s is not None else
                   self.cfg["settle_close_s" if tag == "close" else "settle_wide_s"])
        self.check()
        image, depth, K, pose = self.snapshot()
        self.K = K   # (look_at plans a look again with it)
        with metrics.span("detect"):
            model = m3kinematics.frames(pose)["arm4"] @ m3kinematics.camera_optical()
            normal, height, _ = m3vision.floor_plane(depth, K)
            camera = m3vision.level(model, normal, height, m3kinematics.FLOOR)
            found = m3vision.objects(depth, K, camera, floor=(normal, height))
            m3vision.refine_turns(image, K, camera, found)   # the turn from the colour edges, where sure
        aimed(found, self.AXIS, self.cfg["trim_out_mm"], self.cfg["trim_left_mm"])
        with metrics.span("kit detector"):   # its second opinion, logged only (m3kit)
            detections = m3kit.ask(image)
        if detections is not None:
            self.say(f"{tag} look, " + m3kit.line(m3kit.compare(found, detections)))
        base = np.asarray(self.AXIS)
        for o in found:   # cheap, and hopeful: the pick plans the one it takes in full
            radius = float(np.hypot(*(o["centre"][:2] - base)))
            o["reach"] = bool(o["kind"]) and radius <= furthest(o["height"]) + 0.005
        near = None
        if target_near is not None:
            near = min(found, key=lambda o: float(np.linalg.norm(o["centre"][:2] - target_near[:2])), default=None)
        marked = mark(image, K, camera, found, near)
        cv2.putText(marked, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        self.show(marked, found)
        if self.out is not None:
            self.out.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(self.out / f"{stamp}_{tag}.jpg"), marked)
            cv2.imwrite(str(self.out / f"{stamp}_{tag}_rgb.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(self.out / f"{stamp}_{tag}_depth.png"), depth)
            (self.out / f"{stamp}_{tag}.json").write_text(json.dumps({   # for looking again later; the
                # clean picture's labels for the kit detector (PLAN.md M5) and the depth duel (S1)
                "pose": {str(j): v for j, v in pose.items()}, "K": np.asarray(K).tolist(),
                "camera": camera.tolist(),
                "found": [{"kind": o["kind"], "size": o["size"], "length": o["length"], "top": o["top"],
                           "centre": [float(c) for c in o["seen"]], "aimed": [float(c) for c in o["centre"]],
                           "yaw": o["yaw"], "whole": o["whole"],
                           "outline_px": o["outline_px"]} for o in found],
                "detector": detections}))   # the kit detector's answer (m3kit), None with no service
        return camera, found

    def view(self):
        """The picture from where the arm is, the camera levelled by the depth's floor when there is
        one in it and it is the floor (level_trusted), else the arm model's: (image, K, camera, (depth,
        floor normal, floor height)). The box's looks, kept in `out` as boxview_* to look at again."""
        image, depth, K, pose = self.snapshot()
        model = m3kinematics.frames(pose)["arm4"] @ m3kinematics.camera_optical()
        try:
            normal, height, _ = m3vision.floor_plane(depth, K)
            camera = m3vision.level(model, normal, height, m3kinematics.FLOOR)
            if not level_trusted(camera, model):
                raise ValueError("the plane found is no floor the arm could be over")
        except ValueError:
            camera, normal, height = model, None, None
        if self.out is not None:
            self.out.mkdir(parents=True, exist_ok=True)
            stem = f"boxview_{time.strftime('%Y%m%d_%H%M%S')}"
            cv2.imwrite(str(self.out / f"{stem}_rgb.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(self.out / f"{stem}_depth.png"), depth)
            (self.out / f"{stem}.json").write_text(json.dumps({
                "pose": {str(j): v for j, v in pose.items()}, "K": np.asarray(K).tolist(),
                "model": model.tolist(), "camera": np.asarray(camera).tolist(), "levelled": normal is not None}))
        return image, K, camera, (depth, normal, height)

    def things_in_view(self, image, K, camera, extra) -> list[dict]:
        depth, normal, height = extra
        if normal is None:
            return []
        return m3vision.objects(depth, K, camera, floor=(normal, height))

    # --------------------------------------------------------------------------- doing --
    def engaged(self) -> None:
        """The arm's pose is unknown until its first command: home first, ~3 s."""
        try:
            self.call("read")
        except Exception:  # noqa: BLE001 -- the bridge's ArmError, or the HTTP wrapper's exit
            with self.step("engage"):
                self.call("engage")

    def home(self) -> None:
        self.engaged()
        with self.step("home"):
            # the way the guard allows (2026-10-02: straight home from a look to the right was refused)
            self.go(dict(zip(m3arm.JOINTS, m3arm.POSES["home"])), speed=self.dps("free_dps"))

    def wide_look(self, sides: bool = False) -> list[dict]:
        """To the wide look, the gripper opening on the way, and what it shows; `sides`: the looks
        either side too (WIDE_SIDES), each thing as the look most facing it saw it whole -- one only
        another look cut short left out (PART_M). What lies where this job lets go is left out too
        (dropped_here), and what lies behind J1's axis or past SEARCH_FAR."""
        axis = np.asarray(self.AXIS)

        def off_facing(o, j1):   # deg between the thing's bearing from J1's axis and the look's
            return abs(90 - math.degrees(math.atan2(o["centre"][1] - axis[1], o["centre"][0] - axis[0])) - j1)
        readings = []   # (thing, the J1 of the look that saw it)
        for look in ((*WIDE_SIDES, LOOK) if sides else (LOOK,)):
            here = self.here()
            for leg in look_way(here, {**look, m3arm.GRIPPER: m3arm.gripper_angle(OPEN_MM)}):
                self.through(leg, speed=self.dps("free_dps"))
            tag = "wide" if look is LOOK else f"wide J1 {look[1]}"
            try:
                _camera, seen = self.look(tag)
            except ValueError as exc:   # no floor in the picture: that look skipped, the others count (2026-10-07:
                # the left look saw only the robot's own deck, nearer than the depth reaches, and the job ended)
                self.say(f"the {tag} look skipped: {exc}")
                continue
            readings += [(o, look[1]) for o in seen]
        kept = []
        for o, j1 in sorted(readings, key=lambda r: not r[0]["whole"]):   # the whole ones first
            same = next((i for i, (k, kj) in enumerate(kept) if np.linalg.norm(k["centre"][:2] - o["centre"][:2])
                         < (MATCH_M if o["whole"] or kj == j1 else PART_M)), None)
            if same is None:
                kept.append((o, j1))
            elif o["whole"] and off_facing(o, j1) < off_facing(*kept[same]):
                kept[same] = (o, j1)
        found = [o for o, _ in kept if not self.dropped_here(o) and o["centre"][0] >= axis[0]
                 and float(np.hypot(*(o["centre"][:2] - axis))) <= SEARCH_FAR]
        self.say(("looked round: " if sides else "wide look: ") + describe(found))
        return found

    def grip(self, target: dict, planned, found: list[dict]) -> dict:
        """The close look, the aim corrected, the turn read, the grip replanned -- down the line, shut,
        back up it (Task.grip)."""
        others = [o for o in found if o is not target]
        self.floor_m = target.get("floor")   # the plan's fingertips over the table, down to the put back
        # The close look with the wrist straight: the open fingers left and right of the thing in the picture.
        # Turned for the grip already, one came up across it (2026-10-05: a 30 mm cube seen in part, aimed by
        # the wide look and missed; the next time not seen at all) -- it turns on the way down.
        # Never turned over (J5 270): the finger motor's housing hid the picture's bottom third and the thing
        # in it (2026-10-09, a 30 mm cube "seen in part") -- the wrist turns at the line's top instead.
        pre = planned[1][0]
        look_from = {**pre, 5: 90}
        if m3kinematics.problem(look_from) is not None or m3arm.glide_problem(look_from, [pre]) is not None:
            look_from = pre   # (the wrist turned back where it is, or the way down goes over a carry pose)
        with self.step("pregrasp"), refusable(target["centre"]):   # the fingers opened on the way
            self.go(look_from, speed=self.dps("near_dps"))
        with self.step("look.close"):
            _camera, close = self.look("close", target["centre"])
        again = [o for o in close if o["kind"] and np.linalg.norm(o["centre"][:2] - target["centre"][:2]) < MATCH_M]
        looked_again = False
        if not (again and again[0]["whole"]):   # cut by the picture's edge, or not there: the camera onto it
            relook = look_at(look_from, pre, target, self.K)
            if relook is not None:
                with self.step("look.close"):
                    self.through([relook], speed=self.dps("near_dps"))
                    try:
                        _camera, close2 = self.look("close", target["centre"])
                    except ValueError as exc:   # no floor in it: the first look's, then
                        self.say(f"the look again failed ({exc})")
                        close2 = []
                again2 = [o for o in close2 if o["kind"] and np.linalg.norm(o["centre"][:2] - target["centre"][:2]) < MATCH_M]
                if again2 and (again2[0]["whole"] or not again):
                    close, again, looked_again = close2, again2, True
        reading, yaw_from = turn_of(again[0] if again else None, target)
        # the aim: the close look's -- unless it saw the thing only in part and the wide look whole: a cut
        # outline is grown to size on a guess, 11-14 mm off (2026-10-02), the arm repeats to ~5-9
        aim = again[0] if again and (again[0]["whole"] or not target["whole"]) else target
        centre = aim["centre"]
        if reading is not aim:   # the turn read from the other look: its height
            centre = np.r_[centre[:2], reading["centre"][2]]
        found_how = ("found it" if aim is not target
                     else "seen in part, the wide look aimed" if again else "not found, the wide look kept") \
            + (", looked again turned onto it" if looked_again else "")
        self.say(f"close look: {found_how}, moved {1000 * np.linalg.norm(centre[:2] - target['centre'][:2]):.1f} mm")
        # what is round it: the close look's, and the wide look's -- the close one sees less round it
        others = [o for o in close if np.linalg.norm(o["centre"][:2] - centre[:2]) > MATCH_M] + others
        careful = target.get("careful", False)   # a careful grip stays one; a plain one may become one, a few mm on
        replanned = self.grasp(reading, others, centre, prefer=(planned[0], planned[2]), whole=True, careful=careful)
        if replanned is None and not careful:
            replanned = self.grasp(reading, others, centre, prefer=(planned[0], planned[2]), whole=True, careful=True)
            careful = replanned is not None
        if replanned is None:   # at the edge whole servo degrees make it all or nothing (2026-10-02: a
            # plan at y -0.1297, none at -0.1300): the first plan, a few mm off, its wrist as now read
            squares, off = self.wrist(reading, others, whole=True, opened=planned[1][-1].get(m3arm.GRIPPER))(planned[1][-1])
            for j5 in [*squares, *off]:
                poses = [{**p, 5: j5} for p in planned[1]]
                shut = [{**p, 6: m3arm.gripper_angle(reading["size"] * 1000)} for p in poses]
                if m3arm.glide_problem(poses[0], poses[1:], self.floor_clear()) is None \
                        and m3arm.glide_problem(shut[-1], shut[-2::-1], self.floor_clear()) is None:
                    replanned = (planned[0], poses, planned[2])
                    break
        if replanned is None:   # the corrected aim, the turn read: no way down the arm allows
            raise Refused(centre, "refused: no way down with the fingers turned as this one needs")
        if careful and not (again and again[0]["whole"]):   # a finger a few mm off it: on a guess it came
            # down on the thing's edge and rolled it over (2026-10-05)
            raise Refused(centre, "refused: the careful grip goes down only on a whole close look, and the close"
                                  " looks saw it cut or not at all")
        pitch, poses, _lift = replanned
        j5 = poses[-1][5]
        off = off_square(poses[-1], reading["yaw"], reading["period"])
        self.say(f"the {name(reading)}: fingers along {reading['yaw']:.0f} deg ({yaw_from} look), wrist J5 {j5}"
                 + (f" -- {off:.0f} deg off square: square put a finger on the table" if off > 3 else "")
                 + (f" -- careful grip: opened {CAREFUL_SPARE_MM} mm over it, aimed {1000 * reading.get('shift', target.get('shift', 0)):.0f} mm"
                    " further out (no other keeps the fingers off the robot)" if careful else "")
                 + (f" -- fingertips {1000 * self.floor_clear():.0f} mm over the table (no grip higher)"
                    if self.floor_clear() < self.floors()[0] else ""))
        with self.step("descend"), refusable(centre):   # to the line's top, then down it in one glide
            here = self.here()
            top = {**here, **poses[0]}
            for leg in turn_way(here, top) or [way_to(here, top)]:   # a wrist turned over: the fingers closed
                self.through(leg, speed=self.dps("near_dps"))
            self.through(poses[1:], floor_clear=self.arm_clear(), speed=self.dps("descend_dps"))
        with self.step("close"):
            self.call("close_gripper", width_mm=round(reading["size"] * 1000), floor_clear=self.arm_clear(),
                      speed_dps=self.dps("gripper_dps"), settle=False)
            time.sleep(self.cfg["grip_s"])
        holding = [{j: v for j, v in p.items() if j != m3arm.GRIPPER} for p in poses]   # the gripper as it is now:
        with self.step("lift"):                                                            # shut on the thing
            try:   # (2026-10-02: the line's poses carry the open gripper -- played back, it let go going up)
                self.through(list(reversed(holding[:-1])), floor_clear=self.arm_clear(), speed=self.dps("lift_dps"))
            except Stopped:
                raise
            except Exception as exc:   # refused back up the line: up by another way
                if "refused" not in str(exc):
                    raise
                self.say(f"back up the line refused ({exc}): straight up instead")
                self.straight_up(speed=self.dps("lift_dps"))
        self.held_below = float(m3kinematics.grip(poses[-1])[0][2])   # its bottom was on the floor
        self.held_size = reading["size"]
        return {"reading": reading, "centre": centre, "line": poses,
                "record": {"yaw_from": yaw_from, "pitch": pitch, "j5": j5, "grip_pose": poses[-1],
                           "close_look_found": bool(again), "careful": careful}}

    def straight_up(self, speed: float, rise: float = 0.10) -> None:
        """Up from where the hand is, holding what it holds: the grip point straight up 2 cm at a
        time to `rise`, the hand as steep as it is (or 5-10 deg off where that will not do) -- the
        way out the 2026-10-02 recovery took by hand when the way back up the line was refused."""
        here = self.here()
        point, pitch = m3kinematics.grip(here)
        way = []
        for k in range(1, round(rise / 0.02) + 1):
            for p in (pitch, pitch + 5, pitch - 5, pitch + 10, pitch - 10):
                pose = m3kinematics.reach(point + np.array([0.0, 0.0, 0.02 * k]), p, floor_clear=m3kinematics.MIN_FLOOR_CLEAR)
                if pose is not None:
                    way.append({**here, **{j: pose[j] for j in (1, 2, 3, 4)}})
                    break
        if not way or m3arm.glide_problem(here, way, self.arm_clear()) is not None:
            raise Refused(point, "refused: no way up from here -- the arm holds it; take it off by hand")
        self.through(way, floor_clear=self.arm_clear(), speed=speed)

    def carry_high(self) -> float:
        """m the grip point is carried round at: what it holds hangs below it by as much as the grip
        point was over the floor when the fingers shut on it (held_below; 4 cm when not known) --
        its bottom at the "carry_cm" setting."""
        return self.cfg["carry_cm"] / 100 + getattr(self, "held_below", 0.04)

    # ------------------------------------------------------------------- the drop-off --
    def drop_on_line(self, extra: float) -> None:
        """Up and round to the line out on the right (carry_to; way_to when that will not do), let go as
        low as the arm reaches there: what is held LINE_OVER over the table, else higher, DROP_UP at
        most (2026-10-21, the user's: from DROP_UP's 11 cm blocks bounced off the table)."""
        here = self.here()
        low = getattr(self, "held_below", 0.04) + LINE_OVER
        for release in [low, *(h for h in LINE_RELEASES if h > low)]:
            way = carry_to(here, DROP_OUT + extra, 180.0, release, self.carry_high())
            if way is not None:
                break
        else:
            way = way_to(here, drop_at(DROP_OUT + extra, DROP_UP, 180.0) or drop_at(DROP_OUT, DROP_UP, 180.0), direct=False)
        self.through(way, speed=self.dps("free_dps"))

    def go_box_look(self, j1: float, holding: bool) -> None:
        self.go({**SIDE_LOOK, 1: j1}, speed=self.dps("free_dps"), holding=holding)

    def to_box_view(self) -> None:
        """SIDE_LOOK at box_dir, risen and turned, no up and down (view_way), the wrist straight: turned
        over (J5 270) the finger motor's housing filled the picture's bottom and the camera's exposure
        smeared the marker (2026-10-09, a cube gripped at J5 232: box not found, put back)."""
        here = self.here()
        view = {**here, **SIDE_LOOK, 1: self.box_dir}
        for leg in view_way(here, view, getattr(self, "held_below", 0.04)):
            self.through(leg, speed=self.dps("free_dps"))

    def way_into(self, here: dict, spot: np.ndarray) -> list[dict] | None:
        """carry_to over `spot`, BOX_DROP_OVER over the walls, if J1 turns there."""
        axis = np.asarray(self.AXIS)
        j1 = 90 - math.degrees(math.atan2(spot[1] - axis[1], spot[0] - axis[0]))
        if not 0 <= j1 <= 180:
            return None
        return carry_to(here, float(np.hypot(*(spot - axis))), j1, BOX_INSIDE[2] + BOX_DROP_OVER, self.carry_high())

    def carry(self, way: list[dict]) -> None:
        self.through(way, speed=self.dps("free_dps"))

    def let_go(self) -> None:
        """Open RELEASE_SPARE_MM over what is held (wide open when not known), and a moment for it to fall."""
        size = getattr(self, "held_size", None)
        opened = m3arm.GRIPPER_OPEN if size is None else opened_for(size, RELEASE_SPARE_MM)
        self.call("set_gripper", opened, speed_dps=self.dps("gripper_dps"), settle=False)
        time.sleep(self.cfg["release_s"])

    def put_back(self, back) -> None:
        """What is held put back down where it was picked (`back`: the base's offset then, the pick's
        line, pre-grasp to grip), let go, and back up the line."""
        if back is None or back[1] is None:
            raise Refused(np.zeros(2), "refused: holding something with nowhere to put it -- take it off by hand")
        offset, line = back
        with self.step("put back"):
            if float(np.linalg.norm(offset - self.offset)) > 0.005:   # where the base was when it picked it
                self.base(*(offset - self.offset))
            shut = [{j: v for j, v in q.items() if j != m3arm.GRIPPER} for q in line]   # the gripper as it is
            self.go(shut[0], speed=self.dps("near_dps"), holding=True)
            self.through(shut[1:], floor_clear=self.arm_clear(), speed=self.dps("descend_dps"))
            self.call("set_gripper", line[-1].get(m3arm.GRIPPER, m3arm.GRIPPER_OPEN), floor_clear=self.arm_clear(),
                      speed_dps=self.dps("gripper_dps"), settle=False)   # (2026-10-05: at the arm's own
            # 1.5 cm, refused at the bottom -- the job stopped holding it)
            time.sleep(self.cfg["release_s"])
            self.through(list(reversed(line[:-1])), floor_clear=self.arm_clear(), speed=self.dps("lift_dps"))
        self.say("put back where it was picked")

    # -------------------------------------------------------------------------- planning --
    def plan(self, thing: dict, around: list[dict]):
        """plan_grasp for `thing` as it will be picked: square to it, the fingers clear of `around` --
        where that has none, the careful grip; each at the fingertips' first height, then -- nothing
        planned there -- the least (floors); thing["careful"], thing["floor"] say which."""
        planned = None
        for floor in self.floors():
            self.floor_m = floor
            thing["careful"] = False
            planned = self.grasp(thing, around)
            if planned is None:
                planned = self.grasp(thing, around, careful=True)
                thing["careful"] = planned is not None
            if planned is not None:
                thing["floor"] = floor
                break
        self.floor_m = None
        return planned

    def grasp(self, thing: dict, around: list[dict], centre=None, prefer=None, whole: bool | None = None,
              careful: bool = False):
        """plan_grasp for `thing` at `centre` (its own by default; `prefer`, `whole`: plan_grasp's,
        wrist's), the fingers opened SPARE_MM over it -- or, `careful`, CAREFUL_SPARE_MM, the aim
        CAREFUL_SHIFTS further from J1's axis along a line they close on (either of a cube's two):
        thing["shift"] says how far."""
        centre = np.asarray(thing["centre"] if centre is None else centre, float)
        opened = opened_for(thing["size"], CAREFUL_SPARE_MM if careful else SPARE_MM)
        kwargs = {"prefer": prefer, "floor_clear": self.floor_clear(), "wrist": self.wrist(thing, around, whole, opened),
                  "gripper": opened, "held": m3arm.gripper_angle(thing["size"] * 1000)}
        if not careful:
            return plan_grasp(centre, thing["height"], **kwargs)
        out = centre[:2] - np.asarray(self.AXIS)
        for shift, turn in itertools.product(CAREFUL_SHIFTS, np.arange(thing["yaw"], thing["yaw"] + 180 - 1e-6, thing["period"])):
            along = np.array([math.cos(math.radians(turn)), math.sin(math.radians(turn))])
            along = along if along @ out >= 0 else -along
            planned = plan_grasp(np.r_[centre[:2] + shift * along, centre[2]], thing["height"], **kwargs)
            if planned is not None:
                thing["shift"] = float(shift)
                return planned
        return None

    def approach(self, refused: list[dict], found: list[dict], kit: list[dict], at):
        """The base slid right clear of the right lidar (sidestep), else driven towards the nearest (nudge)
        -- and never into the drop-off box or the line it has dropped onto (obstacles): the box's black
        walls give the depth nothing, so nudge would not see it in the way (2026-10-21, the fetch)."""
        move = self.sidestep(refused, found, at)
        if move:
            return move, "to reach it with the arm clear of the right lidar"
        move = nudge(found, kit, at, self.cfg["nudge_cm"] / 100, self.right_room())
        obstacles = self.obstacles()
        if move is None or not obstacles:
            return move, "to bring it in reach"
        dx, dy = move   # shortened till the base's footprint keeps DRIVE_GAP from them, half way and at the end

        def clear(dx):
            return all(footprint_gap(self.offset + np.array([dx, dy]) * t, 0.0, obstacles, self.FOOTPRINT) >= DRIVE_GAP
                       for t in (0.5, 1.0))
        while dx > 0 and not clear(dx):
            dx = round(dx - 0.005, 3)
        dx = max(dx, 0.0)
        return (None if not clear(dx) or (dx < 0.01 and abs(dy) < 0.01) else (dx, dy)), "to bring it in reach"

    def sidestep(self, refused: list[dict], found: list[dict], at=None) -> tuple[float, float] | None:
        """(0, -slide) m: the base slid right so far that one of `refused` on the right -- in reach,
        no grip keeping the arm off the lidar -- has a plan (the nearest first, or the one nearest
        `at`); None when no slide right_room() allows does it."""
        most = self.right_room()
        if most < SIDESTEP:
            return None
        start = at if at is not None else np.zeros(2)
        for o in sorted((o for o in refused if o["centre"][1] < 0), key=lambda o: float(np.linalg.norm(o["centre"][:2] - start))):
            slide = slide_for(o, found, most, min(self.floors()), self.plan)
            if slide is not None:
                return 0.0, -round(slide, 3)
        return None

    def why_not(self, kit: list[dict], found: list[dict]) -> str:
        """Why none of `kit` is picked from here, each: out of reach (no grip at all reaches it,
        square or not -- 2026-10-02: two at the very edge were blamed on the fingers); at the edge of
        the arm's reach, its middle past it, and how much closer the robot would have to come
        (2026-10-23: a cube 24.0 cm out on the left, its middle reached to 23.2, was "too close to the
        robot's body"); too near the right lidar (a plan from further left: the slide that would do);
        in reach but too close to the robot's body (2026-10-05: a cube beside the left wheel, the
        elbow over the arm's base, was "out of reach"); else the square grip."""
        base, floor = np.asarray(self.AXIS), min(self.floors())   # the lowest a plan tries
        far, lidar, edge, body, awkward = [], [], [], [], []
        for o in kit:
            out = float(np.hypot(*(o["centre"][:2] - base)))
            if out > furthest(o["height"]) + 0.005:
                far.append(o)
            elif plan_grasp(o["centre"], o["height"], floor_clear=floor) is not None:
                awkward.append(o)
            elif out > furthest(max(o["centre"][2], GRIP_Z_MIN)):   # its middle past the arm's reach, either
                # side (2026-10-21: a 40 mm cube 24.6 cm out, its middle reached to 23.5, "near the lidar")
                edge.append((o, closer_for(o, floor)))
            else:
                slide = slide_for(o, found, SIDESTEP_MOST, floor) if o["centre"][1] < 0 else None
                if slide is None:
                    body.append(o)
                else:
                    lidar.append((o, slide))
        parts = [f"out of reach: {describe(far)}"] if far else []
        if edge:
            parts.append("at the edge of the arm's reach (only its top in reach from here): " + "; ".join(
                describe([o]) + (f" -- the robot {100 * closer:.0f} cm closer would reach it" if closer else "")
                for o, closer in edge) + ("" if self.may_drive else " (may drive closer lets the base drive up)"))
        for o in body:   # the guard's own words for the grip from above
            grip = np.r_[o["centre"][:2], max(o["centre"][2], GRIP_Z_MIN)]
            poses = (m3kinematics.reach(grip, pitch, floor_clear=floor, guard=False) for pitch in (90, 70, 50))
            why = next((m3kinematics.problem(q, floor) for q in poses if q is not None), None)
            parts.append(f"in reach but too close to the robot's body for the arm: {describe([o])}" + (f" ({why})" if why else ""))
        if lidar:
            room = self.right_room()
            how = ("may drive closer lets the base slide right" if not self.may_drive
                   else f"the base may slide only {100 * room:.0f} cm right" + (", the drop-off box is there" if room < self.cfg["right_cm"] / 100 else ""))
            parts.append(f"too near the right lidar for the arm (it keeps {100 * m3kinematics.LIDAR_ROOM:.1f} cm off it), "
                         + "; ".join(f"{describe([o])} -- picked from {100 * slide:.0f} cm further left" for o, slide in lidar)
                         + f" ({how})")
        if awkward:
            parts.append("no square grip from here, a finger would come down on the table or over the bumper "
                         f"(turned so, at this distance): {describe(awkward)}")
        return "; ".join(parts)
