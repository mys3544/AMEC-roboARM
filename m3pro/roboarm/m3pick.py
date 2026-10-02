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
  5. the drop-off, the X3 Plus's (roboarm/config.py DROP_POSE): the base a full 90 deg to the right
     (J1 180), the fingertips DROP_OUT out and DROP_UP above the floor -- the thing falls the last
     bit -- and a random 0..DROP_SPREAD further out along that line so they do not land on each
     other; up over the chassis and turned there, never swung across it low with something in hand;
  6. the wide look again: nothing where it stood = picked -- and it is the next pick's wide look.

Every move is one smooth command, or a glide through several (M3Arm.move_to / follow), quicker in
the open and slower near the floor -- the speeds and pauses are SETTINGS, changed from the console;
each step's time goes to roboarm.metrics. Where the base drove up to something out of reach, it
drives back to where the job began before each drop: the drop-off box stays there.

The arm picks off the floor 16-27 cm ahead of base_link (3.5-14 cm before the bumper; nearer, the
hand cannot point down steeply enough), less far aside: further out the shoulder (J2) would have to
lean past its end at 0 (2026-10-01).

ROS-free: the arm is whatever `call(method, *args, **kwargs)` reaches -- roboarm.web.bridge.Bridge
in the bridge itself (tools/m3bridge.py serves /pick/*), or the bridge over HTTP from the laptop
(tools/m3_pick.py) -- and the camera whatever `snapshot()` returns: (colour BGR, depth mm aligned to
it, K, the believed joint pose). Every step is timed (roboarm.metrics: one run per job).
"""

from __future__ import annotations

import collections
import contextlib
import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from roboarm import m3arm, m3kinematics, m3vision, metrics

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
FLOOR_CLEAR_PLACE = 0.003   # m the fingertips stay above the floor on the descent (the floor just measured):
                            # the "fingertips_mm" setting's default. 8 for a day (2026-10-02, they had touched
                            # the table) -- it was the turned wrist's lower finger the model did not have;
                            # with both fingers in it, back to 3 (the user's call): blocks turned far, further
                            # out, had no square grip at 8
OPEN_MM = 55        # the fingers apart at the wide look; at the pre-grasp opened_for() the target
SPARE_MM = 25       # the opening over the target's width (2026-10-01: a 40 mm cube -- 57 mm across
                    # its diagonal -- missed at 55; it opens to ~67 now, the gripper's widest)
FINGER_M = 0.008    # m a fingertip reaches past the opening's edge (its own thickness and a margin)
DROP_OUT, DROP_UP, DROP_SPREAD = 0.189, 0.110, 0.100   # m: the X3 Plus's drop-off (config.DROP_POSE)
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
    "right_cm": ("driving right, at most (the drop-off box is there)", "cm", 0, 15, 1, 3),
}
MATCH_M = 0.03   # the close look's thing is the wide look's when within this
MAX_MISSES = 2   # clean_up gives a spot up after this many misses in a row
APPROACH_X = 0.21    # m ahead of base_link a thing out of reach is driven to (and to y 0): mid-reach
BUMPER_X = 0.124     # m: the chassis' front edge (m3kinematics), and what the drive keeps clear of it
BUMPER_GAP = 0.04
APPROACHES = 3       # drives towards things before a pick gives up


class Stopped(Exception):
    """The job was stopped (pick_stop): the arm holds where it was."""


class NothingToPick(Exception):
    """No kit thing in view, or none in reach."""


class Refused(Exception):
    """The arm refused a move to it (before moving): the thing at `spot` is left for now."""

    def __init__(self, spot: np.ndarray, why: str):
        super().__init__(why)
        self.spot, self.why = np.asarray(spot[:2], float), why


@contextlib.contextmanager
def refusable(spot):
    """The arm's guard refusing a move -- always before it moves -- as Refused(spot)."""
    try:
        yield
    except Stopped:
        raise
    except Exception as exc:   # the bridge's ArmError, or the HTTP wrapper's error
        if "refused" not in str(exc):
            raise
        raise Refused(spot, str(exc)) from exc


# ------------------------------------------------------------------------------------- geometry --
def name(o: dict) -> str:
    if o["kind"] == "cube":
        return f"{o['size'] * 1000:.0f} mm cube"
    return "block " + ("lying" if o["length"] > o["size"] else "standing")


def describe(found: list[dict]) -> str:
    return "; ".join(f"{name(o) if o['kind'] else 'something %.0f mm tall' % (o['top'] * 1000)}"
                     f" at x {o['centre'][0]:.3f} y {o['centre'][1]:+.3f}, turned {o['yaw']:.0f} deg"
                     for o in found) or "nothing"


def opened_for(size_m: float) -> int:
    """The gripper angle that opens the fingers SPARE_MM wider than a thing `size_m` across --
    beyond Yahboom's table (60 mm), its widest, GRIPPER_OPEN (~67 mm)."""
    width = size_m * 1000 + SPARE_MM
    return m3arm.gripper_angle(width) if width <= max(m3arm.GRIPPER_WIDTH) else m3arm.GRIPPER_OPEN


def opening_m(angle: int) -> float:
    """m between the fingers at a gripper angle (m3kinematics.finger_span)."""
    return m3kinematics.finger_span(angle)


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


def wrist_for(pose: dict, yaw: float, period: float = 90, opening: float = OPEN_MM / 1000,
              others: list[dict] = ()) -> int:
    """J5 that squares the fingers with the object: they close along the wrist's y axis (the
    finger links sit either side of it, URDF), at J5 90 square to the arm's own plane -- with the
    arm turned to reach a cube that sits square to the robot, 25 deg off its faces: they caught its
    corners and shoved it (2026-10-01). `yaw` is a line the fingers may close along, the same every
    `period` deg (90 a cube, 180 across a lying block): the smallest turn -- but of the ways that
    square them (a cube's two), the one whose open fingers keep furthest from `others` (2026-10-01:
    fingers opening left and right of a cube hit a 40 mm one 6.5 cm beside it)."""
    def mismatch(j5):
        y = m3kinematics.frames({**pose, 5: j5})["arm5"][:3, 1]
        closing = math.degrees(math.atan2(y[1], y[0]))
        return abs((closing - yaw + period / 2) % period - period / 2)
    best = min(range(181), key=lambda j5: (round(mismatch(j5)), abs(j5 - 90)))
    if not others:
        return best
    squared = [j5 for j5 in {best, best - 90, best + 90} if 0 <= j5 <= 180 and round(mismatch(j5)) <= round(mismatch(best)) + 1]
    return max(squared, key=lambda j5: (min(finger_clearance({**pose, 5: j5}, opening, others), 0.03), -abs(j5 - 90)))


def off_square(pose: dict, yaw: float, period: float = 90) -> float:
    """deg the fingers at `pose` close off the line `yaw` (the same every `period` deg)."""
    y = m3kinematics.frames(pose)["arm5"][:3, 1]
    return abs((math.degrees(math.atan2(y[1], y[0])) - yaw + period / 2) % period - period / 2)


WRIST_SLACK = {90: 20, 180: 12}   # deg the fingers may close off square, by the thing's period: a 30 mm
                                  # cube 20 deg off is 40 mm across, a 30 x 60 block 12 deg off 42 --
                                  # both inside a 55 mm opening, and the closing fingers square them


def wrist_choices(pose: dict, yaw: float, period: float = 90, opening: float = OPEN_MM / 1000,
                  others: list[dict] = ()) -> list[int]:
    """J5s to try, best first: wrist_for()'s square one, then, a step at a time, off square towards
    the wrist straight (J5 90) up to WRIST_SLACK -- a turned wrist tilts the open fingers, one
    lower, and with the hand not straight down that one can reach the table (2026-10-02)."""
    best = wrist_for(pose, yaw, period, opening, others)
    step = 5 if best < 90 else -5
    slack = WRIST_SLACK.get(int(period), 0)
    return [best] + [best + step * k for k in range(1, slack // 5 + 1) if (best + step * k - 90) * step < 0]


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


def line(a: np.ndarray, b: np.ndarray, pitch: float, floor_clear: float, step: float = 0.02):
    """IK poses along the straight line a -> b, every `step`, the hand at `pitch` over the last
    EXACT_M and up to 15 deg off it before that -- at the far edge the shoulder is at its end and the
    elbow runs into its own (J3 0) going back up the line at one pitch (2026-10-01); None if an end,
    or two points in a row, are out of reach."""
    n = max(1, math.ceil(np.linalg.norm(b - a) / step))
    out, skipped = [], False
    for i in range(n, -1, -1):   # from the grip back up: each point's pitch near the one below it
        point = a + (b - a) * i / n
        last = pitch if not out else out[-1][1]
        tilts = (0,) if np.linalg.norm(b - point) <= EXACT_M + 1e-9 else (0, *TILTS)
        for tilt in tilts:
            if abs(last + tilt - pitch) > 15:
                continue
            p = m3kinematics.reach(point, last + tilt, floor_clear=floor_clear)
            if p is not None:
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
    `wrist(grip pose) -> J5 or [J5s, best first]` and `gripper` (J6): the plan as it will be driven
    -- the turned wrist tilts the open fingers, one lower (m3kinematics.outline): checked with them,
    set in the poses; `held` (J6 shut on the thing): the way back up checked with that (2026-10-02:
    up with the open fingers passed, with them shut on a 40 mm cube the arm refused)."""
    base = np.asarray(m3kinematics.CHAIN[0][0][:2])
    yaw = math.atan2(centre[1] - base[1], centre[0] - base[0])
    radius = float(np.hypot(*(np.asarray(centre[:2]) - base)))
    tries = [(back, lift, pitch) for back in BACKS for lift in LIFTS for pitch in PITCHES + FINE]
    if prefer is not None:
        tries.insert(0, (BACK, prefer[1], prefer[0]))
    hand = {} if gripper is None else {6: gripper}
    reached, lines = {}, {}   # (grip, pitch) -> the grip pose; (back, lift, pitch) -> the line: both passes
    # square first, everywhere -- every approach and grip height -- and only then off square: a
    # cube gripped 20 deg off turns in the fingers, and the grip was luck (2026-10-02)
    for squared in ((True, False) if wrist is not None else (True,)):
        for back, lift, pitch in tries:
            z = max(centre[2], GRIP_Z_MIN) + lift
            if (top is not None and z > top + 1e-6) or radius > furthest(z) + 0.005:   # no pitch reaches it
                continue
            grip = np.array([centre[0], centre[1], z])
            if (lift, pitch) not in reached:   # the grip first: cheap
                reached[lift, pitch] = m3kinematics.reach(grip, pitch, floor_clear=floor_clear)
            bottom = reached[lift, pitch]
            if bottom is None:
                continue
            j5s = wrist(bottom) if wrist is not None else [bottom[5]]
            j5s = j5s if isinstance(j5s, list) else [j5s]
            j5s = j5s[:1] if squared else j5s[1:]
            j5s = [j5 for j5 in j5s   # at the grip a turned finger is lowest: the line only if one fits
                   if m3kinematics.problem({**bottom, 5: j5, **hand}, floor_clear) is None]
            if not j5s:
                continue
            if (back, lift, pitch) not in lines:
                r = math.radians(pitch)
                along = np.array([math.cos(yaw) * math.cos(r), math.sin(yaw) * math.cos(r), -math.sin(r)])
                lines[back, lift, pitch] = line(grip - back * along, grip, pitch, floor_clear)
            poses = lines[back, lift, pitch]
            if not poses:
                continue
            for j5 in j5s:
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


def drop_pose(extra: float) -> dict | None:
    """The drop-off pose, `extra` m further out than DROP_OUT along the line 90 deg right."""
    base = np.asarray(m3kinematics.CHAIN[0][0][:2])
    target = (base[0], base[1] - (DROP_OUT + extra), DROP_UP)
    for pitch in (70, 60, 50, 80, 40):
        pose = m3kinematics.reach(target, pitch)
        if pose:
            return pose
    return None


# The drop-off box (2026-10-02): a printed marker inside its lid, which stands up at the far side --
# the box found from it with one look to the right, empty or full (its black walls give the depth
# camera nothing; the marker reads from 28 cm, 109 px). The inside: `along` the lid's hinge, `across`
# from the hinge towards the robot, the walls' height; drops keep BOX_MARGIN from the walls and let
# go BOX_DROP_OVER over their tops.
BOX_MARKER = ("DICT_4X4_50", 10, 0.064)   # family, id, side (m, measured: 64.0-64.7 from the depth)
BOX_INSIDE = (0.15, 0.11, 0.038)          # m: along, across, walls
BOX_MARGIN = 0.025
BOX_DROP_OVER = 0.04
SIDE_LOOK = {1: 180, 2: 95, 3: 30, 4: 0, 5: 90}   # the look at the box: lid, marker and inside in view
CARRY_OVER = 0.02   # m the carry swings round above the drop-off's height
ARC_STEP = 15       # deg of J1 between the carry's points


def drop_at(radius: float, height: float, j1: float) -> dict | None:
    """The grip point `radius` m from J1's axis, `height` m up, the arm turned to `j1`."""
    base = np.asarray(m3kinematics.CHAIN[0][0][:2])
    yaw = math.radians(90 - j1)   # J1 is mirrored: bigger turns right
    target = (base[0] + radius * math.cos(yaw), base[1] + radius * math.sin(yaw), height)
    for pitch in (70, 60, 50, 80, 40, 65, 55, 75, 45, 35, 85):   # at full reach whole degrees leave gaps
        pose = m3kinematics.reach(target, pitch)
        if pose:
            return pose
    return None


def carry_arc(here: dict, extra: float) -> list[dict] | None:
    """Holding something, from `here` to the drop-off line's spot `extra` m further out than
    DROP_OUT (carry_to it, J1 180)."""
    return carry_to(here, DROP_OUT + extra, 180.0, DROP_UP)


def carry_to(here: dict, radius: float, j1: float, release: float) -> list[dict] | None:
    """Holding something, from `here` to `radius` m from J1's axis with the arm turned to `j1`,
    let go `release` m up: up and out to that distance in the arm's own direction, CARRY_OVER over
    the release height, round at that height and distance -- never over the chassis (its front
    corner is 16.2 cm from J1's axis; the drop-offs are further out) -- and down the last bit (the
    user's, 2026-10-02: rather than folded high and down again). None when the guard would not."""
    high = release + CARRY_OVER
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


def box_from_marker(centre: np.ndarray, axes: np.ndarray) -> dict:
    """The box's inside from its marker (find_marker): the hinge where the lid's plane meets the
    walls' top, the inside from there towards the robot -- {"centre" (x, y), "along", "into" (unit
    x, y), "turn" (deg, of "along")}."""
    _along_m, across_m, wall = BOX_INSIDE
    face, right, up = axes[:, 2], axes[:, 0], axes[:, 1]
    hinge = centre - up * (centre[2] - wall) / up[2]
    into = np.array([face[0], face[1]]) / np.hypot(face[0], face[1])
    along = np.array([right[0], right[1]]) / np.hypot(right[0], right[1])
    middle = hinge[:2] + into * across_m / 2
    return {"centre": middle, "along": along, "into": into, "turn": math.degrees(math.atan2(along[1], along[0]))}


def box_corners(box: dict, lift: float = 0.0) -> list[np.ndarray]:
    along_m, across_m, wall = BOX_INSIDE
    return [np.r_[box["centre"] + box["along"] * a * along_m / 2 + box["into"] * b * across_m / 2, wall + lift]
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def box_spots(box: dict, step: float = 0.02) -> list[np.ndarray]:
    """Spots inside the box to let go over, BOX_MARGIN clear of its walls, `step` m apart."""
    along_m, across_m, _ = BOX_INSIDE
    a_half, b_half = along_m / 2 - BOX_MARGIN, across_m / 2 - BOX_MARGIN
    return [box["centre"] + box["along"] * a + box["into"] * b
            for a in np.arange(-a_half, a_half + 1e-9, step) for b in np.arange(-b_half, b_half + 1e-9, step)]


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


def nudge(found: list[dict], kit: list[dict], at=None, most: float = 0.10, right: float = 0.03) -> tuple[float, float] | None:
    """(forward, left) m to drive so that the kit thing nearest the robot (or `at`) comes to
    APPROACH_X ahead, straight ahead -- forward and sideways only, at most `most` each way and
    `right` to the right (the drop-off box is on that side), never so far that anything in the
    robot's way ends up within BUMPER_GAP of the bumper; None when that is under a centimetre."""
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


def mark(image: np.ndarray, K, camera: np.ndarray, found: list[dict], target: dict | None = None) -> np.ndarray:
    """The picture with what was found: kit green (the target thicker), the rest orange."""
    marked = image.copy()
    inv = np.linalg.inv(camera)
    for o in found:
        top = inv @ np.r_[o["centre"][:2], o["top"], 1.0]
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
class Picker:
    """One job at a time (pick, clean_up, home, drop), on a thread of its own; what it is doing,
    its log and the last look's marked picture for whoever watches (the bridge's /pick/*)."""

    def __init__(self, call, snapshot, out: Path | None = None, settle_s: float | None = None, rng=None,
                 drive=None, battery=None):
        self.call, self.snapshot, self.out = call, snapshot, out
        self.battery = battery      # () -> volts, or None: the board's reading, without waiting for the arm
        self.settle_s = settle_s    # None: the settings' (the tests set 0)
        self.drive = drive          # drive(dx, dy, speed m/s) the base, m in its own frame -> what it went; None: it cannot
        self.may_drive = False      # per job: what the caller allowed
        self.box: dict | None = None   # the drop-off box this job (box_look), and where it has let go into it
        self.box_drops: list[np.ndarray] = []
        self.offset = np.zeros(2)   # m the base has driven since the job began (forward, left): the drop-off
                                    # box stays where it was, so the base goes back before a drop
        self.cfg = {key: spec[5] for key, spec in SETTINGS.items()}
        saved = None if out is None else out / "settings.json"
        if saved is not None and saved.exists():
            with contextlib.suppress(ValueError, OSError):
                self.set_settings(json.loads(saved.read_text()), save=False)
        self.rng = rng or np.random.default_rng()
        self.frame: bytes | None = None   # the last look, marked, as JPEG
        self.found: list[dict] = []
        self.state = {"job": None, "step": "idle", "since": time.time(), "busy": False, "error": None,
                      "last": None, "picked": 0, "missed": 0, "looks": 0}
        self._log: collections.deque = collections.deque(maxlen=300)
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ for the watchers --
    def say(self, text: str) -> None:
        with self._lock:
            self._seq += 1
            self._log.append({"n": self._seq, "t": round(time.time(), 2), "text": text})
        print(text, flush=True)
        if self.out is not None:   # a restart loses the one above
            with contextlib.suppress(OSError), open(self.out / "log.txt", "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text}\n")

    def log(self, since: int = 0) -> list[dict]:
        with self._lock:
            return [e for e in self._log if e["n"] > since]

    def status(self) -> dict:
        with self._lock:
            state = dict(self.state)
        state["now"] = time.time()   # the robot's clock, for how long the step has run
        volts = self.battery() if self.battery is not None else None
        state["battery_v"] = None if volts is None else round(float(volts), 2)
        state["moved_cm"] = [round(100 * float(v), 1) for v in self.offset]
        state["box"] = None if self.box is None else {
            "x": round(float(self.box["centre"][0]), 3), "y": round(float(self.box["centre"][1]), 3),
            "turn": round(self.box["turn"], 1), "drops": len(self.box_drops), "in_it": len(self.box.get("items", ()))}
        state["found"] = [{"kind": o["kind"], "name": name(o) if o["kind"] else None,
                           "x": round(float(o["centre"][0]), 3), "y": round(float(o["centre"][1]), 3),
                           "yaw": round(o["yaw"], 1), "reach": o.get("reach")} for o in self.found]
        return state

    def settings(self) -> list[dict]:
        return [{"key": key, "label": spec[0], "unit": spec[1], "min": spec[2], "max": spec[3], "step": spec[4],
                 "default": spec[5], "value": self.cfg[key]} for key, spec in SETTINGS.items()]

    def set_settings(self, values: dict, save: bool = True) -> list[dict]:
        """Change settings (unknown keys refused, values held to their range); they count from the
        next move on, and are kept for the next start."""
        for key, value in values.items():
            if key not in SETTINGS:
                raise ValueError(f"no setting {key!r}")
            lo, hi = SETTINGS[key][2], SETTINGS[key][3]
            self.cfg[key] = min(max(float(value), lo), hi)
        if save and self.out is not None:
            self.out.mkdir(parents=True, exist_ok=True)
            (self.out / "settings.json").write_text(json.dumps(self.cfg, indent=1))
        return self.settings()

    def wrist(self, thing: dict, around: list[dict], whole: bool | None = None):
        """grip pose -> J5 for `thing`: square to it, the fingers clear of `around` -- or straight
        (90) while its turn is not known (seen in part)."""
        if not (thing["whole"] if whole is None else whole):
            return lambda grip: [90]
        others = [o for o in around if o is not thing]
        span = opening_m(opened_for(thing["size"]))
        return lambda grip: wrist_choices(grip, thing["yaw"], thing["period"], span, others)

    def floor_clear(self) -> float:
        """m the fingertips keep above the measured floor on the descent: grips rise to keep it."""
        return self.cfg["fingertips_mm"] / 1000

    def arm_clear(self) -> float:
        """What the arm's guard is asked to keep on the descent: the plan's, less GUARD_SLACK."""
        return max(self.floor_clear() - GUARD_SLACK, m3kinematics.MIN_FLOOR_CLEAR)

    def dps(self, key: str) -> float:
        return self.cfg[key] * self.cfg["speed"] / 100

    def stopping(self) -> bool:
        """A stop is under way: only while a job runs -- a STOP pressed with nothing running must
        not abandon the next move someone else makes through the bridge."""
        return self._stop.is_set() and self.state["busy"]

    # --------------------------------------------------------------------------- jobs --
    def start(self, job: str, **kwargs) -> dict:
        """Start `job` ("pick_once", "clean_up", "home", "drop") on its own thread; an error if one runs."""
        work = {"pick_once": self.pick_once, "clean_up": self.clean_up, "home": self.home, "drop": self.drop}[job]
        with self._lock:
            if self.state["busy"]:
                return {"error": f"busy: {self.state['job']} ({self.state['step']})"}
            self._stop.clear()
            self.state.update(job=job, step="starting", since=time.time(), busy=True, error=None)
        self._thread = threading.Thread(target=self._run, args=(job, work, kwargs), daemon=True)
        self._thread.start()
        return {"ok": True, "job": job}

    def stop(self) -> None:
        self._stop.set()

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self, job: str, work, kwargs: dict) -> None:
        error = None
        with metrics.run(job) as run:
            try:
                work(**kwargs)
                run["status"] = "done"
            except Stopped:
                error, run["status"] = "stopped: the arm holds where it was", "stopped"
            except NothingToPick as exc:
                error, run["status"] = str(exc), "nothing"
            except Exception as exc:  # noqa: BLE001 -- shown to the watcher, whatever it is
                error, run["status"] = f"{type(exc).__name__}: {exc}", "failed"
                if self._stop.is_set():   # a move abandoned by the stop
                    error, run["status"] = "stopped: the arm holds where it was", "stopped"
        if error:
            self.say(error)
        with self._lock:
            self.state.update(step="idle", since=time.time(), busy=False, error=error)
        self._stop.clear()   # it stopped this job only: the arm takes other calls again

    @contextlib.contextmanager
    def step(self, step: str):
        self.check()
        with self._lock:
            self.state.update(step=step, since=time.time())
        with metrics.span(step):
            yield

    def check(self) -> None:
        if self._stop.is_set():
            raise Stopped

    def move(self, pose: dict, floor_clear: float | None = None, speed: float = 20) -> None:
        """One smooth move (M3Arm.move_to: one command, every step of the way checked first)."""
        self.check()
        kwargs = {"speed_dps": speed, "settle": False}
        if floor_clear is not None:
            kwargs["floor_clear"] = floor_clear
        self.call("move_to", {int(j): v for j, v in pose.items()}, **kwargs)
        self.check()

    def go(self, goal: dict, speed: float = 20) -> None:
        """To `goal` the quickest way the guard allows (way_to): straight, or round over the chassis
        (2026-10-01: the look -> a straight-down pre-grasp, straight, passed 17 cm over the arm's base)."""
        here = {int(j): v for j, v in self.call("read").items()}
        self.through(way_to(here, {**here, **{int(j): v for j, v in goal.items()}}), speed=speed)

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
        with metrics.span("detect"):
            model = m3kinematics.frames(pose)["arm4"] @ m3kinematics.camera_optical()
            normal, height, _ = m3vision.floor_plane(depth, K)
            camera = m3vision.level(model, normal, height, m3kinematics.FLOOR)
            found = m3vision.objects(depth, K, camera, floor=(normal, height))
            m3vision.refine_turns(image, K, camera, found)   # the turn from the colour edges, where sure
        base = np.asarray(m3kinematics.CHAIN[0][0][:2])
        for o in found:   # cheap, and hopeful: the pick plans the one it takes in full
            radius = float(np.hypot(*(o["centre"][:2] - base)))
            o["reach"] = bool(o["kind"]) and radius <= furthest(o["height"]) + 0.005
        near = None
        if target_near is not None:
            near = min(found, key=lambda o: float(np.linalg.norm(o["centre"][:2] - target_near[:2])), default=None)
        marked = mark(image, K, camera, found, near)
        cv2.putText(marked, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        with self._lock:
            self.frame = cv2.imencode(".jpg", marked, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            self.found = found
            self.state["looks"] += 1
        if self.out is not None:
            self.out.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%H%M%S")
            cv2.imwrite(str(self.out / f"{stamp}_{tag}.jpg"), marked)
            cv2.imwrite(str(self.out / f"{stamp}_{tag}_depth.png"), depth)
            (self.out / f"{stamp}_{tag}.json").write_text(json.dumps(   # for looking again later
                {"pose": {str(j): v for j, v in pose.items()}, "K": np.asarray(K).tolist()}))
        return camera, found

    # --------------------------------------------------------------------------- doing --
    def engaged(self) -> None:
        """The arm is limp until its first command and its pose unknown: home first, ~3 s."""
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

    def pick_once(self, kind: str | None = None, at=None, drive: bool = False) -> dict:
        """One pick and the drop-off, then home; `drive`: the base may drive up to it."""
        self.may_drive = drive and self.drive is not None
        self.offset = np.zeros(2)
        self.engaged()
        self.box_look()
        try:
            return self.pick(kind, at)[0]
        finally:
            if not self._stop.is_set():
                self.drive_back()
                self.home()

    def wide_look(self) -> list[dict]:
        """To the wide look, the gripper opening on the way, and what it shows."""
        here = {int(j): v for j, v in self.call("read").items()}
        self.through(way_to(here, {**LOOK, m3arm.GRIPPER: m3arm.gripper_angle(OPEN_MM)}), speed=self.dps("free_dps"))
        _camera, found = self.look("wide")
        self.say("wide look: " + describe(found))
        return found

    def pick(self, kind: str | None = None, at=None, found: list[dict] | None = None,
             skip: list[np.ndarray] = ()) -> tuple[dict, list[dict]]:
        """Pick the nearest kit thing in reach (of `kind`, or the one nearest `at`, base x, y m),
        drop it off and look again from the wide look: (the record of the attempt, what that look
        found -- the next pick's wide look). `found`: the wide look already taken, the arm there
        and the gripper open. Things at the `skip` spots are left; a move the arm refuses on the
        way to the grip raises Refused."""
        started = time.monotonic()
        if found is None:
            with self.step("look.wide"):
                found = self.wide_look()
        at = None if at is None else np.asarray(at, float)
        for approach in range(APPROACHES + 1):
            kit = [o for o in found if o["kind"] and (kind is None or o["kind"] == kind)
                   and all(np.linalg.norm(o["centre"][:2] - spot) > MATCH_M for spot in skip)]
            if not kit:
                raise NothingToPick(f"no kit {kind or 'cube or block'} in view")
            target = planned = None
            with metrics.span("plan"):
                for o in sorted((o for o in kit if o["reach"]),
                                key=lambda o: float(np.linalg.norm(o["centre"][:2] - (at if at is not None else [0, 0])))):
                    planned = plan_grasp(o["centre"], o["height"], floor_clear=self.floor_clear(),
                                         wrist=self.wrist(o, found), gripper=opened_for(o["size"]),
                                         held=m3arm.gripper_angle(o["size"] * 1000))
                    if planned:
                        target = o
                        break
                    o["reach"] = False
            if target is not None:
                break
            move = self.may_drive and approach < APPROACHES and \
                nudge(found, kit, at, self.cfg["nudge_cm"] / 100, self.cfg["right_cm"] / 100)
            if not move:
                base = np.asarray(m3kinematics.CHAIN[0][0][:2])
                far = [o for o in kit if float(np.hypot(*(o["centre"][:2] - base))) > furthest(o["height"]) + 0.005]
                awkward = [o for o in kit if o not in far]
                raise NothingToPick("; ".join(
                    ([f"out of reach: {describe(far)}"] if far else [])
                    + ([("no square grip from here, a finger would come too low (turned so, at this distance): "
                         f"{describe(awkward)}")] if awkward else [])))
            with self.step("drive"):
                self.say(f"driving {100 * move[0]:.0f} cm forward, {100 * move[1]:+.0f} cm left, to bring it in reach")
                went = self.base(*move)
                if at is not None:
                    at = at - went
                found = self.wide_look()
        others = [o for o in found if o is not target]
        with self.step("pregrasp"), refusable(target["centre"]):   # the wrist turned, the fingers opened on the way
            self.go(planned[1][0], speed=self.dps("near_dps"))
        with self.step("look.close"):
            _camera, close = self.look("close", target["centre"])
        again = [o for o in close if o["kind"] and np.linalg.norm(o["centre"][:2] - target["centre"][:2]) < MATCH_M]
        reading, yaw_from = turn_of(again[0] if again else None, target)
        # the aim: the close look's -- unless it saw the thing only in part and the wide look whole: a cut
        # outline is grown to size on a guess, 11-14 mm off (2026-10-02), the arm repeats to ~5-9
        aim = again[0] if again and (again[0]["whole"] or not target["whole"]) else target
        centre = aim["centre"]
        if reading is not aim:   # the turn read from the other look: its height
            centre = np.r_[centre[:2], reading["centre"][2]]
        found_how = ("found it" if aim is not target
                     else "seen in part, the wide look aimed" if again else "not found, the wide look kept")
        self.say(f"close look: {found_how}, moved {1000 * np.linalg.norm(centre[:2] - target['centre'][:2]):.1f} mm")
        # what is round it: the close look's, and the wide look's -- the close one sees less round it
        others = [o for o in close if np.linalg.norm(o["centre"][:2] - centre[:2]) > MATCH_M] + others
        replanned = plan_grasp(centre, reading["height"], prefer=(planned[0], planned[2]), floor_clear=self.floor_clear(),
                               wrist=self.wrist(reading, others, whole=True), gripper=opened_for(reading["size"]),
                               held=m3arm.gripper_angle(reading["size"] * 1000))
        if replanned is None:   # at the edge whole servo degrees make it all or nothing (2026-10-02: a
            # plan at y -0.1297, none at -0.1300): the first plan, a few mm off, its wrist as now read
            for j5 in self.wrist(reading, others, whole=True)(planned[1][-1]):
                poses = [{**p, 5: j5} for p in planned[1]]
                shut = [{**p, 6: m3arm.gripper_angle(reading["size"] * 1000)} for p in poses]
                if m3arm.glide_problem(poses[0], poses[1:], self.floor_clear()) is None \
                        and m3arm.glide_problem(shut[-1], shut[-2::-1], self.floor_clear()) is None:
                    replanned = (planned[0], poses, planned[2])
                    break
        if replanned is None:   # the corrected aim, the turn read: no way down the arm allows
            raise Refused(centre, "refused: no way down with the fingers turned as this one needs")
        pitch, poses, _lift = replanned
        j5 = poses[-1][5]
        off = off_square(poses[-1], reading["yaw"], reading["period"])
        self.say(f"the {name(reading)}: fingers along {reading['yaw']:.0f} deg ({yaw_from} look), wrist J5 {j5}"
                 + (f" -- {off:.0f} deg off square: square put a finger on the table" if off > 3 else ""))
        with self.step("descend"), refusable(centre):   # to the line's top, then down it in one glide
            self.go(poses[0], speed=self.dps("near_dps"))
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
        self.drop()
        with self.step("look.after"):   # the next pick's wide look, too
            after = self.wide_look()
        left = [o for o in after if np.linalg.norm(o["centre"][:2] - centre[:2]) < MATCH_M]
        record = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.monotonic() - started, 1),
                  "kind": name(reading), "target": np.round(centre, 4).tolist(), "size_mm": round(reading["size"] * 1000),
                  "yaw_deg": round(reading["yaw"], 1), "yaw_from": yaw_from, "pitch": pitch, "j5": j5,
                  "grip_pose": poses[-1], "close_look_found": bool(again), "picked": not left}
        self._remember(record)
        with self._lock:
            self.state["missed" if left else "picked"] += 1
        if left:
            self.say(f"NOT picked: still there -- {describe(left)}")
        else:
            self.say(f"picked the {name(reading)}: {record['seconds']:.0f} s from the wide look to the next")
        return record, after

    def straight_up(self, speed: float, rise: float = 0.10) -> None:
        """Up from where the hand is, holding what it holds: the grip point straight up 2 cm at a
        time to `rise`, the hand as steep as it is (or 5-10 deg off where that will not do) -- the
        way out the 2026-10-02 recovery took by hand when the way back up the line was refused."""
        here = {int(j): v for j, v in self.call("read").items()}
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

    def drop(self) -> None:
        """What is held to the X3 Plus's drop-off and let go, in one glide: round at just over the
        drop-off's height (carry_arc), or -- where the guard would not have that -- up to a carry
        height and turned there, then out and down; never swung across the chassis low with
        something in hand. Somewhere along the line out on the right, like the X3's (0-DROP_SPREAD
        further out, at random): not a tower on one spot. Back to the wide look from there."""
        self.drive_back()   # the box did not drive with it
        with self.step("drop"):
            here = {int(j): v for j, v in self.call("read").items()}
            into = self.box_way(here) if self.box is not None else None
            if into is not None:
                spot, way = into
            else:
                extra = float(self.rng.uniform(0.0, DROP_SPREAD))
                pose = drop_pose(extra) or drop_pose(0.0)
                way = carry_arc(here, extra) or way_to(here, pose, direct=False)
            self.through(way, speed=self.dps("free_dps"))
            self.call("set_gripper", m3arm.GRIPPER_OPEN, speed_dps=self.dps("gripper_dps"), settle=False)
            time.sleep(self.cfg["release_s"])
        if into is not None:
            self.box_drops.append(spot)
            self.say(f"dropped into the box at x {spot[0]:.3f} y {spot[1]:+.3f}")
        else:
            self.say(f"dropped {1000 * (DROP_OUT + extra):.0f} mm out on the right"
                     + ("" if self.box is None else " -- no spot in the box the arm could reach"))

    def box_way(self, here: dict) -> tuple[np.ndarray, list[dict]] | None:
        """The spot in the box to let go over -- the one furthest from what is in it already and
        from this job's drops, of those the arm reaches -- and the way there; None if none."""
        axis = np.asarray(m3kinematics.CHAIN[0][0][:2])
        taken = list(self.box.get("items", ())) + self.box_drops
        spots = box_spots(self.box)
        self.rng.shuffle(spots)   # ties broken anyhow: not always the same corner
        spots.sort(key=lambda s: -min((float(np.linalg.norm(s - t)) for t in taken), default=1.0))
        release = BOX_INSIDE[2] + BOX_DROP_OVER
        for spot in spots:
            j1 = 90 - math.degrees(math.atan2(spot[1] - axis[1], spot[0] - axis[0]))
            if not 0 <= j1 <= 180:
                continue
            way = carry_to(here, float(np.hypot(*(spot - axis))), j1, release)
            if way is not None:
                return spot, way
        return None

    def box_look(self) -> dict | None:
        """The look to the right at the drop-off box (SIDE_LOOK): where its marker puts it, and what
        stands in it already -- or None, and the drops go on the line out on the right."""
        with self.step("look.box"):
            self.go(SIDE_LOOK, speed=self.dps("free_dps"))
            time.sleep(self.settle_s if self.settle_s is not None else self.cfg["settle_wide_s"])
            image, depth, K, pose = self.snapshot()
            model = m3kinematics.frames(pose)["arm4"] @ m3kinematics.camera_optical()
            try:
                normal, height, _ = m3vision.floor_plane(depth, K)
                camera = m3vision.level(model, normal, height, m3kinematics.FLOOR)
            except ValueError:
                camera, normal, height = model, None, None
            try:   # the box is a help, never a reason for the job to stop
                seen = m3vision.find_marker(image, K, camera, *BOX_MARKER)
            except Exception as exc:  # noqa: BLE001 -- 2026-10-02: the robot's OpenCV 4.5.4 had no ArucoDetector
                self.say(f"the look for the box failed: {type(exc).__name__}: {exc}")
                seen = None
        self.box_drops = []
        if seen is None:
            self.box = None
            self.say(f"no drop-off box: marker {BOX_MARKER[1]} not seen -- dropping on the line out on the right")
            return None
        box = box_from_marker(*seen)
        if normal is not None:   # what stands in it already
            inside = []
            for o in m3vision.objects(depth, K, camera, floor=(normal, height)):
                rel = o["centre"][:2] - box["centre"]
                if abs(rel @ box["along"]) < BOX_INSIDE[0] / 2 and abs(rel @ box["into"]) < BOX_INSIDE[1] / 2:
                    inside.append(o["centre"][:2])
            box["items"] = inside
        self.box = box
        inv = np.linalg.inv(camera)
        marked = image.copy()
        corners = [K @ ((inv @ np.r_[c, 1.0])[:3] / (inv @ np.r_[c, 1.0])[2]) for c in box_corners(box)]
        cv2.polylines(marked, [np.array([[int(c[0]), int(c[1])] for c in corners])], True, (0, 255, 0), 2)
        cv2.putText(marked, "box", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        with self._lock:
            self.frame = cv2.imencode(".jpg", marked, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            self.state["looks"] += 1
        if self.out is not None:
            cv2.imwrite(str(self.out / f"{time.strftime('%H%M%S')}_box.jpg"), marked)
        self.say(f"drop-off box at x {box['centre'][0]:.3f} y {box['centre'][1]:+.3f}, turned {box['turn']:.0f} deg,"
                 f" {len(box.get('items', ()))} thing(s) in it")
        return box

    def base(self, dx: float, dy: float) -> np.ndarray:
        """Drive the base (m forward, left), at the settings' speed; what it went, added to the
        offset from where the job began; then still for a moment before anyone looks."""
        went = self.drive(dx, dy, self.cfg["drive_cm_s"] / 100)
        went = np.asarray(went if went is not None else (dx, dy), float)
        self.offset = self.offset + went
        time.sleep(self.cfg["driven_s"])
        return went

    def drive_back(self) -> None:
        """Back to where the job began (the drop-off box is there), a move at most at a time."""
        if self.drive is None or float(np.linalg.norm(self.offset)) < 0.005:
            return
        with self.step("drive back"):
            most = self.cfg["nudge_cm"] / 100
            for _ in range(4):
                if float(np.linalg.norm(self.offset)) < 0.005:
                    break
                self.say(f"driving back {100 * self.offset[0]:.0f} cm, {100 * -self.offset[1]:+.0f} cm left, to the start")
                self.base(*np.clip(-self.offset, -most, most))

    def clean_up(self, kind: str | None = None, limit: int = 20, drive: bool = False) -> None:
        """Pick every kit thing in reach, one by one, until none is left in reach (or `limit`);
        each pick's last look is the next one's wide look. A spot missed MAX_MISSES times in a row
        is given up. `drive`: the base may drive up to what is out of reach."""
        self.may_drive = drive and self.drive is not None
        self.offset = np.zeros(2)
        self.engaged()
        self.box_look()
        found, misses, skip = None, [], []
        for _ in range(limit):
            try:
                record, found = self.pick(kind, found=found, skip=skip)
            except NothingToPick as exc:
                self.say(f"clean-up done: {exc}" + (f" ({len(skip)} left: the arm refused the way)" if skip else ""))
                break
            except Refused as exc:   # nothing moved: on with the others, that one left
                self.say(f"left the one at x {exc.spot[0]:.3f} y {exc.spot[1]:+.3f}: {exc.why}")
                skip.append(exc.spot)
                found = None
                continue
            if record["picked"]:
                misses = []
                continue
            misses.append(np.asarray(record["target"][:2]))
            if len(misses) >= MAX_MISSES and np.linalg.norm(misses[-1] - misses[-2]) < MATCH_M:
                self.say("clean-up stopped: the same spot missed twice")
                break
        self.drive_back()
        self.home()

    def _remember(self, record: dict) -> None:
        with self._lock:
            self.state["last"] = record
        if self.out is not None:
            log = self.out / "picks.json"
            records = json.loads(log.read_text()) if log.exists() else []
            records.append(record)
            log.write_text(json.dumps(records, indent=1))
