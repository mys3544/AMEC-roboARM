#!/usr/bin/env python3
"""Fit the pixel -> table mapping from a survey pose, and check it on the arm.

    ... calibrate_table.py                 # fit and save the PRIMARY look
    ... calibrate_table.py --verify        # drive the fingertip to a board corner
    ... calibrate_table.py --look outer    # fit the OUTER look as well (--outer works too)
    ... calibrate_table.py --look near --yaws=-20,-10,0,10,20   # the NEAR look
    ... calibrate_table.py --look outer --verify

THREE LOOKS since 2026-09-22: the NEAR look sees the band the primary's near
edge (129 mm) cuts off, which the arm can grasp in since J3's floor went to 0
(nearest grasp point 122 mm). Its fingertip target is 110 mm out and 110 mm up
with the tool straight down: lens 236 mm up over a nadir 110 mm out, seeing
about 56..166 mm by the model, J2 75 / J3 11 / J4 0, 163 mm clear of the mast.

TWO LOOKS, AND WHY. Yawing the base rotates a calibrated look for free -- the table
is invariant under a rotation about J1, so roboarm/sweep.py turns one calibration
into a ring covering every bearing the arm can reach. What a yaw cannot change is
how FAR the camera looks: the lens sits at a fixed height and tilt, so its footprint
lands on a fixed band of distances however the base is turned.

Measured against the primary look, the sweep can measure a tagged cube anywhere
from 129 to 189 mm of the 131..210 mm the arm can actually grasp in. The missing
outer rim is about a third of the workspace, and no number of extra stations
touches it.

The OUTER look closes it. Its fingertip target is 40 mm further out and 25 mm
lower, which carries the lens 30 mm further from the base (nadir 146 -> 176 mm)
while keeping the tool within 10 degrees of straight down and the fingers 92 mm
above the table. It is a genuine second calibration, not a rotation, so it needs
the board -- but once fitted it lives in the same file and sweep.stations() rings
it exactly like the first.

Fitting needs the board taped flat and square to the robot, at the measured place
(see BOARD_X0 / BOARD_Y0). Verifying is the part that actually proves it: the arm is
sent to a corner the camera picked out, and you say whether the fingertip lands on
it. A homography that fits its own points beautifully can still be wrong if the board
placement is off, and only the arm can tell us that.
"""

import argparse
import math
import sys
import time

import numpy as np

from roboarm import camera, sweep
from roboarm import config as cfg
from roboarm import kinematics as kin
from roboarm import workspace as ws
from roboarm.arm import Arm, ArmError

SURVEY_FWD = 0.150
SURVEY_LIFT = 0.090

# The OUTER look. Chosen 2026-09-10 by searching fingertip targets for the one that
# carries the lens furthest from the base while keeping the tool within 10 degrees
# of straight down (past that the camera starts looking at the room rather than the
# table, which is exactly what makes the mast camera useless), the arm clear of the
# mast, and the fingers at least 75 mm above the table so the ring can sweep over a
# standing cube. This target wins on the balance of those:
#
#     target 190 fwd / 65 up -> lens 188 mm up, nadir 176 mm out, pitch 170,
#                               fingers 92 mm above the table
#
# against 146 mm of nadir for the primary look, so it sees about 30 mm further out
# -- enough to cover the 188..210 mm rim the primary cannot reach, with overlap.
# Pushing harder (195/50) buys 5 mm more but drops the fingers to 77 mm and the lens
# to 173 mm, which shrinks the footprint; not worth it.
# 2026-09-14: raised and tilted out. The 190/65 pose (lens 179 mm up, 10 deg off
# vertical, fitted) lost a 40 mm cube's top out of the frame past ~198 mm -- a low
# lens lifts the top more and sees less table per degree. Searched again with the
# J2 floor at 5 and tilt allowed to 20 deg: 190 fwd / 125 up -> lens ~241 mm up,
# nadir ~155, near edge ~185 (overlaps the primary band), and a 40 mm cube stays
# whole to ~260 mm by the model, past the arm's reach. Fit it with --yaws.
OUTER_FWD = 0.190
OUTER_LIFT = 0.125

# What the outer look is called in the calibration file. sweep.stations() rings
# every look it finds, so the name is only for humans and for re-fitting in place.
OUTER_NAME = "outer"

# The NEAR look, 2026-09-22 (see the module docstring). Searched the same way as
# the outer: fingertip targets 20..150 mm out, 40..230 mm up, tool 150..180 deg,
# inside the safe limits and clear of the mast, ranked by how near the picture's
# near edge lands with the lens at least 150 mm up. 100/110 sees 42..151 mm but
# sits at J3 = 1; 110/110 straight down is the next and keeps J3 at 11.
NEAR_FWD = 0.110
NEAR_LIFT = 0.110
NEAR_NAME = "near"
LOOKS = ("primary", OUTER_NAME, NEAR_NAME)


def survey_pose() -> dict[int, int]:
    """The pose the homography is fitted from. Its exact height does not matter --
    the homography absorbs wherever the camera actually ended up -- but its
    REPEATABILITY does, and so does never changing it, because a saved calibration
    is only valid from the joint angles it was taken at. So this deliberately solves
    with the default (closed) tool length even though it then opens the gripper:
    changing that would move the pose and silently invalidate the saved fit."""
    pose, _pitch = kin.solve(SURVEY_FWD, 0.0, -cfg.TABLE_BELOW_PLATE + SURVEY_LIFT)
    pose[6] = cfg.GRIPPER_OPEN
    return pose


def outer_pose() -> dict[int, int]:
    """The second look, further out. Same reasoning as survey_pose() about the
    closed tool length: solved closed, then the gripper is opened, so that the pose
    never moves when the gripper table is re-measured."""
    pose, _pitch = kin.solve(OUTER_FWD, 0.0, -cfg.TABLE_BELOW_PLATE + OUTER_LIFT)
    pose[6] = cfg.GRIPPER_OPEN
    return pose


def near_pose() -> dict[int, int]:
    """The third look, close in and straight down. Solved with the tool at 180
    (down) rather than solve()'s pitch search, so the picture is as square to
    the table as the arm can hold it; the closed tool length, as above."""
    pose = kin.inverse(NEAR_FWD, 0.0, -cfg.TABLE_BELOW_PLATE + NEAR_LIFT, 180.0)
    pose[6] = cfg.GRIPPER_OPEN
    return pose


def look_pose(name: str) -> dict[int, int]:
    return {"primary": survey_pose, OUTER_NAME: outer_pose, NEAR_NAME: near_pose}[name]()


def _spun(points: np.ndarray, degrees: float) -> np.ndarray:
    """Table points turned about the base by `degrees` (+ = left)."""
    turn = math.radians(degrees)
    spin = np.array([[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]])
    return np.asarray(points, dtype=float).reshape(-1, 2) @ spin.T


def do_fit(arm: Arm, look: str = "primary", yaws: tuple[float, ...] = (0.0,)) -> int:
    """Fit one look. With several `yaws` the look is fitted from the board seen at
    each of those base yaws, pooled.

    WHY POOL. The outer look sits low and close: its picture holds one whole
    marker (four corners, 38 mm across) in a frame that spans 150 mm of table.
    A homography through four points is exactly determined -- zero residual,
    no redundancy -- and its perspective terms are set by the foreshortening
    across one small patch, so it extrapolates badly to the frame's edges.
    Yawing the base by d moves that same marker across the picture (and swings
    others in), and because the table is invariant under a yaw about J1 (see
    roboarm/sweep.py), a pixel that sees table point q at yaw d sees R(-d) q at
    yaw 0. So every frame's correspondences can be turned back into the
    unyawed look and pooled: one fit, points spread across the whole width of
    the picture, and a residual that now also measures J1's repeatability.
    """
    pose = look_pose(look)
    print(f"the {look} look: pose {pose}")
    reached = None
    pooled_px, pooled_tb = [], []
    for index, yaw in enumerate(yaws):
        at = sweep.pose_at(pose, yaw)
        here = arm.move_to(at, speed_dps=15, repeatable=True)
        if yaw == 0.0:
            reached = here
        time.sleep(1.5 if index == 0 else 0.5)
        frames = camera.grab(6)
        pixels, table = ws.average_corners(frames)
        print(f"yaw {yaw:+5.1f}: {len(pixels)} marker corners ({len(pixels) // 4} whole markers)")
        if len(pixels):
            pooled_px.append(pixels)
            pooled_tb.append(_spun(table, -yaw))
    if reached is None:
        reached = arm.move_to(pose, speed_dps=15, repeatable=True)
    # SAVED IS THE COMMAND, not the readback: the sweep drives to the saved pose,
    # and what the arm repeats is a command. With the C930e on the wrist (2026-09-24)
    # the arm settles a degree or two short of it -- commanded J2 56 / J3 24, read
    # back 54 / 22 -- so re-commanding the readback put the arm lower still, and the
    # homography was 7..10 mm out there. The Sonix, lighter, read back within a
    # degree, which is why its files hold readbacks and got away with it.
    print(f"commanded {pose}, read back {reached}; saving the command")
    pixels = np.concatenate(pooled_px) if pooled_px else np.empty((0, 2))
    table = np.concatenate(pooled_tb) if pooled_tb else np.empty((0, 2))
    print(f"{len(pixels)} marker corners in all")
    if len(pixels) < 4:
        print("not enough corners -- move the board so more of it is in view",
              file=sys.stderr)
        return 1

    matrix, worst, n = ws.fit_points(pixels, table)
    frames = camera.grab(1)
    if look != "primary":
        # Appended beside the primary rather than replacing it: the primary is the
        # only look validated end to end to 2 mm, and sweep.best_refine() gives it
        # first refusal for exactly that reason.
        ws.add_look(look, matrix, pose, worst, n)
    else:
        ws.save(matrix, pose, worst, n)
    print(f"fitted the {look} look on {n} points, worst residual {worst * 1000:.1f} mm")
    print(f"board area seen: {table[:, 0].min() * 1000:.0f}..{table[:, 0].max() * 1000:.0f} mm "
          f"forward, {table[:, 1].min() * 1000:+.0f}..{table[:, 1].max() * 1000:+.0f} mm left")

    height, width = frames[0].shape[:2]
    centre = ws.apply(matrix, [[width / 2, height / 2]])[0]
    print(f"image centre maps to {centre[0] * 1000:.0f} mm forward, "
          f"{centre[1] * 1000:+.0f} mm left")
    print(f"\nsaved to {ws.CALIBRATION_PATH}")
    print("Now run with --verify: a fit can look perfect and still be wrong if the")
    print("board placement is off. Only the arm can tell us.")
    return 0


def do_verify(arm: Arm, look: str = "primary") -> int:
    looks = ws.load_all()
    if look != "primary":
        chosen = [entry for entry in looks if entry[2] == look]
        if not chosen:
            print(f"no {look!r} look saved yet -- run --look {look} first", file=sys.stderr)
            return 1
        matrix, saved_pose, _name = chosen[0]
    else:
        matrix, saved_pose, _name = looks[0]
    arm.move_to({int(k): v for k, v in saved_pose.items()}, speed_dps=15, repeatable=True)
    time.sleep(1.5)
    frames = camera.grab(6)
    frame = frames[-1]

    pixels, _table = ws.average_corners(frames)
    if len(pixels) < 1:
        print("no corners visible", file=sys.stderr)
        return 1

    # Pick the corner nearest the image centre that the arm can touch: the
    # centre is the most reliable, least distorted part of the picture, but the
    # near look's centre (about 120 mm out) is inside the nearest grasp point.
    centre = np.array([frame.shape[1] / 2, frame.shape[0] / 2])
    z = -cfg.TABLE_BELOW_PLATE + 0.004  # just clear of the paper
    pose = pitch = target = None
    for pick in np.argsort(np.linalg.norm(pixels - centre, axis=1)):
        candidate = ws.apply(matrix, [pixels[pick]])[0]
        print(f"corner at pixel ({pixels[pick][0]:.0f}, {pixels[pick][1]:.0f})"
              f" -> table {candidate[0] * 1000:.0f} mm forward, {candidate[1] * 1000:+.0f} mm left")
        try:
            pose, pitch = kin.solve(float(candidate[0]), float(candidate[1]), z)
        except kin.Unreachable as exc:
            print(f"  not reachable ({exc}); the next one")
            continue
        target = candidate
        break
    if target is None:
        print("no visible corner is reachable", file=sys.stderr)
        return 1
    pose[6] = cfg.GRIPPER_CLOSED  # closed: fingertips on the axis, easy to eyeball

    print(f"\nmoving there (pitch {pitch:.0f})... watch where the fingertips land.")
    arm.move_to(pose, speed_dps=12)
    time.sleep(1.0)
    tip = kin.forward(arm.read())
    print(f"fingertip now at {tip[0] * 1000:.0f} mm forward, {tip[1] * 1000:+.0f} mm left, "
          f"{(tip[2] + cfg.TABLE_BELOW_PLATE) * 1000:.0f} mm above the table")
    print("\nHolding 45s. How far is the fingertip from that corner, and which way?")
    time.sleep(45)
    arm.move_to({int(k): v for k, v in saved_pose.items()}, speed_dps=15, repeatable=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--look", choices=LOOKS, default="primary",
                        help="which look to fit (or verify): primary, outer or near")
    parser.add_argument("--outer", action="store_true",
                        help="the same as --look outer")
    parser.add_argument("--yaws", default="0",
                        help="comma-separated base yaws (deg, + = left) to pool the fit "
                             "over, e.g. -24,-12,0,12,24 for a look that sees one marker")
    args = parser.parse_args()
    look = OUTER_NAME if args.outer else args.look
    yaws = tuple(float(v) for v in args.yaws.split(",") if v.strip())
    if 0.0 not in yaws:
        yaws = (0.0,) + yaws
    try:
        with Arm() as arm:
            if args.verify:
                return do_verify(arm, look)
            return do_fit(arm, look, yaws)
    except (ArmError, camera.CameraError, ValueError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
