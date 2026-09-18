"""Pick an object up and put it down. The motion half of the demo.

Everything about WHERE comes in as a `detect.Target`; nothing here looks at a camera.
That split is deliberate -- the detector can be swapped for a better one without
touching any code that moves the arm.

The sequence is the standard one, and each step earns its place:

    open the fingers to just wider than the object
    hover 60 mm above it
    descend straight down to grasp height, CORRECTING IN MILLIMETRES as it goes
    close
    ask whether anything is actually held
    lift

The correction on the descent is not decoration. Measured on 2026-09-10, the
joints land within a degree of the pose they were given -- converged, by any
joint-space test -- and the fingertip is 9.9 mm short of the point that was asked
for, because at a top-down pitch one degree at the wrist is 3.2 mm of reach. Two
picks failed on exactly that, nudging the object aside instead of straddling it.
See _reach_to() below, and tools/reach_check.py, which measures it.

Three numbers are the whole design:

  * OPENING. Fully open for the approach (since 2026-09-18). It used to be "just
    wider than the object" to keep the tool long and the fingers tidy, but 6 mm
    a side is less than the alignment error on a bad day, and a finger that lands
    ON the cube sticks there. The price -- 23 mm of closing travel -- is paid by
    descending that much higher; see closing_travel().

  * GRASP_HEIGHT_M. Where the fingertips stop. The arm's height error is up to 4 mm
    and grows with reach (see config.py), so this has to be high enough that 4 mm low
    does not press into the table, and low enough that 4 mm high still leaves the
    fingers beside the object rather than above it. 8 mm gives that margin both ways
    for anything taller than about 15 mm.

  * The PITCH is chosen once, at the grasp point, and reused for the hover. Solving
    the two independently would let solve() pick different pitches for them, and the
    tool would swing through an arc on the way down instead of descending.

TWO RULES ABOUT THE GRIPPER, both about not stalling a servo:

  * No move may command J6 once something is held. Re-sending "closed" to fingers
    already stopped by an object just leans on it again. Poses are stripped to the
    arm joints with `_arm_only()` before being commanded.
  * After a successful close, the command is backed off to just past where the
    fingers actually stopped. Left commanded at 170 against an object that stopped
    them at 130, the servo pulls full current for as long as it holds -- which is how
    the battery went 12.0 V to 10.6 V against the camera mast.

NOT HANDLED, and worth saying out loud rather than discovering during a demo:

  * Gripper YAW. J5 could rotate the fingers to match the object's short axis, but
    J5's zero has never been measured, so the fingers always close along the same
    heading. Round and square objects do not care; a long thin one has to be laid
    across the fingers' closing direction by hand.
  * Tall objects. The wrist camera sticks out 50 mm sideways above the fingertips;
    nothing here checks it against an object more than about 60 mm tall.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import NamedTuple

from roboarm import config as cfg
from roboarm import detect
from roboarm import kinematics as kin
from roboarm.arm import Arm, ArmError
from roboarm.config import Pose

HOVER_M = 0.060
# 2026-09-18: 8 -> 4 mm. The touch probe reads the fingertips ~10 mm HIGHER than the
# model at 176 mm reach (readback 29 mm with the tips on a 40 mm cube), and a 20 mm
# cube was taken by its top corners. Aiming 4 mm puts the real tips near 10-14 mm:
# mid-height of a 20 mm cube, lower third of a 40 mm one.
GRASP_HEIGHT_M = 0.004
LIFT_M = 0.080

# How much wider than the held object the fingers open to RELEASE it (place()).
# The approach itself opens fully -- see plan().
FINGER_CLEARANCE_M = 0.012

# Once an object has stopped the fingers, back the command off to just past where
# they actually are. A bus servo commanded 40 degrees past its stop applies full
# torque for as long as it is held there -- that is how the battery went from 12.0 V
# to 10.6 V against the camera mast. A few degrees of overlap grips just as well and
# draws a fraction of the current.
SQUEEZE_DEG = 6

# Transit above the table, with nothing near the fingers. MEASURED 2026-09-11:
# a base yaw at 40 deg/s lands the camera within 2.5 mm of where 20 deg/s
# does, so 30 is a safe middle for every move that is not the descent.
APPROACH_DPS = 30.0
DESCEND_DPS = 8.0   # slow: this is the step that can touch something
# Putting down: the object is already held, the height is 6 mm above where it
# was picked, and nothing has to be measured on the way -- so faster than the
# pick's descent, slower than a transit.
PLACE_DESCEND_DPS = 12.0
# The fingers. Closing speed does not change the grip -- the servo's torque
# does -- so both directions run at the speed the panel's buttons use.
GRIPPER_DPS = 60


class GraspError(ArmError):
    """The pick could not be attempted, or could not be verified."""


class Plan(NamedTuple):
    """Everything decided before the arm moves."""

    hover: Pose
    grasp: Pose
    opening: int
    pitch: float
    roll: int = 90        # J5: turns the fingers to close across the object's narrow way



# THE TOOL GROWS AS THE FINGERS SHUT, and the arm does not move while they do.
#
# The fingers pivot, so closing them pushes the tips FURTHER from J4 along the tool
# axis -- the same effect cfg.GRIPPER_TOOL_MM records, read in the direction nobody
# had read it in. MEASURED there: 162 mm at J6=30, 190 mm at J6=150. So on a
# top-down grasp, where the tool points at the table, shutting the fingers drives
# the tips DOWNWARD by the difference:
#
#     opened to 70 mm, closing onto a 27 mm object   tool grows 23 mm
#     opened to 66 mm, closing onto a 27 mm object   tool grows 14 mm
#     opened to 57 mm, closing onto a 27 mm object   tool grows  5 mm
#
# plan() used to solve the grasp pose at the OPEN tool length and stop there, so a
# fingertip placed 8 mm above the table finished the close somewhere below it, with
# the fingers scraping the paper instead of closing round the object. The wider the
# opening the worse it got, which is the opposite of the intuition that opening
# wider is safer.
#
# The fix is to descend by the extra distance the tips are about to travel, so that
# they arrive at GRASP_HEIGHT_M when they are SHUT rather than when they are open.
# `closing_travel()` is that distance, and it needs the angle the fingers will
# actually stop at -- which is set by the object, not by GRIPPER_CLOSED.


def closing_travel(opening: int, width_m: float) -> float:
    """How much further the fingertips reach once shut on an object this wide.

    Never negative: the opening is chosen wider than the object, so the angle the
    fingers settle at is always at least the angle they started from.
    """
    settled = cfg.gripper_for_gap(width_m)
    return max(0.0, cfg.tool_length(settled) - cfg.tool_length(opening))


def _arm_only(pose: Pose) -> Pose:
    """The arm joints without the gripper, so a move cannot squeeze what is held."""
    return {joint: angle for joint, angle in pose.items() if joint != cfg.GRIPPER_ID}


def outward(x: float, y: float, offset_m: float) -> tuple[float, float]:
    """(x, y) moved `offset_m` further from the base along its own bearing."""
    radius = math.hypot(x, y)
    if radius < 1e-9 or offset_m == 0.0:
        return x, y
    return x + offset_m * x / radius, y + offset_m * y / radius


def reach_offset(x: float, y: float, base_m: float = cfg.REACH_OFFSET_M) -> float:
    """The reach offset for a point this far out: `base_m` plus the slope beyond
    cfg.REACH_OFFSET_AT_M (see config.py). Zero base means zero, the raw model."""
    if base_m == 0.0:
        return 0.0
    beyond = max(0.0, math.hypot(x, y) - cfg.REACH_OFFSET_AT_M)
    return base_m + cfg.REACH_OFFSET_SLOPE * beyond


def aim_for(target: detect.Target, reach_offset_m: float = cfg.REACH_OFFSET_M) -> detect.Target:
    """Where to SEND the fingertip so that it actually arrives on the target.

    The model lands the tips about cfg.REACH_OFFSET_M short of where it thinks
    they are, along the reach. Aiming beyond the object by that much is what puts
    the fingers either side of it instead of on its near face.
    """
    x, y = outward(target.x, target.y, reach_offset(target.x, target.y, reach_offset_m))
    return replace(target, x=x, y=y)


def _rolled(pose: Pose, roll: int) -> Pose:
    """The same pose with the wrist turned to `roll`. inverse() always answers
    J5=90; the roll is decided once by plan() and every move keeps it, so the
    fingers stay square to the object from hover to lift."""
    return {**pose, 5: int(roll)}


# Objects this close to square get the 90-degree symmetry: a cube does not care
# which pair of faces the fingers take, and the smaller turn is the safer one.
SQUARE_TOLERANCE = 0.15


def _solve_near(x: float, y: float, z: float, pitch: float,
                gripper: int) -> tuple[Pose, float]:
    """Solve for a point, preferring `pitch` but taking the nearest that works.

    Every vertical move in a pick wants to keep the pitch the grasp chose, so the
    tool goes straight up and down instead of swinging through an arc. Insisting on
    it exactly would refuse perfectly good points, though -- the same pitch is not
    reachable at every height -- so the pitches are simply tried nearest-first.
    """
    nearest_first = tuple(sorted(kin.GRASP_PITCHES, key=lambda p: abs(p - pitch)))
    return kin.solve(x, y, z, pitches=nearest_first, gripper=gripper)



# Cartesian correction. MEASURED 2026-09-10 at 176 mm reach with the tool 10
# degrees off vertical: the joints land within ONE degree of what they were told,
# which arm.py rightly calls converged -- and the fingertip is still 6.7 mm short
# of the point that was asked for, because at that pitch a degree at the wrist is
# 3.2 mm of reach and two joints were a degree low.
#
# So joint-space correction cannot fix this, and a droop loop in arm.py provably
# did not: driven open loop and closed loop, the same pose measured 6.7 mm short
# BOTH TIMES, because the 1 degree error is inside any honest deadband. Chasing a
# degree walks a joint about and changes the load on its neighbours, so the
# correction has to happen where the error actually matters, in millimetres at
# the fingertip.
#
# The arm settles at (aimed - shortfall), so to LAND on the goal, aim at
# (goal + shortfall). Aiming at the remaining error instead would forget the
# offset already applied and under-correct on every pass after the first.
#
# Two passes: the first removes almost all of it, the second catches what changing
# the pose changed about the load.
REACH_PASSES = 2

# Stop correcting inside this. The kinematics' own worst residual is 6.8 mm and the
# camera is good to about 2 mm and the IK rounds to whole degrees (3.2 mm at the
# wrist), so pushing below 4 mm is chasing noise the rest of the chain cannot honour.
REACH_TOLERANCE_M = 0.004   # 2026-09-18: was tighter, and a third pass chasing 3.3 mm of
# rounding landed 7.6 mm off -- the IK rounds to whole degrees, 3.2 mm at the wrist


def _reach_to(arm: Arm, x: float, y: float, z: float, pitch: float, gripper: int,
              speed_dps: float, passes: int = REACH_PASSES, roll: int | None = None) -> Pose:
    """Put the FINGERTIP at (x, y, z), correcting in millimetres, not degrees.

    Returns the pose finally commanded. The gripper is never commanded here -- only
    the arm joints move -- so this is safe to run with something already held.
    """
    aim = [x, y, z]
    pose, _used = kin.solve(x, y, z, pitches=(pitch,), gripper=gripper)
    roll = arm.read()[5] if roll is None else roll
    for _ in range(passes + 1):
        arm.move_to(_arm_only(_rolled(pose, roll)), speed_dps=speed_dps)
        time.sleep(0.4)
        landed = kin.forward({**arm.read(), cfg.GRIPPER_ID: gripper})
        short = [goal - got for goal, got in zip((x, y, z), landed)]
        # Narrated, because a miss here is the commonest way a pick fails and the
        # numbers say whether the arm is short, wide or high -- and whether the
        # correction is converging or fighting something.
        print(f"    reach pass: fingertip {math.dist(landed, (x, y, z)) * 1000:.1f} mm off "
              f"(short by {short[0] * 1000:+.1f} fwd, {short[1] * 1000:+.1f} left, "
              f"{short[2] * 1000:+.1f} up)", flush=True)
        if math.dist(landed, (x, y, z)) <= REACH_TOLERANCE_M:
            break
        aim = [a + s for a, s in zip(aim, short)]
        # THE AIM MAY NOT GO BELOW THE TABLE, and it wants to. An arm that lands
        # high is corrected by aiming low, so a 15 mm shortfall at 8 mm asks to aim
        # at -7 mm -- which kin.inverse refuses outright, being under the table.
        # Left to raise, that abandoned the whole correction, SIDEWAYS COMPONENT AND
        # ALL, and quietly returned an arm 17 mm off target: exactly the failure
        # this function exists to prevent, hidden by the fact that it had already
        # worked elsewhere. Clamp to the table surface, which is the most downward
        # correction there is, and let x and y go on being corrected regardless.
        aim[2] = max(aim[2], -cfg.TABLE_BELOW_PLATE)
        try:
            pose, _used = kin.solve(*aim, pitches=(pitch,), gripper=gripper)
        except kin.Unreachable:
            # Holding the pitch is a preference, not a requirement -- it keeps the
            # descent vertical. Near the edge of the workspace it is the thing worth
            # giving up to keep the POSITION, so try the neighbouring pitches before
            # settling for a pose that is metres of degrees from where it should be.
            try:
                pose, _used = _solve_near(*aim, pitch, gripper)
            except kin.Unreachable:
                # Genuinely nowhere left to aim. The pose we are in is the best
                # available, so stop rather than raise: the caller has a usable arm.
                break
    return pose


def plan(target: detect.Target, reach_offset_m: float = cfg.REACH_OFFSET_M) -> Plan:
    """Work out the hover pose, the grasp pose and the opening -- or raise.

    All of it before anything moves, so an unreachable object is a message rather
    than an arm that sets off and stops halfway. The poses aim `reach_offset_m`
    beyond the object (see aim_for); pass 0 to plan for the raw model.
    """
    target = aim_for(target, reach_offset_m)
    if not target.graspable:
        raise GraspError(f"{target.label}: {target.why_not()}")

    # FULLY open for the approach, whatever the object's width (2026-09-18, on
    # the user's observation): the fingers only close once the arm is at the
    # grasp position, and an opening of width + 12 mm left 6 mm a side, which
    # a few mm of alignment error turned into a finger landing ON the cube and
    # sticking. 70 mm leaves 15 mm a side for a 40 mm cube. The longer close
    # costs about a second and 23 mm of closing travel, which the descent
    # height below already allows for.
    opening = cfg.GRIPPER_OPEN
    table = -cfg.TABLE_BELOW_PLATE
    # Aim HIGH by however far the tips are about to travel on their own, so they
    # finish the close at GRASP_HEIGHT_M instead of starting there and burrowing.
    reach_z = table + GRASP_HEIGHT_M + closing_travel(opening, target.width_m)
    try:
        grasp_pose, pitch = kin.solve(
            target.x, target.y, reach_z, gripper=opening
        )
    except kin.Unreachable as exc:
        raise GraspError(f"{target.label} cannot be grasped: {exc}") from exc
    # The hover wants the SAME pitch as the grasp, so the descent is straight down
    # rather than an arc. It is not always available at hover height, so fall back to
    # the nearest pitch that is -- a few degrees of swing is harmless, and refusing
    # the object outright would not be.
    try:
        hover_pose, hover_pitch = _solve_near(
            target.x, target.y, table + HOVER_M, pitch, opening
        )
    except kin.Unreachable as exc:
        raise GraspError(
            f"{target.label} is reachable but there is no room above it to "
            f"approach from: {exc}"
        ) from exc
    if abs(hover_pitch - pitch) > 10.0:
        raise GraspError(
            f"{target.label} can only be approached from {hover_pitch:.0f} degrees "
            f"but grasped at {pitch:.0f} -- the tool would swing into it on the way down"
        )
    # Turn the fingers to close across the object's narrow way. The base yaw is
    # part of this: a cube square to the table, off to one side, is only square
    # to the FINGERS once the wrist undoes the yaw.
    square = abs(target.length_m - target.width_m) <= SQUARE_TOLERANCE * target.length_m
    roll = kin.roll_for(target.angle_deg, grasp_pose[1], square=square)
    return Plan(_rolled(hover_pose, roll), _rolled(grasp_pose, roll), opening, pitch, roll)


def pick(arm: Arm, target: detect.Target, verbose: bool = True,
         reach_offset_m: float = cfg.REACH_OFFSET_M) -> bool:
    """Pick the target up. Returns whether anything is actually held.

    False is a normal outcome, not an error: the fingers closed and found nothing
    between them. Either way the arm ends up lifted and clear, so the caller can
    retry or go back and look again.
    """
    seen = target
    target = aim_for(seen, reach_offset_m)   # everything below aims here
    step = plan(target, reach_offset_m=0.0)  # already offset; do not add it twice

    def say(message: str) -> None:
        if verbose:
            print(f"  {message}", flush=True)

    if reach_offset_m:
        say(f"aiming {reach_offset(seen.x, seen.y, reach_offset_m) * 1000:.0f} mm beyond the object along the reach "
            f"(the model lands the tips that much short): "
            f"{target.x * 1000:.0f} mm fwd, {target.y * 1000:+.0f} mm left")

    # Set the opening BEFORE the arm sets off, and let it ARRIVE.
    #
    # Travelling with the arm saved no time worth having and cost two things that
    # both bite at the worst moment. The fingers were still moving during the
    # descent, and a finger that is closing as it arrives sweeps the object aside
    # instead of straddling it: seen on the robot 2026-09-11, a 40 mm cube shoved
    # 24 mm along the closing axis and then "closed on nothing". And while J6 is in
    # transit the TOOL LENGTH is unknown exactly where it matters -- 30 -> 90 grows
    # it 162 -> 180 mm, so a lagging gripper puts the real tips up to 18 mm from
    # where the descent's own forward kinematics believe they are.
    #
    # Doing it here spends that second and a half with the arm still clear of the
    # table, which is the one place where nothing can be hit, and the fingers are
    # then STATIONARY from the approach until the grasp. (This is the "opening
    # FIRST" that was tried and rejected once for making a fully open gripper
    # visibly close before the arm had gone anywhere. It looks odd; it is correct.)
    say(f"opening to {cfg.GRIPPER_GAP_MM[step.opening]} mm for a "
        f"{target.width_m * 1000:.0f} mm object, before moving")
    arm.set_gripper(step.opening, speed_dps=GRIPPER_DPS)
    say("moving above it")
    arm.move_to(_arm_only(step.hover), speed_dps=APPROACH_DPS)
    say(f"fingers turned to close along {kin.finger_heading(step.grasp):+.0f} deg "
        f"(object's narrow way is at {target.angle_deg:+.0f}; J5={step.roll})")

    table = -cfg.TABLE_BELOW_PLATE
    travel = closing_travel(step.opening, target.width_m)
    say(f"descending to {(GRASP_HEIGHT_M + travel) * 1000:.0f} mm, so the tips land "
        f"at {GRASP_HEIGHT_M * 1000:.0f} mm once they have shut "
        f"({travel * 1000:.0f} mm of closing travel)")
    # Two errors, two fixes, BOTH always on. The servos land short of what they
    # were sent (1.6..16 mm on 2026-09-14, different every pick) -- that shows in
    # the readback and the passes remove it. The real tips then sit short of
    # where the readback puts them (the offset's job; tools/touch_probe.py
    # measures it) -- that does NOT show in the readback and no pass can touch it.
    # The passes used to be switched off whenever an offset was set, on the
    # theory that the offset had been tuned with the readback error baked in. It
    # had, which is why it only worked when that error happened to be ~14 mm:
    # at 1.6 mm the tips went 20 mm past the cube and closed on nothing.
    _reach_to(arm, target.x, target.y, table + GRASP_HEIGHT_M + travel, step.pitch,
              step.opening, DESCEND_DPS, roll=step.roll)
    if verbose:
        tip = kin.forward({**arm.read(), cfg.GRIPPER_ID: step.opening})
        say(f"fingertip landed {math.dist(tip[:2], (target.x, target.y)) * 1000:.1f} mm "
            f"from the aim point, {(tip[2] - table) * 1000:.1f} mm up")
    time.sleep(0.2)

    say("closing")
    arm.close_gripper(speed_dps=GRIPPER_DPS)
    # The servo has to have stopped against the object before grasped() asks
    # where it is; move_to already waited a quarter second after the glide.
    time.sleep(0.4)
    holding = arm.grasped()
    say("holding something" if holding else "closed on nothing")
    if holding:
        # Ask grasped() BEFORE this, never after: relaxing the squeeze changes the
        # gripper target, and grasped() only answers about a close it asked for.
        settled = arm.read()[cfg.GRIPPER_ID]
        arm.set_gripper(min(settled + SQUEEZE_DEG, cfg.GRIPPER_CLOSED))

    lift(arm, target.x, target.y, step.pitch)
    return holding


def lift(arm: Arm, x: float, y: float, pitch: float) -> None:
    """Raise as far straight up as the closed gripper can reach, fingers undisturbed.

    Two things have to be held constant or the object gets dragged sideways along
    the table before it clears:

      * the PITCH, which stays whatever the grasp used. Letting solve() choose again
        would let it pick a different one and swing the tool through an arc.
      * the gripper's ACTUAL angle, which after a grasp is wherever the object
        stopped the fingers -- not the opening we asked for, and not closed. It sets
        the tool length, so using the wrong one aims the whole move at the wrong
        height.
    """
    now = arm.read()
    held = now[cfg.GRIPPER_ID]
    # Take the highest lift that is actually reachable, not just the nominal one.
    #
    # A CLOSED gripper is a LONGER tool -- up to 28 mm longer than the open one the
    # approach was solved with -- so a point that was reachable on the way down can
    # be out of reach on the way up, and the nearer the object the worse it is: at
    # 140 mm the full 80 mm lift is unreachable at every legal pitch. Raising an
    # exception here would abandon an object the fingers are already holding, in
    # mid-air, having done the hard part. Half a lift is worth immeasurably more
    # than a traceback, so the height is what gives way.
    table = -cfg.TABLE_BELOW_PLATE
    attempts = [LIFT_M * fraction for fraction in (1.0, 0.75, 0.5, 0.35, 0.25)]
    for height in attempts:
        try:
            up_pose, _used = _solve_near(x, y, table + height, pitch, held)
        except kin.Unreachable:
            continue
        # The wrist keeps its roll too: un-turning it here would twist what is held.
        arm.move_to(_arm_only(_rolled(up_pose, now[5])), speed_dps=APPROACH_DPS)
        return
    # Nothing above it is reachable while holding this. Stay put rather than raise:
    # the caller still has the object, and can place it from where it stands.
    raise GraspError(
        f"holding something at ({x * 1000:.0f}, {y * 1000:+.0f}) mm but nothing "
        f"between {attempts[-1] * 1000:.0f} and {LIFT_M * 1000:.0f} mm above it is "
        f"reachable with the fingers closed -- place it without lifting"
    )


def place(arm: Arm, x: float, y: float, verbose: bool = True,
          reach_offset_m: float = cfg.REACH_OFFSET_M) -> None:
    """Put down whatever is held, at (x, y) on the table.

    Released slightly higher than it was picked from: the object sits somewhere
    between the fingers rather than exactly at the fingertips, so dropping the last
    few millimetres is kinder than pressing it into the table. Aims beyond (x, y)
    by the same reach offset the pick used, so the object lands where asked.
    """
    x, y = outward(x, y, reach_offset(x, y, reach_offset_m))
    table = -cfg.TABLE_BELOW_PLATE
    now = arm.read()
    held = now[cfg.GRIPPER_ID]
    # Nearest-pitch rather than exact: an exception raised HERE would abandon the
    # object in mid-air, which is worse than a few degrees of tool swing.
    try:
        down_pose, pitch = kin.solve(x, y, table + GRASP_HEIGHT_M + 0.006, gripper=held)
        over_pose, _used = _solve_near(x, y, table + HOVER_M, pitch, held)
    except kin.Unreachable as exc:
        raise GraspError(
            f"cannot place at ({x * 1000:.0f}, {y * 1000:+.0f}) mm: {exc}"
        ) from exc

    if verbose:
        print(f"  placing at {x * 1000:.0f} mm forward, {y * 1000:+.0f} mm left", flush=True)
    # The roll the object was picked with stays for the put-down, so it is set
    # down the way it was lifted.
    over_pose, down_pose = _rolled(over_pose, now[5]), _rolled(down_pose, now[5])
    arm.move_to(_arm_only(over_pose), speed_dps=APPROACH_DPS)
    arm.move_to(_arm_only(down_pose), speed_dps=PLACE_DESCEND_DPS)
    time.sleep(0.2)
    # Open only as far as the fingers need to clear what they hold, not all the
    # way: releasing a 40 mm cube needs 12 mm of clearance, not 30, and the
    # fingers are then already about right for the next pick.
    try:
        release = cfg.gripper_for_gap(cfg.gripper_gap(held) + 2 * FINGER_CLEARANCE_M)
    except ValueError:
        release = cfg.GRIPPER_OPEN
    arm.set_gripper(min(release, held), speed_dps=GRIPPER_DPS)
    time.sleep(0.3)
    # Straight back up, fingers left open, before anything else moves -- so they
    # clear the object instead of dragging it along.
    arm.move_to(_arm_only(over_pose), speed_dps=APPROACH_DPS)
