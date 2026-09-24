"""Forward and inverse kinematics for the X3 Plus arm. No hardware, no solver.

Frame: origin on the J1 rotation axis at the level of the rotating base plate.
    +x forward (away from the mast)   +y left   +z up
All lengths in metres, all joint angles in SERVO degrees -- the units the SDK and
arm.py speak, so nothing here needs converting before it can be commanded.

Structure. J1 yaws the whole arm; J2, J3 and J4 all pitch within the vertical plane
that yaw defines; J5 rolls about the forearm axis. A roll joint does not bend the
chain, so J4 -> fingertip is ONE straight segment and the arm is really a three-link
planar chain sitting on a turntable. That is the textbook case with a closed-form
solution -- no numerical solver, no MoveIt.

That third link is NOT a constant: the gripper fingers pivot, so opening them pulls
the tips back and shortens the arm from 190 mm to 162 mm (measured -- see
cfg.tool_length). Everything here therefore takes the gripper angle as part of the
pose, and inverse() returns the gripper angle it solved for, so that
forward(inverse(...)) is self-consistent.

Directions were measured on the robot by moving one joint at a time and watching:
    J1 90->120 swings RIGHT      J3 90->120 folds BACKWARD, towards the mast
    J2 90->60  tips FORWARD      J4 90->120 tips the gripper up
so a rising servo angle tips a joint BACKWARD, and the angle of each link from
vertical (positive = forward) is (90 - servo).

Link lengths are the MEASURED ones in config.py, not the URDF's -- see the warning
there; the URDF's forearm is more than twice too long.
"""

from __future__ import annotations

import math

from roboarm import config as cfg
from roboarm.config import Pose

# The planar chain. J4 -> fingertip is one rigid segment because J5 only rolls,
# but its LENGTH depends on how far the gripper is open, so it is a function
# (cfg.tool_length).
L1 = cfg.L_J2_J3
L2 = cfg.L_J3_J4


# Guards the reach comparison against float error: an exactly-extended arm gives
# hypot(...) == 0.17000000000000004 against L1 + L2 == 0.17, which would otherwise
# be rejected as out of reach.
_EPS = 1e-9


class Unreachable(ValueError):
    """The requested point cannot be reached, or only by an illegal pose."""


def _link_angles(pose: Pose) -> tuple[float, float, float]:
    """Angle of each link from vertical, in radians, positive = forward.

    JOINT_OFFSET_DEG corrects for the joints not being mechanically straight at
    servo 90 -- see config.py. Without it the model is right at the parked pose and
    drifts up to 20 mm once the arm is bent.
    """
    off = cfg.JOINT_OFFSET_DEG
    a2 = math.radians(90 - pose[2] + off[2])
    a3 = a2 + math.radians(90 - pose[3] + off[3])
    a4 = a3 + math.radians(90 - pose[4] + off[4])
    return a2, a3, a4


def forward(pose: Pose) -> tuple[float, float, float]:
    """Servo angles -> fingertip position (x, y, z) in metres.

    Needs the gripper angle too, because it sets the tool length.
    """
    a2, a3, a4 = _link_angles(pose)
    l3 = cfg.tool_length(pose[cfg.GRIPPER_ID])
    # Reach and height within the arm's own vertical plane.
    reach = L1 * math.sin(a2) + L2 * math.sin(a3) + l3 * math.sin(a4)
    height = cfg.BASE_PLATE_TO_J2 + L1 * math.cos(a2) + L2 * math.cos(a3) + l3 * math.cos(a4)
    # J1 = 90 points along +x; rising J1 swings right, which is -y.
    yaw = math.radians(90 - pose[1])
    return reach * math.cos(yaw), reach * math.sin(yaw), height


def camera_nadir(pose: Pose) -> tuple[float, float]:
    """Where on the table the wrist lens is directly above, for a given pose.

    This is the point parallax pushes elevated things AWAY from, so it is what
    detect.markers() has to correct about. It is emphatically NOT the middle of the
    picture: the lens sits off the tool axis and the optical axis is not
    exactly vertical, so at the survey pose the image centre lands about 70 mm from
    here. Assuming the two were the same left 20 mm of a 20 mm reach error in place.

    Independent of the gripper, unlike almost everything else on this arm -- the lens
    is bolted to the forearm, so opening the fingers moves the fingertips and leaves
    the lens where it is.
    """
    a2, a3, a4 = _link_angles(pose)
    # Out to J4 within the arm's own vertical plane, then on to the lens along the
    # tool axis and across it. Only J2 and J3 place J4; J4's own angle aims the
    # segment beyond it.
    reach = (L1 * math.sin(a2) + L2 * math.sin(a3) + cfg.CAMERA_FROM_J4 * math.sin(a4)
             - cfg.CAMERA_ABOVE_TOOL * math.cos(a4))
    yaw = math.radians(90 - pose[1])
    # The lens also sits off to one side of that plane, which is a sideways offset
    # in the horizontal plane, perpendicular to the direction the arm points.
    across = cfg.CAMERA_SIDE * cfg.CAMERA_OFF_AXIS
    return (
        reach * math.cos(yaw) - across * math.sin(yaw),
        reach * math.sin(yaw) + across * math.cos(yaw),
    )


def camera_height(pose: Pose) -> float:
    """Height of the wrist lens above the TABLE, in metres. Diagnostic only."""
    a2, a3, a4 = _link_angles(pose)
    above_plate = (
        cfg.BASE_PLATE_TO_J2
        + L1 * math.cos(a2)
        + L2 * math.cos(a3)
        + cfg.CAMERA_FROM_J4 * math.cos(a4)
        + cfg.CAMERA_ABOVE_TOOL * math.sin(a4)
    )
    return above_plate + cfg.TABLE_BELOW_PLATE


def tool_pitch(pose: Pose) -> float:
    """Angle of the gripper axis from vertical, degrees, positive = forward.

    0 means the gripper points straight up; 90 means straight forward; -90 straight
    down, which is the pitch a top-down grasp wants.
    """
    return math.degrees(_link_angles(pose)[2])


def inverse(
    x: float,
    y: float,
    z: float,
    pitch_deg: float = 90.0,
    elbow_up: bool = True,
    gripper: int = cfg.GRIPPER_CLOSED,
    lowest_z: float | None = None,
) -> Pose:
    """Fingertip position -> servo angles. Raises Unreachable if it cannot be done.

    `pitch_deg` fixes the gripper's approach angle from vertical (90 = pointing
    forward, 180 = pointing straight down at the table). It has to be specified
    because the arm has 5 degrees of freedom, not 6: choosing a position uses three
    of them and the pitch uses the fourth, leaving only roll free. Tool yaw is
    always tied to base yaw and cannot be chosen.

    `elbow_up` picks between the two ways to bend the elbow through the same point.

    `gripper` is not optional information: an open gripper is 28 mm shorter than a
    closed one, so solving without it puts the fingertips 28 mm off. It defaults to
    CLOSED because that is the state every calibration on this arm was measured in.

    `lowest_z` lowers the floor below the table surface, for tools/touch_probe.py
    only: it is LOOKING for the table, and out at the rim the model puts the
    real table below its own (18 mm, on the hand-set pose of 2026-09-23).
    """
    floor = -cfg.TABLE_BELOW_PLATE if lowest_z is None else lowest_z
    if z < floor:
        raise Unreachable(
            f"z={z * 1000:.0f} mm is below the table surface at "
            f"{floor * 1000:.0f} mm"
        )

    yaw = math.atan2(y, x)
    reach = math.hypot(x, y)
    a4 = math.radians(pitch_deg)

    # Back off the J4->fingertip segment to find where J4 must be, measured from J2
    # rather than from the base plate. Its length is whatever this gripper opening
    # makes it.
    l3 = cfg.tool_length(gripper)
    wrist_r = reach - l3 * math.sin(a4)
    wrist_z = (z - cfg.BASE_PLATE_TO_J2) - l3 * math.cos(a4)

    distance = math.hypot(wrist_r, wrist_z)
    if distance > L1 + L2 + _EPS or distance < abs(L1 - L2) - _EPS:
        raise Unreachable(
            f"wrist would sit {distance * 1000:.0f} mm from the shoulder; "
            f"the upper arm can only span {abs(L1 - L2) * 1000:.0f}"
            f"..{(L1 + L2) * 1000:.0f} mm"
        )

    # Two-link solution, angles measured from the +reach axis.
    cos_elbow = (distance**2 - L1**2 - L2**2) / (2 * L1 * L2)
    elbow = math.acos(max(-1.0, min(1.0, cos_elbow)))
    if elbow_up:
        elbow = -elbow
    shoulder = math.atan2(wrist_z, wrist_r) - math.atan2(
        L2 * math.sin(elbow), L1 + L2 * math.cos(elbow)
    )

    # Back to angles from vertical, then to servo degrees.
    a2 = math.pi / 2 - shoulder
    a3 = math.pi / 2 - (shoulder + elbow)
    # Undo exactly what _link_angles() does, JOINT_OFFSET_DEG included. Applying
    # the offsets in forward() but not here left IK solving the uncalibrated arm,
    # so its answers missed by up to 22 mm.
    off = cfg.JOINT_OFFSET_DEG
    pose = {
        1: round(90 - math.degrees(yaw)),
        2: round(90 + off[2] - math.degrees(a2)),
        3: round(90 + off[3] - math.degrees(a3 - a2)),
        4: round(90 + off[4] - math.degrees(a4 - a3)),
        5: 90,
        # The opening we solved for, so forward() on this pose agrees with the
        # target we were given. Returning a different one would silently reintroduce
        # the very error this argument exists to remove.
        6: gripper,
    }

    for joint in (1, 2, 3, 4):
        lo, hi = cfg.SAFE_LIMITS[joint]
        if not (lo <= pose[joint] <= hi):
            raise Unreachable(
                f"needs J{joint}={pose[joint]}, outside its safe range {lo}..{hi}"
            )

    clearance = cfg.mast_clearance(pose)
    if clearance < cfg.MIN_MAST_CLEARANCE_M:
        raise Unreachable(
            f"pose would come within {clearance * 1000:.0f} mm of the camera mast"
        )
    return pose


def finger_heading(pose: Pose) -> float:
    """Heading of the line the fingers close along, degrees CCW from +x (forward).

    0 means fore-aft, 90 (or -90: the line has no direction) means left-right.
    Combines the base yaw with the wrist roll, per the measurements in config.py.
    """
    yaw = 90.0 - pose[1]
    # At rest the fingers close ACROSS the arm: 90 degrees from the yaw.
    heading = yaw + 90.0 + cfg.J5_ROLL_SENSE * (pose[5] - cfg.J5_FINGERS_ACROSS_ARM)
    return (heading + 90.0) % 180.0 - 90.0


def roll_for(closing_deg: float, j1: int, square: bool = False) -> int:
    """The J5 that closes the fingers along `closing_deg` (table heading) at base J1.

    Measured from the REST position (J5_FINGERS_ACROSS_ARM, where survey and home
    leave the wrist), so an object that needs no turn gets none. A closing line
    repeats every 180 degrees, so the turn is at most 90 either way; a SQUARE
    object repeats every 90, so at most 45 -- a cube sitting square in front of
    the arm stays at rest rather than swinging to the equally good position 90
    degrees round. Both alternatives are tried against J5's safe range, and the
    nearer legal one wins.
    """
    rest_heading = (90.0 - j1) + 90.0
    period = 90.0 if square else 180.0
    delta = (closing_deg - rest_heading + period / 2) % period - period / 2
    lo, hi = cfg.SAFE_LIMITS[5]
    choices = []
    for turn in (delta, delta - period, delta + period):
        j5 = round(cfg.J5_FINGERS_ACROSS_ARM + cfg.J5_ROLL_SENSE * turn)
        if lo <= j5 <= hi:
            choices.append((abs(turn), j5))
    if not choices:
        raise Unreachable(f"no wrist roll inside {lo}..{hi} closes along {closing_deg:.0f} deg")
    return min(choices)[1]


# Tried most-vertical first. Pitch and reach are coupled on this arm -- the rigid
# 185 mm forearm is longer than the 170 mm upper arm -- so near targets must be
# taken straight down and far ones at a slant. Fixing one pitch throws most of the
# workspace away.
# 185 and 190 (the tool leaning back toward the base) added 2026-09-22, tried last:
# straight down the arm cannot get its fingertips nearer than 122 mm (J3 would have
# to go negative), leaning back 10 deg it reaches 90 mm (J2 23, J3 19, J4 34 at
# 107 mm, 222 mm clear of the mast). Not beyond 190: 195 would allow 74 mm and 210
# 23 mm, and nothing here models the base plate, whose edge is about 60 mm out.
GRASP_PITCHES = (180.0, 175.0, 170.0, 165.0, 160.0, 155.0, 150.0, 145.0, 140.0,
                 185.0, 190.0)


def solve(
    x: float,
    y: float,
    z: float,
    pitches: tuple[float, ...] = GRASP_PITCHES,
    elbow_up: bool = True,
    gripper: int = cfg.GRIPPER_CLOSED,
    lowest_z: float | None = None,
) -> tuple[Pose, float]:
    """Find a legal pose for a point, letting the tool pitch follow the distance.

    Returns (pose, pitch). Raises Unreachable if no offered pitch works.
    """
    for pitch in pitches:
        try:
            return inverse(x, y, z, pitch, elbow_up=elbow_up, gripper=gripper,
                           lowest_z=lowest_z), pitch
        except Unreachable:
            continue
    raise Unreachable(
        f"({x * 1000:.0f}, {y * 1000:.0f}, {z * 1000:.0f}) mm is not reachable at any "
        f"pitch from {pitches[0]:.0f} to {pitches[-1]:.0f} degrees"
    )
