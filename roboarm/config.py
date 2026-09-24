"""Single source of truth for the robot's measured constants.

Every number here was verified against the physical robot or read out of the
URDF at /home/jetson/breznicky-hetes/yahboomcar_description/urdf/yahboomcar_X3plus.urdf.
Nothing else in the project is allowed to hard-code a limit or a link length.
"""

import math
import os

import numpy as np

# Servo angles by joint id, the shape every module passes around.
Pose = dict[int, int]

# ---------------------------------------------------------------- devices ---
# Names inside the container. compose.yaml maps the stable host /dev/*/by-id/
# paths onto these, so USB re-enumeration cannot shuffle them.
SERIAL_PORT = os.environ.get("ROBOARM_SERIAL", "/dev/robot_serial")

# The wrist camera, on arm_link4: the only camera that can see the workspace (the
# mast camera is fixed horizontal and the table sits 72 degrees below its axis).
WRIST_CAM = int(os.environ.get("ROBOARM_WRIST_CAM", "0"))

# WHICH wrist camera is fitted. Everything that changes with the camera is keyed by
# this: its picture size and V4L2 settings, where its lens sits (the lens section
# below), and its calibration and empty-table photos (data/cameras/<name>/).
#   "sonix"  the stock Yahboom camera, USB 2 YUYV-only, ~7 fps at 640x480. Fitted
#            until 2026-09-24; every measurement in this project before that date
#            was made through it.
#   "c930e"  Logitech Webcam C930e, fitted 2026-09-24. 848x480 at ~24 fps, and a
#            much wider view: 305 x 171 mm of table from the survey pose (the
#            Sonix saw ~157 mm across).
# Going back to the Sonix: cameras/README.md.
WRIST_CAMERA = os.environ.get("ROBOARM_CAMERA", "c930e")
WRIST_CAMERAS = ("sonix", "c930e")
if WRIST_CAMERA not in WRIST_CAMERAS:
    raise ValueError(f"ROBOARM_CAMERA={WRIST_CAMERA!r} -- expected one of {WRIST_CAMERAS}")

# The C930e's picture sizes are CROPS of different width. Measured 2026-09-24 at the
# survey pose, table footprint through the board:
#     640x480, 800x600 (4:3)                 228 x 172 mm, ~24 fps
#     640x360 .. 1920x1080 (all 16:9)        305 x 171 mm, ~24 fps   <- the widest usable
#     2304x1536 (3:2, the whole sensor)      303 x 204 mm, 1 fps, uncompressed only
# So every 16:9 size sees the same table, and the 4:3 ones are its middle. 848x480
# keeps the pixels per mm (and so every pixel-sized threshold) of the 640x480 the
# pipeline was built at, for 1.3x the pixels. The depth engine is built at the same
# aspect (models/da2s_518x910).
WRIST_CAM_SIZE = {"sonix": (640, 480), "c930e": (848, 480)}[WRIST_CAMERA]

# V4L2 settings applied every time the camera is opened, as OpenCV properties, in
# this order (the format before the size). MJPG because 848x480 uncompressed at
# 30 fps is most of a USB 2 bus the mast camera shares. The C930e's continuous
# autofocus hunts on a plain table (and refocusing changes the magnification the
# calibration was fitted at), so its focus is fixed: 45 was the sharpest at the
# survey pose, lens ~150 mm above the table (0 = infinity, 255 = the nearest). The
# camera KEEPS these after it is closed, until it is unplugged, so they are set on
# every open rather than trusted to still be there.
WRIST_CAM_PROPS = {
    "sonix": {},
    "c930e": {"CAP_PROP_FOURCC": "MJPG", "CAP_PROP_AUTOFOCUS": 0, "CAP_PROP_FOCUS": 45},
}[WRIST_CAMERA]

# Lens distortion, undone on every frame as it is captured (camera.py), so that
# everything downstream -- the homographies above all, which cannot model it --
# sees a straight-line picture. (K 3x3 at WRIST_CAM_SIZE, k1 k2 p1 p2 k3), or None.
# The C930e's, from 23 board views at 848x480 over 8 poses and +-20 deg of yaw,
# 2026-09-24: a homography fitted to one view was out by 1.4 mm on average and
# 3.1 mm at worst at the frame's edges, and by 0.8 / 0.95 mm once undistorted.
# The focal length is poorly pinned by views that all look down (416 here, 501
# with more distortion terms, which undistort no better) -- do not read the lens
# height off it. The Sonix was never corrected; its narrow view did not need it.
WRIST_CAM_LENS = {
    "sonix": None,
    "c930e": ([[416.5, 0.0, 425.5], [0.0, 416.5, 249.5], [0.0, 0.0, 1.0]],
              [0.0305, -0.0429, 0.0, 0.0, 0.0]),
}[WRIST_CAMERA]

# ------------------------------------------------------------------- arm ----
# Six bus servos, addressed 1..6. The SDK speaks degrees in these ranges;
# HARD_LIMITS are the servos' own range, SAFE_LIMITS are ours and are what
# arm.py actually enforces. Narrow SAFE_LIMITS after the day-2 reach survey.
JOINT_IDS = (1, 2, 3, 4, 5, 6)
GRIPPER_ID = 6

# What set_uart_servo_angle_array() itself accepts, transcribed from the vendored
# SDK. Anything outside makes it print "angle_s input error!" and do NOTHING -- a
# silently dropped command. arm.py refuses to send such a pose rather than let it
# vanish. Note this is WIDER than HARD_LIMITS for the gripper (SDK allows 0).
SDK_RANGE = {1: (0, 180), 2: (0, 180), 3: (0, 180), 4: (0, 180), 5: (0, 270), 6: (0, 180)}

# What set_uart_servo() -- one servo, raw position pulses -- accepts. The angle
# calls stop at 0 degrees, but that is the SDK's conversion, not the servo: on
# J1-J4 pulse 3100 IS 0 degrees and 4000 is -74. arm.py sends a joint below its
# SDK_RANGE this way (only J2 is allowed there -- see SAFE_LIMITS).
RAW_PULSE_RANGE = (96, 4000)

# J2 -73: pulse 4000, the end of RAW_PULSE_RANGE (2026-09-23; was 0, the angle API's end).
HARD_LIMITS = {1: (0, 180), 2: (-73, 180), 3: (0, 180), 4: (0, 180), 5: (0, 270), 6: (30, 180)}
# J2 floor 15 -> 5 on 2026-09-14: it was only a margin from the servo end (0), and it
# capped the fingertips at 210 mm when the links stretch to 272. 5 keeps the margin.
# J1 ceiling 170 -> 180 on 2026-09-21 (user's wish): the drop point is a full 90 deg
# to the right of straight ahead (J1 = 90), i.e. the servo's own end, and nothing is
# in the way there. The left end keeps its 10 deg margin, untested. The sweep ring is
# unchanged: the next 25 deg station to the right would need J1 = 190.
# J3 floor 10 -> 0 on 2026-09-22 (user's wish: more range; the SDK discards anything
# below 0, so 0 is as far as it goes). The 10 was only a margin from the servo end,
# like J2's. It buys the NEAR side: the nearest grasp point moves 132 -> 122 mm and
# the grasp grid grows 6 % (3679 -> 3893 points); the 238 mm reach is unchanged.
# J4 floor 10 -> 0 the same day, same reason; it changes the grasp envelope not at all
# (J4 only aims the tool, and the pitches a grasp uses never need it that low).
# J2 floor 5 -> -14 on 2026-09-23. The far rim was J2's floor: at the old 232 mm edge
# (open tips, grasp height) the solution sat at J2 = 5 exactly. The user set the arm by
# hand to "limit reach, fingers on the ground" and it read raw 3288 = -15.4 deg, so -14
# is a degree inside a pose shown clear, at J1 = 89 only. Below 0 the angle API refuses
# and arm.py sends raw pulses. Grasp edge 232 -> 253 mm (model; the model put that hand
# pose's tips 18 mm UNDER the table, so it is not calibrated out there -- touch_probe).
SAFE_LIMITS = {1: (10, 180), 2: (-14, 108), 3: (0, 170), 4: (0, 170), 5: (10, 260), 6: (30, 180)}

# How far the base yaws when SEARCHING, in all: 70 deg either side of the calibrated
# look (J1 20..160). J1 itself still goes to 180 for the drop pose; a station there
# only ever saw the drop-off pile (2026-09-22, user: 160, then 140 the same evening).
SWEEP_SPAN_DEG = 140.0

# Where a picked object is let go: a FIXED pose, not a table point. Chosen by eye
# on 2026-09-21 with the arm driven there and looked at: base a full 90 deg to the
# right, fingertips 189 mm out and 110 mm above the table (model), so the object
# falls the last bit. The one-click and the pick job end here; the "place at x/y"
# button is the way to set something down at a table point.
DROP_POSE = {1: 180, 2: 44, 3: 38, 4: 19, 5: 90}

# ... and each drop lets go a random distance up to this much FURTHER OUT along
# the same bearing, same height, so the cubes spread instead of landing on each
# other (user, 2026-09-24: "current position +10 cm max"). The far end needs the
# tool tilted out to 130 deg (grasp.DROP_PITCHES); the mast stays 200+ mm clear.
DROP_SPREAD_M = 0.100

# J2's upper bound is a COLLISION with the camera mast, not a torque limit.
# 108 keeps ~30 mm of modelled fingertip clearance and sits 13 deg below the angle
# where contact was actually observed. See the note under MAST_OFFSET_X: the model
# and the observation do NOT fully agree, so this is deliberately conservative.
#
# The -6 deg "droop" once measured at J2=120 was NOT gravity: it was the arm pushing
# into the mast. Cantilevered forwards at J2=30, where the gravity load is far
# higher, droop is only -1 deg. Real droop on this arm is about a degree, so
# arm.py's correction loop is a safety net rather than something we depend on.
# With the other joints straight (all at 90) the arm reaches the tower at about
# 121 deg: commanded 130 -> 121, 140 -> 121, 150 -> 121, i.e. the joint stops dead
# and the servo simply stalls against the mast, heating up and draining the pack.
# 112 keeps roughly 9 deg of clearance, including the few degrees the droop
# correction adds on top of the goal.
#
# This is the WORST CASE and therefore the right one to encode: an outstretched
# arm reaches the mast soonest. Folded up it would clear further, but a limit that
# depends on the rest of the pose is not something a single number can express, and
# guessing high risks driving into the camera.
#
# Note the earlier reach survey reported "no collisions" -- that swept one joint at
# a time from a folded pose, which never brought the arm near the mast. Single-joint
# sweeps cannot clear a workspace.

# Reach survey, 2026-09-03, 1064 frames with torque off. J1-J4 can be backdriven
# well past what the SDK can command (J1 reached -64..249, J4 -44..193, with the
# extremes sustained over many frames, so they are physical, not glitches). There
# is therefore no mechanical stop to discover inside 0-180: the commandable range
# IS the limit, and SAFE_LIMITS above only needs to keep clear of its ends.
# No self-collision was observed sweeping any single joint across that whole span.
# NOT established: whether a *combination* of J2/J3/J4 can fold the arm into the
# base or the camera mast. Single-joint sweeps cannot rule that out.
#
# Corollary worth remembering: an unpowered arm sags past the commandable range
# (seen on J3, J4 and J6). The SDK then refuses to command it and it looks like a
# dead servo. Recovery is to command a legal angle -- the SDK validates the angle
# sent, not the angle the joint is currently at.

HOME_POSE = {1: 90, 2: 90, 3: 90, 4: 90, 5: 90, 6: 150}  # upright, gripper open

# MEASURED 2026-09-07 by stepping J6 and measuring the fingertip gap. A RISING
# servo angle CLOSES the gripper -- the opposite of what this file assumed, which
# meant close_gripper() was opening the fingers and grasped() reported the reverse
# of reality.
#
#     J6:   30    60    90   120   150   177
#     gap:  70    66    57    42    25    13 mm
#
# At 177 the finger bodies collide -- that is the mechanical stop, so the gripper
# cannot close below a 13 mm gap and cannot hold anything thinner than that.
# GRIPPER_CLOSED stops at 170 (~16 mm) to avoid stalling a servo against the stop,
# which is how we cooked the battery on the mast.
GRIPPER_OPEN = 30      # 70 mm gap -- widest
GRIPPER_CLOSED = 170   # ~16 mm gap, clear of the 177 collision

# Servo angle -> fingertip gap in mm, measured. Objects must be between about
# 15 mm and 65 mm across for this gripper to hold them at all.
GRIPPER_GAP_MM = {30: 70, 60: 66, 90: 57, 120: 42, 150: 25, 177: 13}

# ------------------------------------------------------------ wrist roll ----
# J5 turns the fingers about the tool axis. MEASURED 2026-09-10 through the wrist
# camera at the survey pose, with the gripper closing and opening at several J5
# values and J5 swept across its range (the finger passes through the picture,
# and the homography says which way the picture points):
#
#   * at J5 = 179 the fingers close ALONG THE ARM (fore-aft): closed tips sit on
#     the roll axis, and opening moved the visible finger straight forward.
#   * at J5 = 89 -- the survey and home value -- they close LEFT-RIGHT, across the
#     arm: no finger is anywhere in the picture, open or closed, which is where
#     the camera geometry puts them when they are beside the axis.
#   * RISING J5 turns the closing line CLOCKWISE seen from above (a finger seen
#     at the left of the picture at J5 = 110 had crossed to the right by 210).
#
# Until this was measured every grasp closed across the arm, so a cube off to one
# side, its faces square to the table, was taken at the base's yaw angle -- corner
# first. J5 is assumed to be 1 servo degree per degree of roll; the sweep is
# consistent with that but perspective made it too coarse to prove better than
# about 20%.
#
# The REST value is the zero every roll is measured from: it is where the survey
# and home poses leave the wrist, so an object that needs no turn gets none. For a
# cube the four faces repeat every 90 degrees, so the wrist never turns more than
# 45 either way from here -- it must not swing 90 degrees to reach the equally
# good along-the-arm position when the cube is sitting square in front of it.
J5_FINGERS_ACROSS_ARM = 89   # rest: closes left-right across the arm
J5_FINGERS_ALONG_ARM = 179   # 90 further on: closes fore-aft along the arm
J5_ROLL_SENSE = -1   # +1 would mean rising J5 turns the closing line to the LEFT

# Servo angle at which each joint is actually straight. Nominally 90, but forcing
# these to zero raises the fit RMS from 3.7 mm to 8.3 mm, so the joints really are
# mounted a few degrees off. J4 is essentially straight; J2 and J3 are not, and
# they lean opposite ways, which is why the arm still hangs vertical at all-90 and
# why the parked pose looked perfect while bent poses did not.
JOINT_OFFSET_DEG = {2: 7.5, 3: -10.5, 4: -1.3}

# HEIGHT ACCURACY, measured 2026-09-08 at five points, all commanded 20 mm up:
#
#     target                model   measured   error
#     140 mm fwd, centre     20       20        0
#     180 mm fwd, centre     15       17       +2
#     215 mm fwd, centre     11       15       +4
#     175 mm, 55 left        13       14       +1
#     175 mm, 55 right       13       14       +1
#
# The error GROWS WITH REACH, so there is deliberately no height-offset constant
# here: a single value would zero the error at one distance and double it at
# another. Worst case is 4 mm, inside the kinematics' own 3.7 mm RMS, so the model
# is used as-is and grasping is designed to tolerate about 5 mm of height error
# rather than pretend to correct it. Left and right agree exactly, which rules out
# any asymmetry in J1.
#
# (An earlier single reading suggested 7 mm, but it was taken at a commanded 4 mm,
# where eyeballing the gap is unreliable. Hence measuring several points at 20 mm.)

# REACH OFFSET. Observed 2026-09-10 over several real picks: with the model
# reporting the fingertip within a millimetre or two of the object's centre, the
# fingers were consistently touching the NEAR face of the cube -- the real tips
# land about 20 mm closer to the base than the model says. The cause is not
# pinned down (tool length, a joint offset, the fingers' own splay), so rather
# than guess at a parameter, every grasp and place aims this much further out
# along the bearing from the base. Adjustable from the control panel; set it to
# zero to see the raw model again.
# 2026-09-14: 20 -> 5. The touch probe (tools/touch_probe.py, closed tips driven
# down along the radius across a 40 mm cube) found the cube under the model's
# 0 and +10 mm points and table at -10 and +20: the real tips sit ~5 mm short of
# the model, not 20. The other ~15 mm the old value covered was the servos'
# readback shortfall, which varies 2..16 mm per pick and which grasp.pick()'s
# correction passes now remove on every descent.
#
# PER CAMERA since 2026-09-24: this is about how the arm carries its own weight, and
# the C930e is far heavier than the Sonix. With it on, fingertips put down by the
# kinematics landed 4..8 mm FURTHER out than their readback said (seen through the
# board, five corners), and a 26 mm cube at 197 mm out was missed twice with 5 mm
# (6 mm grown and capped) and held first time with 0, passes 14.3 -> 4.2 -> 2.2 mm.
# One cube, so watch the grips and tune it on the panel ("reach offset").
REACH_OFFSET_M = {"sonix": 0.005, "c930e": 0.0}[WRIST_CAMERA]
# ... and it GROWS WITH REACH, like the height error above. By eye on 2026-09-18:
# 5 mm centred every grip at 150-165 mm; at 174 mm the same 5 mm took the cube by
# its near third (tips ~8 mm short). So the offset used is
#     REACH_OFFSET_M + REACH_OFFSET_SLOPE * (reach - REACH_OFFSET_AT_M), never less
# -> 5 mm at 160, 11 at 180, 17 at 200. Two observations, one slope: PROVISIONAL,
# refine with tools/touch_probe.py --at at a far point when there is time.
REACH_OFFSET_AT_M = 0.160
REACH_OFFSET_SLOPE = 0.30

# Readback tolerance. The SDK documents 1-2 degrees of deviation between the
# commanded and reported angle, so anything inside this is not a fault.
READBACK_TOLERANCE_DEG = 4.0

# Largest single step arm.py will command. Bigger moves are interpolated, so no
# command can ever slam a joint across its range.
MAX_STEP_DEG = 8.0

# --------------------------------------------------------- kinematics (m) ---
# MEASURED on the physical robot 2026-09-03, from a photo with dimensions marked.
# These supersede the URDF at
# /home/jetson/breznicky-hetes/yahboomcar_description/urdf/yahboomcar_X3plus.urdf,
# which a previous team left behind and which does NOT describe this arm:
#
#     segment      URDF        measured
#     J2 -> J3     82.9 mm     85 mm
#     J3 -> J4     82.9 mm     85 mm
#     J4 -> J5    174.55 mm    80 mm     <-- URDF is more than 2x too long
#
# The J4->J5 error alone would have put the fingertip ~95 mm from where the IK
# thought it was, so nothing downstream may use the URDF numbers.
# CALIBRATED 2026-09-07 by least-squares over 8 measured fingertip positions
# (16 equations, 7 parameters, RMS 3.7 mm, worst 6.8 mm). The tape measurements
# were 83.5 / 83.5; the fit prefers 89.4 / 75.2, whose SUM (164.6) is within 2.4 mm
# of the measured 167. Individually L1 and L2 trade off against the joint offsets
# below, so treat the pair as one calibrated quantity rather than two lengths you
# could go and re-measure. What matters is that the model predicts the fingertip.
#
# Independent check that the fit is tracking reality and not absorbing noise: it
# recovered J4->tip = 189.9 mm at J6=150, against 80 + 110 = 190 measured by tape.
L_J2_J3 = 0.0894      # tape said 0.0835
L_J3_J4 = 0.0752      # tape said 0.0835
L_J4_J5 = 0.080

# Horizontal distance from the arm's base to the base of the camera mast --
# what the arm collides with. See SAFE_LIMITS[2].
MAST_OFFSET_X = 0.140

# Measured base plate -> J2, and J5 -> the end of the fingertips.
BASE_PLATE_TO_J2 = 0.030
# J5 to the fingertips, measured at J6=150 (a 25 mm gap).
L_J5_FINGERTIP = 0.110

# TOOL LENGTH VARIES WITH GRIPPER OPENING. The fingers pivot, so opening them pulls
# the tips back along the tool axis and the arm gets effectively shorter.
#
# MEASURED 2026-09-09 (tools/tool_length.py, since deleted: the arm held ONE pose with the tool
# pointing straight down, so fingertip height moves one-for-one with tool length,
# and only J6 changed between readings.
#
#     J6:            30     90    150
#     finger gap:    70     57     25   mm
#     tip height:    48     30     20   mm   (the model predicted 20 at all three)
#     J4 -> tip:    162    180    190   mm
#
# At J6=150 this lands on 190 mm, exactly what the independent least-squares fit of
# the arm geometry produced -- two unrelated methods agreeing. The same fit had
# GUESSED 155 mm at J6=30; the truth is 162.
#
# NOT LINEAR: 18 mm over the first 60 degrees, 10 mm over the second. A circle
# through the three points puts a 29 mm finger on a pivot 82 mm out from J5, which
# is a physically sensible mechanism rather than a curve that happens to fit -- so
# the curvature is real and interpolating between the measured points is right,
# while a straight line would be about 4 mm out in the middle.
#
# This is why every calibration so far was run with the gripper CLOSED, and why
# grasping could not simply reuse those numbers: it has to approach OPEN, where the
# uncorrected model is 28 mm wrong.
GRIPPER_TOOL_MM = {30: 162.0, 90: 180.0, 150: 190.0}


def _between(table: dict[int, float], angle: int) -> float:
    """Read a measured-at-a-few-angles table, straight-line between the points.

    np.interp is flat outside the measured range, which is what we want:
    extrapolating a curve from a handful of points would be inventing data. Both
    gripper tables are consulted at angles nobody measured -- GRIPPER_CLOSED is
    170, which is in neither.
    """
    angles = sorted(table)
    return float(np.interp(angle, angles, [table[a] for a in angles]))


def tool_length(gripper_angle: int) -> float:
    """J4 -> fingertip in metres, for a given gripper opening.

    The flat ends cost us at most ~2 mm at GRIPPER_CLOSED (170), where the curve has
    nearly levelled off anyway.
    """
    return _between(GRIPPER_TOOL_MM, gripper_angle) / 1000


def gripper_gap(gripper_angle: int) -> float:
    """Finger gap in metres at a given servo angle. The inverse of gripper_for_gap.

    Needed because the interesting angles are not the measured ones: GRIPPER_CLOSED
    is 170 and the table jumps 150 -> 177, so a direct lookup raises KeyError.
    """
    return _between(GRIPPER_GAP_MM, gripper_angle) / 1000


def gripper_for_gap(gap_m: float) -> int:
    """Widest gripper angle whose finger gap still exceeds `gap_m`.

    Grasping wants the fingers only just wider than the object, not flung fully
    open: a smaller opening means a longer, better-known tool and less splay to
    catch on neighbouring objects.
    """
    wide_enough = [a for a, gap in sorted(GRIPPER_GAP_MM.items()) if gap / 1000 >= gap_m]
    if not wide_enough:
        raise ValueError(f"nothing opens to {gap_m * 1000:.0f} mm; widest is "
                         f"{max(GRIPPER_GAP_MM.values())} mm")
    return max(wide_enough)

# Measured: the rotating base plate sits this far ABOVE the table surface, so an
# object on the table is at z = -0.190 in the kinematics frame. Nothing may be
# commanded below this -- that is the table.
TABLE_BELOW_PLATE = 0.190

# LAYOUT (confirmed from a photo taken from the robot's right side): the mast
# stands at the CENTRE of the robot and the arm is mounted at the FRONT, so the
# mast is BEHIND the arm. J2 above 90 tips the arm backwards, towards the mast --
# that is the direction that collided. J2 below 90 tips it forwards over the
# table, where the mast is irrelevant.
#
# So the 108 cap costs us almost nothing: it only limits reaching backwards, which
# the demo never needs. Working reach is on the forward side and is unobstructed:
#     J2=75 -> 92 mm forward     J2=45 -> 251 mm forward
#     J2=60 -> 177 mm forward    J2=30 -> 307 mm forward
# MEASURED 2026-09-03 by sweeping J2 from 90 down to 15 in 5 degree steps with J3
# and J4 held straight (worst case for reaching far): NO collision anywhere, and
# droop never worse than -1 deg. Max reach 344 mm forward with the fingertips still
# 116 mm above the base plate. SAFE_LIMITS[2] lower bound is 15 because that is what
# was actually tested -- going lower buys only ~4 mm of reach, so it is not worth
# leaving the tested envelope for.
#
# NOT tested: folding J3/J4 while J2 is low. The sweep held them straight, so a
# folded wrist could still bring the gripper down into the base plate or the table.
#
# UNRESOLVED, and the reason the backward cap is set conservatively:
# straight out, J2 -> fingertip is 355 mm, so the fingertip crosses the 140 mm mast
# radius at J2 = 113 deg. But contact was actually observed at 121, where the model
# says the fingertip is already 43 mm PAST the mast. So the fingertips are evidently
# not what touches -- they must pass beside or above the pole -- and our 2D model of
# this collision is incomplete. It is fine for showing that the margin is small; it
# must NOT be trusted to extrapolate to other J1 angles.
#
# Bigger point: reaching TOWARD the mast is nearly useless anyway -- at the cap the
# arm only reaches ~110 mm horizontally. The workspace has to sit on a J1 heading
# that points AWAY from the mast, where J2 is limited by the table and by torque
# rather than by the tower. The real limit is a J1/J2 pair, not a J2 number.

# --------------------------------------------------------- wrist camera ----
# This is the EYE-IN-HAND camera and, as it turns out, the only one that can see the
# workspace: the mast camera is fixed horizontal, and the table sits about 72 degrees
# below its axis -- far outside any lens. In a top-down grasp pose this one looks
# straight down at the table.
#
# MEASURED 2026-09-10 with the arm parked upright (tools/camera_offset.py, since
# deleted), ruler from the lens to the fingertips at both gripper ends:
#
#              J4 -> fingertip   lens -> fingertip   =>  J4 -> lens
#     closed        190 mm            125 mm              65 mm
#     open          162 mm             96 mm              66 mm
#
# The two agree, so the lens is 65 mm out from J4.
#
# THIS FILE PREVIOUSLY HAD THOSE TWO NUMBERS SWAPPED -- it said 65 mm back from the
# fingertips and therefore 125 mm out from J4. The lens is 60 mm closer to the elbow
# than every calculation assumed, which is what made a reach for a tagged cube miss
# sideways by 20 mm.
#
# Measured back from the FINGERTIPS is the trap: that distance is not a constant,
# because the fingers pivot and the tool grows 28 mm as they close. Only the offset
# from J4 is fixed, so that is what is stored. Anything wanting lens-to-fingertip
# should compute tool_length(gripper) - CAMERA_FROM_J4.
#
# Cross-check from a completely different direction: at the survey pose this puts
# the lens 213 mm above the table, against 222 mm derived from how much a tag of
# known size is magnified by being 40 mm closer to it. With the old swapped value
# the same calculation gave 153 mm, nowhere near.
# (All of the above is the Sonix camera's. The C930e's numbers follow below.)
#
# Where the lens sits, per camera, in metres, relative to the link it is bolted to.
# Both cameras are on arm_link4: they move with J4 but NOT with the wrist roll J5
# (for the C930e, rolling J5 by 30 degrees swung a finger through the picture and
# left the table where it was).
#   FROM_J4     along the tool axis, out from J4.
#   ABOVE_TOOL  across the tool axis, within the arm's own vertical plane. + is the
#               side that is UP when the tool points forward -- which is FORWARD,
#               away from the base, when the tool points down.
#   OFF_AXIS    across the arm's plane, sideways, on the CAMERA_SIDE.
#   SIDE        -1 = the robot's RIGHT (-y), +1 = its LEFT (+y), seen from behind
#               the robot looking the way the arm points.
#
# These decide where parallax pushes an elevated object (away from the LENS), so
# they are what the unlift corrects about. The Sonix sat 50 mm to the right of the
# forearm; its SIDE was INFERRED from the direction of a 20 mm miss, the left
# hypothesis moving the answer the wrong way entirely.
_LENS = {
    #          FROM_J4  ABOVE_TOOL  OFF_AXIS  SIDE
    "sonix":  (0.065,   0.0,        0.050,    -1),
    "c930e":  (0.085,   0.039,      0.0,      -1),
}
# The C930e's, 2026-09-24. The user's ruler: 85 mm from J4 towards J5, 50 mm off the
# axis on top of the arm, centred. The pictures agree on the side -- with the tool
# pointing down the fingertips show at the BOTTOM of the frame, the base side, so
# the lens is forward of them -- and on two of the numbers: the lens position
# solved from 23 board views (8 poses, +-20 deg of yaw, horizontal only, since the
# height rides on a focal length those views cannot pin) gave 90 mm along the tool,
# 3 mm sideways and 38 mm across, 3.1 mm rms; with 85 held, 39 mm across, 3.1 mm rms,
# against 8.3 mm rms for the ruler's 50. 50 is presumably to the camera's body;
# parallax is about the OPTICAL centre, so 39 it is. (Its 11 mm moves the unlift of
# a 40 mm cube by ~2 mm.)
CAMERA_FROM_J4, CAMERA_ABOVE_TOOL, CAMERA_OFF_AXIS, CAMERA_SIDE = _LENS[WRIST_CAMERA]

# The board in the lab is DICT_5X5 -- verified by testing every predefined
# dictionary against a captured frame. 4X4 finds nothing.
ARUCO_DICT = "DICT_5X5_250"

# Tags on OBJECTS are a different family from the calibration board's, established
# 2026-09-10 by running all 27 predefined dictionaries against a frame: the lab's
# cube is an AprilTag 36h11, and only 36h11 matched. The board is DICT_5X5_250.
#
# Being different families is a happy accident worth keeping: a board marker can
# never be mistaken for an object, so the board may stay on the table during a pick
# without having to filter its ids out.
OBJECT_TAG_DICT = "DICT_APRILTAG_36H11"

# MEASURED: the black outer square of the tag on the cube. Needed because a tag of
# KNOWN size doubles as a rangefinder -- see detect.markers(). Everything about
# parallax correction depends on this number being right.
OBJECT_TAG_M = 0.026

# ------------------------------------------------------- detector service ---
# The neural detector (YOLO26 / YOLOE-26) runs in the SEPARATE `vision` container
# and is reached over HTTP -- core holds no torch. compose.yaml sets this to
# http://vision:8760; the default here is for running a tool against the service
# by hand on the robot. If nothing is listening, detect.objects() raises
# DetectorOffline and the caller drops down the ladder to ArUco / change detection.
DETECTOR_URL = os.environ.get("ROBOARM_DETECTOR_URL", "http://vision:8760")

# Generous: first inference after a class-prompt change reloads the text encoder,
# which is seconds on the Orin. A settled model answers in tens of milliseconds.
DETECTOR_TIMEOUT_S = float(os.environ.get("ROBOARM_DETECTOR_TIMEOUT_S", "15"))

# ------------------------------------------------------------- the board ----
# A ChArUco board on A3, taped flat and square to the robot.
BOARD_SQUARES = (7, 5)      # 7 across (left-right), 5 deep (forward)
BOARD_SQUARE_M = 0.0530     # measured: 53 x 53 mm black squares
# MEASURED: the code's modules are 5.5 mm, and a DICT_5X5 marker is 7 modules
# across (a 5x5 code plus a one-module black border), so 7 x 5.5 = 38.5 mm.
# (Inferring it from image ratios gave 39.8 mm -- close, but this is exact.)
BOARD_MARKER_M = 0.0385

# Placement, from the measured paper position: A3 (420 x 297 mm) with its near edge
# 40 mm forward of the rotation axis and centred left-right (210 mm each side). At
# 53 mm squares the 7x5 pattern is 371 x 265 mm, centred on the paper, so the side
# margins are (420-371)/2 = 24.5 mm and the front margin (297-265)/2 = 16 mm.
#
# Board axes were established from the data, not assumed: moving the arm FORWARD
# brought higher board-Y markers into view, and moving LEFT brought higher board-X
# into view. So board +Y is forward and board +X is left.
BOARD_X0 = 0.0560           # table x (forward) where board Y = 0: 40 + 16 mm
BOARD_Y0 = -0.1965          # table y (left) where board X = 0: -371/2 mm, then -11 (below)

# THE BOARD HAD MOVED by 2026-09-24 (between the paper going on and coming off, or
# the robot being nudged while the C930e was fitted): 11 mm to the robot's right of
# the -185.5 mm it was measured and validated at. Found two ways that do not depend
# on any camera calibration: the fingertip, closed, put down by the kinematics on
# five board corners and located in each frame through that frame's own markers
# (lateral 7.7..13.1 mm, mean 10.9; forward +2.7..+5.6, left alone -- that is the
# arm's own reach bias; rotation -0.8 deg, negligible); and the J1 axis, found as the
# centre of the camera's circle as the base yawed +-20 deg, 10..17 mm to the left of
# where the board said it was. The Sonix's saved calibration was fitted before the
# move and stays valid as it is: it is in robot coordinates, not board ones.


def board_to_table(board_x: float, board_y: float) -> tuple[float, float]:
    """Board coordinates (metres, OpenCV convention) -> table (forward, left)."""
    return board_y + BOARD_X0, board_x + BOARD_Y0

# ---------------------------------------------------------------- limits ----
# Below this the servos brown out under load and readback gets unreliable.
MIN_BATTERY_V = 10.0


def angle_from_raw(joint: int, raw: int) -> int:
    """Raw servo count -> degrees, the SDK's own conversion, but rounded honestly.

    Reimplemented rather than calling get_uart_servo_angle() because that clamps
    anything outside the joint's range to -1 -- the same value it uses for "no
    reply" -- and we need the honest angle, out-of-range ones included. Verified
    against the SDK on real readings: raw 1988/2051/2014/2035/1481/1283 give
    91/86/89/87/90/31, matching get_uart_servo_angle_array() exactly.

    The SDK rounds with int(x + 0.5), which truncates toward zero, so every
    NEGATIVE angle comes out a degree high (raw 3288 is -15.4 deg; the SDK says
    -14). floor() is the same above zero and right below it.
    """
    if joint <= 4:
        return math.floor((raw - 900) * (0 - 180) / (3100 - 900) + 180 + 0.5)
    if joint == 5:
        return math.floor((270 - 0) * (raw - 380) / (3700 - 380) + 0 + 0.5)
    return math.floor((180 - 0) * (raw - 900) / (3100 - 900) + 0 + 0.5)


def raw_from_angle(joint: int, angle: int) -> int:
    """Degrees -> raw position pulse, exactly as the SDK's angle calls convert it
    (Rosmaster.__arm_convert_value), for a joint sent below its SDK_RANGE."""
    if joint <= 4:
        return int((3100 - 900) * (angle - 180) / (0 - 180) + 900)
    if joint == 5:
        return int((3700 - 380) * (angle - 0) / (270 - 0) + 380)
    return int((3100 - 900) * (angle - 0) / (180 - 0) + 900)


# ------------------------------------------------------- self-collision ----
# Minimum horizontal gap we insist on between any part of the arm and the mast.
MIN_MAST_CLEARANCE_M = 0.025


def mast_extent(pose: dict[int, int]) -> float:
    """How far BACKWARD (towards the mast) the arm reaches, in metres.

    Measured directions, from moving one joint at a time and watching:
      J1 90->120 swings RIGHT
      J2 90->60  tips FORWARD, so rising J2 tips BACKWARD, towards the mast
      J3 90->120 folds towards the mast -- at exactly 120 it came within ~5 mm
      J4 90->120 tips the gripper up; its sense relative to the mast is ASSUMED
                 to match J2/J3 here, which is the conservative choice.

    Validated against both collisions we have actually seen. For J3=120 it predicts
    135 mm against a 140 mm mast, i.e. 5 mm clear -- exactly what was observed. For
    J2=121 it predicts contact somewhat earlier than it happened, erring safe
    (the fingertips evidently slip past the pole where this flat model says they
    would not).

    Deliberately ignores J1: the arm only actually threatens the mast on some
    headings, but we do not know the mast's bearing well enough to rely on that, so
    every pose is treated as though it were aimed straight at it.
    """
    a2 = math.radians(pose[2] - 90)
    a3 = a2 + math.radians(pose[3] - 90)
    a4 = a3 + math.radians(pose[4] - 90)
    x3 = L_J2_J3 * math.sin(a2)
    x4 = x3 + L_J3_J4 * math.sin(a3)
    x5 = x4 + L_J4_J5 * math.sin(a4)
    tip = x5 + L_J5_FINGERTIP * math.sin(a4)
    return max(x3, x4, x5, tip)


def mast_clearance(pose: dict[int, int]) -> float:
    """Metres between the arm and the mast. Negative means they overlap."""
    return MAST_OFFSET_X - mast_extent(pose)
