"""See the whole reachable table by turning the base under one fixed look.

The wrist camera sees about 150 mm of table at a time, and the arm can grasp
anywhere in an annulus 131..210 mm from the base and +-80 degrees wide -- roughly
380 cm2. One survey pose covers 27.7% of that. This module covers the rest.

THE ONE IDEA HERE. J1 is a pure yaw about the origin of the kinematics frame, and
the camera is bolted downstream of it on arm_link4. So changing ONLY J1, with
J2..J6 held exactly as the calibration was taken, rotates the camera rigidly about
the same axis the table coordinates are measured from. The table plane is invariant
under that rotation. Therefore

    a pixel that meant table point q at the calibrated pose
    means R(dyaw) q at the pose yawed by dyaw

exactly -- and `rotate()` is that one line. A ring of survey poses costs NO new
calibration, no second board placement, and no second homography fit. The 2 mm
end-to-end accuracy validated at the reference pose transfers to every bearing,
because it is literally the same measurement in a rotated frame.

WHAT THAT ARGUMENT RESTS ON, so it can be checked rather than believed:

  * J1's axis is vertical and passes through the frame origin. True by definition
    -- kinematics.py puts the origin ON the J1 axis, and table coordinates are in
    that frame.
  * The camera is rigid with respect to J1. True: it is bolted to arm_link4.
  * J2..J6 are identical across the ring. Enforced by pose_at(), which copies the
    calibrated pose and touches only J1.
  * The table is level. Already assumed by the single-pose homography; the ring
    does not make it any more of an assumption than it already was.

A CONSTANT error in J1's zero cancels and is not a problem: the homography was
fitted from real board observations at J1=90, so it already absorbs whatever the
true yaw was there, and every ring pose is a DIFFERENCE from it. What does not
cancel is a scale error in J1 degrees, and ordinary joint repeatability -- about
1 degree, which is 3.5 mm at the 210 mm outer edge. That is the same order as the
kinematics' own 4 mm height error, and it is the reason for the refine look below.

THE REFINE LOOK is the second half of the design and the part that keeps accuracy
honest. The sweep says roughly where things are, from whatever bearing happened to
see them -- possibly at the edge of the frame, where uncorrected lens distortion is
worst. Before grasping, the base turns so the chosen object sits on the bearing
that runs through the middle of the picture (`look_bearing()`), and it is measured
again from there. So every object is finally measured from the same relative
geometry, near the image centre, no matter where on the table it sits. Objects at
131..209 mm can all be centred this way; only the outermost millimetre of reach
cannot, and the arm can barely grasp there anyway.

MEASURED COVERAGE of a tagged 40 mm cube, computed against the real calibration
on 2026-09-10 by `tools/sweep_check.py` and checked against the arm the same day:

    ring step   looks   reachable grasp area that can be MEASURED
       none       1        13.8%      (one fixed look, as before the sweep)
       25 deg     7        68.9%
       15 deg    11        70.5%
       10 deg    17        71.3%

25 degrees is the default; past that, extra looks buy almost nothing, and the
reason is worth being precise about.

WHAT THE RING DOES AND DOES NOT FIX. Turning the base sweeps the camera round in
BEARING, and bearing is where the old single look was starving: it saw one 68
degree wedge of a 160 degree workspace. The ring closes that completely -- every
bearing the arm can reach is now looked at. What a yaw cannot change is RADIUS.
The camera sits at a fixed height and tilt, so its footprint lands on a fixed band
of distances, and no amount of turning moves that band. The result:

    the arm can grasp        131..210 mm from the base, bearing -80..+80
    the sweep can measure    129..189 mm from the base, bearing -80..+80

So the ring delivers the whole angular range and about three quarters of the
radial one. The missing 189..210 mm rim is a third of the area, and closing it
needs a SECOND calibrated look with the arm tilted further out -- see
`tools/calibrate_table.py --outer`. That is a real calibration, not a rotation,
because tilting the camera does not map the table to itself the way yawing does.
The machinery to carry more than one calibrated look is already here (`stations()`
takes as many as the calibration file holds); what is missing is the calibration
itself, which needs the ChArUco board face-up on the table.

SAFETY. Every ring pose shares J2..J6 with the calibrated survey pose, so all of
them are equally clear of the camera mast (190 mm, against a 25 mm minimum) and all
of them hold the fingertips 115 mm above the table -- 75 mm of clearance over a
40 mm cube. Sweeping the base through the ring therefore cannot strike the mast and
cannot sweep the fingers through anything standing on the table. That property is
inherited from the calibrated pose rather than assumed, and `ring()` refuses to
build a pose that leaves SAFE_LIMITS.
"""

from __future__ import annotations

import functools
import math
from typing import NamedTuple

import numpy as np

from roboarm import config as cfg
from roboarm import kinematics as kin
from roboarm import workspace as ws
from roboarm.config import Pose

# How far apart the ring's looks are. See the coverage table above.
SURVEY_STEP_DEG = 25

# A safety buffer at the frame edge, on TOP of the space the object itself takes
# up -- see sees(). Small on purpose: an object is measured against its real
# apparent size rather than against a fat constant standing in for one. 8 px is
# about 2 mm of table.
EDGE_MARGIN_PX = 8

# The demo object: a 40 mm cube with a 26 mm AprilTag on its top face.
#
# ITS HEIGHT IS NOT A DETAIL. The homography maps the TABLE, so something standing
# 40 mm above it images outward from the point under the lens, magnified by
#
#     camera height / (camera height - object height) = 212 / 172 = 1.23
#
# MEASURED the same day on the real cube: apparent tag 32 mm against 26 mm real,
# i.e. 1.22. So a cube near the edge of the picture is thrown further out still,
# and is clipped while the table point beneath it is comfortably in shot.
# Overlooking this is what made the first version of this module claim 99%
# coverage when the honest figure for a tagged cube was 91%.
OBJECT_HEIGHT_M = 0.040
OBJECT_SPAN_M = 0.040

# What has to be wholly in shot differs by rung: the tag rung needs only the
# printed square, change detection needs the entire silhouette -- which is why it
# reaches about 7 mm less far.
TAG_SPAN_M = cfg.OBJECT_TAG_M

# Two detections closer than this are the same object seen from two overlapping
# looks. Comfortably under the 22 mm narrowest object the gripper can hold, so it
# can never fuse two genuinely different objects, and comfortably over the few mm
# the two views can disagree by.
MERGE_M = 0.020

# How far out a centred look can still measure a tagged cube, against the real
# calibration. Quoted when an object cannot be centred. The limit is parallax, not
# the frame: a 40 mm cube out at 190 mm is thrown a further 23 mm outward by
# standing above the table plane, and that is what puts it over the edge.
CENTRE_REACH_MM = 189


class Look(NamedTuple):
    """One station of the sweep: where to stand, and what pixels mean from there."""

    dyaw: float          # degrees the base is yawed from the calibrated pose, + = left
    pose: Pose           # what to command
    matrix: np.ndarray   # pixel -> table, already rotated into the base frame
    name: str = "primary"   # which calibrated look this is a yaw of

    @property
    def nadir(self) -> tuple[float, float]:
        """The table point under the lens here -- what parallax is measured about."""
        return kin.camera_nadir(self.pose)


class NoLook(ValueError):
    """There is no legal survey pose that can see this point."""


def rotate(matrix: np.ndarray, dyaw_deg: float) -> np.ndarray:
    """The pixel -> table homography for the same look, base yawed by `dyaw_deg`.

    Left-multiplying by a rotation rotates the OUTPUT of the homography, which is
    exactly right: the pixel means the same ray, the ray now points somewhere else
    on the table, and the somewhere-else is the old answer turned about the base.
    """
    turn = math.radians(dyaw_deg)
    cos, sin = math.cos(turn), math.sin(turn)
    spin = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return spin @ np.asarray(matrix, dtype=float)


def pose_at(survey: Pose, dyaw_deg: float) -> Pose:
    """The calibrated survey pose, yawed by `dyaw_deg`. Only J1 moves.

    kinematics.py defines yaw as (90 - J1), so yawing LEFT by d means J1 goes DOWN
    by d. Every other joint is copied verbatim -- that is what makes the rotated
    homography exact, so it is done here rather than left to callers.
    """
    pose = dict(survey)
    pose[1] = round(survey[1] - dyaw_deg)
    low, high = cfg.SAFE_LIMITS[1]
    if not (low <= pose[1] <= high):
        raise NoLook(
            f"yawing {dyaw_deg:+.0f} deg needs J1={pose[1]}, outside its safe "
            f"range {low}..{high}"
        )
    return pose


def yaw_range(survey: Pose) -> tuple[float, float]:
    """How far the base can yaw either way from the calibrated pose, in degrees."""
    low, high = cfg.SAFE_LIMITS[1]
    return float(survey[1] - high), float(survey[1] - low)


def ring(survey: Pose, matrix: np.ndarray,
         step_deg: float = SURVEY_STEP_DEG, name: str = "primary") -> list[Look]:
    """The sweep: every legal look, `step_deg` apart, centred on the calibrated one.

    Centred on the calibrated pose rather than on the middle of the joint's range,
    so that the look the calibration was actually taken at is always one of the
    stations -- it is the only one whose accuracy has been measured end to end.
    """
    if step_deg <= 0:
        raise ValueError(f"step must be positive, got {step_deg}")
    left, right = yaw_range(survey)
    first = math.floor(left / step_deg)
    last = math.ceil(right / step_deg)
    looks = []
    for index in range(first, last + 1):
        dyaw = index * step_deg
        try:
            pose = pose_at(survey, dyaw)
        except NoLook:
            continue
        looks.append(Look(dyaw, pose, rotate(matrix, dyaw), name))
    return looks


def stations(calibrated: list[tuple[np.ndarray, Pose, str]],
             step_deg: float = SURVEY_STEP_DEG) -> list[Look]:
    """The whole sweep: a yaw ring around every calibrated look there is.

    One calibrated look plus its ring covers every bearing at one band of
    distances. A second calibrated look, tilted further out, adds a second band --
    and its ring covers every bearing too. Together they tile the workspace in
    both directions, which one look and any number of yaws cannot do.
    """
    looks: list[Look] = []
    for matrix, survey, name in calibrated:
        looks.extend(ring(survey, matrix, step_deg, name))
    return looks


def to_pixel(matrix: np.ndarray, x: float, y: float) -> np.ndarray:
    """Table point -> the pixel that sees it, under `matrix`. The inverse map."""
    return ws.apply(np.linalg.inv(np.asarray(matrix, dtype=float)), [(x, y)])[0]


def magnification(pose: Pose, height_m: float) -> float:
    """How much larger something `height_m` above the table images than the table.

    Similar triangles about the lens, nothing more. Cross-checked against the real
    cube on 2026-09-10: this gives 1.23 at 40 mm, and the tag measured 1.22.
    """
    lens = kin.camera_height(pose)
    if height_m <= 0.0:
        return 1.0
    if height_m >= lens:
        raise ValueError(f"an object {height_m * 1000:.0f} mm tall is at or above "
                         f"the lens, {lens * 1000:.0f} mm up")
    return lens / (lens - height_m)


def apparent(look: Look, x: float, y: float, height_m: float) -> tuple[float, float]:
    """Where a point `height_m` above (x, y) lands on the table plane in the image.

    Outward from the NADIR -- the table point under the lens, not the middle of the
    picture. detect._unlift() is this same relation solved the other way, and the
    two have to keep agreeing: this one predicts where an object will appear, that
    one undoes it once the object has been found.
    """
    nadir = np.asarray(look.nadir, dtype=float)
    point = np.asarray([x, y], dtype=float)
    shown = nadir + (point - nadir) * magnification(look.pose, height_m)
    return float(shown[0]), float(shown[1])


def pixels_per_metre(matrix: np.ndarray, x: float, y: float) -> float:
    """Local scale of the mapping at one table point. About 4.4 px/mm mid-frame.

    Differenced rather than derived, because a homography is projective: its scale
    varies across the frame -- 0.217 to 0.240 mm per pixel, corner to corner -- so
    there is no single number to quote. The smaller of the two directions is taken,
    which is the conservative one for asking whether something fits.
    """
    step = 0.001
    here = to_pixel(matrix, x, y)
    along = to_pixel(matrix, x + step, y)
    across = to_pixel(matrix, x, y + step)
    return float(min(np.linalg.norm(along - here),
                     np.linalg.norm(across - here))) / step


def sees(look: Look, x: float, y: float, span_m: float = TAG_SPAN_M,
         height_m: float = OBJECT_HEIGHT_M,
         margin_px: float = EDGE_MARGIN_PX) -> bool:
    """Whether a feature `span_m` across, standing `height_m` above (x, y), is
    WHOLLY in shot from this look.

    This is the question that decides coverage, and it is not "does the point land
    in the picture". Two effects push the answer inward: the feature is thrown
    outward by parallax because it stands above the table plane, and it occupies
    real pixels rather than one. A 40 mm cube images 225 px across, so its centre
    must stay 112 px from the edge -- nearly a quarter of the frame -- before the
    detector has a whole object to measure.

    `span_m` is whatever must be complete, and differs per rung: 26 mm of printed
    tag, or the whole 40 mm silhouette.
    """
    try:
        shown = apparent(look, x, y, height_m)
        pixel = to_pixel(look.matrix, *shown)
        scale = pixels_per_metre(look.matrix, *shown)
        grown = magnification(look.pose, height_m)
    except (np.linalg.LinAlgError, ValueError):
        return False
    if not np.all(np.isfinite(pixel)) or not math.isfinite(scale):
        return False
    half = 0.5 * span_m * grown * scale + margin_px
    width, height = cfg.WRIST_CAM_SIZE
    return bool(half < pixel[0] < width - half and half < pixel[1] < height - half)


def look_bearing(matrix: np.ndarray) -> float:
    """Bearing, in degrees, of the table point the middle of the picture sees.

    NOT zero and not the camera nadir: the lens sits 50 mm off the forearm axis, so
    at the calibrated pose the picture is centred +6.6 degrees round from straight
    ahead while the nadir is 20 degrees the other way. The refine look aims an
    object at THIS bearing, because it is the middle of the frame -- the least
    distorted, best-conditioned part of the mapping -- not the nadir.
    """
    width, height = cfg.WRIST_CAM_SIZE
    centre = ws.apply(matrix, [(width / 2, height / 2)])[0]
    return math.degrees(math.atan2(centre[1], centre[0]))


def yaw_to_centre(matrix: np.ndarray, x: float, y: float) -> float:
    """The yaw that brings (x, y) onto the bearing through the middle of the frame.

    Only the BEARING can be chosen -- turning the base cannot change how far away
    something is -- so the object lands on the centre line at whatever radius it
    has. That is enough: along that line the frame covers 119..209 mm, which is the
    whole graspable annulus bar its outermost millimetre.
    """
    return math.degrees(math.atan2(y, x)) - look_bearing(matrix)


def refine_look(survey: Pose, matrix: np.ndarray, x: float, y: float,
                span_m: float = TAG_SPAN_M,
                height_m: float = OBJECT_HEIGHT_M, name: str = "primary") -> Look:
    """The look that puts (x, y) in the middle of the picture. Raises NoLook.

    Raising rather than falling back is deliberate: the caller already HAS a usable
    measurement from the sweep, and silently returning a worse look would hide the
    fact that this object never got the good one.
    """
    dyaw = yaw_to_centre(matrix, x, y)
    try:
        pose = pose_at(survey, dyaw)
    except NoLook as exc:
        raise NoLook(
            f"({x * 1000:.0f}, {y * 1000:+.0f}) mm cannot be centred: {exc}"
        ) from exc
    # pose_at rounds to whole servo degrees, so recompute the yaw actually commanded
    # rather than the one asked for -- otherwise the homography would describe a
    # pose the arm is not in, which is the whole failure mode this module avoids.
    landed = float(survey[1] - pose[1])
    spun = rotate(matrix, landed)
    look = Look(landed, pose, spun, name)
    if not sees(look, x, y, span_m, height_m):
        raise NoLook(
            f"({x * 1000:.0f}, {y * 1000:+.0f}) mm is {math.hypot(x, y) * 1000:.0f} mm "
            f"out; a centred look reaches about {CENTRE_REACH_MM} mm for a "
            f"{span_m * 1000:.0f} mm feature standing {height_m * 1000:.0f} mm up"
        )
    return look


def best_refine(calibrated: list[tuple[np.ndarray, Pose, str]], x: float, y: float,
                span_m: float = TAG_SPAN_M,
                height_m: float = OBJECT_HEIGHT_M) -> Look:
    """A centred look at (x, y), from whichever calibrated look can measure it.

    Tried in file order, so the primary -- the only look validated end to end to
    2 mm -- gets first refusal, and a secondary is used only for the band the
    primary cannot reach.
    """
    reasons = []
    for matrix, survey, name in calibrated:
        try:
            return refine_look(survey, matrix, x, y, span_m, height_m, name)
        except NoLook as exc:
            reasons.append(f"{name}: {exc}")
    raise NoLook("; ".join(reasons) if reasons else "no calibrated looks at all")


class Hint(NamedTuple):
    """Where to look next, on the strength of something seen cut off at an edge."""

    look: Look
    why: str


def frame_sides(matrix: np.ndarray) -> dict[str, str]:
    """Which side of the PICTURE faces which way on the TABLE, for one look.

    Returns {"far": side, "near": side, "left": side, "right": side} where each
    side is one of "top", "bottom", "left", "right" of the image. Decided from
    where the middle of each picture edge lands on the table, rather than
    assumed, because the lens sits off the forearm axis and the picture is a
    little turned: at the primary look the top row is the far edge, but nothing
    downstream should have to know that.
    """
    width, height = cfg.WRIST_CAM_SIZE
    middles = {
        "top": (width / 2, 0.0), "bottom": (width / 2, height - 1.0),
        "left": (0.0, height / 2), "right": (width - 1.0, height / 2),
    }
    on_table = {side: ws.apply(matrix, [pixel])[0] for side, pixel in middles.items()}
    radius = {side: math.hypot(*point) for side, point in on_table.items()}
    bearing = {side: math.atan2(point[1], point[0]) for side, point in on_table.items()}
    far = max(radius, key=radius.get)
    near = min(radius, key=radius.get)
    across = [side for side in middles if side not in (far, near)]
    # +y is left on the table, so the larger bearing is the left-hand side.
    left = max(across, key=bearing.get)
    right = next(side for side in across if side != left)
    return {"far": far, "near": near, "left": left, "right": right}


def reach_of(matrix: np.ndarray) -> float:
    """How far out, in metres from the base, the far edge of this look's picture lies."""
    width, height = cfg.WRIST_CAM_SIZE
    sides = frame_sides(matrix)
    middles = {
        "top": (width / 2, 0.0), "bottom": (width / 2, height - 1.0),
        "left": (0.0, height / 2), "right": (width - 1.0, height / 2),
    }
    return float(math.hypot(*ws.apply(matrix, [middles[sides["far"]]])[0]))


def hints(look: Look, clipped: list, calibrated: list[tuple[np.ndarray, Pose, str]],
          ) -> list[Hint]:
    """Looks worth taking next, from what a station saw cut off at its edges.

    A clipped detection cannot be measured -- its size is unknown and its
    position is biased -- but which edge it went out of is real information
    that the search used to throw away. Two cases are worth acting on:

      * out of the FAR edge: the object is beyond this look's band of
        distances, and no yaw will bring it in. If a calibrated look that
        reaches further exists (the outer look), take it, turned so the
        object's bearing runs through the middle of its picture. One station
        instead of a whole second ring.
      * out of a SIDE edge: the object is in this look's band but off to one
        side. Turn this same look to centre its bearing -- a yaw, which costs
        nothing in calibration -- rather than wait for the next station,
        which may cut it off at the other side.

    Out of the NEAR edge means closer than the arm can grasp; nothing to do.
    The clipped position's bearing is used and its radius is not: a cut-off
    outline still points the right way to within a few degrees, which is all
    a centring yaw needs, whereas its distance is whatever the visible part
    happened to average to.
    """
    sides = frame_sides(look.matrix)
    by_name = {name: (matrix, survey) for matrix, survey, name in calibrated}
    found: list[Hint] = []
    for target in clipped:
        edges = getattr(target, "edges", frozenset())
        if not edges:
            continue
        bearing = math.degrees(math.atan2(target.y, target.x))
        if sides["far"] in edges:
            further = sorted(
                ((reach_of(matrix), matrix, survey, name)
                 for matrix, survey, name in calibrated
                 if name != look.name and reach_of(matrix) > reach_of(look.matrix) + 0.005),
                key=lambda item: item[0])
            if not further:
                continue
            _reach, matrix, survey, name = further[0]
            why = (f"{target.label} runs out of the far edge at bearing {bearing:+.0f} deg; "
                   f"the {name} look reaches further")
        elif sides["near"] in edges:
            continue
        elif sides["left"] in edges or sides["right"] in edges:
            matrix, survey = by_name[look.name]
            name = look.name
            side = "left" if sides["left"] in edges else "right"
            why = (f"{target.label} runs out of the {side} edge at bearing {bearing:+.0f} deg; "
                   f"turning to centre it")
        else:
            continue
        dyaw = yaw_to_centre(matrix, target.x, target.y)
        try:
            pose = pose_at(survey, dyaw)
        except NoLook:
            continue
        landed = float(survey[1] - pose[1])
        if name == look.name and abs(landed - look.dyaw) < 1.0:
            continue   # that is where we already are
        candidate = Look(landed, pose, rotate(matrix, landed), name)
        if any(h.look.name == name and h.look.pose[1] == pose[1] for h in found):
            continue
        found.append(Hint(candidate, why))
    return found


def beyond_reach(look: Look, clipped: list,
                 calibrated: list[tuple[np.ndarray, Pose, str]]) -> list:
    """The clipped detections that ran out of the far edge of the FURTHEST look.

    No calibrated look reaches past this one, so nothing the search can do
    will bring them in: they are beyond what the camera can measure, which
    for the outer look is also about as far as the arm can grasp. Worth
    saying in the log, so a miss reads "out of reach" rather than "not found".
    """
    sides = frame_sides(look.matrix)
    if any(name != look.name and reach_of(matrix) > reach_of(look.matrix) + 0.005
           for matrix, _survey, name in calibrated):
        return []
    return [t for t in clipped if sides["far"] in getattr(t, "edges", frozenset())]


def merge(seen: list[tuple[Look, object]], tol_m: float = MERGE_M) -> list[object]:
    """Collapse one object seen from several overlapping looks into one target.

    Adjacent stations overlap on purpose -- that is what makes the coverage
    continuous -- so anything in the overlap is detected twice. Where two views
    disagree, the one that saw the object NEARER THE MIDDLE OF ITS FRAME wins: that
    is the better-conditioned measurement, for the same reason the refine look
    exists at all.

    `seen` is (look, target) pairs; the targets are returned in the caller's own
    type, sorted nearest first.
    """
    width, height = cfg.WRIST_CAM_SIZE
    middle = np.array([width / 2, height / 2])

    def offness(look: Look, target) -> float:
        """How far from the middle of its own frame this detection sat, in pixels."""
        try:
            return float(np.linalg.norm(
                to_pixel(look.matrix, target.x, target.y) - middle))
        except np.linalg.LinAlgError:
            return float("inf")

    best: list[tuple[float, object]] = []
    for look, target in seen:
        score = offness(look, target)
        for index, (kept_score, kept) in enumerate(best):
            if math.hypot(kept.x - target.x, kept.y - target.y) <= tol_m:
                if score < kept_score:
                    best[index] = (score, target)
                break
        else:
            best.append((score, target))
    return [target for _score, target in
            sorted(best, key=lambda pair: math.hypot(pair[1].x, pair[1].y))]


def reachable_grasp_points(step_m: float = 0.004, gripper: int | None = None,
                           height_m: float = 0.008) -> np.ndarray:
    """Every table point the arm can actually put its fingertips on, on a grid.

    Remembered per process (it is pure, and the IK over the grid is seconds of
    work); the array comes back read-only so the memo cannot be corrupted.

    The honest denominator for a coverage number. Reachability is not a radius: it
    depends on the gripper opening (an open gripper is a 28 mm shorter tool) and on
    whether any legal tool pitch works at that distance, so it is enumerated by
    asking the IK rather than by a formula.
    """
    if gripper is None:
        # The opening a 40 mm cube is approached with -- the demo object.
        gripper = cfg.gripper_for_gap(0.040 + 0.012)
    return _reachable_grid(step_m, gripper, height_m)


@functools.lru_cache(maxsize=8)
def _reachable_grid(step_m: float, gripper: int, height_m: float) -> np.ndarray:
    z = -cfg.TABLE_BELOW_PLATE + height_m
    reach = round(0.400 / step_m)
    found = []
    for xi in range(reach + 1):
        for yi in range(-reach, reach + 1):
            x, y = xi * step_m, yi * step_m
            try:
                kin.solve(x, y, z, gripper=gripper)
            except kin.Unreachable:
                continue
            found.append((x, y))
    points = np.array(found)
    points.setflags(write=False)
    return points


def coverage(looks: list[Look], points: np.ndarray | None = None,
             span_m: float = TAG_SPAN_M,
             height_m: float = OBJECT_HEIGHT_M) -> tuple[float, np.ndarray]:
    """What fraction of the graspable table this set of looks can actually measure.

    Returns (fraction, the points nobody sees) so a caller can say WHERE the holes
    are rather than only how big they are -- a 91% that is missing the near edge is
    a different problem from a 91% missing the far rim.

    The answer depends on WHAT is being looked for, which is why the span is an
    argument rather than a constant: a 26 mm tag can be measured about 7 mm further
    out than a 40 mm silhouette, simply because less of it has to fit in the frame.
    """
    if points is None:
        points = reachable_grasp_points()
    if len(points) == 0:
        return 1.0, points
    seen = np.zeros(len(points), dtype=bool)
    for look in looks:
        for index, (x, y) in enumerate(points):
            if not seen[index] and sees(look, float(x), float(y), span_m, height_m):
                seen[index] = True
    return float(seen.mean()), points[~seen]


def at_look(calibrated: list[tuple[np.ndarray, Pose, str]], pose: Pose,
            tolerance_deg: int = 3) -> Look | None:
    """The calibrated look the arm is standing at right now, or None.

    A look is valid at ANY base yaw -- that is the point of this module -- so only
    J2..J5 have to match the survey pose; J1 just sets the dyaw. The gripper does
    not move the lens, so J6 is ignored. This is how a live view decides whether
    the pixels it is showing can be mapped to the table at all.
    """
    for matrix, survey, name in calibrated:
        if all(abs(pose[j] - survey[j]) <= tolerance_deg for j in (2, 3, 4, 5)):
            dyaw = float(survey[1] - pose[1])
            return Look(dyaw, dict(pose), rotate(matrix, dyaw), name)
    return None
