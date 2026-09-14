"""Find graspable objects on the table, in table coordinates.

Everything here works on a frame taken from the SURVEY POSE and the homography
fitted there (workspace.py). A pixel means nothing on its own; it only means
something from the pose the mapping was fitted at.

Three detectors, deliberately kept independent, in ascending order of capability
and descending order of guarantee:

  * `changes()` -- compare against a photo of the EMPTY table and report what is new.
    No colour to tune, no model to train, no marker to print. Its failure mode is
    visible rather than silent: if the lighting moved, everything lights up and the
    count is obviously wrong, instead of one object quietly landing 20 mm off.

  * `markers()` -- tags. Gives each object an IDENTITY, which is what "pick up the
    red one" needs, and is as reliable as detection gets. Needs a printer. Objects
    carry AprilTag 36h11 (cfg.OBJECT_TAG_DICT); the calibration board is ArUco
    DICT_5X5_250. Different families on purpose, so the board may stay on the table.

  * `objects()` -- the neural detector (YOLO26, or YOLOE-26 for open-vocabulary
    "pick up the <anything>"). Runs in the separate `vision` container and is
    reached over HTTP; core carries no torch. If that container is not up,
    `objects()` raises `DetectorOffline` and the caller falls back to a lower rung.
    The frame is posted across and boxes come back in PIXELS -- the homography is
    applied HERE, so the vision service never needs the calibration.

All three return the same `Target`, so the grasp code does not know or care which
one ran.

NEITHER GEOMETRIC DETECTOR IS GOOD AT BOTH POSITION AND SIZE, which is why `fuse()`
exists. Measured on a 40 mm cube at the middle of the view, 2026-09-10:

    tag         position good; width is the TAG's 26 mm, not the object's 40
    silhouette  size covers the whole object, but ALSO its shadow: read 51 x 94 mm,
                centre dragged 12 mm towards the shadow

THREE KNOWN ERRORS, measured rather than assumed -- see tools/detect_check.py:

  1. PARALLAX. The homography describes the TABLE PLANE. Anything ON TOP of an
     object is above that plane, so it projects larger and pushed away from the
     point directly under the lens. `markers()` CORRECTS this exactly when told the
     tag's true printed size -- the tag doubles as a rangefinder (see `_unlift`).

  2. SILHOUETTES ARE NOT PARALLAX-FREE EITHER. An earlier note here claimed they
     were; that was wrong. A cube seen off-axis shows its base AND its top, so the
     silhouette is the union of the two -- wider than the base and biased outward.
     Only directly beneath the camera do they coincide. There is no known-size trick
     available here, so silhouette SIZE is trusted and silhouette POSITION is not.

  3. GRIPPER YAW is not used. J5 rolls about the forearm axis, so in a straight-down
     pose it IS yaw about vertical and could align the fingers with an object's short
     axis. We have never measured where J5's zero points, so `angle_deg` is reported
     and not acted on. Round or square objects do not care; a long thin one will.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from roboarm import config as cfg
from roboarm import workspace as ws

# Object widths this gripper can actually deal with, both ends derived from the
# MEASURED gap table rather than guessed.
#
# MIN is not "what the fingers can hold" -- they close to 13 mm. It is what a grasp
# can be VERIFIED at. grasped() decides by asking whether an object stopped the
# fingers short of GRIPPER_CLOSED (170) by more than the 4 degree readback
# tolerance, i.e. below 166. Interpolating the gap table, a 22 mm object stops them
# at about 157 -- clear. An 18 mm one stops them at 166, exactly on the threshold,
# so a real grasp would report as a miss about half the time.
#
# MAX is the 70 mm the fingers open to, less the 12 mm of clearance the grasp leaves
# around the object (grasp.FINGER_CLEARANCE_M), rounded down.
MIN_WIDTH_M = 0.022
MAX_WIDTH_M = 0.055

# The photograph of the empty table that `changes()` compares against. Kept beside
# the homography because the two go together: both are only valid from the survey
# pose, and both are invalidated by moving the robot, the board or the lamp.
#
# ONE PER LOOK. A background is a picture of a particular patch of table from a
# particular pose, so the sweep's ring of looks needs a ring of backgrounds --
# point the camera 25 degrees further round and every pixel has moved, which is
# precisely the situation `changes()` reads as one table-sized object. The
# calibrated look keeps the unsuffixed name, so a single-pose setup captured
# before the sweep existed still loads.
BACKGROUND_PATH = Path("/app/data/table_background.png")

# Below this a "change" is noise -- a shadow edge, a compression artefact.
MIN_AREA_M2 = 0.00020   # about 14 x 14 mm

# Pixel difference that counts as changed. Well above sensor noise (a few counts),
# well below the contrast of any real object against the table.
CHANGE_THRESHOLD = 35

# Shadow test, see _shadow(). A shadow dims a pixel without colouring it.
#
# MEASURED on the real scene 2026-09-10, sampling shadow, cube and bare table:
#
#                  value ratio   hue gap   saturation gain
#     shadow          0.51         46         +14  (p90 +53)
#     blue cube       0.48         50         +91  (p90 +226)
#     bare table      1.12         34          -9
#
# HUE IS USELESS HERE and is deliberately not tested. It scatters 46 degrees inside
# the shadow ITSELF, because hue is numerically unstable on near-neutral pixels and
# this table is near-neutral everywhere. A first attempt used a 14 degree tolerance,
# which no shadow pixel could ever satisfy, so the filter did nothing at all.
#
# SATURATION is what actually separates them: a shadow barely changes it, an object
# brings its own colour. That is the whole test, alongside the value ratio.
#
#   DARKEST  below this a pixel is too dark to be a shadow on white paper and is
#            kept -- the only thing saving a genuinely dark OBJECT from erasure.
#   FAINTEST above this it is barely dimmed and would be noise either way.
SHADOW_DARKEST = 0.35
SHADOW_FAINTEST = 0.92
SHADOW_SATURATION_GAIN = 60


@dataclass(frozen=True)
class Target:
    """Something to pick up, already in table coordinates (metres, +x fwd, +y left)."""

    x: float
    y: float
    width_m: float      # the narrow way -- what the gripper has to span
    length_m: float
    angle_deg: float    # heading of the SHORT axis; reported, not yet acted on
    label: str
    # 1.0 for the geometric detectors (a change either is or is not there); the
    # neural detector's own score for `objects()`. Carried so a caller can prefer
    # the surest target, and so `annotate()` can show it.
    confidence: float = 1.0
    # How far the object's top stands above the table, when the detector could
    # tell (a cube-ranged silhouette: it is the cube's edge). None means unknown,
    # and callers fall back to sweep.OBJECT_HEIGHT_M. It decides how much
    # parallax throws the object outward, and so whether it is wholly in shot.
    height_m: float | None = None
    # The outline touched the edge of the picture, so only part of the object
    # was measured: its size is unknown and its position is biased. A clipped
    # blob used to be caught only by asking whether an object of the MEASURED
    # size fits in the frame -- which a 60 mm cube cut down to a 24 mm sliver
    # passes. Whether the contour touches the border is the honest test.
    clipped: bool = False

    @property
    def graspable(self) -> bool:
        return not self.clipped and MIN_WIDTH_M <= self.width_m <= MAX_WIDTH_M

    def why_not(self) -> str:
        if self.clipped:
            return "partly out of frame -- its size is not known"
        if self.width_m < MIN_WIDTH_M:
            return f"{self.width_m * 1000:.0f} mm is too thin for the fingers to hold"
        if self.width_m > MAX_WIDTH_M:
            return f"{self.width_m * 1000:.0f} mm is wider than the gripper opens"
        return ""


def _target_from_quad(quad: np.ndarray, label: str, confidence: float = 1.0,
                      height_m: float | None = None, clipped: bool = False) -> Target:
    """Build a Target from four TABLE-frame corners of a rectangle.

    Measured on the table, not in pixels: the homography is a projection, so pixel
    distances are not proportional to real ones and a pixel-space width would be
    wrong by however much the view is tilted.
    """
    centre = quad.mean(axis=0)
    sides = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    short_i = int(np.argmin(sides[:2]))          # opposite sides are equal
    short = float(min(sides[0], sides[1]))
    long = float(max(sides[0], sides[1]))
    edge = quad[(short_i + 1) % 4] - quad[short_i]
    return Target(
        x=float(centre[0]),
        y=float(centre[1]),
        width_m=short,
        length_m=long,
        angle_deg=float(np.degrees(np.arctan2(edge[1], edge[0]))),
        label=label,
        confidence=float(confidence),
        height_m=height_m,
        clipped=clipped,
    )


def light_scale(frame, background) -> float:
    """How much brighter the room is now than when the reference was photographed.

    The lamp, the sun and the auto-exposure all drift. Measured on this table, the
    bare paper came back 12 percent brighter than its own reference photo a few hours
    later -- enough to widen every silhouette and, left long enough, to light up the
    whole frame as one enormous object.

    The MEDIAN is what makes this safe: objects occupy a minority of the picture, so
    the middle ratio describes the table rather than the things on it. A mean would
    be dragged around by whatever was placed down.
    """
    now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    was = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lit = was > 20  # near-black reference pixels give meaningless ratios
    if not lit.any():
        return 1.0
    return float(np.median(now[lit] / was[lit]))


def _shadow(frame, background, scale: float) -> np.ndarray:
    """Pixels merely DARKER than the background, rather than different from it.

    A shadow dims; an object brings colour. So a pixel is shadow when its VALUE has
    dropped by a plausible amount and its SATURATION has not risen. Hue is not tested
    at all -- see the constants above for the measurements that ruled it out.

    Worth the lines: without this a 40 mm cube measured 55 x 113 mm and was refused
    as "wider than the gripper opens", rejected on the size of its own shadow.

    LIMIT, and a real one: this separates object from shadow by COLOUR. A grey or
    white object on white paper has no colour to separate and would be read as
    shadow and vanish. The value ratio is the only defence there -- anything darker
    than SHADOW_DARKEST is kept -- so demo objects should be coloured or dark, not
    pale grey.
    """
    now = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    was = cv2.cvtColor(background, cv2.COLOR_BGR2HSV).astype(np.float32)

    ratio = now[:, :, 2] / np.maximum(was[:, :, 2] * scale, 1.0)
    dimmed = (ratio > SHADOW_DARKEST) & (ratio < SHADOW_FAINTEST)
    # Signed on purpose: shadows do not add colour, objects do. A drop in saturation
    # is no reason to call something an object.
    duller = (now[:, :, 1] - was[:, :, 1]) < SHADOW_SATURATION_GAIN
    return (dimmed & duller).astype(np.uint8) * 255


def foreground(frame, background) -> np.ndarray:
    """Binary mask of what is genuinely new on the table, shadows removed.

    Public because it is the thing to LOOK AT when detection misbehaves -- a picture
    of this mask says immediately whether the problem is the threshold, the shadow
    test or the object itself. tools/detect_check.py writes it out.
    """
    scale = light_scale(frame, background)
    now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Match the reference to the CURRENT light level before comparing, or a room
    # that merely got brighter reads as a table covered in new objects.
    was = np.clip(
        cv2.cvtColor(background, cv2.COLOR_BGR2GRAY).astype(np.float32) * scale, 0, 255
    ).astype(np.uint8)
    # Blur first: the camera is noisy and 1-pixel speckle survives thresholding,
    # then survives the open, then shows up as a swarm of tiny false objects.
    diff = cv2.absdiff(cv2.GaussianBlur(now, (5, 5), 0), cv2.GaussianBlur(was, (5, 5), 0))
    _, mask = cv2.threshold(diff, CHANGE_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(_shadow(frame, background, scale)))
    # Open then close: drop speckle, then heal an object split by a highlight -- or
    # by the shadow test biting into a dark face of the object itself.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def changes(frame, background, matrix: np.ndarray,
            nadir: tuple[float, float] | None = None,
            lens_m: float | None = None) -> list[Target]:
    """Objects that appeared on the table since `background` was taken.

    Both frames must come from the same survey pose, or every pixel has moved and
    the whole image reads as one enormous object.

    With `nadir` and `lens_m` each silhouette is read as a CUBE and ranged --
    see cube_range(); without them the outline is reported as the table plane
    saw it, which for anything standing up is a little big and a little far.
    """
    return outlines(foreground(frame, background), matrix, "object",
                    nadir=nadir, lens_m=lens_m, sort_by_x=True)



# The COLOUR rung. Needs no empty-table photo and no tag -- only that the object
# brings its own colour to a table that has none.
#
# It earns its place because the other two geometric rungs each have a hole this
# fills. `changes()` needs a background photograph, which is per-pose, so a sweep
# of seven stations needs seven of them and an empty table to take them on.
# `markers()` needs the tag to be facing up, which stops being true the moment
# anything knocks the object over -- which is exactly how it was needed here on
# 2026-09-10.
#
# MEASURED on the real scene the same day, sampling the lab's blue box, its shadow
# and the bare paper (the same sampling that set the shadow constants above):
#
#                  saturation
#     blue box        +91 over the table (p90 +226)
#     its shadow      +14 over the table (p90 +53)
#     bare paper       baseline, near-neutral everywhere
#
# so a threshold between the two separates object from shadow outright, where a
# brightness test cannot: a shadow is exactly a darker version of the table, and
# a saturation test simply does not see it. 70 sits well clear of both.
MIN_SATURATION = 70


def colour_mask(frame, min_saturation: int = MIN_SATURATION) -> np.ndarray:
    """Pixels that bring their own colour to a colourless table, cleaned up.

    Public so the panel's mask view shows exactly what the detector measured.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > min_saturation).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def coloured(frame, matrix: np.ndarray, min_saturation: int = MIN_SATURATION,
             nadir: tuple[float, float] | None = None,
             height_m: float = 0.0, lens_m: float | None = None) -> list[Target]:
    """Strongly coloured objects on a colourless table, in table coordinates.

    Two ways to undo the parallax of an object standing above the table plane:

      * `lens_m` (with `nadir`): the object is taken to be a CUBE, and its own
        outline says how tall it is -- cube_range(). Size and position both come
        out right, for any size of cube, and nothing has to be declared.
      * `height_m` (with `nadir`): the old way, for something whose height is
        known and whose width is not its height. The outline is unlifted about
        the nadir by a fixed ratio, exactly as markers() does for a tag.

    Without either the outline is reported as the table plane saw it.

    The outline of a solid object includes whichever SIDE faces the lens can see,
    not just its top; cube_range() uses that, the fixed-height path is made a
    little generous by it -- the safe direction for choosing a gripper opening.
    """
    if height_m > 0.0 and nadir is None:
        raise ValueError(
            "coloured() needs nadir= to correct for an object standing "
            "above the table; pass kin.camera_nadir(pose)"
        )
    return outlines(colour_mask(frame, min_saturation), matrix, "colour",
                    nadir=nadir, height_m=height_m, lens_m=lens_m)


def outlines(mask: np.ndarray, matrix: np.ndarray, label: str, *,
             nadir: tuple[float, float] | None = None, height_m: float = 0.0,
             lens_m: float | None = None, confidence: float = 1.0,
             sort_by_x: bool = False) -> list[Target]:
    """Every blob in a binary mask, as a Target on the table.

    The one place a silhouette becomes a position and a size, shared by the
    colour and change rungs so they cannot drift apart. The correction for
    standing above the table is chosen as in coloured().
    """
    # CHAIN_APPROX_NONE, not SIMPLE: SIMPLE compresses a straight run to its two
    # endpoints, so a blob clipped along a whole frame edge keeps just two points
    # there and looks intact to any test that counts them. That cost real time.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    found = []
    for index, contour in enumerate(contours):
        name = f"{label} {index + 1}"
        clipped = touches_edge(contour.reshape(-1, 2), mask.shape)
        if lens_m is not None and nadir is not None:
            points = ws.apply(matrix, contour.reshape(-1, 2))
            quad, _width, tall, agreement = range_block(
                points, nadir, lens_m, chamfer_m=chamfer_for(matrix, contour.reshape(-1, 2)))
            target = _target_from_quad(quad, name, 0.0 if clipped else confidence * agreement,
                                       height_m=tall, clipped=clipped)
        else:
            quad = ws.apply(matrix, cv2.boxPoints(cv2.minAreaRect(contour)))
            if height_m > 0.0:
                centre = np.asarray(nadir, dtype=float)
                quad = centre + (quad - centre) / _lift_ratio(height_m)
            target = _target_from_quad(quad, name, 0.0 if clipped else confidence,
                                       clipped=clipped)
        if target.width_m * target.length_m < MIN_AREA_M2:
            continue
        found.append(target)
    if sort_by_x:
        return sorted(found, key=lambda t: t.x)
    return sorted(found, key=lambda t: math.hypot(t.x, t.y))


# Pixels from the picture's edge within which an outline counts as cut off.
CLIPPED_MARGIN_PX = 3


def chamfer_for(matrix: np.ndarray, pixels: np.ndarray) -> float:
    """CHAMFER_PX at this outline's place in the picture, in metres on the table."""
    centre = np.asarray(pixels, dtype=float).reshape(-1, 2).mean(axis=0)
    here, step = ws.apply(matrix, [centre, centre + [1.0, 0.0]])
    return float(CHAMFER_PX * np.linalg.norm(step - here))


def touches_edge(pixels: np.ndarray, shape: tuple[int, ...],
                 margin_px: int = CLIPPED_MARGIN_PX) -> bool:
    """Whether an outline in PIXELS runs into the border of a picture this shape."""
    pts = np.asarray(pixels, dtype=float).reshape(-1, 2)
    height, width = shape[0], shape[1]
    return bool(pts[:, 0].min() <= margin_px or pts[:, 1].min() <= margin_px
                or pts[:, 0].max() >= width - 1 - margin_px
                or pts[:, 1].max() >= height - 1 - margin_px)


# A cube measured from its own outline: how well its two independent size
# estimates agreed. Below this the blob is not a clean cube -- a shadow is
# attached, two objects touch, or it is clipped -- and the target is reported
# with that as its confidence so the caller can prefer a surer one.
CUBE_AGREEMENT_OK = 0.7


# An outline edge shorter than this is a chamfer -- the pixel grid and the
# mask's opening rounding a corner -- not a side of anything. A pixel-scale
# thing, so callers that know the picture's scale pass CHAMFER_PX of it;
# this default is that at the wrist camera's 4.4 px/mm.
CHAMFER_M = 0.002
CHAMFER_PX = 8


def cube_range(outline: np.ndarray, nadir, lens_m: float) -> tuple[np.ndarray, float, float]:
    """A cube's footprint and edge length from its apparent outline. See range_block().

    Returns (footprint quad on the table, edge length in metres, agreement 0..1).
    """
    quad, width, _height, agreement = range_block(outline, nadir, lens_m)
    return quad, width, agreement


def range_block(outline: np.ndarray, nadir, lens_m: float,
                outline_is_top: bool = False,
                chamfer_m: float = CHAMFER_M) -> tuple[np.ndarray, float, float, float]:
    """Where a CUBE really is and how big it is, from its apparent outline.

    The homography maps the TABLE plane, so a solid object images as its top
    thrown outward from the point under the lens (the nadir) by

        m = lens height / (lens height - object height)

    plus whichever side faces the lens can see, which trail from the top back
    towards the nadir. One thing in that picture is robust: the corner of the
    outline FARTHEST from the nadir is a corner of the top, and the two outline
    edges meeting there are edges of the top -- complete, and s * m long for a
    cube of edge s. (Square-on to the radial, one of the two is instead the
    seam down to the base, which points straight back at the nadir and is
    recognised by that.) With s * m = w that is one equation in s,
    s = w * H / (H + w), the edge gives the yaw, the corner places the top, and
    the top's centre unlifts to the base's: nadir + (top - nadir) / m.

    A first version instead measured the height from where the near face met
    the table -- three extents, three unknowns, closed form. It is exact on a
    drawn outline and useless on a real one: for a 40 mm cube 100 mm from the
    nadir the number it divides by is 17 mm, and a few millimetres of nadir
    error or mask edge turned the lab's cube into a 1 mm tall, 50 mm wide tile.
    A second read the outline's minimum-area rectangle as the top, which is
    right only when the cube is square-on to the radial. The far-corner reading
    needs no near face at all -- a mask that catches only the top and one side,
    as the lab's segmentation masks do, is enough -- and depends on the nadir
    only through the final unlift, where 10 mm of nadir error is 2 mm of
    position. Objects are taken to be cubes (height = edge); a tag on top,
    where there is one, measures the height outright -- see markers().

    `outline_is_top` says the outline is a box round the object rather than
    its silhouette (a detector that sends no mask); the box's short side is
    then read as the top's.

    Agreement is how well the cube found explains the outline seen: its
    predicted outline -- base plus magnified top -- overlapped with the
    observation. A clean cube scores high; touching objects or a wide shadow
    do not. Returns (footprint quad on the table, width, height, agreement).
    """
    points = np.asarray(outline, dtype=float).reshape(-1, 2)
    nadir = np.asarray(nadir, dtype=float)
    lens = float(lens_m)
    if len(points) < 3 or lens <= 0.0:
        raise ValueError("range_block needs an outline of at least 3 points and a lens height")

    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2).astype(float)
    top_side, yaw, top_centre = _top_from_far_corner(hull, nadir, outline_is_top, chamfer_m)

    size = top_side * lens / (lens + top_side)
    size = float(min(max(size, 0.001), 0.75 * lens))
    grown = lens / (lens - size)
    centre = nadir + (top_centre - nadir) / grown
    agreement = _overlap(hull, _block_outline(centre, size, size, yaw, nadir, lens))
    return _footprint(centre, size, yaw), size, size, float(agreement)


# An outline edge within this of pointing straight back at the nadir is the seam
# from a top corner down to the base, not an edge of the top. A seam points at
# the nadir exactly; a top edge can lie at any angle, so keep this small.
SEAM_DEG = 6.0
# Outline simplification before corners are read: a rasterised blob's edge is
# jagged at the pixel scale, and a jaggy would otherwise pass for a corner.
SMOOTH_M = 0.0015


def _top_from_far_corner(hull: np.ndarray, nadir: np.ndarray, boxed: bool,
                         chamfer_m: float) -> tuple[float, float, np.ndarray]:
    """(apparent side, yaw, apparent centre) of the top face, read from the
    outline corner farthest from the nadir and the edges meeting there.

    Corners of a real outline are chamfered: the mask's morphological opening
    rounds them, a segmentation mask is smooth, the pixel grid is coarse. A
    chamfer is an edge much shorter than its neighbours, and the corner it cut
    off is where the lines of the edges either side of it meet -- so every
    corner used here is restored that way before anything is measured.
    """
    if boxed:
        (cx, cy), (rect_w, rect_h), angle = cv2.minAreaRect(hull.astype(np.float32))
        return float(min(rect_w, rect_h)), math.radians(angle), np.array([cx, cy])

    poly = cv2.approxPolyDP(hull.astype(np.float32), SMOOTH_M, True).reshape(-1, 2).astype(float)
    if len(poly) < 3:
        poly = hull
    count = len(poly)
    far = int(np.argmax(np.linalg.norm(poly - nadir, axis=1)))

    def vertex(i: int) -> np.ndarray:
        return poly[i % count]

    def is_chamfer(i: int, step: int) -> bool:
        """Whether the edge leaving vertex i in direction `step` is a chamfer."""
        return float(np.linalg.norm(vertex(i + step) - vertex(i))) < chamfer_m

    def edge_end(i: int, step: int) -> tuple[np.ndarray, np.ndarray]:
        """The corner where the edge leaving vertex i in direction `step` ends,
        with its chamfer cut back: (corner, unit direction from vertex i)."""
        start, end = vertex(i), vertex(i + step)
        if is_chamfer(i + step, step):
            end = _meet(start, end, vertex(i + 2 * step), vertex(i + 3 * step))
        direction = end - start
        return end, direction / max(float(np.linalg.norm(direction)), 1e-9)

    # The far corner itself, cut back if a chamfer sits on it.
    corner = vertex(far)
    for step in (1, -1):
        if is_chamfer(far, step):
            corner = _meet(vertex(far - step), vertex(far),
                           vertex(far + step), vertex(far + 2 * step))
            break
    back = nadir - corner
    back /= max(float(np.linalg.norm(back)), 1e-9)

    sides, units = [], []
    for step in (1, -1):
        start = far + step if is_chamfer(far, step) else far
        end, unit = edge_end(start, step)
        # Measured from the restored far corner, not from wherever the chamfer left us.
        length = float(np.linalg.norm(end - corner))
        seam = math.degrees(math.acos(max(-1.0, min(1.0, float(unit @ back)))))
        if seam > SEAM_DEG and length > 1e-6:
            sides.append(length)
            units.append(unit)
    if not sides:
        # Nothing but seams: read the rectangle instead, better than nothing.
        (cx, cy), (rect_w, rect_h), angle = cv2.minAreaRect(hull.astype(np.float32))
        return float(min(rect_w, rect_h)), math.radians(angle), np.array([cx, cy])
    side = max(sides)
    if len(units) == 1:
        # Square-on: the other side of the top runs from this edge towards the nadir.
        inward = np.array([-units[0][1], units[0][0]])
        if inward @ back < 0:
            inward = -inward
        units.append(inward)
    top_centre = corner + (side / 2) * (units[0] + units[1])
    yaw = math.atan2(units[0][1], units[0][0])
    return side, yaw, top_centre


def _meet(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    """Where the line through p1, p2 meets the line through p3, p4; p2 if parallel."""
    d1, d2 = p2 - p1, p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        return p2
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    return p1 + t * d1


def _footprint(centre: np.ndarray, width: float, yaw: float) -> np.ndarray:
    half = width / 2
    cos, sin = math.cos(yaw), math.sin(yaw)
    spin = np.array([[cos, -sin], [sin, cos]])
    return centre + np.array([[-half, -half], [half, -half], [half, half], [-half, half]]) @ spin.T


def _block_outline(centre: np.ndarray, width: float, height: float, yaw: float,
                   nadir: np.ndarray, lens_m: float) -> np.ndarray:
    """The table-plane outline a block this size, here, would show: the hull of
    its base and its top magnified about the nadir. sweep.apparent() is this
    same relation for a single point."""
    base = _footprint(centre, width, yaw)
    top = nadir + (base - nadir) * (lens_m / (lens_m - height))
    return cv2.convexHull(np.vstack([base, top]).astype(np.float32)).reshape(-1, 2).astype(float)


def _overlap(a: np.ndarray, b: np.ndarray, cell_m: float = 0.0005) -> float:
    """Intersection over union of two outlines, 0..1, by rasterising both onto
    a half-millimetre grid. cv2.intersectConvexConvex was tried first and gave
    overlaps above 1 on these metre-scale polygons; pixels do not lie."""
    pts = np.vstack([a, b])
    lo = pts.min(axis=0) - cell_m
    size = np.ceil((pts.max(axis=0) - lo) / cell_m).astype(int) + 2
    if size.max() > 4000:
        cell_m *= size.max() / 4000
        size = np.ceil((pts.max(axis=0) - lo) / cell_m).astype(int) + 2
    def raster(poly):
        canvas = np.zeros((int(size[1]), int(size[0])), np.uint8)
        cv2.fillPoly(canvas, [((poly - lo) / cell_m).astype(np.int32)], 1)
        return canvas.astype(bool)
    ra, rb = raster(a), raster(b)
    union = np.count_nonzero(ra | rb)
    return float(np.count_nonzero(ra & rb) / union) if union else 0.0


def _lift_ratio(height_m: float, lens_m: float = 0.212) -> float:
    """How much larger something `height_m` up images than the table beneath it.

    The lens height defaults to the calibrated survey pose's 212 mm. Callers that
    look from anywhere else should pass their own, from kin.camera_height(pose).
    """
    if height_m <= 0.0:
        return 1.0
    if height_m >= lens_m:
        raise ValueError(f"an object {height_m * 1000:.0f} mm tall is at or above "
                         f"the lens, {lens_m * 1000:.0f} mm up")
    return lens_m / (lens_m - height_m)


def _unlift(quad: np.ndarray, nadir: np.ndarray, tag_m: float) -> np.ndarray:
    """Undo the parallax of a tag sitting ABOVE the table plane.

    The homography maps the TABLE. A tag on top of a 40 mm cube is nearer the camera
    than that, so it projects onto the table larger than life and pushed away from
    the point directly beneath the lens. Both effects are the same magnification m:

        apparent = nadir + (true - nadir) * m

    and a tag of KNOWN printed size measures m for us -- it is however many times
    bigger it came out than it really is. So m needs no camera height, no object
    height and no lens model; it is one division, and it is why cfg.OBJECT_TAG_M has
    to be measured rather than guessed.

    `nadir` is the table point directly beneath the LENS, from kin.camera_nadir().
    It used to be taken as the image centre, which is wrong: the lens is 50 mm off
    the forearm axis and its optical axis is not exactly vertical, so at the survey
    pose the two are about 70 mm apart. Correcting about the middle of the picture
    left 20 mm of a 20 mm reach error in place.
    """
    sides = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    magnification = float(np.mean(sides)) / tag_m
    if magnification <= 0:
        return quad
    return nadir + (quad - nadir) / magnification


def markers(
    frame,
    matrix: np.ndarray,
    ignore: set[int] | None = None,
    dictionary_name: str | None = None,
    tag_m: float | None = None,
    nadir: tuple[float, float] | None = None,
    lens_m: float | None = None,
) -> list[Target]:
    """Tags on the table, one Target each.

    Defaults to the OBJECT tag family (AprilTag 36h11), not the calibration board's
    ArUco -- they are deliberately different, so a board left on the table cannot be
    read as an object. `ignore` is kept for the case where they ever coincide.

    `tag_m` is the tag's true printed size. Given it, positions are corrected for the
    tag being raised above the table (see `_unlift`); without it they are reported
    as the homography saw them, which for a tag on a 40 mm cube is several
    millimetres out. Pass None only when the tag really is lying flat.

    Given `lens_m` (the lens's height above the table) as well, the tag's
    magnification says how high up it sits -- m = H / (H - h) -- and that is the
    object's HEIGHT, measured, with no nadir in it. The object is taken to be a
    cube, so its width is reported as that too. Measured 2026-09-11: the lab's
    40 mm cube read 38.4 mm this way. Without `lens_m` the width reported is
    the TAG's, not the object's.
    """
    if tag_m and nadir is None:
        # Deliberately not defaulting to the image centre. That default was wrong by
        # about 70 mm and the error it caused was invisible -- positions simply came
        # back a little off. Better to refuse than to quietly do the wrong thing.
        raise ValueError(
            "correcting for tag height needs the lens position: pass "
            "nadir=kin.camera_nadir(survey_pose)"
        )

    name = dictionary_name or cfg.OBJECT_TAG_DICT
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None or not len(ids):
        return []

    ignore = ignore or set()
    found = []
    for marker, corner in zip(ids.ravel().astype(int), corners):
        if marker in ignore:
            continue
        quad = ws.apply(matrix, corner.reshape(4, 2))
        height = None
        if tag_m:
            apparent = float(np.mean([np.linalg.norm(quad[(i + 1) % 4] - quad[i])
                                      for i in range(4)]))
            if lens_m is not None and apparent > tag_m:
                height = lens_m * (1.0 - tag_m / apparent)
            quad = _unlift(quad, np.asarray(nadir, dtype=float), tag_m)
        target = _target_from_quad(quad, f"tag {marker}", height_m=height)
        if height is not None:
            target = replace(target, width_m=height, length_m=height)
        found.append(target)
    return sorted(found, key=lambda t: t.x)


# A silhouette within this of a tag is the object the tag is stuck to. Half a
# 55 mm cube plus a few mm of disagreement between the two measurements.
FUSE_RADIUS_M = 0.040


def fuse(tagged: list[Target], silhouettes: list[Target],
         radius_m: float = FUSE_RADIUS_M) -> list[Target]:
    """Take each tag's POSITION and the overlapping silhouette's SIZE.

    Neither detector is good at both. A tag is exact about where it is and what it
    is, but it only spans its own printed square -- it knows nothing about the
    object underneath. A silhouette covers the whole object, but it also covers the
    object's shadow, so its centre is dragged towards the shadow and its outline is
    too big. Measured on a 40 mm cube: the tag landed within a few mm, the
    silhouette read 51 x 94 mm and its centre was 12 mm off.

    Over-reading the width is the safe direction -- it opens the fingers wider than
    needed, which still grips -- so the silhouette's size is used as-is rather than
    guessed at. A tag with no silhouette near it keeps its own width, which will be
    too small for anything bigger than the tag; that is why the tag-only path needs
    an object size supplied from somewhere.
    """
    fused = []
    for tag in tagged:
        near = [s for s in silhouettes
                if not s.clipped and math.dist((s.x, s.y), (tag.x, tag.y)) <= radius_m]
        if not near:
            fused.append(tag)
            continue
        biggest = max(near, key=lambda s: s.width_m * s.length_m)
        fused.append(replace(tag, width_m=biggest.width_m, length_m=biggest.length_m,
                             height_m=biggest.height_m))
    return sorted(fused, key=lambda t: t.x)


class DetectorOffline(RuntimeError):
    """The vision container did not answer -- fall back to a lower rung."""


def _vision_post(path: str, body: bytes | None, headers: dict[str, str],
                 url: str, timeout: float) -> dict:
    """One request to the vision service. Raises DetectorOffline if nothing answers.

    An HTTP error (e.g. 400 for a prompt a fixed-class model cannot honour) means
    the service DID reply and the request was wrong -- that is a ValueError, not an
    outage, and must not trigger a silent fall back to a worse detector.
    """
    request = urllib.request.Request(
        url.rstrip("/") + path, data=body, headers=headers,
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise ValueError(f"vision service rejected the request ({exc.code}): {detail}") from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise DetectorOffline(
            f"no vision service at {url} ({exc}). Start it with "
            f"`docker compose --profile vision up -d vision`, or use a lower rung."
        ) from exc


def vision_available(url: str | None = None, timeout: float = 2.0) -> bool:
    """Whether the vision container is up and its model loaded. Never raises."""
    try:
        health = _vision_post("/health", None, {}, url or cfg.DETECTOR_URL, timeout)
    except (DetectorOffline, ValueError):
        return False
    return health.get("status") == "ok"


# An outline covering this much of the picture IS the picture: a prompt-free
# segmenter labels the whole frame ("studio shot", "chemistry lab") on nearly
# every shot. Never an object on the table, and never allowed to swallow one.
PICTURE_FRACTION = 0.8
# An outline with this much of its area inside a larger one is a part of that
# object, not an object of its own.
NESTED_FRACTION = 0.9


def whole_objects(outlines: list[tuple[dict, np.ndarray, bool]],
                  shape: tuple[int, ...]) -> list[tuple[dict, np.ndarray, bool]]:
    """Keep the neural outlines that are whole objects: not the picture, not a part.

    An open-vocabulary segmenter labels PARTS as readily as wholes. The lab's
    tagged cube came back as a 265 px "traffic sign" AND a 178 px "direct" for
    the tag's inner square, at 0.61 and 0.53 -- and ranged as a cube of its own
    the inner square is a 27 mm block at the same spot. Whichever of the two
    scored higher after ranging would then set the gripper opening, and 45 mm
    of opening jams on a 40 mm cube. Only the outermost outline is the object;
    what lies within another outline is part of it. The exception is an outline
    that is the whole frame, which the same models produce on nearly every shot
    and which would otherwise swallow everything on the table -- that one is
    dropped, not honoured.
    """
    height, width = shape[:2]
    masks = []
    for _item, polygon, _boxed in outlines:
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32).reshape(-1, 1, 2)], 1)
        masks.append(mask)
    areas = [int(mask.sum()) for mask in masks]
    picture = [area >= PICTURE_FRACTION * height * width for area in areas]
    kept = []
    for i, entry in enumerate(outlines):
        if areas[i] == 0 or picture[i]:
            continue
        inside = any(
            j != i and not picture[j] and areas[j] > areas[i]
            and int(np.count_nonzero(masks[i] & masks[j])) >= NESTED_FRACTION * areas[i]
            for j in range(len(outlines))
        )
        if not inside:
            kept.append(entry)
    return kept


def objects(
    frame,
    matrix: np.ndarray,
    *,
    prompt: str | None = None,
    url: str | None = None,
    conf: float | None = None,
    timeout: float | None = None,
    nadir: tuple[float, float] | None = None,
    lens_m: float | None = None,
) -> list[Target]:
    """Objects found by the neural detector, in table coordinates.

    `prompt` is a comma-separated list of things to look for and only works against
    an open-vocabulary (YOLOE) model; against a fixed-class model it raises
    ValueError rather than quietly ignoring it. Omit it to get the model's own
    classes.

    Boxes come back in pixels and are mapped through the SAME homography the other
    detectors use -- so the parallax note in this module's header applies here too.
    Given `nadir` and `lens_m` each box is read as a cube's outline and ranged
    (cube_range), which puts the base where it is and sizes the cube; without
    them a tall object's box is its TOP, a few millimetres too far out.
    """
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("could not JPEG-encode the frame to send to the vision service")

    headers = {"Content-Type": "image/jpeg"}
    if prompt:
        headers["X-Vision-Prompt"] = prompt
    if conf is not None:
        headers["X-Vision-Conf"] = str(conf)

    reply = _vision_post(
        "/detect", buffer.tobytes(), headers,
        url or cfg.DETECTOR_URL, cfg.DETECTOR_TIMEOUT_S if timeout is None else timeout,
    )

    outlines: list[tuple[dict, np.ndarray, bool]] = []
    for item in reply.get("detections", []):
        x, y, w, h = item["box"]
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)
        # A segmentation model sends the outline itself, which is what ranging
        # wants; a box only circumscribes it and ranges a little generously.
        polygon = np.asarray(item.get("polygon") or [], dtype=float).reshape(-1, 2)
        boxed = len(polygon) < 3
        if boxed:
            polygon = corners
        outlines.append((item, polygon, boxed))

    found: list[Target] = []
    for item, polygon, boxed in whole_objects(outlines, frame.shape):
        clipped = touches_edge(polygon, frame.shape)
        score = 0.0 if clipped else item.get("confidence", 1.0)
        if nadir is not None and lens_m is not None:
            quad, _width, tall, agreement = range_block(
                ws.apply(matrix, polygon), nadir, lens_m, outline_is_top=boxed,
                chamfer_m=chamfer_for(matrix, polygon))
            score = score * agreement
            target = _target_from_quad(quad, item["label"], score, height_m=tall,
                                       clipped=clipped)
        else:
            quad = ws.apply(matrix, cv2.boxPoints(cv2.minAreaRect(polygon.astype(np.float32))))
            target = _target_from_quad(quad, item["label"], score, clipped=clipped)
        if target.width_m * target.length_m < MIN_AREA_M2:
            continue
        found.append(target)
    return sorted(found, key=lambda t: t.x)


def declared_size(targets: list[Target], width_m: float) -> list[Target]:
    """Replace the measured size with one you measured by hand.

    An escape hatch, and an honest one. Size is the weakest thing vision gives us
    here: a silhouette includes the object's shadow, and separating the two relies on
    the object having colour the shadow does not. On a tan cardboard cube under a
    hard light that separation is thin, and a 40 mm object measured 55 mm -- past the
    gripper's limit, so the pick was refused before it began.

    Position still comes from the camera, which is the part that has to be automatic.
    Telling the robot how big a known object is, is not cheating; a declared object
    size is ordinary practice. What WOULD be cheating is declaring where it is.
    """
    return [replace(t, width_m=width_m, length_m=width_m, height_m=width_m)
            for t in targets]


def annotate(frame, matrix: np.ndarray, targets: list[Target]):
    """Draw the targets back onto the frame, so a human can check the detector.

    Going back through the INVERSE homography means a mistake in the mapping shows
    up as a box in the wrong place, which drawing the original contours would hide.
    """
    out = frame.copy()
    inverse = np.linalg.inv(matrix)
    for target in targets:
        centre = ws.apply(inverse, [[target.x, target.y]])[0]
        colour = (0, 200, 0) if target.graspable else (0, 0, 220)
        cv2.circle(out, (int(centre[0]), int(centre[1])), 6, colour, 2)
        score = "" if target.confidence >= 1.0 else f" {target.confidence:.2f}"
        cv2.putText(
            out,
            f"{target.label}{score} {target.width_m * 1000:.0f}mm",
            (int(centre[0]) + 10, int(centre[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1,
        )
    return out


def background_path(dyaw: float = 0.0, name: str = "primary") -> Path:
    """Where the empty-table photo for one look lives.

    The primary calibrated look (dyaw 0) keeps the original unsuffixed filename so
    that a calibration captured before the sweep existed is still found. Every
    other station gets its yaw in the name, spelled without characters that need
    quoting, and a station of a SECOND calibrated look gets that look's name too
    -- the outer look at dyaw 0 is a different picture from the primary at 0.
    """
    tag = f"{dyaw:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")
    if name != "primary":
        return BACKGROUND_PATH.with_name(f"table_background_{name}_{tag}.png")
    if abs(dyaw) < 0.05:
        return BACKGROUND_PATH
    return BACKGROUND_PATH.with_name(f"table_background_{tag}.png")


def save_background(frame, dyaw: float = 0.0, name: str = "primary") -> None:
    path = background_path(dyaw, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


def load_background(dyaw: float = 0.0, name: str = "primary"):
    path = background_path(dyaw, name)
    if not path.exists():
        which = ("the calibrated look" if abs(dyaw) < 0.05
                 else f"the {dyaw:+.0f} degree look")
        if name != "primary":
            which = f"the {name} look at {dyaw:+.0f} degrees"
        raise FileNotFoundError(
            f"no empty-table photo for {which} at {path} -- clear the table and "
            f"run tools/detect_check.py --background first"
        )
    return cv2.imread(str(path))


# ---------------------------------------------------------------- ladder ----
MODES = ("auto", "changes", "colour", "markers", "yolo")

# What the neural rung is asked for in "auto" mode. The vision container serves
# a YOLOE model whose classes were fixed to these words when it was saved
# (models/yoloe-cubes.pt, see docs), so no prompt goes over the wire -- it is
# recorded here for the day someone re-prompts it.
CUBE_PROMPT = "cube, block, box, dice"

# Two silhouettes of the same object from different rungs land within this of
# each other; two objects the gripper could tell apart never do (22 mm minimum).
SAME_OBJECT_M = 0.020

# How tall the colour rung assumes an object is when it is NOT told the lens
# height and so cannot range it: the lab box lying down. Being a few mm out
# costs a few mm of position, in the safe direction of reporting the object
# nearer the camera than it is.
COLOUR_OBJECT_HEIGHT_M = 0.028

def distinct(targets: list[Target], tol_m: float = SAME_OBJECT_M) -> list[Target]:
    """One target per object when several rungs saw the same thing.

    The surest wins: for a ranged silhouette that is how cube-like its outline
    was, for the neural rung its own score -- so a clean colour blob beats a
    tentative box, and a confident box beats a blob with a shadow on it.
    """
    kept: list[Target] = []
    for target in sorted(targets, key=lambda t: -t.confidence):
        if all(math.dist((t.x, t.y), (target.x, target.y)) > tol_m for t in kept):
            kept.append(target)
    return sorted(kept, key=lambda t: math.hypot(t.x, t.y))


def everything(frame, matrix: np.ndarray, *, nadir: tuple[float, float] | None,
               lens_m: float | None, dyaw: float = 0.0, url: str | None = None,
               look_name: str = "primary", note=print) -> list[Target]:
    """Every rung at once, merged: the "auto" detector.

    Tags say exactly where a tagged object is; silhouettes -- colour, change
    against an empty-table photo if one exists, and the neural detector if it
    is up -- say how big things are and find the ones with no tag. Each
    silhouette is cube-ranged, so a pale cube the colour rung cannot see is
    still sized and placed by the neural one, and a tag on top of any of them
    takes over the position. Nothing here needs a declared size.

    Rungs that are unavailable are skipped, not fallen back through: `url`
    None means the vision service is known to be down, and a missing
    background is simply a missing background.
    """
    tagged = markers(frame, matrix, tag_m=cfg.OBJECT_TAG_M, nadir=nadir, lens_m=lens_m)
    fixed = COLOUR_OBJECT_HEIGHT_M if (nadir is not None and lens_m is None) else 0.0
    shapes = coloured(frame, matrix, nadir=nadir, height_m=fixed, lens_m=lens_m)
    try:
        shapes += changes(frame, load_background(dyaw, look_name), matrix,
                          nadir=nadir, lens_m=lens_m)
    except FileNotFoundError:
        pass
    if url:
        try:
            shapes += objects(frame, matrix, url=url, nadir=nadir, lens_m=lens_m)
        except DetectorOffline as exc:
            note(f"  vision service down ({exc}); untagged pale cubes may be missed")
        except ValueError as exc:
            note(f"  vision service refused the frame: {exc}")
    shapes = distinct(shapes)
    # A tag that measured its own height is a complete reading, better than any
    # silhouette; only a tag that could not (no lens height) borrows a size.
    fused = [t if t.height_m else f for t, f in zip(tagged, fuse(tagged, shapes))]
    untagged = [s for s in shapes
                if all(math.dist((s.x, s.y), (t.x, t.y)) > FUSE_RADIUS_M for t in tagged)]
    return sorted(fused + untagged, key=lambda t: math.hypot(t.x, t.y))


def ladder(frame, mode: str, matrix: np.ndarray, *, prompt: str | None = None,
           nadir: tuple[float, float] | None = None, dyaw: float = 0.0,
           object_mm: float | None = None, url: str | None = None,
           lens_m: float | None = None, look_name: str = "primary",
           note=print) -> list[Target]:
    """Run the chosen detector, dropping down a rung when one is unavailable.

    "auto" is everything() -- the mode that needs no tag, no background and no
    declared size. `lens_m` is the wrist lens's height above the table
    (kin.camera_height), which lets every silhouette be read as a cube.

    `mode` is one of MODES. "yolo" needs the vision container; if nothing answers
    it falls to "markers" with a note rather than failing, because a working lower
    rung beats no detection at all. "markers" fuses the tag's position with the
    silhouette's size when an empty-table photo exists, and says so when it does
    not. "changes" needs that photo and raises FileNotFoundError without it.

    `note` is told about fallbacks -- print for a tool, a log for a server. A
    declared `object_mm` overrides whatever size the detector measured, on every
    rung; the position always stays the camera's.
    """
    if mode not in MODES:
        raise ValueError(f"unknown detector {mode!r}; one of {', '.join(MODES)}")
    found: list[Target] = []
    if mode == "auto":
        found = everything(frame, matrix, nadir=nadir, lens_m=lens_m, dyaw=dyaw,
                           url=url, look_name=look_name, note=note)
    elif mode == "yolo":
        try:
            found = objects(frame, matrix, prompt=prompt, url=url, nadir=nadir, lens_m=lens_m)
        except DetectorOffline as exc:
            note(f"  vision service down ({exc}); falling back to tags.")
            mode = "markers"

    if mode == "colour":
        if lens_m is not None and nadir is not None:
            found = coloured(frame, matrix, nadir=nadir, lens_m=lens_m)
        else:
            found = coloured(frame, matrix, nadir=nadir, height_m=COLOUR_OBJECT_HEIGHT_M)

    if mode == "markers":
        tagged = markers(frame, matrix, tag_m=cfg.OBJECT_TAG_M, nadir=nadir, lens_m=lens_m)
        try:
            found = fuse(tagged, changes(frame, load_background(dyaw, look_name), matrix,
                                         nadir=nadir, lens_m=lens_m))
        except FileNotFoundError:
            if lens_m is None:
                note("  no empty-table photo, so object SIZE comes from the tag itself;")
                note("  anything wider than its tag will be refused. Capture a background first.")
            found = tagged
    elif mode == "changes":
        found = changes(frame, load_background(dyaw, look_name), matrix,
                        nadir=nadir, lens_m=lens_m)

    return declared_size(found, object_mm / 1000) if object_mm else found
