"""The drop-off box both robots share (2026-10-02, the M3's first): a printed marker inside its lid,
which stands up at the far side -- the box found from it with one look, empty or full (its black
walls give a depth camera nothing; the marker reads from 28 cm, 109 px), from the colour picture
alone: no depth. The inside: `along` the lid's hinge, `into` it from the hinge towards the robot,
the walls' height; drops keep BOX_MARGIN from the walls and let go BOX_DROP_OVER over their tops.

Coordinates are the robot's base frame (x forward, y left, z up, m), whose floor is at `floor_z`
(the M3's base_link is on the floor: 0; the X3's base plate is 0.19 m above its table: -0.19).
"""

from __future__ import annotations

import itertools
import math

import cv2
import numpy as np

BOX_MARKER = ("DICT_4X4_50", 10, 0.064)   # family, id, side (m, measured: 64.0-64.7 from the M3's depth)
BOX_INSIDE = (0.15, 0.11, 0.038)          # m: along, across, walls
BOX_MARGIN = 0.025
BOX_LID_MARGIN = 0.045   # m from the lid's side: what is let go comes straight down past the standing lid
                         # (2026-10-05: a drop 2.5 cm from it, and the box was hit and turned)
BOX_DROP_OVER = 0.04
BOX_MOVED_M, BOX_TURNED_DEG = 0.03, 8.0    # seen again further or turned more than this: it was moved
# (2026-10-16: the M3's readings of a box standing still scatter +-1.5 cm and +-3 deg -- the marker is
# ~110 px, the hinge 8 cm under it; 1.5 cm and 5 deg called it moved before nearly every drop)
BOX_WALL = 0.01   # m the walls are thick: what keeps clear of the box keeps clear of them too


def find_marker(image: np.ndarray, K, camera: np.ndarray, family: str, ident: int,
                side: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Where the printed marker `ident` of the OpenCV ArUco family `family` (e.g. "DICT_4X4_50"),
    `side` m square, is: (its centre, base m; its axes as the columns of a rotation -- x right, y
    up, z out of its face towards whoever reads it), in base through `camera` (base <- the camera's
    optical frame, 4 x 4); None if not seen. The picture undistorted already (K alone)."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    family = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, family))
    if hasattr(cv2.aruco, "ArucoDetector"):   # OpenCV 4.7 on
        corners, ids, _ = cv2.aruco.ArucoDetector(family, cv2.aruco.DetectorParameters()).detectMarkers(grey)
    else:                                     # the M3 host's 4.5.4
        corners, ids, _ = cv2.aruco.detectMarkers(grey, family, parameters=cv2.aruco.DetectorParameters_create())
    if ids is None or ident not in ids.ravel():
        return None
    seen = corners[list(ids.ravel()).index(ident)].reshape(4, 2)
    square = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], float) * side / 2
    ok, rvec, tvec = cv2.solvePnP(square, seen, np.asarray(K, float).reshape(3, 3), None,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    return (camera @ np.r_[tvec.ravel(), 1.0])[:3], camera[:3, :3] @ cv2.Rodrigues(rvec)[0]


def box_from_marker(centre: np.ndarray, axes: np.ndarray, floor_z: float = 0.0) -> dict:
    """The box's inside from its marker (find_marker): the hinge where the lid's plane meets the
    walls' top, the inside from there towards the robot -- {"centre" (x, y), "along", "into" (unit
    x, y), "turn" (deg, of "along")}."""
    _along_m, across_m, wall = BOX_INSIDE
    face, right, up = axes[:, 2], axes[:, 0], axes[:, 1]
    hinge = centre - up * (centre[2] - (floor_z + wall)) / up[2]
    into = np.array([face[0], face[1]]) / np.hypot(face[0], face[1])
    along = np.array([right[0], right[1]]) / np.hypot(right[0], right[1])
    middle = hinge[:2] + into * across_m / 2
    return {"centre": middle, "along": along, "into": into, "turn": math.degrees(math.atan2(along[1], along[0]))}


def in_box(box: dict, point) -> np.ndarray:
    """A floor point in the box's own terms: (along, across) from its middle -- what is in it moves
    with it when it is moved."""
    rel = np.asarray(point[:2], float) - box["centre"]
    return np.array([rel @ box["along"], rel @ box["into"]])


def out_of_box(box: dict, rel) -> np.ndarray:
    return box["centre"] + box["along"] * rel[0] + box["into"] * rel[1]


def inside(box: dict, point, margin: float = 0.0) -> bool:
    """`point` (x, y) over the box's inside, grown by `margin` all round."""
    a, b = in_box(box, point)
    return abs(a) < BOX_INSIDE[0] / 2 + margin and abs(b) < BOX_INSIDE[1] / 2 + margin


def box_corners(box: dict, lift: float = 0.0, floor_z: float = 0.0) -> list[np.ndarray]:
    along_m, across_m, wall = BOX_INSIDE
    return [np.r_[box["centre"] + box["along"] * a * along_m / 2 + box["into"] * b * across_m / 2, floor_z + wall + lift]
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def box_spots(box: dict, step: float = 0.02) -> list[np.ndarray]:
    """Spots inside the box to let go over, BOX_MARGIN clear of its walls and BOX_LID_MARGIN of the
    lid's side (the hinge: -`into`), `step` m apart."""
    along_m, across_m, _ = BOX_INSIDE
    a_half = along_m / 2 - BOX_MARGIN
    b_lo, b_hi = -across_m / 2 + BOX_LID_MARGIN, across_m / 2 - BOX_MARGIN
    return [box["centre"] + box["along"] * a + box["into"] * b
            for a in np.arange(-a_half, a_half + 1e-9, step) for b in np.arange(b_lo, b_hi + 1e-9, step)]


def same_box(seen: dict, known: dict) -> bool:
    """`seen` is where `known` was: not moved further than BOX_MOVED_M nor turned more than BOX_TURNED_DEG."""
    return float(np.linalg.norm(seen["centre"] - known["centre"])) < BOX_MOVED_M \
        and abs((seen["turn"] - known["turn"] + 180) % 360 - 180) < BOX_TURNED_DEG


def merged(readings: list[dict]) -> dict:
    """One box from readings of it standing still (same_box): their mean centre and turn -- each
    drop aimed by the mean of the job's looks, not by the last one's scatter."""
    turn = math.atan2(np.mean([math.sin(math.radians(r["turn"])) for r in readings]),
                      np.mean([math.cos(math.radians(r["turn"])) for r in readings]))
    along = np.array([math.cos(turn), math.sin(turn)])
    last = readings[-1]
    side = 1.0 if along[0] * last["into"][1] - along[1] * last["into"][0] > 0 else -1.0   # into: along turned 90 deg
    return {**last, "centre": np.mean([r["centre"] for r in readings], axis=0), "along": along,
            "into": side * np.array([-along[1], along[0]]), "turn": math.degrees(turn)}


def outline(box: dict, grow: float = BOX_WALL) -> list[np.ndarray]:
    """The box's inside's corners grown by `grow` (its walls, by default), floor x, y."""
    along_m, across_m, _ = BOX_INSIDE
    return [box["centre"] + box["along"] * a * (along_m / 2 + grow) + box["into"] * b * (across_m / 2 + grow)
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def outline_points(corners: list[np.ndarray], step: float = 0.005) -> np.ndarray:
    """Points round a polygon's edges, `step` m apart."""
    out = []
    for a, b in itertools.pairwise([*corners, corners[0]]):
        n = max(1, math.ceil(float(np.linalg.norm(b - a)) / step))
        out.append(a + (b - a) * (np.arange(n)[:, None] / n))
    return np.vstack(out)
