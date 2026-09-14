"""Map camera pixels to table coordinates, from one fixed survey pose.

The wrist camera is the only one that sees the workspace, and it moves with the arm
-- so a pixel means nothing on its own. It only means something from a KNOWN pose.
The arm therefore drives to a fixed survey pose, looks down, and from there a single
3x3 homography turns pixels into table coordinates.

A homography is enough because everything of interest lies on one plane (the table).
It absorbs the camera's intrinsics, its pose and the 50 mm offset between lens and
fingertip into one matrix fitted from correspondences, so none of those has to be
measured or trusted separately. What it does NOT absorb is lens distortion, which is
why the fit reports its worst residual: that is where distortion would show up.

The survey pose must be repeatable, and it is -- measured joint repeatability is
1 degree, and the fit is only valid from the pose it was taken at.

VALIDATED on the robot 2026-09-08. The camera picked a marker corner, the arm was
sent to where this mapping said it was, and the fingertip landed 2 mm to its left.
That end-to-end check is what counts: the 0.5 mm fit residual only says the
homography reproduces its own 8 input points, and would look just as good if the
board placement constants were wrong.

The HEIGHT is not validated by this and is not part of the homography -- everything
here lives on the table plane. In that same test the fingertip sat 10 mm above the
table when 4 mm was asked for, a 6 mm error consistent with the kinematics' own
worst-case residual. Grasping will need that dealt with separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from roboarm import config as cfg

CALIBRATION_PATH = Path("/app/data/table_homography.json")


def board() -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.ARUCO_DICT))
    return cv2.aruco.CharucoBoard(
        cfg.BOARD_SQUARES, cfg.BOARD_SQUARE_M, cfg.BOARD_MARKER_M, dictionary
    )


def find_corners(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect MARKER corners. Returns (pixels Nx2, board points Nx2 in metres).

    Marker corners, not chessboard corners. A chessboard corner can only be
    interpolated when markers on BOTH sides of it are fully visible, and the wrist
    camera cannot get high enough for that: at ~157 mm across, a 52.5 mm board shows
    barely three squares and clips every marker but the middle one. We reliably see
    two or three whole markers, which is 8-12 corner points spread over ~105 mm --
    ample for a homography.

    The cost is that marker corner positions depend on BOARD_MARKER_M, so that has to
    be measured rather than inferred.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.ARUCO_DICT))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None or len(ids) == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    layout = {int(i): np.array(o).reshape(4, 3) for i, o in zip(board().getIds().ravel(),
                                                               board().getObjPoints())}
    pixels, points = [], []
    for marker, corner in zip(ids.ravel().astype(int), corners):
        if marker not in layout:
            continue
        for pixel, obj in zip(corner.reshape(4, 2), layout[marker]):
            pixels.append(pixel)
            points.append(cfg.board_to_table(float(obj[0]), float(obj[1])))
    return np.array(pixels), np.array(points)


def average_corners(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Detect across several frames and average each corner's pixel position.

    Only two markers are ever in view, so there is very little redundancy in the fit
    and single-frame detection noise goes straight into the result. Averaging a
    handful of frames costs nothing and takes the jitter out. Corners are matched
    between frames by their table position, which is exact and frame-independent.
    """
    seen: dict[tuple[int, int], list[np.ndarray]] = {}
    table_of: dict[tuple[int, int], np.ndarray] = {}
    for frame in frames:
        pixels, table = find_corners(frame)
        for pixel, point in zip(pixels, table):
            key = (round(point[0] * 1e5), round(point[1] * 1e5))
            seen.setdefault(key, []).append(pixel)
            table_of[key] = point
    if not seen:
        return np.empty((0, 2)), np.empty((0, 2))
    keys = sorted(seen)
    return (
        np.array([np.mean(seen[k], axis=0) for k in keys]),
        np.array([table_of[k] for k in keys]),
    )


def fit(frame) -> tuple[np.ndarray, float, int]:
    """Fit the pixel -> table homography. Accepts one frame or a list of them.

    Returns (H, worst residual m, n points).
    """
    pixels, table = (find_corners(frame) if isinstance(frame, np.ndarray)
                     else average_corners(frame))
    return fit_points(pixels, table)


def fit_points(pixels: np.ndarray, table: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Fit the pixel -> table homography from matched points. See fit().

    Split out so a fit can pool correspondences from SEVERAL frames of the same
    look -- e.g. the outer look, whose picture holds one whole marker, fitted
    from that marker seen at several base yaws (tools/calibrate_table.py --yaws).
    """
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    table = np.asarray(table, dtype=float).reshape(-1, 2)
    if len(pixels) < 4:
        raise ValueError(f"need at least 4 marker corners, found {len(pixels)}")

    matrix, _mask = cv2.findHomography(pixels, table, method=0)
    if matrix is None:
        raise ValueError("homography fit failed")

    predicted = apply(matrix, pixels)
    worst = float(np.max(np.linalg.norm(predicted - table, axis=1)))
    return matrix, worst, len(pixels)


def apply(matrix: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Pixels (Nx2) -> table (forward, left) in metres."""
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    out = cv2.perspectiveTransform(pixels.reshape(-1, 1, 2), matrix)
    return out.reshape(-1, 2)


def save(matrix: np.ndarray, survey_pose: dict[int, int], worst: float, n: int) -> None:
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(
        json.dumps(
            {
                "homography": matrix.tolist(),
                # Recorded because the fit is ONLY valid from this pose. Loading it
                # and shooting from anywhere else would be silently wrong.
                "survey_pose": {str(k): v for k, v in survey_pose.items()},
                "worst_residual_mm": worst * 1000,
                "points": n,
            },
            indent=2,
        )
    )


def load() -> tuple[np.ndarray, dict[int, int]]:
    """The PRIMARY calibrated look -- the one every earlier tool was written against."""
    matrix, pose, _name = load_all()[0]
    return matrix, pose


def _one(entry: dict) -> tuple[np.ndarray, dict[int, int], str]:
    return (np.array(entry["homography"]),
            {int(k): v for k, v in entry["survey_pose"].items()},
            entry.get("name", "primary"))


def parse(data: dict) -> list[tuple[np.ndarray, dict[int, int], str]]:
    """The calibration file's contents -> every look in it, primary first.

    Split from load_all() so the same file can arrive over the network (the
    hardware bridge serves it to a control panel running elsewhere).
    """
    return [_one(data)] + [_one(extra) for extra in data.get("also", [])]


def load_all() -> list[tuple[np.ndarray, dict[int, int], str]]:
    """Every calibrated look in the file, primary first.

    ONE FILE, SEVERAL LOOKS. Yawing the base rotates a look for free (see
    roboarm/sweep.py), which covers every bearing the arm can reach -- but it
    cannot change how FAR the camera is looking, because a yaw maps the table to
    itself and a tilt does not. Reaching the outer rim of the workspace therefore
    needs a genuinely second calibration, at a pose tilted further out, and this
    is where it lives: appended under "also" by tools/calibrate_table.py --outer.

    Backwards compatible on purpose. A file written before any of this existed has
    no "also" key and simply yields one look, so nothing that predates the sweep
    has to know the format grew.
    """
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            f"no calibration at {CALIBRATION_PATH} -- run tools/calibrate_table.py"
        )
    return parse(json.loads(CALIBRATION_PATH.read_text()))


def add_look(name: str, matrix: np.ndarray, survey_pose: dict[int, int],
             worst: float, n: int) -> None:
    """Append (or replace) a secondary calibrated look, keeping the primary intact.

    Replacing by name rather than always appending, so that re-running a
    calibration corrects the file instead of quietly stacking a second, staler
    copy of the same look behind the new one.
    """
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            f"no primary calibration at {CALIBRATION_PATH} -- fit that first"
        )
    if name == "primary":
        raise ValueError("'primary' is the look save() writes; pick another name")
    data = json.loads(CALIBRATION_PATH.read_text())
    entry = {
        "name": name,
        "homography": matrix.tolist(),
        "survey_pose": {str(k): v for k, v in survey_pose.items()},
        "worst_residual_mm": worst * 1000,
        "points": n,
    }
    also = [e for e in data.get("also", []) if e.get("name") != name]
    data["also"] = also + [entry]
    CALIBRATION_PATH.write_text(json.dumps(data, indent=2))
