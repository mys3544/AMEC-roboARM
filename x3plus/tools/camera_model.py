#!/usr/bin/env python3
"""The wrist camera as a measuring instrument (x3plus/PLAN.md X2.1), step one: its lens.

    python tools/camera_model.py intrinsics            # capture, then fit
    python tools/camera_model.py intrinsics --refit    # fit the saved views again

Someone holds a ChArUco board (docs/board_a4.png printed and taped to card, or any other) in
front of the wrist camera at many TILTS and distances while this takes pictures from the
robot's panel (it owns the camera; the pictures are already undistorted by cfg.WRIST_CAM_LENS).
Which board it is -- dictionary, squares, marker size -- is read off the first picture it is
in; its real size does not matter for the lens (a board twice as big twice as far away looks
the same), so it is not asked for. The A3 board on the table shares marker ids with the A4
one, so it must be out of view or COVERED, not moved, meanwhile.

Why tilts: every earlier fit came from views looking straight down at the table, and the focal
length came out anywhere from 416 to 501 px -- a plane seen square-on cannot tell a near small
board from a far big one. Tilted views pin it. The fit is made on the undistorted pictures, so
its K is the camera everything downstream sees, and its distortion is what the current
correction leaves (it should be near zero).

Writes data/cameras/<camera>/intrinsics.json and keeps the views in out/camera_model/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roboarm import config as cfg

ROOT = Path(__file__).resolve().parents[1]
PANEL = "http://192.168.73.210:8080"
A4 = {"squares": (9, 7), "marker_ratio": 0.75, "dict": "DICT_5X5_250"}   # docs/board_a4.png
MIN_CORNERS = 15     # of the A4 board's 48
MAX_VIEWS = 40
TILT_BIN_DEG = 12    # views are kept one per (tilt x, tilt y, distance, where in the picture)
DISTANCE_BIN_M = 0.10


SQUARE_M = 0.030     # nominal: the lens does not depend on it (see above)
# What a held-up board may be: the families our boards and Yahboom's use, at their largest size
# (the smaller dictionaries are their first ids), and the marker sizes boards are printed with.
DICTIONARIES = ("DICT_5X5_1000", "DICT_4X4_1000", "DICT_6X6_1000", "DICT_7X7_1000")
MARKER_RATIOS = (0.75, 0.7, 0.8, 0.6, 0.5)
LAYOUT_RMS_PX = 3.0  # a wrong layout puts corners in the wrong places: they stop lying on a plane


def a4_board(square_m: float = SQUARE_M) -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, A4["dict"]))
    return cv2.aruco.CharucoBoard(A4["squares"], square_m, square_m * A4["marker_ratio"], dictionary)


def identify(frame: np.ndarray):
    """(board, its description) for the ChArUco board in `frame`, or None: the dictionary with
    the most markers, then the layout (squares, marker size, legacy pattern) under which the most
    corners are found AND they lie on one plane."""
    grey = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best = None
    for name in DICTIONARIES:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        _corners, ids, _rejected = cv2.aruco.ArucoDetector(dictionary).detectMarkers(grey)
        if ids is None or len(ids) < 4:
            continue
        top = int(ids.max())
        for cols in range(3, 15):
            for rows in range(3, 15):
                if not top < cols * rows // 2 <= top + 15:   # a board's markers are ids 0..n-1
                    continue
                for ratio in MARKER_RATIOS:
                    for legacy in (False, True) if rows % 2 == 0 else (False,):
                        board = cv2.aruco.CharucoBoard((cols, rows), SQUARE_M, SQUARE_M * ratio, dictionary)
                        board.setLegacyPattern(legacy)
                        found = find_board(grey, board)
                        if found is None or (best is not None and len(found[0]) <= best[0]):
                            continue
                        _H, _mask = cv2.findHomography(found[0][:, :2], found[1], 0)
                        if _H is None:
                            continue
                        mapped = cv2.perspectiveTransform(found[0][None, :, :2], _H)[0]
                        rms = float(np.sqrt(np.mean(np.sum((mapped - found[1]) ** 2, axis=1))))
                        if rms <= LAYOUT_RMS_PX:
                            what = (f"{name}, {cols} x {rows} squares, markers {ratio:.2f} of a square"
                                    + (", legacy pattern" if legacy else ""))
                            best = (len(found[0]), board, what)
    return None if best is None else (best[1], best[2])


def find_board(frame: np.ndarray, board: cv2.aruco.CharucoBoard):
    """(object points Nx3, image points Nx2) of the board's corners in `frame`, or None."""
    grey = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _markers, _marker_ids = cv2.aruco.CharucoDetector(board).detectBoard(grey)
    if ids is None or len(ids) < MIN_CORNERS:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    return obj.reshape(-1, 3).astype(np.float32), img.reshape(-1, 2).astype(np.float32)


def view_key(obj: np.ndarray, img: np.ndarray, K: np.ndarray, size: tuple[int, int]) -> tuple:
    """Which kind of view this is: the board's tilt about the camera's x and y, its distance,
    and the third of the picture it sits in. One view is kept per kind, so a minute of holding
    the board still yields one view, not ninety."""
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None)
    if not ok:
        return None
    normal = cv2.Rodrigues(rvec)[0][:, 2]
    if normal[2] > 0:   # the side facing the camera, so the tilts are small numbers, not +-180
        normal = -normal
    tilt_x = math.degrees(math.atan2(normal[1], -normal[2]))
    tilt_y = math.degrees(math.atan2(normal[0], -normal[2]))
    centre = img.mean(axis=0)
    return (round(tilt_x / TILT_BIN_DEG), round(tilt_y / TILT_BIN_DEG),
            round(float(tvec[2][0]) / DISTANCE_BIN_M),
            int(3 * centre[0] / size[0]), int(3 * centre[1] / size[1]))


def coverage(views: list, size: tuple[int, int], cells: tuple[int, int] = (6, 4)) -> float:
    """Share of a grid over the picture that some view put a corner in: a fit only knows
    the lens where it saw corners."""
    seen = set()
    for _obj, img in views:
        for x, y in img:
            seen.add((min(int(x * cells[0] / size[0]), cells[0] - 1),
                      min(int(y * cells[1] / size[1]), cells[1] - 1)))
    return len(seen) / (cells[0] * cells[1])


def fit(views: list, size: tuple[int, int]) -> dict:
    """K and the residual distortion (k1 k2 p1 p2; k3 held at 0) from the kept views."""
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [o for o, _ in views], [i for _, i in views], size, None, None, flags=cv2.CALIB_FIX_K3)
    per_view = []
    for (obj, img), rvec, tvec in zip(views, rvecs, tvecs, strict=True):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - img) ** 2, axis=1)))))
    return {"K": K.tolist(), "dist": dist.ravel().tolist(), "rms_px": float(rms),
            "per_view_px": [round(e, 3) for e in per_view], "views": len(views),
            "coverage": round(coverage(views, size), 2), "size": list(size)}


def grab(panel: str) -> np.ndarray | None:
    with urllib.request.urlopen(f"{panel}/snapshot.jpg?view=raw", timeout=5) as reply:
        data = np.frombuffer(reply.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def capture(panel: str, board, folder: Path, seconds: float, wait: float = 0.0):
    """Keep one picture per kind of view of the held-up board; the board, recognised from the
    first picture it is in when not given. The `seconds` start when the board is first seen,
    up to `wait` seconds from now: whoever holds it need not be told when to start."""
    K = np.array(cfg.WRIST_CAM_LENS[0], np.float64) if cfg.WRIST_CAM_LENS else None
    folder.mkdir(parents=True, exist_ok=True)
    kept: dict[tuple, int] = {}
    end = time.monotonic() + wait + seconds
    seen = False
    print(f"hold the board up to the camera; tilt it, move it near and far, into the corners "
          f"({seconds:.0f} s, up to {MAX_VIEWS} views)", flush=True)
    while time.monotonic() < end and len(kept) < MAX_VIEWS:
        frame = grab(panel)
        if frame is None:
            continue
        size = frame.shape[1::-1]
        if K is None:
            K = np.array([[size[0], 0, size[0] / 2], [0, size[0], size[1] / 2], [0, 0, 1]], np.float64)
        if board is None:
            known = identify(frame)
            if known is None:
                time.sleep(0.2)
                continue
            board, what = known
            print(f"  the board: {what}", flush=True)
            (folder / "board.txt").write_text(what)
        found = find_board(frame, board)
        if found is None:
            time.sleep(0.2)
            continue
        if not seen:   # the clock starts now
            seen, end = True, time.monotonic() + seconds
        key = view_key(*found, K, size)
        if key is None or key in kept:
            time.sleep(0.2)
            continue
        kept[key] = len(kept)
        cv2.imwrite(str(folder / f"view_{len(kept):02d}.png"), frame)
        print(f"  view {len(kept):2d}: {len(found[0])} corners, tilt {key[0] * TILT_BIN_DEG:+d}/"
              f"{key[1] * TILT_BIN_DEG:+d} deg, ~{key[2] * DISTANCE_BIN_M:.1f} m", flush=True)
    print(f"{len(kept)} views kept", flush=True)
    return board


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="what", required=True)
    intr = sub.add_parser("intrinsics", help="the lens: K and residual distortion")
    intr.add_argument("--board", choices=("auto", "a4"), default="auto",
                      help="recognise the board from the pictures (default), or docs/board_a4.png")
    intr.add_argument("--panel", default=PANEL, help="the robot's panel (it owns the camera)")
    intr.add_argument("--seconds", type=float, default=120, help="capturing, from the board's first sighting")
    intr.add_argument("--wait", type=float, default=0, help="how long to wait for the board to be held up")
    intr.add_argument("--refit", action="store_true", help="fit the views saved last time again")
    args = parser.parse_args()

    board = a4_board() if args.board == "a4" else None
    folder = ROOT / "out" / "camera_model" / "intrinsics"
    if not args.refit:
        for old in folder.glob("view_*.png"):
            old.unlink()
        board = capture(args.panel, board, folder, args.seconds, args.wait)
    views, size, what = [], None, "docs/board_a4.png"
    for path in sorted(folder.glob("view_*.png")):
        frame = cv2.imread(str(path))
        size = frame.shape[1::-1]
        if board is None:
            known = identify(frame)
            if known is None:
                continue
            board, what = known
        found = find_board(frame, board)
        if found is not None:
            views.append(found)
    if len(views) < 8:
        print(f"only {len(views)} usable views: at least 8, tilted different ways, are needed")
        return 1
    result = fit(views, size)
    if (folder / "board.txt").is_file() and args.board == "auto":
        what = (folder / "board.txt").read_text()
    result.update(camera=cfg.WRIST_CAMERA, board=what,
                  frames="undistorted by cfg.WRIST_CAM_LENS", when=time.strftime("%Y-%m-%d %H:%M"),
                  before=None if cfg.WRIST_CAM_LENS is None else cfg.WRIST_CAM_LENS[0])
    out = ROOT / "data" / "cameras" / cfg.WRIST_CAMERA / "intrinsics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    K = result["K"]
    print(f"fx {K[0][0]:.1f}  fy {K[1][1]:.1f}  cx {K[0][2]:.1f}  cy {K[1][2]:.1f}  "
          f"rms {result['rms_px']:.2f} px over {result['views']} views, "
          f"picture covered {100 * result['coverage']:.0f} %")
    print(f"residual distortion k1 k2 p1 p2: {' '.join(f'{v:+.4f}' for v in result['dist'][:4])}")
    if result["before"]:
        print(f"before: fx {result['before'][0][0]}  (cfg.WRIST_CAM_LENS)")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
