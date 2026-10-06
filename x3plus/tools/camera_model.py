#!/usr/bin/env python3
"""The wrist camera as a measuring instrument (x3plus/PLAN.md X2.1), step one: its lens.

    python tools/camera_model.py intrinsics --square-mm 29.7            # capture, then fit
    python tools/camera_model.py intrinsics --square-mm 29.7 --refit    # fit the saved views again

Someone holds the A4 ChArUco board (docs/board_a4.png, printed and taped to card) in front
of the wrist camera at many TILTS and distances while this takes pictures from the robot's
panel (it owns the camera; the pictures are already undistorted by cfg.WRIST_CAM_LENS). The
A3 board on the table shares its marker ids, so it must be COVERED, not moved, meanwhile.

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


def a4_board(square_m: float) -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, A4["dict"]))
    return cv2.aruco.CharucoBoard(A4["squares"], square_m, square_m * A4["marker_ratio"], dictionary)


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


def capture(panel: str, board, folder: Path, seconds: float) -> None:
    K = np.array(cfg.WRIST_CAM_LENS[0], np.float64) if cfg.WRIST_CAM_LENS else None
    folder.mkdir(parents=True, exist_ok=True)
    kept: dict[tuple, int] = {}
    end = time.monotonic() + seconds
    print(f"hold the A4 board up to the camera; tilt it, move it near and far, into the corners "
          f"({seconds:.0f} s, up to {MAX_VIEWS} views)", flush=True)
    while time.monotonic() < end and len(kept) < MAX_VIEWS:
        frame = grab(panel)
        if frame is None:
            continue
        size = frame.shape[1::-1]
        if K is None:
            K = np.array([[size[0], 0, size[0] / 2], [0, size[0], size[1] / 2], [0, 0, 1]], np.float64)
        found = find_board(frame, board)
        if found is None:
            time.sleep(0.2)
            continue
        key = view_key(*found, K, size)
        if key is None or key in kept:
            time.sleep(0.2)
            continue
        kept[key] = len(kept)
        cv2.imwrite(str(folder / f"view_{len(kept):02d}.png"), frame)
        print(f"  view {len(kept):2d}: {len(found[0])} corners, tilt {key[0] * TILT_BIN_DEG:+d}/"
              f"{key[1] * TILT_BIN_DEG:+d} deg, ~{key[2] * DISTANCE_BIN_M:.1f} m", flush=True)
    print(f"{len(kept)} views kept", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="what", required=True)
    intr = sub.add_parser("intrinsics", help="the lens: K and residual distortion")
    intr.add_argument("--square-mm", type=float, required=True,
                      help="one black square of the PRINTED A4 board, measured")
    intr.add_argument("--panel", default=PANEL, help="the robot's panel (it owns the camera)")
    intr.add_argument("--seconds", type=float, default=120)
    intr.add_argument("--refit", action="store_true", help="fit the views saved last time again")
    args = parser.parse_args()

    board = a4_board(args.square_mm / 1000)
    folder = ROOT / "out" / "camera_model" / "intrinsics"
    if not args.refit:
        for old in folder.glob("view_*.png"):
            old.unlink()
        capture(args.panel, board, folder, args.seconds)
    views, size = [], None
    for path in sorted(folder.glob("view_*.png")):
        frame = cv2.imread(str(path))
        size = frame.shape[1::-1]
        found = find_board(frame, board)
        if found is not None:
            views.append(found)
    if len(views) < 8:
        print(f"only {len(views)} usable views: at least 8, tilted different ways, are needed")
        return 1
    result = fit(views, size)
    result.update(camera=cfg.WRIST_CAMERA, square_mm=args.square_mm, board="docs/board_a4.png",
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
