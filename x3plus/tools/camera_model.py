#!/usr/bin/env python3
"""The wrist camera as a measuring instrument (x3plus/PLAN.md X2.1): its lens, and where it sits.

    python tools/camera_model.py intrinsics            # the lens: capture, then fit
    python tools/camera_model.py intrinsics --refit    # fit the saved views again
    python tools/camera_model.py handeye               # the lens on the arm (roboarm/lens.py), from
                                                       # board views with their poses
    python tools/camera_model.py handeye --capture     # first take those views: the ARM MOVES over
                                                       # the A3 board, ~60 poses, read back at each

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
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roboarm import camera, lens, sweep
from roboarm import config as cfg
from roboarm import kinematics as kin
from roboarm import workspace as ws

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


FX_SD_GOOD = 0.01    # the focal length pinned to 1 %: at 0.3 m that is 3 mm of range
TILT_GOOD_DEG = 30   # what pins it: views tilted this much (2026-10-06: tilts of 2-17 deg left
                     # fx at 476 +- 9 px, and 476 / 487 / 501 by the distortion terms fitted)


def fit(views: list, size: tuple[int, int]) -> dict:
    """K and the residual distortion (k1 k2 p1 p2; k3 held at 0) from the kept views, with
    how well the views pin them: the focal length's standard deviation and the strongest tilt."""
    rms, K, dist, rvecs, tvecs, sd, _sd_extr, _per = cv2.calibrateCameraExtended(
        [o for o, _ in views], [i for _, i in views], size, None, None, flags=cv2.CALIB_FIX_K3)
    tilts = [math.degrees(math.acos(min(1.0, abs(cv2.Rodrigues(r)[0][2, 2])))) for r in rvecs]
    per_view = []
    for (obj, img), rvec, tvec in zip(views, rvecs, tvecs, strict=True):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - img) ** 2, axis=1)))))
    return {"K": K.tolist(), "dist": dist.ravel().tolist(), "rms_px": float(rms),
            "per_view_px": [round(e, 3) for e in per_view], "views": len(views),
            "coverage": round(coverage(views, size), 2), "size": list(size),
            "fx_sd_px": round(float(sd[0, 0]), 2), "tilts_deg": [round(v) for v in tilts]}


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
    started = time.strftime("%H%M%S")   # names that a later --add capture does not overwrite
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
        cv2.imwrite(str(folder / f"view_{started}_{len(kept):02d}.png"), frame)
        print(f"  view {len(kept):2d}: {len(found[0])} corners, tilt {key[0] * TILT_BIN_DEG:+d}/"
              f"{key[1] * TILT_BIN_DEG:+d} deg, ~{key[2] * DISTANCE_BIN_M:.1f} m", flush=True)
    print(f"{len(kept)} views kept", flush=True)
    return board


DATA = ROOT / "data" / "cameras" / cfg.WRIST_CAMERA
# The views for the lens on the arm: (fingertip forward, lift above the table, tool pitch or None
# for solve's choice) x base yaws. The first eight are 2026-09-24's (handeye_views.json, which kept
# the COMMANDED pose -- the arm stops up to 2 deg short -- and fitted no better than 10 px); the
# other four add camera heights of 17-27 cm at tilts of 150-165 deg. Steeper than 150 nothing
# both reachable and looking at the board was found (2026-10-06, a grid over 130-145 deg).
EYE_TARGETS = ((0.150, 0.090, None), (0.150, 0.140, None), (0.110, 0.110, 180.0), (0.190, 0.125, None),
               (0.210, 0.120, 150.0), (0.190, 0.080, 160.0), (0.150, 0.100, 170.0), (0.170, 0.130, 160.0),
               (0.170, 0.160, 150.0), (0.230, 0.060, 150.0), (0.150, 0.160, 155.0), (0.130, 0.140, 165.0))
EYE_YAWS = (-25, -10, 0, 10, 25)
EYE_DPS = 20
EYE_SETTLE_S = 0.8
MIN_EYE_CORNERS = 8


def panel_call(panel: str, path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(panel + path, data=data, headers={"Content-Type": "application/json"},
                                     method="GET" if body is None else "POST")
    with urllib.request.urlopen(request, timeout=60) as reply:
        return json.loads(reply.read())


def capture_eye(panel: str, path: Path, min_lift: float = 0.0) -> int:
    """Drive the arm over the A3 board through EYE_TARGETS x EYE_YAWS (through the panel, which
    checks every move), and keep each view's marker corners with the pose READ BACK there."""
    panel_call(panel, "/api/mode", {"mode": "manual"})
    views = []
    for fwd, lift, pitch in EYE_TARGETS:
        if lift < min_lift:   # things on the table: keep the hand well above them
            continue
        try:
            pose, _ = kin.solve(fwd, 0.0, -cfg.TABLE_BELOW_PLATE + lift,
                                pitches=kin.GRASP_PITCHES if pitch is None else (pitch,))
        except kin.Unreachable as exc:
            print(f"  {fwd * 1000:.0f}/{lift * 1000:.0f}: not reachable ({exc})", flush=True)
            continue
        pose[cfg.GRIPPER_ID] = cfg.GRIPPER_OPEN
        for yaw in EYE_YAWS:
            label = f"{fwd * 1000:.0f}/{lift * 1000:.0f}/{pitch or 'auto'} yaw {yaw:+d}"
            try:
                command = sweep.pose_at(pose, yaw)
            except sweep.NoLook:
                continue
            try:
                panel_call(panel, "/api/arm/move", {"joints": {str(j): v for j, v in command.items()},
                                                    "speed_dps": EYE_DPS})
            except urllib.error.HTTPError as exc:
                reason = json.loads(exc.read()).get("error", exc.reason)
                print(f"  {label}: refused or stopped -- {reason}", flush=True)
                if "interrupt" in str(reason) or "stopped" in str(reason):
                    return 1
                continue
            time.sleep(EYE_SETTLE_S)
            read = panel_call(panel, "/api/state")["arm"]["pose"]
            try:   # the joints to 0.08 deg (Arm.read_fine): whole degrees left this fit at 9 px
                read = {**read, **panel_call(panel, "/api/arm/fine")["pose"]}
            except urllib.error.HTTPError:
                pass
            frames = [grab(panel) for _ in range(3)]
            pixels, table = ws.average_corners([f for f in frames if f is not None])
            print(f"  {label}: {len(pixels)} corners, read J2..J4 {read['2']}/{read['3']}/{read['4']} "
                  f"(sent {command[2]}/{command[3]}/{command[4]})", flush=True)
            if len(pixels) >= MIN_EYE_CORNERS:
                views.append({"label": label, "pose": read, "command": {str(j): v for j, v in command.items()},
                              "pixels": pixels.tolist(), "table": table.tolist()})
            path.write_text(json.dumps(views))   # kept as it goes: a stop loses nothing taken
    print(f"{len(views)} views saved to {path}", flush=True)
    panel_call(panel, "/api/arm/preset", {"name": "survey"})
    return 0


def handeye(path: Path) -> int:
    """Fit roboarm/lens.py's mount to the views in `path`, check it on views it was not fitted
    to (each pose's three yaws left out in turn), and save it beside the lens fit."""
    known = camera.intrinsics(DATA / "intrinsics.json")
    if known is None:
        print("no lens yet: run `intrinsics` first")
        return 1
    K, dist = known
    views = json.loads(path.read_text())
    result = lens.fit(views, K, dist)
    groups = sorted({v["label"].split(" yaw")[0] for v in views})
    held_out = []
    for group in groups:
        rest = [v for v in views if not v["label"].startswith(group + " yaw")]
        left = [v for v in views if v["label"].startswith(group + " yaw")]
        part = lens.fit(rest, K, dist)
        err = lens.residuals(np.array(part["mount_m"] + part["turn_rad"] + part["board_shift"]), left, K, dist)
        held_out.append(float(np.sqrt(np.mean(err.reshape(-1, 2) ** 2) * 2)))
    result.update(held_out_px=[round(v, 2) for v in held_out], groups=groups, views_file=path.name,
                  K=K, dist=dist, when=time.strftime("%Y-%m-%d %H:%M"))
    mount, turn, shift, sd = result["mount_m"], result["turn_rad"], result["board_shift"], result["sd"]
    print(f"{result['views']} views, {sum(len(v['pixels']) for v in views)} corners: "
          f"rms {result['rms_px']:.2f} px, held out {np.mean(held_out):.2f} px "
          f"(worst pose {max(held_out):.2f})")
    mm, mm_sd = [1000 * v for v in mount], [1000 * v for v in sd[:3]]
    print(f"lens on the forearm, mm: along the tool {mm[0]:.1f} +- {mm_sd[0]:.1f}, "
          f"across {mm[1]:.1f} +- {mm_sd[1]:.1f}, sideways {mm[2]:+.1f} +- {mm_sd[2]:.1f}")
    print("  (cfg._LENS: {:.0f}, {:.0f}, {:+.0f})".format(*(1000 * v for v in lens.NOMINAL_MOUNT)))
    print("turned from the tool axis, deg: {:+.2f} {:+.2f} {:+.2f}".format(*(math.degrees(v) for v in turn)))
    print(f"board moved: {1000 * shift[0]:+.1f} mm forward, {1000 * shift[1]:+.1f} mm left, turned {shift[2]:+.2f} deg")
    out = DATA / "lens_on_arm.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"saved {out}")
    return 0


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
    intr.add_argument("--add", action="store_true", help="capture more views next to the saved ones")
    eye = sub.add_parser("handeye", help="the lens on the arm, from board corners seen from many poses")
    eye.add_argument("--views", type=Path, default=DATA / "handeye_views_read.json",
                     help="a JSON list of {pose, pixels, table}: the corners seen and where they lie")
    eye.add_argument("--capture", action="store_true", help="take the views first: THE ARM MOVES")
    eye.add_argument("--panel", default=PANEL, help="the robot's panel (it owns the arm and camera)")
    eye.add_argument("--min-lift-mm", type=float, default=0,
                     help="leave out targets with the fingertips lower than this over the table")
    args = parser.parse_args()
    if args.what == "handeye":
        if args.capture and capture_eye(args.panel, args.views, args.min_lift_mm / 1000):
            return 1
        return handeye(args.views)

    board = a4_board() if args.board == "a4" else None
    folder = ROOT / "out" / "camera_model" / "intrinsics"
    if not args.refit:
        for old in () if args.add else folder.glob("view_*.png"):
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
    pinned = result["fx_sd_px"] / K[0][0]
    steepest = max(result["tilts_deg"])
    print(f"focal length +- {result['fx_sd_px']:.1f} px ({100 * pinned:.1f} %), steepest view {steepest} deg: "
          + ("good enough" if pinned <= FX_SD_GOOD else
             f"NOT pinned -- tilt the board more (at least {TILT_GOOD_DEG} deg) and into the corners"))
    if result["before"]:
        print(f"before: fx {result['before'][0][0]}  (cfg.WRIST_CAM_LENS)")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
