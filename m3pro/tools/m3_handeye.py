"""Where the M3's camera sits on its arm, measured: the hand-eye calibration, with the joints' own
errors. The lab's ChArUco board (roboarm/config.py BOARD_*) lies flat in front of the robot; the
arm, through the bridge (tools/m3bridge.py), looks at it from ~24 poses; the fit finds together

  * the camera on arm4 (the link it is fixed to), 6 numbers,
  * the board in base_link, 6 numbers,
  * each joint's servo, real angle = 90 + scale * (command - 90) + offset: the scales of J1-J4 and
    the offsets of J2 and J3 -- 6 numbers,

by making the board's corners, carried through the arm's geometry (roboarm/m3kinematics.py),
land where the camera saw them. J1's offset cannot be told from the board's own turn, nor J4's from
the camera's mounting: those two are folded into the board and the camera. A pure rigid fit
(the URDF's joints) left 13 px (~12 mm) on 2026-09-30: J1 turned ~0.875 deg a commanded degree,
and the camera's height erred linearly with J2-J4 -- servo scales, not offsets alone. The scales
and offsets are held near 1 and 0 by a mild prior, so that a direction the pictures cannot tell
does not wander off.

    python tools/m3_handeye.py capture [--board-x 0.56]   # the arm moves; -> out/handeye/STAMP/
    python tools/m3_handeye.py solve out/handeye/STAMP     # -> data/m3_handeye.json

Runs on this laptop (the project venv: OpenCV with ArUco); only the bridge's HTTP crosses to the
robot. The robot and the board must not move between capture's first and last picture.
"""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roboarm import config as cfg
from roboarm import m3arm, m3kinematics

BRIDGE = "http://192.168.73.66:8761"
LINK_TO_OPTICAL = m3kinematics.OPTICAL   # optical x right, y down, z ahead
REAL_SAG_DEG = 4.0   # what camcheck found: the arm sits this much lower than the URDF (to predict views)
OUT = ROOT / "data" / "m3_handeye.json"


# --------------------------------------------------------------------------------- geometry --
JOINTS0 = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])   # offsets J2, J3 (deg); scales J1-J4


def real_angles(pose: dict[int, float], joints=JOINTS0) -> dict[int, float]:
    """The servo commands turned into the angles the joints really reach (see the docstring)."""
    o2, o3, *scales = joints
    q = dict(pose)
    for j, scale in zip((1, 2, 3, 4), scales):
        q[j] = 90 + scale * (pose[j] - 90)
    q[2] += o2
    q[3] += o3
    return q


def arm4(pose: dict[int, float], joints=JOINTS0) -> np.ndarray:
    """base_link -> arm4 for servo degrees, through the joints' measured scales and offsets."""
    return m3kinematics.frames(real_angles(pose, joints))["arm4"]


def urdf_camera() -> np.ndarray:
    """arm4 -> the camera's optical frame, as the URDF has it: the starting guess."""
    return m3kinematics.camera_optical()


def rt(params: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = cv2.Rodrigues(np.asarray(params[:3], float))[0]
    m[:3, 3] = params[3:6]
    return m


def params_of(m: np.ndarray) -> np.ndarray:
    return np.concatenate([cv2.Rodrigues(m[:3, :3])[0].ravel(), m[:3, 3]])


# ------------------------------------------------------------------------------------ board --
def board():
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.ARUCO_DICT))
    return cv2.aruco.CharucoBoard(cfg.BOARD_SQUARES, cfg.BOARD_SQUARE_M, cfg.BOARD_MARKER_M, dictionary)


def corners(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The board's chessboard corners seen: (pixels N x 2, board points N x 3, metres)."""
    b = board()
    found, ids, _m, _mi = cv2.aruco.CharucoDetector(b).detectBoard(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if ids is None or len(ids) == 0:
        return np.empty((0, 2)), np.empty((0, 3))
    return found.reshape(-1, 2), b.getChessboardCorners()[ids.ravel()]


# ---------------------------------------------------------------------------------- capture --
def http(path: str, body: dict | None = None):
    request = urllib.request.Request(BRIDGE + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as reply:
            return json.load(reply)
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def plan_poses(board_x: float, count: int = 24) -> list[tuple[int, ...]]:
    """Poses the guard allows from which the board's middle is well inside the picture (the
    URDF lowered by REAL_SAG_DEG), as different from each other as possible, in driving order."""
    K = np.array([[480.8, 0, 320], [0, 480.8, 240], [0, 0, 1]])
    target = np.array([board_x, 0.0, m3kinematics.FLOOR, 1.0])
    ok = []
    for j1, j2, j3, j4 in itertools.product(range(76, 105, 4), range(100, 146, 5), range(0, 46, 5), (0, 10, 20)):
        pose = {1: j1, 2: j2, 3: j3, 4: j4, 5: 90}
        if m3kinematics.problem(pose):
            continue
        cam = arm4({**pose, 2: j2 - REAL_SAG_DEG}) @ urdf_camera()
        p = np.linalg.inv(cam) @ target
        if p[2] < 0.2:
            continue
        u, v = (K @ (p[:3] / p[2]))[:2]
        if 110 < u < 530 and 90 < v < 390:
            ok.append((j1, j2, j3, j4))
    if not ok:
        raise SystemExit("no pose sees the board there -- is --board-x right?")
    scale = np.array([1.0, 1.0, 1.0, 0.5])   # degrees; J4 counts half
    chosen = [min(ok, key=lambda p: abs(p[0] - 90) + abs(p[1] - 120) + abs(p[2] - 30) + p[3])]
    while len(chosen) < min(count, len(ok)):   # farthest point first: spread the poses out
        far = max(ok, key=lambda p: min(np.abs((np.array(p) - np.array(c)) * scale).sum() for c in chosen))
        chosen.append(far)
    order, left = [chosen[0]], chosen[1:]
    while left:
        nxt = min(left, key=lambda p: np.abs(np.array(p) - np.array(order[-1])).max())
        left.remove(nxt)
        order.append(nxt)
    return order


def capture(board_x: float) -> Path:
    folder = ROOT / "out" / "handeye" / time.strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True)
    poses = plan_poses(board_x)
    print(f"{len(poses)} poses; the arm moves", flush=True)
    shots = []
    for k, (j1, j2, j3, j4) in enumerate(poses):
        reply = http("/arm/call", {"method": "move_to",
                                   "args": [{"1": j1, "2": j2, "3": j3, "4": j4, "5": 90}],
                                   "kwargs": {"speed_dps": 30}})
        if "error" in reply:
            print(f"pose {k}: {reply['error']}", flush=True)
            continue
        time.sleep(1.5)   # the arm stops swaying; the depth trails the colour by ~0.17 s
        snap = http("/rgbd")
        if "error" in snap:
            print(f"pose {k}: {snap['error']}", flush=True)
            continue
        image = cv2.imdecode(np.frombuffer(base64.b64decode(snap["color_jpeg"]), np.uint8), cv2.IMREAD_COLOR)
        seen, _points = corners(image)
        name = f"{k:02d}.jpg"
        cv2.imwrite(str(folder / name), image)
        shots.append({"image": name, "joints": {int(j): v for j, v in snap["joints"].items()},
                      "corners": len(seen)})
        print(f"pose {k}: {snap['joints']} -> {len(seen)} corners", flush=True)
    http("/arm/call", {"method": "move_to", "args": [dict(zip("123456", m3arm.POSES["survey"]))],
                       "kwargs": {"speed_dps": 30}})
    (folder / "shots.json").write_text(json.dumps({"K": snap["K"], "board_x": board_x, "shots": shots}, indent=1))
    print(f"-> {folder}")
    return folder


# ------------------------------------------------------------------------------------ solve --
PRIOR = (5.0, 0.1, 3.0)   # an offset of 5 deg, or a scale 0.1 off 1, costs as much as 3 px


def residuals(x: np.ndarray, views, K: np.ndarray) -> np.ndarray:
    """Pixels between where each seen corner is and where the model puts it; then the prior."""
    X, board_pose, joints = rt(x[:6]), rt(x[6:12]), x[12:18]
    out = []
    for pose, pixels, points in views:
        cam_board = np.linalg.inv(arm4(pose, joints) @ X) @ board_pose
        p = (cam_board[:3, :3] @ points.T).T + cam_board[:3, 3]
        uv = (K @ (p / p[:, 2:3]).T).T[:, :2]
        out.append((uv - pixels).ravel())
    deg, scale, px = PRIOR
    out.append(px * np.concatenate([joints[:2] / deg, (joints[2:] - 1) / scale]))
    return np.concatenate(out)


def least_squares(f, x: np.ndarray, iterations: int = 60) -> np.ndarray:
    """Levenberg-Marquardt with a numerical Jacobian: small problems, no scipy needed."""
    lam, r = 1e-3, f(x)
    for _ in range(iterations):
        J = np.empty((len(r), len(x)))
        for i in range(len(x)):
            h = 1e-6 * max(1.0, abs(x[i]))
            step = np.zeros_like(x)
            step[i] = h
            J[:, i] = (f(x + step) - r) / h
        A, g = J.T @ J, J.T @ r
        while True:
            dx = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            r_new = f(x + dx)
            if r_new @ r_new < r @ r:
                x, r, lam = x + dx, r_new, lam / 3
                break
            lam *= 4
            if lam > 1e8:
                return x
        if np.abs(dx).max() < 1e-9:
            break
    return x


def fit(views, K: np.ndarray, x0: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    if x0 is None:   # the URDF's camera and joints, the board where the first picture puts it;
        X0 = urdf_camera()   # then the camera and board alone first, the joints after
        pose0, pixels0, points0 = views[0]
        _ok, rvec, tvec = cv2.solvePnP(points0, pixels0, K, None)
        cam_board = rt(np.concatenate([rvec.ravel(), tvec.ravel()]))
        rigid = np.concatenate([params_of(X0), params_of(arm4(pose0) @ X0 @ cam_board)])
        rigid = least_squares(lambda v: residuals(np.concatenate([v, JOINTS0]), views, K), rigid)
        x0 = np.concatenate([rigid, JOINTS0])
    x = least_squares(lambda v: residuals(v, views, K), x0)
    return x, rms_px(x, views, K)


def rms_px(x: np.ndarray, views, K: np.ndarray) -> float:
    r = residuals(x, views, K)[:-6].reshape(-1, 2)   # the corners only, not the prior
    return float(np.sqrt((r ** 2).sum(axis=1).mean()))


def solve(folder: Path) -> dict:
    data = json.loads((folder / "shots.json").read_text())
    K = np.asarray(data["K"], float)
    views = []
    for shot in data["shots"]:
        pixels, points = corners(cv2.imread(str(folder / shot["image"])))
        if len(pixels) >= 8:
            views.append(({int(j): float(v) for j, v in shot["joints"].items()}, pixels, points))
    if len(views) < 6:
        raise SystemExit(f"only {len(views)} pictures with the board: too few")
    x, rms = fit(views, K)
    held_out = []   # each picture predicted by the fit to the others
    for i in range(len(views)):
        xi, _ = fit(views[:i] + views[i + 1:], K, x)   # from the full fit: it only moves a little
        held_out.append(rms_px(xi, [views[i]], K))
    X, board_pose = rt(x[:6]), rt(x[6:12])
    urdf = urdf_camera()
    turn = math.degrees(np.linalg.norm(cv2.Rodrigues(urdf[:3, :3].T @ X[:3, :3])[0]))
    result = {
        "pictures": len(views), "corners": int(sum(len(v[1]) for v in views)),
        "rms_px": round(rms, 3), "held_out_rms_px": round(float(np.median(held_out)), 3),
        "held_out_worst_px": round(float(max(held_out)), 3),
        "joints": {"offset_deg": {"2": round(float(x[12]), 3), "3": round(float(x[13]), 3)},
                   "scale": {str(j): round(float(x[13 + j]), 4) for j in (1, 2, 3, 4)}},
        "camera_in_arm4": X.round(6).tolist(),
        "camera_vs_urdf": {"moved_mm": round(float(np.linalg.norm(X[:3, 3] - urdf[:3, 3])) * 1000, 1),
                           "turned_deg": round(turn, 2)},
        "board_in_base": board_pose.round(6).tolist(),
        "K": K.tolist(), "source": folder.resolve().relative_to(ROOT).as_posix(),
        "measured": time.strftime("%Y-%m-%d %H:%M"),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--board-x", type=float, default=0.56, help="the board's middle, m ahead of base_link")
    sol = sub.add_parser("solve")
    sol.add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.cmd == "capture":
        capture(args.board_x)
    else:
        result = solve(args.folder)
        print(json.dumps({k: v for k, v in result.items() if k not in ("camera_in_arm4", "board_in_base", "K")},
                         indent=1))
        print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
