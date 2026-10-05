"""The M3's touch test: how far the open-loop chain -- camera, arm geometry, IK -- puts the fingertip
from a point the camera saw. The lab's ChArUco board lies flat in front of the robot; the arm looks
at it (the look pose), finds its corners in base_link through roboarm/m3kinematics.py and the URDF's
camera mounting -- the camera's height and tilt taken from the floor the depth sees (m3vision) -- then holds the grip point a few cm above a corner while someone measures the miss.

    python tools/m3_touch.py look            # the arm moves to the look pose; -> out/touch/look.json
    python tools/m3_touch.py go N [--up 0.05]   # the grip point to --up m above corner N, and holds
    python tools/m3_touch.py home

The corners are numbered as `look` prints them, nearest first. Through the bridge (tools/m3bridge.py);
the robot and the board stay put between `look` and the last `go`.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import m3_handeye as he

from roboarm import m3arm, m3kinematics, m3vision

LOOK = {"1": 88, "2": 105, "3": 25, "4": 0, "5": 90}   # the board in the pick zone fills the view
STATE = ROOT / "out" / "touch" / "look.json"


def look() -> None:
    if "error" in he.http("/arm/call", {"method": "read"}):   # pose unknown since power-on: home first
        print("engage:", he.http("/arm/call", {"method": "engage"}).get("result"))
    reply = he.http("/arm/call", {"method": "move_to", "args": [LOOK], "kwargs": {"speed_dps": 25}})
    if "error" in reply:
        raise SystemExit(f"look pose refused: {reply['error']}")
    time.sleep(1.5)
    snap = he.http("/rgbd")
    K = np.asarray(snap["K"], float)
    image = cv2.imdecode(np.frombuffer(base64.b64decode(snap["color_jpeg"]), np.uint8), cv2.IMREAD_COLOR)
    pixels, points = he.corners(image)
    if len(pixels) < 8:
        raise SystemExit(f"only {len(pixels)} board corners seen -- is the board in view?")
    _ok, rvec, tvec = cv2.solvePnP(points, pixels, K, None)
    cam_board = he.rt(np.concatenate([rvec.ravel(), tvec.ravel()]))
    pose = {int(j): float(v) for j, v in snap["joints"].items()}
    model = m3kinematics.frames(pose)["arm4"] @ he.urdf_camera()
    depth = cv2.imdecode(np.frombuffer(base64.b64decode(snap["depth_png"]), np.uint8), cv2.IMREAD_UNCHANGED)
    normal, height, share = m3vision.floor_plane(depth, K)
    base_cam = m3vision.level(model, normal, height, m3kinematics.FLOOR)
    # the board lies on the same floor: its own plane, as a check on the depth's
    board_normal = cam_board[:3, 2] * np.sign(-cam_board[:3, 2] @ cam_board[:3, 3])
    board_height = float(abs(board_normal @ cam_board[:3, 3]))
    apart = np.degrees(np.arccos(np.clip(board_normal @ normal, -1, 1)))
    print(f"floor from depth: camera {height * 1000:.0f} mm above it ({share:.0%} of the points on it); "
          f"the board's plane: {board_height * 1000:.0f} mm, the two {apart:.2f} deg apart")
    allpoints = he.board().getChessboardCorners()
    in_base = (base_cam @ cam_board @ np.c_[allpoints, np.ones(len(allpoints))].T).T[:, :3]
    order = sorted(range(len(in_base)), key=lambda i: (round(in_base[i][0], 2), -in_base[i][1]))
    corners = [in_base[i].round(4).tolist() for i in order]
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"joints": snap["joints"], "corners": corners}, indent=1))
    cv2.imwrite(str(STATE.with_suffix(".jpg")), image)
    turned = np.degrees(np.arccos(np.clip((np.trace(model[:3, :3].T @ base_cam[:3, :3]) - 1) / 2, -1, 1)))
    print(f"{len(pixels)} corners seen; the table's plane put the camera {1000 * (base_cam[2, 3] - model[2, 3]):+.0f} mm"
          f" and {turned:.1f} deg off the arm model")
    for n, c in enumerate(corners):
        print(f"  corner {n:2d}: x {c[0]:.3f}  y {c[1]:+.3f}  z {c[2]:+.3f}")


def go(n: int, up: float) -> None:
    corners = json.loads(STATE.read_text())["corners"]
    corner = np.asarray(corners[n])
    target = corner + np.array([0.0, 0.0, up])
    for pitch in (75, 65, 55, 85, 45):
        pose = m3kinematics.reach(target, pitch)
        if pose:
            break
    else:
        raise SystemExit(f"corner {n} at {corner.round(3)}: no allowed pose puts the grip {up * 100:.0f} cm above it")
    reply = he.http("/arm/call", {"method": "move_to", "args": [{str(j): v for j, v in pose.items()}],
                                  "kwargs": {"speed_dps": 20}})
    if "error" in reply:
        raise SystemExit(f"refused: {reply['error']}")
    point, real_pitch = m3kinematics.grip(pose)
    print(f"corner {n} at x {corner[0]:.3f} y {corner[1]:+.3f}: the grip point should now be {up * 100:.0f} cm "
          f"straight above it, the fingers {real_pitch:.0f} deg below level (pose {pose}; model grip "
          f"{np.round(point, 3).tolist()})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("look")
    g = sub.add_parser("go")
    g.add_argument("n", type=int)
    g.add_argument("--up", type=float, default=0.05)
    sub.add_parser("home")
    args = parser.parse_args()
    if args.cmd == "look":
        look()
    elif args.cmd == "go":
        go(args.n, args.up)
    else:
        print(he.http("/arm/call", {"method": "move_to", "args": [dict(zip("123456", m3arm.POSES["home"]))],
                                    "kwargs": {"speed_dps": 25}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
