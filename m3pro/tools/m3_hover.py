"""The hover test: how far the pick's open-loop chain -- the wide look, the levelled camera, the arm
model with its sag, IK -- puts the fingers from a kit piece it sees. The arm takes the pick's wide
look, finds the nearest kit piece in it (m3vision.objects), and holds the grip point straight above
that piece's centre, the fingers squared with it and opened as for the pick, `--up` m over its top,
while someone measures the miss with a ruler: forward/back (away from the robot) and left/right.
(2026-10-09: grips came down short of the cube, the fingers closed on its near corners and turned it.)

    python tools/m3_hover.py look           # the wide look; -> out/hover/look.json + .jpg, pieces listed
    python tools/m3_hover.py go [N] [--up 0.02] [--fwd MM] [--left MM]   # over piece N (nearest 0), held
    python tools/m3_hover.py home

`--fwd` / `--left` shift the aim (mm, away from the robot / to its left) to try a correction by
hand before it goes into the model. Through the bridge (tools/m3bridge.py), no job running.
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

from roboarm import m3arm, m3kinematics, m3pick, m3vision

STATE = ROOT / "out" / "hover" / "look.json"


def call(method: str, *args, **kwargs):
    reply = he.http("/arm/call", {"method": method, "args": list(args), "kwargs": kwargs})
    if "error" in reply:
        raise SystemExit(f"{method} refused: {reply['error']}")
    return reply.get("result")


def look() -> None:
    if "error" in he.http("/arm/call", {"method": "read"}):   # pose unknown since power-on: home first
        call("engage")
    call("move_to", {str(j): v for j, v in m3pick.LOOK.items()}, speed_dps=30)
    time.sleep(1.0)
    snap = he.http("/rgbd")
    K = np.asarray(snap["K"], float)
    image = cv2.imdecode(np.frombuffer(base64.b64decode(snap["color_jpeg"]), np.uint8), cv2.IMREAD_COLOR)
    depth = cv2.imdecode(np.frombuffer(base64.b64decode(snap["depth_png"]), np.uint8), cv2.IMREAD_UNCHANGED)
    pose = {int(j): float(v) for j, v in snap["joints"].items()}
    model = m3kinematics.frames(pose)["arm4"] @ m3kinematics.camera_optical()
    normal, height, _ = m3vision.floor_plane(depth, K)
    camera = m3vision.level(model, normal, height, m3kinematics.FLOOR)
    found = [o for o in m3vision.objects(depth, K, camera, floor=(normal, height)) if o["kind"]]
    m3vision.refine_turns(image, K, camera, found)
    pieces = [{"centre": o["centre"].round(4).tolist(), "top": round(o["top"], 4), "yaw": round(o["yaw"], 1),
               "period": o["period"], "size": o["size"], "kind": o["kind"]} for o in found]
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"joints": snap["joints"], "pieces": pieces}, indent=1))
    cv2.imwrite(str(STATE.with_suffix(".jpg")), m3pick.mark(image, K, camera, found))
    for n, p in enumerate(pieces):
        c = p["centre"]
        print(f"  piece {n}: {p['kind']} {1000 * p['size']:.0f} mm at x {c[0]:.3f} y {c[1]:+.3f}, top {1000 * p['top']:.0f} mm,"
              f" turned {p['yaw']:.0f} deg")


def go(n: int, up: float, fwd: float, left: float) -> None:
    piece = json.loads(STATE.read_text())["pieces"][n]
    centre = np.asarray(piece["centre"], float)
    axis = np.asarray(m3kinematics.CHAIN[0][0][:2])
    out = (centre[:2] - axis) / np.linalg.norm(centre[:2] - axis)   # away from the arm's axis
    aim = centre[:2] + out * fwd / 1000 + np.array([-out[1], out[0]]) * left / 1000
    target = np.r_[aim, piece["top"] + up + m3kinematics.FINGERS]   # the fingertips `up` over the top
    opened = m3pick.opened_for(piece["size"], m3pick.OPEN_MM - 1000 * piece["size"])
    for pitch in (88, 85, 80, 75, 70, 65):
        pose = m3kinematics.reach(target, pitch, guard=False)
        if pose is None:
            continue
        j5 = m3pick.wrist_squares(pose, piece["yaw"], piece["period"])[0]
        pose = {**pose, 5: j5, 6: opened}
        if m3kinematics.problem(pose) is None:
            break
    else:
        raise SystemExit(f"piece {n} at {centre.round(3)}: no allowed pose over it")
    call("move_to", {str(j): v for j, v in pose.items()}, speed_dps=20)
    point, pitch = m3kinematics.grip(pose)
    print(f"piece {n} ({piece['kind']}) seen at x {centre[0]:.3f} y {centre[1]:+.3f}; aimed {fwd:+.0f} mm out,"
          f" {left:+.0f} mm left: the fingertips' middle should be {1000 * up:.0f} mm straight over its centre,"
          f" squared with it (pose {pose}, fingers {pitch:.0f} deg down, model grip {np.round(point, 3).tolist()})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("look")
    g = sub.add_parser("go")
    g.add_argument("n", type=int, nargs="?", default=0)
    g.add_argument("--up", type=float, default=0.02)
    g.add_argument("--fwd", type=float, default=0.0)
    g.add_argument("--left", type=float, default=0.0)
    sub.add_parser("home")
    args = parser.parse_args()
    if args.cmd == "look":
        look()
    elif args.cmd == "go":
        go(args.n, args.up, args.fwd, args.left)
    else:
        call("move_to", dict(zip("123456", m3arm.POSES["home"])), speed_dps=25)
    return 0


if __name__ == "__main__":
    sys.exit(main())
