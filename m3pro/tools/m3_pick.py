"""The M3's look-again pick, run from this laptop through the bridge: roboarm/m3pick.py does the
work (the same code the bridge runs itself for the console: POST /pick/start), here with the arm
over /arm/call and the pictures over /rgbd.

    python tools/m3_pick.py look                          # the wide look only: what it finds
    python tools/m3_pick.py pick [--at X,Y] [--kind K]    # the nearest in reach (or nearest X,Y), then home
    python tools/m3_pick.py clean-up [--kind K]           # every kit thing in reach, one by one
    python tools/m3_pick.py drop                          # take what is held to the drop-off

K is cube or block. Pictures, picks.json and metrics.jsonl go to out/pick/. Through the bridge
(tools/m3bridge.py); on the robot itself the console's M3 arm card does the same faster.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import m3_handeye as he

from roboarm import m3pick, metrics

OUT = ROOT / "out" / "pick"


def call(method: str, *args, **kwargs):
    jsonable = [{str(k): v for k, v in a.items()} if isinstance(a, dict) else a for a in args]
    reply = he.http("/arm/call", {"method": method, "args": jsonable, "kwargs": kwargs})
    if "error" in reply:
        raise RuntimeError(f"{method}: {reply['error']}")
    return reply.get("result")


def snapshot():
    snap = he.http("/rgbd")
    if "error" in snap:
        raise RuntimeError(snap["error"])
    image = cv2.imdecode(np.frombuffer(base64.b64decode(snap["color_jpeg"]), np.uint8), cv2.IMREAD_COLOR)
    depth = cv2.imdecode(np.frombuffer(base64.b64decode(snap["depth_png"]), np.uint8), cv2.IMREAD_UNCHANGED)
    pose = {int(j): float(v) for j, v in snap["joints"].items()}
    return image, depth, np.asarray(snap["K"], float), pose


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("look")
    p = sub.add_parser("pick")
    p.add_argument("--at", help="X,Y in base_link (m): the object nearest this")
    p.add_argument("--kind", choices=("cube", "block"), help="only this kind")
    c = sub.add_parser("clean-up")
    c.add_argument("--kind", choices=("cube", "block"), help="only this kind")
    sub.add_parser("drop")
    args = parser.parse_args()
    metrics.REGISTRY.log_path = OUT / "metrics.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    picker = m3pick.Picker(call, snapshot, OUT)
    if args.cmd == "look":
        picker.engaged()
        picker.move(m3pick.LOOK, speed=25)
        _camera, found = picker.look("look")
        print(m3pick.describe(found))
        return 0
    if args.cmd == "pick":
        at = None if args.at is None else [float(v) for v in args.at.split(",")]
        picker.start("pick_once", kind=args.kind, at=at)
    elif args.cmd == "clean-up":
        picker.start("clean_up", kind=args.kind)
    else:
        picker.start("drop")
    try:
        picker.wait()
    except KeyboardInterrupt:
        picker.stop()
        he.http("/arm/interrupt", {})
        picker.wait()
    runs = metrics.snapshot(events=False)["runs"]
    if runs:
        print("time:", json.dumps({k: v["total_s"] for k, v in runs[0]["top"].items()}), f"total {runs[0]['total_s']} s")
    return 1 if picker.state["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
