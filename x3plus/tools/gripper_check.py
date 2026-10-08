"""Where a piece of known width really stops the X3's fingers, against what cfg.GRIPPER_GAP_MM says.

    uv run python tools/gripper_check.py open [--at 100]      # fingers to --at: put the piece between them
    uv run python tools/gripper_check.py close --mm 30        # closes 2 deg at a time, stops at contact

Over the X3 bridge (:8761, /arm/call), no job running. The close reads the gripper's raw count after
every step (read_fine) and stops the moment the piece holds it back -- it lags the command by
LAG_DEG twice running -- then leaves the fingers where they stopped (no squeeze) and says which
angle that was against grasp.TOUCH_DEG and the tip-gap table. 2026-10-08: the fingers "still over closing" at the
table's 28 mm for a 30 mm cube.
"""

import argparse
import json
import sys
import urllib.request

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from roboarm import config as cfg
from roboarm import grasp

BRIDGE = "http://192.168.73.210:8761"
LAG_DEG = 2.5


def call(method, *args, **kwargs):
    body = json.dumps({"method": method, "args": list(args), "kwargs": kwargs}).encode()
    request = urllib.request.Request(BRIDGE + "/arm/call", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as reply:
        answer = json.loads(reply.read())
    if "error" in answer:
        raise SystemExit(f"bridge: {answer['error']}")
    return answer.get("result", answer)


def gripper_now() -> float:
    return float(call("read_fine", [cfg.GRIPPER_ID])[str(cfg.GRIPPER_ID)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("open", "close"))
    parser.add_argument("--at", type=int, default=100, help="open: the angle to open to (100 ~ 52 mm)")
    parser.add_argument("--mm", type=float, default=30.0, help="close: the piece's width")
    parser.add_argument("--start", type=int, default=110)
    parser.add_argument("--stop", type=int, default=cfg.GRIPPER_CLOSED)
    parser.add_argument("--step", type=int, default=2)
    args = parser.parse_args()
    if args.what == "open":
        call("set_gripper", args.at, speed_dps=60)
        print(f"fingers at {args.at} (table: {cfg.gripper_gap(args.at) * 1000:.0f} mm) -- read {gripper_now():.1f}")
        return 0
    table = round(grasp.touch_angle(args.mm / 1000))
    print(f"a {args.mm:.0f} mm piece: grasp.TOUCH_DEG says first touch at J6 {table}; closing {args.start}..{args.stop} "
          f"by {args.step}")
    lagging = 0
    for angle in range(args.start, args.stop + 1, args.step):
        call("set_gripper", angle, speed_dps=30)
        now = gripper_now()
        lag = angle - now
        print(f"  J6 {angle:3d}  read {now:6.1f}  lag {lag:+5.1f}")
        lagging = lagging + 1 if lag >= LAG_DEG else 0
        if lagging >= 2:
            call("set_gripper", round(now), speed_dps=30)
            print(f"CONTACT: the {args.mm:.0f} mm piece stops the fingers at J6 {now:.1f} "
                  f"(grasp.TOUCH_DEG: first touch at {table}; the tip-gap table says "
                  f"{cfg.gripper_gap(round(now)) * 1000:.1f} mm there). Fingers left there, no squeeze.")
            return 0
    print("no contact before the stop: nothing between the fingers, or the piece is narrower than that")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
