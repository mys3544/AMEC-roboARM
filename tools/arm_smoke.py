#!/usr/bin/env python3
"""Exercise arm.py against the real hardware, in stages of increasing reach.

    ... arm_smoke.py                 # gripper only -- cannot collide with anything
    ... arm_smoke.py --joints        # small base/wrist moves; needs clear space
    ... arm_smoke.py --home          # go to the home pose; needs clear space
    ... arm_smoke.py --cycles 10     # repeatability: 10 round trips, measured

Stops at the first failure and leaves the arm holding, never limp.
"""

import argparse
import sys
import time

from roboarm import config as cfg
from roboarm.arm import Arm, ArmError


def show(label: str, pose: dict[int, int]) -> None:
    print(f"  {label:<22}" + "  ".join(f"J{j}={pose[j]:>3}" for j in cfg.JOINT_IDS))


def gripper_stage(arm: Arm) -> None:
    print("\n[gripper] open / close, verified by readback")
    show("start", arm.read())
    show("opened", arm.open_gripper())
    time.sleep(0.3)
    arm.close_gripper()
    show("closed", arm.read())
    print(f"  grasped()               {arm.grasped()}  (expected False on empty fingers)")
    show("re-opened", arm.open_gripper())


def joints_stage(arm: Arm) -> None:
    """Base yaw and wrist roll only. Both sweep a small arc and return."""
    print("\n[joints] small base + wrist moves, returning to start each time")
    start = arm.read()
    for joint, delta in ((1, 15), (5, 15)):
        target = start[joint] + delta
        lo, hi = cfg.SAFE_LIMITS[joint]
        if not (lo <= target <= hi):
            print(f"  J{joint}: skipped, {target} would leave the safe range")
            continue
        show(f"J{joint} +{delta}", arm.move_to({joint: target}))
        show(f"J{joint} back", arm.move_to({joint: start[joint]}))


def home_stage(arm: Arm) -> None:
    print("\n[home] move to the home pose and back")
    start = arm.read()
    show("home", arm.home())
    show("back to start", arm.move_to(start))


def cycles_stage(arm: Arm, cycles: int) -> None:
    """Repeatability: return to the same pose N times and measure the spread.

    This is what bounds grasp accuracy -- a degree of scatter at the shoulder is
    several millimetres at the fingertip.
    """
    print(f"\n[cycles] {cycles} home/start round trips, measuring repeatability")
    start = arm.read()
    at_home: list[dict[int, int]] = []
    at_start: list[dict[int, int]] = []

    for i in range(1, cycles + 1):
        at_home.append(arm.home())
        at_start.append(arm.move_to(start))
        print(f"  cycle {i}/{cycles} ok", flush=True)

    for label, samples in (("home", at_home), ("start", at_start)):
        spread = {j: max(s[j] for s in samples) - min(s[j] for s in samples)
                  for j in cfg.JOINT_IDS}
        worst = max(spread.values())
        print(f"  spread at {label:<6}" + "  ".join(f"J{j}={spread[j]}" for j in cfg.JOINT_IDS)
              + f"   worst {worst} deg")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joints", action="store_true", help="also sweep base and wrist")
    parser.add_argument("--home", action="store_true", help="also move to the home pose")
    parser.add_argument("--cycles", type=int, default=0,
                        help="repeat home/start N times and report repeatability")
    args = parser.parse_args()

    try:
        with Arm() as arm:
            print(f"connected. battery {arm.bot.get_battery_voltage():.1f} V")
            gripper_stage(arm)
            if args.joints:
                joints_stage(arm)
            if args.home:
                home_stage(arm)
            if args.cycles:
                cycles_stage(arm, args.cycles)
            print("\nall stages passed; arm left holding")
    except ArmError as exc:
        print(f"\nARM ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
