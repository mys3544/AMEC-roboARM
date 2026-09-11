#!/usr/bin/env python3
"""Show every step of the droop-compensation loop, and find where a joint saturates.

    docker compose run --rm core python tools/droop_probe.py

Two questions this answers:

  * Is the correction loop working?  Each pass prints what it commanded and what
    the joint actually did, so under-correction is visible directly.
  * Or has the servo simply run out of torque?  If commanding further and further
    past the target buys less and less actual movement, the joint is saturated and
    no amount of control will fix it -- that region is outside the usable workspace.
"""

import sys
import time

from roboarm import config as cfg
from roboarm.arm import Arm, ArmError

NEUTRAL = {1: 90, 2: 90, 3: 90, 4: 90, 5: 90, 6: 120}
JOINT = 2
GOAL = 110


def settle(arm: Arm, joint: int) -> int:
    time.sleep(1.2)
    return arm.read()[joint]


def main() -> int:
    try:
        with Arm() as arm:
            arm.move_to(NEUTRAL, speed_dps=25)
            time.sleep(1.0)

            print(f"--- compensation loop, J{JOINT} -> {GOAL} ---")
            commanded = GOAL
            arm.move_to({JOINT: GOAL}, speed_dps=20, verify=False)
            actual = settle(arm, JOINT)
            print(f"  commanded {commanded:>3}  actual {actual:>3}  droop {commanded - actual:+d}")

            for step in range(1, 5):
                droop = commanded - actual
                lo, hi = cfg.SAFE_LIMITS[JOINT]
                commanded = int(min(max(GOAL + droop, lo), hi))
                arm.move_to({JOINT: commanded}, speed_dps=20, verify=False)
                actual = settle(arm, JOINT)
                print(
                    f"  pass {step}: commanded {commanded:>3}  actual {actual:>3}  "
                    f"droop {commanded - actual:+d}  error vs goal {actual - GOAL:+d}"
                )

            print(f"\n--- open-loop saturation sweep, J{JOINT} ---")
            print("  if actual stops rising as commanded rises, the servo is saturated")
            for command in (110, 120, 130, 140, 150):
                lo, hi = cfg.SAFE_LIMITS[JOINT]
                if not (lo <= command <= hi):
                    continue
                arm.move_to({JOINT: command}, speed_dps=20, verify=False)
                got = settle(arm, JOINT)
                print(f"  commanded {command:>3}  ->  actual {got:>3}   (droop {command - got:+d})")

            arm.move_to(NEUTRAL, speed_dps=25, verify=False)
            print("\nback to neutral")
    except ArmError as exc:
        print(f"\nARM ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
