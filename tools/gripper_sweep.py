#!/usr/bin/env python3
"""Map the gripper servo angle to the actual finger gap, and to tool length.

    docker compose run --rm core python tools/gripper_sweep.py

Two things we got wrong by assuming, both fixed by measuring:

  * WHICH END IS OPEN. GRIPPER_OPEN was set to 150 and GRIPPER_CLOSED to 30, but at
    J6=150 the fingers measure 25 mm apart, and fitting the arm geometry gives a
    tool length of 191 mm there against 155 mm at J6=30 -- a shorter tool means
    wider splay, so 30 is the OPEN end. grasped() is inverted as a result.

  * TOOL LENGTH VARIES WITH OPENING. J5 -> fingertip is 110 mm at J6=150. Splayed
    open the tips move off-axis and the effective length shrinks, which is why
    validation runs disagreed depending on gripper state.

The arm is parked upright throughout -- only the gripper moves, so this is safe to
run with the arm anywhere clear.
"""

import sys
import time

from roboarm import config as cfg
from roboarm.arm import Arm, ArmError

PARKED_ARM = {1: 90, 2: 90, 3: 90, 4: 90, 5: 90}
STEPS = (30, 60, 90, 120, 150, 180)
DWELL = 12.0


def main() -> int:
    lo, hi = cfg.HARD_LIMITS[cfg.GRIPPER_ID]
    try:
        with Arm() as arm:
            print("parking the arm upright (only the gripper will move after this)...")
            arm.move_to({**PARKED_ARM, cfg.GRIPPER_ID: STEPS[0]}, speed_dps=25, verify=False)
            time.sleep(1.0)

            print(f"\nstepping J6 through {STEPS}, holding {DWELL:.0f}s each.")
            print("measure the GAP BETWEEN THE FINGERTIPS at every stop.\n")
            for angle in STEPS:
                if not (lo <= angle <= hi):
                    print(f"  J6={angle:>3}  skipped, outside {lo}..{hi}")
                    continue
                arm.move_to({cfg.GRIPPER_ID: angle}, speed_dps=30, verify=False)
                time.sleep(1.0)
                actual = arm.read()[cfg.GRIPPER_ID]
                print(f"  >>> J6 commanded {angle:>3}, reading {actual:>3}  "
                      f"-- measure the gap now ({DWELL:.0f}s)", flush=True)
                time.sleep(DWELL)

            arm.move_to({cfg.GRIPPER_ID: 150}, speed_dps=30, verify=False)
            print("\ndone -- gripper left at 150, arm upright.")
            print("Report the gap at each step; that fixes both the polarity and the")
            print("tool length, and tells us the widest object the gripper can take.")
    except ArmError as exc:
        print(f"\nARM ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
