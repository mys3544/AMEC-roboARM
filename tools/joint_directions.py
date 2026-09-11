#!/usr/bin/env python3
"""Establish which way each joint physically turns as its servo angle changes.

    docker compose run --rm core python tools/joint_directions.py

The link lengths are measured, but the SIGN of each joint -- whether a rising servo
angle tips the arm one way or the other -- is recorded nowhere and cannot be read
off the hardware. So we measure it: move one joint at a time and have a human say
which way it went.

Only J1-J4 are tested. J5 is wrist roll, which changes tool orientation but not
fingertip position, so its sign does not affect position kinematics.

The swing direction is chosen per joint to stay inside SAFE_LIMITS -- J2 in
particular may only swing forwards, since backwards is where the mast is.
"""

import sys
import time

from roboarm import config as cfg
from roboarm.arm import Arm, ArmError

NEUTRAL = {1: 90, 2: 90, 3: 90, 4: 90, 5: 90, 6: 120}
SWING = 30

# Asked once the arm has moved, phrased for the direction actually commanded.
QUESTIONS = {
    1: "base yaw    -- did the whole arm swing to its LEFT or its RIGHT?",
    2: "shoulder    -- did the upper arm tip FORWARD (away from the mast) or BACK?",
    3: "elbow       -- did the forearm fold UP or DOWN?",
    4: "wrist pitch -- did the gripper tip UP or DOWN?",
}


def pick_target(joint: int) -> int | None:
    """Swing whichever way stays legal, preferring the positive direction."""
    lo, hi = cfg.SAFE_LIMITS[joint]
    for target in (NEUTRAL[joint] + SWING, NEUTRAL[joint] - SWING):
        if lo <= target <= hi:
            return target
    return None


def main() -> int:
    moved: dict[int, tuple[int, int]] = {}
    try:
        with Arm() as arm:
            print("moving to the neutral pose (all joints 90, gripper part-open)...")
            arm.move_to(NEUTRAL, speed_dps=25)
            time.sleep(1.0)
            print("neutral reached. Watch the arm.\n")

            for joint in (1, 2, 3, 4):
                target = pick_target(joint)
                if target is None:
                    print(f"J{joint}: SKIPPED, no legal {SWING} degree swing")
                    continue

                print(f">>> J{joint}: {NEUTRAL[joint]} -> {target} in 3 seconds. WATCH.", flush=True)
                time.sleep(3)
                arm.move_to({joint: target}, speed_dps=20)
                moved[joint] = (NEUTRAL[joint], target)
                print(f"    J{joint} is at {target}. Holding 4 seconds.", flush=True)
                time.sleep(4)
                arm.move_to({joint: NEUTRAL[joint]}, speed_dps=20)
                print(f"    J{joint} back to {NEUTRAL[joint]}.\n", flush=True)
                time.sleep(1.5)

            print("done -- arm left at the neutral pose.\n")
            print("Report what you saw:")
            for joint, (start, target) in moved.items():
                arrow = "increasing" if target > start else "decreasing"
                print(f"  J{joint} ({start} -> {target}, {arrow})  {QUESTIONS[joint]}")
    except ArmError as exc:
        print(f"\nARM ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
