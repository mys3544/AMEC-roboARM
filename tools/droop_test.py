#!/usr/bin/env python3
"""Measure how far the gravity-loaded joints fall short of their commanded angle.

    docker compose run --rm core python tools/droop_test.py

J2 (shoulder) and J3 (elbow) carry the arm's weight, and a servo holding a load
settles wherever its torque balances gravity rather than exactly on target. This
distinguishes two very different causes of the same symptom:

  * still settling  -> the error shrinks over a couple of seconds. Fix: wait longer.
  * genuine droop   -> the error is stable and load-dependent. Fix: model it, or
                       accept it as a floor on positioning accuracy.

Six degrees at the shoulder is roughly 26 mm at the fingertip, so which one this
is decides whether the 10 mm positioning target is reachable at all.
"""

import sys
import time

from roboarm.arm import Arm, ArmError

NEUTRAL = {1: 90, 2: 90, 3: 90, 4: 90, 5: 90, 6: 120}
SETTLE_SAMPLES = 12       # 12 x 0.4s = ~5s of watching after the move stops
SAMPLE_PERIOD = 0.4


def probe(arm: Arm, joint: int, target: int) -> None:
    """Command one joint twice: open-loop, then with the correction loop on."""
    # Open loop: verify=False also skips the correction, so this is the raw droop.
    arm.move_to({joint: target}, speed_dps=20, verify=False)
    readings = []
    for _ in range(SETTLE_SAMPLES):
        readings.append(arm.read()[joint])
        time.sleep(SAMPLE_PERIOD)
    raw = readings[-1]
    settled = "settling" if abs(raw - readings[0]) >= 2 else "stable"

    # Closed loop: re-command with the observed error added on.
    try:
        arm.move_to({joint: target}, speed_dps=20)
        corrected, verdict = arm.read()[joint], "ok"
    except ArmError:
        corrected, verdict = arm.read()[joint], "STILL SHORT"

    print(
        f"  J{joint} -> {target:>3}   open-loop {raw:>3} ({raw - target:+d}, {settled})"
        f"   closed-loop {corrected:>3} ({corrected - target:+d})   {verdict}"
    )


def main() -> int:
    try:
        with Arm() as arm:
            print("to neutral...")
            arm.move_to(NEUTRAL, speed_dps=25)
            time.sleep(1.0)

            for joint in (2, 3):
                print(f"\n--- J{joint} ---")
                for target in (75, 90, 105):
                    probe(arm, joint, target)
                arm.move_to({joint: NEUTRAL[joint]}, speed_dps=20, verify=False)
                time.sleep(1.0)

            print("\nback to neutral")
            arm.move_to(NEUTRAL, speed_dps=25, verify=False)
    except ArmError as exc:
        print(f"\nARM ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
