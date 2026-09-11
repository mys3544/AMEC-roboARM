#!/usr/bin/env python3
"""Find how far the arm can reach FORWARD before it hits the robot or the table.

    ... forward_sweep.py --to 60      # first stage, stays high
    ... forward_sweep.py --to 30      # second stage
    ... forward_sweep.py --to 15      # third, arm nearly horizontal

The mast is BEHIND the arm, so J2 above 90 is the direction that collides with it
and J2 below 90 is the working direction. This measures the working direction,
which is bounded by the robot's own body and the table -- and which has never been
measured. SAFE_LIMITS[2]'s lower bound is currently a guess.

J3 and J4 are held straight at 90 on purpose: fully extended is the worst case for
reaching far, so a limit found here is safe for every folded pose too.

Steps 10 degrees at a time and pauses so a human can watch. Aborts if the battery
sags, because a weak pack droops further than commanded -- which on this side of
vertical means the arm ends up LOWER than asked, towards the table.
"""

import argparse
import math
import sys
import time

from roboarm import config as cfg
from roboarm.arm import Arm, ArmError

START = {1: 90, 2: 90, 3: 90, 4: 90, 5: 90, 6: 120}
REACH_MM = 1000 * (cfg.L_J2_J3 + cfg.L_J3_J4 + cfg.L_J4_J5 + cfg.L_J5_FINGERTIP)
PLATE_TO_J2_MM = 1000 * cfg.BASE_PLATE_TO_J2
ABORT_VOLTS = 10.2


def predict(j2: int) -> tuple[float, float]:
    """Where a straight arm's fingertip should be: (forward mm, mm above the plate)."""
    angle = math.radians(90 - j2)
    return REACH_MM * math.sin(angle), REACH_MM * math.cos(angle) + PLATE_TO_J2_MM


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", type=int, required=True, help="lowest J2 to try")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--dwell", type=float, default=3.0, help="seconds to hold each step")
    args = parser.parse_args()

    lo = cfg.SAFE_LIMITS[2][0]
    if args.to < lo:
        print(f"--to {args.to} is below SAFE_LIMITS[2] lower bound {lo}", file=sys.stderr)
        return 1

    try:
        with Arm() as arm:
            print("to the start pose (arm straight up)...")
            arm.move_to(START, speed_dps=25)
            print(f"{'J2':>4} {'actual':>7} {'droop':>6} {'forward':>8} {'above plate':>12} {'batt':>6}")

            j2 = 90
            while j2 > args.to:
                j2 = max(args.to, j2 - args.step)
                volts = arm.bot.get_battery_voltage()
                if volts < ABORT_VOLTS:
                    print(f"\nABORT: battery {volts:.1f} V is below {ABORT_VOLTS} V")
                    break

                arm.move_to({2: j2}, speed_dps=15, verify=False)
                time.sleep(args.dwell)
                actual = arm.read()[2]
                forward, above = predict(actual)
                print(
                    f"{j2:>4} {actual:>7} {actual - j2:>+6} {forward:>7.0f}mm {above:>11.0f}mm "
                    f"{arm.bot.get_battery_voltage():>5.1f}V",
                    flush=True,
                )

            print("\nreturning to the start pose")
            arm.move_to(START, speed_dps=20, verify=False)
            print("done -- arm upright and holding")
    except ArmError as exc:
        print(f"\nARM ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
