#!/usr/bin/env python3
"""Step through four deliberately dissimilar arm shapes for geometry calibration.

    docker compose run --rm core python tools/fit_poses.py

Every earlier measurement was taken at a similar arm shape, so the least-squares fit
could not tell J3 and J4 apart -- it returned offsets of +17.6 and -17.9 degrees that
simply cancelled. These four poses spread the link angles as widely as the workspace
and the mast clearance allow, which is what makes the joints separable.

The gripper is held at 150 throughout, ON PURPOSE: that is the opening at which
J5->fingertip was physically measured (110 mm), and the tool length changes with
opening, so letting it vary would put an unknown back into the fit.

Measure the MIDPOINT between the two fingertips. At J6=150 they sit 25 mm apart, so
the midpoint is about 12 mm inboard of either tip.
"""

import sys
import time

from roboarm import config as cfg
from roboarm import kinematics as kin
from roboarm.arm import Arm, ArmError

GRIPPER = 150
PARKED = {1: 90, 2: 90, 3: 90, 4: 90, 5: 90, 6: GRIPPER}
HOLD = 60.0

# Chosen by searching the safe workspace for the four poses whose link angles are
# furthest apart -- see the module docstring.
POSES = [(20, 20, 100), (90, 50, 25), (20, 100, 20), (25, 30, 160)]


def main() -> int:
    try:
        with Arm() as arm:
            arm.move_to(PARKED, speed_dps=25, verify=False)
            time.sleep(1.0)

            for index, (j2, j3, j4) in enumerate(POSES, start=1):
                wanted = {**PARKED, 2: j2, 3: j3, 4: j4}
                if cfg.mast_clearance(wanted) < cfg.MIN_MAST_CLEARANCE_M:
                    print(f"[{index}] SKIPPED {j2},{j3},{j4} -- too close to the mast")
                    continue

                actual = arm.move_to(wanted, speed_dps=15, verify=False)
                x, _y, z = kin.forward(actual)
                print(
                    f"\n[{index}/{len(POSES)}] reached J2={actual[2]} J3={actual[3]} "
                    f"J4={actual[4]}  (asked {j2},{j3},{j4})"
                )
                print(
                    f"      model: {x * 1000:.0f} mm forward, "
                    f"{(z + cfg.TABLE_BELOW_PLATE) * 1000:.0f} mm above the table"
                )
                print(f"      MEASURE NOW -- holding {HOLD:.0f}s", flush=True)
                time.sleep(HOLD)

            arm.move_to(PARKED, speed_dps=20, verify=False)
            print("\nparked. Report four pairs of numbers, in order:")
            print("  forward from the rotation centre, and height above the table,")
            print("  measured to the MIDPOINT between the fingertips.")
    except ArmError as exc:
        print(f"\nARM ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
