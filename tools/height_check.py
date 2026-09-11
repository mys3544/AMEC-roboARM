#!/usr/bin/env python3
"""Measure how far off the fingertip's HEIGHT is, across the workspace.

    docker compose run --rm core python tools/height_check.py

The table-plane calibration is validated to 2 mm, but height is not part of it --
a homography only describes the plane. Height comes from the kinematics, whose own
worst residual was 6.8 mm, and one spot check found the fingertip 10 mm up when
4 mm was asked for.

One measurement cannot tell a constant bias from a pose-dependent one, and guessing
wrong would be worse than not correcting at all. So: the same commanded height at
several places, and see whether the error moves.

Commands 20 mm rather than a few mm -- easier to measure against a ruler, and it
keeps the fingertips clear of the paper.
"""

import sys
import time

from roboarm import config as cfg
from roboarm import kinematics as kin
from roboarm.arm import Arm, ArmError

COMMANDED_MM = 20.0
HOLD = 40.0

# Spread across the reachable strip: three forward distances on the centre line,
# then two off to the sides, so a bias that varies with reach or with yaw shows up.
POINTS_MM = [(140, 0), (180, 0), (215, 0), (175, 55), (175, -55)]


def main() -> int:
    z = -cfg.TABLE_BELOW_PLATE + COMMANDED_MM / 1000
    plan = []
    for fwd, side in POINTS_MM:
        try:
            pose, pitch = kin.solve(fwd / 1000, side / 1000, z)
        except kin.Unreachable as exc:
            print(f"  ({fwd}, {side:+}) unreachable: {exc}")
            continue
        pose[6] = cfg.GRIPPER_CLOSED  # closed: fingertips on the axis, easy to sight
        plan.append((fwd, side, pose, pitch))

    if not plan:
        print("nothing reachable", file=sys.stderr)
        return 1
    print(f"{len(plan)} points, commanding {COMMANDED_MM:.0f} mm above the table each\n")

    try:
        with Arm() as arm:
            for index, (fwd, side, pose, pitch) in enumerate(plan, start=1):
                arm.move_to(pose, speed_dps=15, verify=False)
                time.sleep(1.2)
                tip = kin.forward(arm.read())
                print(
                    f"[{index}/{len(plan)}] target {fwd} mm forward, {side:+} mm left, "
                    f"pitch {pitch:.0f}"
                )
                print(
                    f"      model says the fingertip is at "
                    f"{(tip[2] + cfg.TABLE_BELOW_PLATE) * 1000:.0f} mm above the table"
                )
                print(f"      MEASURE the real height now -- holding {HOLD:.0f}s", flush=True)
                time.sleep(HOLD)

            arm.move_to({1: 90, 2: 90, 3: 90, 4: 90, 5: 90, 6: cfg.GRIPPER_OPEN},
                        speed_dps=20, verify=False)
            print("\nparked. Report the five measured heights in order.")
            print("All the same error -> one constant offset fixes it.")
            print("Error growing with reach -> it is pose-dependent and a constant")
            print("offset would be wrong; better to feel for the table instead.")
    except ArmError as exc:
        print(f"\nARM ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
