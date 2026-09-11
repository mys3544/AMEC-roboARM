#!/usr/bin/env python3
"""Measure how much SHORTER the tool gets as the gripper opens.

    docker compose run --rm core python tools/tool_length.py

The fingers pivot, so the fingertips are not a fixed distance from J5: opening the
gripper pulls them back along the tool axis. The kinematics uses one fixed length
(L_J5_FINGERTIP, calibrated at J6=150), which is why every check so far -- the 2 mm
table verify, the 4 mm height check -- was run with the gripper closed and looked fine.

Grasping cannot do that. It has to approach OPEN, and a fit of the arm geometry
suggested the tool is ~36 mm shorter there. If that is right, a commanded 5 mm above
the table would put the fingertips 40 mm up and the gripper would close on nothing.
36 mm is too big to leave inferred, so: measure it directly.

Method. The arm holds ONE pose with the tool pointing straight DOWN (pitch 180), so
the tool axis is vertical and fingertip height moves one-for-one with tool length.
Only J6 changes between readings. Whatever the arm's absolute error is, it is the
same at every step, so the DIFFERENCES are clean even if the absolute is not.
"""

import sys
import time

from roboarm import config as cfg
from roboarm import kinematics as kin
from roboarm.arm import Arm, ArmError

FORWARD_M = 0.150   # straight down is only reachable between ~140 and ~160 mm
COMMANDED_MM = 20.0
HOLD = 35.0

# The same grid GRIPPER_GAP_MM was measured on, so the two tables line up. Three
# points is enough: if the middle one sits on the line between the outer two, the
# relationship is linear and two constants describe it.
GRIPPER_ANGLES = (30, 90, 150)


def main() -> int:
    z = -cfg.TABLE_BELOW_PLATE + COMMANDED_MM / 1000
    try:
        pose = kin.inverse(FORWARD_M, 0.0, z, pitch_deg=180.0)
    except kin.Unreachable as exc:
        print(f"measurement pose unreachable: {exc}", file=sys.stderr)
        return 1

    print(f"pose J1={pose[1]} J2={pose[2]} J3={pose[3]} J4={pose[4]}, tool straight down")
    print(f"model puts the fingertips {COMMANDED_MM:.0f} mm above the table, using its")
    print(f"one fixed tool length of {(cfg.L_J4_J5 + cfg.L_J5_FINGERTIP) * 1000:.0f} mm "
          f"(J4 -> tip).\n")
    print("Measure the tip of ONE finger each time -- they splay apart as they open.\n")

    try:
        with Arm() as arm:
            for index, angle in enumerate(GRIPPER_ANGLES, start=1):
                arm.move_to({**pose, 6: angle}, speed_dps=15, verify=False)
                time.sleep(1.5)
                gap = cfg.GRIPPER_GAP_MM.get(angle, "?")
                print(f"[{index}/{len(GRIPPER_ANGLES)}] J6={angle}  (finger gap {gap} mm)")
                print(f"      MEASURE fingertip height above the table -- holding "
                      f"{HOLD:.0f}s", flush=True)
                time.sleep(HOLD)

            arm.move_to({1: 90, 2: 90, 3: 90, 4: 90, 5: 90, 6: cfg.GRIPPER_OPEN},
                        speed_dps=20, verify=False)
    except ArmError as exc:
        print(f"\nARM ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nparked. Report the three heights in order.")
    print("A HIGHER fingertip means a SHORTER tool, by the same amount.")
    print("If they are all equal, the tool length does not move and the model is fine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
