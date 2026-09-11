#!/usr/bin/env python3
"""Measure where the wrist camera actually sits on the forearm.

    docker compose run --rm core python tools/camera_offset.py

ANSWERED 2026-09-10: 125 mm closed, 96 mm open, so the lens is 65 mm out from
J4 -- config had that swapped with the 125 mm. Kept because it is the only direct
measurement of the mount, and re-running it is how to check a knock has not moved it.

The reasons it was run:

  * the magnification of a tag of known size puts the camera about 220 mm above the
    table at the survey pose, where 65 mm predicts only 181 mm;
  * a reach for a tagged cube missed sideways by 20 mm, which is what correcting
    parallax about the wrong lens position would do.

The arm parks UPRIGHT so a ruler can reach both the lens and the fingertips, and
holds at each gripper end. Measure along the forearm axis, from the LENS to the
FINGERTIPS, both times.

The gap between the two readings is a prediction, not a question: the fingers pivot,
so closing them extends the tool by a measured 28 mm (cfg.GRIPPER_TOOL_MM), which
must make the camera-to-fingertip distance 28 mm SHORTER when closed. If your two
numbers differ by about that, the tool-length model is confirmed from a second,
independent direction. If they do not, one of the two models is wrong.
"""

import sys
import time

from roboarm import config as cfg
from roboarm.arm import Arm, ArmError

UPRIGHT = {1: 90, 2: 90, 3: 90, 4: 90, 5: 90}
HOLD = 45.0


def main() -> int:
    stages = [
        (cfg.GRIPPER_CLOSED, "CLOSED"),
        (cfg.GRIPPER_OPEN, "OPEN"),
    ]
    try:
        with Arm() as arm:
            print("parking upright -- only the gripper moves after this.\n")
            arm.move_to({**UPRIGHT, cfg.GRIPPER_ID: cfg.GRIPPER_CLOSED},
                        speed_dps=20, verify=False)
            time.sleep(1.0)

            for angle, name in stages:
                arm.set_gripper(angle, speed_dps=60, verify=False)
                time.sleep(1.0)
                tool = cfg.tool_length(angle) * 1000
                print(f"--- gripper {name} (J6={angle}, finger gap "
                      f"{cfg.gripper_gap(angle) * 1000:.0f} mm) ---")
                print(f"    J4 -> fingertip is {tool:.0f} mm here, so a camera "
                      f"{cfg.CAMERA_FROM_J4 * 1000:.0f} mm out from J4")
                print(f"    would sit {tool - cfg.CAMERA_FROM_J4 * 1000:.0f} mm "
                      f"back from the fingertips.")
                print(f"    MEASURE lens -> fingertips now. Holding {HOLD:.0f}s.",
                      flush=True)
                time.sleep(HOLD)

            arm.set_gripper(cfg.GRIPPER_OPEN, speed_dps=60, verify=False)
    except ArmError as exc:
        print(f"\nARM ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nReport both distances, and two more things while you are there:")
    print("  * how far the lens sits OFF the forearm centre line (config says 50 mm)")
    print("  * WHICH SIDE it is on, seen from behind the robot looking the way the")
    print("    arm points -- left or right. That sign is what decides which way the")
    print("    parallax correction pulls, and it is the one thing no photo has")
    print("    settled yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
