#!/usr/bin/env python3
"""Does the fingertip actually arrive where the grasp asked it to? Measure it.

    ... reach_check.py                   # at a default empty patch of table
    ... reach_check.py --bearing 20 --radius 190

Run over an EMPTY patch: the arm descends to grasp height three times and nothing
is touched either way.

WHY THIS TOOL EXISTS. On 2026-09-10 the gripper closed on nothing, twice, on a
cube the camera had located to within 2 mm (tools/parallax_probe.py proves that
half). The arm was the half at fault, and in a way that no existing check could
have caught: every joint landed within ONE degree of its commanded angle, which
arm.py rightly calls converged -- and the fingertip was still 9.9 mm short.

The reason is the tool pitch. A top-down grasp at 176 mm needs the tool about 10
degrees off vertical, and there the forearm contributes reach as l3*sin(pitch), so

    d(reach)/d(pitch) = l3 * cos(pitch) = 185 mm * cos(170 deg) = -3.2 mm/degree

Two joints a degree low is most of a centimetre at the fingertip. Correcting in
JOINT space cannot help -- arm.py's droop loop (since deleted) recovered 3.2 mm of
the 9.9 and then stopped, because 1 degree is inside any honest deadband, and
chasing a degree walks a joint about and changes the load on its neighbours.
The fix has to be applied where the error is measured in millimetres, which is
grasp._reach_to(), and this tool is how that claim was established and how it
should be re-established after any change to the arm or its calibration.

MEASURED 2026-09-10 at 176 mm, bearing -40, tool pitch 170, gripper open to 42 mm:

    open loop      9.9 mm short,  fingertip 3.0 mm up when 8 mm was asked
    joint-space    6.7 mm short,  fingertip 3.5 mm up   (the old droop loop)
    cartesian      3.1 mm short,  fingertip 8.8 mm up

The 3.1 mm that remains is the arm's resolution, not slack in the method: the IK
rounds to whole servo degrees and one degree here is 3.2 mm.
"""

import argparse
import math
import sys
import time

from roboarm import config as cfg
from roboarm import grasp
from roboarm import kinematics as kin
from roboarm import workspace as ws
from roboarm.arm import Arm, ArmError


def landed(arm: Arm, want: tuple[float, float, float], pose: dict[int, int],
           opening: int, label: str) -> float:
    actual = arm.read()
    x, y, z = kin.forward({**actual, cfg.GRIPPER_ID: opening})
    miss = math.hypot(x - want[0], y - want[1])
    drift = {j: actual[j] - pose[j] for j in (1, 2, 3, 4) if actual[j] != pose[j]}
    print(f"\n{label}")
    print(f"  joints off by {drift or 'nothing'}")
    print(f"  fingertip {x * 1000:6.1f}, {y * 1000:+6.1f}, "
          f"{(z + cfg.TABLE_BELOW_PLATE) * 1000:5.1f} mm above the table")
    print(f"  horizontal miss {miss * 1000:5.1f} mm "
          f"(reached {math.hypot(x, y) * 1000:.1f} of {math.hypot(*want[:2]) * 1000:.0f} mm)")
    return miss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bearing", type=float, default=-40.0,
                        help="degrees left of straight ahead; pick EMPTY table")
    parser.add_argument("--radius", type=float, default=176.0, help="mm from the base")
    parser.add_argument("--object-mm", type=float, default=27.0,
                        help="the object the gripper would be opened for")
    args = parser.parse_args()

    x = args.radius / 1000 * math.cos(math.radians(args.bearing))
    y = args.radius / 1000 * math.sin(math.radians(args.bearing))
    z = -cfg.TABLE_BELOW_PLATE + grasp.GRASP_HEIGHT_M
    opening = cfg.gripper_for_gap(args.object_mm / 1000 + grasp.FINGER_CLEARANCE_M)

    try:
        _matrix, survey = ws.load()
        pose, pitch = kin.solve(x, y, z, gripper=opening)
        print(f"target {x * 1000:.1f} mm forward, {y * 1000:+.1f} mm left, "
              f"{grasp.GRASP_HEIGHT_M * 1000:.0f} mm up")
        print(f"pitch {pitch:.0f} deg, gripper J6={opening} "
              f"({cfg.GRIPPER_GAP_MM[opening]} mm), tool {cfg.tool_length(opening) * 1000:.0f} mm")
        print(f"one degree of pitch here is "
              f"{abs(cfg.tool_length(opening) * math.cos(math.radians(pitch))) * 1000 * math.pi / 180:.1f} mm of reach")

        arm_only = {j: a for j, a in pose.items() if j != cfg.GRIPPER_ID}
        with Arm() as arm:
            arm.move_to(survey, speed_dps=20, verify=False)
            arm.set_gripper(opening, speed_dps=60, verify=False)
            time.sleep(1.0)

            hover, _p = kin.solve(x, y, -cfg.TABLE_BELOW_PLATE + grasp.HOVER_M,
                                  pitches=(pitch,), gripper=opening)
            arm.move_to({j: a for j, a in hover.items() if j != cfg.GRIPPER_ID},
                        speed_dps=15, verify=False)

            arm.move_to(arm_only, speed_dps=grasp.DESCEND_DPS)
            time.sleep(0.8)
            loose = landed(arm, (x, y, z), pose, opening, "OPEN LOOP")

            grasp._reach_to(arm, x, y, z, pitch, opening, grasp.DESCEND_DPS)
            time.sleep(0.5)
            cartesian = landed(arm, (x, y, z), pose, opening,
                               "CARTESIAN LOOP (what grasp.pick does)")

            print(f"\ncartesian correction recovered {(loose - cartesian) * 1000:+5.1f} mm")
            arm.move_to(survey, speed_dps=20, verify=False)
    except (ArmError, ValueError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
