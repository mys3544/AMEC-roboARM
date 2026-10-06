#!/usr/bin/env python3
"""Check the kinematics against reality: command a known point, measure where the
fingertip actually lands.

    ... ik_validate.py                    # list the points, solve them, move nothing
    ... ik_validate.py --point 3          # go to point 3 and hold while you measure
    ... ik_validate.py --point 3 --hold 60
    ... ik_validate.py --home             # return to the upright parked pose

Everything in the model traces back to measurements taken off a photo, and J4's
direction is still ASSUMED rather than observed, so the numbers being
self-consistent does not make them right. This is the test that decides.

HOW TO MEASURE
    Forward  is measured from the CENTRE OF THE ROTATING BASE PLATE (the J1 axis),
             horizontally, along the direction the arm is pointing.
    Height   is measured from the TABLE SURFACE up to the tip of the fingers.
    Gripper  is held CLOSED, so the fingertips sit on the tool axis and the
             "fingertip" is unambiguous.
    Targets all sit on the centre line (y = 0), so J1 stays at 90 and you only ever
    have to measure two numbers.

READING THE RESULT
    A consistent error in FORWARD distance points at the link lengths.
    A consistent error in HEIGHT that grows with reach points at a joint offset.
    An error that flips sign with pitch points at the assumed J4 direction.
"""

import argparse
import sys
import time

from roboarm import config as cfg
from roboarm import kinematics as kin
from roboarm.arm import Arm, ArmError

PARKED = {1: 90, 2: 90, 3: 90, 4: 90, 5: 90, 6: cfg.GRIPPER_OPEN}

# Across the usable strip, all on the centre line, 30 mm above the table -- roughly
# where the middle of a small object would sit.
LIFT = 0.030
REACHES_MM = (140, 160, 180, 200, 220, 240)


def plan() -> list[tuple[int, float, dict[int, int], float]]:
    """(index, reach_m, pose, pitch) for every target we can actually solve."""
    z = -cfg.TABLE_BELOW_PLATE + LIFT
    out = []
    for reach_mm in REACHES_MM:
        reach = reach_mm / 1000
        try:
            pose, pitch = kin.solve(reach, 0.0, z)
            # Validate with the gripper CLOSED. L_J5_FINGERTIP was measured closed,
            # and an open gripper splays its fingers off the axis -- at a top-down
            # pitch that lateral splay lands squarely in the forward direction we
            # are measuring, which is what put ~21 mm on the first two readings.
            pose[6] = cfg.GRIPPER_CLOSED
        except kin.Unreachable as exc:
            print(f"  r={reach_mm:>3} mm  UNREACHABLE: {exc}")
            continue
        out.append((len(out) + 1, reach, pose, pitch))
    return out


def describe(index: int, reach: float, pose: dict[int, int], pitch: float) -> None:
    x, y, z = kin.forward(pose)
    print(
        f"  [{index}] target {reach * 1000:>3.0f} mm forward, "
        f"{(z + cfg.TABLE_BELOW_PLATE) * 1000:>2.0f} mm above the table   "
        f"pitch {pitch:>3.0f}   "
        f"J1={pose[1]} J2={pose[2]} J3={pose[3]} J4={pose[4]}   "
        f"(model says {x * 1000:.0f}, {y * 1000:.0f} from the axis)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--point", type=int, help="which target to move to")
    parser.add_argument("--hold", type=float, default=45.0, help="seconds to hold it")
    parser.add_argument("--home", action="store_true", help="just park the arm upright")
    parser.add_argument("--pose", help="command J2,J3,J4 directly, e.g. --pose 40,60,110")
    parser.add_argument("--gripper", type=int, help="gripper servo angle to hold")
    args = parser.parse_args()

    if args.pose:
        # Fitting the geometry needs deliberately DIFFERENT arm shapes. Poses that
        # all come from solve() look alike, so the fit cannot separate J3 from J4.
        j2, j3, j4 = (int(v) for v in args.pose.split(","))
        wanted = {1: 90, 2: j2, 3: j3, 4: j4, 5: 90,
                  6: args.gripper if args.gripper else cfg.GRIPPER_OPEN}
        try:
            with Arm() as arm:
                arm.move_to(PARKED, speed_dps=25)
                actual = arm.move_to(wanted, speed_dps=15)
                x, _y, z = kin.forward(actual)
                print(f"reached J2={actual[2]} J3={actual[3]} J4={actual[4]} J6={actual[6]}")
                print(f"model: {x * 1000:.0f} mm forward, "
                      f"{(z + cfg.TABLE_BELOW_PLATE) * 1000:.0f} mm above the table")
                print(f"\nholding {args.hold:.0f}s -- measure forward and height.", flush=True)
                time.sleep(args.hold)
                arm.move_to(PARKED, speed_dps=20)
                print("parked.")
        except ArmError as exc:
            print(f"\nARM ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    targets = plan()
    if not args.point and not args.home:
        print(f"table at z = {-cfg.TABLE_BELOW_PLATE * 1000:.0f} mm, "
              f"targets {LIFT * 1000:.0f} mm above it\n")
        for entry in targets:
            describe(*entry)
        print("\nnothing moved. Re-run with --point N to go to one of these.")
        return 0

    try:
        with Arm() as arm:
            if args.home:
                arm.move_to(PARKED, speed_dps=20)
                print("parked:", arm.read())
                return 0

            match = [t for t in targets if t[0] == args.point]
            if not match:
                print(f"no target {args.point}; there are {len(targets)}", file=sys.stderr)
                return 1
            index, reach, pose, pitch = match[0]

            describe(index, reach, pose, pitch)
            print("\nmoving...", flush=True)
            arm.move_to(PARKED, speed_dps=25)
            actual = arm.move_to(pose, speed_dps=15)

            drift = {j: actual[j] - pose[j] for j in (1, 2, 3, 4) if actual[j] != pose[j]}
            print(f"reached: J1={actual[1]} J2={actual[2]} J3={actual[3]} J4={actual[4]}"
                  + (f"   (off by {drift})" if drift else "   (exactly as commanded)"))
            if drift:
                x, _y, z = kin.forward(actual)
                print(f"model for the pose ACTUALLY reached: {x * 1000:.0f} mm forward, "
                      f"{(z + cfg.TABLE_BELOW_PLATE) * 1000:.0f} mm above the table")

            print(f"\nholding {args.hold:.0f}s -- measure forward distance from the base "
                  f"plate centre, and fingertip height above the table.", flush=True)
            time.sleep(args.hold)
            arm.move_to(PARKED, speed_dps=20)
            print("parked.")
    except ArmError as exc:
        print(f"\nARM ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
