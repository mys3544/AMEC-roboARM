#!/usr/bin/env python3
"""Is the object really where the camera says? Ask the arm, by touching it.

    ... touch_probe.py                   # probe the coloured object it can see
    ... touch_probe.py --height-mm 45    # if the object stands taller than 28 mm
    ... touch_probe.py --at 155,24       # probe a point you name instead
    ... touch_probe.py --staircase --at 230,0 --offsets-mm 0,10,20
                                         # where is the table really, out at the rim?

WHY THIS EXISTS. When a grasp closes on nothing there are exactly two suspects, and
arguing geometry against geometry cannot separate them: either the camera is
reporting the wrong place, or the arm is not going where it is sent. Every other
check in this project compares one model against another. This one compares a model
against the world.

It drives the CLOSED gripper straight down onto the nominated point and reads what
stops it:

    stopped at about the object's height  -> the object is there; the camera is right
    reached the table                     -> nothing under the fingertip

and repeats along the RADIUS, either side, because that is the direction a reach
error lies in -- the tool is nearly vertical, so an error in the arm's reach moves
the fingertip along the radius and barely at all across it.

Pressing straight down on a flat top is the gentlest probe available: there is no
sideways component to slide the object, and the descent stops the moment the joints
stop tracking.

USED IN ANGER 2026-09-10. Two picks had closed on nothing and the silhouette was
suspected. The probe stopped dead at 27.6 mm on the camera's nominated point and
found bare table 10 mm either side -- so the camera was exonerated, and the search
moved to the arm, where tools/reach_check.py found 9.9 mm of open-loop reach error.
"""

import argparse
import math
import sys
import time

from roboarm import camera, detect, sweep
from roboarm import config as cfg
from roboarm import kinematics as kin
from roboarm import workspace as ws
from roboarm.arm import Arm, ArmError

PROBE_Z_M = 0.012          # well below any object's top, so a miss is unmistakable
CLEARANCE_M = 0.070        # height to travel between probes
STOPPED_MARGIN_MM = 8.0    # above the commanded height counts as blocked

# --staircase: step down until the joints stop following, for where the model's
# height cannot be trusted to 8 mm. 2026-09-23: the user set the arm by hand with
# its fingers on the paper at the far rim and the model put them 18 mm UNDER it.
START_M = 0.040            # first step, model height
STEP_M = 0.003
LOWEST_M = -0.035          # how far under its own table the model may be sent looking
CONTACT_LAG_M = 0.008      # the readback trailing this much further behind = touching


def find(arm: Arm, look: sweep.Look, height_m: float) -> tuple[float, float]:
    arm.move_to(look.pose, speed_dps=20)
    time.sleep(2.0)
    frame = camera.grab(8)[-1]
    found = detect.coloured(frame, look.matrix, nadir=look.nadir, height_m=height_m)
    if not found:
        raise ValueError("no coloured object in view -- put one on the table, "
                         "or name a point with --at")
    target = found[0]
    print(f"camera says {target.x * 1000:.1f} mm forward, {target.y * 1000:+.1f} mm "
          f"left, outline {target.width_m * 1000:.0f} x {target.length_m * 1000:.0f} mm")
    return target.x, target.y


def probe(arm: Arm, x: float, y: float, closed: int) -> float | None:
    """Descend at (x, y) and report the height the fingertip actually reached."""
    table = -cfg.TABLE_BELOW_PLATE
    try:
        above, pitch = kin.solve(x, y, table + CLEARANCE_M, gripper=closed)
        down, _p = kin.solve(x, y, table + PROBE_Z_M, pitches=(pitch,), gripper=closed)
    except kin.Unreachable as exc:
        print(f"    unreachable: {exc}")
        return None
    arm.move_to({j: a for j, a in above.items() if j != cfg.GRIPPER_ID},
                speed_dps=18)
    arm.move_to({j: a for j, a in down.items() if j != cfg.GRIPPER_ID},
                speed_dps=6)
    time.sleep(1.0)
    reached = kin.forward({**arm.read(), cfg.GRIPPER_ID: closed})[2] - table
    arm.move_to({j: a for j, a in above.items() if j != cfg.GRIPPER_ID},
                speed_dps=18)
    return reached


def staircase(arm: Arm, x: float, y: float, closed: int) -> float | None:
    """Step the closed tips down at (x, y) until the joints stop following, and
    return the MODEL's height of the tips at that moment -- 0 if the model is
    right about this point, -18 mm if it is as wrong as the hand-set rim pose.

    In free air the readback trails the command by a few mm (servo shortfall,
    1-degree readback steps) and drifts slowly; once the tips are on something
    it trails by one more step every step and stops falling. Contact is both:
    the lag CONTACT_LAG_M beyond its smallest so far, and the last step's fall
    under half a step. None if nothing stopped them above LOWEST_M.
    """
    table = -cfg.TABLE_BELOW_PLATE
    lowest = table + LOWEST_M
    try:
        above, pitch = kin.solve(x, y, table + CLEARANCE_M, gripper=closed)
    except kin.Unreachable as exc:
        print(f"    unreachable: {exc}")
        return None
    arm_only = {j: a for j, a in above.items() if j != cfg.GRIPPER_ID}
    arm.move_to(arm_only, speed_dps=18)
    nearest = tuple(sorted(kin.GRASP_PITCHES, key=lambda p: abs(p - pitch)))
    least_lag = previous = None
    z = table + START_M
    try:
        while z >= lowest:
            pose, _p = kin.solve(x, y, z, pitches=nearest, gripper=closed, lowest_z=lowest)
            got = kin.forward({**arm.move_to({j: pose[j] for j in (1, 2, 3, 4)},
                                             speed_dps=6), cfg.GRIPPER_ID: closed})
            lag = got[2] - z
            least_lag = lag if least_lag is None else min(least_lag, lag)
            stalled = previous is not None and previous - got[2] < STEP_M / 2
            if lag - least_lag > CONTACT_LAG_M and stalled:
                print(f"    contact: asked {(z - table) * 1000:5.1f}, joints say "
                      f"{(got[2] - table) * 1000:5.1f} mm at "
                      f"{math.hypot(got[0], got[1]) * 1000:.0f} mm out (J2={pose[2]})")
                return got[2] - table
            previous = got[2]
            z -= STEP_M
        print(f"    nothing stopped the tips down to {LOWEST_M * 1000:.0f} mm (model)")
        return None
    except kin.Unreachable as exc:
        print(f"    ran out of reach at {(z - table) * 1000:.0f} mm (model): {exc}")
        return None
    finally:
        arm.move_to(arm_only, speed_dps=18)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--at", default=None,
                        help="probe 'x,y' in mm instead of what the camera finds")
    parser.add_argument("--height-mm", type=float, default=28.0,
                        help="how tall the object is, for undoing its parallax")
    parser.add_argument("--offsets-mm", default="-20,-10,0,10,20",
                        help="radial offsets to probe, comma separated")
    parser.add_argument("--staircase", action="store_true",
                        help="step down until contact and report the model's height "
                             "there, instead of one descent to a fixed height")
    args = parser.parse_args()
    offsets = [float(v) for v in args.offsets_mm.split(",")]

    try:
        matrix, survey = ws.load()
        look = sweep.Look(0.0, survey, matrix)
        with Arm() as arm:
            if args.at:
                x, y = (float(v) / 1000 for v in args.at.split(","))
                print(f"probing the point given: {x * 1000:.1f}, {y * 1000:+.1f} mm")
            else:
                x, y = find(arm, look, args.height_mm / 1000)

            radius, bearing = math.hypot(x, y), math.degrees(math.atan2(y, x))
            arm.set_gripper(cfg.GRIPPER_CLOSED, speed_dps=60)
            time.sleep(1.0)
            closed = arm.read()[cfg.GRIPPER_ID]
            print(f"gripper shut to J6={closed}, tool "
                  f"{cfg.tool_length(closed) * 1000:.0f} mm\n")
            if args.staircase:
                touched = []
                for offset in offsets:
                    r = radius + offset / 1000
                    px = r * math.cos(math.radians(bearing))
                    py = r * math.sin(math.radians(bearing))
                    print(f"{offset:+6.0f}   {px * 1000:6.1f},{py * 1000:+6.1f} mm")
                    height = staircase(arm, px, py, closed)
                    if height is not None:
                        touched.append((r, height))
                arm.move_to(survey, speed_dps=20)
                print("\nwhere the tips stopped, by the model (0 = the model is right; "
                      "negative = the real tips sit that much HIGHER than it says):")
                for r, height in touched:
                    print(f"    {r * 1000:5.0f} mm out: {height * 1000:+6.1f} mm")
                return 0
            print("offset    point (mm)       asked   reached   verdict")

            hits = []
            for offset in offsets:
                r = radius + offset / 1000
                px = r * math.cos(math.radians(bearing))
                py = r * math.sin(math.radians(bearing))
                reached = probe(arm, px, py, closed)
                if reached is None:
                    continue
                blocked = reached * 1000 > PROBE_Z_M * 1000 + STOPPED_MARGIN_MM
                print(f"{offset:+6.0f}   {px * 1000:6.1f},{py * 1000:+6.1f}    "
                      f"{PROBE_Z_M * 1000:5.0f}   {reached * 1000:6.1f}   "
                      f"{'STOPPED, something is there' if blocked else 'reached the table'}")
                if blocked:
                    hits.append((offset, reached * 1000))

            arm.move_to(survey, speed_dps=20)
            print()
            if not hits:
                print("Nothing stopped the fingertip anywhere along that radius.")
                print("The camera and the arm disagree about where this object is.")
            else:
                middle = sum(o for o, _h in hits) / len(hits)
                print(f"Stopped at {[f'{o:+.0f}' for o, _h in hits]} mm along the "
                      f"radius, tops around {max(h for _o, h in hits):.1f} mm.")
                if abs(middle) <= 5:
                    print("Centred on the nominated point: the camera is right, and a")
                    print("grasp that still misses is the arm -- see reach_check.py.")
                else:
                    print(f"Centred {middle:+.0f} mm off the nominated point, so that "
                          f"is the offset to explain.")
    except (ArmError, camera.CameraError, ValueError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
