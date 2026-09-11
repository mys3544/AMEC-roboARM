#!/usr/bin/env python3
"""Is the object really where the camera says? Ask the arm, by touching it.

    ... touch_probe.py                   # probe the coloured object it can see
    ... touch_probe.py --height-mm 45    # if the object stands taller than 28 mm
    ... touch_probe.py --at 155,24       # probe a point you name instead

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


def find(arm: Arm, look: sweep.Look, height_m: float) -> tuple[float, float]:
    arm.move_to(look.pose, speed_dps=20, verify=False)
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
                speed_dps=18, verify=False)
    arm.move_to({j: a for j, a in down.items() if j != cfg.GRIPPER_ID},
                speed_dps=6, verify=False)
    time.sleep(1.0)
    reached = kin.forward({**arm.read(), cfg.GRIPPER_ID: closed})[2] - table
    arm.move_to({j: a for j, a in above.items() if j != cfg.GRIPPER_ID},
                speed_dps=18, verify=False)
    return reached


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--at", default=None,
                        help="probe 'x,y' in mm instead of what the camera finds")
    parser.add_argument("--height-mm", type=float, default=28.0,
                        help="how tall the object is, for undoing its parallax")
    parser.add_argument("--offsets-mm", default="-20,-10,0,10,20",
                        help="radial offsets to probe, comma separated")
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
            arm.set_gripper(cfg.GRIPPER_CLOSED, speed_dps=60, verify=False)
            time.sleep(1.0)
            closed = arm.read()[cfg.GRIPPER_ID]
            print(f"gripper shut to J6={closed}, tool "
                  f"{cfg.tool_length(closed) * 1000:.0f} mm\n")
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

            arm.move_to(survey, speed_dps=20, verify=False)
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
