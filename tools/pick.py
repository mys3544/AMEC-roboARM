#!/usr/bin/env python3
"""The demo: look over the whole table, find an object, pick it up, put it down.

    ... detect_check.py --background     # once, on an EMPTY table (sweeps the ring)
    ... pick.py --dry-run                # find and plan, but do not touch anything
    ... pick.py                          # do it (change detection)
    ... pick.py --colour                 # find it by its colour: no tag, no background
    ... pick.py --markers                # use ArUco tags instead of change detection
    ... pick.py --yolo                   # use the neural detector (needs the vision container)
    ... pick.py --yolo --prompt "mug"    # open-vocabulary: pick the thing you name

The detector is a ladder. --yolo is the top rung; if the vision container is not
up it drops to --markers with a warning rather than failing, because a working
lower rung beats no pick at all. --markers, --colour and (default) change
detection never touch the network.

--colour is the rung to reach for when a sweep has to find something the other two
cannot. Change detection needs an empty-table photograph, and that is per-station,
so a seven-station ring needs seven of them and a clear table to take them on;
tags need the tag to be facing up, which stops being true the moment the object is
knocked over. Colour needs neither -- only that the object is coloured and the
table is not. It reports a slightly generous width, because the outline of a solid
object includes whatever side faces the lens can see, so pair it with --object-mm
when the real size is known.

HOW IT LOOKS. The wrist camera sees about 150 mm of table at a time and the arm
can reach a great deal more than that, so a single glance used to leave three
quarters of the workspace invisible. Instead the base now TURNS under a fixed
look -- same J2..J6, only J1 moving -- and photographs the table from a ring of
stations. Because J1 is a pure yaw about the origin the table coordinates are
measured from, each station's pixel-to-table mapping is the calibrated one turned
by the same angle, so the whole ring costs no extra calibration (roboarm/sweep.py
sets out the argument). The chosen object is then re-measured with the base turned
to put it in the MIDDLE of the picture, which is where the mapping is best
conditioned -- so an object at the far edge of reach is measured just as carefully
as one straight ahead.

Everything this needs was measured on the robot first, and the numbers are worth
knowing before running it:

    pick area        131..210 mm from the base, bearing -80..+80 deg (about 380 cm2)
    seen by a sweep  99.1% of that, at the default 25 degree step
    object size      22..55 mm across the narrow way, ideally under 60 mm tall
    accuracy         2 mm sideways (measured), up to 4 mm in height (measured)

The 0.9% the sweep misses is at 209..210 mm, the outermost millimetre of reach,
where the arm is straight out and a grasp is marginal anyway. Objects outside the
pick area are reported rather than attempted.

--dry-run is the right way to start. It runs the whole chain -- sweep, detect,
merge, reachability, gripper opening -- and prints what it WOULD do, without
moving the arm past the survey stations.
"""

import argparse
import math
import sys
import time

from roboarm import camera, detect, grasp, sweep
from roboarm import config as cfg
from roboarm import kinematics as kin
from roboarm import workspace as ws
from roboarm.arm import Arm, ArmError

# Where a picked object is put down. Unlike the old single-glance version, the
# sweep CAN see this spot -- with the whole workspace covered there is nowhere out
# of shot left to hide it. That is an improvement rather than a problem: run
# pick.py twice and the second sweep finds the cube where the first one left it,
# which is a free end-to-end check that the place actually worked.
DROP_X = 0.150
DROP_Y = -0.070

# How far from the sweep's estimate the refine look may find the object and still
# be believed to be the same one. Generous next to the few mm the two views should
# disagree by, tight enough that it cannot lock on to a different object: the
# nearest neighbour would have to be under 30 mm away, which is closer than two
# graspable objects can physically sit.
REFIND_M = 0.030

def detect_targets(frame, mode: str, matrix, prompt: str | None,
                   nadir: tuple[float, float], dyaw: float = 0.0,
                   object_mm: float | None = None, lens_m: float | None = None,
                   look_name: str = "primary") -> list[detect.Target]:
    """Run the chosen detector, dropping down the ladder if a rung is unavailable."""
    return detect.ladder(frame, mode, matrix, prompt=prompt, nadir=nadir, dyaw=dyaw,
                         object_mm=object_mm, lens_m=lens_m, look_name=look_name)


def look_once(arm: Arm, look: sweep.Look, mode: str, prompt: str | None,
              object_mm: float | None, settle: float = 1.2) -> list[detect.Target]:
    """Drive to one station and report what it can see, in table coordinates."""
    arm.move_to(look.pose, speed_dps=20, verify=False)
    time.sleep(settle)
    frame = camera.grab(6)[-1]
    found = detect_targets(frame, mode, look.matrix, prompt, look.nadir,
                           look.dyaw, object_mm, kin.camera_height(look.pose), look.name)
    # Anything touching the frame edge has an unknowable width -- we only measured
    # the part that was in shot -- and sits where the lens distorts most. Judged
    # against the object's OWN apparent size, parallax included, because a 40 mm
    # cube images 225 px across: its centre can be a long way inside the frame
    # while a third of it is off the edge. An adjacent station sees it properly,
    # so dropping it here costs nothing.
    # The tag rung only needs the printed square in shot -- its position comes
    # from that, and a declared --object-mm says nothing about what was measured.
    kept = [t for t in found if not t.clipped
            and sweep.sees(look, t.x, t.y,
                           sweep.TAG_SPAN_M if mode == "markers" and t.label.startswith("tag")
                           else t.width_m, t.height_m or sweep.OBJECT_HEIGHT_M)]
    tag = f"/out/sweep_{look.dyaw:+05.1f}.jpg".replace("+", "p").replace("-", "m")
    camera.write_image(tag, detect.annotate(frame, look.matrix, found))
    return kept


def survey(arm: Arm, looks: list[sweep.Look], mode: str, prompt: str | None,
           object_mm: float | None, stop_early: bool) -> list[detect.Target]:
    """Sweep the ring and return one merged list of everything on the table."""
    seen: list[tuple[sweep.Look, detect.Target]] = []
    print(f"sweeping {len(looks)} stations...")
    for look in looks:
        found = look_once(arm, look, mode, prompt, object_mm)
        print(f"  J1={look.pose[1]:3d} (dyaw {look.dyaw:+6.1f}): "
              f"{len(found)} object(s)")
        seen.extend((look, target) for target in found)
        if stop_early and any(_plannable(t) for t in found):
            print("  --first: stopping here, something graspable is in view.")
            break
    merged = sweep.merge(seen)
    print(f"\n{len(seen)} detection(s) merged into {len(merged)} object(s).")
    return merged


def _plannable(target: detect.Target) -> bool:
    try:
        grasp.plan(target)
    except grasp.GraspError:
        return False
    return True


def describe(targets: list[detect.Target]) -> None:
    if not targets:
        print("nothing on the table.")
        return
    print(f"{len(targets)} object(s):")
    for target in targets:
        bearing = math.degrees(math.atan2(target.y, target.x))
        radius = math.hypot(target.x, target.y)
        print(f"  {target.label:<12} {target.x * 1000:6.0f} mm forward, "
              f"{target.y * 1000:+6.0f} mm left   "
              f"({radius * 1000:3.0f} mm out, {bearing:+4.0f} deg)   "
              f"{target.width_m * 1000:3.0f} x {target.length_m * 1000:3.0f} mm")
        try:
            step = grasp.plan(target)
        except grasp.GraspError as exc:
            print(f"               SKIP: {exc}")
            continue
        print(f"               ok -- fingers to {cfg.GRIPPER_GAP_MM[step.opening]} mm, "
              f"approach {step.pitch:.0f} deg")


def choose(targets: list[detect.Target]) -> detect.Target | None:
    """First object we can actually pick, nearest first."""
    for target in targets:
        if _plannable(target):
            return target
    return None


def refine(arm: Arm, calibrated, target: detect.Target,
           mode: str, prompt: str | None,
           object_mm: float | None) -> detect.Target:
    """Measure the chosen object again, head-on, before reaching for it.

    The sweep found it from whatever station happened to see it, possibly out at
    the edge of that frame. Turning the base to put it in the middle of the picture
    costs one move and makes the measurement that the grasp is actually based on
    the best-conditioned one available -- the same geometry the 2 mm end-to-end
    validation was done in.

    Falls back to the sweep's own answer, loudly, rather than giving up: that
    answer is a real measurement too, and a grasp that closes on nothing is a
    reported miss rather than a crash.
    """
    try:
        look = sweep.best_refine(calibrated, target.x, target.y)
    except sweep.NoLook as exc:
        print(f"  cannot centre it ({exc}); using the sweep's measurement.")
        return target

    print(f"  looking again with the base at J1={look.pose[1]} "
          f"(dyaw {look.dyaw:+.1f}), object in the middle of the frame")
    found = look_once(arm, look, mode, prompt, object_mm)
    near = [t for t in found
            if math.hypot(t.x - target.x, t.y - target.y) <= REFIND_M]
    if not near:
        print(f"  WARNING: nothing within {REFIND_M * 1000:.0f} mm of where the "
              f"sweep put it. Using the sweep's measurement; if the gripper closes")
        print("  on nothing, the object was probably nudged or is mis-sized.")
        return target
    best = min(near, key=lambda t: math.hypot(t.x - target.x, t.y - target.y))
    moved = math.hypot(best.x - target.x, best.y - target.y)
    print(f"  refined to {best.x * 1000:.0f} mm forward, {best.y * 1000:+.0f} mm "
          f"left ({moved * 1000:.1f} mm from the sweep's answer)")
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="find and plan, but do not reach for anything")
    parser.add_argument("--auto", action="store_true",
                        help="every rung at once, each cube sized from its own outline")
    parser.add_argument("--markers", action="store_true",
                        help="detect ArUco tags instead of changes to the table")
    parser.add_argument("--colour", action="store_true",
                        help="find it by colour -- needs no tag and no background")
    parser.add_argument("--yolo", action="store_true",
                        help="use the neural detector (falls back to --markers if it is down)")
    parser.add_argument("--prompt", default=None,
                        help="with --yolo: comma-separated things to look for (YOLOE model)")
    parser.add_argument("--object-mm", type=float, default=None,
                        help="trust this object width instead of the silhouette's")
    parser.add_argument("--step", type=float, default=sweep.SURVEY_STEP_DEG,
                        help="degrees between sweep stations (default %(default)s)")
    parser.add_argument("--first", action="store_true",
                        help="stop sweeping as soon as something graspable is in view")
    parser.add_argument("--no-refine", action="store_true",
                        help="grasp on the sweep's measurement, without looking again")
    parser.add_argument("--drop-x", type=float, default=DROP_X)
    parser.add_argument("--drop-y", type=float, default=DROP_Y)
    args = parser.parse_args()

    mode = ("auto" if args.auto else "yolo" if args.yolo else "colour" if args.colour
            else "markers" if args.markers else "changes")
    if args.prompt and not args.yolo:
        parser.error("--prompt only means something with --yolo")

    try:
        calibrated = ws.load_all()
        _matrix, survey_pose, _name = calibrated[0]
        looks = sweep.stations(calibrated, args.step)
        covered, _missed = sweep.coverage(looks)
        print(f"{len(looks)} stations from {len(calibrated)} calibrated look(s), "
              f"{args.step:.0f} deg apart,")
        print(f"covering {covered:.1%} of what the arm can reach "
              f"(a tagged {sweep.OBJECT_SPAN_M * 1000:.0f} mm cube).\n")

        with Arm() as arm:
            targets = survey(arm, looks, mode, args.prompt, args.object_mm,
                             args.first)
            describe(targets)
            print("\nwrote /out/sweep_*.jpg, one per station")
            if args.dry_run:
                return 0

            target = choose(targets)
            if target is None:
                print("\nnothing here can be picked up.", file=sys.stderr)
                arm.move_to(survey_pose, speed_dps=20, verify=False)
                return 1

            print(f"\npicking {target.label}")
            if not args.no_refine:
                target = refine(arm, calibrated, target, mode,
                                args.prompt, args.object_mm)

            holding = grasp.pick(arm, target)
            if not holding:
                print("\nthe gripper closed on nothing. Common causes, in order:")
                print("  * the object is shorter than the 8 mm the fingers stop at")
                print("  * it is tall enough that the camera saw its top, not its base,")
                print("    so the reported position is pushed outward (parallax)")
                print("  * it was nudged aside by a finger on the way down")
                print("Run --dry-run and compare /out/sweep_*.jpg against reality.")
                arm.move_to(survey_pose, speed_dps=20, verify=False)
                return 1

            grasp.place(arm, args.drop_x, args.drop_y)
            arm.move_to(survey_pose, speed_dps=20, verify=False)
            print("\ndone.")
    except (ArmError, camera.CameraError, ValueError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
