#!/usr/bin/env python3
"""What can the arm see, and what can it reach? Answer both, and compare them.

    ... sweep_check.py                  # geometry only, no hardware touched
    ... sweep_check.py --steps          # how coverage trades against sweep time
    ... sweep_check.py --look           # actually drive the ring and photograph it

The first form needs nothing but the saved calibration, so it is the right thing
to run after any recalibration: it says whether the ring still covers the table
before anyone waits for the arm to prove it does not.

--look is the hardware check. It drives every station of the ring and saves an
annotated frame per station, so a human can confirm three things a coverage
percentage cannot: that the poses are safe to sweep through, that the camera
really sees the table at every bearing, and that adjacent stations overlap rather
than leaving a strip of table nobody looks at.
"""

import argparse
import sys
import time

import numpy as np

from roboarm import camera, sweep
from roboarm import config as cfg
from roboarm import workspace as ws
from roboarm.arm import Arm, ArmError


def describe_envelope(points: np.ndarray) -> None:
    radii = np.hypot(points[:, 0], points[:, 1])
    bearings = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    area = len(points) * 0.004 * 0.004 * 1e4
    print(f"the arm can put its fingertips on {area:.0f} cm2 of table:")
    print(f"  {radii.min() * 1000:.0f}..{radii.max() * 1000:.0f} mm from the base, "
          f"bearing {bearings.min():.0f}..{bearings.max():+.0f} deg")


def report(calibrated, step_deg: float) -> list:
    points = sweep.reachable_grasp_points()
    describe_envelope(points)
    matrix, survey, _name = calibrated[0]

    print("\nwhat can be MEASURED, for a 40 mm cube standing on the table:")
    print("  (the tag rung needs only the 26 mm printed square in shot; change")
    print("   detection needs the whole silhouette, so it reaches less far)")
    alone, _ = sweep.coverage([sweep.Look(0.0, survey, matrix)], points)
    print(f"\n  one look, the calibrated pose alone:  {alone:6.1%}")

    looks = sweep.stations(calibrated, step_deg)
    fraction, missed = sweep.coverage(looks, points)
    silhouette, _ = sweep.coverage(looks, points, sweep.OBJECT_SPAN_M)
    print(f"  {len(looks)} looks from {len(calibrated)} calibration(s), "
          f"{step_deg:.0f} deg apart:")
    print(f"      by tag         {fraction:6.1%}")
    print(f"      by silhouette  {silhouette:6.1%}")
    print("\nstations (J1, and where each one is looking):")
    for look in looks:
        centre = ws.apply(look.matrix, [(cfg.WRIST_CAM_SIZE[0] / 2,
                                         cfg.WRIST_CAM_SIZE[1] / 2)])[0]
        print(f"  dyaw {look.dyaw:+6.1f}  J1={look.pose[1]:3d}   frame centred on "
              f"{centre[0] * 1000:6.1f} mm fwd, {centre[1] * 1000:+6.1f} mm left")

    if len(missed):
        radii = np.hypot(missed[:, 0], missed[:, 1])
        bearings = np.degrees(np.arctan2(missed[:, 1], missed[:, 0]))
        print(f"\nnot covered: {len(missed)} cells, "
              f"r {radii.min() * 1000:.0f}..{radii.max() * 1000:.0f} mm, "
              f"bearing {bearings.min():.0f}..{bearings.max():+.0f} deg")
        # A RIM and a HOLE need opposite fixes, so decide which this is rather
        # than guessing from a radius threshold. The test has to be made BEARING
        # BY BEARING: the footprint is a quadrilateral, not an annulus, so how far
        # out the camera reaches genuinely varies with heading, and comparing one
        # global maximum against another calls a rim a hole.
        #
        # At a given bearing, a missed cell with a SEEN cell further out is a real
        # hole -- the camera reaches past it and simply skipped it, which a finer
        # ring fixes. A missed cell with nothing seen beyond it is rim: no yaw can
        # make the camera look further out, so more stations cannot help.
        missed_keys = {(round(px, 6), round(py, 6)) for px, py in missed}
        seen = np.array([p for p in points
                         if (round(p[0], 6), round(p[1], 6)) not in missed_keys])
        holes = 0
        if len(seen):
            seen_bearing = np.degrees(np.arctan2(seen[:, 1], seen[:, 0]))
            seen_radius = np.hypot(seen[:, 0], seen[:, 1])
            for bearing, radius in zip(bearings, radii):
                nearby = np.abs(seen_bearing - bearing) < 3.0
                if nearby.any() and seen_radius[nearby].max() > radius + 1e-6:
                    holes += 1
        if holes <= 0.05 * len(missed):
            deep = (radii.max() - radii.min()) * 1000
            print(f"  -- an outer RIM {deep:.0f} mm deep, not a hole: every bearing")
            print("     is looked at, but no yaw can make the camera look further")
            print("     out. A smaller --step will not help. Closing it needs a")
            print("     second calibrated look, tilted out -- calibrate_table.py --outer.")
        else:
            print(f"  -- {holes} of them are HOLES inside the covered band, with table")
            print("     seen further out at the same bearing. Try a smaller --step.")
    else:
        print("\nnothing reachable is unseen.")
    return looks


def compare_steps(calibrated) -> None:
    points = sweep.reachable_grasp_points()
    print("\nstep    looks   coverage   sweep time (about 3.5 s a look)")
    for step in (40, 35, 30, 25, 20, 15, 10):
        looks = sweep.stations(calibrated, step)
        fraction, _ = sweep.coverage(looks, points)
        print(f"  {step:2d}      {len(looks):3d}     {fraction:6.1%}     "
              f"{len(looks) * 3.5:5.0f} s")
    print("\nPast about 25 deg the extra looks buy almost nothing: a yaw changes")
    print("the BEARING the camera looks at, never the distance, so more of them")
    print("cannot widen the radial band. That needs a second calibrated look --")
    print("see calibrate_table.py --outer.")


def do_look(arm: Arm, calibrated, step_deg: float) -> int:
    """Drive the ring for real and photograph every station."""
    from roboarm import detect

    survey = calibrated[0][1]
    looks = sweep.stations(calibrated, step_deg)
    print(f"\ndriving {len(looks)} stations. The fingertips stay "
          f"{115:.0f} mm above the table throughout.")
    for look in looks:
        arm.move_to(look.pose, speed_dps=20)
        time.sleep(1.2)
        frame = camera.grab(6)[-1]
        found = detect.markers(frame, look.matrix, tag_m=cfg.OBJECT_TAG_M,
                               nadir=look.nadir)
        name = f"/out/sweep_{look.dyaw:+05.1f}.jpg".replace("+", "p").replace("-", "m")
        camera.write_image(name, detect.annotate(frame, look.matrix, found))
        seen = ", ".join(f"{t.label} at {t.x * 1000:.0f},{t.y * 1000:+.0f}"
                         for t in found) or "nothing"
        print(f"  J1={look.pose[1]:3d} (dyaw {look.dyaw:+6.1f}): {seen}")
    arm.move_to(survey, speed_dps=20)
    print("\nwrote /out/sweep_*.jpg, one per station.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=sweep.SURVEY_STEP_DEG,
                        help="degrees between looks (default %(default)s)")
    parser.add_argument("--steps", action="store_true",
                        help="compare step sizes instead of reporting one")
    parser.add_argument("--look", action="store_true",
                        help="drive the ring and photograph each station")
    args = parser.parse_args()

    try:
        calibrated = ws.load_all()
        for matrix, survey, name in calibrated:
            print(f"calibrated look {name!r}: {survey}")
            print(f"  the middle of its frame looks "
                  f"{sweep.look_bearing(matrix):+.1f} deg round from straight ahead")
        if args.steps:
            compare_steps(calibrated)
            return 0
        report(calibrated, args.step)
        if args.look:
            with Arm() as arm:
                return do_look(arm, calibrated, args.step)
    except (ArmError, camera.CameraError, ValueError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
