#!/usr/bin/env python3
"""Is the parallax correction right? Ask the same object from several yaws.

    ... parallax_probe.py        # put ONE tagged object on the table first

The homography maps the TABLE, so a tag on top of a cube images larger than life
and thrown OUTWARD from the point directly under the lens. detect._unlift() undoes
that, measuring the magnification from the tag's own known size -- so it needs no
camera height and no object height, but it does need the NADIR, and the nadir comes
from the kinematic model rather than from anything measured.

This tool tests that model without a ruler. Turning the base moves the nadir while
leaving the object exactly where it is, so the correction changes and the answer
must not. Any fan-out in the corrected positions is the nadir model being wrong;
the uncorrected positions are printed alongside to show what is being undone.

MEASURED 2026-09-10, cube at 176 mm forward, six stations spanning 20 degrees of
yaw, over which the nadir travelled 50 mm:

    uncorrected   spread 5.1 mm
    corrected     spread 1.6 mm

So the nadir model and the unlift are both sound, and a pick that misses is the
ARM's fault, not the camera's -- which is what sent the investigation to
tools/reach_check.py and found 9.9 mm of open-loop reach error.
"""

import argparse
import sys
import time

import numpy as np

from roboarm import camera, detect, sweep
from roboarm import config as cfg
from roboarm import workspace as ws
from roboarm.arm import Arm, ArmError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--span", type=float, default=24.0,
                        help="degrees of yaw either side to try")
    parser.add_argument("--step", type=float, default=4.0)
    args = parser.parse_args()

    try:
        matrix, survey = ws.load()
        with Arm() as arm:
            arm.move_to(survey, speed_dps=20)
            time.sleep(1.5)
            home = sweep.Look(0.0, survey, matrix)
            seed = detect.markers(camera.grab(6)[-1], matrix,
                                  tag_m=cfg.OBJECT_TAG_M, nadir=home.nadir)
            if not seed:
                print("no tag in view from the calibrated pose", file=sys.stderr)
                return 1
            cube = seed[0]
            print(f"object at {cube.x * 1000:.1f} mm forward, "
                  f"{cube.y * 1000:+.1f} mm left")

            usable = []
            offset = -args.span
            while offset <= args.span:
                try:
                    pose = sweep.pose_at(survey, offset)
                except sweep.NoLook:
                    offset += args.step
                    continue
                look = sweep.Look(float(offset), pose, sweep.rotate(matrix, offset))
                if sweep.sees(look, cube.x, cube.y, cfg.OBJECT_TAG_M):
                    usable.append(look)
                offset += args.step
            print(f"{len(usable)} stations can see it: "
                  f"{[int(look.dyaw) for look in usable]}\n")

            print("dyaw   nadir (mm)        raw tag (mm)       corrected (mm)     mag")
            raws, fixed = [], []
            for look in usable:
                arm.move_to(look.pose, speed_dps=20)
                time.sleep(1.3)
                frame = camera.grab(6)[-1]
                flat = detect.markers(frame, look.matrix, tag_m=None)
                lifted = detect.markers(frame, look.matrix, tag_m=cfg.OBJECT_TAG_M,
                                        nadir=look.nadir)
                if not flat or not lifted:
                    print(f"{look.dyaw:+5.0f}   -- no tag decoded --")
                    continue
                grown = flat[0].width_m / cfg.OBJECT_TAG_M
                nx, ny = look.nadir
                print(f"{look.dyaw:+5.0f}  ({nx * 1000:6.1f},{ny * 1000:+6.1f})  "
                      f"({flat[0].x * 1000:6.1f},{flat[0].y * 1000:+6.1f})  "
                      f"({lifted[0].x * 1000:6.1f},{lifted[0].y * 1000:+6.1f})  {grown:.3f}")
                raws.append((flat[0].x, flat[0].y))
                fixed.append((lifted[0].x, lifted[0].y))
            arm.move_to(survey, speed_dps=20)

        for name, points in (("uncorrected", raws), ("corrected", fixed)):
            if len(points) < 2:
                continue
            seen = np.array(points)
            spread = np.linalg.norm(seen - seen.mean(axis=0), axis=1).max() * 1000
            print(f"\n{name:12s} mean ({seen[:, 0].mean() * 1000:6.1f},"
                  f"{seen[:, 1].mean() * 1000:+6.1f}) mm, worst deviation {spread:5.1f} mm")
        if len(fixed) >= 2 and len(raws) >= 2:
            print("\nIf 'corrected' is the tighter of the two, the nadir model and the")
            print("unlift are both doing their job, and a pick that misses is the arm.")
    except (ArmError, camera.CameraError, ValueError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
