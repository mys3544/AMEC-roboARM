#!/usr/bin/env python3
"""Move to a survey pose, look down at the board, and report what the camera sees.

    docker compose run --rm core python tools/survey.py
    docker compose run --rm core python tools/survey.py --fwd 0.150 --lift 0.090

The wrist camera is the only one that can see the workspace -- the mast camera is
fixed horizontal, with the table roughly 72 degrees below its axis. So the arm has
to move to a fixed "survey" pose, look down, and find the object from there.

From ONE fixed survey pose the camera has a fixed relationship to the table, so a
single homography maps image pixels straight to table coordinates. No camera
intrinsics, no full hand-eye solve. The pose has to be repeatable, which it is:
measured joint repeatability is 1 degree.

Writes an annotated frame so we can see which markers are usable.
"""

import argparse
import sys
import time

import cv2

from roboarm import camera
from roboarm import config as cfg
from roboarm import kinematics as kin
from roboarm.arm import Arm, ArmError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fwd", type=float, default=0.150, help="metres in front of the axis")
    parser.add_argument("--lift", type=float, default=0.090, help="metres above the table")
    parser.add_argument("--out", default="/out/survey.jpg")
    args = parser.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.ARUCO_DICT))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    try:
        pose, _pitch = kin.solve(args.fwd, 0.0, -cfg.TABLE_BELOW_PLATE + args.lift)
    except kin.Unreachable as exc:
        print(f"survey pose unreachable: {exc}", file=sys.stderr)
        return 1
    pose[6] = cfg.GRIPPER_OPEN

    try:
        with Arm() as arm:
            reached = arm.move_to(pose, speed_dps=15, verify=False)
            time.sleep(1.5)
            tip = kin.forward(reached)
            print(
                f"survey pose: J1={reached[1]} J2={reached[2]} J3={reached[3]} "
                f"J4={reached[4]}   pitch {kin.tool_pitch(reached):.0f}"
            )
            print(
                f"fingertip at {tip[0] * 1000:.0f} mm forward, "
                f"{(tip[2] + cfg.TABLE_BELOW_PLATE) * 1000:.0f} mm above the table"
            )

            frame = camera.grab(1)[0]

            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if ids is None or not len(ids):
                print(f"\nNO MARKERS using {cfg.ARUCO_DICT}")
            else:
                print(f"\n{len(ids)} marker(s) with {cfg.ARUCO_DICT}:")
                for marker, corner in zip(ids.ravel(), corners):
                    pts = corner.reshape(4, 2)
                    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                    side = max(
                        float(((pts[i] - pts[(i + 1) % 4]) ** 2).sum() ** 0.5) for i in range(4)
                    )
                    print(f"  id {int(marker):>3}  centre ({cx:>6.1f}, {cy:>6.1f}) px"
                          f"   {side:.0f} px across")
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            camera.write_image(args.out, frame)
            print(f"\nwrote {args.out} ({frame.shape[1]}x{frame.shape[0]})")
    except (ArmError, camera.CameraError) as exc:
        print(f"\nARM ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
