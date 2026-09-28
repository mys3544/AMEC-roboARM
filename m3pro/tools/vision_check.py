#!/usr/bin/env python3
"""See what the neural detector finds, and prove it by touching one of them.

    # start the service (own container, GPU):
    docker compose --profile vision up -d vision

    ... vision_check.py --health                 # is the model up? which classes?
    ... vision_check.py                          # objects on the table: what do we see?
    ... vision_check.py --reach                  # drive the fingertip over the first one

Same idea as detect_check.py: a neat box only proves the detector can SEE an object,
not that it knows where it is. Parallax, the homography and the tool length all live
between the box and the table, so --reach (stops 25 mm ABOVE, never touches) is the
only step that settles it.

The frame comes from the survey pose the homography was fitted at -- boxes come back
in pixels and roboarm.detect maps them to the table with that same homography, so the
vision container never needs the calibration.
"""

import argparse
import json
import sys
import time

from roboarm import camera, detect
from roboarm import config as cfg
from roboarm import kinematics as kin
from roboarm import workspace as ws
from roboarm.arm import Arm, ArmError

HOVER_M = 0.025
HOLD = 40.0


def show_health(url: str) -> int:
    try:
        info = detect._vision_post("/health", None, {}, url, timeout=3.0)
    except (detect.DetectorOffline, ValueError) as exc:
        print(f"vision service not usable: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(info, indent=2))
    return 0


def report(targets: list[detect.Target]) -> None:
    if not targets:
        print("nothing found.")
        return
    print(f"{len(targets)} object(s):")
    for target in targets:
        note = "" if target.graspable else f"   NOT GRASPABLE: {target.why_not()}"
        print(
            f"  {target.label:<16} {target.confidence:4.2f}   "
            f"{target.x * 1000:6.0f} mm forward, {target.y * 1000:+6.0f} mm left   "
            f"{target.width_m * 1000:.0f} x {target.length_m * 1000:.0f} mm{note}"
        )


def do_find(arm: Arm, pose: dict[int, int], matrix, reach: bool, url: str) -> int:
    arm.move_to(pose, speed_dps=15)
    time.sleep(1.5)
    frame = camera.grab(6)[-1]

    try:
        targets = detect.objects(frame, matrix, url=url)
    except detect.DetectorOffline as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"\nvision service rejected the request: {exc}", file=sys.stderr)
        return 1

    report(targets)
    camera.write_image("/out/vision.jpg", detect.annotate(frame, matrix, targets))
    print("\nwrote /out/vision.jpg -- check the boxes land on the objects")

    if not reach:
        return 0
    graspable = [t for t in targets if t.graspable]
    if not graspable:
        print("\nnothing graspable to reach for", file=sys.stderr)
        return 1

    target = graspable[0]
    z = -cfg.TABLE_BELOW_PLATE + HOVER_M
    try:
        arm_pose, pitch = kin.solve(target.x, target.y, z)
    except kin.Unreachable as exc:
        print(f"\n{target.label} is not reachable: {exc}", file=sys.stderr)
        return 1
    arm_pose[cfg.GRIPPER_ID] = cfg.GRIPPER_CLOSED  # fingertips on the tool axis, eyeball-able

    print(f"\nreaching for {target.label} ({target.confidence:.2f}) at "
          f"{target.x * 1000:.0f} mm forward, {target.y * 1000:+.0f} mm left "
          f"(pitch {pitch:.0f})")
    print(f"stopping {HOVER_M * 1000:.0f} mm above the table -- it will NOT touch.")
    arm.move_to(arm_pose, speed_dps=12)
    time.sleep(1.0)
    print(f"\nHolding {HOLD:.0f}s. How far are the fingertips from the object, "
          f"and which way?", flush=True)
    time.sleep(HOLD)
    arm.move_to(pose, speed_dps=15)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health", action="store_true",
                        help="just query the service and print what it reports")
    parser.add_argument("--reach", action="store_true",
                        help="drive the fingertip over the first object found")
    parser.add_argument("--url", default=cfg.DETECTOR_URL,
                        help=f"vision service base URL (default {cfg.DETECTOR_URL})")
    args = parser.parse_args()

    if args.health:
        return show_health(args.url)

    try:
        matrix, pose = ws.load()
        with Arm() as arm:
            return do_find(arm, pose, matrix, args.reach, args.url)
    except (ArmError, camera.CameraError, ValueError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
