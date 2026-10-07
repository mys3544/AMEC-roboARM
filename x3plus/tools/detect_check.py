#!/usr/bin/env python3
"""See what the detector finds, and prove it by touching one of them.

    ... detect_check.py                  # tags on the table: what do we see?
    ... detect_check.py --reach          # drive the fingertip to the first object

The last one is the only step that actually proves anything. A detector that draws a
neat box around an object has shown that it can see it, not that it knows WHERE it
is: parallax, a homography fitted from a slightly wrong board position, and the
gripper's own tool length all live between the box and the table. Only the arm can
settle it -- exactly as with calibrate_table.py --verify, which is how we learned the
table mapping is good to 2 mm.

--reach stops 25 mm ABOVE the object rather than at it. If something is wrong, that
is the height at which it is obvious and harmless.
"""

import argparse
import sys
import time
from pathlib import Path

from roboarm import camera, detect
from roboarm import config as cfg
from roboarm import kinematics as kin
from roboarm import workspace as ws
from roboarm.arm import Arm, ArmError

# Height the fingertips stop at during --reach. This MUST clear the object: the
# fingers arrive closed and directly above its centre, so hovering lower than the
# object is tall drives them into its side. 25 mm suits a flat tag on the table;
# anything with height needs --hover-mm above it.
HOVER_M = 0.025
HOLD = 40.0


def at_survey_pose(arm: Arm, matrix_pose: dict[int, int]) -> list:
    """Drive to the pose the homography was fitted at, and grab settled frames."""
    arm.move_to(matrix_pose, speed_dps=15)
    time.sleep(1.5)
    return camera.grab(6)


def report(targets: list[detect.Target]) -> None:
    if not targets:
        print("nothing found.")
        return
    print(f"{len(targets)} object(s):")
    for target in targets:
        note = "" if target.graspable else f"   NOT GRASPABLE: {target.why_not()}"
        print(
            f"  {target.label:<12} {target.x * 1000:6.0f} mm forward, "
            f"{target.y * 1000:+6.0f} mm left   "
            f"{target.width_m * 1000:.0f} x {target.length_m * 1000:.0f} mm   "
            f"short axis {target.angle_deg:+.0f} deg{note}"
        )


def do_find(arm: Arm, pose: dict[int, int], matrix, reach: bool,
            hover_m: float, object_mm: float | None) -> int:
    frames = at_survey_pose(arm, pose)
    frame = frames[-1]

    nadir = kin.camera_nadir(pose)
    print(f"lens is above {nadir[0] * 1000:.0f} mm forward, {nadir[1] * 1000:+.0f} mm "
          f"left, {kin.camera_height(pose) * 1000:.0f} mm up -- parallax pushes "
          f"objects away from THERE, not from the middle of the picture.\n")
    raw_tags = detect.markers(frame, matrix, tag_m=None)
    tagged = detect.markers(frame, matrix, tag_m=cfg.OBJECT_TAG_M, nadir=nadir)
    print("--- tags, as the homography saw them (uncorrected) ---")
    report(raw_tags)
    print("\n--- tags, corrected for sitting above the table ---")
    report(tagged)
    for raw, fixed in zip(raw_tags, tagged):
        print(f"  {fixed.label}: correction moved it "
              f"{(fixed.x - raw.x) * 1000:+.0f} mm forward, "
              f"{(fixed.y - raw.y) * 1000:+.0f} mm left; "
              f"apparent tag size {raw.width_m * 1000:.0f} mm vs "
              f"{cfg.OBJECT_TAG_M * 1000:.0f} mm real "
              f"(magnification {raw.width_m / cfg.OBJECT_TAG_M:.2f})")

    targets = tagged
    if object_mm:
        targets = detect.declared_size(targets, object_mm / 1000)
        print(f"\n(size overridden to {object_mm:.0f} mm by hand; position is still "
              f"the camera's)")
        report(targets)

    out = Path(__file__).resolve().parents[1] / "out" / "detected.jpg"
    camera.write_image(str(out), detect.annotate(frame, matrix, targets))
    print(f"\nwrote {out} -- check the boxes land on the objects")

    if not reach:
        return 0
    graspable = [t for t in targets if t.graspable]
    if not graspable:
        print("\nnothing graspable to reach for", file=sys.stderr)
        return 1

    target = graspable[0]
    z = -cfg.TABLE_BELOW_PLATE + hover_m
    try:
        arm_pose, pitch = kin.solve(target.x, target.y, z)
    except kin.Unreachable as exc:
        print(f"\n{target.label} is not reachable: {exc}", file=sys.stderr)
        return 1
    # Closed, so the fingertips sit on the tool axis and can be eyeballed against
    # the object. Open they splay, and there is no single point to compare.
    arm_pose[6] = cfg.GRIPPER_CLOSED

    print(f"\nreaching for {target.label} at {target.x * 1000:.0f} mm forward, "
          f"{target.y * 1000:+.0f} mm left (pitch {pitch:.0f})")
    print(f"stopping {hover_m * 1000:.0f} mm above the table -- it will NOT touch.")
    arm.move_to(arm_pose, speed_dps=12)
    time.sleep(1.0)

    # Did the arm actually GO there? Moves are open loop, so nothing checks the
    # result on the way. Without this readback a miss
    # cannot be blamed: "the camera said the wrong place" and "the arm did not go
    # where it was told" look identical from the far side of the ruler.
    actual = arm.read()
    drift = {j: arm_pose[j] - actual[j] for j in (1, 2, 3, 4) if arm_pose[j] != actual[j]}
    landed = kin.forward(actual)
    print("\njoints commanded vs reached: "
          + (", ".join(f"J{j} off by {d:+d}" for j, d in drift.items()) or "exact"))
    print(f"so the model puts the fingertips at {landed[0] * 1000:.0f} mm forward, "
          f"{landed[1] * 1000:+.0f} mm left, "
          f"{(landed[2] + cfg.TABLE_BELOW_PLATE) * 1000:.0f} mm up")
    print(f"  vs commanded {target.x * 1000:.0f}, {target.y * 1000:+.0f}, "
          f"{hover_m * 1000:.0f}  -> arm itself is out by "
          f"{(landed[0] - target.x) * 1000:+.0f} mm forward, "
          f"{(landed[1] - target.y) * 1000:+.0f} mm left")
    print("  (any remaining miss is the camera's; this part is the arm's.)")

    print(f"\nHolding {HOLD:.0f}s. How far are the fingertips from the object, "
          f"and which way?", flush=True)
    time.sleep(HOLD)
    arm.move_to(pose, speed_dps=15)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reach", action="store_true",
                        help="drive the fingertip over the first object found")
    parser.add_argument("--hover-mm", type=float, default=HOVER_M * 1000,
                        help="fingertip height during --reach; MUST clear the object")
    parser.add_argument("--object-mm", type=float, default=None,
                        help="trust this width instead of the tag's own")
    args = parser.parse_args()

    try:
        matrix, pose = ws.load()
        with Arm() as arm:
            return do_find(arm, pose, matrix, args.reach,
                           args.hover_mm / 1000, args.object_mm)
    except (ArmError, camera.CameraError, ValueError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
