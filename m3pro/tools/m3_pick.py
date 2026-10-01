"""The M3's look-again pick: a cube found by the depth on whatever floor, picked up.

  1. the gripper opens (OPEN_MM: a 30 mm cube fits turned any way, its diagonal is 42 mm);
  2. the wide look: the depth's floor levels the camera, roboarm.m3vision.objects finds the cubes
     on it -- no board, no tags; the nearest kit cube (or the one nearest --at X,Y) is the target;
  3. the pre-grasp pose: the fingertips m3_place.BACK short along the approach line;
  4. the close look: the cube found again from there, the aim corrected (the arm's repeatability);
  5. down the line to the grip point -- the cube's middle, at least GRIP_Z_MIN up, so the
     fingertips stop ~4 mm over the floor -- the fingers close to the cube's width, back up;
  6. the wide look again: no cube where it stood = picked. The arm goes home holding it.

    python tools/m3_pick.py look              # the wide look only: what it finds (nothing moves after)
    python tools/m3_pick.py pick [--at X,Y]   # pick the nearest cube, or the one nearest X,Y (base_link, m)
    python tools/m3_pick.py drop              # open the gripper (at home: hand the cube over)

Pictures and what was found go to out/pick/. Through the bridge (tools/m3bridge.py).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import m3_handeye as he
import m3_place as mp

from roboarm import m3arm, m3kinematics, m3vision

OPEN_MM = 55
GRIP_Z_MIN = 0.018   # m: the grip point's lowest; the fingertips reach ~1.4 cm below it at 70 deg
OUT = ROOT / "out" / "pick"


def look(tag: str):
    """(levelled camera, the objects found, the picture) from where the arm is."""
    image, depth, K, pose = mp.snapshot()
    model = m3kinematics.frames(pose)["arm4"] @ he.urdf_camera()
    normal, height, _ = m3vision.floor_plane(depth, K)
    camera = m3vision.level(model, normal, height, m3kinematics.FLOOR)
    found = m3vision.objects(depth, K, camera)
    marked = image.copy()
    inv = np.linalg.inv(camera)
    for o in found:
        top = inv @ np.r_[o["centre"][:2], o["top"], 1.0]
        u, v = (K @ (top[:3] / top[2]))[:2]
        colour = (0, 255, 0) if o["size"] else (0, 165, 255)
        cv2.drawMarker(marked, (int(np.rint(u)), int(np.rint(v))), colour, cv2.MARKER_CROSS, 22, 2)
        label = f"{o['size'] * 1000:.0f} mm" if o["size"] else f"? {o['top'] * 1000:.0f} mm tall"
        cv2.putText(marked, label, (int(u) + 10, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{tag}.jpg"), marked)
    cv2.imwrite(str(OUT / f"{tag}_depth.png"), depth)
    return camera, found


def describe(found: list[dict]) -> str:
    return "; ".join(f"{'%.0f mm cube' % (o['size'] * 1000) if o['size'] else 'something %.0f mm tall' % (o['top'] * 1000)}"
                     f" at x {o['centre'][0]:.3f} y {o['centre'][1]:+.3f}, turned {o['yaw']:.0f} deg"
                     for o in found) or "nothing"


def plan_grasp(centre: np.ndarray, size: float):
    """(pitch, the line's poses pre-grasp -> grip) with the grip point in the cube's middle."""
    grip = np.array([centre[0], centre[1], max(size / 2, GRIP_Z_MIN)])
    yaw = math.atan2(grip[1] - m3kinematics.CHAIN[0][0][1], grip[0] - m3kinematics.CHAIN[0][0][0])
    for pitch in mp.PITCHES:
        r = math.radians(pitch)
        along = np.array([math.cos(yaw) * math.cos(r), math.sin(yaw) * math.cos(r), -math.sin(r)])
        poses = mp.line(grip - mp.BACK * along, grip, pitch, mp.FLOOR_CLEAR_PLACE)
        if poses and m3kinematics.problem(poses[0]) is None:
            return pitch, poses
    return None


def pick(at: np.ndarray | None) -> None:
    stamp = time.strftime("%H%M%S")
    mp.call("set_gripper", m3arm.gripper_angle(OPEN_MM))
    mp.move({int(j): v for j, v in mp.LOOK.items()}, speed=25)            # 2. the wide look
    _camera, found = look(f"{stamp}_1_wide")
    print("wide look:", describe(found))
    cubes = [o for o in found if o["size"]]
    if not cubes:
        raise SystemExit("no kit cube in view -- see out/pick/*_1_wide.jpg")
    target = min(cubes, key=lambda o: 0 if at is None else float(np.linalg.norm(o["centre"][:2] - at)))
    planned = plan_grasp(target["centre"], target["size"])
    if not planned:
        raise SystemExit(f"the cube at {np.round(target['centre'], 3).tolist()} is out of reach")
    mp.move(planned[1][0])                                                # 3. the pre-grasp pose
    _camera, close = look(f"{stamp}_2_close")                             # 4. look again from close
    again = [o for o in close if o["size"] and np.linalg.norm(o["centre"][:2] - target["centre"][:2]) < 0.03]
    centre = again[0]["centre"] if again else target["centre"]
    print(f"close look: {'found it' if again else 'not found: the wide look is kept'}, "
          f"moved {1000 * np.linalg.norm(centre[:2] - target['centre'][:2]):.1f} mm")
    planned = plan_grasp(centre, target["size"]) or planned
    pitch, poses = planned
    mp.move(poses[0])
    for p in poses[1:]:                                                   # 5. down, close, up
        mp.move(p, floor_clear=mp.FLOOR_CLEAR_PLACE, speed=15)
    mp.call("close_gripper", width_mm=round(target["size"] * 1000), floor_clear=mp.FLOOR_CLEAR_PLACE)
    time.sleep(0.6)
    for p in reversed(poses[:-1]):
        mp.move(p, floor_clear=mp.FLOOR_CLEAR_PLACE, speed=15)
    mp.move({int(j): v for j, v in mp.LOOK.items()}, speed=25)            # 6. is it gone?
    _camera, after = look(f"{stamp}_3_after")
    left = [o for o in after if np.linalg.norm(o["centre"][:2] - centre[:2]) < 0.03]
    record = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "target": np.round(centre, 4).tolist(),
              "size_mm": round(target["size"] * 1000), "yaw_deg": round(target["yaw"], 1), "pitch": pitch,
              "grip_pose": poses[-1], "close_look_found": bool(again), "picked": not left}
    log = OUT / "picks.json"
    records = json.loads(log.read_text()) if log.exists() else []
    records.append(record)
    log.write_text(json.dumps(records, indent=1))
    if left:
        print(f"NOT picked: still on the floor -- {describe(left)}")
        mp.call("home")
        return
    print(f"picked the {record['size_mm']} mm cube from x {centre[0]:.3f} y {centre[1]:+.3f} (approach {pitch} deg)")
    mp.call("home")   # holding it; `drop` opens the gripper


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("look")
    p = sub.add_parser("pick")
    p.add_argument("--at", help="X,Y in base_link (m): the cube nearest this")
    sub.add_parser("drop")
    args = parser.parse_args()
    if "error" in he.http("/arm/call", {"method": "read"}):
        print("engage:", mp.call("engage"))   # limp since the bridge started: home first, 3 s
    if args.cmd == "look":
        mp.move({int(j): v for j, v in mp.LOOK.items()}, speed=25)
        _camera, found = look(time.strftime("%H%M%S") + "_look")
        print(describe(found))
    elif args.cmd == "pick":
        pick(None if args.at is None else np.array([float(v) for v in args.at.split(",")]))
    else:
        print(mp.call("set_gripper", m3arm.gripper_angle(OPEN_MM)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
