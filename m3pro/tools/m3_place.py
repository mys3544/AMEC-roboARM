"""Place-and-look: where the M3's gripper really puts something it was told to put where the camera
looked -- the look-again pick's own error, measured without a ruler. The arm holds a 30 mm cube;
for a ChArUco corner on the board in front of it the arm

  1. goes to a pre-place pose: the fingertips BACK m short of the corner along the approach line,
     the camera ~24 cm from it (the depth starts at 0.2 m);
  2. looks: the floor in the depth levels the camera (roboarm/m3vision.py), the board gives the
     corner -- the target, in base_link;
  3. places the cube on it along that straight line (IK every 2 cm; the last stretch keeps only
     FLOOR_CLEAR_PLACE from the floor just measured), lets go, and backs out the same way;
  4. looks again, from the wide look (from close the depth sees too little of the cube's top
     face): the cube against the corner IN THE BOARD'S OWN FRAME -- the corner and
     the board's plane from the colour picture, the cube's top face from the depth's points 3 cm
     above the floor, turned onto the board's plane (cube_on_board). The arm's repeatability and
     the depth sensor's ~2 deg against the colour camera stay out of the miss.

    python tools/m3_place.py open             # the gripper opens: put the cube between the fingers
    python tools/m3_place.py hold [30]        # the gripper closes to 30 mm
    python tools/m3_place.py plan             # from the wide look: the corners a placement can reach
    python tools/m3_place.py run N            # place on corner N (an id from `plan`) and measure
    python tools/m3_place.py measure N        # a cube already on the board, against corner N
    python tools/m3_place.py home

Each run appends to out/place/results.json; then `open` and `hold` again for the next. Through the
bridge (tools/m3bridge.py).
"""

from __future__ import annotations

import argparse
import base64
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

from roboarm import m3arm, m3kinematics, m3vision

LOOK = {"1": 88, "2": 105, "3": 25, "4": 0, "5": 90}
CUBE = 0.030               # m, the kit's small cube
BACK = 0.12                # m the fingertips stop short along the approach: the camera ~24 cm off
DROP = 0.004               # m between the cube's bottom and the board when the fingers open
FLOOR_CLEAR_PLACE = 0.003  # m the fingertips may come to the floor, the floor just measured
RELEASE_MM = 45            # the fingers open to this to let go (wider than the cube, not wide open)
PITCHES = (70, 65, 75, 60, 80, 55)
OUT = ROOT / "out" / "place"


def call(method: str, *args, **kwargs):
    reply = he.http("/arm/call", {"method": method, "args": list(args), "kwargs": kwargs})
    if "error" in reply:
        raise SystemExit(f"{method}: {reply['error']}")
    return reply.get("result")


def move(pose: dict, floor_clear: float | None = None, speed: float = 20) -> None:
    kwargs = {"speed_dps": speed} if floor_clear is None else {"speed_dps": speed, "floor_clear": floor_clear}
    call("move_to", {str(j): v for j, v in pose.items()}, **kwargs)


# ------------------------------------------------------------------------------------- seeing --
def snapshot():
    time.sleep(1.2)   # still: the arm stops swaying, the depth (0.17 s behind the colour) catches up
    snap = he.http("/rgbd")
    if "error" in snap:
        raise SystemExit(snap["error"])
    image = cv2.imdecode(np.frombuffer(base64.b64decode(snap["color_jpeg"]), np.uint8), cv2.IMREAD_COLOR)
    depth = cv2.imdecode(np.frombuffer(base64.b64decode(snap["depth_png"]), np.uint8), cv2.IMREAD_UNCHANGED)
    pose = {int(j): float(v) for j, v in snap["joints"].items()}
    return image, depth, np.asarray(snap["K"], float), pose


def see(image, depth, K, pose):
    """The levelled camera (base <- camera), the board (camera <- board) and the board's normal
    in the camera frame, pointing at it."""
    model = m3kinematics.frames(pose)["arm4"] @ he.urdf_camera()
    normal, height, _share = m3vision.floor_plane(depth, K)
    camera = m3vision.level(model, normal, height, m3kinematics.FLOOR)
    pixels, points = he.corners(image)
    if len(pixels) < 6:
        raise SystemExit(f"only {len(pixels)} board corners in view")
    _ok, rvec, tvec = cv2.solvePnP(points, pixels, K, None)
    board = he.rt(np.concatenate([rvec.ravel(), tvec.ravel()]))
    up = board[:3, 2] * np.sign(-board[:3, 2] @ board[:3, 3])
    return camera, board, up


def corner_in_base(camera, board, n: int) -> np.ndarray:
    c = he.board().getChessboardCorners()[n]
    return (camera @ board @ np.r_[c, 1.0])[:3]


def cube_on_board(depth, K, board, up, near: np.ndarray, size: float = CUBE) -> tuple[np.ndarray, tuple] | None:
    """The cube's centre on the board, in the board's frame (m), and its footprint as measured
    ((w, h) m, turn deg): the depth's points over the floor within 6 cm of `near` (board frame),
    turned from the depth sensor's frame into the colour camera's by the rotation that lays the
    depth's floor onto the board's plane (the PnP's), dropped onto the board, the smallest
    rectangle round the blob nearest `near` -- every face of a cube standing there lies over its
    footprint. The DCW2
    at ~24 cm sees only the top face, and not its band nearest the camera (2026-10-01: 30.9 x 23.0
    mm for a 30 mm cube): a side shorter than `size` is grown to it on the camera's side."""
    normal, height, _ = m3vision.floor_plane(depth, K)
    p = m3vision.points(depth, K, step=1)
    above = height + p @ normal            # the floor's normal points at the camera
    axis = np.cross(normal, up)
    sin, cos = float(np.linalg.norm(axis)), float(normal @ up)
    turn = cv2.Rodrigues(axis / sin * np.arctan2(sin, cos))[0] if sin > 1e-9 else np.eye(3)
    inv = np.linalg.inv(board)
    in_board = (inv[:3, :3] @ (turn @ p.T)).T + inv[:3, 3]
    on = (above > 0.005) & (above < size + 0.01) & (np.linalg.norm(in_board[:, :2] - near[:2], axis=1) < 0.06)
    if on.sum() < 30:
        return None
    # one object only: the points over the floor fall into blobs a few mm apart at most; another
    # cube nearby (2026-10-01: the last one, left on the board) is a blob of its own
    cells = {tuple(c) for c in np.floor(in_board[on, :2] / 0.004).astype(int)}
    blobs, seen = [], set()
    for start in cells:
        if start in seen:
            continue
        todo, blob = [start], []
        seen.add(start)
        while todo:
            c = todo.pop()
            blob.append(c)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if n in cells and n not in seen:
                        seen.add(n)
                        todo.append(n)
        blobs.append(blob)
    cell_of = np.floor(in_board[:, :2] / 0.004).astype(int)
    nearest = min(blobs, key=lambda b: np.linalg.norm((np.mean(b, axis=0) + 0.5) * 0.004 - near[:2]))
    keep = {tuple(c) for c in nearest}
    on &= np.array([tuple(c) in keep for c in cell_of])
    if on.sum() < 30:
        return None
    (cx, cy), (w, h), angle = cv2.minAreaRect(in_board[on, :2].astype(np.float32))
    centre = np.array([cx, cy])
    to_camera = inv[:2, 3] - centre                 # the camera's foot on the board, from the cube
    for length, a in ((w, np.radians(angle)), (h, np.radians(angle) + np.pi / 2)):
        along = np.array([np.cos(a), np.sin(a)])
        if length < size:
            centre = centre + np.sign(to_camera @ along) * (size - length) / 2 * along
    return np.array([centre[0], centre[1], size / 2]), ((float(w), float(h)), float(angle))


def mark(image, K, board, up, corner: np.ndarray, cube: np.ndarray | None, path: Path) -> None:
    """The picture with the corner (red) and the measured cube top centre (green), to check by eye."""
    def px(point, lift):
        c = (board @ np.r_[point, 1.0])[:3] + lift * up
        uv = K @ (c / c[2])
        return int(np.rint(uv[0])), int(np.rint(uv[1]))
    out = image.copy()
    cv2.drawMarker(out, px(corner, 0.0), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    if cube is not None:
        cv2.drawMarker(out, px(cube, CUBE), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
    cv2.putText(out, "red: the corner   green: the cube's top centre, measured", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), out)


def measure(n: int, camera, board, up, image, depth, K, tag: str) -> dict | None:
    """The cube against corner n, in the board's frame and as forward/left in base_link; the
    pictures saved under out/place/<tag>."""
    corner = he.board().getChessboardCorners()[n]
    found = cube_on_board(depth, K, board, up, corner)
    cv2.imwrite(str(OUT / f"{tag}.jpg"), image)
    cv2.imwrite(str(OUT / f"{tag}_depth.png"), depth)
    mark(image, K, board, up, corner, None if found is None else found[0] - [0, 0, CUBE / 2], OUT / f"{tag}_marked.jpg")
    if found is None:
        return None
    cube, ((w, h), angle) = found
    miss = (cube - corner)[:2]
    forward, left = ((camera @ board)[:3, :2] @ miss)[:2]
    return {"corner": n, "miss_board_m": miss.round(4).tolist(), "forward_mm": round(1000 * forward, 1),
            "left_mm": round(1000 * left, 1), "off_mm": round(1000 * float(np.linalg.norm(miss)), 1),
            "footprint_mm": [round(1000 * w, 1), round(1000 * h, 1)], "turned_deg": round(angle, 1)}


# ----------------------------------------------------------------------------------- planning --
def line(a: np.ndarray, b: np.ndarray, pitch: float, floor_clear: float, step: float = 0.02):
    """IK poses along the straight line a -> b at `pitch`, every `step`; None if any is out of reach."""
    n = max(1, math.ceil(np.linalg.norm(b - a) / step))
    out = []
    for i in range(n + 1):
        p = m3kinematics.reach(a + (b - a) * i / n, pitch, floor_clear=floor_clear)
        if p is None:
            return None
        out.append(p)
    return out


def plan(target: np.ndarray):
    """(pitch, the line's poses pre-place -> place) for putting the cube's centre on `target`."""
    place = target + np.array([0.0, 0.0, CUBE / 2 + DROP])
    yaw = math.atan2(place[1] - m3kinematics.CHAIN[0][0][1], place[0] - m3kinematics.CHAIN[0][0][0])
    for pitch in PITCHES:
        r = math.radians(pitch)
        along = np.array([math.cos(yaw) * math.cos(r), math.sin(yaw) * math.cos(r), -math.sin(r)])
        poses = line(place - BACK * along, place, pitch, FLOOR_CLEAR_PLACE)
        if poses and m3kinematics.problem(poses[0]) is None:   # the pre-place pose: the usual margin
            return pitch, poses
    return None


# ----------------------------------------------------------------------------------- commands --
def wide_look():
    move({int(j): v for j, v in LOOK.items()}, speed=25)
    image, depth, K, pose = snapshot()
    camera, board, _up = see(image, depth, K, pose)
    return camera, board


def cmd_plan() -> None:
    camera, board = wide_look()
    for n in range(len(he.board().getChessboardCorners())):
        c = corner_in_base(camera, board, n)
        found = plan(c)
        if found:
            print(f"corner {n:2d} at x {c[0]:.3f} y {c[1]:+.3f}: placeable at {found[0]} deg, {len(found[1])} steps")


def cmd_run(n: int) -> None:
    camera, board = wide_look()
    found = plan(corner_in_base(camera, board, n))
    if not found:
        raise SystemExit(f"corner {n}: no placement plan")
    pitch, poses = found
    move(poses[0])                                          # 1. the pre-place pose
    image, depth, K, pose = snapshot()                      # 2. look close
    camera, board, up = see(image, depth, K, pose)
    target = corner_in_base(camera, board, n)
    found = plan(target)
    if not found:
        raise SystemExit(f"corner {n} from close: no placement plan")
    pitch, poses = found
    move(poses[0])
    for p in poses[1:]:                                     # 3. down the line, let go, back up it
        move(p, floor_clear=FLOOR_CLEAR_PLACE, speed=15)
    call("set_gripper", m3arm.gripper_angle(RELEASE_MM), floor_clear=FLOOR_CLEAR_PLACE)
    time.sleep(0.5)
    for p in reversed(poses[:-1]):
        move(p, floor_clear=FLOOR_CLEAR_PLACE, speed=15)
    move({int(j): v for j, v in LOOK.items()}, speed=25)    # 4. look again, from the wide look:
    image, depth, K, pose = snapshot()                      #    from close the depth misses most of
    camera, board, up = see(image, depth, K, pose)          #    the cube's top face (2026-10-01)
    result = measure(n, camera, board, up, image, depth, K, f"run_{n}_{time.strftime('%H%M%S')}")
    call("home")   # out of the way: the cube can be taken off the board and handed back
    if result is None:
        raise SystemExit("the cube was not found next to the corner -- see out/place/*_marked.jpg")
    result.update({"pitch": pitch, "pose": poses[-1], "when": time.strftime("%Y-%m-%d %H:%M:%S")})
    results = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else []
    results.append(result)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(f"corner {n} at {pitch} deg: the cube's centre {result['forward_mm']:+.1f} mm forward,"
          f" {result['left_mm']:+.1f} mm left of the corner ({result['off_mm']:.1f} mm off)")


def cmd_measure(n: int) -> None:
    """The cube already on the board, measured against corner n -- nothing placed: from corner n's
    pre-place pose when it has one and sees enough of the board, else from the wide look."""
    camera, board = wide_look()
    found = plan(corner_in_base(camera, board, n))
    view = None
    if found:
        move(found[1][0])
        image, depth, K, pose = snapshot()
        try:
            view = see(image, depth, K, pose)
        except SystemExit as why:   # the cube hides too much of the board from close
            print(f"{why}; from the wide look instead")
    if view is None:
        move({int(j): v for j, v in LOOK.items()}, speed=25)
        image, depth, K, pose = snapshot()
        view = see(image, depth, K, pose)
    result = measure(n, *view, image, depth, K, f"measure_{n}_{time.strftime('%H%M%S')}")
    call("home")
    print("not found" if result is None else
          f"corner {n}: the cube's centre {result['forward_mm']:+.1f} mm forward, {result['left_mm']:+.1f} mm left"
          f" ({result['off_mm']:.1f} mm off; footprint {result['footprint_mm']} mm)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("open")
    h = sub.add_parser("hold")
    h.add_argument("width", type=float, nargs="?", default=30)
    sub.add_parser("plan")
    r = sub.add_parser("run")
    r.add_argument("n", type=int)
    m = sub.add_parser("measure")
    m.add_argument("n", type=int)
    sub.add_parser("home")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd != "home" and "error" in he.http("/arm/call", {"method": "read"}):
        print("engage:", call("engage"))   # limp since the bridge started: home first, 3 s
    if args.cmd == "open":
        print(call("open_gripper"))
    elif args.cmd == "hold":
        print(call("close_gripper", width_mm=args.width))
    elif args.cmd == "plan":
        cmd_plan()
    elif args.cmd == "run":
        cmd_run(args.n)
    elif args.cmd == "measure":
        cmd_measure(args.n)
    else:
        print(call("home"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
