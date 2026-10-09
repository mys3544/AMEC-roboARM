"""Teach a thing by showing it (PLAN.md track A, A1-A2), and pick it by its name: the M3's jobs over
the teach model (tools/teach_service.py, YOLOE-26 visual prompts, 127.0.0.1:8762).

- teach(name): one thing stands alone in front of the robot. The wide look finds it in the depth;
  the arm then looks at it from views round it -- the wide look itself turned on J1 either side
  (find's own view of things, at a slant), and the few steep views the arm can take over it
  (m3scan.view: at the pick's 20 cm only ~82 deg) -- and in every view the depth boxes it: the box is the visual prompt; the
  views' embeddings, averaged, are what `name` is (A2: the user's choice of YOLOE, 2026-10-09; one
  view of one face named an apricot cube the news one in the next layout). Every view is kept
  (out/teach/NAME/) for training a detector on later (A4).
- find(): what each thing on the table is -- the wide look's things (the depth), each named by the
  taught thing whose detection covers it best, or unknown.
- pick_named(name): find, then the pick of the thing that is `name`.
"""

from __future__ import annotations

import base64
import json
import math
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

from roboarm import m3kinematics, m3scan, m3vision

SERVICE = "http://127.0.0.1:8762"
LOOK_TURNS = (-20, -10, 0, 10, 20)   # deg of J1 off the thing's own direction: the wide look turned
STEEP_TURNS = (-15, 0, 15)           # and the views from over it
STEEP = 82                           # deg the camera looks down from over it (at 20 cm the only height
                                     # the arm and the depth's 0.2 m allow, 2026-10-09)
SURE = 0.3                    # a detection's confidence that names a thing; below it, unknown
COVER = 0.3                   # how much a detection's box must overlap the thing's (IoU)


def ask(path: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    request = urllib.request.Request(SERVICE + path, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            answer = json.loads(reply.read())
    except urllib.error.HTTPError as exc:   # it answered, with what went wrong
        raise RuntimeError(f"the teach model: {exc.read().decode(errors='replace')[:300]}") from exc
    except OSError as exc:
        raise RuntimeError(f"the teach model does not answer ({exc}): is roboarm-m3teach running?") from exc
    if "error" in answer:
        raise RuntimeError(f"the teach model: {answer['error']}")
    return answer


def jpeg(image: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])[1]).decode()


def box_around(depth_mm: np.ndarray, K, cam: np.ndarray, centre, radius: float, margin: int = 6):
    """The pixel box [x1, y1, x2, y2] round what stands within `radius` m of `centre` (x, y, base)
    6 mm-0.3 m over the floor, seen through the levelled camera `cam`; None if under 80 points."""
    K = np.asarray(K, float)
    rows, cols = depth_mm.shape
    v, u = np.mgrid[0:rows, 0:cols]
    z = depth_mm.astype(np.float64) / 1000
    ok = (z > m3vision.NEAR) & (z < m3vision.FAR)
    p = np.stack([(u[ok] - K[0, 2]) * z[ok] / K[0, 0], (v[ok] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], axis=1)
    world = p @ cam[:3, :3].T + cam[:3, 3]
    up = world[:, 2] - m3kinematics.FLOOR
    there = (up > 0.006) & (up < 0.3) & (np.hypot(world[:, 0] - centre[0], world[:, 1] - centre[1]) < radius)
    if there.sum() < 80:
        return None
    us, vs = u[ok][there], v[ok][there]
    return [max(0, int(us.min()) - margin), max(0, int(vs.min()) - margin),
            min(cols - 1, int(us.max()) + margin), min(rows - 1, int(vs.max()) + margin)]


def in_view(pose: dict, point, K=None, inside: float = 0.85) -> bool:
    """Whether `point` (x, y, z, base) is in the camera's picture at `pose`, within `inside` of
    the way from its middle to its edge."""
    K = np.array([[480.8, 0, 323.8], [0, 480.9, 240.0], [0, 0, 1]]) if K is None else np.asarray(K, float)
    p = np.linalg.inv(m3scan.camera(pose)) @ np.r_[point[:3], 1.0]
    if p[2] <= 0.1:
        return False
    u, v = K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]
    return abs(u - K[0, 2]) < inside * K[0, 2] and abs(v - K[1, 2]) < inside * K[1, 2]


def views_of(centre, radius: float, top: float) -> list[dict]:
    """The arm's poses that look at a thing at `centre` (x, y): the wide look turned LOOK_TURNS off
    it, where the thing stays in the picture, then the views over it at STEEP the arm can take."""
    base = np.asarray(m3kinematics.CHAIN[0][0][:2])
    j1 = 90 - math.degrees(math.atan2(centre[1] - base[1], centre[0] - base[0]))
    middle = np.r_[centre[:2], top / 2]
    poses = []
    for turn in LOOK_TURNS:
        # the fingers shut: they keep what they had, and open wide from home they came down over the
        # chassis with the arm turned (the guard, J1 67)
        pose = {**m3scan.LOOK_AT, 1: round(min(180, max(0, j1 + turn))), 6: m3scan.SLIM}
        if in_view(pose, middle) and m3kinematics.problem(pose) is None and pose not in poses:
            poses.append(pose)
    nearest = m3scan.NEAREST + math.hypot(radius, top / 2)
    for turn in STEEP_TURNS:
        pose = m3scan.view(j1 + turn, middle, STEEP, nearest + 0.02, radius, top)             or m3scan.view(j1 + turn, middle, STEEP, nearest, radius, top)
        if pose is not None:
            poses.append(pose)
    return poses


def the_thing(p) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """The wide look's nearest thing standing on the table: (it, the picture, the depth, K)."""
    p.move(m3scan.LOOK_AT, speed=p.dps("free_dps"))
    time.sleep(p.settle_s if p.settle_s is not None else p.cfg["settle_wide_s"])
    image, depth, K, pose = p.snapshot()
    cam = m3scan.levelled(depth, K, pose)
    normal, height, _ = m3vision.floor_plane(depth, K)
    found = m3vision.objects(depth, K, cam, floor=(normal, height))
    if not found:
        raise RuntimeError("nothing standing in front of the robot to look at")
    return found[0], image, depth, np.asarray(K, float)


def teach(p, name: str) -> None:
    """Learn the thing standing alone in front of the robot as `name` (see the module)."""
    name = str(name).strip()
    if not name:
        raise RuntimeError("a taught thing needs a name")
    p.engaged()
    with p.step("look.teach"):
        thing, *_ = the_thing(p)
    w, h = thing["footprint"]
    radius = 0.5 * math.hypot(w, h) + 0.008
    centre = thing["centre"][:2]
    poses = views_of(centre, radius, thing["top"])
    p.say(f"to teach '{name}': {100 * thing['top']:.0f} cm tall, {100 * max(w, h):.0f} cm long, "
          f"at x {centre[0]:.3f} y {centre[1]:+.3f}; {len(poses)} views")
    folder = None if p.out is None else p.out.parent / "teach" / name.replace("/", "_")
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    embeddings = []
    for i, pose in enumerate(poses):
        p.check()
        with p.step("look.teach"):
            p.move(pose, speed=p.dps("free_dps"))
            time.sleep(p.settle_s if p.settle_s is not None else p.cfg["settle_close_s"])
            image, depth, K, joints = p.snapshot()
            box = box_around(depth, K, m3scan.levelled(depth, K, joints), centre, radius)
        if box is None:
            continue
        with p.step("teach.embed"):
            embeddings.append(ask("/embed", {"jpeg": jpeg(image), "boxes": [box]})["embeddings"][0])
        if folder is not None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(folder / f"{stamp}_{i}_rgb.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(folder / f"{stamp}_{i}_depth.png"), depth)
            (folder / f"{stamp}_{i}.json").write_text(json.dumps({"name": name, "box": box, "K": np.asarray(K).tolist(),
                                                                  "pose": {str(j): v for j, v in joints.items()}}))
    if not embeddings:
        raise RuntimeError(f"'{name}' was not seen in any view")
    mean = np.mean(np.asarray(embeddings), axis=0)
    mean = mean / np.linalg.norm(mean)
    spread = float(np.mean(np.asarray(embeddings) @ mean))
    names = ask("/teach", {"name": name, "embedding": mean.tolist()})["names"]
    p.home()
    p.say(f"taught '{name}' from {len(embeddings)} views (they agree {spread:.2f}); known: {', '.join(names)}")


def overlap(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / union if union > 0 else 0.0


def name_things(found: list[dict], detections: list[dict]) -> list[dict]:
    """Each thing the depth found, with "taught" (the name whose detection covers it best, at least
    COVER and SURE) and "sure" (that detection's confidence); None and 0 when none does."""
    out = []
    for o in found:
        pts = np.asarray(o["outline_px"], float)
        box = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
        covering = [d for d in detections if d["conf"] >= SURE and overlap(d["box"], box) >= COVER]
        best = max(covering, key=lambda d: d["conf"], default=None)
        out.append({**o, "taught": best["name"] if best else None, "sure": best["conf"] if best else 0.0})
    return out


def find(p) -> list[dict]:
    """What each thing on the table in front is, by the taught names (said, and returned)."""
    p.engaged()
    with p.step("look.find"):
        p.move(m3scan.LOOK_AT, speed=p.dps("free_dps"))
        time.sleep(p.settle_s if p.settle_s is not None else p.cfg["settle_wide_s"])
        image, depth, K, pose = p.snapshot()
        cam = m3scan.levelled(depth, K, pose)
        normal, height, _ = m3vision.floor_plane(depth, K)
        found = m3vision.objects(depth, K, cam, floor=(normal, height))
    with p.step("teach.detect"):
        named = name_things(found, ask("/detect", {"jpeg": jpeg(image), "conf": 0.1})["detections"])
    p.say("found: " + ("; ".join(f"{o['taught'] or 'unknown'} ({o['sure']:.2f}) at x {o['centre'][0]:.3f} "
                                 f"y {o['centre'][1]:+.3f}" for o in named) or "nothing"))
    return named


def pick_named(p, name: str) -> None:
    """Find the thing taught as `name` and pick it (the surest, when several are)."""
    named = [o for o in find(p) if o["taught"] == name]
    if not named:
        raise RuntimeError(f"no '{name}' in view")
    target = max(named, key=lambda o: o["sure"])
    p.say(f"picking '{name}' ({target['sure']:.2f}) at x {target['centre'][0]:.3f} y {target['centre'][1]:+.3f}")
    p.pick_once(at=[float(target["centre"][0]), float(target["centre"][1])])
