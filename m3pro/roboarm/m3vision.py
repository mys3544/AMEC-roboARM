"""What the M3's depth camera (the DCW2, depth aligned to colour) measures for the arm: the floor's
plane under the camera, which fixes the camera's height, tilt and roll exactly whatever the arm got
wrong (roboarm/m3kinematics.py SAG: ~1-2 cm, ~1-2 deg), and the things standing on that floor.

Depth images are 16-bit millimetres in the colour camera's pixels; K is the colour camera's matrix.
The camera frame is the optical one: x right, y down, z ahead. No ROS here.
"""

from __future__ import annotations

import cv2
import numpy as np

NEAR, FAR = 0.15, 1.2   # m: the DCW2 answers from ~0.19 (a cut at 0.2 took the near half of a cube's
                        # top face at the pick's close look, 2026-10-01); past ~1.2 the floor is too
                        # oblique to trust


def points(depth_mm: np.ndarray, K, step: int = 4) -> np.ndarray:
    """The depth image's points in the camera frame (m), every `step`-th pixel, NEAR..FAR only."""
    K = np.asarray(K, float).reshape(3, 3)
    v, u = np.mgrid[0:depth_mm.shape[0]:step, 0:depth_mm.shape[1]:step]
    z = depth_mm[v, u].astype(float) / 1000
    ok = (z > NEAR) & (z < FAR)
    u, v, z = u[ok], v[ok], z[ok]
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)


def floor_plane(depth_mm: np.ndarray, K, tolerance: float = 0.006, rounds: int = 300,
                seed: int = 0) -> tuple[np.ndarray, float, float]:
    """The floor under the camera: (unit normal pointing at the camera, the camera's height above
    it in m, the share of the points on it). The largest plane in view, by RANSAC, then a least-
    squares fit to its points -- what stands on the floor is left out. Raises ValueError when
    there is too little depth, or when the plane found does not face the camera from below."""
    p = points(depth_mm, K)
    if len(p) < 500:
        raise ValueError(f"only {len(p)} depth points between {NEAR} and {FAR} m")
    rng = np.random.default_rng(seed)
    best, best_count = None, 0
    for _ in range(rounds):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inliers = np.abs((p - a) @ n) < tolerance
        if inliers.sum() > best_count:
            best, best_count = inliers, int(inliers.sum())
    floor = p[best]
    centre = floor.mean(axis=0)
    normal = np.linalg.svd(floor - centre, full_matrices=False)[2][2]
    if normal @ centre > 0:   # point it at the camera, which sits at the origin
        normal = -normal
    height = float(-normal @ centre)
    if normal[1] > -0.1:   # the floor's normal must point up, i.e. against the image's y (down)
        raise ValueError("the largest plane in view is not a floor below the camera")
    return normal, height, best_count / len(p)


def level(model: np.ndarray, normal: np.ndarray, height: float, floor_z: float = 0.0) -> np.ndarray:
    """base_link <- camera with its height, tilt and roll taken from the floor it sees (`normal`
    in the camera frame, pointing at it; `height` above it) -- the floor level, at `floor_z` --
    and its place along the floor and its heading from the arm `model`, which a plane cannot
    tell. The look-again step's first correction."""
    seen_up = model[:3, :3] @ np.asarray(normal, float)   # where the model says the floor's normal points
    axis = np.cross(seen_up, [0.0, 0.0, 1.0])
    s, c = float(np.linalg.norm(axis)), float(seen_up[2])
    out = np.array(model, float)
    if s > 1e-9:
        out[:3, :3] = cv2.Rodrigues(axis / s * np.arctan2(s, c))[0] @ out[:3, :3]
    out[2, 3] = floor_z + height
    return out


# The kit (plan v2.2): (kind, width across, length, height) in m -- the cubes, and the 30 x 30 x 60
# block (two cubes glued) lying and standing. The fingers close across `width`.
KIT = (("cube", 0.030, 0.030, 0.030), ("cube", 0.040, 0.040, 0.040),
       ("block", 0.030, 0.060, 0.030), ("block", 0.030, 0.030, 0.060))


def kit_item(top: float, long: float):
    """The kit item a footprint `long` m at its longest and `top` m tall is, or None: the nearest
    height within 7 mm, no longer than the item (+12 mm -- seen partly a thing looks smaller,
    never bigger: 2026-10-01, 150 x 46 mm and 35 tall was not a cube)."""
    fits = [item for item in KIT if abs(item[3] - top) <= 0.007 and long <= item[2] + 0.012]
    return min(fits, key=lambda item: (abs(item[3] - top), item[2])) if fits else None


def objects(depth_mm: np.ndarray, K, camera: np.ndarray, cell: float = 0.004,
            min_points: int = 40) -> list[dict]:
    """The things standing on the floor, from one depth image and the camera levelled on that
    floor (level()): each the depth's points 6-100 mm over the floor, grouped by where they stand
    (cells of `cell` m that touch), dropped onto the floor -- every face of a box standing there
    lies over its footprint -- and the smallest rectangle round them. Its top (the 95th percentile
    of the heights) and its longest side name the kit item (kit_item(): a cube, a block lying or
    standing), else None. A side shorter than the item's is grown to it on the camera's side: a
    thing cut by the picture's edge is seen only in part. Such a rectangle's turn is the edge's,
    not the thing's (2026-10-01: 0.7 deg for a cube turned 45) -- "whole" says it was seen whole,
    its turn worth using. Nearest the camera first; each {"centre" (base, m, at half height),
    "kind" ("cube", "block" or None), "size" (m the fingers close across, or None), "length",
    "height" (m), "top" (m, measured), "footprint" ((w, h) m as seen), "yaw" (deg: a line the
    fingers may close along -- any face of a square footprint, across a lying block), "period"
    (deg: 90 square, 180 a lying block), "whole" (bool), "points"}."""
    K = np.asarray(K, float).reshape(3, 3)
    rows, cols = depth_mm.shape[:2]
    normal, height, _ = floor_plane(depth_mm, K)
    p = points(depth_mm, K, step=1)
    above = height + p @ normal
    keep = (above > 0.006) & (above < 0.10)
    p, above = p[keep], above[keep]
    if len(p) < min_points:
        return []
    base = (camera[:3, :3] @ p.T).T + camera[:3, 3]
    cells = np.floor(base[:, :2] / cell).astype(int)
    index: dict[tuple, list[int]] = {}
    for i, c in enumerate(map(tuple, cells)):
        index.setdefault(c, []).append(i)
    found, seen = [], set()
    for start in index:
        if start in seen:
            continue
        todo, members = [start], []
        seen.add(start)
        while todo:   # the touching cells: one object
            c = todo.pop()
            members += index[c]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if n in index and n not in seen:
                        seen.add(n)
                        todo.append(n)
        if len(members) < min_points:
            continue
        xy = base[members, :2].astype(np.float32)
        top = float(np.percentile(above[members], 95))
        (cx, cy), (w, h), angle = cv2.minAreaRect(xy)   # w lies along `angle`, h across it
        item = kit_item(top, max(w, h))
        kind, size, length, tall = item if item else (None, None, None, top)
        centre = np.array([cx, cy])
        mine = p[members]
        u, v = K[0, 0] * mine[:, 0] / mine[:, 2] + K[0, 2], K[1, 1] * mine[:, 1] / mine[:, 2] + K[1, 2]
        clipped = bool(np.any((u < 3) | (u > cols - 4) | (v < 3) | (v > rows - 4)))
        lying = item is not None and length > size
        long_deg = angle if w >= h else angle + 90             # the footprint's long side
        whole = False
        if item:
            want = {True: length, False: size}                 # each side's length, the longer one's first
            whole = not clipped and all(side >= want[side == max(w, h)] - 0.006 for side in (w, h))
            to_camera = camera[:2, 3] - centre
            for side, a in ((w, np.radians(angle)), (h, np.radians(angle) + np.pi / 2)):
                along = np.array([np.cos(a), np.sin(a)])
                need = want[lying and side == max(w, h)]
                if side < need:
                    centre = centre + np.sign(to_camera @ along) * (need - side) / 2 * along
        found.append({"centre": np.array([centre[0], centre[1], tall / 2]), "kind": kind, "size": size,
                      "length": length, "height": tall, "top": top, "footprint": (float(w), float(h)),
                      "yaw": float(long_deg + 90) % 180 if lying else float(angle) % 90,
                      "period": 180 if lying else 90, "whole": whole, "points": len(members)})
    return sorted(found, key=lambda o: float(np.linalg.norm(o["centre"][:2] - camera[:2, 3])))
