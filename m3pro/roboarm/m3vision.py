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


SCORED = 3000   # points each RANSAC guess is scored on: the floor is most of the picture, a sample
                # tells the best guess as well as all ~19 000 (and is 6x quicker)


def floor_plane(depth_mm: np.ndarray, K, tolerance: float = 0.006, rounds: int = 120,
                seed: int = 0) -> tuple[np.ndarray, float, float]:
    """The floor under the camera: (unit normal pointing at the camera, the camera's height above
    it in m, the share of the points on it). The largest plane in view, by RANSAC, then a least-
    squares fit to its points -- what stands on the floor is left out. Raises ValueError when
    there is too little depth, or when the plane found does not face the camera from below."""
    p = points(depth_mm, K)
    if len(p) < 500:
        raise ValueError(f"only {len(p)} depth points between {NEAR} and {FAR} m")
    rng = np.random.default_rng(seed)
    sample = p[rng.choice(len(p), min(SCORED, len(p)), replace=False)]
    best, best_count = None, 0
    for _ in range(rounds):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        count = int((np.abs((sample - a) @ n) < tolerance).sum())
        if count > best_count:
            best, best_count = (a, n), count
    a, n = best
    inliers = np.abs((p - a) @ n) < tolerance   # the best guess's points among all of them
    best_count = int(inliers.sum())
    floor = p[inliers]
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
            min_points: int = 40, floor: tuple[np.ndarray, float] | None = None) -> list[dict]:
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
    (deg: 90 square, 180 a lying block), "whole" (bool), "points"}. `floor`: (normal, height) from
    floor_plane() when the caller has it already."""
    K = np.asarray(K, float).reshape(3, 3)
    rows, cols = depth_mm.shape[:2]
    normal, height = floor if floor is not None else floor_plane(depth_mm, K)[:2]
    p = points(depth_mm, K, step=1)
    above = height + p @ normal
    keep = (above > 0.006) & (above < 0.10)
    p, above = p[keep], above[keep]
    if len(p) < min_points:
        return []
    base = (camera[:3, :3] @ p.T).T + camera[:3, 3]
    cells = np.floor(base[:, :2] / cell).astype(int)
    cells -= cells.min(axis=0)
    grid = np.zeros(cells.max(axis=0)[::-1] + 1, np.uint8)   # the floor seen from above, a cell a pixel
    grid[cells[:, 1], cells[:, 0]] = 1
    _count, labels = cv2.connectedComponents(grid, connectivity=8)   # the touching cells: one object
    label = labels[cells[:, 1], cells[:, 0]]
    order = np.argsort(label, kind="stable")
    bounds = np.flatnonzero(np.diff(label[order])) + 1
    found = []
    for members in np.split(order, bounds):
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


def find_marker(image: np.ndarray, K, camera: np.ndarray, family: str, ident: int,
                side: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Where the printed marker `ident` of the OpenCV ArUco family `family` (e.g. "DICT_4X4_50"),
    `side` m square, is: (its centre, base m; its axes as the columns of a rotation -- x right, y
    up, z out of its face towards whoever reads it), in base through the levelled `camera`; None if
    not seen."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    family = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, family))
    if hasattr(cv2.aruco, "ArucoDetector"):   # OpenCV 4.7 on: this laptop
        corners, ids, _ = cv2.aruco.ArucoDetector(family, cv2.aruco.DetectorParameters()).detectMarkers(grey)
    else:                                     # the robot's 4.5.4
        corners, ids, _ = cv2.aruco.detectMarkers(grey, family, parameters=cv2.aruco.DetectorParameters_create())
    if ids is None or ident not in ids.ravel():
        return None
    seen = corners[list(ids.ravel()).index(ident)].reshape(4, 2)
    square = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], float) * side / 2
    ok, rvec, tvec = cv2.solvePnP(square, seen, np.asarray(K, float).reshape(3, 3), None,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    return (camera @ np.r_[tvec.ravel(), 1.0])[:3], camera[:3, :3] @ cv2.Rodrigues(rvec)[0]


TOPDOWN_MM = 0.25   # mm a pixel of the picture laid onto a top face


def colour_turn(image: np.ndarray, K, camera: np.ndarray, o: dict, inside: float = 0.004, outside: float = 0.0015,
                window: float = 15.0) -> tuple[float, float] | None:
    """The thing's turn from the colour picture: (yaw deg, how sure), or None. The picture is laid
    onto the plane of its top face as seen from straight above, blurred over a camera pixel (laid
    out 2-4x larger, its pixels' steps pulled the edges 4-5 deg towards the picture's own axes), and
    the brightness edges from `inside` within to `outside` beyond where the depth puts its outline
    (the depth's rectangle, its turn and size: what lies beyond -- a chessboard's edges -- has no
    vote) vote for their direction, modulo 90; the strongest within `window` deg of the depth's wins. The
    depth's outline of a 30 mm square is a few noisy pixels: its turn wandered 6-16 deg from one
    look to the next where the colour's held to 1-5 (2026-10-02, a clean-up's wide looks). "How
    sure": the winning direction's votes over the average direction's."""
    if not o["kind"]:
        return None
    K = np.asarray(K, float).reshape(3, 3)
    across, along = o["size"] / 2, o["length"] / 2 if o["length"] > o["size"] else o["size"] / 2
    half = max(across, along) * 1.6
    n = int(2 * half * 1000 / TOPDOWN_MM)
    rows, cols = np.mgrid[0:n, 0:n]
    dx = half - rows * TOPDOWN_MM / 1000          # up the patch: +x (forward)
    dy = half - cols * TOPDOWN_MM / 1000          # left in the patch: +y (left)
    world = np.stack([o["centre"][0] + dx, o["centre"][1] + dy, np.full_like(dx, o["top"]), np.ones_like(dx)], axis=-1)
    seen = world @ np.linalg.inv(camera).T
    if np.any(seen[..., 2] <= 0.05):
        return None
    u = K[0, 0] * seen[..., 0] / seen[..., 2] + K[0, 2]
    v = K[1, 1] * seen[..., 1] / seen[..., 2] + K[1, 2]
    if u.min() < 0 or v.min() < 0 or u.max() > image.shape[1] - 1 or v.max() > image.shape[0] - 1:
        return None   # the top face runs off the picture
    patch = cv2.remap(image, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)
    step = float(np.median(np.hypot(np.diff(u, axis=1), np.diff(v, axis=1))))   # camera pixels a patch pixel
    grey = cv2.GaussianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0),
                            max(1.5, 1.3 / max(step, 1e-6)))
    gx, gy = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    # the outline the depth found: `yaw` is the line across (the fingers'), `along` the other way
    a = np.radians(o["yaw"])
    p_across = dx * np.cos(a) + dy * np.sin(a)
    p_along = -dx * np.sin(a) + dy * np.cos(a)
    edge = np.maximum(np.abs(p_across) - across, np.abs(p_along) - along)   # 0 on the outline
    ring_mask = (edge > -inside) & (edge < outside)
    direction = np.degrees(np.arctan2(-gx, -gy)) % 90   # a gradient in base: rows go -x, columns -y
    votes = (np.hypot(gx, gy) * ring_mask).ravel() ** 2
    hist = np.bincount((direction.ravel() * 2).astype(int) % 180, weights=votes, minlength=180)
    smooth = np.convolve(np.r_[hist[-6:], hist, hist[:6]], np.ones(7) / 7, "same")[6:-6]
    near = [i for i in range(180) if abs((i / 2 - o["yaw"] + 45) % 90 - 45) <= window]
    best = max(near, key=lambda i: smooth[i])
    sure = float(smooth[best] / max(smooth.mean(), 1e-12))
    turn = o["yaw"] + ((best / 2 - o["yaw"] + 45) % 90 - 45)   # near the depth's: a block keeps its period
    return turn % o["period"], sure


def refine_turns(image: np.ndarray, K, camera: np.ndarray, found: list[dict], sure: float = 1.5) -> None:
    """Each whole kit thing's turn from the colour picture where it is sure (colour_turn); the
    depth's kept as "yaw_depth", and "turn_from" says which counts."""
    for o in found:
        o["yaw_depth"], o["turn_from"] = o["yaw"], "depth"
        if not (o["kind"] and o["whole"]):
            continue
        got = colour_turn(image, K, camera, o)
        if got is not None and got[1] >= sure:
            o["yaw"], o["turn_from"] = got[0], "colour"
