"""Check a saved arena map against the fence: four walls, 3.2 x 2.8 m, square corners.

    python3 fence_fit.py MAP.yaml [--png MAP_fit.png]

Reads map_saver's trinary map (PGM: 0 occupied, 254 free, 205 unknown) and prints, as JSON,
the fitted fence -- its corners, side lengths, corner angles, how straight each wall is and how
much of it the lidar has seen -- and the blobs standing inside it (the landmarks, or anything
else) in the arena frame: origin at a fence corner, x along a long wall, y into the arena.
Runs on the robot (numpy and OpenCV from JetPack) and in the project venv.
"""

import argparse
import json
import pathlib

import cv2
import numpy as np

NOMINAL = (3.2, 2.8)   # the fence, m
GATE = 0.10            # a point belongs to a wall when it is this close to it, m
CLEAR = 0.06           # a blob point stands at least this far inside the walls, m


def load(yaml_path):
    """map_saver's YAML is flat `key: value` lines; its image name is relative to the YAML."""
    yaml_path = pathlib.Path(yaml_path)
    meta = {}
    for line in yaml_path.read_text().splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    image = pathlib.Path(meta["image"])
    if not image.is_absolute():
        image = yaml_path.parent / image
    grid = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if grid is None:
        raise SystemExit(f"cannot read {image}")
    return grid, float(meta["resolution"]), json.loads(meta["origin"])[:2]


def cells_to_xy(rows, cols, shape, res, origin):
    """Cell centres in the map frame (row 0 is the top of the image, the largest y)."""
    return np.stack([origin[0] + (cols + 0.5) * res,
                     origin[1] + (shape[0] - rows - 0.5) * res], axis=1)


def rot90(u):
    return np.array([-u[1], u[0]])


def near_wall(points, a, b, margin):
    """The points within GATE of segment a-b, leaving `margin` clear of both ends."""
    length = np.linalg.norm(b - a)
    u = (b - a) / length
    rel = points - a
    along, across = rel @ u, np.abs(rel @ rot90(u))
    return points[(across < GATE) & (along > margin) & (along < length - margin)]


def fit_line(points, a, b):
    """A Huber line through the wall's points; the box side itself when too few were seen."""
    if len(points) < 10:
        return a, (b - a) / np.linalg.norm(b - a)
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0], float), np.array([vx, vy], float)


def intersect(p1, d1, p2, d2):
    t, _s = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
    return p1 + t * d1


def wall_direction(points, res):
    """The angle whose two axes pile the points into the sharpest peaks: the walls' direction.
    Works while the far walls are still a few dots (a map from one spot)."""
    best_score, best = -1.0, 0.0
    for angle in np.radians(np.arange(0, 90, 0.25)):
        u = np.array([np.cos(angle), np.sin(angle)])
        score = 0.0
        for axis in (u, rot90(u)):
            proj = points @ axis
            score += float((np.bincount(((proj - proj.min()) / res).astype(int)) ** 2).sum())
        if score > best_score:
            best_score, best = score, angle
    return np.array([np.cos(best), np.sin(best)])


def wall_pair(proj, res):
    """Along one axis: the two walls, as the pair of dense bins whose spacing is nearest a side
    of the fence (within 15 cm) -- so a landmark or the other robot is never taken for a wall.
    None when no such pair is in the map yet."""
    lo = proj.min()
    counts = np.convolve(np.bincount(((proj - lo) / res).astype(int)), np.ones(3), "same")
    best = None
    for side in NOMINAL:
        for step in range(round((side - 0.15) / res), round((side + 0.15) / res) + 1):
            if step >= len(counts):
                break
            both = np.minimum(counts[:-step], counts[step:])
            i = int(np.argmax(counts[:-step] + counts[step:] - 1e6 * (both < 3)))
            score = counts[i] + counts[i + step]
            if both[i] >= 3 and (best is None or score > best[0]):
                best = (score, lo + (i + 0.5) * res, lo + (i + step + 0.5) * res)
    return None if best is None else (best[1], best[2])


def one_wall(proj, free_proj, res, side):
    """Along an axis where only one wall has been seen (a map from one spot: far walls are hit
    at a grazing angle and the rays next to them wipe the hits out): the densest bin is the
    wall, and the other one is put `side` away from it, on the side where the free space is."""
    lo = proj.min()
    counts = np.convolve(np.bincount(((proj - lo) / res).astype(int)), np.ones(3), "same")
    wall = lo + (int(np.argmax(counts)) + 0.5) * res
    other = wall + side * np.sign(free_proj.mean() - wall)
    return min(wall, other), max(wall, other)


def fit(grid, res, origin):
    occupied = (grid < 50).astype(np.uint8)
    rows, cols = np.nonzero(occupied)
    if len(rows) < 20:
        raise SystemExit("the map has almost nothing occupied yet")
    points = cells_to_xy(rows, cols, grid.shape, res, origin)

    # A first box from the walls' direction and the two facing pairs, then each wall refitted
    # from its own points (the lidar cannot see over the 30 cm fence, so every point is the fence
    # or something standing inside it).
    u = wall_direction(points, res)
    v = rot90(u)
    pairs = [wall_pair(points @ u, res), wall_pair(points @ v, res)]
    placed = None   # the axis whose second wall is not in the map yet
    unmeasured = None   # then the side, 0 long or 1 short, that is nominal, not measured
    if pairs[0] is None and pairs[1] is None:
        raise SystemExit("no two facing walls in the map yet: drive where the lidars see them")
    if None in pairs:
        placed = pairs.index(None)
        seen = pairs[1 - placed]
        near_long = abs(seen[1] - seen[0] - NOMINAL[0]) < abs(seen[1] - seen[0] - NOMINAL[1])
        rows_f, cols_f = np.nonzero(grid > 250)
        free = cells_to_xy(rows_f, cols_f, grid.shape, res, origin)
        axis = (u, v)[placed]
        pairs[placed] = one_wall(points @ axis, free @ axis, res, NOMINAL[1 if near_long else 0])
        unmeasured = 1 if near_long else 0
    (a1, a2), (b1, b2) = pairs
    corners = np.array([a1 * u + b1 * v, a2 * u + b1 * v, a2 * u + b2 * v, a1 * u + b2 * v])
    fence_points = points
    for _ in range(3):
        walls = [near_wall(fence_points, corners[i], corners[(i + 1) % 4], GATE) for i in range(4)]
        lines = [fit_line(w, corners[i], corners[(i + 1) % 4]) for i, w in enumerate(walls)]
        corners = np.array([intersect(*lines[i - 1], *lines[i]) for i in range(4)])

    wall_stats = []
    for i, (p0, d) in enumerate(lines):
        a, b = corners[i], corners[(i + 1) % 4]
        length = float(np.linalg.norm(b - a))
        pts = near_wall(fence_points, a, b, GATE)
        across = (pts - p0) @ rot90(d) if len(pts) else np.zeros(1)
        # seen: the share of 2-cell stretches along the wall with at least one point in them
        along = (pts - a) @ ((b - a) / length)
        stretches = max(1, int((length - 2 * GATE) / (2 * res)))
        seen = np.isin(np.arange(stretches), np.floor((along - GATE) / (2 * res))).mean()
        # thick: how far apart the scans put the wall (5th to 95th percentile across it) -- the
        # direct measure of how well the scans were registered
        wall_stats.append({"length_m": round(length, 3),
                           "rms_cm": round(float(np.sqrt(np.mean(across ** 2))) * 100, 2),
                           "thick_cm": round(float(np.ptp(np.percentile(across, [5, 95]))) * 100, 1),
                           "seen": round(float(seen), 3)})
    angles = [round(float(np.degrees(np.arccos(abs(lines[i - 1][1] @ lines[i][1])))), 2)
              for i in range(4)]

    # Arena frame: x along the long walls, pointing +x on the map (so it stays put as the map
    # grows), y to its left, origin at the corner that puts the arena at +x, +y. Which of the
    # two such corners is meant is the map's orientation until the landmarks name one.
    lengths = [w["length_m"] for w in wall_stats]
    first = 0 if lengths[0] + lengths[2] >= lengths[1] + lengths[3] else 1
    centre = corners.mean(axis=0)
    ex = (corners[first + 1] - corners[first]) / np.linalg.norm(corners[first + 1] - corners[first])
    if ex[0] < 0:
        ex = -ex
    ey = rot90(ex)
    o = corners[np.argmin((corners - centre) @ (ex + ey))]

    # Blobs: occupied cells well inside the walls, grouped.
    poly = corners.astype(np.float32).reshape(-1, 1, 2)
    inside = np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), True) > CLEAR
                       for x, y in points])
    mask = np.zeros_like(occupied)
    mask[rows[inside], cols[inside]] = 1
    j = max(1, round(0.03 / res))
    n_labels, blob_labels = cv2.connectedComponents(
        cv2.dilate(mask, np.ones((2 * j + 1,) * 2, np.uint8)))
    blobs = []
    for label in range(1, n_labels):
        pick = inside & (blob_labels[rows, cols] == label)
        if pick.sum() < 3:
            continue
        pts = points[pick]
        local = (pts - o) @ np.array([ex, ey]).T
        extent = local.max(axis=0) - local.min(axis=0) + res
        mid = pts.mean(axis=0)
        blobs.append({"arena_xy_m": [round(float(v), 3) for v in (mid - o) @ np.array([ex, ey]).T],
                      "map_xy_m": [round(float(v), 3) for v in mid],
                      "size_cm": [round(float(v) * 100, 1) for v in extent],
                      "cells": int(pick.sum())})
    blobs.sort(key=lambda b: -b["cells"])

    long_side = (lengths[first] + lengths[first + 2]) / 2
    short_side = (lengths[1 - first] + lengths[3 - first]) / 2
    error = [round((long_side - NOMINAL[0]) * 100, 1), round((short_side - NOMINAL[1]) * 100, 1)]
    if unmeasured is not None:
        error[unmeasured] = None
    return {
        "resolution_m": res,
        "fence": {
            "size_m": [round(long_side, 3), round(short_side, 3)],
            "nominal_m": list(NOMINAL),
            "error_cm": error,   # None: that side's second wall is not in the map yet
            "complete": unmeasured is None,
            "corner_angles_deg": angles,
            "walls": wall_stats,
            "corners_map": [[round(float(v), 3) for v in c] for c in corners],
        },
        "arena_frame": {"origin_map": [round(float(v), 3) for v in o],
                        "x_axis_map": [round(float(v), 4) for v in ex]},
        "blobs": blobs,
    }


def render(grid, res, origin, result, path=None):
    """The map cropped to the fence, the fitted walls in green, blobs in red, the arena axes;
    the whole map when there is no fit (`result` None). Returns the image and the function that
    takes a map-frame point to its pixel, for drawing more on it."""

    def to_px(xy):   # map frame -> fractional (col, row) of the full grid
        return np.array([(xy[0] - origin[0]) / res, grid.shape[0] - (xy[1] - origin[1]) / res])

    if result is None:
        c0, r0, (c1, r1) = 0, 0, grid.shape[::-1]
    else:
        px = np.array([to_px(c) for c in result["fence"]["corners_map"]])
        pad = 0.3 / res
        c0, r0 = np.maximum(np.floor(px.min(axis=0) - pad), 0).astype(int)
        c1, r1 = np.minimum(np.ceil(px.max(axis=0) + pad), grid.shape[::-1]).astype(int)
    lut = np.full(256, 215, np.uint8)
    lut[:50], lut[250:] = 0, 255
    crop = lut[grid[r0:r1, c0:c1]]
    scale = int(np.clip(900 // max(crop.shape), 2, 8))
    img = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    band = 62
    img = cv2.copyMakeBorder(img, band, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    def P(xy):
        c, r = to_px(xy)
        # np.rint, not round(): numpy 1.26 on the robot rounds a float64 to a float64
        return int(np.rint((c - c0) * scale)), int(np.rint((r - r0) * scale)) + band

    font = cv2.FONT_HERSHEY_SIMPLEX
    if result is None:
        cv2.putText(img, "no fence fit yet: two facing walls are not in the map",
                    (8, 18), font, 0.45, (0, 0, 200), 1)
        return finish(img, P, path)
    corners = np.array(result["fence"]["corners_map"])
    cv2.polylines(img, [np.array([P(c) for c in corners], np.int32)], True, (0, 160, 0), 2)
    centre = corners.mean(axis=0)
    for i, wall in enumerate(result["fence"]["walls"]):
        mid = (corners[i] + corners[(i + 1) % 4]) / 2
        at = np.array(P(mid), float)
        out = np.array(P(mid + (mid - centre) / np.linalg.norm(mid - centre) * 0.1)) - at
        out /= np.linalg.norm(out)
        text = f"{wall['length_m']:.3f} m"
        (tw, th), _base = cv2.getTextSize(text, font, 0.45, 1)
        x, y = at + out * (abs(out[0]) * tw / 2 + abs(out[1]) * th / 2 + 6)
        cv2.putText(img, text, (int(x - tw / 2), int(y + th / 2)), font, 0.45, (0, 120, 0), 1)
    o = np.array(result["arena_frame"]["origin_map"])
    ex = np.array(result["arena_frame"]["x_axis_map"])
    cv2.arrowedLine(img, P(o), P(o + 0.4 * ex), (0, 0, 220), 2, tipLength=0.2)
    cv2.arrowedLine(img, P(o), P(o + 0.4 * rot90(ex)), (220, 0, 0), 2, tipLength=0.2)
    for n, blob in enumerate(result["blobs"], 1):
        x, y = P(blob["map_xy_m"])
        radius = int(max(blob["size_cm"]) / 100 / res * scale / 2) + 6
        cv2.circle(img, (x, y), radius, (0, 0, 230), 2)
        a, b = blob["arena_xy_m"]
        cv2.putText(img, f"{n}: ({a:.2f}, {b:.2f})", (x + radius + 2, y + 4), font, 0.4,
                    (0, 0, 200), 1)
    f = result["fence"]
    seen = min(w["seen"] for w in f["walls"])
    rms = max(w["rms_cm"] for w in f["walls"])
    cv2.putText(img, f"fence {f['size_m'][0]:.3f} x {f['size_m'][1]:.3f} m (nominal 3.2 x 2.8), "
                f"corners {', '.join(f'{a:.1f}' for a in f['corner_angles_deg'])} deg",
                (8, 18), font, 0.45, (0, 0, 0), 1)
    if not f["complete"]:
        cv2.putText(img, "one wall is not in the map yet: drawn at the nominal distance",
                    (8, 54), font, 0.45, (0, 0, 200), 1)
    cv2.putText(img, f"walls: worst rms {rms:.1f} cm, least seen {seen * 100:.0f} %;  "
                f"{len(result['blobs'])} blobs inside;  {res * 100:.0f} cm cells",
                (8, 36), font, 0.45, (0, 0, 0), 1)
    return finish(img, P, path)


def finish(img, P, path):
    if path:
        cv2.imwrite(str(path), img)
    return img, P


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("yaml")
    parser.add_argument("--png", help="write the map with the fit drawn on it")
    args = parser.parse_args()
    grid, res, origin = load(args.yaml)
    result = fit(grid, res, origin)
    if args.png:
        render(grid, res, origin, result, args.png)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
