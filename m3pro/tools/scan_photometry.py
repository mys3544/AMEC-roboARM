"""How much an object scan's views disagree in brightness and colour, read off the floor they all see
(NEXT.md, "The scan's colours": the question PPISP answers, 2026-10-27).

Every view of a scan's splat export (m3_scan.py fuse --export: transforms.json, depths/, images/) puts its
floor pixels into the world through its depth and corrected camera; the floor is binned into 1 cm cells
round the thing (3-25 cm past its radius, 6 mm of the floor's height, none saturated). A matte floor under
fixed light looks the same from every view, so what differs is the camera's: log(linear colour) = the cell's
own + a gain for the view, a channel each, fitted by alternating means over the cells three views or more saw.
Printed: the views' exposure (gains' range), their colour (R/G, B/G), the vignetting (what is left by the
distance from the picture's middle), and the same cell's spread across views raw and with the gains -- on
plain floor and on the print and tape edges, where what is left is the views landing mm apart, not colour.

2026-10-27, the two saved scans: exposure x0.89-1.20 (box 20261012_172202) and x0.93-1.14 (can 165048),
B/G 0.89-1.18 and 0.90-1.08, the corners 9 and 7 % darker; the spread 8.2 -> 6.9 % and 10.1 -> 9.2 % with
the gains, plain floor 4.5 / 8.3 %, printed 8.9 / 10.2 %. A scan with the exposure locked should show it.

    python tools/scan_photometry.py out/scan/STAMP/splat [out/scan/STAMP2/splat ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CELL = 0.01            # floor cells, m
FLOOR_BAND = 0.006     # a floor point is within this of the floor's height, m
RING = (0.03, 0.25)    # the floor used: this far past the thing's radius, m
MIN_PIXELS = 5         # pixels a view must put in a cell to count
MIN_VIEWS = 3          # views a cell must be seen in to count


def linear(srgb: np.ndarray) -> np.ndarray:
    c = srgb / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def floor_readings(folder: Path) -> tuple[np.ndarray, int]:
    """One row a view and cell: view, cell id, distance from the picture's middle (1 = its corner), the
    texture inside the cell (sd of log luminance) and the median log linear R, G, B."""
    t = json.loads((folder / "transforms.json").read_text())
    centre, radius = np.array(t["centre"]), t["radius"]
    views = []
    for f in t["frames"]:
        img = cv2.imread(str(folder / f["file"]))[:, :, ::-1].astype(np.float64)
        depth = cv2.imread(str(folder / f["depth"]), cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
        K, c2w = np.array(f["K"]), np.array(f["c2w"])
        v, u = np.nonzero(depth > 0)
        z = depth[v, u]
        cam = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
        views.append((img, v, u, (c2w[:3, :3] @ cam + c2w[:3, 3:4]).T))
    floor = np.median(np.concatenate([p[:, 2] for *_, p in views]))
    rows = []
    for i, (img, v, u, p) in enumerate(views):
        out = np.linalg.norm(p[:, :2] - centre, axis=1)
        m = (np.abs(p[:, 2] - floor) < FLOOR_BAND) & (out > radius + RING[0]) & (out < radius + RING[1])
        colour = img[v[m], u[m]]
        ok = (colour.max(1) < 250) & (colour.min(1) > 5)
        cells = np.floor(p[m][ok, :2] / CELL).astype(np.int64)
        key = cells[:, 0] * 100000 + cells[:, 1]
        logc = np.log(linear(colour[ok]) + 1e-4)
        rad = np.hypot((u[m][ok] - img.shape[1] / 2) / 400.0, (v[m][ok] - img.shape[0] / 2) / 400.0)
        order = np.argsort(key)
        key, logc, rad = key[order], logc[order], rad[order]
        uniq, start = np.unique(key, return_index=True)
        for k, s, e in zip(uniq, start, list(start[1:]) + [len(key)]):
            if e - s >= MIN_PIXELS:
                rows.append((i, k, np.median(rad[s:e]), logc[s:e].mean(1).std(), *np.median(logc[s:e], axis=0)))
    return np.array(rows), len(views)


def report(folder: Path) -> None:
    rows, n = floor_readings(folder)
    view, cell, rad, tex, val = rows[:, 0].astype(int), rows[:, 1].astype(np.int64), rows[:, 2], rows[:, 3], rows[:, 4:]
    _, ci = np.unique(cell, return_inverse=True)
    keep = np.bincount(ci)[ci] >= MIN_VIEWS
    view, rad, tex, val = view[keep], rad[keep], tex[keep], val[keep]
    _, ci = np.unique(ci[keep], return_inverse=True)
    cells = ci.max() + 1
    seen = np.maximum(np.bincount(ci, minlength=cells), 1)[:, None]
    per_view = np.maximum(np.bincount(view, minlength=n), 1)
    gain = np.zeros((n, 3))
    for _ in range(50):
        own = np.stack([np.bincount(ci, val[:, c] - gain[view, c], minlength=cells) for c in range(3)], 1) / seen
        gain = np.stack([np.bincount(view, val[:, c] - own[ci, c], minlength=n) / per_view for c in range(3)], 1)
        gain -= gain.mean(0)
    raw = (val - (np.stack([np.bincount(ci, val[:, c], minlength=cells) for c in range(3)], 1) / seen)[ci]).mean(1)
    left = (val - gain[view] - own[ci]).mean(1)

    def spread(res: np.ndarray, which: np.ndarray) -> float:
        return float(np.median([res[ci == c].std() for c in np.unique(ci[which]) if (ci == c).sum() >= MIN_VIEWS]))

    lum, rg, bg = gain.mean(1), gain[:, 0] - gain[:, 1], gain[:, 2] - gain[:, 1]
    textured = (np.bincount(ci, tex) / seen[:, 0])[ci] >= np.percentile(np.bincount(ci, tex) / seen[:, 0], 40)
    every = np.ones(len(ci), bool)
    edges = [0, .2, .4, .6, .8, 1.01]
    print(f"== {folder.parent.name}: {n} views, {cells} floor cells seen by {MIN_VIEWS}+ views")
    print(f"  exposure view to view: x{np.exp(lum.min()):.2f} .. x{np.exp(lum.max()):.2f} (sd {lum.std() * 100:.0f} %)")
    print(f"  colour: R/G {np.exp(rg.min()):.2f} .. {np.exp(rg.max()):.2f}, B/G {np.exp(bg.min()):.2f} .. {np.exp(bg.max()):.2f}")
    print("  vignetting (left over, by distance from the middle; 1 = corner): "
          + "  ".join(f"{edges[i]:.1f}-{min(edges[i + 1], 1):.1f} {np.median(left[(rad >= edges[i]) & (rad < edges[i + 1])]) * 100:+.1f} %"
                      for i in range(5)))
    print(f"  the same cell across views: {spread(raw, every) * 100:.1f} % raw -> {spread(left, every) * 100:.1f} % with the gains"
          f" (plain floor {spread(left, ~textured) * 100:.1f} %, print and tape edges {spread(left, textured) * 100:.1f} %)")
    print("  exposure a view (%): " + " ".join(f"{x * 100:+.0f}" for x in lum))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("splat", nargs="+", type=Path, help="a scan's splat export (out/scan/STAMP/splat)")
    for folder in p.parse_args().splat:
        report(folder)


if __name__ == "__main__":
    main()
