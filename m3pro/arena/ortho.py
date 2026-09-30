"""A top-down picture of the arena floor -- the printed track -- from RTAB-Map's coloured cloud:
the points within 3 cm of the floor, seen from above at 1 cm per pixel, each pixel the median
colour of its points. Also a view from above of everything (fence, landmarks) by height.

    python3 ortho.py CLOUD.ply OUT_PREFIX     (tools/arena_map.py rtab-export runs it)
"""

import sys

import cv2
import numpy as np

TYPES = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1", "uint8": "u1",
         "int": "i4", "uint": "u4", "short": "i2", "ushort": "u2"}


def read_ply(path):
    """x, y, z and r, g, b of a PLY's vertices (binary little-endian or ASCII)."""
    with open(path, "rb") as f:
        props, count, fmt, element = [], 0, "ascii", None
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):   # PCL adds a "camera" element after the vertices
                element = line.split()[1]
                if element == "vertex":
                    count = int(line.split()[2])
            elif line.startswith("property") and element == "vertex":
                _p, kind, name = line.split()
                props.append((name, "<" + TYPES[kind]))
            elif line == "end_header":
                break
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=count)
            v = {name: data[:, i] for i, (name, _k) in enumerate(props)}
        else:
            v = np.frombuffer(f.read(np.dtype(props).itemsize * count), dtype=props, count=count)
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(float)
    rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
    return xyz, rgb


def main():
    xyz, rgb = read_ply(sys.argv[1])
    out = sys.argv[2]
    floor_z = np.median(xyz[:, 2][np.abs(xyz[:, 2]) < 0.1])
    res = 0.01
    lo = np.percentile(xyz[:, :2], 1, axis=0) - 0.2
    hi = np.percentile(xyz[:, :2], 99, axis=0) + 0.2
    w, h = np.ceil((hi - lo) / res).astype(int)
    col = ((xyz[:, 0] - lo[0]) / res).astype(int)
    row = h - 1 - ((xyz[:, 1] - lo[1]) / res).astype(int)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)

    # the floor print: median colour per pixel of the points on the floor
    on_floor = inside & (np.abs(xyz[:, 2] - floor_z) < 0.03)
    key = row[on_floor] * w + col[on_floor]
    order = np.argsort(key, kind="stable")
    key, colours = key[order], rgb[on_floor][order]
    starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1]
    image = np.full((h * w, 3), 40, np.uint8)
    for s, e in zip(starts, np.r_[starts[1:], len(key)]):
        image[key[s]] = np.median(colours[s:e], axis=0)
    cv2.imwrite(f"{out}_floor.png", cv2.cvtColor(image.reshape(h, w, 3), cv2.COLOR_RGB2BGR))

    # everything from above, the highest point of each pixel, coloured by height (0 - 40 cm)
    top = np.full(h * w, -1.0)
    np.maximum.at(top, (row * w + col)[inside], xyz[inside, 2] - floor_z)
    heights = np.clip(top.reshape(h, w) / 0.4 * 255, 0, 255).astype(np.uint8)
    view = cv2.applyColorMap(heights, cv2.COLORMAP_TURBO)
    view[top.reshape(h, w) < 0] = 0
    cv2.imwrite(f"{out}_heights.png", view)
    print(f"{len(xyz)} points, floor at z = {floor_z:+.3f} m, {on_floor.sum()} on it;"
          f" {w} x {h} px at 1 cm -> {out}_floor.png, {out}_heights.png")


if __name__ == "__main__":
    main()
