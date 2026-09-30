"""A top-down picture of the arena floor -- the printed track -- from RTAB-Map's coloured cloud:
the points within 3 cm of the floor, seen from above at 1 cm per pixel, each pixel the median
colour of its points. Also a view from above of everything (fence, landmarks) by height, the
cloud cropped to the fence (OUT_arena.ply), and a SCORE line to compare stitchings by: the fence
as the cloud has it -- its size against 3.2 x 2.8 m and how thick its walls (the spread of the
same wall across frames). A Laplacian "sharpness" of the print was tried and dropped: it rose
and fell with speckle and coverage, and ranked the visibly blurred print first.

    python3 ortho.py CLOUD.ply OUT_PREFIX     (tools/arena_map.py rtab-export runs it)
"""

import sys

import cv2
import fence_fit
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


def write_ply(path, xyz, rgb):
    data = np.zeros(len(xyz), [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                               ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = xyz.T
    data["red"], data["green"], data["blue"] = rgb.T
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(data)}\n"
                 "property float x\nproperty float y\nproperty float z\nproperty uchar red\n"
                 "property uchar green\nproperty uchar blue\nend_header\n").encode())
        f.write(data.tobytes())


def main():
    xyz, rgb = read_ply(sys.argv[1])
    out = sys.argv[2]
    floor_z = np.median(xyz[:, 2][np.abs(xyz[:, 2]) < 0.1])

    # The fence as the cloud has it -- the points 5-25 cm above the floor, fitted like a map --
    # scores the stitching (how thick its walls, how near 3.2 x 2.8 m) and crops the cloud to
    # the arena (the camera also saw the room over the fence).
    band = (xyz[:, 2] - floor_z > 0.05) & (xyz[:, 2] - floor_z < 0.25)
    fence = fence_fit.fit_points(xyz[band, :2])
    o = np.array(fence["arena_frame"]["origin_map"])
    ex = np.array(fence["arena_frame"]["x_axis_map"])
    length, width = fence["fence"]["size_m"]
    a, b = (xyz[:, :2] - o) @ ex, (xyz[:, :2] - o) @ fence_fit.rot90(ex)
    arena = (a > -0.05) & (a < length + 0.05) & (b > -0.05) & (b < width + 0.05)
    xyz, rgb = xyz[arena], rgb[arena]
    write_ply(f"{out}_arena.ply", xyz, rgb)

    res = 0.01
    lo = xyz[:, :2].min(axis=0) - 0.05
    hi = xyz[:, :2].max(axis=0) + 0.05
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
    f = fence["fence"]
    print(f"{len(xyz)} points in the arena, floor at z = {floor_z:+.3f} m, {on_floor.sum()} on it;"
          f" {w} x {h} px at 1 cm -> {out}_floor.png, {out}_heights.png, {out}_arena.ply")
    print(f"SCORE fence {f['size_m'][0]:.3f} x {f['size_m'][1]:.3f} m"
          f" (off {f['error_cm'][0]:+.1f} / {f['error_cm'][1]:+.1f} cm), walls thick"
          f" {[wall['thick_cm'] for wall in f['walls']]} cm, corners {f['corner_angles_deg']}")


if __name__ == "__main__":
    main()
