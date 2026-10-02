"""The M3's arm and body as the guard sees them (roboarm/m3kinematics.py): Yahboom's meshes, made small.

  * the arm's shell: arm3, arm4, arm5, the DCW2 camera and the gripper's palm, each in its own link's
    frame, its mesh's vertices one per 6 mm cube -- the guard's outline is a line through the
    joints, the parts reach up to ~5 cm past it (the camera 4.5 cm to either side);
  * the body: the highest point of base_link and the front camera in each 5 mm cell seen from
    above, and the wheels (r 0.04 on axles 0.04 up, 4 cm wide).

The lidars are not here: they are no part of Yahboom's model, and m3kinematics.LIDARS has them as
measured. 2026-10-02: the arm hit the front-right one once -- its camera, most likely, which
the outline had as a point.

    python tools/m3_body_model.py [MESHES]   # -> roboarm/m3body.npz

MESHES: the folder with Yahboom's STL files, by default where the robot has them; on this laptop,
a copy of it (the files are binary STL in the links' own frames, the URDF's visual origins all 0).
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

ROBOT_MESHES = "/home/jetson/yahboomcar_ws/src/yahboom_M3Pro_description/meshes"
OUT = Path(__file__).resolve().parents[1] / "roboarm" / "m3body.npz"
SHELL = {"arm3": "arm3", "arm4": "arm4", "arm5": "arm5", "camera": "DCW2", "gripping": "Gripping"}   # frame: mesh
VOXEL = 0.006   # m: one shell point per cube this size
CELL, X0, Y0, NX, NY = 0.005, -0.30, -0.27, 120, 108   # the body's height map, base_link
FRONT_CAMERA = ((0.0909154688773379, 0.000198272120369507, 0.0934842113675051), (1.5707963267949, 0, 0))   # URDF
WHEELS_X, WHEEL_Y, WHEEL_R = (0.0775753787975409, -0.0814246212024597), 0.09775, 0.04


def triangles(path: Path) -> np.ndarray:
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    if 84 + 50 * n != len(data):
        raise SystemExit(f"{path}: not a binary STL")
    rows = np.frombuffer(data[84:], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]), count=n)
    return rows["v"].reshape(-1, 3, 3).astype(float)


def shell(path: Path) -> np.ndarray:
    v = triangles(path).reshape(-1, 3)
    _, first = np.unique(np.floor(v / VOXEL).astype(int), axis=0, return_index=True)
    return v[np.sort(first)].astype(np.float32)


def surface(t: np.ndarray) -> np.ndarray:
    """Points over the triangles, ~1 a square mm (a 5 mm cell under a big flat triangle has none
    of its corners), and the corners."""
    rng = np.random.default_rng(0)
    area = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) / 2
    idx = np.repeat(np.arange(len(t)), np.clip((area * 1e6).astype(int), 1, 400))
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return np.vstack([t[idx, 0] + u[:, None] * (t[idx, 1] - t[idx, 0]) + v[:, None] * (t[idx, 2] - t[idx, 0]),
                      t.reshape(-1, 3)])


def rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def body(meshes: Path) -> np.ndarray:
    """mm, the highest point in each cell; -1 where nothing is."""
    top = np.full((NX, NY), -np.inf)

    def put(points):
        ix = np.floor((points[:, 0] - X0) / CELL).astype(int)
        iy = np.floor((points[:, 1] - Y0) / CELL).astype(int)
        ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
        np.maximum.at(top, (ix[ok], iy[ok]), points[ok, 2])

    put(surface(triangles(meshes / "base_link.STL")))
    xyz, angles = FRONT_CAMERA
    put(surface(triangles(meshes / "Camera.STL")) @ rpy(*angles).T + xyz)
    xs = X0 + (np.arange(NX) + 0.5) * CELL
    ys = Y0 + (np.arange(NY) + 0.5) * CELL
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    for wx in WHEELS_X:
        wheel = (np.abs(gx - wx) < WHEEL_R) & (np.abs(np.abs(gy) - WHEEL_Y) < 0.02)
        top = np.where(wheel, np.maximum(top, 2 * WHEEL_R), top)
    return np.where(np.isfinite(top), np.round(top * 1000), -1).astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("meshes", nargs="?", default=ROBOT_MESHES)
    meshes = Path(parser.parse_args().meshes)
    parts = {f"shell_{frame}": shell(meshes / f"{mesh}.STL") for frame, mesh in SHELL.items()}
    heights = body(meshes)
    np.savez_compressed(OUT, body_mm=heights, body_origin=np.array([X0, Y0]), body_cell=np.array(CELL), **parts)
    print(f"{OUT}: shell {', '.join(f'{k[6:]} {len(v)}' for k, v in parts.items())} points;"
          f" body {int((heights >= 0).sum())} cells, highest {heights.max()} mm")


if __name__ == "__main__":
    main()
