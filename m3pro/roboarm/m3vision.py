"""What the M3's depth camera (the DCW2, depth aligned to colour) measures for the arm: the floor's
plane under the camera, which fixes the camera's height, tilt and roll exactly whatever the arm got
wrong (roboarm/m3kinematics.py SAG: ~1-2 cm, ~1-2 deg), and the things standing on that floor.

Depth images are 16-bit millimetres in the colour camera's pixels; K is the colour camera's matrix.
The camera frame is the optical one: x right, y down, z ahead. No ROS here.
"""

from __future__ import annotations

import cv2
import numpy as np

NEAR, FAR = 0.2, 1.2   # m: the DCW2's depth starts at 0.2; past ~1.2 the floor is too oblique to trust


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
