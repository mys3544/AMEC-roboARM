"""Depth Anything 3's camera pose code, computed here rather than in its TensorRT engine.

DA3 can be told where each camera was (pose-conditioned multi-view depth). Its camera encoder first
turns each camera into nine numbers -- the camera-to-world translation, the rotation as a
quaternion (x, y, z, w, real part last and kept >= 0) and the two fields of view -- and only then
runs learnt layers. The quaternion picks one of four candidate formulas by argmax / one-hot /
boolean indexing, which an engine reproduces badly, so tools/export_da3.py builds engines that take
these nine numbers as their "pose" input, and the robot computes them here. Matches DA3's own
extri_intri_to_pose_encoding to 5e-7 on 2000 random poses (2026-10-07).
"""

from __future__ import annotations

import numpy as np


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrices (N, 3, 3) -> quaternions (N, 4), x y z w, w >= 0 (DA3's mat_to_quat)."""
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = R.reshape(-1, 9).T
    q_abs = np.sqrt(np.maximum(0.0, np.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                                              1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1)))
    candidates = np.stack([
        np.stack([q_abs[:, 0] ** 2, m21 - m12, m02 - m20, m10 - m01], -1),
        np.stack([m21 - m12, q_abs[:, 1] ** 2, m10 + m01, m02 + m20], -1),
        np.stack([m02 - m20, m10 + m01, q_abs[:, 2] ** 2, m12 + m21], -1),
        np.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[:, 3] ** 2], -1),
    ], -2) / (2.0 * np.maximum(q_abs, 0.1))[..., None]
    q = candidates[np.arange(len(q_abs)), q_abs.argmax(-1)][:, [1, 2, 3, 0]]   # rijk -> ijkr
    return np.where(q[:, 3:4] < 0, -q, q)


def pose_code(world_to_camera: np.ndarray, K: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """(N, 9) float32 for N cameras: world_to_camera (N, 4, 4), K (N, 3, 3) for the network's
    input of size_hw (height, width)."""
    R = np.transpose(world_to_camera[:, :3, :3], (0, 2, 1))          # camera to world
    t = -np.einsum("nij,nj->ni", R, world_to_camera[:, :3, 3])
    h, w = size_hw
    fov_h = 2 * np.arctan((h / 2) / K[:, 1, 1])
    fov_w = 2 * np.arctan((w / 2) / K[:, 0, 0])
    return np.concatenate([t, mat_to_quat(R), fov_h[:, None], fov_w[:, None]], -1).astype(np.float32)


def normalised(world_to_camera: np.ndarray) -> np.ndarray:
    """(N, 4, 4) poses as DA3's API hands them to the network (DepthAnything3._normalize_extrinsics):
    relative to the first camera, whose pose becomes the identity, and the translations divided by
    the cameras' median distance from it (torch's median: the lower middle), at least 0.1."""
    rel = world_to_camera @ np.linalg.inv(world_to_camera[0])
    centres = -np.einsum("nji,nj->ni", rel[:, :3, :3], rel[:, :3, 3])
    dists = np.sort(np.linalg.norm(centres, axis=1))
    rel[:, :3, 3] /= max(float(dists[(len(dists) - 1) // 2]), 0.1)
    return rel


def world_to_camera(R: np.ndarray, c: np.ndarray) -> np.ndarray:
    """roboarm.lens.camera_pose's (R with the camera axes as columns, centre c) as a 4x4 matrix."""
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ c
    return T
