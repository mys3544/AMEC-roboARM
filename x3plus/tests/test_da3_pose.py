"""roboarm/da3_pose.py: the pose code Depth Anything 3's engines take."""

import math

import numpy as np

from roboarm import da3_pose

K = np.array([[300.0, 0, 252.0], [0, 300.0, 140.0], [0, 0, 1]])


def rotation_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def test_quaternions_are_xyzw_with_the_real_part_kept_positive():
    q = da3_pose.mat_to_quat(np.stack([np.eye(3), rotation_z(90), rotation_z(-90), rotation_z(180)]))
    s = math.sqrt(0.5)
    assert np.allclose(q[0], [0, 0, 0, 1])
    assert np.allclose(q[1], [0, 0, s, s])
    assert np.allclose(q[2], [0, 0, -s, s])
    assert np.allclose(np.abs(q[3]), [0, 0, 1, 0]) and q[3][3] >= 0


def test_every_branch_gives_a_unit_quaternion_of_the_same_rotation():
    rng = np.random.default_rng(3)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        x, y, z, w = q
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        back = da3_pose.mat_to_quat(R[None])[0]
        assert abs(np.linalg.norm(back) - 1) < 1e-6
        assert np.allclose(back, q if w >= 0 else -q, atol=1e-6)


def test_the_code_is_the_camera_centre_its_turn_and_its_fields_of_view():
    R = rotation_z(30) @ np.diag([1.0, -1.0, -1.0])     # looking straight down, turned 30 deg
    c = np.array([0.1, -0.02, 0.3])
    code = da3_pose.pose_code(da3_pose.world_to_camera(R, c)[None], K[None], (280, 504))[0]
    assert code.shape == (9,) and code.dtype == np.float32
    assert np.allclose(code[:3], c, atol=1e-6), "the translation is the camera's centre"
    assert np.allclose(code[7], 2 * math.atan(140 / 300)) and np.allclose(code[8], 2 * math.atan(252 / 300))
    assert np.allclose(da3_pose.mat_to_quat(R[None])[0], code[3:7])


def test_poses_are_normalised_like_the_api():
    R = rotation_z(30) @ np.diag([1.0, -1.0, -1.0])
    cams = [da3_pose.world_to_camera(rotation_z(a) @ R, np.array(c))
            for a, c in ((0, [0.1, 0.0, 0.3]), (10, [0.1, 0.4, 0.3]), (-10, [0.1, -0.2, 0.3]))]
    rel = da3_pose.normalised(np.stack(cams))
    assert np.allclose(rel[0], np.eye(4)), "the first camera is the origin"
    centres = [np.linalg.norm(-r[:3, :3].T @ r[:3, 3]) for r in rel]
    assert np.allclose(sorted(centres), [0.0, 1.0, 2.0]), "distances in units of the median one (0.2 m)"
    near = da3_pose.normalised(np.stack([cams[0], da3_pose.world_to_camera(R, np.array([0.13, 0.0, 0.3]))]))
    assert np.isclose(np.linalg.norm(near[1][:3, 3]), 0.03 / 0.1), "a baseline under 0.1 is divided by 0.1"
    # relative rotations are kept: camera i's turn from camera 0
    assert np.allclose(rel[1][:3, :3], cams[1][:3, :3] @ cams[0][:3, :3].T)
