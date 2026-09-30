"""tools/m3_handeye.py on made-up pictures: a camera mounted a little off the URDF, joints with
their own scales and offsets, the board where capture expects it -- the fit explains the corners
to a fraction of a pixel, and the poses it plans are all ones the guard lets the arm into."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import m3_handeye as h

from roboarm import m3kinematics

K = np.array([[480.0, 0, 320], [0, 480.0, 240], [0, 0, 1]])


def test_the_planned_poses_are_allowed_and_see_the_board():
    poses = h.plan_poses(0.56, count=12)
    assert len(poses) == 12
    for j1, j2, j3, j4 in poses:
        assert m3kinematics.problem({1: j1, 2: j2, 3: j3, 4: j4, 5: 90}) is None


def test_the_fit_explains_pictures_made_with_other_joints_and_mounting():
    truth_X = h.urdf_camera() @ h.rt(np.array([0.02, -0.03, 0.01, 0.004, -0.006, 0.003]))
    truth_joints = np.array([3.0, -2.0, 0.93, 0.97, 1.02, 0.9])
    board_pose = np.eye(4)
    board_pose[:3, 3] = (0.43, 0.19, m3kinematics.FLOOR)
    board_pose[:3, :3] = np.diag([1.0, -1.0, -1.0])   # flat, face up
    points = h.board().getChessboardCorners()
    views = []
    for j1, j2, j3, j4 in h.plan_poses(0.56, count=8):
        pose = {1: j1, 2: j2, 3: j3, 4: j4, 5: 90}
        cam_board = np.linalg.inv(h.arm4(pose, truth_joints) @ truth_X) @ board_pose
        p = (cam_board[:3, :3] @ points.T).T + cam_board[:3, 3]
        uv = (K @ (p / p[:, 2:3]).T).T[:, :2]
        inside = (p[:, 2] > 0) & (uv[:, 0] > 0) & (uv[:, 0] < 640) & (uv[:, 1] > 0) & (uv[:, 1] < 480)
        if inside.sum() >= 8:
            views.append((pose, uv[inside], points[inside]))
    assert len(views) >= 6
    assert h.rms_px(np.concatenate([h.params_of(truth_X), h.params_of(board_pose), truth_joints]), views, K) < 1e-6
    _x, rms = h.fit(views, K)
    assert rms < 0.5
