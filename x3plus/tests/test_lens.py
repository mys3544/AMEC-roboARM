"""roboarm/lens.py: the lens on the arm, fitted back from views rendered through a known mount."""

import math

import numpy as np

from roboarm import config as cfg
from roboarm import kinematics as kin
from roboarm import lens, sweep

K = [[506.9, 0, 436.0], [0, 506.2, 254.4], [0, 0, 1]]
DIST = [0.003, -0.028, 0.005, 0.004, 0.0]
SIZE = (848, 480)


def views_through(mount, turn, shift, noise_px=0.3, seed=1):
    """What the camera mounted at `mount`, turned by `turn`, sees of a 7 x 5 grid of table points
    (moved by `shift`, as a board that slid) from a spread of poses: noise added, the parts
    outside the picture dropped."""
    rng = np.random.default_rng(seed)
    grid = [(0.056 + 0.053 * i, -0.1965 + 0.053 * j) for i in range(6) for j in range(8)]
    views = []
    for fwd, lift, pitch in [(0.150, 0.090, None), (0.110, 0.110, 180.0), (0.190, 0.125, None),
                             (0.210, 0.120, 150.0), (0.170, 0.130, 160.0), (0.190, 0.080, 160.0)]:
        pose, _ = kin.solve(fwd, 0.0, -cfg.TABLE_BELOW_PLATE + lift,
                            pitches=kin.GRASP_PITCHES if pitch is None else (pitch,))
        for yaw in (-20, 0, 20):
            at = sweep.pose_at(pose, yaw)
            pixels = lens.project(lens.table_points(grid, shift), at, K, DIST, mount, turn)
            pixels += rng.normal(0, noise_px, pixels.shape)
            keep = (pixels[:, 0] > 0) & (pixels[:, 0] < SIZE[0]) & (pixels[:, 1] > 0) & (pixels[:, 1] < SIZE[1])
            if keep.sum() >= 8:
                views.append({"label": f"{fwd} yaw {yaw}", "pose": {str(j): v for j, v in at.items()},
                              "pixels": pixels[keep].tolist(), "table": np.array(grid)[keep].tolist()})
    return views


def test_the_mount_turn_and_board_shift_come_back_from_the_views():
    mount, turn, shift = (0.088, 0.041, 0.007), (math.radians(-9), math.radians(2), 0.0), (-0.012, 0.003, -1.0)
    views = views_through(mount, turn, shift)
    assert len(views) >= 12
    fitted = lens.fit(views, K, DIST)
    assert np.allclose(fitted["mount_m"], mount, atol=0.0015), fitted["mount_m"]
    assert np.allclose(fitted["turn_rad"], turn, atol=math.radians(0.3)), fitted["turn_rad"]
    assert np.allclose(fitted["board_shift"][:2], shift[:2], atol=0.0015)
    assert abs(fitted["board_shift"][2] - shift[2]) < 0.3
    assert fitted["rms_px"] < 0.6


def test_the_camera_looks_along_the_tool_with_the_base_at_the_bottom():
    """With the tool straight down and the base turned to straight ahead, the picture's bottom is
    the base side and its right is the robot's right."""
    pose, _ = kin.solve(0.110, 0.0, -cfg.TABLE_BELOW_PLATE + 0.110, pitches=(180.0,))
    R, c = lens.camera_pose(pose, (0.085, 0.0, 0.0))
    assert np.allclose(R[:, 2], [0, 0, -1], atol=0.05), "looking down"
    assert R[0, 1] < -0.95, "picture down = towards the base (-x)"
    assert R[1, 0] < -0.95, "picture right = the robot's right (-y)"
    nadir = kin.camera_nadir({**pose})
    assert abs(c[0] - nadir[0]) < 0.06 and abs(c[1]) < 0.01
