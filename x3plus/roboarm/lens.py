"""Where the wrist camera is, for ANY arm pose: the lens on arm_link4 (x3plus/PLAN.md X2.1).

The calibrated looks map pixels to the table with one homography each, which holds only at those
poses. Depth from one camera needs more: the camera as a 3D instrument -- its matrix (measured by
tools/camera_model.py intrinsics) and its position and turn in the base frame for whatever pose the
arm is in. This module is that second part: the kinematics up to J4, then a fixed lens mount on
the forearm, fitted to board corners seen from many poses (tools/camera_model.py handeye).

Frames. The base frame is the kinematics' own: x forward, y left, z up from the base plate, so
the table is at z = -cfg.TABLE_BELOW_PLATE. At J4, in the arm's vertical plane turned by J1:
`u` is the tool axis (out from J4 towards the fingers), `w` is across it in that plane (UP when
the tool points forward, FORWARD when it points down -- cfg's ABOVE_TOOL side), `s` is the
horizontal normal to the plane (the robot's left when the arm points forward). The lens sits at
J4 + mount[0] u + mount[1] w + mount[2] s, looking along u, with the picture's DOWN towards -w
(with the tool pointing down, the bottom of the picture is the base side), turned further by the
small rotation `turn` (a Rodrigues vector in the camera's own frame).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from roboarm import config as cfg
from roboarm.config import Pose

# cfg._LENS as a mount: along the tool, across it (ABOVE_TOOL), sideways (+ = left)
NOMINAL_MOUNT = (cfg.CAMERA_FROM_J4, cfg.CAMERA_ABOVE_TOOL, cfg.CAMERA_SIDE * cfg.CAMERA_OFF_AXIS)
# Gravity sag. The servos read their motor shafts; the links hang a little further, by more the
# further the load beyond a joint reaches out: J2..J4 are taken to give k_j degrees per metre of
# horizontal lever. 2026-10-06, 55 views read to 0.08 deg: 8.5 px without it, 6.0 px with it
# (J2 ~ +23, J3 ~ -16 deg/m: 2-3.5 deg at full reach); a tilted base explained nothing more.
HAND_LEVER_M = 0.10    # J4 to the hand's and camera's weight, roughly


def sagged(pose: Pose, sag=(0.0, 0.0, 0.0)) -> Pose:
    """`pose` as the links really hang: J2..J4 moved by k * the horizontal lever beyond each."""
    if not any(sag):
        return pose
    off = cfg.JOINT_OFFSET_DEG
    a2 = math.radians(90 - pose[2] + off[2])
    a3 = a2 + math.radians(90 - pose[3] + off[3])
    a4 = a3 + math.radians(90 - pose[4] + off[4])
    lever = {4: HAND_LEVER_M * math.sin(a4)}
    lever[3] = cfg.L_J3_J4 * math.sin(a3) + lever[4]
    lever[2] = cfg.L_J2_J3 * math.sin(a2) + lever[3]
    out = dict(pose)
    for joint, k in zip((2, 3, 4), sag):
        out[joint] = pose[joint] + k * lever[joint]
    return out


def j4_frame(pose: Pose) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(J4's position, u, w, s) in the base frame for `pose` (servo degrees)."""
    off = cfg.JOINT_OFFSET_DEG
    a2 = math.radians(90 - pose[2] + off[2])
    a3 = a2 + math.radians(90 - pose[3] + off[3])
    a4 = a3 + math.radians(90 - pose[4] + off[4])
    yaw = math.radians(90 - pose[1])
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    side = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    reach = cfg.L_J2_J3 * math.sin(a2) + cfg.L_J3_J4 * math.sin(a3)
    height = cfg.BASE_PLATE_TO_J2 + cfg.L_J2_J3 * math.cos(a2) + cfg.L_J3_J4 * math.cos(a3)
    u = math.sin(a4) * forward + math.cos(a4) * up
    w = -math.cos(a4) * forward + math.sin(a4) * up
    return reach * forward + height * up, u, w, side


def camera_pose(pose: Pose, mount=NOMINAL_MOUNT, turn=(0.0, 0.0, 0.0),
                sag=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """(R, c): the camera's axes as the columns of R (x right, y down, z along the view) and its
    centre c, in the base frame, for the arm at `pose` (read back), sagging by `sag`."""
    j4, u, w, s = j4_frame(sagged(pose, sag))
    y = -w
    nominal = np.column_stack([np.cross(y, u), y, u])
    R = nominal @ cv2.Rodrigues(np.asarray(turn, np.float64))[0]
    return R, j4 + mount[0] * u + mount[1] * w + mount[2] * s


def project(points: np.ndarray, pose: Pose, K, dist, mount=NOMINAL_MOUNT, turn=(0.0, 0.0, 0.0),
            sag=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Base-frame points (N x 3) -> pixels (N x 2) of the camera at `pose`."""
    R, c = camera_pose(pose, mount, turn, sag)
    rvec = cv2.Rodrigues(R.T)[0]
    tvec = -R.T @ c
    pixels, _ = cv2.projectPoints(np.asarray(points, np.float64), rvec, tvec,
                                  np.asarray(K, np.float64), np.asarray(dist, np.float64))
    return pixels.reshape(-1, 2)


def table_points(table_xy, shift=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Table points (x forward, y left, metres) on the table plane, moved by `shift` (dx, dy, and a
    turn in degrees about the base's axis): the board may not lie exactly where it was measured."""
    xy = np.asarray(table_xy, np.float64)
    a = math.radians(shift[2])
    turned = xy @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
    z = np.full((len(xy), 1), -cfg.TABLE_BELOW_PLATE)
    return np.hstack([turned + np.asarray(shift[:2]), z])


def residuals(params: np.ndarray, views: list[dict], K, dist) -> np.ndarray:
    """Pixel errors of every corner of every view for params = mount (3) + turn (3) + board shift (3)
    [+ sag (3)]."""
    mount, turn, shift = params[:3], params[3:6], params[6:9]
    sag = params[9:12] if len(params) > 9 else (0.0, 0.0, 0.0)
    out = []
    for view in views:
        pose = {int(j): float(a) for j, a in view["pose"].items()}
        seen = project(table_points(view["table"], shift), pose, K, dist, mount, turn, sag)
        out.append((seen - np.asarray(view["pixels"], np.float64)).ravel())
    return np.concatenate(out)


def fit(views: list[dict], K, dist, steps: int = 60, sag: bool = False) -> dict:
    """The lens mount, its turn and the board's shift that best explain `views` (each a dict of
    pose, pixels and the matching table points), by Levenberg-Marquardt from the nominal mount;
    with `sag`, the gravity sag too -- started from the fit without it, since from the nominal
    mount the two together fall into a wrong minimum (61 px on rendered views)."""
    params = np.array([*NOMINAL_MOUNT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Where the board lies matters most and may be far from where it was measured (it had slid
    # 57 mm on 2026-10-06): start from the best of a coarse grid, or the fit can settle wrong.
    starts = [(dx, dy) for dx in np.arange(-0.08, 0.081, 0.02) for dy in np.arange(-0.08, 0.081, 0.02)]
    best = min(starts, key=lambda d: float(np.sum(residuals(np.r_[params[:6], d[0], d[1], 0.0], views, K, dist) ** 2)))
    params[6:8] = best
    if sag:
        first = fit(views, K, dist, steps)
        params = np.array(first["mount_m"] + first["turn_rad"] + first["board_shift"] + [0.0, 0.0, 0.0])
    scale = np.array([0.01] * 3 + [0.02] * 3 + [0.01, 0.01, 1.0] + ([5.0] * 3 if sag else []))   # step sizes
    damping = 1e-3
    error = residuals(params, views, K, dist)
    for _ in range(steps):
        jac = np.empty((error.size, params.size))
        for i in range(params.size):
            h = np.zeros_like(params)
            h[i] = 1e-4 * scale[i]
            jac[:, i] = (residuals(params + h, views, K, dist) - residuals(params - h, views, K, dist)) / (2 * h[i])
        normal = jac.T @ jac
        while True:
            step = np.linalg.solve(normal + damping * np.diag(np.diag(normal)), -jac.T @ error)
            trial = residuals(params + step, views, K, dist)
            if trial @ trial < error @ error:
                params, error, damping = params + step, trial, max(damping / 3, 1e-9)
                break
            damping *= 4
            if damping > 1e6:
                break
        if damping > 1e6 or np.abs(step / scale).max() < 1e-6:
            break
    rms_view = []
    start = 0
    for view in views:
        n = 2 * len(view["pixels"])
        part = error[start:start + n].reshape(-1, 2)
        rms_view.append(float(np.sqrt(np.mean(np.sum(part ** 2, axis=1)))))
        start += n
    covariance = np.linalg.pinv(jac.T @ jac) * (error @ error) / max(1, error.size - params.size)
    sd = np.sqrt(np.diag(covariance))
    return {"mount_m": params[:3].tolist(), "turn_rad": params[3:6].tolist(),
            "board_shift": params[6:9].tolist(), "sd": sd.tolist(),
            "sag_deg_per_m": params[9:12].tolist() if sag else [0.0, 0.0, 0.0],
            "rms_px": float(np.sqrt(np.mean(error.reshape(-1, 2) ** 2) * 2)),
            "per_view_px": [round(v, 2) for v in rms_view], "views": len(views)}
