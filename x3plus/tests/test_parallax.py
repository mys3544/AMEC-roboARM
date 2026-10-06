"""roboarm/parallax.py: a raised square's height from two rendered views, through a pose error."""

import math

import cv2
import numpy as np

from roboarm import parallax

K = np.array([[506.9, 0, 424.0], [0, 506.2, 240.0], [0, 0, 1]])
FRAME = (848, 480)
TOP = (0.20, 0.25, -0.025, 0.025)   # x0, x1, y0, y1 of the raised square (m)


def texture():
    rng = np.random.default_rng(7)
    noise = cv2.GaussianBlur(rng.uniform(0, 255, (800, 800)).astype(np.float32), (0, 0), 2.0)
    return cv2.normalize(noise, None, 20, 235, cv2.NORM_MINMAX)


TEX = texture()     # 1 texel = 1 mm, the table from x 0 to 0.8 and y -0.4 to 0.4


def camera(yaw_deg: float, error_deg: float = 0.0):
    """A camera 0.30 m up, 0.10 m out, looking forward and down 60 deg, the base turned by yaw;
    error_deg turns it a little more about its own x axis -- the arm's pose error."""
    t = math.radians(60)
    z = np.array([math.cos(t), 0.0, -math.sin(t)])
    x = np.array([0.0, -1.0, 0.0])
    R = np.column_stack([x, np.cross(z, x), z])
    R = R @ cv2.Rodrigues(np.array([math.radians(error_deg), 0, 0]))[0]
    a = math.radians(yaw_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return Rz @ R, Rz @ np.array([0.10, 0.0, 0.30])   # the X3's wide look: lens ~10 cm off J1's axis


def render(R, c, h):
    """The textured table, with the square raised by h -- its top only, which is what the
    parallax of a top face is about."""
    vs, us = np.mgrid[0:FRAME[1], 0:FRAME[0]]
    rays = np.stack([us, vs, np.ones_like(us)], -1).reshape(-1, 3) @ np.linalg.inv(K).T @ R.T
    t_top = (h - c[2]) / rays[:, 2]
    top = c + rays * t_top[:, None]
    on_top = (top[:, 0] > TOP[0]) & (top[:, 0] < TOP[1]) & (top[:, 1] > TOP[2]) & (top[:, 1] < TOP[3])
    t_floor = (0.0 - c[2]) / rays[:, 2]
    hit = np.where(on_top[:, None], top, c + rays * t_floor[:, None])
    # the top carries the texture of 0.5 m to the side, so it is not a copy of the table under it
    tx = (hit[:, 0] * 1000).astype(np.float32) + np.where(on_top, 500.0, 0.0).astype(np.float32)
    ty = (hit[:, 1] * 1000 + 400).astype(np.float32)
    img = cv2.remap(TEX, tx.reshape(FRAME[1], FRAME[0]) % 800, ty.reshape(FRAME[1], FRAME[0]) % 800,
                    cv2.INTER_LINEAR)
    table = ((hit[:, 0] > 0.0) & (hit[:, 0] < 0.6) & ~on_top).reshape(FRAME[1], FRAME[0])
    return img.astype(np.float32), table, on_top.reshape(FRAME[1], FRAME[0])


def views(h, error_deg):
    out = []
    for yaw, err in ((0, 0.0), (-15, error_deg), (15, -error_deg)):
        R_true, c = camera(yaw)
        grey, table, top = render(R_true, c, h)
        R_believed, _ = camera(yaw, err)     # what the arm says, a little off
        out.append({"grey": grey, "table": table, "top": top, "R": R_believed, "c": c})
    return out


def test_a_raised_top_comes_out_at_its_height_through_a_pose_error():
    for h in (0.030, 0.040):
        a, *others = views(h, error_deg=0.8)
        height, agreement = parallax.height_map(a, others, K, table_z=0.0)
        region = cv2.erode(a["top"].astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        found = parallax.object_height(height, agreement, region)
        assert found is not None and abs(found - h) < 0.003, (h, found)
        table = cv2.erode(a["table"].astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        assert abs(parallax.object_height(height, agreement, table)) < 0.002, "the table reads flat"


def test_without_the_table_correction_the_pose_error_would_show():
    """The same error, uncorrected, moves the table off zero: what table_correction is for."""
    a, b, _ = views(0.030, error_deg=0.8)
    H0 = parallax.plane_map(K, a["R"], a["c"], b["R"], b["c"], 0.0)
    C, info = parallax.table_correction(a["grey"].astype(np.uint8), b["grey"].astype(np.uint8),
                                        a["table"], b["table"], H0)
    assert info["pose_off_px"] > 2.0, info
    assert not np.allclose(C, np.eye(3), atol=1e-3)
