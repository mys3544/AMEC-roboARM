"""roboarm/monodepth.py: Depth Anything's relative map anchored to the floor, on rendered scenes."""

import math

import numpy as np

from roboarm import monodepth

K = [[506.9, 0, 436.0], [0, 506.2, 254.4], [0, 0, 1]]
NO_DIST = [0.0] * 5
FRAME = (848, 480)
GRID = (212, 120)    # a quarter of the frame: enough, and quick


def looking(down_deg: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """A camera `height` m over the floor (z = 0), turned `down_deg` below the horizontal,
    facing +x: (R with the camera axes as columns, centre)."""
    t = math.radians(down_deg)
    z = np.array([math.cos(t), 0.0, -math.sin(t)])
    x = np.array([0.0, -1.0, 0.0])
    return np.column_stack([x, np.cross(z, x), z]), np.array([0.0, 0.0, height])


def scene_depth(R, c, boxes) -> np.ndarray:
    """True Z on GRID of the floor and the TOPS of `boxes` ((x0, x1, y0, y1, height) in m)."""
    ray_cam = monodepth.rays(K, NO_DIST, FRAME, GRID)
    ray = ray_cam @ R.T
    z = (0.0 - c[2]) / ray[..., 2]
    for x0, x1, y0, y1, h in boxes:
        t = (h - c[2]) / ray[..., 2]
        hit = c + ray * t[..., None]
        on_top = (hit[..., 0] >= x0) & (hit[..., 0] <= x1) & (hit[..., 1] >= y0) & (hit[..., 1] <= y1) & (t > 0)
        z = np.where(on_top & (t < z), t, z)
    return z


def disparity(z, a=3.7, b=0.42, warp=(0.05, -0.02, 0.04), noise=0.002, seed=3) -> np.ndarray:
    """What Depth Anything would hand back: a / Z + b, a slow quadratic warp, a little noise."""
    ray_cam = monodepth.rays(K, NO_DIST, FRAME, GRID)
    x, y = ray_cam[..., 0], ray_cam[..., 1]
    rng = np.random.default_rng(seed)
    return a / z + b + warp[0] * x * x + warp[1] * x * y + warp[2] * y * y + rng.normal(0, noise, z.shape)


CUBE30 = (0.30, 0.33, -0.05, -0.02, 0.030)
CUBE40 = (0.24, 0.28, 0.03, 0.07, 0.040)
BLOCK = (0.36, 0.42, 0.04, 0.07, 0.030)   # a 30 x 30 x 60 lying


def top_height(result, R, c, box) -> float:
    ray = result["rays"] @ R.T
    t = (box[4] - c[2]) / ray[..., 2]
    hit = c + ray * t[..., None]
    inside = ((hit[..., 0] > box[0] + 0.004) & (hit[..., 0] < box[1] - 0.004)
              & (hit[..., 1] > box[2] + 0.004) & (hit[..., 1] < box[3] - 0.004))
    return float(np.nanmedian(result["height"][inside]))


def test_an_oblique_look_pins_the_scale_and_the_kit_heights_come_out():
    R, c = looking(down_deg=40, height=0.28)
    z = scene_depth(R, c, [CUBE30, CUBE40, BLOCK])
    result = monodepth.anchor(disparity(z), FRAME, K, NO_DIST, R, c, floor_z=0.0, warp=True)
    assert result["pinned"], result["scale_sd"]
    assert abs(result["scale"] - 3.7) < 0.1
    for box in (CUBE30, CUBE40, BLOCK):
        assert abs(top_height(result, R, c, box) - box[4]) < 0.003, box
    floor = result["floor"]
    assert np.nanmax(np.abs(result["height"][floor])) < 0.004, "the floor comes out flat"


def test_looking_straight_down_cannot_pin_the_scale():
    """September's failure, reproduced: the floor square to the picture leaves a unknown."""
    R, c = looking(down_deg=90, height=0.22)
    z = scene_depth(R, c, [(0.0, 0.03, -0.015, 0.015, 0.030)])
    result = monodepth.anchor(disparity(z), FRAME, K, NO_DIST, R, c, floor_z=0.0)
    assert not result["pinned"]


def test_without_the_warp_term_the_hills_bend_the_floor():
    R, c = looking(down_deg=40, height=0.28)
    z = scene_depth(R, c, [CUBE40])
    plain = monodepth.anchor(disparity(z), FRAME, K, NO_DIST, R, c, floor_z=0.0, warp=False)
    warped = monodepth.anchor(disparity(z), FRAME, K, NO_DIST, R, c, floor_z=0.0, warp=True)
    assert warped["sigma"] < plain["sigma"] / 2


def test_a_big_thing_in_view_does_not_drag_the_scale():
    """2026-10-15, the M3's close looks: a box over a third of the floor the pose predicts, or the
    gripper's finger a few cm from the lens, and the robust fit alone took it in (its sigma grows
    with it): the cube came out 32 mm, or a negative. The refits leave it out."""
    R, c = looking(down_deg=40, height=0.28)
    box = (0.15, 0.60, -0.40, 0.0, 0.12)        # 12 cm tall, 30 % of the floor in view
    finger = (0.10, 0.20, -0.30, 0.30, 0.24)    # 4 cm under the lens, across the near edge
    for big in (box, finger):
        d = disparity(scene_depth(R, c, [big, CUBE40]), warp=(0, 0, 0))
        once = monodepth.anchor(d, FRAME, K, NO_DIST, R, c, floor_z=0.0, refits=0)
        assert abs(top_height(once, R, c, CUBE40) - 0.040) > 0.005, big   # the failure, reproduced
        result = monodepth.anchor(d, FRAME, K, NO_DIST, R, c, floor_z=0.0)
        assert abs(result["scale"] - 3.7) < 0.1, big
        assert abs(top_height(result, R, c, CUBE40) - 0.040) < 0.003, big
