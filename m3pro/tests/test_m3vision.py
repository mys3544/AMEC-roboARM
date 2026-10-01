"""roboarm/m3vision.py on made-up depth: a camera 0.33 m over a floor, looking 52 deg down and rolled a
little, a 30 mm cube on the floor -- the floor's plane comes back to a millimetre and a tenth of a
degree, and level() puts a model camera that was 16 mm and 5 deg off back on the floor, but for its yaw."""

import math

import cv2
import numpy as np

from roboarm import m3vision

K = np.array([[480.0, 0, 320], [0, 480.0, 240], [0, 0, 1]])


def camera(height, down_deg, roll_deg):
    """base <- camera (optical: x right, y down, z ahead), looking along +x of the base, tilted down."""
    look = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # optical axes in base, level
    tilt = cv2.Rodrigues(np.array([0, math.radians(down_deg), 0]))[0]
    roll = cv2.Rodrigues(np.array([math.radians(roll_deg), 0, 0]))[0]
    m = np.eye(4)
    m[:3, :3] = tilt @ roll @ look
    m[:3, 3] = (0.15, 0.0, height)
    return m


def render(cam, cube=None, noise=0.002, seed=1):
    """A depth image (mm) of the floor z = 0 -- and a 30 mm cube on it -- from `cam`."""
    rng = np.random.default_rng(seed)
    v, u = np.mgrid[0:480, 0:640]
    rays = np.stack([(u - 320) / 480.0, (v - 240) / 480.0, np.ones_like(u, float)], axis=-1)
    d = rays @ cam[:3, :3].T                       # ray directions in base
    t = -cam[2, 3] / np.where(np.abs(d[..., 2]) > 1e-9, d[..., 2], -1e-9)
    z = np.where(t > 0, t, 0.0)                    # depth along the optical axis = t (ray z = 1)
    if cube is not None:   # the cube's top face, 30 mm up
        t2 = (0.03 - cam[2, 3]) / np.where(np.abs(d[..., 2]) > 1e-9, d[..., 2], -1e-9)
        hit = cam[:3, 3] + d * t2[..., None]
        on = (t2 > 0) & (np.abs(hit[..., 0] - cube[0]) < 0.015) & (np.abs(hit[..., 1] - cube[1]) < 0.015)
        z = np.where(on, t2, z)
    z = z + rng.normal(0, noise, z.shape)
    return np.clip(z * 1000, 0, 65535).astype(np.uint16)


def test_the_floor_comes_back_from_noisy_depth_with_a_cube_on_it():
    cam = camera(0.33, 52, 2.0)
    normal, height, share = m3vision.floor_plane(render(cam, cube=(0.40, 0.02)), K)
    truth = cam[:3, :3].T @ np.array([0, 0, 1.0])   # the floor's up, in the camera frame
    assert abs(height - 0.33) < 0.001
    assert math.degrees(math.acos(min(1.0, normal @ truth))) < 0.15
    assert share > 0.8


def test_level_puts_a_model_camera_back_on_the_measured_floor():
    real = camera(0.315, 56.5, 1.0)
    model = camera(0.331, 51.5, 0.0)                # the arm model: 16 mm high, 5 deg shallow, no roll
    normal, height, _ = m3vision.floor_plane(render(real), K)
    fixed = m3vision.level(model, normal, height)
    assert abs(fixed[2, 3] - 0.315) < 0.001
    up = fixed[:3, :3] @ normal                      # the floor, seen through the fixed camera: level
    assert math.degrees(math.acos(min(1.0, up[2]))) < 0.15
    # what is left is a turn about the vertical only -- the roll of a camera looking down is mostly
    # a yaw, which no floor can show (here ~0.8 deg): the arm model's to keep
    left = fixed[:3, :3] @ real[:3, :3].T
    axis = cv2.Rodrigues(left)[0].ravel()
    assert np.linalg.norm(axis) < math.radians(1.5) and abs(axis[2]) > 0.9 * np.linalg.norm(axis)


def test_a_cube_on_the_floor_is_found_where_it_stands_and_named_by_its_height():
    cam = camera(0.33, 52, 1.0)
    found = m3vision.objects(render(cam, cube=(0.42, -0.03)), K, cam)
    assert len(found) == 1
    cube = found[0]
    assert cube["size"] == 0.030 and abs(cube["top"] - 0.030) < 0.004
    assert np.linalg.norm(cube["centre"][:2] - [0.42, -0.03]) < 0.004
    assert m3vision.objects(render(cam), K, cam) == []       # a bare floor: nothing


def test_something_as_tall_as_a_cube_but_longer_is_not_one():
    cam = camera(0.33, 52, 1.0)
    depth = render(cam, cube=(0.42, -0.03))
    plank = render(cam, cube=(0.42, -0.03))   # a second render to borrow from: a 30 mm tall plank 12 cm long
    v, u = np.mgrid[0:480, 0:640]
    rays = np.stack([(u - 320) / 480.0, (v - 240) / 480.0, np.ones_like(u, float)], axis=-1) @ cam[:3, :3].T
    t = (0.03 - cam[2, 3]) / np.where(np.abs(rays[..., 2]) > 1e-9, rays[..., 2], -1e-9)
    hit = cam[:3, 3] + rays * t[..., None]
    on = (t > 0) & (np.abs(hit[..., 0] - 0.55) < 0.015) & (np.abs(hit[..., 1] - 0.10) < 0.06)
    plank = np.where(on, (t * 1000).astype(np.uint16), depth)
    found = m3vision.objects(plank, K, cam)
    assert sorted(o["size"] is None for o in found) == [False, True]   # the cube, and not-a-cube
