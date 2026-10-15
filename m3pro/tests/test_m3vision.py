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
    """A depth image (mm) of the floor z = 0 -- and a box on it, (x, y[, turn deg[, length, height]]):
    a 30 mm cube unless told, 30 mm wide always -- from `cam`."""
    rng = np.random.default_rng(seed)
    v, u = np.mgrid[0:480, 0:640]
    rays = np.stack([(u - 320) / 480.0, (v - 240) / 480.0, np.ones_like(u, float)], axis=-1)
    d = rays @ cam[:3, :3].T                       # ray directions in base
    t = -cam[2, 3] / np.where(np.abs(d[..., 2]) > 1e-9, d[..., 2], -1e-9)
    z = np.where(t > 0, t, 0.0)                    # depth along the optical axis = t (ray z = 1)
    if cube is not None:   # the box's top face
        length, tall = (tuple(cube[3:5]) + (0.03, 0.03))[:2]
        t2 = (tall - cam[2, 3]) / np.where(np.abs(d[..., 2]) > 1e-9, d[..., 2], -1e-9)
        hit = cam[:3, 3] + d * t2[..., None]
        turn = math.radians(cube[2] if len(cube) > 2 else 0.0)
        dx, dy = hit[..., 0] - cube[0], hit[..., 1] - cube[1]
        along, aside = dx * math.cos(turn) + dy * math.sin(turn), -dx * math.sin(turn) + dy * math.cos(turn)
        on = (t2 > 0) & (np.abs(along) < length / 2) & (np.abs(aside) < 0.015)
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


def test_the_robots_own_front_wheel_is_no_thing_but_a_cube_before_the_bumper_is():
    cam = camera(0.33, 52, 0.0)
    cam[0, 3] = 0.0   # further back than the others: the floor just past the bumper in view
    tyre = render(cam, cube=(0.130, 0.115, 0.0, 0.02, 0.06))   # 2 cm of it, 6 cm tall, by the left wheel
    assert m3vision.objects(tyre, K, cam) == []
    found = m3vision.objects(render(cam, cube=(0.145, 0.0)), K, cam)   # 2 cm out from the bumper
    assert len(found) == 1 and found[0]["size"] == 0.030


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


def test_a_turned_cube_reads_its_turn_and_one_cut_by_the_pictures_edge_says_so():
    cam = camera(0.33, 52, 1.0)
    cube = m3vision.objects(render(cam, cube=(0.42, -0.03, 40)), K, cam)[0]
    assert cube["whole"] and abs((cube["yaw"] - 40 + 45) % 90 - 45) < 3
    # the picture's bottom edge meets the cube's top face at x ~0.211: a third of it out of view, and
    # the rectangle round the rest turns with the edge (2026-10-01: 0.7 deg for a cube turned 45)
    cut = m3vision.objects(render(cam, cube=(0.215, 0.0, 40)), K, cam)[0]
    assert cut["size"] == 0.030 and not cut["whole"]


def test_the_blocks_lying_and_standing_are_named_and_gripped_across():
    cam = camera(0.33, 52, 1.0)
    lying = m3vision.objects(render(cam, cube=(0.42, -0.03, 30, 0.06, 0.03)), K, cam)[0]
    assert (lying["kind"], lying["size"], lying["length"], lying["period"]) == ("block", 0.030, 0.060, 180)
    assert lying["whole"] and abs((lying["yaw"] - 120 + 90) % 180 - 90) < 3   # across its length (30 + 90)
    assert np.linalg.norm(lying["centre"] - [0.42, -0.03, 0.015]) < 0.004
    standing = m3vision.objects(render(cam, cube=(0.42, -0.03, 10, 0.03, 0.06)), K, cam)[0]
    assert (standing["kind"], standing["size"], standing["height"], standing["period"]) == ("block", 0.030, 0.060, 90)
    assert abs(standing["centre"][2] - 0.030) < 1e-9


def picture(cam, x, y, turn, fine=4):
    """The colour picture to render()'s depth: a white 30 mm cube's top face turned `turn` deg on a
    chessboard of 30 mm squares square to the robot -- edges at 0 and 90 deg all round it. Drawn
    `fine` times finer and averaged down: a camera's optics soften its edges, no hard steps."""
    v, u = np.mgrid[0:480 * fine, 0:640 * fine]
    u, v = (u + 0.5) / fine - 0.5, (v + 0.5) / fine - 0.5
    d = np.stack([(u - 320) / 480.0, (v - 240) / 480.0, np.ones_like(u, float)], axis=-1) @ cam[:3, :3].T
    o = cam[:3, 3]
    dz = np.where(np.abs(d[..., 2]) > 1e-9, d[..., 2], -1e-9)
    floor = o + d * (-o[2] / dz)[..., None]
    board = ((np.floor(floor[..., 0] / 0.03) + np.floor(floor[..., 1] / 0.03)) % 2) * 255
    top = o + d * ((0.03 - o[2]) / dz)[..., None]
    t = math.radians(turn)
    dx, dy = top[..., 0] - x, top[..., 1] - y
    on = (np.abs(dx * math.cos(t) + dy * math.sin(t)) < 0.015) & (np.abs(-dx * math.sin(t) + dy * math.cos(t)) < 0.015)
    grey = cv2.resize(np.where(on, 200.0, board), (640, 480), interpolation=cv2.INTER_AREA).astype(np.uint8)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def test_the_turn_from_the_colour_edges_beats_the_depths_and_ignores_the_board():
    cam = camera(0.33, 52, 1.0)
    for true in (5, 37, 60, 85):   # near the board's own edges too
        cube = m3vision.objects(render(cam, cube=(0.40, -0.02, true)), K, cam)[0]
        image = picture(cam, 0.40, -0.02, true)
        for off in (-8, 8):        # the depth's turn, 8 deg out -- as it wandered on the robot
            turn, sure = m3vision.colour_turn(image, K, cam, {**cube, "yaw": (true + off) % 90})
            assert abs((turn - true + 45) % 90 - 45) <= 2 and sure > 1.5, (true, off, turn)
    image = picture(cam, 0.40, -0.02, 37)
    found = m3vision.objects(render(cam, cube=(0.40, -0.02, 37)), K, cam)
    m3vision.refine_turns(image, K, cam, found)
    assert found[0]["turn_from"] == "colour" and abs((found[0]["yaw"] - 37 + 45) % 90 - 45) < 1.5
    blank = np.zeros_like(image)                 # nothing to see: the depth's turn stands
    found = m3vision.objects(render(cam, cube=(0.40, -0.02, 37)), K, cam)
    m3vision.refine_turns(blank, K, cam, found)
    assert found[0]["turn_from"] == "depth"


def test_a_printed_marker_is_found_where_it_stands_and_which_way_it_faces():
    """The drop-off box's marker (DICT_4X4_50, 10, 64 mm), standing up and leaning back 21 deg as the
    lid does, drawn into the picture of a camera 0.31 m up looking at it from 28 cm."""
    cam = camera(0.31, 40, 0.0)
    lean = math.radians(21)
    face = np.array([-math.cos(lean), 0.0, math.sin(lean)])      # towards the camera, and up a little
    right = np.array([0.0, -1.0, 0.0])
    up = np.cross(face, right)                                     # the marker's up, along the lid
    centre = np.array([0.45, 0.02, 0.10])
    corners = [centre + 0.032 * (sx * right + sy * up) for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1))]
    inv = np.linalg.inv(cam)
    px = np.array([(K @ ((inv @ np.r_[c, 1])[:3] / (inv @ np.r_[c, 1])[2]))[:2] for c in corners], np.float32)
    art = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 10, 360)
    art = cv2.copyMakeBorder(art, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)   # the paper's white edge
    src = np.array([[60, 60], [420, 60], [420, 420], [60, 420]], np.float32)
    warp = cv2.warpPerspective(art, cv2.getPerspectiveTransform(src, px), (640, 480), borderValue=90)
    found = m3vision.find_marker(cv2.cvtColor(warp, cv2.COLOR_GRAY2BGR), K, cam, "DICT_4X4_50", 10, 0.064)
    assert found is not None
    where, axes = found
    assert np.linalg.norm(where - centre) < 0.004
    assert math.degrees(math.acos(min(1.0, float(axes[:, 2] @ face)))) < 3
    assert m3vision.find_marker(cv2.cvtColor(warp, cv2.COLOR_GRAY2BGR), K, cam, "DICT_4X4_50", 11, 0.064) is None
