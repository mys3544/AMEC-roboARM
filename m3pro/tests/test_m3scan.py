"""roboarm/m3scan.py: the stops go round the thing facing it, a stop's views look at it from a few
heights and never nearer than the depth starts, a box ahead is found and measured, and the scan job
drives round it with a pretend base and keeps every view."""

import json
import math

import cv2
import numpy as np
import pytest
from test_m3pick import HOME, K, render

from roboarm import m3arm, m3kinematics, m3pick, m3scan


def test_the_stops_go_round_the_thing_each_as_far_and_facing_it():
    plan = m3scan.stops((0.0, 0.0, 0.0), (0.38, 0.0), 12)
    assert len(plan) == 12
    assert np.allclose(plan[0], (0.0, 0.0, 0.0), atol=1e-9)   # the first: where the robot is, if right
    for x, y, yaw in plan:
        assert math.hypot(x - 0.38, y) == np.float64(0.38) or abs(math.hypot(x - 0.38, y) - 0.38) < 1e-9
        ahead = 0.38 * np.array([math.cos(yaw), math.sin(yaw)])
        assert np.allclose((x, y) + ahead, (0.38, 0.0), atol=1e-9)   # facing it
    assert plan[3][1] < -0.3   # anticlockwise from above: a quarter round, the robot is on its right


def test_a_stop_looks_at_the_thing_from_a_few_heights_never_nearer_than_the_depth_starts():
    views = m3scan.stop_views(0.03, 0.06, 0.38)
    assert len(views) >= 2
    middle = np.array([0.38, 0.0, 0.03])
    heights = []
    for pose in views:
        cam = m3scan.camera(pose)
        ray = middle - cam[:3, 3]
        assert math.degrees(math.acos(ray @ cam[:3, 2] / np.linalg.norm(ray))) < 4     # on it
        assert np.linalg.norm(ray) - math.hypot(0.03, 0.03) >= m3scan.NEAREST - 0.002   # the depth sees it
        assert m3kinematics.problem(pose) is None and m3scan.clear_of(pose, (0.38, 0.0), 0.03, 0.06)
        heights.append(cam[2, 3])
    assert max(heights) - min(heights) > 0.02


def test_a_box_ahead_is_found_and_measured():
    cam = m3scan.camera(m3scan.LOOK_FIRST)
    depth = render(cam, [(0.36, 0.02, 0.0, 0.06, 0.05)])   # 60 x 30 mm, 50 mm tall (its top face)
    thing = m3scan.find_thing(depth, K, cam, (0.38, 0.0))
    assert thing is not None and thing["whole"]
    assert np.allclose(thing["centre"], (0.36, 0.02), atol=0.006)
    assert abs(thing["top"] - 0.05) < 0.004
    assert 0.025 < thing["radius"] < 0.04
    assert m3scan.find_thing(depth, K, cam, (0.25, 0.15)) is None   # nothing there
    tall = m3scan.find_thing(render(cam, [(0.38, 0.0, 0.0, 0.06, 0.20)]), K, cam, (0.38, 0.0))
    assert tall["whole"] and abs(tall["top"] - 0.20) < 0.004        # the first look holds a 20 cm thing


def test_a_thing_the_picture_cuts_is_told_from_a_whole_one():
    # the pick's wide look, as the scan's first look until 2026-10-12: its top edge cut a 10 cm box at 42 cm
    cam = m3scan.camera(m3scan.LOOK_AT)
    cut = m3scan.find_thing(render(cam, [(0.42, 0.0, 0.0, 0.10, 0.05)]), K, cam, (0.42, 0.0))
    assert cut is not None and not cut["whole"] and cut["radius"] < 0.03   # read short, as the can was


class Round:
    """The arm as the bridge would call it, a base that goes exactly where it is sent, and one box
    on the floor (odometry frame)."""

    def __init__(self, box):
        self.pose, self.box, self.odom, self.trail = dict(HOME), box, (0.0, 0.0, 0.0), []

    def call(self, method, *args, **kwargs):
        if method in ("move_to", "follow"):
            for goal in ([args[0]] if method == "move_to" else args[0]):
                goal = {**self.pose, **{int(j): v for j, v in goal.items()}}
                assert m3kinematics.problem(goal) is None
                self.pose = goal
        elif method == "engage":
            self.pose = dict(HOME)
        return dict(self.pose) if method == "read" else None

    def goto(self, x, y, yaw):
        assert math.hypot(x - self.odom[0], y - self.odom[1]) <= 0.30   # the bridge's GOTO_MOST
        self.odom = (x, y, yaw)
        self.trail.append(self.odom)
        return self.odom

    def snapshot(self):
        x, y, turn, length, tall = self.box
        c, s = math.cos(self.odom[2]), math.sin(self.odom[2])
        dx, dy = x - self.odom[0], y - self.odom[1]
        here = (c * dx + s * dy, -s * dx + c * dy, turn - math.degrees(self.odom[2]), length, tall)
        depth = render(m3scan.camera(self.pose), [here])
        return np.zeros((480, 640, 3), np.uint8), depth, K, dict(self.pose)


def test_the_scan_drives_round_the_thing_and_keeps_every_view(tmp_path):
    world = Round((0.37, -0.01, 30.0, 0.06, 0.05))
    picker = m3pick.Picker(world.call, world.snapshot, tmp_path / "pick", settle_s=0.0,
                           odom=lambda: world.odom, goto=world.goto)
    picker.scan(radius=0.38, stops=4)
    folder = next((tmp_path / "scan").iterdir())
    plan = json.loads((folder / "scan.json").read_text())
    assert np.allclose(plan["centre"], (0.37, -0.01), atol=0.008)
    views = sorted(folder.glob("*.json"))
    assert len(views) >= 4 * 2 + 1   # scan.json, and two views or more at each of the four stops
    assert len(world.trail) == 1 + 4 * 3   # onto the circle, then round it 30 deg a go
    assert np.allclose(world.trail[-1], world.trail[0])   # back to the first
    for x, y, yaw in world.trail:
        assert abs(math.hypot(x - plan["centre"][0], y - plan["centre"][1]) - 0.38) < 1e-6
    assert world.pose == dict(zip(m3arm.JOINTS, m3arm.POSES["home"]))


def test_a_thing_off_the_picture_is_refused_and_a_given_size_is_kept(tmp_path):
    world = Round((0.46, 0.0, 0.0, 0.12, 0.25))   # its top face past the first look's top edge
    picker = m3pick.Picker(world.call, world.snapshot, tmp_path / "pick", settle_s=0.0,
                           odom=lambda: world.odom, goto=world.goto)
    with pytest.raises(RuntimeError, match="give its height and width"):
        picker.scan(radius=0.38, stops=4)
    assert world.trail == []   # the base did not move

    world = Round((0.37, -0.01, 30.0, 0.06, 0.05))
    picker = m3pick.Picker(world.call, world.snapshot, tmp_path / "pick", settle_s=0.0,
                           odom=lambda: world.odom, goto=world.goto)
    picker.scan(radius=0.38, stops=4, height=0.08, width=0.10)
    plan = json.loads((max((tmp_path / "scan").iterdir()) / "scan.json").read_text())
    assert plan["top"] == 0.08 and plan["radius"] == 0.05
    # it read ~3 cm round; the 2 cm more it is given lie behind its near side, away from the robot
    assert 0.37 + 0.012 < plan["centre"][0] < 0.37 + 0.026


def test_the_views_fuse_into_the_box_where_it_stands(tmp_path):
    pytest.importorskip("open3d")   # the laptop's scan group; not on the robot
    world = Round((0.37, -0.01, 30.0, 0.06, 0.05))
    m3pick.Picker(world.call, world.snapshot, tmp_path / "pick", settle_s=0.0,
                  odom=lambda: world.odom, goto=world.goto).scan(radius=0.38, stops=6)
    folder = next((tmp_path / "scan").iterdir())
    plan = json.loads((folder / "scan.json").read_text())
    views = []
    for meta in sorted(folder.glob("[0-9][0-9]_*.json")):
        v = json.loads(meta.read_text())
        depth = cv2.imread(str(meta.with_name(meta.stem + "_depth.png")), cv2.IMREAD_UNCHANGED)
        pose = {int(j): a for j, a in v["pose"].items()}
        views.append({"rgb": np.full((480, 640, 3), 128, np.uint8), "depth": depth, "K": K,
                      "camera": m3scan.world_of(v["odom"]) @ m3scan.levelled(depth, K, pose)})
    xyz, _rgb, _mesh, _cameras = m3scan.fuse(views, plan["centre"], plan["radius"], plan["top"])
    top = xyz[xyz[:, 2] > 0.04]
    assert len(top) > 500
    assert abs(np.median(top[:, 2]) - 0.05) < 0.002
    assert np.allclose(top[:, :2].mean(axis=0), (0.37, -0.01), atol=0.004)
