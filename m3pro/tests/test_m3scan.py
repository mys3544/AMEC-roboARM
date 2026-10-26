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
    assert thing is not None and thing["whole"] and thing["top_seen"] > m3scan.TOP_SEEN   # its top in view
    assert np.allclose(thing["centre"], (0.36, 0.02), atol=0.006)
    assert abs(thing["top"] - 0.05) < 0.004
    assert 0.025 < thing["radius"] < 0.04
    assert m3scan.find_thing(depth, K, cam, (0.25, 0.15)) is None   # nothing there
    tall = m3scan.find_thing(render(cam, [(0.38, 0.0, 0.0, 0.06, 0.20)]), K, cam, (0.38, 0.0))
    assert tall["whole"] and abs(tall["top"] - 0.20) < 0.004        # the first look holds a 20 cm thing


def test_the_views_are_shifted_so_the_thing_lands_on_one_point_whatever_the_wheels_drifted():
    # the thing at (0.40, 0.0); three views from round it, the odometry drifted 0, 3 and 6 cm by them;
    # each reading leans toward its camera by SIGHT_PUSH of its radius
    views, r = [], 0.03
    for k, angle in enumerate((0.0, 2.0, 4.0)):
        drift = np.array([0.03 * k, -0.01 * k])
        cam_at = np.array([0.40, 0.0]) + 0.38 * np.array([math.cos(angle), math.sin(angle)]) + drift
        thing_at = np.array([0.40, 0.0]) + drift
        sight = (thing_at - cam_at) / np.linalg.norm(thing_at - cam_at)
        camera = np.eye(4)
        camera[:2, 3] = cam_at
        views.append({"camera": camera, "seen": {"centre": thing_at - m3scan.SIGHT_PUSH * r * sight,
                                                   "radius": r, "top": 0.1}})
    views.append({"camera": views[-1]["camera"].copy(), "seen": None})   # read nothing: moves as the one before
    point, radius, top = m3scan.anchor(views)
    assert (radius, top) == (r, 0.1)
    assert np.allclose(point, (0.43, -0.01), atol=1e-9)   # the median: the middle view's drift
    for k, v in enumerate(views[:3]):   # every camera moved by its own drift, less the median's
        cam_at = np.array([0.40, 0.0]) + 0.38 * np.array([math.cos(2.0 * k), math.sin(2.0 * k)])
        assert np.allclose(v["camera"][:2, 3], cam_at + (0.03, -0.01), atol=1e-9)
    assert np.allclose(views[3]["camera"], views[2]["camera"])

    # a view that saw the thing's top whole reads it true: no push
    camera = np.eye(4)
    camera[:2, 3] = (0.02, 0.0)
    whole_top = {"camera": camera, "seen": {"centre": np.array([0.40, 0.0]), "radius": r, "top": 0.1, "top_seen": 1.0}}
    leaning = {"camera": camera.copy(), "seen": {**whole_top["seen"], "top_seen": 0.8}}
    m3scan.anchor([whole_top, leaning])
    # the two marks a push apart, the point between them: the views shifted half of it each way
    assert np.isclose(whole_top["camera"][0, 3] - leaning["camera"][0, 3], m3scan.SIGHT_PUSH * r)


def test_a_thing_the_picture_cuts_is_told_from_a_whole_one():
    # the pick's wide look, as the scan's first look until 2026-10-12: its top edge cut a 10 cm box at 42 cm
    cam = m3scan.camera(m3scan.LOOK_AT)
    cut = m3scan.find_thing(render(cam, [(0.42, 0.0, 0.0, 0.10, 0.05)]), K, cam, (0.42, 0.0))
    assert cut is not None and not cut["whole"] and cut["radius"] < 0.03   # read short, as the can was


class Round:
    """The arm as the bridge would call it, a base that goes where it is sent -- by its odometry,
    which may drift by `drift` (x, y m) more with every leg -- and one box on the floor (the floor's
    frame, where the odometry began)."""

    def __init__(self, box, drift=(0.0, 0.0)):
        self.pose, self.box, self.odom, self.trail = dict(HOME), box, (0.0, 0.0, 0.0), []
        self.drift, self.error = np.asarray(drift, float), np.zeros(2)

    def true(self):
        """Where the base really is: the odometry less its error."""
        return (self.odom[0] - self.error[0], self.odom[1] - self.error[1], self.odom[2])

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
        self.error = self.error + self.drift
        self.trail.append(self.odom)
        return self.odom

    def snapshot(self):
        x, y, turn, length, tall = self.box
        at = self.true()
        c, s = math.cos(at[2]), math.sin(at[2])
        dx, dy = x - at[0], y - at[1]
        here = (c * dx + s * dy, -s * dx + c * dy, turn - math.degrees(at[2]), length, tall)
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
    assert np.allclose(world.trail[-1], world.trail[0], atol=0.005)   # back to the first (as re-read)
    for x, y, yaw in world.trail:   # round the thing as the stops read it again: mm off the first look's
        assert abs(math.hypot(x - plan["centre"][0], y - plan["centre"][1]) - 0.38) < 0.005
    assert len(plan["centres"]) == 4
    assert world.pose == dict(zip(m3arm.JOINTS, m3arm.POSES["home"]))
    view = json.loads((folder / "00_0.json").read_text())
    assert view["thing"]["whole"] and abs(view["thing"]["top"] - 0.05) < 0.004   # each view's own reading


def test_the_scan_goes_round_the_thing_though_the_wheels_drift(tmp_path):
    # 1 cm more odometry error with every leg: 13 legs, 13 cm by the end -- the first live scan's
    # wheels made a can seem to move 9 cm in half a circle (2026-10-12)
    world = Round((0.37, -0.01, 30.0, 0.06, 0.05), drift=(0.008, 0.006))
    away = []
    goto = world.goto

    def watched(x, y, yaw):
        out = goto(x, y, yaw)
        at = world.true()
        away.append(math.hypot(at[0] - 0.37, at[1] + 0.01))
        return out
    world.goto = watched
    picker = m3pick.Picker(world.call, world.snapshot, tmp_path / "pick", settle_s=0.0,
                           odom=lambda: world.odom, goto=world.goto)
    picker.scan(radius=0.38, stops=4)
    assert max(abs(d - 0.38) for d in away) < 0.035   # one stop's 3 legs of drift at most, not 13 cm


def test_a_thing_off_the_picture_is_refused_and_a_given_size_is_kept(tmp_path):
    world = Round((0.44, 0.0, 0.0, 0.12, 0.25))   # its top face past the first look's top edge (46 cm until
    # 2026-10-26: the camera as the depth sees it, 1.4 deg steeper, sees no part of it there)
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
    assert plan["top"] == 0.08 and plan["radius"] == 0.05   # kept: not read again
    # it read ~3 cm round; the 2 cm more it is given lie behind its near side: the first circle goes
    # round a centre that much further back -- until the stop's own views have read where it stands
    first = world.trail[0]   # 0.38 straight back from that centre
    assert 0.37 + 0.012 < first[0] + 0.38 < 0.37 + 0.026
    assert np.allclose(plan["centre"], (0.37, -0.01), atol=0.008)


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
    # 4.8 mm off across since 2026-10-26 (2.1 before): the camera 1.4 deg steeper, as the depth sees it, and
    # SIGHT_PUSH, the per-stop re-read's correction, fitted with the old one on a real can
    assert np.allclose(top[:, :2].mean(axis=0), (0.37, -0.01), atol=0.006)
