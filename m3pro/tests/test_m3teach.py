"""roboarm/m3teach.py: the box round a thing in a view, the views round it, a thing named by the
detection that covers it, and the teach and the pick by name against a pretend arm and model."""

import math

import numpy as np
import pytest
from test_m3pick import HOME, K, render

from roboarm import m3kinematics, m3pick, m3scan, m3teach


def test_the_box_is_round_the_thing_in_the_picture():
    cam = m3scan.camera(m3scan.LOOK_AT)
    depth = render(cam, [(0.22, 0.03, 0.0, 0.03, 0.03), (0.30, -0.08, 0.0, 0.03, 0.03)])
    box = m3teach.box_around(depth, K, cam, (0.22, 0.03), 0.03)
    centre = np.linalg.inv(cam) @ np.r_[0.22, 0.03, 0.03, 1.0]
    u, v = K[0, 0] * centre[0] / centre[2] + K[0, 2], K[1, 1] * centre[1] / centre[2] + K[1, 2]
    assert box[0] < u < box[2] and box[1] < v < box[3]
    assert box[2] - box[0] < 120 and box[3] - box[1] < 120   # the one thing, not both
    assert m3teach.box_around(depth, K, cam, (0.40, 0.15), 0.03) is None


def test_the_views_look_at_the_thing_from_round_it():
    poses = m3teach.views_of((0.22, 0.0), 0.025, 0.03)
    assert len(poses) >= 3
    assert len({p[1] for p in poses}) >= 2   # from more than one J1
    straight = 90 - round(math.degrees(math.atan2(0.0, 0.22 - m3kinematics.CHAIN[0][0][0])))
    for pose in poses:
        cam = m3scan.camera(pose)
        ray = np.array([0.22, 0.0, 0.015]) - cam[:3, 3]
        off = math.degrees(math.acos(ray @ cam[:3, 2] / np.linalg.norm(ray)))
        assert off < (4 if pose[1] == straight else 25)   # J1 turned: off the middle, still in view
        assert m3kinematics.problem(pose) is None


def test_a_thing_takes_the_name_of_the_surest_detection_covering_it():
    found = [{"outline_px": [[100, 100], [140, 100], [140, 140], [100, 140]], "centre": np.zeros(3)},
             {"outline_px": [[300, 300], [340, 300], [340, 340], [300, 340]], "centre": np.zeros(3)},
             {"outline_px": [[500, 50], [540, 50], [540, 90], [500, 90]], "centre": np.zeros(3)}]
    detections = [{"name": "apricot", "conf": 0.8, "box": [98, 99, 141, 142]},
                  {"name": "news", "conf": 0.5, "box": [101, 98, 139, 140]},     # the same thing, less sure
                  {"name": "news", "conf": 0.6, "box": [298, 301, 342, 338]},
                  {"name": "apricot", "conf": 0.2, "box": [500, 50, 540, 90]}]   # too unsure
    named = m3teach.name_things(found, detections)
    assert [o["taught"] for o in named] == ["apricot", "news", None]
    assert named[0]["sure"] == 0.8


class Table:
    """The arm as the bridge calls it, a box on the table, and a pretend teach model."""

    def __init__(self, things):
        self.pose, self.things, self.taught, self.embedded = dict(HOME), things, {}, []

    def call(self, method, *args, **kwargs):
        if method in ("move_to", "follow"):
            for goal in ([args[0]] if method == "move_to" else args[0]):
                goal = {**self.pose, **{int(j): v for j, v in goal.items()}}
                assert m3kinematics.problem(goal) is None
                self.pose = goal
        elif method == "engage":
            self.pose = dict(HOME)
        return dict(self.pose) if method == "read" else None

    def snapshot(self):
        return np.zeros((480, 640, 3), np.uint8), render(m3scan.camera(self.pose), self.things), K, dict(self.pose)

    def ask(self, path, payload=None, timeout=30.0):
        if path == "/embed":
            self.embedded.append(payload["boxes"][0])
            return {"embeddings": [list(np.eye(4)[0] + 0.1 * len(self.embedded))]}
        if path == "/teach":
            self.taught[payload["name"]] = payload["embedding"]
            return {"names": list(self.taught)}
        raise AssertionError(path)


def test_the_teach_looks_from_round_the_thing_and_stores_the_average(tmp_path, monkeypatch):
    table = Table([(0.22, 0.01, 20.0, 0.03, 0.03)])
    monkeypatch.setattr(m3teach, "ask", table.ask)
    picker = m3pick.Picker(table.call, table.snapshot, tmp_path / "pick", settle_s=0.0)
    picker.teach("apricot")
    assert list(table.taught) == ["apricot"] and len(table.embedded) >= 3
    assert abs(np.linalg.norm(table.taught["apricot"]) - 1) < 1e-6
    assert len(list((tmp_path / "teach" / "apricot").glob("*_rgb.jpg"))) == len(table.embedded)
    with pytest.raises(RuntimeError, match="name"):
        picker.teach("  ")
