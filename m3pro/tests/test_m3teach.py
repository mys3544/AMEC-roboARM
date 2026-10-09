"""roboarm/m3teach.py: the box round a thing in a view, the views round it, a thing named by the
detection that covers it, and the teach and the pick by name against a pretend arm and model."""


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
    poses = m3teach.views_of((0.20, -0.03), 0.03, 0.03)
    assert len(poses) >= 4
    assert len({p[1] for p in poses}) >= 3                    # from round it on J1
    assert any(p[2] == m3scan.LOOK_AT[2] for p in poses)      # find's own look among them
    assert any(p[2] != m3scan.LOOK_AT[2] for p in poses)      # and from over it
    for pose in poses:
        assert m3teach.in_view(pose, (0.20, -0.03, 0.015))
        assert m3kinematics.problem(pose) is None


def test_a_thing_is_named_only_when_its_views_agree_and_no_other_name_comes_close():
    found = [{"outline_px": [[100, 100], [140, 140]], "centre": np.zeros(3)} for _ in range(3)]
    identified = [{"name": "apricot", "sure": 0.78, "margin": 0.12},
                  {"name": "news", "sure": 0.74, "margin": 0.02},      # cosmetics nearly as like it
                  {"name": "fish", "sure": 0.52, "margin": 0.10}]      # like nothing taught
    named = m3teach.name_things(found, identified)
    assert [o["taught"] for o in named] == ["apricot", None, None]
    assert m3teach.box_of(found[0]) == [96, 96, 144, 144]


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
