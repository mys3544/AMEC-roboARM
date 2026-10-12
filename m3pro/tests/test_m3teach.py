"""roboarm/m3teach.py: the box round a thing in a view, the views round it, a thing named by the
detection that covers it, and the teach and the pick by name against a pretend arm and model; the
teach from a scan's views all round the thing."""


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


def test_a_scan_with_a_name_teaches_the_thing_from_its_views_all_round(tmp_path, monkeypatch):
    from test_m3scan import Round

    world = Round((0.37, -0.01, 30.0, 0.06, 0.05))
    embedded, taught = [], {}

    def ask(path, payload=None, timeout=30.0):
        if path == "/embed":
            embedded.append(payload["boxes"][0])
            return {"embeddings": [list(np.eye(4)[len(embedded) % 4] + 1.0)]}
        if path == "/teach":
            taught[payload["name"]] = payload["embedding"]
            return {"names": list(taught)}
        raise AssertionError(path)
    monkeypatch.setattr(m3teach, "ask", ask)
    picker = m3pick.Picker(world.call, world.snapshot, tmp_path / "pick", settle_s=0.0,
                           odom=lambda: world.odom, goto=world.goto)
    with pytest.raises(RuntimeError, match="name"):
        picker.scan(radius=0.38, stops=4, name="  ")
    assert world.trail == []   # refused before the drive

    picker.scan(radius=0.38, stops=4, name="jetson box")
    scan = next((tmp_path / "scan").iterdir())
    views = list(scan.glob("[0-9][0-9]_*_rgb.jpg"))
    kept = list((tmp_path / "teach" / "jetson box").glob("*_rgb.jpg"))
    assert len(views) >= 8 and len(kept) == len(views) == len(embedded)   # every view, from all round
    assert list(taught) == ["jetson box"] and abs(np.linalg.norm(taught["jetson box"]) - 1) < 1e-6
    for box in embedded:
        assert 0 <= box[0] < box[2] < 640 and 0 <= box[1] < box[3] < 480

    driven = len(world.trail)   # a scan already made: taught again, the base still
    picker.teach_scan("box again")
    assert len(world.trail) == driven
    assert len(list((tmp_path / "teach" / "box again").glob("*_rgb.jpg"))) == len(views)
    with pytest.raises(RuntimeError, match="no scan 20990101_000000"):
        picker.teach_scan("x", stamp="20990101_000000")


def test_find_looks_again_from_nearer_at_what_the_wide_look_could_not_name(tmp_path, monkeypatch):
    table = Table([(0.24, 0.05, 0.0, 0.03, 0.03), (0.30, -0.07, 30.0, 0.03, 0.03)])
    looks = []

    def ask(path, payload=None, timeout=30.0):   # unsure from afar, sure from near
        assert path == "/identify"
        looks.append((dict(table.pose), len(payload["boxes"])))
        sure, margin = (0.60, 0.02) if len(looks) == 1 else (0.80, 0.20)
        return {"things": [{"name": "news", "sure": sure, "margin": margin, "scores": {}} for _ in payload["boxes"]]}
    monkeypatch.setattr(m3teach, "ask", ask)
    picker = m3pick.Picker(table.call, table.snapshot, tmp_path / "pick", settle_s=0.0)
    named = m3teach.find(picker)
    assert len(named) == 2 and all(o["taught"] == "news" and o["closer"] for o in named)
    assert looks[0][1] == 2 and [n for _pose, n in looks[1:]] == [1, 1]   # the wide look, then one each
    for (pose, _n), o in zip(looks[1:], sorted(named, key=lambda o: float(np.hypot(*o["centre"][:2]))), strict=True):
        assert pose[2] != m3scan.LOOK_AT[2] and m3kinematics.problem(pose) is None
        for x in named:   # clear of every thing on the table, not just the one looked at
            assert m3scan.clear_of(pose, x["centre"][:2], m3teach.reach_of(x), x["top"])
    assert {j: table.pose[j] for j in m3scan.LOOK_AT} == m3scan.LOOK_AT   # back to the table in view

    looks.clear()   # named from afar: no nearer look
    monkeypatch.setattr(m3teach, "ask", lambda path, payload=None, timeout=30.0: (looks.append(1), {"things": [
        {"name": "news", "sure": 0.8, "margin": 0.2, "scores": {}} for _ in payload["boxes"]]})[1])
    assert all(o["taught"] == "news" and not o.get("closer") for o in m3teach.find(picker)) and len(looks) == 1
