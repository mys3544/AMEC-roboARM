"""roboshared.box and roboshared.task against a pretend robot with nothing of either real robot in it:
the flow's own decisions (the box, the line, the misses, the stop, what the console reads)."""

import math
import time
from typing import ClassVar

import numpy as np
import pytest

from roboshared import box as boxes
from roboshared import task


def marker_axes(turn_deg: float, lean_deg: float = 0.0) -> np.ndarray:
    """A marker standing up facing the robot (its face along -x turned by `turn_deg`), leaning back."""
    t, lean = math.radians(turn_deg), math.radians(lean_deg)
    face = np.array([-math.cos(t) * math.cos(lean), -math.sin(t) * math.cos(lean), math.sin(lean)])
    up = np.array([math.cos(t) * math.sin(lean), math.sin(t) * math.sin(lean), math.cos(lean)])
    right = np.cross(up, face)
    return np.column_stack([right, up, face])


def test_the_box_is_the_same_box_on_any_floor():
    """The hinge is where the lid meets the walls' top, wherever the robot's floor is (the M3's at 0,
    the X3's 0.19 m below its base plate)."""
    axes = marker_axes(0.0, lean_deg=20.0)
    on_m3 = boxes.box_from_marker(np.array([0.30, 0.0, 0.10]), axes)
    on_x3 = boxes.box_from_marker(np.array([0.30, 0.0, 0.10 - 0.19]), axes, floor_z=-0.19)
    assert np.allclose(on_m3["centre"], on_x3["centre"]) and np.allclose(on_m3["into"], on_x3["into"])
    assert on_m3["into"][0] < -0.99, "into the box: back towards the robot"
    for spot in boxes.box_spots(on_m3):
        assert boxes.inside(on_m3, spot)
    assert all(abs(c[2] - (boxes.BOX_INSIDE[2] - 0.19)) < 1e-12 for c in boxes.box_corners(on_x3, floor_z=-0.19))
    moved = {**on_m3, "centre": on_m3["centre"] + [0.04, 0.0]}
    assert boxes.same_box(on_m3, on_m3) and not boxes.same_box(moved, on_m3)


def test_a_box_standing_still_is_the_mean_of_its_readings():
    """2026-10-16: the M3's readings of a box standing still scattered +-1.5 cm and +-3 deg, and each
    drop was aimed by the last one, "moved" nearly every time. Within 3 cm and 8 deg it is the same box,
    and the job aims by the mean."""
    readings = [boxes.box_from_marker(np.array([0.30 + dx, dy, 0.10]), marker_axes(turn, lean_deg=20.0))
                for dx, dy, turn in ((0.012, -0.008, 183.0), (-0.010, 0.006, 177.0), (0.001, 0.002, 180.5))]
    assert all(boxes.same_box(r, readings[0]) for r in readings)
    mean = boxes.merged(readings)
    assert np.allclose(mean["centre"], np.mean([r["centre"] for r in readings], axis=0))
    unwrapped = [readings[0]["turn"] + (r["turn"] - readings[0]["turn"] + 180) % 360 - 180 for r in readings]
    assert abs((mean["turn"] - np.mean(unwrapped) + 180) % 360 - 180) < 0.05   # the mean, round the +-180 seam
    assert np.allclose(mean["into"], readings[0]["into"], atol=0.06) and abs(mean["along"] @ mean["into"]) < 1e-12


def thing(x, y, kind="cube", size=0.03):
    return {"centre": np.array([x, y, size / 2]), "kind": kind, "size": size, "length": size, "height": size,
            "top": size, "footprint": (size, size), "yaw": 0.0, "period": 90, "whole": True, "reach": True}


class Pretend(task.Task):
    """A robot that finds what its list says, picks what is in reach, and lets go anywhere."""

    SETTINGS: ClassVar[dict] = {"speed": ("", "%", 30, 150, 5, 100), "free_dps": ("", "", 1, 200, 1, 90),
                "near_dps": ("", "", 1, 200, 1, 60), "gripper_dps": ("", "", 1, 300, 1, 150),
                "settle_close_s": ("", "s", 0, 2, 0.05, 0.0), "release_s": ("", "s", 0, 1, 0.05, 0.0),
                "drive_cm_s": ("", "", 1, 10, 0.5, 5), "driven_s": ("", "s", 0, 2, 0.05, 0.0),
                "nudge_cm": ("", "cm", 1, 15, 1, 10), "right_cm": ("", "cm", 0, 15, 1, 10)}
    DROP_OUT, DROP_SPREAD = 0.19, 0.10
    BOX_LOOKS = (180, 90, 0)

    def __init__(self, table, box_at=None, misses=0, **kw):
        super().__init__(call=lambda m, *a, **k: {1: 90} if m == "read" else None, snapshot=None,
                         settle_s=0.0, rng=np.random.default_rng(0), **kw)
        self.table, self.box_at, self.misses, self.moves = table, box_at, misses, []

    def engaged(self):
        self.moves.append("engage")

    def home(self):
        self.moves.append("home")

    def wide_look(self, sides=False):
        found = [o for o in self.table if not self.dropped_here(o)]
        self.found = found
        return found

    def plan(self, thing, around):
        return "plan" if float(np.hypot(*thing["centre"][:2])) < 0.25 else None

    def grip(self, target, planned, found):
        self.moves.append(("grip", tuple(np.round(target["centre"][:2], 3))))
        if self.misses:
            self.misses -= 1
        else:
            self.table = [o for o in self.table if o is not target]
        return {"reading": target, "centre": target["centre"], "line": ["down"], "record": {"how": "pretend"}}

    def view(self):
        return np.zeros((10, 10, 3), np.uint8), np.eye(3), np.eye(4), None

    def go_box_look(self, j1, holding):
        self.moves.append(("box look", j1))
        self.looking_at = j1

    def to_box_view(self):
        self.looking_at = self.box_dir

    def see_box(self, tag):   # no picture to find a marker in: the box is where the test put it, at its J1
        if self.box_at is None or self.looking_at != self.box_at[0]:
            return None
        return {"centre": np.array(self.box_at[1]), "along": np.array([0.0, 1.0]), "into": np.array([1.0, 0.0]),
                "turn": 90.0, "items": []}

    def way_into(self, here, spot):
        return [("over", tuple(np.round(spot, 3)))]

    def carry(self, way):
        self.moves.append(("carry", way[-1]))

    def drop_on_line(self, extra):
        self.moves.append(("line", round(extra, 3)))

    def let_go(self):
        self.moves.append("let go")

    def put_back(self, back):
        self.moves.append(("put back", back[1]))


def run(job, robot, **kw):
    robot.start(job, **kw)
    robot.wait(10)
    return robot.status()


def test_clean_up_takes_what_is_in_reach_into_the_box_and_says_why_it_stopped():
    robot = Pretend([thing(0.18, 0.02), thing(0.20, -0.05), thing(0.40, 0.0)], box_at=(90, (-0.05, 0.30)))
    state = run("clean_up", robot)
    assert state["picked"] == 2 and state["missed"] == 0 and state["box"]["drops"] == 2
    assert [m for m in robot.moves if isinstance(m, tuple) and m[0] == "box look"] == [("box look", 180), ("box look", 90)]
    assert sum(1 for m in robot.moves if isinstance(m, tuple) and m[0] == "carry") == 2
    assert robot.moves[-1] == "home" and state["job"] == "clean_up" and not state["busy"]
    assert any("clean-up done" in e["text"] for e in robot.log())
    assert state["last"]["how"] == "pretend" and state["last"]["picked"]


def test_with_no_box_the_drops_go_on_the_line_and_are_not_picked_again():
    robot = Pretend([thing(0.18, 0.02)])
    robot.box_dir = 180
    state = run("pick_once", robot)
    assert state["picked"] == 1 and robot.line_drops == 1
    assert any(isinstance(m, tuple) and m[0] == "line" for m in robot.moves)
    robot.table = [thing(float(robot.AXIS[0]), -0.22)]   # where it fell, on the line
    assert robot.dropped_here(robot.table[0])


def test_what_reads_by_the_box_walls_is_the_box_not_a_piece():
    """2026-10-21, the X3: a "30 mm cube" read 3.6 cm past the box's inside by its wall towards the robot
    (the box at x 0.140 y -0.191, turned -172 deg) was picked -- air. A piece further off is still one."""
    robot = Pretend([])
    t = math.radians(-172.0)
    along = np.array([math.cos(t), math.sin(t)])
    robot.box = {"centre": np.array([0.140, -0.191]), "along": along, "into": np.array([along[1], -along[0]]),
                 "turn": -172.0, "items": []}
    assert robot.dropped_here(thing(0.115, -0.103))
    assert not robot.dropped_here(thing(0.115, -0.07))


def test_a_sort_puts_the_blocks_on_the_line_and_the_cubes_into_the_box():
    robot = Pretend([thing(0.18, 0.02), thing(0.20, 0.08, kind="block"), thing(0.16, 0.12)], box_at=(90, (-0.05, 0.30)))
    state = run("sort", robot)
    assert state["picked"] == 3 and state["box"]["drops"] == 2 and robot.line_drops == 1 and not state["error"]
    lines = [m for m in robot.moves if isinstance(m, tuple) and m[0] == "line"]
    carries = [m for m in robot.moves if isinstance(m, tuple) and m[0] == "carry"]
    assert len(lines) == 1 and len(carries) == 2 and robot.to_line is None
    assert robot.dropped_here(thing(0.0, -0.22, kind="block"))   # where the block fell: not picked again
    assert len(robot.obstacles()) == 2   # the base keeps off both


def test_a_sort_is_refused_without_a_box_or_with_the_box_on_the_line():
    robot = Pretend([thing(0.18, 0.02)])
    state = run("sort", robot)
    assert "cannot sort: no drop-off box" in state["error"] and state["picked"] == 0 and robot.moves[-1] == "home"
    robot = Pretend([thing(0.18, 0.02)], box_at=(180, (0.0, -0.30)))
    state = run("sort", robot)
    assert "stands where the line" in state["error"] and state["picked"] == 0
    assert run("clean_up", robot)["picked"] == 1   # a clean-up minds no line: everything into the box


def test_a_spot_missed_twice_is_left_and_the_job_goes_on():
    robot = Pretend([thing(0.18, 0.02), thing(0.21, -0.04)], misses=2)
    state = run("clean_up", robot)
    assert state["missed"] == 2 and state["picked"] == 1
    assert any("missed twice" in e["text"] for e in robot.log())


class Shoving(Pretend):
    """Its grip shoves the target `by` (x, y m) instead of taking it, once (None: takes it); its middle
    look (wide_look()) sees only what lies within MIDDLE of y 0, its look round everything."""

    MIDDLE = 0.10

    def __init__(self, table, by=None, **kw):
        super().__init__(table, **kw)
        self.by, self.looks = by, []

    def grip(self, target, planned, found):
        out = super().grip(target, planned, found)
        if self.by is not None:
            self.table.append({**target, "centre": target["centre"] + np.r_[self.by, 0.0]})
            self.by = None
        return out

    def wide_look(self, sides=False):
        self.looks.append("round" if sides else "middle")
        seen = super().wide_look(sides)
        return seen if sides else [o for o in seen if abs(o["centre"][1]) <= self.MIDDLE]

    def round_sees_more(self, spot, radius):
        return abs(spot[1]) + radius > self.MIDDLE


def test_a_piece_shoved_off_its_spot_is_not_picked():
    """2026-10-23: a 30 mm cube shoved 4.3 cm out was counted picked -- its spot was clear."""
    robot = Shoving([thing(0.18, 0.02)], by=(0.043, 0.0))
    state = run("pick_once", robot)
    assert state["picked"] == 0 and state["missed"] == 1
    assert state["last"]["picked"] is False and state["last"]["pushed_to"] == [0.223, 0.02]
    assert any("shoved 4.3 cm" in e["text"] for e in robot.log())
    assert robot.looks == ["round", "middle"]   # the middle look after saw it: no look round


def test_a_neighbour_left_where_it_was_is_not_the_shoved_target():
    robot = Shoving([thing(0.18, 0.02), thing(0.18, 0.07)])   # 5 cm apart, as in 2026-10-23's round B
    state = run("pick_once", robot)
    assert state["picked"] == 1 and state["missed"] == 0 and "pushed_to" not in state["last"]


def test_where_the_look_after_may_miss_a_shoved_piece_it_looks_round():
    robot = Shoving([thing(0.18, 0.06)], by=(0.0, 0.06))   # shoved out of the middle look
    state = run("pick_once", robot)
    assert state["missed"] == 1 and state["last"]["pushed_to"] == [0.18, 0.12]
    assert robot.looks == ["round", "middle", "round"]
    robot = Shoving([thing(0.18, 0.06)])   # near the middle look's edge, taken: picked, after a look round
    assert run("pick_once", robot)["picked"] == 1 and robot.looks == ["round", "middle", "round"]
    robot = Shoving([thing(0.18, 0.0)])    # in the middle: its ring all in the middle look, no look round
    assert run("pick_once", robot)["picked"] == 1 and robot.looks == ["round", "middle"]


def test_a_box_gone_at_the_drop_means_what_is_held_goes_back():
    robot = Pretend([thing(0.18, 0.02)], box_at=(180, (0.0, -0.30)))

    def gone(*_a):
        robot.box_at = None   # moved away while the thing was being picked
    robot.drive_back = gone
    state = run("pick_once", robot)
    assert ("put back", ["down"]) in robot.moves
    assert "not to be found" in state["error"]


def test_a_stop_stops_the_job_and_frees_the_robot():
    robot = Pretend([thing(0.18, 0.02)])

    def slow(*_a, **_k):
        for _ in range(50):
            robot.check()
            time.sleep(0.01)
    robot.grip = slow
    robot.start("pick_once")
    time.sleep(0.05)
    robot.stop()
    robot.wait(5)
    state = robot.status()
    assert state["error"].startswith("stopped") and not state["busy"] and not robot.stopping()


def test_settings_are_held_to_their_range_and_unknown_ones_refused():
    robot = Pretend([])
    robot.set_settings({"speed": 500})
    assert robot.cfg["speed"] == 150 and robot.dps("free_dps") == 90 * 1.5
    with pytest.raises(ValueError):
        robot.set_settings({"nope": 1})
