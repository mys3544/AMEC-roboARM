"""roboarm/m3pick.py against a pretend arm and a depth camera rendered from the arm model: a turned
cube is picked with the fingers square to it and dropped off, a lying block is gripped across,
something out of reach is left and said so, and a stop stops the job between two moves."""

import math
import time

import numpy as np

from roboarm import m3arm, m3kinematics, m3pick

K = np.array([[480.0, 0, 320], [0, 480.0, 240], [0, 0, 1]])
HOME = {1: 90, 2: 180, 3: 5, 4: 0, 5: 90, 6: 0}


def render(camera, boxes):
    """Depth (mm) of the floor z = 0 and boxes (x, y, turn deg, length, height; 30 mm wide, top
    faces only) as `camera` (base <- optical) sees them."""
    v, u = np.mgrid[0:480, 0:640]
    rays = np.stack([(u - 320) / 480.0, (v - 240) / 480.0, np.ones_like(u, float)], axis=-1)
    d, o = rays @ camera[:3, :3].T, camera[:3, 3]
    dz = np.where(np.abs(d[..., 2]) > 1e-9, d[..., 2], -1e-9)
    t = np.where(-o[2] / dz > 0, -o[2] / dz, 0.0)
    for x, y, turn, length, tall in boxes:
        t2 = (tall - o[2]) / dz
        hit = o + d * t2[..., None]
        c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
        dx, dy = hit[..., 0] - x, hit[..., 1] - y
        on = (t2 > 0) & (np.abs(dx * c + dy * s) < length / 2) & (np.abs(-dx * s + dy * c) < 0.015)
        t = np.where(on, t2, t)
    return np.clip(t * 1000, 0, 65535).astype(np.uint16)


class World:
    """The arm as the bridge would call it, and the things on the floor."""

    def __init__(self, boxes, move_s=0.0):
        self.pose, self.boxes, self.move_s = dict(HOME), list(boxes), move_s
        self.held, self.dropped, self.calls = None, [], []

    def call(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        if method == "move_to":
            goal = {**self.pose, **{int(j): v for j, v in args[0].items()}}
            why = m3kinematics.problem(goal, kwargs.get("floor_clear") or m3kinematics.FLOOR_CLEAR)
            assert why is None, why
            time.sleep(self.move_s)
            self.pose = goal
        elif method == "follow":
            time.sleep(self.move_s)
            for pose in args[0]:
                goal = {**self.pose, **{int(j): v for j, v in pose.items()}}
                why = m3kinematics.problem(goal, kwargs.get("floor_clear") or m3kinematics.FLOOR_CLEAR)
                assert why is None, why
                self.pose = goal
        elif method in ("home", "engage"):
            self.pose = dict(HOME)
        elif method == "close_gripper":
            point, _ = m3kinematics.grip(self.pose)
            near = [b for b in self.boxes if math.hypot(b[0] - point[0], b[1] - point[1]) < 0.012]
            if near:
                self.held = near[0]
                self.boxes.remove(near[0])
        elif method == "set_gripper" and self.held and args[0] <= 40:   # opened wide: let go
            self.dropped.append(self.held)
            self.held = None
        return dict(self.pose) if method == "read" else None

    def snapshot(self):
        camera = m3kinematics.frames(self.pose)["arm4"] @ m3kinematics.camera_optical()
        return np.zeros((480, 640, 3), np.uint8), render(camera, self.boxes), K, dict(self.pose)


def run(world, job, **kwargs):
    picker = m3pick.Picker(world.call, world.snapshot, settle_s=0.0, rng=np.random.default_rng(0))
    picker.start(job, **kwargs)
    picker.wait(120)
    return picker


def closing(pose):
    y = m3kinematics.frames(pose)["arm5"][:3, 1]
    return math.degrees(math.atan2(y[1], y[0]))


def test_a_turned_cube_is_picked_square_and_dropped_off():
    world = World([(0.21, 0.03, 30, 0.03, 0.03)])
    picker = run(world, "pick_once")
    assert picker.state["error"] is None and picker.state["picked"] == 1
    assert len(world.dropped) == 1 and world.pose == HOME
    record = picker.state["last"]
    assert record["kind"] == "30 mm cube" and record["yaw_from"] == "close"
    off = (closing(record["grip_pose"]) - 30 + 45) % 90 - 45
    assert abs(off) < 4, off
    width = next(kw["width_mm"] for m, _, kw in world.calls if m == "close_gripper")
    assert width == 30


def test_a_lying_block_is_gripped_across_its_length():
    world = World([(0.21, -0.02, 20, 0.06, 0.03)])
    picker = run(world, "pick_once", kind="block")
    assert picker.state["error"] is None and picker.state["picked"] == 1
    off = (closing(picker.state["last"]["grip_pose"]) - 110 + 90) % 180 - 90   # across 20 deg: 110
    assert abs(off) < 4, off


def test_out_of_reach_is_left_and_said():
    world = World([(0.40, 0.0, 0, 0.03, 0.03)])
    picker = run(world, "pick_once")
    assert "out of reach" in picker.state["error"] and world.boxes and not world.dropped


def test_clean_up_takes_every_one_in_reach():
    world = World([(0.21, 0.05, 10, 0.03, 0.03), (0.22, -0.05, 40, 0.03, 0.03)])
    picker = run(world, "clean_up")
    assert picker.state["picked"] == 2 and not world.boxes and len(world.dropped) == 2
    assert picker.state["error"] is None and world.pose == HOME


def test_stop_stops_between_moves():
    world = World([(0.21, 0.03, 0, 0.03, 0.03)], move_s=0.2)
    picker = m3pick.Picker(world.call, world.snapshot, settle_s=0.0)
    picker.start("pick_once")
    time.sleep(0.3)
    picker.stop()
    picker.wait(30)
    assert picker.state["error"].startswith("stopped") and not picker.state["busy"]
    assert world.boxes and not picker.stopping()   # the next job may run


def test_a_stop_with_nothing_running_does_not_linger():
    world = World([])
    picker = m3pick.Picker(world.call, world.snapshot, settle_s=0.0)
    picker.stop()
    assert not picker.stopping()


def test_the_way_to_the_drop_off_carries_low_and_turns_while_rising_and_the_guard_passes_it():
    lifted = [{1: 80, 2: 76, 3: 15, 4: 19, 5: 90, 6: 134}, {1: 136, 2: 59, 3: 34, 4: 18, 5: 44, 6: 134}]
    for extra in (0.0, 0.05, 0.1):
        drop = m3pick.drop_pose(extra)
        for here in lifted:
            way = m3pick.way_to(here, drop, direct=False)
            assert way[-1] == drop and len(way) >= 2 and m3arm.glide_problem(here, way) is None
            assert way[0][2] < m3pick.FOLDED[2]   # not folded right up: a carry below it
        back = m3pick.way_to(drop, {**m3pick.LOOK, 6: 76})
        assert m3arm.glide_problem(drop, back) is None and back[-1][2] == m3pick.LOOK[2]
    deck = {1: 90, 2: 60, 3: 10, 4: 10, 5: 90}   # down onto the front deck: refused, and said why
    assert "chassis" in m3arm.glide_problem(dict(HOME), [deck])
