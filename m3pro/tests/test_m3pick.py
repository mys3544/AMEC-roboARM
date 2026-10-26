"""roboarm/m3pick.py against a pretend arm and a depth camera rendered from the arm model: a turned
cube is picked with the fingers square to it and dropped off, a lying block is gripped across,
something out of reach is left and said so, and a stop stops the job between two moves."""

import itertools
import math
import time

import numpy as np
import pytest
from roboshared.box import outline

from roboarm import m3aim, m3arm, m3kinematics, m3pick

K = np.array([[480.0, 0, 320], [0, 480.0, 240], [0, 0, 1]])
HOME = {1: 90, 2: 180, 3: 5, 4: 0, 5: 90, 6: 0}


@pytest.fixture(autouse=True)
def arm_as_modelled(monkeypatch):
    """The pretend arm goes exactly where the model says: nothing taught (the real one's m3aim)."""
    monkeypatch.setattr(m3aim, "TAUGHT", ())


def test_the_aim_moves_a_thing_out_and_left_of_where_it_was_seen():
    things = [{"centre": np.array([0.20, 0.0, 0.015])}, {"centre": np.array([0.0, 0.20, 0.015])}]
    m3pick.aimed(things, (0.0, 0.0), 14, 7)
    assert np.allclose(things[0]["centre"], [0.214, 0.007, 0.015])   # ahead: out = +x, left = +y
    assert np.allclose(things[1]["centre"], [-0.007, 0.214, 0.015])  # on the left: out = +y, left = -x
    assert np.allclose(things[0]["seen"], [0.20, 0.0, 0.015])


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
        self.held, self.dropped, self.calls, self.driven, self.released = None, [], [], [], []
        self.base = np.zeros(2)   # where the base is from where it began
        self.refuse = None        # a spot the arm refuses to come down to
        self.slip = []            # spots where the fingers shut on the thing and it is not held
        self.dropped_at = []
        self.opened_to = []       # the gripper's angle at each let-go

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
            last = {**self.pose, **{int(j): v for j, v in args[0][-1].items()}}
            point, _ = m3kinematics.grip(last)
            if self.refuse is not None and np.hypot(*(point[:2] - self.refuse)) < 0.02 and point[2] < 0.06:
                raise RuntimeError("follow: refused: as the arm's guard would")
            for pose in args[0]:
                goal = {**self.pose, **{int(j): v for j, v in pose.items()}}
                why = m3kinematics.problem(goal, kwargs.get("floor_clear") or m3kinematics.FLOOR_CLEAR)
                assert why is None, why
                if self.held and goal.get(6, 180) < 110:   # opened wider than 40 mm: it falls where it is
                    point, _ = m3kinematics.grip(goal)
                    self.boxes.append((float(point[0]), float(point[1]), *self.held[2:]))
                    self.held = None
                self.pose = goal
        elif method in ("home", "engage"):
            self.pose = dict(HOME)
        elif method == "close_gripper":
            goal = {**self.pose, 6: m3arm.gripper_angle(kwargs["width_mm"])}
            why = m3kinematics.problem(goal, kwargs.get("floor_clear") or m3kinematics.FLOOR_CLEAR)
            assert why is None, why   # as the arm's own move_to checks it
            self.pose[6] = goal[6]
            point, _ = m3kinematics.grip(self.pose)
            near = [b for b in self.boxes if math.hypot(b[0] - point[0], b[1] - point[1]) < 0.012
                    and all(math.hypot(b[0] - x, b[1] - y) > 0.02 for x, y in self.slip)]
            if near:
                self.held, self.shut_mm = near[0], kwargs["width_mm"]
                self.boxes.remove(near[0])
        elif method == "set_gripper":
            why = m3kinematics.problem({**self.pose, 6: args[0]}, kwargs.get("floor_clear") or m3kinematics.FLOOR_CLEAR)
            assert why is None, why   # as the arm's own move_to checks it (2026-10-05: the put-back's open, refused)
            self.pose[6] = args[0]
            if self.held and m3kinematics.finger_span(args[0]) * 1000 > self.shut_mm + 5:   # opened past it:
                point = m3kinematics.grip(self.pose)[0]
                if point[2] < 0.04:   # low, set down where the hand is
                    self.boxes.append((float(point[0]), float(point[1]), *self.held[2:]))
                else:                 # let go, it falls
                    self.released.append(point)
                    self.dropped.append(self.held)
                    self.dropped_at.append(self.base.copy())
                    self.opened_to.append(args[0])
                self.held = None
        return dict(self.pose) if method == "read" else None

    def drive(self, dx, dy, speed=0.05):
        """The base moves: everything on the floor moves the other way, in its frame -- the things
        it holds and has dropped go with it, the drop-off box stays."""
        self.driven.append((round(float(dx), 4), round(float(dy), 4)))
        self.boxes = [(x - dx, y - dy, *rest) for x, y, *rest in self.boxes]
        self.base += (dx, dy)
        return dx, dy

    def snapshot(self):
        camera = m3kinematics.frames(self.pose)["arm4"] @ m3kinematics.camera_optical()
        return np.zeros((480, 640, 3), np.uint8), render(camera, self.boxes), K, dict(self.pose)


def run(world, job, **kwargs):
    picker = m3pick.Picker(world.call, world.snapshot, settle_s=0.0, rng=np.random.default_rng(0), drive=world.drive)
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
    assert abs(off) < m3pick.WRIST_SLACK[90] + 4, off   # square, or off it towards straight: the fingers' tilt
    width = next(kw["width_mm"] for m, _, kw in world.calls if m == "close_gripper")
    assert width == 30


def test_a_look_with_no_floor_in_it_is_skipped_and_the_job_goes_on():
    """2026-10-07: the left look saw only the robot's own deck, nearer than the depth reaches -- no floor,
    and the whole clean-up ended there."""
    world = World([(0.21, 0.03, 30, 0.03, 0.03)])
    seen = world.snapshot

    def blind_on_the_left():
        image, depth, k, pose = seen()
        return image, (np.zeros_like(depth) if pose[1] == m3pick.WIDE_SIDES[1][1] else depth), k, pose
    world.snapshot = blind_on_the_left
    picker = run(world, "pick_once")
    assert picker.state["error"] is None and picker.state["picked"] == 1
    assert any(f"wide J1 {m3pick.WIDE_SIDES[1][1]} look skipped" in e["text"] for e in picker._log)


def test_a_lying_block_is_gripped_across_its_length():
    world = World([(0.21, -0.02, 20, 0.06, 0.03)])
    picker = run(world, "pick_once", kind="block")
    assert picker.state["error"] is None and picker.state["picked"] == 1
    off = (closing(picker.state["last"]["grip_pose"]) - 110 + 90) % 180 - 90   # across 20 deg: 110
    assert abs(off) < m3pick.WRIST_SLACK[180] + 4, off


def test_out_of_reach_is_left_and_said():
    world = World([(0.40, 0.0, 0, 0.03, 0.03)])
    picker = run(world, "pick_once")
    assert "out of reach" in picker.state["error"] and world.boxes and not world.dropped


def test_clean_up_takes_every_one_in_reach():
    world = World([(0.21, 0.05, 10, 0.03, 0.03), (0.22, -0.05, 40, 0.03, 0.03)])
    picker = run(world, "clean_up")
    assert picker.state["picked"] == 2 and not world.boxes and len(world.dropped) == 2
    assert picker.state["error"] is None and world.pose == HOME


def test_the_look_job_looks_and_picks_nothing():
    """/pick/look on the M3: the shared job was named look(), as the M3's own looks are -- 2026-10-16
    it called that one with no tag and failed before the arm moved."""
    world = World([(0.21, 0.05, 10, 0.03, 0.03)])
    picker = run(world, "look")
    assert picker.state["error"] is None and picker.state["picked"] == 0 and len(world.boxes) == 1
    kit = [o for o in picker.status()["found"] if o["kind"]]
    assert len(kit) == 1 and abs(kit[0]["x"] - 0.21) < 0.015 and kit[0]["top"]
    assert world.pose == HOME


def test_a_thing_no_plan_reaches_is_planned_once_a_job_not_before_every_pick(monkeypatch):
    # nearest first, so the one with no grip comes up first before each of the other two
    world = World([(0.16, -0.03, 0, 0.03, 0.03), (0.21, 0.05, 10, 0.03, 0.03), (0.22, 0.10, 40, 0.03, 0.03)])
    tried = []
    plan = m3pick.Picker.plan

    def no_grip_on_the_right(self, thing, around):
        tried.append(float(thing["centre"][1]))
        return None if thing["centre"][1] < 0 else plan(self, thing, around)

    monkeypatch.setattr(m3pick.Picker, "plan", no_grip_on_the_right)
    picker = run(world, "clean_up")
    assert picker.state["picked"] == 2 and len(world.boxes) == 1   # the other two, it left
    assert sum(y < 0 for y in tried) == 1, tried                   # (2026-10-15: before every pick)


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
        drop = m3pick.drop_at(m3pick.DROP_OUT + extra, m3pick.DROP_UP, 180.0)
        for here in lifted:
            way = m3pick.way_to(here, drop, direct=False)
            assert way[-1] == drop and len(way) >= 2 and m3arm.glide_problem(here, way) is None
            assert way[0][2] < m3pick.FOLDED[2]   # not folded right up: a carry below it
        back = m3pick.way_to(drop, {**m3pick.LOOK, 6: 76})
        assert m3arm.glide_problem(drop, back) is None and back[-1][2] == m3pick.LOOK[2]
    deck = {1: 90, 2: 60, 3: 10, 4: 10, 5: 90}   # down onto the front deck: refused, and said why
    assert "chassis" in m3arm.glide_problem(dict(HOME), [deck])


def test_out_of_reach_it_drives_up_when_allowed_and_only_then():
    far = (0.29, -0.05, 0, 0.03, 0.03)   # 29 cm out: no shoulder angle reaches it from here
    # (at y -0.13 then: since 2026-10-02 driven up to it is in front of the right lidar -- a slide too)
    world = World([far])
    picker = run(world, "pick_once")
    assert "out of reach" in picker.state["error"] and not world.driven
    world = World([far])
    picker = run(world, "pick_once", drive=True)
    assert picker.state["error"] is None and picker.state["picked"] == 1 and world.dropped
    (dx, dy), back = world.driven
    assert 0.05 < dx <= 0.10 and dy == 0.0     # forward was enough: no sideways
    assert back == (-dx, -dy)                  # and back before the drop: the box did not drive along
    assert np.allclose(world.dropped_at[0], 0) and np.allclose(world.base, 0)


def test_a_cube_at_the_edge_of_reach_is_not_blamed_on_the_lidar():
    """2026-10-21, the user: "why is the cube too near the lidar? both are out of reach" -- a 40 mm cube
    24.6 cm from the arm's axis, its middle reached to 23.5: a slide right brings it nearer, not off the lidar."""
    picker = m3pick.Picker(lambda *a, **k: None, None, settle_s=0.0)
    cube = {"kind": "cube", "centre": np.array([0.262, -0.092, 0.02]), "height": 0.04, "size": 0.04, "length": 0.04,
            "yaw": 66.0, "period": 90.0, "whole": True, "footprint": (0.04, 0.04)}
    said = picker.why_not([cube], [cube])
    assert "edge of the arm's reach" in said and "lidar for the arm" not in said and "cm closer would reach it" in said, said


def test_a_cube_at_the_edge_of_reach_on_the_left_is_not_blamed_on_the_body():
    """2026-10-23, the user: why was this cube not picked? 24.0 cm from the arm's axis on the left, its
    middle reached to 23.2: only a grip at its very top, the arm low over the chassis's corner -- "too
    close to the robot's body". It is the reach, and 3 cm closer would do."""
    picker = m3pick.Picker(lambda *a, **k: None, None, settle_s=0.0)
    cube = {"kind": "cube", "centre": np.array([0.187, 0.184, 0.015]), "height": 0.03, "size": 0.03, "length": 0.03,
            "yaw": 40.0, "period": 90.0, "whole": True, "footprint": (0.03, 0.03)}
    said = picker.why_not([cube], [cube])
    assert "edge of the arm's reach (only its top in reach from here): 30 mm cube at x 0.187" in said and "robot's body" not in said, said
    assert "the robot 3 cm closer would reach it" in said and "may drive closer" in said, said


def test_the_fetch_drives_up_to_a_cube_far_off_and_back_to_the_box_beside_it():
    """2026-10-21, the event's fetch (the user's): a cube 42 cm out, the box at the front-left out of the
    base's way -- driven up to it in moves of nudge_cm, picked, driven back, dropped into the box."""
    world = World([(0.42, 0.0, 0, 0.03, 0.03)])
    picker = m3pick.Picker(world.call, world.snapshot, settle_s=0.0, rng=np.random.default_rng(0), drive=world.drive)
    left = {"centre": np.array([0.13, 0.23]), "along": np.array([1.0, 0.0]), "into": np.array([0.0, -1.0]),
            "turn": 0.0, "items": []}   # its lid along x on the far (left) side
    picker.see_box = lambda tag: left
    picker.start("pick_once", drive=True)
    picker.wait(120)
    assert picker.state["error"] is None and picker.state["picked"] == 1 and len(world.released) == 1
    out = [d for d in world.driven if d[0] > 0]
    assert len(out) >= 2 and all(dx <= picker.cfg["nudge_cm"] / 100 + 1e-9 for dx, _ in out)   # up, a move at a time
    assert np.allclose(world.base, 0, atol=0.002)                       # back where it began, the box there
    along, across = m3pick.in_box(left, world.released[0])             # and into it
    assert abs(along) < m3pick.BOX_INSIDE[0] / 2 and abs(across) < m3pick.BOX_INSIDE[1] / 2


def test_the_fetch_never_drives_into_the_box_the_depth_does_not_see():
    """The box's black walls give the depth nothing: a box ahead in the base's way stops the drive short
    of it (the footprint DRIVE_GAP off), wherever the cube is -- 2026-10-21, the fetch."""
    world = World([(0.42, 0.06, 0, 0.03, 0.03)])
    picker = m3pick.Picker(world.call, world.snapshot, settle_s=0.0, rng=np.random.default_rng(0), drive=world.drive)
    ahead = {**box_here(), "centre": np.array([0.30, -0.10])}   # its walls y -0.165 to -0.035: in the way
    picker.see_box = lambda tag: ahead
    picker.start("pick_once", drive=True)
    picker.wait(120)
    nearest = min(float(p[0]) for p in outline(ahead))   # the box's near wall, outside
    forward = sum(dx for dx, _ in world.driven if dx > 0)
    assert 0 < forward <= nearest - m3pick.ROBOT_X[1] - m3pick.DRIVE_GAP + 1e-6, (forward, nearest)
    assert "out of reach" in picker.state["error"]   # driven as far as that, the cube still too far


def test_the_drive_keeps_clear_of_what_is_in_the_way():
    def at(x, y):
        return {"kind": "cube", "centre": np.array([x, y, 0.015]), "height": 0.03}
    target, blocker = at(0.30, 0.0), at(0.18, 0.03)
    dx, _dy = m3pick.nudge([target, blocker], [target])
    assert abs(dx - (0.18 - m3pick.BUMPER_X - m3pick.BUMPER_GAP)) < 1e-3
    assert m3pick.nudge([at(0.17, 0.0)], [at(0.17, 0.0)]) is None   # already near: nowhere to go


def test_the_fingers_square_a_cube_the_way_that_keeps_them_off_its_neighbour_and_open_wider_for_40_mm():
    """2026-10-01: a 30 mm cube with a 40 mm one 6.5 cm to its left -- fingers opening left-right hit it.
    (Then at y -0.063, turned 70: since 2026-10-02 in front of the right lidar -- the scene here is
    that one turned 39 deg left round the arm's axis.)"""
    cube = np.array([0.212, 0.063, 0.015])
    big = {"centre": np.array([0.191, 0.124, 0.02]), "footprint": (0.04, 0.04)}
    _pitch, poses, _lift = m3pick.plan_grasp(cube, 0.03)
    width = m3kinematics.finger_span(m3pick.opened_for(0.030))
    alone = m3pick.wrist_squares(poses[-1], 109, 90, width)[0]
    beside = m3pick.wrist_squares(poses[-1], 109, 90, width, [big])[0]
    assert m3pick.finger_clearance({**poses[-1], 5: alone}, width, [big]) < 0.005
    assert m3pick.finger_clearance({**poses[-1], 5: beside}, width, [big]) > 0.02
    assert abs(abs(beside - alone) - 90) <= 2           # the cube's other square way
    assert abs(m3kinematics.finger_span(m3pick.opened_for(0.030)) - 0.055) < 0.001
    assert m3kinematics.finger_span(m3pick.opened_for(0.040)) > 0.065   # a 40 mm cube's diagonal is 57 mm


def test_the_settings_are_held_to_their_ranges_kept_and_count(tmp_path):
    world = World([(0.21, 0.03, 0, 0.03, 0.03)])
    picker = m3pick.Picker(world.call, world.snapshot, out=tmp_path)
    assert {s["key"] for s in picker.settings()} == set(m3pick.SETTINGS)
    picker.set_settings({"speed": 500, "descend_dps": 30})
    assert picker.cfg["speed"] == m3pick.SETTINGS["speed"][3] and picker.dps("descend_dps") == 30 * 1.5
    with pytest.raises(ValueError, match="no setting"):
        picker.set_settings({"warp": 9})
    again = m3pick.Picker(world.call, world.snapshot, out=tmp_path)   # a restart keeps them
    assert again.cfg["descend_dps"] == 30
    again.set_settings({"settle_close_s": 0.0, "settle_wide_s": 0.0, "speed": 100})
    again.start("pick_once")
    again.wait(60)
    descent = [kw["speed_dps"] for m, _a, kw in world.calls if m == "follow" and kw.get("floor_clear")]
    assert descent[0] == 30   # down the line at the setting


def test_the_fingertips_keep_their_height_over_the_table_the_grip_rises_instead():
    """Asked for more room over the table, the grip rises on the cube (8 mm, the default from 2026-10-05:
    they still touched it, the user saw; 5 from 2026-10-21, the user's, for more grips)."""
    for centre in ((0.20, 0.03, 0.015), (0.25, -0.05, 0.015), (0.16, 0.0, 0.015)):
        low = m3pick.plan_grasp(np.array(centre), 0.03, floor_clear=0.003)
        high = m3pick.plan_grasp(np.array(centre), 0.03, floor_clear=0.008)
        for _pitch, poses, _lift in (low, high):
            assert m3kinematics.grip(poses[-1])[0][2] <= 0.03 + 1e-6   # never above the cube's top
        assert min(m3kinematics.outline(p)[:, 2].min() for p in high[1]) >= 0.008
        assert m3kinematics.grip(high[1][-1])[0][2] > m3kinematics.grip(low[1][-1])[0][2]
    assert m3pick.SETTINGS["fingertips_mm"][5] == 8 and m3pick.SETTINGS["fingertips_least_mm"][5] == 5


def test_on_the_line_it_lets_go_just_over_the_table():
    """2026-10-21, the user's: let go from DROP_UP's 11 cm, blocks bounced off the table -- now what is held
    LINE_OVER over it where the arm reaches that low, else the lowest of LINE_RELEASES."""
    world = World([(0.20, 0.03, 0, 0.06, 0.03)])   # a block lying; no box: onto the line
    picker = run(world, "pick_once")
    assert picker.state["error"] is None and picker.state["picked"] == 1
    on_line = [b for b in world.boxes if b[0] < 0.10 and b[1] < -0.15]   # set down low (World: under 4 cm)
    assert on_line or (world.released and world.released[-1][2] <= 0.06), (world.boxes, world.released)
    here = {1: 80, 2: 76, 3: 15, 4: 19, 5: 90, 6: 134}
    for extra in (0.0, 0.05, 0.10):   # further out the arm reaches only higher: still on the line
        way = next(w for h in (0.04, *m3pick.LINE_RELEASES) if (w := m3pick.carry_to(here, m3pick.DROP_OUT + extra, 180.0, h, 0.18)))
        assert way[-1][1] == 180


def test_the_fingertips_come_lower_only_where_nothing_plans_higher():
    """2026-10-21, the user's: at 5 mm for every grip the fingers touched the table by small cubes and
    shoved the robot; at 8 two blocks had no grip. 8 first, 5 for those only."""
    picker = m3pick.Picker(lambda *a, **k: None, None, settle_s=0.0)
    cube = {"centre": np.array([0.20, 0.03, 0.015]), "height": 0.03, "size": 0.03, "length": 0.03, "yaw": 20.0,
            "period": 90.0, "whole": True, "kind": "cube"}
    assert picker.plan(cube, [cube]) is not None and cube["floor"] == 0.008
    block = {"centre": np.array([0.225, 0.117, 0.015]), "height": 0.03, "size": 0.03, "length": 0.06, "yaw": 37.0,
             "period": 180.0, "whole": True, "kind": "block", "footprint": (0.03, 0.06)}   # that day's, no grip at 8
    assert picker.plan(block, [block]) is not None and block["floor"] == 0.005
    assert picker.floor_m is None and picker.floor_clear() == 0.008   # planning leaves the first set
    picker.set_settings({"fingertips_least_mm": 8}, save=False)
    assert picker.floors() == [0.008] and picker.plan(dict(block), [block]) is None


def test_a_plan_is_one_the_arm_will_glide_down_and_back_up():
    """2026-10-01: lines whose points all passed, the glide between them refused by the arm (the
    forearm 17 cm over its base) -- the job failed. The planner checks the glides as the arm does.
    (y -0.115 then: since 2026-10-02 in front of the right lidar, its mirror here.)"""
    for centre, top in (((0.171, 0.081, 0.015), 0.03), ((0.169, 0.115, 0.015), 0.03), ((0.167, -0.032, 0.015), 0.03),
                        ((0.225, 0.051, 0.02), 0.04), ((0.26, -0.062, 0.015), 0.03)):
        for clear in (0.003, 0.008):
            _pitch, poses, _lift = m3pick.plan_grasp(np.array(centre), top, floor_clear=clear)
            assert m3arm.glide_problem(poses[0], poses[1:], clear) is None
            assert m3arm.glide_problem(poses[-1], poses[-2::-1], clear) is None


def test_a_refused_way_leaves_that_one_and_the_clean_up_goes_on():
    world = World([(0.20, 0.05, 0, 0.03, 0.03), (0.21, -0.05, 20, 0.03, 0.03)])
    world.refuse = np.array([0.20, 0.05])
    picker = run(world, "clean_up")
    assert picker.state["error"] is None and picker.state["picked"] == 1
    assert world.boxes == [(0.20, 0.05, 0, 0.03, 0.03)] and len(world.dropped) == 1


def test_with_the_wrist_turned_both_fingertips_keep_off_the_table():
    """2026-10-02: J5 45 at 60 deg tilted the open fingers -- one 4 mm into the table while the hand's
    middle was 8.5 mm over it; it touched and pushed the robot. Planned with the wrist it will use."""
    opening = m3pick.opened_for(0.030)
    for centre in ((0.25, 0.0, 0.015), (0.21, 0.13, 0.015), (0.175, 0.07, 0.015), (0.20, 0.05, 0.015)):
        for yaw in (0, 20, 45, 70):
            def wrist(grip, yaw=yaw):
                return m3pick.wrist_choices(grip, yaw, 90, m3kinematics.finger_span(opening))
            planned = m3pick.plan_grasp(np.array(centre), 0.03, floor_clear=0.008, wrist=wrist, gripper=opening)
            if centre[0] >= 0.25 and yaw in (20, 45, 70):
                # at the far edge the hand comes in at ~55 deg: a cube turned much tilts the fingers more
                # than a grip at its very top makes up -- no plan, rather than one into the table (the
                # robot drives closer when it may)
                assert planned is None or min(m3kinematics.outline(p)[:, 2].min() for p in planned[1]) >= 0.008 - 1e-9
                continue
            assert planned is not None, (centre, yaw)
            for pose in planned[1]:
                assert pose[6] == opening and m3kinematics.outline(pose)[:, 2].min() >= 0.008 - 1e-9, (centre, yaw)
    turned = {1: 90, 2: 20, 3: 40, 4: 60, 5: 45, 6: opening}
    tips = m3kinematics.outline(turned)[-5:-1]          # the fingers, as the guard sees them
    assert np.ptp(tips[[1, 3], 2]) > 0.01               # the turned wrist: one fingertip well below the other


def test_the_way_back_up_is_planned_with_the_fingers_shut_on_the_thing():
    """2026-10-02: a 40 mm cube by the right front corner -- up the line with the fingers open passed,
    with them shut on it the arm refused, and the arm stood at the bottom holding it."""
    cube, opened, shut = np.array([0.163, -0.115, 0.02]), m3pick.opened_for(0.040), m3arm.gripper_angle(40)
    for yaw in (75, 30, 0):
        def wrist(grip, yaw=yaw):
            return m3pick.wrist_choices(grip, yaw, 90, m3kinematics.finger_span(opened))
        planned = m3pick.plan_grasp(cube, 0.04, floor_clear=0.003, wrist=wrist, gripper=opened, held=shut)
        if planned is None:
            continue
        _pitch, poses, _lift = planned
        lifted = [{**q, 6: shut} for q in poses]
        assert m3arm.glide_problem(lifted[-1], lifted[-2::-1], 0.003) is None, yaw


def test_the_carry_goes_round_low_just_over_the_drop_off_height():
    """2026-10-02, the user: not up high and down again -- round at the carry height (15 cm and what
    hangs below the grip: a 30 mm cube's 3 cm here), then down onto the spot. (Round just over the
    drop-off's 11 cm, the camera rode over the front-right lidar.)"""
    high = 0.15 + 0.03
    for here in ({1: 80, 2: 76, 3: 15, 4: 19, 5: 90, 6: 134}, {1: 60, 2: 72, 3: 20, 4: 25, 5: 120, 6: 115},
                 {1: 136, 2: 59, 3: 34, 4: 18, 5: 44, 6: 134}):
        for extra in (0.0, 0.05, 0.10):
            way = m3pick.carry_to(here, m3pick.DROP_OUT + extra, 180.0, m3pick.DROP_UP, high)
            assert way is not None and m3arm.glide_problem(here, way) is None
            grips = [m3kinematics.grip(q)[0] for q in way]
            assert max(g[2] for g in grips) <= high + 0.005
            assert way[-1][1] == 180 and abs(grips[-1][2] - m3pick.DROP_UP) < 0.006
            axis = np.asarray(m3kinematics.CHAIN[0][0][:2])
            assert all(np.hypot(*(g[:2] - axis)) > 0.17 for g in grips[1:])   # round, clear of the chassis
            assert all(q[6] == here[6] for q in way)                          # still holding it


def box_here():
    """The drop-off box as the user put it: its inside 15 x 11 cm, centred 10 cm ahead and 21 cm out on
    the right, the lid's hinge along x at the far side."""
    return {"centre": np.array([0.10, -0.21]), "along": np.array([-1.0, 0.0]), "into": np.array([0.0, 1.0]),
            "turn": 180.0, "items": []}


def test_into_the_box_its_marker_puts_it_clear_of_the_walls_and_spread_out():
    world = World([(0.20, 0.06, 0, 0.03, 0.03), (0.21, -0.05, 30, 0.03, 0.03), (0.24, 0.01, 60, 0.03, 0.03)])
    picker = m3pick.Picker(world.call, world.snapshot, settle_s=0.0, rng=np.random.default_rng(0), drive=world.drive)
    picker.see_box = lambda tag: box_here()   # every look sees marker 10, the box where it stands
    picker.start("clean_up")
    picker.wait(120)
    assert picker.state["error"] is None and picker.state["picked"] == 3 and len(world.released) == 3
    box = box_here()
    for point in world.released:
        rel = point[:2] - box["centre"]
        assert abs(rel @ box["along"]) <= 0.075 - m3pick.BOX_MARGIN + 0.006
        assert abs(rel @ box["into"]) <= 0.055 - m3pick.BOX_MARGIN + 0.006
        assert abs(point[2] - (m3pick.BOX_INSIDE[2] + m3pick.BOX_DROP_OVER)) < 0.006   # over the walls
    gaps = [np.linalg.norm(a[:2] - b[:2]) for i, a in enumerate(world.released) for b in world.released[i + 1:]]
    assert min(gaps) > 0.03   # not a tower: spread over the inside


def test_a_floor_that_turns_the_camera_far_off_the_arm_model_is_no_floor():
    """2026-10-16: a box look at J1 135 levelled by another plane (the room's floor past the table's
    edge in view) put the box 10 cm off; the arm model's camera put it where it stood."""
    pose = {**m3pick.SIDE_LOOK, 1: 135}
    model = m3kinematics.frames(pose)["arm4"] @ m3kinematics.camera_optical()

    def turned(deg, dz):
        out = np.array(model)
        out[:3, :3] = m3pick.cv2.Rodrigues(np.array([math.radians(deg), 0.0, 0.0]))[0] @ model[:3, :3]
        out[2, 3] += dz
        return out
    assert m3pick.level_trusted(turned(1.7, -0.012), model)     # the usual: 12 mm lower, 1.7 deg
    assert not m3pick.level_trusted(turned(12.0, -0.012), model)
    assert not m3pick.level_trusted(turned(1.0, -0.06), model)


def test_the_box_from_its_marker():
    lean = math.radians(21)
    face = np.array([0.0, math.cos(lean), math.sin(lean)])   # towards the robot (+y), leaning back
    right = np.array([-1.0, 0.0, 0.0])
    up = np.cross(face, right)
    box = m3pick.box_from_marker(np.array([0.10, -0.31, 0.10]), np.column_stack([right, up, face]))
    assert np.allclose(box["into"], [0.0, 1.0]) and np.allclose(box["along"], [-1.0, 0.0])
    hinge_y = -0.31 + (0.10 - m3pick.BOX_INSIDE[2]) * math.tan(lean)   # down the lid to the walls' top
    assert abs(box["centre"][1] - (hinge_y + m3pick.BOX_INSIDE[1] / 2)) < 0.002 and abs(box["centre"][0] - 0.10) < 1e-9


def test_held_things_go_round_15_cm_up_and_straight_down_into_the_box():
    """2026-10-02, the user: just over the release height it nearly hit the box (its lid stands up) --
    15 cm from the floor, the setting's default, measured to the bottom of what is held."""
    picker = m3pick.Picker(lambda *a, **k: None, None, settle_s=0.0, rng=np.random.default_rng(1))
    assert picker.cfg["carry_cm"] == 15
    picker.box, picker.box_drops, picker.held_below = box_here(), [], 0.022   # gripped 22 mm up
    here = {1: 80, 2: 76, 3: 15, 4: 19, 5: 90, 6: 134}
    spot, way = picker.box_way(here)
    grips = [m3kinematics.grip(q)[0] for q in way]
    assert np.linalg.norm(grips[-1][:2] - spot) < 0.008                    # let go over the spot
    assert abs(max(g[2] for g in grips) - (0.15 + 0.022)) < 0.006            # the cube's bottom at 15 cm
    assert abs(grips[-1][2] - (m3pick.BOX_INSIDE[2] + m3pick.BOX_DROP_OVER)) < 0.006
    assert np.linalg.norm(grips[-2][:2] - grips[-1][:2]) < 0.01            # straight down onto the spot
    assert m3arm.glide_problem(here, way) is None


def test_both_of_a_cubes_square_wrists_are_tried():
    """2026-10-02: a cube 26 cm out turned 74 deg with another 6 cm off -- the square wrist kept
    clear of that one stood the fingers on end (into the table there); the other was never tried."""
    cube = np.array([0.258, -0.067, 0.015])
    other = {"centre": np.array([0.285, -0.012, 0.015]), "footprint": (0.03, 0.03)}
    opened = m3pick.opened_for(0.030)

    def wrist(grip):
        return m3pick.wrist_choices(grip, 74, 90, m3kinematics.finger_span(opened), [other])
    squares, _off = wrist(m3pick.plan_grasp(cube, 0.03, floor_clear=0.003)[1][-1])
    assert len(squares) >= 2 and all(abs((a - b + 45) % 90 - 45) <= 2 for a in squares for b in squares)   # 90 apart
    assert sum(j5 <= 180 for j5 in squares) == 2   # both within 0-180 (and, past it, the same turned over)
    planned = m3pick.plan_grasp(cube, 0.03, floor_clear=0.003, wrist=wrist, gripper=opened, held=m3arm.gripper_angle(30))
    assert planned is not None and planned[1][-1][5] in squares


def box_at(x, y):
    return {**box_here(), "centre": np.array([x, y])}


def test_a_box_moved_between_picks_is_dropped_into_where_it_is_now():
    world = World([(0.20, 0.06, 0, 0.03, 0.03), (0.21, -0.05, 30, 0.03, 0.03)])
    picker = m3pick.Picker(world.call, world.snapshot, settle_s=0.0, rng=np.random.default_rng(0), drive=world.drive)
    looks = []

    def see(tag):   # the job's look and the first drop's see it at 10 cm ahead; then it was moved 6 cm on
        looks.append(tag)
        return box_at(0.10, -0.21) if len(looks) <= 2 else box_at(0.16, -0.22)
    picker.see_box = see
    picker.start("clean_up")
    picker.wait(120)
    assert picker.state["error"] is None and picker.state["picked"] == 2
    first, second = world.released
    assert abs(in_box_of(box_at(0.10, -0.21), first)).max() <= 0.075
    assert abs(m3pick.in_box(box_at(0.16, -0.22), second)[0]) <= 0.075 - m3pick.BOX_MARGIN + 0.006
    assert abs(m3pick.in_box(box_at(0.16, -0.22), second)[1]) <= 0.055 - m3pick.BOX_MARGIN + 0.006


def in_box_of(box, point):
    return m3pick.in_box(box, point)


def test_no_box_to_be_found_puts_it_back_and_stops():
    cube = (0.21, 0.03, 20, 0.03, 0.03)
    world = World([cube])
    picker = m3pick.Picker(world.call, world.snapshot, settle_s=0.0, rng=np.random.default_rng(0), drive=world.drive)
    looks = []

    def see(tag):   # there at the job's start, gone when it comes to the drop
        looks.append(tag)
        return box_at(0.10, -0.21) if len(looks) == 1 else None
    picker.see_box = see
    picker.start("clean_up")
    picker.wait(120)
    assert not world.released and not world.dropped and world.held is None   # nothing let go over nothing
    (x, y, *_rest), = world.boxes                                              # back down where it was
    assert np.hypot(x - cube[0], y - cube[1]) < 0.012
    assert looks == ["box", "box view"] + ["box"] * (len(m3pick.BOX_LOOKS) - 1)   # the start's, the box view
    # where it was found, then the other looks round
    assert world.pose[2] == 180                          # home again


def test_left_out_of_reach_by_the_lidar_without_a_square_grip_and_by_the_body_is_said():
    """2026-10-02: the message for a cube out of reach and a block lying lengthwise to the bumper
    (its square grip puts a finger over it) raised ValueError -- `in` compared the dicts' arrays.
    All four said at once: out of reach; too near the right lidar; a block at the left edge with no
    square grip; a cube beside the left wheel too close to the body -- "out of reach" until 2026-10-05.
    (Those blocks by the bumper of 2026-10-02 have the careful grip since.)"""
    world = World([(0.40, 0.0, 0, 0.03, 0.03), (0.169, -0.115, 0, 0.03, 0.03), (0.15, 0.12, 90, 0.06, 0.03),
                   (0.09, 0.19, 0, 0.03, 0.03)])
    picker = run(world, "clean_up")
    said = picker._log[-1]["text"]
    assert picker.state["error"] is None, picker.state["error"]
    assert "out of reach: 30 mm cube at x 0.400" in said and "too near the right lidar" in said, said
    assert "no square grip" in said and "too close to the robot's body for the arm: 30 mm cube at x 0.090" in said, said
    assert len(world.boxes) == 4 and not world.dropped


def test_the_arm_keeps_off_the_front_right_lidar():
    """2026-10-02: the arm had hit the front-right lidar. Reaching front-right its camera -- 4.5 cm
    to either side of the arm -- rode over that corner, in poses the chassis' steps allowed: right
    in front of the lidar no grip is planned now, the arm itself refuses the pose that went in, and
    what it uses -- home, the looks, the carries -- still passes. (2026-10-05: 1.5 cm off it, not 2.5,
    the user having watched the arm 2.0-2.4 cm off it down a pick's path: a little more is picked.)"""
    for centre in ((0.175, -0.07, 0.015), (0.169, -0.115, 0.015), (0.17, -0.10, 0.015)):
        assert m3pick.plan_grasp(np.array(centre), 0.03, floor_clear=0.003) is None, centre
    for centre in ((0.212, -0.063, 0.015), (0.21, -0.13, 0.015)):   # 1.0 cm off it, the hand as seen
        assert m3pick.plan_grasp(np.array(centre), 0.03, floor_clear=0.003) is not None, centre
    # the 40 mm cube just by it: only with the hand turned over (J5 past 180, the finger motor's housing
    # away from the lidar), the line down planned with that wrist (2026-10-05)
    cube, opened = np.array([0.160, -0.075, 0.02]), m3pick.opened_for(0.04)   # 0.165 until the fingers'
    # margin went to 6 mm (2026-10-05): there J5 173 does too
    def wrist(grip):
        return m3pick.wrist_choices(grip, 67, 90, m3kinematics.finger_span(opened))
    planned = m3pick.plan_grasp(cube, 0.04, floor_clear=0.003, wrist=wrist, gripper=opened, held=m3arm.gripper_angle(40))
    assert planned is not None and planned[1][-1][5] > 180, planned
    why = m3kinematics.problem({1: 115, 2: 30, 3: 30, 4: 30, 5: 90})
    assert why and "front-right lidar" in why
    for pose in (m3pick.FOLDED, m3pick.LOOK, *({**m3pick.SIDE_LOOK, 1: j} for j in m3pick.BOX_LOOKS),
                 *({1: j, 2: c[0], 3: c[1], 4: c[2], 5: 90} for c in m3pick.CARRIES for j in range(0, 181, 15))):
        assert m3kinematics.problem(pose) is None, pose


def test_in_front_of_the_right_lidar_the_base_slides_right_when_it_may_and_says_so_when_not():
    """2026-10-02, the user's choice: the arm keeps LIDAR_ROOM off the front-right lidar, and what is just
    in front of it is picked after the base slides right -- back before the drop."""
    cube = (0.20, -0.15, 0, 0.03, 0.03)
    world = World([cube])
    picker = run(world, "pick_once")
    said = " ".join(line["text"] for line in picker._log)
    assert "too near the right lidar" in picker.state["error"] and "may drive closer" in picker.state["error"], said
    assert not world.driven and world.boxes == [cube]
    world = World([cube])
    picker = run(world, "pick_once", drive=True)
    assert picker.state["error"] is None and picker.state["picked"] == 1 and world.dropped
    (dx, dy), back = world.driven
    assert dx == 0.0 and -0.10 <= dy <= -m3pick.SIDESTEP    # right only, no further than right_cm
    assert back == (-dx, -dy) and np.allclose(world.base, 0)


def test_the_slide_right_keeps_off_the_drop_off_box_and_the_line_dropped_onto():
    world = World([])
    picker = m3pick.Picker(world.call, world.snapshot, settle_s=0.0)
    assert picker.right_room() == pytest.approx(0.10)          # nothing there: right_cm
    picker.box = box_here()                                      # its near wall 2.5 cm off the wheels: not a cm
    assert picker.right_room() == 0.0
    picker.box = box_at(0.10, -0.30)                             # 11.5 cm off: 8.5 cm, 3 short of it
    assert abs(picker.right_room() - 0.085) <= 0.0055                # in 5 mm steps
    picker.offset = np.array([0.0, -0.04])                       # already slid 4 cm: what is left
    assert abs(picker.right_room() - 0.045) <= 0.0055
    picker.box, picker.offset, picker.line_drops = None, np.zeros(2), 1
    near = m3pick.DROP_OUT - 0.03 - m3pick.ROBOT_Y - m3pick.DRIVE_GAP   # short of the line, 3 cm round its spots
    assert abs(picker.right_room() - near) <= 0.0055


def test_the_side_looks_find_what_the_middle_one_cannot_see():
    """2026-10-05, the user: every cube in reach across the front. At J1 24 deg, 21 cm out, the middle
    look does not see it; the job's first look looks round."""
    world = World([(0.12, 0.19, 0, 0.03, 0.03)])
    picker = run(world, "pick_once")
    said = [line["text"] for line in picker._log]
    assert picker.state["error"] is None and picker.state["picked"] == 1, said
    assert any(t.startswith("looked round:") and "x 0.12" in t for t in said), said


def test_what_lies_where_the_job_lets_go_is_not_picked_again():
    """The right side look sees into the drop-off box and onto the drop line: those are left."""
    world = World([])
    picker = m3pick.Picker(world.call, world.snapshot, settle_s=0.0)
    def at(x, y):
        return {"centre": np.array([x, y, 0.015])}
    picker.box = box_here()
    assert picker.dropped_here(at(0.10, -0.21)) and picker.dropped_here(at(0.17, -0.25))   # in it, by its wall
    assert not picker.dropped_here(at(0.21, 0.03))
    picker.offset = np.array([0.05, 0.0])            # the base 5 cm on: the box 5 cm nearer in its frame
    assert picker.dropped_here(at(0.05, -0.21)) and not picker.dropped_here(at(0.10, -0.10))
    picker.box, picker.offset = None, np.zeros(2)
    x = float(m3kinematics.CHAIN[0][0][0])
    assert not picker.dropped_here(at(x, -0.22))     # the line: only once dropped onto
    picker.line_drops = 1
    assert picker.dropped_here(at(x, -0.22)) and not picker.dropped_here(at(0.21, -0.03))


def test_a_square_grip_turned_over_is_tried_once_j5_turns_past_180(monkeypatch):
    """2026-10-05, the user's: the finger motor's housing turned away from the lidar -- J5 turned
    180 deg. Yahboom drives the M3's J5 to 270; so does ours since (the user's, the same day)."""
    grip = m3kinematics.reach(np.array([0.21, 0.03, 0.02]), 70, floor_clear=0.003)
    turned, off = m3pick.wrist_choices(grip, 20, 90)
    monkeypatch.setitem(m3arm.SAFE_LIMITS, 5, (0, 180))   # as the X3's: none turned over
    squares, _off = m3pick.wrist_choices(grip, 20, 90)
    assert max(squares) <= 180
    assert any(j5 > 180 for j5 in turned) and set(squares) <= set(turned)
    assert all(abs((j5 - s + 180) % 360 - 180) in (0, 90, 180) for j5 in turned for s in squares)
    assert all(j5 <= 270 for j5 in off)


def test_the_looks_go_round_without_up_and_down():
    """2026-10-05, the user: searching, the arm went up and down several times -- the side looks in
    shapes of their own, each way between them over a carry pose. Now one turn of J1 from the box
    look, and a settle into LOOK."""
    opened = m3arm.gripper_angle(m3pick.OPEN_MM)

    def glides(here, goal):
        legs = m3pick.look_way(here, goal)
        assert legs[-1][-1] == goal
        for leg in legs:
            assert m3arm.glide_problem(here, leg) is None
            assert all(q[2] <= 105 for q in leg), legs    # never up to a carry pose (J2 130-180)
            here = leg[-1]
    here = {**m3pick.SIDE_LOOK, 6: opened}
    for look in (*m3pick.WIDE_SIDES, m3pick.LOOK):
        glides(here, {**look, 6: opened})
        here = {**look, 6: opened}
    glides({**m3pick.LOOK, 6: opened}, {**m3pick.WIDE_SIDES[0], 6: opened})   # round again, from LOOK


def test_a_cube_missed_twice_is_left_and_the_others_still_picked():
    """2026-10-05: the clean-up stopped at the second miss of one cube, two others left untried."""
    world = World([(0.19, 0.0, 0, 0.03, 0.03), (0.22, 0.05, 10, 0.03, 0.03)])
    world.slip = [(0.19, 0.0)]   # the nearer one: its grip never holds
    picker = run(world, "clean_up")
    said = " ".join(line["text"] for line in picker._log)
    assert "missed twice" in said and picker.state["missed"] == 2 and picker.state["picked"] == 1, said
    assert [b[:2] for b in world.boxes] == [(0.19, 0.0)] and len(world.dropped) == 1


def test_a_block_lying_across_in_front_of_the_wheel_has_the_careful_grip():
    """2026-10-05, the user: a block lying across 11 mm in front of the bumper, by the left wheel, "we
    have to pick things from such position" -- no grip kept the near finger off the robot. The careful
    one, as it was read (turned 2-6 deg, or 178): opened 15 mm over it, aimed 3-5 mm further out."""
    picker = m3pick.Picker(lambda *a, **k: None, None, settle_s=0.0)
    for yaw in (2.0, 4.0, 6.0, 178.0):
        block = {"centre": np.array([0.150, 0.079, 0.015]), "height": 0.03, "size": 0.03, "length": 0.06, "yaw": yaw,
                 "period": 180.0, "whole": True, "kind": "block", "footprint": (0.03, 0.06)}
        planned = picker.plan(block, [block])
        assert planned is not None and block["careful"], yaw
        grip = m3kinematics.grip(planned[1][-1])[0]
        assert 0.002 <= grip[0] - 0.150 <= 0.0085 and abs(grip[1] - 0.079) < 0.003, (yaw, grip)   # further out along
        # x: 3-5 mm, and the whole servo degrees' few
        assert all(q[6] == m3pick.opened_for(0.03, m3pick.CAREFUL_SPARE_MM) for q in planned[1])
    world = World([(0.150, 0.079, 94, 0.06, 0.03)])
    picker = run(world, "pick_once")
    said = " ".join(line["text"] for line in picker._log)
    assert picker.state["error"] is None and picker.state["picked"] == 1 and "careful grip" in said, said
    assert "looked again turned onto it" in said, said   # by the bumper the first close look cuts it


def test_a_block_by_the_front_corner_is_picked_beside_the_tyre():
    """2026-10-21, the user's: a block lying ahead of the left front wheel, beside the chassis, refused --
    the arm's line 1.4 cm over and beside the tyre, inside the front deck's 17.5 cm box. The front corners
    keep 1 cm over the tyres now; the deck itself, the body's width, its 17.5."""
    picker = m3pick.Picker(lambda *a, **k: None, None, settle_s=0.0)
    block = {"centre": np.array([0.129, 0.187, 0.015]), "height": 0.03, "size": 0.03, "length": 0.06, "yaw": 74.0,
             "period": 180.0, "whole": True, "kind": "block", "footprint": (0.03, 0.06)}
    assert picker.plan(block, [block]) is not None
    def kept(x, y):   # the height the arm keeps over the floor at (x, y)
        return max((top for x0, x1, top, half in m3kinematics.CHASSIS
                    if x0 < x < x1 and abs(y) < half + m3kinematics.SPARE_Y), default=0.0)
    assert kept(0.10, 0.12) == kept(0.10, -0.12) == 0.175 and kept(0.10, 0.13) == kept(0.10, -0.13) == 0.09
    assert kept(0.0, 0.13) == 0.205 and kept(0.13, 0.0) == 0.0   # the arm's base full width; in front: none


def test_the_careful_grip_goes_down_only_on_a_whole_close_look(monkeypatch):
    """2026-10-05: the blue block's close look cut it, the wide look aimed, and the careful grip's near
    finger -- 4.5 mm off it by plan -- came down on its edge and rolled it over. Not seen whole: left."""
    monkeypatch.setattr(m3pick, "look_at", lambda *a: None)   # no look again
    block = (0.150, 0.079, 94, 0.06, 0.03)
    world = World([block])
    picker = run(world, "pick_once")
    assert "careful grip goes down only on a whole close look" in picker.state["error"], picker.state["error"]
    assert world.boxes == [block] and not any(m == "close_gripper" for m, _a, _k in world.calls)


def test_a_look_again_turns_the_camera_onto_what_the_first_cut():
    K = np.array([[480.84, 0, 323.79], [0, 480.89, 239.99], [0, 0, 1]])
    block = {"centre": np.array([0.150, 0.079, 0.015]), "height": 0.03, "size": 0.03, "length": 0.06, "yaw": 4.0,
             "period": 180.0, "whole": True, "kind": "block", "footprint": (0.06, 0.03)}
    picker = m3pick.Picker(lambda *a, **k: None, None, settle_s=0.0)
    pre = picker.plan(block, [block])[1][0]
    look = {**pre, 5: 90}
    assert not m3pick.in_picture(look, block, K)   # its near half below the picture, as on the robot
    again = m3pick.look_at(look, pre, block, K)
    assert again is not None and m3pick.in_picture(again, block, K)
    assert all(abs(again[j] - look[j]) <= 30 for j in (1, 2, 3, 4)) and (again[5], again[6]) == (look[5], look[6])
    camera = m3kinematics.frames(again)["arm4"] @ m3kinematics.camera_optical()
    assert camera[2, 1] <= -0.15   # at a slant: floor_plane takes no floor seen straight from above


def test_a_cube_is_not_picked_careful_where_the_plain_grip_does():
    picker = m3pick.Picker(lambda *a, **k: None, None, settle_s=0.0)
    cube = {"centre": np.array([0.21, 0.03, 0.015]), "height": 0.03, "size": 0.03, "yaw": 20.0, "period": 90.0,
            "whole": True, "kind": "cube", "footprint": (0.03, 0.03)}
    planned = picker.plan(cube, [cube])
    assert planned is not None and not cube["careful"] and planned[1][-1][6] == m3pick.opened_for(0.03)


def test_the_close_look_is_taken_with_the_wrist_straight():
    """2026-10-05: at the pre-grasp the wrist turned for the grip brought a finger up across the cube in
    the close look's picture -- seen in part, aimed by the wide look, missed. The look with the fingers
    left and right of it; the wrist turns on the way down."""
    world = World([(0.21, 0.03, 30, 0.03, 0.03)])
    picker = m3pick.Picker(world.call, world.snapshot, settle_s=0.0, rng=np.random.default_rng(0), drive=world.drive)
    looked, look = [], picker.look

    def recording(tag, target_near=None):
        looked.append((tag, dict(world.pose)))
        return look(tag, target_near)
    picker.look = recording
    picker.start("pick_once")
    picker.wait(120)
    assert picker.state["error"] is None and picker.state["picked"] == 1
    (_tag, pose), = [x for x in looked if x[0] == "close"]
    assert pose[5] in (90, 270) and picker.state["last"]["j5"] not in (90, 270)   # looked straight, gripped turned


def test_after_the_lift_the_box_is_looked_at_from_where_it_was_found_no_up_and_down():
    """2026-10-05, the user: "after the lift ... semi home position ... so it sees more" -- over the spot
    the marker read close and askew and the job looked for the box all round, holding: up and down.
    Risen in place, turned with the hand raised, tilted down into SIDE_LOOK at the box's J1: never up to a
    carry pose, and what is held never down over the chassis."""
    world = World([(0.21, 0.03, 30, 0.03, 0.03)])
    picker = m3pick.Picker(world.call, world.snapshot, settle_s=0.0, rng=np.random.default_rng(0), drive=world.drive)
    views = []

    def see(tag):
        views.append((tag, dict(world.pose)))
        return box_here()
    picker.see_box = see
    picker.start("pick_once")
    picker.wait(120)
    assert picker.state["error"] is None and picker.state["picked"] == 1 and len(world.released) == 1
    (_tag, pose), = [v for v in views if v[0] == "box view"]
    assert {j: pose[j] for j in (1, 2, 3, 4)} == {j: v for j, v in m3pick.SIDE_LOOK.items() if j != 5}
    closed = next(i for i, (m, _a, _k) in enumerate(world.calls) if m == "close_gripper")
    let_go = next(i for i, (m, _a, _k) in enumerate(world.calls) if m == "set_gripper" and i > closed)
    carried = [q for m, args, _kw in world.calls[closed:let_go] if m == "follow" for q in args[0]]
    assert carried and all(q[2] <= 105 for q in carried), carried   # never up to a carry pose (J2 130-180)
    assert m3kinematics.finger_span(world.opened_to[0]) <= 0.03 + 0.016   # let go just over its width


def test_the_raised_turn_holds_the_thing_over_the_chassis():
    held = m3arm.gripper_angle(30)
    top = {1: 90, 2: 60, 3: 40, 4: 60, 5: 90, 6: held}
    view = {**top, **m3pick.SIDE_LOOK, 1: 180}
    legs = m3pick.view_way(top, view, 0.02)
    assert len(legs) == 3 and legs[-1] == [view] and legs[1][0][4] == view[4] + m3pick.VIEW_TURN, legs
    assert m3pick.held_problem(top, [leg[-1] for leg in legs], 0.035) is None
    flat = {**top, **{j: view[j] for j in (2, 3, 4, 5)}}   # in SIDE_LOOK's shape what is held comes over the
    assert m3pick.held_problem(top, [flat, {**flat, 1: 180}], 0.035) is not None   # front deck's corner, low


def test_to_a_box_at_j1_45_or_135_the_arm_swings_round_low():
    """2026-10-21, the user's: with the box at J1 135 every pick went up to a carry pose (J2 130) on the way
    to the box view -- now round an arc VIEW_ARCS high, what is held over the front deck."""
    top = {1: 46, 2: 63, 3: 38, 4: 10, 5: 270, 6: m3arm.gripper_angle(30)}   # a line's top by the bumper, turned over
    for j1, below in itertools.product((135, 45), (0.018, 0.035)):
        view = {**top, **m3pick.SIDE_LOOK, 1: j1}
        legs = m3pick.view_way(top, view, below)
        assert legs[-1][-1] == view and all(p[2] < 130 for leg in legs for p in leg), (j1, below, legs)


def test_the_wrist_turns_over_with_the_fingers_closed_not_up_high():
    """2026-10-21, the user's: from the close look (J5 90) to a line's top turned over (J5 270) the arm went
    up to a carry pose and back -- the open fingers, swung across, came over the front deck."""
    look = {1: 46, 2: 63, 3: 38, 4: 10, 5: 90, 6: 76}
    top = {**look, 5: 270}
    assert m3arm.glide_problem(look, [top]) is not None   # turned open: over the deck
    legs = m3pick.turn_way(look, top)
    assert legs is not None and legs[-1] == [top] and legs[0][0][6] >= m3pick.TURN_SHUT and legs[1][0][6] >= m3pick.TURN_SHUT
    assert all(p[2] == look[2] for leg in legs for p in leg)   # no carry pose: turned where it is
    assert m3pick.turn_way(look, {**look, 5: 100}) is None   # a small turn goes straight (way_to's)


def test_the_drop_spots_keep_off_the_standing_lid_and_clear_of_the_walls():
    box = box_here()
    across = [float(m3pick.in_box(box, s)[1]) for s in m3pick.box_spots(box)]
    assert min(across) >= -m3pick.BOX_INSIDE[1] / 2 + m3pick.BOX_LID_MARGIN - 1e-9   # the lid: -into
    assert max(across) <= m3pick.BOX_INSIDE[1] / 2 - m3pick.BOX_FRONT_MARGIN + 1e-9   # the front: +into
    along = [abs(float(m3pick.in_box(box, s)[0])) for s in m3pick.box_spots(box)]
    assert max(along) <= m3pick.BOX_INSIDE[0] / 2 - m3pick.BOX_MARGIN + 1e-9   # the sides
    assert len(m3pick.box_spots(box)) >= 6   # room to spread the drops still


def test_after_a_pick_out_at_the_sides_it_looks_round_for_a_shoved_piece_not_by_the_bumper():
    """The look after is LOOK's alone; a piece shoved 8 cm out of it at the sides is seen only by a look
    round (roboshared.task.shoved), which costs ~4.4 s -- not where the side looks see no more of the ring."""
    picker = m3pick.Picker(lambda *a, **k: None, None, settle_s=0.0)
    for x, y in ((0.20, 0.15), (0.22, -0.12), (0.16, 0.10)):
        assert picker.round_sees_more(np.array([x, y, 0.015]), 0.08), (x, y)
    for x, y in ((0.25, 0.0), (0.20, 0.0), (0.30, 0.05)):   # straight ahead: LOOK, or nothing, sees it all
        assert not picker.round_sees_more(np.array([x, y, 0.015]), 0.08), (x, y)


def test_the_taught_aim_follows_the_grips_near_a_thing_and_their_mean_far_from_them():
    taught = ((0.20, 0.00, 0.020, 0.010), (0.20, -0.10, 0.008, -0.004))
    mean = np.array([0.014, 0.003])
    assert np.allclose(m3aim.offset(0.20, -0.10, taught), [0.008, -0.004], atol=0.0025)   # the corner's own
    assert np.allclose(m3aim.offset(0.20, 0.00, taught), [0.020, 0.010], atol=0.0025)
    assert np.allclose(m3aim.offset(0.60, 0.40, taught), mean)                           # nothing near: the mean
    assert np.allclose(m3aim.offset(0.2, 0.0, ()), 0.0)
