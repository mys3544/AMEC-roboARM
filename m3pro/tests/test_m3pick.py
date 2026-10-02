"""roboarm/m3pick.py against a pretend arm and a depth camera rendered from the arm model: a turned
cube is picked with the fingers square to it and dropped off, a lying block is gripped across,
something out of reach is left and said so, and a stop stops the job between two moves."""

import math
import time

import numpy as np
import pytest

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
        self.held, self.dropped, self.calls, self.driven, self.released = None, [], [], [], []
        self.base = np.zeros(2)   # where the base is from where it began
        self.refuse = None        # a spot the arm refuses to come down to
        self.dropped_at = []

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
            if self.refuse is not None and np.hypot(*(point[:2] - self.refuse)) < 0.02:
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
            self.pose[6] = m3arm.gripper_angle(kwargs["width_mm"])
            point, _ = m3kinematics.grip(self.pose)
            near = [b for b in self.boxes if math.hypot(b[0] - point[0], b[1] - point[1]) < 0.012]
            if near:
                self.held = near[0]
                self.boxes.remove(near[0])
        elif method == "set_gripper":
            self.pose[6] = args[0]
            if self.held and args[0] < 110:   # opened past it, low: set down where the hand is
                point = m3kinematics.grip(self.pose)[0]
                if point[2] < 0.04:
                    self.boxes.append((float(point[0]), float(point[1]), *self.held[2:]))
                    self.held = None
            if self.held and args[0] <= 40:   # opened wide: let go
                self.released.append(m3kinematics.grip(self.pose)[0])
                self.dropped.append(self.held)
                self.dropped_at.append(self.base.copy())
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
    width = m3pick.opening_m(m3pick.opened_for(0.030))
    alone = m3pick.wrist_for(poses[-1], 109, 90, width)
    beside = m3pick.wrist_for(poses[-1], 109, 90, width, [big])
    assert m3pick.finger_clearance({**poses[-1], 5: alone}, width, [big]) < 0.005
    assert m3pick.finger_clearance({**poses[-1], 5: beside}, width, [big]) > 0.02
    assert abs(abs(beside - alone) - 90) <= 2           # the cube's other square way
    assert abs(m3pick.opening_m(m3pick.opened_for(0.030)) - 0.055) < 0.001
    assert m3pick.opening_m(m3pick.opened_for(0.040)) > 0.065   # a 40 mm cube's diagonal is 57 mm


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
    """Asked for more room over the table, the grip rises on the cube (8 mm here; the default is 3)."""
    for centre in ((0.20, 0.03, 0.015), (0.25, -0.05, 0.015), (0.16, 0.0, 0.015)):
        low = m3pick.plan_grasp(np.array(centre), 0.03, floor_clear=0.003)
        high = m3pick.plan_grasp(np.array(centre), 0.03, floor_clear=0.008)
        for _pitch, poses, _lift in (low, high):
            assert m3kinematics.grip(poses[-1])[0][2] <= 0.03 + 1e-6   # never above the cube's top
        assert min(m3kinematics.outline(p)[:, 2].min() for p in high[1]) >= 0.008
        assert m3kinematics.grip(high[1][-1])[0][2] > m3kinematics.grip(low[1][-1])[0][2]
    assert m3pick.SETTINGS["fingertips_mm"][5] == 3   # the default: the lower finger, so 3 will do


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
                return m3pick.wrist_choices(grip, yaw, 90, m3pick.opening_m(opening))
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
            return m3pick.wrist_choices(grip, yaw, 90, m3pick.opening_m(opened))
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
            way = m3pick.carry_arc(here, extra, high)
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
        return m3pick.wrist_choices(grip, 74, 90, m3pick.opening_m(opened), [other])
    squares, _off = wrist(m3pick.plan_grasp(cube, 0.03, floor_clear=0.003)[1][-1])
    assert len(squares) == 2 and abs(abs(squares[0] - squares[1]) - 90) <= 2
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
    assert len(looks) == 1 + 1 + len(m3pick.BOX_LOOKS)   # the start's, over the spot, then all round
    assert world.pose[2] == 180                          # home again


def test_left_both_out_of_reach_and_without_a_square_grip_is_said():
    """2026-10-02: the message for a cube out of reach and a block lying lengthwise to the bumper
    (its square grip puts a finger over it) raised ValueError -- `in` compared the dicts' arrays.
    (2026-10-02, later: that block is too near the right lidar now -- its mirror on the left is the
    square grip's; all three said at once.)"""
    world = World([(0.40, 0.0, 0, 0.03, 0.03), (0.156, -0.056, 107, 0.06, 0.03),   # read as turned 17
                   (0.156, 0.056, 73, 0.06, 0.03)])
    picker = run(world, "clean_up")
    said = picker._log[-1]["text"]
    assert picker.state["error"] is None, picker.state["error"]
    assert "out of reach" in said and "too near the right lidar" in said and "no square grip" in said, said
    assert len(world.boxes) == 3 and not world.dropped


def test_the_arm_keeps_off_the_front_right_lidar():
    """2026-10-02: the arm had hit the front-right lidar. Reaching front-right its camera -- 4.5 cm
    to either side of the arm -- rode over that corner, in poses the chassis' steps allowed: right
    in front of the lidar no grip is planned now, the arm itself refuses the pose that went in, and
    what it uses -- home, the looks, the carries -- still passes."""
    for centre in ((0.21, -0.13, 0.015), (0.175, -0.07, 0.015), (0.169, -0.115, 0.015), (0.212, -0.063, 0.015)):
        assert m3pick.plan_grasp(np.array(centre), 0.03, floor_clear=0.003) is None, centre
    why = m3kinematics.problem({1: 115, 2: 30, 3: 30, 4: 30, 5: 90})
    assert why and "front-right lidar" in why
    for pose in (m3pick.FOLDED, m3pick.LOOK, *({**m3pick.SIDE_LOOK, 1: j} for j in m3pick.BOX_LOOKS),
                 *({1: j, 2: c[0], 3: c[1], 4: c[2], 5: 90} for c in m3pick.CARRIES for j in range(0, 181, 15))):
        assert m3kinematics.problem(pose) is None, pose


def test_in_front_of_the_right_lidar_the_base_slides_right_when_it_may_and_says_so_when_not():
    """2026-10-02, the user's choice: the arm keeps 2.5 cm off the front-right lidar, and what is just
    in front of it is picked after the base slides right -- back before the drop."""
    cube = (0.20, -0.11, 0, 0.03, 0.03)
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
