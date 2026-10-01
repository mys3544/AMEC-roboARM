"""roboarm/m3arm.py with a pretend board: nothing moves before engage(), every pose sent is legal
and clear of the chassis, moves go in small steps, an interrupt holds where it stopped, the
gripper closes only to a width."""

import threading
import time

import pytest

from roboarm import m3arm, m3kinematics
from roboarm.arm import ArmError

HOME = dict(zip(m3arm.JOINTS, m3arm.POSES["home"]))
SURVEY = dict(zip(m3arm.JOINTS, m3arm.POSES["survey"]))


class Board:
    """What the M3's board is sent, on a pretend clock that sleeping moves on."""

    def __init__(self, volts=11.8):
        self.sent, self.t, self.volts = [], 100.0, volts

    def send(self, pose, ms):
        self.sent.append((dict(pose), ms))

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def arm(self, **kw):
        return m3arm.M3Arm(self.send, lambda: self.volts, self.clock, self.sleep, **kw)


def test_nothing_moves_before_engage_and_engage_sends_a_whole_known_pose():
    board = Board()
    with board.arm() as arm:
        with pytest.raises(ArmError, match="engage"):
            arm.read()
        with pytest.raises(ArmError, match="engage"):
            arm.move_to({1: 100})
        assert board.sent == []
        with pytest.raises(ArmError, match="six"):
            arm.engage({1: 90})
        assert arm.engage() == HOME
        assert board.sent == [(HOME, m3arm.ENGAGE_MS)] and arm.read() == HOME


def test_a_move_goes_in_small_legal_steps_and_is_believed_on_the_way():
    board = Board()
    with board.arm(step_deg=5) as arm:
        arm.engage()
        board.sent.clear()
        assert arm.move_to(SURVEY) == SURVEY
    poses = [p for p, _ms in board.sent]
    assert poses[-2] == SURVEY and poses[-1] == SURVEY   # the last step, then the hold on exit
    last = HOME
    for p in poses[:-1]:
        assert max(abs(p[j] - last[j]) for j in m3arm.JOINTS) <= 5
        assert m3kinematics.problem(p) is None
        last = p
    arm2 = board.arm()
    arm2.engage()
    t0 = board.t
    arm2._send(SURVEY, 1000)   # halfway through a 1 s command, read() is halfway there
    board.t = t0 + 0.5
    assert abs(arm2.read()[2] - (HOME[2] + SURVEY[2]) / 2) <= 1


def test_what_is_not_safe_is_refused_before_anything_is_sent():
    board = Board()
    arm = board.arm()
    arm.engage()
    board.sent.clear()
    with pytest.raises(ArmError, match="safe range"):
        arm.move_to({2: 200})
    with pytest.raises(ArmError, match="chassis"):      # folding the arm down onto the robot
        arm.move_to({3: 180, 4: 90})
    with pytest.raises(ArmError, match="chassis"):   # down onto the front deck and its lidar
        arm.move_to({2: 60, 3: 10, 4: 10})
    with pytest.raises(ArmError, match="floor"):     # the fingertips into the floor in front
        arm.move_to({2: 0, 3: 30, 4: 60})
    assert board.sent == [] and arm.read() == HOME


def test_an_interrupt_holds_where_the_move_got_to():
    board = Board()
    arm = board.arm(step_deg=5)
    arm.engage()
    board.sent.clear()
    arm.interrupt = lambda: len(board.sent) >= 3
    with pytest.raises(ArmError, match="interrupted"):
        arm.move_to(SURVEY)
    (third, _ms), (held, ms) = board.sent[-2], board.sent[-1]
    assert ms == 0 and held == third != SURVEY   # stopped at the last step sent, not the goal


def test_a_glide_returns_at_once_and_hold_stops_it():
    board = Board()
    arm = m3arm.M3Arm(board.send, lambda: 11.8, time.monotonic, time.sleep, step_deg=5)
    arm._send(HOME, 0)   # as engage() leaves it, without its 3.3 s on the real clock
    goal = arm.glide(SURVEY, speed_dps=50)     # 60 deg of J2 at 50 deg/s: ~1.2 s
    assert goal == SURVEY and arm.moving()
    time.sleep(0.35)
    held = arm.hold()
    assert not arm.moving() and held != SURVEY and board.sent[-1] == (held, 0)


def test_the_gripper_closes_only_to_a_width_and_cannot_say_it_holds():
    assert m3arm.gripper_angle(30) == 134 and m3arm.gripper_angle(35) in (124, 125)
    with pytest.raises(ArmError, match="table"):
        m3arm.gripper_angle(80)
    board = Board()
    arm = board.arm()
    arm.engage()
    with pytest.raises(ArmError, match="width"):
        arm.close_gripper()
    assert arm.close_gripper(width_mm=40)[m3arm.GRIPPER] == 115
    with pytest.raises(ArmError, match="camera"):
        arm.grasped()
    with pytest.raises(ArmError, match="torque"):
        arm.release()


def test_a_low_battery_refuses_to_start():
    with pytest.raises(ArmError, match="charge"), Board(volts=10.3).arm():
        pass


def test_the_guard_passes_the_poses_the_arm_has_been_in_on_the_robot():
    for pose in (m3arm.POSES["home"], m3arm.POSES["survey"], (90, 150, 12, 20, 90, 0), (90, 90, 90, 90, 90, 0)):
        assert m3kinematics.problem(dict(zip(m3arm.JOINTS, pose))) is None
    _position, down = m3kinematics.camera(SURVEY)
    assert 33 < down < 37   # measured on the robot: 34.9 (the URDF alone: 30; the SAG adds the rest)


def test_threads_are_not_left_behind():
    assert not [t for t in threading.enumerate() if t.name == "m3-arm-glide" and t.is_alive()]


def test_a_move_planned_on_the_measured_floor_may_come_closer_to_it_but_not_too_close():
    board = Board()
    arm = board.arm()
    arm.engage()
    low = m3kinematics.reach((0.224, 0.007, 0.019), 70, floor_clear=0.003)   # a 30 mm cube placed
    assert low is not None and m3kinematics.problem(low) is not None         # the default refuses it
    with pytest.raises(ArmError, match="floor"):
        arm.move_to(low)
    with pytest.raises(ArmError, match="minimum"):
        arm.move_to(low, floor_clear=0.001)
    above = m3kinematics.reach((0.224, 0.007, 0.08), 70)
    arm.move_to(above)                                   # from just above, a short way down
    assert arm.move_to(low, floor_clear=0.003)[2] == low[2]
