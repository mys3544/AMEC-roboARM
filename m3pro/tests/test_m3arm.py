"""roboarm/m3arm.py with a pretend board: nothing moves before engage(), every pose sent is legal
and clear of the chassis, every step on the way is checked and then a move is one command, an
interrupt holds where it got to, follow() goes through poses without stopping, the gripper closes
only to a width."""

import math

import pytest

from roboarm import m3arm, m3kinematics
from roboarm.m3arm import ArmError

HOME = dict(zip(m3arm.JOINTS, m3arm.POSES["home"]))
SURVEY = dict(zip(m3arm.JOINTS, m3arm.POSES["survey"]))


class Board:
    """What the M3's board is sent, on a pretend clock that sleeping moves on."""

    def __init__(self, volts=11.8):
        self.sent, self.t, self.volts, self.times = [], 100.0, volts, []

    def send(self, pose, ms):
        self.sent.append((dict(pose), ms))
        self.times.append(self.t)

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def arm(self):
        return m3arm.M3Arm(self.send, lambda: self.volts, self.clock, self.sleep)


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


def test_a_move_is_one_command_over_its_whole_time_and_is_believed_on_the_way():
    board = Board()
    with board.arm() as arm:
        arm.engage()
        board.sent.clear()
        t0 = board.t
        assert arm.move_to(SURVEY, speed_dps=40, settle=False) == SURVEY
        assert abs(board.t - t0 - 60 / 40) < 0.05          # J2 180 -> 120 at 40 deg/s
    assert board.sent == [(SURVEY, 1500), (SURVEY, 0)]   # one sweep, then the hold on exit
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
    arm = board.arm()
    arm.engage()
    board.sent.clear()
    t0 = board.t
    arm.interrupt = lambda: board.t >= t0 + 0.75   # halfway through 1.5 s
    with pytest.raises(ArmError, match="interrupted"):
        arm.move_to(SURVEY, speed_dps=40)
    held, ms = board.sent[-1]
    assert ms == 0 and abs(held[2] - (HOME[2] + SURVEY[2]) / 2) <= 2   # frozen halfway, not at the goal


def test_follow_goes_through_without_stopping_and_checks_the_rounded_corners():
    board = Board()
    arm = board.arm()
    arm.engage()
    board.sent.clear()
    via = {1: 90, 2: 150, 3: 20, 4: 0, 5: 90}
    t0 = board.t
    assert arm.follow([via, SURVEY], speed_dps=60) == SURVEY
    (first, ms1), (second, _ms2) = board.sent
    assert first == {**via, 6: 0} and second == SURVEY
    assert board.times[1] - t0 < ms1 / 1000 - 0.05      # the second went out before the first arrived
    board.sent.clear()
    with pytest.raises(ArmError, match="chassis"):     # a way that dips onto the deck: nothing sent
        arm.follow([{2: 60, 3: 10, 4: 10}, SURVEY])
    assert board.sent == []


def test_the_gripper_closes_only_to_a_width():
    assert m3arm.gripper_angle(30) == 134 and m3arm.gripper_angle(35) in (124, 125)
    with pytest.raises(ArmError, match="table"):
        m3arm.gripper_angle(80)
    board = Board()
    arm = board.arm()
    arm.engage()
    with pytest.raises(ArmError, match="width"):
        arm.close_gripper()
    assert arm.close_gripper(width_mm=40)[m3arm.GRIPPER] == 115


def test_a_low_battery_refuses_to_start():
    with pytest.raises(ArmError, match="charge"), Board(volts=10.3).arm():
        pass


def test_the_guard_passes_the_poses_the_arm_has_been_in_on_the_robot():
    for pose in (m3arm.POSES["home"], m3arm.POSES["survey"], (90, 150, 12, 20, 90, 0), (90, 90, 90, 90, 90, 0)):
        assert m3kinematics.problem(dict(zip(m3arm.JOINTS, pose))) is None
    down = -math.degrees(math.asin(m3kinematics.frames(SURVEY)["camera"][2, 0]))   # the DCW2 link looks along x
    assert 33 < down < 37   # measured on the robot: 34.9 (the URDF alone: 30; the SAG adds the rest)


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
