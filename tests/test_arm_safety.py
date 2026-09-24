"""Safety-layer tests. No hardware: a fake board stands in for the SDK.

These are the rules we do not want to rediscover on a real arm.
"""

import itertools
import struct
import threading

import pytest

from roboarm import arm as arm_mod
from roboarm import config as cfg
from roboarm.arm import Arm, ArmError

RESTING = {1: 91, 2: 86, 3: 89, 4: 87, 5: 90, 6: 31}


def _raw_for_angle(joint: int, angle: int) -> int:
    """Inverse of cfg.angle_from_raw, so the fake board can answer direct queries."""
    if joint <= 4:
        return round((angle - 180) * (3100 - 900) / (0 - 180) + 900)
    if joint == 5:
        return round(angle * (3700 - 380) / 270 + 380)
    return round(angle * (3100 - 900) / 180 + 900)


class FakeBoard:
    """Stands in for Rosmaster. Records commands and mimics servo motion."""

    def __init__(self, pose=None, version=3.5, voltage=12.0, sticky=None, drop=None, sag=None,
                 batch_blind=None, dead=None):
        self.pose = dict(pose or RESTING)
        self.version = version
        self.voltage = voltage
        self.sticky = sticky or {}  # joint -> (lo, hi) it can physically reach
        self.drop = dict(drop or {})  # joint -> how many reads still report -1
        self.sag = sag or {}          # joint -> degrees it falls short under load
        self.batch_blind = set(batch_blind or ())  # joints the BATCH call never returns
        self.dead = set(dead or ())               # joints that answer NEITHER reader
        self.sent: list[tuple[list[int], int]] = []
        self.raw_sent: list[tuple[list[int], int]] = []   # hand-built packets, pulses
        self.torque = True
        self.ctrl_enabled = False

    # -- the handful of SDK calls Arm actually uses --
    def create_receive_threading(self):
        pass

    def get_version(self):
        return self.version

    def get_battery_voltage(self):
        return self.voltage

    def set_uart_servo_ctrl_enable(self, on):
        self.ctrl_enabled = on

    def set_uart_servo_torque(self, on):
        self.torque = on

    def get_uart_servo_angle_array(self):
        out = []
        for j in cfg.JOINT_IDS:
            if j in self.batch_blind or j in self.dead or self.drop.get(j, 0) > 0:
                if j not in self.batch_blind and j not in self.dead:
                    self.drop[j] -= 1
                out.append(-1)
            else:
                out.append(self.pose[j])
        return out

    def get_uart_servo_value(self, sid):
        # Direct query still works for a servo the batch call cannot see -- unless
        # the servo is genuinely off the bus, when nothing answers.
        if sid in self.dead:
            return 0, -1
        return sid, _raw_for_angle(sid, self.pose[sid])

    def set_uart_servo_angle_array(self, angles, run_time=500):
        if not all(cfg.SDK_RANGE[j][0] <= a <= cfg.SDK_RANGE[j][1]
                   for j, a in zip(cfg.JOINT_IDS, angles)):
            return  # "angle_s input error!" and nothing happens, as the real SDK does
        self.sent.append((list(angles), run_time))
        for joint, angle in zip(cfg.JOINT_IDS, angles):
            self._put(joint, angle)

    # What arm._arm_ctrl() reaches for to build the array packet by hand.
    _Rosmaster__HEAD = 0xFF
    _Rosmaster__DEVICE_ID = 0xFC
    _Rosmaster__COMPLEMENT = 257 - 0xFC
    _Rosmaster__delay_time = 0.002
    _Rosmaster__arm_ctrl_enable = True
    FUNC_ARM_CTRL = 0x23

    @property
    def ser(self):
        return self

    def write(self, cmd):
        """The FUNC_ARM_CTRL packet: six little-endian pulses, the run time, a checksum."""
        assert cmd[:4] == [0xFF, 0xFC, len(cmd) - 2, 0x23]
        assert cmd[-1] == sum(cmd[:-1], 257 - 0xFC) & 0xFF
        *pulses, run_time = struct.unpack("<7h", bytes(cmd[4:18]))
        self.raw_sent.append((list(pulses), run_time))
        for joint, pulse in zip(cfg.JOINT_IDS, pulses):
            self._put(joint, cfg.angle_from_raw(joint, pulse))

    def _put(self, joint, angle):
        lo, hi = self.sticky.get(joint, (-999, 999))
        self.pose[joint] = min(max(angle - self.sag.get(joint, 0), lo), hi)


def build(monkeypatch, **kw) -> tuple[Arm, FakeBoard]:
    board = FakeBoard(**kw)
    monkeypatch.setattr(arm_mod, "Rosmaster", lambda com=None: board)
    monkeypatch.setattr(arm_mod.time, "sleep", lambda _s: None)
    return Arm(), board


# ------------------------------------------------------------------ startup --
def test_refuses_when_board_silent(monkeypatch):
    a, _ = build(monkeypatch, version=-1)
    with pytest.raises(ArmError, match="holding the port"):
        a.__enter__()


def test_refuses_on_flat_battery(monkeypatch):
    a, _ = build(monkeypatch, voltage=9.4)
    with pytest.raises(ArmError, match="below"):
        a.__enter__()


def test_connecting_to_a_sagged_arm_is_allowed(monkeypatch):
    # An unpowered arm sags past its limits routinely (seen on J2, J3, J4 and J6).
    # Refusing to connect would block the only thing that recovers it.
    a, _ = build(monkeypatch, pose={**RESTING, 3: -57})
    with a:
        assert a.out_of_range()[3] == -57, "it should still be reported, just not fatal"


def test_moving_eases_a_sagged_joint_back_into_range(monkeypatch):
    a, board = build(monkeypatch, pose={**RESTING, 3: -57})
    with a:
        a.move_to({1: 100})
        assert board.pose[3] >= cfg.SAFE_LIMITS[3][0]


def test_enables_the_command_gate(monkeypatch):
    a, board = build(monkeypatch)
    with a:
        assert board.ctrl_enabled is True, "commands are silently dropped when this is off"


# ------------------------------------------------------------------- motion --
def test_rejects_target_outside_safe_limits(monkeypatch):
    a, _ = build(monkeypatch)
    with a, pytest.raises(ArmError, match="safe range"):
        a.move_to({2: cfg.SAFE_LIMITS[2][1] + 5})


def test_rejects_unknown_joint(monkeypatch):
    a, _ = build(monkeypatch)
    with a, pytest.raises(ArmError, match="no such joint"):
        a.move_to({9: 90})


def test_no_step_exceeds_the_limit(monkeypatch):
    a, board = build(monkeypatch)
    with a:
        board.sent.clear()
        a.move_to({1: 150})  # 59 degrees away from 91
        previous = RESTING[1]
        for angles, _run_time in board.sent:
            assert abs(angles[0] - previous) <= cfg.MAX_STEP_DEG + 1
            previous = angles[0]
        assert previous == 150


def test_untouched_joints_keep_their_angle(monkeypatch):
    a, board = build(monkeypatch)
    with a:
        a.move_to({1: 120})
        assert {j: board.pose[j] for j in (2, 3, 4, 5)} == {
            j: RESTING[j] for j in (2, 3, 4, 5)
        }


def test_a_jammed_joint_is_reported_honestly_not_raised(monkeypatch):
    # J2 jams well short of a legal target. Moves are open loop: the pose that
    # comes back is the one the arm is actually in, and the caller decides.
    goal = cfg.SAFE_LIMITS[2][1]
    a, _ = build(monkeypatch, sticky={2: (-999, goal - 20)})
    with a:
        assert a.move_to({2: goal})[2] == goal - 20


def test_move_to_current_pose_sends_nothing(monkeypatch):
    a, board = build(monkeypatch)
    with a:
        board.sent.clear()
        a.move_to({1: RESTING[1]})
        assert board.sent == []


# ------------------------------------------------------------------ gripper --
def test_closing_on_an_object_is_not_a_failure(monkeypatch):
    # A rising servo angle CLOSES this gripper, so an object stops the fingers
    # BELOW the closed target -- here at 120, a 42 mm gap.
    a, _ = build(monkeypatch, pose={**RESTING, 6: cfg.GRIPPER_OPEN}, sticky={6: (-999, 120)})
    with a:
        a.close_gripper()  # must not raise
        assert a.grasped() is True


def test_closing_on_nothing_reports_no_grasp(monkeypatch):
    a, _ = build(monkeypatch, pose={**RESTING, 6: cfg.GRIPPER_OPEN})
    with a:
        a.close_gripper()
        assert a.grasped() is False


def test_gripper_angle_and_gap_run_in_opposite_directions():
    """Measured on the robot: a rising servo angle closes the fingers."""
    gaps = [cfg.GRIPPER_GAP_MM[a] for a in sorted(cfg.GRIPPER_GAP_MM)]
    assert all(b < a for a, b in itertools.pairwise(gaps)), "gap must shrink as J6 rises"
    assert cfg.GRIPPER_GAP_MM[cfg.GRIPPER_OPEN] == max(gaps)
    assert cfg.GRIPPER_OPEN < cfg.GRIPPER_CLOSED
    # 177 is where the finger bodies collide; never command into it.
    assert cfg.GRIPPER_CLOSED < 177


# ------------------------------------------------------------------- torque --
def test_exit_leaves_the_arm_holding_not_limp(monkeypatch):
    a, board = build(monkeypatch)
    with a:
        pass
    assert board.torque is True
    assert board.sent, "exit should command a hold"


def test_engage_targets_the_present_pose_so_nothing_snaps(monkeypatch):
    a, board = build(monkeypatch)
    with a:
        board.sent.clear()
        a.engage()
    commanded, _run_time = board.sent[0]
    assert commanded == [RESTING[j] for j in cfg.JOINT_IDS]


def test_engage_pulls_an_out_of_range_joint_back_inside(monkeypatch):
    # The whole arm collapsed while unpowered: J2 past 180, J3 and J4 below zero.
    collapsed = {1: 100, 2: 189, 3: -23, 4: -2, 5: 88, 6: 28}
    a, board = build(monkeypatch, pose=collapsed)
    with a:
        a.engage()
    for joint in cfg.JOINT_IDS:
        lo, hi = cfg.SAFE_LIMITS[joint]
        assert lo <= board.pose[joint] <= hi, f"J{joint} left outside its range"


def test_hold_does_not_command_an_out_of_range_joint(monkeypatch):
    a, board = build(monkeypatch)
    with a:
        board.pose[3] = -57  # drifts out of range while we are connected
        board.sent.clear()
        a.hold()
        assert board.sent == [], "commanding an out-of-range pose is silently dropped"


# ------------------------------------------------------------------ reading --
def test_a_flaky_read_is_retried_not_reported_as_a_fault(monkeypatch):
    # J3 drops out of two consecutive frames, then recovers -- exactly what the
    # real bus did right after a move to the home pose.
    a, _ = build(monkeypatch, drop={3: 2})
    with a:
        assert a.read()[3] == RESTING[3]


def test_a_genuinely_dead_joint_still_raises(monkeypatch):
    # Answers neither the batch call nor a direct query: that is a real fault and
    # must not be papered over by the fallback.
    # Connecting no longer reads, so the fault surfaces on the first read rather
    # than at connect -- but it must still surface, not be papered over.
    a, _ = build(monkeypatch, dead={3})
    with a, pytest.raises(ArmError, match="check the servo bus"):
        a.read()


# ------------------------------------------------------------------- droop --
def test_a_drooping_joint_is_not_chased(monkeypatch):
    # J2 settles a degree short of whatever it is told. Chasing that walks the
    # joint about and changes the load on its neighbours; the fingertip is
    # corrected in millimetres by grasp._reach_to() instead. So: one command
    # sequence to the goal, and the honest readback.
    goal = cfg.SAFE_LIMITS[2][1] - 8
    a, board = build(monkeypatch, sag={2: 1})
    with a:
        board.sent.clear()
        assert a.move_to({2: goal})[2] == goal - 1
        assert max(cmd[1] for cmd, _ in board.sent) == goal, "never commanded past the goal"


def test_the_gripper_never_squeezes_harder_on_an_object(monkeypatch):
    # A rising angle closes, so "harder" means commanding ABOVE the closed target.
    a, board = build(monkeypatch, pose={**RESTING, 6: cfg.GRIPPER_OPEN}, sticky={6: (-999, 120)})
    with a:
        a.close_gripper()
        overshoot = [cmd for cmd, _ in board.sent if cmd[5] > cfg.GRIPPER_CLOSED]
        assert overshoot == [], "must not command past the closed limit"
        assert a.grasped() is True


def test_raw_conversion_round_trips():
    for joint in cfg.JOINT_IDS:
        for angle in (30, 60, 90, 120):
            assert abs(cfg.angle_from_raw(joint, _raw_for_angle(joint, angle)) - angle) <= 1


def test_negative_angles_are_rounded_honestly():
    # The SDK's int(x + 0.5) truncates toward zero: raw 3288 is -15.4 deg and it
    # says -14. That was the hand-set rim pose of 2026-09-23.
    assert cfg.angle_from_raw(2, 3288) == -15
    for angle in (-1, -14, -30, 0, 5):
        assert cfg.angle_from_raw(2, cfg.raw_from_angle(2, angle)) == angle


class SdkRoundingBoard(FakeBoard):
    """The batch reader rounding the way the real SDK does below zero."""

    def get_uart_servo_angle_array(self):
        return [int(v + 0.5) if v != -1 else v for v in super().get_uart_servo_angle_array()]


def test_j2_below_zero_reads_true_not_the_sdks_degree_high(monkeypatch):
    board = SdkRoundingBoard(pose={**RESTING, 2: -15})
    monkeypatch.setattr(arm_mod, "Rosmaster", lambda com=None: board)
    monkeypatch.setattr(arm_mod.time, "sleep", lambda _s: None)
    assert board.get_uart_servo_angle_array()[1] == -14, "the SDK's reading"
    with Arm() as a:
        assert a.read()[2] == -15


def test_minus_one_is_an_angle_not_a_dead_servo(monkeypatch):
    # -1 is the SDK's "no reply" -- and, with J2 allowed below zero, a real angle.
    a, _ = build(monkeypatch, pose={**RESTING, 2: -1})
    with a:
        assert a.read()[2] == -1


def test_j2_below_zero_goes_as_one_hand_built_packet(monkeypatch):
    # The SDK's angle check drops the whole array for it, but the board takes
    # the pulse: the same packet, built by hand, all six joints at once.
    goal = cfg.SAFE_LIMITS[2][0]
    assert goal < 0
    a, board = build(monkeypatch, pose={**RESTING, 2: 10})
    with a:
        assert a.move_to({2: goal})[2] == goal
        pulses, _run_time = board.raw_sent[-1]
        assert pulses == [cfg.raw_from_angle(j, board.pose[j]) for j in cfg.JOINT_IDS]
        assert pulses[1] > 3100, "past the angle API's 0 degrees"
        assert all(angles[1] >= 0 for angles, _t in board.sent), "the SDK path stays legal"
        a.move_to({2: 20})
        assert board.pose[2] == 20
        assert board.sent[-1][0][1] == 20, "back above zero, the SDK's own call again"


def test_no_hand_built_packet_when_the_command_gate_is_off(monkeypatch):
    a, board = build(monkeypatch, pose={**RESTING, 2: 10})
    with a:
        board._Rosmaster__arm_ctrl_enable = False
        arm_mod.send_pose(board, {**RESTING, 2: -5}, 100)
        assert board.raw_sent == [] and board.pose[2] == 10


def test_hold_and_engage_keep_a_joint_below_zero(monkeypatch):
    a, board = build(monkeypatch, pose={**RESTING, 2: -10})
    with a:
        board.raw_sent.clear()
        a.hold()
        assert board.raw_sent and board.pose[2] == -10
        a.engage()
        assert board.pose[2] == -10


def test_below_the_raw_range_is_still_refused(monkeypatch):
    a, board = build(monkeypatch)
    with a, pytest.raises(ArmError, match="discard"):
        arm_mod.send_pose(board, {**RESTING, 2: -80}, 100)


def test_a_joint_the_batch_reader_never_returns_is_read_directly(monkeypatch):
    # Exactly what J4 did on the real arm: absent from every batch frame, but a
    # direct query answers with a stable value.
    a, _ = build(monkeypatch, batch_blind={4})
    with a:
        assert abs(a.read()[4] - RESTING[4]) <= 1


def test_an_untouched_joint_outside_the_safe_range_does_not_block_a_move(monkeypatch):
    # J2 rests at 114 while the safe cap is 112 -- moving J1 must still work, and
    # should ease J2 back inside rather than refusing.
    a, board = build(monkeypatch, pose={**RESTING, 2: cfg.SAFE_LIMITS[2][1] + 2})
    with a:
        a.move_to({1: 120})
        assert board.pose[1] == 120
        assert board.pose[2] <= cfg.SAFE_LIMITS[2][1]


# ----------------------------------------------------------- mast safety --
def test_clearance_model_matches_the_observed_near_miss():
    # J3=120 with everything else at 90 came within about 5 mm on the real robot.
    pose = {1: 90, 2: 90, 3: 120, 4: 90, 5: 90, 6: 120}
    assert 0.0 < cfg.mast_clearance(pose) < 0.010


def test_a_pose_that_would_foul_the_mast_is_refused(monkeypatch):
    # Every joint here is individually inside SAFE_LIMITS; only the combination
    # is dangerous, which is exactly what per-joint limits cannot catch.
    a, _ = build(monkeypatch)
    with a, pytest.raises(ArmError, match="camera mast"):
        a.move_to({2: 105, 3: 120, 4: 90})


def test_reaching_forward_is_unaffected(monkeypatch):
    a, board = build(monkeypatch)
    with a:
        a.move_to({2: 40, 3: 90, 4: 90})
        assert board.pose[2] == 40


def test_an_open_gripper_does_not_report_a_grasp(monkeypatch):
    """Open reads below the closed target, which used to look like a grasp."""
    a, _ = build(monkeypatch)
    with a:
        a.open_gripper()
        assert a.grasped() is False, "opening is not grasping"


# ---------------------------------------------------------------- interrupt --
def test_an_interrupted_move_stops_at_a_legal_intermediate_pose(monkeypatch):
    """The web app's stop button: trip `interrupt` and the glide abandons its next
    step, leaving the arm at the last pose it was sent -- never limp, never at the
    goal, and move_to() says so."""
    a, board = build(monkeypatch)
    sent_before_stop = 3
    a.interrupt = lambda: len(board.sent) >= sent_before_stop
    with a, pytest.raises(ArmError, match="interrupted"):
        a.move_to({1: 20})
    moves = [angles for angles, run_time in board.sent if run_time != 0]  # 0 = the exit hold
    assert len(moves) == sent_before_stop
    assert 20 < board.pose[1] < RESTING[1], "stopped part way, not at the goal"
    assert cfg.SAFE_LIMITS[1][0] <= board.pose[1] <= cfg.SAFE_LIMITS[1][1]


def test_without_an_interrupt_hook_moves_complete(monkeypatch):
    a, board = build(monkeypatch)
    with a:
        assert a.interrupt is None
        a.move_to({1: 20})
        assert board.pose[1] == 20


def test_a_glide_moves_in_the_background_and_answers_reads_meanwhile(monkeypatch):
    """A continuous scan yaws the base while frames are taken: glide() returns at
    once, read() works while it runs, and it ends at the goal in the same
    bounded steps as move_to."""
    a, board = build(monkeypatch)
    with a:
        assert a.glide({1: 20}, speed_dps=1000)[1] == 20
        assert a.read()[1] >= 20  # answered, not refused, while under way
        deadline = threading.Event()
        for _ in range(200):
            if not a.moving():
                break
            deadline.wait(0.01)
        assert not a.moving()
        assert board.pose[1] == 20
        steps = [angles for angles, run_time in board.sent if run_time != 0]
        assert len(steps) == 9 and all(
            abs(b[0] - c[0]) <= cfg.MAX_STEP_DEG for b, c in itertools.pairwise(steps))


def test_hold_stops_a_glide_where_it_is(monkeypatch):
    a, board = build(monkeypatch)
    with a:
        a.glide({1: 20}, speed_dps=100)  # 80 ms a step, 9 steps
        threading.Event().wait(0.12)
        a.hold()
        assert not a.moving()
        assert 20 < board.pose[1] < RESTING[1], "stopped part way, not at the goal"
        assert board.sent[-1][1] == 0, "the hold command"
        a.move_to({1: 20})
        assert board.pose[1] == 20


# --------------------------------------------------------- repeatable moves --
def _sent_poses(board):
    return [dict(zip(cfg.JOINT_IDS, angles)) for angles, _ in board.sent]


def test_a_repeatable_move_comes_down_onto_the_loaded_joints_from_above(monkeypatch):
    # A look pose must be arrived at the same way every time, or the servos'
    # deadband leaves the camera a degree or two elsewhere (2026-09-24, C930e).
    # Coming UP to it here, so the detour past the goal is visible.
    a, board = build(monkeypatch, pose={**RESTING, 2: 60, 3: 20, 4: 10})
    goal = {2: 70, 3: 30, 4: 20}
    with a:
        board.sent.clear()
        a.move_to(goal, repeatable=True)
        sent = _sent_poses(board)
        for joint, target in goal.items():
            path = [pose[joint] for pose in sent]
            peak = path.index(max(path))
            assert max(path) == target + Arm.APPROACH_DEG
            assert path[-1] == target
            assert path[peak:] == sorted(path[peak:], reverse=True), "only down after the peak"
        assert board.pose[2] == 70 and board.pose[3] == 30 and board.pose[4] == 20


def test_an_ordinary_move_takes_no_detour(monkeypatch):
    a, board = build(monkeypatch, pose={**RESTING, 2: 60, 3: 20, 4: 10})
    with a:
        board.sent.clear()
        a.move_to({2: 70, 3: 30, 4: 20})
        assert max(pose[2] for pose in _sent_poses(board)) == 70
