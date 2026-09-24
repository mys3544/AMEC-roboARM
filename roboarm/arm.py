"""The only place in the project that commands the arm servos.

Every safety rule lives here, so a demo script written in a hurry cannot skip one.
Angles are servo degrees throughout -- the units the SDK speaks. Converting to and
from URDF joint angles is kinematics.py's job, not this module's.

Three rules learned the hard way against the real hardware:

  1. Never fire two read calls back to back with no gap -- that confuses the SDK's
     async receive thread and both start returning spurious -1s. Spaced ~50ms
     apart they coexist fine, which matters because neither reader is reliable
     alone: the batch call has been seen to drop one joint (J4) indefinitely while
     a direct query of that same servo answered with a stable value. read() uses
     the batch call and falls back to a spaced direct query for whatever it drops.

  2. The batch reader is unclamped and therefore honest: a joint parked outside its
     range reports its real angle. get_uart_servo_angle() masks that as -1 -- the
     same value it uses for a servo that never replied.

  3. Before enabling torque, command the pose the arm is already in. Otherwise a
     servo may snap to a stale setpoint.
"""

from __future__ import annotations

import contextlib
import math
import struct
import threading
import time
from collections.abc import Callable

try:
    from Rosmaster_Lib import Rosmaster
except ImportError:
    # No vendored SDK on this machine (vendor/ is populated by tools/vendor_sdk.sh
    # on the robot). Everything that does not touch hardware -- the tests, the
    # simulator behind `tools/webapp.py --sim` -- still works; connecting does not.
    Rosmaster = None

from roboarm import config as cfg
from roboarm.config import Pose

_PORT_BUSY_HINT = "pkill -f 'rosmaster_mai[n]'"


def read_pose(bot: Rosmaster, attempts: int = 4) -> dict[int, int | None]:
    """Read every joint. None means that joint answered neither reader.

    The batch call is one transaction and normally the right tool, but it has been
    seen to drop a joint (J4) from every frame while a direct query of that same
    servo answered with a stable value. So: batch first, then ask the stragglers
    directly. The 50ms gaps are load-bearing -- fired back to back the two readers
    corrupt each other, spaced they do not. Measured 10/10 complete poses this way.

    Values are never merged across attempts: a stitched frame would describe a pose
    the arm was never actually in.

    Anything the batch gives at or below ZERO is asked for directly too. Since J2
    may go below zero, -1 is a real angle as well as the SDK's "no reply", and the
    SDK rounds every negative angle a degree high (see cfg.angle_from_raw); the
    direct query returns the raw count, which settles both.
    """
    pose: dict[int, int | None] = {}
    for _ in range(attempts):
        pose = dict(zip(cfg.JOINT_IDS, bot.get_uart_servo_angle_array()))
        doubtful = [j for j, v in pose.items() if v <= 0]
        if not doubtful:
            return pose
        for joint in doubtful:
            time.sleep(0.05)
            for _ in range(3):
                read_id, raw = bot.get_uart_servo_value(joint)
                if read_id == joint and raw > 0:
                    pose[joint] = cfg.angle_from_raw(joint, raw)
                    break
                time.sleep(0.05)
            else:
                if pose[joint] == -1:
                    pose[joint] = None   # answered neither reader
        if all(v is not None for v in pose.values()):
            return pose
    return pose


def send_pose(bot: Rosmaster, pose: Pose, run_time: int) -> None:
    """Command all six joints, refusing anything the SDK would silently drop.

    set_uart_servo_angle_array() validates its input and, when something is out
    of range, prints "angle_s input error!" and returns having done NOTHING. The
    move usually still looks fine because the next command lands, so the failure
    is invisible. We have seen those errors appear intermittently; this turns
    them into a loud one instead of a dropped command.

    A joint below its SDK_RANGE (J2 near the far rim) is refused by the SDK's
    ANGLE check, not by the board: the array packet carries pulses, and the
    firmware moves J2 to pulse 3136 (-3 deg) as readily as to 3100 (0). So that
    pose goes as the very same packet, built by _arm_ctrl(). Measured 2026-09-23,
    arm stretched level: the hand-built packet reached -4 for -3 (the usual
    degree of sag); a raw single-servo command 2 ms after the array was DROPPED
    (J2 went to the array's 0), and 50 ms after worked but swung toward 0 first.
    """
    below, bad = False, []
    for j in cfg.JOINT_IDS:
        lo, hi = cfg.SDK_RANGE[j]
        pulse = cfg.raw_from_angle(j, pose[j])
        if pose[j] < lo and cfg.RAW_PULSE_RANGE[0] <= pulse <= cfg.RAW_PULSE_RANGE[1]:
            below = True
        elif not (lo <= pose[j] <= hi):
            bad.append(f"J{j}={pose[j]} outside the SDK's {cfg.SDK_RANGE[j]}")
    if bad:
        raise ArmError("the SDK would discard this command: " + ", ".join(bad))
    if below:
        _arm_ctrl(bot, [cfg.raw_from_angle(j, pose[j]) for j in cfg.JOINT_IDS], run_time)
    else:
        bot.set_uart_servo_angle_array([pose[j] for j in cfg.JOINT_IDS], run_time=run_time)


def _arm_ctrl(bot: Rosmaster, pulses: list[int], run_time: int) -> None:
    """The packet set_uart_servo_angle_array() sends, byte for byte, minus its
    angle check. Reaches into the vendored SDK's private fields, so a re-vendored
    SDK that renamed them fails here, loudly, instead of moving anything."""
    try:
        if not bot._Rosmaster__arm_ctrl_enable:
            return  # the SDK drops every arm command then; so do we
        cmd = [bot._Rosmaster__HEAD, bot._Rosmaster__DEVICE_ID, 0x00, bot.FUNC_ARM_CTRL]
        for value in (*pulses, min(max(int(run_time), 0), 2000)):
            cmd += struct.pack("<h", int(value))
        cmd[2] = len(cmd) - 1
        cmd.append(sum(cmd, bot._Rosmaster__COMPLEMENT) & 0xFF)
        bot.ser.write(cmd)
        time.sleep(bot._Rosmaster__delay_time)
    except AttributeError as exc:
        raise ArmError(f"cannot build the arm packet by hand on this SDK: {exc}") from exc


class ArmError(RuntimeError):
    """Raised instead of moving when something is not safe or not verifiable."""


class Arm:
    """Safe wrapper around the arm servos. Use as a context manager."""

    def __init__(self, port: str = cfg.SERIAL_PORT, step_deg: float = cfg.MAX_STEP_DEG):
        self.port = port
        self.step_deg = step_deg
        self._bot: Rosmaster | None = None
        # What the gripper was last told to do. grasped() is only meaningful after
        # a close, so without this an OPEN gripper reads "below closed" and reports
        # a grasp that never happened.
        self._gripper_target: int | None = None
        # Asked between the steps of every glide. Return True to abandon the move:
        # the arm stops at the last intermediate pose it was sent -- a legal one,
        # since every step is -- and move_to() raises ArmError. This is how a
        # remote stop button reaches into a move already under way, without the
        # SDK having any notion of cancelling a command.
        self.interrupt: Callable[[], bool] | None = None
        # One talker on the serial line at a time. A glide() runs its steps on a
        # thread of its own so read() can be answered meanwhile; the two must not
        # interleave bytes (two readers fired back to back already corrupt each
        # other, see read_pose), so every SDK call goes through this lock.
        self._io = threading.RLock()
        self._glide_thread: threading.Thread | None = None
        self._glide_cancel = threading.Event()

    # ------------------------------------------------------------- lifecycle --
    # ruff wants `Self` here, which needs Python 3.11 or a typing_extensions
    # dependency; the container is 3.10 and this is not worth either.
    def __enter__(self) -> Arm:
        if Rosmaster is None:
            raise ArmError(
                "the Rosmaster SDK is not available here -- run tools/vendor_sdk.sh on "
                "the robot, or use the simulator (tools/webapp.py --sim)"
            )
        self._bot = Rosmaster(com=self.port)
        self._bot.create_receive_threading()
        time.sleep(1.0)

        if self._bot.get_version() == -1:
            raise ArmError(
                f"no reply from the board on {self.port}. Something else is holding "
                f"the port -- on the host run:  {_PORT_BUSY_HINT}"
            )
        # Commands are silently discarded when this gate is off, which looks
        # exactly like broken hardware. Assert it rather than assume it.
        self._bot.set_uart_servo_ctrl_enable(True)

        voltage = self._bot.get_battery_voltage()
        if voltage < cfg.MIN_BATTERY_V:
            raise ArmError(f"battery {voltage:.1f} V is below {cfg.MIN_BATTERY_V} V -- charge it")

        return self

    def __exit__(self, *exc) -> None:
        if self._bot is not None:
            # Leave the arm holding, never limp. If the hold itself fails we are
            # already in trouble and cannot do better -- but we must not let a
            # cleanup failure replace the error that actually got us here.
            with contextlib.suppress(ArmError):
                self.hold()
            del self._bot
            self._bot = None

    @property
    def bot(self) -> Rosmaster:
        if self._bot is None:
            raise ArmError("arm is not connected -- use `with Arm() as arm:`")
        return self._bot

    # ----------------------------------------------------------------- state --
    def read(self, attempts: int = 4) -> Pose:
        """Current servo angles, as one self-consistent frame. Raises if a joint is
        answering neither reader -- that is a real fault, not a dropped frame."""
        with self._io:
            pose = read_pose(self.bot, attempts)
        silent = [j for j, v in pose.items() if v is None]
        if silent:
            raise ArmError(
                f"no data from joint(s) {silent} after {attempts} batch reads and "
                f"direct queries -- check the servo bus"
            )
        return {j: v for j, v in pose.items() if v is not None}

    def out_of_range(self, pose: Pose | None = None) -> Pose:
        """Joints sitting outside their hard limits. Real positions, not faults."""
        pose = self.read() if pose is None else pose
        return {
            j: v
            for j, v in pose.items()
            if not (cfg.HARD_LIMITS[j][0] <= v <= cfg.HARD_LIMITS[j][1])
        }

    # ---------------------------------------------------------------- motion --
    def _validate(self, goal: Pose) -> Pose:
        for joint, angle in goal.items():
            if joint not in cfg.JOINT_IDS:
                raise ArmError(f"no such joint: {joint}")
            lo, hi = cfg.SAFE_LIMITS[joint]
            if not (lo <= angle <= hi):
                raise ArmError(f"J{joint}={angle} is outside its safe range {lo}..{hi}")
        return goal

    def _assert_clear_of_mast(self, goal: Pose) -> None:
        """Per-joint limits cannot express this: whether the arm fouls the mast
        depends on J2, J3 and J4 together. J3=120 alone came within 5 mm of it while
        every joint was individually well inside its range."""
        clearance = cfg.mast_clearance(goal)
        if clearance < cfg.MIN_MAST_CLEARANCE_M:
            raise ArmError(
                f"pose would come within {clearance * 1000:.0f} mm of the camera mast "
                f"(minimum {cfg.MIN_MAST_CLEARANCE_M * 1000:.0f} mm): "
                + " ".join(f"J{j}={goal[j]}" for j in (2, 3, 4))
            )

    def _send(self, pose: Pose, run_time: int) -> None:
        with self._io:
            send_pose(self.bot, pose, run_time)

    def battery(self) -> float:
        with self._io:
            return float(self.bot.get_battery_voltage())

    def _plan(self, targets: Pose) -> tuple[Pose, Pose]:
        """(where we are, the legal goal) for `targets`, partial poses allowed."""
        start = self.read()
        # Only what the caller actually asked for has to be a legal target. Joints
        # they did not mention keep their current angle, eased back inside the safe
        # envelope if they have drifted out of it -- otherwise one joint resting
        # past a limit (or a limit we later tightened) blocks every future move.
        self._validate(targets)
        goal = dict(start)
        for joint in cfg.JOINT_IDS:
            lo, hi = cfg.SAFE_LIMITS[joint]
            goal[joint] = int(min(max(targets.get(joint, start[joint]), lo), hi))
        self._assert_clear_of_mast(goal)
        return start, goal

    def move_to(self, targets: Pose, speed_dps: float = 40.0, settle: bool = True,
                repeatable: bool = False) -> Pose:
        """Move to `targets` (partial poses allowed), interpolated so no single step
        exceeds step_deg. Bounding the step bounds the speed, and keeps every
        intermediate pose a legal one.

        Open loop on purpose. Real droop on this arm is about a degree (the -6 deg
        once seen at J2=120 was the arm pushing into the mast), which is inside the
        readback tolerance, and a joint-space correction cannot fix what matters --
        the fingertip landing millimetres short at a top-down pitch. grasp._reach_to()
        corrects that where it is measured, in millimetres.

        `settle=False` skips the quarter-second pause and the readback at the end
        and returns the goal: for a transit that the next move follows straight
        on from (the lift, the trip to the drop pose, the way back), where the
        pause is a visible stop and nothing needs the readback.

        `repeatable=True` makes the last approach to J2..J4 always come down from
        above (see APPROACH_DEG): for the camera's look poses, where what matters
        is landing in the same place every time.
        """
        self._cancel_glide()
        start, goal = self._plan(targets)
        if repeatable:
            above = self._above(goal)
            # Leaning back is leaning toward the mast: the detour must be as clear
            # of it as any goal (raises before anything moves if not).
            self._assert_clear_of_mast(above)
            if above != goal:
                self._glide(start, above, speed_dps)
                time.sleep(0.15)
                start = self.read()
        if not self._glide(start, goal, speed_dps):
            return start
        if not settle:
            return goal

        time.sleep(0.25)  # let the last segment settle before believing the readback
        return self.read()

    # The joints gravity loads, and how far above its goal each one is sent first
    # when a move must be repeatable. These joints stop inside a deadband of about a
    # degree either side of a command, WHICH side depending on the way they came:
    # with the C930e on the wrist (2026-09-24) J3 read 22 arriving from below and 24
    # from above for the same command 24, and at a look pose that difference moved
    # the picture 10..20 mm on the table. Re-commanding by the shortfall does not
    # help -- a one-degree nudge lands inside the deadband or jumps across it -- but
    # arriving from the same side every time lands in the same place. From above is
    # the side that read the goal: coming down with gravity.
    APPROACH_JOINTS = (2, 3, 4)
    APPROACH_DEG = 4

    def _above(self, goal: Pose) -> Pose:
        above = dict(goal)
        for joint in self.APPROACH_JOINTS:
            above[joint] = min(goal[joint] + self.APPROACH_DEG, cfg.SAFE_LIMITS[joint][1])
        return above

    def glide(self, targets: Pose, speed_dps: float = 12.0) -> Pose:
        """Start moving toward `targets` and return at once, with the legal goal.

        The steps are sent from a thread of their own, so read() keeps answering
        while the arm is under way -- that is what a continuous camera scan needs:
        the base yawing steadily while frames are taken and their yaw read. It is
        the same interpolated, interruptible glide as move_to(); moving() says
        whether it is still going, and hold() or any move_to() stops it where it
        is. A glide the SDK refuses mid-way ends quietly: moving() goes False
        short of the goal, and the caller reads where it got to.
        """
        self._cancel_glide()
        start, goal = self._plan(targets)
        cancel = threading.Event()
        self._glide_cancel = cancel

        def run() -> None:
            with contextlib.suppress(ArmError):
                self._glide(start, goal, speed_dps, cancel)

        self._glide_thread = threading.Thread(target=run, name="arm-glide", daemon=True)
        self._glide_thread.start()
        return goal

    def moving(self) -> bool:
        thread = self._glide_thread
        return thread is not None and thread.is_alive()

    def _cancel_glide(self) -> None:
        thread = self._glide_thread
        if thread is not None and thread.is_alive():
            self._glide_cancel.set()
            thread.join(timeout=5.0)
        self._glide_thread = None

    def _glide(self, start: Pose, goal: Pose, speed_dps: float,
               cancel: threading.Event | None = None) -> bool:
        """Interpolated move, no step larger than step_deg. Returns False if already there.

        Bounding the step bounds the speed and keeps every intermediate pose legal.
        A loaded joint also simply cannot complete a large jump in one command.
        With `cancel`, the move stops at its last sent step as soon as the event
        is set (that is a glide() being held or superseded).
        """
        # A joint that has sagged out of range cannot be glided back: every
        # intermediate angle is out of range too, and the SDK discards the lot.
        # Clamping the START means the first command jumps it to the nearest legal
        # angle and the rest of the move glides normally -- the same compromise
        # engage() makes, and the only one available.
        start = {
            j: int(min(max(v, *cfg.SAFE_LIMITS[j][:1]), cfg.SAFE_LIMITS[j][1]))
            for j, v in start.items()
        }

        biggest = max(abs(goal[j] - start[j]) for j in cfg.JOINT_IDS)
        if biggest == 0:
            return False

        steps = max(1, math.ceil(biggest / self.step_deg))
        run_time = max(20, int(1000 * (biggest / steps) / speed_dps))
        for i in range(1, steps + 1):
            if self.interrupt is not None and self.interrupt():
                raise ArmError("move interrupted -- the arm is holding where it stopped")
            frac = i / steps
            self._send(
                {j: round(start[j] + frac * (goal[j] - start[j])) for j in cfg.JOINT_IDS},
                run_time,
            )
            if cancel is None:
                time.sleep(run_time / 1000)
            elif cancel.wait(run_time / 1000):
                return True
        return True

    def home(self, **kw) -> Pose:
        return self.move_to(dict(cfg.HOME_POSE), **kw)

    def set_gripper(self, angle: int, **kw) -> Pose:
        self._gripper_target = angle
        return self.move_to({cfg.GRIPPER_ID: angle}, **kw)

    def open_gripper(self, **kw) -> Pose:
        return self.set_gripper(cfg.GRIPPER_OPEN, **kw)

    def close_gripper(self, **kw) -> Pose:
        # An object stops the fingers short of the target, which is a successful
        # grasp, not a fault. grasped() tells the two apart.
        return self.set_gripper(cfg.GRIPPER_CLOSED, **kw)

    def grasped(self) -> bool:
        """True if a close attempt was stopped short by an object between the fingers.

        A RISING servo angle closes this gripper, so an object holds it BELOW the
        closed target. Two ways to get this wrong, both of which we did:

          * the comparison used to run the other way, reporting the opposite of
            reality once the polarity was corrected;
          * position alone cannot tell "open" from "holding something", because a
            wide object stops the fingers exactly where open would leave them. So
            this only answers after close_gripper() actually asked them to close.
        """
        if self._gripper_target != cfg.GRIPPER_CLOSED:
            return False
        return self.read()[cfg.GRIPPER_ID] < cfg.GRIPPER_CLOSED - cfg.READBACK_TOLERANCE_DEG

    # ---------------------------------------------------------------- torque --
    def hold(self) -> Pose:
        """Freeze where we are. This is the e-stop: holding is safer than going limp,
        because a limp arm falls."""
        self._cancel_glide()
        pose = self.read()
        if not self.out_of_range(pose):
            self._send(pose, run_time=0)
        return pose

    def release(self) -> None:
        """Go limp so the arm can be moved by hand. IT WILL SAG -- support it."""
        self.bot.set_uart_servo_torque(False)

    def engage(self) -> Pose:
        """Re-stiffen, easing any joint that has drifted out of range back inside.

        An out-of-range joint cannot be glided back: the intermediate angles are out
        of range too, and the SDK silently drops them. So the clamped pose is
        commanded directly, with the longest run_time the firmware accepts.
        """
        start = self.read()
        safe = {}
        for joint, value in start.items():
            lo, hi = cfg.SAFE_LIMITS[joint]
            safe[joint] = int(min(max(value, lo), hi))

        self.bot.set_uart_servo_torque(True)
        time.sleep(0.3)
        # Pinning the target to where the arm already is (clamped) also stops a
        # servo snapping to whatever stale setpoint it still held.
        self._send(safe, run_time=2000)
        time.sleep(2.5)
        return self.read()
