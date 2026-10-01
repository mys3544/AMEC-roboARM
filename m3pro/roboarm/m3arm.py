"""The M3 Pro's arm, behind the same calls as roboarm.arm.Arm (the X3's), so that the bridge and
everything above it stay as they are. It is commanded over ROS 2 -- /arm6_joints, six servo
angles in degrees and a time in ms, to the micro-ROS board -- through a `send(pose, ms)` given
by the caller (tools/m3bridge.py publishes it); this module has no ROS in it.

What the M3's board does not do, and what that changes (surveyed 2026-09-30):
  * it reports no servo angles: read() is the pose last commanded, interpolated through a move
    under way -- believed, not measured; the camera is what checks where the arm really is;
  * the arm is limp at power-up until the first command, at a pose nobody knows: nothing moves
    until engage() has sent a known pose, slowly;
  * there is no torque switch: release() refuses;
  * the gripper servo stalls and can burn when told to close past an object: it closes only to
    a width, through Yahboom's width table, and grasped() cannot tell -- look instead.

Every pose sent is checked by roboarm.m3kinematics.problem (the chassis, the lidars, the floor)
and against per-joint limits -- and so is every pose on the way, every `step_deg`, before anything
is sent. Then a move is ONE command over its whole time: the board's servos sweep there together,
joint by joint in a straight line, which is the path checked. (Until 2026-10-01 each 5 deg step was
a command of its own, and the arm went stop-and-go the whole way.) follow() runs through several
poses without stopping at each. An interrupt freezes the arm where it is believed to be. Angles
are servo degrees; 90 is the URDF's zero.
"""

from __future__ import annotations

import contextlib
import itertools
import math
import threading
import time
from collections.abc import Callable

from roboarm import m3kinematics
from roboarm.arm import ArmError

Pose = dict[int, int]
JOINTS = (1, 2, 3, 4, 5, 6)
GRIPPER = 6
# the URDF's limits (+-90 deg about 90) for 1-5; the gripper 0 (widest) .. 180 (shut)
SAFE_LIMITS = {1: (0, 180), 2: (0, 180), 3: (0, 180), 4: (0, 180), 5: (0, 180), 6: (0, 180)}
POSES = {
    "home": (90, 180, 5, 0, 90, 0),     # Yahboom's mapping pose: folded over the base, camera level
    "survey": (90, 120, 30, 0, 90, 0),  # the camera 34.9 deg down, 0.358 m up (measured)
}
# Yahboom's table, object width (mm) -> the gripper angle that just holds it; 0 is widest open
GRIPPER_WIDTH = {0: 180, 20: 152, 30: 134, 40: 115, 50: 95, 60: 57}
GRIPPER_OPEN = 30          # wider than the 60 mm entry: the kit's biggest piece is 40 mm
ENGAGE_MS = 3000           # the first command, from wherever the limp arm is
MIN_BATTERY_V = 10.5       # the M3's board went silent at ~10.4 V under load (2026-09-30)
POLL_S = 0.03              # how often a move under way looks for an interrupt
BLEND = 0.8                # follow(): the next pose goes out this far through the one before


def glide_problem(start: Pose, poses: list[Pose], floor_clear: float | None = None,
                  step_deg: float = 5.0) -> str | None:
    """Why gliding from `start` through `poses` (whole poses, joints 1-5 at least) would not be
    allowed -- every pose, and every step_deg on the way as follow() drives it, its corners cut at
    BLEND -- or None. For planners to choose a way before asking the arm for it."""
    clear = m3kinematics.FLOOR_CLEAR if floor_clear is None else floor_clear
    last = start
    for i, goal in enumerate(poses):
        biggest = max(abs(goal[j] - last[j]) for j in (1, 2, 3, 4, 5))
        n = max(1, math.ceil(biggest / step_deg))
        for k in range(1, n + 1):
            why = m3kinematics.problem({j: round(last[j] + (goal[j] - last[j]) * k / n) for j in (1, 2, 3, 4, 5)}, clear)
            if why:
                return why
        if i < len(poses) - 1:
            last = {j: round(last[j] + BLEND * (goal[j] - last[j])) for j in goal}
    return None


def gripper_angle(width_mm: float) -> int:
    """The angle that closes the fingers to `width_mm`, from Yahboom's table (interpolated)."""
    widths = sorted(GRIPPER_WIDTH)
    if not widths[0] <= width_mm <= widths[-1]:
        raise ArmError(f"no gripper angle for {width_mm} mm: the table covers {widths[0]}-{widths[-1]} mm")
    for lo, hi in itertools.pairwise(widths):
        if lo <= width_mm <= hi:
            t = (width_mm - lo) / (hi - lo)
            return round(GRIPPER_WIDTH[lo] + t * (GRIPPER_WIDTH[hi] - GRIPPER_WIDTH[lo]))
    return GRIPPER_WIDTH[widths[-1]]


class M3Arm:
    """Use as a context manager, like roboarm.arm.Arm."""

    def __init__(self, send: Callable[[Pose, int], None], battery: Callable[[], float | None] = lambda: None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 step_deg: float = 5.0):
        self._send_pose, self._battery, self._clock, self._sleep = send, battery, clock, sleep
        self.step_deg = step_deg
        self.interrupt: Callable[[], bool] | None = None   # set by the bridge; see Arm.interrupt
        # what was last sent, and the move under way: (start, goal, t0, seconds)
        self._pose: Pose | None = None
        self._move: tuple[Pose, Pose, float, float] | None = None
        self._lock = threading.RLock()
        self._glide_thread: threading.Thread | None = None
        self._glide_cancel = threading.Event()

    # ------------------------------------------------------------- lifecycle --
    def __enter__(self) -> M3Arm:
        volts = self._battery()
        if volts is not None and volts < MIN_BATTERY_V:
            raise ArmError(f"battery {volts:.2f} V is below {MIN_BATTERY_V} V -- charge it")
        return self

    def __exit__(self, *exc) -> None:
        if self._pose is not None:
            with contextlib.suppress(ArmError):
                self.hold()

    # ----------------------------------------------------------------- state --
    def read(self, attempts: int = 4) -> Pose:   # attempts: Arm.read's signature, unused
        """The pose believed now: the last command, interpolated through a move under way."""
        with self._lock:
            if self._pose is None:
                raise ArmError("the arm's pose is unknown until the first command (it cannot be"
                               " read back): engage() first")
            if self._move is None:
                return dict(self._pose)
            start, goal, t0, seconds = self._move
            frac = 1.0 if seconds <= 0 else min(max((self._clock() - t0) / seconds, 0.0), 1.0)
            return {j: round(start[j] + frac * (goal[j] - start[j])) for j in JOINTS}

    def out_of_range(self, pose: Pose | None = None) -> Pose:
        pose = self.read() if pose is None else pose
        return {j: v for j, v in pose.items() if not SAFE_LIMITS[j][0] <= v <= SAFE_LIMITS[j][1]}

    def battery(self) -> float:
        volts = self._battery()
        if volts is None:
            raise ArmError("no battery reading from the board yet")
        return float(volts)

    # ---------------------------------------------------------------- motion --
    def _check(self, pose: Pose, floor_clear: float | None = None) -> None:
        for joint, angle in pose.items():
            if joint not in SAFE_LIMITS:
                raise ArmError(f"no such joint: {joint}")
            lo, hi = SAFE_LIMITS[joint]
            if not lo <= angle <= hi:
                raise ArmError(f"J{joint}={angle} is outside its safe range {lo}..{hi}")
        why = m3kinematics.problem(pose, m3kinematics.FLOOR_CLEAR if floor_clear is None else floor_clear)
        if why:
            raise ArmError(f"refused: {why} ({' '.join(f'J{j}={pose[j]}' for j in JOINTS)})")

    def _send(self, pose: Pose, ms: int) -> None:
        with self._lock:
            start = self.read() if self._pose is not None else dict(pose)
            self._send_pose(dict(pose), int(ms))
            self._pose = dict(pose)
            self._move = (start, dict(pose), self._clock(), ms / 1000)

    def _plan(self, targets: Pose, floor_clear: float | None = None) -> tuple[Pose, Pose]:
        if floor_clear is not None and floor_clear < m3kinematics.MIN_FLOOR_CLEAR:
            raise ArmError(f"floor_clear {floor_clear} m is under the {m3kinematics.MIN_FLOOR_CLEAR} m minimum")
        start = self.read()
        targets = {int(j): int(v) for j, v in targets.items()}
        goal = {j: targets.get(j, start[j]) for j in JOINTS}
        self._check(goal, floor_clear)
        return start, goal

    def _steps(self, start: Pose, goal: Pose, floor_clear: float | None = None) -> list[Pose]:
        biggest = max(abs(goal[j] - start[j]) for j in JOINTS)
        n = max(1, math.ceil(biggest / self.step_deg)) if biggest else 0
        steps = [{j: round(start[j] + (goal[j] - start[j]) * k / n) for j in JOINTS} for k in range(1, n + 1)]
        for step in steps:   # every pose on the way, before any is sent
            self._check(step, floor_clear)
        return steps

    def _run(self, steps: list[Pose], speed_dps: float, cancel: threading.Event | None = None) -> None:
        if steps:   # all checked already: one command to the last of them
            self._sweep(steps[-1], speed_dps, cancel)

    def _sweep(self, goal: Pose, speed_dps: float, cancel: threading.Event | None = None,
               early: float = 0.0) -> None:
        """Send `goal` to arrive when the joint with the furthest to go gets there at speed_dps,
        and wait for it -- or only (1 - early) of the way, for follow() to send the next."""
        start = self.read()
        seconds = max(0.05, max(abs(goal[j] - start[j]) for j in JOINTS) / speed_dps)
        self._send(goal, int(1000 * seconds))
        self._wait(seconds * (1.0 - early), cancel)

    def _wait(self, seconds: float, cancel: threading.Event | None = None) -> None:
        end = self._clock() + seconds
        while True:
            if self.interrupt is not None and self.interrupt():
                self._freeze()   # not hold(): a glide's own thread cannot wait for itself
                raise ArmError("move interrupted -- the arm is holding where it stopped")
            if cancel is not None and cancel.is_set():
                self._freeze()
                return
            left = end - self._clock()
            if left <= 0:
                return
            if cancel is None:
                self._sleep(min(POLL_S, left))
            else:
                cancel.wait(min(POLL_S, left))

    def move_to(self, targets: Pose, speed_dps: float = 40.0, settle: bool = True,
                repeatable: bool = False, floor_clear: float | None = None) -> Pose:   # repeatable: Arm's
        """Move to `targets` (partial poses allowed) in checked steps of at most step_deg. A move
        planned on a floor the depth has just measured may keep less than FLOOR_CLEAR from it
        (`floor_clear`, m; never under MIN_FLOOR_CLEAR)."""
        self._cancel_glide()
        start, goal = self._plan(targets, floor_clear)
        self._run(self._steps(start, goal, floor_clear), speed_dps)
        if settle:
            self._sleep(0.25)
        return goal

    def follow(self, poses: list[Pose], speed_dps: float = 30.0, floor_clear: float | None = None,
               settle: bool = False) -> Pose:
        """Through `poses` (partial poses allowed) one after another without stopping at each: the
        next goes out when the arm is BLEND of the way to the one before, so it rounds the corner
        -- and that rounded way is checked too, with every stretch, before anything moves. For a
        straight line of the hand: IK poses along it, a few cm apart."""
        self._cancel_glide()
        if floor_clear is not None and floor_clear < m3kinematics.MIN_FLOOR_CLEAR:
            raise ArmError(f"floor_clear {floor_clear} m is under the {m3kinematics.MIN_FLOOR_CLEAR} m minimum")
        here = self.read()
        goals, last = [], here
        for pose in poses:
            goal = {j: int({int(k): v for k, v in pose.items()}.get(j, last[j])) for j in JOINTS}
            self._check(goal, floor_clear)
            goals.append(goal)
            last = goal
        last = here
        for i, goal in enumerate(goals):   # the stretches as they will be driven, corners cut
            self._steps(last, goal, floor_clear)
            if i < len(goals) - 1:
                last = {j: round(last[j] + BLEND * (goal[j] - last[j])) for j in JOINTS}
        for i, goal in enumerate(goals):
            self._sweep(goal, speed_dps, early=0.0 if i == len(goals) - 1 else 1.0 - BLEND)
        if settle:
            self._sleep(0.25)
        return goals[-1] if goals else here

    def glide(self, targets: Pose, speed_dps: float = 12.0) -> Pose:
        """Start the move and return at once with the goal; moving() says whether it goes on."""
        self._cancel_glide()
        start, goal = self._plan(targets)
        steps = self._steps(start, goal)
        cancel = threading.Event()
        self._glide_cancel = cancel

        def run() -> None:
            with contextlib.suppress(ArmError):
                self._run(steps, speed_dps, cancel)

        self._glide_thread = threading.Thread(target=run, name="m3-arm-glide", daemon=True)
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

    def home(self, **kw) -> Pose:
        return self.move_to(dict(zip(JOINTS, POSES["home"])), **kw)

    def set_gripper(self, angle: int, **kw) -> Pose:
        return self.move_to({GRIPPER: int(angle)}, **kw)

    def open_gripper(self, **kw) -> Pose:
        return self.set_gripper(GRIPPER_OPEN, **kw)

    def close_gripper(self, width_mm: float | None = None, **kw) -> Pose:
        """Close to the object's width -- never further: the servo stalls and can burn."""
        if width_mm is None:
            raise ArmError("the M3's gripper closes only to a width: close_gripper(width_mm=30)")
        return self.set_gripper(gripper_angle(width_mm), **kw)

    def grasped(self) -> bool:
        raise ArmError("the M3 cannot read its gripper back: check the grasp with the camera")

    # ---------------------------------------------------------------- torque --
    def hold(self) -> Pose:
        """Stop where the arm is believed to be: the e-stop."""
        self._cancel_glide()
        return self._freeze()

    def _freeze(self) -> Pose:
        with self._lock:
            pose = self.read()
            self._send_pose(dict(pose), 0)
            self._pose, self._move = dict(pose), None
        return pose

    def release(self) -> None:
        raise ArmError("the M3's board has no torque switch: the arm cannot be let go limp")

    def engage(self, pose: Pose | None = None) -> Pose:
        """The first command after power-up: the limp arm, wherever it is, goes to `pose`
        (default: home) over ENGAGE_MS. From then on its pose is known."""
        self._cancel_glide()
        goal = {int(j): int(v) for j, v in (pose or dict(zip(JOINTS, POSES["home"]))).items()}
        if set(goal) != set(JOINTS):
            raise ArmError("engage needs all six joints: nothing is known about the others")
        self._check(goal)
        with self._lock:
            self._send_pose(dict(goal), ENGAGE_MS)
            self._pose, self._move = dict(goal), (dict(goal), dict(goal), self._clock(), 0.0)
        self._sleep(ENGAGE_MS / 1000 + 0.3)
        return goal
