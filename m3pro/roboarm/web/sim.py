"""A pretend arm and a pretend wrist camera, so the web app runs without the robot.

Not a physics simulation -- just enough of the real modules' contract that every
code path in session.py gets exercised: the arm glides through interpolated steps
and can be interrupted, the gripper stops on an object, grasped() answers about it,
and the camera renders what the REAL calibrated homography says it would see from
the current pose. So a sweep in the simulator finds the coloured blocks where they
were put, a pick lifts one, and a place puts it down somewhere else, all through
the same detect/grasp/sweep code the robot runs.

The homography and survey pose are the ones fitted on the robot on 2026-09-08
(the same numbers tests/test_sweep.py asserts coverage against).
"""

from __future__ import annotations

import math
import threading
import time

import cv2
import numpy as np

from roboarm import camera, sweep
from roboarm import config as cfg
from roboarm import kinematics as kin
from roboarm.arm import ArmError
from roboarm.config import Pose

# data/table_homography.json, fitted 2026-09-08, worst residual 0.48 mm over 8 points.
REAL_H = np.array([
    [1.5806220846933396e-05, -2.412854587985502e-04, 2.1607999921844648e-01],
    [-2.2744926875526548e-04, -3.5041628047172227e-07, 9.179117812253038e-02],
    [5.9373685046794684e-05, -8.77513698668719e-05, 1.0],
])
REAL_SURVEY = {1: 90, 2: 56, 3: 23, 4: 10, 5: 89, 6: 30}
# A background glide moves one degree per step and never faster than this per
# step, whatever time_scale says: a scan reads the pose before and after each
# frame and takes the mean, so the arm must not jump far between the two (the
# robot yaws 12 deg/s and reads 4 times a second: 3 deg between reads).
SIM_GLIDE_STEP_DEG = 1.0
SIM_GLIDE_STEP_S = 0.01


def calibration() -> list[tuple[np.ndarray, Pose, str]]:
    """What workspace.load_all() would return on the robot."""
    return [(REAL_H.copy(), dict(REAL_SURVEY), "primary")]


class Block:
    """Something on the table: a coloured square, `size` metres across."""

    def __init__(self, x: float, y: float, size: float, colour: tuple[int, int, int],
                 length: float | None = None, height: float | None = None):
        self.x = x
        self.y = y
        self.size = size
        self.length = size if length is None else length
        self.height = size if height is None else height   # a cube unless told otherwise
        self.colour = colour  # BGR
        self.held = False


class World:
    """The table and what is on it. Shared by the arm and the camera."""

    def __init__(self, blocks: list[Block] | None = None):
        self.blocks = blocks if blocks is not None else default_blocks()
        self.lock = threading.Lock()


def default_blocks() -> list[Block]:
    # Kept apart: two blobs that touch in the picture merge into one silhouette,
    # and a block's outline includes its side faces, which reach towards the
    # point under the lens (about 50 mm to the right of straight ahead).
    return [
        Block(0.150, 0.010, 0.040, (0, 120, 255)),          # orange, straight ahead
        Block(0.170, -0.055, 0.030, (255, 100, 0)),         # blue, off to the right
        Block(0.140, 0.095, 0.060, (0, 0, 220)),            # red, too wide to grasp
    ]


class SimArm:
    """Enough of roboarm.arm.Arm for session.py, moving a World instead of servos.

    `time_scale` stretches or squashes how long a glide takes: 1.0 feels like the
    robot, 0 makes every move instant for tests.
    """

    def __init__(self, world: World | None = None, pose: Pose | None = None,
                 time_scale: float = 1.0, step_deg: float = cfg.MAX_STEP_DEG):
        self.world = world or World()
        self.pose: Pose = dict(pose or cfg.HOME_POSE)
        self.time_scale = time_scale
        self.step_deg = step_deg
        self.interrupt = None
        self.torque = True
        self.voltage = 12.3
        # The real arm lands its tips this much nearer the base than the model
        # says (cfg.REACH_OFFSET_M). The simulator reproduces it, so the offset the
        # grasp code applies is exercised rather than merely tolerated.
        self.reach_short_m = cfg.REACH_OFFSET_M
        self._gripper_target: int | None = None
        self._connected = False
        self.port = "sim"
        self._glide_thread: threading.Thread | None = None
        self._glide_cancel = threading.Event()

    # Session calls these the way it would on a real Arm.
    def __enter__(self) -> SimArm:
        self._connected = True
        return self

    def __exit__(self, *exc) -> None:
        self._connected = False

    @property
    def bot(self) -> SimArm:
        return self

    def get_battery_voltage(self) -> float:
        return self.voltage

    def battery(self) -> float:
        return self.voltage

    # ----------------------------------------------------------------- state --
    def read(self, attempts: int = 4) -> Pose:
        if not self._connected:
            raise ArmError("arm is not connected -- use `with Arm() as arm:`")
        return dict(self.pose)

    def out_of_range(self, pose: Pose | None = None) -> Pose:
        pose = self.read() if pose is None else pose
        return {j: v for j, v in pose.items()
                if not (cfg.HARD_LIMITS[j][0] <= v <= cfg.HARD_LIMITS[j][1])}

    # ---------------------------------------------------------------- motion --
    def move_to(self, targets: Pose, speed_dps: float = 40.0, settle: bool = True,
                repeatable: bool = False) -> Pose:
        # The simulated servos always reach their command, so there is nothing for
        # `repeatable` to make repeatable.
        self._cancel_glide()
        start, goal = self._plan(targets)
        self._glide(start, goal, speed_dps)
        return self.read()

    def glide(self, targets: Pose, speed_dps: float = 12.0) -> Pose:
        """As Arm.glide(): move in the background, read() answers meanwhile.
        With time_scale 0 the steps still take SIM_GLIDE_STEP_S each, so a scan
        that reads the pose as it goes sees the arm pass every station."""
        self._cancel_glide()
        start, goal = self._plan(targets)
        cancel = threading.Event()
        self._glide_cancel = cancel

        def run() -> None:
            try:
                self._glide(start, goal, speed_dps, cancel, min_step_s=SIM_GLIDE_STEP_S,
                            step_deg=SIM_GLIDE_STEP_DEG)
            except ArmError:
                pass

        self._glide_thread = threading.Thread(target=run, name="sim-glide", daemon=True)
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

    def _plan(self, targets: Pose) -> tuple[Pose, Pose]:
        start = self.read()
        for joint, angle in targets.items():
            if joint not in cfg.JOINT_IDS:
                raise ArmError(f"no such joint: {joint}")
            lo, hi = cfg.SAFE_LIMITS[joint]
            if not (lo <= angle <= hi):
                raise ArmError(f"J{joint}={angle} is outside its safe range {lo}..{hi}")
        goal = dict(start)
        for joint in cfg.JOINT_IDS:
            lo, hi = cfg.SAFE_LIMITS[joint]
            goal[joint] = int(min(max(targets.get(joint, start[joint]), lo), hi))
        clearance = cfg.mast_clearance(goal)
        if clearance < cfg.MIN_MAST_CLEARANCE_M:
            raise ArmError(
                f"pose would come within {clearance * 1000:.0f} mm of the camera mast "
                f"(minimum {cfg.MIN_MAST_CLEARANCE_M * 1000:.0f} mm)"
            )
        return start, goal

    def _glide(self, start: Pose, goal: Pose, speed_dps: float,
               cancel: threading.Event | None = None, min_step_s: float = 0.0,
               step_deg: float | None = None) -> None:
        biggest = max(abs(goal[j] - start[j]) for j in cfg.JOINT_IDS)
        if biggest == 0:
            return
        steps = max(1, math.ceil(biggest / (step_deg or self.step_deg)))
        run_time = max(20, int(1000 * (biggest / steps) / speed_dps))
        for i in range(1, steps + 1):
            if self.interrupt is not None and self.interrupt():
                raise ArmError("move interrupted -- the arm is holding where it stopped")
            frac = i / steps
            self._settle({j: round(start[j] + frac * (goal[j] - start[j]))
                          for j in cfg.JOINT_IDS})
            delay = max(run_time / 1000 * self.time_scale, min_step_s)
            if cancel is not None:
                if cancel.wait(delay):
                    return
            elif delay > 0:
                time.sleep(delay)

    def _settle(self, pose: Pose) -> None:
        """Adopt `pose`, with the gripper stopping on whatever it is holding, and
        anything held travelling with the fingertips."""
        with self.world.lock:
            held = next((b for b in self.world.blocks if b.held), None)
            if held is not None:
                # Fingers already shut on it: they cannot close further, and a
                # narrower command just leans on the block, exactly as on the robot.
                stop_at = _angle_for_gap(held.size)
                pose[cfg.GRIPPER_ID] = min(pose[cfg.GRIPPER_ID], stop_at)
            self.pose = dict(pose)
            if held is not None:
                held.x, held.y = self._real_tip()[:2]

    def home(self, **kw) -> Pose:
        return self.move_to(dict(cfg.HOME_POSE), **kw)

    def set_gripper(self, angle: int, **kw) -> Pose:
        self._gripper_target = angle
        pose = self.move_to({cfg.GRIPPER_ID: angle}, **kw)
        self._grab_or_drop()
        return pose

    def open_gripper(self, **kw) -> Pose:
        return self.set_gripper(cfg.GRIPPER_OPEN, **kw)

    def close_gripper(self, **kw) -> Pose:
        return self.set_gripper(cfg.GRIPPER_CLOSED, **kw)

    def _real_tip(self) -> tuple[float, float, float]:
        """Where the fingertips really are: the model's answer, pulled back along
        the reach by the measured shortfall."""
        x, y, z = kin.forward(self.pose)
        radius = math.hypot(x, y)
        if radius > 1e-9 and self.reach_short_m:
            x -= self.reach_short_m * x / radius
            y -= self.reach_short_m * y / radius
        return x, y, z

    def _grab_or_drop(self) -> None:
        """Closing over a block on the table picks it up; opening lets it go."""
        with self.world.lock:
            x, y, z = self._real_tip()
            above_table = z + cfg.TABLE_BELOW_PLATE
            gap = cfg.gripper_gap(self.pose[cfg.GRIPPER_ID])
            for block in self.world.blocks:
                if block.held:
                    if gap > block.size + 0.004:
                        block.held = False
                    continue
                near = math.hypot(block.x - x, block.y - y) < 0.012
                low = above_table < 0.030
                if (near and low and self._gripper_target == cfg.GRIPPER_CLOSED
                        and cfg.gripper_gap(cfg.GRIPPER_OPEN) > block.size > 0.013):
                    block.held = True
                    self.pose[cfg.GRIPPER_ID] = _angle_for_gap(block.size)

    def grasped(self) -> bool:
        if self._gripper_target != cfg.GRIPPER_CLOSED:
            return False
        return self.read()[cfg.GRIPPER_ID] < cfg.GRIPPER_CLOSED - cfg.READBACK_TOLERANCE_DEG

    # ---------------------------------------------------------------- torque --
    def hold(self) -> Pose:
        self._cancel_glide()
        self.torque = True
        return self.read()

    def release(self) -> None:
        self.torque = False

    def engage(self) -> Pose:
        self.torque = True
        return self.read()


def _angle_for_gap(gap_m: float) -> int:
    """The servo angle at which the fingers are `gap_m` apart. Inverse of
    cfg.gripper_gap, found by scanning -- the table is small and monotonic."""
    for angle in range(cfg.GRIPPER_OPEN, cfg.GRIPPER_CLOSED + 1):
        if cfg.gripper_gap(angle) <= gap_m:
            return angle
    return cfg.GRIPPER_CLOSED


class SimStream(camera.Stream):
    """A camera.Stream fed by rendering the World from the arm's current pose.

    At a survey station the picture is what the real homography says the lens
    would see -- blocks land on the pixels detect.py maps back to their table
    positions. Anywhere else the mapping is undefined, so the view is dimmed and
    says so; that is also what the page shows on the robot when the arm is off
    station, because table coordinates mean nothing there.
    """

    def __init__(self, arm: SimArm, fps: float = 8.0,
                 calibrated: list[tuple[np.ndarray, Pose, str]] | None = None):
        super().__init__(device=-1)
        self.arm = arm
        self.frame_period = 1.0 / fps
        self.calibrated = calibrated or calibration()
        self._rng = np.random.default_rng(0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.publish(self.render())
            time.sleep(self.frame_period)

    def settled(self, settle_s: float = 1.2, count: int = 1) -> list:
        """A frame asked for with no settle time is rendered right now, from the
        pose the arm is at this instant. The robot's camera hands over a frame
        at most 60 ms old (16 fps), under a degree of yaw during a scan; the
        simulated arm sweeps a station in that time, which would make its
        frames far staler than the real ones."""
        if settle_s > 0:
            return super().settled(settle_s, count)
        frame = self.render()
        self.publish(frame)
        return [frame] * count

    def render(self):
        width, height = cfg.WRIST_CAM_SIZE
        frame = np.full((height, width, 3), 168, np.uint8)
        # Soft vignette so the picture is not perfectly flat.
        yy, xx = np.mgrid[0:height, 0:width]
        shade = 1.0 - 0.25 * (((xx - width / 2) / width) ** 2 + ((yy - height / 2) / height) ** 2)
        frame = (frame * shade[..., None]).astype(np.uint8)

        # Straight off the arm, not read(): the camera runs whether or not anyone
        # has opened the arm, exactly as the real one does.
        pose = dict(self.arm.pose)
        look = sweep.at_look(self.calibrated, pose)
        if look is None:
            frame = (frame * 0.55).astype(np.uint8)
            cv2.putText(frame, "sim: arm is off the survey station", (40, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)
            matrix = sweep.rotate(self.calibrated[0][0], self.calibrated[0][1][1] - pose[1])
        else:
            matrix = look.matrix
        inverse = np.linalg.inv(matrix)

        with self.arm.world.lock:
            for block in self.arm.world.blocks:
                if block.held:
                    continue
                hx, hy = block.size / 2, block.length / 2
                corners = np.array([
                    [block.x - hx, block.y - hy], [block.x + hx, block.y - hy],
                    [block.x + hx, block.y + hy], [block.x - hx, block.y + hy],
                ])
                # A solid block: its top stands above the table plane and images
                # pushed outward from the nadir, and the side faces towards the
                # lens fill the gap between base and top. So the outline is the
                # hull of the base and the magnified top -- exactly the parallax
                # detect.range_block() undoes.
                grown = sweep.magnification(pose, block.height)
                nadir = np.array(kin.camera_nadir(pose))
                top = nadir + (corners - nadir) * grown
                pixels = cv2.perspectiveTransform(
                    np.vstack([corners, top]).reshape(-1, 1, 2).astype(np.float64), inverse
                ).reshape(-1, 2)
                if not np.all(np.isfinite(pixels)):
                    continue
                hull = cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2)
                cv2.fillPoly(frame, [hull.astype(np.int32)], block.colour)

        noise = self._rng.integers(-6, 7, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return frame

