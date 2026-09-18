"""Everything the browser can see or do, in one object.

The page is a thin client: it polls `snapshot()` a couple of times a second, shows
the MJPEG stream, and POSTs intentions. All the judgement is here, and three rules
keep the arm safe with several browser tabs and threads all talking at once:

  1. ONE LOCK ON THE ARM. The SDK is a serial conversation and cannot interleave
     two callers, so every servo command goes through `_with_arm()`. A manual
     request that finds the lock taken gets "busy" back instead of waiting behind a
     move it cannot see -- otherwise a burst of jog clicks queues up and the arm
     keeps moving after the operator has let go.

  2. ONE JOB AT A TIME, AND ONLY IN AUTO MODE. Sweep, pick, place and background
     run on a worker thread that holds the arm for the duration. Manual moves are
     refused while it runs; the mode switch is refused too. Stop is the one thing
     always allowed: it trips `Arm.interrupt`, which abandons the glide in progress
     at its last legal step, and then holds.

  3. THE CAMERA IS OWNED, NOT BORROWED. camera.Stream keeps the device open for the
     live view, so the automatic pipeline takes frames from the stream rather than
     re-opening the camera (which V4L2 would refuse). `_look_once()` is
     tools/pick.py's look_once() with that one substitution.

Nothing here is hardware-specific: the arm and the camera are handed in, so
sim.py can stand in for both and the whole thing runs on a laptop.
"""

from __future__ import annotations

import collections
import contextlib
import io
import math
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from roboarm import camera, detect, grasp, sweep
from roboarm import config as cfg
from roboarm import kinematics as kin
from roboarm.arm import ArmError

Pose = dict[int, int]

MODES = ("manual", "auto")
VIEWS = ("raw", "detect", "mask", "edges")

# Where a picked object is put down, as in tools/pick.py.
DROP_X = 0.150
DROP_Y = -0.070

# How far from the sweep's estimate the refine look may find the object and still
# be believed to be the same one (tools/pick.py).
REFIND_M = 0.030

# Detection on the live view runs at most this often. The wrist camera is ~7 fps
# and the detectors take tens of milliseconds, so this is plenty and leaves the
# CPU for the stream.
DETECT_PERIOD_S = 0.35

# How long a manual request will wait for the arm lock before giving up. Long
# enough to sit out the background pose read, which over WiFi to the bridge can
# take a few hundred milliseconds; a job holds the lock for far longer and gets
# "busy" back, which is right.
MANUAL_WAIT_S = 1.0

# Seconds to let the wrist camera's exposure settle after the arm stops, before
# a frame is trusted for detection. Only needed when the SCENE changed -- the
# arm came from the home pose, or from a place, and the auto-exposure has a
# different picture to adapt to. MEASURED 2026-09-11 on the robot: after a
# base yaw, which keeps the same view of the same table, a tag read from the
# very next frame lands within 1.7 mm of a frame taken 1.2 s later -- the
# same 1.7 mm -- so a station-to-station look waits only for the frame after
# the move (move_to itself already pauses a quarter second).
SETTLE_S = 1.2
YAW_SETTLE_S = 0.15

# Base yaw between stations, and any other move that is a transit rather than
# a grasp. Measured the same day: 40 deg/s lands the camera 0.8 mm further
# from where 20 deg/s does; 30 is the compromise.
STATION_DPS = 30.0

# A refine look is skipped when the base is already within this many degrees
# of the centred bearing: at 160 mm a degree is 2.8 mm or 12 px, so 8 deg
# leaves the object under 100 px from the middle of a 640 px frame, where
# the mapping is as good as it is at the centre. Measured 2026-09-11: a
# refine from 10 deg off moved the estimate 0.7 mm and cost two seconds.
REFINE_IF_OFF_DEG = 8


class Busy(RuntimeError):
    """The arm is doing something else; try again in a moment."""


class Refused(RuntimeError):
    """The request is not allowed in the current mode."""


class PoseTracking:
    """The arm, with every call that yields a pose also reporting it.

    grasp.pick() and friends move the arm through their own sequence of calls,
    and nothing else may read the servos while they do -- the SDK cannot
    interleave a read with a glide. So the page's picture of the pose would freeze
    for the whole job. Instead, every method that already returns a pose passes it
    to `on_pose` on the way out, and the page follows along at no extra cost.
    """

    _REPORTING = frozenset({"read", "move_to", "home", "set_gripper", "open_gripper",
                            "close_gripper", "hold", "engage"})

    def __init__(self, arm, on_pose: Callable[[Pose], None]):
        object.__setattr__(self, "_arm", arm)
        object.__setattr__(self, "_on_pose", on_pose)

    def __getattr__(self, name: str):
        attr = getattr(self._arm, name)
        if name not in self._REPORTING:
            return attr

        def reporting(*args, **kwargs):
            pose = attr(*args, **kwargs)
            if isinstance(pose, dict):
                self._on_pose(pose)
            return pose
        return reporting

    def __setattr__(self, name: str, value) -> None:
        setattr(self._arm, name, value)


class Job:
    """One automatic task running on the worker thread."""

    def __init__(self, name: str):
        self.name = name
        self.status = "running"      # running | done | failed | stopped
        self.message = ""
        self.started = time.time()
        self.finished: float | None = None

    def snapshot(self) -> dict:
        end = self.finished or time.time()
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "elapsed_s": round(end - self.started, 1),
        }


class Session:
    def __init__(
        self,
        *,
        arm_factory: Callable[[], object],
        streams: dict[str, Callable[[], camera.Stream]],
        calibration: Callable[[], list[tuple[np.ndarray, Pose, str]]],
        out_dir: Path | None = None,
        detector_url: str = cfg.DETECTOR_URL,
    ):
        self._arm_factory = arm_factory
        self._stream_factories = streams
        self._calibration_loader = calibration
        self.out_dir = out_dir
        self.detector_url = detector_url

        self.mode = "manual"
        self.view = "detect"
        self.detector = "auto"
        self.prompt: str | None = None
        self.object_mm: float | None = None
        self.step_deg = float(sweep.SURVEY_STEP_DEG)
        self.speed_dps = 30.0
        self.settle_s = SETTLE_S
        self.reach_offset_mm = cfg.REACH_OFFSET_M * 1000

        self.arm = None
        self.arm_error: str | None = None
        self._lock = threading.RLock()
        self._pose: Pose | None = None
        self._pose_when = 0.0
        self._battery: float | None = None
        self._torque = True
        self._holding = False

        self.camera_name = next(iter(streams))
        self.stream: camera.Stream | None = None

        self._stop = threading.Event()
        self.job: Job | None = None
        self._job_thread: threading.Thread | None = None

        self._log: collections.deque = collections.deque(maxlen=500)
        self._log_seq = 0
        self._log_lock = threading.Lock()

        self.calibrated: list[tuple[np.ndarray, Pose, str]] = []
        self.calibration_error: str | None = None
        self.vision_ok = False

        self.detection: dict = {"targets": [], "at_station": False, "error": None,
                                "when": 0.0}
        self._annotated = None       # newest frame with detections drawn on
        self._mask = None            # newest detector mask / filter output
        self._frame_seq = 0
        self.sweep_targets: list[detect.Target] = []
        self.sweep_when = 0.0
        self._coverage: dict[tuple, float] = {}
        self._coverage_lock = threading.Lock()
        self._lap_started = 0.0

        self._threads: list[threading.Thread] = []
        self._closing = threading.Event()

    # ------------------------------------------------------------- lifecycle --
    def start(self) -> Session:
        self.reload_calibration()
        self.set_camera(self.camera_name)
        self.connect()
        for name, target in (("detect", self._detect_loop), ("poll", self._poll_loop)):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        return self

    def close(self) -> None:
        self._closing.set()
        self.stop()
        if self.stream is not None:
            self.stream.stop()
        self.disconnect()

    # ------------------------------------------------------------------ log --
    def log(self, text: str) -> None:
        with self._log_lock:
            self._log_seq += 1
            self._log.append((self._log_seq, time.time(), text.rstrip()))

    def log_since(self, seq: int) -> list[dict]:
        with self._log_lock:
            return [{"seq": s, "t": t, "text": x} for s, t, x in self._log if s > seq]

    # ------------------------------------------------------------- hardware --
    def reload_calibration(self) -> None:
        try:
            self.calibrated = self._calibration_loader()
            self.calibration_error = None
        except (FileNotFoundError, ValueError, KeyError) as exc:
            self.calibrated = []
            self.calibration_error = str(exc)
            self.log(f"calibration: {exc}")

    def connect(self) -> None:
        with self._lock:
            if self.arm is not None:
                return
            try:
                arm = self._arm_factory()
                arm.__enter__()
            except Exception as exc:  # noqa: BLE001 -- shown to the operator, whatever it is
                self.arm_error = str(exc)
                self.log(f"arm: {exc}")
                return
            arm.interrupt = self._stop.is_set
            self.arm = PoseTracking(arm, self._note_pose)
            self.arm_error = None
            self._torque = True
            self.log(f"arm connected on {getattr(arm, 'port', '?')}")
            self._refresh(read_battery=True)

    def disconnect(self) -> None:
        with self._lock:
            if self.arm is None:
                return
            with contextlib.suppress(Exception):
                self.arm.__exit__(None, None, None)
            self.arm = None
            self.log("arm disconnected")

    def set_camera(self, name: str) -> None:
        if name not in self._stream_factories:
            raise ValueError(f"no camera called {name!r}")
        if self.stream is not None and name == self.camera_name:
            return
        if self.stream is not None:
            self.stream.stop()
        self.camera_name = name
        self.stream = self._stream_factories[name]().start()
        self.log(f"camera: {name}")

    # ----------------------------------------------------------------- arm ---
    def _with_arm(self, fn, wait: float = MANUAL_WAIT_S):
        if not self._lock.acquire(timeout=wait):
            raise Busy("the arm is busy -- wait for the current move to finish")
        try:
            if self.arm is None:
                raise ArmError(self.arm_error or "arm is not connected")
            return fn(self.arm)
        finally:
            self._lock.release()

    def _note_pose(self, pose: Pose) -> None:
        self._pose = dict(pose)
        self._pose_when = time.time()

    def _refresh(self, read_battery: bool = False) -> None:
        """Re-read the pose (and battery) from the arm. Caller holds the lock."""
        if self.arm is None:
            return
        try:
            self.arm.read()  # PoseTracking notes the answer
            if read_battery:
                self._battery = float(self.arm.bot.get_battery_voltage())
        except ArmError as exc:
            self.arm_error = str(exc)
            self.log(f"arm: {exc}")

    def _manual(self, fn, what: str):
        """Run one manual arm action: refused in auto mode, busy if a job holds the arm.

        Every action is logged, so the log answers "what moved the arm just now?"
        -- with several browser tabs and a keyboard, that question does come up.
        """
        if self.mode != "manual":
            raise Refused("switch to manual mode first")
        if self._job_running():
            raise Busy("an automatic job is running -- stop it first")

        def run(arm):
            self._stop.clear()
            self.log(f"manual: {what}")
            try:
                return fn(arm)
            finally:
                self._refresh()
        return self._with_arm(run)

    def move_joints(self, targets: dict, speed_dps: float | None = None) -> Pose:
        goal = {int(j): round(float(a)) for j, a in targets.items()}
        speed = float(speed_dps or self.speed_dps)
        what = "move " + " ".join(f"J{j}={a}" for j, a in sorted(goal.items()))
        return self._manual(lambda arm: arm.move_to(goal, speed_dps=speed, verify=False), what)

    def jog(self, joint: int, delta: float, speed_dps: float | None = None) -> Pose:
        pose = self.pose()
        if pose is None:
            raise ArmError("no pose from the arm yet")
        lo, hi = cfg.SAFE_LIMITS[joint]
        goal = int(min(max(pose[joint] + delta, lo), hi))
        return self.move_joints({joint: goal}, speed_dps)

    def gripper(self, angle: int | None = None, action: str | None = None) -> Pose:
        if action == "open":
            return self._manual(lambda arm: arm.open_gripper(speed_dps=60, verify=False),
                                "open gripper")
        if action == "close":
            return self._manual(lambda arm: arm.close_gripper(speed_dps=40), "close gripper")
        if angle is None:
            raise ValueError("give an angle or an action")
        return self._manual(
            lambda arm: arm.set_gripper(int(angle), speed_dps=60, verify=False),
            f"gripper to {int(angle)}")

    def cartesian(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                  x: float | None = None, y: float | None = None,
                  z: float | None = None) -> Pose:
        """Move the fingertip by (dx, dy, dz) mm, or to (x, y, z) mm absolute.

        z is height ABOVE THE TABLE, as the page shows it -- not the kinematics
        frame's height above the base plate, which sits 190 mm higher.

        Keeps the tool pitch the arm already has where it can, so a jog is a
        translation rather than a swing; if that pitch cannot reach, tries the
        neighbouring ones, then whatever solve() finds.
        """
        pose = self.pose()
        if pose is None:
            raise ArmError("no pose from the arm yet")
        here = kin.forward(pose)
        goal = [
            here[0] + dx / 1000 if x is None else x / 1000,
            here[1] + dy / 1000 if y is None else y / 1000,
            here[2] + dz / 1000 if z is None else z / 1000 - cfg.TABLE_BELOW_PLATE,
        ]
        gripper = pose[cfg.GRIPPER_ID]
        pitch = kin.tool_pitch(pose)
        candidates = [pitch] + [pitch + d for d in (5, -5, 10, -10, 15, -15)]
        target: Pose | None = None
        for candidate in candidates:
            try:
                target = kin.inverse(*goal, pitch_deg=candidate, gripper=gripper)
                break
            except kin.Unreachable:
                continue
        if target is None:
            target, _pitch = kin.solve(*goal, gripper=gripper)  # raises Unreachable
        arm_joints = {j: a for j, a in target.items() if j != cfg.GRIPPER_ID}
        # inverse() always answers J5=90; a translation must not also un-roll the wrist.
        arm_joints[5] = pose[5]
        return self.move_joints(arm_joints)

    def preset(self, name: str) -> Pose:
        if name == "home":
            return self._manual(lambda arm: arm.home(speed_dps=self.speed_dps, verify=False),
                                "home")
        if name == "survey":
            look = self._primary_look()
            return self._manual(
                lambda arm: arm.move_to(look, speed_dps=min(self.speed_dps, 20.0),
                                        verify=False), "survey pose")
        raise ValueError(f"no preset called {name!r}")

    def hold(self) -> Pose | None:
        """The stop button. Interrupts whatever is moving, then freezes the arm.

        Allowed in every mode, from any thread. Sets the interrupt flag first so a
        glide in progress abandons its next step, then takes the lock -- which the
        interrupted caller releases as soon as its ArmError propagates.
        """
        self._stop.set()
        try:
            if not self._lock.acquire(timeout=6.0):
                raise Busy("could not get the arm to hold -- it did not stop")
            try:
                if self.arm is None:
                    return None
                pose = self.arm.hold()
                self._torque = True
                return pose
            finally:
                self._lock.release()
        finally:
            self._stop.clear()

    def stop(self) -> None:
        """Stop the running job, if any, and hold. Idempotent."""
        thread = self._job_thread
        if thread is not None and thread.is_alive():
            self._stop.set()
            thread.join(timeout=8.0)
        self.hold()

    def release(self) -> None:
        if self._job_running():
            raise Busy("an automatic job is running -- stop it first")

        def run(arm):
            arm.release()
            self._torque = False
            self.log("torque OFF -- the arm will sag, support it")
        self._with_arm(run)

    def engage(self) -> Pose:
        if self._job_running():
            raise Busy("an automatic job is running -- stop it first")

        def run(arm):
            pose = arm.engage()
            self._torque = True
            self.log("torque on")
            return pose
        return self._with_arm(run, wait=1.0)

    # ---------------------------------------------------------------- state --
    def pose(self) -> Pose | None:
        return dict(self._pose) if self._pose else None

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode != self.mode and self._job_running():
            raise Busy("stop the running job before changing mode")
        self.mode = mode
        self.log(f"mode: {mode}")

    def set_view(self, view: str | None = None, detector: str | None = None,
                 prompt: str | None = None, object_mm: float | None = None,
                 camera_name: str | None = None, step_deg: float | None = None,
                 speed_dps: float | None = None,
                 reach_offset_mm: float | None = None) -> None:
        if view is not None:
            if view not in VIEWS:
                raise ValueError(f"view must be one of {VIEWS}")
            self.view = view
        if detector is not None:
            if detector not in detect.MODES:
                raise ValueError(f"detector must be one of {detect.MODES}")
            self.detector = detector
        if prompt is not None:
            self.prompt = prompt.strip() or None
        if object_mm is not None:
            self.object_mm = float(object_mm) if float(object_mm) > 0 else None
        if camera_name is not None:
            self.set_camera(camera_name)
        if step_deg is not None:
            if not (5 <= float(step_deg) <= 90):
                raise ValueError("sweep step must be 5..90 degrees")
            self.step_deg = float(step_deg)
        if speed_dps is not None:
            if not (5 <= float(speed_dps) <= 60):
                raise ValueError("speed must be 5..60 deg/s")
            self.speed_dps = float(speed_dps)
        if reach_offset_mm is not None:
            if not (-30 <= float(reach_offset_mm) <= 40):
                raise ValueError("reach offset must be -30..40 mm")
            self.reach_offset_mm = float(reach_offset_mm)

    def snapshot(self) -> dict:
        pose = self.pose()
        tip = pitch = fingers = None
        if pose is not None:
            fx, fy, fz = kin.forward(pose)
            tip = {"x": round(fx * 1000, 1), "y": round(fy * 1000, 1),
                   "z": round((fz + cfg.TABLE_BELOW_PLATE) * 1000, 1)}
            pitch = round(kin.tool_pitch(pose), 1)
            fingers = round(kin.finger_heading(pose))
        stream = self.stream
        with self._log_lock:
            log_seq = self._log_seq
        return {
            "mode": self.mode,
            "view": self.view,
            "detector": self.detector,
            "detectors": list(detect.MODES),
            "prompt": self.prompt or "",
            "object_mm": self.object_mm,
            "step_deg": self.step_deg,
            "speed_dps": self.speed_dps,
            "reach_offset_mm": self.reach_offset_mm,
            "camera": self.camera_name,
            "cameras": list(self._stream_factories),
            "camera_fps": round(stream.fps, 1) if stream else 0.0,
            "camera_error": stream.error if stream else "no camera",
            "arm": {
                "connected": self.arm is not None,
                "error": self.arm_error,
                "pose": {str(j): a for j, a in pose.items()} if pose else None,
                "pose_age_s": round(time.time() - self._pose_when, 1) if pose else None,
                "battery": self._battery,
                "tip_mm": tip,
                "pitch": pitch,
                "fingers_deg": fingers,
                "torque": self._torque,
                "busy": self._lock_held(),
                "limits": {str(j): list(cfg.SAFE_LIMITS[j]) for j in cfg.JOINT_IDS},
                "gripper": {"open": cfg.GRIPPER_OPEN, "closed": cfg.GRIPPER_CLOSED},
            },
            "job": self.job.snapshot() if self.job else None,
            "detection": {
                "targets": self.detection["targets"],
                "at_station": self.detection["at_station"],
                "error": self.detection["error"],
                "age_s": round(time.time() - self.detection["when"], 1)
                if self.detection["when"] else None,
            },
            "sweep": {
                "targets": [self._target_dict(t) for t in self.sweep_targets],
                "age_s": round(time.time() - self.sweep_when, 1) if self.sweep_when else None,
            },
            "calibration": {
                "ok": bool(self.calibrated),
                "looks": [name for _m, _p, name in self.calibrated],
                "error": self.calibration_error,
            },
            "vision": {"available": self.vision_ok, "url": self.detector_url},
            "log_seq": log_seq,
        }

    def _lock_held(self) -> bool:
        if self._lock.acquire(blocking=False):
            self._lock.release()
            return False
        return True

    @staticmethod
    def _target_dict(target: detect.Target) -> dict:
        try:
            grasp.plan(target)
            plan_error = None
        except grasp.GraspError as exc:
            plan_error = str(exc)
        return {
            "label": target.label,
            "x_mm": round(target.x * 1000, 1),
            "y_mm": round(target.y * 1000, 1),
            "range_mm": round(math.hypot(target.x, target.y) * 1000),
            "bearing_deg": round(math.degrees(math.atan2(target.y, target.x))),
            "width_mm": round(target.width_m * 1000),
            "length_mm": round(target.length_m * 1000),
            "height_mm": round(target.height_m * 1000) if target.height_m else None,
            "angle_deg": round(target.angle_deg),
            "confidence": round(target.confidence, 2),
            "graspable": target.graspable and plan_error is None,
            "why_not": target.why_not() or plan_error or "",
        }

    # ------------------------------------------------------------- frames ----
    def frame(self, view: str | None = None):
        """The newest frame in `view` (default: the page's current view). (frame, seq)."""
        view = view or self.view
        stream = self.stream
        if stream is None:
            return None, 0
        raw, seq = stream.latest()
        if raw is None:
            return None, seq
        if view == "raw":
            return raw, seq
        if view == "edges":
            edges = cv2.Canny(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), 60, 160)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), seq
        if view == "mask" and self._mask is not None:
            return self._mask, seq
        if self._annotated is not None:
            return self._annotated, seq
        return raw, seq

    def wait_frame(self, seen: int, timeout: float = 1.0, view: str | None = None):
        stream = self.stream
        if stream is None:
            time.sleep(timeout)
            return None, seen
        _frame, seq = stream.wait(seen, timeout)
        if seq <= seen:
            return None, seen
        return self.frame(view)

    def _detect_loop(self) -> None:
        seen = 0
        while not self._closing.is_set():
            stream = self.stream
            if stream is None:
                time.sleep(0.2)
                continue
            frame, seq = stream.wait(seen, timeout=1.0)
            if frame is None:
                continue
            seen = seq
            started = time.monotonic()
            try:
                self._detect_frame(frame)
            except Exception as exc:  # noqa: BLE001 -- a detector bug must not kill the view
                self.detection = {"targets": [], "at_station": False,
                                  "error": f"{type(exc).__name__}: {exc}", "when": time.time()}
                self._annotated = frame
                self._mask = None
            elapsed = time.monotonic() - started
            if elapsed < DETECT_PERIOD_S:
                time.sleep(DETECT_PERIOD_S - elapsed)

    def _detect_frame(self, frame) -> None:
        """Run the chosen detector on one live frame and keep the pictures.

        Table coordinates are only honest when the arm is at a calibrated look;
        elsewhere the primary matrix is still used so the DRAWING lands on the
        right pixels, and `at_station` says the numbers mean nothing.
        """
        pose = self.pose()
        look = sweep.at_look(self.calibrated, pose) if (pose and self.calibrated) else None
        at_station = look is not None
        if look is not None:
            matrix, dyaw = look.matrix, look.dyaw
        elif self.calibrated:
            matrix, dyaw = self.calibrated[0][0], 0.0
        else:
            matrix, dyaw = np.diag([0.001, 0.001, 1.0]), 0.0
        nadir = kin.camera_nadir(pose) if pose else (0.0, 0.0)
        lens_m = kin.camera_height(pose) if pose else None

        error = None
        notes: list[str] = []
        try:
            targets = detect.ladder(frame, self.detector, matrix, prompt=self.prompt,
                                    nadir=nadir, dyaw=dyaw, object_mm=self.object_mm,
                                    url=self._vision_url(), lens_m=lens_m,
                                    look_name=look.name if look else "primary",
                                    note=notes.append)
        except FileNotFoundError as exc:
            targets, error = [], str(exc)
        except ValueError as exc:
            targets, error = [], f"detector: {exc}"
        if notes and not error:
            error = " ".join(n.strip() for n in notes)

        self._annotated = detect.annotate(frame, matrix, targets)
        self._mask = self._mask_for(frame, dyaw, targets)
        listed = []
        for target in targets:
            info = self._target_dict(target)
            # The sweep silently drops anything touching the frame edge, because
            # only the part in shot was measured. The live view shows it, but
            # must not call a clipped 60 mm block a graspable 35 mm one.
            if look is not None and not sweep.sees(look, target.x, target.y,
                                                   self._span(target), self._height(target)):
                info["graspable"] = False
                info["why_not"] = "partly out of frame -- its size is not known"
            listed.append(info)
        self.detection = {
            "targets": listed,
            "at_station": at_station,
            "error": error,
            "when": time.time(),
        }

    def _mask_for(self, frame, dyaw: float, targets: list[detect.Target] = ()):
        """How the detector saw the frame: its intermediate picture (the colour
        mask, the change mask, the tags it read) with every outline it drew on
        top -- filled green when the object is graspable, red when it is cut
        off or refused. The thing to look at when a pick misbehaves."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.detector in ("colour", "auto"):
            out = cv2.cvtColor(detect.colour_mask(frame), cv2.COLOR_GRAY2BGR) // 2
        elif self.detector == "changes":
            try:
                background = detect.load_background(dyaw)
            except FileNotFoundError:
                return None
            out = cv2.cvtColor(detect.foreground(frame, background), cv2.COLOR_GRAY2BGR) // 2
        else:
            out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) // 2
        if self.detector in ("markers", "auto"):
            for name, colour in ((cfg.OBJECT_TAG_DICT, (0, 200, 0)),
                                 (cfg.ARUCO_DICT, (200, 120, 0))):
                dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
                detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
                corners, ids, _ = detector.detectMarkers(gray)
                if ids is not None and len(ids):
                    cv2.aruco.drawDetectedMarkers(out, corners, ids, colour)
        for target in targets:
            if not target.pixels:
                continue
            pts = np.round(np.asarray(target.pixels)).astype(np.int32).reshape(-1, 1, 2)
            colour = (0, 200, 0) if target.graspable else (0, 0, 220)
            fill = out.copy()
            cv2.fillPoly(fill, [pts], colour)
            out = cv2.addWeighted(fill, 0.35, out, 0.65, 0)
            cv2.polylines(out, [pts], True, colour, 2)
            top = pts.reshape(-1, 2)[pts.reshape(-1, 2)[:, 1].argmin()]
            cv2.putText(out, f"{target.label} {target.width_m * 1000:.0f}mm"
                        + (" clipped" if target.clipped else ""),
                        (int(top[0]), max(12, int(top[1]) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        return out

    def _poll_loop(self) -> None:
        """Keep the pose fresh while nothing else is using the arm, and check the
        vision service now and then."""
        last_vision = 0.0
        last_battery = 0.0
        while not self._closing.is_set():
            time.sleep(0.5)
            now = time.time()
            if now - last_vision > 10.0:
                last_vision = now
                self.vision_ok = detect.vision_available(self.detector_url, timeout=1.5)
            if self.arm is not None and self._lock.acquire(blocking=False):
                try:
                    self._refresh(read_battery=now - last_battery > 15.0)
                    if now - last_battery > 15.0:
                        last_battery = now
                finally:
                    self._lock.release()

    # ------------------------------------------------------------- auto -----
    def _job_running(self) -> bool:
        thread = self._job_thread
        return thread is not None and thread.is_alive()

    def start_job(self, name: str, **params) -> Job:
        if self.mode != "auto":
            raise Refused("switch to automatic mode first")
        if self._job_running():
            raise Busy(f"{self.job.name} is still running -- stop it first")
        if self.arm is None:
            raise ArmError(self.arm_error or "arm is not connected")
        runner = {
            "sweep": self._job_sweep,
            "pick": self._job_pick,
            "place": self._job_place,
            "background": self._job_background,
            "survey": self._job_survey,
        }.get(name)
        if runner is None:
            raise ValueError(f"no job called {name!r}")
        self.job = Job(name)
        self._stop.clear()
        self._job_thread = threading.Thread(
            target=self._run_job, args=(self.job, runner, params), name=f"job-{name}",
            daemon=True)
        self._job_thread.start()
        return self.job

    def one_click(self, detector: str = "auto", object_mm: float | None = None,
                  drop_x: float | None = None, drop_y: float | None = None) -> Job:
        """The whole demo behind one button: choose the detector and the declared
        object width (None: measure it), switch to automatic, and run
        sweep -> refine -> pick -> place.

        Refused while a job runs, like any other start. Switching the mode here is
        deliberate: the button is meant for someone who does not want to know that
        there is a mode."""
        if self._job_running():
            raise Busy(f"{self.job.name} is still running -- stop it first")
        self.set_view(detector=detector, object_mm=object_mm, view="detect")
        self.set_mode("auto")
        return self.start_job("pick", refine=True, first=True, drop_x=drop_x, drop_y=drop_y)

    def _run_job(self, job: Job, runner, params: dict) -> None:
        self.log(f"--- {job.name} ---")
        started = time.monotonic()
        with self._lock:
            try:
                runner(**params)
                job.status = "done"
            except ArmError as exc:
                if self._stop.is_set():
                    job.status, job.message = "stopped", "stopped by the operator"
                else:
                    job.status, job.message = "failed", str(exc)
            except (camera.CameraError, kin.Unreachable, ValueError,
                    FileNotFoundError, sweep.NoLook) as exc:
                job.status, job.message = "failed", str(exc)
            except Exception as exc:  # noqa: BLE001 -- must reach the page, whatever it is
                job.status, job.message = "failed", f"{type(exc).__name__}: {exc}"
                self.log(traceback.format_exc())
            finally:
                job.finished = time.time()
                self._refresh()
        self.log(f"{job.name}: {job.status} in {time.monotonic() - started:.1f} s"
                 + (f" -- {job.message}" if job.message else ""))

    def _primary_look(self) -> Pose:
        if not self.calibrated:
            raise FileNotFoundError(self.calibration_error or "no table calibration loaded")
        return dict(self.calibrated[0][1])

    def _look_once(self, look: sweep.Look) -> list[detect.Target]:
        """What this station can measure: whole objects, wholly in shot."""
        return self._look_all(look)[0]

    def _look_all(self, look: sweep.Look) -> tuple[list[detect.Target], list[detect.Target]]:
        """tools/pick.py look_once(), taking the frame from the live stream.

        Returns (measurable, clipped): the objects wholly in shot, and the ones
        cut off at a frame edge -- unusable as targets, but sweep.hints() reads
        which edge they went out of. The exposure wait depends on what the move
        changed: a yaw from one station to the next keeps the same view and
        needs none (see YAW_SETTLE_S); anything that moved J2..J5 changed the
        picture."""
        before = self.pose()
        yaw_only = before is not None and all(
            abs(before[j] - look.pose[j]) <= 1 for j in (2, 3, 4, 5))
        self.arm.move_to(look.pose, speed_dps=STATION_DPS, verify=False)
        self._refresh()
        settle = min(self.settle_s, YAW_SETTLE_S) if yaw_only else self.settle_s
        frame = self.stream.settled(settle, count=1)[-1]
        found = detect.ladder(frame, self.detector, look.matrix, prompt=self.prompt,
                              nadir=look.nadir, dyaw=look.dyaw, object_mm=self.object_mm,
                              url=self._vision_url(), lens_m=kin.camera_height(look.pose),
                              look_name=look.name, note=self.log)
        kept = [t for t in found if not t.clipped
                and sweep.sees(look, t.x, t.y, self._span(t), self._height(t))]
        clipped = [t for t in found if t.clipped]
        if self.out_dir is not None:
            tag = f"sweep_{look.dyaw:+05.1f}.jpg".replace("+", "p").replace("-", "m")
            if look.name != "primary":
                tag = f"sweep_{look.name}_{tag[6:]}"
            with contextlib.suppress(camera.CameraError):
                camera.write_image(str(self.out_dir / tag),
                                   detect.annotate(frame, look.matrix, found))
        return kept, clipped

    def _span(self, target: detect.Target) -> float:
        """What has to be wholly in shot for this rung's measurement to be trusted.

        A tagged target was located by its printed square, and sized by that
        square's magnification, so only the TAG must be complete; a declared or
        inferred width says nothing about what was measured. Anything whose
        size came from a silhouette needs all of it.
        """
        if target.label.startswith("tag"):
            return sweep.TAG_SPAN_M
        return target.width_m

    @staticmethod
    def _height(target: detect.Target) -> float:
        """How tall the object stands, for the parallax that decides whether it
        is wholly in shot: measured when the detector could, else the demo cube."""
        return target.height_m or sweep.OBJECT_HEIGHT_M

    def _vision_url(self) -> str | None:
        """Where the neural rung is, or None when it is known to be down.

        The auto detector skips the rung rather than waiting on a dead socket at
        every station; the explicit yolo rung keeps trying, and falls back with a
        note, because the operator asked for it by name."""
        if self.detector == "auto" and not self.vision_ok:
            return None
        return self.detector_url

    def _job_survey(self) -> None:
        self.arm.move_to(self._primary_look(), speed_dps=STATION_DPS, verify=False)

    def _lap(self, what: str | None = None) -> None:
        """Log how long the stage since the last call took, and start the next.

        The numbers are the point of this: 'kinda slow' is not something that
        can be worked on, 'the search took 9.2 s and the place 10.1 s' is."""
        now = time.monotonic()
        if what and self._lap_started:
            self.log(f"  [{what} took {now - self._lap_started:.1f} s]")
        self._lap_started = now

    def _coverage_of(self, looks: list[sweep.Look]) -> float:
        """sweep.coverage(), remembered: it cost two seconds at the start of every
        search. (A warm-up thread at session start was tried and dropped: it
        starved the frame and detection loops for those two seconds, which is
        exactly when a freshly started session is asked what it sees. The IK
        grid under it is now memoised per process anyway.)"""
        key = (self.step_deg, tuple((round(look.dyaw, 3), tuple(look.pose.items()))
                               for look in looks))
        # One computation at a time: a job that arrives while the warm-up is
        # still running waits for its answer rather than starting a second.
        with self._coverage_lock:
            if key not in self._coverage:
                self._coverage[key], _missed = sweep.coverage(looks)
            return self._coverage[key]

    def _job_sweep(self, first: bool = False, **_ignored) -> list[detect.Target]:
        """Scan the ring of stations and list what is on the table.

        With `first`, the scan is a SEARCH rather than a survey: it starts at the
        station nearest to where the base already points -- so an object already
        in view costs no move at all -- works outward alternately left and right,
        and stops the moment something graspable is seen. The arm then stays at
        the station that saw it, ready to approach. Without `first` every station
        is visited and the arm returns to the survey pose.
        """
        self._lap()
        looks = sweep.stations(self._all_calibrated(), self.step_deg)
        pose = self.pose()
        if first and pose is not None:
            # Nearest station first -- but one calibrated look's whole ring before
            # the next: alternating between looks at every bearing would move
            # J2..J5 each time, which is the slow kind of move and the one that
            # needs the camera's exposure to settle again.
            order = [name for _m, _p, name in self._all_calibrated()]
            looks.sort(key=lambda look: (order.index(look.name), abs(look.pose[1] - pose[1])))
        covered = self._coverage_of(looks)
        self.log(f"{'searching' if first else 'sweeping'} {len(looks)} stations "
                 f"{self.step_deg:.0f} deg apart, covering {covered:.1%} of reach, "
                 f"detector = {self.detector}"
                 + (", nearest station first, stopping when something is found" if first else ""))
        seen: list[tuple[sweep.Look, detect.Target]] = []
        stopped_early = False
        # A search follows HINTS: something seen cut off at a frame edge says
        # where to look next (sweep.hints), and that look goes to the front of
        # the queue. Each station is visited once, hint or not.
        queue = list(looks)
        visited: set[tuple[str, int]] = set()
        while queue:
            look = queue.pop(0)
            key = (look.name, look.pose[1])
            if key in visited:
                continue
            visited.add(key)
            found, clipped = self._look_all(look)
            self.log(f"  J1={look.pose[1]:3d} (dyaw {look.dyaw:+6.1f}, {look.name}): "
                     f"{len(found)} object(s)"
                     + (f", {len(clipped)} cut off at the edge" if clipped else ""))
            seen.extend((look, t) for t in found)
            if first and any(self._plannable(t) for t in found):
                self.log("  found something graspable; stopping the scan here")
                stopped_early = True
                break
            if first and clipped:
                for lost in sweep.beyond_reach(look, clipped, self._all_calibrated()):
                    bearing = math.degrees(math.atan2(lost.y, lost.x))
                    self.log(f"  {lost.label} at bearing {bearing:+.0f} deg runs out of the "
                             f"far edge of the {look.name} look, the furthest there is: "
                             f"beyond reach")
                for hint in reversed(sweep.hints(look, clipped, self._all_calibrated())):
                    if (hint.look.name, hint.look.pose[1]) in visited:
                        continue
                    self.log(f"  hint: {hint.why} -> {hint.look.name} J1={hint.look.pose[1]}")
                    queue.insert(0, hint.look)
        merged = sweep.merge(seen)
        self.sweep_targets = merged
        self.sweep_when = time.time()
        self.log(f"{len(seen)} detection(s) merged into {len(merged)} object(s)")
        for target in merged:
            info = self._target_dict(target)
            verdict = "ok" if info["graspable"] else f"SKIP: {info['why_not']}"
            self.log(f"  {target.label:<12} {info['x_mm']:6.0f} mm fwd, {info['y_mm']:+6.0f} mm "
                     f"left  {info['width_mm']} x {info['length_mm']} mm   {verdict}")
        if not stopped_early:
            self.arm.move_to(self._primary_look(), speed_dps=STATION_DPS, verify=False)
        self._lap("search" if first else "sweep")
        return merged

    def _all_calibrated(self):
        if not self.calibrated:
            raise FileNotFoundError(self.calibration_error or "no table calibration loaded")
        return self.calibrated

    @staticmethod
    def _plannable(target: detect.Target) -> bool:
        try:
            grasp.plan(target)
        except grasp.GraspError:
            return False
        return True

    def _job_pick(self, index: int | None = None, dry_run: bool = False,
                  refine: bool = True, first: bool = True,
                  drop_x: float | None = None, drop_y: float | None = None,
                  **_ignored) -> None:
        if index is None or not self.sweep_targets:
            targets = self._job_sweep(first=first)
            # The SUREST plannable target, not the nearest. 2026-09-18: the outer
            # look returned the red cube at 0.45 and its shadow, ranged as a 45 mm
            # cube, at 0.20; the shadow was 4 mm nearer and got picked -- "closed
            # on nothing". Confidence is the one number that told them apart.
            target = max((t for t in targets if self._plannable(t)),
                         key=lambda t: t.confidence, default=None)
            if target is None:
                raise ValueError("nothing here can be picked up")
        else:
            if not (0 <= int(index) < len(self.sweep_targets)):
                raise ValueError(f"no target #{index}")
            target = self.sweep_targets[int(index)]
            grasp.plan(target)  # raises GraspError with the reason

        self.log(f"picking {target.label} at {target.x * 1000:.0f} mm fwd, "
                 f"{target.y * 1000:+.0f} mm left")
        if dry_run:
            step = grasp.plan(target)
            self.log(f"dry run: would open to {cfg.GRIPPER_GAP_MM[step.opening]} mm and "
                     f"approach at pitch {step.pitch:.0f}; not moving")
            return

        if refine:
            self._lap()
            target = self._refine(target)
            self._lap("refine")
        self._lap()
        held = grasp.pick(self.arm, target, reach_offset_m=self.reach_offset_mm / 1000)
        self._refresh()
        self._lap("pick")
        if not held:
            self.log("the gripper closed on nothing")
            self.arm.move_to(self._primary_look(), speed_dps=STATION_DPS, verify=False)
            raise ValueError("closed on nothing -- see the log for the usual causes")
        self.log("holding it")
        grasp.place(self.arm, DROP_X if drop_x is None else drop_x / 1000,
                    DROP_Y if drop_y is None else drop_y / 1000,
                    reach_offset_m=self.reach_offset_mm / 1000)
        self._lap("place")
        self.arm.move_to(self._primary_look(), speed_dps=STATION_DPS, verify=False)
        self.sweep_targets = []
        self._lap("return to survey")
        self.log("done")

    def _refine(self, target: detect.Target) -> detect.Target:
        try:
            look = sweep.best_refine(self.calibrated, target.x, target.y,
                                     self._span(target), self._height(target))
        except sweep.NoLook as exc:
            self.log(f"  cannot centre it ({exc}); using the sweep's measurement")
            return target
        pose = self.pose()
        if pose is not None and (abs(look.pose[1] - pose[1]) <= REFINE_IF_OFF_DEG
                                 and all(abs(look.pose[j] - pose[j]) <= 3
                                         for j in (2, 3, 4, 5))):
            self.log(f"  already looking at it head-on from J1={pose[1]}; no second look needed")
            return target
        self.log(f"  looking again with J1={look.pose[1]} (dyaw {look.dyaw:+.1f})")
        found = self._look_once(look)
        near = [t for t in found if math.hypot(t.x - target.x, t.y - target.y) <= REFIND_M]
        if not near:
            self.log(f"  WARNING: nothing within {REFIND_M * 1000:.0f} mm of the sweep's "
                     f"estimate; using it anyway")
            return target
        best = min(near, key=lambda t: math.hypot(t.x - target.x, t.y - target.y))
        moved = math.hypot(best.x - target.x, best.y - target.y)
        self.log(f"  refined to {best.x * 1000:.0f} mm fwd, {best.y * 1000:+.0f} mm left "
                 f"({moved * 1000:.1f} mm from the sweep)")
        return best

    def _job_place(self, x: float, y: float, **_ignored) -> None:
        grasp.place(self.arm, float(x) / 1000, float(y) / 1000,
                    reach_offset_m=self.reach_offset_mm / 1000)
        self.arm.move_to(self._primary_look(), speed_dps=STATION_DPS, verify=False)

    def _job_background(self, **_ignored) -> None:
        looks = sweep.stations(self._all_calibrated(), self.step_deg)
        self.log(f"photographing the empty table from {len(looks)} stations")
        for look in looks:
            self.arm.move_to(look.pose, speed_dps=STATION_DPS, verify=False)
            self._refresh()
            detect.save_background(self.stream.settled(self.settle_s)[-1],
                                   look.dyaw, look.name)
            self.log(f"  {look.name} dyaw {look.dyaw:+6.1f} -> "
                     f"{detect.background_path(look.dyaw, look.name).name}")
        self.arm.move_to(self._primary_look(), speed_dps=STATION_DPS, verify=False)
        self.log("do not move the board, the lamp or the robot before detecting")


class LogTee(io.TextIOBase):
    """Send every print() to the session log as well as the terminal.

    grasp.pick() and friends narrate what they are doing with print(). That is
    exactly what the operator wants to read in the browser, so rather than thread
    a callback through every module, stdout is tapped once here.
    """

    def __init__(self, session: Session, inner):
        self.session = session
        self.inner = inner
        self._buffer = ""

    def write(self, text: str) -> int:
        self.inner.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.session.log(line)
        return len(text)

    def flush(self) -> None:
        self.inner.flush()

    @classmethod
    def install(cls, session: Session) -> None:
        if not isinstance(sys.stdout, cls):
            sys.stdout = cls(session, sys.stdout)
