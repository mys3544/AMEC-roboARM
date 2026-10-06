"""Grab settled frames from the wrist camera.

Small enough to inline, except that getting it wrong is silent: the first frames off
this camera come back dark or half-exposed, and a homography fitted to them is wrong
in a way nothing downstream can detect. It was already copied into two tools; one
copy is easier to keep honest than three.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from roboarm import config as cfg

# The camera auto-exposes over the first second or so after it is opened. Counted in
# time, not frames: the Sonix ran ~7 fps and the C930e runs 30.
SETTLE_S = 1.2


class CameraError(RuntimeError):
    """The camera gave us nothing usable."""


def open_capture(device: int):
    """VideoCapture on `device` at cfg.WRIST_CAM_SIZE, with the fitted camera's
    settings applied (cfg.WRIST_CAM_PROPS: the C930e's focus is fixed, see there)."""
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    props = dict(cfg.WRIST_CAM_PROPS)
    if "CAP_PROP_FOURCC" in props:  # before the size: the sizes on offer depend on it
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*props.pop("CAP_PROP_FOURCC")))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.WRIST_CAM_SIZE[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.WRIST_CAM_SIZE[1])
    for name, value in props.items():
        capture.set(getattr(cv2, name), value)
    return capture


class Undistort:
    """Takes cfg.WRIST_CAM_LENS's distortion out of a frame; frames of another size
    (or with no lens model) pass through untouched. The maps are built once.
    Same pixel scale out as in: the corners lose a few pixels, and the one-pixel
    sliver the middle of the top edge would be short of is filled from the edge
    rather than left black, which the depth rung would take for a step."""

    def __init__(self):
        self._maps = None
        if cfg.WRIST_CAM_LENS is not None:
            K, dist = (np.array(v, np.float64) for v in cfg.WRIST_CAM_LENS)
            self._maps = cv2.initUndistortRectifyMap(K, dist, None, K, cfg.WRIST_CAM_SIZE,
                                                     cv2.CV_16SC2)

    def __call__(self, frame):
        if self._maps is None or frame.shape[1::-1] != cfg.WRIST_CAM_SIZE:
            return frame
        return cv2.remap(frame, *self._maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class Stream:
    """One camera held open, its newest frame always available.

    grab() below opens the device, takes a few frames and lets it go, which suits a
    tool that looks once. A live view cannot work that way: V4L2 gives a device to
    ONE process, so a second VideoCapture on a camera the viewer already holds just
    returns nothing -- the exact failure grab() reports as "is another process
    holding it?". So while the web app runs, this object owns the camera, and
    anything that wants a frame asks it rather than opening the device again.

    `latest()` hands back the newest frame with a sequence number; `wait()` blocks
    until a NEWER one than the caller has seen exists, which is what an MJPEG
    client needs so it neither spins nor sends the same frame twice. `settled()`
    is grab()'s exposure-settling wait, rephrased: discard frames for a moment
    after the arm has stopped, then return the freshest.
    """

    def __init__(self, device: int | None = None):
        self.device = cfg.WRIST_CAM if device is None else device
        self._frame = None
        self._seq = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stamps: list[float] = []
        self.error: str | None = None

    # ------------------------------------------------------------ lifecycle --
    def start(self) -> Stream:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _open(self):
        return open_capture(self.device)

    def _run(self) -> None:
        capture = self._open()
        undistort = Undistort()
        failures = 0
        try:
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    failures += 1
                    if failures >= 20:
                        # The device went away (unplugged, or another process took
                        # it). Say so where the UI can see it and keep trying, so
                        # plugging it back in needs no restart.
                        self.error = f"camera {self.device} is not delivering frames"
                        capture.release()
                        time.sleep(1.0)
                        capture = self._open()
                        failures = 0
                    else:
                        time.sleep(0.02)
                    continue
                failures = 0
                self.error = None
                self.publish(undistort(frame))
        finally:
            capture.release()

    # --------------------------------------------------------------- frames --
    def publish(self, frame) -> None:
        """Make `frame` the newest one. Public so a simulator can feed a Stream."""
        now = time.monotonic()
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._stamps.append(now)
            self._stamps = [t for t in self._stamps if now - t < 2.0]
            self._cond.notify_all()

    def latest(self):
        """(frame, seq). The frame is None until the camera has delivered one."""
        with self._cond:
            return self._frame, self._seq

    def wait(self, seen: int, timeout: float = 1.0):
        """Block until a frame newer than `seen` exists. Returns (frame, seq); the
        seq is unchanged and the frame None if nothing arrived within `timeout`."""
        with self._cond:
            if self._seq <= seen:
                self._cond.wait(timeout)
            if self._seq <= seen:
                return None, seen
            return self._frame, self._seq

    def settled(self, settle_s: float = 1.2):
        """The first new frame after waiting `settle_s` for exposure to settle.
        Raises CameraError if the camera is not producing anything."""
        _frame, seen = self.latest()
        deadline = time.monotonic() + settle_s
        while True:
            frame, seen = self.wait(seen, timeout=1.0)
            if frame is None:
                if self.error or time.monotonic() > deadline + 3.0:
                    raise CameraError(
                        self.error or f"camera {self.device} produced no frames"
                    )
                continue
            if time.monotonic() >= deadline:
                return frame

    @property
    def fps(self) -> float:
        with self._cond:
            if len(self._stamps) < 2:
                return 0.0
            span = self._stamps[-1] - self._stamps[0]
            return (len(self._stamps) - 1) / span if span > 0 else 0.0


def grab(count: int = 1) -> list:
    """Return `count` frames, after discarding the ones taken while exposure settles."""
    index = cfg.WRIST_CAM
    capture = open_capture(index)
    undistort = Undistort()
    frames = []
    try:
        settled = time.monotonic() + SETTLE_S
        # A dead device fails every read at once; bound the attempts, not the time.
        for _attempt in range(200 + count):
            ok, frame = capture.read()
            if ok and time.monotonic() >= settled:
                frames.append(undistort(frame))
                if len(frames) == count:
                    break
    finally:
        capture.release()
    if len(frames) < count:
        raise CameraError(
            f"wanted {count} frames from camera {index}, got {len(frames)} -- "
            f"is another process holding it?"
        )
    return frames


# The lens as measured by tools/camera_model.py intrinsics, for the frames Stream hands out.
INTRINSICS_PATH = Path("/app/data/cameras") / cfg.WRIST_CAMERA / "intrinsics.json"


def intrinsics(path: Path = INTRINSICS_PATH) -> tuple[list, list] | None:
    """(K, distortion) of the pictures this module hands out -- already undistorted, so the
    distortion is only what that correction leaves: tools/camera_model.py's fit when there is
    one. Else cfg.WRIST_CAM_LENS's matrix with none left, which is what the correction assumes
    but whose focal length was never pinned (416.5 px; the C930e measured 506.9 +- 1.4 on
    2026-10-06). None when neither is known."""
    if path.is_file():
        fitted = json.loads(path.read_text())
        return fitted["K"], fitted["dist"]
    if cfg.WRIST_CAM_LENS is not None:
        return [list(row) for row in cfg.WRIST_CAM_LENS[0]], [0.0] * 5
    return None


def write_image(path: str, frame) -> None:
    """Save a frame, loudly.

    cv2.imwrite RETURNS False on failure rather than raising, so writing to a
    directory that does not exist quietly produces nothing -- which is exactly what
    happened to every annotated frame this project wrote before /out was mounted.
    """
    if not cv2.imwrite(path, frame):
        raise CameraError(
            f"could not write {path} -- does its directory exist inside the "
            f"container? (see the volumes in compose.yaml)"
        )
