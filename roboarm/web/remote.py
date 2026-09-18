"""The real robot's arm and camera, used from another machine through the bridge.

RemoteArm has the same surface as roboarm.arm.Arm, so the session, grasp.pick()
and everything else are none the wiser: a call becomes one POST to the robot
(roboarm.web.bridge) and the pose comes back. RemoteStream is a camera.Stream fed
by the bridge's MJPEG instead of a V4L2 device.

The one thing that needs care is the stop button. Arm.interrupt is a callable the
arm polls between glide steps -- but the arm is on the robot and the callable is
here. So while a call is in flight, a watcher thread polls `interrupt()` locally
and, the moment it answers True, POSTs /arm/interrupt; the bridge's Arm sees its
own flag on the next step and abandons the move exactly as it would locally.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable

import cv2
import numpy as np

from roboarm import camera
from roboarm import workspace as ws
from roboarm.arm import ArmError
from roboarm.web.bridge import ALLOWED

# A move can legitimately take a while: 180 degrees at 8 deg/s is over twenty
# seconds, and engage() sleeps for two and a half. Generous, not infinite.
CALL_TIMEOUT_S = 120.0
INTERRUPT_POLL_S = 0.05


def _request(url: str, body: dict | None = None, timeout: float = 10.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return json.loads(reply.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read() or b"{}")
        except json.JSONDecodeError:
            payload = {}
        payload.setdefault("error", f"HTTP {exc.code} from {url}")
        payload.setdefault("type", "http")
        return payload
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise ArmError(f"cannot reach the robot bridge at {url}: {exc}") from exc


def _pose(value):
    if isinstance(value, dict):
        return {int(k): v for k, v in value.items()}
    return value


class RemoteArm:
    """roboarm.arm.Arm's interface, executed on the robot.

    Every method the bridge ALLOWS (read, move_to, home, set_gripper, grasped,
    hold, release, engage, ...) is one POST, resolved by `__getattr__`; only the
    lifecycle and the battery need spelling out.
    """

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.port = self.url
        self.interrupt: Callable[[], bool] | None = None
        self._connected = False

    def __enter__(self) -> RemoteArm:
        reply = _request(self.url + "/arm/connect", {}, timeout=15.0)
        if not reply.get("ok"):
            raise ArmError(reply.get("error") or "the bridge could not open the arm")
        self._connected = True
        return self

    def __exit__(self, *exc) -> None:
        # The robot's own Arm keeps holding; nothing to release here.
        self._connected = False

    def __getattr__(self, name: str):
        if name not in ALLOWED:
            raise AttributeError(name)
        return lambda *args, **kwargs: self._call(name, *args, **kwargs)

    # ------------------------------------------------------------- plumbing --
    def _call(self, method: str, *args, **kwargs):
        stop = threading.Event()
        watcher = None
        if self.interrupt is not None:
            watcher = threading.Thread(target=self._watch, args=(stop,), daemon=True)
            watcher.start()
        try:
            reply = _request(self.url + "/arm/call",
                             {"method": method, "args": list(args), "kwargs": kwargs},
                             timeout=CALL_TIMEOUT_S)
        finally:
            stop.set()
            if watcher is not None:
                watcher.join(timeout=1.0)
        if "error" in reply:
            raise ArmError(reply["error"])
        return _pose(reply.get("result"))

    def _watch(self, stop: threading.Event) -> None:
        while not stop.wait(INTERRUPT_POLL_S):
            if self.interrupt():
                with contextlib.suppress(ArmError):
                    _request(self.url + "/arm/interrupt", {}, timeout=5.0)
                return

    @property
    def bot(self) -> RemoteArm:
        return self

    def get_battery_voltage(self) -> float:
        return float(self._call("battery"))


class RemoteStream(camera.Stream):
    """A camera.Stream whose frames arrive as the bridge's MJPEG."""

    def __init__(self, url: str):
        super().__init__(device=-1)
        self.url = f"{url.rstrip('/')}/stream.mjpg"

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                request = urllib.request.Request(self.url)
                with urllib.request.urlopen(request, timeout=10.0) as reply:
                    content_type = reply.headers.get("Content-Type", "")
                    if "boundary=" not in content_type:
                        raise ValueError(f"not an MJPEG stream: {content_type!r}")
                    self.error = None
                    self._read_parts(reply)
            except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
                self.error = f"camera via bridge: {exc}"
                time.sleep(1.0)

    def _read_parts(self, reply) -> None:
        """One multipart body: header lines, a blank line, `Content-Length` bytes."""
        while not self._stop.is_set():
            length = None
            while True:
                line = reply.readline()
                if not line:
                    return  # the bridge went away; reconnect
                stripped = line.strip()
                if stripped.lower().startswith(b"content-length:"):
                    length = int(stripped.split(b":", 1)[1])
                elif stripped == b"" and length is not None:
                    break
            data = reply.read(length)
            if len(data) < length:
                return
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                self.publish(frame)


def calibration(url: str):
    """The robot's table calibration, as workspace.load_all() would return it."""
    reply = _request(url.rstrip("/") + "/calibration", timeout=10.0)
    if "error" in reply and "homography" not in reply:
        raise FileNotFoundError(reply["error"])
    return ws.parse(reply)


def health(url: str) -> dict:
    return _request(url.rstrip("/") + "/health", timeout=5.0)
