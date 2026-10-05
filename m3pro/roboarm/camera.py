"""The camera's newest frame, held for whoever wants it.

tools/m3bridge.py publishes each colour frame from ROS into a Stream, and the bridge's
/stream.mjpg waits on it (roboarm/web/bridge.py).
"""

from __future__ import annotations

import threading
import time


class Stream:
    """The newest frame, always available.

    `wait()` blocks until a NEWER frame than the caller has seen exists, which is what an
    MJPEG client needs so it neither spins nor sends the same frame twice.
    """

    def __init__(self):
        self._frame = None
        self._seq = 0
        self._cond = threading.Condition()
        self._stamps: list[float] = []
        self.error: str | None = None

    def publish(self, frame) -> None:
        """Make `frame` the newest one."""
        now = time.monotonic()
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._stamps.append(now)
            self._stamps = [t for t in self._stamps if now - t < 2.0]
            self._cond.notify_all()

    def wait(self, seen: int, timeout: float = 1.0):
        """Block until a frame newer than `seen` exists. Returns (frame, seq); the
        seq is unchanged and the frame None if nothing arrived within `timeout`."""
        with self._cond:
            if self._seq <= seen:
                self._cond.wait(timeout)
            if self._seq <= seen:
                return None, seen
            return self._frame, self._seq

    @property
    def fps(self) -> float:
        with self._cond:
            if len(self._stamps) < 2:
                return 0.0
            span = self._stamps[-1] - self._stamps[0]
            return (len(self._stamps) - 1) / span if span > 0 else 0.0
