"""The robot's arm and camera over HTTP, so that code on another machine can drive them.

Runs ON THE ROBOT, inside tools/m3bridge.py, and holds the arm and the camera's frames. It has
no page and makes no decisions: every request is one method call on the arm
(roboarm.m3arm.M3Arm) or the camera's frames.

    GET  /health                       {ok, arm: {connected, error}, camera}
    GET  /stream.mjpg                  multipart MJPEG of the arm's camera
    POST /arm/call                     {"method": "move_to", "args": [...], "kwargs": {...}}
                                       -> {"result": ...}; an ArmError is {"error": ..., "type": "ArmError"}
    POST /arm/interrupt                abandon the move in progress (M3Arm.interrupt)

Two things it does guard. Calls are serialised on one lock, one move at a time -- a second
caller waits. And /arm/interrupt is handled OUTSIDE that lock, which is the whole point of it:
it is how a stop reaches a move already under way.

No authentication: the lab network only. Do not expose it further.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2

from roboarm import camera
from roboarm.m3arm import ArmError

BOUNDARY = "roboarmframe"
JPEG_QUALITY = 80

# Everything a caller may invoke on the arm. Anything else is a 400, so a bug on
# the laptop cannot reach into the object behind it.
ALLOWED = frozenset({
    "read", "move_to", "step_to", "follow", "home", "set_gripper", "open_gripper", "close_gripper",
    "hold", "engage", "battery",
})


def send_json(handler: BaseHTTPRequestHandler, payload, status: int = 200) -> None:
    body = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def send_mjpeg(handler: BaseHTTPRequestHandler, wait_frame,
               max_frames: int | None = None) -> None:
    """Multipart MJPEG: one JPEG per camera frame, for as long as the client stays.

    `wait_frame(seen, timeout)` returns (frame, seq) with frame None on timeout.
    Each part is written as one buffer, so a slow client stalls between frames
    rather than mid-frame. A disconnect surfaces as a socket error, which simply
    ends the loop.
    """
    handler.send_response(200)
    handler.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Connection", "close")
    handler.end_headers()
    seen = 0
    sent = 0
    try:
        while max_frames is None or sent < max_frames:
            frame, seq = wait_frame(seen, 1.0)
            if frame is None:
                continue
            seen = seq
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                continue
            body = jpeg.tobytes()
            handler.wfile.write(
                f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                f"Content-Length: {len(body)}\r\n\r\n".encode() + body + b"\r\n"
            )
            handler.wfile.flush()
            sent += 1
    except (TimeoutError, BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
        pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # Tests set this so a stream request finishes instead of running forever.
    max_stream_frames: int | None = None


def _int_keys(value):
    """JSON turned the pose's joint numbers into strings; turn them back."""
    if isinstance(value, dict) and value and all(str(k).lstrip("-").isdigit() for k in value):
        return {int(k): v for k, v in value.items()}
    return value


class Bridge:
    def __init__(self, arm_factory: Callable[[], object], stream: Callable[[], camera.Stream]):
        self._arm_factory = arm_factory
        self._stream_factory = stream
        self.arm = None
        self.arm_error: str | None = None
        self._lock = threading.RLock()
        self._interrupt = threading.Event()
        self._stream: camera.Stream | None = None
        self._stream_lock = threading.Lock()

    # ---------------------------------------------------------------- arm ---
    def connect(self) -> None:
        with self._lock:
            if self.arm is not None:
                return
            try:
                arm = self._arm_factory()
                arm.__enter__()
            except Exception as exc:  # noqa: BLE001 -- reported to the caller, whatever it is
                self.arm_error = str(exc)
                return
            arm.interrupt = self._interrupt.is_set
            self.arm = arm
            self.arm_error = None

    def call(self, method: str, args: list, kwargs: dict):
        if method not in ALLOWED:
            raise ValueError(f"{method!r} is not something the bridge will call")
        with self._lock:
            if self.arm is None:
                self.connect()
            if self.arm is None:
                raise ArmError(self.arm_error or "arm is not connected")
            # A stale interrupt from the last stop must not abort this new call.
            self._interrupt.clear()
            fn = getattr(self.arm, method)
            return fn(*[_int_keys(a) for a in args],
                      **{k: _int_keys(v) for k, v in kwargs.items()})

    def interrupt(self) -> None:
        self._interrupt.set()

    def close(self) -> None:
        with self._lock:
            if self.arm is not None:
                try:
                    self.arm.__exit__(None, None, None)
                finally:
                    self.arm = None

    # -------------------------------------------------------------- camera --
    def stream(self) -> camera.Stream:
        with self._stream_lock:
            if self._stream is None:
                self._stream = self._stream_factory()
            return self._stream

    def health(self) -> dict:
        stream = self._stream
        return {
            "ok": True,
            "arm": {"connected": self.arm is not None, "error": self.arm_error},
            "camera": {"open": stream is not None,
                       "fps": round(stream.fps, 1) if stream else 0,
                       "error": stream.error if stream else None},
        }


class Handler(BaseHTTPRequestHandler):
    bridge: Bridge
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            send_json(self, self.bridge.health())
        elif path == "/stream.mjpg":
            send_mjpeg(self, self.bridge.stream().wait,
                       getattr(self.server, "max_stream_frames", None))
        else:
            send_json(self, {"error": f"no such endpoint: GET {path}"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            params = json.loads(body or b"{}")
        except json.JSONDecodeError as exc:
            send_json(self, {"error": f"bad JSON: {exc}"}, 400)
            return

        if path == "/arm/interrupt":
            self.bridge.interrupt()
            send_json(self, {"ok": True})
        elif path == "/arm/call":
            started = time.perf_counter()
            try:
                result = self.bridge.call(params.get("method", ""), params.get("args", []),
                                          params.get("kwargs", {}))
            except ArmError as exc:
                send_json(self, {"error": str(exc), "type": "ArmError"}, 400)
            except (ValueError, TypeError, KeyError) as exc:
                send_json(self, {"error": str(exc), "type": type(exc).__name__}, 400)
            except Exception as exc:  # noqa: BLE001 -- the laptop must hear about it
                send_json(self, {"error": f"{type(exc).__name__}: {exc}", "type": "bug"}, 500)
            else:
                if isinstance(result, dict):
                    result = {str(k): v for k, v in result.items()}
                # How long it ran here, so a caller can tell the arm from the network.
                # Includes the wait for the one-call-at-a-time lock.
                send_json(self, {"result": result,
                                 "elapsed_s": round(time.perf_counter() - started, 4)})
        else:
            send_json(self, {"error": f"no such endpoint: POST {path}"}, 404)
