"""The robot's hardware over HTTP, so the control panel can run on another machine.

Runs ON THE ROBOT (`tools/hwbridge.py`, the `bridge` compose service) and holds
the arm and the cameras. It has no page and makes no decisions: every request is
one method call on roboarm.arm.Arm, one camera's frames, or the calibration file.
The panel on the laptop (roboarm.web.remote) is what turns those into a session.

    GET  /health                       {ok, arm: {connected, error}, cameras}
    GET  /camera/<name>/stream.mjpg    multipart MJPEG, opened on first use
    GET  /calibration                  data/table_homography.json, verbatim
    GET  /vision/health                forwarded to the vision container
    POST /vision/detect                raw JPEG in, forwarded to the vision container
    POST /arm/connect                  open the serial port (idempotent)
    POST /arm/call                     {"method": "move_to", "args": [...], "kwargs": {...}}
                                       -> {"result": ...}; an ArmError is {"error": ..., "type": "ArmError"}
    POST /arm/interrupt                abandon the move in progress (Arm.interrupt)

Two things it does guard. Calls are serialised on one lock, because the SDK cannot
interleave two conversations -- a second caller waits. And /arm/interrupt is
handled OUTSIDE that lock, which is the whole point of it: it is how a stop button
on the laptop reaches a glide already under way here.

No authentication: this is the robot's WiFi, and the port is only published there.
Do not expose it further.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from roboarm import camera
from roboarm import config as cfg
from roboarm.arm import ArmError
from roboarm.web.server import send_json, send_mjpeg

# Everything a caller may invoke on the arm. Anything else is a 400, so a bug on
# the laptop cannot reach into the SDK object behind it.
ALLOWED = frozenset({
    "read", "move_to", "home", "set_gripper", "open_gripper", "close_gripper",
    "grasped", "hold", "release", "engage", "out_of_range", "assert_ready", "battery",
})


def _int_keys(value):
    """JSON turned the pose's joint numbers into strings; turn them back."""
    if isinstance(value, dict) and value and all(str(k).lstrip("-").isdigit() for k in value):
        return {int(k): v for k, v in value.items()}
    return value


class Bridge:
    def __init__(self, arm_factory: Callable[[], object],
                 streams: dict[str, Callable[[], camera.Stream]],
                 calibration_path: Path):
        self._arm_factory = arm_factory
        self._stream_factories = streams
        self.calibration_path = calibration_path
        self.arm = None
        self.arm_error: str | None = None
        self._lock = threading.RLock()
        self._interrupt = threading.Event()
        self._streams: dict[str, camera.Stream] = {}
        self._streams_lock = threading.Lock()

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
            if method == "battery":
                return float(self.arm.bot.get_battery_voltage())
            fn = getattr(self.arm, method)
            return fn(*[_int_keys(a) for a in args],
                      **{k: _int_keys(v) for k, v in kwargs.items()})

    def interrupt(self) -> None:
        self._interrupt.set()

    def close(self) -> None:
        with self._streams_lock:
            for stream in self._streams.values():
                stream.stop()
            self._streams.clear()
        with self._lock:
            if self.arm is not None:
                try:
                    self.arm.__exit__(None, None, None)
                finally:
                    self.arm = None

    # ------------------------------------------------------------- cameras --
    def stream(self, name: str) -> camera.Stream:
        if name not in self._stream_factories:
            raise KeyError(name)
        with self._streams_lock:
            if name not in self._streams:
                self._streams[name] = self._stream_factories[name]().start()
            return self._streams[name]

    def health(self) -> dict:
        return {
            "ok": True,
            "arm": {"connected": self.arm is not None, "error": self.arm_error},
            "cameras": {name: {"open": name in self._streams,
                               "fps": round(self._streams[name].fps, 1) if name in self._streams else 0,
                               "error": self._streams[name].error if name in self._streams else None}
                        for name in self._stream_factories},
            "calibration": self.calibration_path.exists(),
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
        elif path == "/vision/health":
            self._forward_vision("/health")
        elif path.startswith("/camera/") and path.endswith("/stream.mjpg"):
            name = path[len("/camera/"):-len("/stream.mjpg")]
            try:
                stream = self.bridge.stream(name)
            except KeyError:
                send_json(self, {"error": f"no camera called {name!r}"}, 404)
                return
            send_mjpeg(self, stream.wait, getattr(self.server, "max_stream_frames", None))
        elif path == "/calibration":
            if not self.bridge.calibration_path.exists():
                send_json(self, {"error": f"no calibration at {self.bridge.calibration_path}"}, 404)
                return
            send_json(self, json.loads(self.bridge.calibration_path.read_text()))
        else:
            send_json(self, {"error": f"no such endpoint: GET {path}"}, 404)

    def _forward_vision(self, tail: str, body: bytes | None = None) -> None:
        """Relay one request to the vision container and its reply back.

        A panel on a laptop cannot reach the compose-internal `vision` host; the
        bridge can. Only the headers the service reads are passed on."""
        headers = {name: self.headers[name] for name in
                   ("Content-Type", "X-Vision-Prompt", "X-Vision-Conf") if self.headers.get(name)}
        status, reply = vision_request(cfg.DETECTOR_URL + tail, body, headers)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if path == "/vision/detect":
            self._forward_vision("/detect", body)
            return
        try:
            params = json.loads(body or b"{}")
        except json.JSONDecodeError as exc:
            send_json(self, {"error": f"bad JSON: {exc}"}, 400)
            return

        if path == "/arm/interrupt":
            self.bridge.interrupt()
            send_json(self, {"ok": True})
        elif path == "/arm/connect":
            self.bridge.connect()
            send_json(self, {"ok": self.bridge.arm is not None, "error": self.bridge.arm_error})
        elif path == "/arm/call":
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
                send_json(self, {"result": result})
        else:
            send_json(self, {"error": f"no such endpoint: POST {path}"}, 404)


def vision_request(url: str, body: bytes | None, headers: dict[str, str],
                   timeout: float = cfg.DETECTOR_TIMEOUT_S) -> tuple[int, bytes]:
    """(status, JSON bytes) from the vision service; 502 with a message if it
    cannot be reached. A function so a test can stand in for the service."""
    request = urllib.request.Request(url, data=body, headers=headers,
                                     method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return reply.status, reply.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read() or json.dumps({"error": str(exc)}).encode()
    except (urllib.error.URLError, OSError) as exc:
        return 502, json.dumps({"error": f"vision service unreachable: {exc}"}).encode()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    max_stream_frames: int | None = None


def make_server(bridge: Bridge, host: str = "0.0.0.0", port: int = 8761) -> Server:
    handler = type("BoundBridgeHandler", (Handler,), {"bridge": bridge})
    return Server((host, port), handler)


def serve(bridge: Bridge, host: str = "0.0.0.0", port: int = 8761) -> None:
    server = make_server(bridge, host, port)
    bridge.connect()
    status = "arm connected" if bridge.arm is not None else f"arm NOT connected: {bridge.arm_error}"
    print(f"roboarm bridge on port {server.server_port}: {status}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        bridge.close()
        print("bridge stopped; arm left holding", flush=True)
