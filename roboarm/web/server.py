"""HTTP in front of a Session. Standard library only, on purpose.

The core container deliberately carries no web framework -- it has to build in a
minute and start in a second -- and nothing here needs one: a few JSON endpoints,
one static page, and an MJPEG stream, which is just a response that never ends.
ThreadingHTTPServer gives each client its own thread, so a browser holding the
stream open does not block the JSON calls the same page is making.

    GET  /                      the page
    GET  /stream.mjpg           multipart MJPEG of the current view
    GET  /snapshot.jpg          one frame of the current view
    GET  /api/state             everything the page shows, as JSON
    GET  /api/log?since=N       log lines newer than N

    POST /api/mode              {"mode": "manual" | "auto"}
    POST /api/view              {view, detector, object_mm, step_deg, speed_dps, reach_offset_mm}
    POST /api/arm/move          {"joints": {"1": 90, ...}, "speed_dps": 30}
    POST /api/arm/jog           {"joint": 2, "delta": -5}
    POST /api/arm/cartesian     {"dx": 5, "dy": 0, "dz": 0}  or  {"x": 160, "y": 0, "z": 40}
                                (mm; x forward, y left, z above the table)
    POST /api/arm/gripper       {"action": "open" | "close"}  or  {"angle": 120}
    POST /api/arm/preset        {"name": "home" | "survey"}
    POST /api/arm/hold          the stop button; always allowed
    POST /api/arm/release       torque off (the arm sags)
    POST /api/arm/engage        torque on
    POST /api/arm/connect       (re)open the serial port
    POST /api/auto/start        {"job": "sweep"|"pick"|"place"|"background"|"survey", ...params}
    POST /api/auto/oneclick     {detector, object_mm, drop_x, drop_y}: set up, go automatic, pick & place
    POST /api/auto/stop
    POST /api/calibration/reload

Errors come back as {"error": "..."} with 400 (bad request / arm refused it),
409 (busy, or wrong mode) or 500 (a bug).
"""

from __future__ import annotations

import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from roboarm import camera, sweep
from roboarm import kinematics as kin
from roboarm.arm import ArmError
from roboarm.web.session import VIEWS, Busy, Refused, Session

STATIC = Path(__file__).parent / "static"
BOUNDARY = "roboarmframe"
JPEG_QUALITY = 80


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
    ends the loop. Shared by the page's stream and the hardware bridge's.
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


def _routes(session: Session) -> dict:
    """(method, path) -> callable(params) -> JSON-able."""

    def pose_reply(pose):
        return {"ok": True, "pose": {str(j): a for j, a in (pose or {}).items()}}

    return {
        ("GET", "/api/state"): lambda p: session.snapshot(),
        ("GET", "/api/log"): lambda p: {"lines": session.log_since(int(p.get("since", 0)))},
        ("POST", "/api/mode"): lambda p: (session.set_mode(p["mode"]), {"ok": True})[1],
        ("POST", "/api/view"): lambda p: (session.set_view(
            view=p.get("view"), detector=p.get("detector"), object_mm=p.get("object_mm"),
            step_deg=p.get("step_deg"), speed_dps=p.get("speed_dps"),
            reach_offset_mm=p.get("reach_offset_mm")), {"ok": True})[1],
        ("POST", "/api/arm/move"): lambda p: pose_reply(
            session.move_joints(p["joints"], p.get("speed_dps"))),
        ("POST", "/api/arm/jog"): lambda p: pose_reply(
            session.jog(int(p["joint"]), float(p["delta"]), p.get("speed_dps"))),
        ("POST", "/api/arm/cartesian"): lambda p: pose_reply(session.cartesian(
            float(p.get("dx", 0)), float(p.get("dy", 0)), float(p.get("dz", 0)),
            p.get("x"), p.get("y"), p.get("z"))),
        ("POST", "/api/arm/gripper"): lambda p: pose_reply(
            session.gripper(p.get("angle"), p.get("action"))),
        ("POST", "/api/arm/preset"): lambda p: pose_reply(session.preset(p["name"])),
        ("POST", "/api/arm/hold"): lambda p: pose_reply(session.hold()),
        ("POST", "/api/arm/release"): lambda p: (session.release(), {"ok": True})[1],
        ("POST", "/api/arm/engage"): lambda p: pose_reply(session.engage()),
        ("POST", "/api/arm/connect"): lambda p: (session.connect(), {
            "ok": session.arm is not None, "error": session.arm_error})[1],
        ("POST", "/api/auto/start"): lambda p: session.start_job(
            p["job"], **{k: v for k, v in p.items() if k != "job"}).snapshot(),
        ("POST", "/api/auto/oneclick"): lambda p: session.one_click(
            p.get("detector") or "auto", p.get("object_mm"),
            p.get("drop_x"), p.get("drop_y")).snapshot(),
        ("POST", "/api/auto/stop"): lambda p: (session.stop(), {"ok": True})[1],
        ("POST", "/api/calibration/reload"): lambda p: (
            session.reload_calibration(), session.snapshot()["calibration"])[1],
    }


class Handler(BaseHTTPRequestHandler):
    session: Session
    routes: dict
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet: one line per poll is noise
        pass

    # ------------------------------------------------------------- replies --
    def _file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            send_json(self, {"error": "not found"}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _params(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"bad JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise TypeError("JSON body must be an object")
        return data

    def _dispatch(self, method: str, params: dict) -> None:
        path = urlparse(self.path).path
        route = self.routes.get((method, path))
        if route is None:
            send_json(self, {"error": f"no such endpoint: {method} {path}"}, 404)
            return
        try:
            send_json(self, route(params))
        except (Busy, Refused) as exc:
            send_json(self, {"error": str(exc)}, HTTPStatus.CONFLICT)
        except (ArmError, camera.CameraError, kin.Unreachable, sweep.NoLook,
                ValueError, KeyError, TypeError, FileNotFoundError) as exc:
            detail = f"missing field {exc}" if isinstance(exc, KeyError) else str(exc)
            send_json(self, {"error": detail}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001 -- the page must hear about it
            self.session.log(f"server error on {method} {path}: {type(exc).__name__}: {exc}")
            send_json(self, {"error": f"{type(exc).__name__}: {exc}"},
                      HTTPStatus.INTERNAL_SERVER_ERROR)

    # ------------------------------------------------------------ methods --
    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._file(STATIC / "index.html", "text/html; charset=utf-8")
        elif url.path in ("/stream.mjpg", "/snapshot.jpg"):
            # ?view=mask|detect|raw|edges renders that view for THIS client only,
            # so a second pane on the page can show the detector's mask while
            # the main one shows the detections.
            view = parse_qs(url.query).get("view", [None])[-1]
            view = view if view in VIEWS else None
            if url.path == "/stream.mjpg":
                self._stream(view)
            else:
                self._snapshot(view)
        else:
            params = {k: v[-1] for k, v in parse_qs(url.query).items()}
            self._dispatch("GET", params)

    def do_POST(self) -> None:
        try:
            params = self._params()
        except (ValueError, TypeError) as exc:
            send_json(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._dispatch("POST", params)

    # ------------------------------------------------------------- frames --
    def _snapshot(self, view: str | None = None) -> None:
        frame, _seq = self.session.frame(view)
        if frame is None:
            send_json(self, {"error": "no frame yet"}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            send_json(self, {"error": "could not encode the frame"}, 500)
            return
        body = jpeg.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, view: str | None = None) -> None:
        send_mjpeg(self, lambda seen, timeout=1.0: self.session.wait_frame(seen, timeout, view),
                   getattr(self.server, "max_stream_frames", None))


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # Tests set this so a stream request finishes instead of running forever.
    max_stream_frames: int | None = None


def make_server(session: Session, host: str = "0.0.0.0", port: int = 8080) -> Server:
    handler = type("BoundHandler", (Handler,), {"session": session, "routes": _routes(session)})
    return Server((host, port), handler)


def serve(session: Session, host: str = "0.0.0.0", port: int = 8080) -> None:
    server = make_server(session, host, port)
    shown = "localhost" if host in ("0.0.0.0", "") else host
    print(f"roboarm web: http://{shown}:{server.server_port}/  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        started = time.time()
        server.shutdown()
        session.close()
        print(f"stopped in {time.time() - started:.1f}s; arm left holding", flush=True)
