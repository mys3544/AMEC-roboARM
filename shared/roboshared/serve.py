"""The robot side's HTTP surface for the jobs, the same on both robots (m3pro/tools/m3bridge.py,
x3plus/tools/x3bridge.py) -- what the console's arm card drives and watches:

    POST /pick/start    {"kind": "cube"|"block"|null, "at": [x, y]|null, "drive": bool}   one pick, then home
    POST /pick/cleanup  {"kind": ..., "drive": bool}   every kit thing in reach, one by one
    POST /pick/home     POST /pick/stop (the arm holds where it is, at once)
    POST /pick/scan     {"radius": m, "stops": n}   the M3 only: drive round the thing ahead, the
                        camera on it, and keep the views (m3pro/roboarm/m3scan.py)
    POST /pick/teach    {"name": str}   the M3 only: learn the thing alone in front as `name`
    POST /pick/find     name what is on the table in front    POST /pick/named {"name"}: pick it
                        (m3pro/roboarm/m3teach.py)
    GET  /pick/state    the job, its step, the counts, the last attempt, what the last look found
    GET  /pick/log?since=N    the narrated steps after N
    GET  /pick/frame.jpg      the last look, marked
    GET  /pick/settings       [{key, label, unit, min, max, step, default, value}]; POST {key: value} changes them
    GET  /pick/taught         {"names": [...]}: the things taught by showing them (the M3 only)
    GET  /api/metrics         where the time went (roboshared.metrics: one run a job)

A bridge's handler asks reply() first and sends what it gives (respond()); None means the path is
the bridge's own. No authentication: the lab network only.
"""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from roboshared import metrics


def blank_frame(text: str = "no look yet") -> bytes:
    image = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(image, text, (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", image)[1].tobytes()


_BLANK = None


def _json(payload, status: int = 200) -> tuple[int, str, bytes]:
    return status, "application/json", json.dumps(payload).encode()


def reply(picker, interrupt, method: str, path: str, body: bytes = b"") -> tuple[int, str, bytes] | None:
    """(status, content type, body) for `method` `path` (with its query) when it is one of the
    module's endpoints, else None. `picker`: the robot's roboshared.task.Task; `interrupt()`: the
    arm's stop, outside its call lock (the bridge's)."""
    global _BLANK
    url = urlparse(path)
    if url.path == "/api/metrics" and method == "GET":
        return _json(metrics.snapshot(events="events=0" not in url.query))
    if picker is None or not url.path.startswith("/pick/"):
        return None
    if method == "GET":
        if url.path == "/pick/state":
            return _json(picker.status())
        if url.path == "/pick/log":
            since = parse_qs(url.query).get("since", ["0"])[0]
            return _json({"lines": picker.log(int(since) if since.isdigit() else 0)})
        if url.path == "/pick/settings":
            return _json({"settings": picker.settings()})
        if url.path == "/pick/taught" and hasattr(picker, "taught"):
            return _json({"names": picker.taught()})
        if url.path == "/pick/frame.jpg":
            if picker.frame is None and _BLANK is None:
                _BLANK = blank_frame()
            return 200, "image/jpeg", picker.frame or _BLANK
        return _json({"error": f"no such endpoint: GET {url.path}"}, 404)
    try:
        params = json.loads(body or b"{}")
    except ValueError:
        return _json({"error": "the body is not JSON"}, 400)
    kind = params.get("kind") if params.get("kind") in ("cube", "block") else None
    if url.path == "/pick/stop":
        picker.stop()
        interrupt()
        answer = {"ok": True}
    elif url.path == "/pick/start":
        answer = picker.start("pick_once", kind=kind, at=params.get("at"), drive=bool(params.get("drive")))
    elif url.path == "/pick/cleanup":
        answer = picker.start("clean_up", kind=kind, drive=bool(params.get("drive")))
    elif url.path == "/pick/home":
        answer = picker.start("home")
    elif url.path == "/pick/scan" and "scan" in picker.jobs():
        answer = picker.start("scan", **{k: float(params[k]) for k in ("radius", "stops") if k in params})
    elif url.path == "/pick/teach" and "teach" in picker.jobs():
        answer = picker.start("teach", name=str(params.get("name") or ""))
    elif url.path == "/pick/find" and "find" in picker.jobs():
        answer = picker.start("find")
    elif url.path == "/pick/named" and "pick_named" in picker.jobs():
        answer = picker.start("pick_named", name=str(params.get("name") or ""))
    elif url.path == "/pick/settings":
        try:
            answer = {"settings": picker.set_settings(params)}
        except ValueError as exc:
            answer = {"error": str(exc)}
    else:
        return _json({"error": f"no such endpoint: POST {url.path}"}, 404)
    return _json(answer, 409 if "error" in answer else 200)


def respond(handler, status: int, kind: str, data: bytes) -> None:
    """Send one reply() on a BaseHTTPRequestHandler: never cached, readable from the console's page."""
    handler.send_response(status)
    handler.send_header("Content-Type", kind)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(data)


def handle(handler, picker, interrupt, method: str) -> bool:
    """reply() for `handler`'s request, sent; False when the path is not the module's (the body,
    for a POST, not read then)."""
    path = urlparse(handler.path).path
    if not (path.startswith("/pick/") and picker is not None or path == "/api/metrics"):
        return False
    body = b""
    if method == "POST":
        length = int(handler.headers.get("Content-Length") or 0)
        body = handler.rfile.read(length) if length else b""
    answer = reply(picker, interrupt, method, handler.path, body)
    if answer is None:
        return False
    respond(handler, *answer)
    return True


def arm_caller(the_bridge, picker):
    """The job's way to the arm: the bridge's own calls (one at a time, on its lock), with the arm's
    interrupt also asking the job -- a stop between two calls would otherwise be wiped by the next
    call, which clears the bridge's interrupt before it starts. `picker()`: the job, once made."""
    def call(method, *args, **kwargs):
        if the_bridge.arm is None:
            the_bridge.connect()
        arm = the_bridge.arm
        if arm is not None and not getattr(arm.interrupt, "asks_picker", False):
            plain = arm.interrupt

            def interrupt() -> bool:
                return picker().stopping() or bool(plain is not None and plain())
            interrupt.asks_picker = True
            arm.interrupt = interrupt
        return the_bridge.call(method, list(args), kwargs)
    return call
