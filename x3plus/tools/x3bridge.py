#!/usr/bin/env python3
"""The X3 Plus's arm, wrist camera and jobs over HTTP on port 8761 -- the M3's surface
(m3pro/tools/m3bridge.py), for the console's arm card: the same /pick/* (roboshared.serve), the
same jobs (roboshared.task, the X3's body roboarm/x3pick.py).

On the robot (robot/roboarm-x3bridge.service runs it at boot):

    .venv/bin/python tools/x3bridge.py                 # the real arm and camera
    python tools/x3bridge.py --sim --port 8180         # anywhere: the simulator (console --fake)

    GET  /health /stream.mjpg /calibration /vision/health   POST /arm/call /arm/interrupt /arm/connect
                        /vision/detect /vision/raised        (roboarm.web.bridge: the arm for tools)
    GET  /snapshot.jpg  the newest frame, as the camera gives it (undistorted)
    the jobs: POST /pick/start /pick/cleanup /pick/home /pick/stop, GET /pick/state /pick/log
              /pick/frame.jpg /pick/settings (POST to change), GET /api/metrics -- roboshared.serve

It replaces the X3's own panel (tools/webapp.py, port 8080), retired on the user's word 2026-10-07:
one app, the console, for both robots. No authentication: the lab network only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from roboshared import metrics, serve

from roboarm import camera, x3pick
from roboarm import config as cfg
from roboarm import workspace as ws
from roboarm.web import bridge

ROOT = Path(__file__).resolve().parents[1]
BATTERY_EVERY_S = 10.0


def make_handler(the_bridge: bridge.Bridge, picker: x3pick.Picker):
    class Handler(bridge.Handler):
        def do_POST(self) -> None:
            if not serve.handle(self, picker, the_bridge.interrupt, "POST"):
                super().do_POST()

        def do_GET(self) -> None:
            if serve.handle(self, picker, the_bridge.interrupt, "GET"):
                return
            if urlparse(self.path).path == "/snapshot.jpg":
                frame, _seen = the_bridge.stream().latest()
                if frame is None:
                    serve.respond(self, 503, "application/json", b'{"error": "no frame from the camera yet"}')
                    return
                serve.respond(self, 200, "image/jpeg", cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
                return
            super().do_GET()

    Handler.bridge = the_bridge
    return Handler


def calibration_loader(load):
    """The calibrated looks, or [] (the job then says so) -- read at every job's look, so a new
    calibration counts without a restart."""
    def calibrated():
        try:
            return load()
        except (FileNotFoundError, ValueError, KeyError):
            return []
    return calibrated


def battery_watch(the_bridge: bridge.Bridge):
    """The board's voltage, read every BATTERY_EVERY_S on a thread of its own (through the bridge's
    lock), so /pick/state never waits for the arm."""
    volts = [None]

    def run():
        while True:
            with contextlib.suppress(Exception):   # no arm yet: the last reading stands
                volts[0] = float(the_bridge.call("battery", [], {}))
            time.sleep(BATTERY_EVERY_S)
    threading.Thread(target=run, daemon=True).start()
    return lambda: volts[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8761)
    parser.add_argument("--sim", action="store_true", help="the simulator, no hardware")
    parser.add_argument("--out", default=str(ROOT / "out" / "pick"), help="where the jobs keep their looks ('' for none)")
    args = parser.parse_args()
    out = Path(args.out) if args.out else None

    if args.sim:
        from roboarm.web import sim
        world = sim.World()
        the_bridge = bridge.Bridge(arm_factory=lambda: sim.SimArm(world, time_scale=1.0),
                                   stream=lambda: sim.SimStream(the_bridge.arm, fps=8),
                                   calibration_path=Path("sim"))
        calibration, detector_url, lens_fit = sim.calibration, None, None
    else:
        from roboarm.arm import Arm
        the_bridge = bridge.Bridge(arm_factory=Arm, stream=lambda: camera.Stream(cfg.WRIST_CAM),
                                   calibration_path=ws.CALIBRATION_PATH)
        calibration, detector_url = calibration_loader(ws.load_all), cfg.DETECTOR_URL
        fit = ROOT / "data" / "cameras" / cfg.WRIST_CAMERA / "lens_on_arm.json"
        lens_fit = json.loads(fit.read_text()) if fit.exists() else None

    the_bridge.connect()
    picker_box: list = []
    picker = x3pick.Picker(serve.arm_caller(the_bridge, lambda: picker_box[0]), the_bridge.stream, calibration,
                           out=out, battery=battery_watch(the_bridge), detector_url=detector_url,
                           lens_fit=lens_fit, apart=not args.sim)
    picker_box.append(picker)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        metrics.REGISTRY.log_path = out / "metrics.jsonl"
    server = bridge.Server((args.host, args.port), make_handler(the_bridge, picker))
    looks = [name for _m, _p, name in calibration()]
    state = "arm connected" if the_bridge.arm is not None else f"arm NOT connected: {the_bridge.arm_error}"
    print(f"x3bridge on port {server.server_port}: {state}; calibrated looks {looks or 'NONE'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        the_bridge.close()   # the arm left holding
        print("x3bridge stopped; arm left holding", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
