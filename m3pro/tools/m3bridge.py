#!/usr/bin/env python3
"""The M3 Pro's arm and camera over HTTP: the X3's bridge contract (roboarm/web/bridge.py) on port
8761, so that our ROS-free code drives the M3 as it drives the X3 -- plus GET /rgbd, a colour
frame with its aligned depth, for the depth detector and for recording test sets.

On the robot, from ~/roboarm (python tools/sync_robot.py puts it there), with ROS 2 and Yahboom's
workspaces sourced (arm_msgs) -- tools/m3bridge.sh does both:

    python3 tools/m3bridge.py [--camera]     # --camera: start the DCW2's driver too (depth
                                             # aligned to colour); leave it off when the mapping
                                             # stack already runs the camera

The arm: roboarm.m3arm.M3Arm, publishing on /arm6_joints (arm_msgs/ArmJoints) to the micro-ROS
board, the battery from /battery. The camera: /camera/color/image_raw into the MJPEG stream, and
/camera/depth/image_raw (aligned, 16-bit mm) with /camera/color/camera_info for /rgbd:

    GET /rgbd  {"stamp", "depth_lag_s", "joints" (believed; null before engage), "K" (3 x 3),
                "color_jpeg", "depth_png" (base64; the PNG is 16-bit, millimetres)}

And the pick itself (roboarm.m3pick.Picker), here beside the arm and the camera so no picture
crosses the network -- what the console's M3 arm card drives and watches:

    POST /pick/start    {"kind": "cube"|"block"|null, "at": [x, y]|null, "drive": bool}   one pick, then home
    POST /pick/cleanup  {"kind": ..., "drive": bool}   every kit thing in reach, one by one
                        "drive": the base may drive up to what is out of reach -- forward and
                        sideways, 10 cm at most a go at 5 cm/s, on /odom_raw (off unless asked)
    POST /pick/home     POST /pick/stop (the arm holds where it is, at once)
    GET  /pick/state    the job, its step, the counts, the last attempt, what the last look found
    GET  /pick/log?since=N    the narrated steps after N
    GET  /pick/frame.jpg      the last look, marked
    GET  /pick/settings       the speeds and times the console may change: [{key, label, unit, min,
                              max, step, default, value}]; POST {key: value, ...} changes them
    GET  /api/metrics         where the time went (roboarm.metrics: one run a job)

No authentication: the lab network only, as the X3's bridge.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from roboarm import camera, m3arm, m3pick, metrics
from roboarm.m3arm import ArmError
from roboarm.web import bridge
from roboarm.web.bridge import Server, send_json

HOST, PORT = "0.0.0.0", 8761
OUT = Path(__file__).resolve().parents[1] / "out" / "pick"


class Frames:
    """The newest colour frame, aligned depth and intrinsics, as they come from ROS."""

    def __init__(self):
        self.lock = threading.Lock()
        self.color = self.color_stamp = self.depth = self.depth_stamp = self.K = None

    def arrays(self):
        """(colour BGR, depth mm, K 3 x 3) as they are now, copies; an ArmError while one is missing."""
        with self.lock:
            color, depth, K = self.color, self.depth, self.K
        if color is None or depth is None or K is None:
            raise ArmError("no picture from the camera yet: was the bridge started with --camera?")
        return color.copy(), depth.copy(), np.asarray(K, float).reshape(3, 3)

    def snapshot(self, joints) -> dict:
        with self.lock:
            color, depth, K = self.color, self.depth, self.K
            color_stamp, depth_stamp = self.color_stamp, self.depth_stamp
        if color is None or depth is None or K is None:
            missing = [n for n, v in (("colour", color), ("depth", depth), ("camera info", K)) if v is None]
            return {"error": f"no {', '.join(missing)} from the camera yet"}
        return {
            "stamp": color_stamp, "depth_lag_s": round(color_stamp - depth_stamp, 4),
            "joints": joints, "K": np.asarray(K, float).reshape(3, 3).tolist(),
            "color_jpeg": base64.b64encode(cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 92])[1]).decode(),
            "depth_png": base64.b64encode(cv2.imencode(".png", depth)[1]).decode(),
        }


def blank_frame() -> bytes:
    image = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(image, "no look yet", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", image)[1].tobytes()


def make_handler(the_bridge: bridge.Bridge, frames: Frames, picker: m3pick.Picker | None = None):
    blank = blank_frame()

    class Handler(bridge.Handler):
        def send_bytes(self, data: bytes, kind: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            url = urlparse(self.path)
            if picker is None or not url.path.startswith("/pick/"):
                super().do_POST()
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            except ValueError:
                send_json(self, {"error": "the body is not JSON"}, 400)
                return
            kind = body.get("kind") if body.get("kind") in ("cube", "block") else None
            if url.path == "/pick/stop":
                picker.stop()
                the_bridge.interrupt()
                reply = {"ok": True}
            elif url.path == "/pick/start":
                reply = picker.start("pick_once", kind=kind, at=body.get("at"), drive=bool(body.get("drive")))
            elif url.path == "/pick/cleanup":
                reply = picker.start("clean_up", kind=kind, drive=bool(body.get("drive")))
            elif url.path == "/pick/home":
                reply = picker.start("home")
            elif url.path == "/pick/settings":
                try:
                    reply = {"settings": picker.set_settings(body)}
                except ValueError as e:
                    reply = {"error": str(e)}
            else:
                send_json(self, {"error": f"no {url.path}"}, 404)
                return
            send_json(self, reply, 409 if "error" in reply else 200)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if picker is not None and url.path == "/pick/state":
                send_json(self, picker.status())
                return
            if picker is not None and url.path == "/pick/log":
                since = parse_qs(url.query).get("since", ["0"])[0]
                send_json(self, {"lines": picker.log(int(since) if since.isdigit() else 0)})
                return
            if picker is not None and url.path == "/pick/settings":
                send_json(self, {"settings": picker.settings()})
                return
            if picker is not None and url.path == "/pick/frame.jpg":
                self.send_bytes(picker.frame or blank, "image/jpeg")
                return
            if url.path == "/api/metrics":
                send_json(self, metrics.snapshot(events="events=0" not in url.query))
                return
            if url.path == "/rgbd":
                arm = the_bridge.arm
                try:
                    joints = arm.read() if arm is not None else None
                except ArmError:
                    joints = None   # not engaged yet: the pose is unknown
                reply = frames.snapshot(None if joints is None else {str(k): v for k, v in joints.items()})
                send_json(self, reply, 503 if "error" in reply else 200)
                return
            super().do_GET()

    Handler.bridge = the_bridge
    return Handler


def make_node(frames: Frames, stream: camera.Stream):
    """The ROS side: /arm6_joints out; /battery and the camera in."""
    from arm_msgs.msg import ArmJoints
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import Float32

    class M3Node(Node):
        def __init__(self):
            super().__init__("m3bridge")
            self.pub = self.create_publisher(ArmJoints, "/arm6_joints", 10)
            self.wheels = self.create_publisher(Twist, "/cmd_vel", 10)
            self.odom = None   # (x, y, yaw) of base_footprint in odom_raw, the board's wheel odometry
            self.create_subscription(Odometry, "/odom_raw", self.on_odom, 10)
            self.volts = None
            self.create_subscription(Float32, "/battery", self.on_battery, 10)
            self.create_subscription(Image, "/camera/color/image_raw", self.on_color, qos_profile_sensor_data)
            self.create_subscription(Image, "/camera/depth/image_raw", self.on_depth, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, "/camera/color/camera_info", self.on_info, qos_profile_sensor_data)

        def on_battery(self, msg):
            self.volts = float(msg.data)

        def on_odom(self, msg):
            q = msg.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

        def wheel(self, vx, vy):
            msg = Twist()
            msg.linear.x, msg.linear.y = float(vx), float(vy)
            self.wheels.publish(msg)

        @staticmethod
        def stamp(msg) -> float:
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        def on_color(self, msg):
            img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else img.copy()
            with frames.lock:
                frames.color, frames.color_stamp = img, self.stamp(msg)
            stream.publish(img)

        def on_depth(self, msg):
            depth = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width).copy()
            with frames.lock:
                frames.depth, frames.depth_stamp = depth, self.stamp(msg)

        def on_info(self, msg):
            with frames.lock:
                frames.K = list(msg.k)

        def send(self, pose, ms):
            msg = ArmJoints()
            msg.joint1, msg.joint2, msg.joint3, msg.joint4, msg.joint5, msg.joint6 = (int(pose[j]) for j in m3arm.JOINTS)
            msg.time = int(ms)
            self.pub.publish(msg)

        def board_listening(self, timeout=5.0) -> bool:
            deadline = time.monotonic() + timeout
            while self.pub.get_subscription_count() == 0:
                if time.monotonic() > deadline:
                    return False
                time.sleep(0.1)
            return True

    return M3Node()


def arm_caller(the_bridge: bridge.Bridge, picker):
    """The picker's way to the arm: the bridge's own calls (one at a time, on its lock), with the
    arm's interrupt also asking the picker -- a stop between two calls would otherwise be wiped by
    the next call, which clears the bridge's interrupt before it starts."""
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


def snapshot(the_bridge: bridge.Bridge, frames: Frames):
    color, depth, K = frames.arrays()
    return color, depth, K, the_bridge.call("read", [], {})


MAX_DRIVE = 0.15   # m one drive() may go, each way, whatever the settings say
MAX_V = 0.10       # m/s, likewise


def drive(node, dx: float, dy: float, speed: float = 0.05, stopping=lambda: False) -> tuple[float, float]:
    """Drive the base (dx forward, dy left, m, in its own frame as it starts) on the board's wheel
    odometry, 20 commands a second at `speed` m/s, slowing over the last 2 cm; stops at once when
    `stopping()`. What the wheels say it went."""
    speed = min(max(float(speed), 0.01), MAX_V)
    if abs(dx) > MAX_DRIVE or abs(dy) > MAX_DRIVE:
        raise ArmError(f"drive ({dx:+.3f}, {dy:+.3f}) m: at most {MAX_DRIVE} m each way in one go")
    if node.odom is None:
        raise ArmError("no wheel odometry (/odom_raw) from the board: not driving blind")
    x0, y0, yaw0 = node.odom
    c, s = math.cos(yaw0), math.sin(yaw0)
    goal = np.array([dx, dy])
    deadline = time.monotonic() + 2.0 * float(np.linalg.norm(goal)) / speed + 1.0
    done = np.zeros(2)
    try:
        while time.monotonic() < deadline and not stopping():
            x, y, _ = node.odom
            done = np.array([c * (x - x0) + s * (y - y0), -s * (x - x0) + c * (y - y0)])
            left = goal - done
            far = float(np.linalg.norm(left))
            if far < 0.004:
                break
            v = min(speed, 0.01 + 2.0 * far) * left / far
            node.wheel(v[0], v[1])
            time.sleep(0.05)
    finally:
        for _ in range(3):
            node.wheel(0.0, 0.0)
            time.sleep(0.02)
    return float(done[0]), float(done[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", action="store_true", help="start the DCW2's driver too")
    args = parser.parse_args()

    import rclpy
    from rclpy.signals import SignalHandlerOptions

    driver = None
    if args.camera:
        # in the bridge's own process group: m3bridge.sh stop signals both
        driver = subprocess.Popen(["ros2", "launch", "orbbec_camera", "dabai_dcw2.launch.py",
                                   "depth_registration:=true"])
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    frames, stream = Frames(), camera.Stream()
    node = make_node(frames, stream)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    def make_arm():
        if not node.board_listening():
            raise ArmError("nothing listens on /arm6_joints: is the micro-ROS agent up?")
        return m3arm.M3Arm(node.send, lambda: node.volts)

    the_bridge = bridge.Bridge(make_arm, lambda: stream)
    picker = m3pick.Picker(arm_caller(the_bridge, lambda: picker), lambda: snapshot(the_bridge, frames), OUT,
                           drive=lambda dx, dy, v: drive(node, dx, dy, v, lambda: picker.stopping()),
                           battery=lambda: node.volts)
    metrics.REGISTRY.log_path = OUT / "metrics.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    server = Server((HOST, PORT), make_handler(the_bridge, frames, picker))
    the_bridge.connect()
    state = "arm ready (engage before the first move)" if the_bridge.arm else f"arm NOT connected: {the_bridge.arm_error}"
    print(f"m3bridge on port {server.server_port}: {state}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        the_bridge.close()   # the arm holds where it is, if it was ever engaged
        node.destroy_node()
        rclpy.try_shutdown()
        if driver is not None:
            driver.terminate()
        print("m3bridge stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
