"""The drive page's logic and web API, without ROS: drive_server.py feeds it the robot's scans,
map, battery and pose; fake_m3.py feeds it a simulated robot's. It turns the page's commands --
by hand, or a slow loop round the arena -- into velocity commands through `send(vx, vy, wz)`.

Both modes are held to the same limits: at most 0.15 m/s and 0.6 rad/s; no motion towards a
lidar return in the robot's path (22 cm either side of its centre line) nearer than 0.35 m from
its centre, slowing from 0.6 m; no turning with anything within 0.25 m; nothing at all without a
scan from the last 0.5 s. A manual command lapses 0.4 s after the page stops repeating it, so a
closed tab stops the robot, and a hand on the controls ends the loop.

    GET  /                  the drive page (drive.html)
    GET  /api/state         mode, pose, command, nearest return ahead, battery, loop, fence
    GET  /api/map.png       the map with the fence fit, the robot, its scan and the loop
    POST /api/drive         {"vx", "vy", "wz"}: -1..1 of full speed, repeated while held
    POST /api/stop
    POST /api/auto          {"on": true | false}
"""

import json
import math
import pathlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import fence_fit
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MAX_V, MAX_W, AUTO_V = 0.15, 0.6, 0.08            # m/s, rad/s, m/s
STOP_AT, SLOW_FROM, TURN_CLEAR = 0.35, 0.6, 0.25  # m from the robot's centre
HALF_PATH = 0.22                                  # half the width swept, m (chassis ~0.3 m)
DEADMAN, SCAN_STALE = 0.4, 0.5                    # s
INSET = 0.45                                      # the loop's distance from the walls, m
BLOB_MARGIN = 0.33      # the loop keeps a blob's edge this far from the robot's centre, m:
                        # beyond HALF_PATH passing by, so a landmark never blocks the loop
WALL_CLEAR = 0.30                                 # and the walls this far, m
STUCK = 4.0                                       # s blocked before a loop point is given up


def limit(vx, vy, wz, scan_xy):
    """The command scaled down for what the lidar sees (points in base_link), and the distance
    to the nearest return in the robot's path."""
    ranges = np.hypot(scan_xy[:, 0], scan_xy[:, 1])
    if len(ranges) and ranges.min() < TURN_CLEAR:
        wz = 0.0
    speed = math.hypot(vx, vy)
    if speed < 1e-3:
        return 0.0, 0.0, wz, math.inf
    d = np.array([vx, vy]) / speed
    along = scan_xy @ d
    across = np.abs(scan_xy @ np.array([-d[1], d[0]]))
    ahead = (along > 0) & (across < HALF_PATH)
    nearest = float(along[ahead].min()) if ahead.any() else math.inf
    k = min(max((nearest - STOP_AT) / (SLOW_FROM - STOP_AT), 0.0), 1.0)
    return vx * k, vy * k, wz, nearest


def detour(route, blobs, length, width, start):
    """The route (arena frame) from `start`, the robot, kept clear of every blob -- (centre,
    radius) pairs; a circle of radius + BLOB_MARGIN around each -- and WALL_CLEAR from the
    walls: a point inside a circle is pushed out of it (or dropped, when a wall is in the way),
    and a leg through a circle -- the first one, from the robot, too -- gets a point on its
    edge, on the leg's side, or on the other when that side is against a wall. The robot's own
    position is not in the result."""
    circles = [(np.asarray(c, float), r + BLOB_MARGIN) for c, r in blobs]

    def inside(p):
        return np.clip(p, WALL_CLEAR, [length - WALL_CLEAR, width - WALL_CLEAR])

    def edge(c, clear, towards):
        return inside(c + (towards - c) / max(np.linalg.norm(towards - c), 1e-6) * clear)

    def clear_of(p, c, clear):
        return np.linalg.norm(p - c) > 0.95 * clear

    route = [inside(p) for p in route]
    for c, clear in circles:
        pushed = [p if clear_of(p, c, clear) else edge(c, clear, p) for p in route]
        route = [p for p in pushed if clear_of(p, c, clear)]
    start = np.asarray(start, float)
    for c, clear in circles:   # a robot inside a circle first backs straight out of it
        if not clear_of(start, c, clear / 0.95):
            route.insert(0, edge(c, clear, start))

    def via(p, q, depth=0):
        """The points to put between p and q so that no leg cuts into a circle: one on the edge,
        then the same for the two new legs (a chord between edge points cuts in too)."""
        d = q - p
        for c, clear in sorted(circles, key=lambda cc: (cc[0] - p) @ d):   # in the leg's order
            t = float(np.clip((c - p) @ d / max(d @ d, 1e-9), 0.0, 1.0))
            closest = p + t * d
            gap = np.linalg.norm(closest - c)
            if 0.0 < t < 1.0 and not clear_of(closest, c, clear) and depth < 6:
                away = (closest - c) / gap if gap > 1e-6 else fence_fit.rot90(d) / np.linalg.norm(d)
                for side in (away, -away):
                    point = inside(c + side * clear)
                    if clear_of(point, c, clear):
                        return via(p, point, depth + 1) + [point] + via(point, q, depth + 1)
                return []   # no way round inside the walls: the lidar limit stops it there
        return []

    out = [start]
    for q in route:
        out += via(out[-1], q) + [q]
    return out[1:]


class Driver:
    """What the page does, fed by a robot: `pose()` gives (x, y, yaw) in the map frame or None,
    `send(vx, vy, wz)` commands the base once; call set_scan / set_map as they come, set
    `battery`, and tick() at 20 Hz."""

    def __init__(self, pose, send, clock=time.monotonic):
        self.pose, self.send, self.clock = pose, send, clock
        self.lock = threading.Lock()
        self.scan_xy, self.scan_time = None, -math.inf
        self.map = None                  # (grid as map_saver writes it, resolution, origin)
        self.fit_cache = (None, None)    # (the map it was fitted to, the fit)
        self.battery = None
        self.manual, self.manual_time = (0.0, 0.0, 0.0), -math.inf
        self.auto = None                 # the loop: waypoints, index, heading, blocked_since
        self.mode, self.note = "idle", ""
        self.cmd, self.nearest, self.moving = (0.0, 0.0, 0.0), math.inf, False

    # --- inputs -------------------------------------------------------------------------------

    def set_scan(self, xy):
        """The latest scan, as points (N x 2) in base_link."""
        with self.lock:
            self.scan_xy, self.scan_time = xy, self.clock()

    def set_map(self, grid, resolution, origin):
        """The latest map, as map_saver writes it (0 occupied, 254 free, 205 unknown; row 0 at
        the top, +y)."""
        with self.lock:
            self.map = (grid, resolution, origin)

    def fit(self):
        """The fence fitted to the latest map (once per map), or None."""
        with self.lock:
            m = self.map
            cached_map, cached_fit = self.fit_cache
        if m is None or m is cached_map:
            return m, cached_fit
        try:
            result = fence_fit.fit(*m)
        except SystemExit:   # fence_fit's "not enough of the fence yet"
            result = None
        with self.lock:
            self.fit_cache = (m, result)
        return m, result

    # --- commands from the page ---------------------------------------------------------------

    def drive(self, body):
        def unit(key):
            return max(-1.0, min(1.0, float(body.get(key, 0.0))))
        with self.lock:
            self.auto = None
            self.manual = (unit("vx") * MAX_V, unit("vy") * MAX_V, unit("wz") * MAX_W)
            self.manual_time = self.clock()

    def stop(self, note="stopped"):
        with self.lock:
            self.auto, self.manual_time, self.note = None, -math.inf, note

    def start_auto(self):
        """Plan the loop: 45 cm inside the fitted fence, from the corner nearest the robot, round
        and back, then once along the middle, clear of the blobs (the landmarks)."""
        m, result = self.fit()
        pose = self.pose()
        if m is None or pose is None:
            return "no map or no pose yet"
        if result is None:
            return "no fence fit yet: drive by hand until two facing walls are in the map"
        o = np.array(result["arena_frame"]["origin_map"])
        ex = np.array(result["arena_frame"]["x_axis_map"])
        ey = fence_fit.rot90(ex)
        length, width = result["fence"]["size_m"]
        # planned in the arena frame, driven in the map frame
        corners = [np.array(c) for c in ((INSET, INSET), (length - INSET, INSET),
                                         (length - INSET, width - INSET), (INSET, width - INSET))]
        here = np.array([(np.array(pose[:2]) - o) @ ex, (np.array(pose[:2]) - o) @ ey])
        k = int(np.argmin([np.linalg.norm(c - here) for c in corners]))
        route = corners[k:] + corners[:k] + [corners[k]]
        middle = [np.array((INSET, width / 2)), np.array((length - INSET, width / 2))]
        if np.linalg.norm(middle[0] - route[-1]) > np.linalg.norm(middle[1] - route[-1]):
            middle.reverse()
        blobs = [(b["arena_xy_m"], max(b["size_cm"]) / 200) for b in result["blobs"]]
        route = [o + p[0] * ex + p[1] * ey
                 for p in detour(route + middle, blobs, length, width, here)]
        with self.lock:
            self.manual_time = -math.inf
            self.auto = {"waypoints": [(float(p[0]), float(p[1])) for p in route], "i": 0,
                         "heading": pose[2], "blocked_since": None, "skips": 0}
            self.note = f"auto: {len(route)} points"
        return None

    # --- the control loop, 20 Hz ----------------------------------------------------------------

    def tick(self):
        now = self.clock()
        with self.lock:
            scan = self.scan_xy if now - self.scan_time < SCAN_STALE else None
            manual = self.manual if now - self.manual_time < DEADMAN else None
            auto = self.auto
        want, mode = None, "idle"
        if manual is not None:
            want, mode = manual, "manual"
        elif auto is not None:
            want, mode = self.auto_step(auto), "auto"
        self.mode = mode if want is not None else "idle"
        if want is None:
            if self.moving:   # a few zeros, then the base is the gamepad's again
                for _ in range(3):
                    self.send(0.0, 0.0, 0.0)
                self.moving = False
            self.cmd, self.nearest = (0.0, 0.0, 0.0), math.inf
            return
        if scan is None:
            vx = vy = wz = 0.0
            self.nearest, self.note = math.inf, "no lidar scan: holding still"
        else:
            vx, vy, wz, self.nearest = limit(*want, scan)
        if mode == "auto":
            self.watch_blocked(auto, want, (vx, vy), now)
        self.send(vx, vy, wz)
        self.cmd, self.moving = (vx, vy, wz), True

    def auto_step(self, auto):
        """The loop's next command: straight at the next point at AUTO_V, keeping the heading
        it started with (the base is omnidirectional). None when the loop is done."""
        pose = self.pose()
        if pose is None:
            return (0.0, 0.0, 0.0)
        x, y, yaw = pose
        while True:
            if auto["i"] >= len(auto["waypoints"]):
                self.stop("auto: loop done")
                return None
            dx, dy = auto["waypoints"][auto["i"]][0] - x, auto["waypoints"][auto["i"]][1] - y
            dist = math.hypot(dx, dy)
            if dist > 0.05:
                break
            auto["i"] += 1
            auto["blocked_since"], auto["skips"] = None, 0
        speed = min(AUTO_V, 0.8 * dist)
        c, s = math.cos(yaw), math.sin(yaw)
        turn = math.atan2(math.sin(auto["heading"] - yaw), math.cos(auto["heading"] - yaw))
        return (speed * (c * dx + s * dy) / dist, speed * (-s * dx + c * dy) / dist,
                max(-0.3, min(0.3, turn)))

    def watch_blocked(self, auto, want, got, now):
        """Give a loop point up after STUCK s held still by the lidar; stop after two in a row."""
        if math.hypot(*want[:2]) < 0.01 or math.hypot(*got) > 0.005:
            auto["blocked_since"] = None
        elif auto["blocked_since"] is None:
            auto["blocked_since"] = now
        elif now - auto["blocked_since"] > STUCK:
            auto["i"] += 1
            auto["skips"] += 1
            auto["blocked_since"] = None
            if auto["skips"] >= 2:
                self.stop("auto stopped: blocked twice in a row -- drive it by hand")

    # --- what the page shows ------------------------------------------------------------------

    def state(self):
        pose = self.pose()
        _m, result = self.fit()
        with self.lock:
            auto, scan_age = self.auto, self.clock() - self.scan_time
        return {
            "mode": self.mode, "note": self.note, "battery_v": self.battery,
            "pose": None if pose is None else [round(pose[0], 3), round(pose[1], 3),
                                               round(math.degrees(pose[2]), 1)],
            "cmd": [round(v, 3) for v in self.cmd],
            "nearest_ahead_m": None if math.isinf(self.nearest) else round(self.nearest, 2),
            "scan_age_s": round(scan_age, 2) if math.isfinite(scan_age) else None,
            "auto": None if auto is None else [auto["i"], len(auto["waypoints"])],
            "fence": None if result is None else {"size_m": result["fence"]["size_m"],
                                                  "complete": result["fence"]["complete"],
                                                  "blobs": len(result["blobs"])},
        }

    def picture(self):
        """The map with the fence fit, the robot, its current scan and the loop drawn on it."""
        m, result = self.fit()
        if m is None:
            return None
        img, P = fence_fit.render(*m, result)
        pose = self.pose()
        with self.lock:
            scan, auto = self.scan_xy, self.auto
        if auto is not None:
            pts = [P(p) for p in auto["waypoints"]]
            cv2.polylines(img, [np.array(pts, np.int32)], False, (200, 120, 0), 1)
            for i, p in enumerate(pts):
                cv2.circle(img, p, 4, (200, 0, 200) if i == auto["i"] else (200, 120, 0), -1)
        if pose is not None:
            x, y, yaw = pose
            c, s = math.cos(yaw), math.sin(yaw)
            if scan is not None:
                for p in scan[::2] @ np.array([[c, s], [-s, c]]) + [x, y]:
                    cv2.circle(img, P(p), 1, (0, 0, 255), -1)
            radius = int(0.17 * (P((x + 1, y))[0] - P((x, y))[0]))   # the chassis, ~17 cm
            cv2.circle(img, P((x, y)), max(radius, 4), (0, 140, 255), 2)
            cv2.line(img, P((x, y)), P((x + 0.25 * c, y + 0.25 * s)), (0, 140, 255), 2)
        return cv2.imencode(".png", img)[1].tobytes()


def serve(driver, port, host="0.0.0.0"):
    """The page and the API (above) on `port`, until the process ends. CORS is open: the
    console on the laptop calls it from another origin."""
    page = (HERE / "drive.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, body, kind="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self.reply(200, page, "text/html; charset=utf-8")
            elif path == "/api/state":
                self.reply(200, json.dumps(driver.state()).encode())
            elif path == "/api/map.png":
                png = driver.picture()
                if png is None:
                    self.reply(503, b'{"error": "no map yet"}')
                else:
                    self.reply(200, png, "image/png")
            else:
                self.reply(404, b'{"error": "not found"}')

        def do_POST(self):
            size = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(size) or b"{}")
            path, error = self.path.split("?")[0], None
            if path == "/api/drive":
                driver.drive(body)
            elif path == "/api/stop":
                driver.stop()
            elif path == "/api/auto":
                error = driver.start_auto() if body.get("on") else driver.stop()
            else:
                return self.reply(404, b'{"error": "not found"}')
            self.reply(200, json.dumps({"ok": error is None, "error": error}).encode())

    ThreadingHTTPServer((host, port), Handler).serve_forever()
