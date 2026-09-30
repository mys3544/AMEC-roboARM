"""The drive page's logic and web API, without ROS: drive_server.py feeds it the robot's scans,
map, battery and pose; fake_m3.py feeds it a simulated robot's. It turns the page's commands --
by hand, or a slow loop round the arena -- into velocity commands through `send(vx, vy, wz)`.

Both modes are held to at most 0.15 m/s and 0.6 rad/s. The loop also to the distance limits: no
motion towards a lidar return in the robot's path (22 cm either side of its centre line) nearer
than 0.35 m from its centre, slowing from 0.6 m; no turning with anything within 0.25 m; nothing
at all without a scan from the last 0.5 s. Manual driving has no distance limits -- whoever
drives judges how close (the user's call, 2026-09-30) -- but a manual command lapses 0.4 s after
the page stops repeating it, so a closed tab stops the robot, and a hand on the controls ends the
loop.

    GET  /                  the drive page (drive.html)
    GET  /api/state         mode, pose, command, nearest return ahead, battery, loop, fence,
                            motion (false in bench mode)
    GET  /api/map.png       the map with the fence fit, the robot, its scan and the loop
    POST /api/drive         {"vx", "vy", "wz"}: -1..1 of full speed, repeated while held
                            (refused in bench mode, as is the loop)
    POST /api/stop
    POST /api/auto          {"on": true | false}; {"on": true, "step": 0.5} stops and goes: it
                            pauses after every 0.5 m until {"resume": true};
                            {"on": true, "plan": "survey"}: the survey for a 3D scan
"""

import heapq
import itertools
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
WALL_CLEAR = 0.30       # and the walls this far, m (0.42 left no room between a landmark and
                        # the wall, and the planner zig-zagged: 93 points)
STUCK = 4.0                                       # s blocked before a loop point is given up
BENCH = "bench mode: motion off (arena_slam.launch.py motion:=false)"
# The survey for a 3D scan (camera tilted down): stations with a heading; at each the robot
# stands SETTLE s (the arm stops swaying), then still until DWELL s: the frames taken then are
# the sharp ones -- frames taken moving carry the board clock's 55-75 ms lag as blur. The map
# records all along (paused while moving, RTAB-Map lost its map, 2026-09-30); the export picks
# the standing frames by their odometry speed. It moves between stations faster than the loop,
# and sideways.
SURVEY_STEP, SURVEY_V = 0.35, 0.12   # m at most between stations on the way; m/s
# What the camera sees from a station, on the floor (the DCW2 35 deg down on the arm, 0.385 m up,
# 0.11 m ahead of the robot's centre; its depth 91 x 62 deg, kept to 1.2 m): from CAM_NEAR to
# CAM_FAR ahead of the camera, CAM_HALF either side. The survey picks its views by this.
CAM_AHEAD, CAM_NEAR, CAM_FAR, CAM_HALF = 0.11, 0.17, 1.14, math.radians(45)
SURVEY_CLEAR = 0.45     # m the survey keeps from the walls and the landmarks' edges: nothing
                        # nearer than hypot(STOP_AT, HALF_PATH) = 0.41 can stop it, and at 0.45
                        # the slow-down leaves it moving -- passing between a landmark and a
                        # wall with 0.33 either side, any leg angled a little towards the wall
                        # put the wall in the lidar stop (2026-09-30). Corners behind landmarks
                        # near a wall are then out of reach: the camera looks into them.
SETTLE, DWELL = 0.5, 1.8   # s at a station: standing settled from SETTLE to DWELL, 1.3 s --
                           # RTAB-Map takes a frame about once a second (DetectionRate 1 Hz)
AT, AT_YAW = 0.04, math.radians(5)                # arrived: within 4 cm and 5 degrees


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


def free_space(blobs, length, width, start, res=0.04):
    """Where the survey may put the robot's centre (arena frame, cells of `res`): SURVEY_CLEAR
    from the walls and from each blob's edge -- (centre, radius) pairs -- and reachable from
    `start` (or, from outside, from the free cell nearest it) without leaving that space.
    Returns (free, res)."""
    a = (np.arange(math.ceil(length / res)) + 0.5) * res
    b = (np.arange(math.ceil(width / res)) + 0.5) * res
    A, B = np.meshgrid(a, b, indexing="ij")
    free = ((A >= SURVEY_CLEAR - 1e-9) & (A <= length - SURVEY_CLEAR + 1e-9)
            & (B >= SURVEY_CLEAR - 1e-9) & (B <= width - SURVEY_CLEAR + 1e-9))
    for c, r in blobs:
        free &= np.hypot(A - c[0], B - c[1]) >= r + SURVEY_CLEAR - 1e-9
    cells = np.argwhere(free)
    if not len(cells):
        return free, res
    seed = tuple(cells[np.argmin(np.hypot(*((cells + 0.5) * res - np.asarray(start)).T))])
    reach, todo = np.zeros_like(free), [seed]
    reach[seed] = True
    while todo:   # flood fill, 8 neighbours
        c = todo.pop()
        for da, db in itertools.product((-1, 0, 1), repeat=2):
            n = (c[0] + da, c[1] + db)
            if 0 <= n[0] < free.shape[0] and 0 <= n[1] < free.shape[1] and free[n] and not reach[n]:
                reach[n] = True
                todo.append(n)
    return reach, res


def free_path(free, res, p, q):
    """The shortest way from p to q (m) through free cells -- A* on the grid, then cut short
    wherever a straight line stays free -- as the points to drive through, ending at q; None
    when q is not free or cannot be reached from p. A p outside the free space first goes
    straight to the nearest free cell."""
    def cell(pt):
        return (min(max(int(pt[0] / res), 0), free.shape[0] - 1),
                min(max(int(pt[1] / res), 0), free.shape[1] - 1))

    def centre(c):
        return np.array([(c[0] + 0.5) * res, (c[1] + 0.5) * res])

    def clear(u, v):   # the straight line from u to v (m) through free cells only
        n = max(1, int(np.linalg.norm(v - u) / (res / 2)))
        return all(free[cell(u + (v - u) * k / n)] for k in range(n + 1))

    goal = cell(q)
    if not free[goal]:
        return None
    lead = []
    s = cell(p)
    if not free[s]:
        cells = np.argwhere(free)
        s = tuple(cells[np.argmin(np.hypot(*(cells * res + res / 2 - np.asarray(p)).T))])
        lead = [centre(s)]
    came, cost, todo = {s: None}, {s: 0.0}, [(0.0, s)]
    while todo:
        _f, c = heapq.heappop(todo)
        if c == goal:
            break
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            n = (c[0] + da, c[1] + db)
            if not (0 <= n[0] < free.shape[0] and 0 <= n[1] < free.shape[1]) or not free[n]:
                continue
            g = cost[c] + math.hypot(da, db)
            if g < cost.get(n, math.inf):
                cost[n], came[n] = g, c
                heapq.heappush(todo, (g + math.hypot(goal[0] - n[0], goal[1] - n[1]), n))
    if goal not in came:
        return None
    cells = [goal]
    while came[cells[-1]] is not None:
        cells.append(came[cells[-1]])
    points = [centre(c) for c in cells[::-1]][1:-1] + [np.asarray(q, float)]
    out, here = [], lead[-1] if lead else np.asarray(p, float)
    i = 0
    while i < len(points):   # the farthest point still in a straight line, then from there on
        j = len(points) - 1
        while j > i and not clear(here, points[j]):
            j -= 1
        out.append(points[j])
        here, i = points[j], j + 1
    return lead + out


def seen_by(views, targets, blobs):
    """How well each view (a, b, heading) sees each target point (a, b) on the floor: 1 up to
    0.6 m from the camera, falling to 0.4 at CAM_FAR, 0 outside the view or behind a landmark
    (they stand taller than the camera sees over). Returns (quality, bearing), views x targets:
    bearing = the direction the target is seen from, for telling sides apart."""
    views = np.asarray(views, float)
    cam = views[:, :2] + CAM_AHEAD * np.stack([np.cos(views[:, 2]), np.sin(views[:, 2])], axis=1)
    d = targets[None, :, :] - cam[:, None, :]
    dist = np.hypot(d[..., 0], d[..., 1])
    bearing = np.arctan2(d[..., 1], d[..., 0])
    off = np.abs((bearing - views[:, 2:3] + np.pi) % (2 * np.pi) - np.pi)
    quality = np.where((off < CAM_HALF) & (dist > CAM_NEAR) & (dist < CAM_FAR),
                       np.clip(1 - 0.6 * (dist - 0.6) / (CAM_FAR - 0.6), 0.4, 1.0), 0.0)
    for c, r in blobs:   # hidden: the line of sight passes the landmark, the target beyond it
        c = np.asarray(c, float)
        to_c = c - cam
        t = np.clip(np.einsum("vtk,vk->vt", d, to_c) / np.maximum(dist ** 2, 1e-9), 0, 1)
        gap = np.hypot(*(cam[:, None, :] + t[..., None] * d - c).transpose(2, 0, 1))
        quality[(gap < r) & (dist > np.hypot(*to_c.T)[:, None])] = 0.0
    return quality, bearing


def survey(blobs, length, width, start):
    """Stations (a, b, heading) in the arena frame, from `start`, for a 3D scan with the camera
    looking down ahead. The views are chosen for what they see (seen_by): from every reachable
    place (free_space, 12 cm apart) in 8 headings, greedily the view that adds most -- floor not
    yet seen, or seen only from another side, closer counting more; the landmarks' faces, once
    each -- until nothing adds much: every patch of floor the camera can reach is seen, most of
    it from two sides, and pockets out of reach are looked into from their edge. Then a tour
    through the views, nearest next, by free paths, with a station every SURVEY_STEP on the way
    (facing the next view: its frames overlap for the stitching)."""
    free, res = free_space(blobs, length, width, start)
    places = [((i + 0.5) * res, (j + 0.5) * res) for i, j in np.argwhere(free) if i % 3 == 0 and j % 3 == 0]
    if not places:
        return []
    heads = np.arange(8) * math.pi / 4
    views = np.array([(a, b, h) for a, b in places for h in heads])
    step = 0.05
    A, B = np.meshgrid((np.arange(int(length / step)) + 0.5) * step,
                       (np.arange(int(width / step)) + 0.5) * step, indexing="ij")
    floor = np.ones_like(A, bool)
    for c, r in blobs:
        floor &= np.hypot(A - c[0], B - c[1]) > r + 0.02
    rims = [np.asarray(c, float) + (r + 0.02) * np.array([math.cos(t), math.sin(t)])
            for c, r in blobs for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    targets = np.concatenate([np.stack([A[floor], B[floor]], axis=1), np.reshape(rims, (-1, 2))])
    rim_of = np.concatenate([np.full(floor.sum(), -1),
                             np.repeat(np.arange(len(blobs)), 16)]) if blobs else np.full(floor.sum(), -1)
    quality, bearing = seen_by(views, targets, blobs)
    for k, (c, _r) in enumerate(blobs):   # a landmark's face is seen only from its own side
        on = rim_of == k
        outward = targets[on] - np.asarray(c, float)
        cam = views[:, :2] + CAM_AHEAD * np.stack([np.cos(views[:, 2]), np.sin(views[:, 2])], axis=1)
        facing = np.einsum("vtk,tk->vt", cam[:, None, :] - targets[None, on], outward) > 0
        quality[:, on] *= facing
    side = ((bearing + math.pi / 4) // (math.pi / 2)).astype(int) % 4   # which of 4 sides
    need = np.where(rim_of < 0, 2, 1)                                    # floor from 2 sides
    got = np.zeros((len(targets), 4), bool)
    chosen = []
    while True:
        fresh = ~got[np.arange(len(targets))[None, :], side] & (got.sum(axis=1) < need)[None, :]
        gain = (quality * fresh).sum(axis=1)
        k = int(np.argmax(gain))
        if gain[k] < 3.0:   # adds less than about three patches of floor: enough
            break
        chosen.append(tuple(views[k]))
        hit = quality[k] >= 0.4
        got[np.flatnonzero(hit), side[k, hit]] = True
    # the tour: nearest next (a turn counted as 0.2 m a radian), by free paths
    out, here, yaw = [], np.asarray(start, float), None
    left = list(chosen)
    while left:
        v = min(left, key=lambda v, here=here, yaw=yaw: math.hypot(v[0] - here[0], v[1] - here[1])
                + (0.0 if yaw is None else 0.2 * abs((v[2] - yaw + math.pi) % (2 * math.pi) - math.pi)))
        left.remove(v)
        points = free_path(free, res, here, np.array(v[:2]))
        if points is None:
            continue
        for q in points:
            n = max(1, math.ceil(np.linalg.norm(q - here) / SURVEY_STEP))
            out.extend((*(here + (q - here) * j / n), v[2]) for j in range(1, n + 1))
            here = q
        if not out or not np.allclose(out[-1][:2], v[:2]):
            out.append(v)
        yaw = v[2]
    return out


class Driver:
    """What the page does, fed by a robot: `pose()` gives (x, y, yaw) in the map frame or None,
    `send(vx, vy, wz)` commands the base once; call set_scan / set_map as they come, set
    `battery`, and tick() at 20 Hz."""

    def __init__(self, pose, send, clock=time.monotonic, motion=True):
        self.pose, self.send, self.clock = pose, send, clock
        # motion=False: bench mode -- everything shown, every command to move refused (on a
        # table the lidars cannot see the edge; the 35 cm stop is for obstacles, not drops)
        self.motion = motion
        self.lock = threading.Lock()
        self.scan_xy, self.scan_time = None, -math.inf
        self.map = None                  # (grid as map_saver writes it, resolution, origin)
        self.fit_cache = (None, None)    # (the map it was fitted to, the fit)
        self.battery = None
        self.manual, self.manual_time = (0.0, 0.0, 0.0), -math.inf
        self.auto = None                 # the loop: waypoints, index, heading, blocked_since
        self.mode, self.note = "idle", "" if motion else BENCH
        self.cmd, self.nearest, self.moving = (0.0, 0.0, 0.0), math.inf, False
        # on_record(on): told when the camera's frames are the sharp ones -- during a survey
        # only while the robot stands settled at a station; otherwise always (`recording` in
        # the state). Nothing pauses the map with it any more (drive_server.py)
        self.on_record, self.recording = None, True

    def record(self, on):
        if on != self.recording:
            self.recording = on
            if self.on_record is not None:
                self.on_record(on)

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
        """A manual command (-1..1 of full speed); an error message instead in bench mode."""
        if not self.motion:
            return BENCH

        def unit(key):
            return max(-1.0, min(1.0, float(body.get(key, 0.0))))
        with self.lock:
            self.auto = None
            self.manual = (unit("vx") * MAX_V, unit("vy") * MAX_V, unit("wz") * MAX_W)
            self.manual_time = self.clock()

    def stop(self, note="stopped"):
        with self.lock:
            self.auto, self.manual_time, self.note = None, -math.inf, note

    def start_auto(self, step=None, plan="loop"):
        """Plan the loop: 45 cm inside the fitted fence, from the corner nearest the robot, round
        and back, then once along the middle, clear of the blobs (the landmarks). Or, with plan
        "survey", the stations of survey() for a 3D scan, each held DWELL s, the map recording
        only while it stands. With `step` (m), stop and go: pause after every `step` of travel
        until resume() -- for a robot on a charging cable, which someone moves while it waits."""
        if not self.motion:
            return BENCH
        m, result = self.fit()
        pose = self.pose()
        if m is None or pose is None:
            return "no map or no pose yet"
        if result is None:
            return "no fence fit yet: drive by hand until two facing walls are in the map"
        if plan not in ("loop", "survey"):
            return f"no plan {plan!r}: loop or survey"
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
        if plan == "loop":
            waypoints = [tuple(float(v) for v in o + p[0] * ex + p[1] * ey)
                         for p in detour(route + middle, blobs, length, width, here)]
        else:   # headings turned from the arena frame into the map's
            turn = math.atan2(ex[1], ex[0])
            waypoints = [(*(float(v) for v in o + a * ex + b * ey), h + turn)
                         for a, b, h in survey(blobs, length, width, here)]
        step = None if step is None else min(max(float(step), 0.1), 5.0)
        with self.lock:
            self.manual_time = -math.inf
            self.auto = {"plan": plan, "waypoints": waypoints, "i": 0,
                         "heading": pose[2], "blocked_since": None, "skips": 0, "arrived": None,
                         "step": step, "travelled": 0.0, "from": pose[:2], "paused": False}
            self.note = (f"auto: {plan}, {len(waypoints)}"
                         f" {'points' if plan == 'loop' else 'stations'}"
                         + (f", pausing every {step:.2f} m" if step else ""))
        if plan == "survey":
            self.record(False)
        return None

    def resume(self):
        """Go on after a stop-and-go pause (the cable moved, the arena clear again)."""
        with self.lock:
            auto = self.auto
        if auto is None or not auto["paused"]:
            return "nothing to continue: the loop is not paused"
        pose = self.pose()
        if pose is not None:
            auto["from"] = pose[:2]
        auto["travelled"], auto["blocked_since"], auto["paused"] = 0.0, None, False
        self.note = f"auto: going on from point {auto['i'] + 1} of {len(auto['waypoints'])}"
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
        if want is None:
            mode = "paused" if auto is not None and auto["paused"] and manual is None else "idle"
        if mode != "auto" or self.auto is None:   # the survey over, however: record as usual
            self.record(True)
        self.mode = mode
        if want is None:
            if self.moving:   # a few zeros, then the base is the gamepad's again
                for _ in range(3):
                    self.send(0.0, 0.0, 0.0)
                self.moving = False
            self.cmd, self.nearest = (0.0, 0.0, 0.0), math.inf
            return
        if mode == "manual":   # the driver is the judge: no distance limits (the user's call,
            vx, vy, wz = want  # 2026-09-30); the speed cap and the dead-man stay
            self.nearest = math.inf if scan is None else limit(*want, scan)[3]   # shown only
        elif scan is None:
            vx = vy = wz = 0.0
            self.nearest, self.note = math.inf, "no lidar scan: holding still"
        else:
            vx, vy, wz, self.nearest = limit(*want, scan)
        if mode == "auto":
            self.watch_blocked(auto, want, (vx, vy, wz), now)
        self.send(vx, vy, wz)
        self.cmd, self.moving = (vx, vy, wz), True

    def auto_step(self, auto):
        """The next command: straight at the next point at AUTO_V -- keeping the heading the
        loop started with (the base is omnidirectional), or turning to a survey station's own
        and standing there DWELL s. None when done, or paused."""
        if auto["paused"]:
            return None
        pose = self.pose()
        if pose is None:
            return (0.0, 0.0, 0.0)
        x, y, yaw = pose
        if auto["step"]:   # stop and go: pause once `step` has been driven since the last one
            auto["travelled"] += math.hypot(x - auto["from"][0], y - auto["from"][1])
            auto["from"] = (x, y)
            if auto["travelled"] >= auto["step"]:
                auto["paused"] = True
                self.note = (f"paused after {auto['travelled']:.2f} m, at point {auto['i'] + 1}"
                             f" of {len(auto['waypoints'])}: move the cable, clear the arena,"
                             " then Continue")
                return None
        while True:
            if auto["i"] >= len(auto["waypoints"]):
                self.stop(f"auto: {auto.get('plan', 'loop')} done")
                return None
            point = auto["waypoints"][auto["i"]]
            dx, dy = point[0] - x, point[1] - y
            dist = math.hypot(dx, dy)
            station = len(point) > 2
            aim = point[2] if station else auto["heading"]
            turn = math.atan2(math.sin(aim - yaw), math.cos(aim - yaw))
            if not station:
                if dist > 0.05:
                    break
            elif dist > AT or abs(turn) > AT_YAW:
                auto["arrived"] = None
                break
            else:   # at a station: stand, then let the map record, then on to the next
                now = self.clock()
                if auto.get("arrived") is None:
                    auto["arrived"] = now
                if now - auto["arrived"] < DWELL:
                    self.record(now - auto["arrived"] >= SETTLE)
                    return (0.0, 0.0, 0.0)
                self.record(False)
                auto["arrived"] = None
            auto["i"] += 1
            auto["blocked_since"], auto["skips"] = None, 0
        top = SURVEY_V if station else AUTO_V
        speed = min(top, 0.8 * dist) if dist > (AT / 2 if station else 0.0) else 0.0
        c, s = math.cos(yaw), math.sin(yaw)
        spin = max(-0.4, min(0.4, 1.2 * turn)) if station else max(-0.3, min(0.3, turn))
        if speed == 0.0:
            return (0.0, 0.0, spin)
        return (speed * (c * dx + s * dy) / dist, speed * (-s * dx + c * dy) / dist, spin)

    def watch_blocked(self, auto, want, got, now):
        """Give a point up after STUCK s held still by the lidar -- a move, or a station's turn
        on the spot; stop after two in a row."""
        moving = math.hypot(*want[:2]) >= 0.01
        blocked = (math.hypot(*got[:2]) <= 0.005 if moving
                   else abs(want[2]) >= 0.01 and got[2] == 0.0)
        if not blocked:
            auto["blocked_since"] = None
        elif auto["blocked_since"] is None:
            auto["blocked_since"] = now
        elif now - auto["blocked_since"] > STUCK:
            pose = self.pose()
            # into the robot's log (the launch's): what stopped the loop, for next time
            print(f"auto: blocked {STUCK:.0f} s short of point {auto['i'] + 1} of"
                  f" {len(auto['waypoints'])} ({auto['waypoints'][auto['i']][0]:.2f},"
                  f" {auto['waypoints'][auto['i']][1]:.2f}): robot at"
                  f" {None if pose is None else tuple(round(v, 2) for v in pose)}, going"
                  f" ({want[0]:.3f}, {want[1]:.3f}) m/s, nearest return ahead"
                  f" {self.nearest:.2f} m", flush=True)
            auto["i"] += 1
            auto["skips"] += 1
            auto["blocked_since"], auto["arrived"] = None, None
            if auto["skips"] >= 2:
                self.stop("auto stopped: blocked twice in a row -- drive it by hand")

    # --- what the page shows ------------------------------------------------------------------

    def state(self):
        pose = self.pose()
        _m, result = self.fit()
        with self.lock:
            auto, scan_age = self.auto, self.clock() - self.scan_time
        return {
            "mode": self.mode, "note": self.note, "battery_v": self.battery, "motion": self.motion,
            "pose": None if pose is None else [round(pose[0], 3), round(pose[1], 3),
                                               round(math.degrees(pose[2]), 1)],
            "cmd": [round(v, 3) for v in self.cmd],
            "nearest_ahead_m": None if math.isinf(self.nearest) else round(self.nearest, 2),
            "scan_age_s": round(scan_age, 2) if math.isfinite(scan_age) else None,
            "auto": None if auto is None else [auto["i"], len(auto["waypoints"])],
            "plan": None if auto is None else auto.get("plan", "loop"),
            "recording": self.recording,
            "paused": auto is not None and auto["paused"],
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
            pts = [P(p[:2]) for p in auto["waypoints"]]
            cv2.polylines(img, [np.array(pts, np.int32)], False, (200, 120, 0), 1)
            for i, (p, w) in enumerate(zip(pts, auto["waypoints"])):
                colour = (200, 0, 200) if i == auto["i"] else (200, 120, 0)
                if len(w) > 2:   # a station: a short tick the way the camera will look
                    tip = (w[0] + 0.08 * math.cos(w[2]), w[1] + 0.08 * math.sin(w[2]))
                    cv2.circle(img, p, 2, colour, -1)
                    cv2.line(img, p, P(tip), colour, 1)
                else:
                    cv2.circle(img, p, 4, colour, -1)
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
                error = driver.drive(body)
            elif path == "/api/stop":
                driver.stop()
            elif path == "/api/auto":
                if body.get("resume"):
                    error = driver.resume()
                elif body.get("on"):
                    error = driver.start_auto(body.get("step"), body.get("plan", "loop"))
                else:
                    error = driver.stop()
            else:
                return self.reply(404, b'{"error": "not found"}')
            self.reply(200, json.dumps({"ok": error is None, "error": error}).encode())

    ThreadingHTTPServer((host, port), Handler).serve_forever()
