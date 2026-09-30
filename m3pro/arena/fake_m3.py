"""A pretend M3 Pro, for the drive page and the console without the robot: drive_core -- the
real page's logic, safety limits and web API -- driving a simulated base on a saved arena map,
with lidar scans ray-cast from the map's walls and landmarks.

    python m3pro/arena/fake_m3.py [--map m3pro/arena/maps/arena_v2.yaml] [--port 8195]
    -> http://localhost:8195/      (the project venv: numpy and OpenCV)

The map stays as saved (no SLAM here); the robot starts in the middle of the fence, facing
along it, and moves exactly as commanded. The simulation also refuses a move that would put a
wall inside the chassis -- which the page's own limits should never come near.
"""

import argparse
import math
import pathlib
import threading
import time

import drive_core
import fence_fit
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RATE, SCAN_HZ = 20, 7   # the page's control loop and the T-mini Plus, Hz
CHASSIS = 0.17          # m
BEAMS = np.radians(np.arange(0.0, 360.0, 0.5))


class FakeM3:
    def __init__(self, map_yaml, seed=None):
        self.grid, self.res, self.origin = fence_fit.load(map_yaml)
        self.occupied = self.grid < 50
        fit = fence_fit.fit(self.grid, self.res, self.origin)
        centre = np.array(fit["fence"]["corners_map"]).mean(axis=0)
        ex = fit["arena_frame"]["x_axis_map"]
        self.lock = threading.Lock()
        self.x, self.y, self.yaw = float(centre[0]), float(centre[1]), math.atan2(ex[1], ex[0])
        self.vel = (0.0, 0.0, 0.0)
        self.rng = np.random.default_rng(seed)
        steps = np.arange(-CHASSIS, CHASSIS + self.res, self.res)
        dx, dy = np.meshgrid(steps, steps)
        self.disc = np.stack([dx[dx ** 2 + dy ** 2 <= CHASSIS ** 2],
                              dy[dx ** 2 + dy ** 2 <= CHASSIS ** 2]], axis=1)
        self.core = drive_core.Driver(self.pose, self.command)
        self.core.set_map(self.grid, self.res, self.origin)
        self.core.battery = 12.1

    def pose(self):
        with self.lock:
            return self.x, self.y, self.yaw

    def command(self, vx, vy, wz):
        with self.lock:
            self.vel = (vx, vy, wz)

    def walls_at(self, xs, ys):
        """Occupied map cells at the points; outside the map counts as free."""
        cols = np.floor((xs - self.origin[0]) / self.res).astype(int)
        rows = self.grid.shape[0] - 1 - np.floor((ys - self.origin[1]) / self.res).astype(int)
        inside = (rows >= 0) & (rows < self.grid.shape[0]) & (cols >= 0) & (cols < self.grid.shape[1])
        out = np.zeros(np.shape(xs), bool)
        out[inside] = self.occupied[rows[inside], cols[inside]]
        return out

    def scan(self):
        """720 beams all round, to the first wall within 6 m, 5 mm of noise; in base_link."""
        x, y, yaw = self.pose()
        steps = np.arange(0.18, 6.0, self.res / 2)
        world = yaw + BEAMS
        hit = self.walls_at(x + np.outer(np.cos(world), steps), y + np.outer(np.sin(world), steps))
        seen = hit.any(axis=1)
        r = steps[hit.argmax(axis=1)][seen] + self.rng.normal(0, 0.005, seen.sum())
        return np.stack([r * np.cos(BEAMS[seen]), r * np.sin(BEAMS[seen])], axis=1)

    def step(self, dt):
        with self.lock:
            (vx, vy, wz), x, y, yaw = self.vel, self.x, self.y, self.yaw
        nx = x + (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        ny = y + (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        free = not self.walls_at(nx + self.disc[:, 0], ny + self.disc[:, 1]).any()
        with self.lock:
            if free:
                self.x, self.y = nx, ny
            self.yaw = yaw + wz * dt

    def run(self):
        last, next_scan = time.monotonic(), 0.0
        while True:
            now = time.monotonic()
            self.step(now - last)
            last = now
            if now >= next_scan:
                self.core.set_scan(self.scan())
                next_scan = now + 1 / SCAN_HZ
            self.core.tick()
            time.sleep(1 / RATE)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", default=str(HERE / "maps" / "arena_v2.yaml"))
    parser.add_argument("--port", type=int, default=8195)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    robot = FakeM3(args.map)
    threading.Thread(target=robot.run, daemon=True).start()
    print(f"fake M3: http://localhost:{args.port}/  (Ctrl-C to stop)", flush=True)
    drive_core.serve(robot.core, args.port, args.host)


if __name__ == "__main__":
    main()
