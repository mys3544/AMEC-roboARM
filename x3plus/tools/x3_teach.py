"""Teach the X3's grip by hand, from the keyboard, as the M3's (m3pro/tools/m3_teach.py): the X3
looks the table over (the bridge's /pick/look), the arm goes over a kit piece it found -- where the
aim sends it now (roboarm.x3aim) -- and the teacher steers it, in mm steps, until the open fingers sit
where the ideal grip is, and saves. Every pose sent is known, so each save is a pair: where the
camera saw the piece, and the pose that really reaches it (out/teach/points.jsonl); then
`python -m roboarm.x3aim out/teach/points.jsonl` makes x3aim's table from them.

    python tools/x3_teach.py [N] [--robot URL]   # over piece N of the look (nearest the robot 0), then the keys:

    arrows up / down     out / in (away from the robot / towards it)
    arrows left / right  left / right, as the arm sees it
    PgUp / PgDn          up / down
    [ / ]                the wrist turned -/+ 3 deg
    1 / 5                steps of 1 mm or 5 mm
    Enter                save this pose as the ideal grip of the piece nearest it
    n                    the next piece (from 20 mm over it again)
    h                    up 3 cm, then home, and quit
    q                    quit where it is

Windows console only (msvcrt). Through the X3's bridge (tools/x3bridge.py), no job running.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboarm import config as cfg
from roboarm import kinematics as kin
from roboarm import x3aim

ROBOT = "http://192.168.73.210:8761"
POINTS = ROOT / "out" / "teach" / "points.jsonl"
START_UP = 0.020    # m: the fingertips over the piece's top at the start
LOWEST = 0.004      # m: the fingertips may come this close to the table, no closer
TABLE = -cfg.TABLE_BELOW_PLATE
SPEED_DPS = 15
KEYS = {"H": (1, 0, 0), "P": (-1, 0, 0), "K": (0, 1, 0), "M": (0, -1, 0), "I": (0, 0, 1), "Q": (0, 0, -1)}


def http(robot: str, path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(robot + path, data=data, headers={"Content-Type": "application/json"},
                                     method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as reply:
            return json.loads(reply.read())
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read())


def look(robot: str) -> list[dict]:
    """The kit pieces the X3 finds on the table, nearest the robot first."""
    started = http(robot, "/pick/look", {})
    if "error" in started:
        raise SystemExit(f"the look did not start: {started['error']}")
    time.sleep(1.0)
    while (state := http(robot, "/pick/state"))["busy"]:
        time.sleep(0.5)
    if state.get("error"):
        raise SystemExit(f"the look failed: {state['error']}")
    return sorted((o for o in state["found"] if o["kind"]), key=lambda o: math.hypot(o["x"], o["y"]))


class Teach:
    def __init__(self, robot: str, piece: dict, n: int):
        self.robot, self.piece, self.n = robot, piece, n
        self.centre = np.array([piece["x"], piece["y"]], float)
        self.aim = self.centre + x3aim.offset(*self.centre)   # where the aim sends the fingers now
        self.out = self.centre / np.linalg.norm(self.centre)
        self.left = np.array([-self.out[1], self.out[0]])
        self.shift = np.zeros(3)          # m: out, left, up -- from the aim, START_UP over the top
        self.turn = 0.0                   # deg on the squared wrist
        self.pitch = None                 # the first pose's, kept: the tool goes straight up and down
        self.pose = None
        self.top = piece.get("top") or 0.030
        self.square = piece["kind"] == "cube" or "standing" in (piece.get("name") or "")

    def target(self) -> np.ndarray:
        xy = self.aim + self.out * self.shift[0] + self.left * self.shift[1]
        return np.r_[xy, TABLE + self.top + START_UP + self.shift[2]]

    def plan(self) -> dict | None:
        x, y, z = self.target()
        if z < TABLE + LOWEST:
            return None
        # the pitch kept while it works -- straight up and down -- else the one there that does, as
        # grasp.plan's own kin.solve (2026-10-15: kept from 50 mm up, it stopped the descent at 26 mm, 25 cm out)
        pose = None
        if self.pitch is not None:
            try:
                pose = kin.inverse(x, y, z, self.pitch, gripper=cfg.GRIPPER_OPEN, lowest_z=TABLE + LOWEST)
            except kin.Unreachable:
                pose = None
        if pose is None:
            try:
                pose, self.pitch = kin.solve(x, y, z, gripper=cfg.GRIPPER_OPEN, lowest_z=TABLE + LOWEST)
            except kin.Unreachable:
                return None
        roll = kin.roll_for(self.piece["yaw"], pose[1], square=self.square) + round(self.turn)
        return {**pose, 5: int(roll), cfg.GRIPPER_ID: cfg.GRIPPER_OPEN}

    def move(self) -> bool:
        pose = self.plan()
        if pose is None:
            print("   no allowed pose there -- not moved")
            return False
        reply = http(self.robot, "/arm/call", {"method": "move_to", "args": [{str(j): v for j, v in pose.items()}],
                                               "kwargs": {"speed_dps": SPEED_DPS}})
        if "error" in reply:
            print(f"   refused: {reply['error']}")
            return False
        self.pose = pose
        print(f"   out {1000 * self.shift[0]:+5.0f}  left {1000 * self.shift[1]:+5.0f} mm, fingertips "
              f"{1000 * (self.target()[2] - TABLE):3.0f} mm over the table, wrist {self.turn:+.0f}  (pose {pose})")
        return True

    def save(self, pieces: list[dict]) -> None:
        """The pose as the ideal grip of the piece nearest it -- steered over to another piece without
        `n`, it is that one's (as on the M3)."""
        point = np.array(kin.forward(self.pose))
        n = int(np.argmin([math.hypot(point[0] - p["x"], point[1] - p["y"]) for p in pieces]))
        centre = np.array([pieces[n]["x"], pieces[n]["y"]])
        out = centre / np.linalg.norm(centre)
        error = point[:2] - centre
        record = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "piece": pieces[n], "n": n,
                  "turn_deg": self.turn, "pose": self.pose, "model_grip": np.round(point, 4).tolist(),
                  "pitch": round(float(self.pitch), 1)}
        POINTS.parent.mkdir(parents=True, exist_ok=True)
        with POINTS.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"   SAVED piece {n}: seen at {centre.round(3).tolist()}, the ideal grip "
              f"{1000 * error @ out:+.0f} mm out, {1000 * error @ np.array([-out[1], out[0]]):+.0f} mm left of it")


def key() -> str:
    import msvcrt
    k = msvcrt.getwch()
    return "\xe0" + msvcrt.getwch() if k in ("\x00", "\xe0") else k


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("n", nargs="?", type=int, default=0)
    ap.add_argument("--robot", default=ROBOT)
    a = ap.parse_args()
    print("the X3 looks the table over ...")
    pieces = look(a.robot)
    if not pieces:
        raise SystemExit("no kit piece found on the table")
    for i, p in enumerate(pieces):
        print(f"  {i}: {p['name']} at {p['x']:.3f}, {p['y']:+.3f}")
    print(__doc__.split("\n\n")[2])
    n = a.n % len(pieces)
    teach = Teach(a.robot, pieces[n], n)
    print(f"piece {n}: {pieces[n]['name']} at {pieces[n]['x']:.3f}, {pieces[n]['y']:+.3f}")
    teach.move()
    step = 0.001
    while True:
        k = key()
        if k[0] == "\xe0" and k[1:] in KEYS:
            before = teach.shift.copy()
            teach.shift += np.array(KEYS[k[1:]]) * step
            if not teach.move():
                teach.shift = before
        elif k in ("[", "]"):
            teach.turn += -3 if k == "[" else 3
            if not teach.move():
                teach.turn -= -3 if k == "[" else 3
        elif k in ("1", "5"):
            step = 0.001 if k == "1" else 0.005
            print(f"   steps of {1000 * step:.0f} mm")
        elif k == "\r":
            if teach.pose is not None:
                teach.save(pieces)
        elif k == "n":
            teach.shift[2] = max(teach.shift[2], 0.0)
            teach.move()   # up first, over this piece
            n = (n + 1) % len(pieces)
            teach = Teach(a.robot, pieces[n], n)
            print(f"piece {n}: {pieces[n]['name']} at {pieces[n]['x']:.3f}, {pieces[n]['y']:+.3f}")
            teach.move()
        elif k == "h":
            teach.shift[2] += 0.03
            teach.move()
            http(a.robot, "/arm/call", {"method": "home", "args": [], "kwargs": {}})
            return 0
        elif k == "q":
            return 0


if __name__ == "__main__":
    sys.exit(main())
