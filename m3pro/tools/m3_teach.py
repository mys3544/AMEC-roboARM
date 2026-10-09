"""Teach the grip by hand, from the keyboard: the M3 cannot go limp or read its joints back
(roboarm/m3arm.py), so instead of moving the arm, the teacher steers it -- from over a kit piece the
wide look found (tools/m3_hover.py look), in mm steps, until the fingers sit where the ideal grip
is -- and saves. Every pose sent is known, so each save is a pair: where the camera saw the piece,
and the pose that really reaches it. The pairs fit the correction (out/teach/points.jsonl).

    python tools/m3_teach.py [N]      # over piece N of out/hover/look.json (nearest 0), then the keys:

    arrows up / down     out / in (away from the robot / towards it)
    arrows left / right  left / right, as the arm sees it
    PgUp / PgDn          up / down
    [ / ]                the wrist turned -/+ 3 deg
    1 / 5                steps of 1 mm or 5 mm
    Enter                save this pose as the ideal grip of the piece nearest it
    n                    the next piece (from 20 mm over it again)
    h                    up 3 cm, then home, and quit
    q                    quit where it is

Windows console only (msvcrt). Through the bridge (tools/m3bridge.py), no job running.
"""

from __future__ import annotations

import json
import msvcrt
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import m3_handeye as he

from roboarm import m3arm, m3kinematics, m3pick

LOOK = ROOT / "out" / "hover" / "look.json"
POINTS = ROOT / "out" / "teach" / "points.jsonl"
START_UP = 0.020    # m: the fingertips over the piece's top at the start
LOWEST = 0.004      # m: the fingertips may come this close to the table, no closer
KEYS = {"H": (1, 0, 0), "P": (-1, 0, 0), "K": (0, 1, 0), "M": (0, -1, 0), "I": (0, 0, 1), "Q": (0, 0, -1)}


def call(method: str, *args, **kwargs):
    return he.http("/arm/call", {"method": method, "args": list(args), "kwargs": kwargs})


class Teach:
    def __init__(self, piece: dict, n: int):
        self.piece, self.n = piece, n
        self.centre = np.asarray(piece["centre"], float)
        axis = np.asarray(m3kinematics.CHAIN[0][0][:2])
        self.out = (self.centre[:2] - axis) / np.linalg.norm(self.centre[:2] - axis)
        self.left = np.array([-self.out[1], self.out[0]])
        self.shift = np.zeros(3)          # m: out, left, up -- from the fingertips START_UP over the top
        self.turn = 0.0                   # deg on the squared wrist
        self.opened = m3pick.opened_for(piece["size"], m3pick.OPEN_MM - 1000 * piece["size"])
        self.pose = None

    def target(self) -> np.ndarray:
        xy = self.centre[:2] + self.out * self.shift[0] + self.left * self.shift[1]
        return np.r_[xy, self.piece["top"] + START_UP + self.shift[2] + m3kinematics.FINGERS]

    def plan(self) -> dict | None:
        target = self.target()
        for pitch in (88, 85, 80, 75, 70, 65, 60):
            pose = m3kinematics.reach(target, pitch, guard=False)
            if pose is None:
                continue
            j5 = m3pick.wrist_squares(pose, self.piece["yaw"], self.piece["period"])[0] + round(self.turn)
            pose = {**pose, 5: j5, 6: self.opened}
            if m3kinematics.problem(pose, LOWEST) is None:
                return pose
        return None

    def move(self) -> bool:
        pose = self.plan()
        if pose is None:
            print("   no allowed pose there -- not moved")
            return False
        reply = call("move_to", {str(j): v for j, v in pose.items()}, speed_dps=15)
        if "error" in reply:
            print(f"   refused: {reply['error']}")
            return False
        self.pose = pose
        tips = self.piece["top"] + START_UP + self.shift[2]
        print(f"   out {1000 * self.shift[0]:+5.0f}  left {1000 * self.shift[1]:+5.0f} mm, fingertips "
              f"{1000 * tips:3.0f} mm over the table, wrist {self.turn:+.0f}  (pose {pose})")
        return True

    def save(self, pieces: list[dict]) -> None:
        """The pose as the ideal grip of the piece nearest it -- steered over to another piece without
        `n`, it is that one's (2026-10-09: three of six saves were made so)."""
        point, pitch = m3kinematics.grip(self.pose)
        n = int(np.argmin([np.linalg.norm(point[:2] - np.asarray(p["centre"][:2])) for p in pieces]))
        centre = np.asarray(pieces[n]["centre"][:2])
        axis = np.asarray(m3kinematics.CHAIN[0][0][:2])
        out = (centre - axis) / np.linalg.norm(centre - axis)
        error = point[:2] - centre
        record = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "piece": pieces[n], "n": n,
                  "turn_deg": self.turn, "pose": self.pose,
                  "model_grip": np.round(point, 4).tolist(), "pitch": round(pitch, 1)}
        POINTS.parent.mkdir(parents=True, exist_ok=True)
        with POINTS.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"   SAVED piece {n}: seen at {centre.round(3).tolist()}, the ideal grip "
              f"{1000 * error @ out:+.0f} mm out, {1000 * error @ np.array([-out[1], out[0]]):+.0f} mm left of it")


def key() -> str:
    k = msvcrt.getwch()
    return "\xe0" + msvcrt.getwch() if k in ("\x00", "\xe0") else k


def main() -> int:
    pieces = json.loads(LOOK.read_text())["pieces"]
    if not pieces:
        raise SystemExit("no kit piece in out/hover/look.json: run tools/m3_hover.py look first")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(__doc__.split("\n\n")[2])
    teach = Teach(pieces[n], n)
    print(f"piece {n}: {pieces[n]['kind']} at {np.round(pieces[n]['centre'][:2], 3).tolist()}")
    teach.move()
    step = 0.001
    while True:
        k = key()
        if k[0] == "\xe0" and k[1] in KEYS:
            before = teach.shift.copy()
            teach.shift += np.array(KEYS[k[1]]) * step
            if not teach.move():
                teach.shift = before
        elif k in "[]":
            teach.turn += -3 if k == "[" else 3
            if not teach.move():
                teach.turn -= -3 if k == "[" else 3
        elif k in "15":
            step = 0.001 if k == "1" else 0.005
            print(f"   steps of {1000 * step:.0f} mm")
        elif k == "\r":
            teach.save(pieces)
        elif k == "n":
            n = (n + 1) % len(pieces)
            teach.shift[2] = max(teach.shift[2], 0.0)
            teach.move()   # up first, over this piece
            teach = Teach(pieces[n], n)
            print(f"piece {n}: {pieces[n]['kind']} at {np.round(pieces[n]['centre'][:2], 3).tolist()}")
            teach.move()
        elif k == "h":
            teach.shift[2] += 0.03
            teach.move()
            call("move_to", dict(zip("123456", m3arm.POSES["home"])), speed_dps=25)
            return 0
        elif k == "q":
            return 0


if __name__ == "__main__":
    sys.exit(main())
