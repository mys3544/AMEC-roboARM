"""One real trial of a policy, scored as the simulator's grid scores one (m3_sim.py evaluate): where the cube is
(the picker's look, from the policy's start pose), the policy run on the robot (m3_policy.py robot), then the
closest the tool point came to the cube by the run's log (the poses sent, through the arm model), the J1 at
1.2 s, and what a look afterwards finds there -- moved, gone (picked), or as it was.

    ~/.roboarm/lerobot/Scripts/python.exe tools/m3_policy_trial.py CHECKPOINT [--seconds 25] [--max-dps 100]

Each trial -> out/policy/trials.jsonl. The cube alone on the floor in front, the box where the jobs find it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roboarm import m3kinematics

BRIDGE = "http://192.168.73.66:8761"
OUT = ROOT / "out" / "policy"


def call(path: str, body: dict | None = None, timeout: float = 30.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BRIDGE + path, data=data, headers={"Content-Type": "application/json"},
                                 method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def look() -> list[dict]:
    """The picker's look over the table (POST /pick/look): the kit things it found."""
    call("/pick/look", {})
    time.sleep(2.0)
    while call("/pick/state").get("busy"):
        time.sleep(1.0)
    return [f for f in call("/pick/state").get("found", []) if f.get("kind") and f.get("reach")]   # out of reach: not it


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--max-dps", type=float, default=100.0)
    ap.add_argument("--near", type=float, nargs=2, metavar=("X", "Y"),
                    help="the trial's cube: the kit thing nearest this (m), where the look finds more than one")
    a = ap.parse_args()
    before = look()
    if a.near is not None and before:
        before = [min(before, key=lambda f: np.hypot(f["x"] - a.near[0], f["y"] - a.near[1]))]
    if len(before) != 1:
        raise SystemExit(f"want one kit thing on the floor, the look found {len(before)}: {before}")
    cube = np.array([before[0]["x"], before[0]["y"], before[0].get("top", 0.03) / 2])
    axis = np.asarray(m3kinematics.CHAIN[0][0][:2])
    j1_to = 90 - np.degrees(np.arctan2(cube[1] - axis[1], cube[0] - axis[0]))
    print(f"the {before[0]['name']} at x {cube[0]:.3f} y {cube[1]:+.3f} (J1 to it ~{j1_to:.0f}); the policy for "
          f"{a.seconds:.0f} s", flush=True)
    started, t_start = time.strftime("%Y%m%d_%H%M%S"), time.time()
    subprocess.run([sys.executable, str(ROOT / "tools" / "m3_policy.py"), "robot", str(a.checkpoint),
                    "--seconds", str(a.seconds), "--max-dps", str(a.max_dps)], check=False)
    runs = sorted((p for p in OUT.glob("run_*.jsonl") if p.stat().st_mtime >= t_start - 1),   # the robot's clock
                  key=lambda p: p.stat().st_mtime)                                                # is not the laptop's
    if not runs:
        raise SystemExit("no run log came back")
    ticks = [json.loads(line) for line in runs[-1].read_text().splitlines()]
    closest, lowest, j1_12 = np.inf, np.inf, None
    for t in ticks:
        pose = t.get("sent") or t.get("here")
        if pose is None:
            continue
        point = m3kinematics.grip({j: pose[j - 1] for j in range(1, 6)})[0]
        closest = min(closest, float(np.linalg.norm(point - cube)))
        lowest = min(lowest, float(point[2]))
        if j1_12 is None and t["t"] >= 1.2:
            j1_12 = pose[0]
    call("/pick/home", {})
    time.sleep(1.0)
    while call("/pick/state").get("busy"):
        time.sleep(1.0)
    after = look()
    moved = min((float(np.hypot(f["x"] - cube[0], f["y"] - cube[1])) for f in after), default=None)
    outcome = "gone (picked?)" if moved is None or moved > 0.05 else f"there, moved {1000 * moved:.0f} mm"
    row = {"when": started, "checkpoint": str(a.checkpoint), "cube": cube.round(4).tolist(), "j1_to": round(float(j1_to), 1),
           "j1_at_1_2s": j1_12, "closest_mm": round(1000 * closest), "lowest_grip_mm": round(1000 * lowest),
           "after": outcome, "run": runs[-1].name}
    with open(OUT / "trials.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"J1 to it ~{j1_to:.0f}, J1 after 1.2 s {j1_12}: the grip point came within {1000 * closest:.0f} mm "
          f"(lowest {1000 * lowest:.0f} mm up); the cube after: {outcome}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
