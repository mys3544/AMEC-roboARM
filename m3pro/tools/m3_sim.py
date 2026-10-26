"""The M3's arm in Isaac Sim, from the laptop (NEXT.md, "After the event"): the plans the simulated arm
plays -- made by the picker's own geometry and grasp planner -- run on Dionysos by m3pro/sim/isaac_m3.py,
and what came of them.

    python tools/m3_sim.py setup                 # Yahboom's description (from the robot) + isaac_m3.py to
                                                 # Dionysos, imported there as m3pro.usd
    python tools/m3_sim.py grip [--x 0.20] [--y 0.0] [--yaw 0] [--size 0.03] [--mass 0.015] [--friction 0.6]
        a cube picked as the M3 picks it -- open, over it, down the line, shut, up, to the right, let go --
        played in the simulator; says whether it came up with the fingers, stayed in them, and where it fell

The simulated arm is the URDF's, without the real one's sag (m3kinematics.SAG is for the real arm: off
here). Servo degrees -> URDF radians as m3kinematics turns them (J1 mirrored, 90 the zero); the gripper's
J6 -> the finger joint through the opening measured on 2026-10-23 (NEXT.md) and the URDF's linkage.
Output in out/sim/ (plan_*.json, result_*.json).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roboarm import m3arm, m3kinematics, m3pick

OUT = ROOT / "out" / "sim"
DESC = OUT / "yahboom_M3Pro_description"
HOST, REMOTE = "Hercules@147.232.40.154", "C:/Users/Hercules/roboarm-sim"
ISAAC = r"C:\isaac-sim-standalone-5.0.0-windows-x86_64\python.bat"
ARM = ["arm1_Joint", "arm2_Joint", "arm3_Joint", "arm4_Joiint", "arm5_Joint"]
FINGER = "rlink1_Joint"
# J6 -> the finger joint (rad): the gap measured between the rubber tips (2026-10-23) matched to the gap
# between the URDF's fingertips over the joint's travel; 0-30 is the open stop (61.5 mm, the URDF's 60 at 0.6)
J6_TO_FINGER = ((0, 0.6), (30, 0.6), (60, 0.242), (90, -0.022), (120, -0.392), (150, -0.721), (165, -0.907),
                (170, -0.974), (180, -0.974))
SPEEDS = {"free": 90, "near": 60, "descend": 25, "lift": 60, "gripper": 150}   # the picker's defaults, deg/s
GRIP_S = 0.3
K = (480.8, 480.9, 323.8, 240.0)   # the DCW2's colour camera at 640 x 480 (its camera_info)
TEXTURES = "C:/Users/Hercules/roboarm-synth/textures"   # train/synth/textures, put there for kit_synth.py
# the lab table as the wide look saw it on 2026-10-26 (its edges in the picture, through the model): the paper's
# middle and turn, the light
TABLE = {"textures": TEXTURES, "paper": [0.12, -0.105, 0.0], "paper_texture": "paper_faint.png",
         "dome": 250.0, "sun": 700.0, "sun_from": [0.5, 0.4, 0.8]}   # its shadows fall towards the robot, right
GREEN = (0.23, 0.46, 0.23)          # the lab's green picture cube's paint (kit_synth.py's CUBE_PAINT)


def green_cube() -> list[dict]:
    """The lab's green 30 mm picture cube: paint top and bottom, a picture on each side."""
    return [{"colour": GREEN}, {"colour": GREEN}] + [{"texture": f"{TEXTURES}/pic_green_{p}.png"}
                                                     for p in ("apple", "egg", "fish", "melon")]
FLIP = np.diag([1.0, -1.0, -1.0])  # the optical frame (x right, y down, z ahead) -> a USD camera's (+y up, -z ahead)


def wrist_view() -> dict:
    """The simulated wrist camera as the robot's model has it: on the URDF's DCW2 link, at its optical frame
    (m3kinematics.camera_optical -- the hand-eye fit of 2026-09-30 was degenerate and is not used)."""
    T = np.eye(4)
    T[:3, :3] = m3kinematics.OPTICAL @ FLIP
    return {"mount": "DCW2", "T": T.tolist(), "K": list(K), "size": [640, 480]}


def project(pose: dict, point) -> np.ndarray:
    """Where `point` (base x, y, z) falls in the real camera's picture at servo `pose`, by the robot's model."""
    p = np.linalg.inv(m3kinematics.frames(pose)["arm4"] @ m3kinematics.camera_optical()) @ np.r_[point[:3], 1.0]
    return np.array([K[0] * p[0] / p[2] + K[2], K[1] * p[1] / p[2] + K[3]])


def urdf_targets(pose: dict) -> dict:
    """Servo degrees {1..6} -> the simulated joints' targets, rad."""
    out = {ARM[j - 1]: math.radians(90 - pose[j] if j in m3kinematics.MIRRORED else pose[j] - 90) for j in range(1, 6)}
    out[FINGER] = float(np.interp(pose[6], *zip(*J6_TO_FINGER)))
    return out


class Plan:
    """Keys (time, servo pose) built as the arm sweeps: each move timed by its furthest joint."""

    def __init__(self, start: dict):
        self.keys, self.t, self.pose, self.marks = [(0.0, dict(start))], 0.0, dict(start), {}

    def go(self, pose: dict, speed: float, mark: str | None = None, hold: float = 0.0) -> None:
        pose = {**self.pose, **{int(j): float(v) for j, v in pose.items()}}
        self.t += max(0.05, max(abs(pose[j] - self.pose[j]) for j in range(1, 7)) / speed)
        self.keys.append((self.t, pose))
        if hold:
            self.t += hold
            self.keys.append((self.t, pose))
        self.pose = pose
        if mark:
            self.marks[mark] = round(self.t, 3)

    def json(self, pieces: list[dict], **extra) -> dict:
        return {"keys": [{"t": round(t, 3), "q": urdf_targets(p), "servo": {str(j): round(v, 1) for j, v in p.items()}}
                         for t, p in self.keys],
                "marks": self.marks, "pieces": pieces, **extra}


@contextlib.contextmanager
def no_sag():
    """The arm model without the real arm's sag: the simulated arm goes where it is told."""
    sag = dict(m3kinematics.SAG)
    m3kinematics.SAG.clear()
    try:
        yield
    finally:
        m3kinematics.SAG.update(sag)


def grip_plan(x: float, y: float, yaw: float, size: float, mass: float, friction: float, close: bool = False) -> dict:
    """A cube at (x, y) on the table picked as the picker plans it, carried to the right and let go."""
    with no_sag():
        return _grip_plan(x, y, yaw, size, mass, friction, close)


def _grip_plan(x: float, y: float, yaw: float, size: float, mass: float, friction: float, close: bool) -> dict:
    centre = np.array([x, y, size / 2])
    opened = m3pick.opened_for(size)
    shut = m3arm.gripper_angle(size * 1000)
    span = m3kinematics.finger_span(opened)
    planned = m3pick.plan_grasp(centre, size, floor_clear=0.008, gripper=opened, held=shut,
                                wrist=lambda grip: m3pick.wrist_choices(grip, yaw, 90, span, []))
    if planned is None:
        raise SystemExit(f"no grasp plan for a {size * 1000:.0f} mm cube at x {x} y {y}")
    pitch, line, _lift = planned
    home = dict(zip(m3arm.JOINTS, m3arm.POSES["home"]))
    plan = Plan(home)
    plan.go({**dict(zip(m3arm.JOINTS, m3arm.POSES["survey"])), 6: opened}, SPEEDS["free"], "look")
    plan.go(line[0], SPEEDS["near"], "pregrasp")
    for pose in line[1:]:
        plan.go(pose, SPEEDS["descend"])
    plan.marks["down"] = round(plan.t, 3)
    plan.go({6: shut}, SPEEDS["gripper"], "shut", hold=GRIP_S)
    for pose in line[-2::-1]:
        plan.go({**pose, 6: shut}, SPEEDS["lift"])
    plan.marks["up"] = round(plan.t, 3)
    # up to the look's height before turning, as the picker carries (carry_cm): swung round low, what
    # was held hit the robot's front-right corner (2026-10-23, the first runs)
    plan.go({**dict(zip(m3arm.JOINTS, m3arm.POSES["survey"])), 6: shut}, SPEEDS["lift"], "raised")
    plan.go({1: 160, 6: shut}, SPEEDS["free"], "carried")
    plan.go({6: opened}, SPEEDS["gripper"], "let_go", hold=0.5)
    plan.go(home, SPEEDS["free"], "home")
    print(f"plan: pitch {pitch}, line {len(line)} poses, grip J6 {opened} -> {shut}; "
          f"{plan.t:.1f} s, marks {plan.marks}", flush=True)
    return plan.json([{"size": [size, size, size], "centre": centre.tolist(), "yaw": yaw, "mass": mass}],
                     friction=friction, settle_s=1.0, grip_point=centre.tolist(),
                     pictures={"from": [x + 0.03, y - 0.13, 0.05], "at": [x, y, 0.03]} if close else
                     {"from": [x + 0.30, y - 0.30, 0.32], "at": [x - 0.05, y, 0.08]})


def remote(cmd: str, log: str, tries: int = 3) -> None:
    """`cmd` run on Dionysos, its log fetched -- again while its pictures came out black: one run in three
    renders black all through with an ACT training on the same GPU (2026-10-26; isaac_m3.py's _picture)."""
    for attempt in range(1, tries + 1):
        began = time.time()
        subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, f"cd /d {REMOTE.replace('/', chr(92))} && {cmd} > {log} 2>&1"],
                       check=False)
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{HOST}:{REMOTE}/{log}", str(OUT / log)], check=True)
        print(f"({time.time() - began:.0f} s on Dionysos; log out/sim/{log})", flush=True)
        if "the picture stayed black" not in (OUT / log).read_text(errors="replace"):
            return
        print(f"  the pictures came out black (run {attempt} of {tries})", flush=True)
    raise SystemExit("the simulator's pictures stayed black: see the log")


def put(*paths: Path) -> None:
    for p in paths:
        subprocess.run(["scp", "-q", "-r", "-o", "BatchMode=yes", str(p), f"{HOST}:{REMOTE}/"], check=True)


def setup() -> None:
    if not DESC.exists():
        subprocess.run(["scp", "-q", "-r", "-o", "BatchMode=yes",
                        "jetson@192.168.73.66:/home/jetson/yahboomcar_ws/src/yahboom_M3Pro_description", str(OUT)], check=True)
    subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, f"if not exist {REMOTE.replace('/', chr(92))} mkdir {REMOTE.replace('/', chr(92))}"],
                   check=False)
    put(DESC, ROOT / "sim" / "isaac_m3.py")
    remote(f"{ISAAC} isaac_m3.py import yahboom_M3Pro_description m3pro.usd", "import.log")


def grip(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    plan_path, result = OUT / f"plan_{stamp}.json", f"result_{stamp}.json"
    plan_path.write_text(json.dumps(grip_plan(args.x, args.y, args.yaw, args.size, args.mass, args.friction, args.close),
                                    indent=1))
    put(plan_path, ROOT / "sim" / "isaac_m3.py")
    remote(f"{ISAAC} isaac_m3.py play m3pro.usd {plan_path.name} {result}", f"play_{stamp}.log")
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{HOST}:{REMOTE}/{result}", str(OUT / result)], check=True)
    played = json.loads((OUT / result).read_text())
    sheet(played, OUT / result.replace(".json", ".jpg"))
    summary(played)


def look(args) -> None:
    """One picture from the simulated wrist camera at servo `pose`, the cube where given -- the arm held where the
    real one sits (the command plus m3kinematics.SAG, as the robot's model has it) -- and, given a real frame
    taken there, the two side by side and blended, the model's projection of the cube's top marked on both."""
    import cv2

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pose = {j: float(v) for j, v in zip(m3arm.JOINTS, args.pose.split(","), strict=True)}
    held = {j: v + m3kinematics.SAG.get(j, 0.0) for j, v in pose.items()}
    plan = Plan(held)
    plan.go(held, SPEEDS["free"], hold=0.6)
    plan.marks["look"] = 0.5
    x, y, yaw = (float(v) for v in args.cube.split(","))
    centre = [x, y, args.size / 2]
    plan_path, result = OUT / f"look_{stamp}.json", f"lookresult_{stamp}.json"
    table = {**TABLE, **json.loads(args.table or "{}")}
    plan_path.write_text(json.dumps(plan.json([{"size": [args.size] * 3, "centre": centre, "yaw": yaw, "mass": 0.015,
                                                "faces": green_cube()}], settle_s=0.0, pictures=wrist_view(),
                                              table=table), indent=1))
    put(plan_path, ROOT / "sim" / "isaac_m3.py")
    remote(f"{ISAAC} isaac_m3.py play m3pro.usd {plan_path.name} {result}", f"look_{stamp}.log")
    npy = OUT / f"lookresult_{stamp}_look.npy"
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{HOST}:{REMOTE}/{npy.name}", str(npy)], check=True)
    sim = cv2.cvtColor(np.load(npy).astype(np.uint8), cv2.COLOR_RGB2BGR)
    top = project(pose, [x, y, args.size])
    tiles = [sim]
    if args.real:
        real = cv2.imread(str(args.real))
        tiles = [real, sim, cv2.addWeighted(real, 0.5, sim, 0.5, 0)]
    for tile in tiles:
        cv2.drawMarker(tile, tuple(round(float(v)) for v in top), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    out = OUT / f"look_{stamp}.jpg"
    cv2.imwrite(str(out), np.hstack(tiles))
    print(f"the cube's top by the model at pixel {np.round(top, 1).tolist()}; picture: {out}", flush=True)


def sheet(played: dict, out: Path) -> None:
    """The pictures taken at the plan's marks, fetched and put side by side with their names."""
    import cv2

    tiles = []
    for name in played["plan"]["marks"]:
        npy = out.with_name(f"{out.stem}_{name}.npy")
        if subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{HOST}:{REMOTE}/{npy.name}", str(npy)], check=False).returncode:
            continue
        picture = np.load(npy)
        npy.unlink()
        if picture.ndim != 3 or not picture.size:
            print(f"  no picture at {name}: {picture.shape}", flush=True)
            continue
        tile = cv2.cvtColor(picture.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.putText(tile, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(tile, (320, 240)))
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
        print(f"pictures: {out}", flush=True)


def summary(result: dict) -> None:
    """What became of the cube: up with the fingers, still in them at the drop-off, where it fell."""
    marks, samples = result["plan"]["marks"], result["samples"]

    def at(t):
        return min(samples, key=lambda s: abs(s["t"] - t))

    start = np.array(samples[0]["pieces"][0][:3])
    down = at(marks["down"])
    servo = next(k["servo"] for k in result["plan"]["keys"] if abs(k["t"] - marks["down"]) < 1e-3)
    with no_sag():
        want = m3kinematics.frames({int(j): v for j, v in servo.items()})["gripping"][:3, 3]
    print(f"  the tool point at the grip: simulated {np.round(np.array(down['grip']) * 1000).astype(int)} mm, "
          f"the model's {np.round(want * 1000).astype(int)} mm, the cube's centre {np.round(start * 1000).astype(int)}", flush=True)
    for name in ("down", "shut", "up", "raised", "carried", "let_go", "home"):
        s = at(marks[name])
        cube, tips = np.array(s["pieces"][0][:3]), np.array(s["tips"])
        between = np.linalg.norm(cube - tips.mean(0))
        print(f"  {name:8s} t {s['t']:5.1f}: cube at {np.round(cube * 1000).astype(int)} mm "
              f"({1000 * (cube[2] - start[2]):+.0f} up), {1000 * between:.0f} mm from between the fingertips, "
              f"fingertips {1000 * np.linalg.norm(tips[0] - tips[1]):.0f} mm apart", flush=True)
    lifted = at(marks["up"])["pieces"][0][2] - start[2]
    carried = at(marks["carried"])["pieces"][0][2] - start[2]
    end = np.array(samples[-1]["pieces"][0][:3])
    print(f"lifted {1000 * lifted:.0f} mm, at the drop-off {1000 * carried:.0f} mm up, "
          f"ended at {np.round(end * 1000).astype(int)} mm -> "
          + ("PICKED AND CARRIED" if lifted > 0.03 and carried > 0.03 else "NOT carried"), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    g = sub.add_parser("grip")
    g.add_argument("--x", type=float, default=0.20)
    g.add_argument("--y", type=float, default=0.0)
    g.add_argument("--yaw", type=float, default=0.0)
    g.add_argument("--size", type=float, default=0.03)
    g.add_argument("--mass", type=float, default=0.015)
    g.add_argument("--friction", type=float, default=0.6)
    g.add_argument("--close", action="store_true", help="the pictures from beside the cube, not over the robot")
    lk = sub.add_parser("look")
    lk.add_argument("--pose", default="90,105,10,0,90,76", help="servo degrees J1..J6 (default: the wide look)")
    lk.add_argument("--cube", default="0.2377,-0.0158,4.5", help="x,y,yaw of a cube on the table (m, m, deg)")
    lk.add_argument("--size", type=float, default=0.03)
    lk.add_argument("--real", type=Path, help="a real frame taken at that pose, to lay beside and over it")
    lk.add_argument("--table", help='JSON over TABLE\'s keys, e.g. {"sun": 0, "paper_texture": "paper.png"}')
    a = ap.parse_args()
    if a.cmd == "setup":
        setup()
    elif a.cmd == "look":
        look(a)
    else:
        grip(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
