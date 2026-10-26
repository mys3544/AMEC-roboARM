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
import itertools
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


CUBE_PAINT = {"yellow": (0.85, 0.72, 0.15), "green": GREEN, "red": (0.72, 0.16, 0.19), "blue": (0.22, 0.41, 0.70)}
PICTURES = {"yellow": ("chopsticks", "cigarette", "peach", "toilet"), "green": ("apple", "egg", "fish", "melon"),
            "red": ("batteries", "cosmetics", "tablets", "tablets"), "blue": ("bag", "book", "can", "news")}
SIDES40 = [(0.80, 0.60, 0.26), (0.15, 0.45, 0.22), (0.85, 0.26, 0.30), (0.12, 0.42, 0.62)]   # the 40 mm cube's paint
EPISODES = ROOT / "out" / "sim_episodes"   # the simulated recordings, in roboarm.m3record's layout
HZ = 10
# where a pick attempt is, by its step: what the hand-back gate learns to predict (RESEARCH.md 9.4, 9.7)
STAGES = {"pregrasp": 0.1, "look.close": 0.25, "descend": 0.4, "close": 0.55, "lift": 0.7, "drop": 0.85,
          "look.after": 1.0}


def piece_faces(piece: dict, rng) -> list[dict]:
    """The lab's look for a piece (kit_synth.py's): a 30 mm picture cube (paint top and bottom, its colour's
    pictures round), the 40 mm cube (wood ends, four paints), a block (red, a tag on an end, or blue)."""
    if piece["name"] == "cube30":
        colour = str(rng.choice(list(CUBE_PAINT)))
        paint = {"colour": CUBE_PAINT[colour]}
        return [paint, paint] + [{"texture": f"{TEXTURES}/pic_{colour}_{p}.png"} for p in rng.permutation(PICTURES[colour])]
    if piece["name"] == "cube40":
        return [{"texture": f"{TEXTURES}/wood_a.png"}, {"texture": f"{TEXTURES}/wood_b.png"}] + [{"colour": c} for c in SIDES40]
    if rng.random() < 0.5:
        red = {"colour": (0.69, 0.13, 0.15)}
        return [red, red, {"texture": f"{TEXTURES}/tag_{rng.choice(list('abcde'))}.png"}, red, red, red]
    return [{"colour": (0.18, 0.38, 0.66)}] * 6


def frame_of(origin, x_axis, z_axis=(0.0, 0.0, 1.0)) -> np.ndarray:
    """A 4 x 4 (column vectors) at `origin`, x along `x_axis`, z as near `z_axis` as square allows."""
    x = np.asarray(x_axis, float) / np.linalg.norm(x_axis)
    y = np.cross(np.asarray(z_axis, float), x)
    y /= np.linalg.norm(y)
    m = np.eye(4)
    m[:3, :3] = np.c_[x, y, np.cross(x, y)]
    m[:3, 3] = origin
    return m


def box_parts(box: dict, lean_deg: float = 20.0) -> list[dict]:
    """The drop-off box as fixed parts (isaac_m3.py's add_static): a floor, four walls round roboshared.box's
    inside, and the lid standing up from the hinge on the far side, leaning back, marker 10 inside it."""
    from roboshared import box as boxes

    along_m, across_m, wall_h = boxes.BOX_INSIDE
    t = boxes.BOX_WALL
    c = np.r_[box["centre"], 0.0]
    a, i = np.r_[box["along"], 0.0], np.r_[box["into"], 0.0]
    dark = (0.03, 0.03, 0.03)
    parts = [{"T": frame_of(c + [0, 0, 0.002], a).tolist(), "size": [along_m + 2 * t, across_m + 2 * t, 0.004], "colour": dark}]
    for side in (-1, 1):
        parts.append({"T": frame_of(c + i * side * (across_m + t) / 2 + [0, 0, wall_h / 2], a).tolist(),
                      "size": [along_m + 2 * t, t, wall_h], "colour": dark})
        parts.append({"T": frame_of(c + a * side * (along_m + t) / 2 + [0, 0, wall_h / 2], a).tolist(),
                      "size": [t, across_m + 2 * t, wall_h], "colour": dark})
    lean = math.radians(lean_deg)
    up = math.cos(lean) * np.array([0, 0, 1.0]) - math.sin(lean) * i   # leaning back, away from the robot
    hinge = c - i * (across_m / 2 + t) + [0, 0, wall_h]
    lid_h = 0.12
    marker = {"texture": f"{TEXTURES}/box_lid_marker10.png"}
    parts.append({"T": frame_of(hinge + up * lid_h / 2, a, z_axis=up).tolist(), "size": [along_m + 2 * t, 0.004, lid_h],
                  # +z -z +x +y -x -y: the marker on its face towards the robot (-y of a frame whose y is up x along)
                  "faces": [{"colour": dark}, {"colour": dark}, {"colour": dark}, {"colour": dark}, {"colour": dark}, marker]})
    return parts


def board_keys(sends: list) -> list[tuple[float, dict]]:
    """The board's sweeps as straight-line keys (t, servo pose): each command from where the arm is believed
    then (M3Arm.read) to its pose over its ms -- cut short by the next command, as the board does."""
    keys, cur = [], None
    for k, (t, pose, ms) in enumerate(sends):
        if cur is None:
            here = {int(j): float(v) for j, v in pose.items()}
        else:
            start, goal, t0, dur = cur
            f = 1.0 if dur <= 0 else min(max((t - t0) / dur, 0.0), 1.0)
            here = {j: start[j] + f * (goal[j] - start[j]) for j in goal}
        goal = {**here, **{int(j): float(v) for j, v in pose.items()}}
        keys.append((t, here))
        cur = (here, goal, t, ms / 1000)
        if t + ms / 1000 < (sends[k + 1][0] if k + 1 < len(sends) else math.inf):
            keys.append((t + ms / 1000, goal))
    return keys


def pose_at(keys: list, t: float) -> dict:
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, p0), (t1, p1) in itertools.pairwise(keys):
        if t <= t1:
            f = (t - t0) / max(t1 - t0, 1e-9)
            return {j: p0[j] + f * (p1[j] - p0[j]) for j in p1}
    return keys[-1][1]


# The real arm against its model (m3kinematics.SAG), degrees a joint sits further: the fingertip touches of
# 2026-10-26 (tools/m3_touch.py, ten over the ChArUco board, reach 16-27 cm, the fingers 55-90 deg down) put the
# tips 12 mm short and 19 mm low of the model, nearly alike in every pose. J2 2.9 deg further down (and the shut
# fingertips 9 mm longer) explains them to 5.3 mm held out (13.8 before), keeps the camera's readings
# (m3_twin_gap.py) at 4.4 mm and foretells the look's camera 6 mm low -- the depth saw it 7 mm low. The short part
# is the real picker's taught aim (m3aim: 8-22 mm further out). Each card's offsets centre on it, so the simulated
# arm is the real one and the expert compensates it as that aim does; zero once m3kinematics.SAG takes it --
# which it did the same day (SAG J2 4.0 -> 1.1): the episodes of 16:00-17:00 carry {2: -2.9} on the old SAG,
# the same arm.
TWIN_SAG: dict[int, float] = {}


def true_sag() -> dict[int, float]:
    """Degrees the real arm sits off its commands: the model's SAG and the twin's offset."""
    return {j: m3kinematics.SAG.get(j, 0.0) + TWIN_SAG.get(j, 0.0) for j in sorted({*m3kinematics.SAG, *TWIN_SAG})}


def draw_card(rng) -> dict:
    """A robot card (RESEARCH.md 9.3): how this episode's simulated M3 differs from its model -- the servos'
    delay, gain round 90 and zero, backlash, the gripper's zero, the camera's mount. The expert compensates the
    joints' part (compensated(): the arm lands where planned, the commands recorded carry the card), the camera's
    shows in the pictures. Ranges from the servo probe of 2026-10-23 (lag 0.10-0.20 s, backlash 0.36-1.09 deg,
    15 deg moves turning 0.87-0.93 of the command); the joints' zeros round the real arm's (TWIN_SAG)."""
    return {"delay_s": round(float(rng.uniform(0.10, 0.20)), 3),
            "gain": {j: round(float(rng.uniform(0.86, 1.0)), 4) for j in (1, 2, 3, 4)},
            "offset_deg": {j: round(float(rng.normal(TWIN_SAG.get(j, 0.0), 1.5)), 2) for j in (1, 2, 3, 4, 5)},
            "twin_sag": dict(TWIN_SAG),
            "backlash_deg": {j: round(float(rng.uniform(0.3, 1.2)), 2) for j in (1, 2, 3, 4)},
            "gripper_offset_deg": round(float(rng.normal(0, 2.0)), 2),
            "camera": {"shift_m": np.round(np.clip(rng.normal(0, 0.008, 3), -0.02, 0.02), 4).tolist(),
                       "turn_deg": np.round(np.clip(rng.normal(0, 1.5, 3), -3, 3), 2).tolist()}}


def sim_keys(keys: list, card: dict) -> list[dict]:
    """The board's keys as the simulated arm follows them (isaac_m3.py's keys): where the plan has the arm
    sit -- its commands plus SAG, the robot's model -- the card's delay late. The rest of the card is in the
    commands recorded (compensated): the expert knows its arm, as the real picker's taught aim does."""
    return [{"t": round(t + card["delay_s"], 4),
             "q": urdf_targets({**{j: pose[j] + m3kinematics.SAG.get(j, 0.0) for j in range(1, 6)}, 6: pose[6]})}
            for t, pose in keys]


def compensated(keys: list, card: dict) -> list[tuple[float, dict]]:
    """The commands an arm with `card` must be sent to sit where the plan has it (command + SAG): its gain
    round 90 and its zero undone, half its backlash added the way each joint last moved, the gripper's zero
    taken off -- what a simulated episode records as the state, as the real recorder records what was sent."""
    out, last, way = [], None, dict.fromkeys(range(1, 7), 0.0)
    for t, pose in keys:
        if last is not None:
            for j in way:
                if abs(pose[j] - last[j]) > 1e-6:
                    way[j] = math.copysign(1.0, pose[j] - last[j])
        sent = {}
        for j in range(1, 6):
            sits = pose[j] + m3kinematics.SAG.get(j, 0.0)
            sent[j] = (90 + (sits - 90 - card["offset_deg"].get(j, 0.0) + way[j] * card["backlash_deg"].get(j, 0.0) / 2)
                       / card["gain"].get(j, 1.0) - m3kinematics.SAG.get(j, 0.0))
        sent[6] = pose[6] - card["gripper_offset_deg"]
        # inside the servos' range, as a real command is (a gain round 90 sends a folded J4 below 0)
        sent = {j: min(max(v, m3arm.SAFE_LIMITS[j][0]), m3arm.SAFE_LIMITS[j][1]) for j, v in sent.items()}
        out.append((t, sent))
        last = pose
    return out


def carded_view(card: dict) -> dict:
    """The wrist camera with the card's mount: shifted and turned in the DCW2 link's frame."""
    view = wrist_view()
    off = np.eye(4)
    rx, ry, rz = np.radians(card["camera"]["turn_deg"])
    off[:3, :3] = (np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
                   @ np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
                   @ np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]]))
    off[:3, 3] = card["camera"]["shift_m"]
    view["T"] = (off @ np.array(view["T"])).tolist()
    return view


def first_afters(steps: list) -> list[float]:
    """When each attempt's look after began: the first after each pregrasp (a look round may follow it)."""
    starts = [t for t, s in steps if s == "pregrasp"]
    return [next((t for t, s in steps if s == "look.after" and t > t0), math.inf) for t0 in starts]


def verdicts(plan: dict, played: dict, pieces: list[dict], box: dict, delay: float) -> list[bool]:
    """Per pick record: did the simulated arm put its piece in the box -- that piece (the layout's nearest the
    record's target) over the box's inside and low, half a second after its attempt's look after began."""
    from roboshared import box as boxes

    afters = first_afters(plan["steps"])
    samples = played["samples"]
    out = []
    for k, record in enumerate(plan["records"]):
        target = np.asarray(record["target"][:2])
        which = min(range(len(pieces)), key=lambda i: float(np.hypot(*(pieces[i]["centre"][:2] - target))))
        when = min(afters[k] if k < len(afters) else math.inf, samples[-1]["t"] - delay - 0.5) + delay + 0.5
        s = min(samples, key=lambda x: abs(x["t"] - when))
        x, y, z = s["pieces"][which][:3]
        out.append(bool(boxes.inside(box, (x, y)) and z < 0.06))
    return out


def episode(args) -> None:
    """One simulated clean-up recorded as the real recorder keeps one (roboarm.m3record): a layout drawn, the
    picker told it (m3_expert), its commands played with a robot card drawn, the wrist camera ten times a
    second -> out/sim_episodes/<stamp>_clean_up/ (000000.jpg ..., ticks.jsonl, episode.json), its pick records
    marked picked or not by what the simulated pieces did. tools/m3_episodes.py build --episodes out/sim_episodes."""
    import cv2
    import m3_expert

    rng = np.random.default_rng(args.seed)
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_s{args.seed}"   # the seed in it: workers side by side never share a name
    box = m3_expert.facing_box(np.array(m3_expert.BOX_AT) + rng.uniform(-0.02, 0.02, 2), rng.uniform(-10, 10))
    pieces = m3_expert.layout(rng, args.pieces, box)
    began = time.time()
    plan = m3_expert.plan_episode(pieces, box, args.seed)
    print(f"planned: {plan['seconds']} s, {len(plan['records'])} picks ({time.time() - began:.0f} s); "
          f"error {plan['error']}", flush=True)
    if not plan["records"]:
        raise SystemExit("nothing picked in the plan: another seed")
    card = draw_card(rng)
    if args.card == "zero":   # the arm exactly as its model: whether the plan itself picks
        card = {"delay_s": 0.0, "gain": {}, "offset_deg": {}, "backlash_deg": {}, "gripper_offset_deg": 0.0,
                "camera": {"shift_m": [0.0, 0.0, 0.0], "turn_deg": [0.0, 0.0, 0.0]}}
    keys = board_keys(plan["sends"])
    table = {**TABLE, "paper": [TABLE["paper"][0] + rng.uniform(-0.03, 0.03), TABLE["paper"][1] + rng.uniform(-0.03, 0.03),
                                rng.uniform(-5, 5)],
             "dome": float(rng.uniform(150, 350)), "sun": float(rng.uniform(400, 1000)),
             "sun_from": [float(rng.uniform(0.2, 0.7)), float(rng.uniform(-0.2, 0.6)), 0.8]}
    name = f"{stamp}_clean_up"
    sim_plan = {"keys": sim_keys(keys, card), "settle_s": 0.5, "friction": float(rng.uniform(0.5, 0.8)),
                "pieces": [{"size": [p["size"], p["length"], p["height"]], "centre": [float(v) for v in p["centre"]],
                            "yaw": p["yaw"], "mass": 0.015,   # its width (x) along yaw, as in_picture has it
                            "faces": piece_faces(p, rng)} for p in pieces],
                "statics": box_parts(box), "table": table, "pictures": carded_view(card),
                "record": {"dir": name, "hz": HZ}}
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path, result = OUT / f"episode_{stamp}.json", f"episoderesult_{stamp}.json"
    plan_path.write_text(json.dumps(sim_plan))
    put(plan_path, ROOT / "sim" / "isaac_m3.py")
    remote(f"{ISAAC} isaac_m3.py play m3pro.usd {plan_path.name} {result}", f"episode_{stamp}.log")
    subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, f"cd /d {REMOTE.replace('/', chr(92))} && tar -czf {name}.tgz {name}"],
                   check=True)
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{HOST}:{REMOTE}/{result}", str(OUT / result)], check=True)
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{HOST}:{REMOTE}/{name}.tgz", str(OUT / f"{name}.tgz")], check=True)
    played = json.loads((OUT / result).read_text())
    folder = EPISODES / name
    folder.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(OUT / f"{name}.tgz"), "-C", str(EPISODES)], check=True)
    (OUT / f"{name}.tgz").unlink()
    subprocess.run(["ssh", "-o", "BatchMode=yes", HOST,   # its pictures are here now: Dionysos keeps only the logs
                    f"cd /d {REMOTE.replace('/', chr(92))} && rmdir /s /q {name} & del {name}.tgz {plan_path.name}"], check=False)
    picked = verdicts(plan, played, pieces, box, card["delay_s"])
    # the ticks as the real recorder writes them: the pose the arm was sent (believed), the job's step
    steps, records = plan["steps"], plan["records"]
    starts = [t for t, s in steps if s == "pregrasp"]
    afters = first_afters(steps)
    sent = compensated(keys, card)
    began_unix = time.time()
    with open(folder / "ticks.jsonl", "w", encoding="utf-8") as f:
        for i in range(played["ticks"]):
            t = i / HZ
            step = next((s for t0, s in reversed(steps) if t0 <= t + 1e-9), "engage")
            k = next((k for k, t0 in enumerate(starts) if t0 <= t + 1e-9 and (k + 1 >= len(starts) or t < starts[k + 1])), None)
            within = k is not None and k < len(records) and (k >= len(afters) or t < afters[k] + 3.0)
            f.write(json.dumps({"i": i, "t": round(t, 3), "pose": [round(pose_at(sent, t)[j]) for j in range(1, 7)],
                                "step": step, "odom": None, "frame_age": 0.0,
                                "progress": STAGES.get(step) if within else None,
                                "verdict": picked[k] if within else None}) + "\n")
    log = [{"n": n + 1, "t": round(began_unix + t, 3), "text": text} for n, (t, text) in enumerate(plan["said"])]
    (folder / "episode.json").write_text(json.dumps({
        "job": "clean_up", "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(began_unix)),
        "started_unix": round(began_unix, 3), "seconds": plan["seconds"], "ticks": played["ticks"], "error": plan["error"],
        "log": log, "attempts": [{**r, "picked": bool(ok), "planned_picked": r.get("picked")} for r, ok in zip(records, picked, strict=True)],
        "robot_card": card, "sim": {"seed": args.seed, "box": {k: np.asarray(v).tolist() for k, v in box.items() if k != "items"},
                                    "layout": [{**p, "centre": np.asarray(p["centre"]).tolist()} for p in pieces]}},
        indent=1, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else float(v)), encoding="utf-8")
    sheet_path = OUT / f"{name}_sheet.jpg"
    frames = sorted(folder.glob("*.jpg"))
    tiles = [cv2.resize(cv2.imread(str(frames[int(i)])), (320, 240)) for i in np.linspace(0, len(frames) - 1, 8)]
    cv2.imwrite(str(sheet_path), np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))
    print(f"{played['ticks']} pictures, picks {sum(picked)} of {len(picked)} in the box (planned {len(records)}) "
          f"-> {folder}; sheet {sheet_path}", flush=True)


def green_cube() -> list[dict]:
    """The lab's green 30 mm picture cube: paint top and bottom, a picture on each side."""
    return [{"colour": GREEN}, {"colour": GREEN}] + [{"texture": f"{TEXTURES}/pic_green_{p}.png"}
                                                     for p in ("apple", "egg", "fish", "melon")]
FLIP = np.diag([1.0, -1.0, -1.0])  # the optical frame (x right, y down, z ahead) -> a USD camera's (+y up, -z ahead)


def wrist_view() -> dict:
    """The simulated wrist camera as the robot's model has it: on the URDF's DCW2 link, at its optical frame
    turned by CAMERA_TILT (m3kinematics.camera_optical -- the hand-eye fit of 2026-09-30 was degenerate)."""
    T = np.eye(4)
    T[:3, :3] = m3kinematics.CAMERA_TURN @ m3kinematics.OPTICAL @ FLIP
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


LEROBOT_PY = r"C:\Users\Hercules\roboarm-lerobot\.venv\Scripts\python.exe"   # LeRobot's venv on Dionysos
POLICY_URL = "http://127.0.0.1:8764"   # tools/policy_http.py there
START = {1: 90, 2: 105, 3: 10, 4: 0, 5: 90, 6: 76}   # where every recorded episode begins (m3policy.START)


def grid_cells(box: dict) -> list[np.ndarray]:
    """The evaluation grid: a 30 mm cube at three distances from J1's axis, every 10 deg of bearing -- those a
    pick reaches, clear of the chassis' front and of the box -- 20 of them, spread evenly."""
    from roboshared import box as boxes
    from roboshared import task as shared_task

    axis = np.asarray(m3pick.Picker.AXIS)
    cells = []
    for r in (0.16, 0.19, 0.22):
        for bearing in range(-60, 61, 10):
            at = axis + r * np.array([math.cos(math.radians(bearing)), math.sin(math.radians(bearing))])
            if at[0] >= m3pick.BUMPER_X + 0.03 and r <= m3pick.furthest(0.03) \
                    and not boxes.inside(box, at, boxes.BOX_WALL + shared_task.BOX_NEAR + 0.03):
                cells.append(at)
    return [cells[int(i)] for i in np.linspace(0, len(cells) - 1, min(20, len(cells)))]   # 20, evenly over them


def evaluate(args) -> None:
    """A checkpoint in the simulator's closed loop (isaac_m3.py's run_policy, the policy behind
    tools/policy_http.py on Dionysos's CPU): a 30 mm cube on each cell of the grid in turn, the arm from the
    episodes' start for --seconds; per cell whether the tool point came within 2 cm of the cube (reached), the
    cube rose 3 cm (lifted), and ended in the box. Ranks checkpoints before the robot (the research: a matched
    simulation orders policies as reality does, r ~0.9; offline action error does not)."""
    import cv2
    import m3_expert

    ckpt = Path(args.checkpoint)
    model = ckpt / "pretrained_model" if (ckpt / "pretrained_model").exists() else ckpt
    name = f"{ckpt.parent.parent.name}_{ckpt.name}" if ckpt.parent.name == "checkpoints" else ckpt.name
    folder = f"{REMOTE.replace('/', chr(92))}\\policies\\{name}"
    have = subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, f"if exist {folder}\\model.safetensors (echo yes)"],
                          capture_output=True, text=True, check=False)
    if "yes" not in have.stdout:
        subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, f"mkdir {folder}"], check=False)
        for f in model.iterdir():
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(f), f"{HOST}:{REMOTE}/policies/{name}/"], check=True)
    put(ROOT / "tools" / "policy_http.py", ROOT / "sim" / "isaac_m3.py")
    rng = np.random.default_rng(args.seed)
    box = m3_expert.facing_box(np.array(m3_expert.BOX_AT))
    cells = grid_cells(box)[: args.trials]
    trials = [{"pieces": [{"centre": [float(x), float(y), 0.015], "yaw": float(rng.uniform(0, 90))}], "seconds": args.seconds}
              for x, y in cells]
    sag = true_sag()   # the policy's commands move an arm that sits as the real one does
    held = {**{j: START[j] + sag.get(j, 0.0) for j in range(1, 6)}, 6: START[6]}
    plan = {"keys": [{"t": 0.0, "q": urdf_targets(held)}], "settle_s": 0.0, "friction": 0.7,
            "pieces": [{"size": [0.03] * 3, "centre": trials[0]["pieces"][0]["centre"], "yaw": 0.0, "mass": 0.015,
                        "faces": green_cube()}],
            "statics": box_parts(box), "table": TABLE, "pictures": wrist_view(),
            "policy": {"url": POLICY_URL, "n_action_steps": 10, "hz": HZ, "max_dps": args.max_dps},
            "start": START, "trials": trials,
            "servo_map": {"sag": {str(j): v for j, v in sag.items()}, "mirrored": list(m3kinematics.MIRRORED),
                          "j6": [list(p) for p in J6_TO_FINGER]}}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path, result = OUT / f"eval_{stamp}.json", f"evalresult_{stamp}.json"
    plan_path.write_text(json.dumps(plan))
    put(plan_path)
    print(f"{name}: {len(trials)} cells x {args.seconds:.0f} s in the simulator", flush=True)
    remote(f"start /b {LEROBOT_PY} policy_http.py policies\\{name} > policy_http_{stamp}.log 2>&1 & "
           f"{ISAAC} isaac_m3.py play m3pro.usd {plan_path.name} {result}", f"eval_{stamp}.log")
    subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, f"curl -s {POLICY_URL}/quit"], capture_output=True, check=False)
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{HOST}:{REMOTE}/{result}", str(OUT / result)], check=True)
    played = json.loads((OUT / result).read_text())
    from roboshared import box as boxes

    rows, tiles = [], []
    for trial in played["trials"]:
        cube0 = np.array(trial["pieces"][0]["centre"])
        cubes = np.array([t["pieces"][0][:3] for t in trial["ticks"]])
        grips = np.array([t["grip"] for t in trial["ticks"]])
        reached = float(np.min(np.linalg.norm(cubes - grips, axis=1)))
        lifted = float(cubes[:, 2].max() - cube0[2])
        in_box = bool(boxes.inside(box, cubes[-1][:2]) and cubes[-1][2] < 0.06)
        j1 = [t["sent"][0] for t in trial["ticks"][:12]]
        rows.append((cube0, reached, lifted, in_box, j1[-1]))
        if len(tiles) < 8:
            for tick in trial["sheet"][1:3]:
                jpg = f"trial{trial['trial']:02d}_{tick:04d}.jpg"
                local = OUT / f"eval_{stamp}_{jpg}"
                if subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{HOST}:{REMOTE}/{jpg}", str(local)], check=False).returncode == 0:
                    tile = cv2.resize(cv2.imread(str(local)), (320, 240))
                    cv2.putText(tile, f"{trial['trial']} t{tick / HZ:.0f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    tiles.append(tile)
                    local.unlink()
    axis = np.asarray(m3pick.Picker.AXIS)
    for (c, reached, lifted, in_box, j1), trial in zip(rows, played["trials"], strict=True):
        aim = 90 - math.degrees(math.atan2(c[1] - axis[1], c[0] - axis[0]))
        print(f"  cell {trial['trial']:2d} x {c[0]:.3f} y {c[1]:+.3f} (J1 to it ~{aim:3.0f}, J1 after 1.2 s {j1:5.1f}): "
              f"closest {1000 * reached:3.0f} mm, lifted {1000 * lifted:3.0f} mm, {'IN THE BOX' if in_box else '-'}", flush=True)
    n = len(rows)
    print(f"{name}: reached {sum(r[1] < 0.02 for r in rows)}/{n}, lifted {sum(r[2] > 0.03 for r in rows)}/{n}, "
          f"in the box {sum(r[3] for r in rows)}/{n}", flush=True)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet_path = OUT / f"eval_{stamp}_{name}.jpg"
        cv2.imwrite(str(sheet_path), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
        print(f"sheet: {sheet_path}", flush=True)


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
    ep = sub.add_parser("episode")
    ep.add_argument("--seed", type=int, default=1)
    ep.add_argument("--pieces", type=int, default=4)
    ep.add_argument("--card", choices=["drawn", "zero"], default="drawn")
    ev = sub.add_parser("evaluate")
    ev.add_argument("checkpoint", help="a run's checkpoints/NNNNNN (its pretrained_model inside)")
    ev.add_argument("--trials", type=int, default=20, help="cells of the grid (at most its 21)")
    ev.add_argument("--seconds", type=float, default=25.0)
    ev.add_argument("--max-dps", type=float, default=100.0)
    ev.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "setup":
        setup()
    elif a.cmd == "look":
        look(a)
    elif a.cmd == "episode":
        episode(a)
    elif a.cmd == "evaluate":
        evaluate(a)
    else:
        grip(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
