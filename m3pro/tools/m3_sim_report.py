"""What the simulated episodes say, from the laptop's copies (out/sim_episodes, out/sim/episoderesult_*.json):

    python tools/m3_sim_report.py misses       # every simulated pick: in the box, or where it went wrong --
                                               # not lifted (how far the grip point was off the piece when the
                                               # fingers shut, how far the descent shoved it), lost on the way,
                                               # beside the box -- and the misses by kind, pitch and the card
    python tools/m3_sim_report.py placement DATASET   # where the pieces were picked from, real against simulated
                                               # (a dataset of tools/m3_episodes.py build: its m3_attempts.jsonl),
                                               # -> DATASET/placement.png: the table seen from above, the J1 spread

The simulated arm follows the plan's keys late by the card's delay: plan time t is the simulation's t + delay.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "shared"))
from roboshared import box as boxes

from roboarm import m3kinematics

EPISODES = ROOT / "out" / "sim_episodes"
RESULTS = ROOT / "out" / "sim"
HZ = 10
LIFTED = 0.02    # m the piece must rise to count as up with the fingers
IN_BOX_Z = 0.06  # m: lower than this over the box's inside is in it (m3_sim.verdicts)


def segments(ticks: list[dict]) -> list[dict[str, tuple[int, int]]]:
    """Each attempt's steps: {step: (first tick, last tick)}, an attempt from one pregrasp to the next."""
    out, run_start, current = [], 0, None
    for i, t in enumerate(ticks + [{"step": None}]):
        if i and t["step"] != ticks[i - 1]["step"]:
            step = ticks[i - 1]["step"]
            if step == "pregrasp":
                current = {}
                out.append(current)
            if current is not None and step not in current:
                current[step] = (run_start, i - 1)
            run_start = i
    return out


def sample_at(samples: list[dict], t: float) -> dict:
    i = int(np.clip(np.searchsorted([s["t"] for s in samples], t), 0, len(samples) - 1))
    return samples[i]


def misses(folder: Path = EPISODES) -> None:
    rows = []
    for ep in sorted(p for p in folder.iterdir() if (p / "episode.json").exists()):
        stamp = ep.name.removesuffix("_clean_up")
        result, plan = RESULTS / f"episoderesult_{stamp}.json", RESULTS / f"episode_{stamp}.json"
        if not result.exists():
            continue
        meta = json.loads((ep / "episode.json").read_text())
        samples = json.loads(result.read_text())["samples"]
        friction = json.loads(plan.read_text()).get("friction") if plan.exists() else None
        ticks = [json.loads(line) for line in (ep / "ticks.jsonl").read_text().splitlines()]
        card, layout = meta["robot_card"], meta["sim"]["layout"]
        box = {k: np.asarray(v, float) if isinstance(v, list) else v for k, v in meta["sim"]["box"].items()}
        delay = card["delay_s"]
        for k, (attempt, steps) in enumerate(zip(meta["attempts"], segments(ticks), strict=False)):
            target = np.asarray(attempt["target"][:2])
            which = min(range(len(layout)), key=lambda i: float(np.hypot(*(np.asarray(layout[i]["centre"][:2]) - target))))
            if "close" not in steps:
                continue
            t0 = steps["pregrasp"][0] / HZ + delay
            shut = steps["close"][1] / HZ + delay
            start, closed = sample_at(samples, t0), sample_at(samples, shut)
            p0, pc = np.asarray(start["pieces"][which][:3]), np.asarray(closed["pieces"][which][:3])
            grip = np.asarray(closed["grip"])
            up = [s["pieces"][which][2] for s in samples
                  if shut <= s["t"] <= (steps.get("drop", steps["close"])[1] / HZ + delay)]
            end = steps.get("look.after", steps.get("drop", steps["close"]))[0] / HZ + delay + 0.5
            final = np.asarray(sample_at(samples, end)["pieces"][which][:3])
            lifted = bool(up) and max(up) > p0[2] + LIFTED
            in_box = bool(boxes.inside(box, final[:2])) and final[2] < IN_BOX_Z
            if in_box:
                how = "in the box"
            elif not lifted:
                how = "not lifted"
            elif boxes.inside(box, final[:2]):
                how = "on the box, not in it"
            elif "drop" in steps and max(s["pieces"][which][2] for s in samples
                                         if steps["drop"][1] / HZ + delay - 0.3 <= s["t"] <= steps["drop"][1] / HZ + delay) > p0[2] + LIFTED:
                how = "let go beside the box"
            else:
                how = "lost on the way"
            off = grip - pc   # the grip point against the piece's middle as the fingers shut
            shove = float(np.hypot(*(pc[:2] - p0[:2])))
            rows.append({"episode": ep.name, "k": k, "kind": attempt["kind"], "pitch": attempt["pitch"], "how": how,
                         "careful": attempt.get("careful"), "yaw_from": attempt.get("yaw_from"), "friction": friction,
                         "gripper_offset": card["gripper_offset_deg"], "delay": delay,
                         "off_mm": np.round(1000 * off, 1).tolist(), "shove_mm": round(1000 * shove, 1),
                         "radius": round(float(np.hypot(*(p0[:2] - np.asarray(m3kinematics.CHAIN[0][0][:2])))), 3),
                         "final": np.round(final, 3).tolist()})
    if not rows:
        raise SystemExit(f"no simulated episodes with their results in {folder} and {RESULTS}")
    count = collections.Counter(r["how"] for r in rows)
    print(f"{len(rows)} simulated picks in {len({r['episode'] for r in rows})} episodes: " +
          ", ".join(f"{n} {how}" for how, n in count.most_common()), flush=True)
    for key in ("kind", "careful", "yaw_from"):
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r[key]].append(r["how"] == "in the box")
        print(f"by {key}: " + "; ".join(f"{g} {sum(v)}/{len(v)}" for g, v in sorted(groups.items(), key=lambda kv: str(kv[0]))))
    for name, edges in (("pitch", (40, 60, 70, 80, 91)), ("radius", (0.0, 0.17, 0.20, 0.23, 0.30)),
                        ("friction", (0.0, 0.6, 0.7, 0.81)), ("gripper_offset", (-9, -2, 0, 2, 9))):
        parts = []
        for lo, hi in itertools.pairwise(edges):
            v = [r["how"] == "in the box" for r in rows if r[name] is not None and lo <= r[name] < hi]
            if v:
                parts.append(f"{lo}-{hi} {sum(v)}/{len(v)}")
        print(f"by {name}: " + "; ".join(parts))
    print("the misses:")
    for r in rows:
        if r["how"] != "in the box":
            print(f"  {r['episode']} #{r['k']}: {r['kind']}, pitch {r['pitch']}, {r['radius'] * 100:.0f} cm out, "
                  f"{'careful, ' if r['careful'] else ''}friction {r['friction']}: {r['how']} -- the grip point "
                  f"{r['off_mm']} mm off its middle as the fingers shut, shoved {r['shove_mm']} mm on the way down; "
                  f"ended at {r['final']}")
    out = RESULTS / "misses.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"-> {out}")


# ------------------------------------------------------------------------------ placement --
INK, MUTED, SURFACE, GRID = (11, 11, 11), (78, 81, 82), (251, 252, 252), (226, 228, 228)
REAL, SIM = (214, 120, 42), (52, 104, 235)   # BGR: the palette's categorical slots 1 (blue) and 2 (orange)


def text(img, s, at, size=0.5, colour=INK, thick=1):
    cv2.putText(img, s, at, cv2.FONT_HERSHEY_SIMPLEX, size, colour, thick, cv2.LINE_AA)


def placement(dataset: Path) -> None:
    rows = [json.loads(line) for line in (dataset / "m3_attempts.jsonl").read_text().splitlines()]
    axis = np.asarray(m3kinematics.CHAIN[0][0][:2])
    pts = {"real": [], "sim": []}
    for r in rows:
        which = "sim" if re.search(r"_s\d+_", r["recording"]) else "real"
        pts[which].append(np.asarray(r["record"]["target"][:2], float))
    img = np.full((640, 1280, 3), SURFACE, np.uint8)
    left = img[:, :620].copy()   # the table from above, drawn on its own so nothing runs into the J1 panel
    # x ahead (up the picture), y to the left (to the left), 5 cm grid
    ox, oy, s = 310, 560, 1400.0   # pixel of base_link's origin; px per m

    def px(p):
        return int(ox - s * p[1]), int(oy - s * p[0])
    for x in np.arange(0.05, 0.31, 0.05):
        cv2.line(left, px((x, 0.20)), px((x, -0.17)), GRID, 1, cv2.LINE_AA)
        text(left, f"{100 * x:.0f}", (px((x, -0.17))[0] + 6, px((x, 0))[1] + 4), 0.42, MUTED)
    for y in np.arange(-0.15, 0.16, 0.05):
        cv2.line(left, px((0.0, y)), px((0.30, y)), GRID, 1, cv2.LINE_AA)
        text(left, f"{100 * y:+.0f}", (px((0, y))[0] - 14, oy + 22), 0.42, MUTED)
    for r in (0.15, 0.20, 0.25):   # the reach round J1's axis
        cv2.ellipse(left, px(axis), (int(s * r), int(s * r)), 0, 180, 360, (200, 202, 202), 1, cv2.LINE_AA)
        text(left, f"{100 * r:.0f} cm", (px(axis)[0] + 4, px(axis)[1] - int(s * r) - 4), 0.38, MUTED)
    cv2.rectangle(left, px((0.124, 0.112)), px((-0.02, -0.112)), (190, 192, 192), 1, cv2.LINE_AA)
    text(left, "robot", (px((0.05, 0.03))[0], px((0.05, 0))[1]), 0.5, MUTED)
    for name, colour in (("real", REAL), ("sim", SIM)):
        for p in pts[name]:
            cv2.circle(left, px(p), 5, SURFACE, -1, cv2.LINE_AA)   # the surface ring round each marker
            cv2.circle(left, px(p), 4, colour, -1, cv2.LINE_AA)
    text(left, "Where the pieces were picked from (seen from above)", (20, 28), 0.6, INK, 1)
    text(left, "x ahead (cm)", (px((0.30, -0.17))[0] - 60, px((0.31, 0))[1] - 4), 0.42, MUTED)
    text(left, "y left (cm)", (ox - 40, oy + 46), 0.42, MUTED)
    img[:, :620] = left
    # J1 spread: the arm's turn to each piece, 10 deg bins, the share of each set
    gx, gy, gw, gh = 720, 100, 500, 390
    j1 = {k: [90 - math.degrees(math.atan2(p[1] - axis[1], p[0] - axis[0])) for p in v] for k, v in pts.items()}
    bins = np.arange(30, 151, 10)
    share = {k: np.histogram(v, bins)[0] / max(1, len(v)) for k, v in j1.items()}
    top = max(0.05, max(float(v.max()) for v in share.values()))
    top = math.ceil(top * 20) / 20
    for f in np.arange(0, top + 1e-9, 0.05):
        y = int(gy + gh - gh * f / top)
        cv2.line(img, (gx, y), (gx + gw, y), GRID, 1, cv2.LINE_AA)
        text(img, f"{100 * f:.0f}%", (gx - 40, y + 4), 0.42, MUTED)
    bw = gw / (len(bins) - 1)
    for i in range(len(bins) - 1):
        for j, (name, colour) in enumerate((("real", REAL), ("sim", SIM))):
            h = gh * share[name][i] / top
            x0 = int(gx + i * bw + 3 + j * (bw - 6) / 2)
            x1 = int(gx + i * bw + 3 + (j + 1) * (bw - 6) / 2) - 2   # a 2 px gap between the pair
            if h > 0:
                cv2.rectangle(img, (x0, int(gy + gh - h)), (x1, gy + gh), colour, -1)
        if i % 2 == 0:
            text(img, f"{bins[i]:.0f}", (int(gx + i * bw) - 8, gy + gh + 20), 0.42, MUTED)
    text(img, f"{bins[-1]:.0f}", (gx + gw - 12, gy + gh + 20), 0.42, MUTED)
    cv2.line(img, (gx, gy + gh), (gx + gw, gy + gh), (160, 162, 162), 1, cv2.LINE_AA)
    text(img, "The arm's turn to each piece (J1, deg)", (gx - 50, 28), 0.6, INK)
    text(img, "90 = straight ahead; less = left, more = right", (gx - 50, 54), 0.45, MUTED)
    text(img, "share of each set's picks", (gx - 50, gy - 14), 0.42, MUTED)
    text(img, "J1 (deg)", (gx + gw // 2 - 30, gy + gh + 44), 0.42, MUTED)
    for j, (name, colour) in enumerate((("real", REAL), ("sim", SIM))):   # the legend
        x = 20 + j * 260
        cv2.circle(img, (x + 6, 52), 5, colour, -1, cv2.LINE_AA)
        mean = float(np.mean(j1[name])) if j1[name] else float("nan")
        text(img, f"{name}: {len(pts[name])} picks, J1 mean {mean:.0f}", (x + 18, 57), 0.45, INK)
    out = dataset / "placement.png"
    cv2.imwrite(str(out), img)
    for name in ("real", "sim"):
        v = np.asarray(j1[name])
        print(f"{name}: {len(v)} picks, J1 {v.mean():.0f} mean, {np.percentile(v, 10):.0f}-{np.percentile(v, 90):.0f} "
              f"(10-90 %), left of 90: {np.mean(v < 90):.0%}; out {np.mean([np.hypot(*(p - axis)) for p in pts[name]]) * 100:.1f} cm mean")
    print(f"-> {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("misses")
    p = sub.add_parser("placement")
    p.add_argument("dataset", type=Path)
    a = parser.parse_args()
    if a.cmd == "misses":
        misses()
    else:
        placement(a.dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
