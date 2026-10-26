"""The M3's recorded jobs (roboarm/m3record.py: out/episodes/ on the robot) as a LeRobot dataset a
VLA trains on (NEXT.md, "After the event"): every pick attempt one episode, from the arm about to leave
the look pose for the piece (LEAD ticks before the step "pregrasp") to its arrival back there after the
drop (the look after's first tick at that pose: attempts()).

    python tools/m3_episodes.py fetch                  # the robot's episodes not here yet -> out/episodes/
    <LeRobot's python> tools/m3_episodes.py build OUT [--fps 10] [--keep-failed] [--jobs clean_up sort ...]

`build` needs LeRobot -- its own venv, outside the workspace: `uv venv --python 3.12 ~/.roboarm/lerobot`
and `uv pip install --python ~/.roboarm/lerobot/Scripts/python.exe lerobot`; the rest is stdlib. Each
frame: observation.images.wrist (the DCW2's colour as recorded), observation.state (J1..J6 as sent, in
servo degrees -- the M3 reads none back), action (the next frame's state: the command as the board
sweeps through it; the last frame's its own) and the task ("pick up the 30 mm cube and put it in the
box"). The ticks (~10 Hz, a little uneven) are taken at --fps by the nearest. OUT/m3_attempts.jsonl:
each episode's job, recording, pick record and success. Attempts the robot itself found missed are left
out unless --keep-failed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EPISODES = ROOT / "out" / "episodes"
ROBOT, REMOTE = "jetson@192.168.73.66", "/home/jetson/roboarm/out/episodes"
ROBOT_TZ = datetime.timezone(datetime.timedelta(hours=8))   # the M3's clock: Asia/Shanghai, Yahboom's
JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]


def load(folder: Path) -> tuple[list[dict], dict]:
    ticks = [json.loads(line) for line in (folder / "ticks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return ticks, json.loads((folder / "episode.json").read_text(encoding="utf-8"))


def started_unix(summary: dict) -> float:
    """When the job began, unix s: kept since 2026-10-23 afternoon; before, from its local time."""
    if "started_unix" in summary:
        return float(summary["started_unix"])
    local = datetime.datetime.strptime(summary["started"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=ROBOT_TZ)
    return local.timestamp() + 0.5   # the string drops the fraction


LEAD = 3   # ticks kept before an attempt's pregrasp, the arm still standing at the look pose


def attempts(ticks: list[dict], summary: dict) -> list[dict]:
    """The job's pick attempts: {"first", "last" (tick indices, inclusive), "record", "drop" ("box",
    "line" or None)} -- each from LEAD ticks before its pregrasp (the arm standing at the look pose, as
    long as it stood) to the arm back at that pose after the drop: the first tick of the look after
    there. The rest of that look -- the arm standing while the picker looks, 6-64 ticks -- is left out:
    kept until 2026-10-26, it outnumbered the departures from the look pose 718 to 25, and ACT learnt to
    stay there (a run on the M3 never left it). One that never got to its look after (a move refused
    on the way) has no record and is left out; the rest are the records in order."""
    starts = [i for i, t in enumerate(ticks) if t["step"] == "pregrasp" and (i == 0 or ticks[i - 1]["step"] != "pregrasp")]
    whole = []   # (pregrasp, look after's first tick, its last)
    for k, first in enumerate(starts):
        stop = starts[k + 1] if k + 1 < len(starts) else len(ticks)
        after = [i for i in range(first, stop) if ticks[i]["step"] == "look.after"]
        if not after:
            continue
        last = after[0]
        while last + 1 < stop and ticks[last + 1]["step"] == "look.after":
            last += 1
        whole.append((first, after[0], last))
    records = summary.get("attempts", [])
    if len(records) != len(whole):
        raise ValueError(f"{len(whole)} attempts in the ticks, {len(records)} records: not matched")
    t0 = started_unix(summary)
    drops = [(e["t"] - t0, "box" if e["text"].startswith("dropped into the box") else "line")
             for e in summary.get("log", []) if e["text"].startswith("dropped ")]
    looks = [(e["t"] - t0, seen(e["text"])) for e in summary.get("log", [])
             if e["text"].startswith(("looked round:", "wide look:"))]
    out, kept = [], -1   # kept: the last tick of the episode before
    for k, ((first, look, last), record) in enumerate(zip(whole, records)):
        inside = [where for t, where in drops if ticks[first]["t"] <= t <= ticks[last]["t"]]
        until = ticks[whole[k + 1][0]]["t"] if k + 1 < len(whole) else math.inf
        pushed = record.get("picked") and pushed_away(record, looks, ticks[first]["t"], ticks[last]["t"], until)
        if pushed:
            record = {**record, "picked": False, "pushed_to": pushed}
        rest, lead = ticks[first - 1]["pose"] if first else None, first
        while first - lead < LEAD and lead - 2 > kept and ticks[lead - 1]["pose"] == rest == ticks[lead - 2]["pose"]:
            lead -= 1
        back = next(i for i in range(look, last + 1) if ticks[i]["pose"] == ticks[last]["pose"])
        out.append({"first": lead, "last": back, "record": record, "drop": inside[0] if len(inside) == 1 else None})
        kept = back
    return out


PIECE = re.compile(r"(30 mm cube|40 mm cube|block) (?:lying |standing )?at x ([-\d.]+) y ([-+\d.]+)")
PUSH_M, SAME_M = 0.08, 0.015


def seen(text: str) -> list[tuple[str, float, float]]:
    """The kit pieces a look's log line names: (kind, x, y)."""
    return [(k, float(x), float(y)) for k, x, y in PIECE.findall(text)]


def pushed_away(record: dict, looks: list, start: float, end: float, until: float) -> list | None:
    """Where the piece an attempt counted as picked turns up again: one of its kind within PUSH_M of the
    target, in a look from the attempt's own look after to the next attempt, where no look before it had
    one (SAME_M). The picker counts a piece gone from where it was as picked; on 2026-10-23 a 30 mm
    cube shoved 4.3 cm out, past the next look's middle view, was "picked". None: not seen."""
    if "target" not in record:
        return None
    kind = (record.get("kind") or "").replace(" lying", "").replace(" standing", "")
    target = np.asarray(record["target"][:2], float)
    before = [p for t, pieces in looks if t <= start + 0.5 for p in pieces]   # every look since the job began:
    # the one just before may name a piece only "something" (a middle view, seen in part)
    for t, pieces in looks:
        if not end - 4.0 <= t <= until:
            continue
        for k, x, y in pieces:
            here = np.array([x, y])
            new = not any(np.linalg.norm(here - np.array([bx, by])) < SAME_M for _k, bx, by in before)
            if k == kind and new and np.linalg.norm(here - target) < PUSH_M:
                return [x, y]
    return None


def resample(ticks: list[dict], first: int, last: int, fps: float) -> list[int]:
    """Tick indices first..last at 1/fps apart by the nearest (a tick late repeats the one before)."""
    t0, t1 = ticks[first]["t"], ticks[last]["t"]
    chosen, i = [], first
    for k in range(int((t1 - t0) * fps + 1e-6) + 1):
        want = t0 + k / fps
        while i < last and abs(ticks[i + 1]["t"] - want) <= abs(ticks[i]["t"] - want):
            i += 1
        chosen.append(i)
    return chosen


def task_for(attempt: dict) -> str:
    kind = attempt["record"].get("kind") or "piece"
    for posture in (" lying", " standing"):
        kind = kind.removesuffix(posture)
    where = {"box": "put it in the box", "line": "put it on the line on the right"}.get(attempt["drop"], "put it down")
    return f"pick up the {kind} and {where}"


def frames(ticks: list[dict], indices: list[int]) -> list[tuple[int, list[float], list[float]]]:
    """(tick, state, action) for each chosen tick: the action is the next one's state, the last its own."""
    states = [[float(v) for v in ticks[i]["pose"]] for i in indices]
    return [(i, s, states[k + 1] if k + 1 < len(states) else s) for k, (i, s) in enumerate(zip(indices, states))]


# ------------------------------------------------------------------------------- fetch --
def fetch() -> None:
    have = {p.name for p in EPISODES.iterdir()} if EPISODES.exists() else set()
    listing = subprocess.run(["ssh", "-o", "BatchMode=yes", ROBOT, f"ls {REMOTE}; ls {REMOTE}/*/episode.json"],
                             capture_output=True, text=True, check=True).stdout.split()
    done = {Path(p).parent.name for p in listing if p.endswith("episode.json")}   # a job still running has none
    new = sorted(done - have)
    EPISODES.mkdir(parents=True, exist_ok=True)
    for name in new:
        subprocess.run(["scp", "-q", "-r", "-o", "BatchMode=yes", f"{ROBOT}:{REMOTE}/{name}", str(EPISODES)], check=True)
        print("fetched", name, flush=True)
    print(f"{len(new)} new, {len(done)} on the robot", flush=True)


# ------------------------------------------------------------------------------- build --
def build(out: Path, fps: float, keep_failed: bool, jobs: list[str] | None, episodes: tuple[Path, ...] = (EPISODES,)) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    chosen = []   # (folder, ticks, summary, attempt)
    for folder in sorted((p for e in episodes for p in e.iterdir() if (p / "episode.json").exists()), key=lambda p: p.name):
        ticks, summary = load(folder)
        if jobs and summary["job"] not in jobs:
            continue
        if "robot_card" in summary and "twin_sag" not in summary["robot_card"]:   # m3_sim.TWIN_SAG: 2026-10-26
            print(f"{folder.name}: left out -- simulated before the real arm's offset was known", flush=True)
            continue
        try:
            found = attempts(ticks, summary)
        except ValueError as exc:
            print(f"{folder.name}: left out -- {exc}", flush=True)
            continue
        chosen += [(folder, ticks, summary, a) for a in found if keep_failed or a["record"].get("picked")]
    if not chosen:
        raise SystemExit(f"no attempts in {episodes}: fetch first?")
    height, width = np.asarray(Image.open(chosen[0][0] / f"{chosen[0][3]['first']:06d}.jpg")).shape[:2]
    features = {
        "observation.images.wrist": {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINTS},
    }
    dataset = LeRobotDataset.create(repo_id=f"local/{out.name}", fps=int(fps), features=features, root=out,
                                    robot_type="m3pro", use_videos=True)
    meta = []
    for folder, ticks, summary, attempt in chosen:
        task = task_for(attempt)
        for i, state, action in frames(ticks, resample(ticks, attempt["first"], attempt["last"], fps)):
            picture = np.asarray(Image.open(folder / f"{i:06d}.jpg").convert("RGB"))
            dataset.add_frame({"observation.images.wrist": picture, "observation.state": np.float32(state),
                               "action": np.float32(action), "task": task})
        dataset.save_episode()
        meta.append({"episode": len(meta), "recording": folder.name, "job": summary["job"], "task": task,
                     "ticks": [attempt["first"], attempt["last"]], "picked": bool(attempt["record"].get("picked")),
                     "drop": attempt["drop"], "record": attempt["record"]})
        print(f"episode {len(meta) - 1}: {folder.name} ticks {attempt['first']}-{attempt['last']}: {task}", flush=True)
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    with open(out / "m3_attempts.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(m) + "\n" for m in meta)
    print(f"-> {out}: {len(meta)} episodes at {fps:g} fps", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    b = sub.add_parser("build")
    b.add_argument("out", type=Path, help="the dataset's folder (must not exist yet)")
    b.add_argument("--fps", type=float, default=10.0)
    b.add_argument("--keep-failed", action="store_true")
    b.add_argument("--jobs", nargs="+", help="only these jobs (clean_up, sort, pick, named ...)")
    b.add_argument("--episodes", type=Path, nargs="+", default=[EPISODES],
                   help="the recordings' folders (out/sim_episodes: the simulator's, m3_sim.py episode; "
                        "several: one dataset of them all -- sim and real together)")
    args = parser.parse_args()
    if args.cmd == "fetch":
        fetch()
    else:
        build(args.out, args.fps, args.keep_failed, args.jobs, tuple(args.episodes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
