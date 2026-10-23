"""The M3's recorded jobs (roboarm/m3record.py: out/episodes/ on the robot) as a LeRobot dataset a
VLA trains on (NEXT.md, "After the event"): every pick attempt one episode, from the arm leaving the
look pose for the piece (the step "pregrasp") to it back at the look after the drop ("look.after").

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
import subprocess
import sys
from pathlib import Path

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


def attempts(ticks: list[dict], summary: dict) -> list[dict]:
    """The job's pick attempts: {"first", "last" (tick indices, inclusive), "record", "drop" ("box",
    "line" or None)} -- each from its pregrasp to the end of the look after it. One that never got
    to its look after (a move refused on the way) has no record and is left out; the rest are the
    records in order."""
    starts = [i for i, t in enumerate(ticks) if t["step"] == "pregrasp" and (i == 0 or ticks[i - 1]["step"] != "pregrasp")]
    whole = []
    for k, first in enumerate(starts):
        stop = starts[k + 1] if k + 1 < len(starts) else len(ticks)
        after = [i for i in range(first, stop) if ticks[i]["step"] == "look.after"]
        if not after:
            continue
        last = after[0]
        while last + 1 < stop and ticks[last + 1]["step"] == "look.after":
            last += 1
        whole.append((first, last))
    records = summary.get("attempts", [])
    if len(records) != len(whole):
        raise ValueError(f"{len(whole)} attempts in the ticks, {len(records)} records: not matched")
    t0 = started_unix(summary)
    drops = [(e["t"] - t0, "box" if e["text"].startswith("dropped into the box") else "line")
             for e in summary.get("log", []) if e["text"].startswith("dropped ")]
    out = []
    for (first, last), record in zip(whole, records):
        inside = [where for t, where in drops if ticks[first]["t"] <= t <= ticks[last]["t"]]
        out.append({"first": first, "last": last, "record": record, "drop": inside[0] if len(inside) == 1 else None})
    return out


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
def build(out: Path, fps: float, keep_failed: bool, jobs: list[str] | None) -> None:
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    chosen = []   # (folder, ticks, summary, attempt)
    for folder in sorted(p for p in EPISODES.iterdir() if (p / "episode.json").exists()):
        ticks, summary = load(folder)
        if jobs and summary["job"] not in jobs:
            continue
        try:
            found = attempts(ticks, summary)
        except ValueError as exc:
            print(f"{folder.name}: left out -- {exc}", flush=True)
            continue
        chosen += [(folder, ticks, summary, a) for a in found if keep_failed or a["record"].get("picked")]
    if not chosen:
        raise SystemExit(f"no attempts in {EPISODES}: fetch first?")
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
    args = parser.parse_args()
    if args.cmd == "fetch":
        fetch()
    else:
        build(args.out, args.fps, args.keep_failed, args.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
