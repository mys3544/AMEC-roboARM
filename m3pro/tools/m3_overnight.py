"""The night's work on Dionysos with the laptop away (2026-10-27): started there detached (a WMI-created process
outlives SSH: m3_overnight.ps1), in LeRobot's venv, with ROBOARM_SIM_LOCAL=1 (tools/m3_sim.py runs Isaac right
there) and PYTHONPATH holding roboarm-code\\shared and roboarm-code\\m3pro:

  1. waits for the co-training to end (its log says "End of training");
  2. its last checkpoint in the sim grid (m3_sim.py evaluate, 18 cells);
  3. simulated clean-ups, WORKERS side by side, until --until (local time) or out/sim/STOP: the box of its own
     each episode, the firmer grip (m3_sim.py episode);
  4. dataset v2: all the simulated episodes made with the real arm's offset + the real ones (m3_episodes.py
     build -> roboarm-lerobot\\<--dataset>);
  5. ACT co-trained on it from scratch, as the first (150k steps, 4 loader workers) -> roboarm-lerobot\\<--run>;
  6. its last checkpoint in the sim grid too.

Each step's line in out/sim/overnight.log (times are Dionysos's own clock, the laptop's too).

    python tools/m3_overnight.py --after m3_act_cotrain_20261026 --until 05:30 --dataset m3_cotrain_20261027
        --run m3_act_cotrain_20261027
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEROBOT = Path(r"C:\Users\Hercules\roboarm-lerobot")
LOG = ROOT / "out" / "sim" / "overnight.log"
WORKERS = 3
STEPS = 150_000


def say(text: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")


def run(args: list[str], log: Path, cwd: Path = ROOT) -> int:
    with open(log, "a", encoding="utf-8") as f:
        return subprocess.run(args, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=False).returncode


def trained(run_name: str) -> bool:
    log = LEROBOT / f"{run_name}.log"
    return log.exists() and "End of training" in log.read_text(errors="replace")


def evaluate(run_name: str) -> None:
    ckpt = LEROBOT / run_name / "checkpoints" / f"{STEPS:06d}"
    if not ckpt.exists():
        say(f"no checkpoint {ckpt}: not evaluated")
        return
    say(f"evaluating {ckpt} in the sim grid")
    out = ROOT / "out" / "sim" / f"eval_{run_name}.log"
    code = run([sys.executable, "tools/m3_sim.py", "evaluate", str(ckpt), "--trials", "18", "--seconds", "25",
                "--max-dps", "100"], out)
    tail = [line for line in out.read_text(errors="replace").splitlines() if "reached" in line]
    say(f"evaluate exit {code}: {tail[-1] if tail else 'no summary (see ' + out.name + ')'}")


def worker(w: int, until: float, counts: dict) -> None:
    seed = 800_000 + 10_000 * w
    out = ROOT / "out" / "sim" / f"overnight_w{w}.log"
    while time.time() < until and not (ROOT / "out" / "sim" / "STOP").exists():
        code = run([sys.executable, "tools/m3_sim.py", "episode", "--seed", str(seed), "--pieces", str(3 + seed % 3)], out)
        counts[w] = counts.get(w, 0) + (code == 0)
        seed += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--after", required=True, help="the training to wait for (its run name)")
    ap.add_argument("--until", default="05:30", help="local time the episodes stop at (HH:MM, the next one after now)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    os.environ["ROBOARM_SIM_LOCAL"] = "1"
    say(f"start: after {a.after}, episodes until {a.until}, then {a.dataset} -> {a.run}")
    while not trained(a.after):
        time.sleep(120)
    say(f"{a.after} ended")
    evaluate(a.after)
    hh, mm = (int(v) for v in a.until.split(":"))
    now = time.localtime()
    until = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hh, mm, 0, 0, 0, -1))   # local time, this machine's
    if until <= time.time():
        until += 86_400
    counts: dict[int, int] = {}
    say(f"{WORKERS} workers making episodes until {time.strftime('%Y-%m-%d %H:%M', time.localtime(until))}")
    threads = [threading.Thread(target=worker, args=(w, until, counts)) for w in range(1, WORKERS + 1)]
    for t in threads:
        t.start()
        time.sleep(30)   # their Isaac starts staggered
    for t in threads:
        t.join()
    say(f"episodes done: {sum(counts.values())} ({counts})")
    target = LEROBOT / a.dataset
    if target.exists():
        say(f"{target} exists already: not built, not trained")
        return 1
    code = run([sys.executable, "tools/m3_episodes.py", "build", str(target), "--episodes",
                str(ROOT / "out" / "sim_episodes"), str(ROOT / "out" / "episodes")], ROOT / "out" / "sim" / f"build_{a.dataset}.log")
    attempts = target / "m3_attempts.jsonl"
    episodes = len(attempts.read_text().splitlines()) if attempts.exists() else 0
    say(f"build exit {code}: {episodes} episodes in {target}")
    if code or not episodes:
        return 1
    say(f"training {a.run} ({STEPS} steps)")
    code = run([str(LEROBOT / ".venv" / "Scripts" / "python.exe"), "lerobot_train_win.py",
                f"--dataset.repo_id=local/{a.dataset}", f"--dataset.root={target.as_posix()}", "--dataset.eval_split=0.1",
                "--policy.type=act", "--policy.device=cuda", "--policy.use_amp=true", "--policy.push_to_hub=false",
                "--policy.chunk_size=20", "--policy.n_action_steps=10", "--batch_size=8", "--num_workers=4",
                f"--steps={STEPS}", "--log_freq=500", "--eval_steps=5000", "--save_freq=25000",
                f"--output_dir={(LEROBOT / a.run).as_posix()}", "--wandb.enable=false", f"--job_name={a.run}"],
               LEROBOT / f"{a.run}.log", cwd=LEROBOT)
    say(f"training exit {code}")
    evaluate(a.run)
    say("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
