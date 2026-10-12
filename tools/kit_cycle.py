"""The kit detector's whole cycle from this laptop (PLAN.md M5): the M3's labelled pick looks ->
the training set -> hekate trains it -> the weights and ONNX back here.

    uv run python tools/kit_cycle.py [--size n|s] [--int8] [--epochs 150] [--no-fetch]

Run it from the repo root (the workspace's venv: inside train/ `uv run` would install hekate's CUDA
torch here). The steps:
  1. the M3's looks (m3pro/tools/m3_duel.py fetch: every day's; the SSH password from
     ROBOARM_SSH_PASSWORD, or asked) -- skipped with --no-fetch;
  2. the set (m3pro/tools/kit_dataset.py -> m3pro/out/kit);
  3. it replaces datasets\\kit on hekate (`ssh hekate`, CLAUDE.md), with train/'s project and lock;
  4. hekate trains YOLO26-seg from the pretrained weights as run NAME (train/train_kit.py), and
     with --int8 then fine-tunes that with the quantizers in as NAME-int8 (quantization-aware);
  5. each run exported to ONNX at 480x640, and its best.pt, best.onnx, results.csv and validation
     pictures fetched to train/runs/kit/NAME/.
Training is GPU work on a shared PC: hekate's CLAUDE.md asks first when under ~16 GB is free --
this stops then.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = "hekate"
REMOTE = "Desktop/Sim Arm 2026/train"          # under daniel's home, for scp
REMOTE_WIN = r"$HOME\Desktop\Sim Arm 2026\train"  # the same, for PowerShell
FREE_MIB = 16_000
FETCHED = ("weights/best.pt", "weights/best.onnx", "results.csv", "val_batch0_pred.jpg")


def ssh(command: str, timeout: float = 600) -> str:
    done = subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, command], capture_output=True, text=True,
                          timeout=timeout, check=False)
    if done.returncode != 0:
        raise SystemExit(f"hekate: {(done.stderr or done.stdout).strip()[-800:]}")
    return done.stdout


def scp(*paths: str, recursive: bool = False, timeout: float = 600) -> None:
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", *(["-r"] if recursive else []), *paths],
                   check=True, timeout=timeout)


def on_hekate(args: str, timeout: float = 7200) -> str:
    """train_kit.py on hekate with `args`: its last lines."""
    return ssh(f'cd "{REMOTE_WIN}"; & .\\.venv\\Scripts\\python.exe train_kit.py {args} 2>&1 | '
               'ForEach-Object { "$_" } | Select-Object -Last 8', timeout=timeout)


def best_row(results: Path) -> dict:
    with results.open(newline="") as f:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
    return max(rows, key=lambda r: float(r.get("metrics/mAP50-95(M)") or 0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", choices=("n", "s"), default="n")
    ap.add_argument("--int8", action="store_true", help="then quantization-aware training, and its INT8 ONNX")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--qat-epochs", type=int, default=40)
    ap.add_argument("--name", default=None, help="the run's name (default: SIZE and today's date)")
    ap.add_argument("--no-fetch", action="store_true", help="use the looks already fetched")
    a = ap.parse_args()
    name = a.name or f"{a.size}_{time.strftime('%Y%m%d')}"
    m3 = ROOT / "m3pro"

    if not a.no_fetch:
        print("1. the M3's looks", flush=True)
        subprocess.run([sys.executable, "tools/m3_duel.py", "fetch", "20"], cwd=m3, check=True)
    print("2. the set", flush=True)
    subprocess.run([sys.executable, "tools/kit_dataset.py"], cwd=m3, check=True)

    free = int(ssh("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits").split()[0])
    if free < FREE_MIB:
        raise SystemExit(f"hekate's GPU has {free} MiB free (under {FREE_MIB}): someone is training -- ask first")
    print(f"3. to hekate ({free} MiB free on its GPU)", flush=True)
    ssh(f'Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "{REMOTE_WIN}\\datasets\\kit"; '
        f'New-Item -ItemType Directory -Force "{REMOTE_WIN}\\datasets" | Out-Null')
    scp(*(str(ROOT / "train" / f) for f in ("pyproject.toml", "uv.lock", "train_kit.py")), f"{HOST}:{REMOTE}/")
    scp(str(m3 / "out" / "kit"), f"{HOST}:{REMOTE}/datasets/", recursive=True)
    print(ssh(f'cd "{REMOTE_WIN}"; & $HOME\\.local\\bin\\uv.exe sync --frozen --no-progress 2>&1 | Select-Object -Last 1'))

    runs = [(name, f"--model yolo26{a.size}-seg.pt --epochs {a.epochs} --name {name}", "")]
    if a.int8:
        qat = f"--int8 --model runs\\kit\\{name}\\weights\\best.pt --epochs {a.qat_epochs} --name {name}-int8"
        runs.append((f"{name}-int8", qat, "--int8 "))
    for run, train_args, export_args in runs:
        print(f"4. training {run}", flush=True)
        print(on_hekate(train_args))
        print(f"5. exporting {run}", flush=True)
        print(on_hekate(f"{export_args}--export runs\\kit\\{run}\\weights\\best.pt", timeout=1200))
        here = ROOT / "train" / "runs" / "kit" / run
        (here / "weights").mkdir(parents=True, exist_ok=True)
        for f in FETCHED:
            scp(f"{HOST}:{REMOTE}/runs/kit/{run}/{f}", str(here / f))
        row = best_row(here / "results.csv")
        print(f"   {run}: best epoch {row['epoch']}: mask mAP50 {float(row['metrics/mAP50(M)']):.3f}, "
              f"mAP50-95 {float(row['metrics/mAP50-95(M)']):.3f} -> {here}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
