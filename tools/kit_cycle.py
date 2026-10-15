"""The kit detector's whole cycle from this laptop (PLAN.md M5): the M3's labelled pick looks ->
the training set -> hekate trains it -> the weights and ONNX back here.

    uv run python tools/kit_cycle.py [--size n|s] [--synth] [--int8] [--x3] [--epochs N] [--no-fetch]

Run it from the repo root (the workspace's venv: inside train/ `uv run` would install hekate's CUDA
torch here). The steps:
  1. the M3's looks (m3pro/tools/m3_duel.py fetch: every day's; the SSH password from
     ROBOARM_SSH_PASSWORD, or asked) -- skipped with --no-fetch;
  2. the set (m3pro/tools/kit_dataset.py -> m3pro/out/kit); with --x3 the X3's looks added to it
     (x3plus/tools/kit_dataset_x3.py: fetched over SSH unless --no-fetch, labelled by projecting
     what the X3 found back into its pictures; on hekate SAM 2 then draws each piece's exact
     outline from that, train/refine_x3_sam.py, and takes out the pictures where it disagrees);
  3. it replaces datasets\\kit on hekate (`ssh hekate`, CLAUDE.md), with train/'s project and lock;
  4. hekate trains YOLO26-seg from the pretrained weights as run NAME (train/train_kit.py) -- with
     --synth first on the real set and Isaac Sim's pictures together as NAME-mix (synth_to_yolo.py
     on hekate's datasets\\synth_raw2: rendered with the camera at the robot's looks, train/synth/),
     then NAME fine-tuned from it on the real set alone, 60 epochs each (the best way on 2026-10-12:
     mask mAP50 0.653 -> 0.791); with --int8 then NAME fine-tuned with the quantizers in as
     NAME-int8 (quantization-aware; no faster on the Orin NX, PLAN.md M5);
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
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = "hekate"
REMOTE = "Desktop/Sim Arm 2026/train"          # under daniel's home, for scp
REMOTE_WIN = r"$HOME\Desktop\Sim Arm 2026\train"  # the same, for PowerShell
FREE_MIB = 16_000
FETCHED = ("weights/best.pt", "weights/best.onnx", "results.csv", "val_batch0_pred.jpg")


def ssh(command: str, timeout: float = 600, check: bool = True) -> str:
    """`command` in hekate's PowerShell: what it printed. check=False for a program whose stderr is
    redirected (2>&1): Windows PowerShell then fails the command whatever the program did -- uv sync
    writes its progress there (2026-10-12) -- and a real failure shows as the files not there after."""
    done = subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, command], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout, check=False)   # (2026-10-13: cp1252
    # met a byte it has no character for in Ultralytics' output, and the step's report came back None)
    if check and done.returncode != 0:
        raise SystemExit(f"hekate: {(done.stderr or done.stdout).strip()[-800:]}")
    return done.stdout


def scp(*paths: str, recursive: bool = False, timeout: float = 600) -> None:
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", *(["-r"] if recursive else []), *paths],
                   check=True, timeout=timeout)


def send_folder(local: Path, remote: str, remote_win: str) -> None:
    """`local` into hekate's folder `remote` (`remote_win` for PowerShell) as one tar, unpacked there:
    scp -r goes file by file, and hekate's SSH server took over 20 min for the X3's ~1,900 small
    pictures and labels (2026-10-15) -- the tar 30 s."""
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / f"{local.name}.tar"
        with tarfile.open(pack, "w") as tar:
            tar.add(local, arcname=local.name)
        scp(str(pack), f"{HOST}:{remote}/")
    ssh(f'cd "{remote_win}"; tar -xf {local.name}.tar; Remove-Item {local.name}.tar')


def on_hekate(args: str, timeout: float = 7200) -> str:
    """train_kit.py on hekate with `args`: its last lines."""
    return ssh(f'cd "{REMOTE_WIN}"; $env:PYTHONUTF8 = "1"; & .\\.venv\\Scripts\\python.exe train_kit.py {args} 2>&1 | '
               'ForEach-Object { "$_" } | Select-Object -Last 8', timeout=timeout, check=False)


def best_row(results: Path) -> dict:
    with results.open(newline="") as f:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
    return max(rows, key=lambda r: float(r.get("metrics/mAP50-95(M)") or 0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", choices=("n", "s"), default="n")
    ap.add_argument("--int8", action="store_true", help="then quantization-aware training, and its INT8 ONNX")
    ap.add_argument("--synth", nargs="?", const=r"datasets\synth_raw2", default=None, metavar="RAW",
                    help=r"first on real + Isaac Sim's pictures (hekate's train\RAW, default datasets\synth_raw2)")
    ap.add_argument("--epochs", type=int, default=None, help="each run's (default 150; 60 with --synth)")
    ap.add_argument("--qat-epochs", type=int, default=40)
    ap.add_argument("--name", default=None, help="the run's name (default: SIZE and today's date)")
    ap.add_argument("--x3", action="store_true", help="the X3's looks in the set too")
    ap.add_argument("--no-fetch", action="store_true", help="use the looks already fetched")
    a = ap.parse_args()
    name = a.name or f"{a.size}_{time.strftime('%Y%m%d')}"
    m3 = ROOT / "m3pro"

    if not a.no_fetch:
        print("1. the M3's looks", flush=True)
        subprocess.run([sys.executable, "tools/m3_duel.py", "fetch", "20"], cwd=m3, check=True)
    print("2. the set", flush=True)
    subprocess.run([sys.executable, "tools/kit_dataset.py"], cwd=m3, check=True)
    if a.x3:
        x3 = ROOT / "x3plus"
        if not a.no_fetch:
            subprocess.run([sys.executable, "tools/kit_dataset_x3.py", "fetch"], cwd=x3, check=True)
        subprocess.run([sys.executable, "tools/kit_dataset_x3.py", "build"], cwd=x3, check=True)

    free = int(ssh("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits").split()[0])
    if free < FREE_MIB:
        raise SystemExit(f"hekate's GPU has {free} MiB free (under {FREE_MIB}): someone is training -- ask first")
    print(f"3. to hekate ({free} MiB free on its GPU)", flush=True)
    ssh(f'Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "{REMOTE_WIN}\\datasets\\kit"; '
        f'New-Item -ItemType Directory -Force "{REMOTE_WIN}\\datasets" | Out-Null')
    scp(*(str(ROOT / "train" / f) for f in ("pyproject.toml", "uv.lock", "train_kit.py", "refine_x3_sam.py")),
        f"{HOST}:{REMOTE}/")
    send_folder(m3 / "out" / "kit", f"{REMOTE}/datasets", f"{REMOTE_WIN}\\datasets")
    print(ssh(f'cd "{REMOTE_WIN}"; & $HOME\\.local\\bin\\uv.exe sync --frozen --no-progress 2>&1 | Select-Object -Last 1', check=False))
    if a.x3:   # the X3's projected outlines made exact by SAM 2 (train/refine_x3_sam.py)
        print(ssh(f'cd "{REMOTE_WIN}"; $env:PYTHONUTF8 = "1"; & .\\.venv\\Scripts\\python.exe refine_x3_sam.py '
                  '--set datasets\\kit 2>&1 | ForEach-Object { "$_" } | Select-String "X3 pictures"',
                  timeout=3600, check=False))

    epochs = a.epochs or (60 if a.synth else 150)
    first = f"--model yolo26{a.size}-seg.pt --epochs {epochs} --name {name}"
    if a.synth:
        print(f"3b. Isaac Sim's pictures ({a.synth}) as labels, and the real + synthetic set", flush=True)
        ssh(f'New-Item -ItemType Directory -Force "{REMOTE_WIN}\\synth" | Out-Null')
        scp(*(str(p) for p in (ROOT / "train" / "synth").glob("*.py")), f"{HOST}:{REMOTE}/synth/")
        print(ssh(f'cd "{REMOTE_WIN}"; & .\\.venv\\Scripts\\python.exe synth\\synth_to_yolo.py --raw {a.synth} 2>&1 | '
                  'Select-Object -Last 1', timeout=1200, check=False))
        print(f"4a. training {name}-mix (real + synthetic)", flush=True)
        print(on_hekate(f"--data datasets\\kit_synth.yaml --model yolo26{a.size}-seg.pt --epochs {epochs} "
                        f"--name {name}-mix"))
        first = f"--model runs\\kit\\{name}-mix\\weights\\best.pt --epochs {epochs} --name {name}"   # then real only
    runs = [(name, first, "")]
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
