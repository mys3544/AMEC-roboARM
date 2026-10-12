"""An object scan's Gaussian splat from this laptop, in one go (PLAN.md M4 C): the scan's views ->
hekate fits the splat -> the console's 3D view shows it.

    uv run python tools/splat_cycle.py [STAMP] [--name can] [--fetch] [--steps 7000] [--depth-weight 0.5]
                                       [--pose-lr 1e-4]

Run it from the repo root (the workspace's venv, as tools/kit_cycle.py). The steps:
  1. the scan: m3pro/out/scan/STAMP -- the newest there (the console's Scan button fetches and fuses
     each scan into it) -- or, with --fetch or when STAMP is not here, the robot's (m3_scan.py fetch;
     the SSH password from ROBOARM_SSH_PASSWORD, or asked);
  2. its views as the splat's input (m3_scan.py fuse STAMP --export -> STAMP/splat/; ~20 s, Open3D);
  3. to hekate (`ssh hekate`, CLAUDE.md) as train\\datasets\\scans\\NAME, with train/splat/ and
     train/'s project and lock;
  4. hekate fits it (train/splat/splat.bat: gsplat, the depth camera's depth held to the render,
     each view's camera refined with it; ~1 min for a 36-view scan), its progress shown here;
  5. NAME.splat -> m3pro/out/arena/rtab/splat_NAME.splat, which the console's 3D view lists; its
     turntable, held-out view, refined cameras and log beside the scan (STAMP/splat_turntable.jpg,
     splat_poses.json, ...).
NAME is STAMP, or THING_STAMP with --name THING. Fitting is GPU work on a shared PC: hekate's
CLAUDE.md asks first when under ~16 GB is free -- this stops then.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from kit_cycle import FREE_MIB, HOST, REMOTE, REMOTE_WIN, scp, ssh

ROOT = Path(__file__).resolve().parents[1]
M3 = ROOT / "m3pro"
SCANS = M3 / "out" / "scan"
CLOUDS = M3 / "out" / "arena" / "rtab"   # the console's 3D view lists these (console/serve.py)
# torch's own warnings (its check of cl's version fails, harmlessly), and ssh's about its key exchange
NOISE = re.compile(r"^(W\d{4} |\*\* )")


def fit(name: str, steps: int, depth_weight: float, pose_lr: float, log: Path) -> None:
    """splat.bat on hekate's copy of the scan, its lines shown as they come (and kept in `log`)."""
    command = (f'cd "{REMOTE_WIN}"; cmd /c "splat\\splat.bat datasets\\scans\\{name} --name splat '
               f'--steps {steps} --depth-weight {depth_weight} --pose-lr {pose_lr} 2>&1"')
    lines = []
    with subprocess.Popen(["ssh", "-o", "BatchMode=yes", HOST, command], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors="replace") as proc:
        for line in proc.stdout:
            line = line.rstrip()
            if line and not NOISE.match(line):
                lines.append(line)
                print(f"   {line}", flush=True)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if proc.returncode != 0 or not any(line.startswith("held-out views") for line in lines):
        raise SystemExit(f"the fit failed (exit {proc.returncode}): its lines in {log}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stamp", nargs="?", help="the scan (default: the newest in m3pro/out/scan)")
    ap.add_argument("--name", help="the thing's name: the splat is splat_NAME_STAMP.splat")
    ap.add_argument("--fetch", action="store_true", help="fetch the scan from the robot first")
    ap.add_argument("--steps", type=int, default=7000)
    ap.add_argument("--depth-weight", type=float, default=0.5, help="0: fitted to the pictures only")
    ap.add_argument("--pose-lr", type=float, default=1e-4, help="the cameras' refinement; 0: the scan's poses")
    a = ap.parse_args()
    if a.name and not re.fullmatch(r"[A-Za-z0-9_-]+", a.name):
        ap.error("--name: letters, digits, _ and - only")

    if a.fetch or (a.stamp and not (SCANS / a.stamp).is_dir()):
        print("1. the scan, from the robot", flush=True)
        subprocess.run([sys.executable, "tools/m3_scan.py", "fetch", *([a.stamp] if a.stamp else [])], cwd=M3, check=True)
    scans = sorted(d.name for d in SCANS.iterdir() if (d / "scan.json").is_file()) if SCANS.is_dir() else []
    stamp = a.stamp or (scans[-1] if scans else None)
    if stamp not in scans:
        raise SystemExit(f"no scan {stamp or ''} in {SCANS}: scan one (the console's Scan), or --fetch")
    name = f"{a.name}_{stamp}" if a.name else stamp
    here = SCANS / stamp
    print(f"2. scan {stamp}: its views as the splat's input", flush=True)
    subprocess.run([sys.executable, "tools/m3_scan.py", "fuse", stamp, "--export"], cwd=M3, check=True)

    free = int(ssh("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits").split()[0])
    if free < FREE_MIB:
        raise SystemExit(f"hekate's GPU has {free} MiB free (under {FREE_MIB}): someone is training -- ask first")
    print(f"3. to hekate as datasets\\scans\\{name} ({free} MiB free on its GPU)", flush=True)
    ssh(f'Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "{REMOTE_WIN}\\datasets\\scans\\{name}"; '
        f'New-Item -ItemType Directory -Force "{REMOTE_WIN}\\datasets\\scans", "{REMOTE_WIN}\\splat" | Out-Null')
    scp(*(str(ROOT / "train" / f) for f in ("pyproject.toml", "uv.lock")), f"{HOST}:{REMOTE}/")
    scp(*(str(f) for f in (ROOT / "train" / "splat").iterdir() if f.suffix in (".py", ".bat")), f"{HOST}:{REMOTE}/splat/")
    scp(str(here / "splat"), f"{HOST}:{REMOTE}/datasets/scans/{name}", recursive=True)
    synced = ssh(f'cd "{REMOTE_WIN}"; & $HOME\\.local\\bin\\uv.exe sync --frozen --no-progress 2>&1 | Select-Object -Last 1',
                 check=False).strip()
    if synced:   # uv says nothing when train/'s environment is as its lock already
        print(f"   {synced}", flush=True)

    print("4. fitting the splat on hekate", flush=True)
    fit(name, a.steps, a.depth_weight, a.pose_lr, here / "splat_train.log")

    print("5. back here", flush=True)
    CLOUDS.mkdir(parents=True, exist_ok=True)
    out = CLOUDS / f"splat_{name}.splat"
    there = f"{HOST}:{REMOTE}/datasets/scans/{name}"
    scp(f"{there}/splat.splat", str(out))
    for f in ("splat_turntable.jpg", "splat_held_view.jpg", "splat_poses.json"):
        scp(f"{there}/{f}", str(here / f))
    print(f"-> {out.relative_to(ROOT)} ({out.stat().st_size // 32:,} Gaussians, {out.stat().st_size / 1e6:.1f} MB): "
          f"the console's 3D view lists it; its turntable and held-out view in {here.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
