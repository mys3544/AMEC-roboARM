"""The depth duel (PLAN.md S1, stage 1 -- the user's choice: the same picture, the M3 only): the
M3's stereo depth (the DCW2) against Depth Anything V2 Small on the very same colour picture,
anchored to the floor as the X3 has to (x3plus/roboarm/monodepth.py: d = a/Z + b + a quadratic warp,
fitted on the floor the arm's pose predicts). The pictures are the pick's own looks, kept clean
with the kit's outlines since 2026-10-09 (m3pick.look: *_rgb.jpg, *_depth.png, *.json).

    python tools/m3_duel.py fetch [DAY]      # the robot's out/pick/DAY_*  (today) -> out/duel/frames/
    python tools/m3_duel.py run [--panels 6] # -> out/duel/results.md, out/duel/panel_*.jpg

Two anchors, so the network's share of the error is told from the arm model's: "arm" -- the floor
where the arm model puts it (what the X3, with no depth camera, has); "floor" -- where the DCW2's
own floor plane puts it (the camera levelled, an upper bound). For every kit piece in a picture: its
height from each (the 90th percentile over its outline, as x3plus's parallax reads it) against the
kit's true height; and the floor's flatness away from the pieces. Depth Anything runs on the
laptop's CPU (onnxruntime, the X3's ONNX export: 0.6 s; 26 ms as the X3's TensorRT engine).

A second contestant since 2026-10-12, when there is its ONNX (PLAN.md M5 model 8; exported on hekate
to train/runs/extra/): YOLO26n-depth, 3.8 ms as an engine on either robot. Its log-depth head reads
the scene's shape, its absolute scale a fit to its training cameras -- so it is scored three ways:
as it reads (its depth taken for metres, heights through the DCW2-levelled camera), and anchored
exactly as Depth Anything is (its 1/depth as the disparity), on each floor.
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roboshared.kit import KIT

from roboarm import m3kinematics

OUT = ROOT / "out" / "duel"
FRAMES = OUT / "frames"
MODEL = Path.home() / "Desktop" / "AMEC releases" / "RoboARM-X3Plus-2026-09-30" / "models" / "da2s_518x686.onnx"
HOST, REMOTE = "192.168.73.66", "/home/jetson/roboarm/out/pick"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
NAMED_MM = 7   # a height within this of a kit piece's names it (m3vision.kit_item)
YOLO_MODEL = ROOT.parent / "train" / "runs" / "extra" / "yolo26n-depth.onnx"   # 480 x 640, RGB / 255 in, depth out
COLUMNS = [("dcw2", "DCW2"), ("arm", "DA, arm's floor"), ("floor", "DA, DCW2's floor"),
           ("y_raw", "YOLO26n-depth as it reads"), ("y_arm", "YOLO26n-depth, arm's floor"),
           ("y_floor", "YOLO26n-depth, DCW2's floor")]

# x3plus/roboarm/monodepth.py by its path: both robots' packages are called roboarm
_spec = importlib.util.spec_from_file_location("monodepth", ROOT.parent / "x3plus" / "roboarm" / "monodepth.py")
monodepth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(monodepth)


def fetch(day: str) -> None:
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username="jetson", look_for_keys=False, timeout=15,
                   password=os.environ.get("ROBOARM_SSH_PASSWORD") or getpass.getpass(f"jetson@{HOST} password: "))
    sftp = client.open_sftp()
    FRAMES.mkdir(parents=True, exist_ok=True)
    stems = sorted(n[:-8] for n in sftp.listdir(REMOTE) if n.startswith(day) and n.endswith("_rgb.jpg"))
    for stem in stems:
        for suffix in ("_rgb.jpg", "_depth.png", ".json"):
            if not (FRAMES / (stem + suffix)).exists():
                sftp.get(f"{REMOTE}/{stem}{suffix}", str(FRAMES / (stem + suffix)))
    client.close()
    print(f"{len(stems)} frames -> {FRAMES}")


def true_height(o: dict) -> float | None:
    """The kit piece's real height (m), from what the depth named it: its kind, width and length."""
    for kind, width, length, height in KIT:
        if o["kind"] == kind and o["size"] is not None and abs(o["size"] - width) < 1e-6 \
                and abs((o["length"] or 0) - length) < 1e-6:
            return height
    return None


def kit_piece(o: dict, bgr: np.ndarray, inside: np.ndarray) -> bool:
    """Whether a thing the depth named a kit piece is one to score: seen whole (not cut by the
    picture's edge), and not black -- the kit has no black pieces, and the depth "named" bits of the
    black drop-off box's wall and lid 30-40 mm cubes (2026-10-09)."""
    return bool(o["whole"]) and float(np.median(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2][inside])) > 70


def dcw2_heights(depth_mm: np.ndarray, K, cam: np.ndarray) -> np.ndarray:
    """Height over the floor (m) of every pixel's point from the DCW2's own depth; NaN where none."""
    rows, cols = depth_mm.shape
    v, u = np.mgrid[0:rows, 0:cols]
    z = depth_mm.astype(np.float64) / 1000
    p = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=-1)
    h = (p @ cam[:3, :3].T + cam[:3, 3])[..., 2] - m3kinematics.FLOOR
    return np.where(z > 0.15, h, np.nan)


def run(panels: int) -> None:
    import onnxruntime

    session = onnxruntime.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    yolo = onnxruntime.InferenceSession(str(YOLO_MODEL), providers=["CPUExecutionProvider"]) \
        if YOLO_MODEL.exists() else None
    rows, floors, spent, spent_yolo = [], [], [], []
    shown = 0
    for meta_path in sorted(FRAMES.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        stem = meta_path.stem
        bgr = cv2.imread(str(FRAMES / f"{stem}_rgb.jpg"))
        depth = cv2.imread(str(FRAMES / f"{stem}_depth.png"), cv2.IMREAD_UNCHANGED)
        K = np.asarray(meta["K"], float)
        h, w = depth.shape
        x = (cv2.cvtColor(cv2.resize(bgr, (686, 518), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
             .astype(np.float32) / 255.0 - MEAN) / STD
        began = time.perf_counter()
        disp = session.run(None, {"image": np.ascontiguousarray(x.transpose(2, 0, 1)[None])})[0][0]
        spent.append(time.perf_counter() - began)
        pose = {int(j): v for j, v in meta["pose"].items()}
        cameras = {"arm": m3kinematics.frames(pose)["arm4"] @ m3kinematics.camera_optical(),
                   "floor": np.asarray(meta["camera"], float)}
        real = dcw2_heights(depth, K, cameras["floor"])
        maps = {}
        for name, cam in cameras.items():
            try:
                fit = monodepth.anchor(disp, (w, h), K, np.zeros(5), cam[:3, :3], cam[:3, 3], m3kinematics.FLOOR)
            except ValueError:
                maps[name] = None
                continue
            maps[name] = cv2.resize(fit["height"].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        if yolo is not None:
            rgb = cv2.cvtColor(cv2.resize(bgr, (640, 480)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            began = time.perf_counter()
            depth_y = yolo.run(None, {"images": np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])})[0][0, 0]
            spent_yolo.append(time.perf_counter() - began)
            depth_y = cv2.resize(depth_y.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            maps["y_raw"] = dcw2_heights(np.clip(depth_y * 1000, 0, 65535), K, cameras["floor"])
            for name, cam in cameras.items():
                try:
                    fit = monodepth.anchor(1.0 / np.maximum(depth_y, 1e-3), (w, h), K, np.zeros(5), cam[:3, :3],
                                           cam[:3, 3], m3kinematics.FLOOR)
                    maps[f"y_{name}"] = fit["height"].astype(np.float32)
                except ValueError:
                    maps[f"y_{name}"] = None
        pieces = np.zeros((h, w), bool)
        look = "close" if "close" in stem else "wide"
        for o in meta["found"]:
            truth = true_height(o)
            if truth is None or len(o["outline_px"]) < 3:
                continue
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [np.asarray(o["outline_px"], np.int32)], 1)
            inside = mask.astype(bool)
            pieces |= inside
            if not kit_piece(o, bgr, inside):
                continue
            row = {"frame": stem, "look": look, "truth": truth, "whole": o["whole"],
                   "dcw2": float(np.nanpercentile(real[inside], 90))}
            for name, hm in maps.items():
                row[name] = None if hm is None else float(np.nanpercentile(hm[inside], 90))
            rows.append(row)
        flat = ~cv2.dilate(pieces.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) \
            & (np.abs(real) < 0.005)
        floors.append({"look": look, "dcw2": float(np.nanstd(real[flat]))} |
                      {n: (None if hm is None else float(np.nanstd(hm[flat]))) for n, hm in maps.items()})
        if shown < panels and rows and rows[-1]["frame"] == stem and maps["arm"] is not None:
            write_panel(OUT / f"panel_{shown}_{stem}.jpg", bgr, real, maps, meta, stem)
            shown += 1
    report(rows, floors, spent, spent_yolo)


def colour(hm: np.ndarray, top: float = 0.065) -> np.ndarray:
    scaled = np.clip(np.nan_to_num(hm, nan=0.0) / top, 0, 1)
    out = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[np.isnan(hm)] = 0
    return out


def write_panel(path: Path, bgr, real, maps, meta, stem) -> None:
    """colour | the DCW2's heights | Depth Anything's, anchored on the arm's floor | on the DCW2's floor;
    0 (blue) to 65 mm (red), each kit piece's height written on it."""
    tiles = [bgr.copy(), colour(real), colour(maps["arm"]), colour(maps["floor"]) if maps["floor"] is not None else colour(real * np.nan)]
    titles = ["colour", "DCW2 (stereo)", "Depth Anything, arm's floor", "Depth Anything, DCW2's floor"]
    for o in meta["found"]:
        truth = true_height(o)
        if truth is None or len(o["outline_px"]) < 3:
            continue
        mask = np.zeros(real.shape, np.uint8)
        cv2.fillPoly(mask, [np.asarray(o["outline_px"], np.int32)], 1)
        if not kit_piece(o, bgr, mask.astype(bool)):
            continue
        u, v = np.asarray(o["outline_px"]).mean(axis=0).astype(int)
        for tile, hm in zip(tiles[1:], (real, maps["arm"], maps["floor"]), strict=True):
            if hm is None:
                continue
            text = f"{1000 * np.nanpercentile(hm[mask.astype(bool)], 90):.0f}"
            cv2.putText(tile, text, (u - 14, v + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(tiles[0], f"{1000 * truth:.0f}", (u - 14, v + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    for tile, title in zip(tiles, titles, strict=True):
        cv2.putText(tile, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))


def report(rows: list[dict], floors: list[dict], spent: list[float], spent_yolo: list[float] = ()) -> None:
    keys = [(k, label) for k, label in COLUMNS if any(r.get(k) is not None for r in rows)]
    lines = ["# The depth duel: the DCW2 against Depth Anything V2 Small (and YOLO26n-depth), the same pictures", "",
             (f"{len({r['frame'] for r in rows})} pictures, {len(rows)} kit pieces (m3pro/tools/m3_duel.py, "
              f"{time.strftime('%Y-%m-%d')}). Heights: the 90th percentile over each piece's outline; "
              f"'named' = within {NAMED_MM} mm of its true height (what m3vision.kit_item needs)."), "",
             ("Mind: the pieces are the ones the DCW2's depth found and named (whole, not black), so its own "
              "column cannot miss by much -- it says how exact it is, not how often it finds them."), "",
             "Error mm (median / p90), named:", "",
             "| look | pieces | " + " | ".join(label for _k, label in keys) + " |",
             "|---|---|" + "---|" * len(keys)]
    for look in ("wide", "close", "all"):
        chosen = [r for r in rows if look in ("all", r["look"])]
        if not chosen:
            continue
        cells = []
        for key, _label in keys:
            err = np.array([abs(r[key] - r["truth"]) * 1000 for r in chosen if r.get(key) is not None])
            cells.append(f"{np.median(err):.1f} / {np.percentile(err, 90):.1f}, {np.mean(err <= NAMED_MM):.0%}"
                         if len(err) else "--")
        lines.append(f"| {look} | {len(chosen)} | " + " | ".join(cells) + " |")
    lines += ["", "| look | floor flatness, sd mm: " + " | ".join(label for _k, label in keys) + " |",
              "|---|" + "---|" * len(keys)]
    for look in ("wide", "close"):
        chosen = [f for f in floors if f["look"] == look]
        if chosen:
            cells = [f"{1000 * np.median(v):.1f}" if (v := [f[k] for f in chosen if f.get(k) is not None]) else "--"
                     for k, _label in keys]
            lines.append(f"| {look} | " + " | ".join(cells) + " |")
    by_truth = {}
    for r in rows:
        by_truth.setdefault(round(r["truth"] * 1000), []).append(r)
    lines += ["", "| true height mm | pieces | reads (median): " + " | ".join(label for _k, label in keys) + " |",
              "|---|---|" + "---|" * len(keys)]
    for truth, chosen in sorted(by_truth.items()):
        cells = [f"{1000 * np.median(v):.0f}" if (v := [r[k] for r in chosen if r.get(k) is not None]) else "--"
                 for k, _label in keys]
        lines.append(f"| {truth} | {len(chosen)} | " + " | ".join(cells) + " |")
    lines += ["", f"Depth Anything on the laptop's CPU: {1000 * np.median(spent):.0f} ms a picture (median)"
              + (f"; YOLO26n-depth {1000 * np.median(spent_yolo):.0f} ms." if len(spent_yolo) else ".")]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch").add_argument("day", nargs="?", default=time.strftime("%Y%m%d"))
    sub.add_parser("run").add_argument("--panels", type=int, default=6)
    args = parser.parse_args()
    if args.cmd == "fetch":
        fetch(args.day)
    else:
        run(args.panels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
