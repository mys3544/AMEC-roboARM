"""The depth duel (PLAN.md S1, stage 1 -- the user's choice: the same picture, the M3 only): the
M3's stereo depth (the DCW2) against Depth Anything V2 Small on the very same colour picture,
anchored to the floor as the X3 has to (x3plus/roboarm/monodepth.py: d = a/Z + b, fitted on the
floor the arm's pose predicts, then again without what that fit reads as raised -- since 2026-10-15,
when the quadratic warp went and the refits came: tools/duel_floorfit.py). The pictures are the
pick's own looks, kept clean with the kit's outlines since 2026-10-09 (m3pick.look: *_rgb.jpg,
*_depth.png, *.json).

    python tools/m3_duel.py fetch [DAY]      # the robot's out/pick/DAY_*  (today) -> out/duel/frames/
    python tools/m3_duel.py run [--panels 6] [--versus da2l]   # -> out/duel/results.md, out/duel/panel_*.jpg

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

The big contestants since 2026-10-13, too big for the robots: train/depth/predict_depth.py on
hekate's 5090 (Depth Anything V2 Large and its metric indoor model, Depth Pro, ...), each model's
maps in out/duel/extra/MODEL/ (`scp -r "hekate:Desktop/Sim Arm 2026/train/depth/out/*" out/duel/extra/`).
A relative one is anchored as Depth Anything is; a metric one is also scored as it reads.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tarfile
import threading
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
EXTRA = OUT / "extra"   # the big contestants' maps from hekate, a folder each (kind.txt, ms.json, STEM.npy)
EXTRA_LABELS = {"da2s": "DA V2 Small on hekate", "da2l": "DA V2 Large",
                "da2l_metric": "DA V2 Metric Indoor Large", "depthpro": "Depth Pro"}

# x3plus/roboarm/monodepth.py by its path: both robots' packages are called roboarm
_spec = importlib.util.spec_from_file_location("monodepth", ROOT.parent / "x3plus" / "roboarm" / "monodepth.py")
monodepth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(monodepth)


def fetch(day: str) -> None:
    """The robot's looks of `day` (a prefix of their stamps) not here yet: over SSH with the laptop's
    key (on the M3 since 2026-10-15; a password before), the missing files as one tar stream."""
    FRAMES.mkdir(parents=True, exist_ok=True)
    ssh = ["ssh", "-o", "BatchMode=yes", f"jetson@{HOST}"]
    names = subprocess.run([*ssh, f"ls {REMOTE}"], capture_output=True, text=True, check=True).stdout.splitlines()
    stems = sorted(n[:-8] for n in names if n.startswith(day) and n.endswith("_rgb.jpg"))
    there = set(names)
    wanted = [stem + suffix for stem in stems for suffix in ("_rgb.jpg", "_depth.png", ".json")
              if stem + suffix in there and not (FRAMES / (stem + suffix)).exists()]
    if wanted:
        with subprocess.Popen([*ssh, f"cd {REMOTE} && tar --null -T - -cf -"], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE) as proc:
            def send() -> None:   # beside the reading, so neither side waits on a full pipe
                proc.stdin.write(b"\0".join(n.encode() for n in wanted))
                proc.stdin.close()
            threading.Thread(target=send, daemon=True).start()
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                tar.extractall(FRAMES, filter="data")
    print(f"{len(stems)} frames of {day}, {len(wanted)} files new -> {FRAMES}")


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


def anchored(disp: np.ndarray, size: tuple[int, int], K, cameras: dict) -> dict:
    """Heights (m) from a relative disparity, fitted to each camera's floor: {camera: map, or None
    where the fit fails}, at `size` (w, h)."""
    maps = {}
    for name, cam in cameras.items():
        try:
            fit = monodepth.anchor(disp, size, K, np.zeros(5), cam[:3, :3], cam[:3, 3], m3kinematics.FLOOR)
        except ValueError:
            maps[name] = None
            continue
        maps[name] = cv2.resize(fit["height"].astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    return maps


def extras() -> dict[str, str]:
    """The big contestants fetched from hekate: {model: "disparity" | "depth"}."""
    return {d.name: (d / "kind.txt").read_text().strip() for d in sorted(EXTRA.glob("*")) if (d / "kind.txt").exists()}


def extra_columns(kinds: dict[str, str]) -> list[tuple[str, str]]:
    columns = []
    for name, kind in kinds.items():
        label = EXTRA_LABELS.get(name, name)
        if kind == "depth":
            columns.append((f"{name}_raw", f"{label} as it reads"))
        columns += [(f"{name}_arm", f"{label}, arm's floor"), (f"{name}_floor", f"{label}, DCW2's floor")]
    return columns


def run(panels: int, versus: str | None = None) -> None:
    import onnxruntime

    session = onnxruntime.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    yolo = onnxruntime.InferenceSession(str(YOLO_MODEL), providers=["CPUExecutionProvider"]) \
        if YOLO_MODEL.exists() else None
    kinds = extras()
    rows, floors, spent, spent_yolo = [], [], [], []
    shown = {"wide": 0, "close": 0}
    for old in OUT.glob("panel_*.jpg") if panels else ():   # this run's panels only
        old.unlink()
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
        maps = anchored(disp, (w, h), K, cameras)
        if yolo is not None:
            rgb = cv2.cvtColor(cv2.resize(bgr, (640, 480)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            began = time.perf_counter()
            depth_y = yolo.run(None, {"images": np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])})[0][0, 0]
            spent_yolo.append(time.perf_counter() - began)
            depth_y = cv2.resize(depth_y.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            maps["y_raw"] = dcw2_heights(np.clip(depth_y * 1000, 0, 65535), K, cameras["floor"])
            maps |= {f"y_{n}": hm for n, hm in anchored(1.0 / np.maximum(depth_y, 1e-3), (w, h), K, cameras).items()}
        for model, kind in kinds.items():
            path = EXTRA / model / f"{stem}.npy"
            if not path.exists():
                continue
            out = cv2.resize(np.nan_to_num(np.load(path).astype(np.float32)), (w, h), interpolation=cv2.INTER_LINEAR)
            if kind == "depth":   # metres: as it reads, then its 1/depth anchored as a disparity
                maps[f"{model}_raw"] = dcw2_heights(np.clip(out * 1000, 0, 65535), K, cameras["floor"])
                out = 1.0 / np.maximum(out, 1e-3)
            maps |= {f"{model}_{n}": hm for n, hm in anchored(out, (w, h), K, cameras).items()}
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
        if shown[look] < (panels + 1) // 2 and rows and rows[-1]["frame"] == stem and maps["arm"] is not None \
                and (versus is None or maps.get(f"{versus}_arm") is not None):   # half wide looks, half close
            write_panel(OUT / f"panel_{look}_{shown[look]}_{stem}.jpg", bgr, real, maps, meta, versus)
            shown[look] += 1
    times = [f"Depth Anything V2 Small {1000 * np.median(spent):.0f} ms (this laptop's CPU, onnxruntime)"]
    if spent_yolo:
        times.append(f"YOLO26n-depth {1000 * np.median(spent_yolo):.0f} ms (the same)")
    for model in kinds:
        ms = json.loads((EXTRA / model / "ms.json").read_text())["median_ms"]
        times.append(f"{EXTRA_LABELS.get(model, model)} {ms:.0f} ms (hekate's RTX 5090, PyTorch fp32)")
    report(rows, floors, COLUMNS + extra_columns(kinds), times)


def colour(hm: np.ndarray, top: float = 0.065) -> np.ndarray:
    scaled = np.clip(np.nan_to_num(hm, nan=0.0) / top, 0, 1)
    out = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[np.isnan(hm)] = 0
    return out


def write_panel(path: Path, bgr, real, maps, meta, versus: str | None = None) -> None:
    """colour | the DCW2's heights | Depth Anything V2 Small's on the arm's floor (what the robots run)
    | `versus`'s on the arm's floor (a big contestant from hekate) -- without one, Depth Anything on
    the DCW2's floor; 0 (blue) to 65 mm (red), each kit piece's height written on it."""
    last = (f"{versus}_arm", f"{EXTRA_LABELS.get(versus, versus)}, arm's floor") if versus \
        else ("floor", "Depth Anything, DCW2's floor")
    shown = [real, maps["arm"], maps.get(last[0])]
    tiles = [bgr.copy()] + [colour(real * np.nan if hm is None else hm) for hm in shown]
    titles = ["colour", "DCW2 (stereo)", "Depth Anything V2 Small, arm's floor", last[1]]
    for o in meta["found"]:
        truth = true_height(o)
        if truth is None or len(o["outline_px"]) < 3:
            continue
        mask = np.zeros(real.shape, np.uint8)
        cv2.fillPoly(mask, [np.asarray(o["outline_px"], np.int32)], 1)
        if not kit_piece(o, bgr, mask.astype(bool)):
            continue
        u, v = np.asarray(o["outline_px"]).mean(axis=0).astype(int)
        for tile, hm in zip(tiles[1:], shown, strict=True):
            if hm is None:
                continue
            text = f"{1000 * np.nanpercentile(hm[mask.astype(bool)], 90):.0f}"
            cv2.putText(tile, text, (u - 14, v + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(tiles[0], f"{1000 * truth:.0f}", (u - 14, v + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    for tile, title in zip(tiles, titles, strict=True):
        cv2.putText(tile, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))


def report(rows: list[dict], floors: list[dict], columns: list[tuple[str, str]], times: list[str]) -> None:
    """results.md: a row per contestant (and anchor) -- its height error on each look, the floor's
    flatness, and what the pieces read -- then each network's time a picture."""
    keys = [(k, label) for k, label in columns if any(r.get(k) is not None for r in rows)]
    looks = [(look, [r for r in rows if look in ("all", r["look"])]) for look in ("wide", "close", "all")]
    looks = [(look, chosen) for look, chosen in looks if chosen]
    truths = sorted({round(r["truth"] * 1000) for r in rows})
    lines = ["# The depth duel: the DCW2 against monocular depth networks, the same pictures", "",
             (f"{len({r['frame'] for r in rows})} pictures, {len(rows)} kit pieces (m3pro/tools/m3_duel.py, "
              f"{time.strftime('%Y-%m-%d')}). Heights: the 90th percentile over each piece's outline; "
              f"'named' = within {NAMED_MM} mm of its true height (what m3vision.kit_item needs). "
              "'Arm's floor': the network's map fitted to the floor where the arm model puts it (what the X3, "
              "with no depth camera, has); 'DCW2's floor': to the depth camera's own floor plane (an upper "
              "bound); 'as it reads': a metric network's metres taken as they are."), "",
             ("Mind: the pieces are the ones the DCW2's depth found and named (whole, not black), so its own "
              "row cannot miss by much -- it says how exact it is, not how often it finds them."), "",
             ("Height error mm, median / p90, and the share named; the floor's flatness away from the pieces "
              "(sd mm, median over the pictures); what the pieces read (median mm) by true height:"), "",
             "| contestant | " + " | ".join(f"{look}, {len(chosen)} pieces" for look, chosen in looks)
             + " | floor sd, wide | floor sd, close | " + " | ".join(f"reads, {t} mm" for t in truths) + " |",
             "|---|" + "---|" * (len(looks) + 2 + len(truths))]
    for key, label in keys:
        cells = []
        for _look, chosen in looks:
            err = np.array([abs(r[key] - r["truth"]) * 1000 for r in chosen if r.get(key) is not None])
            cells.append(f"{np.median(err):.1f} / {np.percentile(err, 90):.1f}, {np.mean(err <= NAMED_MM):.0%}"
                         if len(err) else "--")
        for look in ("wide", "close"):
            v = [f[key] for f in floors if f["look"] == look and f.get(key) is not None]
            cells.append(f"{1000 * np.median(v):.1f}" if v else "--")
        for t in truths:
            v = [r[key] for r in rows if round(r["truth"] * 1000) == t and r.get(key) is not None]
            cells.append(f"{1000 * np.median(v):.0f}" if v else "--")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "Time a picture (median): " + "; ".join(times) + "."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch").add_argument("day", nargs="?", default=time.strftime("%Y%m%d"))
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--panels", type=int, default=6, help="pictures to draw, half wide looks, half close")
    run_parser.add_argument("--versus", default=None, metavar="MODEL",
                            help="the panels' last tile: this big contestant (out/duel/extra/MODEL, e.g. da2l)")
    args = parser.parse_args()
    if args.cmd == "fetch":
        fetch(args.day)
    else:
        run(args.panels, args.versus)
    return 0


if __name__ == "__main__":
    sys.exit(main())
