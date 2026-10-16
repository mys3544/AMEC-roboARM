"""Why the depth duel's close look fails (m3_duel.py, 2026-10-15): the floor fit's variants, each
scored exactly as m3_duel scores the arm's floor -- per kit piece the 90th percentile of its heights
over its outline against its true height, median / p90 error and the share 'named' (within 7 mm).

    python tools/duel_floorfit.py [--data DIR] [--day 20261009] [--models da2s,da2l] [--frames]
                                  [--set main|sweep] [--variants "A;B"] [--camera arm|floor]

--data: the duel's folder (frames/, extra/); the saved looks are not in git, so a worktree points
it at the main checkout's m3pro/out/duel. Depth Anything V2 Small's maps are made once on the CPU
(m3_duel's ONNX, its preprocessing) and kept in out/floorfit/da2s/. --frames: a line per close
look. --set sweep: the self-exclusion's knobs. --camera floor: anchor on the DCW2's levelled pose
(m3_duel's "DCW2's floor" column) instead of the arm model's. A process a frame (--workers).

What it found (2026-10-15, the 69 pictures of 2026-10-09, then the 54 of 2026-10-15 as a check):
the quadratic warp cost on every look and both networks; a third of a close look's predicted floor
is not floor (box, tape, the gripper's finger before the lens) and the Huber fit kept 98 % of it;
the fit made again without what it reads over 10 mm (monodepth.anchor's refits) fixes the gripper
looks; what is left close is the network's own relief -- the "oracle: ... true floor + its Z" rows,
the best any floor anchor can do, are about where the robot-realistic fit already is.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # a process a frame instead
    os.environ.setdefault(_var, "1")

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m3_duel as duel

monodepth, m3kinematics = duel.monodepth, duel.m3kinematics
FLOOR = m3kinematics.FLOOR
CACHE = HERE.parent / "out" / "floorfit"
RAISED_M = 0.005     # the oracle: what the DCW2 sees this far over the floor is not floor
GROW_FRAME_PX = 15   # and its neighbourhood, in frame pixels (640 x 480)


# --- the maps -------------------------------------------------------------------------------------

def da2s_disp(stem: str, bgr: np.ndarray, session) -> np.ndarray:
    path = CACHE / "da2s" / f"{stem}.npy"
    if path.exists():
        return np.load(path)
    x = (cv2.cvtColor(cv2.resize(bgr, (686, 518), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
         .astype(np.float32) / 255.0 - duel.MEAN) / duel.STD
    disp = session.run(None, {"image": np.ascontiguousarray(x.transpose(2, 0, 1)[None])})[0][0]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, disp)
    return disp


def extra_disp(data: Path, model: str, stem: str, size: tuple[int, int]) -> np.ndarray | None:
    """As m3_duel: the map resized to the frame, 1/depth for a metric one."""
    path = data / "extra" / model / f"{stem}.npy"
    if not path.exists():
        return None
    out = cv2.resize(np.nan_to_num(np.load(path).astype(np.float32)), size, interpolation=cv2.INTER_LINEAR)
    if (data / "extra" / model / "kind.txt").read_text().strip() == "depth":
        out = 1.0 / np.maximum(out, 1e-3)
    return out


# --- the fits -------------------------------------------------------------------------------------

def on_grid(mask: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), grid.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)


def grow(mask: np.ndarray, px: int) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)).astype(bool) if px else mask


class Look:
    """One picture, one camera pose, one network's map: what every variant needs."""

    def __init__(self, disp, size, K, cam, real, depth_mm, cam_dcw2=None):
        self.disp, self.size, self.K, self.cam = disp, size, K, cam
        self.cam_dcw2 = cam if cam_dcw2 is None else cam_dcw2   # the DCW2-levelled pose, for its own Z
        self.R, self.c = cam[:3, :3], cam[:3, 3]
        self.ray = monodepth.rays(K, np.zeros(5), size, disp.shape[::-1])
        self.inv_floor = monodepth.floor_inverse_depth(self.ray, self.R, self.c, FLOOR)
        self.real = real                    # the DCW2's heights, frame size
        self.depth_mm = depth_mm
        self.grid_px = max(1, round(GROW_FRAME_PX * disp.shape[1] / size[0]))

    def anchor(self, exclude=None, warp=True) -> dict:
        """monodepth.anchor on the rays already made."""
        fit = monodepth.fit_floor(self.disp, self.inv_floor, self.ray, exclude, warp)
        z = monodepth.metric_depth(self.disp, fit, self.ray)
        return {**fit, "height": monodepth.heights(z, self.ray, self.R, self.c, FLOOR)}

    def heights_from(self, params, warp) -> np.ndarray:
        z = monodepth.metric_depth(self.disp, {"params": params, "warp": warp}, self.ray)
        return monodepth.heights(z, self.ray, self.R, self.c, FLOOR)

    def frame(self, h_grid: np.ndarray) -> np.ndarray:
        return cv2.resize(h_grid.astype(np.float32), self.size, interpolation=cv2.INTER_LINEAR)

    def floor_px(self) -> np.ndarray:
        return (self.inv_floor > 1 / monodepth.FAR_M) & (self.inv_floor < 1 / monodepth.NEAR_M) & np.isfinite(self.disp)


def irls(A: np.ndarray, d: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
    """monodepth.fit_floor's loop (Huber, 4 sigma out), from `params` if given instead of from OLS."""
    weights = np.ones_like(d)
    if params is not None:
        weights = _weights(d - A @ params)
    for _ in range(monodepth.FIT_ROUNDS):
        w = np.sqrt(weights)[:, None]
        params, *_ = np.linalg.lstsq(A * w, d * w[:, 0], rcond=None)
        weights = _weights(d - A @ params)
    return params


def _weights(residual):
    sigma = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-12
    scaled = np.abs(residual) / (monodepth.HUBER_K * sigma)
    weights = np.where(scaled <= 1, 1.0, 1.0 / scaled)
    weights[np.abs(residual) > 4 * sigma] = 0.0
    return weights


def design(look: Look, warp: bool, offset: bool = True) -> np.ndarray:
    A = monodepth._design(look.inv_floor, look.ray, warp)
    return A if offset else np.delete(A, 1, axis=-1)


def v_base(look, warp=True):
    return look.frame(look.anchor(warp=warp)["height"])


def v_oracle(look, warp=True, with_nan=False):
    """Fit only where the DCW2 sees floor: nothing over RAISED_M, grown; with_nan: nor where it sees nothing."""
    bad = look.real > RAISED_M
    if with_nan:
        bad |= ~np.isfinite(look.real)
    return look.frame(look.anchor(on_grid(grow(bad, GROW_FRAME_PX), look.disp), warp)["height"])


def v_self(look, warp=True, thr=0.010, rounds=3, below=False, keep=False, grow_px=GROW_FRAME_PX,
           start="plain", final_warp=False, cap=1.0, cap_upright_only=False):
    """Robot-realistic: fit, drop what the fit itself reads over thr (and below -thr if `below`) and
    where it reads nothing, grown by grow_px frame pixels, refit; `keep`: the dropped pixels stay
    dropped; start "lmeds": the first fit is v_lmeds's; final_warp: one last fit with the warp on
    the floor the plain fits left; cap: stop (keep the last fit) once more than this share of the
    predicted floor would be dropped -- with cap_upright_only, only while the last fit's scale a is
    positive (an upside-down fit is no fit to keep)."""
    px = round(grow_px * look.disp.shape[1] / look.size[0])
    if start == "plain":
        fit = look.anchor(warp=warp)
        h, a = fit["height"], fit["scale"]
    else:
        h, a = v_lmeds(look, warp, grid=True), 1.0
    exclude = np.zeros(look.disp.shape, bool)
    floor = look.floor_px()
    for _ in range(rounds):
        bad = ~np.isfinite(h) | (h > thr)
        if below:
            bad |= h < -thr
        bad = grow(bad, px)
        if (bad & floor).sum() > cap * floor.sum() and (a > 0 or not cap_upright_only):
            break
        exclude = exclude | bad if keep else bad
        try:
            fit = look.anchor(exclude, warp)
        except ValueError:
            break
        h, a = fit["height"], fit["scale"]
    if final_warp:
        try:
            h = look.anchor(exclude, True)["height"]
        except ValueError:
            pass
    return look.frame(h)


def v_lmeds(look, warp=True, tries=400, sample=40, seed=0, grid=False):
    """Least median of squares over random floor subsets (any thing under half the floor cannot
    pull it), then monodepth's Huber loop from there."""
    floor = look.floor_px()
    A, d = design(look, warp)[floor], look.disp[floor]
    rng = np.random.default_rng(seed)
    score_idx = rng.choice(len(d), min(len(d), 20000), replace=False)
    best, best_med = None, np.inf
    for _ in range(tries):
        idx = rng.choice(len(d), sample, replace=False)
        p, *_ = np.linalg.lstsq(A[idx], d[idx], rcond=None)
        med = np.median(np.abs(d[score_idx] - A[score_idx] @ p))
        if med < best_med:
            best, best_med = p, med
    params = irls(A, d, best)
    h = look.heights_from(params, warp)
    return h if grid else look.frame(h)


def v_no_offset(look, warp=True):
    """d = a / Z (+ warp): b taken as 0, so a comes from the floor's level, not its tilt."""
    floor = look.floor_px()
    A, d = design(look, warp, offset=False)[floor], look.disp[floor]
    p = irls(A, d)
    params = np.insert(p, 1, 0.0)
    return look.frame(look.heights_from(params, warp))


def _fit_true_z(look, mask, warp):
    """d = a / Z + b (+ warp) with the DCW2's own Z on `mask` (grid); heights through the DCW2's
    levelled pose, so the pose cannot add its own error."""
    z = cv2.resize(look.depth_mm.astype(np.float32), look.disp.shape[::-1], interpolation=cv2.INTER_NEAREST) / 1000
    ok = mask & (z > 0.15) & np.isfinite(look.disp)
    inv = np.where(ok, 1 / np.maximum(z, 1e-3), 0)
    params = irls(monodepth._design(inv, look.ray, warp)[ok], look.disp[ok])
    zz = monodepth.metric_depth(look.disp, {"params": params, "warp": warp}, look.ray)
    R, c = look.cam_dcw2[:3, :3], look.cam_dcw2[:3, 3]
    return look.frame(monodepth.heights(zz, look.ray, R, c, FLOOR))


def v_oracle_global(look, warp=False):
    """a, b (+ warp) fitted on EVERY pixel the DCW2 has depth for, with its own Z."""
    return _fit_true_z(look, np.ones(look.disp.shape, bool), warp)


def v_oracle_floor_z(look, warp=False):
    """The best ANY floor-only anchor could do with this map: a, b (+ warp) on the DCW2's true
    floor (within 3 mm, away from anything raised), with its own Z."""
    real = look.real
    floor = np.isfinite(real) & (np.abs(real) < 0.003) & ~grow(real > RAISED_M, 10)
    return _fit_true_z(look, on_grid(floor, look.disp), warp)


VARIANTS = {
    "base (warp)": lambda L: v_base(L, True),
    "no warp": lambda L: v_base(L, False),
    "oracle: DCW2 >5mm out, warp": lambda L: v_oracle(L, True),
    "oracle: DCW2 >5mm out, no warp": lambda L: v_oracle(L, False),
    "oracle: DCW2 >5mm or none out, warp": lambda L: v_oracle(L, True, True),
    "oracle: DCW2 >5mm or none out, no warp": lambda L: v_oracle(L, False, True),
    "self-exclude 10mm, warp": lambda L: v_self(L, True),
    "self-exclude 10mm, no warp": lambda L: v_self(L, False),
    "self-exclude 5mm, warp": lambda L: v_self(L, True, 0.005),
    "self-exclude +-10mm, warp": lambda L: v_self(L, True, 0.010, below=True),
    "LMedS + Huber, warp": lambda L: v_lmeds(L, True),
    "LMedS + Huber, no warp": lambda L: v_lmeds(L, False),
    "b = 0, warp": lambda L: v_no_offset(L, True),
    "b = 0, no warp": lambda L: v_no_offset(L, False),
    "oracle: a,b on all DCW2 px, no warp": lambda L: v_oracle_global(L, False),
    "oracle: a,b on all DCW2 px, warp": lambda L: v_oracle_global(L, True),
    "oracle: a,b on DCW2 true floor + its Z, no warp": lambda L: v_oracle_floor_z(L, False),
    "oracle: a,b on DCW2 true floor + its Z, warp": lambda L: v_oracle_floor_z(L, True),
}
SWEEP = {   # --set sweep: the self-exclusion's knobs, no warp
    "no warp": lambda L: v_base(L, False),
    "self 5mm": lambda L: v_self(L, False, 0.005),
    "self 10mm": lambda L: v_self(L, False, 0.010),
    "self 15mm": lambda L: v_self(L, False, 0.015),
    "self 20mm": lambda L: v_self(L, False, 0.020),
    "self 10mm, 1 round": lambda L: v_self(L, False, 0.010, rounds=1),
    "self 10mm, 6 rounds": lambda L: v_self(L, False, 0.010, rounds=6),
    "self 10mm, kept": lambda L: v_self(L, False, 0.010, keep=True),
    "self +-10mm": lambda L: v_self(L, False, 0.010, below=True),
    "self 10mm, grow 0": lambda L: v_self(L, False, 0.010, grow_px=0),
    "self 10mm, grow 30": lambda L: v_self(L, False, 0.010, grow_px=30),
    "self 10mm, LMedS start": lambda L: v_self(L, False, 0.010, start="lmeds"),
    "self 10mm, warp after": lambda L: v_self(L, False, 0.010, final_warp=True),
    "self nan-only": lambda L: v_self(L, False, 10.0),
    "self 30mm": lambda L: v_self(L, False, 0.030),
    "self 10mm, cap 50%": lambda L: v_self(L, False, 0.010, cap=0.5),
    "self 10mm, cap 50% while a > 0": lambda L: v_self(L, False, 0.010, cap=0.5, cap_upright_only=True),
    "self +-10mm, cap 50%": lambda L: v_self(L, False, 0.010, below=True, cap=0.5),
    "LMedS, no warp": lambda L: v_lmeds(L, False),
    "LMedS start, self +-10mm": lambda L: v_self(L, False, 0.010, below=True, start="lmeds"),
}
ALL = VARIANTS | SWEEP


# --- the run --------------------------------------------------------------------------------------

def pieces(meta, bgr, shape):
    for o in meta["found"]:
        truth = duel.true_height(o)
        if truth is None or len(o["outline_px"]) < 3:
            continue
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [np.asarray(o["outline_px"], np.int32)], 1)
        inside = mask.astype(bool)
        if duel.kit_piece(o, bgr, inside):
            yield truth, inside


_session = None


def one_frame(job) -> tuple[list, list]:
    """A frame's rows (model, variant, look, stem, truth, read) and diagnostics."""
    global _session
    data, meta_path, models, variants, camera = job
    stem = meta_path.stem
    meta = json.loads(meta_path.read_text())
    bgr = cv2.imread(str(data / "frames" / f"{stem}_rgb.jpg"))
    depth = cv2.imread(str(data / "frames" / f"{stem}_depth.png"), cv2.IMREAD_UNCHANGED)
    K = np.asarray(meta["K"], float)
    h, w = depth.shape
    found = list(pieces(meta, bgr, (h, w)))
    rows, diag = [], []
    if not found:
        return rows, diag
    pose = {int(j): v for j, v in meta["pose"].items()}
    cam_dcw2 = np.asarray(meta["camera"], float)
    cam = m3kinematics.frames(pose)["arm4"] @ m3kinematics.camera_optical() if camera == "arm" else cam_dcw2
    real = duel.dcw2_heights(depth, K, cam_dcw2)
    look_kind = "close" if "close" in stem else "wide"
    for model in models:
        if model == "da2s":
            if _session is None and not (CACHE / "da2s" / f"{stem}.npy").exists():
                _session = duel.onnx_session(duel.MODEL)
            disp = da2s_disp(stem, bgr, _session)
        else:
            disp = extra_disp(data, model, stem, (w, h))
            if disp is None:
                continue
        look = Look(disp, (w, h), K, cam, real, depth, cam_dcw2)
        fl = look.floor_px()
        try:
            base_fit = look.anchor()
            sd, pinned, a = base_fit["scale_sd"], base_fit["pinned"], base_fit["scale"]
        except ValueError:
            sd, pinned, a = float("nan"), False, float("nan")
        contam = on_grid(grow(look.real > RAISED_M, GROW_FRAME_PX) | ~np.isfinite(look.real), disp)
        diag.append({"model": model, "stem": stem, "look": look_kind,
                     "tilt": float(np.degrees(np.arccos(-cam[2, 2]))), "height": float(cam[2, 3]),
                     "scale_sd": sd, "pinned": pinned, "a": a,
                     "contam": float((contam & fl).sum() / max(1, fl.sum()))})
        for name in variants:
            try:
                hm = ALL[name](look)
            except (ValueError, np.linalg.LinAlgError):
                hm = None
            for truth, inside in found:
                read = None if hm is None else float(np.nanpercentile(hm[inside], 90))
                rows.append((model, name, look_kind, stem, truth, read))
    return rows, diag


def run(data: Path, day: str, models: list[str], variants: list[str], per_frame: bool, workers: int,
        camera: str = "arm") -> None:
    jobs = [(data, m, models, variants, camera) for m in sorted((data / "frames").glob(f"{day}*.json"))]
    missing = [m.stem for _, m, *_ in jobs if not (CACHE / "da2s" / f"{m.stem}.npy").exists()]
    if "da2s" in models and missing:   # the maps first, here on the GPU -- not a session a worker
        session = duel.onnx_session(duel.MODEL)
        for stem in missing:
            da2s_disp(stem, cv2.imread(str(data / "frames" / f"{stem}_rgb.jpg")), session)
    rows, diag = [], []
    with Pool(workers) as pool:
        for r, d in pool.imap(one_frame, jobs):
            rows += r
            diag += d
    print(f"(anchored on {'the arm model' if camera == 'arm' else 'the DCW2'}'s floor; frames {day or 'all'}*)")
    report(rows, diag, models, variants, per_frame)


def summary(errs: list[float | None]) -> str:
    e = np.array([x for x in errs if x is not None and np.isfinite(x)])
    if not len(e):
        return "--"
    fails = len(errs) - len(e)
    return (f"{np.median(e):.1f} / {np.percentile(e, 90):.1f}, {np.mean(e <= duel.NAMED_MM):.0%}"
            + (f" ({fails} no fit)" if fails else ""))


def report(rows, diag, models, variants, per_frame) -> None:
    for model in models:
        mine = [r for r in rows if r[0] == model]
        if not mine:
            continue
        n = {lk: len({(r[3], id(r)) for r in mine if r[2] == lk and r[1] == variants[0]}) for lk in ("wide", "close")}
        print(f"\n## {model}: {len({r[3] for r in mine})} pictures; "
              f"height error mm, median / p90, named (<= {duel.NAMED_MM} mm)\n")
        print(f"| variant | wide, {n['wide']} pieces | close, {n['close']} pieces | all |")
        print("|---|---|---|---|")
        for name in variants:
            cells = []
            for lk in ("wide", "close", "all"):
                errs = [None if r[5] is None else abs(r[5] - r[4]) * 1000 for r in mine
                        if r[1] == name and lk in ("all", r[2])]
                cells.append(summary(errs))
            print(f"| {name} | " + " | ".join(cells) + " |")
        d = [x for x in diag if x["model"] == model]
        for lk in ("wide", "close"):
            dd = [x for x in d if x["look"] == lk]
            if dd:
                print(f"\n{lk}: tilt from straight down {np.median([x['tilt'] for x in dd]):.0f} deg (median), "
                      f"camera {100 * np.median([x['height'] for x in dd]):.0f} cm up; scale_sd median "
                      f"{np.median([x['scale_sd'] for x in dd]):.3f}, pinned {np.mean([x['pinned'] for x in dd]):.0%}; "
                      f"floor px the DCW2 says are raised/none {np.median([x['contam'] for x in dd]):.0%} median, "
                      f"{np.max([x['contam'] for x in dd]):.0%} max")
        if per_frame:
            print(f"\n{model}, close looks: stem | tilt | scale_sd | contam | reads (truth 30): "
                  + " ; ".join(variants))
            for x in d:
                if x["look"] != "close":
                    continue
                reads = []
                for name in variants:
                    rr = [r[5] for r in mine if r[3] == x["stem"] and r[1] == name]
                    reads.append(",".join("--" if v is None else f"{1000 * v:.0f}" for v in rr))
                print(f"{x['stem']} | {x['tilt']:.0f} | {x['scale_sd']:.3f} | {x['contam']:.0%} | " + " ; ".join(reads))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=duel.OUT)
    parser.add_argument("--day", default="20261009", help="frames whose stem starts so ('' = all)")
    parser.add_argument("--models", default="da2s,da2l")
    parser.add_argument("--variants", default=None, help="semicolon-separated names (default all)")
    parser.add_argument("--frames", action="store_true", help="a line per close look")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--set", choices=("main", "sweep"), default="main")
    parser.add_argument("--camera", choices=("arm", "floor"), default="arm",
                        help="the pose the anchor uses: the arm model's, or the DCW2's levelled one")
    args = parser.parse_args()
    chosen = VARIANTS if args.set == "main" else SWEEP
    variants = list(chosen) if args.variants is None else [v.strip() for v in args.variants.split(";")]
    run(args.data, args.day, args.models.split(","), variants, args.frames, args.workers, args.camera)
    return 0


if __name__ == "__main__":
    sys.exit(main())
