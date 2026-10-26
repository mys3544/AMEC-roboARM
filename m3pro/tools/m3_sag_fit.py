"""A sag that depends on the arm's shape, fitted to the ChArUco readings of tools/m3_twin_gap.py (NEXT.md, "This
week": the twin's constant SAG left ~1 cm in height in the low, reaching poses where the grips are).

The models, each as degrees the joints sit off their command (m3kinematics.SAG's sense) -- or the mount:
    constant   J2, J3 fixed (m3kinematics.SAG today: 4.0, -8.5) -- as it is, and refitted
    constant3  J2, J3, J4 fixed, refitted
    centred    today's constant at a 0.20 m lever, a slope a joint (J2-J4) fitted
    gravity    J2, J3, J4 each c + k x the grip point's level distance from that joint's axis (m): what the
               links beyond a joint weigh on it, the more the further they reach
    mount      the joints as today, the camera's mount on arm4 turned and shifted (6 numbers)

2026-10-26, 38 readings from 3 runs (12 high looks, 26 low grip-like poses), held out: today's constant 2.5 mm
median on the table, 4.8 mm rms; constant refitted 5.5, with J4 5.5, centred 4.4, mount 4.6; gravity 3.7 --
but through J2 -10..-18 deg and J4 +9 deg, offsets the arm could not hide: it trades the joints against the
camera's mount, which a camera alone cannot tell apart. None adopted: the twin's model holds to ~5 mm rms,
near what the readings themselves repeat to; the gripper's sag wants the fingertips measured (m3_touch.py).
Fitted with each run's board pose (6 numbers a run: the board may have moved between runs) by Levenberg-
Marquardt on the board's corners reprojected through the model's camera, against where they were seen; judged
by 5-fold cross-validation -- the held-out poses' board place against the run's, in mm (on the table, height).

    python tools/m3_sag_fit.py out/twin/A.json out/twin/B.json ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import m3_handeye as he

from roboarm import m3kinematics

JOINTS = (2, 3, 4)
NOW = {2: 4.0, 3: -8.5}   # m3kinematics.SAG today


def levers(pose: dict) -> dict[int, float]:
    """The grip point's level distance (m) from the axes of joints 2-4, the arm as commanded (no sag)."""
    saved = dict(m3kinematics.SAG)
    m3kinematics.SAG.clear()
    try:
        f = m3kinematics.frames(pose)
        tip = (f["arm5"] @ np.r_[m3kinematics.GRIPPING[0], 1.0])[:3]
        return {j: float(np.hypot(*(tip[:2] - f[f"arm{j}"][:2, 3]))) for j in JOINTS}
    finally:
        m3kinematics.SAG.update(saved)


def sag_of(model: str, theta: np.ndarray, pose: dict) -> dict[int, float]:
    if model == "mount":   # the joints as today, the camera's mount turned and shifted (rad, m)
        MOUNT[:] = rt(np.r_[theta[:3], theta[3:6]])
        return dict(NOW)
    MOUNT[:] = np.eye(4)   # every other model: the URDF's mount
    if model == "constant":
        return {2: float(theta[0]), 3: float(theta[1])}
    if model == "constant3":
        return {2: float(theta[0]), 3: float(theta[1]), 4: float(theta[2])}
    lv = levers(pose)
    if model == "centred":   # today's constant at a 0.20 m lever, each joint's slope fitted
        return {j: NOW.get(j, 0.0) + float(theta[i]) * (lv[j] - 0.20) for i, j in enumerate(JOINTS)}
    return {j: float(theta[2 * i] + theta[2 * i + 1] * lv[j]) for i, j in enumerate(JOINTS)}


MOUNT = np.eye(4)   # the camera's mount corrected (the "mount" model sets it while it is fitted)


def camera(pose: dict, sag: dict) -> np.ndarray:
    saved = dict(m3kinematics.SAG)
    m3kinematics.SAG.clear()
    m3kinematics.SAG.update(sag)
    try:
        return m3kinematics.frames(pose)["arm4"] @ m3kinematics.camera_optical() @ MOUNT
    finally:
        m3kinematics.SAG.clear()
        m3kinematics.SAG.update(saved)


def rt(v) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = cv2.Rodrigues(np.asarray(v[:3], float))[0]
    m[:3, 3] = v[3:6]
    return m


def residuals(params, model, n_theta, readings, runs) -> np.ndarray:
    theta, boards = params[:n_theta], params[n_theta:].reshape(-1, 6)
    out = []
    for r in readings:
        cam = camera(r["pose"], sag_of(model, theta, r["pose"]))
        b = rt(boards[runs.index(r["run"])])
        q = np.linalg.inv(cam) @ b @ np.c_[r["points"], np.ones(len(r["points"]))].T
        uv = (r["K"] @ (q[:3] / q[2]))[:2].T
        out.append((uv - r["pixels"]).ravel())
    return np.concatenate(out)


def fit(model, theta0, readings, runs, iters: int = 40):
    """Levenberg-Marquardt over the sag's numbers and each run's board pose."""
    boards0 = []
    for run in runs:   # each run's board where the starting model puts it, the median of its poses
        places = [camera(r["pose"], sag_of(model, theta0, r["pose"])) @ r["board_in_camera"]
                  for r in readings if r["run"] == run]
        rv = np.median([cv2.Rodrigues(p[:3, :3])[0].ravel() for p in places], axis=0)
        boards0.append(np.r_[rv, np.median([p[:3, 3] for p in places], axis=0)])
    p = np.r_[theta0, np.concatenate(boards0)]
    lam = 1e-2
    res = residuals(p, model, len(theta0), readings, runs)
    for _ in range(iters):
        jac = np.empty((len(res), len(p)))
        for i in range(len(p)):
            step = 1e-4 if i >= len(theta0) else 1e-3
            q = p.copy()
            q[i] += step
            jac[:, i] = (residuals(q, model, len(theta0), readings, runs) - res) / step
        g, h = jac.T @ res, jac.T @ jac
        while True:
            delta = np.linalg.solve(h + lam * np.diag(np.diag(h) + 1e-9), -g)
            trial = residuals(p + delta, model, len(theta0), readings, runs)
            if trial @ trial < res @ res:
                p, res, lam = p + delta, trial, lam / 3
                break
            lam *= 4
            if lam > 1e8:
                return p, res
        if np.abs(delta).max() < 1e-6:
            break
    return p, res


def place_errors(model, p, n_theta, readings, runs) -> np.ndarray:
    """Each reading's board middle through the model against its run's fitted board: (n x 3) m."""
    theta, boards = p[:n_theta], p[n_theta:].reshape(-1, 6)
    middle = np.r_[np.mean(he.board().getChessboardCorners(), axis=0), 1.0]
    out = []
    for r in readings:
        cam = camera(r["pose"], sag_of(model, theta, r["pose"]))
        out.append(((cam @ r["board_in_camera"]) @ middle)[:3] - (rt(boards[runs.index(r["run"])]) @ middle)[:3])
    return np.array(out)


def report(name: str, errors: np.ndarray) -> str:
    flat, up = 1000 * np.hypot(errors[:, 0], errors[:, 1]), 1000 * errors[:, 2]
    return (f"{name:28s} on the table {np.median(flat):5.1f} mm median, {flat.max():5.1f} max; height "
            f"{np.median(np.abs(up)):5.1f} mm median |.|, {np.abs(up).max():5.1f} max (rms 3D {1000 * np.sqrt(np.mean(np.sum(errors ** 2, 1))):.1f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    a = ap.parse_args()
    readings = []
    for path in a.runs:
        for r in json.loads(path.read_text())["readings"]:
            readings.append({"run": path.stem, "pose": {int(j): float(v) for j, v in r["pose"].items()},
                             "K": np.asarray(r["K"], float), "board_in_camera": np.asarray(r["board_in_camera"], float),
                             "pixels": np.asarray(r["pixels"], float), "points": np.asarray(r["points"], float)})
    runs = sorted({r["run"] for r in readings})
    print(f"{len(readings)} readings from {len(runs)} runs", flush=True)
    models = {"constant as now": ("constant", np.array([4.0, -8.5]), False),
              "constant refitted": ("constant", np.array([4.0, -8.5]), True),
              "camera mount": ("mount", np.zeros(6), True),
              "constant with J4": ("constant3", np.array([4.0, -8.5, 0.0]), True),
              "centred slopes": ("centred", np.array([0.0, 0.0, 0.0]), True),
              "gravity": ("gravity", np.array([4.0, 0.0, -8.5, 0.0, 0.0, 0.0]), True)}
    rng = np.random.default_rng(0)
    folds = rng.permutation(len(readings)) % 5
    for name, (model, theta0, refit) in models.items():
        if refit:
            p, _ = fit(model, theta0, readings, runs)
        else:   # the sag as it is: only the boards fitted
            p = np.r_[theta0, fit_boards_only(model, theta0, readings, runs)]
        theta = p[: len(theta0)]
        held = np.zeros((len(readings), 3))
        for k in range(5):
            train = [r for r, f in zip(readings, folds, strict=True) if f != k]
            test = [i for i, f in enumerate(folds) if f == k]
            if refit:
                pk, _ = fit(model, theta0, train, runs)
            else:
                pk = np.r_[theta0, fit_boards_only(model, theta0, train, runs)]
            held[test] = place_errors(model, pk, len(theta0), [readings[i] for i in test], runs)
        print(f"{name}: {np.round(theta, 3).tolist()}", flush=True)
        print("  fitted   " + report(name, place_errors(model, p, len(theta0), readings, runs)), flush=True)
        print("  held out " + report(name, held), flush=True)
    return 0


def fit_boards_only(model, theta, readings, runs) -> np.ndarray:
    """Each run's board pose with the sag held: Levenberg-Marquardt over the boards alone."""
    def res_b(b):
        return residuals(np.r_[theta, b], model, len(theta), readings, runs)
    boards = []
    for run in runs:
        places = [camera(r["pose"], sag_of(model, theta, r["pose"])) @ r["board_in_camera"] for r in readings if r["run"] == run]
        rv = np.median([cv2.Rodrigues(p[:3, :3])[0].ravel() for p in places], axis=0)
        boards.append(np.r_[rv, np.median([p[:3, 3] for p in places], axis=0)])
    b = np.concatenate(boards)
    res, lam = res_b(b), 1e-2
    for _ in range(30):
        jac = np.empty((len(res), len(b)))
        for i in range(len(b)):
            q = b.copy()
            q[i] += 1e-4
            jac[:, i] = (res_b(q) - res) / 1e-4
        g, h = jac.T @ res, jac.T @ jac
        while True:
            delta = np.linalg.solve(h + lam * np.diag(np.diag(h) + 1e-9), -g)
            trial = res_b(b + delta)
            if trial @ trial < res @ res:
                b, res, lam = b + delta, trial, lam / 3
                break
            lam *= 4
            if lam > 1e8:
                return b
        if np.abs(delta).max() < 1e-7:
            break
    return b


if __name__ == "__main__":
    sys.exit(main())
