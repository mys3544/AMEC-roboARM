#!/usr/bin/env python3
"""Least-squares fit of the arm's geometry to measured fingertip positions.

    docker compose run --rm core python tools/fit_kinematics.py

Three measurements cannot distinguish six candidate faults, which is why hypothesis-
at-a-time got us nowhere. This fits them all at once and reports what the data
actually supports.

Free parameters:
    L1, L2      the two upper-arm links
    L3          J4 -> fingertip, fitted PER GRIPPER STATE, because open fingers
                splay off the axis and shorten the tool
    d2, d3, d4  a constant offset on each joint, i.e. the servo angle at which that
                joint is really straight (nominally 90)

Add every measurement to OBSERVATIONS as you take it. The fit needs more
observations than parameters to mean anything -- with 4 points it is merely
consistent, with 8 it is evidence.
"""

from __future__ import annotations

import math
import sys

import numpy as np
from scipy.optimize import least_squares

from roboarm import config as cfg

# (label, pose, gripper_servo, forward_mm, height_above_table_mm)
OBSERVATIONS = [
    ("parked",  {2: 90, 3: 90, 4: 90}, 150,   0.0, 575.0),
    ("point 1", {2: 37, 3: 21, 4: 32}, 150, 160.0,  32.0),
    ("point 4", {2: 23, 3: 37, 4: 44}, 150, 215.0,  30.0),
    ("point 4b", {2: 22, 3: 39, 4: 43},  30, 208.0,  65.0),
    # Four deliberately dissimilar shapes, gripper fixed at 150, measured to the
    # MIDPOINT between the fingertips. Angles are the ones ACTUALLY REACHED, read
    # back from the arm -- it landed 1-2 degrees under target on several joints.
    ("shape A", {2: 19, 3: 19, 4: 99},  150, 285.0,  65.0),
    ("shape B", {2: 89, 3: 48, 4: 25},  150, 242.0, 315.0),
    ("shape C", {2: 19, 3: 99, 4: 20},  150, 300.0, 165.0),
    ("shape D", {2: 22, 3: 29, 4: 159}, 150, 303.0, 308.0),
]

GRIPPERS = sorted({obs[2] for obs in OBSERVATIONS})
TABLE = cfg.TABLE_BELOW_PLATE * 1000
BASE = cfg.BASE_PLATE_TO_J2 * 1000


def predict(params: np.ndarray, pose: dict[int, int], gripper: int) -> tuple[float, float]:
    l1, l2, d2, d3, d4 = params[:5]
    l3 = params[5 + GRIPPERS.index(gripper)]
    a2 = math.radians(90 - pose[2] + d2)
    a3 = a2 + math.radians(90 - pose[3] + d3)
    a4 = a3 + math.radians(90 - pose[4] + d4)
    forward = l1 * math.sin(a2) + l2 * math.sin(a3) + l3 * math.sin(a4)
    height = BASE + l1 * math.cos(a2) + l2 * math.cos(a3) + l3 * math.cos(a4) + TABLE
    return forward, height


def residuals(params: np.ndarray) -> np.ndarray:
    out = []
    for _label, pose, gripper, fwd, hgt in OBSERVATIONS:
        pf, ph = predict(params, pose, gripper)
        out += [pf - fwd, ph - hgt]
    return np.array(out)


def main() -> int:
    n_obs = 2 * len(OBSERVATIONS)
    n_par = 5 + len(GRIPPERS)
    print(f"{len(OBSERVATIONS)} observations ({n_obs} equations), {n_par} parameters")
    if n_obs <= n_par:
        print("  WARNING: not over-determined -- the fit can match the data without\n"
              "  being right. Take more measurements before believing it.\n")

    start = np.array([83.5, 83.5, 0.0, 0.0, 0.0] + [190.0] * len(GRIPPERS))
    bounds = (
        np.array([60, 60, -20, -20, -20] + [100.0] * len(GRIPPERS)),
        np.array([120, 120, 20, 20, 20] + [260.0] * len(GRIPPERS)),
    )
    fit = least_squares(residuals, start, bounds=bounds)

    l1, l2, d2, d3, d4 = fit.x[:5]
    print("fitted geometry (measured value in brackets):")
    print(f"  L1  J2->J3      {l1:6.1f} mm   [83.5]")
    print(f"  L2  J3->J4      {l2:6.1f} mm   [83.5]")
    for gripper in GRIPPERS:
        l3 = fit.x[5 + GRIPPERS.index(gripper)]
        print(f"  L3  J4->tip     {l3:6.1f} mm   at gripper J6={gripper}")
    print(f"  d2  J2 offset   {d2:+6.1f} deg  [0 if the servo is straight at 90]")
    print(f"  d3  J3 offset   {d3:+6.1f} deg")
    print(f"  d4  J4 offset   {d4:+6.1f} deg")

    # Occam: are the joint offsets earning their keep, or just soaking up noise?
    # Three extra parameters can always reduce the residual; the question is whether
    # they reduce it enough to be believed over the simpler, measured geometry.
    def fixed_residuals(free: np.ndarray) -> np.ndarray:
        return residuals(np.concatenate([free[:2], [0.0, 0.0, 0.0], free[2:]]))

    fixed = least_squares(
        fixed_residuals,
        np.concatenate([start[:2], start[5:]]),
        bounds=(
            np.concatenate([bounds[0][:2], bounds[0][5:]]),
            np.concatenate([bounds[1][:2], bounds[1][5:]]),
        ),
    )
    full_rms = float(np.sqrt(np.mean(fit.fun**2)))
    fixed_rms = float(np.sqrt(np.mean(fixed.fun**2)))
    print(f"\nRMS, joint offsets free      : {full_rms:5.1f} mm   (7 parameters)")
    print(f"RMS, offsets forced to zero  : {fixed_rms:5.1f} mm   (4 parameters)")
    if fixed_rms < full_rms * 1.35:
        print("  -> the offsets buy little; the simpler model is the honest one:")
        print(f"     L1 {fixed.x[0]:.1f} mm   L2 {fixed.x[1]:.1f} mm   " + "   ".join(
            f"L3@J6={g} {fixed.x[2 + GRIPPERS.index(g)]:.1f} mm" for g in GRIPPERS))
    else:
        print("  -> the offsets genuinely help: the joints are NOT mechanically")
        print("     straight at servo 90.")

    print("\nresiduals (fitted minus measured):")
    worst = 0.0
    for _i, (label, pose, gripper, fwd, hgt) in enumerate(OBSERVATIONS):
        pf, ph = predict(fit.x, pose, gripper)
        worst = max(worst, abs(pf - fwd), abs(ph - hgt))
        print(f"  {label:<9} J6={gripper:<3}  forward {pf - fwd:+6.1f} mm   height {ph - hgt:+6.1f} mm")
    print(f"\nworst residual {worst:.1f} mm")
    print("A good fit with few observations proves little -- add points and re-run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
