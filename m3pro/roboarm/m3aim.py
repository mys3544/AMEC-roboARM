"""Where the arm must aim to have a thing between its fingers: the grips the user steered to by hand
(tools/m3_teach.py -- the M3 cannot go limp or read its joints back), each the place the wide look
saw a piece and how far the model's grip point sat from it when the fingers were right.

2026-10-09: grips came down short and closed on a cube's near corners (6 misses in 8). Eight taught
grips sat 8-22 mm further out and -4..+11 mm further left than the camera saw the piece -- less by
the robot's right front, more in the middle -- and the spread is not the wrist's (a grip point off
J5's axis fitted them better but foretold a left-out one no better: 7.4 mm against 7.3). So the aim
is the taught offsets near the thing, Gaussian-weighted over SIGMA, pulled towards their mean by
PRIOR: it foretold each left-out grip to 5.6 mm (rms; the mean alone 7.3).

To teach more: tools/m3_hover.py look, tools/m3_teach.py, then `python -m roboarm.m3aim
out/teach/points.jsonl` prints the table below anew.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# (seen x, seen y, aim dx, dy): m in base_link -- out/teach/points.jsonl
TAUGHT = (
    (0.2383, -0.0592, +0.0187, +0.0036),   # 2026-10-09 12:20 cube
    (0.1814, -0.0478, +0.0101, +0.0053),   # 2026-10-09 12:21 cube
    (0.2150, +0.0107, +0.0182, +0.0102),   # 2026-10-09 12:22 cube
    (0.1718, -0.0261, +0.0134, +0.0046),   # 2026-10-09 12:29 cube
    (0.1666, +0.0553, +0.0136, +0.0075),   # 2026-10-09 12:30 cube
    (0.2277, +0.0155, +0.0181, +0.0106),   # 2026-10-09 12:32 cube
    (0.1992, -0.1002, +0.0083, -0.0035),   # 2026-10-09 13:47 block, by the right front
    (0.2057, +0.0085, +0.0220, +0.0114),   # 2026-10-09 13:49 block (taught 1 cm high: the floor guard)
)
SIGMA = 0.03   # m: how far a taught grip speaks for its neighbours
PRIOR = 0.3    # the mean's weight, as many taught grips right on the spot


def offset(x: float, y: float, taught=None) -> np.ndarray:
    """(dx, dy) m: where to aim past a thing the camera saw at (x, y) -- zero with nothing taught."""
    table = np.asarray(TAUGHT if taught is None else taught, float).reshape(-1, 4)
    if not len(table):
        return np.zeros(2)
    w = np.exp(-0.5 * ((table[:, 0] - x) ** 2 + (table[:, 1] - y) ** 2) / SIGMA ** 2)
    return (w @ table[:, 2:] + PRIOR * table[:, 2:].mean(axis=0)) / (w.sum() + PRIOR)


def table_from(points: str) -> list[str]:
    """TAUGHT's lines from tools/m3_teach.py's saves."""
    from roboarm import m3kinematics  # only to make the table
    lines = []
    for line in Path(points).read_text().splitlines():
        r = json.loads(line)
        seen = np.asarray(r["piece"]["centre"][:2])
        grip = m3kinematics.grip({int(j): v for j, v in r["pose"].items()})[0]
        d = grip[:2] - seen
        lines.append(f"    ({seen[0]:.4f}, {seen[1]:+.4f}, {d[0]:+.4f}, {d[1]:+.4f}),   # {r['when'][:16]} {r['piece']['kind']}")
    return lines


if __name__ == "__main__":
    print("\n".join(table_from(sys.argv[1])))
