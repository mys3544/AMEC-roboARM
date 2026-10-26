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

2026-10-26: the table re-derived for m3kinematics.SAG's J2 1.1 (was 4.0) -- the same taught poses and seen
places, each through the new model (the seen places 3 mm further out with the wide look's camera): the
offsets 8-22 mm out shrink to -3..+10 (most of them were the model's J2), the left ones stay (+1..+10).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# (seen x, seen y, aim dx, dy): m in base_link -- out/teach/points.jsonl
TAUGHT = (   # re-derived 2026-10-26 for SAG J2 1.1 (the 2026-10-09 offsets were 8-22 mm out)
    (0.2413, -0.0592, +0.0065, +0.0059),   # 2026-10-09 12:20 cube
    (0.1844, -0.0478, -0.0021, +0.0078),   # 2026-10-09 12:21 cube
    (0.2180, +0.0107, +0.0057, +0.0092),   # 2026-10-09 12:22 cube
    (0.1748, -0.0261, +0.0009, +0.0059),   # 2026-10-09 12:29 cube
    (0.1696, +0.0553, +0.0018, +0.0038),   # 2026-10-09 12:30 cube
    (0.2307, +0.0155, +0.0057, +0.0094),   # 2026-10-09 12:32 cube
    (0.2022, -0.1002, -0.0026, +0.0013),   # 2026-10-09 13:47 block, by the right front
    (0.2087, +0.0085, +0.0098, +0.0104),   # 2026-10-09 13:49 block (taught 1 cm high: the floor guard)
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
