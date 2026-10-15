"""Where the X3 must aim to have a thing between its fingers: the grips the user steered to by hand
(tools/x3_teach.py, as the M3's tools/m3_teach.py), each the place the wide look saw a piece and how
far the arm model's fingertip point sat from it when the fingers were right.

The user, 2026-10-15: far out the grips came down a little off and at a slant -- a cube survived it,
the yellow cylinder slipped out; the reach offset was 0 with the C930e, the raw model. Nine grips
taught that day: the arm reaches FURTHER than the model says, more the further out (1 mm at 14.5 cm,
11-14 mm at 25 cm), and a few mm left. So the aim: back towards the base by a line in the reach, and
a constant sideways -- each grip left out foretold by the rest to 8.7 mm rms, against 12.5 aiming at
the raw model; the M3's nearness-weighted table (roboarm/m3aim.py) did 9.7-11.1 on these.

Once it holds anything it replaces the reach offset (config.REACH_OFFSET_*, grasp.aim_for) while ON
(the console's "taught aim" setting, x3pick; 0 aims the old way).

To teach more: tools/x3_teach.py, then `python -m roboarm.x3aim out/teach/points.jsonl` (from
x3plus/) prints the table below anew.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# (seen x, seen y, aim dx, dy): m in the arm's base frame -- out/teach/points.jsonl
TAUGHT: tuple = (
    (0.2490, -0.0610, -0.0109, +0.0104),   # 2026-10-15 15:12 30 mm cube
    (0.2030, +0.1520, -0.0143, +0.0008),   # 2026-10-15 15:17 30 mm cube
    (0.2480, +0.0460, -0.0069, +0.0141),   # 2026-10-15 15:18 30 mm cube
    (0.2460, -0.0770, -0.0122, +0.0055),   # 2026-10-15 15:20 30 mm cube
    (0.1340, -0.0550, +0.0018, +0.0082),   # 2026-10-15 15:22 30 mm cube
    (0.1420, +0.1370, -0.0063, -0.0013),   # 2026-10-15 15:22 30 mm cube
    (0.1990, +0.0480, -0.0125, -0.0084),   # 2026-10-15 15:23 30 mm cube
    (0.1880, -0.1190, -0.0075, +0.0018),   # 2026-10-15 15:24 30 mm cube
    (0.2490, -0.0470, -0.0118, -0.0034),   # 2026-10-15 15:25 30 mm cube
)
ON = True        # the console's "taught aim" (x3pick sets it from its setting)
REACH_AT = 0.20  # m: the line's reference reach


def model(taught=None) -> tuple[float, float, float]:
    """(out at REACH_AT, out per m of reach, left), m: the line through the taught grips' errors
    along their reach, their mean across it; a constant out with fewer than three."""
    table = np.asarray(TAUGHT if taught is None else taught, float).reshape(-1, 4)
    r = np.hypot(table[:, 0], table[:, 1])
    u = table[:, :2] / r[:, None]
    out = (table[:, 2:] * u).sum(axis=1)
    left = (table[:, 2:] * np.c_[-u[:, 1], u[:, 0]]).sum(axis=1)
    if len(table) < 3:
        return float(out.mean()), 0.0, float(left.mean())
    a, b = np.linalg.lstsq(np.c_[np.ones_like(r), r - REACH_AT], out, rcond=None)[0]
    return float(a), float(b), float(left.mean())


def offset(x: float, y: float, taught=None) -> np.ndarray:
    """(dx, dy) m: where to aim past a thing the camera saw at (x, y) -- zero with nothing taught."""
    if not len(TAUGHT if taught is None else taught):
        return np.zeros(2)
    a, b, left = model(taught)
    r = float(np.hypot(x, y))
    u = np.array([x, y]) / r
    return (a + b * (r - REACH_AT)) * u + left * np.array([-u[1], u[0]])


def table_from(points: str) -> list[str]:
    """TAUGHT's lines from tools/x3_teach.py's saves."""
    from roboarm import kinematics as kin  # only to make the table
    lines = []
    for line in Path(points).read_text().splitlines():
        r = json.loads(line)
        seen = np.array([r["piece"]["x"], r["piece"]["y"]], float)
        grip = np.array(kin.forward({int(j): v for j, v in r["pose"].items()})[:2])
        d = grip - seen
        lines.append(f"    ({seen[0]:.4f}, {seen[1]:+.4f}, {d[0]:+.4f}, {d[1]:+.4f}),   "
                     f"# {r['when'][:16]} {r['piece']['name']}")
    return lines


if __name__ == "__main__":
    print("\n".join(table_from(sys.argv[1])))
