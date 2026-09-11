#!/usr/bin/env python3
"""Analyse a reach-survey log: are the extremes real, or single-frame spikes?

    python tools/analyse_survey.py /tmp/survey.log

Raw min/max is maximally sensitive to one bad sample, so this reports the
distribution alongside a median-filtered range, which survives a genuine
position held for ~1s but discards an isolated glitch.
"""

import re
import statistics
import sys

from roboarm import config as cfg


def median_filter(values: list[int], width: int = 5) -> list[int]:
    """Rolling median. A one-frame spike vanishes; a pose held for width frames
    (about a second at 5 Hz) survives."""
    half = width // 2
    return [
        statistics.median(values[max(0, i - half) : i + half + 1])
        for i in range(len(values))
    ]


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/survey.log"
    with open(path, errors="replace") as handle:
        text = handle.read().replace("\r", "\n")

    frames = []
    for line in text.split("\n"):
        found = re.findall(r"J(\d):\s*(-?\d+)", line)
        if len(found) == len(cfg.JOINT_IDS):
            frames.append([int(v) for _, v in found])

    print(f"parsed {len(frames)} frames from {path}\n")
    if not frames:
        return 1

    print(f"{'joint':<6} {'raw range':>14} {'filtered range':>16} {'outliers':>10}  held-extremes")
    for index, joint in enumerate(cfg.JOINT_IDS):
        series = [f[index] for f in frames]
        lo_hard, hi_hard = cfg.HARD_LIMITS[joint]
        outside = sum(1 for v in series if not (lo_hard <= v <= hi_hard))
        filtered = median_filter(series)

        # Was the filtered extreme actually dwelt on, or just brushed past?
        f_lo, f_hi = min(filtered), max(filtered)
        dwell_lo = sum(1 for v in filtered if v <= f_lo + 2)
        dwell_hi = sum(1 for v in filtered if v >= f_hi - 2)

        print(
            f"J{joint:<5} {f'{min(series)}..{max(series)}':>14} {f'{f_lo}..{f_hi}':>16} "
            f"{f'{outside}/{len(series)}':>10}  {dwell_lo} lo / {dwell_hi} hi frames"
        )

    print("\nA large 'outliers' count means the joint really was moved past its")
    print("commandable range by hand. A count of 1-2 means a glitched frame.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
