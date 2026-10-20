"""The M3's fixed test set for the kit detector (2026-10-20): looks no model is trained on, every
model judged on them alike. The M3 val (tools/kit_dataset.py's every 5th scene) moves as frames come
in -- on 2026-10-19 the served model had seen part of it in training, and 40 pictures with the 40 mm
cube in 8 could not tell the models apart. As the X3's x3plus/out/looks_test.

    python tools/kit_test_m3.py add 20261020_09 20261020_10   # those looks (stamp prefixes) of out/duel/frames
    python tools/kit_test_m3.py build                         # out/kit_test/{all,wide,close}: images + labels
    ...python train/eval_kit.py MODEL.pt ... --set SET --test m3pro/out/kit_test/all --test .../wide --test .../close

The looks are copied to out/kit_test/frames; kit_dataset.py leaves out every look kept there, even
fetched again. Labels as kit_dataset.py makes them (the depth's, the box and the cylinder left out).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kit_dataset

ROOT = HERE.parent
FRAMES = ROOT / "out" / "duel" / "frames"
TEST = ROOT / "out" / "kit_test"
KEPT = TEST / "frames"


def held_out() -> set[str]:
    """The stems of the looks in the test set."""
    return {p.name[:-5] for p in KEPT.glob("*.json")}


def add(prefixes: list[str]) -> None:
    KEPT.mkdir(parents=True, exist_ok=True)
    n = 0
    for meta in sorted(FRAMES.glob("*.json")):
        stem = meta.name[:-5]
        if stem[16:] in kit_dataset.LOOKS and any(stem.startswith(p) for p in prefixes):
            for suffix in ("_rgb.jpg", "_depth.png", ".json"):
                if (FRAMES / (stem + suffix)).exists():
                    shutil.copy2(FRAMES / (stem + suffix), KEPT / (stem + suffix))
            n += 1
    print(f"{n} looks added -> {KEPT} ({len(held_out())} in all)")


def build() -> None:
    counts = {}
    for name in ("all", "wide", "close"):
        if (TEST / name).exists():
            shutil.rmtree(TEST / name)
    for stem in sorted(held_out()):
        bgr = cv2.imread(str(KEPT / f"{stem}_rgb.jpg"))
        lines = kit_dataset.label_lines(json.loads((KEPT / f"{stem}.json").read_text()), bgr=bgr)
        for name in ("all", stem[16:]):
            for sub in ("images", "labels"):
                (TEST / name / sub).mkdir(parents=True, exist_ok=True)
            shutil.copy2(KEPT / f"{stem}_rgb.jpg", TEST / name / "images" / f"{stem}.jpg")
            (TEST / name / "labels" / f"{stem}.txt").write_text("".join(f"{line}\n" for line in lines))
            per = counts.setdefault(name, [0] * (len(kit_dataset.NAMES) + 1))
            per[0] += 1
            for line in lines:
                per[1 + int(line.split()[0])] += 1
    for name, (pictures, *per) in counts.items():
        print(f"{name}: {pictures} pictures, " + ", ".join(f"{c} {k}" for c, k in zip(kit_dataset.NAMES, per)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("add").add_argument("prefixes", nargs="+")
    sub.add_parser("build")
    a = ap.parse_args()
    add(a.prefixes) if a.cmd == "add" else build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
