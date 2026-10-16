"""The kit detector's training set (PLAN.md M5) from the M3's own pick looks: every look keeps its
clean picture and the depth's outlines, named by the kit's sizes (m3pick.look: *_rgb.jpg + *.json;
fetched to out/duel/frames/ by `python tools/m3_duel.py fetch 2026`). This writes them as an
Ultralytics segmentation set -- three classes, colour comes from HSV -- to be trained on hekate
(train/ at the top of the repo).

    python tools/kit_dataset.py [--frames out/duel/frames] [--out out/kit] [--val-every 5]

What the depth could not name (tape rolls, the box, the remote, table edges) stays unlabelled, i.e.
background -- and so does what it named but is no kit piece (not_kit(): the black drop-off box's
walls and lid, and the yellow cylinder: 123 of the 694 labels of 2026-10-09/15/16). The box views
("wide J1 160") are left out: the pieces lying in the box are not outlined there, and unlabelled
they would be taught as background. Looks a minute or more apart are separate
scenes; every --val-every'th scene goes to val, so a pick's wide and close looks never straddle the split.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from roboshared.kit import KIT

ROOT = Path(__file__).resolve().parents[1]

NAMES = ["cube30", "cube40", "block"]
LOOKS = ("wide", "close")
SCENE_GAP_S = 60


def kit_class(o: dict) -> int | None:
    """The class of a thing the depth named (KIT's rows: the 30 and 40 mm cubes, the block lying or
    standing), or None for what it could not name."""
    for kind, width, length, _height in KIT:
        if o["kind"] == kind and o["size"] is not None and abs(o["size"] - width) < 1e-6 \
                and abs((o["length"] or 0) - length) < 1e-6:
            return 2 if kind == "block" else NAMES.index(f"cube{round(width * 1000)}")
    return None


def not_kit(outline_px, hsv: np.ndarray) -> bool:
    """A thing the depth named a kit piece that is none, by its colours inside its outline: the black
    drop-off box's wall or lid (dark, hardly a coloured or a bright pixel -- where the navy cubes, the
    tag cubes and pieces in its shadow keep theirs) or the yellow cylinder (the M3's kit has no yellow
    piece). Checked on the contact sheets of every such label, 2026-10-16."""
    inside = np.zeros(hsv.shape[:2], np.uint8)
    cv2.fillPoly(inside, [np.asarray(outline_px, np.int32)], 1)
    h, s, v = (hsv[..., i][inside.astype(bool)].astype(float) for i in range(3))
    if not len(v):
        return False
    if np.median(v) <= 70:
        return np.mean((s > 90) & (v > 40)) < 0.10 and np.mean(v > 150) < 0.15
    return 18 <= np.median(h) <= 35 and np.median(s) >= 120


def label_lines(meta: dict, w: int = 640, h: int = 480, bgr: np.ndarray | None = None) -> list[str]:
    hsv = None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lines = []
    for o in meta["found"]:
        c = kit_class(o)
        if c is None or len(o["outline_px"]) < 3 or (hsv is not None and not_kit(o["outline_px"], hsv)):
            continue
        xy = " ".join(f"{min(max(x / w, 0), 1):.5f} {min(max(y / h, 0), 1):.5f}" for x, y in o["outline_px"])
        lines.append(f"{c} {xy}")
    return lines


def scenes(stems: list[str]) -> list[list[str]]:
    """The looks grouped into scenes: a pick's wide look and the close looks after it -- a new one at
    every wide look, and wherever a minute or more passes. (A minute's gap alone made one scene of a
    whole session of picks in a row, and every 5th of those put that one session in val, 2026-10-12.)"""
    out, last = [], None
    for stem in sorted(stems):
        t = time.mktime(time.strptime(stem[:15], "%Y%m%d_%H%M%S"))
        if last is None or t - last >= SCENE_GAP_S or stem[16:].startswith("wide"):
            out.append([])
        out[-1].append(stem)
        last = t
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", type=Path, default=ROOT / "out" / "duel" / "frames")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "kit")
    ap.add_argument("--val-every", type=int, default=5)
    a = ap.parse_args()

    stems = [p.name[:-5] for p in a.frames.glob("*.json")
             if p.name[16:-5] in LOOKS and (a.frames / (p.name[:-5] + "_rgb.jpg")).exists()]
    if a.out.exists():
        shutil.rmtree(a.out)
    counts = {"train": [0, 0, [0] * len(NAMES)], "val": [0, 0, [0] * len(NAMES)]}
    for i, scene in enumerate(scenes(stems)):
        split = "val" if i % a.val_every == a.val_every - 1 else "train"
        for sub in ("images", "labels"):
            (a.out / sub / split).mkdir(parents=True, exist_ok=True)
        for stem in scene:
            lines = label_lines(json.loads((a.frames / f"{stem}.json").read_text()),
                                bgr=cv2.imread(str(a.frames / f"{stem}_rgb.jpg")))
            shutil.copy(a.frames / f"{stem}_rgb.jpg", a.out / "images" / split / f"{stem}.jpg")
            (a.out / "labels" / split / f"{stem}.txt").write_text("".join(f"{line}\n" for line in lines))
            counts[split][0] += 1
            counts[split][1] += 0 if lines else 1
            for line in lines:
                counts[split][2][int(line.split()[0])] += 1
    # no "path:": train/train_kit.py writes the set's absolute folder in where it trains
    (a.out / "kit.yaml").write_text("train: images/train\nval: images/val\nnames:\n"
                                     + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    for split, (n, empty, per) in counts.items():
        print(f"{split}: {n} pictures ({empty} with no kit piece), "
              + ", ".join(f"{name} {k}" for name, k in zip(NAMES, per)))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
