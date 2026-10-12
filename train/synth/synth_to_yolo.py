"""Isaac Sim's synthetic kit pictures (kit_synth.py: BasicWriter's rgb + instance segmentation) as the
real set's labels -- Ultralytics segmentation, the same three classes as m3pro/tools/kit_dataset.py --
and a set that trains on both and is judged on the real pictures only. On hekate, in train/:

    uv run python synth/synth_to_yolo.py [--raw datasets\\synth_raw] [--out datasets\\synth]
    uv run python train_kit.py --data datasets\\kit_synth.yaml --name n-synth

Each piece's polygon is the outline of what is seen of it (its instance mask: a piece behind another
is cut); a piece under MIN_PX pixels in view is left out (it is not labelled in the real set either).
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
NAMES = ["cube30", "cube40", "block"]   # m3pro/tools/kit_dataset.py's order
MIN_PX = 40


def polygons(seg_rgba: np.ndarray, semantics: dict) -> list[tuple[int, np.ndarray]]:
    """(class, polygon in pixels) for every kit piece in an instance-segmentation picture."""
    out = []
    for key, value in semantics.items():
        if value.get("class") not in NAMES:
            continue
        mask = np.all(seg_rgba == np.array(ast.literal_eval(key), np.uint8), axis=-1).astype(np.uint8)
        if int(mask.sum()) < MIN_PX:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest = max(contours, key=cv2.contourArea)
        poly = cv2.approxPolyDP(biggest, 1.0, True).reshape(-1, 2)
        if len(poly) >= 3:
            out.append((NAMES.index(value["class"]), poly))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", type=Path, default=HERE / "datasets" / "synth_raw")
    ap.add_argument("--out", type=Path, default=HERE / "datasets" / "synth")
    a = ap.parse_args()
    if a.out.exists():
        shutil.rmtree(a.out)
    (a.out / "images" / "train").mkdir(parents=True)
    (a.out / "labels" / "train").mkdir(parents=True)
    counts, pictures = [0] * len(NAMES), 0
    for rgb_file in sorted(a.raw.glob("rgb_*.png")):
        n = rgb_file.stem.split("_")[-1]
        seg = cv2.imread(str(a.raw / f"instance_segmentation_{n}.png"), cv2.IMREAD_UNCHANGED)
        semantics = json.loads((a.raw / f"instance_segmentation_semantics_mapping_{n}.json").read_text())
        if seg is None:
            continue
        rgb = cv2.imread(str(rgb_file))
        h, w = rgb.shape[:2]
        lines = []
        for cls, poly in polygons(cv2.cvtColor(seg, cv2.COLOR_BGRA2RGBA), semantics):
            lines.append(f"{cls} " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in poly))
            counts[cls] += 1
        cv2.imwrite(str(a.out / "images" / "train" / f"synth_{n}.jpg"), rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (a.out / "labels" / "train" / f"synth_{n}.txt").write_text("".join(f"{line}\n" for line in lines))
        pictures += 1
    kit = (a.out.parent / "kit").resolve().as_posix()   # the real set, beside it
    (a.out.parent / "kit_synth.yaml").write_text(
        f"train:\n  - {kit}/images/train\n  - {(a.out / 'images' / 'train').resolve().as_posix()}\n"
        f"val: {kit}/images/val\nnames:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(NAMES)))
    print(f"{pictures} pictures, " + ", ".join(f"{name} {k}" for name, k in zip(NAMES, counts)) + f" -> {a.out}")


if __name__ == "__main__":
    main()
