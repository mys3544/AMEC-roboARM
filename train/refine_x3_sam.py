"""The X3's pictures in the kit set (x3plus/tools/kit_dataset_x3.py, files x3_*) get their outlines
from SAM 2. The X3 puts each piece's box into its pictures from where its survey found it, which
lands 1-2 cm off on the pictures taken while the arm turns (2026-10-15); SAM, prompted with a box
grown around that outline, finds the piece's own edges. A picture where a mask does not agree
with its outline -- far smaller or larger, or mostly elsewhere -- is taken out of the set: a wrong
label, or a piece left unlabelled, teaches worse than no picture.

The X3's surveys miss pieces now and then -- the user's tagged cubes and blocks among them, left
unlabelled on 2026-10-08 -- so the kit detector as it is (--checker, trained on the M3's pictures)
looks too: a piece it is sure of where no outline is takes the picture out. Surer still (CHECK_EMPTY)
on a picture with no outline at all, where its false cubes on the X3's tag board (0.3-0.6) would
otherwise take out the very pictures that are to teach it better.

    .venv\\Scripts\\python.exe refine_x3_sam.py [--set datasets\\kit] [--model sam2.1_b.pt] [--draw 24]

`--draw N` keeps N pictures in SET\\x3_sam_check\\: the X3's outline yellow, SAM's green (red: taken
out), what the checker saw unlabelled magenta.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"
import cv2
import numpy as np
from ultralytics import SAM, YOLO

CHECKER = Path("runs") / "kit" / "n_cycle" / "weights" / "best.pt"   # 0.791 mask mAP50 on the M3's val
CHECK, CHECK_EMPTY = 0.6, 0.75   # the checker's confidence for a piece left unlabelled
GROW, GROW_PX = 0.25, 15   # the prompt box grown by this share of its size each way, at least this many px
# With the box alone SAM often took the white paper round a white picture cube (2026-10-15, 507 of
# 704 pictures out); the outline's centre as a point on the piece holds it to the piece.
AREA = (0.4, 2.2)          # a mask's area against its outline's, to agree
MIN_IOU = 0.2


def pieces(label: Path, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    out = []
    for line in label.read_text().split("\n"):
        if line.strip():
            c, *xy = line.split()
            out.append((int(c), (np.array(xy, float).reshape(-1, 2) * [w, h]).astype(np.float32)))
    return out


def refine(model, image: np.ndarray, found: list) -> tuple[list[tuple[int, np.ndarray]] | None, list]:
    """(the pieces with SAM's outlines, or None when one disagrees; SAM's masks for drawing)."""
    h, w = image.shape[:2]
    boxes, centres = [], []
    for _c, poly in found:
        (x1, y1), (x2, y2) = poly.min(axis=0), poly.max(axis=0)
        gx, gy = max(GROW_PX, GROW * (x2 - x1)), max(GROW_PX, GROW * (y2 - y1))
        boxes.append([max(0, x1 - gx), max(0, y1 - gy), min(w - 1, x2 + gx), min(h - 1, y2 + gy)])
        centres.append([float(np.clip(poly[:, 0].mean(), 0, w - 1)), float(np.clip(poly[:, 1].mean(), 0, h - 1))])
    result = model(image, bboxes=boxes, points=centres, labels=[1] * len(centres), verbose=False)[0]
    masks = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
             for m in result.masks.data.cpu().numpy() > 0.5]
    out = []
    for (c, poly), mask in zip(found, masks, strict=True):
        drawn = np.zeros((h, w), np.uint8)
        cv2.fillPoly(drawn, [poly.astype(np.int32)], 1)
        drawn = drawn > 0
        iou = (mask & drawn).sum() / max(1, (mask | drawn).sum())
        if not AREA[0] <= mask.sum() / max(1, drawn.sum()) <= AREA[1] or iou < MIN_IOU:
            return None, masks
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True).reshape(-1, 2)
        if len(edge) < 3:
            return None, masks
        out.append((c, edge.astype(np.float32)))
    return out, masks


def unlabelled(checker, image: np.ndarray, labels: list[tuple[int, np.ndarray]]) -> list[np.ndarray]:
    """The boxes (x1, y1, x2, y2) of pieces the checker is sure of that no label covers: its box's
    centre in none of the labels' outlines."""
    sure = CHECK if labels else CHECK_EMPTY
    boxes = checker(image, conf=sure, verbose=False)[0].boxes.xyxy.cpu().numpy()
    return [b for b in boxes
            if not any(cv2.pointPolygonTest(poly.reshape(-1, 1, 2), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), False) >= 0
                       for _c, poly in labels)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", type=Path, default=Path("datasets") / "kit")
    ap.add_argument("--model", default="sam2.1_b.pt")
    ap.add_argument("--checker", type=Path, default=CHECKER)
    ap.add_argument("--no-checker", action="store_true",
                    help="no second look (a test set: the checker would keep the pictures it gets right)")
    ap.add_argument("--draw", type=int, default=0)
    a = ap.parse_args()
    model, checker = SAM(a.model), None if a.no_checker else YOLO(a.checker)
    check = a.set / "x3_sam_check"
    check.mkdir(exist_ok=True)
    kept = disagreed = missed = drawn = 0
    for image_path in sorted(a.set.glob("images/*/x3_*.jpg")):
        label = a.set / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        found = pieces(label, w, h)
        refined, masks = refine(model, image, found) if found else ([], [])
        missing = unlabelled(checker, image, refined) if refined is not None and checker is not None else []
        if drawn < a.draw and (found or missing):
            shown = image.copy()
            for (_c, poly), mask in zip(found, masks, strict=False):
                shown[mask] = 0.5 * shown[mask] + ((0, 110, 0) if refined is not None else (0, 0, 110))
                cv2.polylines(shown, [poly.astype(np.int32)], True, (0, 255, 255), 2)
            for x1, y1, x2, y2 in missing:
                cv2.rectangle(shown, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
            cv2.imwrite(str(check / image_path.name), shown)
            drawn += 1
        if refined is None or missing:
            image_path.unlink()
            label.unlink()
            disagreed += refined is None
            missed += refined is not None
            continue
        if found:
            label.write_text("".join(f"{c} " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in poly) + "\n"
                                     for c, poly in refined))
        kept += 1
    print(f"X3 pictures: {kept} kept (outlines by SAM), {disagreed} taken out where a mask disagreed, "
          f"{missed} where the checker saw a piece unlabelled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
