"""The kit detector judged by where its pictures come from (PLAN.md M5). The set's val holds the M3's
looks and, since 2026-10-15, the X3's (x3_*): one mask mAP50 over both says little about either.
Each model on each part -- and on a held-out set (x3plus/tools/kit_dataset_x3.py's test set) with
--test -- overall and per class.

    .venv\\Scripts\\python.exe eval_kit.py MODEL.pt [MODEL.pt ...] [--set datasets\\kit] [--test DIR ...]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"
from ultralytics import YOLO

NAMES = ["cube30", "cube40", "block"]
OUT = Path("runs") / "kit_eval"


def part(name: str, images: list[Path]) -> Path:
    """A data yaml whose val is exactly `images` (their labels beside them, as Ultralytics finds them)."""
    listed = OUT / f"{name}.txt"
    listed.write_text("".join(f"{p.resolve()}\n" for p in images))
    data = OUT / f"{name}.yaml"
    data.write_text(f"train: {listed.resolve()}\nval: {listed.resolve()}\nnames:\n"
                    + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--set", type=Path, default=Path("datasets") / "kit")
    ap.add_argument("--test", type=Path, action="append", default=[],
                    help="a held-out set (DIR/images, labels beside), named by its folder; again for more")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    val = sorted((a.set / "images" / "val").glob("*.jpg"))
    parts = {"M3 val": [p for p in val if not p.name.startswith("x3_")],
             "X3 val": [p for p in val if p.name.startswith("x3_")]}
    for test in a.test:   # the X3's looks_test, the M3's kit_test sets (m3pro/tools/kit_test_m3.py)
        parts[test.name] = sorted((test / "images").rglob("*.jpg"))
    for model in a.models:
        label = model.parent.parent.name if model.stem in ("best", "last") else model.stem
        for name, images in parts.items():
            if not images:
                continue
            m = YOLO(model).val(data=str(part(name.replace(" ", "_"), images)), imgsz=640, batch=16, plots=False,
                                verbose=False, project=str(OUT), name="val", exist_ok=True)
            per = dict(zip(m.ap_class_index, m.seg.ap50, strict=True))
            print(f"{label:22s} {name:8s} {len(images):4d} pictures  mask mAP50 {m.seg.map50:.3f}  "
                  + "  ".join(f"{n} {per[i]:.3f}" if i in per else f"{n} --" for i, n in enumerate(NAMES)),
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
