"""Train the kit detector (PLAN.md M5) on hekate: YOLO26n/s-seg fine-tuned on the M3's own labelled
looks (m3pro/tools/kit_dataset.py builds the set; it is copied here as datasets\\kit).

    uv run python train_kit.py [--model yolo26n-seg.pt] [--epochs 150] [--name n]
    uv run python train_kit.py --export runs\\kit\\n\\weights\\best.pt   # -> best.onnx at 480x640

Runs land in runs\\kit\\NAME. The engine is built on each robot from the ONNX, in the venv that
serves it (TensorRT differs between the robots and from this PC's).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=HERE / "datasets" / "kit" / "kit.yaml")
    ap.add_argument("--model", default="yolo26n-seg.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default=None, help="run name (default: the model's size letter)")
    ap.add_argument("--export", type=Path, default=None, help="export these weights to ONNX instead")
    a = ap.parse_args()

    if a.export:
        # the cameras' 4:3 at 640 wide: a quarter less work than 640x640 (PLAN.md M5)
        print(YOLO(str(a.export)).export(format="onnx", imgsz=(480, 640), simplify=True))
        return
    # the set's own folder as its root: without an absolute "path:" Ultralytics looks for the
    # images under the working folder or its datasets folder, not next to the yaml
    data = a.data.resolve()
    lines = [line for line in data.read_text().splitlines() if not line.startswith("path:")]
    data.write_text("\n".join([f"path: {data.parent.as_posix()}", *lines]) + "\n")
    name = a.name or a.model.removeprefix("yolo26").split("-")[0]
    YOLO(a.model).train(data=str(data), epochs=a.epochs, imgsz=640, batch=a.batch,
                        patience=50, workers=4, project=str(HERE / "runs" / "kit"), name=name,
                        exist_ok=True, plots=True)


if __name__ == "__main__":
    main()
