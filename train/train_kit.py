"""Train the kit detector (PLAN.md M5) on hekate: YOLO26n/s-seg fine-tuned on the M3's own labelled
looks (m3pro/tools/kit_dataset.py builds the set; it is copied here as datasets\\kit).

    uv run python train_kit.py [--model yolo26n-seg.pt] [--epochs 150] [--name n]
    uv run python train_kit.py --export runs\\kit\\n\\weights\\best.pt   # -> best.onnx at 480x640

INT8 by quantization-aware training (PLAN.md M5; Ultralytics' quantize=8, NVIDIA ModelOpt): fine-tune
the float model with the quantizers in, then export its Q/DQ ranges straight into the ONNX:

    uv run python train_kit.py --int8 --model runs\\kit\\n\\weights\\best.pt --epochs 40 --name n-int8
    uv run python train_kit.py --int8 --export runs\\kit\\n-int8\\weights\\best.pt

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
    ap.add_argument("--int8", action="store_true", help="quantization-aware training / its INT8 export")
    a = ap.parse_args()
    quantize = {"quantize": 8} if a.int8 else {}

    if a.export:
        # the cameras' 4:3 at 640 wide: a quarter less work than 640x640 (PLAN.md M5)
        print(YOLO(str(a.export)).export(format="onnx", imgsz=(480, 640), simplify=True, **quantize))
        return
    # the set's own folder as its root: without an absolute "path:" Ultralytics looks for the
    # images under the working folder or its datasets folder, not next to the yaml
    data = a.data.resolve()
    lines = [line for line in data.read_text().splitlines() if not line.startswith("path:")]
    data.write_text("\n".join([f"path: {data.parent.as_posix()}", *lines]) + "\n")
    name = a.name or Path(a.model).stem.removeprefix("yolo26").split("-")[0] + ("-int8" if a.int8 else "")
    YOLO(a.model).train(data=str(data), epochs=a.epochs, imgsz=640, batch=a.batch,
                        patience=50, workers=4, project=str(HERE / "runs" / "kit"), name=name,
                        exist_ok=True, plots=True, **quantize)


if __name__ == "__main__":
    main()
