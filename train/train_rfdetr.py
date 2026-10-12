"""RF-DETR-Seg on the kit set, on hekate: the model table's transformer row (PLAN.md M5, model 3;
Apache-2.0, said to fine-tune well on small data) beside YOLO26 (train_kit.py), on the same set
(datasets\\kit, which m3pro/tools/kit_dataset.py builds; RF-DETR reads its Ultralytics layout and
polygons as they are, from a data.yaml).

    uv run python train_rfdetr.py [--size n|s] [--epochs 100] [--batch 8]

At the variant's own resolution (Seg-Nano 312, Seg-Small 384 px square), as published. Runs land in
runs\\rfdetr\\NAME, the ONNX beside the checkpoints; the robots build their engines from it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rfdetr import RFDETRSegNano, RFDETRSegSmall

HERE = Path(__file__).resolve().parent
SIZES = {"n": RFDETRSegNano, "s": RFDETRSegSmall}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=HERE / "datasets" / "kit")
    ap.add_argument("--size", choices=sorted(SIZES), default="n")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default=None, help="run name (default: seg-SIZE)")
    a = ap.parse_args()

    data = a.data.resolve()
    # RF-DETR finds a YOLO set by its data.yaml; the folders resolve against an absolute "path:"
    lines = [line for line in (data / "kit.yaml").read_text().splitlines() if not line.startswith("path:")]
    (data / "data.yaml").write_text("\n".join([f"path: {data.as_posix()}", *lines]) + "\n")
    out = HERE / "runs" / "rfdetr" / (a.name or f"seg-{a.size}")
    model = SIZES[a.size]()
    model.train(dataset_dir=str(data), epochs=a.epochs, batch_size=a.batch, output_dir=str(out), device="cuda")
    model.export(output_dir=str(out))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
