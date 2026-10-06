#!/usr/bin/env python3
"""Build a TensorRT engine from a .pt model, for faster inference on the Orin.

Runs INSIDE the vision container (it needs CUDA torch + TensorRT + the GPU):

    docker compose --profile vision run --rm vision python tools/export_engine.py

The engine is written next to the .pt in /app/models (a bind mount, so it
survives the container) and is what ROBOARM_VISION_MODEL should then point at.

Why bother: FP16 on this Orin is roughly 2x the throughput of the PyTorch model
at no accuracy cost worth measuring for detection. Why in-container: a TensorRT
engine is tied to the exact TensorRT build, CUDA version and GPU it was created
on -- an engine exported on the host can fail to deserialize in the container
even when both are "JetPack 6.2". Export where you run.

The engine bakes in its input size, so it must be rebuilt to change imgsz, and a
mismatched ROBOARM_VISION_IMGSZ at serve time is ignored in favour of the engine.
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    default_model = os.environ.get("ROBOARM_VISION_MODEL", "/app/models/yoloe-26s-seg-pf.pt")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=default_model,
                        help=f"source .pt (default {default_model})")
    parser.add_argument("--imgsz", type=int,
                        default=int(os.environ.get("ROBOARM_VISION_IMGSZ", "640")))
    parser.add_argument("--fp32", action="store_true",
                        help="skip FP16 (larger, slower, marginally more precise)")
    parser.add_argument("--int8", action="store_true",
                        help="INT8 -- needs a calibration set; only if you know you want it")
    args = parser.parse_args()

    source = Path(args.model)
    if source.suffix == ".engine":
        print(f"{source} is already an engine.", file=sys.stderr)
        return 1
    if not source.exists():
        print(f"no model at {source} -- fetch the weights into ./models first",
              file=sys.stderr)
        return 1

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        print("CUDA is not available in here -- are you running this inside the "
              "vision container with the GPU passed through?", file=sys.stderr)
        return 1

    print(f"exporting {source.name}  imgsz={args.imgsz}  "
          f"{'int8' if args.int8 else 'fp32' if args.fp32 else 'fp16'}")
    out = YOLO(str(source)).export(
        format="engine",
        imgsz=args.imgsz,
        half=not args.fp32 and not args.int8,
        int8=args.int8,
        device=0,
        verbose=False,
    )
    engine = Path(out)
    print(f"\nwrote {engine}  ({engine.stat().st_size / 1e6:.1f} MB)")
    print(f"point the service at it:  ROBOARM_VISION_MODEL={engine}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
