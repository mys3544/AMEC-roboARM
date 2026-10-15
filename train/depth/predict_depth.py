"""The depth duel's big contestants (PLAN.md S1): monocular depth networks too big for the robots,
run on hekate's 5090 over the M3's saved looks (m3pro/out/duel/frames, copied to datasets\\duel).

    .venv\\Scripts\\python.exe predict_depth.py [--models da2s,da2l,...] [--frames DIR] [--out DIR]

For every look and model: out\\MODEL\\STEM.npy, float16 at the look's own size, and out\\MODEL\\kind.txt
saying what it holds -- "disparity" (relative, larger = nearer: anchored to the floor on the laptop
as Depth Anything V2 Small is) or "depth" (metres as the model reads them; Depth Pro's with the
camera's own focal length, not the one it guessed); out\\MODEL\\ms.json the milliseconds a picture
took on the GPU. m3pro/tools/m3_duel.py run scores them beside the others.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
MODELS = {   # name: (Hugging Face id, what it gives)
    "da2s": ("depth-anything/Depth-Anything-V2-Small-hf", "disparity"),   # the robots' own, through this path
    "da2l": ("depth-anything/Depth-Anything-V2-Large-hf", "disparity"),
    "da2l_metric": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", "depth"),
    "depthpro": ("apple/DepthPro-hf", "depth"),
}


def load(name: str, device: str):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    repo, _kind = MODELS[name]
    processor = AutoImageProcessor.from_pretrained(repo, use_fast=True)
    model = AutoModelForDepthEstimation.from_pretrained(repo).to(device).eval()
    return processor, model


@torch.inference_mode()
def predict(name: str, processor, model, image: Image.Image, fx: float, device: str) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    post = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])[0]
    out = post["predicted_depth"].float()
    if name == "depthpro":   # metric depth goes with the focal length: ours, not its guess
        out = out * fx / float(post["focal_length"])
    return out.cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--frames", type=Path, default=HERE.parent / "datasets" / "duel")
    ap.add_argument("--out", type=Path, default=HERE / "out")
    a = ap.parse_args()
    device = "cuda"
    looks = sorted(a.frames.glob("*_rgb.jpg"))
    print(f"{len(looks)} looks in {a.frames}, on {torch.cuda.get_device_name()}", flush=True)
    for name in a.models.split(","):
        try:
            processor, model = load(name, device)
        except Exception as exc:  # noqa: BLE001 -- one model failing must not stop the others
            print(f"{name}: not loaded -- {type(exc).__name__}: {exc}", flush=True)
            continue
        folder = a.out / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "kind.txt").write_text(MODELS[name][1])
        spent = []
        for i, path in enumerate(looks):
            stem = path.name[: -len("_rgb.jpg")]
            fx = json.loads((path.parent / f"{stem}.json").read_text())["K"][0][0]
            image = Image.open(path).convert("RGB")
            torch.cuda.synchronize()
            began = time.perf_counter()
            out = predict(name, processor, model, image, fx, device)
            torch.cuda.synchronize()
            if i:   # the first one warms up
                spent.append(1000 * (time.perf_counter() - began))
            np.save(folder / f"{stem}.npy", out.astype(np.float16))
        ms = statistics.median(spent) if spent else float("nan")
        (folder / "ms.json").write_text(json.dumps({"median_ms": ms, "pictures": len(looks)}))
        print(f"{name}: {len(looks)} looks, {ms:.0f} ms a picture, {torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak",
              flush=True)
        del processor, model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return 0


if __name__ == "__main__":
    sys.exit(main())
