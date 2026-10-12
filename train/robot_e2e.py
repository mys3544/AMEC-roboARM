"""The kit detector end to end on a robot (PLAN.md M5): picture in, masks out -- Ultralytics' preprocess,
inference and postprocess -- on the robot's own pick looks (~/roboarm/out/pick/*_rgb.jpg), as a TensorRT
FP16 engine built here and in PyTorch. In the robot's ML venv, with the weights at
~/roboarm/models/kit/yolo26n_kit.pt (train/runs/kit/NAME/weights/best.pt):

    cd ~/roboarm && .venv-ml/bin/python models/kit/robot_e2e.py    # ~8 min: the engine build

On the M3 2026-10-12: engine 8.3 ms a picture (inference 3.2, the masks' postprocess 3.5), PyTorch 32.6.
"""
import glob
import os
import time

os.environ["YOLO_AUTOINSTALL"] = "false"   # never pip into the robot's pinned venv
import cv2
import numpy as np
from ultralytics import YOLO

PT = os.path.expanduser("~/roboarm/models/kit/yolo26n_kit.pt")
frames = [cv2.imread(f) for f in sorted(glob.glob(os.path.expanduser("~/roboarm/out/pick/*_rgb.jpg")))]
print(len(frames), "frames", frames[0].shape, flush=True)
t = time.time()
engine = YOLO(PT).export(format="engine", half=True, imgsz=(480, 640), device=0, simplify=True)
print(f"engine built in {time.time() - t:.0f} s: {engine}", flush=True)

def bench(model, label, rounds=3):
    for f in frames[:10]:
        model.predict(f, imgsz=(480, 640), verbose=False, half=True)
    stage = {"pre": [], "inf": [], "post": [], "wall": []}
    for _ in range(rounds):
        for f in frames:
            t = time.perf_counter()
            r = model.predict(f, imgsz=(480, 640), verbose=False, half=True)[0]
            stage["wall"].append(1000 * (time.perf_counter() - t))
            stage["pre"].append(r.speed["preprocess"]); stage["inf"].append(r.speed["inference"]); stage["post"].append(r.speed["postprocess"])
    print(label + ": " + ", ".join(f"{k} {np.median(v):.2f} (p99 {np.percentile(v, 99):.2f})" for k, v in stage.items()) + " ms", flush=True)

bench(YOLO(engine, task="segment"), "TensorRT FP16 engine")
bench(YOLO(PT), "PyTorch (CUDA, half)")
print("DONE_E2E")
