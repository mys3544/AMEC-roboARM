"""The M3's policy (ACT, trained on its own recordings: tools/m3_policy.py) in TensorRT, in its own
process from ~/roboarm/.venv-ml (tools/ml_env.sh: JetPack's CUDA torch and TensorRT), so the bridge
needs neither. The bridge's policy job (roboarm/m3policy.py) asks it over 127.0.0.1:8763, once a chunk:

    GET  /health   {"model" (act.json), "precision", "ms" (the last inference), "engine_check" (below)}
    POST /act      {"state": [J1..J6 as sent], "jpeg": base64 (the colour picture, BGR as the camera
                   gives it)} -> {"chunk": [[J1..J6] x chunk_size], "ms"}

The model: models/policy/act.onnx with act.json and act_check.npz (`m3_policy.py robot` puts them
there). Its engine, act.engine, is built beside it when act.onnx is newer (FP16 unless --fp32: a few
minutes on the Orin), then checked against act_check.npz -- LeRobot's own chunks for held-out frames:

    .venv-ml/bin/python tools/policy_service.py --build [--fp32]   # build when needed, check, exit
    .venv-ml/bin/python tools/policy_service.py                    # serve (robot/roboarm-m3policy.service)

Standard library HTTP, as the teach model's (tools/teach_service.py).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch

ROOT = Path(__file__).resolve().parents[1]
HOST, PORT = "127.0.0.1", 8763
MODEL = ROOT / "models" / "policy"
LOGGER = trt.Logger(trt.Logger.WARNING)


def build(onnx: Path, fp32: bool) -> bytes:
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse(onnx.read_bytes()):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if not fp32:
        config.set_flag(trt.BuilderFlag.FP16)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT could not build the engine (its log above)")
    return bytes(plan)


class Policy:
    """The engine on its own CUDA stream, its inputs and output in torch tensors on the GPU."""

    def __init__(self, folder: Path = MODEL, fp32: bool = False):
        self.meta = json.loads((folder / "act.json").read_text())
        onnx, engine = folder / "act.onnx", folder / "act.engine"
        flag = folder / "act.engine.precision"   # what the engine was built as
        self.precision = "fp32" if fp32 else "fp16"
        if (not engine.exists() or engine.stat().st_mtime < onnx.stat().st_mtime
                or not flag.exists() or flag.read_text() != self.precision):
            print(f"building {engine.name} ({self.precision}) from {onnx.name}: a few minutes", flush=True)
            started = time.time()
            engine.write_bytes(build(onnx, fp32))
            flag.write_text(self.precision)
            print(f"built in {time.time() - started:.0f} s", flush=True)
        self.engine = trt.Runtime(LOGGER).deserialize_cuda_engine(engine.read_bytes())
        self.context = self.engine.create_execution_context()
        rows, cols = self.meta["image"]
        self.io = {"image": torch.empty((1, 3, rows, cols), dtype=torch.float32, device="cuda"),
                   "state": torch.empty((1, 6), dtype=torch.float32, device="cuda"),
                   "chunk": torch.empty((1, self.meta["chunk_size"], 6), dtype=torch.float32, device="cuda")}
        for name, tensor in self.io.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        self.stream = torch.cuda.Stream()
        self.lock = threading.Lock()
        self.ms = None
        self.check = None

    def __call__(self, rgb: np.ndarray, state) -> np.ndarray:
        """RGB H x W x 3 uint8 and the six joints -> chunk_size x 6 joints (degrees)."""
        with self.lock, torch.cuda.stream(self.stream):
            started = time.perf_counter()
            self.io["image"].copy_(torch.from_numpy(np.ascontiguousarray(rgb)).to("cuda").permute(2, 0, 1)[None]
                                   .float().div_(255))
            self.io["state"].copy_(torch.as_tensor(np.asarray(state, np.float32)).view(1, 6))
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT did not run the engine")
            out = self.io["chunk"][0].cpu().numpy()   # waits for the stream
            self.ms = round(1000 * (time.perf_counter() - started), 1)
        return out

    def checked(self, folder: Path = MODEL) -> dict:
        """The engine on act_check.npz's frames against LeRobot's chunks: max and mean |diff| (deg)."""
        data = np.load(folder / "act_check.npz")
        got = np.stack([self(image, state) for image, state in zip(data["images"], data["states"])])
        diff = np.abs(got - data["chunks"])
        times = []
        for _ in range(20):
            self(data["images"][0], data["states"][0])
            times.append(self.ms)
        self.check = {"frames": len(got), "max_deg": round(float(diff.max()), 3), "mean_deg": round(float(diff.mean()), 4),
                      "max_per_joint": np.round(diff.max(axis=(0, 1)), 3).tolist(), "ms": float(np.median(times))}
        return self.check


def make_handler(policy: Policy):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def reply(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self.reply({"model": policy.meta, "precision": policy.precision, "ms": policy.ms,
                            "engine_check": policy.check})
            else:
                self.reply({"error": f"no such endpoint: GET {self.path}"}, 404)

        def do_POST(self) -> None:
            if self.path != "/act":
                self.reply({"error": f"no such endpoint: POST {self.path}"}, 404)
                return
            try:
                ask = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                bgr = cv2.imdecode(np.frombuffer(base64.b64decode(ask["jpeg"]), np.uint8), cv2.IMREAD_COLOR)
                if bgr is None or list(bgr.shape[:2]) != policy.meta["image"]:
                    raise ValueError(f"the picture is not a {policy.meta['image']} JPEG")
                chunk = policy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ask["state"])
            except (ValueError, KeyError, TypeError) as exc:
                self.reply({"error": f"{type(exc).__name__}: {exc}"}, 400)
                return
            self.reply({"chunk": np.round(chunk, 3).tolist(), "ms": policy.ms})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", action="store_true", help="build the engine when needed, check it, exit")
    parser.add_argument("--fp32", action="store_true", help="an FP32 engine (FP16 by default)")
    args = parser.parse_args()
    policy = Policy(fp32=args.fp32)
    check = policy.checked()
    print(f"{policy.meta['checkpoint']} ({policy.precision}) against LeRobot on {check['frames']} held-out frames: "
          f"max |diff| {check['max_deg']} deg (per joint {check['max_per_joint']}), mean {check['mean_deg']}; "
          f"{check['ms']} ms an inference", flush=True)
    if args.build:
        return 0
    server = ThreadingHTTPServer((HOST, PORT), make_handler(policy))
    print(f"policy on {HOST}:{PORT}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
