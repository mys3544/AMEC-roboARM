"""A LeRobot policy (ACT: tools/m3_policy.py) over HTTP for the simulator's closed loop (m3_sim.py evaluate), in
LeRobot's own python on Dionysos (C:/Users/Hercules/roboarm-lerobot/.venv), the robot's policy service's API
(tools/policy_service.py) on 127.0.0.1:8764:

    GET  /health   {"checkpoint", "chunk_size", "n_action_steps", "device"}
    POST /act      {"state": [J1..J6 as sent], "jpeg": base64 (RGB as JPEG)} -> {"chunk": [[J1..J6] x chunk], "ms"}
    GET  /quit     the server ends (the simulator says so when it is done)

    python policy_http.py CHECKPOINT [--device cpu|cuda] [--task TEXT]

On the CPU by default: the simulator's workers keep the GPU's memory (an inference a second, ~0.1-0.2 s on the
CPU, is no burden). LeRobot's own pipeline (preprocessor, predict_action_chunk, postprocessor), as m3_policy's
eval: what the robot's TensorRT engine matches to 0.2 deg.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

PORT = 8764
TASK = "pick up the 30 mm cube and put it in the box"


def load(checkpoint: Path, device: str):
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    path = checkpoint / "pretrained_model" if (checkpoint / "pretrained_model").exists() else checkpoint
    policy = ACTPolicy.from_pretrained(str(path))
    policy.config.device = device
    policy.to(torch.device(device)).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(path),
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def main() -> int:
    import torch
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--task", default=TASK)
    a = ap.parse_args()
    policy, pre, post = load(a.checkpoint, a.device)
    lock = threading.Lock()
    health = {"checkpoint": str(a.checkpoint), "chunk_size": policy.config.chunk_size,
              "n_action_steps": policy.config.n_action_steps, "device": a.device}

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
                self.reply(health)
            elif self.path == "/quit":
                self.reply({"ok": True})
                threading.Thread(target=server.shutdown, daemon=True).start()
            else:
                self.reply({"error": f"no such endpoint: GET {self.path}"}, 404)

        def do_POST(self) -> None:
            if self.path != "/act":
                self.reply({"error": f"no such endpoint: POST {self.path}"}, 404)
                return
            ask = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(ask["jpeg"]))).convert("RGB"))
            began = time.perf_counter()
            with lock, torch.inference_mode():
                batch = {"observation.images.wrist": torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255,
                         "observation.state": torch.tensor(ask["state"], dtype=torch.float32)[None], "task": [a.task]}
                chunk = post(policy.predict_action_chunk(pre(batch)))[0].float().cpu().numpy()
            self.reply({"chunk": np.round(chunk, 3).tolist(), "ms": round(1000 * (time.perf_counter() - began), 1)})

    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"policy {a.checkpoint} on 127.0.0.1:{PORT} ({a.device})", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
