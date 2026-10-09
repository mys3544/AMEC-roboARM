"""The teach's model on the M3 (PLAN.md track A, A1-A2): YOLOE-26 with visual prompts, in its own
process from ~/roboarm/.venv-ml (tools/ml_env.sh: JetPack's CUDA torch, Ultralytics 8.4), so the
bridge needs no torch. The bridge's jobs (roboarm/m3teach.py) ask it over 127.0.0.1:8762:

    GET  /health    {"model", "device", "names"}: the taught things
    POST /embed     {"jpeg": base64, "boxes": [[x1, y1, x2, y2], ...]} -> {"embeddings": [[512], ...]}
                    each box's visual-prompt embedding (YOLOE's SAVPE), unit length
    POST /teach     {"name", "embedding": [512]} -> {"names"}: name taught (or taught anew)
    POST /forget    {"name"} -> {"names"}
    POST /detect    {"jpeg": base64, "conf": 0.1} -> {"detections": [{"name", "conf", "box"}]}, pixels

The taught things are kept in out/teach/library.npz (names, embeddings) and loaded at start. A
systemd user unit runs it (robot/roboarm-m3teach.service). Standard library HTTP, as the X3's
vision service.

2026-10-09, the first probe (one look, five things taught from their depth boxes): right in that
layout, wrong in the next (an apricot cube called the news one at 0.85) -- one view of one face is
too little, so the teach averages several views (A2), and the library's other things compete.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

ROOT = Path(__file__).resolve().parents[1]
HOST, PORT = "127.0.0.1", 8762
MODEL = os.environ.get("ROBOARM_TEACH_MODEL", str(ROOT / "models" / "yoloe-26s-seg.pt"))
LIBRARY = ROOT / "out" / "teach" / "library.npz"


class Teacher:
    def __init__(self, weights: str):
        self.lock = threading.Lock()
        self.model = YOLOE(weights)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.prompter = YOLOEVPSegPredictor(overrides={"task": "segment", "mode": "predict", "save": False,
                                                       "batch": 1, "verbose": False, "device": self.device})
        self.prompter.set_prompts({"bboxes": np.zeros((1, 4), np.float32), "cls": np.zeros(1, int)})
        self.prompter.setup_model(model=self.model.model, verbose=False)
        self.names, self.embeddings = [], np.zeros((0, 0), np.float32)
        if LIBRARY.exists():
            with np.load(LIBRARY, allow_pickle=False) as data:
                self.names, self.embeddings = data["names"].tolist(), data["embeddings"].astype(np.float32)
            self._classes()

    def embed(self, image: np.ndarray, boxes: list) -> np.ndarray:
        """Each box's visual-prompt embedding, unit length (n x D)."""
        with self.lock:
            boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
            self.prompter.set_prompts({"bboxes": boxes, "cls": np.arange(len(boxes))})
            self.prompter.model.names = {i: f"object{i}" for i in range(len(boxes))}
            vpe = self.prompter.get_vpe(image)[0].float()
            return torch.nn.functional.normalize(vpe, dim=-1).cpu().numpy()

    def teach(self, name: str, embedding) -> list[str]:
        e = np.asarray(embedding, np.float32)
        e = e / np.linalg.norm(e)
        with self.lock:
            keep = [i for i, n in enumerate(self.names) if n != name]
            self.names = [self.names[i] for i in keep] + [name]
            self.embeddings = np.vstack([self.embeddings[keep].reshape(len(keep), -1), e[None]])
            self._save()
            return list(self.names)

    def forget(self, name: str) -> list[str]:
        with self.lock:
            keep = [i for i, n in enumerate(self.names) if n != name]
            self.names, self.embeddings = [self.names[i] for i in keep], self.embeddings[keep]
            self._save()
            return list(self.names)

    def _save(self) -> None:
        LIBRARY.parent.mkdir(parents=True, exist_ok=True)
        np.savez(LIBRARY, names=np.asarray(self.names, dtype=np.str_), embeddings=self.embeddings)
        if self.names:
            self._classes()

    def _classes(self) -> None:
        pe = torch.from_numpy(self.embeddings[None]).to(self.device)
        self.model.set_classes(list(self.names), pe)

    def detect(self, image: np.ndarray, conf: float) -> list[dict]:
        with self.lock:
            if not self.names:
                return []
            r = self.model.predict(image, conf=conf, verbose=False, device=self.device)[0]
        if r.boxes is None:
            return []
        return [{"name": self.names[int(c)], "conf": round(float(s), 3), "box": [round(float(v), 1) for v in b]}
                for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                   strict=True)]


def decode(b64: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)


def make_handler(teacher: Teacher):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def reply(self, payload, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self.reply({"model": Path(MODEL).name, "device": teacher.device, "names": teacher.names})
            self.reply({"error": f"no such endpoint: GET {self.path}"}, 404)

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                began = time.perf_counter()
                if self.path == "/embed":
                    out = {"embeddings": teacher.embed(decode(body["jpeg"]), body["boxes"]).tolist()}
                elif self.path == "/teach":
                    out = {"names": teacher.teach(str(body["name"]), body["embedding"])}
                elif self.path == "/forget":
                    out = {"names": teacher.forget(str(body["name"]))}
                elif self.path == "/detect":
                    out = {"detections": teacher.detect(decode(body["jpeg"]), float(body.get("conf", 0.1)))}
                else:
                    return self.reply({"error": f"no such endpoint: POST {self.path}"}, 404)
                self.reply({**out, "ms": round(1000 * (time.perf_counter() - began), 1)})
            except Exception as exc:  # noqa: BLE001 -- the caller sees what went wrong
                self.reply({"error": f"{type(exc).__name__}: {exc}"}, 400)

    return Handler


def main() -> int:
    teacher = Teacher(MODEL)
    print(f"teach service on {HOST}:{PORT}: {Path(MODEL).name} on {teacher.device}, "
          f"{len(teacher.names)} taught: {teacher.names}", flush=True)
    ThreadingHTTPServer((HOST, PORT), make_handler(teacher)).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
