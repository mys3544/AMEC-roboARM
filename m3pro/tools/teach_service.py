"""The teach's models on the M3 (PLAN.md track A, A1-A2), in their own process from ~/roboarm/.venv-ml
(tools/ml_env.sh: JetPack's CUDA torch, Ultralytics 8.4), so the bridge needs no torch. The bridge's
jobs (roboarm/m3teach.py) ask it over 127.0.0.1:8762:

    GET  /health    {"models", "device", "names", "views"}: the taught things, and views of each
    POST /identify  {"jpeg": base64, "boxes": [[x1, y1, x2, y2], ...]} -> {"things": [{"scores":
                    {name: similarity}, "name", "sure", "margin"}, ...]}: which taught thing each
                    box holds -- DINOv2 on its crop against the taught views' crops
    POST /teach     {"name", "embedding": [512]} -> {"names"}: `name` taught (anew) -- its YOLOE
                    prompt, and its DINOv2 views from out/teach/NAME/ (the bridge has put them there)
    POST /forget    {"name"} -> {"names"}
    POST /embed     {"jpeg", "boxes"} -> {"embeddings": [[512], ...]}: YOLOE-26's visual-prompt
                    embedding (SAVPE) of each box, unit length
    POST /detect    {"jpeg", "conf": 0.1} -> {"detections": [{"name", "conf", "box"}]}: YOLOE with
                    the taught prompts as its classes, pixels

Who is who: DINOv2 (ViT-S/14, the CLS token and the mean patch token of a square crop round the box,
unit length), scored against the views a thing was taught from -- the mean similarity of its BEST
three (all of them until 2026-10-12, when things were first taught from scans all round). 2026-10-09,
the four picture cubes taught, a scene of four: DINOv2 named 4 of 4, the right name 0.09-0.17 ahead;
YOLOE's visual prompts 1 of 4, at 640 px and at 1280 (one name won everywhere: its prompts tell a
cube from a cup, not one print from another). So YOLOE stays for finding things where there is no
depth, and for A3 (prompts as an engine input); DINOv2 says which. The user's choice.

The library: out/teach/library.npz (the names and their YOLOE prompts); each name's DINOv2 views are
made from its folder of views at start and at /teach. A systemd user unit runs it
(robot/roboarm-m3teach.service). Standard library HTTP, as the X3's vision service.
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
DINO = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"   # torch.hub's, once fetched
TEACH = ROOT / "out" / "teach"
LIBRARY = TEACH / "library.npz"
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
CROP_GROW, CROP_PX = 0.15, 224
BEST = 3   # the taught views of a name a crop is scored on: its most like it


def crop(image: np.ndarray, box) -> np.ndarray:
    """The square round `box`, CROP_GROW larger, as DINOv2's normalised CROP_PX input (H x W x 3)."""
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1 + CROP_GROW)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(image.shape[1], cx + side / 2)), int(min(image.shape[0], cy + side / 2))
    patch = cv2.resize(image[b:d, a:c], (CROP_PX, CROP_PX), interpolation=cv2.INTER_CUBIC)
    return ((cv2.cvtColor(patch, cv2.COLOR_BGR2RGB) / 255.0 - MEAN) / STD).astype(np.float32)


class Teacher:
    def __init__(self, weights: str):
        self.lock = threading.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLOE(weights)
        self.model.to(self.device)
        self.prompter = YOLOEVPSegPredictor(overrides={"task": "segment", "mode": "predict", "save": False,
                                                       "batch": 1, "verbose": False, "device": self.device})
        self.prompter.set_prompts({"bboxes": np.zeros((1, 4), np.float32), "cls": np.zeros(1, int)})
        self.prompter.setup_model(model=self.model.model, verbose=False)
        self.dino = torch.hub.load(str(DINO), "dinov2_vits14", source="local").to(self.device).eval()
        self.names, self.embeddings, self.views = [], np.zeros((0, 0), np.float32), {}
        if LIBRARY.exists():
            with np.load(LIBRARY, allow_pickle=False) as data:
                self.names, self.embeddings = data["names"].tolist(), data["embeddings"].astype(np.float32)
            self._classes()
            for name in self.names:
                self.views[name] = self._views_of(name)

    @torch.no_grad()
    def features(self, image: np.ndarray, boxes) -> np.ndarray:
        """DINOv2's unit embedding of each box's crop (n x 768: CLS, then the mean patch token)."""
        x = torch.from_numpy(np.stack([crop(image, b) for b in boxes]).transpose(0, 3, 1, 2)).to(self.device)
        out = self.dino.forward_features(x)
        both = torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(1)], dim=1)
        return torch.nn.functional.normalize(both, dim=-1).cpu().numpy()

    def _views_of(self, name: str) -> np.ndarray:
        """DINOv2 embeddings of every view in out/teach/NAME/ (its colour picture and the box in its JSON)."""
        rows = []
        for meta in sorted((TEACH / name.replace("/", "_")).glob("*.json")):
            picture = meta.with_name(meta.stem + "_rgb.jpg")
            if picture.exists():
                rows.append(self.features(cv2.imread(str(picture)), [json.loads(meta.read_text())["box"]])[0])
        return np.asarray(rows, np.float32).reshape(len(rows), -1)

    def identify(self, image: np.ndarray, boxes) -> list[dict]:
        with self.lock:
            if not boxes or not self.views:
                return [{"scores": {}, "name": None, "sure": 0.0, "margin": 0.0} for _ in boxes]
            seen = self.features(image, boxes)
            out = []
            for f in seen:
                # a name's score: its BEST views' mean -- taught from a scan, its views go all round it,
                # and a look sees one side (the user's choice, 2026-10-12; on the taught cubes left out
                # one at a time: 24/24 either way, own 0.88 and the next name's 0.70 at the median)
                scores = {n: float(np.mean(np.sort(v @ f)[-BEST:])) for n, v in self.views.items() if len(v)}
                ranked = sorted(scores, key=scores.get, reverse=True)
                margin = scores[ranked[0]] - (scores[ranked[1]] if len(ranked) > 1 else 0.0)
                out.append({"scores": {n: round(s, 3) for n, s in scores.items()}, "name": ranked[0],
                            "sure": round(scores[ranked[0]], 3), "margin": round(margin, 3)})
            return out

    def embed(self, image: np.ndarray, boxes: list) -> np.ndarray:
        """Each box's YOLOE visual-prompt embedding, unit length (n x D)."""
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
            rest = self.embeddings[keep] if keep else np.zeros((0, len(e)), np.float32)   # (none yet: 0 x 0)
            self.names = [self.names[i] for i in keep] + [name]
            self.embeddings = np.vstack([rest, e[None]])
            self._save()
            self.views[name] = self._views_of(name)
            return list(self.names)

    def forget(self, name: str) -> list[str]:
        with self.lock:
            keep = [i for i, n in enumerate(self.names) if n != name]
            self.names, self.embeddings = [self.names[i] for i in keep], self.embeddings[keep]
            self.views.pop(name, None)
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
        boxes, classes, confs = (r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())
        return [{"name": self.names[int(c)], "conf": round(float(s), 3), "box": [round(float(v), 1) for v in b]}
                for b, c, s in zip(boxes, classes, confs, strict=True)]


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
                return self.reply({"models": [Path(MODEL).name, "dinov2_vits14"], "device": teacher.device,
                                   "names": teacher.names, "views": {n: len(v) for n, v in teacher.views.items()}})
            self.reply({"error": f"no such endpoint: GET {self.path}"}, 404)

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                began = time.perf_counter()
                if self.path == "/identify":
                    out = {"things": teacher.identify(decode(body["jpeg"]), body["boxes"])}
                elif self.path == "/teach":
                    out = {"names": teacher.teach(str(body["name"]), body["embedding"])}
                elif self.path == "/forget":
                    out = {"names": teacher.forget(str(body["name"]))}
                elif self.path == "/embed":
                    out = {"embeddings": teacher.embed(decode(body["jpeg"]), body["boxes"]).tolist()}
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
    print(f"teach service on {HOST}:{PORT}: {Path(MODEL).name} + DINOv2 ViT-S/14 on {teacher.device}, "
          f"taught: {({n: len(v) for n, v in teacher.views.items()})}", flush=True)
    ThreadingHTTPServer((HOST, PORT), make_handler(teacher)).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
