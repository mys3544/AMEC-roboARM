"""The depth rung: what STANDS UP on the table, from a monocular depth model.

Depth Anything V2 Small gives RELATIVE inverse depth -- up to a scale and a
shift, and warping slowly across a flat table (tested offline 2026-09-21: a
global metric fit on the table is ill-conditioned and gave 70..120 mm heights
for 20..40 mm cubes). What it does reliably is draw a sharp step at the rim of
anything raised, with no shadow: a white cube on white paper came out as a
crisp plateau where the colour rung read 15..29 mm on the same day. So this
rung finds EDGES, not levels: regions enclosed by depth edges that are nearer
than their surround are raised things; the client maps their outline through
the homography and cube-ranges it exactly like a neural mask.

Served from a TensorRT engine built with trtexec from the ONNX export
(models/da2s_518x686.onnx -> models/da2s_518x686.fp16.engine, 26 ms on the
Orin, output matches the CPU model to 0.1 %). No engine, no rung: /raised
answers 503 and the client skips it.

    ROBOARM_DEPTH_ENGINE   path to the engine, default /app/models/da2s_518x686.fp16.engine

`raised_regions()` is pure numpy + OpenCV so the core environment can test it.
"""

from __future__ import annotations

import base64
import os

import cv2
import numpy as np

DEFAULT_ENGINE = "/app/models/da2s_518x686.fp16.engine"
INPUT_W, INPUT_H = 686, 518          # the wrist camera's 4:3, in multiples of 14
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

EDGE_SIGMAS = 8.0      # a depth edge is this many MADs of gradient above the median
MIN_AREA_PX = 1500     # in the 686x518 map; a 20 mm cube top is ~5000 px at the primary look
RING_PX = 31           # the surround a region is compared against
MIN_STEP = 5.0         # nearer than the surround by this many noise sigmas = it stands up
FULL_SCORE_STEP = 40.0  # a step this strong scores 1.0 (cubes measured 19..120, one 7)


def raised_regions(disp: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Outlines (N x 2 pixel polygons in `disp`'s frame) of regions enclosed by
    depth edges that are nearer than their surround, with the step contrast in
    noise sigmas. Regions touching the border are left out: a thing cut off by
    the frame is not whole, and the table itself always touches the border."""
    disp = np.asarray(disp, np.float32)
    disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-9)
    # The fp16 engine's map is quantised to ~1500 distinct values; the steps
    # between them are gradients everywhere, which lifted the edge threshold past
    # a real cube's rim on two of three frames (2026-09-21). A 3x3 blur removes
    # them and leaves a cube's rim, which is tens of times larger, intact.
    disp = cv2.GaussianBlur(disp, (3, 3), 0)
    height, width = disp.shape
    gx = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    median = float(np.median(grad))
    mad = 1.4826 * float(np.median(np.abs(grad - median))) + 1e-9
    edges = cv2.dilate((grad > median + EDGE_SIGMAS * mad).astype(np.uint8), np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((1 - edges).astype(np.uint8))
    found = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < MIN_AREA_PX or x == 0 or y == 0 or x + w >= width or y + h >= height:
            continue
        mask = labels == i
        ring = cv2.dilate(mask.astype(np.uint8), np.ones((RING_PX, RING_PX), np.uint8)).astype(bool)
        ring &= ~mask & ~edges.astype(bool)
        if ring.sum() < 100:
            continue
        # The surround is a sloping, slowly warping table: fit it with a plane,
        # measure the region against the plane at its own centre, and take the
        # plane's residual as the noise. Dividing by the raw spread instead let
        # the slope itself hide a 0.3 step (step 2.7 on a synthetic ramp).
        ys, xs = np.nonzero(ring)
        design = np.stack([xs, ys, np.ones_like(xs)], axis=1).astype(np.float64)
        coeff = np.linalg.lstsq(design, disp[ring].astype(np.float64), rcond=None)[0]
        residual = disp[ring] - design @ coeff
        noise = 1.4826 * float(np.median(np.abs(residual))) + 1e-6
        cy, cx = (float(v) for v in np.mean(np.nonzero(mask), axis=1))
        step = (float(np.median(disp[mask])) - float(np.dot(coeff, (cx, cy, 1.0)))) / noise
        if step < MIN_STEP:
            continue
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
        found.append((outline, step))
    return sorted(found, key=lambda r: -r[1])


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """Frame -> the engine's 1x3x518x686 float32 input (RGB, ImageNet mean/std)."""
    rgb = cv2.cvtColor(cv2.resize(bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_CUBIC),
                       cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


class Depth:
    """The engine, loaded once. `infer(jpeg)` answers in the /detect wire format."""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import torch

        self.name = os.path.basename(engine_path)
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._ctx = self._engine.create_execution_context()
        self._ctx.set_input_shape("image", (1, 3, INPUT_H, INPUT_W))
        self._out = torch.empty(tuple(self._ctx.get_tensor_shape("depth")), device="cuda",
                                dtype=torch.float32)
        self._in = torch.empty((1, 3, INPUT_H, INPUT_W), device="cuda", dtype=torch.float32)
        self._stream = torch.cuda.Stream()
        self._ctx.set_tensor_address("image", self._in.data_ptr())
        self._ctx.set_tensor_address("depth", self._out.data_ptr())
        self._torch = torch
        self.depth_map(np.full((INPUT_H, INPUT_W, 3), 127, np.uint8))  # warm up

    @classmethod
    def from_env(cls) -> Depth | None:
        path = os.environ.get("ROBOARM_DEPTH_ENGINE", DEFAULT_ENGINE)
        if not os.path.exists(path):
            print(f"vision: no depth engine at {path}; the depth rung is off", flush=True)
            return None
        return cls(path)

    def depth_map(self, bgr: np.ndarray) -> np.ndarray:
        """Relative inverse depth, INPUT_H x INPUT_W float32 (larger = nearer)."""
        torch = self._torch
        with torch.cuda.stream(self._stream):
            self._in.copy_(torch.from_numpy(preprocess(bgr)))
            self._ctx.execute_async_v3(self._stream.cuda_stream)
            out = self._out.clone()
        self._stream.synchronize()
        return out.cpu().numpy().reshape(INPUT_H, INPUT_W)

    def infer(self, jpeg: bytes, want_map: bool = False) -> dict:
        """The raised regions; with `want_map` also "map", the relative depth as
        a base64 PNG at half size, 8-bit, nearer = brighter -- for a picture on
        the panel, never for measuring."""
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("the body is not a decodable JPEG")
        height, width = frame.shape[:2]
        scale = np.array([width / INPUT_W, height / INPUT_H])
        disp = self.depth_map(frame)
        detections = []
        for outline, step in raised_regions(disp):
            pixels = outline * scale
            x, y, w, h = cv2.boundingRect(pixels.astype(np.float32))
            detections.append({
                "label": "raised",
                "confidence": round(min(1.0, step / FULL_SCORE_STEP), 4),
                "box": [float(x), float(y), float(w), float(h)],
                "polygon": [[round(float(px), 1), round(float(py), 1)] for px, py in pixels],
            })
        reply = {"model": self.name, "width": width, "height": height, "detections": detections}
        if want_map:
            small = cv2.resize(disp, (INPUT_W // 2, INPUT_H // 2), interpolation=cv2.INTER_AREA)
            small = (255 * (small - small.min()) / (small.max() - small.min() + 1e-9)).astype(np.uint8)
            ok, png = cv2.imencode(".png", small)
            if ok:
                reply["map"] = base64.b64encode(png.tobytes()).decode("ascii")
        return reply
