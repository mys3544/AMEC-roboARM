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
answers 503 and the client skips it. One engine per wrist camera, at that
camera's aspect: the Sonix's 640x480 is 4:3 (518x686), the C930e's 848x480 is
16:9 (518x910, 2026-09-24). A frame of another shape is stretched to fit.

    ROBOARM_DEPTH_ENGINE   path to the engine; default by ROBOARM_CAMERA, see ENGINES

`raised_regions()` is pure numpy + OpenCV so the core environment can test it.
"""

from __future__ import annotations

import base64
import os

import cv2
import numpy as np

ENGINES = {"sonix": "/app/models/da2s_518x686.fp16.engine",
           "c930e": "/app/models/da2s_518x910.fp16.engine"}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

EDGE_SIGMAS = 8.0      # a depth edge is this many MADs of gradient above the median
# A second, coarser pass at this threshold (Sobel units of the 0..1-normalised map)
# mends tops the first one cut up. On a clear table the MAD is tiny, and the
# threshold it sets (0.013 on 2026-09-24) sat under the slope the model draws
# ACROSS a top face (~0.03): a 30 x 60 block's top was edge all over but for a
# 57 px strip, dropped as too small from one look and ranged 11 mm wide from the
# next. The floor cannot simply replace the first threshold, though: the map is
# normalised to its whole range, so where the robot's own wheel is in view the
# cubes' rims fall under 0.05 and a floor-only pass lost every one of them.
EDGE_FLOOR = 0.05
MIN_AREA_PX = 1500     # in the map; a 20 mm cube top was ~5000 px at the Sonix's primary look
RING_PX = 31           # the surround a region is compared against
MIN_STEP = 5.0         # nearer than the surround by this many noise sigmas = it stands up
FULL_SCORE_STEP = 40.0  # a step this strong scores 1.0 (cubes measured 19..120, one 7)


def raised_regions(disp: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Outlines (N x 2 pixel polygons in `disp`'s frame) of regions enclosed by
    depth edges that are nearer than their surround, with the step contrast in
    noise sigmas. Regions touching the border are left out: a thing cut off by
    the frame is not whole, and the table itself always touches the border.

    Two passes: edges above the noise (EDGE_SIGMAS), and, when that threshold is
    below EDGE_FLOOR, edges above the floor. A region of the coarse pass stands
    in for the fine regions inside it only when there is at most ONE of them --
    it mends a top the fine pass cut to a sliver, or finds one it missed, but it
    never merges two objects the fine pass kept apart."""
    disp = np.asarray(disp, np.float32)
    disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-9)
    # The fp16 engine's map is quantised to ~1500 distinct values; the steps
    # between them are gradients everywhere, which lifted the edge threshold past
    # a real cube's rim on two of three frames (2026-09-21). A 3x3 blur removes
    # them and leaves a cube's rim, which is tens of times larger, intact.
    disp = cv2.GaussianBlur(disp, (3, 3), 0)
    gx = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    median = float(np.median(grad))
    mad = 1.4826 * float(np.median(np.abs(grad - median))) + 1e-9
    fine_threshold = median + EDGE_SIGMAS * mad
    kept = _regions(disp, grad, fine_threshold)
    if fine_threshold < EDGE_FLOOR:
        fine = list(kept)
        for mask, outline, step in _regions(disp, grad, EDGE_FLOOR):
            inside = [f for f in fine if np.count_nonzero(f[0] & mask) > 0.5 * np.count_nonzero(f[0])]
            if len(inside) > 1:
                continue
            kept = [f for f in kept if all(f is not g for g in inside)] + [(mask, outline, step)]
    return sorted(((outline, step) for _mask, outline, step in kept), key=lambda r: -r[1])


def _regions(disp: np.ndarray, grad: np.ndarray,
             threshold: float) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """(mask, outline, step) of every region enclosed by gradients above
    `threshold` that is big enough, clear of the border and stands up."""
    height, width = disp.shape
    edges = cv2.dilate((grad > threshold).astype(np.uint8), np.ones((5, 5), np.uint8))
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
        found.append((mask, outline, step))
    return found


def preprocess(bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Frame -> the engine's 1x3xHxW float32 input (RGB, ImageNet mean/std);
    `size` is (W, H)."""
    rgb = cv2.cvtColor(cv2.resize(bgr, size, interpolation=cv2.INTER_CUBIC),
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
        # The engine's own input size: each is built at one camera's aspect.
        _n, _c, self.height, self.width = (int(v) for v in self._engine.get_tensor_shape("image"))
        self._ctx.set_input_shape("image", (1, 3, self.height, self.width))
        self._out = torch.empty(tuple(self._ctx.get_tensor_shape("depth")), device="cuda",
                                dtype=torch.float32)
        self._in = torch.empty((1, 3, self.height, self.width), device="cuda", dtype=torch.float32)
        self._stream = torch.cuda.Stream()
        self._ctx.set_tensor_address("image", self._in.data_ptr())
        self._ctx.set_tensor_address("depth", self._out.data_ptr())
        self._torch = torch
        self.depth_map(np.full((self.height, self.width, 3), 127, np.uint8))  # warm up

    @classmethod
    def from_env(cls) -> Depth | None:
        camera = os.environ.get("ROBOARM_CAMERA", "c930e")
        path = os.environ.get("ROBOARM_DEPTH_ENGINE", ENGINES.get(camera, ENGINES["c930e"]))
        if not os.path.exists(path):
            print(f"vision: no depth engine at {path}; the depth rung is off", flush=True)
            return None
        return cls(path)

    def depth_map(self, bgr: np.ndarray) -> np.ndarray:
        """Relative inverse depth at the engine's size, float32 (larger = nearer)."""
        torch = self._torch
        with torch.cuda.stream(self._stream):
            self._in.copy_(torch.from_numpy(preprocess(bgr, (self.width, self.height))))
            self._ctx.execute_async_v3(self._stream.cuda_stream)
            out = self._out.clone()
        self._stream.synchronize()
        return out.cpu().numpy().reshape(self.height, self.width)

    def infer(self, jpeg: bytes, want_map: bool = False) -> dict:
        """The raised regions; with `want_map` also "map", the relative depth as
        a base64 PNG at half size, 8-bit, nearer = brighter -- for a picture on
        the panel, never for measuring."""
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("the body is not a decodable JPEG")
        height, width = frame.shape[:2]
        scale = np.array([width / self.width, height / self.height])
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
            small = cv2.resize(disp, (self.width // 2, self.height // 2), interpolation=cv2.INTER_AREA)
            small = (255 * (small - small.min()) / (small.max() - small.min() + 1e-9)).astype(np.uint8)
            ok, png = cv2.imencode(".png", small)
            if ok:
                reply["map"] = base64.b64encode(png.tobytes()).decode("ascii")
        return reply
