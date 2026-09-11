"""Load one detection model and run it. No HTTP here, no table geometry.

Kept free of top-level torch / ultralytics / PIL imports on purpose: everything
heavy is imported inside the method that needs it, so `extract_detections()` --
the part most likely to break on an Ultralytics version bump -- can be tested from
the core environment with a fake results object.

Configuration is entirely from the environment, so the container needs no config
file mounted:

    ROBOARM_VISION_MODEL     path to weights, or a name Ultralytics can fetch.
                             Default /app/models/yolo26n.pt. A name containing
                             "yoloe" is loaded as an open-vocabulary model.
    ROBOARM_VISION_IMGSZ     inference size, default 640.
    ROBOARM_VISION_CONF      default confidence floor, default 0.25.
    ROBOARM_VISION_CLASSES   comma-separated default prompt for a YOLOE model,
                             used when a request carries no X-Vision-Prompt.
"""

from __future__ import annotations

import io
import os

DEFAULT_MODEL = "/app/models/yolo26n.pt"


class PromptNotSupported(RuntimeError):
    """A text prompt was given to a fixed-class model that cannot honour it."""


def extract_detections(boxes, names, masks=None) -> list[dict]:
    """Ultralytics `Results.boxes` -> our wire format: label, confidence, pixel box.

    `box` is [x, y, w, h] with (x, y) the top-left corner, matching what
    roboarm.detect.objects() expects to feed straight into the homography.
    With `masks` (a segmentation model's `Results.masks`) each detection also
    carries `polygon`, its outline in pixels -- what the client's block ranging
    wants, since a box only circumscribes the object.

    Written against the per-box tensors Ultralytics yields (`.cls`, `.conf` shape
    (1,); `.xyxy` shape (1, 4)) but tolerant of plain scalars/lists so a test can
    fake it without torch.
    """
    polygons = list(getattr(masks, "xy", None) or []) if masks is not None else []
    out: list[dict] = []
    for index, box in enumerate(boxes):
        cls = int(_first(box.cls))
        score = float(_first(box.conf))
        x1, y1, x2, y2 = (float(v) for v in _row(box.xyxy))
        label = names[cls] if not isinstance(names, dict) else names.get(cls, str(cls))
        item = {
            "label": str(label),
            "confidence": round(score, 4),
            "box": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
        }
        if index < len(polygons) and len(polygons[index]) >= 3:
            item["polygon"] = [[round(float(px), 1), round(float(py), 1)]
                               for px, py in polygons[index]]
        out.append(item)
    return out


def _first(value):
    """`value[0]` if it is a sequence/tensor, else `value`."""
    try:
        return value[0]
    except (TypeError, IndexError, KeyError):
        return value


def _row(value):
    """The first row of an (N, 4) tensor/list, or `value` if it is already 4 long."""
    row = _first(value)
    # A bare (4,) tensor indexes to a scalar on [0]; detect that and use it whole.
    try:
        if len(row) == 4:
            return row
    except TypeError:
        pass
    return value


class Detector:
    """A loaded model plus the small amount of state prompting needs."""

    def __init__(self, model, name: str, open_vocab: bool, imgsz: int, conf: float):
        self._model = model
        self.name = name
        self.open_vocab = open_vocab
        self.imgsz = imgsz
        self.conf = conf
        self._applied: tuple[str, ...] | None = None
        self.cuda = _cuda_available()
        self.classes = self._model_classes()

    # ------------------------------------------------------------- loading --
    @classmethod
    def from_env(cls) -> Detector:
        path = _resolve_model(os.environ.get("ROBOARM_VISION_MODEL", DEFAULT_MODEL))
        imgsz = int(os.environ.get("ROBOARM_VISION_IMGSZ", "640"))
        conf = float(os.environ.get("ROBOARM_VISION_CONF", "0.25"))
        open_vocab = "yoloe" in os.path.basename(path).lower()

        model = _load_model(path, open_vocab)
        detector = cls(model, os.path.basename(path), open_vocab, imgsz, conf)

        default_prompt = os.environ.get("ROBOARM_VISION_CLASSES", "").strip()
        if open_vocab and default_prompt:
            detector._set_classes(_split(default_prompt))
        detector._warmup()
        return detector

    def _model_classes(self) -> list[str]:
        names = getattr(self._model, "names", None)
        if isinstance(names, dict):
            return [str(v) for v in names.values()]
        if names:
            return [str(v) for v in names]
        return []

    # ----------------------------------------------------------- inference --
    def infer(self, jpeg: bytes, prompt: str | None = None,
              conf: float | None = None) -> dict:
        # Reject a bad request before doing any decoding or inference.
        if prompt and not self.open_vocab:
            raise PromptNotSupported(
                f"{self.name} has a fixed class list; a text prompt needs a "
                f"YOLOE model (set ROBOARM_VISION_MODEL to one)."
            )

        from PIL import Image

        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        if prompt:
            self._set_classes(_split(prompt))

        results = self._model.predict(
            image, imgsz=self.imgsz, conf=self.conf if conf is None else conf,
            verbose=False,
        )
        first = results[0]
        detections = extract_detections(first.boxes, first.names,
                                        getattr(first, "masks", None))
        return {
            "model": self.name,
            "open_vocab": self.open_vocab,
            "prompt": prompt,
            "width": image.width,
            "height": image.height,
            "detections": detections,
        }

    def health(self) -> dict:
        return {
            "status": "ok",
            "model": self.name,
            "open_vocab": self.open_vocab,
            "classes": self.classes,
            "cuda": self.cuda,
            "imgsz": self.imgsz,
            "conf": self.conf,
        }

    # ------------------------------------------------------------ prompting --
    def _set_classes(self, classes: list[str]) -> None:
        key = tuple(classes)
        if key == self._applied:
            return
        # get_text_pe() runs the (slow) text encoder; skip it when unchanged.
        self._model.set_classes(classes, self._model.get_text_pe(classes))
        self._applied = key
        self.classes = list(classes)

    def _warmup(self) -> None:
        from PIL import Image

        blank = Image.new("RGB", (self.imgsz, self.imgsz), (127, 127, 127))
        try:
            self._model.predict(blank, imgsz=self.imgsz, conf=0.99, verbose=False)
        except (RuntimeError, OSError, ValueError) as exc:  # best-effort: a cold
            print(f"vision: warmup inference failed ({exc}); first /detect will be slow")


def _split(prompt: str) -> list[str]:
    return [part.strip() for part in prompt.split(",") if part.strip()]


def _resolve_model(path: str) -> str:
    """Prefer a sibling .engine when the configured model is a .pt.

    tools/export_engine.py drops `<name>.engine` next to `<name>.pt`; once it is
    there it is ~1.5x faster and should be what serves. The .pt stays the
    configured default because an engine is welded to the exact TensorRT build and
    GPU it was made on -- a stale one is useless on a different machine, whereas
    the .pt always loads. Set ROBOARM_VISION_NO_AUTO_ENGINE=1 to force the .pt.
    """
    if os.environ.get("ROBOARM_VISION_NO_AUTO_ENGINE") == "1":
        return path
    base, ext = os.path.splitext(path)
    if ext == ".pt":
        engine = base + ".engine"
        if os.path.exists(engine):
            print(f"vision: using {os.path.basename(engine)} "
                  f"(sibling of the configured {os.path.basename(path)})")
            return engine
    return path


def _load_model(path: str, open_vocab: bool):
    if open_vocab:
        try:
            from ultralytics import YOLOE as Loader
        except ImportError:
            from ultralytics import YOLO as Loader
    else:
        from ultralytics import YOLO as Loader
    try:
        return Loader(path)
    except Exception as exc:  # Ultralytics raises a variety of errors here
        raise RuntimeError(
            f"could not load the detection model '{path}': {exc}. Put the weights "
            f"in ./models (mounted at /app/models), or set ROBOARM_VISION_MODEL to "
            f"a name Ultralytics can download while the robot has WiFi."
        ) from exc


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError, AttributeError):
        # torch absent or broken -> report no CUDA rather than crash /health
        return False
