"""HTTP wrapper around one Detector. `python3 -m vision_service.serve`.

Two endpoints, both deliberately tiny:

    GET  /health   is the model loaded, on which device, with which classes.
                   roboarm.detect.vision_available() polls this to decide whether
                   the neural rung is usable.
    POST /detect   body is a JPEG (Content-Type: image/jpeg). Optional headers:
                     X-Vision-Prompt  comma-separated things to look for (YOLOE)
                     X-Vision-Conf    confidence floor for this one request
                   returns {model, open_vocab, prompt, width, height, detections},
                   each detection {label, confidence, box:[x,y,w,h]} in PIXELS.

The model is loaded once at startup, not per request, so a bad ROBOARM_VISION_MODEL
fails the container immediately rather than on the first pick attempt.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from vision_service.detector import Detector, PromptNotSupported

_detector: Detector | None = None


def _get_detector() -> Detector:
    global _detector
    if _detector is None:
        _detector = Detector.from_env()
    return _detector


@asynccontextmanager
async def lifespan(_app: FastAPI):
    detector = _get_detector()  # load + warmup now; a bad model is a boot failure
    # CUDA is injected by CDI at `docker run`, so this is the first point it can
    # be checked. Fail here rather than serve slow CPU inference that surprises
    # someone mid-demo. ROBOARM_VISION_REQUIRE_CUDA=0 opts into CPU on purpose.
    if not detector.cuda and os.environ.get("ROBOARM_VISION_REQUIRE_CUDA", "1") != "0":
        raise RuntimeError(
            "the vision container has no CUDA -- the GPU was not passed through. "
            "Check `devices: [nvidia.com/gpu=all]` in compose.yaml and that the "
            "host has a CDI spec (`nvidia-ctk cdi generate`). Set "
            "ROBOARM_VISION_REQUIRE_CUDA=0 to run on CPU anyway."
        )
    yield


app = FastAPI(title="roboarm vision", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return _get_detector().health()


@app.post("/detect")
async def detect(request: Request) -> dict:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty body -- POST a JPEG frame")

    prompt = request.headers.get("x-vision-prompt") or None
    raw_conf = request.headers.get("x-vision-conf")
    try:
        conf = float(raw_conf) if raw_conf else None
    except ValueError:
        raise HTTPException(status_code=400, detail=f"bad X-Vision-Conf: {raw_conf!r}")

    try:
        return _get_detector().infer(body, prompt=prompt, conf=conf)
    except PromptNotSupported as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    # Binds all interfaces INSIDE the container; compose publishes it to
    # 127.0.0.1 only, and on the compose network `core` reaches it as `vision`.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("ROBOARM_VISION_PORT", "8760")),
        log_level=os.environ.get("ROBOARM_VISION_LOG", "info"),
    )


if __name__ == "__main__":
    main()
