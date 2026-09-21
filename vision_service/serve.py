"""HTTP wrapper around one Detector. `python3 -m vision_service.serve`.

Standard library only, like roboarm.web.server: two endpoints do not need a
framework.

    GET  /health   is the model loaded, on which device, with which classes.
                   roboarm.detect.vision_available() polls this to decide whether
                   the neural rung is usable.
    POST /detect   body is a JPEG (Content-Type: image/jpeg).
                   returns {model, width, height, detections}, each detection
                   {label, confidence, box:[x,y,w,h], polygon?} in PIXELS.
    POST /raised   same body, same reply shape, from the depth model: the
                   outlines of what stands up on the table (vision_service.depth).
                   503 when no depth engine is built; the client then skips it.

The model is loaded once at startup, not per request, so a bad ROBOARM_VISION_MODEL
fails the container immediately rather than on the first pick attempt.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vision_service.depth import Depth
from vision_service.detector import Detector


class Handler(BaseHTTPRequestHandler):
    detector: Detector
    depth: Depth | None = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # one line per frame is noise
        pass

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json({**self.detector.health(),
                        "depth": self.depth.name if self.depth else None})
        else:
            self._json({"error": f"no such endpoint: GET {self.path}"}, 404)

    def do_POST(self) -> None:
        if self.path == "/detect":
            model = self.detector
        elif self.path == "/raised":
            model = self.depth
            if model is None:
                self._json({"error": "no depth engine is built -- see vision_service/depth.py"}, 503)
                return
        else:
            self._json({"error": f"no such endpoint: POST {self.path}"}, 404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if not body:
            self._json({"error": "empty body -- POST a JPEG frame"}, 400)
            return
        try:
            self._json(model.infer(body))
        except Exception as exc:  # noqa: BLE001 -- the client must hear about it
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main() -> None:
    detector = Detector.from_env()  # load + warmup now; a bad model is a boot failure
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
    depth = Depth.from_env() if detector.cuda else None   # the engine is CUDA-only
    handler = type("BoundHandler", (Handler,), {"detector": detector, "depth": depth})
    # Binds all interfaces INSIDE the container; compose publishes it to
    # 127.0.0.1 only, and on the compose network `core` reaches it as `vision`.
    port = int(os.environ.get("ROBOARM_VISION_PORT", "8760"))
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    server.daemon_threads = True
    print(f"roboarm vision: {detector.name} on port {port}, cuda={detector.cuda}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
