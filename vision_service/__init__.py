"""The GPU detector service: YOLO26 / YOLOE-26 behind a tiny HTTP endpoint.

Runs in its own container (docker/vision.Dockerfile) so the CUDA + torch stack
never touches the demo-safe `core` image. Core posts a JPEG to /detect and gets
pixel-space boxes back; core owns the homography, so this service knows nothing
about the table or the arm.

`detector.py` is import-safe without torch/ultralytics/fastapi installed -- only
the parts that actually run a model import them, lazily -- so its pure helpers can
be unit-tested from the core environment.
"""
