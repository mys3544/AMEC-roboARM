# GPU detector service. Base image already ships CUDA 12.6 + cuDNN + PyTorch
# built for JetPack 6 / L4T r36.4 -- matching the host's L4T R36.4.7.
# libcuda.so.1 itself is injected at runtime by the NVIDIA container runtime (CDI),
# NOT during build -- so CUDA *availability* is checked at startup by serve.py,
# not here. This build only proves the imports resolve and torch is a CUDA build.
FROM dustynv/l4t-pytorch:r36.4.0

COPY --from=ghcr.io/astral-sh/uv:0.12.8 /uv /bin/uv

ENV UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Two-stage install (see the two requirements files):
#   1. --no-deps: just ultralytics + its two helpers, kept off uv's resolver so
#      it can never swap the base image's CUDA torch for a PyPI wheel.
#   2. resolved: everything ultralytics needs on top of the base, plus the HTTP
#      layer. None of it depends on torch, so uv resolving its sub-deps is safe.
COPY docker/vision-requirements.txt docker/vision-extra-requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps -r vision-requirements.txt && \
    uv pip install --system -r vision-extra-requirements.txt

# Fail the build, not the demo, if an import the service needs is unresolvable or
# if torch is no longer a CUDA build. (torch.cuda.is_available() would be False
# here regardless -- no GPU is attached during build -- so it is not asserted.)
RUN python3 -c "import torch, torchvision, ultralytics, yaml, cv2, fastapi, uvicorn; from PIL import Image; assert torch.version.cuda, 'torch lost its CUDA build'; print('vision build ok: torch', torch.__version__, 'cuda', torch.version.cuda, '| ultralytics', ultralytics.__version__)"

CMD ["python3", "-m", "vision_service.serve"]
