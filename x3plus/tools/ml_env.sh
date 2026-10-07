#!/usr/bin/env bash
# The robot's model environment, ~/roboarm/.venv-ml (x3plus/PLAN.md X9): newer models next to
# JetPack's own GPU stack, without touching it. The vision service runs from it (robot/).
#
#   bash tools/ml_env.sh            # on the robot, from ~/roboarm
#
# Built on JetPack 6.2's system packages (--system-site-packages): torch 2.5 (NVIDIA's build,
# CUDA), TensorRT 10.7 -- the newest that runs on JetPack 6; newer needs JetPack 7 -- and ONNX
# Runtime 1.20 with the TensorRT and CUDA providers. None of these can come from PyPI here: its
# torch / tensorrt wheels are for servers (an install of ultralytics would pull torch 2.14 for
# CUDA 13 over the working one), so everything that depends on torch goes in with --no-deps and
# numpy stays at the system's 1.23.5, which torch and cv2 are built against.
set -euo pipefail
VENV=${VENV:-$HOME/roboarm/.venv-ml}
PY=$VENV/bin/python
export PATH=$HOME/.local/bin:$PATH

[ -d "$VENV" ] || uv venv "$VENV" --python /usr/bin/python3.10 --system-site-packages
# Exactly the pins (robot/ml-requirements.txt, frozen from a working venv), and nothing else: sync
# installs what is listed without resolving dependencies, so no PyPI torch can come in. The pins
# were first got by installing the pure-Python helpers under numpy==1.23.5 and the models' own
# packages (ultralytics, polygraphy, Depth Anything 3, MoGe) with --no-deps.
uv pip sync --python "$PY" "$(dirname "$0")/../robot/ml-requirements.txt"

# pycolmap has no aarch64 wheels; Depth Anything 3 imports it only for its COLMAP export.
mkdir -p "$VENV/lib/python3.10/site-packages/pycolmap"
cat > "$VENV/lib/python3.10/site-packages/pycolmap/__init__.py" <<'EOF'
"""Placeholder: pycolmap has no aarch64 build; Depth Anything 3 imports it only for its COLMAP
export, which the X3 never uses. Any real use fails loudly."""


def __getattr__(name):
    raise ImportError(f"pycolmap is a placeholder on this robot; pycolmap.{name} is not available")
EOF

"$PY" - <<'EOF'
import numpy, tensorrt, torch, onnxruntime, ultralytics
from depth_anything_3.api import DepthAnything3  # noqa: F401
print("numpy", numpy.__version__, "| torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "| tensorrt", tensorrt.__version__, "| onnxruntime", onnxruntime.get_available_providers(),
      "| ultralytics", ultralytics.__version__)
EOF
# Not here: MoGe imports torch.distributed parts JetPack's torch is built without (GradBucket);
# MoGe-3's refiner needs Triton, which has no JetPack build.
