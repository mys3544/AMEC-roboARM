#!/usr/bin/env bash
# The robot's model environment, ~/roboarm/.venv-ml (x3plus/PLAN.md X9): newer models next to
# JetPack's own GPU stack, without touching it or the Docker services.
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

uv venv "$VENV" --python /usr/bin/python3.10 --system-site-packages
echo 'numpy==1.23.5' > /tmp/ml-constraints.txt

# Pure-Python helpers the models need (none of them depends on torch).
uv pip install --python "$PY" -c /tmp/ml-constraints.txt \
    polars einops huggingface_hub hf_xet safetensors addict omegaconf plyfile imageio trimesh evo moviepy

# The models' own packages, without their torch / opencv requirements.
uv pip install --python "$PY" --no-deps \
    ultralytics==8.4.174 polygraphy \
    'git+https://github.com/ByteDance-Seed/Depth-Anything-3' \
    'git+https://github.com/microsoft/MoGe.git' 'git+https://github.com/EasternJournalist/utils3d-moge.git'

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
