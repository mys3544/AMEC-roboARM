#!/usr/bin/env python3
"""Depth Anything 3 -> ONNX that TensorRT builds RIGHT (x3plus/PLAN.md X9).

    python tools/export_da3.py DA3-SMALL                  # 3 views + their poses, 280 x 504
    python tools/export_da3.py DA3-LARGE --views 3 --size 280 504
    python tools/export_da3.py DA3-BASE --free            # 3 views, no poses

Runs where torch and depth_anything_3 are installed (the robot's ~/roboarm/.venv-ml from
tools/ml_env.sh, or a laptop). Then, on the robot, build with the TensorRT that will run it:

    /usr/src/tensorrt/bin/trtexec --onnx=da3-small_enc_3x280x504.onnx --fp16 \\
        --saveEngine=da3-small_enc.fp16.engine --timingCacheFile=trt_timing.cache

What a plain torch.onnx.export of DA3 gets wrong, found 2026-10-06/07 on the X3's Orin (an engine
that disagreed with PyTorch by 6.5 % median, in FP32 as in FP16, on TensorRT 10.4 and 10.7):
  * ATTENTION SCALE. torch's scaled_dot_product_attention, exported without a scale, computes
    1/sqrt(head_dim) from the tensor's shape at run time, and TensorRT misread it: the first
    block's softmax already differed by 100 %. Here the attention is written out with the scale
    a constant. With this alone the engine matches PyTorch to 0.07 % (corr 0.999995).
  * The camera POSE CODE (quaternion by argmax / one-hot / boolean indexing) is computed outside
    the engine -- roboarm/da3_pose.py -- and the engine takes it as its "pose" input.
  * The SKY step (100k random samples, a quantile) is left out: the table fit needs none of it.
  * The REFERENCE VIEW is fixed to the middle one ("saddle_balanced" picks it by an argmax).
  * x.mT and torch.cartesian_prod (RoPE's grid) have no ONNX opset-17 export: mapped by hand.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
from torch.onnx import register_custom_op_symbolic, symbolic_helper

os.environ.setdefault("XFORMERS_DISABLED", "1")


def _mT(g, self):
    rank = symbolic_helper._get_tensor_rank(self)
    return g.op("Transpose", self, perm_i=list(range(rank - 2)) + [rank - 1, rank - 2])


def _cartesian_prod(a, b):
    ga, gb = torch.meshgrid(a, b, indexing="ij")
    return torch.stack([ga.reshape(-1), gb.reshape(-1)], dim=-1)


def _attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    s = scale if scale is not None else 1.0 / math.sqrt(int(q.shape[-1]))   # a constant in the graph
    attn = torch.matmul(q * s, k.transpose(-2, -1))
    if attn_mask is not None:
        attn = attn.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else attn + attn_mask
    return torch.matmul(attn.softmax(dim=-1), v)


class Exportable(torch.nn.Module):
    """DA3's network as the engine runs it: images (1, N, 3, H, W), ImageNet-normalised, and with
    poses a (1, N, 9) pose code -> depth and confidence (1, N, H, W)."""

    def __init__(self, net, posed: bool):
        super().__init__()
        net._process_mono_sky_estimation = lambda output: output
        if posed:
            enc = net.cam_enc
            enc.forward = lambda code, _ixt, _size: enc.trunk_norm(enc.trunk(enc.token_norm(enc.pose_branch(code))))
        self.net, self.posed = net, posed

    def forward(self, image, pose=None):
        out = self.net(image, pose if self.posed else None, None, ref_view_strategy="middle")
        return out["depth"], out["depth_conf"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="DA3-SMALL, DA3-BASE, DA3-LARGE (the Hugging Face name after depth-anything/)")
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--size", type=int, nargs=2, default=(280, 504), metavar=("H", "W"), help="multiples of 14")
    parser.add_argument("--free", action="store_true", help="no poses: DA3 estimates them itself")
    args = parser.parse_args()
    h, w = args.size
    if h % 14 or w % 14:
        parser.error("the size must be a multiple of the 14 px patch")

    from depth_anything_3.api import DepthAnything3
    register_custom_op_symbolic("aten::mT", _mT, 17)
    torch.cartesian_prod = _cartesian_prod

    net = DepthAnything3.from_pretrained(f"depth-anything/{args.model}").eval().model
    rng = np.random.default_rng(0)
    image = torch.tensor(rng.normal(0, 1, (1, args.views, 3, h, w)), dtype=torch.float32)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from roboarm import da3_pose
    w2c = np.stack([da3_pose.world_to_camera(np.eye(3), np.array([0.03 * (i - 1), 0.0, 0.0]))
                    for i in range(args.views)])
    K = np.array([[300.0, 0, w / 2], [0, 300.0, h / 2], [0, 0, 1]])[None].repeat(args.views, 0)
    pose = torch.tensor(da3_pose.pose_code(w2c, K, (h, w))[None])
    model = Exportable(net, posed=not args.free).eval()
    inputs = (image,) if args.free else (image, pose)
    with torch.no_grad():
        before = model(*inputs)[0]
        torch.nn.functional.scaled_dot_product_attention, original = _attention, torch.nn.functional.scaled_dot_product_attention
        after = model(*inputs)[0]
    print(f"the written-out attention changes the depth by at most {float((after - before).abs().max()):.1e}")
    name = f"{args.model.lower()}_{'free' if args.free else 'enc'}_{args.views}x{h}x{w}.onnx"
    torch.onnx.export(model, inputs, name, input_names=["image"] + ([] if args.free else ["pose"]),
                      output_names=["depth", "conf"], opset_version=17, dynamo=False)
    torch.nn.functional.scaled_dot_product_attention = original
    print(f"exported {name}, {os.path.getsize(name) / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
