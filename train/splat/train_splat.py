"""A Gaussian splat of an object scan (PLAN.md M4 C), on hekate: m3pro/tools/m3_scan.py fuse --export
writes the scan's splat/ folder (its pictures and depths, the cameras as the fuse corrected them, the
starting points); this fits 3D Gaussians to the pictures with gsplat 1.5 -- whose CUDA kernels build the
first time it runs, with hekate's CUDA 12.8 toolkit and Visual C++ -- and writes NAME.ply (the 3DGS
layout the viewers read), NAME.splat (the console's 3D view reads it: copy it to the laptop's
m3pro/out/arena/rtab), a turntable of rendered frames, and how well it renders the views it did not see.

    splat\\splat.bat SPLAT_DIR [--steps 7000] [--depth-weight 0.5] [--name splat]
    splat\\splat.bat SPLAT_DIR --name splat --convert      # NAME.ply -> NAME.splat again, no training

Depth supervision: the splat's expected depth is held to the depth camera's (the M3's DCW2, aligned to
the colour) wherever that read anything, weighted --depth-weight (0: pictures only) -- the first can's
splat, pictures only, floated smears where the scan saw least. Every HOLD_EVERY-th view is held out and
only scored: PSNR, and the median depth error against the camera's: the honest numbers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def build_gsplat_for_msvc() -> None:
    """gsplat's own build of its kernels (gsplat.cuda._backend, on first use) passes GCC's
    -Wno-attributes to the C++ compiler, which MSVC refuses (D8021; 2026-10-12, gsplat 1.5.3): on
    Windows build the same sources here with MSVC's flags, as the module gsplat looks for first."""
    import torch.utils.cpp_extension as ext

    cuda = os.path.join(importlib.util.find_spec("gsplat").submodule_search_locations[0], "cuda")
    sources = [os.path.join(cuda, "csrc", f) for f in sorted(os.listdir(os.path.join(cuda, "csrc")))
               if f.endswith((".cu", ".cpp"))] + [os.path.join(cuda, "ext.cpp")]
    module = ext.load(name="gsplat_cuda", sources=sources, extra_cflags=["/O2"],
                      extra_cuda_cflags=["-O3", "-use_fast_math"],
                      extra_include_paths=[os.path.join(cuda, "include"),
                                           os.path.join(cuda, "csrc", "third_party", "glm")],
                      build_directory=ext._get_build_directory("gsplat_cuda", verbose=False), verbose=False)
    sys.modules["gsplat.csrc"] = module


if os.name == "nt":
    build_gsplat_for_msvc()

# gsplat after its kernels are built (above)
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

C0 = 0.28209479177387814   # the 0th spherical harmonic
HOLD_EVERY = 6
START_SIZE = 0.003         # m: the starting points' spacing (m3_scan's export, 3 mm)
DEPTH_NEAR, DEPTH_FAR = 0.15, 1.5   # m: the depth camera's readings believed (it starts at ~0.17-0.19)


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """m3_scan's PLY (binary: float x y z, uchar r g b) -> (n x 3 m, n x 3 in 0..1)."""
    raw = path.read_bytes()
    body = raw[raw.index(b"end_header\n") + len(b"end_header\n"):]
    rows = np.frombuffer(body, [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    return (np.stack([rows["x"], rows["y"], rows["z"]], 1).astype(np.float32),
            np.stack([rows["r"], rows["g"], rows["b"]], 1).astype(np.float32) / 255)


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SSIM of two 1 x 3 x H x W pictures (an 11-px Gaussian window)."""
    g = torch.exp(-((torch.arange(11, device=a.device) - 5.0) ** 2) / (2 * 1.5 ** 2))
    w = (g[:, None] * g[None, :] / g.sum() ** 2).expand(3, 1, 11, 11)
    blur = lambda x: F.conv2d(x, w, padding=5, groups=3)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma ** 2, blur(b * b) - mb ** 2, blur(a * b) - ma * mb
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean()


def look_at(eye: np.ndarray, at: np.ndarray) -> np.ndarray:
    """world -> camera (OpenCV: x right, y down, z ahead), the world's z up."""
    z = (at - eye) / np.linalg.norm(at - eye)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, :3] = np.stack([x, y, z])
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def save_ply(path: Path, p: dict) -> None:
    """The 3DGS layout: x y z, normals, f_dc, f_rest, opacity, scale, rot -- as the splat viewers read."""
    n = p["means"].shape[0]
    rest = p["shN"].detach().transpose(1, 2).reshape(n, -1).cpu().numpy()   # channel-major, as 3DGS writes
    cols = ([p["means"].detach().cpu().numpy(), np.zeros((n, 3), np.float32), p["sh0"].detach()[:, 0].cpu().numpy(),
             rest, p["opacities"].detach()[:, None].cpu().numpy(), p["scales"].detach().cpu().numpy(),
             F.normalize(p["quats"].detach(), dim=-1).cpu().numpy()])
    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(rest.shape[1])]
             + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    data = np.concatenate(cols, 1).astype("<f4")
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" \
        + "".join(f"property float {name}\n" for name in names) + "end_header\n"
    path.write_bytes(header.encode() + data.tobytes())


def read_ply(path: Path) -> dict:
    """save_ply's file back, as torch tensors on the CPU (for --convert)."""
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    names = [line.split()[-1] for line in raw[:end].decode().splitlines() if line.startswith("property")]
    data = np.frombuffer(raw[end:], "<f4").reshape(-1, len(names))
    col = {name: data[:, k] for k, name in enumerate(names)}
    stack = lambda keys: torch.tensor(np.stack([col[k] for k in keys], 1))
    return {"means": stack(["x", "y", "z"]), "scales": stack(["scale_0", "scale_1", "scale_2"]),
            "quats": stack(["rot_0", "rot_1", "rot_2", "rot_3"]), "opacities": torch.tensor(col["opacity"]),
            "sh0": stack(["f_dc_0", "f_dc_1", "f_dc_2"])[:, None, :]}


def inside(folder: Path, means: np.ndarray) -> np.ndarray:
    """Which Gaussians lie where the scan's starting points do: within as far round the thing as
    those reach (m3_scan's SPLAT_REACH), and between their lowest and highest (2 cm under, 3 over).
    Outside, they only paint the room behind (0.4 of the first can's splat, a fringe of smears)."""
    meta = json.loads((folder / "transforms.json").read_text())
    xyz, _rgb = read_points(folder / "points.ply")
    centre = np.asarray(meta["centre"][:2])
    reach = np.hypot(*(xyz[:, :2] - centre).T).max()
    return ((np.hypot(*(means[:, :2] - centre).T) <= reach)
            & (means[:, 2] >= xyz[:, 2].min() - 0.02) & (means[:, 2] <= xyz[:, 2].max() + 0.03))


def save_splat(path: Path, p: dict, keep: np.ndarray | None = None) -> int:
    """The compact .splat the console's viewer reads (antimatter15's layout, 32 bytes a Gaussian):
    position (3 float32, m), size (3 float32, m), colour and opacity (4 uint8), rotation (4 uint8:
    w x y z, each q * 128 + 128) -- the view-dependent colour (shN) left out. Only those `keep` says
    (all, without it), and of them not the barely visible (opacity under 2 %). Returns how many it kept."""
    alpha = torch.sigmoid(p["opacities"].detach()).cpu().numpy()
    keep = (alpha >= 0.02) & (True if keep is None else keep)
    rows = np.zeros(int(keep.sum()), [("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("q", "u1", 4)])
    rows["p"] = p["means"].detach().cpu().numpy()[keep]
    rows["s"] = torch.exp(p["scales"].detach()).cpu().numpy()[keep]
    rgb = np.clip(0.5 + C0 * p["sh0"].detach()[:, 0].cpu().numpy(), 0, 1)
    rows["c"] = np.round(np.concatenate([rgb, alpha[:, None]], 1)[keep] * 255).astype(np.uint8)
    q = F.normalize(p["quats"].detach(), dim=-1).cpu().numpy()[keep]
    rows["q"] = np.clip(np.round(q * 128 + 128), 0, 255).astype(np.uint8)
    path.write_bytes(rows.tobytes())
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--steps", type=int, default=7000)
    ap.add_argument("--max-points", type=int, default=150_000)
    ap.add_argument("--depth-weight", type=float, default=0.5, help="the depth loss's weight (m); 0: pictures only")
    ap.add_argument("--name", default="splat", help="the outputs' name: NAME.ply, NAME_turntable.jpg, ...")
    ap.add_argument("--convert", action="store_true", help="only write DIR/NAME.ply's .splat, no training")
    a = ap.parse_args()
    if a.convert:
        p = read_ply(a.dir / f"{a.name}.ply")
        print(f"{save_splat(a.dir / f'{a.name}.splat', p, inside(a.dir, p['means'].numpy()))} of "
              f"{len(p['means'])} Gaussians -> "
              f"{a.dir / (a.name + '.splat')}")
        return
    dev = "cuda"
    meta = json.loads((a.dir / "transforms.json").read_text())
    W, H = meta["width"], meta["height"]
    frames = meta["frames"]
    images = torch.stack([torch.from_numpy(cv2.cvtColor(cv2.imread(str(a.dir / f["file"])), cv2.COLOR_BGR2RGB))
                          for f in frames]).float().div(255).to(dev)
    Ks = torch.tensor([f["K"] for f in frames], dtype=torch.float32, device=dev)
    c2w = np.array([f["c2w"] for f in frames], float)
    viewmats = torch.linalg.inv(torch.tensor(c2w, dtype=torch.float32, device=dev))
    depths = None
    if all("depth" in f and (a.dir / f["depth"]).exists() for f in frames):
        depths = torch.stack([torch.from_numpy(cv2.imread(str(a.dir / f["depth"]), cv2.IMREAD_UNCHANGED).astype(np.float32))
                              for f in frames]).div(1000).to(dev)
    valid = None if depths is None else (depths > DEPTH_NEAR) & (depths < DEPTH_FAR)
    held = list(range(0, len(frames), HOLD_EVERY))
    trained = [i for i in range(len(frames)) if i not in held]

    xyz, rgb = read_points(a.dir / "points.ply")
    if len(xyz) > a.max_points:
        keep = np.random.default_rng(0).choice(len(xyz), a.max_points, replace=False)
        xyz, rgb = xyz[keep], rgb[keep]
    centre = np.r_[meta["centre"], meta["top"] / 2]
    scene_scale = float(np.median(np.linalg.norm(c2w[:, :3, 3] - centre, axis=1))) * 1.1
    n = len(xyz)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(xyz, device=dev)),
        "scales": torch.nn.Parameter(torch.full((n, 3), math.log(START_SIZE), device=dev)),
        "quats": torch.nn.Parameter(F.normalize(torch.randn(n, 4, device=dev), dim=-1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.1, device=dev))),
        "sh0": torch.nn.Parameter(((torch.tensor(rgb, device=dev) - 0.5) / C0)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(n, 15, 3, device=dev)),
    })
    rates = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
             "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": r, "name": k}], eps=1e-15) for k, r in rates.items()}
    decay = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / a.steps))
    strategy = DefaultStrategy(verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    def render(p, view, k, sh):
        """(colour H x W x 3, expected depth H x W m, gsplat's info) of the splat through one camera."""
        out, _alpha, info = rasterization(p["means"], F.normalize(p["quats"], dim=-1), torch.exp(p["scales"]),
                                          torch.sigmoid(p["opacities"]), torch.cat([p["sh0"], p["shN"]], 1),
                                          view[None], k[None], W, H, sh_degree=sh, packed=False,
                                          render_mode="RGB+ED")
        return out[0, ..., :3].clamp(0, 1), out[0, ..., 3], info

    rng, began = np.random.default_rng(1), time.time()
    weight = a.depth_weight if depths is not None else 0.0
    print(f"{len(trained)} views to fit, {len(held)} held out; {n} Gaussians to start; scene scale {scene_scale:.2f} m; "
          f"depth weight {weight}" + ("" if depths is not None else " (no depths in this export)"), flush=True)
    for step in range(a.steps):
        i = trained[rng.integers(len(trained))]
        pred, depth, info = render(params, viewmats[i], Ks[i], min(step // 1000, 3))
        strategy.step_pre_backward(params, optimizers, state, step, info)
        gt = images[i]
        loss = 0.8 * F.l1_loss(pred, gt) + 0.2 * (1 - ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        if weight > 0:
            loss = loss + weight * (depth - depths[i]).abs()[valid[i]].mean()
        loss.backward()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        decay.step()
        if step % 1000 == 0 or step == a.steps - 1:
            print(f"step {step}: loss {loss.item():.4f}, {params['means'].shape[0]} Gaussians, {time.time() - began:.0f} s",
                  flush=True)

    with torch.no_grad():
        psnr, depth_err = [], []
        for i in held:
            pred, depth, _ = render(params, viewmats[i], Ks[i], 3)
            psnr.append(float(-10 * torch.log10(F.mse_loss(pred, images[i]))))
            if depths is not None:
                depth_err.append(1000 * float((depth - depths[i]).abs()[valid[i]].median()))
            if i == held[0]:
                side = torch.cat([images[i], pred], 1).mul(255).byte().cpu().numpy()
                cv2.imwrite(str(a.dir / f"{a.name}_held_view.jpg"), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
        save_ply(a.dir / f"{a.name}.ply", params)
        kept = save_splat(a.dir / f"{a.name}.splat", params, inside(a.dir, params["means"].detach().cpu().numpy()))
        k = Ks[0]
        spin = a.dir / f"{a.name}_turntable"
        spin.mkdir(exist_ok=True)
        tiles = []
        for j in range(24):   # round it at the scan's own distance, 45 deg up
            ang = 2 * math.pi * j / 24
            eye = centre + scene_scale / 1.1 * np.array([math.cos(ang) * 0.71, math.sin(ang) * 0.71, 0.71])
            view = torch.tensor(look_at(eye, centre), dtype=torch.float32, device=dev)
            pic = render(params, view, k, 3)[0].mul(255).byte().cpu().numpy()
            cv2.imwrite(str(spin / f"{j:02d}.jpg"), cv2.cvtColor(pic, cv2.COLOR_RGB2BGR))
            tiles.append(cv2.resize(cv2.cvtColor(pic, cv2.COLOR_RGB2BGR), (320, 240)))
        cv2.imwrite(str(a.dir / f"{a.name}_turntable.jpg"),
                    np.vstack([np.hstack(tiles[r * 6:(r + 1) * 6]) for r in range(4)]))
    print(f"held-out views: PSNR {np.mean(psnr):.2f} dB (median {np.median(psnr):.2f})"
          + (f", depth off the camera's by {np.median(depth_err):.1f} mm (median)" if depth_err else "")
          + f"; {params['means'].shape[0]} Gaussians ({kept} in the .splat); {time.time() - began:.0f} s -> "
          f"{a.dir / (a.name + '.ply')}",
          flush=True)


if __name__ == "__main__":
    main()
