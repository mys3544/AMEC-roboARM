"""Metric depth from one camera: Depth Anything's map anchored to the floor (x3plus/PLAN.md X2.3).

Depth Anything V2 gives inverse depth up to an unknown scale and offset: d = a / Z + b. Where
the camera sees the floor, Z is known exactly -- the arm's pose puts the camera in the base frame
(roboarm/lens.py), and the floor is the plane the robot stands on -- so a and b can be fitted in
every frame, and then Z follows everywhere else, the tops of things included.

What makes that fit possible, and what made it fail in September 2026. On a plane, 1/Z is
AFFINE in the normalised picture coordinates (x, y): 1/Z = alpha x + beta y + gamma, with
(alpha, beta) the plane's tilt to the picture. So on the floor, d = a alpha x + a beta y +
(a gamma + b): the scale a shows ONLY in the slopes. Looking straight down, the floor is square
to the picture, alpha = beta = 0, and a cannot be found at all -- the global fit then gave
70-120 mm for 20-40 mm cubes. An oblique look tilts the floor (the M3 looks at it from 12 to
45 cm ahead) and a is pinned. For the same reason the correction for the map's low "hills" may
not have linear terms -- they would absorb the slopes and a with them -- so it is quadratic only;
and since the M3's duel (2026-10-15) it is off unless asked for: it made every look worse there.

Everything here is plain numpy on the engine's grid; vision_service/depth.py makes the map.
"""

from __future__ import annotations

import cv2
import numpy as np

WARP_TERMS = 3          # x^2, xy, y^2 -- see above for why no x, y
HUBER_K = 1.5           # in units of the residual's robust sigma
FIT_ROUNDS = 6
MIN_FLOOR_PX = 500
NEAR_M, FAR_M = 0.05, 2.0   # floor distances taken into the fit
SCALE_SD_GOOD = 0.03    # a pinned to 3 %: a 30 mm step at 25 cm is then good to ~1 mm
REFITS = 3              # anchor(): fits again without what the last fit read as raised ...
RAISED_M = 0.010        # ... this far over the floor (5 mm takes the network's floor bumps out too,
                        #     20 mm leaves more of the big things in: both worse in the duel)
RAISED_SKIRT = 0.023    # ... and this far round it, a share of the map's width (15 px of 640) ...
RAISED_MAX = 0.5        # ... while that leaves at least half the floor the pose predicts


def rays(K, dist, frame_size: tuple[int, int], grid_size: tuple[int, int]) -> np.ndarray:
    """Normalised camera rays (x, y, 1) for each cell of a grid_size (W, H) map that covers a
    frame of frame_size (W, H) taken through K with `dist` left in it. Shape (H, W, 3)."""
    gw, gh = grid_size
    fw, fh = frame_size
    us = (np.arange(gw) + 0.5) * fw / gw - 0.5
    vs = (np.arange(gh) + 0.5) * fh / gh - 0.5
    grid = np.stack(np.meshgrid(us, vs), axis=-1).reshape(-1, 1, 2).astype(np.float64)
    norm = cv2.undistortPoints(grid, np.asarray(K, np.float64), np.asarray(dist, np.float64))
    xy = norm.reshape(gh, gw, 2)
    return np.concatenate([xy, np.ones((gh, gw, 1))], axis=-1)


def floor_inverse_depth(ray_cam: np.ndarray, R: np.ndarray, c: np.ndarray, floor_z: float) -> np.ndarray:
    """1/Z (camera depth, along the optical axis) of the floor plane z = floor_z for every ray;
    0 where a ray does not come down to it."""
    down = ray_cam @ R.T            # rays in the base frame (z component scales with Z)
    drop = floor_z - c[2]           # negative: the floor is below the camera
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = drop / down[..., 2]
    return np.where((down[..., 2] < 0) & np.isfinite(depth) & (depth > 0), 1.0 / np.maximum(depth, 1e-9), 0.0)


def _design(inv_floor: np.ndarray, ray_cam: np.ndarray, warp: bool) -> np.ndarray:
    x, y = ray_cam[..., 0], ray_cam[..., 1]
    columns = [inv_floor, np.ones_like(x)]
    if warp:
        columns += [x * x, x * y, y * y]
    return np.stack(columns, axis=-1)


def fit_floor(disp: np.ndarray, inv_floor: np.ndarray, ray_cam: np.ndarray,
              exclude: np.ndarray | None = None, warp: bool = True) -> dict:
    """d = a / Z_floor + b (+ the quadratic warp), robustly, on the floor pixels. Returns the
    parameters, how sure the scale is (scale_sd: a's relative standard deviation), the residual
    sigma, and the mask of pixels the fit kept as floor."""
    floor = (inv_floor > 1 / FAR_M) & (inv_floor < 1 / NEAR_M) & np.isfinite(disp)
    if exclude is not None:
        floor &= ~exclude
    if floor.sum() < MIN_FLOOR_PX:
        raise ValueError(f"only {int(floor.sum())} floor pixels in view: no fit")
    A = _design(inv_floor, ray_cam, warp)[floor]
    d = disp[floor]
    weights = np.ones_like(d)
    for _ in range(FIT_ROUNDS):
        w = np.sqrt(weights)[:, None]
        params, *_ = np.linalg.lstsq(A * w, d * w[:, 0], rcond=None)
        residual = d - A @ params
        sigma = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-12
        scaled = np.abs(residual) / (HUBER_K * sigma)
        weights = np.where(scaled <= 1, 1.0, 1.0 / scaled)
        weights[np.abs(residual) > 4 * sigma] = 0.0     # things standing on the floor
    kept = weights > 0
    # How well a is known: the noise over what the 1/Z_floor column holds that the other columns
    # cannot explain (Frisch-Waugh). Zero of it -- a floor square to the picture -- is an unknown
    # scale, which a pseudo-inverse would hide behind a finite number.
    rest = A[kept, 1:]
    own = A[kept, 0] - rest @ np.linalg.lstsq(rest, A[kept, 0], rcond=None)[0]
    spread = float(own @ own)
    a = float(params[0])
    scale_sd = float(sigma / np.sqrt(spread) / abs(a)) if spread > 1e-12 and a != 0 else float("inf")
    mask = np.zeros_like(floor)
    mask.flat[np.flatnonzero(floor)[kept]] = True
    return {"params": params, "scale": a, "scale_sd": scale_sd, "sigma": float(sigma), "floor": mask,
            "warp": warp, "pinned": bool(scale_sd <= SCALE_SD_GOOD and a > 0)}


def metric_depth(disp: np.ndarray, fit: dict, ray_cam: np.ndarray) -> np.ndarray:
    """Z (m, along the optical axis) everywhere from the fitted floor; NaN where the anchored
    map says nothing sensible (behind the camera or beyond reach)."""
    params = fit["params"]
    rest = disp - params[1]
    if fit["warp"]:
        x, y = ray_cam[..., 0], ray_cam[..., 1]
        rest = rest - (params[2] * x * x + params[3] * x * y + params[4] * y * y)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = params[0] / rest
    return np.where((z > 0.02) & (z < 5.0), z, np.nan)


def heights(z: np.ndarray, ray_cam: np.ndarray, R: np.ndarray, c: np.ndarray, floor_z: float) -> np.ndarray:
    """Height above the floor (m) of every pixel's point, from its depth Z."""
    points = c + (ray_cam * z[..., None]) @ R.T
    return points[..., 2] - floor_z


def anchor(disp: np.ndarray, frame_size: tuple[int, int], K, dist, R: np.ndarray, c: np.ndarray,
           floor_z: float, exclude: np.ndarray | None = None, warp: bool = False,
           refits: int = REFITS) -> dict:
    """The whole step: the fit, metric depth and heights on disp's grid, and the rays used.

    The fit is made again `refits` times without what the last one read as raised (over RAISED_M)
    or could not read, and a skirt round it -- unless that would leave out more than RAISED_MAX of
    the floor the pose predicts, when the last fit stands if it is the right way up (a cluttered
    desk ran away: the fit chased the last strip of floor). A third of a close look's predicted
    floor is no floor (the M3's duel, 2026-10-15, m3pro/tools/duel_floorfit.py): the drop-off
    box, tape rolls, the gripper's own finger a few cm from the lens -- more than the Huber fit
    sheds on its own, its sigma grows with them; the gripper made a 4-9x too big, cubes read 0 mm.
    No warp by default: alone it cost on every look and both networks there (DA V2 Large median
    error close 9.6 -> 5.7 mm without it, wide 4.8 -> 3.7)."""
    ray_cam = rays(K, dist, frame_size, disp.shape[::-1])
    inv_floor = floor_inverse_depth(ray_cam, R, c, floor_z)
    fit = fit_floor(disp, inv_floor, ray_cam, exclude, warp)
    z = metric_depth(disp, fit, ray_cam)
    height = heights(z, ray_cam, R, c, floor_z)
    skirt = np.ones((2 * round(RAISED_SKIRT * disp.shape[1]) + 1,) * 2, np.uint8)
    predicted = (inv_floor > 1 / FAR_M) & (inv_floor < 1 / NEAR_M) & np.isfinite(disp)
    for _ in range(refits):
        raised = cv2.dilate((~np.isfinite(height) | (height > RAISED_M)).astype(np.uint8), skirt).astype(bool)
        if fit["scale"] > 0 and (raised & predicted).sum() > RAISED_MAX * predicted.sum():
            break               # (an upside-down fit, a < 0, is no fit to keep: on regardless)
        try:
            fit = fit_floor(disp, inv_floor, ray_cam, raised if exclude is None else raised | exclude, warp)
        except ValueError:      # too little floor left: the last fit stands
            break
        z = metric_depth(disp, fit, ray_cam)
        height = heights(z, ray_cam, R, c, floor_z)
    return {**fit, "z": z, "height": height, "rays": ray_cam,
            "floor_z_expected": np.where(inv_floor > 0, 1 / np.maximum(inv_floor, 1e-9), np.nan)}
