"""Heights from two views of the table: plane + parallax (x3plus/PLAN.md X2.4).

Depth networks find WHERE things are, but every one tried flattens a 30-40 mm piece in a wide
picture (x3plus/PLAN.md, the 2026-10-06 table). Geometry does not: two pictures from known poses
see a raised top shifted against the table by an amount that grows with its height.

For a height h, the plane z = table + h maps pixels of view A into view B by a homography the
two camera poses give (plane_map). The poses come from the arm, good to about a degree -- enough
to move a whole picture by several pixels, as much as the parallax itself. But a pose error that
is mostly a ROTATION moves every plane's map alike, so it is measured once on the table, whose
texture both pictures show (table_correction: features, RANSAC homography), and taken out of
every plane's map. Then, pixel by pixel, the h whose corrected map makes A and B agree best
(local normalised cross-correlation) is that pixel's height. Textureless faces say nothing; the
edges of things carry it, so the caller reads heights along an object's outline.
"""

from __future__ import annotations

import cv2
import numpy as np

WINDOW = 11            # px, the correlation window
HEIGHTS = np.arange(-0.006, 0.0801, 0.001)
GOOD_NCC = 0.6         # below this a pixel's best match is not believed


def plane_map(K, Ra, ca, Rb, cb, plane_z: float, frame: tuple[int, int] = (848, 480)) -> np.ndarray:
    """H with x_B ~ H x_A for points on the plane z = plane_z (base frame); R, c as from
    roboarm.lens.camera_pose (camera axes as columns, centre)."""
    w, h = frame
    src = np.array([[0.15 * w, 0.5 * h], [0.85 * w, 0.5 * h], [0.85 * w, 0.98 * h], [0.15 * w, 0.98 * h]])
    rays = np.c_[src, np.ones(4)] @ np.linalg.inv(K).T @ Ra.T
    t = (plane_z - ca[2]) / rays[:, 2]
    points = ca + rays * t[:, None]
    cam = (points - cb) @ Rb
    dst = (cam @ np.asarray(K).T)[:, :2] / cam[:, 2:3]
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))


def table_correction(grey_a, grey_b, mask_a, mask_b, H0) -> tuple[np.ndarray, dict]:
    """C with C @ H0 = the table's map A -> B as the pictures show it, from ORB features on the
    table (masks: where each picture shows the table). Identity when too little matches."""
    orb = cv2.ORB_create(4000)
    ka, da = orb.detectAndCompute(grey_a, mask_a.astype(np.uint8) * 255)
    kb, db = orb.detectAndCompute(grey_b, mask_b.astype(np.uint8) * 255)
    if da is None or db is None:
        return np.eye(3), {"matches": 0}
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < 30:
        return np.eye(3), {"matches": len(matches)}
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    H, inliers = cv2.findHomography(pa, pb, cv2.RANSAC, 1.5)
    if H is None:
        return np.eye(3), {"matches": len(matches)}
    keep = inliers.ravel() > 0
    off = np.linalg.norm(cv2.perspectiveTransform(pa[None], H0)[0] - pb, axis=1)
    return H @ np.linalg.inv(H0), {"matches": int(keep.sum()), "pose_off_px": float(np.median(off[keep]))}


def ncc(x: np.ndarray, y: np.ndarray, window: int = WINDOW) -> np.ndarray:
    k = (window, window)
    mx, my = cv2.blur(x, k), cv2.blur(y, k)
    cov = cv2.blur(x * y, k) - mx * my
    vx = cv2.blur(x * x, k) - mx * mx
    vy = cv2.blur(y * y, k) - my * my
    return cov / np.sqrt(np.maximum(vx * vy, 1e-6))


def height_map(a: dict, others: list[dict], K, table_z: float, heights=HEIGHTS) -> tuple[np.ndarray, np.ndarray]:
    """(height, agreement) per pixel of view A. A view is a dict of grey (float32, undistorted
    through K), R, c and table (a bool mask of where it shows the table)."""
    shape = a["grey"].shape
    frame = (shape[1], shape[0])
    score = np.zeros((len(heights), *shape), np.float32)
    for b in others:
        H0 = plane_map(K, a["R"], a["c"], b["R"], b["c"], table_z, frame)
        C, _info = table_correction(a["grey"].astype(np.uint8), b["grey"].astype(np.uint8),
                                    a["table"], b["table"], H0)
        for i, h in enumerate(heights):
            H = C @ plane_map(K, a["R"], a["c"], b["R"], b["c"], table_z + h, frame)
            warped = cv2.warpPerspective(b["grey"], H, frame, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                                         borderValue=-1)
            s = ncc(a["grey"], warped)
            s[warped < 0] = 0
            score[i] += s
    # the best step, refined between its neighbours by a parabola through the three scores
    heights = np.asarray(heights)
    best = np.clip(np.argmax(score, axis=0), 1, len(heights) - 2)
    s0, s1, s2 = (np.take_along_axis(score, (best + d)[None], 0)[0] for d in (-1, 0, 1))
    curve = s0 - 2 * s1 + s2
    shift = np.where(curve < -1e-6, 0.5 * (s0 - s2) / np.where(curve < -1e-6, curve, -1), 0.0)
    step = heights[1] - heights[0]
    return heights[best] + np.clip(shift, -0.5, 0.5) * step, score.max(axis=0) / max(1, len(others))


def object_height(height: np.ndarray, agreement: np.ndarray, region: np.ndarray) -> float | None:
    """A thing's height from the believed pixels of `region` (its outline, a little grown): the
    median, or None when too few pixels agree."""
    good = height[region & (agreement > GOOD_NCC)]
    return float(np.median(good)) if good.size >= 20 else None
