"""arena/fence_fit.py on a drawn map: a turned 3.2 x 2.8 m fence with gaps and three landmarks."""

import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "arena"))
import fence_fit

RES, ORIGIN, SHAPE = 0.02, (-1.0, -1.5), (260, 250)   # 5.2 x 5.0 m of map, rows x cols
LANDMARKS = [(0.9, 0.7), (2.3, 1.9), (1.6, 2.5)]       # arena frame, m; the last 25 cm off a wall


def to_cell(xy):
    return (int(np.rint((xy[0] - ORIGIN[0]) / RES - 0.5)),
            int(np.rint(SHAPE[0] - (xy[1] - ORIGIN[1]) / RES - 0.5)))


def draw_map(tmp_path):
    turn = np.radians(7)
    ex, ey = np.array([np.cos(turn), np.sin(turn)]), np.array([-np.sin(turn), np.cos(turn)])
    o = np.array([0.2, -0.6])
    corners = [o, o + 3.2 * ex, o + 3.2 * ex + 2.8 * ey, o + 2.8 * ey]
    grid = np.full(SHAPE, 205, np.uint8)
    cv2.fillPoly(grid, [np.array([to_cell(c) for c in corners])], 254)
    rng = np.random.default_rng(1)
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        for t in np.linspace(0, 1, 400):
            if i == 2 and 0.3 < t < 0.5:
                continue   # a stretch the lidar never saw
            c, r = to_cell(a + t * (b - a) + rng.normal(0, 0.006, 2))
            grid[r, c] = 0
    for a, b in LANDMARKS:
        cv2.circle(grid, to_cell(o + a * ex + b * ey), 4, 0, -1)   # ~16 cm across
    cv2.imwrite(str(tmp_path / "arena.pgm"), grid)
    (tmp_path / "arena.yaml").write_text(
        f"image: arena.pgm\nmode: trinary\nresolution: {RES}\n"
        f"origin: [{ORIGIN[0]}, {ORIGIN[1]}, 0]\nnegate: 0\n")
    return [o + a * ex + b * ey for a, b in LANDMARKS]


def test_fence_and_landmarks_are_found(tmp_path):
    truth = draw_map(tmp_path)
    grid, res, origin = fence_fit.load(tmp_path / "arena.yaml")
    result = fence_fit.fit(grid, res, origin)
    fence = result["fence"]
    assert abs(fence["size_m"][0] - 3.2) < 0.015 and abs(fence["size_m"][1] - 2.8) < 0.015
    assert all(abs(a - 90) < 0.5 for a in fence["corner_angles_deg"])
    assert min(w["seen"] for w in fence["walls"]) < 0.9 < max(w["seen"] for w in fence["walls"])
    assert len(result["blobs"]) == 3
    for blob in result["blobs"]:
        assert min(np.linalg.norm(np.array(blob["map_xy_m"]) - t) for t in truth) < 0.02
        # the drawn fence's x axis points +x on the map, so the arena frame is the drawn one
        assert min(np.linalg.norm(np.subtract(blob["arena_xy_m"], lm)) for lm in LANDMARKS) < 0.02
    fence_fit.render(grid, res, origin, result, tmp_path / "fit.png")
    assert cv2.imread(str(tmp_path / "fit.png")) is not None
    json.dumps(result)
