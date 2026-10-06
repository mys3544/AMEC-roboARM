"""tools/camera_model.py: the lens fit, on pictures of the A4 board rendered through a known camera."""

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import camera_model

SIZE = (848, 480)
K_TRUE = np.array([[600.0, 0, 430.0], [0, 600.0, 245.0], [0, 0, 1]])
SQUARE_M = 0.0297


def rotation(tilt_x_deg: float, tilt_y_deg: float) -> np.ndarray:
    rx = cv2.Rodrigues(np.array([math.radians(tilt_x_deg), 0, 0]))[0]
    ry = cv2.Rodrigues(np.array([0, math.radians(tilt_y_deg), 0]))[0]
    return ry @ rx


def render(board, tilt_x: float, tilt_y: float, distance: float, shift=(0.0, 0.0)) -> np.ndarray:
    """The board as the camera K_TRUE would see it, tilted and `distance` m away."""
    cols, rows = camera_model.A4["squares"]
    width_px, height_px = 90 * cols, 90 * rows
    picture = board.generateImage((width_px, height_px), marginSize=0)
    to_metres = np.diag([cols * SQUARE_M / width_px, rows * SQUARE_M / height_px, 1.0])
    R = rotation(tilt_x, tilt_y)
    centre = np.array([cols * SQUARE_M / 2, rows * SQUARE_M / 2, 0.0])
    t = np.array([shift[0], shift[1], distance]) - R @ centre
    H = K_TRUE @ np.column_stack([R[:, 0], R[:, 1], t]) @ to_metres
    return cv2.warpPerspective(picture, H, SIZE, borderValue=200)


def views_for(board) -> list:
    views = []
    for tilt_x, tilt_y, distance, shift in [
        (0, 0, 0.45, (0, 0)), (35, 0, 0.45, (0, 0)), (-35, 0, 0.45, (0, 0)), (0, 35, 0.45, (0, 0)),
        (0, -35, 0.45, (0, 0)), (25, 25, 0.5, (0.08, 0.02)), (-25, 25, 0.5, (-0.08, -0.02)),
        (25, -25, 0.5, (-0.08, 0.03)), (-25, -25, 0.5, (0.08, -0.03)), (15, -30, 0.4, (0.05, 0)),
        (-15, 30, 0.4, (-0.05, 0)), (40, 10, 0.55, (0, 0.04)),
    ]:
        found = camera_model.find_board(render(board, tilt_x, tilt_y, distance, shift), board)
        assert found is not None, (tilt_x, tilt_y)
        views.append(found)
    return views


def test_the_fit_recovers_the_camera_from_tilted_views():
    board = camera_model.a4_board(SQUARE_M)
    result = camera_model.fit(views_for(board), SIZE)
    K = np.array(result["K"])
    assert abs(K[0, 0] - 600) < 6 and abs(K[1, 1] - 600) < 6, K   # within 1 %
    assert abs(K[0, 2] - 430) < 6 and abs(K[1, 2] - 245) < 6, K
    assert result["rms_px"] < 0.5 and max(result["per_view_px"]) < 1.0
    assert max(abs(v) for v in result["dist"][:4]) < 0.05, "no distortion was rendered"
    assert result["views"] == 12 and result["coverage"] > 0.4


def test_holding_the_board_still_keeps_one_view():
    board = camera_model.a4_board(SQUARE_M)
    first = camera_model.find_board(render(board, 20, 0, 0.45), board)
    again = camera_model.find_board(render(board, 21, 1, 0.46), board)
    other = camera_model.find_board(render(board, -30, 0, 0.45), board)
    key = camera_model.view_key(*first, K_TRUE, SIZE)
    assert key == camera_model.view_key(*again, K_TRUE, SIZE)
    assert key != camera_model.view_key(*other, K_TRUE, SIZE)


def test_a_picture_without_enough_of_the_board_is_no_view():
    board = camera_model.a4_board(SQUARE_M)
    assert camera_model.find_board(np.full((480, 848, 3), 200, np.uint8), board) is None
    half = render(board, 0, 0, 0.45)
    half[:, 330:] = 200   # most of the board hidden
    assert camera_model.find_board(half, board) is None
