"""The lab's ChArUco board as the M3's camera sees it, and the bridge's HTTP (tools/m3bridge.py):
what tools/m3_touch.py looks and moves with.

The hand-eye fit that was here (capture, solve -> data/m3_handeye.json) went 2026-10-05: nothing
read its output -- the URDF's camera mounting is the one in use (m3kinematics.camera_optical; the
fit was degenerate, 2026-09-30). It is in git.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roboarm import m3kinematics

BRIDGE = "http://192.168.73.66:8761"
# The lab's board (the X3's roboarm/config.py BOARD_*): A3, taped flat, DICT_5X5 markers.
ARUCO_DICT = "DICT_5X5_250"
BOARD_SQUARES = (7, 5)      # 7 across (left-right), 5 deep (forward)
BOARD_SQUARE_M = 0.0530     # measured: 53 x 53 mm black squares
BOARD_MARKER_M = 0.0385     # measured: 7 modules of 5.5 mm (a 5x5 code and its border)


def urdf_camera() -> np.ndarray:
    """arm4 -> the camera's optical frame, as the URDF has it."""
    return m3kinematics.camera_optical()


def rt(params: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = cv2.Rodrigues(np.asarray(params[:3], float))[0]
    m[:3, 3] = params[3:6]
    return m


def board():
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
    return cv2.aruco.CharucoBoard(BOARD_SQUARES, BOARD_SQUARE_M, BOARD_MARKER_M, dictionary)


def corners(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The board's chessboard corners seen: (pixels N x 2, board points N x 3, metres)."""
    b = board()
    found, ids, _m, _mi = cv2.aruco.CharucoDetector(b).detectBoard(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if ids is None or len(ids) == 0:
        return np.empty((0, 2)), np.empty((0, 3))
    return found.reshape(-1, 2), b.getChessboardCorners()[ids.ravel()]


def http(path: str, body: dict | None = None):
    request = urllib.request.Request(BRIDGE + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as reply:
            return json.load(reply)
    except urllib.error.HTTPError as e:
        return json.loads(e.read())
