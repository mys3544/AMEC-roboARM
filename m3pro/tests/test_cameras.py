"""The wrist-camera switch, ROBOARM_CAMERA (cameras/README.md).

Each case runs in a fresh interpreter: config reads the variable once, at import,
and conftest.py pins the rest of the suite to the Sonix.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROBE = """
import json
import numpy as np
from roboarm import camera, detect
from roboarm import config as cfg
from roboarm import workspace as ws
undistort = camera.Undistort()
w, h = cfg.WRIST_CAM_SIZE
ramp = np.tile(np.arange(w, dtype=np.uint8), (h, 1))[..., None].repeat(3, axis=2)
other = np.zeros((h, w + 16, 3), np.uint8)
print(json.dumps({
    "camera": cfg.WRIST_CAMERA,
    "size": list(cfg.WRIST_CAM_SIZE),
    "lens": [cfg.CAMERA_FROM_J4, cfg.CAMERA_ABOVE_TOOL, cfg.CAMERA_OFF_AXIS, cfg.CAMERA_SIDE],
    "calibration": ws.CALIBRATION_PATH.as_posix(),
    "background": detect.BACKGROUND_PATH.as_posix(),
    "corrects": bool((undistort(ramp) != ramp).any()),
    "other_sizes_untouched": undistort(other) is other,
}))
"""


def probe(camera: str | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("ROBOARM_CAMERA", None)
    if camera is not None:
        env["ROBOARM_CAMERA"] = camera
    return subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120, check=False)


def settings(camera: str | None) -> dict:
    done = probe(camera)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_c930e_is_the_default():
    got = settings(None)
    assert got["camera"] == "c930e"
    assert got["size"] == [848, 480]
    assert got["calibration"] == "/app/data/cameras/c930e/table_homography.json"
    assert got["background"] == "/app/data/cameras/c930e/table_background.png"
    assert got["corrects"] is True
    assert got["other_sizes_untouched"] is True


def test_the_sonix_gets_everything_it_had_before_the_swap():
    got = settings("sonix")
    assert got["size"] == [640, 480]
    assert got["lens"] == [0.065, 0.0, 0.050, -1]
    assert got["calibration"] == "/app/data/cameras/sonix/table_homography.json"
    assert got["corrects"] is False


def test_an_unknown_camera_is_refused_at_import():
    done = probe("webcam")
    assert done.returncode != 0
    assert "ROBOARM_CAMERA" in done.stderr


def test_the_way_back_to_the_sonix_is_kept():
    """cameras/sonix.env is what compose reads to put the Sonix back, and the copy
    of its calibration is what the robot's data/ can be restored from."""
    env = dict(line.split("=", 1) for line in (ROOT / "cameras" / "sonix.env").read_text().splitlines()
               if line and not line.startswith("#"))
    assert env["ROBOARM_CAMERA"] == "sonix"
    assert "Sonix" in env["ROBOARM_WRIST_CAM_DEV"]
    saved = json.loads((ROOT / "cameras" / "sonix" / "table_homography.json").read_text())
    assert [e["name"] for e in saved["also"]] == ["outer", "near"]
    assert len(saved["homography"]) == 3
