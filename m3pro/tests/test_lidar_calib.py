"""arena/lidar_calib.py and arena/scan_relay.py on scans drawn inside a 3.2 x 2.8 m fence: the
two lidars' range biases and the second one's mounting error are found again, and the relay
takes a bias off and blanks the chassis. ROS's modules are stubbed."""

import math
import pathlib
import sys
from unittest import mock

import numpy as np

for name in ("rclpy", "rclpy.node", "rclpy.qos", "rclpy.time", "rclpy.serialization",
             "rosbag2_py", "sensor_msgs", "sensor_msgs.msg", "tf2_ros", "yaml"):   # yaml: ROS's
    sys.modules.setdefault(name, mock.MagicMock())
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "arena"))
import lidar_calib
import scan_relay

FENCE = (3.2, 2.8)
BEAMS = 4.712 / 500 * np.arange(500)                     # the T-mini Plus: 270 deg in 500 beams
NOMINAL = {"laser0_frame": [-0.11617, 0.09156, 0.1253, 0.0],
           "laser1_frame": [0.10766, -0.09078, 0.1253, 0.0]}


def scans(robot, mount, bias, frames=20, seed=0):
    """Ranges a lidar mounted at `mount` (x, y, yaw_deg) on a robot at `robot` (x, y, heading)
    reads inside the fence: the true distance to the wall + its bias + 5 mm of noise."""
    rx, ry, heading = robot
    c, s = math.cos(heading), math.sin(heading)
    px, py = rx + c * mount[0] - s * mount[1], ry + s * mount[0] + c * mount[1]
    phi = heading + math.radians(mount[2]) + BEAMS
    with np.errstate(divide="ignore"):
        tx = np.where(np.cos(phi) > 0, (FENCE[0] - px) / np.cos(phi), -px / np.cos(phi))
        ty = np.where(np.sin(phi) > 0, (FENCE[1] - py) / np.sin(phi), -py / np.sin(phi))
    true = np.minimum(np.abs(tx), np.abs(ty))
    rng = np.random.default_rng(seed)
    return [true + bias + rng.normal(0, 0.005, len(true)) for _ in range(frames)]


def test_calibration_finds_the_biases_and_the_second_lidars_mounting():
    robot = (1.3, 1.1, math.radians(10))
    true_second = [0.10766 + 0.02, -0.09078 + 0.01, 0.5]    # 2 cm, 1 cm, 0.5 deg off nominal
    ranges = {"laser0_frame": scans(robot, [-0.11617, 0.09156, 0.0], 0.008, seed=1),
              "laser1_frame": scans(robot, true_second, 0.012, seed=2)}
    angles = {frame: BEAMS for frame in ranges}
    bias, second, _report = lidar_calib.calibrate(ranges, angles, NOMINAL)
    assert abs(bias["laser0_frame"] - 0.008) < 0.003
    assert abs(bias["laser1_frame"] - 0.012) < 0.003
    assert abs(second[0] - true_second[0]) < 0.005 and abs(second[1] - true_second[1]) < 0.005
    assert abs(second[3] - true_second[2]) < 0.2


def test_the_relay_takes_the_bias_off_and_blanks_the_chassis():
    # a lidar at (0.1, 0) facing +x: beam 0 looks out, beam 1 looks back across the robot
    out = scan_relay.corrected([1.0, 0.2, math.inf], 0.0, math.pi, (0.1, 0.0, 0.0), 0.01)
    assert math.isclose(out[0], 0.99, abs_tol=1e-6)
    assert math.isinf(out[1])    # 0.19 back from the lidar = 0.09 from the centre: the chassis
    assert math.isinf(out[2])
