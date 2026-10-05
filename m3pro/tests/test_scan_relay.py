"""arena/scan_relay.py: the relay takes a lidar's range bias (lidars.yaml) off and blanks the
chassis. ROS's modules are stubbed."""

import math
import pathlib
import sys
from unittest import mock

for name in ("rclpy", "rclpy.node", "rclpy.qos", "rclpy.time", "sensor_msgs", "sensor_msgs.msg",
             "tf2_ros", "yaml"):   # yaml: ROS's
    sys.modules.setdefault(name, mock.MagicMock())
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "arena"))
import scan_relay


def test_the_relay_takes_the_bias_off_and_blanks_the_chassis():
    # a lidar at (0.1, 0) facing +x: beam 0 looks out, beam 1 looks back across the robot
    out = scan_relay.corrected([1.0, 0.2, math.inf], 0.0, math.pi, (0.1, 0.0, 0.0), 0.01)
    assert math.isclose(out[0], 0.99, abs_tol=1e-6)
    assert math.isinf(out[1])    # 0.19 back from the lidar = 0.09 from the centre: the chassis
    assert math.isinf(out[2])
