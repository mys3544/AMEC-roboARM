"""Each lidar's scan with its range bias (from lidars.yaml, measured 2026-09-30) taken off
every range, and returns within 0.18 m of the robot's centre (the chassis) blanked, as Yahboom's
filter does for /scan -- published two ways:

- /scan0_c and /scan1_c, one per lidar: what the merger merges into /scan with bias:=true;
- /scan_lidars, both on one topic, each in its own frame and at its own time, for slam_toolbox
  with scans:=lidars (it keeps one sensor per frame_id). Live, that drops nearly every scan: the
  robot board stamps them 55-75 ms ahead of the Jetson's clock, the odometry transform is never
  there yet, and a scan every 70 ms pushes the waiting one out of slam_toolbox's queue.

    python3 scan_relay.py LIDARS_YAML      (arena_slam.launch.py starts it)
"""

import math
import sys

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

CHASSIS = 0.18   # m from base_link


def corrected(ranges, angle_min, angle_increment, mount, bias):
    """A scan's ranges with the lidar's bias taken off and the chassis blanked (inf): `mount` is
    the lidar's (x, y, yaw) in base_link."""
    x, y, yaw = mount
    r = np.asarray(ranges, dtype=np.float32) - bias
    a = angle_min + angle_increment * np.arange(len(r)) + yaw
    with np.errstate(invalid="ignore"):
        chassis = np.hypot(x + r * np.cos(a), y + r * np.sin(a)) < CHASSIS
    r[chassis] = math.inf
    return r.tolist()


class Relay(Node):
    def __init__(self, bias):
        super().__init__("scan_relay")
        self.bias = bias   # {frame_id: m}
        self.both = self.create_publisher(LaserScan, "/scan_lidars", 20)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.mounts = {}
        for topic in ("/scan0", "/scan1"):
            one = self.create_publisher(LaserScan, f"{topic}_c", qos_profile_sensor_data)
            self.create_subscription(LaserScan, topic, lambda scan, one=one: self.relay(scan, one),
                                     qos_profile_sensor_data)

    def mount(self, frame):
        """The lidar's (x, y, yaw) in base_link, from the static transforms."""
        if frame not in self.mounts:
            t = self.tf.lookup_transform("base_link", frame, Time())
            q = t.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.mounts[frame] = (t.transform.translation.x, t.transform.translation.y, yaw)
        return self.mounts[frame]

    def relay(self, scan, one):
        frame = scan.header.frame_id
        try:
            mount = self.mount(frame)
        except Exception:   # noqa: BLE001 -- the static transforms are not in yet
            return
        scan.ranges = corrected(scan.ranges, scan.angle_min, scan.angle_increment, mount,
                                self.bias.get(frame, 0.0))
        one.publish(scan)
        self.both.publish(scan)


def main():
    rclpy.init()
    with open(sys.argv[1]) as f:
        node = Relay(yaml.safe_load(f).get("range_bias") or {})
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
