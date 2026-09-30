"""Drive the M3 Pro from a browser while it maps -- by hand, or on a slow loop round the arena.

    http://192.168.73.66:8095/     (started with the mapping stack by arena_slam.launch.py)
    python3 drive_server.py [--no-motion]    (--no-motion: bench mode, nothing moves)

The ROS side of the drive page: it feeds drive_core.Driver -- the page's logic, its safety
limits and its web API -- with /scan, /map, /battery and the map -> base_footprint pose, and
publishes its commands on /cmd_vel. It owns /cmd_vel only while it commands, so the gamepad
works whenever it is idle. RTAB-Map is not paused while a survey moves: it lost its map that
way (2026-09-30); the export picks the frames taken standing instead.
"""

import math
import sys
import threading

import drive_core
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener

PORT = 8095


class RosDriver(Node):
    def __init__(self, motion=True):
        super().__init__("arena_drive")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.core = drive_core.Driver(self.pose, self.send, motion=motion)
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, "/map", self.on_map, latched)
        self.create_subscription(Float32, "/battery", self.on_battery, 10)
        self.create_timer(0.05, self.core.tick)

    def on_scan(self, msg):
        r = np.asarray(msg.ranges, dtype=float)
        a = msg.angle_min + msg.angle_increment * np.arange(len(r))
        ok = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)
        self.core.set_scan(np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])], axis=1))

    def on_map(self, msg):
        h, w = msg.info.height, msg.info.width
        data = np.asarray(msg.data, dtype=np.int16).reshape(h, w)[::-1]   # image rows: top = +y
        grid = np.full((h, w), 205, np.uint8)
        grid[(data >= 0) & (data < 19.6)] = 254    # map_saver's --free 0.196
        grid[data > 65] = 0                         # and --occ 0.65
        self.core.set_map(grid, msg.info.resolution,
                          [msg.info.origin.position.x, msg.info.origin.position.y])

    def on_battery(self, msg):
        self.core.battery = round(float(msg.data), 2)

    def pose(self):
        try:
            t = self.tf.lookup_transform("map", "base_footprint", Time())
        except Exception:   # noqa: BLE001 -- tf2 raises several kinds; all mean "no pose yet"
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw

    def send(self, vx, vy, wz):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = vx, vy, wz
        self.pub.publish(msg)


def main():
    # Our own Ctrl-C handling, so the last thing published is a zero while ROS is still up.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = RosDriver(motion="--no-motion" not in sys.argv)
    threading.Thread(target=drive_core.serve, args=(node.core, PORT), daemon=True).start()
    node.get_logger().info(f"drive page on port {PORT}"
                           + ("" if node.core.motion else f" -- {drive_core.BENCH}"))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(3):
            node.send(0.0, 0.0, 0.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
