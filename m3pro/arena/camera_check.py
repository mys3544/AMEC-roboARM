"""Where is the camera, measured from the floor it sees? The floor in the bottom rows of the aligned
depth, fitted as a plane in the camera's own frame (RANSAC), gives the camera's height above the
floor, how far it looks down and its roll -- independent of any transform -- which are then
compared with what the launch's camera:= argument says. Run it with the mapping stack up
(mapper:=rtabmap starts the camera).

    python3 camera_check.py        (tools/arena_map.py camcheck)
"""

import math

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

BOTTOM = 0.15   # the share of the image rows, from the bottom, taken as floor candidates


def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def floor_in_camera(points):
    """The dominant plane among `points` (camera optical frame: x right, y down, z ahead): its
    unit normal towards the camera and the camera's distance to it, and the inlier share."""
    rng = np.random.default_rng(0)
    best, best_n = None, 0
    for _ in range(300):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inliers = np.abs((points - a) @ n) < 0.01
        if inliers.sum() > best_n:
            best, best_n = inliers, inliers.sum()
    p = points[best]
    centre = p.mean(axis=0)
    n = np.linalg.svd(p - centre, full_matrices=False)[2][2]   # the inliers' least-squares normal
    d = -n @ centre
    if d < 0:                               # orient it towards the camera (the origin)
        n, d = -n, -d
    return n, d, best_n / len(points)


def main():
    rclpy.init()
    node = rclpy.create_node("camera_check")
    tf = Buffer()
    _listener = TransformListener(tf, node)
    got = {}
    node.create_subscription(Image, "/camera/depth/image_raw",
                             lambda m: got.setdefault("depth", m), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, "/camera/color/camera_info",
                             lambda m: got.setdefault("info", m), qos_profile_sensor_data)
    t = None
    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.1)
        if len(got) == 2:
            try:
                t = tf.lookup_transform("base_footprint", got["depth"].header.frame_id, Time())
                break
            except Exception:   # noqa: BLE001, S112 -- the transforms are not in yet: wait on
                continue
    if t is None:
        raise SystemExit(f"no depth, camera info or transform yet (got {sorted(got)})")
    d = got["depth"]
    depth = np.frombuffer(d.data, np.uint16).reshape(d.height, d.width).astype(float) / 1000
    fx, _, cx, _, fy, cy = got["info"].k[:6]
    rows = int(d.height * (1 - BOTTOM))
    v, u = np.nonzero((depth[rows:] > 0.2) & (depth[rows:] < 3.0))
    v = v + rows
    z = depth[v, u]
    cam = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)
    if len(cam) < 500:
        raise SystemExit("too little depth in the bottom rows to find the floor")
    n, height, share = floor_in_camera(cam)
    down = math.degrees(math.asin(-n[2]))    # the optical axis below the horizon
    roll = math.degrees(math.asin(n[0]))
    R = quat_to_matrix(t.transform.rotation)
    p = t.transform.translation
    said_down = math.degrees(math.asin(-R[2, 2]))
    said_roll = math.degrees(math.asin(-R[2, 0]))
    print(f"measured from the floor ({share * 100:.0f} % of the bottom rows on it): camera"
          f" {height:.3f} m above the floor, looking {down:.1f} deg down, roll {roll:.1f} deg")
    print(f"the launch says:                                   {p.z:.3f} m, {said_down:.1f} deg"
          f" down, roll {said_roll:.1f} deg")
    ok = abs(height - p.z) < 0.02 and abs(down - said_down) < 1.5 and abs(roll - said_roll) < 1.5
    print("they agree" if ok else
          f"MISMATCH -- for the measured pose: camera:=x,y,{height - 0.02:.3f},0,{down:.1f}"
          " (base_link sits 2 cm above the floor; roll is not an argument)")


if __name__ == "__main__":
    main()
