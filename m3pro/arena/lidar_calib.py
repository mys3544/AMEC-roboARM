"""Calibrate the two lidars against the fence, from a recording's first seconds
(tools/arena_map.py record starts with the robot standing still for 20 s):

    python3 lidar_calib.py BAG [--seconds 15] [--write lidars_cal.yaml]

For each lidar, each beam's median range over those seconds. First each lidar alone against the
fence's measured 3.2 x 2.8 m: every wall it sees comes out `bias` too far, so two facing walls
come out 2 x bias too far apart -- that is its range bias (where it sits on the robot does not
change it). Then, biases taken off, both scans into base_link through lidars.yaml and the second
lidar's points laid onto the first's (ICP against the first scan's outline densified between
neighbouring beams, so it matches points to lines rather than to where beams happened to land).
No SLAM is involved. --write puts the biases and the second lidar's corrected mounting in a
lidars.yaml for arena_slam.launch.py.
"""

import argparse
import json
import math
import pathlib
import warnings

import fence_fit
import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import LaserScan

HERE = pathlib.Path(__file__).resolve().parent


def still_scans(bag, seconds):
    """{frame_id: [ranges, ...]} and the angles, for the first `seconds` of the recording."""
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(bag), storage_id="sqlite3"),
                ConverterOptions(input_serialization_format="cdr",
                                 output_serialization_format="cdr"))
    ranges, angles, t0 = {}, {}, None
    while reader.has_next():
        topic, data, t = reader.read_next()
        t0 = t if t0 is None else t0
        if (t - t0) / 1e9 > seconds:
            break
        if topic in ("/scan0", "/scan1"):
            scan = deserialize_message(data, LaserScan)
            frame = scan.header.frame_id
            ranges.setdefault(frame, []).append(np.asarray(scan.ranges, float))
            angles[frame] = scan.angle_min + scan.angle_increment * np.arange(len(scan.ranges))
    return ranges, angles


def to_base(ranges, angles, mount, bias=0.0):
    """Median range per beam, less the lidar's bias -> points in base_link, the chassis
    (< 0.18 m) and beyond 5 m out; also each point's beam index, to know which points are
    neighbours along the scan."""
    with warnings.catch_warnings():   # beams that never returned anything: NaN, dropped below
        warnings.simplefilter("ignore", RuntimeWarning)
        r = np.nanmedian(np.where(np.isfinite(ranges), ranges, np.nan), axis=0) - bias
    x, y, _z, yaw_deg = mount
    a = angles + math.radians(yaw_deg)
    pts = np.stack([x + r * np.cos(a), y + r * np.sin(a)], axis=1)
    keep = np.isfinite(r) & (r < 5.0) & (np.hypot(pts[:, 0], pts[:, 1]) > 0.18)
    return pts[keep], np.nonzero(keep)[0]


def densify(pts, beams, step=0.005, gap=0.06):
    """The outline: points added every `step` between neighbouring beams closer than `gap`."""
    out = [pts]
    for i in range(len(pts) - 1):
        d = pts[i + 1] - pts[i]
        n = int(np.linalg.norm(d) / step)
        if beams[i + 1] - beams[i] == 1 and np.linalg.norm(d) < gap and n > 1:
            out.append(pts[i] + np.outer(np.arange(1, n) / n, d))
    return np.concatenate(out)


def icp(source, target, iterations=60):
    """The rigid 2D transform (R, t) that best lays `source` onto `target`, and the RMS gap."""
    R, t = np.eye(2), np.zeros(2)
    for i in range(iterations):
        moved = source @ R.T + t
        d2 = ((moved[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
        nearest = target[d2.argmin(axis=1)]
        gap = np.sqrt(d2.min(axis=1))
        pair = gap < (0.10 if i < 10 else 0.04 if i < 30 else 0.02)
        a, b = moved[pair], nearest[pair]
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        u, _s, vt = np.linalg.svd((a - ca).T @ (b - cb))
        dR = (u @ vt).T
        if np.linalg.det(dR) < 0:
            dR = (u @ np.diag([1, -1]) @ vt).T
        R, t = dR @ R, dR @ t + cb - dR @ ca
    d2 = (((source @ R.T + t)[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
    gap = np.sqrt(d2.min(axis=1))
    return R, t, float(np.sqrt(np.mean(gap[gap < 0.02] ** 2))), float(np.mean(gap < 0.02))


def fence_of(points, res=0.01):
    """fence_fit on the points drawn into a grid (no free space needed: all four walls are in)."""
    lo = points.min(axis=0) - 0.2
    size = np.ceil((points.max(axis=0) + 0.2 - lo) / res).astype(int)
    grid = np.full((size[1], size[0]), 205, np.uint8)
    cols = ((points[:, 0] - lo[0]) / res).astype(int)
    rows = size[1] - 1 - ((points[:, 1] - lo[1]) / res).astype(int)
    grid[rows, cols] = 0
    return fence_fit.fit(grid, res, lo)


FIRST, SECOND = "laser0_frame", "laser1_frame"


def calibrate(ranges, angles, mounts):
    """The biases {frame: m}, the second lidar's corrected mounting [x, y, z, yaw_deg], and the
    report printed by main. `ranges` {frame: [scan ranges, ...]}, `angles` {frame: beam angles},
    `mounts` {frame: [x, y, z, yaw_deg]} as in lidars.yaml."""
    first, second = FIRST, SECOND

    def points(frame, bias=0.0):
        return to_base(np.array(ranges[frame]), angles[frame], mounts[frame], bias)

    # 1. each lidar alone against the fence: its range bias
    bias, alone = {}, {}
    for frame in (first, second):
        size = fence_of(points(frame)[0])["fence"]["size_m"]
        bias[frame] = round(float(np.mean(np.subtract(size, fence_fit.NOMINAL)) / 2), 4)
        alone[frame] = {"size_m": size, "range_bias_cm": round(bias[frame] * 100, 2)}
    # 2. the biases off, the second lidar laid onto the first
    p0, b0 = points(first, bias[first])
    p1, _b1 = points(second, bias[second])
    R, t, rms, share = icp(p1, densify(p0, b0))
    dyaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    x1, y1, z1, yaw1 = mounts[second]
    x1n, y1n = R @ np.array([x1, y1]) + t
    corrected = [round(float(x1n), 5), round(float(y1n), 5), z1, round(yaw1 + dyaw, 3)]
    before = fence_of(np.concatenate([points(first)[0], points(second)[0]]))["fence"]
    after = fence_of(np.concatenate([p0, p1 @ R.T + t]))["fence"]
    report = {
        "frames": {k: len(v) for k, v in ranges.items()},
        "each_lidar_alone": alone,
        "second_lidar_correction": {"dx_cm": round(float(t[0]) * 100, 2),
                                    "dy_cm": round(float(t[1]) * 100, 2),
                                    "dyaw_deg": round(dyaw, 3),
                                    "matched_rms_cm": round(rms * 100, 2),
                                    "matched_share": round(share, 3)},
        "fence_standing_still": {"as_mounted": [before["size_m"], before["corner_angles_deg"]],
                                 "calibrated": [after["size_m"], after["corner_angles_deg"]],
                                 "thick_cm": [w["thick_cm"] for w in after["walls"]]},
    }
    return bias, corrected, report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bag")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--lidars", default=str(HERE / "lidars.yaml"))
    parser.add_argument("--write", help="a lidars.yaml with the biases and the second lidar's "
                                            "corrected mounting")
    args = parser.parse_args()
    mounts = {k: v for k, v in yaml.safe_load(pathlib.Path(args.lidars).read_text()).items()
              if k.endswith("_frame")}
    ranges, angles = still_scans(args.bag, args.seconds)
    bias, corrected, report = calibrate(ranges, angles, mounts)
    print(json.dumps(report, indent=1))
    if args.write:
        pathlib.Path(args.write).write_text(
            "# lidars.yaml calibrated by lidar_calib.py: the biases, and the second lidar laid\n"
            "# onto the first\n"
            f"{FIRST}: {mounts[FIRST]}\n{SECOND}: {corrected}\n"
            f"range_bias: {{{FIRST}: {bias[FIRST]}, {SECOND}: {bias[SECOND]}}}   # m, taken off\n")


if __name__ == "__main__":
    main()
