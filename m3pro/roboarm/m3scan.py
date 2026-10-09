"""The object scan (PLAN.md M4 A): the M3 drives round a thing on the floor, its arm keeping the
camera on it, stopping for each view, and the views' colour and depth are fused into one coloured
model -- coloured ICP on the views (Park 2017), then a TSDF (Open3D, on the laptop for now:
tools/m3_scan.py).

Why the base has to move (2026-10-09): the camera is fixed to arm4, before J5, so it looks within
the arm's own upright plane, and the DCW2 sees nothing nearer than ~0.19 m while it reaches only
~0.25 m from the shoulder. From one spot a thing is seen through a cone ~20 deg wide: from straight
above (67-89 deg) at the pick's 20 cm, at 38-56 deg 40 cm out (a 15 cm tall thing). So the robot
stands STOP_R from the thing, facing it, takes a few views across that cone, and goes round it in
STOPS stops -- one goto() each, moving and turning at once -- back to where it began. The user's
choice over a turntable or a raised thing seen from the front only.
"""

from __future__ import annotations

import itertools
import json
import math
import time

import cv2
import numpy as np

from roboarm import m3kinematics, m3vision

NEAREST = 0.20     # m: the DCW2's depth starts at ~0.17-0.19 (m3vision.NEAR, 0.15, is only the cut)
ROOM = 0.03        # m the arm's outline keeps off the thing (the circle round its footprint, its top)
SLIM = 134         # J6: the fingers shut to 30 mm, out of the way (the width table's 30 mm)
STOP_R = 0.38      # m from base_link to the thing's middle at every stop
STOPS = 12         # round it, every 30 deg
ARC_STEP = 30      # deg round the thing at most in one goto(): a straight drive cuts the circle --
                   # between stops 90 deg apart it would pass 27 cm from the thing, the arm out at it
VIEWS = 3          # a stop's views across the cone the arm can look through
LOOK_AT = {1: 90, 2: 105, 3: 10, 4: 0, 5: 90}   # the pick's wide look: the floor 12-45 cm ahead
SHOULDER, CAMERA_REACH = np.array([0.0315, 0.0, 0.218]), 0.25   # m: J2's axis, and how far from it the
                                                                # camera gets (0.248, sampled over all poses)
GUESSES = tuple(itertools.product((40, 90, 140), (20, 70, 120), (20, 70, 120)))   # J2-J4 to start from


def camera(pose: dict) -> np.ndarray:
    """base_link <- the camera's optical frame (x right, y down, z ahead), by the arm model."""
    return m3kinematics.frames(pose)["arm4"] @ m3kinematics.camera_optical()


def _aim(j1: float, centre, elevation: float, distance: float) -> tuple[np.ndarray, np.ndarray]:
    """(where the camera should be, which way it should look) for J1 `j1`: in the arm's plane,
    `distance` from the thing's centre as that plane passes it, `elevation` deg over the table."""
    base = np.asarray(m3kinematics.CHAIN[0][0][:2])
    yaw = math.radians(90 - j1)                    # J1: see m3kinematics.SIGNS
    u = np.array([math.cos(yaw), math.sin(yaw)])
    passing = np.r_[base + float((np.asarray(centre[:2]) - base) @ u) * u, centre[2]]
    e = math.radians(elevation)
    place = passing + distance * np.r_[-math.cos(e) * u, math.sin(e)]
    return place, (passing - place) / distance


def clear_of(pose: dict, centre, radius: float, top: float) -> bool:
    """Whether the arm's outline at `pose` keeps ROOM off a thing `radius` m round `centre` (x, y)
    and `top` m tall."""
    p = m3kinematics.outline(pose)
    low = p[:, 2] < top + ROOM
    return bool(np.all(np.hypot(p[low, 0] - centre[0], p[low, 1] - centre[1]) >= radius + ROOM))


def view(j1: float, centre, elevation: float, distance: float, radius: float, top: float,
         first=None) -> dict | None:
    """The pose (J1-J4, J5 90, the fingers shut) that puts the camera at `elevation` and `distance`
    looking at the thing's `centre` (x, y, z), or None when the arm cannot, or only into the robot,
    the floor or the thing. `first`: a pose to start the search from (a neighbouring view's)."""
    place, look = _aim(j1, centre, elevation, distance)
    if np.linalg.norm(place - SHOULDER) > CAMERA_REACH:
        return None

    def error(q):
        cam = camera({1: j1, 2: q[0], 3: q[1], 4: q[2], 5: 90})
        return np.r_[(cam[:3, 3] - place) / 0.005, np.cross(cam[:3, 2], look) / math.radians(2)]   # 5 mm, 2 deg

    best = None
    starts = [(first[2], first[3], first[4])] if first is not None else []
    for guess in [*starts, *GUESSES]:
        if best is not None and best[0] < 0.003:
            break   # (a neighbour's start found it: the rest would only agree)
        q = np.array(guess, float)
        for _ in range(12):
            e = error(q)
            if np.abs(e).max() < 0.1:
                break
            J = np.column_stack([(error(q + d) - e) / 0.5 for d in np.eye(3) * 0.5])
            q = q + np.clip(np.linalg.lstsq(J, -e, rcond=None)[0], -25, 25)
        if not all(0 <= v <= 180 for v in q):
            continue
        pose = {1: round(j1), 2: round(q[0]), 3: round(q[1]), 4: round(q[2]), 5: 90, 6: SLIM}
        cam = camera(pose)
        miss = float(np.linalg.norm(cam[:3, 3] - place))
        off = math.degrees(math.acos(float(np.clip(cam[:3, 2] @ look, -1, 1))))
        if miss > 0.01 or off > 4:
            continue
        if m3kinematics.problem(pose) is not None or not clear_of(pose, centre, radius, top):
            continue
        if best is None or miss < best[0]:
            best = (miss, pose)
    return None if best is None else best[1]


def stop_views(radius: float, top: float, stop_r: float = STOP_R, count: int = VIEWS) -> list[dict]:
    """A stop's views of a thing `radius` m round and `top` m tall, standing `stop_r` m straight
    ahead of base_link: `count` poses spread over the elevations the arm can look from (low to
    high), each from as far as it can (the depth is better nearer, but the nearest face must stay
    NEAREST off). Empty when there is none."""
    middle = np.array([stop_r, 0.0, top / 2])
    nearest = NEAREST + math.hypot(radius, top / 2)
    found = []
    for elevation in range(20, 90, 5):
        for distance in (nearest + 0.08, nearest + 0.04, nearest):
            pose = view(90, middle, elevation, distance, radius, top, found[-1][1] if found else None)
            if pose is not None:
                found.append((elevation, pose))
                break
    if len(found) <= count:
        return [pose for _e, pose in found]
    picks = np.linspace(0, len(found) - 1, count).round().astype(int)
    return [found[i][1] for i in picks]


def on_circle(centre, angle: float, stop_r: float = STOP_R) -> tuple[float, float, float]:
    """The base's (x, y, yaw) `stop_r` m from `centre` at `angle` (rad, from it), facing it."""
    return (centre[0] + stop_r * math.cos(angle), centre[1] + stop_r * math.sin(angle),
            math.remainder(angle + math.pi, math.tau))


def stops(here, centre, count: int = STOPS, stop_r: float = STOP_R) -> list[tuple[float, float, float]]:
    """`count` stops round `centre` (x, y in the odometry's frame), each `stop_r` m from it and
    facing it (x, y, yaw): the first on the line from it to `here` (x, y, yaw, the robot now), the
    rest anticlockwise seen from above, every 360/count deg."""
    start = math.atan2(here[1] - centre[1], here[0] - centre[0])
    return [on_circle(centre, start + 2 * math.pi * k / count, stop_r) for k in range(count)]


def way_round(centre, a_from: float, a_to: float, stop_r: float = STOP_R) -> list[tuple[float, float, float]]:
    """The base's poses from `a_from` to `a_to` (rad) round `centre`, ARC_STEP deg apart at most,
    on the circle, facing it -- the last at `a_to`."""
    n = max(1, math.ceil(abs(a_to - a_from) / math.radians(ARC_STEP) - 1e-9))
    return [on_circle(centre, a_from + (a_to - a_from) * j / n, stop_r) for j in range(1, n + 1)]


def levelled(depth_mm: np.ndarray, K, pose: dict) -> np.ndarray:
    """The camera for one view: the arm model's, its height, tilt and roll from the floor in view
    (m3vision.level) when there is floor enough."""
    model = camera(pose)
    try:
        normal, height, _ = m3vision.floor_plane(depth_mm, K)
    except ValueError:
        return model
    return m3vision.level(model, normal, height, m3kinematics.FLOOR)


def find_thing(depth_mm: np.ndarray, K, cam: np.ndarray, near, reach: float = 0.15) -> dict | None:
    """The thing standing within `reach` m of `near` (x, y, base_link), from one view through the
    levelled camera `cam`: {"centre" (x, y), "radius" (m round it, its footprint), "top" (m)}, or
    None when fewer than 200 depth points stand there 8 mm-0.45 m over the floor."""
    p = m3vision.points(depth_mm, K, step=2)
    world = p @ cam[:3, :3].T + cam[:3, 3]
    up = world[:, 2] - m3kinematics.FLOOR
    there = (up > 0.008) & (up < 0.45) & (np.hypot(world[:, 0] - near[0], world[:, 1] - near[1]) < reach)
    if there.sum() < 200:
        return None
    xy = world[there, :2].astype(np.float32)
    (cx, cy), _size, _angle = cv2.minAreaRect(xy)
    radius = float(np.percentile(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy), 98))
    return {"centre": (float(cx), float(cy)), "radius": radius, "top": float(np.percentile(up[there], 98))}


def to_world(odom, local) -> np.ndarray:
    """A point (x, y) in base_link, where the odometry put the base at `odom` (x, y, yaw), in the
    odometry's frame."""
    c, s = math.cos(odom[2]), math.sin(odom[2])
    return np.array([odom[0] + c * local[0] - s * local[1], odom[1] + s * local[0] + c * local[1]])


def world_of(odom) -> np.ndarray:
    """odometry frame <- base_link, 4 x 4, for the base at `odom` (x, y, yaw) on the floor."""
    c, s = math.cos(odom[2]), math.sin(odom[2])
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    m[:2, 3] = odom[:2]
    return m


def run(p, radius: float = STOP_R, stops_n: int = STOPS) -> None:
    """The scan as a job of the M3's Picker `p` (its arm, camera, base and narration): find the
    thing about `radius` m ahead, stand `radius` from it facing it, then at each of `stops_n` stops
    round it take stop_views() and keep every view (out/scan/STAMP/: the colour, the depth, a JSON
    of the pose, K and the odometry), and come back to the first stop."""
    if p.goto is None or p.odom is None:
        raise RuntimeError("the base cannot drive here: no wheel odometry")
    stops_n = int(stops_n)
    folder = (p.out.parent if p.out is not None else None)
    folder = None if folder is None else folder / "scan" / time.strftime("%Y%m%d_%H%M%S")
    p.engaged()
    with p.step("look.scan"):
        p.move(LOOK_AT, speed=p.dps("free_dps"))
        time.sleep(p.settle_s if p.settle_s is not None else p.cfg["settle_wide_s"])
        _image, depth, K, pose = p.snapshot()
        thing = find_thing(depth, K, levelled(depth, K, pose), (radius, 0.0))
    if thing is None:
        raise RuntimeError(f"nothing standing about {100 * radius:.0f} cm ahead to scan")
    here = p.odom()
    if here is None:
        raise RuntimeError("no wheel odometry (/odom_raw) from the board: not driving blind")
    centre = to_world(here, thing["centre"])
    p.say(f"to scan: a thing {100 * thing['top']:.0f} cm tall, {200 * thing['radius']:.0f} cm across, "
          f"{100 * float(np.hypot(*thing['centre'])):.0f} cm ahead")
    views = stop_views(thing["radius"], thing["top"], radius)
    if not views:
        raise RuntimeError("the arm has no view of it from a stop: too tall or too wide for this camera")
    plan = stops(here, centre, stops_n, radius)
    start = math.atan2(here[1] - centre[1], here[0] - centre[0])
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "scan.json").write_text(json.dumps({"centre": centre.tolist(), "radius": thing["radius"],
                                                      "top": thing["top"], "stops": plan}, indent=1))
    kept = 0
    for k in range(stops_n + 1):
        p.check()
        with p.step("drive"):   # the first: onto the circle; then round it, the last back to the first
            for way in ([plan[0]] if k == 0 else way_round(centre, start + math.tau * (k - 1) / stops_n,
                                                             start + math.tau * k / stops_n, radius)):
                p.check()
                p.goto(*way)
        if k == stops_n:
            break   # back at the first: the loop closed
        for i, pose in enumerate(views if k % 2 == 0 else views[::-1]):
            with p.step("look.scan"):
                p.move(pose, speed=p.dps("free_dps"))
                time.sleep(p.settle_s if p.settle_s is not None else p.cfg["settle_close_s"])
                image, depth, K, joints = p.snapshot()
                odom = p.odom()
            if folder is not None:
                name = f"{k:02d}_{i}"
                cv2.imwrite(str(folder / f"{name}_rgb.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(folder / f"{name}_depth.png"), depth)
                (folder / f"{name}.json").write_text(json.dumps({"stop": k, "pose": {str(j): v for j, v in joints.items()},
                                                                 "K": np.asarray(K).tolist(), "odom": list(odom)}))
            kept += 1
        p.say(f"stop {k + 1} of {stops_n}: {len(views)} views")
    p.home()
    p.say(f"scan done: {kept} views" + (f" in {folder}" if folder is not None else ""))


def fuse(views: list[dict], centre, radius: float, top: float, voxel: float = 0.0015):
    """One coloured model from the views (each {"rgb" (H x W x 3, RGB), "depth" (mm), "K", "camera"
    (world <- optical, 4 x 4)}): the depth cut to a box round the thing (its floor ring kept, it
    holds the views together), each view's camera corrected by coloured ICP onto the views before
    it (Park 2017, coarse to fine; a correction past 3 cm or 8 deg is not believed), then a TSDF at
    `voxel` m. Returns (points n x 3, colours n x 3 in 0..1, the mesh, the cameras as corrected)."""
    import open3d as o3d  # the laptop's scan group (pyproject): not on the robot yet

    lo = np.r_[np.asarray(centre[:2]) - radius - 0.04, m3kinematics.FLOOR - 0.004]
    hi = np.r_[np.asarray(centre[:2]) + radius + 0.04, m3kinematics.FLOOR + top + 0.02]
    clouds, images, cameras = [], [], []
    for v in views:
        K = np.asarray(v["K"], float)
        depth = cut_to_box(v["depth"], K, v["camera"], lo, hi)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(v["rgb"])), o3d.geometry.Image(depth),
            depth_scale=1000.0, depth_trunc=1.0, convert_rgb_to_intensity=False)
        h, w = depth.shape
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        clouds.append(o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, np.linalg.inv(v["camera"])))
        images.append((rgbd, intrinsic))
        cameras.append(np.array(v["camera"], float))
    model = clouds[0]
    for i in range(1, len(clouds)):
        fix = np.eye(4)
        for size in (0.008, 0.004, 0.002):
            src, dst = clouds[i].voxel_down_sample(size), model.voxel_down_sample(size)
            if len(src.points) < 30 or len(dst.points) < 30:
                break
            for c in (src, dst):
                c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=size * 2.5, max_nn=30))
            try:
                fit = o3d.pipelines.registration.registration_colored_icp(
                    src, dst, size * 1.5, fix, o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6,
                                                                      max_iteration=40))
            except RuntimeError:   # too few correspondences at this scale
                break
            fix = fit.transformation
        turned = math.degrees(math.acos(float(np.clip((np.trace(fix[:3, :3]) - 1) / 2, -1, 1))))
        if np.linalg.norm(fix[:3, 3]) <= 0.03 and turned <= 8:
            cameras[i] = fix @ cameras[i]
            clouds[i].transform(fix)
        model += clouds[i]
        model = model.voxel_down_sample(0.002)
    # a cube over the box (Open3D 0.20's ScalableTSDFVolume gave nothing, even for a flat plane)
    length = float(np.max(hi - lo))
    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length=length, resolution=math.ceil(length / voxel), sdf_trunc=voxel * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8, origin=lo.reshape(3, 1))
    for (rgbd, intrinsic), cam in zip(images, cameras, strict=True):
        volume.integrate(rgbd, intrinsic, np.linalg.inv(cam))
    cloud = volume.extract_point_cloud()
    keep = np.all((np.asarray(cloud.points) >= lo) & (np.asarray(cloud.points) <= hi), axis=1)
    return np.asarray(cloud.points)[keep], np.asarray(cloud.colors)[keep], volume.extract_triangle_mesh(), cameras


def cut_to_box(depth_mm: np.ndarray, K, cam: np.ndarray, lo, hi) -> np.ndarray:
    """The depth image with every pixel whose point (through `cam`) lies outside the box lo..hi
    set to 0."""
    rows, cols = depth_mm.shape
    v, u = np.mgrid[0:rows, 0:cols]
    z = depth_mm.astype(np.float32) / 1000
    p = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=-1)
    world = p @ cam[:3, :3].T + cam[:3, 3]
    inside = np.all((world >= lo) & (world <= hi), axis=-1) & (z > 0)
    return np.where(inside, depth_mm, 0).astype(np.uint16)
