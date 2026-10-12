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
# The first look, for the thing's size: the wide look's top edge is low -- 10 cm up it sees only to
# 34 cm ahead, so a can at 38 cm lost its top and read half its size (2026-10-12). Leaned back 30 deg
# at the shoulder, the camera 5 cm further back and 5 cm higher, it holds all of 30-48 cm ahead,
# +-6 cm across, floor to 20 cm tall, 60 px off every edge and 23 cm or more away (searched over
# J2-J4 with the arm model, the chassis and a 20 cm thing at 0.39 clear).
LOOK_FIRST = {1: 90, 2: 135, 3: 10, 4: 0, 5: 90, 6: SLIM}
SHOULDER, CAMERA_REACH = np.array([0.0315, 0.0, 0.218]), 0.25   # m: J2's axis, and how far from it the
                                                                # camera gets (0.248, sampled over all poses)
GUESSES = tuple(itertools.product((40, 90, 140), (20, 70, 120), (20, 70, 120)))   # J2-J4 to start from
EDGE_PX = 4        # a thing with depth this near the picture's edge runs off it: its size is not known
FIRST_SHORT = 0.015   # m the first look reads a thing's radius short -- the depth's edge halo at its range:
                      # 2.0 against 3.3 cm on a can, the stops' closer views 2.7-3.5 (2026-10-12) -- so the
                      # first stop's views keep that much more off it, until they have read it themselves
SIZE_SLACK = 0.005    # m: a size read again closer than this to the one the views were planned for keeps them
SIGHT_PUSH = 0.25     # of a view's reading radius: how far its reading lies short of the thing's middle along
                      # the line of sight when the view does not see its top whole -- a top with no depth (a
                      # can's shiny lid) leaves only the near side, and the reading leans toward the camera.
                      # Fitted on a can (2026-10-12: 6-8 mm of 31; fused, 25.7 mm round without, 29.6 with,
                      # 31.9 a single view's arc, 33 true). A box whose matte top was seen needs none: pushed,
                      # its fused top and sides disagreed (16.9 x 11.3 against 19.1 x 13.1 cm; 16.9 x 12.2
                      # and 17.1 x 12.3 unpushed; 16.4 x 11 true)
TOP_SEEN = 0.95       # a view sees the thing's top whole when the top's outline covers this much of the
                      # outline of all of it seen (the box's views 0.96-1.00; the can's 0.48-1.00, median 0.87)


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
    n = max(1, math.ceil(abs(a_to - a_from) / math.radians(ARC_STEP) - 0.05))   # a leg a degree or two
                                                                              # over ARC_STEP (re-centred) is one
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
    levelled camera `cam`: {"centre" (x, y), "radius" (m round it, its footprint), "top" (m), "whole"
    (whether it is all in the picture), "top_seen" (how much of its outline seen from above its top's
    covers: 1 when the top is seen whole)}, or None when fewer than 200 depth points stand there 8 mm-0.45 m
    over the floor. A thing the picture's edge cuts reads short and narrow -- a 17 cm can whose top was
    out of the wide look read 9 cm tall and 3 cm across, and 16.8 and 6.4 where it was whole (2026-10-12)."""
    p = m3vision.points(depth_mm, K, step=2)
    world = p @ cam[:3, :3].T + cam[:3, 3]
    up = world[:, 2] - m3kinematics.FLOOR
    there = (up > 0.008) & (up < 0.45) & (np.hypot(world[:, 0] - near[0], world[:, 1] - near[1]) < reach)
    if there.sum() < 200:
        return None
    xy = world[there, :2].astype(np.float32)
    (cx, cy), _size, _angle = cv2.minAreaRect(xy)
    radius = float(np.percentile(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy), 98))
    uv = p[there] @ np.asarray(K, float).reshape(3, 3).T
    u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
    h, w = depth_mm.shape[:2]
    whole = bool(u.min() >= EDGE_PX and v.min() >= EDGE_PX and u.max() < w - EDGE_PX and v.max() < h - EDGE_PX)
    top = float(np.percentile(up[there], 98))
    lid = world[there & (up > top - 0.01), :2].astype(np.float32)
    outline = cv2.contourArea(cv2.convexHull(xy))
    seen = cv2.contourArea(cv2.convexHull(lid)) / outline if len(lid) >= 3 and outline > 0 else 0.0
    return {"centre": (float(cx), float(cy)), "radius": radius, "top": top, "whole": whole, "top_seen": float(seen)}


def to_world(odom, local) -> np.ndarray:
    """A point (x, y) in base_link, where the odometry put the base at `odom` (x, y, yaw), in the
    odometry's frame."""
    c, s = math.cos(odom[2]), math.sin(odom[2])
    return np.array([odom[0] + c * local[0] - s * local[1], odom[1] + s * local[0] + c * local[1]])


def to_local(odom, world) -> tuple[float, float]:
    """A point (x, y) of the odometry's frame in base_link, the base at `odom` (to_world's inverse)."""
    c, s = math.cos(odom[2]), math.sin(odom[2])
    dx, dy = world[0] - odom[0], world[1] - odom[1]
    return (c * dx + s * dy, -s * dx + c * dy)


def world_of(odom) -> np.ndarray:
    """odometry frame <- base_link, 4 x 4, for the base at `odom` (x, y, yaw) on the floor."""
    c, s = math.cos(odom[2]), math.sin(odom[2])
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    m[:2, 3] = odom[:2]
    return m


def run(p, radius: float = STOP_R, stops_n: int = STOPS, height: float | None = None,
        width: float | None = None):
    """The scan as a job of the M3's Picker `p` (its arm, camera, base and narration): find the
    thing about `radius` m ahead, stand `radius` from it facing it, then at each of `stops_n` stops
    round it take stop_views() and keep every view (out/scan/STAMP/: the colour, the depth, a JSON
    of the pose, K, the odometry and the thing as that view read it), and come back to the first stop.
    Returns that folder (None when the Picker keeps no pictures).

    The arm keeps clear of the thing by its size as the first look reads it, FIRST_SHORT wider -- or
    as given (`height`, `width`: m, the widest way across), which a thing that runs off that look's
    picture needs: there its reading is short. From the first stop on, the stop's own views read it
    again: its size (the median so far) and where it stands, and the next leg goes round THAT -- the
    wheel odometry drifts as the base goes round (a can seemed to move 9 cm in half a circle,
    2026-10-12), the thing does not."""
    if p.goto is None or p.odom is None:
        raise RuntimeError("the base cannot drive here: no wheel odometry")
    stops_n = int(stops_n)
    folder = (p.out.parent if p.out is not None else None)
    folder = None if folder is None else folder / "scan" / time.strftime("%Y%m%d_%H%M%S")
    p.engaged()
    with p.step("look.scan"):
        p.move(LOOK_FIRST, speed=p.dps("free_dps"))
        time.sleep(p.settle_s if p.settle_s is not None else p.cfg["settle_wide_s"])
        _image, depth, K, pose = p.snapshot()
        thing = find_thing(depth, K, levelled(depth, K, pose), (radius, 0.0))
    if thing is None:
        raise RuntimeError(f"nothing standing about {100 * radius:.0f} cm ahead to scan")
    given = height is not None and width is not None
    if not given:
        if not thing["whole"]:
            raise RuntimeError(f"the thing runs off the picture (it read {100 * thing['top']:.0f} cm tall, "
                               f"{200 * thing['radius']:.0f} cm across, short): give its height and width")
        size = (thing["radius"] + FIRST_SHORT, thing["top"])
    else:
        # the look saw its near side: what more it is across lies behind, away from the robot
        grown = max(0.0, width / 2 - thing["radius"])
        ahead = np.asarray(thing["centre"], float)
        thing = {**thing, "centre": tuple(ahead + grown * ahead / np.linalg.norm(ahead))}
        size = (width / 2, height)
    here = p.odom()
    if here is None:
        raise RuntimeError("no wheel odometry (/odom_raw) from the board: not driving blind")
    centre = to_world(here, thing["centre"])
    p.say(f"to scan: a thing {100 * thing['top']:.0f} cm tall, {200 * thing['radius']:.0f} cm across, "
          f"{100 * float(np.hypot(*thing['centre'])):.0f} cm ahead" + (" (its size as given)" if given else ""))
    views = stop_views(*size, radius)
    if not views:
        raise RuntimeError("the arm has no view of it from a stop: too tall or too wide for this camera")
    start = math.atan2(here[1] - centre[1], here[0] - centre[0])
    record = {"centre": centre.tolist(), "radius": size[0], "top": size[1], "centres": []}
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "scan.json").write_text(json.dumps(record, indent=1))
    kept, sizes = 0, []
    for k in range(stops_n + 1):
        p.check()
        aim = start + math.tau * k / stops_n
        with p.step("drive"):   # the first: onto the circle; then round it, the last back to the first
            if k == 0:
                legs = [on_circle(centre, aim, radius)]
            else:
                now = p.odom()
                at = math.atan2(now[1] - centre[1], now[0] - centre[0])
                legs = way_round(centre, at, at + math.remainder(aim - at, math.tau), radius)
            for way in legs:
                p.check()
                p.goto(*way)
        if k == stops_n:
            break   # back at the first: the loop closed
        reads = []
        for i, pose in enumerate(views if k % 2 == 0 else views[::-1]):
            with p.step("look.scan"):
                p.move(pose, speed=p.dps("free_dps"))
                time.sleep(p.settle_s if p.settle_s is not None else p.cfg["settle_close_s"])
                image, depth, K, joints = p.snapshot()
                odom = p.odom()
                seen = find_thing(depth, K, levelled(depth, K, joints), to_local(odom, centre))
            if seen is not None and seen["whole"]:
                reads.append((to_world(odom, seen["centre"]), seen["radius"], seen["top"]))
            if folder is not None:
                name = f"{k:02d}_{i}"
                cv2.imwrite(str(folder / f"{name}_rgb.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(folder / f"{name}_depth.png"), depth)
                (folder / f"{name}.json").write_text(json.dumps({
                    "stop": k, "pose": {str(j): v for j, v in joints.items()}, "K": np.asarray(K).tolist(),
                    "odom": list(odom), "thing": seen and {**seen, "centre": list(seen["centre"])}}))
            kept += 1
        moved = 0.0
        if reads:
            was, centre = centre, np.median([c for c, _r, _t in reads], axis=0)
            moved = float(np.linalg.norm(centre - was))
            sizes += [(r, t) for _c, r, t in reads]
            if not given:
                read = (float(np.median([r for r, _t in sizes])), float(np.median([t for _r, t in sizes])))
                if max(abs(read[0] - size[0]), abs(read[1] - size[1])) > SIZE_SLACK:
                    size = read
                    views = stop_views(*size, radius)
                    if not views:
                        raise RuntimeError("the arm has no view of it from a stop: too tall or too wide "
                                           "for this camera")
                    p.say(f"read again: {100 * size[1]:.0f} cm tall, {200 * size[0]:.0f} cm across")
        record["centres"].append(centre.tolist())
        p.say(f"stop {k + 1} of {stops_n}: {len(views)} views" + (f", the thing {100 * moved:.0f} cm off where "
                                                                   f"the wheels had it" if moved >= 0.01 else ""))
    record.update(centre=centre.tolist(), radius=size[0], top=size[1])
    if folder is not None:
        (folder / "scan.json").write_text(json.dumps(record, indent=1))
    p.home()
    p.say(f"scan done: {kept} views" + (f" in {folder}" if folder is not None else ""))
    return folder


def anchor(views: list[dict]) -> tuple[np.ndarray, float, float]:
    """Take the wheels' drift out of the views' cameras ("camera": world <- optical, the odometry's
    world): each is shifted (x, y) so that the thing it read ("seen": {"centre" (x, y, that world),
    "radius", "top", "top_seen"}, or None) lands on one point -- a reading from a view that did not see
    the thing's top whole (top_seen under TOP_SEEN) first moved SIGHT_PUSH of its radius further along
    the line of sight; a view that read nothing moves as the one before it. Between
    the first and the last stop the odometry drifted ~10 cm round a can (2026-10-12), and the views
    fused as a smear. Returns (that point, the thing's radius, its top): the readings' medians."""
    marks = []
    for v in views:
        seen = v.get("seen")
        if seen is None:
            marks.append(None)
            continue
        centre = np.asarray(seen["centre"], float)
        sight = centre - np.asarray(v["camera"], float)[:2, 3]
        push = SIGHT_PUSH * seen["radius"] if seen.get("top_seen", 0.0) < TOP_SEEN else 0.0
        marks.append(centre + push * sight / np.linalg.norm(sight))
    got = [m for m in marks if m is not None]
    if not got:
        raise ValueError("no view read the thing whole")
    point, shift = np.median(got, axis=0), np.zeros(2)
    for v, mark in zip(views, marks, strict=True):
        if mark is not None:
            shift = point - mark
        v["camera"] = np.array(v["camera"], float)
        v["camera"][:2, 3] += shift
    seen = [v["seen"] for v in views if v.get("seen") is not None]
    return point, float(np.median([s["radius"] for s in seen])), float(np.median([s["top"] for s in seen]))


def fuse(views: list[dict], centre, radius: float, top: float, voxel: float = 0.0015):
    """One coloured model from the views (each {"rgb" (H x W x 3, RGB), "depth" (mm), "K", "camera"
    (world <- optical, 4 x 4)}): the depth cut to a box round the thing (its floor ring kept, it
    holds the views together), each view's camera corrected by coloured ICP onto the views before
    it (Park 2017, coarse to fine; a correction past 3 cm or 8 deg is not believed), then a TSDF at
    `voxel` m. Returns (points n x 3, colours n x 3 in 0..1, the mesh, the cameras as corrected)."""
    import open3d as o3d  # the workspace's laptop group: not on the robot yet

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
