"""The object scan from the laptop: start it on the robot (the bridge's job, roboarm/m3scan.py --
the M3 drives round the thing ahead, the camera on it), fetch its views, and fuse them into one
coloured model that the console's 3D view shows.

    python tools/m3_scan.py start [--radius 0.38] [--stops 12] [--height H --width W]   # on the robot;
                                                # follows its log
    python tools/m3_scan.py fetch [STAMP]       # the robot's out/scan/STAMP (the newest) -> out/scan/
    python tools/m3_scan.py fuse [DIR]          # out/scan/DIR (the newest) -> its cloud.ply + mesh.ply,
                                                # and the cloud as object_STAMP.ply among the console's scans
    python tools/m3_scan.py fuse [DIR] --export # and DIR/splat: a Gaussian splat's input (train/splat/)

Put the thing on the floor with its middle `radius` m (0.38: ~26 cm from the bumper) straight ahead
of the robot, nothing else within ~0.7 m of it: the robot goes round it, its back ~0.55 m out.
Up to ~15 cm tall at 0.38; taller, --radius 0.42. The arm keeps clear of it by the size the first
look reads; a thing that runs off that look's picture (a tall can's top) is refused until its
--height and --width (m) are given. fuse needs Open3D (the workspace's laptop group: `uv sync` at the top).
The SSH password comes from ROBOARM_SSH_PASSWORD, or is asked for.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import m3_handeye as he

from roboarm import m3kinematics, m3scan

OUT = ROOT / "out" / "scan"
CLOUDS = ROOT / "out" / "arena" / "rtab"   # the console's 3D view lists the PLYs here (console/serve.py)
HOST, REMOTE = "192.168.73.66", "/home/jetson/roboarm/out/scan"


def start(radius: float, stops: int, height: float | None = None, width: float | None = None) -> None:
    # the log's lines are numbered (n) across jobs, the last 300 kept: start after the ones there now
    since = max((line["n"] for line in he.http("/pick/log?since=0")["lines"]), default=0)
    reply = he.http("/pick/scan", {"radius": radius, "stops": stops, "height": height, "width": width})
    if "error" in reply:
        raise SystemExit(f"not started: {reply['error']}")
    began = time.time()
    while True:
        lines = he.http(f"/pick/log?since={since}")["lines"]
        for line in lines:
            print(f"{line['t'] - began:6.1f} s  {line['text']}", flush=True)
        since = max((line["n"] for line in lines), default=since)
        state = he.http("/pick/state")
        if not state["busy"]:
            print("error:" if state["error"] else "done.", state["error"] or "")
            return
        time.sleep(1.0)


def fetch(stamp: str | None) -> Path:
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username="jetson", look_for_keys=False, timeout=15,
                   password=os.environ.get("ROBOARM_SSH_PASSWORD") or getpass.getpass(f"jetson@{HOST} password: "))
    sftp = client.open_sftp()
    stamp = stamp or max(sftp.listdir(REMOTE))
    here = OUT / stamp
    here.mkdir(parents=True, exist_ok=True)
    names = sftp.listdir(f"{REMOTE}/{stamp}")
    for name in names:
        sftp.get(f"{REMOTE}/{stamp}/{name}", str(here / name))
    client.close()
    print(f"{len(names)} files -> {here}")
    return here


def fuse(folder: Path, export: bool = False) -> None:
    import open3d as o3d

    plan = json.loads((folder / "scan.json").read_text())
    centres = plan.get("centres") or []   # each stop's own reading (scans before 2026-10-12: none)
    views = []
    for meta in sorted(folder.glob("[0-9][0-9]_*.json")):
        v = json.loads(meta.read_text())
        rgb = cv2.cvtColor(cv2.imread(str(meta.with_name(meta.stem + "_rgb.jpg"))), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(meta.with_name(meta.stem + "_depth.png")), cv2.IMREAD_UNCHANGED)
        pose = {int(j): a for j, a in v["pose"].items()}
        K = np.asarray(v["K"], float)
        levelled = m3scan.levelled(depth, K, pose)
        near = centres[v["stop"]] if v["stop"] < len(centres) else plan["centre"]   # where the stop had it
        seen = m3scan.find_thing(depth, K, levelled, m3scan.to_local(v["odom"], near))
        views.append({"rgb": rgb, "depth": depth, "K": K, "camera": m3scan.world_of(v["odom"]) @ levelled,
                      "seen": {**seen, "centre": m3scan.to_world(v["odom"], seen["centre"])}
                      if seen is not None and seen["whole"] else None})
    centre, radius, top = m3scan.anchor(views)
    print(f"the thing: {100 * top:.1f} cm tall, {200 * radius:.1f} cm across; "
          f"{sum(v['seen'] is not None for v in views)} of {len(views)} views read it whole")
    began = time.time()
    xyz, rgb, mesh, cameras = m3scan.fuse(views, centre, radius, top)
    print(f"{len(views)} views -> {len(xyz)} points, {len(mesh.triangles)} triangles in {time.time() - began:.0f} s")
    colours = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
    write_ply(folder / "cloud.ply", xyz, colours)
    o3d.io.write_triangle_mesh(str(folder / "mesh.ply"), mesh)
    CLOUDS.mkdir(parents=True, exist_ok=True)
    write_ply(CLOUDS / f"object_{folder.name}.ply", xyz, colours)
    print(f"-> {folder / 'cloud.ply'}, {folder / 'mesh.ply'}, and the console's 3D view (object_{folder.name}.ply)")
    if export:
        export_splat(folder / "splat", views, cameras, centre, radius, top)


SPLAT_REACH = 0.30   # m round the thing whose floor the splat's first points keep


def export_splat(out: Path, views: list[dict], cameras: list[np.ndarray], centre, radius: float, top: float) -> None:
    """The scan as a Gaussian splat's input (train/splat/train_splat.py, on hekate): every view's picture
    and its depth (mm, the depth camera's), its intrinsics and its camera as the fuse corrected it (world
    <- camera, OpenCV axes: x right, y down, z ahead) in transforms.json, and points.ply -- every view's
    depth within SPLAT_REACH of the thing and under its top, in those cameras, 3 mm apart: where the
    Gaussians start."""
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "depths").mkdir(parents=True, exist_ok=True)
    frames, points, colours = [], [], []
    for i, (v, cam) in enumerate(zip(views, cameras, strict=True)):
        name, depth = f"images/{i:03d}.jpg", f"depths/{i:03d}.png"
        cv2.imwrite(str(out / name), cv2.cvtColor(v["rgb"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / depth), v["depth"])   # the DCW2's, aligned to the colour: mm, 0 where none
        frames.append({"file": name, "depth": depth, "K": np.asarray(v["K"]).tolist(), "c2w": np.asarray(cam).tolist()})
        K, z = np.asarray(v["K"], float), v["depth"].astype(np.float64) / 1000
        rows, cols = z.shape
        vv, uu = np.mgrid[0:rows:2, 0:cols:2]
        zz = z[vv, uu]
        ok = zz > 0.15
        p = np.stack([(uu[ok] - K[0, 2]) * zz[ok] / K[0, 0], (vv[ok] - K[1, 2]) * zz[ok] / K[1, 1], zz[ok]], axis=1)
        world = p @ cam[:3, :3].T + cam[:3, 3]
        keep = (np.hypot(world[:, 0] - centre[0], world[:, 1] - centre[1]) < SPLAT_REACH) \
            & (world[:, 2] > m3kinematics.FLOOR - 0.01) & (world[:, 2] < top + 0.03)
        points.append(world[keep])
        colours.append(v["rgb"][vv[ok][keep], uu[ok][keep]])
    xyz, rgb = np.vstack(points), np.vstack(colours)
    cell = np.floor(xyz / 0.003).astype(np.int64)
    _, first = np.unique(cell, axis=0, return_index=True)
    write_ply(out / "points.ply", xyz[first], rgb[first])
    (out / "transforms.json").write_text(json.dumps({
        "width": int(views[0]["rgb"].shape[1]), "height": int(views[0]["rgb"].shape[0]), "frames": frames,
        "centre": [float(c) for c in centre], "radius": float(radius), "top": float(top)}, indent=1))
    print(f"-> {out}: {len(frames)} views, {len(first)} starting points")



def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """The one PLY the console's viewer reads (console/cloud.js): binary, float x y z, uchar r g b."""
    data = np.zeros(len(xyz), [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = np.asarray(xyz, float).T
    data["red"], data["green"], data["blue"] = np.asarray(rgb).T
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(data)}\n"
                 "property float x\nproperty float y\nproperty float z\nproperty uchar red\n"
                 "property uchar green\nproperty uchar blue\nend_header\n").encode())
        f.write(data.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--radius", type=float, default=m3scan.STOP_R)
    s.add_argument("--stops", type=int, default=m3scan.STOPS)
    s.add_argument("--height", type=float, help="m, the thing's height: needed when it runs off the first look")
    s.add_argument("--width", type=float, help="m, the widest way across it (with --height)")
    sub.add_parser("fetch").add_argument("stamp", nargs="?")
    f = sub.add_parser("fuse")
    f.add_argument("dir", nargs="?")
    f.add_argument("--export", action="store_true", help="and its views as a Gaussian splat's input (DIR/splat)")
    args = parser.parse_args()
    if args.cmd == "start":
        if (args.height is None) != (args.width is None):
            parser.error("--height and --width go together")
        start(args.radius, args.stops, args.height, args.width)
    elif args.cmd == "fetch":
        fetch(args.stamp)
    else:
        fuse(OUT / args.dir if args.dir else max(OUT.iterdir()), export=args.export)
    return 0


if __name__ == "__main__":
    sys.exit(main())
