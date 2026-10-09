"""The object scan from the laptop: start it on the robot (the bridge's job, roboarm/m3scan.py --
the M3 drives round the thing ahead, the camera on it), fetch its views, and fuse them into one
coloured model that the console's 3D view shows.

    python tools/m3_scan.py start [--radius 0.38] [--stops 12]   # on the robot; follows its log
    python tools/m3_scan.py fetch [STAMP]       # the robot's out/scan/STAMP (the newest) -> out/scan/
    python tools/m3_scan.py fuse [DIR]          # out/scan/DIR (the newest) -> its cloud.ply + mesh.ply,
                                                # and the cloud as object_STAMP.ply among the console's scans

Put the thing on the floor ~38 cm straight ahead of the robot with ~1.2 m clear round it: the
robot goes round it. fuse needs Open3D (the workspace's laptop group: `uv sync` at the top).
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

from roboarm import m3scan

OUT = ROOT / "out" / "scan"
CLOUDS = ROOT / "out" / "arena" / "rtab"   # the console's 3D view lists the PLYs here (console/serve.py)
HOST, REMOTE = "192.168.73.66", "/home/jetson/roboarm/out/scan"


def start(radius: float, stops: int) -> None:
    reply = he.http("/pick/scan", {"radius": radius, "stops": stops})
    if "error" in reply:
        raise SystemExit(f"not started: {reply['error']}")
    since, began = 0, time.time()
    while True:
        lines = he.http(f"/pick/log?since={since}")["lines"]
        for line in lines:
            print(f"{line['t'] - began:6.1f} s  {line['text']}", flush=True)
        since += len(lines)
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


def fuse(folder: Path) -> None:
    import open3d as o3d

    plan = json.loads((folder / "scan.json").read_text())
    views = []
    for meta in sorted(folder.glob("[0-9][0-9]_*.json")):
        v = json.loads(meta.read_text())
        rgb = cv2.cvtColor(cv2.imread(str(meta.with_name(meta.stem + "_rgb.jpg"))), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(meta.with_name(meta.stem + "_depth.png")), cv2.IMREAD_UNCHANGED)
        pose = {int(j): a for j, a in v["pose"].items()}
        K = np.asarray(v["K"], float)
        cam = m3scan.world_of(v["odom"]) @ m3scan.levelled(depth, K, pose)
        views.append({"rgb": rgb, "depth": depth, "K": K, "camera": cam})
    began = time.time()
    xyz, rgb, mesh, _cameras = m3scan.fuse(views, plan["centre"], plan["radius"], plan["top"])
    print(f"{len(views)} views -> {len(xyz)} points, {len(mesh.triangles)} triangles in {time.time() - began:.0f} s")
    colours = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
    write_ply(folder / "cloud.ply", xyz, colours)
    o3d.io.write_triangle_mesh(str(folder / "mesh.ply"), mesh)
    CLOUDS.mkdir(parents=True, exist_ok=True)
    write_ply(CLOUDS / f"object_{folder.name}.ply", xyz, colours)
    print(f"-> {folder / 'cloud.ply'}, {folder / 'mesh.ply'}, and the console's 3D view (object_{folder.name}.ply)")


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
    sub.add_parser("fetch").add_argument("stamp", nargs="?")
    sub.add_parser("fuse").add_argument("dir", nargs="?")
    args = parser.parse_args()
    if args.cmd == "start":
        start(args.radius, args.stops)
    elif args.cmd == "fetch":
        fetch(args.stamp)
    else:
        fuse(OUT / args.dir if args.dir else max(OUT.iterdir()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
