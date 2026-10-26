"""How far the simulator's M3 is from the real one where it matters, the camera over the table (NEXT.md, "This
week"): the arm, through the bridge, to POSES round the ChArUco board lying still in the pick zone; at each
the board found in the picture (its pose in the camera) and the camera put where the robot's model -- the
simulator's twin, command + SAG, the URDF's DCW2 mount -- says it is. Were the model right, every pose would put
the board in the same place on the table; how far apart they put it is the twin's error: in mm on the table
and in degrees, pose by pose, and the reprojection of the board's corners in pixels.

    python tools/m3_twin_gap.py [--speed 20]     # the board as for m3_servo_probe.py: flat, in the pick zone,
                                                 # seen whole from its LOOK; no job running

Each run -> out/twin/<stamp>.json, every reading in it.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import m3_handeye as he

from roboarm import m3kinematics

OUT = ROOT / "out" / "twin"
SETTLE_S = 1.0
MIN_CORNERS = 8
# round m3_servo_probe's LOOK (88, 115, 0, 0, 90), each joint the way a pick's poses turn it; the LOOK again
# at the end, for the repeatability beside the spread
POSES = [{1: j1, 2: j2, 3: j3, 4: j4, 5: 90}
         for j1 in (78, 88, 98) for j2, j3, j4 in ((115, 0, 0), (108, 0, 10), (122, 12, 0), (115, 12, 10), (108, 12, 0))] \
    + [{1: 88, 2: 115, 3: 0, 4: 0, 5: 90}]


def board_in_camera(snap: dict):
    """(4 x 4 board -> camera, pixels, board points) from an /rgbd snapshot, or None with too few corners."""
    image = cv2.imdecode(np.frombuffer(base64.b64decode(snap["color_jpeg"]), np.uint8), cv2.IMREAD_COLOR)
    pixels, points = he.corners(image)
    if len(pixels) < MIN_CORNERS:
        return None
    _ok, rvec, tvec = cv2.solvePnP(points, pixels, np.asarray(snap["K"], float), None)
    m = np.eye(4)
    m[:3, :3] = cv2.Rodrigues(rvec)[0]
    m[:3, 3] = tvec.ravel()
    return m, pixels, points


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--speed", type=float, default=20.0)
    a = ap.parse_args()
    if "error" in he.http("/arm/call", {"method": "read"}):
        print("engage:", he.http("/arm/call", {"method": "engage"}).get("result"), flush=True)
    readings = []
    for pose in POSES:
        reply = he.http("/arm/call", {"method": "move_to", "args": [{str(j): v for j, v in pose.items()}],
                                      "kwargs": {"speed_dps": a.speed}})
        if "error" in reply:
            print(f"  {pose}: refused ({reply['error']})", flush=True)
            continue
        time.sleep(SETTLE_S)
        snap = he.http("/rgbd")
        seen = board_in_camera(snap)
        if seen is None:
            print(f"  {pose}: the board not seen", flush=True)
            continue
        bc, pixels, points = seen
        model = m3kinematics.frames(pose)["arm4"] @ m3kinematics.camera_optical()   # base <- camera, as the twin
        readings.append({"pose": pose, "K": snap["K"], "board_in_camera": bc, "camera": model,
                         "board": model @ bc, "pixels": pixels, "points": points})
        print(f"  {pose}: {len(pixels)} corners", flush=True)
    if len(readings) < 4:
        raise SystemExit("too few poses saw the board: put it where the look pose sees it whole")
    # the board's place on the table by every pose; its mean; each pose's error against it
    places = np.array([r["board"][:3, 3] for r in readings])
    mean_t = places.mean(axis=0)
    u, _s, vt = np.linalg.svd(sum(r["board"][:3, :3] for r in readings))
    mean_r = u @ vt
    mean = np.eye(4)
    mean[:3, :3], mean[:3, 3] = mean_r, mean_t
    rows = []
    for r in readings:
        off = r["board"][:3, 3] - mean_t
        turn = np.degrees(np.linalg.norm(cv2.Rodrigues(mean_r.T @ r["board"][:3, :3])[0]))
        # the board's corners where the twin's camera would see the mean board, against where they were seen
        cam = np.linalg.inv(r["camera"]) @ mean
        q = cam[:3, :3] @ r["points"].T + cam[:3, 3:4]
        px = (np.asarray(r["K"], float) @ (q / q[2]))[:2].T
        rows.append({"pose": r["pose"], "table_mm": np.round(1000 * off[:2], 1).tolist(), "height_mm": round(1000 * off[2], 1),
                     "turn_deg": round(float(turn), 2), "reproj_px": round(float(np.sqrt(np.mean(np.sum((px - r["pixels"]) ** 2, 1)))), 1)})
    flat = np.array([np.hypot(*row["table_mm"]) for row in rows])
    print(f"the board by {len(rows)} poses: on the table {np.median(flat):.1f} mm median, {flat.max():.1f} max "
          f"(rms {np.sqrt(np.mean(flat ** 2)):.1f}); turn {np.median([row['turn_deg'] for row in rows]):.2f} deg median; "
          f"reprojected {np.median([row['reproj_px'] for row in rows]):.1f} px median", flush=True)
    for row in rows:
        print(f"  {row['pose']}: {row['table_mm']} mm, {row['height_mm']:+.1f} mm up, {row['turn_deg']:.2f} deg, "
              f"{row['reproj_px']:.1f} px", flush=True)
    first, last = rows[0], rows[-1]   # the LOOK at the start (in the J1 88 block) and the end: repeatability
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"rows": rows, "board_mean": mean.tolist(), "first": first, "last": last,
                               "readings": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
                                            for r in readings]}, indent=1))
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
