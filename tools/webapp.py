#!/usr/bin/env python3
"""Serve the browser control panel: live camera, detector views, manual and auto.

On the robot (the `web` service in compose.yaml does exactly this):

    docker compose --profile web up web          # then open http://<robot-ip>:8080/

or by hand inside the core container:

    docker compose run --rm --no-deps --service-ports core python tools/webapp.py

On a laptop, driving the REAL robot over WiFi -- the page and all the judgement
run here, only servo commands and camera frames cross the network:

    (on the robot)  docker compose --profile bridge up -d bridge
    (here)          python tools/webapp.py --robot http://192.168.73.210:8761

Anywhere else, with no robot at all:

    python tools/webapp.py --sim                 # a pretend arm and camera

The simulator renders the wrist camera's view from the real calibrated homography,
so sweeps find the coloured blocks where they were put and a pick lifts one. It is
the way to try the page, and the way tests/test_web.py exercises the server.

The page talks to roboarm.web.server; what it can do lives in roboarm.web.session.
Everything that moves the arm goes through roboarm.arm, so every safety rule that
holds for the command-line tools holds here too. The one extra: the page's STOP
button interrupts a move already under way, via Arm.interrupt.
"""

import argparse
import sys
import tempfile
from pathlib import Path

# Runnable from a plain checkout (`python tools/webapp.py --sim`), not only inside
# the container, where PYTHONPATH=/app is set for us.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roboarm import camera, detect
from roboarm import config as cfg
from roboarm import workspace as ws
from roboarm.web import server, session


def real_session(out_dir: Path | None) -> session.Session:
    from roboarm.arm import Arm

    return session.Session(
        arm_factory=Arm,
        streams={
            "wrist": lambda: camera.Stream(cfg.WRIST_CAM),
            "mast": lambda: camera.Stream(cfg.TABLE_CAM, cfg.TABLE_CAM_SIZE),
        },
        calibration=ws.load_all,
        out_dir=out_dir,
    )


def remote_session(url: str) -> session.Session:
    from roboarm.web import remote

    # Detection runs HERE, so the empty-table photos and sweep pictures live here.
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    detect.BACKGROUND_PATH = data_dir / "table_background.png"
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    return session.Session(
        arm_factory=lambda: remote.RemoteArm(url),
        streams={
            "wrist": lambda: remote.RemoteStream(url, "wrist"),
            "mast": lambda: remote.RemoteStream(url, "mast"),
        },
        calibration=lambda: remote.calibration(url),
        out_dir=out_dir,
        # The neural rung lives in the robot's compose network; the bridge
        # forwards it, so from here it is reached through the bridge.
        detector_url=url.rstrip("/") + "/vision",
    )


def sim_session(time_scale: float) -> session.Session:
    from roboarm.web import sim

    # Backgrounds and sweep pictures go somewhere harmless rather than /app/data.
    scratch = Path(tempfile.mkdtemp(prefix="roboarm-sim-"))
    detect.BACKGROUND_PATH = scratch / "table_background.png"
    arm = sim.SimArm(time_scale=time_scale)
    return session.Session(
        arm_factory=lambda: arm,
        streams={"wrist": lambda: sim.SimStream(arm)},
        calibration=sim.calibration,
        out_dir=scratch,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--sim", action="store_true",
                        help="pretend arm and camera; no hardware needed")
    parser.add_argument("--robot", metavar="URL", default=None,
                        help="drive a robot running tools/hwbridge.py, e.g. "
                             "http://192.168.73.210:8761")
    parser.add_argument("--sim-speed", type=float, default=1.0,
                        help="with --sim: 1.0 moves as slowly as the robot, 0 is instant")
    parser.add_argument("--out", default="/out",
                        help="where sweep pictures are written (default /out; '' for none)")
    args = parser.parse_args()

    if args.sim:
        sess = sim_session(args.sim_speed)
    elif args.robot:
        sess = remote_session(args.robot.rstrip("/"))
    else:
        out_dir = Path(args.out) if args.out else None
        if out_dir is not None and not out_dir.is_dir():
            print(f"note: {out_dir} does not exist; sweep pictures will not be written",
                  file=sys.stderr)
            out_dir = None
        sess = real_session(out_dir)

    session.LogTee.install(sess)
    sess.start()
    if sess.arm is None:
        print(f"arm not connected: {sess.arm_error}\n"
              f"(the page still works for the camera; use its Connect button to retry)",
              file=sys.stderr)
    server.serve(sess, args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
