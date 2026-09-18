#!/usr/bin/env python3
"""Expose this robot's arm and wrist camera over HTTP, for a control panel elsewhere.

On the robot (the `bridge` compose service does exactly this):

    docker compose --profile bridge up -d bridge     # port 8761

Then on a laptop on the same WiFi:

    python tools/webapp.py --robot http://192.168.73.210:8761

The page, the detectors and every decision run on the laptop; only servo commands
and camera frames cross the network. See roboarm/web/bridge.py for the contract.

    python tools/hwbridge.py --sim      # a pretend robot, to try the remote path
                                        # without hardware
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roboarm import camera
from roboarm import config as cfg
from roboarm import workspace as ws
from roboarm.web import bridge


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8761)
    parser.add_argument("--sim", action="store_true",
                        help="pretend arm and camera, for trying the remote path")
    args = parser.parse_args()

    if args.sim:
        import json
        import tempfile

        from roboarm.web import sim

        arm = sim.SimArm()
        matrix, pose, name = sim.calibration()[0]
        path = Path(tempfile.mkdtemp(prefix="roboarm-bridge-sim-")) / "table_homography.json"
        path.write_text(json.dumps({"homography": matrix.tolist(), "name": name,
                                    "survey_pose": {str(j): a for j, a in pose.items()}}))
        hw = bridge.Bridge(arm_factory=lambda: arm,
                           stream=lambda: sim.SimStream(arm),
                           calibration_path=path)
    else:
        from roboarm.arm import Arm

        hw = bridge.Bridge(
            arm_factory=Arm,
            stream=lambda: camera.Stream(cfg.WRIST_CAM),
            calibration_path=ws.CALIBRATION_PATH,
        )
    bridge.serve(hw, args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
