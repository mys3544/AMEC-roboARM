# Wrist cameras

Two cameras have been on arm_link4. `ROBOARM_CAMERA` says which one is fitted,
and everything that depends on the camera is keyed by it:

| what | where | `sonix` | `c930e` |
|---|---|---|---|
| picture | `cfg.WRIST_CAM_SIZE`, `WRIST_CAM_PROPS` | 640x480 YUYV, ~7 fps | 848x480 MJPG, ~30 fps, focus fixed at 45 |
| lens distortion | `cfg.WRIST_CAM_LENS` | not corrected | undone on capture (`camera.Undistort`) |
| lens position | `cfg._LENS` | 65 mm from J4, 50 mm right | 85 mm from J4, 39 mm across (forward when looking down) |
| reach offset | `cfg.REACH_OFFSET_M` | 5 mm | 0 |
| depth engine | `vision_service/depth.py` | `da2s_518x686` (4:3) | `da2s_518x910` (16:9) |
| calibration, empty-table photos | `data/cameras/<name>/` on the robot | fitted 2026-09-08..22 | fitted 2026-09-24 |
| device | `ROBOARM_WRIST_CAM` (robot/robot.env) | `usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera-video-index0` | `usb-046d_Logitech_Webcam_C930e_3E3B7A2E-video-index0` |

The C930e is the default. The Sonix was fitted until 2026-09-24; every
measurement in this project before that date was made through it.

## Going back to the Sonix

1. Plug the Sonix back onto the wrist, in the old position: 65 mm out from J4,
   50 mm to the right of the forearm. Its calibration is only valid for that
   mounting.
2. On the robot, in `~/roboarm`:

       cp cameras/sonix.env robot.local.env
       systemctl --user restart roboarm-vision roboarm-web

   `robot.local.env` overrides robot/robot.env for both services: it sets
   `ROBOARM_CAMERA=sonix` and `ROBOARM_WRIST_CAM` to the Sonix's device; the
   vision service picks its 4:3 depth engine from it. A tool run by hand needs
   the same two variables (`set -a; . robot/robot.env; . robot.local.env; set +a`).
3. On the laptop, set `ROBOARM_CAMERA=sonix` before starting the panel
   (`tools/webapp.py --robot ...`): detection and grasp planning run there and
   need the Sonix's lens position and reach offset.

Its calibration (three looks: primary, outer, near) is in `data/cameras/sonix/`
on the robot, and a copy is kept here in `cameras/sonix/table_homography.json`
in case the robot's is lost: copy it to `~/roboarm/data/cameras/sonix/`.

To return to the C930e: delete `robot.local.env` on the robot and restart both
services; unset `ROBOARM_CAMERA` on the laptop. A camera that is not plugged in
leaves the panel without a picture (it says so), so `robot.local.env` must match
the camera that is actually on the wrist.

## Recalibrating a camera

Board uncovered (paper off, board not moved), the arm service stopped (it holds
the camera and the serial port):

    systemctl --user stop roboarm-web
    set -a; . robot/robot.env; set +a
    for look in primary outer near; do
      .venv/bin/python tools/calibrate_table.py --look $look
    done
    systemctl --user start roboarm-web

One yaw per look: the C930e sees 5-9 markers in one frame, and pooling yaws
(`--yaws`) adds J1's own error (fits went from ~1 mm to ~11 mm when pooled).
`BOARD_X0 / BOARD_Y0` in `roboarm/config.py` must say where the board is; it
had slid 11 mm by 2026-09-24 (see there for how that was found).
