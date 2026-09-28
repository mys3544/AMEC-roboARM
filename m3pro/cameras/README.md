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
| container device | `ROBOARM_WRIST_CAM_DEV` in compose | `usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera-video-index0` | `usb-046d_Logitech_Webcam_C930e_3E3B7A2E-video-index0` |

The C930e is the default. The Sonix was fitted until 2026-09-24; every
measurement in this project before that date was made through it.

## Going back to the Sonix

1. Plug the Sonix back onto the wrist, in the old position: 65 mm out from J4,
   50 mm to the right of the forearm. Its calibration is only valid for that
   mounting.
2. On the robot, in `~/roboarm`:

       cp cameras/sonix.env .env
       docker compose --profile bridge up -d --force-recreate bridge
       docker compose --profile vision up -d --force-recreate vision

   `.env` sets `ROBOARM_CAMERA=sonix` and points the containers' `/dev/video0`
   at the Sonix; the vision container picks its 4:3 depth engine from it. Every
   `docker compose run ... core` picks it up too.
3. On the laptop, set `ROBOARM_CAMERA=sonix` before starting the panel
   (`tools/webapp.py --robot ...`): detection and grasp planning run there and
   need the Sonix's lens position and reach offset.

Its calibration (three looks: primary, outer, near) is in `data/cameras/sonix/`
on the robot, and a copy is kept here in `cameras/sonix/table_homography.json`
in case the robot's is lost: copy it to `~/roboarm/data/cameras/sonix/`.

To return to the C930e: delete `.env` on the robot and recreate the bridge and
vision; unset `ROBOARM_CAMERA` on the laptop.

A device that is not plugged in stops every container from starting
("error gathering device information"), so `.env` must match the camera that
is actually on the wrist.

## Recalibrating a camera

Board uncovered (paper off, board not moved), bridge stopped:

    docker compose --profile bridge stop bridge
    for look in primary outer near; do
      docker compose run --rm --no-deps core python tools/calibrate_table.py --look $look
    done
    docker compose --profile bridge up -d bridge

One yaw per look: the C930e sees 5-9 markers in one frame, and pooling yaws
(`--yaws`) adds J1's own error (fits went from ~1 mm to ~11 mm when pooled).
`BOARD_X0 / BOARD_Y0` in `roboarm/config.py` must say where the board is; it
had slid 11 mm by 2026-09-24 (see there for how that was found).
