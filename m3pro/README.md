# Yahboom ROSMASTER M3 Pro

The lab's second robot. It has the same Jetson board as the X3 Plus but different hardware
around it, so its software lives in this folder and is changed here. It started on 2026-09-28
as a copy of the X3 Plus software (below); `roboarm/` and `tools/` at the top of the repo stay
the X3 Plus's, and their `sync_robot.py` pushes nothing from here.

## The robot

- **Board:** Jetson Orin NX Engineering Reference Developer Kit Super, the X3 Plus's carrier.
  Ubuntu 22.04.5, L4T R36.4.4 (JetPack 6), GNOME 42 on X11, `jetson` logs in automatically.
- **Network:** IISLab-AMEC at 192.168.73.66 (DHCP; the same address over two reboots so
  far), with internet. `ssh jetson@192.168.73.66` with the factory password; sudo asks for it.
- **Hostname** `yahboom`, the same as the X3 Plus: use the address, not `yahboom.local`.
- **Its own software runs in the desktop session.** `~/.config/autostart` starts the micro-ROS
  agent on `/dev/myserial` at 2 Mbaud (the base controller; the X3 Plus talks to its board
  through Rosmaster_Lib on `/dev/ttyUSB0` instead), the joystick controller
  (`ros2 launch yahboomcar_ctrl yahboomcar_joy_launch.py`) and a sensor self-check, on ROS 2
  Humble with `ROS_DOMAIN_ID=30` and the workspace `~/yahboomcar_ws`. Restarting the desktop
  (gdm3) or logging out stops them until the session is back; a reboot brings them all up.

## The software

It began as the X3 Plus's, byte for byte as of commit 1f94700; what only the X3 used -- its
serial arm, wrist camera, table calibration, web panel and their tools -- went on 2026-10-05
(in git). What is here:

- **The arm and the pick:** `tools/m3bridge.py`, started on the robot by `tools/m3bridge.sh`,
  serves the arm (`roboarm/m3arm.py`, `m3kinematics.py` with `m3body.npz`) and the depth camera
  over HTTP on port 8761 (`roboarm/web/bridge.py`), and runs the pick there (`roboarm/m3pick.py`,
  `m3vision.py`, timed by `metrics.py`) for the console's M3 arm card.
- **Measuring the arm:** `tools/m3_touch.py` (with `m3_handeye.py`) and `tools/m3_body_model.py`.
- **The arena:** `arena/` and `tools/arena_map.py` (below).
- **The vision container:** `vision_service/`, `docker/` and `compose.yaml`, the X3's neural
  detector, kept.

Work from this folder; it has its own `pyproject.toml` and `uv.lock`, so its own `.venv`:

```bash
cd m3pro
uv sync --frozen
uv run --frozen python -m pytest -q     # `python -m` so that roboarm/ here is imported
uv run --frozen ruff check roboarm tools tests vision_service
python tools/sync_robot.py              # system Python (paramiko): push to ~/roboarm on the M3
```

## Mapping the arena

`arena/` runs on the robot's host ROS 2 (not in a container), from `~/arena`: slam_toolbox at
2 cm on each lidar's own scan, their range biases taken off (`scan_relay.py`, `lidars.yaml`),
with Yahboom's EKF at 30 Hz (`arena_slam.launch.py`), and a drive page on port 8095
(`drive_server.py`). `tools/arena_map.py` drives it from this laptop (system Python, paramiko):

```bash
python tools/arena_map.py start        # upload arena/ and start the stack; nothing moves yet
python tools/arena_map.py snap         # the map so far, with the fence fit -> out/arena/
python tools/arena_map.py save NAME    # -> arena/maps/NAME*, and the pose graph on the robot
python tools/arena_map.py stop
```

Tuning, on a recording instead of the robot's battery:

```bash
python tools/arena_map.py record NAME  # 20 s standing, the auto loop, 10 s standing -> a bag,
                                       # printing the live map's fence every 10 s
python tools/arena_map.py tune NAME [CONFIG ...]   # replay it through settings side by side
```

`tune` runs `replay.sh` once per configuration, each on its own ROS domain (so neither each
other nor the live robot interfere), fits the fence every 20 s of the replay and prints a table
and the fence over time. A configuration is a name from `CONFIGS` in the tool, or
`LABEL=arg:=value,arg:=value` with `arena_slam.launch.py`'s arguments (`scans`, `ekf_hz`,
`ekf_predict`, `lidars`, `slam`, `slam_set`).

- **Driving:** http://192.168.73.66:8095/ -- hold the buttons or W/S (forward, back), A/D
  (sideways), Q/E (turn); letting go, Space or closing the tab stops it. "Auto: loop the arena"
  drives a slow loop 45 cm inside the fitted fence, around the landmarks. Both are held to
  0.15 m/s; the loop stops 35 cm short of anything the lidars see in its path, driving by hand
  has no distance limit (whoever drives judges). The gamepad works whenever the page is not
  commanding.
- **Bench mode:** `python tools/arena_map.py start motion:=false` -- everything is shown, every
  command to move is refused (the page and the console grey their controls). Use it whenever
  the robot is on a table: the lidars see obstacles, not the table's edge. The gamepad still
  drives.
- **The fence fit** (`fence_fit.py`) checks every map against the fence's 3.2 x 2.8 m: side
  lengths, corner angles, how straight and how much of each wall is seen, and the blobs inside
  (the landmarks) in an arena frame with its origin at a fence corner.
- **What the robot gives:** the lidars come from the micro-ROS board as `/scan0` and `/scan1`,
  270 degrees each at about 7 Hz, 14.5 cm above the floor; Yahboom publishes their static
  transforms only in `m3pro_bringup/static_tf.launch.py` (the launch publishes them from
  `lidars.yaml`). The board stamps everything 55-75 ms ahead of the Jetson's clock, so the
  EKF trails the scans and slam_toolbox drops most of them -- harmless (it keeps a scan every
  10 cm of travel anyway); keeping them (`scan_queue_size` 10) wrecked the map.
- **First maps, 2026-09-30** (`arena_v0`, `arena_v1`; in git): 3.213 x 2.801 m after half a
  metre of driving, 3.232 x 2.823 m after a full loop. Most of that was the lidars reading long
  (0.8 and 1.2 cm, measured by `lidar_calib.py`, in git too; the biases are in `lidars.yaml`);
  with the biases taken off, a replay of the same drive stays within 1 cm of the measured
  3.2 x 2.8 m all the way (3.198 x 2.799 m at the end).
- **The 3D scan (RTAB-Map):** `arena_map.py arm mapping` (the arm to Yahboom's mapping pose; it
  has no read-back), `start mapper:=rtabmap` (the DaBai with depth aligned to colour, the lidar
  aligning key frames), `camcheck` (the camera's height and tilt measured from the floor,
  against the launch's), drive, then `rtab-export`: the database stitched again visually (best
  of three, by the fence walls' thickness in the cloud), the coloured cloud from depth within
  1.5 m, noise-filtered and cropped to the arena (PLY), the camera poses, the floor print from
  above and a height view in `out/arena/rtab/`. Stop RTAB-Map with SIGINT and let it save its
  database before exporting. First scan, 2026-09-30: the lidar-stitched export put the fence's
  walls 14-16 cm thick and blurred the print; re-stitched visually, 9-11 cm, the print's lines
  single, the fence 3.215 x 2.817 m. `arena_v4` stays the reference map. The crop
  (`ortho.py`) keeps 5 cm either side of each wall for the walls' thickness, but nothing there
  above the fence (35 cm): that is the room seen over it -- in the first scan, something up to
  1.1 m tall standing against one wall, 6.9k points. `arena_map.py rtab-crop [N]` crops the
  kept export (or try N) again and fetches it, without stitching again.
- **The survey, for a better 3D scan** ("Auto: survey for a 3D scan" on the drive page and in
  the console): the camera tilted down (`arena_map.py arm survey`, `[90, 120, 30, 0, 90, 0]`:
  34.9 deg down, 0.358 m up, measured; in the mapping pose half the view is the room over the
  fence). `drive_core.survey()` picks the views -- from every place 45 cm clear of the walls and
  landmarks (nothing nearer can trip the lidar stop), 8 headings each -- by what the camera
  would see of the floor and the landmarks' faces, until the floor is seen, most of it from two
  sides; then a tour through them. The robot stands still 1.8 s at each station: the frames
  taken then are sharp -- frames taken moving carry the board clock's lag. On `arena_v4`: ~70
  stations, ~12 m, 98.6 % of the floor in view (the first scan: the camera never closer than
  0.66 m). Steps: charge, cable out, `arm survey`, `start mapper:=rtabmap
  camera:=0.114,0,Z,0,PITCH` (Z, PITCH from `camcheck`; an existing `rtab/arena.db` is renamed
  first, not deleted), `camlock` (the exposure; the DCW2 does not report its white balance),
  the survey, stop RTAB-Map with SIGINT, `rtab-export NAME`.
  **First runs, 2026-09-30:** 74 stations in 6 min 12 s, nothing blocked. The cloud: 99.8 % of
  the floor (scan 1: 93.1 %), the print's lines single and the arrows readable, the flat floor
  21.5 mm thick (10-90 % of the heights per 2 cm cell; scan 1: 23.7), the walls thicker (8-14
  cm: the camera sees them only at the top of its view). Built from the standing frames alone
  (358 of 573, told by their odometry speed): the floor 12.5 mm thick, the walls back to 6-11
  cm, but patchy -- one or two frames a station. Pausing RTAB-Map while moving (so that it
  takes standing frames only) lost its map within 27 stations -- doubled walls, the robot held
  at a wall by the lidar stop: it records all along. New: the ceiling lights glare on the mat
  in the downward view.
- **The maps to use:** `arena_v4` -- the live run loop4 replayed with today's defaults
  (bias-corrected merged scan, EKF 7 Hz, 20 cm between key scans): 3.200 x 2.785 m, every wall
  seen, the three landmarks; an image and YAML only, no pose graph. `arena_v3` -- the whole
  arena driven by hand, with its pose graph on the robot, but made before the bias correction:
  3.226 x 2.817 m.

## Network: lab WiFi at boot, hotspot off

Out of the box the robot starts its own hotspot, SSID ROSMASTER at 192.168.8.88, with no
internet. Since 2026-09-28 it joins IISLab-AMEC at boot instead (checked with a reboot). The
hotspot, NetworkManager connection `Hotspot`, is still configured but not started, and it has
no password any more: turned on again, it is an open network. Two laptop tools made those
changes while the robot was reachable only on its hotspot, hopping this laptop's WiFi over and
back in one run (`tools/open_hotspot.py` and `tools/lab_wifi.py`, each with a robot half that
logged to `/var/log/m3pro-*.log`; in git, gone 2026-10-05: their job was done). If IISLab-AMEC
is unavailable the robot tries its other saved networks
(MIFI-5BE2, Pi_Hot, ROSMASTER-A1) and otherwise stays offline; then get in over Ethernet, or
with a screen and keyboard. To start the hotspot at boot again, run on the robot and reboot:

```bash
sudo nmcli connection modify IISLab-AMEC connection.autoconnect no
sudo nmcli connection modify Hotspot connection.autoconnect yes
```

## Desktop with or without a monitor

The X3 Plus's setup, installed unchanged: `tools/display_setup.sh` with
`20-headless-virtual.conf` and `monitors.xml` next to it, as `install DP-1` (DP-1 is the
physical port on this carrier too). Copies are on the robot in `~/display_setup`, and
`bash ~/display_setup/display_setup.sh check` shows what Xorg, Mutter and VNC see.
Verified 2026-09-28:

- **No monitor at boot:** a 1920x1080 virtual screen, VNC on 5900 (password) and RDP on 3389
  from gnome-remote-desktop.
- **Monitor plugged in while running:** the desktop appears on it, no restart (an HP P244).
  GNOME keeps the virtual screen's 1920x1080@60 layout until the next boot with the monitor
  attached, so a monitor whose native mode is not 1080p only gets its own modes then.
- **Not tried yet:** unplugging while running.
- The first line of `check`, `/sys/class/drm/card1-DP-1` from the kernel, goes stale while X
  runs; trust the Xorg and xrandr lines below it.
