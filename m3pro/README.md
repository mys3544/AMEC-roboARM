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

## The software, copied from the X3 Plus

`roboarm/`, `tools/`, `tests/`, `vision_service/`, `docker/`, `cameras/` and the project files
(`pyproject.toml`, `uv.lock`, `compose.yaml`, `.dockerignore`, `.gitignore`) are the X3
Plus's as of commit 1f94700, byte for byte, except that nothing here points at the X3 Plus:
`tools/sync_robot.py` pushes to this robot, 192.168.73.66, and the examples in
`tools/webapp.py` and `tools/hwbridge.py` name it. Left out: the X3's docs and README, and
`tools/wifi_hop.ps1`, its old WiFi workaround (`tools/open_hotspot.py` is the M3's).

All of it still assumes the X3 Plus's hardware -- the arm and its kinematics, the Rosmaster
board through Rosmaster_Lib on `/dev/ttyUSB0`, the wrist camera and its calibration -- until
it is changed here for the M3 Pro. Two differences are known already:

- **Serial:** on the M3 Pro, `/dev/ttyUSB0` (`/dev/myserial`, a CP210x) is the micro-ROS base
  board, held by the micro-ROS agent. The copied arm code must not open it until it is
  rewritten for this robot.
- **Cameras:** a depth camera (an Orbbec; its ROS 2 driver is in
  `~/yahboomcar_ws/src/OrbbecSDK_ROS2`) and an RGB camera that differs from the X3 Plus's.
  Neither was plugged in on 2026-09-28. The camera-specific code is `roboarm/config.py`
  (`ROBOARM_CAMERA` and the lens geometry), `roboarm/camera.py`, the profiles in `cameras/`,
  and `vision_service/depth.py`, which estimates depth from one RGB frame and could use the
  real depth camera instead.

Work from this folder; it has its own `pyproject.toml` and `uv.lock`, so its own `.venv`:

```bash
cd m3pro
uv sync --frozen
uv run --frozen python -m pytest -q     # `python -m` so that roboarm/ here is imported
uv run --frozen ruff check roboarm tools tests vision_service
python tools/sync_robot.py              # system Python (paramiko): push to ~/roboarm on the M3
```

## Mapping the arena

`arena/` runs on the robot's host ROS 2 (not in a container), from `~/arena`: slam_toolbox on
the two lidars at 2 cm, with Yahboom's EKF and scan merge (`arena_slam.launch.py`), and a drive
page on port 8095 (`drive_server.py`). `tools/arena_map.py` drives it from this laptop (system
Python, paramiko):

```bash
python tools/arena_map.py start        # upload arena/ and start the stack; nothing moves yet
python tools/arena_map.py snap         # the map so far, with the fence fit -> out/arena/
python tools/arena_map.py save NAME    # -> arena/maps/NAME*, and the pose graph on the robot
python tools/arena_map.py stop
```

- **Driving:** http://192.168.73.66:8095/ -- hold the buttons or W/S (forward, back), A/D
  (sideways), Q/E (turn); letting go, Space or closing the tab stops it. "Auto: loop the arena"
  drives a slow loop 45 cm inside the fitted fence, around the landmarks. Both are held to
  0.15 m/s and stop 35 cm short of anything the lidars see in their path. The gamepad works
  whenever the page is not commanding.
- **The fence fit** (`fence_fit.py`) checks every map against the fence's 3.2 x 2.8 m: side
  lengths, corner angles, how straight and how much of each wall is seen, and the blobs inside
  (the landmarks) in an arena frame with its origin at a fence corner.
- **What the robot gives:** the lidars come from the micro-ROS board as `/scan0` and `/scan1`,
  270 degrees each at about 7 Hz, 14.5 cm above the floor; their static transforms exist only
  while `m3pro_bringup/static_tf.launch.py` runs (the launch includes it).
- **First maps, 2026-09-30,** in `arena/maps/`: 3.213 x 2.801 m after half a metre of driving,
  3.232 x 2.823 m after a full loop -- the map grows with distance driven, which is the next
  thing to tune.

## Network: lab WiFi at boot, hotspot off

Out of the box the robot starts its own hotspot, SSID ROSMASTER at 192.168.8.88, with no
internet. Since 2026-09-28 it joins IISLab-AMEC at boot instead (`tools/lab_wifi.py`, checked
with a reboot). The hotspot, NetworkManager connection `Hotspot`, is still configured but not
started, and it has no password any more (`tools/open_hotspot.py`): turned on again, it is an
open network. If IISLab-AMEC is unavailable the robot tries its other saved networks
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

## WiFi tools (M3 only)

These were needed while the robot was reachable only on its own hotspot, which has no
internet: each hops this Windows laptop's WiFi over to ROSMASTER, does its job and comes back
to IISLab-AMEC in one run, and the robot undoes the change by itself unless the laptop
confirms it can still reach it. Run them with the system Python 3.12 (it has paramiko); the
SSH password comes from `ROBOARM_SSH_PASSWORD` or is asked for.

- `tools/open_hotspot.py`, robot half `open_hotspot.sh`: takes the password off the hotspot,
  and holds the WiFi-hop helpers the other tool uses.
- `tools/lab_wifi.py`, robot half `lab_wifi.sh`: makes IISLab-AMEC the boot network and stops
  the hotspot, with the lab passphrase read from this laptop's saved WiFi profile.

On the robot they log to `/var/log/m3pro-open-hotspot.log` and `/var/log/m3pro-lab-wifi.log`.
