# 1. Error sources on the M3 Pro's base while it turns

Scope: where the pose error of the M3 Pro's mecanum base comes from while it turns, with a
magnitude for each source at our speeds (the object scan's goto: 0.05 m/s and up to 0.3 rad/s,
legs of at most 30 deg; the drive page: up to 0.15 m/s and 0.6 rad/s) and at the ranges the arena
gives (walls 0.3-3 m away; 1.2 m is used as the reference range). It covers the lidar scans as the
base board makes them, the merge of the two lidars, the board-to-Jetson clock, the wheel
odometry, the IMU and the lidars' own range and mounting errors. Fixes are in notes 2-4; the
budget is summed up in section 8. Labels: [REAL-LC] measured on a low-cost robot (ours unless
said), [REAL-other], [SIM], [CLAIM] (vendor or author statement, not measured by us), [2nd]
(second-hand: a search summary or abstract), "est." = our arithmetic from the stated model.
Nothing here was measured on the robot today (research only); every new number is either from
the repo's records or computed.

## 1. The board's lidar scans are snapshots of a rolling buffer, stamped when they are sent

### Takeaway
On the M3 the lidars are not read by a ROS driver on the Jetson: the STM32 base board parses the
T-mini Plus serial stream into a fixed array of angle bins and publishes a copy of that array on
a 140 ms timer, stamped at the moment of publishing. If the shipped firmware does what Yahboom's
published sample does, every scan holds points aged 0-167 ms (mean ~83 ms) with a seam where the
newest and oldest points meet, and `time_increment` is not filled in. This, not the clock
offset, is the largest geometric error between the scans and the odometry while turning.

Evidence:
- The control board is an STM32 with the left-rear lidar on UART4 and the right-front one on
  UART5, a 25 MHz crystal and an ICM-20948 on SPI2
  ([Yahboom, Introduction to the Control Board](https://www.yahboom.net/study/ROSMASTER-M3PRO/20493),
  [Read IMU data](https://www.yahboom.net/study/ROSMASTER-M3PRO/20510)) [CLAIM].
- Yahboom's sample "Read Lidar data" parses each T-mini packet (header, CT, LSN samples, start
  and end angle) and writes its points into `Lidar_Ranges[]` at index `start_angle / interval`,
  each further point `interval` (0.541 deg) on; bins a packet skipped are filled with the next
  packet's first distance ([Read Lidar data](https://www.yahboom.net/study/ROSMASTER-M3PRO/20511))
  [CLAIM, vendor sample code].
- Yahboom's sample "Publish radar data topic" (micro-ROS): `#define LIDAR_DATA_LEN 666`,
  `LIDAR_INTERVAL_POINTS (0.541f)`, `angle_increment = -0.541 deg` (clockwise), a publisher
  timer of `LIDAR_PUBLISHER_TIMEOUT (140)` ms, and in the callback the whole array is copied and
  stamped with `get_ros2_timestamp()` before `rcl_publish`; `time_increment` and `scan_time` are
  never set ([Publish radar data topic](https://www.yahboom.net/study/ROSMASTER-M3PRO/20526))
  [CLAIM, vendor sample code].
- The lidar spins at 6 Hz by default (6-12 Hz adjustable), ranges at 4000 Hz, 0.54 deg angular
  resolution ([T-mini Plus data sheet](https://cdn.soselectronic.com/productdata/6e/cf/9b94bb2f/t-mini-plus.pdf);
  [Yahboom, Lidar introduction](https://www.yahboom.net/study/ROSMASTER-M3PRO/15799): "Scanning
  Frequency 6 (6-12) Hz") [CLAIM].
- What we measured fits this exactly: `/scan0` and `/scan1` come at ~7.1 Hz (= a 140 ms timer,
  not a 6 Hz rotation) with 500 points over 270 deg (500 x 0.541 = 270.5 deg)
  ([PLAN.md M2](../../PLAN.md), 2026-09-30) [REAL-LC].
- est. (6 Hz rotation, 140 ms publishing, a 270 deg sector): the mean age of a scan's points at
  its stamp is 83 ms, between 62 and 104 ms depending on where the write head happens to be;
  16 % of the bins are re-sent unchanged from the previous scan (140/167 of a turn is refreshed
  per scan); the oldest and newest points are one revolution (167 ms) apart, side by side at the
  seam.

| turn rate | mean lag of the points (83 ms) | at 1.2 m | seam (167 ms) | at 1.2 m |
|---|---|---|---|---|
| 0.13 rad/s (a scan leg's average) | 0.62 deg | 1.3 cm | 1.24 deg | 2.6 cm |
| 0.3 rad/s (goto's TURN_V) | 1.43 deg | 3.0 cm | 2.86 deg | 6.0 cm |
| 0.6 rad/s (drive page cap) | 2.86 deg | 6.0 cm | 5.73 deg | 12.0 cm |

  Moving straight, the same timing gives 0.42 / 0.83 / 1.25 cm of mean lag and 0.8 / 1.7 /
  2.5 cm of seam at 0.05 / 0.10 / 0.15 m/s. All est.
- The shipped firmware is not the sample: it publishes 270 deg in 500 bins, on two topics, and
  its timer and stamp may differ. Nothing here is verified on our board yet; note 4 section 5
  lists the 30-minute check (time_increment, stale bins, the seam).
- If the shipped firmware also places points at a fixed 0.541 deg from each packet's start
  angle (as the sample does), any deviation of the motor from 6.00 Hz makes a small saw-tooth
  in angle within each packet (est.: at 6.2 Hz, ~0.017 deg per sample). Visible as a repeating
  ripple on straight walls; low priority.

## 2. Motion distortion is the same thing seen as smear: no consumer in our stack removes it

### Takeaway
A rotating 2D lidar on a turning robot always smears a scan over its sweep (167 ms here); the
usual cure is per-point de-skew from `time_increment` and odometry. Our scans carry no per-point
time, and slam_toolbox, AMCL and the merger do not de-skew anyway, so the full smear and the
mean lag of section 1 reach the scan matcher.

Evidence:
- slam_toolbox (Humble) looks up the odometry pose once per scan, at the scan's stamp, through a
  tf2 `MessageFilter` on the odom frame
  ([slam_toolbox_common.cpp, humble](https://github.com/SteveMacenski/slam_toolbox/blob/humble/src/slam_toolbox_common.cpp));
  AMCL likewise transforms an identity pose stamped at `laser_scan->header.stamp` into odom
  ([amcl_node.cpp, humble](https://github.com/ros-navigation/navigation2/blob/humble/nav2_amcl/src/amcl_node.cpp)).
  Neither has a de-skew step [code read 2026-10-27].
- `laser_geometry::transformLaserScanToPointCloud` (Humble) does interpolate: it looks up the
  target-frame transform at `stamp` and at `stamp + (n-1) * time_increment` and slerps per point
  ("Assume constant motion during the laser-scan")
  ([laser_geometry.cpp, humble](https://github.com/ros-perception/laser_geometry/blob/humble/src/laser_geometry.cpp)).
  With `time_increment` 0, or a target frame that moves with the lidar (`base_link`, as in the
  merger), it changes nothing.
- Cartographer is the one mapper that unwarps: every point gets
  `extrapolator_->ExtrapolatePose(time + point_time)`
  ([local_trajectory_builder_2d.cc](https://github.com/cartographer-project/cartographer/blob/master/cartographer/mapping/internal/2d/local_trajectory_builder_2d.cc)),
  which needs per-point times from `time_increment` (note 3).

## 3. Two lidars merged in base_link: no motion compensation, and a mounting mismatch

### Takeaway
The merge puts both lidars' points into `base_link` through static transforms, so whatever
timing each scan has passes straight through, and the merged scan carries one stamp for points
taken by two unsynchronised lidars over ~170 ms. Separately, the second lidar's measured
mounting correction (2.5 cm, 0.55 deg) was never applied; it does not change the fence's size,
but it puts the two lidars' copies of a wall ~2-3 cm apart, which is a likely share of the
4-6 cm thick walls.

Evidence:
- Yahboom merges with its own copy of `ira_laser_tools` in
  `~/M3Pro_ws/src/M3Pro_core/ira_laser_tools/src/laserscan_multi_merger.cpp`, `destination_frame:
  base_link`, then `yahboom_laser_filter` copies the message (stamp kept) and blanks returns under
  0.18 m ([Yahboom, Dual Lidar fusion](https://www.yahboom.net/study/ROSMASTER-M3PRO/15803))
  [CLAIM]. Our launch uses that merger
  ([arena_slam.launch.py](../../m3pro/arena/arena_slam.launch.py)).
- Which stamp the merged scan gets depends on the port: the original ROS 1 merger copies the
  first cloud's header and then overwrites the scan's stamp with `ros::Time(0)`
  ([kinetic source](https://github.com/iralabdisco/ira_laser_tools/blob/kinetic/src/laserscan_multi_merger.cpp));
  the public ROS 2 port pairs scans whose stamps are within `max_merge_time_diff` (0.05 s),
  publishes an incomplete pile after `max_completion_time` (0.05 s) when `best_effort` is on, and
  the merged header is that of the pile's slot 0
  ([relffok/ira_laser_tools](https://github.com/relffok/ira_laser_tools), CloudPile.cpp). Users
  report the ROS 2 merger "sometimes" publishing one lidar only
  ([ROS Answers 395522](https://answers.ros.org/question/395522/ira_laser_tools-not-consistently-merging-laserscans)) [2nd].
  Yahboom's copy was not read (it is on the robot).
- If the board sends both lidars from one timer (likely, unverified), the two stamps are nearly
  equal and the pairing works; each lidar's write head is still at its own angle, so the two
  halves of a merged scan have different age patterns (section 1).
- `lidar_calib.py` (removed 2026-10-05, in git at 7885fd8^) laid the second lidar onto the first
  by ICP on 15 s of standing scans: 2.5 cm and 0.55 deg; it "changed nothing in replays" of the
  fence size ([lidars.yaml](../../m3pro/arena/lidars.yaml)) [REAL-LC]. Size is the wrong test for
  it: a relative offset moves the two copies of each wall against each other, not the walls'
  mean. est.: 0.55 deg at 2 m = 1.9 cm, plus up to 2.5 cm of shift.
- The data sheet allows the lidar's zero direction to sit within +-3 deg of its marked axis
  ([data sheet](https://cdn.soselectronic.com/productdata/6e/cf/9b94bb2f/t-mini-plus.pdf), 2.7)
  [CLAIM]: each lidar's absolute yaw on the robot needs measuring, not just the pair's relative
  one. 1 deg of common yaw error turns the map against the wheels' heading: 2.1 cm at 1.2 m for
  anything that mixes the two (driving to a map goal, the arm camera's pose from a map pose).

## 4. The board's clock: one sync at connect, then free-running -- an offset that grows

### Takeaway
The board's micro-ROS clock is synced once, NTP-style, when the agent connects, and then runs
on its own crystal: we saw ~75 ms ahead on 2026-09-30 and 1.1 s ahead after ~23 h of uptime on
2026-10-08, i.e. ~13 ppm or ~48 ms per hour (two data points). Because all four board topics
share that clock, the offset puts no geometric error between scans and wheel odometry; it does
three other things: it makes the EKF and slam_toolbox wait (dropped scans), it misaligns
anything stamped by the Jetson (camera frames, RTAB-Map) with the base, and it grows.

Evidence:
- Micro XRCE-DDS (under micro-ROS): `uxr_sync_session` sends one TIMESTAMP and, on the reply,
  sets `session->time_offset = ((t0 + t3) - (t1 + t2)) / 2` from a single exchange; no averaging,
  no timer; `uxr_epoch_nanos` is `uxr_nanos() - session->time_offset`
  ([session.c](https://github.com/eProsima/Micro-XRCE-DDS-Client/blob/master/src/c/core/session/session.c)).
  `rmw_uros_sync_session(timeout_ms)` "Synchronizes the session time using the NTP protocol"
  ([time_sync.h, humble](https://github.com/micro-ROS/rmw_microxrcedds/blob/humble/rmw_microxrcedds_c/include/rmw_microros/time_sync.h)).
  Typical firmware calls it once per connection
  ([micro_ros_platformio #162](https://github.com/micro-ROS/micro_ros_platformio/issues/162)).
- Ours: scans +55 ms, odometry +72, IMU +74 (stamp minus arrival), 2026-09-30
  ([PLAN.md M2](../../PLAN.md)) [REAL-LC]. After ~23 h of uptime, 1.1 s ahead; restarting the agent
  re-synced it ("stamps ~behind arrival again"), 2026-10-08 (project memory,
  m3pro-arena-mapping) [REAL-LC].
- The three differences fit the serial link's transfer time exactly (est., 2 Mbaud = 200 kB/s,
  [Yahboom, Post a topic](https://www.yahboom.net/study/ROSMASTER-M3PRO/20518): "the baud rate is
  changed to 2Mbps"): a 500-range + 500-intensity scan (~4.1 kB) takes 20.4 ms, odometry (~0.7 kB)
  3.6 ms, IMU (~0.3 kB) 1.6 ms; differences 16.8 and 2.0 ms against the measured 17 and 2. So all
  three are stamped just before sending, and the true clock offset that day was ~76-78 ms (74 +
  1.6 ms of transfer + 1-2 ms of agent and DDS).
- Clock drift with the same bridge on a Jetson + Pixhawk: "slow, constant drift" with NTP off,
  "sudden jumps" with NTP on
  ([PX4 forum](https://discuss.px4.io/t/time-sync-drift-between-jetson-and-pixhawk-using-micro-xrce-dds/47961))
  [REAL-other, no numbers].
- Camera (Jetson clock) against odometry (board clock), est.: 65-78 ms is 1.1-1.3 deg at
  0.3 rad/s and 2.2-2.7 deg at 0.6 rad/s (2.3-5.6 cm at 1.2 m); 150 ms (two hours of uptime at
  48 ms/h) doubles it. RTAB-Map's +7 cm grid and its lidar-ICP walls 14-16 cm thick (visual
  stitching: 9-11) are consistent with this ([m3pro/README.md](../../m3pro/README.md)) [REAL-LC].

## 5. Wheel odometry in turns: encoder-only yaw, sampled velocities, uncalibrated scales

### Takeaway
`/odom_raw` integrates encoder velocities only (no IMU), by Euler steps at its publish rate
(~11 Hz live), from velocities sampled at that moment; Yahboom's two scale factors for it were
never calibrated on our robot. Our own scan data put the odometry's error at ~1.6 cm (median,
0.7-5.9) per 30-deg leg at 0.38 m, and constant scale factors explain little of it: it is mostly
per-leg, non-systematic error (slip at starts and stops, sampling), which calibration cannot
remove and an absolute fix (the lidar on the fence) can.

Evidence:
- Yahboom's odometry sample: a 50 ms timer; each tick takes `car_data.v_x, v_y, v_z` and
  `vel_dt = now - prev`, adds `v * dt` rotated by the heading before the step, and publishes;
  "angular speed from encoders"
  ([Release Speed Topic](https://www.yahboom.net/study/ROSMASTER-M3PRO/20524)) [CLAIM, sample code].
  Live, `/odom_raw` comes at ~11 Hz, not 20 ([PLAN.md M2](../../PLAN.md)) [REAL-LC].
- est.: integrating a sampled velocity misplaces each start and stop by up to one step: at
  0.3 rad/s and 91 ms, up to 1.6 deg per start or stop, ~0.6 deg rms per leg; goto's ramp
  (`w = 0.05 + 1.5 |turn|` near the end, [m3bridge.py](../../m3pro/tools/m3bridge.py)) shrinks the
  stop's share.
- Yahboom's firmware has `set_ros_scale_line` and `set_ros_scale_angular` (default 1.0), plus an
  IMU yaw PID for holding heading ([Modify the firmware parameters](https://www.yahboom.net/study/ROSMASTER-M3PRO/15771));
  their angular calibration turns the robot 360 deg by `/odom` and the user judges by eye
  whether it really turned 360 ([Angular velocity calibration](https://www.yahboom.net/study/ROSMASTER-M3PRO/15795))
  [CLAIM]. We have not run it.
- Our scans (12 stops round a fixed thing at 0.38 m, `goto` at 5 cm/s, 0.3 rad/s): the thing's
  reading in the odometry frame moved 0.7-5.9 cm from stop to stop (medians 1.5-1.9 cm per run)
  and 8.2-12.7 cm first to last, in three runs; the reading's own spread within a stop (base
  standing) is 0.3-0.9 cm (`m3pro/out/scan/20261012_165048`, `_165541`, `_172202`; analysis in the
  session scratchpad, `odom_fit.py`) [REAL-LC].
- Fitting constant wheel scales (yaw kw, lateral ky, forward kx, bounded to +-15-25 %) to those
  runs lowers the rms misfit only from 4.6 to 3.2-3.6 cm (one run), 5.4 to 4.6 and 3.9 to 3.3 cm
  (the others), with the scales running into their bounds: no constant scale explains the drift
  [REAL-LC, est.]. Caveat: one point landmark fixes only 2 of the 3 pose degrees, so this data
  cannot separate heading from position error; the lidar on the fence can.
- Literature: mecanum odometry over-reports travel in both axes, much more laterally, and
  under-reports the yaw angle; the error grows with distance (LiDAR used as reference)
  ([Springer chapter, 2021](https://link.springer.com/chapter/10.1007/978-3-030-75793-9_70))
  [2nd, abstract only]; lateral position accuracy "one magnitude" worse, slip in all lateral
  motion, growing with speed ([ResearchGate entry](https://www.researchgate.net/publication/220755973_The_sources_of_position_errors_of_omni-directional_mobile_robot_with_Mecanum_wheel))
  [2nd]; "slipping of the Mecanum wheels results in poor dead-reckoning"
  ([Killpack et al., BYU](https://scholarsarchive.byu.edu/facpub/3222)) [2nd]. No paper found
  with a per-cent yaw error for a small mecanum base turning in place (searched: Google-style web
  search for mecanum + odometry + rotation + slip + UMBmark; publishers' pages were paywalled).

## 6. The IMU: a good gyro, possibly not even in our EKF

### Takeaway
The board's ICM-20948 is set to +-2000 deg/s and its bias is calibrated at every boot; published
at 25 Hz, stamped at sending. Yahboom's EKF reads `/imu/data` from an `imu_filter` node that our
arena launch does not start, so our `/odom` may be wheels only -- check before tuning anything.
With the gyro in, a 1 % scale error is 3.6 deg per full turn: the scale needs calibrating
against the lidar.

Evidence:
- ICM-20948 on SPI2, `ICM20948_gyro_full_scale_select(_2000dps)` (16.4 LSB per deg/s = 0.06 deg/s
  per count), `ICM20948_gyro_calibration()` averages 100 readings at start-up and writes them to
  the offset registers ([Read IMU data](https://www.yahboom.net/study/ROSMASTER-M3PRO/20510))
  [CLAIM, sample code]. A robot that is moved or bumped while booting keeps that bias.
- `/imu/data_raw` on a 40 ms timer, the latest sample, `get_ros2_timestamp()` at sending; the
  orientation fields carry `w = 1` with x, y, z copied in, so the orientation must not be fused
  ([Publish IMU data topic](https://www.yahboom.net/study/ROSMASTER-M3PRO/20525)) [CLAIM].
- Yahboom's graph: `imu_filter` turns the board's IMU into `/imu/data`, and `ekf_filter_node`
  fuses `/odom_raw` and `/imu/data` into `/odom`
  ([Angular velocity calibration, 4.1](https://www.yahboom.net/study/ROSMASTER-M3PRO/15795))
  [CLAIM]. [arena_slam.launch.py](../../m3pro/arena/arena_slam.launch.py) starts only
  `ekf_node` with `yahboom_M3Pro_ekf.yaml`; if that file names `/imu/data`, nothing publishes it in
  our stack. Unverified either way (the yaml is on the robot).
- The data-sheet scale tolerance and bias stability of the ICM-20948 were not fetched (TDK's PDF
  did not load): treat "about 1-2 % scale, tens of mdeg/s residual bias" as [CLAIM, typical MEMS],
  to be replaced by the measurement in note 4.

## 7. The lidars' own errors: range bias, zero direction, tilt

### Takeaway
Range bias is measured and removed (0.77 and 1.15 cm); the data sheet allows 2 cm of systematic
error, so range-dependent bias is worth one check at two distances. The zero-direction tolerance
(+-3 deg) is the larger risk (section 3). Tilt is harmless for the walls.

Evidence:
- Biases 0.77 / 1.15 cm from the standing fence, taken off by `scan_relay.py`
  ([lidars.yaml](../../m3pro/arena/lidars.yaml)) [REAL-LC]; "Systematic error 20 mm, 0.05 m <
  Distance <= 12 m", tilt 0.75 deg typical, 1.5 max, zero direction +-3 deg
  ([data sheet](https://cdn.soselectronic.com/productdata/6e/cf/9b94bb2f/t-mini-plus.pdf)) [CLAIM].
- est.: a 1.5 deg tilt shortens the horizontal range by r(1 - cos 1.5 deg) = 1 mm at 3 m and puts
  the beam 14.5 +- 7.9 cm up the 30 cm fence at 3 m: still on it.

## 8. The budget at our speeds

### Takeaway
While turning, the scan timing (sections 1-2) dominates: ~1.4 deg of lag and +-1.4 deg of smear
at 0.3 rad/s, twice that at 0.6. Standing still, all timing terms vanish and what is left is
mounting (up to 0.55 deg between the lidars, +-3 deg absolute until measured), range bias
residuals and the map. The object scan stops for every view, so it can be served by standing
fixes regardless of how well turning is solved.

| source | at 0.3 rad/s, 0.05 m/s | at 0.6 rad/s, 0.15 m/s | standing | removable by |
|---|---|---|---|---|
| scan points older than the stamp (mean 83 ms) | 1.4 deg, 3.0 cm @1.2 m | 2.9 deg, 6.0 cm | 0 | re-stamp to mean time |
| smear within a scan / seam (167 ms) | +-1.4 deg, seam 6 cm | +-2.9 deg, seam 12 cm | 0 | per-point de-skew (needs the write head) |
| two lidars, different write heads | inside the row above | inside the row above | 0 | per-lidar de-skew before merging |
| relative mounting of lidar 1 (measured, not applied) | 0.55 deg, ~2-3 cm wall doubling | same | same | apply lidar_calib's correction |
| absolute lidar yaw (data sheet +-3 deg) | unknown, constant | same | same | calibrate vs. wheels |
| board clock vs Jetson (camera, RTAB-Map) | 1.1-1.3 deg (2.3-2.8 cm) at 65-78 ms | 2.2-2.7 deg (4.7-5.6 cm) | 0 | restamp board topics |
| ... growing with uptime | +48 ms/h (two points) | | | periodic sync or tracking |
| wheel yaw, per 30-deg leg | ~1.6 cm at 0.38 m (median), mostly random | larger (slip grows with speed) | 0 | absolute fix at stops; gyro |
| sampled-velocity integration | <= 1.6 deg per start/stop, ~0.6 deg rms/leg | 3.1 deg | 0 | gyro, firmware |
| gyro scale (if fused) | 1 % = 3.6 deg per turn | same | 0 | calibrate N turns vs lidar |
| gyro bias after warm-up (if fused) | 0.02 deg/s = 2.4 deg per 2-min scan ([CLAIM], typical) | same | grows with time | re-estimate at every stop |
| range bias residual | ~0.3 cm (est.) | same | same | done; check 2 distances |

All rows est. from the cited inputs except the wheel row ([REAL-LC], ours).
