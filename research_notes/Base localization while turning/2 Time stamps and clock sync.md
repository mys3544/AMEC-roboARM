# 2. Time stamps and clock sync between the base board and the Jetson

Scope: how the nodes we run use message stamps (robot_localization, slam_toolbox, Nav2 AMCL,
the lidar merger, Cartographer), why the board's clock offset makes them wait and drop scans,
and the three ways to fix the timing: re-sync in the firmware, a host-side re-stamper that
tracks the offset, and estimating the remaining scan-to-odometry offset from motion data. The
error sources themselves are in note 1; de-skew and localization in note 3. Labels as in note 1.

## 1. Every consumer trusts the stamp; robot_localization waits for the host clock to reach it

### Takeaway
robot_localization processes a measurement only once the host clock has passed its stamp, and
stamps its `odom -> base_footprint` transform with the last measurement's (board) time. With
the board ~75 ms ahead, the transform for a scan's stamp exists only ~75 ms (plus up to one EKF
period) after the scan arrived; slam_toolbox's queue of one is overwritten by the next scan
first. There is no per-sensor time-offset parameter to correct this inside the EKF.

Evidence:
- `integrateMeasurements`: "if (current_time < measurement->time_) { ... break; }" -- later
  measurements "should wait until a future iteration"; the transform is stamped
  `filtered_position->header.stamp + tf_time_offset_`, where the odometry stamp is
  `filter_.getLastMeasurementTime()`
  ([ros_filter.cpp, humble-devel](https://github.com/cra-ros-pkg/robot_localization/blob/humble-devel/src/ros_filter.cpp))
  [code read 2026-10-27].
- Timing parameters are `frequency, sensor_timeout, transform_time_offset, transform_timeout,
  predict_to_current_time, history_length, smooth_lagged_data, *_queue_size, *_nodelay,
  *_differential, *_relative`; the only offset is the global `transform_time_offset` ("Some
  packages require that your transforms are future-dated by a small time offset")
  ([state_estimation_nodes.rst](https://github.com/cra-ros-pkg/robot_localization/blob/humble-devel/doc/state_estimation_nodes.rst)).
- slam_toolbox (Humble): scans go through `tf2_ros::MessageFilter(..., odom_frame_,
  scan_queue_size_, ..., transform_timeout_)`; one `LaserRangeFinder` per `frame_id`;
  map -> odom is stamped `scan_timestamp + transform_timeout_` (or `now() + transform_timeout_`
  with `restamp_tf`)
  ([slam_toolbox_common.cpp](https://github.com/SteveMacenski/slam_toolbox/blob/humble/src/slam_toolbox_common.cpp));
  the README says `scan_queue_size` "should be 1 in async mode"
  ([README, humble](https://github.com/SteveMacenski/slam_toolbox/blob/humble/README.md)).
- Nav2 AMCL (Humble): odometry at the scan's stamp; map -> odom stamped `stamp +
  transform_tolerance_` ([amcl_node.cpp](https://github.com/ros-navigation/navigation2/blob/humble/nav2_amcl/src/amcl_node.cpp)).
- Ours: "scans:=lidars" (14 scans/s on one topic) dropped nearly all scans live; Yahboom's EKF
  rate 7 Hz with the merged scan kept enough; `scan_queue_size` 10 "wrecked the map"
  ([PLAN.md M2](../../PLAN.md), [scan_relay.py](../../m3pro/arena/scan_relay.py)) [REAL-LC]. A
  likely reading (not tested): dropping scans acted as an accidental filter; with the queue,
  more scans taken while turning -- the most distorted ones (note 1) -- reached the matcher.

## 2. The board syncs once; re-syncing is a firmware change

### Takeaway
micro-ROS's sync is one NTP-style exchange, computed from a single sample, when the firmware
asks; Yahboom's firmware evidently asks only at connection (restarting the agent re-synced it).
A firmware that calls `rmw_uros_sync_session()` every 10-30 s would hold the offset to the
link's asymmetry (~1 ms on a 2 Mbaud serial link, est.) instead of letting it grow ~48 ms per
hour -- but it means rebuilding and flashing Yahboom's STM32 firmware, which PLAN calls a last
resort.

Evidence:
- Single exchange, `time_offset = ((t0 + t3) - (t1 + t2)) / 2`, no averaging, no timer
  ([session.c](https://github.com/eProsima/Micro-XRCE-DDS-Client/blob/master/src/c/core/session/session.c));
  the Micro XRCE-DDS docs suggest averaging several exchanges for accuracy and allow a user
  callback ([time_sync.rst](https://github.com/eProsima/Micro-XRCE-DDS-docs/blob/master/docs/time_sync.rst)).
- The 1.1 s after ~23 h and the re-sync on agent restart (note 1 section 4) [REAL-LC].
- Firmware source and build: Yahboom's control-board course (STM32CubeIDE, micro-ROS library,
  ST-Link recommended since the agent's port doubles as the flashing port)
  ([Publish radar data topic](https://www.yahboom.net/study/ROSMASTER-M3PRO/20526)); factory
  firmware in the "Code-Firmware" download ([course page](https://www.yahboom.net/study/ROSMASTER-M3PRO)) [CLAIM].
- Without flashing, a scheduled agent restart re-syncs too, but it drops the board's topics
  for seconds and once left the board silent (the 2026-10-15 hang needed the reset button):
  not recommended while driving.

## 3. A host-side re-stamper removes the offset without touching the firmware

### Takeaway
Track the board's clock from the data: for each board topic, `stamp - arrival` is the clock
offset minus the transfer time minus a positive, jittery delay, so its running maximum over a
window, plus the known transfer time, is the offset. Re-stamp all four board topics
(`/scan0`, `/scan1`, `/odom_raw`, `/imu/data_raw`) into Jetson time in one node, feed the EKF
and the merger the re-stamped copies, and the waiting, the dropped scans and the camera-to-base
mismatch go away together. Expected accuracy: a few ms (est.).

How (a design, not built):
- Offset `c(t)`: on `/imu/data_raw` (25 Hz, 1.6 ms transfer), `c = max over the last 30-60 s of
  (stamp - arrival) + 1.6 ms` (a one-way minimum-delay estimator: the least-delayed message
  bounds the offset; drift of 13 ppm moves it 0.8 ms per minute, so a window of a minute is
  fine). Log `c(t)` -- it is also the drift measurement of note 4.
- Re-stamp: `stamp_host = stamp_board - c`; for scans, additionally shift to the points' mean
  time (`- 83 ms` by the snapshot model, or per point, note 3).
- Transfer times at 2 Mbaud (est.): scan with intensities 20.4 ms, ranges only 10.4 ms,
  odometry 3.6 ms, IMU 1.6 ms; the measured differences (17 and 2 ms) fit the first, odometry
  and IMU (note 1 section 4).
- Where: `scan_relay.py` already subscribes to both scans; add the IMU and odometry and
  republish `*_h` topics; point the EKF at them (`odom0`, `imu0` in a copy of Yahboom's yaml).
- Watch for: Jetson clock steps by its NTP client (the PX4 thread saw "sudden jumps" with NTP
  on, [PX4 forum](https://discuss.px4.io/t/time-sync-drift-between-jetson-and-pixhawk-using-micro-xrce-dds/47961));
  a step makes `stamp - arrival` jump, and the windowed maximum must be reset on a jump larger
  than ~20 ms. `timedatectl` on the robot tells which client runs.

## 4. The remaining offset between scan geometry and odometry is estimated from motion

### Takeaway
Even with one clock, "when a scan's points were taken" relative to its stamp is a property of
the firmware (note 1: ~83 ms by the sample code), so it should be measured, not assumed:
rotate the robot back and forth in the arena, match every scan to the standing map, and find
the time shift that best aligns the scan-matched yaw rate with the gyro's and the wheels'. The
same data give each sensor's own lag. Turning both ways separates a time offset (it flips sign
with the direction of turn) from a mounting yaw (it does not).

Evidence and method:
- Kalibr's camera-IMU calibration does exactly this for its time-shift prior: `np.correlate` of
  the predicted and measured angular-velocity norms, peak lag times the mean IMU period
  ([IccSensors.py](https://github.com/ethz-asl/kalibr/blob/master/aslam_offline_calibration/kalibr/python/kalibr_imu_camera_calibration/IccSensors.py)),
  then refines it in the batch fit.
- est. resolution: at 0.3 rad/s, 10 ms of offset is 0.17 deg; per-scan ICP heading noise on the
  fence is a few tenths of a degree, so a minute of back-and-forth (hundreds of scans) pins the
  offset to a few ms. Use a sinusoidal yaw command (e.g. +-0.4 rad/s, 0.2 Hz) so the rate keeps
  changing; a constant turn only shows the lag as a constant heading bias, which also a
  mounting yaw makes.
- Refinement: re-run the scan matching with the scan's stamp shifted by `d` and pick the `d`
  that minimises the matched residual; repeat with per-point de-skew once the write head is
  known (note 3).
- Output: three numbers to put in the re-stamper -- the scan's effective lag, the wheel
  odometry's and the IMU's (each relative to its stamp).

## 5. What the camera and the arm need from this

### Takeaway
The DCW2's frames and the arm's commands are on the Jetson's clock; the base's data are on the
board's. Once the board topics are re-stamped, a camera frame can be paired with the base pose
at its own time, and the object scan's views, RTAB-Map's key frames and the survey stop
depending on "stand still 1.8 s and the lag does not matter".

Evidence: the survey stands 1.8 s per station because "frames taken moving carry the board
clock's 55-75 ms lag as blur" ([drive_core.py](../../m3pro/arena/drive_core.py)); RTAB-Map's
lidar-stitched walls 14-16 cm thick vs. 9-11 visually ([m3pro/README.md](../../m3pro/README.md))
[REAL-LC].
