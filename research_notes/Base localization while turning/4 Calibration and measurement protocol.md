# 4. Calibrating the base and measuring its error while it turns

Scope: how to calibrate what is constant (wheel scales, gyro scale and bias, each lidar's
mounting yaw and offset, range bias) and how to measure what is left -- heading and position
error while turning -- with what the lab has: the fence and landmarks as a reference the lidars
see, the floor print, tape marks, and an overhead camera. It ends with a 30-minute read-only
timing audit to run before anything else, and the pass criteria for the object scan (~1 cm,
~0.5 deg) and for docking. Labels as in note 1.

## 1. UMBmark, adapted to a mecanum base

### Takeaway
UMBmark's idea carries over: drive a closed figure both ways, several times, and separate the
systematic part (which a calibration removes) from the random part (which only an absolute fix
removes). For a mecanum base add the motions a differential robot does not have: pure rotation,
crab (lateral) runs, and the scan's own circle of 30-degree legs. Measure end poses with the
standing lidar fit on the fence (note 3), not by eye.

Evidence:
- UMBmark: a 4 x 4 m square, 5 runs clockwise and 5 counter-clockwise, `E_max,syst =
  max(r_cw, r_ccw)` of the centres of gravity of the end positions; the two dominant
  differential-drive errors (wheel-diameter ratio and wheelbase) are corrected from it; "a
  consistent improvement of at least one order of magnitude" in systematic accuracy
  ([Borenstein & Feng 1996, IEEE T-RA 12(5)](https://cs.au.dk/~ocaprani/legolab/DigitalControl.dir/NXT/Lesson10.dir/paper58.pdf))
  [REAL-other].
- For mecanum the systematic parameters are different: per-axis scales (forward, lateral, yaw)
  and, if you go further, per-wheel radius and the roller slip that the kinematic model hides;
  the lateral scale is the weak one ([Springer chapter](https://link.springer.com/chapter/10.1007/978-3-030-75793-9_70)) [2nd].
- Our arena allows a 1.5 x 1.5 m square with 0.65 m clear of the walls; the fence fit gives
  the end pose at every corner (stop 0.5 s, then fit), not just at the end.

## 2. The knobs we have: Yahboom's two scales, or a scale in our relay

### Takeaway
Calibrate with the lidar, not by eye. Either write the measured factors into the board
(`set_ros_scale_line`, `set_ros_scale_angular`, `python3 ~/config_robot.py`, agent stopped) or
apply them in the host relay that re-stamps `/odom_raw` (re-integrating its twist). The relay
is reversible and can hold a separate lateral factor, which the board's single line scale
cannot.

Evidence: [Modify the firmware parameters](https://www.yahboom.net/study/ROSMASTER-M3PRO/15771),
[Angular velocity calibration](https://www.yahboom.net/study/ROSMASTER-M3PRO/15795) (its
`odom_angular_scale_correction`, judged by "the actual rotation angle of the car") [CLAIM].
Expected gain is modest: our scan data show mostly non-systematic per-leg drift (note 1
section 5) [REAL-LC].

## 3. Lidar mounting: the pair, then the pair against the wheels

### Takeaway
Two steps. (1) Apply the second lidar's correction relative to the first (lidar_calib.py's
2.5 cm / 0.55 deg, re-measured) and judge by the walls' thickness, not the fence's size. (2)
Find the pair's common yaw against the wheels: drive straight along a long wall (and crab
along it) and compare the wall's direction in the scans with the direction of travel; or turn
in place both ways and read the yaw offset that is the same for both directions (a timing
error flips sign).

Evidence: [lidar_calib.py at 7885fd8^](../../m3pro/arena/lidars.yaml) (the yaml's comment points
to it) [REAL-LC]; data sheet zero direction +-3 deg ([T-mini Plus](https://cdn.soselectronic.com/productdata/6e/cf/9b94bb2f/t-mini-plus.pdf))
[CLAIM]. Range bias at a second distance: stand 0.5 m and 1.5 m from one wall and compare
(the data sheet allows 20 mm systematic error; our 0.77 / 1.15 cm were fitted at one set of
distances).

## 4. The gyro: bias at rest, scale over many turns

### Takeaway
Bias: 2 minutes standing after the boot calibration, mean of `angular_velocity.z` (the
resolution is 0.06 deg/s per count). Scale: N full turns in place (N = 3-5) both ways at 0.3
rad/s with the lidar fit before and after; `scale = (integrated gyro yaw) / (lidar yaw change +
2 pi N)`. At N = 5 and 0.2 deg of fit error the scale is good to ~0.01 % (est.). Apply it in the
relay; robot_localization has no scale parameter. The bias drifts as the board warms up, and
over a 2-minute scan 0.02 deg/s is 2.4 deg: re-estimate it whenever the wheels report standstill
(every scan stop does), in the relay, rather than trusting the boot-time value.

Evidence: ICM-20948 +-2000 dps, boot-time bias calibration
([Read IMU data](https://www.yahboom.net/study/ROSMASTER-M3PRO/20510)) [CLAIM].

## 5. Before anything: a 30-minute timing audit, robot standing (read-only)

### Takeaway
Six checks decide which fixes are needed; all are `ros2` commands or a small logging node on the
robot with nothing moving.

1. `ros2 topic echo --once /scan0` (and `/scan1`): `angle_min`, `angle_max`, `angle_increment`
   (sign), `time_increment`, `scan_time`, the number of ranges and whether `intensities` is
   filled. Expected from the sample code: `time_increment` and `scan_time` 0.
2. Stale bins: log 200 consecutive `/scan0` messages; per message, the share of ranges exactly
   equal to the previous message's. ~16 % means snapshots of a 6 Hz lidar on a 140 ms timer;
   ~0 % means one scan per revolution. Where the stale arc ends is the write head (note 3
   section 2).
3. `/scan0` vs `/scan1` stamps: one timer (equal stamps) or two.
4. `stamp - arrival` for all four board topics for an hour (a logging node, not `ros2 topic
   echo`): the offset now, its slope (ppm), and the transfer-time differences; repeat right
   after an agent restart. Also `timedatectl` (the Jetson's NTP client).
5. `ros2 node info /ekf_filter_node` and the `imu0` line of `yahboom_M3Pro_ekf.yaml`: is the IMU
   fused at all (Yahboom's graph expects `/imu/data` from `imu_filter`, which our launch does
   not start)? Which fields (`odom0_config`, `imu0_config`), `two_d_mode`?
6. Read Yahboom's merger source on the robot
   (`~/M3Pro_ws/src/M3Pro_core/ira_laser_tools/src/laserscan_multi_merger.cpp`): which stamp the
   merged scan gets, whether it waits for both lidars, `max_merge_time_diff`, `best_effort`.

## 6. The measurement protocol: heading and position error while turning

### Takeaway
Reference = the standing lidar fit on the fence and landmarks (all three degrees, ~0.5 cm /
~0.2 deg, est.), checked once against an independent reference (tape marks or an overhead
camera). Then three short drives give everything: in-place turns both ways at three speeds
(time offsets, smear, gyro and wheel yaw scales), the scan's circle without the arm (odometry
drift per leg, all three degrees), and a square both ways (UMBmark-style systematic error).

Setup:
- Arena empty except the landmarks; robot charged (> 11 V; the board went silent near
  10.4 V, project memory); cable out. Record a bag: `/scan0 /scan1 /odom_raw /imu/data_raw /tf
  /tf_static` and, once built, the re-stamped topics. No SLAM needed during recording.
- Reference model: the fence's four inner lines (3.2 x 2.8 m, measured) and the landmark
  outlines from `arena_v4` ([m3pro/arena/maps/](../../m3pro/arena/maps/)). Fit = point-to-line
  ICP of the merged, bias-corrected standing scan (2-3 scans averaged).

Validation of the reference (once, standing):
- Tape: mark the floor under two points of the chassis (a plumb line or a laser pointer pointing
  down) at 10 poses; move the robot away and back by hand to each mark; the fit must repeat to
  < 0.5 cm and < 0.2 deg. Measure the marks' distances to two walls with a tape for absolute
  error.
- Or overhead camera: a phone or webcam 2.5-3 m above the arena centre, calibrated (OpenCV,
  ChArUco), an ArUco/AprilTag (~10 cm) flat on the base's top; the fence's inner corners or the
  print's corners give the camera-to-arena homography. est. accuracy 3-5 mm and ~0.3 deg at
  2.5 m with a 1080p camera; standing comparisons only (its clock is not the robot's).

Runs (each ~2-5 minutes):
1. In-place turns: 3 full turns CW then CCW at 0.15, 0.3 and 0.6 rad/s, standing 2 s before and
   after each set; then 60 s of sinusoidal yaw (+-0.4 rad/s, 0.2 Hz). Analysis: (a) wheel and
   gyro yaw change vs. the standing fits -> yaw scales; (b) per-scan fit during the turns ->
   heading error vs. the interpolated reference, separately per turn rate and direction; the
   CW/CCW mean is the mounting yaw, half the CW-CCW difference over the rate is the time offset;
   (c) cross-correlate the scan-fit yaw rate with gyro and wheel rates -> each lag (note 2
   section 4); (d) the wall residual against beam index during steady turns -> the seam.
2. The scan's circle, arm parked: `m3scan`'s 12 stops of 30 deg round a marked spot at 0.38 m
   (no thing needed), standing fit at each stop -> per-leg heading and position error of
   `/odom_raw`, of the EKF's `/odom`, and of gyro-heading odometry; repeat 3 times.
3. A 1.5 x 1.5 m square, 3 runs each way, stops at the corners -> the UMBmark-style systematic
   error and the lateral (crab) and forward scales.

Pass criteria:
- Object scan: each stop's base pose from the standing fit within 1 cm and 0.5 deg of the
  reference; then in the fused model the box's and the can's sizes against the ruler (topic 7's
  judge) and no doubled edges.
- Mapping: walls in the map <= 3 cm thick (2 cm cells), fence 3.200 x 2.800 +- 1 cm, stable
  over three loops, with turns at the drive page's 0.6 rad/s allowed.
- Docking: AMCL pose within the staging tolerance (~5 cm, 3 deg) while driving at 0.15 m/s and
  0.3 rad/s; the final approach is the camera's.

## 7. What we expect after the fixes (est., to be confirmed by section 6)

### Takeaway
Standing: ~0.5 cm and ~0.2 deg from the fence fit, i.e. the scan's target is met at its stops
once the lidars' mounting is calibrated. Turning at 0.3 rad/s: re-stamping removes ~1.4 deg of
mean lag; with per-point de-skew the residual should be ~0.2-0.4 deg (10 ms of timing error is
0.17 deg); without de-skew the seam leaves +-1.4 deg of smear that the matcher averages. Wheel
odometry alone stays at ~1-2 cm per leg; gyro yaw with a calibrated scale and its bias
re-estimated at every stop should hold ~0.3-0.5 deg per circle (est.: 0.1 % of 360 deg is
0.36 deg; a residual bias of 0.003 deg/s over the ~2 minutes is another 0.4 deg -- without the
re-estimation, a bias of 0.02 deg/s left from a warm-up would add 2.4 deg).
