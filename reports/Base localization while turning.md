# The M3's error while turning is a timing problem in its base board's lidar scans, and the object scan can sidestep it by localizing on the fence at every stop

**Bottom line.** While the M3 Pro turns, its biggest pose error is not wheel slip but time: the
STM32 base board does not send lidar revolutions, it sends a copy of a rolling buffer of angle
bins every 140 ms, stamped when sent, on a clock that is synced once and then drifts. By
Yahboom's own sample firmware, which matches every number we have measured, a scan's points are
on average ~83 ms older than its stamp and span one 167 ms revolution, and nothing in our stack
(the merger, slam_toolbox, AMCL) corrects for it. The wheel odometry that the object scan drives
on loses ~1.6 cm per 30-degree leg, mostly at random. The fix for the scan is to stop trusting
motion at all: at every stop the base stands still, and a standing scan matched to the fence and
the three landmarks has no timing error -- expected ~0.5 cm and ~0.2 deg, which meets the
~1 cm / ~0.5 deg target. For mapping and driving: re-stamp the board's topics on the Jetson,
feed the mapper only scans taken while turning slowly, calibrate the lidars' mounting and the
gyro, then AMCL with the omnidirectional model. **Error budget** at goto's 0.3 rad/s and
0.05 m/s (in brackets: the drive page's 0.6 rad/s and 0.15 m/s), sideways error at 1.2 m:

| source | heading | position | standing |
|---|---|---|---|
| scan points older than the stamp (~83 ms) | 1.4 deg (2.9) | 3.0 cm (6.0) + 0.4 cm along (1.3) | 0 |
| smear inside one scan, seam (167 ms) | +-1.4 deg, 2.9 deg step (+-2.9, 5.7) | seam 6 cm (12) | 0 |
| board clock vs Jetson, ~77 ms, growing ~48 ms/h | 1.3 deg (2.7), camera vs base only | 2.8 cm (5.6) | 0 |
| second lidar's mounting, measured, not applied | 0.55 deg | 2.5 cm, walls doubled ~2-3 cm | same |
| lidars' absolute zero direction (data sheet) | up to +-3 deg, unmeasured | 6 cm at 1.2 m per 3 deg | same |
| wheel odometry, per 30-deg leg at 0.38 m | ~1-2 deg if all heading | ~1.6 cm median (0.7-5.9), 8-13 cm a circle | 0 |
| gyro (if fused): 1 % scale; 0.02 deg/s bias | 3.6 deg per turn; 2.4 deg per 2 min | -- | bias grows |
| range bias (measured 0.77 / 1.15 cm, corrected) | -- | ~0.3 cm residual (est.) | same |

Labels used below: [REAL-LC] measured on a low-cost robot (ours unless said), [REAL-other],
[CLAIM] (a vendor's or author's statement, including vendor sample code), [2nd] (abstract or
search summary), est. (our arithmetic). The detailed evidence is in four notes:
[1 Error sources](<../research_notes/Base localization while turning/1 Error sources on the M3 while it turns.md>),
[2 Time stamps and clock sync](<../research_notes/Base localization while turning/2 Time stamps and clock sync.md>),
[3 De-skew, scan matching and localization](<../research_notes/Base localization while turning/3 De-skew, scan matching and localization in the arena.md>),
[4 Calibration and measurement protocol](<../research_notes/Base localization while turning/4 Calibration and measurement protocol.md>).
Nothing was run on the robots for this report; the first step of the plan below is a 30-minute
read-only audit that confirms or refutes the firmware model.

## 1. The base board sends snapshots of a rolling buffer, stamped when sent

The two T-mini Plus lidars are wired to the STM32 board (UART4 left rear, UART5 right front),
not to the Jetson ([Yahboom, control board](https://www.yahboom.net/study/ROSMASTER-M3PRO/20493)).
Yahboom's published firmware samples show how they become `/scan0` and `/scan1`
[CLAIM, vendor sample code]:
- the parser writes each T-mini packet into `Lidar_Ranges[]` at bin `start_angle / 0.541`
  ([Read Lidar data](https://www.yahboom.net/study/ROSMASTER-M3PRO/20511));
- a micro-ROS timer of 140 ms copies the whole array into a `LaserScan` (666 bins, `angle_increment
  = -0.541 deg`), stamps it with `get_ros2_timestamp()` and publishes; `time_increment` and
  `scan_time` are never set ([Publish radar data topic](https://www.yahboom.net/study/ROSMASTER-M3PRO/20526));
- the lidar turns at 6 Hz by default ([data sheet](https://cdn.soselectronic.com/productdata/6e/cf/9b94bb2f/t-mini-plus.pdf),
  [Yahboom, lidar](https://www.yahboom.net/study/ROSMASTER-M3PRO/15799)).

Our measurements fit: `/scan0` and `/scan1` arrive at ~7.1 Hz -- the 140 ms timer, not the 6 Hz
rotation -- with 500 points over 270 deg = 500 x 0.541 deg ([PLAN.md M2](../PLAN.md)) [REAL-LC].
Consequences (est.): each scan holds points aged 0-167 ms, mean 83 ms (62-104 depending on
where the write head is); 16 % of the bins are re-sent unchanged; the newest and oldest points
sit side by side at a seam. The odometry and the IMU are stamped the same way, at sending
([odometry sample](https://www.yahboom.net/study/ROSMASTER-M3PRO/20524),
[IMU sample](https://www.yahboom.net/study/ROSMASTER-M3PRO/20525)), so the scan lags the
odometry by the points' age: 1.4 deg at 0.3 rad/s. The stamp-minus-arrival differences we
measured (scans +55, odometry +72, IMU +74 ms) match the serial link's transfer times of the
three message sizes at 2 Mbaud almost exactly (20.4, 3.6, 1.6 ms, est.), so all three are
stamped just before sending and the true clock offset that day was ~76-78 ms.

The merge does not help: Yahboom's `ira_laser_tools` copy merges in `base_link` through static
transforms ([Dual Lidar fusion](https://www.yahboom.net/study/ROSMASTER-M3PRO/15803)); known ports
either zero the merged stamp ([ROS 1 source](https://github.com/iralabdisco/ira_laser_tools/blob/kinetic/src/laserscan_multi_merger.cpp))
or take the first slot's header and publish half-piles after 50 ms
([relffok port](https://github.com/relffok/ira_laser_tools)). Yahboom's own copy is on the robot
and was not read. slam_toolbox and AMCL take one odometry pose per scan, at its stamp, and do not
de-skew ([slam_toolbox](https://github.com/SteveMacenski/slam_toolbox/blob/humble/src/slam_toolbox_common.cpp),
[AMCL](https://github.com/ros-navigation/navigation2/blob/humble/nav2_amcl/src/amcl_node.cpp)).

A likely explanation of past results (not tested): the map grew more with the EKF at 30 Hz than
at 7 Hz and broke with `scan_queue_size` 10 because in both cases more scans taken while
turning, the most distorted ones, reached the matcher; dropping them was an accidental filter.
The walls' 4-6 cm thickness is consistent with the seam (6 cm at 1.2 m and 0.3 rad/s) plus the
unapplied mounting of the second lidar.

## 2. Wheel odometry loses ~1.6 cm per 30-degree leg, mostly at random; only an absolute fix removes it

The board integrates encoder velocities only (no IMU) by Euler steps at its publish rate,
~11 Hz live ([odometry sample](https://www.yahboom.net/study/ROSMASTER-M3PRO/20524) [CLAIM];
[PLAN.md](../PLAN.md) [REAL-LC]); a velocity sampled once per step misplaces each start and
stop by up to one step, 1.6 deg at 0.3 rad/s (est.). Yahboom's scale factors for it
(`set_ros_scale_line`, `set_ros_scale_angular`, default 1.0) were never calibrated on our robot
([firmware parameters](https://www.yahboom.net/study/ROSMASTER-M3PRO/15771)).

Our three object scans of 2026-10-12 measure the result: the fixed thing's reading in the
odometry frame moved 0.7-5.9 cm from stop to stop (median per run 1.5-1.9 cm), 8.2-12.7 cm
first to last, while within a stop it spreads only 0.3-0.9 cm
(`m3pro/out/scan/20261012_165048`, `_165541`, `_172202`) [REAL-LC]. Fitting constant wheel
scales (yaw, lateral, forward) lowers the misfit only from 4.6 to 3.2 cm at best, with the
scales running into their bounds: the drift is mostly per-leg and random, not a scale. The
literature agrees on the shape -- lateral far worse than forward, yaw under-reported, slip
growing with speed ([Springer chapter](https://link.springer.com/chapter/10.1007/978-3-030-75793-9_70),
[ResearchGate entry](https://www.researchgate.net/publication/220755973_The_sources_of_position_errors_of_omni-directional_mobile_robot_with_Mecanum_wheel))
[2nd] -- but gave no per-cent yaw figure for a small mecanum base. A single point landmark (the
scanned thing) fixes only two of the three pose degrees, which is why `m3scan.anchor()` can
correct position but not heading; the fence fixes all three.

The IMU would help turns but may not be in our EKF at all: Yahboom's graph feeds the EKF
`/imu/data` from an `imu_filter` node ([Angular velocity calibration](https://www.yahboom.net/study/ROSMASTER-M3PRO/15795)),
and [arena_slam.launch.py](../m3pro/arena/arena_slam.launch.py) starts only the EKF. The gyro
is an ICM-20948 at +-2000 deg/s with a bias calibration at boot
([Read IMU data](https://www.yahboom.net/study/ROSMASTER-M3PRO/20510)) [CLAIM]; it needs its scale
measured (1 % is 3.6 deg per turn) and its bias re-estimated at stops (0.02 deg/s is 2.4 deg
over a 2-minute scan).

## 3. The clock offset harms the camera and the pipeline, not the scan-to-wheel geometry

micro-ROS's sync is a single NTP-style exchange: `time_offset = ((t0 + t3) - (t1 + t2)) / 2`,
no averaging, no timer ([Micro XRCE-DDS session.c](https://github.com/eProsima/Micro-XRCE-DDS-Client/blob/master/src/c/core/session/session.c),
[rmw_uros_sync_session](https://github.com/micro-ROS/rmw_microxrcedds/blob/humble/rmw_microxrcedds_c/include/rmw_microros/time_sync.h)).
Ours was ~77 ms ahead on 2026-09-30 and 1.1 s ahead after ~23 h of uptime on 2026-10-08 (project
memory), re-synced by an agent restart [REAL-LC]: ~13 ppm, ~48 ms per hour, from two points.
Since all board topics share that clock, it adds no error between the scans and the wheels. It
does three other things:
1. robot_localization processes a measurement only when the host clock passes its stamp and
   stamps its transform at the last measurement's time
   ([ros_filter.cpp](https://github.com/cra-ros-pkg/robot_localization/blob/humble-devel/src/ros_filter.cpp));
   it has no per-sensor offset ([parameters](https://github.com/cra-ros-pkg/robot_localization/blob/humble-devel/doc/state_estimation_nodes.rst)).
   So the transform for a scan arrives ~77 ms late and slam_toolbox's queue of one drops it:
   the live failure of `scans:=lidars`.
2. Anything on the Jetson's clock -- the DCW2's frames, RTAB-Map's key frames, the console --
   is paired with the base 77 ms off: 1.3 deg at 0.3 rad/s, 2.7 at 0.6 (RTAB-Map's +7 cm grid and
   14-16 cm lidar-stitched walls, [m3pro/README.md](../m3pro/README.md)) [REAL-LC].
3. It grows until the agent reconnects, and NTP steps on the Jetson move it suddenly
   ([PX4 forum](https://discuss.px4.io/t/time-sync-drift-between-jetson-and-pixhawk-using-micro-xrce-dds/47961))
   [REAL-other].

The fix without flashing is a host re-stamper: the running maximum of `stamp - arrival` on the
IMU (1.6 ms transfer) over the last minute, plus the transfer time, is the offset; subtract it
from all four board topics and point the EKF and the merger at the copies. The firmware fix --
`rmw_uros_sync_session()` every 10-30 s -- is cleaner but needs a build and a flash.
What is left after re-stamping -- when within a scan the points were taken -- is measured from
motion: match each scan to the standing map while the robot yaws back and forth, and find the
time shift that best aligns the scan-matched yaw rate with the gyro's (the method behind
Kalibr's time-shift prior, [IccSensors.py](https://github.com/ethz-asl/kalibr/blob/master/aslam_offline_calibration/kalibr/python/kalibr_imu_camera_calibration/IccSensors.py)).
Turning both ways separates a time offset (its error flips sign) from a mounting yaw (it does not).

## 4. No ROS 2 node de-skews our scans; a turn-rate gate and a re-stamp get most of the gain

In Humble only Cartographer unwarps 2D scans, point by point, with its pose extrapolator
([local_trajectory_builder_2d.cc](https://github.com/cartographer-project/cartographer/blob/master/cartographer/mapping/internal/2d/local_trajectory_builder_2d.cc));
upstream says it is "no longer actively maintained" ([README](https://github.com/cartographer-project/cartographer)).
`laser_geometry` has the per-point interpolation into any frame
([laser_geometry.cpp](https://github.com/ros-perception/laser_geometry/blob/humble/src/laser_geometry.cpp));
the 2D de-skew node `laser_deskew` is ROS 1 ([repo](https://github.com/rsasaki0109/laser_deskew)).
All of them need per-point times, which our scans lack. In order of cost:
- shift each scan's stamp back to its points' mean time (removes the 1.4 deg lag at 0.3 rad/s)
  and give the mapper only scans taken below ~0.15 rad/s (removes most of the smear from the
  map; the safety stop still gets every scan);
- infer the write head from the ~16 % of bins that did not change since the last scan, and
  de-skew each point into `odom` at its own time (removes the seam);
- change the firmware to publish one scan per revolution, stamped at its first point, with
  `time_increment` -- then any standard tool works, Cartographer included.

## 5. Standing on the fence gives the scan its 1 cm and 0.5 deg; AMCL-Omni drives; the camera docks

**For the object scan (topic 7).** The base stands still at each of the 12 stops while the arm
takes its views; the arm's move to the first view plus `settle_close_s` (0.35 s) already take
longer than one lidar revolution (0.17 s) and the transfer, so the wait is free. A standing
scan, both lidars bias-corrected and mounted right, matched by point-to-line ICP to the fence's
measured 3.2 x 2.8 m and the three landmark outlines from `arena_v4`, gives the base pose in all
three degrees (est. ~0.5 cm and ~0.2 deg: two 3 m walls good to 0.5 cm bound the heading to
~0.1 deg; systematic terms set the floor). That replaces ~1.6 cm per leg and 8-13 cm per circle
of wheel drift, and -- unlike `anchor()`, which can only shift views -- it gives each view's
heading, the error the splat's pose refinement now has to find (cameras moved 0.5-1 deg)
[REAL-LC]. goto() can then close each leg on the fitted pose instead of `/odom_raw`, so the
circle closes. Report the fit's covariance per stop: near one wall, or with the thing and the
arm in the lidar plane, a direction can be unconstrained; then fall back to odometry.

**For driving.** Nav2 AMCL in Humble has `nav2_amcl::OmniMotionModel`
([plugins.xml](https://github.com/ros-navigation/navigation2/blob/humble/nav2_amcl/plugins.xml));
Beluga AMCL 2.1.1 (2026-04-20) is released for Humble with a "thick walls" pre-processing option
([beluga_amcl](https://index.ros.org/p/beluga_amcl/)). slam_toolbox's localization mode needs a
serialised pose graph, which `arena_v4` (a grid) lacks
([README](https://github.com/SteveMacenski/slam_toolbox/blob/humble/README.md)). The mirror
trap: the landmarks are a few per cent of the beams, so use 180-240 beams, recovery off, a known
start, and score any global fix against its 180-degree mirror with all beams.

**For docking near the box.** Nav2's docking (opennav_docking, Humble branch) reaches a staging
pose by global localization and does the rest on a detected dock pose in `odom`
(`docking_threshold` 0.05, `dock_prestaging_tolerance` 0.5)
([README](https://github.com/open-navigation/opennav_docking/blob/humble/README.md)) [CLAIM]: global
localization needs ~5 cm and a few degrees; the last centimetre is the camera's.

## 6. Measurement protocol

**Audit first (30 min, standing, read-only; [note 4 §5](<../research_notes/Base localization while turning/4 Calibration and measurement protocol.md>)).**
(1) `/scan0` fields: `time_increment`, `scan_time`, angle sign, intensities; (2) share of bins
unchanged between consecutive scans (16 % predicted); (3) `/scan0` vs `/scan1` stamps; (4)
`stamp - arrival` of all board topics for an hour and after an agent restart (offset, ppm),
plus `timedatectl`; (5) `ros2 node info /ekf_filter_node` and the yaml's `imu0`; (6) Yahboom's
merger source on the robot.

**Reference.** The standing fit on the fence and landmarks (2-3 scans averaged), validated once
against an independent reference: tape marks under a plumb line at 10 poses (repeatability
< 0.5 cm, < 0.2 deg; absolute by tape to two walls), or an overhead camera 2.5-3 m up with a
~10 cm ArUco/AprilTag on the base and the fence corners as the arena frame (est. 3-5 mm,
~0.3 deg; standing comparisons only, since its clock is not the robot's).

**Runs** (record `/scan0 /scan1 /odom_raw /imu/data_raw /tf /tf_static`, battery > 11 V):
1. *In-place turns*: 3 turns CW, 3 CCW at 0.15, 0.3, 0.6 rad/s with 2 s stands between, then
   60 s of sinusoidal yaw (+-0.4 rad/s, 0.2 Hz). Gives: wheel and gyro yaw scales (against the
   standing fits), heading error per scan vs. turn rate and direction, the mounting yaw (CW/CCW
   mean), the time offset (CW/CCW half-difference over the rate, and the cross-correlation),
   the seam (wall residual against bin index).
2. *The scan's circle without the arm*: 12 stops of 30 deg round a marked spot at 0.38 m, the
   standing fit at each stop, three times. Gives per-leg heading and position error of
   `/odom_raw`, of the EKF, and of gyro-heading odometry.
3. *A 1.5 x 1.5 m square*, 3 times each way, fits at the corners: UMBmark's systematic error
   ([Borenstein & Feng 1996](https://cs.au.dk/~ocaprani/legolab/DigitalControl.dir/NXT/Lesson10.dir/paper58.pdf))
   plus the forward and lateral scales.

**Pass criteria.** Scan stops within 1 cm and 0.5 deg of the reference, then the box's and the
can's sizes against the ruler and no doubled edges in the fused model; maps with walls <= 3 cm
thick and the fence 3.200 x 2.800 +- 1 cm over three loops driven at up to 0.6 rad/s; AMCL
within ~5 cm and 3 deg at 0.15 m/s and 0.3 rad/s.

## 7. Fixes, ranked

| # | fix | where | effort | expected gain (est.) | how to check |
|---|---|---|---|---|---|
| 1 | timing audit (6 checks) | robot, read-only | 30 min | decides 3, 4, 8, 11 | the predictions in §6 |
| 2 | pose from a standing fence fit at every scan stop | new fit module; m3scan.run | 1-2 days | scan poses 8-13 cm/circle -> ~0.5 cm, ~0.2 deg, heading included | run 2; box/can sizes |
| 3 | host re-stamper for the 4 board topics; EKF and merger on the copies | scan_relay.py; EKF yaml copy | 0.5-1 day | no dropped scans; camera vs base 1.3-2.7 deg -> ~0.1; drift tracked | offset log; slam drops; RTAB-Map walls |
| 4 | scan stamp shifted to mean point time; mapper gets only scans below 0.15 rad/s | scan_relay.py | 2 h | 1.4-2.9 deg lag and most smear out of the map | wall thickness at 0.6 rad/s driving |
| 5 | second lidar's mounting applied; common yaw vs wheels measured | lidars.yaml | 2 h | walls ~2-3 cm thinner; no heading bias | wall thickness; straight drive along a wall |
| 6 | EKF fuses gyro yaw rate (wheels vx, vy only); gyro scale calibrated, bias re-estimated at stops | relay; EKF yaml | 0.5 day | per-leg yaw error -> ~0.1-0.2 deg | runs 1-2 |
| 7 | goto() closes each leg on the fitted pose | m3bridge / m3scan | 0.5 day after 2 | the circle closes within ~1 cm | run 2 |
| 8 | per-point de-skew from the inferred write head | scan_relay.py | 1-2 days | seam +-1.4 deg at 0.3 rad/s -> ~0.2-0.4 deg | wall residual vs bin |
| 9 | wheel scales (forward, lateral, yaw) | relay or config_robot.py | 2 h | systematic part only: small by our data | the square |
| 10 | AMCL-Omni or Beluga on the arena map | Nav2 | 1 day | driving pose ~1-3 cm, ~0.5-1 deg | overhead camera / tape at stops |
| 11 | firmware: periodic sync; one scan per revolution with time_increment | STM32 firmware | 2-4 days + flash risk | the clean form of 3 and 8 | same as 3 and 8 |
| 12 | Cartographer with per-point unwarping | after 11 | 2 days | better maps while moving, maybe | wall thickness |

## 8. Configs to try

Re-stamper / relay (new parameters in `scan_relay.py`; all values to be replaced by the audit):
```yaml
offset_window_s: 60.0        # running max of (stamp - arrival) on /imu/data_raw
offset_jump_reset_s: 0.02    # Jetson NTP step: restart the window
transfer_s: {imu: 0.0016, odom: 0.0036, scan: 0.0204}   # 2 Mbaud, by message size
scan_mean_age_s: 0.083       # snapshot model; replace by the cross-correlation result
gate_wz_rad_s: 0.15          # /scan_map gets only scans taken below this turn rate
gyro_scale: 1.000            # from run 1
bias_from_standstill: true   # re-estimate gyro z bias whenever wheel speeds are zero
```

EKF (a copy of Yahboom's yaml; robot_localization Humble):
```yaml
ekf_filter_node:
  ros__parameters:
    frequency: 30.0
    sensor_timeout: 0.2
    two_d_mode: true
    world_frame: odom
    odom_frame: odom
    base_link_frame: base_footprint
    odom0: /odom_raw_h            # re-stamped
    odom0_config: [false, false, false, false, false, false,
                   true,  true,  false, false, false, false,   # vx, vy -- not vyaw
                   false, false, false]
    imu0: /imu/data_raw_h         # re-stamped, scaled, bias removed
    imu0_config: [false, false, false, false, false, false,
                  false, false, false, false, false, true,     # vyaw only
                  false, false, false]
```

slam_toolbox (mapping, after re-stamping): `scan_topic: /scan_map` (gated), `scan_queue_size: 1`,
`transform_timeout: 0.2`, `minimum_travel_distance: 0.2`, `minimum_travel_heading: 0.15`,
`resolution: 0.02` -- and retry `scans:=lidars` (each lidar its own scan, the best in replay,
3.198 x 2.799 m), since the dropping that killed it live should be gone. Save with the pose graph
(`serialize_map`) so localization mode becomes possible.

AMCL (Humble; check names against the Humble docs before use):
```yaml
amcl:
  ros__parameters:
    robot_model_type: "nav2_amcl::OmniMotionModel"
    base_frame_id: base_footprint
    laser_model_type: likelihood_field
    max_beams: 240
    sigma_hit: 0.05
    z_hit: 0.95
    z_rand: 0.05
    laser_likelihood_max_dist: 0.5
    do_beamskip: true
    alpha1: 0.1     # rotation from rotation: set from run 1
    alpha2: 0.05
    alpha3: 0.05
    alpha4: 0.05
    alpha5: 0.1     # strafe from translation: mecanum's weak axis
    min_particles: 300
    max_particles: 2000
    update_min_d: 0.05
    update_min_a: 0.05
    recovery_alpha_slow: 0.0
    recovery_alpha_fast: 0.0
    transform_tolerance: 0.3
    set_initial_pose: true
```

Driving: until per-point de-skew exists, cap turns at ~0.15 rad/s in precise phases (mapping,
the last approach) and keep 0.6 rad/s for transit.

## 9. Risks

- **The firmware model is Yahboom's sample, not our firmware.** The shipped one sends 270 deg in
  500 bins on two topics and may stamp or time differently; the audit (fix 1) comes before any
  change that assumes 83 ms or 16 % stale bins.
- **The drift rate rests on two observations**; part of the 1.1 s may have been a Jetson NTP
  step. The re-stamper's log settles it.
- **A re-stamp with the wrong sign makes things worse.** Develop it on a recorded bag, compare
  wall thickness and fence size before and after, then run it live.
- **Standing fits can degenerate** near one wall or when the scanned thing, the arm or the other
  robot block the lidar plane (14.5 cm); a 15+ cm thing sits in that plane. Use the fit's
  covariance, fall back to odometry for the unconstrained direction.
- **The EKF change can break Yahboom's navigation** if it shares the yaml: keep our copy separate.
- **Flashing the board** (fix 11) needs an ST-Link and can leave the base without odometry until
  re-flashed; PLAN treats it as a last resort.
- **The mirror pose** with few beams or after a kidnapping (the robot carried by hand also
  doubles walls in mapping): many beams, recovery off, an independent check.
- **Battery**: the board went silent near 10.4 V twice; low voltage also changes slip.

## Conclusion

The M3's base is badly timed rather than badly built: its lidar scans are 167 ms snapshots
stamped when sent, on a board clock synced once and drifting ~48 ms an hour, and every consumer
in our stack takes the stamp at face value. At goto's 0.3 rad/s that is ~1.4 deg of lag and
+-1.4 deg of smear per scan, twice that at the drive page's 0.6 rad/s; the wheels add ~1.6 cm
per 30-degree leg, mostly at random. For the object scan the cheapest complete answer is not to
fix motion but to avoid it: localize on the fence and landmarks at every stop, standing, where
none of the timing terms exist, and take the heading from there as well as the position --
expected ~0.5 cm and ~0.2 deg, within the 1 cm / 0.5 deg target. For mapping, Nav2 and later
the Healthcare Challenge, the order is: audit the timing (30 minutes), re-stamp the board's
topics on the Jetson, gate or de-skew scans taken while turning, apply and measure the lidars'
mounting, put a calibrated gyro into the EKF, then AMCL-Omni; firmware changes only if the
host-side fixes leave the seam visible. Open: whether the shipped firmware matches the sample,
whether the EKF has been running without the IMU, the true drift rate, and how far the standing
fit really gets -- each answered by one of the measurements in section 6.
