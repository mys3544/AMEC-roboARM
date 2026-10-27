# 3. De-skew, scan matching and localization in the fenced arena

Scope: what exists in ROS 2 Humble to de-skew 2D scans and whether the mappers and localizers
we can run handle motion inside a scan; how our board's scans could be de-skewed; which
localization fits a 3.2 x 2.8 m fence with three landmarks (standing scan-to-model ICP,
slam_toolbox localization, AMCL / Beluga, Cartographer); the 180-degree mirror trap; and what
docking near the box needs. Timing is in note 2, calibration and the protocol in note 4. Labels
as in note 1.

## 1. In Humble, only Cartographer de-skews; for the rest we write ~100 lines ourselves

### Takeaway
No maintained ROS 2 node de-skews a 2D `LaserScan`. The building block exists
(`laser_geometry::transformLaserScanToPointCloud` interpolates the transform per point into any
frame), Cartographer unwarps internally, and slam_toolbox, AMCL and the merger do none. A
de-skew step in our own `scan_relay.py` -- each point into `odom` at its own time, then back into
`base_link` at one common time -- is the practical route, and it must know each point's time.

Evidence:
- `laser_geometry` (Humble): two `lookupTransform`s, at `stamp` and at `stamp + (n-1) *
  time_increment`, then per point `setInterpolate3` and `slerp` by `ratio = i / (n-1)`
  ([laser_geometry.cpp](https://github.com/ros-perception/laser_geometry/blob/humble/src/laser_geometry.cpp)).
  Use it with target `odom` (a fixed frame) and a correct `time_increment`; its sign follows
  the index order, so a clockwise lidar published with a negative `angle_increment` needs care.
- `laser_deskew` (scan + tf -> `/deskewed_scan`) is ROS 1 (Melodic CI)
  ([rsasaki0109/laser_deskew](https://github.com/rsasaki0109/laser_deskew)). Other hits were 3D
  (Livox, MOLA) [searched: web search "ROS 2 2D laser scan deskew ... humble"].
- Cartographer: each point is transformed by `extrapolator_->ExtrapolatePose(time_point)` with
  `time_point = time + point_time`
  ([local_trajectory_builder_2d.cc](https://github.com/cartographer-project/cartographer/blob/master/cartographer/mapping/internal/2d/local_trajectory_builder_2d.cc));
  the docs: distances "are measured over a certain period of time, while the robot is actually
  moving" and more messages per revolution unwarp better
  ([algorithm walkthrough](https://google-cartographer-ros.readthedocs.io/en/latest/algo_walkthrough.html)).
- For a lidar read on the Jetson, drivers do fill the timing: the YDLIDAR ROS 2 driver stamps
  the first point and sets `time_increment = scan_time / count`, with `scan_time` measured
  between revolutions
  ([ydlidar_ros2_driver](https://github.com/YDLIDAR/ydlidar_ros2_driver/blob/humble/src/ydlidar_ros2_driver_node.cpp),
  [CYdLidar.cpp](https://github.com/YDLIDAR/YDLidar-SDK/blob/master/src/CYdLidar.cpp));
  `rplidar_ros` (the X3's kind) stamps the start of the blocking grab and sets
  `time_increment = scan_time / (node_count - 1)`
  ([rplidar_node.cpp](https://github.com/Slamtec/rplidar_ros/blob/ros2/src/rplidar_node.cpp)).
  The M3's board publishes neither (note 1).

## 2. Our scans first need a point time: three ways to get it

### Takeaway
A snapshot scan (note 1) can be de-skewed only if we know where the write head was when it was
sent. Cheapest first: (a) shift each scan's stamp to the points' mean time and drop or
down-weight scans taken while turning fast; (b) infer the write head from the bins that did not
change since the previous scan (the ~16 % stale arc ends at the head); (c) change the firmware to
publish once per revolution, stamped at the first point, with `time_increment` and `scan_time`.

Notes:
- (a) removes the mean lag (1.4 deg at 0.3 rad/s) but not the seam; with a turn-rate gate (only
  scans taken below ~0.1-0.15 rad/s go to mapping; all go to the drive page's safety stop) it is
  enough for mapping and for the object scan, which stands still at its views.
- (b) works only while the scene changes enough that a refreshed bin differs from its last value
  (1 mm resolution: true for most bins with real noise; check the stale fraction on the robot
  first: ~16 % predicted for a 6 Hz rotation and a 140 ms timer). Point age is then
  `((head - angle) mod 360) / 360 / f_rot`, direction by the lidar's clockwise rotation.
- (c) is the clean fix: the T-mini packet carries a CT byte whose bit 0 marks the packet that
  starts a revolution [CLAIM, YDLIDAR protocol, not re-checked]; the sample parser keeps CT but
  ignores it ([Read Lidar data](https://www.yahboom.net/study/ROSMASTER-M3PRO/20511)). Costs a
  firmware build and flash.

## 3. For the object scan's stops: standing scan-to-model ICP on the fence and landmarks

### Takeaway
At each stop the base stands still; a scan taken at least one revolution plus the latency after
the wheels stop has no timing error at all. Matching the merged, bias-corrected scan (both
lidars with calibrated mountings) to a geometric model of the arena -- four wall lines of the
measured 3.2 x 2.8 m and three landmark outlines from `arena_v4` -- gives the base pose in all
three degrees. Expected: ~0.5 cm and ~0.2 deg (est., to be measured), against ~1.6 cm per leg and
~10 cm per circle from the wheels.

Evidence and reasoning:
- `fence_fit.py` already fits the four walls in maps (rms 1.1-2.1 cm per wall in the map,
  [PLAN.md M2](../../PLAN.md)) [REAL-LC]; a scan sees a few hundred wall points per standing
  scan; est. statistical pose noise is millimetres, so the floor is set by systematic terms:
  the range-bias residual (~0.3 cm), mounting yaw (after calibration ~0.1 deg), the walls'
  straightness, the map's own error.
- Heading from walls is strong: two walls 2.8-3.2 m long seen end to end constrain yaw to
  ~0.1-0.2 deg if each wall's offset is good to ~0.5 cm (est.: atan(0.5 cm / 3 m) = 0.1 deg).
- Near a wall or behind the thing, the view can degenerate (one wall only: position along it
  free); the landmarks then decide. Report the ICP's Hessian / covariance per stop and fall back
  to odometry when one direction is unconstrained.
- Settling: use a scan stamped >= 167 ms (one revolution) + transfer after the wheels stopped;
  the arm's move to the first view plus `settle_close_s` (0.35 s,
  [m3pick.py](../../m3pro/roboarm/m3pick.py)) already take longer, so this costs nothing extra.
- Why this helps the object scan (topic 7): `m3scan.anchor()` shifts each view so the thing's
  reading lands on one point -- it removes position drift but not heading ([m3scan.py](../../m3pro/roboarm/m3scan.py)).
  A wrong heading turns a view's points about the thing's centre: est. 5 deg at a 8 cm
  half-diagonal is 7 mm on the surface. A lidar pose per stop gives the heading too, so the
  fuse starts from poses good to ~0.5 cm / ~0.2 deg instead of leaving the yaw to ICP and the
  splat's pose refinement (which moved cameras 0.5-1 deg on the box and can,
  project memory m3pro-object-scan) [REAL-LC].

## 4. While driving: AMCL (Omni) or Beluga on the fence map, slam_toolbox localization second

### Takeaway
For continuous localization, Nav2's AMCL with the omnidirectional motion model (in Humble as a
plugin), the likelihood field with many beams and no random-particle recovery is the standard
choice; Beluga AMCL 2.1.1 (released for Humble, April 2026) is a drop-in with a "thick walls"
pre-processing option. slam_toolbox's localization mode needs a serialised pose graph, which
`arena_v4` does not have. Cartographer unwarps but is no longer developed upstream; worth it
only if the firmware gives per-point times. None of them fixes the scan timing: re-stamp first.

Evidence:
- AMCL Humble plugins: `nav2_amcl::DifferentialMotionModel`, `nav2_amcl::OmniMotionModel`
  ([plugins.xml](https://github.com/ros-navigation/navigation2/blob/humble/nav2_amcl/plugins.xml));
  Omni noise: rotation `sqrt(alpha1 dr^2 + alpha2 dt^2)`, translation `sqrt(alpha3 dt^2 + alpha4
  dr^2)`, strafe `sqrt(alpha4 dr^2 + alpha5 dt^2)`
  ([omni_motion_model.cpp](https://github.com/ros-navigation/navigation2/blob/humble/nav2_amcl/src/motion_model/omni_motion_model.cpp));
  defaults alpha1-5 0.2, `laser_model_type likelihood_field`, `sigma_hit 0.2`, `max_beams 60`,
  `update_min_d 0.25`, `update_min_a 0.2`, `recovery_alpha_slow/fast 0.0`,
  `transform_tolerance 1.0`, `do_beamskip false`
  ([Nav2 AMCL configuration](https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/others/configuring_amcl/)).
- Beluga AMCL 2.1.1 (2026-04-20), released for Humble, Jazzy and Kilted; 2.1.0 added the
  likelihood-prob model and `model_unknown_space`; 2.1.1 "Thick walls pre-processing feature
  added", "Add paranoid mode"; NDT sensor model since 2.0.0
  ([index.ros.org, beluga_amcl](https://index.ros.org/p/beluga_amcl/)). Its omnidirectional
  motion model is listed in the Beluga docs ([Beluga](https://ekumen-os.github.io/beluga/)) [2nd,
  search summary].
- slam_toolbox localization: `mode: localization`, `map_file_name` (a serialised pose graph),
  `map_start_pose` or `map_start_at_dock`; a rolling buffer of recent scans is added to the
  loaded graph and expires; "needs considerable tuning" and good odometry; the author
  recommends AMCL for most users ([README, humble](https://github.com/SteveMacenski/slam_toolbox/blob/humble/README.md)).
  `arena_v4` is a grid only, no pose graph (project memory m3pro-arena-mapping): a new mapping
  run saved with `serialize_map` is needed first.
- Cartographer: "no longer actively maintained ... no new development"
  ([cartographer README](https://github.com/cartographer-project/cartographer)); packages
  still released into Humble (2.0.9004, May 2025,
  [ros2-gbp/cartographer-release](https://github.com/ros2-gbp/cartographer-release)) [2nd for the
  date].
- Published accuracy of AMCL in small arenas was not found with numbers comparable to ours
  (search budget ran out); our own measurement (note 4) decides. PLAN's estimate for the fence
  cue is 1-3 cm and ~1 deg ([PLAN.md M2.2](../../PLAN.md)) [CLAIM, ours].

## 5. The 180-degree mirror trap

### Takeaway
A bare 3.2 x 2.8 m rectangle fits its 180-degree turn exactly; only the three landmarks tell the
two apart, and they are a few per cent of the beams. Keep enough beams that the landmarks are in
every update, turn random-particle recovery off, start from a known pose, and for any global
(re-)initialisation score the pose and its mirror with all beams and require a clear margin --
or ask the floor print.

Notes:
- est.: a ~25 cm landmark at 1.5 m spans ~10 deg, i.e. ~18 beams of 0.54 deg per lidar, but
  only ~2-3 of AMCL's default 60 beams; at `max_beams` 180-360 the landmarks carry 5-10 % of the
  weight. With `sigma_hit` 0.05 m, a landmark beam that misses by 20 cm counts heavily.
- Recovery off (`recovery_alpha_slow = recovery_alpha_fast = 0`, the defaults) prevents random
  particles from finding the mirror pose ([PLAN.md M2.2](../../PLAN.md) already says so).
- The independent arbiter: the track print seen by the arm camera (PLAN M2.2) or the landmark
  check in our own ICP (residual on the landmark points for both hypotheses).

## 6. Docking near the box needs relative perception, not centimetre global localization

### Takeaway
Nav2's docking server (opennav_docking, with a Humble branch) drives to a staging pose by global
localization and does the last part on a detected dock pose in the `odom` frame. So global
localization needs only to reach staging (it skips navigation within 0.5 in combined x, y, yaw
distance by default); the final ~1 cm comes from the camera on the box (an AprilTag or our
detector) and the odometry over the last ~0.5 m.

Evidence: external detection via `detected_dock_pose`; `docking_threshold` 0.05; "smooth
odometry frame (`odom`) rather than `map`" recommended as `fixed_frame`; `staging_x_offset`
-0.7 m; `dock_prestaging_tolerance` 0.5; the blind (database-pose) approach "may have a reduced
success rate on a real robot" ([opennav_docking README](https://github.com/open-navigation/opennav_docking/blob/humble/README.md);
branches `humble`, `main`) [CLAIM].

## 7. Turn slowly where precision counts

### Takeaway
Every timing error in note 1 scales with the turn rate. Until de-skew exists, cap the turn at
~0.15 rad/s in precise phases (mapping, the last approach) and keep 0.6 rad/s for transit; for
mapping, gate scans by turn rate rather than slow the whole drive.

est.: at 0.15 rad/s the mean lag is 0.7 deg and the seam 1.4 deg (half of 0.3 rad/s); the scan's
legs already average ~0.13 rad/s (30 deg in ~4 s at 5 cm/s).
