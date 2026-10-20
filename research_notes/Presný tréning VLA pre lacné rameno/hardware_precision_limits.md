# Physical precision limits of low-cost serial-bus-servo arms, and what a learned policy can and cannot compensate

Notes compiled 2026-10-20. Conversion used throughout: end-effector error ~ r x angle(rad). At r = 25 cm:
1 deg = 4.36 mm, 0.088 deg (one 12-bit count) = 0.38 mm, 0.25 deg = 1.09 mm, 0.87 deg = 3.8 mm. At r = 30 cm multiply by 1.2.
This is a per-joint, worst-case lever (shoulder/base joints see the full reach; wrist joints see only a few cm),
and errors of several joints add up along the chain. All such mm figures below are derived by me and marked as such.

## 1. Resolution, backlash, deadband, repeatability and load sag of common serial bus servos

### Takeaway
On every common hobby bus servo, the 12-bit encoder (0.088 deg, ~0.4 mm at 25 cm) is NOT the limiting factor; gear backlash
(~0.6-0.9 deg measured on the STS3215, 0.25 deg spec on the XM430) and load-dependent deflection (~2 deg on an STS3215 at
one-third of its stall torque) are each worth several millimetres at 25 cm reach, an order of magnitude more than quantisation.
Potentiometer-feedback servos (Yahboom, Hiwonder LX-series) add sensor noise/nonlinearity on top of that.

### Cited Findings
**Feetech STS3215 (SO-100/SO-101, AhaRobot)**
- 12-bit magnetic encoder, 4096 steps/rev, 0.088 deg per step; 1 count at 10 cm radius ~ 0.153 mm — [Robo9 STS3215 test (2025)](https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/)
- Measured backlash (dial indicator, ~30 g rubber-band tension): 1.3 mm at an 86 mm lever = ~0.87 deg, vs datasheet <=0.5 deg; a second unit measured ~0.57 deg (unit-to-unit variation) — [Robo9 STS3215 test](https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/)
- Second Robo9 measurement series, single STS3215 at 100 mm lever: unloaded 7.03 counts / 0.62 deg / 1.07 mm; loaded (~0.3 kgf, ~10 % of stall) 14.78 counts / 1.30 deg / 2.26 mm — backlash roughly doubles under load — [Robo9 backlash compensation (images dated Mar 2026)](https://robonine.com/backlash-compensation-in-sts3215-servo-actuators/)
- Firmware deadband reported as 10 encoder counts: "commands under 10 steps produce no motion" — [Robo9 STS3215 test](https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/). CONFLICT: a third-party STS memory-table analysis lists Positive/Negative Dead Zone registers (addr 26/27) with default 1, range 0-16, unit 0.087 deg — [AIFitLab STS memory table](https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-sts-servo-memory-table-analysis). The 10-count figure may be an effective (friction + deadband) minimum step rather than the register value; not resolved.
- Repeatability (10 cm vertical lever, 20-step moves, 3 cycles each direction, 3 iterations): tip deviation +-0.3 mm = ~0.17 deg = ~2 counts — [Robo9 STS3215 test](https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/)
- Position error under static load (mean absolute deviation): 22.5 steps (~2 deg) at 10 kg.cm, 30 steps (~2.6 deg) at 15 kg.cm (at 12 V) — [Robo9 STS3215 test](https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/)
- Stall ~35 kg.cm measured vs 30 kg.cm spec (12 V version); overload protection then cuts output to ~20 % of nominal — [Robo9 STS3215 test](https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/)
- Static friction: on AhaRobot (STS3215, 1:345, $15 each), a staircase of 0.175 deg steps (2x the minimum resolution) could NOT be tracked without a dithering feed-forward term; with dithering it tracked — [AhaRobot, arXiv 2503.10070 (Mar 2025)](https://arxiv.org/html/2503.10070v1)
- Variants: C001 7.4 V 19.5 kg.cm; C046 1/147 gear 14.4 kg.cm; C044 1/191 gear 27.4 kg.cm; C018 12 V 30 kg.cm; STS3250-C002 12 V 50 kg.cm 1/345 — [OpenELAB Feetech listing](https://openelab.io/collections/feetech-motor)
- SO-101 leader: back-driven joints "lock" into ~29 distinct positions per revolution even unpowered (mechanical detents), making intermediate values hard to capture in teleoperation; issue closed "not planned" May 2026 — [LeRobot issue #2074 (Sep 2025)](https://github.com/huggingface/lerobot/issues/2074)

**Dynamixel (Koch, ALOHA-class)**
- XM430-W350: backlash 15 arcmin (0.25 deg), 4096 pulse/rev — [ROBOTIS docs XM430-W350](https://docs.robotis.com/docs/dxl/model_reference/x_series/xm_series/xm430-w350/) (value from search excerpt of this page)
- XL430-W250 and XL330 (M077/M288): backlash listed as "-" (not specified) by ROBOTIS — [ROBOTIS XL430 shop page](https://en.robotis.com/shop_en/item.php?it_id=902-0135-000); [ROBOTIS XL330-M288 shop page](https://en.robotis.com/shop_en/item.php?it_id=902-0163-000)
- XL330-M288-T: 4096 pulse/rev (0.088 deg), contactless absolute encoder (ams AS5601, 12-bit), gear ratio 288.4:1, engineering-plastic gears, stall 0.52 N.m at 5 V; default Position P = 400 (KPP = value/128), I = 0, D = 0 — [ROBOTIS e-Manual XL330-M288](https://docs.robotis.com/docs/dxl/model_reference/x_series/xl_series/xl330-m288/)

**Hiwonder / Yahboom (potentiometer class)**
- Hiwonder LX-16A: control range 0-1000 over 0-240 deg (=> 0.24 deg/unit, derived), listed "servo accuracy 0.3 deg", 17 kg.cm @ 6 V / 20 kg.cm @ 7.4 V — [Hiwonder LX-16A manual](https://cdn.robotshop.com/rbm/a5e9ab51-86bc-497b-adac-003de3088fb7/9/92896585-5e83-4c7c-a663-650b81e5978b/67e9729b_lx-16a-serial-bus-servo-user-manual.pdf); [servodatabase LX-16A](https://servodatabase.com/servo/hiwonder/lx16a)
- Hiwonder LX-224: same 0-1000 / 240 deg mapping, accuracy 0.3 deg, 20 kg.cm @ 7.4 V — [servodatabase LX-224](https://servodatabase.com/servo/hiwonder/lx224)
- Yahboom 15 kg serial bus servo (YB-SD15M): range 300 deg (+15 deg) mapped to positions 96-4000, "imported high-precision potentiometer" for angle feedback, position read-back, 15 kgf.cm @ 7.4 V, 0.30 s/60 deg, 115200 baud, locked-rotor protection after 3 s; "Servo Accuracy: 1" with no unit — [Yahboom 15KG bus servo (RobotShop)](https://www.robotshop.com/products/yahboom-yahboom-15kg-serial-bus-smart-servo-robotic-arm-w-english-manual); [servodatabase YB-SD15M](https://servodatabase.com/servo/yahboom/yb-sd15m)
- Potentiometer drift with heat/age and up to "seven degrees non-linear" end-to-end is claimed by a magnetic-encoder servo vendor (vendor marketing, no test data) — [Team Tekin](https://www.teamtekin.com/phone/servos.html); typical hobby servo accuracy "+-1 deg to +-3 deg" — [Kpower article (vendor blog)](https://www.kpower.com/insight_servo/7477.html)

### Inferences
- Translation to millimetres at 25 cm (derived): STS3215 quantisation 0.38 mm/count; backlash 0.6-1.3 deg => 2.7-5.7 mm per proximal joint; static-load deviation 2 deg => ~8.7 mm (at 10 kg.cm load; an SO-101/Yahboom-size arm's shoulder probably sees a few kg.cm, so expect ~1 deg => ~4 mm sag; ESTIMATE by linear interpolation). XM430 0.25 deg spec => ~1.1 mm per joint. LX-16A 0.24 deg/unit => ~1 mm per unit, 0.3 deg accuracy => ~1.3 mm.
- The Yahboom 96-4000 over ~300 deg mapping gives ~13 counts/deg (~0.077 deg/count, derived) — nominally as fine as a 12-bit encoder, but read-back resolution of a potentiometer is not its accuracy; real accuracy is set by pot noise/nonlinearity, for which no number was found.
- An STS3215's repeatability at the joint (+-2 counts) is much better than its backlash (7-15 counts): if a joint always arrives from the same side, it repeats to ~0.3 mm at 10 cm (~0.8 mm at 25 cm, derived), while arrivals from alternating sides scatter by the full backlash.
- For our X3 (pot servos with read-back, 1-4 mm after hand-eye calibration) these magnitudes are consistent: 1-4 mm is roughly 1 backlash band at the proximal joints. For the M3 (reports SENT angles), the ~1.5 cm short grip at ~25 cm reach corresponds to ~3.4 deg of cumulative angular error (derived), the size expected from gravity sag + loaded backlash at shoulder and elbow (inference, not measured).

### Gaps
- No independent backlash/sag measurement found for Yahboom or Hiwonder bus servos; no measured potentiometer drift vs temperature numbers for any of them.
- No measured backlash for XL430/XL330 (ROBOTIS leaves it blank); no independent measurement of XM430 backlash vs its 0.25 deg spec.
- No stiffness (deg per N.m) number for any of these servos beyond the Robo9 load-deviation pair. The PMC test-stand article ("An open-source test stand for backlash measurement in low-cost UART servo motors", https://pmc.ncbi.nlm.nih.gov/articles/PMC13087586/) was behind a reCAPTCHA; its search excerpt says a single STS3215 has ~1-2 mm backlash at 100 mm lever (1.08 mm one direction, 2.27 mm the other), growing under load — unverified beyond the excerpt.

## 2. Measured repeatability/accuracy of SO-100/SO-101, Koch, ALOHA, myCobot, xArm Lite and similar

### Takeaway
Commercial low-cost arms advertise +-0.5 mm repeatability (myCobot 280, UFACTORY Lite 6), but the open-source servo arms
(SO-100/101, Koch) have no rigorous published repeatability figure; the few measurements that exist put an IK-driven SO-101 at
~5 mm steady-state error, and ALOHA's ViperX at 5-8 mm accuracy. Repeatability (same pose, same approach) is far better than
absolute accuracy on all of them.

### Cited Findings
- myCobot 280: repeatability +-0.5 mm; myCobot 320: +-0.3 mm (manufacturer FAQ, notes it depends on length of use) — [ThinkRobotics myCobot 280 listing](https://thinkrobotics.com/products/mycobot-280-jetson-nano-6-dof-collaborative-robot-online); [QVIRO myCobot 280 specs](https://qviro.com/product/elephant-robotics/arduino/specifications)
- UFACTORY Lite 6: repeatability +-0.5 mm, 440 mm reach, 600 g payload; xArm 6: +-0.1 mm — [UFACTORY spec sheet](https://www.ufactory.cc/_files/ugd/896670_42a4e0a7264a494090202c8aadf91b59.pdf); [RobotShop Lite 6](https://www.robotshop.com/products/ufactory-6-axis-robot-arm-lite-6)
- ALOHA (ViperX 300, ~$5,600 per arm): "an accuracy of 5-8mm"; the paper gives no repeatability figure; "Millimeters of error would lead to task failure" for the condiment-cup task — [Zhao et al., ALOHA/ACT, RSS 2023](https://ar5iv.labs.arxiv.org/html/2304.13705)
- SO-101, plain IK at 200 Hz (community project, undated): p99 tracking error while moving 12.6 mm at 10 cm/s and 44.3 mm at 25 cm/s; steady state after settling ~5.5 mm / ~5.4 mm — [so101-residual-grasp (GitHub)](https://github.com/OmarAds2002/so101-residual-grasp)
- SO-100 (FidEx paper): forward kinematics from raw encoders gives end-effector translation error 0.18 vs 0.06-0.07 for image-based state estimation (units not stated in the extracted table); naive move control error 0.37 vs 0.21-0.22; grasp test 9/10 vs 4/10 for middle-pose calibration — [Fiducial Exoskeletons, arXiv 2601.08034 (Jan 2026)](https://arxiv.org/html/2601.08034)
- Forte (3D-printed, cable/belt drive, <$215 material, 0.63 kg payload, 0.467 m reach): "sub-millimeter repeatability" (no number in abstract) — [Chebly et al., arXiv 2507.15693 (Jul 2025)](https://arxiv.org/abs/2507.15693)
- A low-accuracy arm used for depth-based visual servoing is reported (search excerpt of the PDF) at ~3 cm position-control accuracy with low repeatability; with RGB-D visual servoing it grasped successfully on >95 % of trials — [Hofer et al., arXiv 1612.03784 (Dec 2016)](https://arxiv.org/abs/1612.03784) (the 3 cm figure is from the search excerpt, not verified in the abstract)
- Koch follower: 2x XL430-W250 ($100) + 4x XL330-M288 ($96), ~$250 follower — [Koch low_cost_robot repo](https://github.com/AlexanderKoch-Koch/low_cost_robot)
- An open SO-101 "error anatomy" project (ISO 9283-style, ground-truth-independent measurement) was at hardware bring-up with no data as of 2026-09 — [so101-error-anatomy (GitHub)](https://github.com/tucasteixeira-debug/so101-error-anatomy)

### Inferences
- The +-0.5 mm figures for myCobot/Lite 6 are ISO-style repeatability (same pose, same approach), not accuracy; absolute accuracy of these arms is typically several times worse and not published (inference).
- Our X3 at 1-4 mm after hand-eye calibration is already in the band of a ViperX (5-8 mm accuracy) and of an IK-driven SO-101 (~5.5 mm steady state); it is not obviously worse than the open-source reference arms.

### Gaps
- No rigorous repeatability (dial indicator / mocap, ISO 9283) for an assembled SO-100/101 or Koch arm was found; Koch and ALOHA repeatability are unpublished.
- No accuracy figure (as opposed to repeatability) for myCobot 280 or Lite 6.
- Forte's numeric repeatability is in the full paper, not read.

## 3. Systematic (calibratable) vs random error sources

### Takeaway
Link-length and joint-offset errors, gravity sag (a function of configuration and payload) and the direction-dependent half of
backlash are systematic and can be modelled or calibrated; the residual within-backlash position, deadband stops, stiction,
thermal drift, potentiometer noise and grasp slip are random or history-dependent and set the floor. On cheap arms the dominant
error (backlash/sag) depends on configuration, load and motion history, so a single static model only partly removes it.

### Cited Findings
- Backlash: "each joint has a margin of rotation in which encoders register no change"; this unmeasured motion "accumulates down the kinematic chain, leading to large state error at the end-effector"; joint offsets are often set by matching a few target poses, giving imperfect mappings — [FidEx, arXiv 2601.08034](https://arxiv.org/html/2601.08034)
- Backlash roughly doubles under load (STS3215 1.07 mm -> 2.26 mm at 100 mm) — load-dependent, i.e. configuration-dependent — [Robo9 backlash compensation](https://robonine.com/backlash-compensation-in-sts3215-servo-actuators/)
- Load deflection (2 deg at 10 kg.cm) is repeatable for a given load — systematic in configuration and payload — [Robo9 STS3215 test](https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/)
- Thermal: static 15 kg.cm load -> 48 deg C in 10 min, then stable; +-90 deg oscillation -> 60 deg C in 50 min, overheating at ~71 deg C; no built-in thermal shutdown observed — [Robo9 STS3215 test](https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/)
- Configuration-dependent calibration (locally interpolated POE models across the workspace) lists motor backlash among non-geometric, configuration-varying errors and cuts max positioning error by >50 % (industrial arms) — [arXiv 2510.19962 (Oct 2025)](https://arxiv.org/html/2510.19962)
- A low-accuracy arm's error "has low repeatability, so it is not possible to learn a deterministic model for the error" (search excerpt) — [Hofer et al., arXiv 1612.03784](https://arxiv.org/abs/1612.03784)
- myCobot's repeatability "depends on the length of use" (wear) — [ThinkRobotics myCobot 280](https://thinkrobotics.com/products/mycobot-280-jetson-nano-6-dof-collaborative-robot-online)
- Potentiometer drift with heat and age is asserted (vendor, no data) — [Team Tekin](https://www.teamtekin.com/phone/servos.html)

### Inferences
- Systematic, calibratable: link lengths, joint zero offsets (homing), hand-eye transform, camera intrinsics, gravity sag vs configuration and payload, the mean (direction-dependent) backlash offset, potentiometer nonlinearity (fixed curve per servo).
- Random / history-dependent: where within the backlash band a joint rests when the approach direction or external load changes; deadband/stiction stops (the last command below ~2-10 counts is ignored); potentiometer noise; thermal drift during a session; object slip in the gripper; tracking lag while moving (speed-dependent, e.g. 12.6 vs 44.3 mm p99 on the SO-101).
- On the M3, which reports sent rather than measured angles, every one of these errors is invisible to proprioception; only the camera can observe them. On the X3, read-back sees the servo output shaft (after the gears, so it sees sag), but a potentiometer's own error is not observable from it.

### Gaps
- No quantitative thermal drift (deg per deg C) measurement for any bus servo found.
- No study splitting an assembled low-cost arm's error budget into these components (the so101-error-anatomy project intends to, no data yet).

## 4. What a closed-loop visual policy can and cannot compensate

### Takeaway
A wrist-camera closed-loop policy (learned or classical visual servoing) can remove any error that is observable in the image
and larger than the arm's smallest executable step: calibration offsets, sag, the static part of backlash, object position
error. It cannot command motions finer than the deadband/stiction step (~0.2-0.9 deg on an STS3215 => ~1-4 mm at 25 cm),
cannot hold a position against a contact force larger than the servo's stiffness allows, and cannot correct errors that occur
after the camera loses sight (occlusion in the final centimetres, slip during the close).

### Cited Findings
- ALOHA: humans "do not have industrial-grade proprioception" yet do delicate tasks by "actively compensating for errors"; the closed-loop policy maps webcam images to actions so it "can react to different positions of the cup and lid"; ACT on a 5-8 mm-accuracy arm reached 80-90 % on tasks like battery slotting with 10 min of demonstrations — [Zhao et al., ALOHA/ACT, RSS 2023](https://ar5iv.labs.arxiv.org/html/2304.13705)
- ALOHA limits: low-cost motors "cannot generate enough torque" for high-force tasks — [ALOHA/ACT](https://ar5iv.labs.arxiv.org/html/2304.13705)
- Image-based state estimation on an SO-100 (no extra sensors besides a printed fiducial exoskeleton) reduced control error ~40-45 % vs naive FK moves and lifted grasp success from 4/10 to 9/10 — [FidEx, arXiv 2601.08034](https://arxiv.org/html/2601.08034)
- Visual servoing on a low-cost SCARA: "The tolerance cannot be zero because of measurement noise, backlash and limited robot precision"; the authors added compensation at each joint direction change because elbow/shoulder backlash was noticeable — [Visual servoing for low-cost SCARA robots using an RGB-D camera (c. 2018)](https://www.researchgate.net/publication/325655900_Visual_servoing_for_low-cost_SCARA_robots_using_an_RGB-D_camera_as_the_only_sensor)
- Sub-resolution steps: 0.175 deg incremental targets could not be followed without dithering (static friction) — [AhaRobot, arXiv 2503.10070](https://arxiv.org/html/2503.10070v1)
- VLA benchmark on SO-101 (pi0.5, SmolVLA, Wall-X, ACT; 100 demos/task, 20 episodes): average success 32.5-56.25 %; "Precision Pen Placement" 45-80 %; Precision Misalignment is the least frequent failure category (0-15 % of failures), the dominant failures are execution instability (grasp instability, repetition loops, state mismatch); the paper names actuator noise, joint backlash, control latency, jitter, limited repeatability and imperfect visual-action calibration as hardware factors — [Yu & Qiu, arXiv 2606.08881 (Jun 2026)](https://arxiv.org/html/2606.08881)
- SO-101 ACT pick-and-place (simulation): failures ended 1-3 cm off the cube centre vs 1 mm for the scripted expert — lateral grasp precision was the remaining failure mode — [aviadarn/so101-lerobot (GitHub)](https://github.com/aviadarn/so101-lerobot)
- Adding servo load/current to an ACT policy for SO-101 test-tube insertion: 6/20 (positions only) vs 5/20 (with load+current), statistically inconclusive — [FaithQin/so101-tube-insertion (GitHub)](https://github.com/FaithQin/so101-tube-insertion)
- Tactile fusion for 0.1 mm-clearance insertion: ReTac-ACT 80 % vs plain ACT 15 % (not a low-cost arm) — [ReTac-ACT, arXiv 2603.09565](https://arxiv.org/pdf/2603.09565)

### Inferences
- CAN compensate (if visible and above the minimum step): hand-eye/kinematic offsets, sag, mean backlash, object pose error, slow thermal drift — the policy simply closes the loop on the image error. This is why ALOHA works at 5-8 mm arm accuracy.
- CANNOT (hard floors): (a) corrections smaller than the deadband/stiction step — a policy that outputs a 1-count correction gets no motion, so it oscillates or stalls (approach-from-one-side or dithering is the hardware/controller fix, not the policy); (b) the random position inside the backlash band when contact forces flip the gear flank; (c) compliance under contact: at ~2 deg per 10 kg.cm an STS3215 joint yields several mm when pushing; (d) anything after the camera view is occluded by the fingers or the object; (e) tracking lag at speed (12.6 mm p99 at 10 cm/s on SO-101), unless the policy slows down near the target.
- Practical rule (inference): the achievable closed-loop floor is roughly max(1 executable step, half the backlash band, camera pixel footprint) at the working distance — for an STS3215-class arm at 25 cm that is ~1-2 mm, matching our X3's 1-4 mm classical result.
- The SO-101 VLA benchmark suggests that for current VLAs, behaviour-level failures dominate precision failures; hardware precision becomes the bottleneck only for tight-tolerance tasks (insertion, placement).

### Gaps
- No paper quantifies, in mm, the residual error of a learned wrist-camera policy on an SO-101/Koch-class arm versus the open-loop arm error.
- No study measured how deadband interacts with action-chunked policies (e.g., chattering near the goal) on these servos.

## 5. Kinematic calibration, servo control settings and direction-of-approach tricks

### Takeaway
Camera-based calibration or image-based joint-state estimation roughly halves (40-75 %) end-effector error on an SO-100; mechanical
preload (two servos per joint) cuts STS3215 backlash ~7x; software tricks — a fixed approach direction, a dithering
feed-forward, sensible P gain and deadband registers — attack deadband and backlash at zero cost. LeRobot itself lowers the
STS3215 P gain (16 instead of 32) to stop shaking, which trades stiffness for smoothness.

### Cited Findings
- Dual STS3215 with home-offset preload (each gear train on the opposite flank, preload ~3 % of stall ~0.9 kg.cm): loaded backlash 14.78 -> 2 counts (2.26 -> 0.31 mm at 100 mm), unloaded 7.03 -> 1 count (1.07 -> 0.15 mm); coupled without offset 8 / 6 counts; ~80 % of combined torque retained, no overheating — [Robo9 backlash compensation (Mar 2026)](https://robonine.com/backlash-compensation-in-sts3215-servo-actuators/)
- AhaRobot dual-motor anti-backlash: two STS3215 output shafts coupled with opposite bias torques u1 = u0 + ub, u2 = u0 - ub; suppresses oscillation around the target (qualitative); plus dithering u_d = (-1)^floor(t/T) * ub at a 66 Hz PID to overcome static friction — [AhaRobot, arXiv 2503.10070](https://arxiv.org/html/2503.10070v1)
- Direction-change compensation for backlash in a visual-servoed low-cost SCARA — [low-cost SCARA visual servoing](https://www.researchgate.net/publication/325655900_Visual_servoing_for_low-cost_SCARA_robots_using_an_RGB-D_camera_as_the_only_sensor)
- LeRobot SO-101/SO-100 follower: P_Coefficient 16 (default 32), I 0, D 32, set in configure() "to reduce shaking" — [LeRobot SO-101 docs (Mintlify mirror, third-party)](https://www.mintlify.com/huggingface/lerobot/robots/so-101)
- STS register map: Position P (addr 21, Kp scaled 1/4), D (22, 1/8), I (23); Positive/Negative Dead Zone (26/27, default 1, 0-16, 0.087 deg); Minimum Starting Torque (24, 0.1 %); Torque Limit (48, 0-1000, 0.1 %); Acceleration (41, 0 = max, unit 8.7 deg/s^2); Position Offset (31) — [AIFitLab STS memory table (third-party)](https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-sts-servo-memory-table-analysis)
- Dynamixel XL330 position PID defaults P 400 (KPP = 400/128), I 0, D 0; feed-forward gains available (addr 88/90) — [ROBOTIS e-Manual XL330-M288](https://docs.robotis.com/docs/dxl/model_reference/x_series/xl_series/xl330-m288/)
- FidEx (image-based joint state from fiducials on an SO-100): ~75 % lower EE state error, ~40 % (+5 % with delta refinement) lower control error vs raw-encoder FK — [FidEx, arXiv 2601.08034](https://arxiv.org/html/2601.08034)
- General camera-based kinematic calibration reaches ~1-2 mm end-effector error (simulation) — [Robot self-calibration using multiple kinematic chains, arXiv 1805.07263](https://arxiv.org/pdf/1805.07263); after a point, more calibration poses do not help — [space manipulator EKF calibration, Machines 2023](https://doi.org/10.3390/machines11030409)
- Configuration-dependent (local POE) calibration: >50 % max-error reduction on industrial arms — [arXiv 2510.19962](https://arxiv.org/html/2510.19962)

### Inferences
- Cheapest wins for our arms (inference): (1) always approach the final pose from the same side on each joint (e.g. overshoot by > backlash and come back), which turns backlash from random into a constant offset that hand-eye calibration absorbs; (2) final approach from above so gravity pre-loads the gear flanks in a consistent direction; (3) keep the deadband register at 1 count and use a raised P (or I > 0) only for the final hold, since the low P used against shaking increases sag; (4) a configuration-dependent correction table (what our M3 taught-aim table already is) captures sag + mean backlash per region.
- Dual-servo preload is the one mechanical trick with a measured ~7x backlash gain at ~$15-28 per extra STS3215, but it doubles the joint mass and needs a new bracket; worth it only on the shoulder/elbow.
- Kinematic calibration alone cannot go below the backlash/sag floor; it removes the geometric part (link lengths, zero offsets), which on hobby arms is usually smaller than the gear-train part.

### Gaps
- No primary LeRobot source code was fetched (raw GitHub path 404'd); the P=16 value comes from a third-party mirror of the docs.
- No published before/after mm numbers for direction-of-approach backlash compensation on a hobby bus-servo arm.
- No Yahboom-specific documentation of PID/deadband registers for the YB-SD15M class servo was found.

## 6. Cheap hardware upgrades that buy the most precision

### Takeaway
Per euro, the biggest precision gains are (1) mechanical backlash removal on the two proximal joints (dual-servo preload, or
servos with lower spec backlash such as an XM430 at 0.25 deg), (2) better grip geometry (custom fingers + high-friction tape/pads,
which ALOHA found necessary), and (3) a close-range depth camera (D405: <0.7 % depth error at 25 cm, ~1.75 mm) for the final
centimetres. Swapping STS3215 for an STS3250 buys torque, not documented precision.

### Cited Findings
- Prices: STS3215 ~$15 (AhaRobot) / EUR 24.70-27.78 retail; STS3250-C002 (50 kg.cm) EUR 75.85, no stated precision or backlash advantage — [AhaRobot](https://arxiv.org/html/2503.10070v1); [OpenELAB Feetech listing](https://openelab.io/collections/feetech-motor)
- Dynamixel: XL430-W250 ~$50 each, XL330-M288 ~$24 each (Koch BOM) — [Koch low_cost_robot](https://github.com/AlexanderKoch-Koch/low_cost_robot); XM430-W350 0.25 deg backlash spec — [ROBOTIS docs XM430-W350](https://docs.robotis.com/docs/dxl/model_reference/x_series/xm_series/xm430-w350/) (XM430 price not captured; a ROBOTIS US listing exists: [XM430-W350-R](https://www.robotis.us/dynamixel-xm430-w350-r/))
- Dual-servo preload: ~7x backlash reduction for the cost of a second STS3215 per joint — [Robo9](https://robonine.com/backlash-compensation-in-sts3215-servo-actuators/)
- Drivetrain redesign (capstan cable drives, timing belts, tensioners, topology-optimised links) gives "sub-millimeter repeatability" at <$215 material — [Forte, arXiv 2507.15693](https://arxiv.org/abs/2507.15693)
- ALOHA fingers: 3D-printed "see-through" fingers with gripping tape; "The OEM fingers, however, are not versatile enough to handle fine manipulation tasks"; two Logitech C922x webcams wrist-mounted; "do not make use of special encoders, sensors, or machined components" — [ALOHA/ACT](https://ar5iv.labs.arxiv.org/html/2304.13705)
- Third-party SO-ARM100/101 parallel gripper for STS3215 claims 0.5 mm repeatability (vendor claim for the gripper, not the arm) — [roboninecom parallel gripper](https://github.com/roboninecom/SO-ARM100-101-Parallel-Gripper)
- Intel RealSense D405: 7-50 cm range, +-2 % depth accuracy at 50 cm, "better than 0.7 %" at 25 cm, minimum object detection 0.1 mm at 7 cm (press release; a retailer says 0.5 mm) — [FRAMOS D405 release](https://framos.com/en/news/framos-introduces-new-intel-realsense-depth-camera-d405); [Intel D405 page](https://realsense.intel.cn/depth-camera-d405/); [SparkFun (retired listing)](https://www.sparkfun.com/intel-realsensetm-depth-camera-d405.html)
- Image-only state estimation (printed fiducial exoskeleton on the links) as a near-zero-cost "external encoder": grasp 4/10 -> 9/10 on an SO-100 — [FidEx, arXiv 2601.08034](https://arxiv.org/html/2601.08034)
- Tactile sensing lifts 0.1 mm-clearance insertion from 15 % to 80 % for ACT (not low-cost hardware) — [ReTac-ACT, arXiv 2603.09565](https://arxiv.org/pdf/2603.09565)
- Magnetic-encoder servos are claimed to avoid potentiometer wear jitter and drift (vendor claim) — [Team Tekin](https://www.teamtekin.com/phone/servos.html)

### Inferences
- For our two arms (inference, ranked by mm gained per euro): (1) consistent approach direction + taught/table correction — free, attacks the largest systematic term; (2) compliant high-friction fingertip pads (grip tape/silicone) — a few euros, widen the capture range so a 2-4 mm error still yields a stable grasp; (3) fiducial markers on the links seen by a fixed or wrist camera (FidEx-style) — a few euros, gives the M3 the measured joint state it lacks; (4) replacing the shoulder/elbow pot servos with magnetic-encoder servos (STS3215-class, ~EUR 25 each) — removes pot nonlinearity/drift on the X3, but not backlash; (5) dual-servo preload on shoulder/elbow (~EUR 50-60 + bracket); (6) D405 (~a few hundred euros; price not captured) for metric depth in the final 7-30 cm; (7) Dynamixel XM430-class servos — lowest specified backlash but by far the most expensive per joint.
- Stiffer printed links matter less than gears on these arms: the measured servo deflection (~2 deg at 10 kg.cm) already dominates (inference; no link-stiffness measurement found).
- A better wrist RGB camera mostly helps through resolution at the working distance (pixel footprint) and lower motion blur; at 20-30 cm a 640-848 px wide image already gives sub-mm pixels, so the camera is rarely the floor (inference).

### Gaps
- No price for D405 or XM430-W350 was captured in this session; no measured precision gain from swapping STS3215 for STS3250 or for a Dynamixel on the same arm.
- No published measurement of link (PLA/aluminium bracket) compliance on SO-101 or Yahboom arms.
- No source on adding external output-shaft encoders (e.g. AS5600) to a bus-servo arm with measured gains.
