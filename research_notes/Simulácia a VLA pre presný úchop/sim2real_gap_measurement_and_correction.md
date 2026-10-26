# Measuring and closing the sim-to-real gap for an imitation-learned grasping policy on a cheap arm without joint feedback (M3 Pro playbook)

Scope note (researcher, 2026-10-26): this file covers practical procedures, metrics and protocols only. Co-training ratios, Gaussian-splat real2sim and actuator models at a high level were covered in an earlier report and are touched here only where a number or a procedure is needed. In each section, "Cited Findings" are what the sources measured or claim (tagged [measured] or [authors' claim] where it matters). "Inferences" are the researcher's opinion and the concrete M3 procedure built on top of them. Project facts used for concreteness come from the assignment and the repo context (M3 arm = bus servos, commanded angle + move time over ROS 2, no read-back; DaBai DCW2 wrist RGB-D with hand-eye; gripper gap table; URDF; classical picker ~85-90 %, fingertip ~1.5 cm short without taught corrections; LeRobot episodes at 10 Hz with state = last command; a recent M3 ACT judged offline at 6.05 deg joint error vs 7.79 deg, which "never left START" on the robot because of an episode cut, per the 2026-10-26 git log).

## 1. How do recent works replay real trajectories in sim (and vice versa) to quantify kinematic, dynamic and visual gaps, and what tolerances matter for grasping small (30 mm) objects?

### Takeaway
The standard 2024-2026 procedure is to (1) replay the real robot's recorded commands open-loop in sim and fit controller/actuator parameters until simulated end-effector paths match the real ones (SIMPLER), (2) evaluate real and sim on the same fixed grid of initial object poses, and (3) fix colour/texture so the rendered image matches the real camera. Each of these measurably raised sim-real agreement. No paper gives a ready tolerance for 30 mm objects; for the M3 the budget has to come from the gripper gap table (lateral slack = (open gap - object width)/2), and the robot's joint-space errors have to be converted to fingertip millimetres.

### Cited Findings
- [measured, May 2024] SIMPLER identifies the simulated robot controller by replaying real demonstrations open-loop. It optimises PD stiffness and damping so simulated end-effector trajectories match the real ones. The loss is the mean L2 end-effector translation error plus an arcsin rotation error, minimised by simulated annealing over a normalised [0,1] range in 3 narrowing rounds. It reuses existing demonstration data, so nothing new is collected. — [SIMPLER, arXiv 2405.05941](https://arxiv.org/html/2405.05941)
- [measured, May 2024] In SIMPLER, better control identification gave better sim-real ranking. On Pick Coke Can, the identified gains gave control loss 0.131 and MMRV 0.031, against 0.267/0.070 and 0.432/0.100 for two perturbed gain settings. — [SIMPLER](https://arxiv.org/html/2405.05941)
- [measured, May 2024] Visual matching in SIMPLER: green-screening the real background alone cut the real-sim success gap from 0.272 to 0.198. Green screen plus drawer and robot texture matching cut it to 0.136 (MMRV 0.087 to 0.050). Texture matching without the green screen made the gap worse (0.328). The authors conclude that visual fixes must be applied to the whole scene together. — [SIMPLER](https://arxiv.org/html/2405.05941)
- [measured, Nov 2025] Zhang et al. evaluate on a fixed grid of initial object poses shared between sim and real (16-27 configurations per checkpoint). The robot's Gaussian-splat model is aligned to URDF points by ICP+RANSAC (2,000 points per link). Colour is aligned by fitting a degree-2 polynomial RGB transform (IRLS) between renders and real captures, with camera white balance and exposure fixed. Removing colour alignment dropped Pearson r from 0.944 to 0.805 (toy packing), 0.901 to 0.714 (rope) and 0.915 to 0.529 (push-T). Removing physics optimisation dropped r to 0.694, 0.832 and 0.905. Which gap dominates depends on the task. — [Real-to-Sim Policy Evaluation with GS Simulation, arXiv 2511.04665](https://arxiv.org/html/2511.04665v2)
- [authors' claim, Dec 2025] PolaRiS sets metric scale and orientation with a ChArUco board. Robot splats are attached to kinematic links and moved by forward kinematics. The authors recommend joint-position actions because they transfer real-to-sim better by avoiding compounding error. — [PolaRiS, arXiv 2512.16881](https://arxiv.org/html/2512.16881)
- [measured, Oct 2025] GSWorld evaluated real-trained policies in its splat simulator. Sim systematically under-predicted real success while keeping the order: ACT averaged 50.0 % real vs 41.0 % sim, Pi0 55.0 % vs 43.5 %. Per-task ACT real/sim: Place Box 50/44, Pour Sauce 40/28, Stack Cans 50/42, Arrange Cans 60/48. No correlation coefficient is reported. Scale came from a printed ArUco marker, the robot was aligned by ICP of the URDF mesh, and object mass was weighed. — [GSWorld, arXiv 2510.20813](https://arxiv.org/html/2510.20813)
- [measured, Apr/Jul 2025] Real-is-Sim runs the opposite direction (sim follows real). An Embodied-Gaussians twin is corrected at 60 Hz by "visual forces" from the photometric error, and the real robot follows the simulated robot's joints. Reported failure modes: simulated collisions with too little clearance cause impacts that never happen in reality, and the visual correction cannot recover from them. — [Real-is-Sim, arXiv 2504.03597v2](https://arxiv.org/html/2504.03597v2)
- [measured, Jan 2026] Fiducial exoskeletons: a 3D-printed marker on each link of a $100 SO-100 arm, with joint angles solved from one RGB image by L-BFGS over FK. End-effector state error was 0.06 translation / 0.27 rotation, against 0.18 / 1.1 for the arm's own encoder FK (units not stated in the table; about 75 % lower error). A grasp test succeeded 9/10, against 4/10 with middle-pose calibration. The authors describe backlash as an unmeasured dead zone that accumulates down the chain. — [Fiducial Exoskeletons, arXiv 2601.08034](https://arxiv.org/html/2601.08034v1)
- [authors' claim, 2025, third-party blog, low confidence] A survey of SO-100/101 simulation work says that in lerobot-sim2real, friction and inertia identification carried over from the upstream model but the controller did not: the simulated position actuators use gains substantially different from the published values. — [SO-101 sim landscape blog](https://mcobzarenco-fontaine-blog.static.hf.space/papers/so101-sim-landscape.html)

### Inferences
- Opinion: the M3 version of the SIMPLER replay (step 1 of calibration). Take N ≈ 20-50 recorded classical-picker episodes (state = last command, at 10 Hz). Replay the exact command stream, with the same move times, to (a) the real arm while the wrist camera watches a fixed ChArUco board, and (b) the simulated arm in Isaac Sim. The board pose seen in each real frame, combined with the hand-eye calibration, gives the real camera pose and therefore the real end-effector pose at 30 fps, with no joint read-back needed. Fit the simulated actuator parameters (static per-joint offset, gain or move-time profile, delay, backlash) so the simulated end-effector path matches the camera-measured path, using SIMPLER's translation-plus-rotation loss.
- Opinion: report the replay gap as three numbers per episode set: (i) end-effector position error at the moment of grasp closure (mm); (ii) RMS path error over the descent (mm); (iii) timing offset (ms, by cross-correlation). Point (i) is the one that decides a 30 mm pick.
- Opinion, tolerance budget: lateral slack per side = (gripper open gap - 30 mm)/2 from the measured gap table, minus a margin so the fingertip does not land on the top edge. The known 1.5 cm short-landing of the uncorrected picker is already larger than any plausible slack. Set the replay acceptance at a few millimetres, about one third of the slack: "sim fingertip within ~5 mm of camera-measured real fingertip at closure". This is a heuristic, not a published number.
- Opinion: convert joint-space metrics to millimetres before comparing them to the tolerance. Example: the M3 ACT's offline 6.05 deg joint error, if it sat on a shoulder joint with ~0.2 m of lever arm, is ~0.105 rad x 0.2 m ≈ 21 mm at the fingertip. That would miss a 30 mm cube, so a "better" 6.05 vs 7.79 deg can still be a failing policy. Use the URDF Jacobian at the grasp poses.
- Opinion: the reverse replay (sim → real) is cheap and useful. Send the Isaac expert's joint commands for a known layout to the real arm with no cube present, and photograph or measure where the fingertips close relative to taped marks. This isolates the kinematic and actuator gap from perception.

### Gaps
- No 2024-2026 paper found that states a fingertip-error tolerance for small-object grasping as a function of gripper slack; the tolerance above is the researcher's heuristic.
- No paper found that does SIMPLER-style controller identification on a servo arm without read-back using a wrist camera on a board; the procedure combines SIMPLER (fit), Fiducial Exoskeletons (vision as the state sensor) and the existing M3 ChArUco measurements.

## 2. How do you identify servo dynamics without joint feedback (camera-based joint tracking, fiducials on links, step response from video), measure latency, and model backlash/deadband in Isaac Sim (joint drives, actuator nets)?

### Takeaway
Use vision as the joint sensor: per-link fiducials, or the wrist camera looking at a fixed ChArUco board. Measure latency the UMI way (a timestamp on a screen for the camera; commanded-vs-observed cross-correlation for execution). Then pick the simplest Isaac Lab actuator that reproduces the measured response: DelayedPD for delay, DCMotor for a speed ceiling, an actuator net if the response is non-linear. Isaac Lab has no deadband model. Backlash exists only in the new BAM servo model (Newton backend), so backlash and deadband otherwise have to be added on the command side.

### Cited Findings
- [measured, Feb 2024] UMI camera latency: film a rolling QR code on a monitor that shows the current system timestamp. l_camera = t_recv - t_display - l_display. — [UMI, arXiv 2402.10329](https://arxiv.org/html/2402.10329)
- [measured, Feb 2024] UMI execution latency: send sinusoidal position commands, record the measured response, and find the lag by cross-correlation. Execution latency = end-to-end latency - proprioception latency. Without robot timestamps, proprioception latency is approximated as half the ping round-trip. At deployment, actions already outdated by observation + inference + execution latency are discarded. On dynamic tossing, latency matching gave 105/120 (87.5 %), against 69/120 (57.5 %) with all latencies set to 0. — [UMI](https://arxiv.org/html/2402.10329)
- [documentation, accessed 2026-10-26] Isaac Lab actuator models:
  - Implicit (solver-side PD) and IdealPD.
  - DCMotor: PD torque clipped to a four-quadrant torque-speed envelope (`saturation_effort`, `actuator_velocity_limit`).
  - DelayedPDActuator: position, velocity and effort commands delayed by a buffer of `min_delay`..`max_delay` physics steps, sampled once per reset.
  - RemotizedPD: angle-dependent torque ceiling.
  - ActuatorNetMLP/LSTM: a learned net from position-error and velocity history to torque.
  - BamActuatorCfg (Newton backend only): voltage-domain servo model with firmware P law, PWM duty, back-EMF and load-dependent gearbox friction.
  - Common fields: stiffness, damping, armature, static/dynamic/viscous friction.
  - The page has no general backlash model. Backlash exists only in BAM, through `backlash_joint_template`, which binds a passive revolute "play" hinge in series with each servo.
  - No deadband model is described.
  — [Isaac Lab actuators](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/../../concepts/actuators.html)
- [measured, Oct 2024] BAM (Rhoban) fits extended friction models M2-M6 (Stribeck, load-dependent, directional, quadratic) to servo trajectories recorded on a pendulum test bench. MAE fell by 2.93x for the Dynamixel MX-64 and 2.02x for the MX-106 against the default Coulomb-viscous model. The docs list a Feetech STS3215 entry, but only the basic m1 model was identified for it. — [Duclusaud et al., arXiv 2410.08650](https://arxiv.org/pdf/2410.08650); [BAM docs](https://bam.readthedocs.io/en/latest/usage/actuators.html)
- [measured, Jan 2026] Fiducial Exoskeletons estimate the state in ~0.043 s per image, against ~4.9 s for the Dr. Robot baseline. Control loop: move to the target, estimate the actual state by vision, command the target minus the observed error. The base marker and at least one other link marker must be visible; with one visible link and no encoder warm start, the optimisation can fall into a degenerate minimum. — [arXiv 2601.08034](https://arxiv.org/html/2601.08034v1)
- [measured, Dec 2020] Surgical-robot calibration tracked coloured sphere markers with an RGB-D camera and trained a recurrent network from commanded to actual joint configurations. Using the motion history compensated hysteresis and backlash-like effects. — [Hwang et al., arXiv 2012.12844](https://arxiv.org/pdf/2012.12844)
- [measured, May 2024] SIMPLER's identification fits only PD stiffness and damping, from existing trajectories (see Q1). — [SIMPLER](https://arxiv.org/html/2405.05941)

### Inferences
Opinion: an M3 servo-identification protocol that uses only the wrist camera and the ChArUco board already in use. About one lab day.
1. **Static map and backlash per joint.** Hold all joints but one. Step the joint across its range in ~5 deg steps, approaching each target once from below and once from above, with a long move time and 1 s settle. Read the camera pose from the board, then solve the joint angle by IK with the other joints fixed. Results: a commanded→actual offset table, where backlash = actual(from below) - actual(from above) at the same command; plus the gravity-loaded droop of J2/J3 as a function of pose. This is the M3 analogue of the Fiducial Exoskeletons offset calibration (Δθ = θ* - θ_cmd).
2. **Step and move-time response.** Command steps of 5, 15 and 30 deg with the move times the policy uses (the 10 Hz tick = 100 ms). Film at the camera's highest frame rate. Fit dead time, the servo's internal trajectory profile (does it finish in the commanded move time? overshoot?) and the settling error. The bus servo runs its own interpolation over the move time, so a plain PD in Isaac will not match by default.
3. **Latency.** Camera: point the DCW2 at a laptop screen showing a millisecond clock or rolling QR code (UMI formula). Command path: a slow sinusoid on one joint, cross-correlate the commanded angle with the camera-measured angle (UMI), then subtract camera latency. Record the ROS 2 publish timestamp to separate DDS/bridge delay from servo delay.
4. **Deadband.** Send increasingly small increments (0.5, 1, 2 deg) and find the smallest command change that produces visible motion.
5. **Model in Isaac Sim, simplest first.**
   - Joint position drive with high stiffness, fed the command through a small Python "servo emulator" that reproduces the measured move-time interpolation.
   - DelayedPD-style delay sampled per episode from the measured latency spread.
   - The static offset table, randomised by ±(measured error) per episode.
   - Command-side backlash: a hysteresis state per joint that holds the output until the command has moved b/2 past the last reversal.
   - Deadband: ignore command changes smaller than the measured threshold.
   - Consider BamActuatorCfg or ActuatorNet only if residual error remains after this. BAM's built-in backlash needs the Newton backend, and nothing has been identified for these Yahboom servos.
6. **Acceptance.** Re-run the Q1 replay with the fitted emulator. Accept when the fingertip error at closure and the timing offset fall inside the Q1 budget. Keep a held-out set of episodes for this check, not the ones used to fit.

### Gaps
- No public identification data or BAM model exists for the Yahboom M3 Pro bus servos. BAM covers Dynamixel models; for the Feetech STS3215 only m1 was identified.
- No 2024-2026 paper found that measures servo step responses purely from video for imitation-learning sim2real. The protocol above is assembled from UMI, Fiducial Exoskeletons and SIMPLER.
- It was not verified whether the Isaac Sim version on hekate or Dionysos (Dionysos runs 5.0.0) has the Newton backend that BamActuatorCfg needs.

## 3. How should the camera be matched: intrinsics, extrinsics via hand-eye, exposure/white balance/noise, depth noise models for active-stereo RGB-D (Orbbec) in sim, rendering realism vs randomisation, and using depth or segmentation to shrink the visual gap?

### Takeaway
Measured evidence says camera-pose alignment matters (NVIDIA: success fell 67→56 % and 95→70 % with unaligned sim views). Colour alignment matters (r fell to 0.53-0.81 without it). Scene-wide visual matching beats partial matching. A small leftover visual gap is not harmful when a physics gap exists. Model depth with the stereo error law ε_z = z²·ε_d/(f·B) and Isaac Sim's single-view stereo depth sensor, plus edge holes and the minimum-range blind zone, after measuring the DCW2's actual ε_d on a plane.

### Cited Findings
- [measured, Mar 2025] NVIDIA sim-and-real co-training: sim data generated with camera views not aligned to the real ones lowered real success from 67 % to 56 % (Panda C2SPnP) and from 95 % to 70 % (GR-1 CupPnP). Perfect alignment was not necessary: their aligned humanoid camera did not model the real fisheye distortion. — [Maddukuri et al., arXiv 2503.24361](https://arxiv.org/html/2503.24361v2)
- [measured, Nov 2025] A degree-2 polynomial RGB colour transform, with fixed white balance and exposure on the real cameras (RealSense D405 wrist, D455 fixed), raised sim-real correlation strongly (see Q1 numbers). — [arXiv 2511.04665](https://arxiv.org/html/2511.04665v2)
- [measured, Dec 2025] PolaRiS ablations: Gaussian-splat rendering gave the strongest correlation, ahead of ray-traced and untextured variants. Unlike SIMPLER's green screen, a 3D reconstruction supports wrist-camera policies. — [PolaRiS](https://arxiv.org/html/2512.16881)
- [measured, Mar/Aug 2025] Wei et al. (planar pushing, diffusion policy, IROS 2025):
  - Performance was least sensitive to a colour-mean shift. A policy with no visual gap was slightly weaker than one with a small gap.
  - When a physics gap exists, removing the visual gap hurts, and a one-hot sim/real encoding recovers the loss. When no physics gap exists, removing the visual gap helps.
  - A binary probe classified sim vs real at 100 % from the observation embedding of all four probed policies.
  - Closing physics gaps may matter more than visual fidelity for contact-rich tasks.
  — [Wei et al., arXiv 2503.22634](https://arxiv.org/html/2503.22634v2)
- [measured, 2024 paper] ManiSkill3's Koch-arm sim2real green-screens the real background into sim, leaves the arm and cube rendered (via the segmentation map), and randomises camera pose, lighting direction and robot pose. It reports 18/20 real successes zero-shot. — [ManiSkill3, arXiv 2410.00425](https://arxiv.org/pdf/2410.00425)
- [measured, 2017] Stereo depth error grows with the square of distance: |ε_z| = z²·|ε_d|/(f·B), where f is the focal length in pixels and B the baseline. Typical subpixel disparity error is ~0.08 px for static and ~0.05 px for moving targets, with ~0.1 px as a rule of thumb, on a RealSense D4xx. The minimum range is set by the disparity search window (64 disparities → z_min ≈ f·B/64). — [Keselman et al., Intel RealSense Stereoscopic Depth Cameras, arXiv 1705.05548](https://arxiv.org/html/1705.05548v2)
- [measured, 2022] Physics-grounded active-stereo simulation (ray-traced IR pattern + IR noise + stereo matching, in SAPIEN) reproduces material-dependent depth errors in real time. Perception and RL policies trained on it transferred without fine-tuning. Rasterisation cannot reproduce failures on translucent or transparent surfaces. — [Zhang et al., arXiv 2201.11924](https://arxiv.org/abs/2201.11924); related: [DREDS, arXiv 2208.03792](https://arxiv.org/pdf/2208.03792), [RaSim, arXiv 2404.03962](https://arxiv.org/html/2404.03962v1)
- [documentation, Isaac Sim 5.1, accessed 2026-10-26] Isaac Sim models stereo disparity and noise from a single camera view:
  - API: `isaacsim.sensors.camera.SingleViewDepthSensor`; baseline attribute `omni:rtx:post:depthSensor:baselineMM`; a D455 asset `rsd455.usd` as the example.
  - Annotators: `DepthSensorDistance` (simulated stereo depth) and `DistanceToImagePlane` (ground truth) for comparing the two.
  - Isaac Sim 6.x renames the API to `SingleViewDepthCameraSensor` and adds a `StructuredLightCamera`.
  — [Isaac Sim 5.1 depth sensors](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera_depth.html); [Isaac Sim 6.x depth sensors](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/sensors/isaacsim_sensors_camera_depth.html)
- [reseller copy of the vendor spec, low confidence] The DaBai DCW2 is described as a binocular structured-light camera with depth up to 5 m and a 0.15 m near blind zone. No official accuracy figure was found. — [OpenELAB M3 Pro listing](https://openelab.io/products/yahboom-rosmaster-m3-pro-ros2)
- [measured, Jan 2025] Biruduganti et al. score vision encoders on a "Domain Invariance Score" (how well sim and real embeddings align) and an "Action Score" (linear-probe action prediction). DINOv2-B was worst on action inference and MCR was best on both. Embedding similarity alone is therefore not a reliable predictor of policy usefulness. — [arXiv 2501.16389](https://arxiv.org/html/2501.16389v2)

### Inferences
Opinion: an M3 camera-matching checklist, in order of measured payoff.
1. **Intrinsics.** Use the DCW2's factory or ChArUco-calibrated RGB intrinsics and distortion in the Isaac camera (focal length, principal point, resolution, crop). Render at the policy's input resolution.
2. **Extrinsics.** Mount the sim camera on the gripper link with the existing hand-eye transform. Verify it with an overlay test: render the sim view at a few real poses (commanded joints run through the fitted static offset map from Q2) and blend it with the real frame. A board edge or gripper finger misaligned by more than a few pixels means the hand-eye or joint offsets are off. ManiSkill and SIMPLER both use overlay checks.
3. **Exposure and white balance.** Lock auto-exposure and auto-WB on the DCW2 when recording real episodes, then fit a per-channel degree-2 polynomial colour map (sim→real) from ~10 paired views of the board and the kit pieces, as in arXiv 2511.04665. Randomise around the fitted map, not around sim defaults.
4. **Depth.** Measure ε_d first: plane-fit RMS of the DCW2 on a flat board at 0.2, 0.3 and 0.4 m. Solve ε_d = RMS·f·B/z² and use it in the Isaac depth sensor or as additive disparity noise. Add holes at depth discontinuities and dark or specular surfaces, and invalid depth inside 0.15 m. At close range, edge "flying pixels" and holes on a 30 mm cube edge probably hurt more than the per-pixel noise. Example only, with assumed f ≈ 460 px and B ≈ 40 mm (not DCW2 values): z = 0.25 m and ε_d = 0.08-0.5 px give ε_z ≈ 0.3-1.7 mm, so measure.
5. **Shrink the gap by choosing inputs.** Wei et al. show that some visual gap is fine and that physics matters more. For a wrist-camera grasp, feeding depth (or a kit-detector mask from the existing YOLO26-seg) alongside RGB, or as a second stream, removes texture and colour from the gap. The mask channel can come from the same segmentation model on real images and from ground truth in sim, though the detector's real errors must then be injected in sim.
6. **Visual gap probe** (diagnostic, not a pass/fail). Compute FID/KID or a DINOv2/policy-encoder embedding distance between real wrist frames and sim frames at matched poses (from the Q1 replay). Track it as you fix colour, intrinsics and textures. Do not treat a large distance as a predictor of failure: Wei et al. found high-performing policies separate sim from real anyway.

### Gaps
- No official DaBai DCW2 datasheet (baseline, focal length, accuracy vs distance) was found; the only figures are a reseller's copy (5 m, 0.15 m blind zone).
- No 2024-2026 source was found that validates FID or KID between sim and real robot images as a predictor of real policy success. The closest are SIMPLER's real-sim gap ablations and the encoder Domain Invariance Score, which did not predict usefulness.
- The exact noise parameters of Isaac Sim's single-view depth sensor are in a separate document that was not retrieved.

## 4. Which offline predictors of real success correlate with it (SIMPLER, real-to-sim evaluation, Real-is-Sim, correlation studies), and does offline action error on held-out real episodes work?

### Takeaway
Closed-loop evaluation in a carefully matched sim reaches Pearson r ≈ 0.9 with real success: SIMPLER 0.924 with visual matching, splat-based 0.90-0.94, PolaRiS ≈ 0.9 average and worst case 0.81. Offline action MSE on held-out real data is a poor predictor (SIMPLER: r = 0.308, MMRV 0.375). Use action error only as a regression or sanity test, preferably in task space (fingertip mm at grasp), and use a matched sim twin of the real test grid to rank checkpoints.

### Cited Findings
- [measured, May 2024] SIMPLER, Google Robot, six checkpoints: validation action MSE MMRV 0.375 / Pearson r 0.308; Variant Aggregation 0.143 / 0.778; Visual Matching 0.056 / 0.924. Per task: Pick Coke Can 0.031 / 0.976, Move Near 0.111 / 0.855, Drawer 0.027 / 0.942. MMRV weights each mis-ranked pair by its real performance gap, in [0,1], lower is better. "Over ~1500" evaluation episodes in total. — [SIMPLER](https://arxiv.org/html/2405.05941)
- [measured, Nov 2025] Splat + physics twin: Pearson r 0.944 / 0.901 / 0.915 (MMRV 0.076 / 0.174 / 0.108) across ACT, DP, Pi-0 and SmolVLA checkpoints. An Isaac Lab baseline reached r = 0.237 (rope) and 0.649 (push-T). — [arXiv 2511.04665](https://arxiv.org/html/2511.04665v2)
- [measured, Dec 2025] PolaRiS:
  - Average r ≈ 0.9 against real DROID evaluations, r = 0.98 against RoboArena scores, worst per-environment r = 0.81.
  - It needed a brief sim co-fine-tune: ~350 sim demos from 15 environments disjoint from the evaluation scenes, 90 % real / 10 % sim, ~1k steps, under 25 min. Without it, correlation was "too low to accurately rank" policies, and fine-tuning too long reduced correlation.
  - Real evaluation used 20 rollouts per policy per environment; sim used 50.
  - Build time under one hour per environment.
  - The authors say it "does not entirely replace real-world evaluation".
  — [PolaRiS](https://arxiv.org/html/2512.16881)
- [measured, Apr/Jul 2025] Real-is-Sim kept the relative order of 4 checkpoints, using 60 real trials each (20 poses × 3), but reported no coefficient. Later checkpoints were not consistently better than earlier ones. — [Real-is-Sim](https://arxiv.org/html/2504.03597v2)
- [authors' claim, snippet-level, 2025-2026] Several sources report offline loss as a weak proxy:
  - Robomimic-derived evidence: the lowest-validation-loss checkpoint performed far worse on the robot than the best one. — [HF blog, validation loss in robotics](https://huggingface.co/blog/m1b/validation-loss-robotics)
  - "Much Ado About Noising" (Dec 2025): success rose while validation loss stayed flat. — [arXiv 2512.01809](https://arxiv.org/pdf/2512.01809)
  - A 2026 behaviour-cloning scaling study found validation action error significantly correlated with real performance (the strongest offline signal), but validation loss not. — [arXiv 2606.27375](https://arxiv.org/pdf/2606.27375)
  - The ICRA 2026 REAL-I challenge report calls checkpoint selection an open problem, because open-loop error misses compounding error, contact and recovery. — [arXiv 2609.13679](https://arxiv.org/pdf/2609.13679)
  - Counterpoint: "Train Offline, Test Online" (2023) found offline action-error metrics tracked real reward. — [arXiv 2306.00942](https://arxiv.org/pdf/2306.00942)
  - Note: these were seen through search summaries, not read in full.
- [measured, Mar/Aug 2025] Wei et al. fit power laws for co-trained policies: action MSE ∝ |D_sim|^-0.285 · |D_real|^-0.587 (R² = 0.975). In this offline metric, real data is weighted more heavily. — [arXiv 2503.22634](https://arxiv.org/html/2503.22634v2)
- [measured, Oct 2025] GSWorld's sim under-predicted real success by ~6-20 points per task but kept the order (see Q1). — [GSWorld](https://arxiv.org/html/2510.20813)

### Inferences
Opinion: an M3 pre-deployment gate, three tiers, cheapest first.
1. **Tier 0: open-loop on held-out real episodes** (minutes, no robot). Run the policy on real recorded LeRobot episodes the classical picker made, which were not used for training. Report (a) joint error in degrees, as already done (6.05 vs 7.79 deg); (b) more importantly, the FK fingertip position of the predicted grasp pose against the picker's actual grasp pose, in mm; (c) gripper-close timing error in ticks. Use it to catch regressions and broken preprocessing: the "never left START" episode-cut bug would have shown up as a policy that predicts "stay" on real frames. Do not rank checkpoints by it (SIMPLER r = 0.31).
2. **Tier 1: closed-loop in a matched twin** (hours, no robot). Isaac scene with the Q1/Q2-fitted servo emulator, the Q3 camera match, and the real test grid of cube poses (Q6). Run every checkpoint on the same grid with ≥50 sim rollouts. Once 4-6 checkpoints or policies also have real results on the grid, compute Pearson r and MMRV. Trust sim ranking only if r ≳ 0.8 and MMRV ≲ 0.1, the range the cited papers reach. If not, apply PolaRiS's fix first: a short co-fine-tune with ~10 % in-domain sim data.
3. **Tier 2: real "shadow mode"** (robot powered, arm not driven by the policy). Run the policy on live wrist frames while the classical picker drives, logging the policy's proposed next command against the picker's. This tests the policy on the real visual stream without the real-robot risk. It is still open-loop, so it only catches perception problems.
4. **Tier 3: guarded real rollouts** on the fixed grid (Q6).
- Opinion: Real-is-Sim and GSWorld show sim can keep the order while being biased by 6-20 points. Use sim for ranking and gating, never as the success estimate.

### Gaps
- No study found that tests these predictors on a low-cost servo arm without read-back. All correlation numbers come from Franka-, Google-Robot- or DROID-class arms with accurate joint control.
- The "2606.27375" and "2609.13679" findings were read only as search summaries; their exact correlation values were not verified.

## 5. What is the best post-deployment correction loop with limited real data and an available classical expert (DAgger with the classical picker as labeller, HG-DAgger, residual policies, real-only fine-tune vs sim+real co-training), and how many real episodes are typically needed?

### Takeaway
With little real data, mixing sim and real data (co-training) beats sim-pretrain-then-real-fine-tune:
- NVIDIA: real-only 45.3 % → 83.2 %.
- Wei et al.: 10 real demos 2/20 → 14/20, 50 real demos 10/20 → 19/20, no gain at 150.

Targeted on-policy corrections then add 30-60 points with 50-230 correction episodes:
- TRANSIC: 227 trajectories, 81 % average.
- CR-DAgger: 50-100 episodes, +64 %.
- GSWorld planner-DAgger: real 30 → 64 %.

Plain fine-tuning of the base policy on corrections can hurt (CR-DAgger: -30 % on one task). Residual corrections, or retraining on aggregated data, are safer. A classical planner as the DAgger expert, resetting to the last solvable state, is a published pattern (GSWorld, Diff-DAgger). It fits the M3 picker.

### Cited Findings
- [measured, Mar 2025] NVIDIA co-training, real success, average of 6 tasks:
  - Results: real-only 45.3 %; real + task-aware sim ("DC") 81.1 %; real + task-agnostic prior sim 76.8 %; both 83.2 %.
  - Real data: 50 demos per task (Panda) and 20 per task (GR-1). Sim data: 10k per task generated from 100 source demos (Panda), 1k from 10 source demos (GR-1).
  - Mixing ratio: default 90 % sim. Ablation on CupPnP: 99 % sim best (95 %), 99.5-99.9 % sim fell to 60 %, 1:1 suboptimal.
  - Fewer sim demos hurt: 10k → 500 dropped 67 → 53 %; 1k → 100 dropped 95 → 75 %.
  - Generalisation (GR-1): unseen positions 43 → 100 %, unseen objects 10 → 80 %.
  — [arXiv 2503.24361](https://arxiv.org/html/2503.24361v2)
- [measured, Mar/Aug 2025] Wei et al., real planar pushing, 20 trials per policy:
  - 10 real demos: 2/20 → 14/20 with co-training. 50 real demos: 10/20 → 19/20. 150 real demos: 20/20, no meaningful gain.
  - Mixing beat sim-only pretraining followed by real fine-tuning, and fine-tuning co-trained models did not consistently help.
  - The best mixing ratio rises with the amount of real data; sweep it.
  - In their sim experiments, co-training was worth ~2-3x more target data.
  — [arXiv 2503.22634](https://arxiv.org/html/2503.22634v2)
- [measured, Apr/Jul 2025] Real-is-Sim (PushT): 30 real demos ≈ 57 %; 30 real + 30 sim = 80 %. — [arXiv 2504.03597](https://arxiv.org/html/2504.03597v2)
- [measured, Oct 2025] GSWorld DAgger with an MPlib motion-planner expert:
  - Procedure: on each failed rollout, sample states from the failure sequence where the task is still achievable, reset there, and let the planner generate the correction. 100 expert trajectories in iteration 1, then 4 more iterations of 100.
  - Real Franka FR3: average 30 % → 63.75 % after 5 iterations, against 55 % for training from scratch on the same amount of data. Sim: 35 → 75 %.
  - Also: real-to-sim-to-real DAgger, starting from a real ACT policy trained on a few demos, on an xArm.
  — [GSWorld, arXiv 2510.20813](https://arxiv.org/html/2510.20813)
- [measured, Oct 2024] Diff-DAgger uses a rule-based RRT motion planner as the DAgger expert (ManiSkill stacking and plugging) and includes real-robot tasks. — [arXiv 2410.14868](https://arxiv.org/pdf/2410.14868)
- [measured, May/Oct 2024] TRANSIC (CoRL 2024):
  - Data: humans corrected a sim-trained policy on the real robot; 227 corrected trajectories with 1,043 corrections across 4 tasks.
  - Method: a residual policy (predicting the post-minus-pre-intervention state delta) plus a learned gate.
  - Results: average 81 % (100 / 95 / 45 / 85 %), 1.24x the sim policy, beating HG-DAgger and IWR. Learned gate 81 % vs human gating 82 %.
  - Under exaggerated sim-to-real gaps (perception error, underactuated controller, embodiment, dynamics, asset mismatch): 77 % vs 18 % for the best baseline (IWR).
  - Scaling: going from 25 % to 75 % of the correction data improved TRANSIC by 42 %.
  - 20 real trials per task.
  — [TRANSIC, arXiv 2405.10315](https://arxiv.org/html/2405.10315v3)
- [measured, Jun/Dec 2025] Compliant Residual DAgger:
  - Data: 50-100 correction episodes per collection method.
  - Results: +64 % base success across 4 tasks; book flipping 40 → 100 %; belt assembly 15.6 % → 96.8 %. Fine-tuning the base policy on corrections dropped ~30 % (book) and 15.6 % (belt); retraining from scratch gained little.
  - On-policy "delta" corrections beat take-over corrections by 45 %.
  - Their protocol: start collecting once the base policy reaches ~10-20 % success; include no-correction data labelled as zero residual; sample densely right after the intervention start; watch tracking error, which can flip the sign of small corrections.
  - Evaluation: 20-32 rollouts per task.
  — [CR-DAgger, arXiv 2506.16685](https://arxiv.org/html/2506.16685)
- [measured, Sep 2025] Residual off-policy RL on a real humanoid, on top of ACT, sparse binary reward:
  - WoollyBallPnP: 14 → 64 % with 134 episodes (~15 min robot execution time).
  - PackageHandover: 23 → 64 % with 343 episodes (~76 min).
  - Settings: UTD 4, n-step 3, 50/50 demo/online batches, DrQ augmentation.
  - Safety: wrist force-torque limits and collision checks; the frozen base policy acted as an implicit constraint.
  — [Ankile et al., arXiv 2509.19301](https://arxiv.org/html/2509.19301v2)
- [authors' claim, Oct 2018 / 2019] HG-DAgger: the human expert takes full control only when they judge it necessary, because humans give poor labels without control (they perceive actuator lag). It also learns a risk threshold from model uncertainty. Improved on DAgger and BC in sim and real driving. — [Kelly et al., arXiv 1810.02890](https://arxiv.org/abs/1810.02890)
- [documentation, accessed 2026-10-26] LeRobot ships `lerobot-rollout --strategy.type=dagger`:
  - Loop: pause → takeover → return-to-policy within one episode. Autonomous and human segments are recorded together; then fine-tune on demos + HIL data and iterate.
  - It requires a teleoperator with active motors (SO leader, OpenArm Mini).
  - It follows RaC's recovery-then-correction decomposition.
  — [LeRobot HIL data collection](https://huggingface.co/docs/lerobot/hil_data_collection)
- [documentation, accessed 2026-10-26] LeRobot's guidance for first datasets: "at least 50 episodes, with 10 episodes per location". — [LeRobot IL on real robots](https://huggingface.co/docs/lerobot/il_robots)
- [authors' claim, Mar 2026, snippet-level] ExpertGen distils RL-steered sim experts into visuomotor policies with DAgger and reports 90.5 % on real industrial assembly. — [ExpertGen, arXiv 2603.15956](https://arxiv.org/abs/2603.15956)

### Inferences
Opinion: an M3 correction loop, in order.
1. **Before any real correction, co-train.** Train ACT on (Isaac expert demos, thousands) + (real classical-picker episodes, the ~50-200 already recorded). Use sim-heavy sampling (start at 90 % sim, sweep 75 / 90 / 99 %) and a one-hot "sim/real" input token, per Wei et al., so the policy may keep the domains apart. Expect the largest gain if real episodes number in the tens. At ~150+ good real episodes, co-training may add little (Wei et al.). The classical picker is the cheapest source of real episodes, so keep logging every successful real pick.
2. **DAgger with the classical picker as expert** (GSWorld-style "reset to solvable, let the planner finish"). When the guard trips or the policy stalls or drifts:
   - Freeze the arm.
   - Let the classical picker take over from the current pose. If its perception needs a look pose, retreat to the nearest look pose, which is itself a valid recovery trajectory.
   - Let it finish the pick; record the whole take-over as a correction segment with an intervention flag.
   - Pointwise labelling is also possible: at each visited state, the label = the next 10 Hz joint target along the picker's IK path toward its current grasp estimate. It needs the picker's perception to run from off-nominal poses, so take-over is the simpler first version.
   - Critical: the picker lands ~1.5 cm short without taught corrections. Use it with the taught-aim corrections (m3aim table) when labelling, or the policy will learn the bias.
3. **Human HG-DAgger for what the picker cannot label** (e.g., cube knocked over, gripper caught on the box rim). LeRobot's dagger strategy needs an active-motor leader arm, which the M3 lacks. Keyboard teach (already used for taught grips) is the realistic take-over interface. Record recovery and correction separately (RaC).
4. **Update rule.** Prefer retraining on aggregated data (demos + corrections, corrections up-weighted), or a small residual head trained on corrections with zero-residual samples for uncorrected states (TRANSIC / CR-DAgger). Avoid naive fine-tuning of the base policy on correction data only (CR-DAgger -15 to -30 %). Start the DAgger loop once real success is ≥10-20 %; below that, fix the sim gap (Q1-Q3) first.
5. **Budget.**
   - Real DAgger: 50-100 correction episodes per round (CR-DAgger), 2-3 rounds; TRANSIC used 227 trajectories over 4 tasks.
   - Residual RL with binary success: ~130-350 episodes per task in the humanoid study. With a human labelling success on the M3, that is a full day per task. Defer unless DAgger plateaus.
6. **Residual for the systematic offset.** If the policy's grasp is consistently short or offset, a tiny residual on the final pre-grasp pose (a 3-DoF Cartesian offset, bounded to ±1-2 cm) learned from corrections is far cheaper than retraining. It is the learned version of the taught-aim table.

### Gaps
- No paper found that uses a classical depth+IK picker (rather than a motion planner with privileged state) as a real-world DAgger labeller on a servo arm. GSWorld's expert labels in sim; whether the M3 picker's off-nominal perception is good enough for labels is untested.
- No reliable source found for "how many real episodes" specifically for ACT on a 5-DOF servo arm grasping 30 mm cubes. Numbers above come from Franka, humanoid and planar-pushing tasks.

## 6. What evaluation protocol should be used: number of trials, fixed layouts, success criteria, confidence intervals, and how do labs report sim-to-real success?

### Takeaway
Labs typically report 20-75 real trials per policy on a fixed grid of initial poses shared between sim and real:
- SIMPLER: 75 (25 positions × 3 orientations).
- Real-is-Sim: 60 (20 poses × 3).
- TRANSIC, PolaRiS, Wei et al.: 20.
- CR-DAgger: 20-32.

With 20 trials a 95 % interval spans ~30 points, so only large differences are resolvable. Best practice (TRI 2024): pre-registered success criteria, matched initial conditions, blind interleaved A/B in the same session, counts plus intervals, failure-mode taxonomy. Sequential testing (STEP, RSS 2025) cuts trials by up to 32 %.

### Cited Findings
- [authors' recommendations, Sep 2024] Kress-Gazit et al. (TRI), "Robot Learning as an Empirical Science":
  - Report counts, not only percentages.
  - Give interval estimates or a Bayesian posterior (their example uses a Beta(1,1) prior and gives P(B better than A) = 0.11 for 15/18 vs 11/17).
  - Interleave policies blind within one session.
  - Control and match initial conditions with image overlays or scene markings rather than randomising.
  - Define success in advance; the designer writes the criteria and a different person evaluates. Use subgoal rubrics; report failure modes with frequencies.
  - Report smoothness (SPARC) as a secondary metric.
  - Example: on matched initial conditions, a lab reorganisation dropped policy A from 13/20 to 0/8.
  — [arXiv 2409.09491](https://arxiv.org/html/2409.09491v2)
- [measured, Mar/Jun 2025] Snyder et al. (RSS 2025) note that policy comparisons are usually limited to 10-50 trials. Their near-optimal sequential test decides after each batch whether to continue, cutting trials by up to 32 % while keeping error guarantees. — [arXiv 2503.10966](https://arxiv.org/abs/2503.10966)
- [measured] Trial counts in sim-to-real papers:
  - SIMPLER Pick Coke Can: 25 grid positions × 3 orientations = 75 trials. — [SIMPLER](https://arxiv.org/html/2405.05941)
  - Real-is-Sim: 20 start poses × 3 = 60 per checkpoint. — [Real-is-Sim](https://arxiv.org/html/2504.03597v2)
  - PolaRiS: 20 real rollouts per policy per environment, 50 in sim. — [PolaRiS](https://arxiv.org/html/2512.16881)
  - TRANSIC: 20 per task, varied object and robot poses. — [TRANSIC](https://arxiv.org/html/2405.10315v3)
  - Splat soft-body twin: 16-27 fixed configurations per checkpoint, called "limited" by the authors. — [arXiv 2511.04665](https://arxiv.org/html/2511.04665v2)
  - Wei et al.: 20 real trials per policy, 200 in sim. — [arXiv 2503.22634](https://arxiv.org/html/2503.22634v2)
  - CR-DAgger: 20-32 per task. — [arXiv 2506.16685](https://arxiv.org/html/2506.16685)
- [measured, Feb 2024] UMI reports success as counts out of a fixed object set, e.g. 105/120 vs 69/120. — [UMI](https://arxiv.org/html/2402.10329)
- [computed by the researcher, standard binomial formulas, 2026-10-26; arithmetic, not a source claim]
  - Wilson 95 % intervals: 17/20 → [0.64, 0.95]; 18/20 → [0.70, 0.97]; 26/30 → [0.70, 0.95]; 43/50 → [0.74, 0.93]; 85/100 → [0.77, 0.91]; 20/20 → [0.84, 1.00]; 30/30 → [0.89, 1.00].
  - Zero failures in n trials bounds the failure rate at ≈3/n (20 → 14 %, 30 → 9.5 %, 50 → 5.8 %).
  - Trials per arm for 80 % power at two-sided α = 0.05 (unpaired): 50 vs 80 % ≈ 39; 60 vs 90 % ≈ 32; 70 vs 90 % ≈ 62; 85 vs 95 % ≈ 141.

### Inferences
Opinion: an M3 evaluation protocol.
1. **Fixed layout grid**, identical in Isaac and on the real floor or table.
   - Grid: e.g. 5 radial positions × 4 distances inside the reach envelope × 2 cube yaws (0 / 45 deg) = 40 cells, marked with tape or a printed mat registered to the ChArUco board.
   - Kit pieces: 30 mm cube, 40 mm cube, 30×30×60 block.
   - A screening set: 20 cells.
   - An "edge" set near the reach and FOV limits, kept separate (out-of-distribution, reported separately).
   - Set each layout by overlaying a reference photo in the wrist-camera view (TRI's matched-initial-conditions practice).
2. **Pre-registered success definition.** Success = the cube is lifted ≥3 cm, held 3 s, and dropped in the box, with no guard trip and no human touch. Partial rubric: reached pre-grasp / fingers closed on cube / lifted / placed. Continuous metrics:
   - Fingertip-to-cube-centre offset at closure, measured from the wrist frame (mm). This has far more statistical power than binary success at ~10-20 trials.
   - Time to grasp.
   - Number of guard trips.
   Failure taxonomy: perception miss, short or long reach, lateral miss, collision or guard, gripper timing, drop.
3. **Paired, blind, interleaved comparisons.** Run policy A and B (or learned vs classical picker) on the same cells in the same session in random order, with an operator who does not know which is running. Compare with a paired test (McNemar on discordant cells) or a Beta-Binomial posterior, and report counts plus Wilson intervals.
4. **Trial budget.**
   - Screening (does the policy work at all?): 20 trials. 17/20 means "somewhere between 64 and 95 %".
   - Choosing between two candidates with a big gap (≥25-30 points): ~30-40 per arm.
   - "Learned policy ≥ classical picker at ~90 %" needs ≥100-140 per arm unpaired. Not realistic for two people. Use paired cells + sequential stopping (Snyder et al.) + the continuous mm metric, and state the limitation.
   - For sim, run ≥50-200 per policy on the same grid.
5. **Safety protocol for real rollouts.**
   - First every new checkpoint passes Tier 0-1 (Q4).
   - The same safety guard code runs in sim and on the robot (joint limits, workspace box, the chassis-guard mesh), plus a per-tick clamp on command change. The servo executes commands over the move time, so a clamp of a few degrees per 100 ms limits the damage from one bad prediction.
   - Then: first real rollouts at reduced speed (longer move times), with one person on STOP; abort an episode on guard trip or no progress for N ticks; log every abort as a failure with its category.
   - Escalate speed only after ≥10 clean rollouts.
   - The residual-RL study relied on force-torque limits and collision checks; the M3 has no force sensing, so the kinematic guard and STOP are the only layers.
6. **Reporting.** Per checkpoint: sim success on the grid (n), real success (n, Wilson CI), sim-real Pearson r and MMRV across checkpoints once ≥4 exist, mm offsets, failure taxonomy, conditions (lighting, camera exposure locked, floor or table, date).

### Gaps
- No 2024-2026 standard was found for how many trials a low-cost-arm or LeRobot community result should report. Community practice visible in the LeRobot docs is about data collection (≥50 episodes), not evaluation sample size.
- The per-difficulty trial tables of Snyder et al. were not retrieved; only the up-to-32 % saving is cited.
