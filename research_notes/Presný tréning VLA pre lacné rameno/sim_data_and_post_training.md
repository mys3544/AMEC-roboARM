# Training data for a precise VLA / visuomotor policy on a low-cost servo arm: simulated privileged experts, sim-to-real, and post-training (DAgger, RL fine-tuning)

Research date: 2026-10-20. Scope: the data side of the plan "Isaac Sim scene of our arm + table + kit pieces -> privileged scripted expert (classical IK picker fed the simulator's true object poses) generates demos and DAgger corrections -> VLA (SmolVLA / X-VLA / pi0.5) trained on them -> fine-tuned with some real episodes from the real classical picker". Dates are arXiv submission dates unless stated otherwise.

## 1. Privileged-expert / scripted-teacher data generation in simulation (Isaac Lab Mimic, MimicGen family, RoboCasa, GR00T synthetic data, scripted experts in LIBERO/ManiSkill)

### Takeaway
Generating demonstrations automatically in simulation is a mature recipe. MimicGen-style systems turn 10-200 seed demos into 1K-50K demos per task set. Real-world transfer is solid for pick-and-place-class tasks: a co-trained policy averaged 83% against 45% for real-only, and DexMimicGen's real-to-sim-to-real test reached 90%. Gains taper with data volume: past roughly 1K generated demos per task, more data helps little or not at all. A *privileged scripted expert* (motion planner / IK fed true poses) is the limiting case of the same idea, and ManiSkill already ships motion-planning experts. Isaac Lab Mimic's own Franka stacking example reaches only 40-60% policy success from 1,000 generated demos, which shows the tooling is not a precision guarantee.

### Cited Findings
#### MimicGen family (NVIDIA / UT Austin)
- MimicGen (CoRL 2023): >50K demos across 18 tasks generated from ~200 human demos. Using 10, 50 or 200 source demos on Square / Three Piece Assembly changed policy success only modestly (2-21%), but a single source demo was "much worse" on Square. Source demos differ a lot in usefulness: on Threading D0 one source demo yielded >170 generated episodes and another <10. — [MimicGen arXiv 2310.17596](https://arxiv.org/abs/2310.17596)
- SkillMimicGen / SkillGen (CoRL 2024): >24K demos over 18 task variants from 60 human demos. It segments demos into skills, adapts each skill to the new object poses, and connects them with motion-planned free-space transit. Agents are on average 24% more successful than with MimicGen data. It was applied to 3 real tasks, plus zero-shot sim-to-real on one long-horizon assembly task. — [SkillMimicGen arXiv 2410.18907](https://arxiv.org/abs/2410.18907). The appendix shows a much higher *generation* (data-yield) rate than MimicGen, e.g. 99.8% vs 73.7% on Square D0 — [SkillMimicGen PDF](https://arxiv.org/pdf/2410.18907)
- DexMimicGen (2024-10): 60 source demos (10 per gripper task, 5 per dexterous task) -> 1,000 per task, ~21K total. Generation yield on BiGym: FlipCup 29.1%, DishwasherLoadPlates 43.6%, CupBoardsCloseAll 76.4%. Policy success rises clearly from 100 to 500-1,000 generated demos, but "does not always increase from 1000 to 5000". Real-to-sim-to-real Can Sorting on a GR-1 humanoid: 4 human demos -> 40 successful generated demos in a digital twin, replayed on the robot. Diffusion Policy on those 40 reached 90% (10 trials per colour); trained on the 4 source demos it reached 0%. — [DexMimicGen arXiv 2410.24185v2](https://arxiv.org/html/2410.24185v2)

#### Isaac Lab Mimic (the tool we would actually use in Isaac Sim)
- Workflow: 10 human demos for Franka cube stacking, annotated with subtask boundaries (auto heuristics for stacking, or manual). Generation runs with `--num_envs 1000 --generation_num_trials 1000` (state) or `--num_envs 300` (visuomotor), sized for an RTX PRO 6000 Blackwell. About 1,000 demos "gave good results" for BC-RNN. Expected generation success is ~40% (as high as 70% for simple cases, <1% for hard tasks or complex robots). Generation takes ~15 min (state) / ~1 h (visuomotor); training takes ~30 min (state) / ~6 h (visuomotor, 600 epochs). Reported policy success is only ~40-60%. Segments are stitched by linear interpolation; use few subtasks and put boundaries where interpolated motion will not collide. Smooth SpaceMouse demos are preferred over keyboard demos. — [Isaac Lab teleop & imitation docs](https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/teleop_imitation.html)
- Config knobs: `generation_guarantee` (retry until the target count of successes), `max_num_failures` (default 50), `generation_keep_failed` (default False; keep failures to debug low yield). — [Isaac Lab mimic_env_cfg source](https://isaac-sim.github.io/IsaacLab/v2.2.1/_modules/isaaclab/envs/mimic_env_cfg.html)
- Per-subtask action noise is built into Mimic generation. The Franka stack config uses `action_noise=0.03` with `apply_noise_during_interpolation=False` — [Franka stack Mimic cfg](https://isaac-sim.github.io/IsaacLab/v2.3.0/_modules/isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env_cfg.html); the AgiBot place-toy config uses 0.01 on its final subtask — [AgiBot Mimic cfg](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab_mimic/envs/agibot_place_toy2box_mimic_env_cfg.html)

#### Scale examples from RoboCasa / GR00T
- RoboCasa prior data used in the NVIDIA co-training study: 60K Panda demos over 20 RoboCasa tasks (a subset of a 72K set). The humanoid set has 10K (1K per task via DexMimicGen). The task-matched "digital cousin" sets have 10K per task, MimicGen'd from 100 source demos, with 100 textures each for floor, walls, cabinets and counters. — [Sim-and-Real Co-Training arXiv 2503.24361v2](https://arxiv.org/html/2503.24361v2)
- GR00T N1 (2025-03): 780K synthetic trajectories (≈6.5K hours of human-demo equivalent) generated in ~11 h of compute. Combined with real data, they improved GR00T N1 by 40% over real-only. — [NVIDIA blog, synthetic motion generation](https://developer.nvidia.com/blog/building-a-synthetic-motion-generation-pipeline-for-humanoid-robot-learning). Conflict: another NVIDIA post says "over 750K" — [NVIDIA blog, GR00T N1](https://developer.nvidia.com/blog/accelerate-generalist-humanoid-robot-development-with-nvidia-isaac-gr00t-n1)
- GR00T-Dreams blueprint (Cosmos Predict-2 + Cosmos Reason; from DreamGen research): generates trajectory data from one image + a language prompt by generating videos and then extracting actions with an inverse-dynamics model. NVIDIA says it was used to develop GR00T N1.5 in 36 h, versus ~3 months of manual collection. — [NVIDIA blog, GR00T-Dreams](https://developer.nvidia.com/blog/enhance-robot-learning-with-synthetic-trajectory-data-generated-by-world-foundation-models)

#### Scripted experts in benchmarks
- ManiSkill2 shipped 1,000 successful trajectories for PegInsertionSide; success = half the peg inserted — [ManiSkill2 arXiv 2302.04659](https://arxiv.org/pdf/2302.04659). In ManiSkill3 PegInsertionSide-v1 the hole radius = peg radius + 3 mm clearance, peg radius randomized 15-25 mm, success when the peg end is within 15 mm of the box centre — [ManiSkill PegInsertionSide source](https://maniskill.readthedocs.io/en/latest/_modules/mani_skill/envs/tasks/tabletop/peg_insertion_side.html)
- Planners are good at long-horizon structure but poor at contact-rich phases. HITL-TAMP hands contact-rich segments to a human and the rest to a TAMP planner, producing >3x more demos per unit time than plain teleop. — [HITL-TAMP arXiv 2310.16014](https://arxiv.org/pdf/2310.16014)

### Inferences
- Our classical IK picker given true object poses is a stronger expert than MimicGen's "replay a transformed human segment": it can label *any* state, so it supports DAgger relabelling (section 2) and unlimited, always-successful generation. The MimicGen-family lessons still apply: (a) returns flatten around 1K demos per task variant, (b) diversity of poses, objects and backgrounds matters more than count, (c) keep a few failures for debugging the yield.
- Isaac Lab Mimic's own example needs a human seed demo and subtask annotation. With a scripted expert we can skip Mimic and write the generator directly in the running Isaac Sim (as `train/synth/kit_synth.py` already does for detector images). Mimic's per-subtask `action_noise` (0.01-0.03) is the built-in equivalent of DART noise, worth copying.
- The low 40-60% policy success of Isaac Lab's stacking example (BC-RNN, 1K demos) is a warning that the learner and action representation, not the data volume, often set the ceiling.

### Gaps
- No source reports real-world mm-level placement accuracy for a VLA trained on MimicGen-style or scripted sim data. Papers report success rates against task-specific tolerances, not mm error.
- SkillMimicGen real-world success numbers (its Table F.2) were not retrievable in this session.
- No published head-to-head comparison of a *privileged scripted planner* expert against human-seeded MimicGen data on the same task was found.
- GR00T-Dreams/DreamGen quantitative success numbers on real robots were not verified.

## 2. Teacher-student distillation (privileged state teacher -> vision student), and DAgger vs pure BC for precision

### Takeaway
The dominant 2024-2026 recipe is a privileged teacher (an RL policy or a planner with true state) distilled into a pixel student with a BC + DAgger mixture: BC gives fast initialization, DAgger covers the states the student actually visits. For precision tasks, plain BC on chunked policies plateaus even with thousands of demos. DAgger helps (one_leg ~90% vs DP 54%) but is beaten by closed-loop residual RL (98%). Distillation itself can become the bottleneck: going from 10K to 100K synthetic trajectories gave only 78% -> 80%.

### Cited Findings
- ResiP (MIT, 2024-07): BC on precise FurnitureBench assembly *plateaus*: round_table only 56% even at 10,000 demos. DP-DAgger reaches ~90% on one_leg with ~2,000 demos and 71% on round_table, but ResiP (residual RL) beats DP-DAgger by 9 points (one_leg) and 23 points (round_table). In ResiP's sim-to-real pipeline, a state-based teacher at 98% was distilled to an RGB student; scaling synthetic rendered data from 10K to 100K trajectories only moved success 78% -> 80%, "pointing to the distillation process as the bottleneck". — [ResiP arXiv 2407.16677v4](https://arxiv.org/html/2407.16677v4)
- ResiP's diagnosis: chunked BC policies (Ta=32, execute 8, ~1 s per chunk at 10 Hz) "act more like trajectory planners than closed-loop controllers". The per-step residual (~0.1 s) fixes failures like premature insertion. Under random force perturbations ResiP drops 12 points, while chunked methods (DP, DP-DAgger, chunk-level ResiP-C) drop 19-26 points. — [ResiP arXiv 2407.16677v4](https://arxiv.org/html/2407.16677v4)
- VIRAL (humanoid visual sim-to-real, 2025-11): RGB student distilled from a privileged teacher with a *mixture* of online DAgger and BC. Teacher rollouts give clean near-optimal demos, student rollouts cover states off the teacher's distribution, and the mix combines BC's fast initialization with DAgger's state coverage. — [VIRAL arXiv 2511.15200](https://arxiv.org/pdf/2511.15200)
- Rebar insertion under geometric variation (2026-09): same hybrid with a 0.5 mixing ratio taken from a prior ablation. The student has 8 RGB views + proprioception through ResNet-18, and everything visual is randomized during distillation. — [arXiv 2609.20477](https://arxiv.org/pdf/2609.20477)
- Visuotactile insertion with compliant hands (2024-11): BC first, then DAgger with the teacher's control probability decaying; learns faster than direct RL. — [arXiv 2411.06408](https://arxiv.org/pdf/2411.06408)
- Squint (SO-101, 2026-02/09): *BC and visual DAgger from a state-based expert* were the weak baselines. Sim average: BC 41.9% vs direct visual RL (Squint) 96.1%. Real SO-101: BC 38/80 vs Squint 73/80. The authors attribute this to state-distribution mismatch and learning from a state-based expert. — [Squint arXiv 2602.21203v2](https://arxiv.org/html/2602.21203v2)
- TRANSIC (CoRL 2024): sim teacher = PPO with privileged observations + OSC actions in Isaac Gym. The student is BC'd on teacher rollouts relabelled to joint-position actions, with synthetic point clouds and a GMM head. — [TRANSIC arXiv 2405.10315v3](https://arxiv.org/html/2405.10315v3)
- HIL-SERL comparison on real precision tasks (Franka-class, 2024-10): HG-DAgger (human DAgger) averaged 49.7% across 13 tasks vs 100% for HIL-SERL RL. The RL policy's state visitation forms a "funnel" toward the target, while DAgger's is diffuse, because DAgger "explores only near the current policy". — [HIL-SERL arXiv 2410.21845](https://arxiv.org/html/2410.21845)

### Inferences
- For our plan the expert is already privileged (true poses), so "learning by cheating" applies directly. The cheap win is DAgger *in simulation*: roll out the student VLA, and at every visited state query the IK picker for the action/chunk it would take from there. This is the BC+DAgger mixture of VIRAL / the rebar paper, at zero human cost. Expect it to raise robustness and recovery more than final mm precision.
- Watch for an *information asymmetry* trap. The expert sees exact poses; the student sees pixels from a wrist camera. Where the camera cannot resolve the pose (occlusion during descent, motion blur), the student cannot imitate the expert. Fixes: make the expert act on noisy, delayed poses comparable to what vision provides, or add an explicit re-look step, as our classical picker already does.
- Evidence is consistent that the last few mm and the last ~10-40% of success come from closed-loop correction (residual RL, real RL), not from more BC/DAgger data (sections 5 and 6).

### Gaps
- No source quantified DAgger vs pure BC *in mm of placement error*; all comparisons are success rates.
- VIRAL's mixture-coefficient ablation (its Fig. 11) was not inspected.
- No published DAgger relabelling scheme for action-chunked VLAs (SmolVLA / pi0.5 output 50-step chunks) using a scripted expert was found; how to relabel chunks from student-visited states is open.

## 3. Sim-to-real for precise, contact-rich manipulation: randomization, actuator models / system ID, real-to-sim reconstruction, residual gap, co-training ratios

### Takeaway
Three things matter most for low-cost arms in 2025-2026:
1. Match the camera viewpoint, and randomize lighting and colour.
2. Paste the real background into sim (green-screen overlay or Gaussian splats).
3. Co-train heavily on sim data, with real data a small but non-zero fraction: 99% sim : 1% real by sampling weight was best in NVIDIA's study, and 99.9% hurt.

With that, zero-shot RGB sim-to-real reaches ~90% on simple picks on SO-100/SO-101-class arms. Sim-trained RL insertion on industrial arms with force sensing reaches 83-99% (IndustReal), with clearances claimed down to 0.02 mm (InsertAnything, 2026-09). Hobby-servo actuation (backlash, deadband, delay) is the least solved part. NVIDIA's own SO-101 course treats it with a learned actuator-gap model (SAGE + GapONet) rather than an analytic one.

### Cited Findings
#### Co-training ratios and results
- Sim-and-Real Co-Training (NVIDIA/UT Austin, RSS 2025, 2025-03). Real demos: 50 per task (Panda) / 20 per task (GR-1). Average real success: real-only 45.3%, + task-matched "digital cousin" sim 81.1%, + prior multi-task sim 76.8%, + both 83.2%. Ratio ablation (CupPnP, 20 real + 1,000 sim): a 50% ratio is suboptimal, 99% sim gave 95% success, 99.5% / 99.9% dropped to 60%. Sim volume matters: C2SPnP 67% with 10K sim vs 53% with 500. Camera misalignment costs 67% -> 56% (Panda) and 95% -> 70% (GR-1), but perfect calibration is not required. Co-training still beats real-only at 400 real demos. Sim tasks must share the real task's success criterion. Generalization: unseen positions Panda 11% -> 28%, GR-1 43% -> 100%. — [arXiv 2503.24361v2](https://arxiv.org/html/2503.24361v2) (the paper is internally inconsistent on CloseDoor DC demo count, 10K vs 50)
- RL-Co (RLinf, 2026-02, v4 2026-06): SFT on mixed real+sim (ratio α), then RL in sim with an auxiliary SFT loss on real data to prevent forgetting. Four real tasks (pick-and-place, push cube, open/close drawer), average success:
  - OpenVLA: real-only 16.5%, SFT co-train 40.0%, RL-Co 64.0%
  - pi0.5: real-only 26.7%, SFT co-train 45.9%, RL-Co 66.2%

  — [RLinf-Co docs](https://rlinf.readthedocs.io/en/latest/rst_source/resources/publications/rlinf_co.html); [arXiv 2602.12628](https://arxiv.org/abs/2602.12628)
- V-CAGE (2026-04): a VLA with 10 real demos reached 20% real success; adding 250 simulated episodes raised it to 55% (search-snippet level, not verified in full text). — [V-CAGE arXiv 2604.09036](https://arxiv.org/pdf/2604.09036)
- NVIDIA "Train an SO-101 from Sim-to-Real" course: GR00T N1.6 co-trained on 5 real teleop episodes + "a much larger set" (70-100) of sim demos. The four strategies are domain randomization, co-training with real data, Cosmos Transfer 2.5 augmentation, and SAGE + GapONet actuation-gap closing. No success numbers are given. — [NVIDIA SO-101 course, co-training](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/13-strategy2-cotraining.html)

#### Zero-shot visual sim-to-real on low-cost arms
- Squint (SO-101, ManiSkill3, RA-L 2026): visual SAC, wrist camera only, 128x128 render downsampled to 16x16, ~15 min on one RTX 3090. Zero-shot real: 73/80 = 91.3% over 8 tasks (reach / lift / place / stack, cube and can), objects in a 0.2 x 0.2 m box. DR: small wrist-camera pose/FOV perturbations, lighting and colour jitter, object size and friction, gripper closing speed, and joint-position noise σ = 5°. **Removing colour jitter dropped real success 91.3% -> 72.5%.** Sim control is 10 Hz PD joint-delta targets; on the real robot actions are scaled 0.15 and run at 30 Hz. No servo system identification is reported. Failures: can tipping, gripper friction, stack-can 6/10. — [Squint arXiv 2602.21203v2](https://arxiv.org/html/2602.21203v2)
- lerobot-sim2real (SO-100, ManiSkill3): zero-shot RGB cube grasping with no real data. Procedure: overlay the sim render on the live camera and align camera pose/FOV by hand; then capture a background photo with the robot removed and use it as a "greenscreen" behind the simulated arm; randomize robot colour. PPO for 25-40M steps (~1 h on a 4090); over-training can hurt, and checkpoints with equal sim success transfer differently. The repo ships a `system_id_so100.npy` but documents no method, and the author now recommends Squint. — [lerobot-sim2real RGB tutorial](https://github.com/StoneT2000/lerobot-sim2real/blob/main/docs/zero_shot_rgb_sim2real.md); [lerobot-sim2real README](https://github.com/StoneT2000/lerobot-sim2real)
- ManiSkill's documented zero-shot demo on a Koch v1.1 low-cost arm: PPO on RGB + proprioception, ~1 h on a 4090, PickCube 18/20 real. — [ManiSkill docs gallery](https://maniskill.readthedocs.io/en/latest/user_guide/demos/gallery.html) (reported via search summary; page not opened)

#### Real-to-sim reconstruction (Gaussian splats)
- SplatSim (CMU, 2024-09): replaces mesh rendering with Gaussian splats. Zero-shot sim-trained RGB policies averaged 86.25% real success vs 97.5% for policies trained on real data; rigid objects only. — [SplatSim arXiv 2409.10161](https://arxiv.org/html/2409.10161v1)
- NVIDIA Omniverse NuRec: Gaussian-splat reconstructions packaged as OpenUSD ParticleField scenes, rendered by RTX alongside meshes in Isaac Sim (docs for 5.0.0-6.2.0). Splats are trained with 3DGruT; "Fixer" (Cosmos-based) cleans artefacts. — [NuRec overview](https://developer.nvidia.com/omniverse/nurec); [Isaac Sim 5.0 NuRec docs](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/assets/usd_assets_nurec.html)

#### Actuator modelling and system ID
- Isaac Lab actuator classes: implicit (physics-engine PD) and explicit models: IdealPD (torque with saturation), DCMotor (velocity-dependent saturation), DelayedPD (ideal PD with delayed command application), and an MLP-based ActuatorNet. — [Isaac Lab actuators API](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/api/lab/isaaclab.actuators.html)
- SAGE (Tongji/PKU/NVIDIA) measures the per-joint gap from paired real/sim runs of chirp-like scripts (`oscillation_low_freq`, `random_waypoints`, `actuator_bandwidth`, `pick_place`), comparing position, velocity and torque. The example sim runs physics at 200 Hz, control at 50 Hz, kp=100, kd=2. GapONet is a network that maps command -> real motion, trained on ~8 h of real SO-101 data. Advice: tune parameters when the gap is small, use GapONet when it is large; focus on the gripper and wrist; hobby-servo backlash accumulates along the chain. No numeric before/after error is given in the text. — [NVIDIA SO-101 course, SAGE + GapONet](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/15-strategy4-sage.html)
- Plain Coulomb-viscous friction misrepresents servo actuators; extended friction models identified on a pendulum bench improve 2R-arm simulation accuracy. — [Extended friction models for servo actuators, arXiv 2410.08650](https://arxiv.org/html/2410.08650v1)
- NVIDIA's industrial-assembly sim-to-real in Isaac Lab randomized joint friction, damping and controller gains plus observation noise. — [NVIDIA blog, sim-to-real for assembly in Isaac Lab](https://developer.nvidia.com/blog/bridging-the-sim-to-real-gap-for-industrial-robotic-assembly-applications-using-nvidia-isaac-lab/)

#### How far precision sim-to-real goes (industrial arms, mostly force/torque sensing)
- IndustReal (NVIDIA, 2023): RL in Isaac Gym with SDF rewards, a sampling-based curriculum, simulation-aware policy updates and a policy-level action integrator. 83-99% success over 600 real trials on a Franka, no real-world adaptation. Insert-only rows are ~7-8/10 per peg; in full pick-place-insert, pegs reach 80% success, gears 97.5%, connectors 100%. — [IndustReal ar5iv](https://ar5iv.labs.arxiv.org/html/2305.17110); [IndustReal PDF](https://arxiv.org/pdf/2305.17110)
- AutoMate (2024): specialist policies at 86.5% mean on 20 real assemblies, only 4.2 points below sim. — [NVIDIA blog, AutoMate](https://developer.nvidia.com/blog/training-sim-to-real-transferable-robotic-assembly-skills-over-diverse-geometries)
- InsertAnything (2026-09-21): RL trained entirely in sim with target poses + 3D fingertip force feedback. Minimum nominal clearance 0.02 mm; one policy trained on hexagonal insertion reached 95.0% on 8 unseen real tasks. — [InsertAnything arXiv 2609.24511](https://arxiv.org/abs/2609.24511)
- ResiP real one_leg (Franka, 10 trials): RGB with 40 real demos ~20-30% real-only vs ~50-60% real + ~400 synthetic rendered demos. With black parts (unseen colour) real-only failed every trial; real+sim kept working. — [ResiP arXiv 2407.16677v4](https://arxiv.org/html/2407.16677v4)
- TRANSIC (CoRL 2024): sim policy plus a residual learned from real human corrections. 81% average on four FurnitureBench square-table subskills (Stabilize 100%, Reach-and-Grasp 95%, Insert 45%, Screw 85%; 20 trials each). Human effort was 17-100 corrected trajectories per task. Under exaggerated sim-real gaps it averaged 77% vs 18% for IWR. — [TRANSIC arXiv 2405.10315v3](https://arxiv.org/html/2405.10315v3)

### Inferences
- For our arms the biggest visual levers are cheap and already half-built. The wrist camera is the right input (Squint uses only the wrist camera). The existing splat pipeline (`tools/splat_cycle.py`) can produce NuRec/3DGruT-style backgrounds, or simply a lerobot-sim2real-style background photo from the survey pose. Add colour/lighting jitter (worth ~19 points in Squint).
- Co-training ratio to start with: real episodes from the classical picker at ~1-10% of sampled batches. The NVIDIA optimum was 99:1 by sampling weight with 20 real vs 1,000 sim demos; 99.9:0.1 collapsed. Sweep it.
- Actuators: our Yahboom bus servos are position-controlled like the SO-101's Feetech servos. A practical Isaac Lab model is an implicit/IdealPD position actuator with DelayedPD-style command delay, randomized kp/kd, joint friction and deadband. Identify these from logged chirp / waypoint scripts on the real arm (SAGE-style), and add joint-position observation noise of a few degrees (Squint used σ = 5°). Whether a learned GapONet-style model is needed depends on how large the measured gap is.
- Sim-only mm-level transfer is demonstrated almost exclusively with force sensing and stiff industrial arms. For a hobby-servo arm without force sensing, expect the floor of achievable precision to be set by servo backlash/deadband and camera calibration rather than by the policy (see the companion note `hardware_precision_limits.md`).

### Gaps
- No source gives quantitative backlash/deadband models or identified parameters for Feetech/Yahboom-type bus servos in Isaac Lab; SAGE's numbers are only plotted.
- No real-world mm error numbers for SO-100/SO-101 sim-to-real policies were found; Squint and lerobot-sim2real report task success only.
- The optimal sim:real ratio for *VLA* fine-tuning (SmolVLA / X-VLA / pi0.5), rather than for Diffusion Policy-class models, was not found; RL-Co does not state α in the retrieved text.
- No retrieved IndustReal clearance values in mm.

## 4. How many demos, quality vs quantity, and whether "perfect" scripted demos hurt robustness (recoveries via noise injection, DART, DAgger)

### Takeaway
Diversity (poses, objects, scenes) beats raw count. Per-condition demo counts saturate quickly, and around 1K demos per task is typical for sim-generated sets. Clean, always-successful expert trajectories omit recovery states, which causes compounding errors. The standard remedies are:
- DART-style noise injected into the expert's executed actions while recording clean labels (Mimic already applies 0.01-0.03 action noise);
- on-policy relabelling (DAgger, HG-DAgger, CR-DAgger);
- explicitly collected recovery data (RaC: 10x less data than the previous best).

### Cited Findings
- Data Scaling Laws in IL (ICLR 2025, UMI + Diffusion Policy, >40K demos, 32 environments, 64 objects): generalization follows a rough power law in the number of environments and objects. Beyond a threshold of demos per environment/object, extra demos have minimal effect, and saturation comes sooner when both vary. — [arXiv 2410.18647v3](https://arxiv.org/html/2410.18647v3)
- DexMimicGen: success rises 100 -> 500 -> 1,000 generated demos but not reliably 1,000 -> 5,000. — [arXiv 2410.24185v2](https://arxiv.org/html/2410.24185v2). Co-training study: sim count matters (500 vs 10K: 53% vs 67%). — [arXiv 2503.24361v2](https://arxiv.org/html/2503.24361v2)
- DART (CoRL 2017): BC errors compound off the supervisor's distribution. DART injects noise into the supervisor's controls during demonstration so the supervisor shows how to recover; the noise should match the trained policy's error. In MuJoCo it was up to 3x faster in computation than DAgger on Humanoid, with a 5% loss of supervisor reward (vs 80% for DAgger's executed policies). On a real grasping-in-clutter task it gave +62% over BC. — [DART arXiv 1703.09327](https://arxiv.org/abs/1703.09327); [BAIR blog](https://bair.berkeley.edu/blog/2017/10/26/dart)
- Isaac Lab Mimic applies per-subtask action noise (0.03 Franka stack, 0.01 AgiBot) and by default not during interpolation segments. — [Franka stack Mimic cfg](https://isaac-sim.github.io/IsaacLab/v2.3.0/_modules/isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env_cfg.html)
- RaC (CMU, 2025-09): after imitation pre-training, fine-tune on human intervention rollouts that deliberately include *recovery* (rewind to a familiar state) as well as *correction* segments. Beats the prior state of the art with 10x less data-collection time on 3 real bimanual tasks + 1 sim assembly; performance scales linearly with the number of recovery manoeuvres exhibited at test time. — [RaC arXiv 2509.07953](https://arxiv.org/abs/2509.07953)
- CR-DAgger (Stanford, NeurIPS 2025): compliant "nudge" interventions + a force-aware residual policy. +64% base success over four contact-rich tasks (book flipping, belt assembly, cable routing, gear insertion; latest arXiv v5); earlier versions report >50% / >60% on two tasks. Beats retrain-from-scratch and fine-tuning at equal data budget. — [CR-DAgger arXiv 2506.16685v5](https://arxiv.org/html/2506.16685v5)
- Human DAgger alone is not enough for precision: HG-DAgger averaged 49.7% on HIL-SERL's 13 real tasks (RAM insertion 29%, timing belt 2%). — [HIL-SERL arXiv 2410.21845](https://arxiv.org/html/2410.21845)
- RL from 1 vs many demos (SimpleVLA-RL, RoboTwin 2.0): with 0 demos SFT, RL stays at 0% (no successes to learn from). With 100 demos 7.3% -> 25.4%; with 1,000 demos 28.2% -> 50.4%. A competent base policy is a prerequisite for RL post-training. — [SimpleVLA-RL arXiv 2509.09674v1](https://arxiv.org/html/2509.09674v1)

### Inferences
- For the scripted IK expert, generate with: (1) per-step action noise on the *executed* trajectory while recording the expert's clean action for the visited state (DART); (2) random perturbation events (nudged cube, slipped grasp, offset approach) followed by the expert's re-plan, so recoveries appear in the data; (3) DAgger rounds where the student drives and the expert labels. All three are free in sim; the RaC/CR-DAgger evidence suggests recovery content matters more than adding more clean episodes.
- A per-condition budget of ~1K episodes per task variant, spread over many object poses, kit pieces, textures and lighting, is consistent with every scaling result found. Spend extra compute on diversity rather than repeats.
- Real episodes from the classical picker are "clean scripted" too and contain its systematic biases, such as the known ~1.5 cm short grips on the M3. Filter real episodes by verified success, and consider adding small deliberate perturbations on the real robot as well.

### Gaps
- No controlled study was found that measures how *noise level* in demonstrations trades off against final mm precision (noise adds recoveries but can blur the fine-alignment phase).
- No study directly compares "perfect scripted demos" with noisy human demos on mm-tolerance tasks with VLAs.

## 5. Post-training for precision: RL fine-tuning of VLAs and residual RL on top of BC (HIL-SERL, ConRFT, RLinf, pi*0.6 RECAP, ResiP, PA-RL, Policy Decorator, SimpleVLA-RL, VLA-RL), especially insertion / low-cost arms

### Takeaway
RL post-training is where precision tasks go from roughly 30-80% to roughly 95-100%. Representative results:

| Method | Setting | Result |
|---|---|---|
| HIL-SERL (real RL) | 13 real tasks incl. RAM / USB insertion, timing belt | 49.7% -> 100%, ~1-2.5 h per task (timing belt 6 h) |
| ResiP (residual PPO on frozen DP, sim) | Assembly | peg-in-hole 5% -> 99% |
| ConRFT (real VLA fine-tuning) | 8 real tasks | 96.3% after 45-90 min |
| SimpleVLA-RL, πRL (sim VLA RL) | LIBERO | ~97-99% |
| pi*0.6 RECAP | Long tasks | >2x throughput, ~2x fewer failures |

For a low-cost arm the most practical routes are:
- **Residual RL in sim** on top of the frozen VLA/BC policy, small residual scale (0.1 in ResiP), per-step closed loop.
- **LeRobot's HIL-SERL** on the real arm: SO-100 configs exist, 10 Hz end-effector actions, 128x128 crops, reward classifier. LeRobot publishes no SO-100 success numbers.

### Cited Findings
#### Real-world RL / human-in-the-loop RL
- HIL-SERL (Luo et al., 2024-10): 20-30 demos in the buffer (baselines got 200), sparse reward from a binary classifier trained on ~200 positive + ~1,000 negative frames collected in ~5 min (>95% accuracy). ResNet-10 encoder, 128x128 wrist/side crops, one RTX 4090, 6-D Cartesian twist targets to an impedance controller, discrete gripper. Success, HG-DAgger -> HIL-SERL:

  | Task | Success | Training time |
  |---|---|---|
  | RAM insertion | 29% -> 100% | 1.5 h |
  | USB grasp-insertion | 26% -> 100% | 2.5 h |
  | Timing belt | 2% -> 100% | 6 h |
  | Car dashboard | 41% -> 100% | 2 h |
  | Average of 13 tasks | 49.7% -> 100% | - |

  Cycle time 9.6 s -> 5.4 s. Ablations on three tasks show all ingredients are needed: no demos + no interventions = 0%, and without interventions RAM insertion reaches 48%. Diffusion Policy got 27% on RAM insertion; the off-the-shelf residual-RL baseline got 0%. — [HIL-SERL arXiv 2410.21845](https://arxiv.org/html/2410.21845) (DP dashboard: table 18% vs text 28%, an internal inconsistency)
- LeRobot HIL-SERL (current docs): SO-100/SO-101 workflow:
  1. find end-effector bounds with `lerobot-find-joint-limits`;
  2. record demos (example config: 15 episodes; gamepad, keyboard or SO-101 leader; the geared SO-101 leader makes takeover smoother than the SO-100);
  3. crop ROIs and resize to 128x128;
  4. train a ResNet-10 reward classifier, recording with `terminate_on_success=false` to get enough positives;
  5. run the SAC actor/learner over gRPC at fps 10 with episodes ≤20 s (tasks of 5-10 s recommended) and `temperature_init` ~1e-2.

  The docs advise end-effector rather than joint action spaces ("some tasks are nearly impossible to learn in joint space"). Interventions cut the steps needed to reach max reward "by a quarter". Optional motor-current and joint-velocity observations are available for contact. No success-rate numbers for SO-100 are given. — [LeRobot HIL-SERL docs](https://huggingface.co/docs/lerobot/hilserl)
- ConRFT (RSS 2025): offline BC + Q-learning (consistency policy) on a small demo set, then online fine-tuning with human interventions. 96.3% average over 8 real tasks after 45-90 min online, +144% success and 1.9x shorter episodes vs supervised baselines. Only the action head is fine-tuned (encoders/backbone frozen). — [ConRFT arXiv 2502.05450v2](https://arxiv.org/html/2502.05450v2)
- PA-RL (Finn/Kumar groups, 2024-12): critic re-ranks and refines sampled actions, then the policy is distilled with a supervised loss; works for diffusion and autoregressive policies. OpenVLA-7B fine-tuned autonomously with Cal-QL went from 40% to 70% real success in ~40 min. — [PA-RL arXiv 2412.06685](https://arxiv.org/abs/2412.06685)
- RL-100 (2025-10, v4 2026-03): diffusion visuomotor policy, imitation + RL under one clipped-PPO objective inside denoising, then consistency-distilled to one-step control. 1000/1000 successes across its evaluated real tasks (up to 250 consecutive on one task), ~90% zero-shot under shifts, ~7 h failure-free juicing in a mall. — [RL-100 arXiv 2510.14830](https://arxiv.org/abs/2510.14830)
- pi*0.6 with RECAP (Physical Intelligence, 2025-11):
  - Training: offline-RL pretraining, then demo fine-tuning per task, then RL on autonomous episodes with expert teleop corrections and reward labels.
  - Mechanism: a value function predicts (negative) steps to completion, and the policy is conditioned on advantage, being told to act high-advantage at test time; corrections are forced to a positive label.
  - Results: more than 2x throughput on the hardest tasks and failure rates down 2x or more. Espresso throughput and success both more than double; laundry throughput +50% with success saturating early. Demonstrated runs: espresso ~18 h in one day, 59 factory boxes.

  — [PI blog pi*0.6](https://pi.website/blog/pistar06); [pi*0.6 arXiv 2511.14759](https://arxiv.org/pdf/2511.14759). No precision (mm) figures are given.

#### Residual RL on top of a frozen BC/VLA policy
- ResiP (MIT, 2024-07): frozen chunked Diffusion Policy + per-step Gaussian residual trained with PPO from sparse rewards; residual action scale 0.1; up to 500M sim steps with 1,024 envs; 50 demos per task. Success, DP -> ResiP (1,024 eval episodes each):

  | Task | Diffusion Policy | ResiP |
  |---|---|---|
  | one_leg | 54% | 98% |
  | round_table | 12% | 94% |
  | lamp | 7% | 97% |
  | mug-rack | 26% | 88% |
  | peg-in-hole | 5% | 99% |
  | biman-insert | 33% | 93% |

  Medium randomization: 70-77%. Per-step residual beats a chunk-level residual (98% vs 92% on one_leg). — [ResiP arXiv 2407.16677v4](https://arxiv.org/html/2407.16677v4)
- Policy Decorator (ICLR 2025): a model-agnostic small residual (MLP/CNN Gaussian) on a frozen large policy (Behavior Transformer, Diffusion Policy), with a bounded residual and a progressive exploration schedule. Improves 2 SOTA IL models on 8 ManiSkill/Adroit tasks while keeping IL smoothness and avoiding erratic RL behaviour. — [Policy Decorator arXiv 2412.13630](https://arxiv.org/abs/2412.13630)
- TRANSIC: a residual from 17-100 real correction trajectories per task brings a sim-trained policy to 81% real average on furniture subskills; insertion remained the hardest at 45%. — [TRANSIC arXiv 2405.10315v3](https://arxiv.org/html/2405.10315v3)

#### RL fine-tuning of VLAs in simulation
- SimpleVLA-RL (2025-09), OpenVLA-OFT, GRPO-style with binary outcome reward, 8x A800:
  - LIBERO average 91.0% -> 99.1%.
  - One demo per task: average 48.9% -> 96.9%, LIBERO-Long 17.3% -> 91.7% (the abstract says 17.1%).
  - RoboTwin 2.0: 38.3% -> 68.8%.
  - Sim-to-real on two AgileX Piper arms with no real demos: 17.5% -> 38.5% average.
  - Exploration tricks: dynamic sampling (drop all-success/all-fail groups), clip range [0.8, 1.28], rollout temperature 1.6, no KL.
  - Observed "pushcut": the RL policy learns to push objects instead of grasping them.

  — [SimpleVLA-RL arXiv 2509.09674v1](https://arxiv.org/html/2509.09674v1)
- πRL (RLinf, 2025-10): Flow-Noise / Flow-SDE make flow-matching VLAs RL-trainable. LIBERO few-shot SFT: pi0 57.6% -> 97.6%, pi0.5 77.1% -> 98.3%; pi0.5 one-shot LIBERO-Long 43.9% -> 94.0%. ManiSkill 4,352 pick-and-place variations with 320 parallel envs: v2 reports pi0 38.4% -> 78.8% and pi0.5 40.1% -> 90.8% (v1 reported 41.6 -> 85.7 / 40.0 -> 84.8). — [πRL arXiv 2510.25889v1](https://arxiv.org/html/2510.25889v1); [v2 PDF](https://arxiv.org/pdf/2510.25889v2)
- RLinf-VLA: on ManiSkill PPO beats GRPO and is more stable. OpenVLA-OFT 18.29% base -> 60.64% (GRPO) / 77.05% (PPO); OpenVLA 42.11% -> 75.15% / 81.93%. On LIBERO, OpenVLA-OFT 65.43% -> 98.11% (GRPO). — [RLinf-VLA docs](https://rlinf.readthedocs.io/en/latest/rst_source/publications/rlinf_vla.html)
- RL-Co (sim RL + real SFT regularizer) adds ~+20 points on pi0.5 over SFT co-training in the real world (45.9% -> 66.2%). — [RLinf-Co docs](https://rlinf.readthedocs.io/en/latest/rst_source/resources/publications/rlinf_co.html)

### Inferences
- Most promising order for our plan:
  1. VLA SFT on sim expert data (+ DAgger) co-trained with real picker episodes.
  2. RL in Isaac Sim, either as πRL/RLinf-style VLA RL with the action-head only (as ConRFT does), or, cheaper and closer to ResiP / Policy Decorator, a small per-step residual on the end-effector command with bounded scale. The sim reward is free: the simulator knows the true cube pose, so success = within X mm of target.
  3. Optionally HIL-SERL-style real RL of a small residual on the real arm with a reward classifier.

  ResiP's per-step residual directly targets the chunked-VLA "open-loop trajectory planner" failure mode that limits precision.
- Precision RL needs a base policy that already succeeds sometimes; SimpleVLA-RL shows 0 demos -> 0% stays 0%. Our sim expert data provides that.
- Beware reward hacking like "pushcut". For kit pieces, define success with orientation and grasp-held checks, or the policy may shove pieces into place.
- HIL-SERL's evidence is on a Franka with an impedance controller. On a position-servo arm with backlash, the achievable gain is unknown; the LeRobot docs give no SO-100 numbers.

### Gaps
- No published success rates for LeRobot HIL-SERL on SO-100/SO-101 were found (docs only show intervention-rate curves).
- VLA-RL (2025) numbers were not retrieved.
- No source reports RL fine-tuning results on a *hobby-servo* arm for an insertion or tight-tolerance task; precision RL evidence is on Franka/UR-class arms.
- Wall-clock of ResiP training and Policy Decorator per-task numbers were not retrieved.

## 6. Automatic success detection and resets for unattended data collection (real and sim)

### Takeaway
In sim, success and resets are free (privileged state). On real robots the standard pattern is:
- a learned binary success classifier on cropped camera images (HIL-SERL: ~1,200 frames, >95% accuracy; LeRobot ships one);
- or a VLM-based success check;
- plus a reset policy or scripted reset.

AutoEval cut human supervision by >99% on WidowX arms. Heuristic detectors can be unreliable (~20% error in one study), and learned reset policies cover single-step tasks only.

### Cited Findings
- HIL-SERL reward classifier: ~200 positive + ~1,000 negative samples in ~5 min, >95% accuracy, sparse binary reward. — [HIL-SERL arXiv 2410.21845](https://arxiv.org/html/2410.21845)
- LeRobot reward classifier: `helper2424/resnet10` CNN on 128x128 crops, 2 classes, `success_threshold` e.g. 0.7, `terminate_on_success`, a fixed reset joint pose and `reset_time_s` 5 s; manual s/esc labelling works as a fallback. — [LeRobot HIL-SERL docs](https://huggingface.co/docs/lerobot/hilserl)
- AutoEval (Berkeley/NVIDIA, 2025-03): success classifier + reset policy + job queue for around-the-clock real evaluation on BridgeData WidowX scenes. >99% less human supervision, results closely matching hand evaluation. — [AutoEval arXiv 2503.24278v2](https://arxiv.org/html/2503.24278v2). A 2026-09 follow-up (HALTER) says AutoEval's automation covers single-step tasks only, because one learned reset policy cannot handle long-horizon end states. — [HALTER on alphaXiv 2609.19413](https://www.alphaxiv.org/abs/2609.19413)
- "So You Think You Can Scale Up Autonomous Robot Data Collection?" (Stanford, 2024-11): autonomous collection at scale is harder than prior work implies; a heuristic sock-folded detector had ~20% error, so they trained a learned classifier. — [arXiv 2411.01813](https://arxiv.org/pdf/2411.01813)
- RADAR (2026-03): VLM visual-question-answering success evaluation + a finite-state-machine reset; up to 90% success in sim, with real deployment via few-shot adaptation (real numbers not retrieved). — [RADAR arXiv 2603.11811](https://arxiv.org/abs/2603.11811)
- AutoDex (2026, secondary summary): real dexterous-grasp pipeline with lift-and-hold success labelling and active object reset. It reports 4.8x throughput over teleop and 76% grasp success when grasps are validated physically, vs 34% with sim-only validation. — [Emergent Mind AutoDex summary](https://api.emergentmind.com/topics/autodex) (secondary source)
- pi*0.6 RECAP labels success with human annotators and throughput = successful executions per hour. — [pi*0.6 arXiv 2511.14759](https://arxiv.org/pdf/2511.14759)

### Inferences
- Our robots already have most of an AutoEval-style loop. The classical picker is a reset policy: it can pick the piece and drop it at a random reachable pose, seen through its own vision ladder. Success can be checked by privileged-free signals: gripper position after close (the gap table gives the expected TIP gap for 30/40 mm pieces), a re-look with the wrist camera, and the kit detector. Combining these gives a cheap success detector for unattended real episodes and for real RL.
- A learned classifier (LeRobot ResNet-10 on wrist-camera crops) can be trained from the classical picker's logged outcomes, without human labelling, as long as its success checks are reliable. The 20% heuristic-error warning argues for cross-checking two signals (gripper gap + vision).
- In sim, use the simulator's true poses for success/reward and reset by teleporting objects to randomized poses; no learned detector is needed there.

### Gaps
- No source reports long unattended (multi-day) real collection with a VLM success detector on low-cost arms.
- RADAR's real-world numbers and AutoDex's primary paper were not verified.
