# Open sim-to-real pipelines for low-cost arms (Isaac Sim / Isaac Lab, with ManiSkill / MuJoCo / Genesis for comparison) — state as of 2026-10-26

Scope note: everything below was collected on 2026-10-26. "Real" means a result measured on a physical arm; "sim" means measured in simulation only. Where a page carries no date, it is marked "undated, read 2026-10-26". Some lower-weight items come only from search-result summaries (not a full page read); they are marked "(search summary)".

## 1. LeIsaac (LightwheelAI) and NVIDIA's own SO-101 Isaac Lab material: what they provide, versions, Windows, results

### Takeaway
LeIsaac is the most complete open Isaac Lab "playground" for the SO-101 (teleop with the leader arm, scripted state-machine datagen, a MimicGen env, a green-screen "DigitalTwin" env, a direct LeRobot recorder, GR00T inference), but it supports only Isaac Sim 4.5–5.1 / Isaac Lab 2.1.1–2.3.0 and publishes no real-robot success numbers. NVIDIA's "Train an SO-101 Robot From Sim-to-Real" course (Isaac Sim 5.1 / Isaac Lab 2.3 / LeRobot 0.4.3 / GR00T N1.6) is the only official end-to-end sim→real recipe; its sources give real numbers only through third parties: 40 % end-to-end (8/20) with the sim-trained checkpoint, ~60–70 % at the original workshop after camera/calibration tweaks, and 0/150–200 for one user until the physical layout was matched to the sim (then 8/10).

### Cited Findings
**LeIsaac (LightwheelAI)**
- LeIsaac provides teleoperation in Isaac Lab with the SO101Leader driving a simulated SO101Follower, data collection, HDF5→LeRobot conversion and policy training; it calls itself "the official imitation-learning (IL) simulation playground" in LeRobot's EnvHub (news 2025-11-26). Citation version 0.4.0 (2026); 751 stars, Apache-2.0 (read 2026-10-26). — [LeIsaac README](https://github.com/LightwheelAI/leisaac)
- Changelog (dates as written YY/MM/DD): 25/11/27 keyboard + gamepad teleop; 25/12/19 Marble tutorial, LeKiwi teleop; 26/01/12 LeRobot recorder (record straight to LeRobot Dataset format); 26/01/13 LeIsaac × Cosmos "video2action" tutorial; 26/01/16 GR00T N1.6 inference; 26/03/10 `datagen` module (programmatic trajectories / state machine); 26/04/14 remote teleoperation. — [LeIsaac README](https://github.com/LightwheelAI/leisaac)
- The README says it uses "simulation-collected data to fine-tune GR00T N1.5 and deploy it on real hardware", but gives no sim or real success rates or episode counts. — [LeIsaac README](https://github.com/LightwheelAI/leisaac)
- Compatibility table: Isaac Sim 4.5 / Python 3.10 / Isaac Lab v2.1.1 / CUDA 11.8 / PyTorch 2.5.1; Isaac Sim 5.0 / 3.11 / v2.2.1 / 12.8 / 2.7.0; Isaac Sim 5.1 / 3.11 / v2.3.0 / 12.8 / 2.7.0. For RTX 50-series it recommends Isaac Sim 5.0+ and Isaac Lab v2.2.1+. LeRobot is an optional extra, not pinned (needs `numpy==1.26.0` afterwards). OS not stated (commands are Linux-style). — [LeIsaac installation](https://lightwheelai.github.io/leisaac/docs/getting_started/installation)
- Tasks: `LeIsaac-SO101-PickOrange-v0` (3 oranges to a plate), `LeIsaac-SO101-LiftCube-v0` (red cube), `CleanToyTable` (single and bi-arm), `FoldCloth-BiArm`, `LeIsaac-LeKiwi-CleanupTrash-v0`; most have a `-Direct-v0` variant. — [LeIsaac available envs](https://lightwheelai.github.io/leisaac/resources/available_env)
- State-machine datagen: a scripted policy runs the task without teleop and records to HDF5, or directly to LeRobot format with `--use_lerobot_recorder` (default 30 fps); only `LeIsaac-SO101-PickOrange-v0` is named as working; `--num_envs` is "usually 1"; `--num_demos` counts only successful demos; grasp success depends strongly on orange spawn positions. New tasks subclass `StateMachineBase`. — [LeIsaac state machine](https://lightwheelai.github.io/leisaac/docs/features/state_machine)
- Feature index also lists a "MimicGen Env" (generate data from demonstrations), an "EnvHub Support" page and a "DigitalTwin Env" ("Make Sim2Real Simple"). — [LeIsaac features](https://lightwheelai.github.io/leisaac/docs/features)
- DigitalTwin Env replaces the simulated background with real background images while keeping the arm and objects from the sim ("inspired by the greenscreening functionality in SIMPLER and ManiSkill"); configured by subclassing `ManagerBasedRLDigitalTwinEnvCfg` (`overlay_mode`, background image path, foreground components); example `LiftCubeDigitalTwinEnvCfg`; no quantitative results. — [LeIsaac DigitalTwin](https://lightwheelai.github.io/leisaac/docs/features/digitaltwin_env)
- Windows: a community guide runs LeIsaac on Windows 11 with Isaac Sim 5.1 built from source + Isaac Lab `release/2.3.0` + LeIsaac pinned to commit `3bb45ed` (a later commit `58d56c7` "broke recording"); everything is run through Isaac Sim's `python.bat`, from Command Prompt not PowerShell; leader arm on `COM3`; needs Developer Mode/admin symlinks and removal of the "OpenCL, OpenGL, and Vulkan Compatibility Pack"; the HDF5→LeRobot script needed a manual edit because the LeRobot API changed. No policy/real results (undated, read 2026-10-26). — [Windows LeIsaac guide (HackMD)](https://hackmd.io/@asierarranz/rkg1tvT93gx)
- Seeed Studio's LeIsaac tutorial (last updated 2025-08-01): Ubuntu 22.04, RTX 3080, Isaac Sim 4.5 + Isaac Lab v2.1.0, GR00T N1.5 fine-tuned 10,000 steps, batch 2, `so100_dualcam` config; sim-only inference; the author reports "The final trained model failed to control the SOArm101 robotic arm to pick the orange" (with three data sets recorded). — [Seeed wiki](https://wiki.seeedstudio.com/simulate_soarm101_by_leisaac/)

**NVIDIA "Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac" (learning path)**
- Task: unstructured pick-and-place of centrifuge vials from a mat into a rack inside a lightbox; GR00T VLA with language + joint feedback + cameras → motor positions; notes the gripper camera is occluded after the grasp. — [Course overview](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/01-overview.html)
- Stack: Isaac Sim 5.1.0, Isaac Lab 2.3.0, LeRobot 0.4.3, GR00T N1.6; repo `isaac-sim/Sim-to-Real-SO-101-Workshop`; Docker only (`teleop-docker` for sim; `real-robot` container built for `blackwell` or `ada`, build "can take up to an hour"); requires "an NVIDIA GPU (Ada or Blackwell architecture)", e.g. RTX 4090 / RTX PRO 6000; OS not stated. — [Get the code](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/06-get-the-code.html)
- Isaac Lab tasks `Lerobot-So101-Teleop-Vials-To-Rack` and `...-DR`; a physical leader arm teleoperates the simulated follower (`lerobot_agent --task ...`); records straight to LeRobot format; recommends "at least 70 demonstrations"; sim logs `[GRASP]`, `[RELEASE]`, `[RACK]` events for scoring; first launch ~2 min. — [Strategy 1: DR + teleop](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/09-strategy1-dr-teleop.html)
- Sim evaluation (`...-Eval` task, GR00T server + `lerobot_eval`, `--action_horizon 16`): expected sim success "between 50-70%" for a policy trained on 75 demos; failure factors: vials rolling, dark lighting, too little data. — [Sim evaluation](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/11-sim-evaluation.html)
- Real evaluation runs the same checkpoint (`aravindhs-NV/grootn16-finetune_sreetz-so101_teleop_vials_rack_left/checkpoint-10000`) with two OpenCV cameras (wrist + front, 640×480 @ 30 fps); the page reports no success rate. — [Real evaluation](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/12-real-evaluation.html)
- Co-training: "a small amount of real demonstration data (5 episodes)" + "a much larger set of simulation demonstrations (70-100)"; co-trained model `aravindhs-NV/grootn16-finetune_sreetz-so101_teleop_vials_rack_left_sim_and_real/checkpoint-10000`; a real dataset `sreetz-nv/so101_teleop_vials_rack_left_real_50` exists; no mixing ratio and no success rates published. — [Strategy 2: co-training](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/13-strategy2-cotraining.html)
- Cosmos augmentation: 75 sim episodes + 7 or 70 Cosmos-augmented episodes (controls: depth weight 0.2, edge 1.0, segmentation 0.3, vis 0.1; "Cosmos Transfer 2.5" linked); models `..._augment_02` and `so100-orig-groot-vials-rack-left-cosmos-70`; no results published. — [Strategy 3: Cosmos](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/14-strategy3-cosmos.html)
- SAGE (sim-to-real actuation gap estimation) + GapONet: paired real/sim runs of the same joint trajectories (`pick_place`, `actuator_bandwidth`, `oscillation_low_freq`, `random_waypoints`), per-joint position/velocity/torque comparison; SO-101 hobby-servo backlash "adds up through the kinematic chain"; sim defaults `kp` 100, `kd` 2, 50 Hz control, 200 Hz physics; "8 hours of real trajectory data" collected for SO-101; no numeric errors; GapONet+GR00T integration is future work. — [Strategy 4: SAGE + GapONet](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/15-strategy4-sage.html)
- The conclusion gives no numbers; its message is that sim success does not guarantee real success and the four strategies must be combined. — [Conclusion](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/16-conclusion.html)
- Third-party real numbers for the course checkpoint (undated repo, read 2026-10-26): 20 sim + 20 real trials of `checkpoint-10000` (≈75 demos), after a camera-orientation fix: pick 95 % sim / 75 % real, placement|arrival 55.6 % / 53.3 %, end-to-end 50 % sim / 40 % real; remote inference round trip mean ≈39.7 ms (P99 ≈42 ms). — [nairb36/so101-sim2real](https://github.com/nairb36/so101-sim2real)
- NVIDIA forum, 2026-05-28 to 2026-06-17: user on Isaac Sim 5.1 saw 70–90 % in sim but 0 placements in "roughly 150-200 episodes" on the real arm; after matching the physical layout to the sim (no DR, 50 % light, rack 18 cm from left wall / 17 cm from back) got 80 % in 10 episodes. The tutorial author (sreetz, 2026-06-02) said the workshop ran 40 "identical" WowRobo setups, deviations came from calibration or a camera angle too high/low and one tweak usually gave "about 60-70%"; a RealSense D455 "would match the sim camera exactly"; suggested higher-contrast mat or co-training with 5–10 real demos. NVIDIA staff redirected sim2real questions to the HF/LeRobot community. — [NVIDIA forum thread](https://forums.developer.nvidia.com/t/physical-modelling-of-sim2real-so101-arm-project/371627)
- NVIDIA Isaac Capture/Isaac Teleop "LeRobot and SO-101" integration (last updated 2026-10-24, Isaac Lab 3.0 badge): teleop with XR controller or SO-101 leader in sim and real into LeRobot format, then GR00T N1.7; its own status list says "export sim demos to LeRobot format" and GR00T N1.7 training are not done yet; no numbers. — [Isaac Capture LeRobot integration](https://nvidia.github.io/IsaacCapture/main/integrations/lerobot/index.html)
- An earlier NVIDIA/HF blog on GR00T N1.5 for SO-101 announced an Isaac Lab SO-101 environment "to be released soon" (search summary). — [HF blog: GR00T N1.5 SO-101](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
- Isaac Lab Mimic is not SO-101-specific: the official augmented-imitation tutorial is Franka cube stacking (see §4). — [Isaac Lab augmented imitation](https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/augmented_imitation.html)

### Inferences
- Neither LeIsaac nor the NVIDIA course lists Isaac Sim 6.0 / Isaac Lab 3.0 support; our hekate (Isaac Sim 5.0, Windows 11) matches LeIsaac's 5.0 / Isaac Lab 2.2.1 row, our second PC (Isaac Sim 6.0 + Isaac Lab 3.0.0-beta2) matches neither — porting would be needed there.
- Both are SO-101-asset-centric and assume teleop with a leader arm; for the M3 the reusable parts are patterns (DR config values, DigitalTwin green-screen env, state-machine datagen + LeRobot recorder, event-based scoring), not the robot assets.
- The NVIDIA numbers show the main failure is the visual/layout gap, not the dynamics: real pick drops from 95 % to 75 % while transport and placement are close to sim once grasped.

### Gaps
- No published LeIsaac real-robot success rate was found; no official real numbers from NVIDIA's course (sim-only vs co-trained vs real-only) either.
- Official OS support for LeIsaac and the course containers is not stated (only a community Windows guide exists).
- Whether LeIsaac runs unchanged on Isaac Sim 6.0 / Isaac Lab 3.0 was not found.

## 2. Other pipelines: ManiSkill3 SO-100/SO-101/Koch, LeRobot gym envs, Genesis, RoboVerse, community, Yahboom/Dofbot

### Takeaway
The best-documented zero-shot sim→real results on low-cost arms come from ManiSkill3 visual RL: Koch v1.1 cube pick 91.6 % (22/24) with green screen, and Squint on SO-101 with only a wrist camera, 73/80 (91.3 %) across 8 tasks after 15 min training on one RTX 3090. For imitation learning, the strongest concrete numbers are a Genesis-pretrained ACT on SO-101 (50 scripted sim + 10 real episodes → 85 %, vs 25 % real-only) and a Yahboom DOFBOT pipeline (scripted IK expert in Isaac Sim 6.0.1 → LeRobot 0.5.1 Diffusion Policy) that only worked on the real arm after per-joint offset correction.

### Cited Findings
**ManiSkill3 (SAPIEN)**
- ManiSkill3 paper (v2, 2025-05-30, RSS 2025): Koch v1.1 ($300 arm) + phone camera, cube picking; real background green-screened into the sim (robot and cube kept from the sim via segmentation), camera pose roughly aligned with overlay images; DR of camera pose, lighting direction, robot pose, cube size/colour/friction; PPO + Nature CNN on RGB + joint positions (no privileged state), delta target joint position control same as the real robot; 15 M samples, 256 envs, ~1 h on an RTX 4090; real 22/24 = 91.6 % (3 training runs × 8 trials) with varied cube sizes/colours/poses; good sim–real correlation over 14 checkpoints × 3 seeds. — [ManiSkill3 paper v2](https://arxiv.org/html/2410.00425v2)
- LeRobot-Sim2Real tutorial (SO100GraspCube-v1): set sim camera pose in `env_config.json` (example `[0.7, 0.37, 0.28]` m from base), place the real camera at the same offsets, run `camera_alignment.py` (sim render blended over live frame, nudge camera, FOV with arrow keys), capture a background image with the robot removed for the green screen (cropped to 128×128), optional `robot_color` randomisation; `train_ppo_rgb.py` with 1024 envs uses ~8–10 GB VRAM; "around 25 to 40 million [steps] are enough, which take about an hour of training on a 4090"; real eval at `--control-freq=15` for SO100 with a ~2.5 cm cube, step-by-step confirmation mode first; no numeric real success rate given. — [zero_shot_rgb_sim2real.md](https://github.com/StoneT2000/lerobot-sim2real/blob/main/docs/zero_shot_rgb_sim2real.md)
- The LeRobot-Sim2Real README pins an old LeRobot commit, ships `system_id_so100.npy`, and now recommends Squint instead ("better performance and sim2real transfer", SO-101 examples, faster training). — [lerobot-sim2real](https://github.com/StoneT2000/lerobot-sim2real)
- Squint (Almuzairee & Christensen; v1 2026-02-24, v2 2026-09-04, RA-L 2026): visual SAC; 8 SO-101 tasks in ManiSkill3 (reach/lift/place/stack × cube/can) with objects in a 0.2 × 0.2 m box; wrist camera is the only image input; renders 128×128 (16×16 did not transfer); DR: wrist camera pose/rotation/FOV jitter, lighting + colour jitter, object size, friction, gripper closing speed, σ = 5° joint noise; 1024 envs; 10 Hz joint-delta control in sim, deployed at 30 Hz with actions scaled by 0.15; 15 min on one RTX 3090 (most tasks < 6 min); real results 10 trials each: Lift Cube 10/10, Place Cube 10/10, Stack Cube 8/10, Lift Can 9/10, Stack Can 6/10, total 73/80 (91.3 %); sim 96.1 ± 1.6 %; without colour jitter real drops to 58/80 (72.5 %). — [Squint HTML](https://arxiv.org/html/2602.21203); [Squint abstract](https://arxiv.org/abs/2602.21203)
- SO-101 port of the ManiSkill tutorial (`SO101GraspCube-v1`, mani_skill 3.0.0b21, LeRobot 0.3.4 / Sep 2025): same alignment/greenscreen steps, no real numbers ("still in a very early stage"). — [sangnt1552314/lerobot_so101-sim2real](https://github.com/sangnt1552314/lerobot_so101-sim2real)
- Wang (2025-11-13): ManiSkill3 Grasp Cube on SO-101 with a third-person RealSense D435i, PPO CNN; zero-shot real success 0.28 without vs 0.44 with scene randomisation (100 camera poses on a sphere, light 0–0.5, 17 table textures, object colour, table height); trial count not reported; sim scenes were not aligned to the real scene. — [arXiv 2511.09932](https://arxiv.org/html/2511.09932)

**LeRobot sim envs (MuJoCo / Isaac via EnvHub)**
- `lerobot/so101-tabletop-isaac`: EnvHub env built natively in Isaac Sim 6.2 (PhysX + RTX), SO-101 + four painted 3 cm wooden cubes + green box, "Put all the cubes in the green box"; front + wrist 480×640 RGB, 6 absolute joint targets (degrees, gripper 0–100, same units as `so101_follower`), 30 fps; identical layout/obs/actions to `lerobot/so101-tabletop-mujoco`; only `n_envs=1`, ~80 ms per step on an RTX 3080 Laptop; Python 3.12; runs via `lerobot-eval --env.type=envhub` or `lerobot-env-server`/`lerobot-rollout`; no sim-to-real claims or numbers (collection updated ~mid-Oct 2026). — [lerobot/so101-tabletop-isaac](https://huggingface.co/lerobot/so101-tabletop-isaac)
- gym-hil: MuJoCo Franka Panda pick-cube envs (base / gamepad / keyboard) for HIL-SERL in LeRobot; no SO-101, no sim-to-real. — [gym-hil](https://github.com/huggingface/gym-hil)
- Isaac Lab-Arena environments are runnable through `lerobot-eval` via EnvHub; Arena is labelled alpha (v0.3) in a 2026 comparison (search summary). — [nvkartik/isaaclab-arena-envs-lab3](https://huggingface.co/nvkartik/isaaclab-arena-envs-lab3); [Bifrost comparison](https://www.bifrost.ai/blog/robot-policy-evaluation-platforms-compared/)

**MuJoCo real2sim (closest to our Windows + LeRobot 0.6.1 stack)**
- `so101-real2sim` (2026, undated): SO-101 (STS3215), LeRobot 0.6.1 on Windows 11, RTX 5070 Ti laptop; 20 real episodes (30 Hz, wrist + side cams) → 12 reproducible source demos → measured MuJoCo twin (object positions from FK when the jaws stall; actuator ID by 200 Hz excitation: identified stiffness 16–62 vs 998 in the stock model, excitation error 1.8–3.7° → 0.3–0.5°) → MimicGen-style generation (277 attempts, 200 kept) → ACT. Twin replay success 9/16 with stock actuators vs 12/16 with identified ones. Sim-evaluated ACT: 12 real demos 35/31/60 % (inside/beyond/lifted) vs +200 generated 36/34/77 %. Real open-loop replays of generated motions: 4/4 at an interpolation mark, 0/4 at the reach-limit mark; no closed-loop real policy result. — [so101-real2sim](https://github.com/biswajeetpandaisreal-arch/so101-real2sim)

**Genesis**
- Genesis-pretrained ACT for SO-101 (AMD AI DevMaster hackathon; undated card): stage 1 `so101_cube_dr` = 50 scripted sim episodes with DR (cube colour, table, lighting, friction, mass), 60k steps; stage 2 co-train = 10 real teleop episodes repeated 3× + 50 sim (66,539 frames, 33.1 % real), 20k steps, 2 h 22 min, 3.9 GB VRAM; world + wrist cams 640×480 @ 30 fps; real 85 % (17/20) on 20 fixed placements vs 25 % (5/20) for the same 10 real episodes without sim pretraining (Fisher p = 0.00033). Sim-only checkpoints reached the cube but did not complete grasps. — [omiya239532/so101_act_cotrain](https://huggingface.co/omiya239532/so101_act_cotrain)
- A public dataset mixes 534 Genesis sim episodes with 46 real SO-101 teleop episodes (search summary, no results). — [Claru dataset page](https://claru.ai/datasets/shane1027-so101-nerds-combined-v2)

**RoboVerse**
- RoboVerse (2025) is a unified platform/dataset/benchmark that reports improved sim-to-real transfer across IL, RL and world models (search summary); nothing tying it to SO-100/SO-101 hardware was found. — [RoboVerse (BIGAI)](https://eng.bigai.ai/?p=7441)

**Community Isaac Lab SO-101 pipelines**
- `HYEONHEE5739/so101-isaaclab`: Isaac Sim 5.1, Isaac Lab 2.3, LeRobot 0.4.1; leader-arm teleop into `SO101-PickPlace-Mimic-v1`, 20 source demos, Mimic generation 20/22 (90.9 %), LeRobot dataset with side + wrist cams, ACT sim eval 34/50 (68 %; 20 % grasp failures); no real results. — [so101-isaaclab](https://github.com/HYEONHEE5739/so101-isaaclab)
- `dreamerarun/cube-to-tray-so101-pipeline`: scripted state machine in Isaac Sim 5.1 generating SO-101 cube-pick demos, converter to LeRobot v3 that skips failed and < 10-frame episodes (search summary). — [cube-to-tray-so101-pipeline](https://github.com/dreamerarun/cube-to-tray-so101-pipeline)
- A Hackaday log: 1 recorded SO-101 cube-lift demo → 10 MimicGen demos → LeRobot v3.0 dataset (search summary). — [Hackaday ChefMate log](https://hackaday.io/project/204187-chefmate-kitchen-robot-using-vla-act-diffusion/log/243819-mimicgen-data-augmentation-pipeline-for-robotic-manipulation)
- `kabilankb/...`: SO-101 leader teleop for a bimanual YAM sim on Isaac Lab 3; its MimicGen annotate/generate steps "ported and reviewed, but not yet run end to end" (search summary). — [kabilankb repo](https://github.com/kabilankb/Isaac-Lab-3-and-add-SO-101-leader-teleoperation)
- Nebius/NVIDIA Berlin hackathon "sim-to-table" (2026-10-18): Isaac Sim 6.1 + Isaac Lab 3.0 scripted expert (143 sim episodes) + Cosmos Transfer 2.5 (12) + 30 real teleop, Cosmos Reason 2 as episode auditor; GR00T fine-tune 1000 steps, batch 16; sim-only GR00T gave erratic joint targets on the real arm; ACT 1/2 and 0/3; GR00T's loader rejected LeRobot v3.0 datasets. — [ETohidi/physical-ai-hackathon-sim-to-table](https://github.com/ETohidi/physical-ai-hackathon-sim-to-table)

**Yahboom DOFBOT (closest robot family)**
- `natsutan/Dofbot_diffusion_policy` (2026, undated): Isaac Sim 6.0.1, LeRobot v0.5.1 (Dataset v3), Ubuntu 24.04 + ROS 2 Jazzy, RTX 4060 Ti, real Raspberry Pi 4 + Yahboom DOFBOT; scripted IK expert (ROS 2 node `collect_imitation_episodes`, not teleop) for box pick-and-lift; box spawned X 0.17–0.22 m, Y −0.08…−0.02 m, ±2° home jitter; success = lifted ≥ 5 cm with box–gripper ≤ 6 cm for 0.5 s; 10 Hz record/command; state = 11 joints (rad) + box XY, action = 11 absolute joint targets; Diffusion Policy 50,000 steps, batch 32 (n_obs 2, horizon 16, n_action 8), 10 diffusion steps at inference. Results: sim pick success; real pick FAILED with raw joint feedback; replaying a sim-successful command trajectory on the real arm succeeded; with offsets added to `observation.state` (arm2 +0.0527 rad ≈ +3.02°, arm3 +0.0284 rad ≈ +1.63°) and gripper close angle 145° the real pick succeeded. No success-rate statistics; the real servo driver is not in the repo. — [Dofbot_diffusion_policy](https://github.com/natsutan/Dofbot_diffusion_policy)

### Inferences
- For a wrist-camera-only arm, Squint is the strongest evidence that zero-shot RGB transfer works on a cheap arm when the sim is a "digital twin" and colour/lighting are jittered; it is RL (not LeRobot IL) and ManiSkill (not Isaac), so it is a recipe to copy, not code to drop in.
- The DOFBOT and Genesis results both say the same thing for IL from a scripted expert: sim-only fails at the grasp; a few degrees of joint offset and a few real episodes fix most of it.

### Gaps
- No myCobot or xArm Lite 6 sim→real pipeline with Isaac Sim/LeRobot was found.
- No official Genesis SO-101 sim2real task or RoboVerse SO-100/101 real result was found.
- No Yahboom M3 Pro (or ROSMASTER arm) sim2real work was found.

## 3. How they calibrate the sim to the real robot (camera, joints, gripper, servos, latency, visuals)

### Takeaway
Working pipelines calibrate in this order: (1) motor zero/limits carefully, (2) camera pose by overlaying the sim render on the live frame (ManiSkill tool) or by physically moving the real camera to the sim pose, (3) background by green screen (ManiSkill, LeIsaac DigitalTwin) or HDRI/lighting DR (NVIDIA), (4) actuators by system ID only when replays fail (SAGE, so101-real2sim). Camera placement is the single most cited cause of failure; servo PD-gain mismatch and latency are the most cited dynamic causes.

### Cited Findings
- Camera, ManiSkill: put the real camera at the sim's offsets from the base, then blend sim over real and nudge pose/FOV until aligned ("alignment that looks worse than the ones here will struggle a lot more"); green-screen background captured with the robot removed; motors calibrated with zero set "as accurately as possible". — [zero_shot_rgb_sim2real.md](https://github.com/StoneT2000/lerobot-sim2real/blob/main/docs/zero_shot_rgb_sim2real.md)
- Camera, ManiSkill3 paper: green-screen the real background, keep robot/cube from sim via segmentation, rough camera alignment, plus camera-pose DR. — [ManiSkill3 paper v2](https://arxiv.org/html/2410.00425v2)
- Camera, NVIDIA: viewport set to 4:3 to match the real cameras; operators watch only camera viewports while teleoperating; external camera pose DR ±0.02 m (x, y), ±0.01 m (z), ±0.05 rad roll/pitch/yaw. — [Strategy 1](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/09-strategy1-dr-teleop.html)
- Camera, NVIDIA in practice: real failures traced to "calibration or camera angle that was too high or too low"; a RealSense D455 "would match the sim camera exactly"; wider FOV reduces sensitivity to framing. — [NVIDIA forum](https://forums.developer.nvidia.com/t/physical-modelling-of-sim2real-so101-arm-project/371627); early real runs "badly degraded by a camera-orientation mismatch". — [nairb36/so101-sim2real](https://github.com/nairb36/so101-sim2real)
- Wrist camera, Squint: small random position/rotation/FOV perturbations of the wrist camera; 128×128 render. — [Squint HTML](https://arxiv.org/html/2602.21203)
- Background, LeIsaac: DigitalTwin env overlays real background images on the sim render, keeping arm/objects. — [LeIsaac DigitalTwin](https://lightwheelai.github.io/leisaac/docs/features/digitaltwin_env)
- Lighting/visual DR, NVIDIA: sky-light exposure (−4.0, 3.0), colour temperature (2500, 9500) K, random HDRI `.exr` per reset; mat yaw (−0.1, 0.1); physics DR listed only as examples (mass, friction, damping, delays, noise) with no ranges. — [Strategy 1](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/09-strategy1-dr-teleop.html)
- Visual DR value: Squint real success 73/80 with colour jitter vs 58/80 without. — [Squint HTML](https://arxiv.org/html/2602.21203); Isaac Lab Mimic + Cosmos lifts sim robustness under light-intensity change from 11.1 % (Mimic 1k) / 20 % (Mimic 2k) to 62.2 % (Cosmos-Mimic 2k), but table texture (20 %) and arm texture (4.4 %) stay weak. — [Isaac Lab augmented imitation](https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/augmented_imitation.html)
- Robot colour: ManiSkill exposes `robot_color` ("random" or e.g. black `[0.1, 0.1, 0.1]`). — [zero_shot_rgb_sim2real.md](https://github.com/StoneT2000/lerobot-sim2real/blob/main/docs/zero_shot_rgb_sim2real.md); NVIDIA author: the black robot may blend into the mat; sim DR uses black robots. — [NVIDIA forum](https://forums.developer.nvidia.com/t/physical-modelling-of-sim2real-so101-arm-project/371627); Genesis ACT: success correlates 0.956 with object–background RGB contrast and collapses when cube colour matches the table. — [so101_act_cotrain](https://huggingface.co/omiya239532/so101_act_cotrain)
- Joint offsets: DOFBOT needed +3.02° (arm2) and +1.63° (arm3) added to the policy's state input, individual to that unit. — [Dofbot_diffusion_policy](https://github.com/natsutan/Dofbot_diffusion_policy); so101-real2sim had to widen the wrist limit (real rest −102° vs stock ±95°). — [so101-real2sim](https://github.com/biswajeetpandaisreal-arch/so101-real2sim)
- Gripper: DOFBOT gripper driver open 60°, close 145°; Squint randomises gripper closing speed and object friction. — [Dofbot_diffusion_policy](https://github.com/natsutan/Dofbot_diffusion_policy); [Squint HTML](https://arxiv.org/html/2602.21203)
- Servo PD / speed: so101-real2sim identified stiffness 16–62 vs 998 stock, a "soft position loop and a 10°/frame command clamp" with lag up to 40° on fast moves; static error ≤ 0.4°. — [so101-real2sim](https://github.com/biswajeetpandaisreal-arch/so101-real2sim); Genesis ACT: sim KP = 120 vs servo P = 16 named as a cause of sim-only grasp failure. — [so101_act_cotrain](https://huggingface.co/omiya239532/so101_act_cotrain); NVIDIA SAGE default sim gains kp 100 / kd 2 at 50 Hz. — [Strategy 4](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/15-strategy4-sage.html)
- Control rate: Squint 10 Hz sim → 30 Hz real with 0.15 action scale; ManiSkill SO100 eval at 15 Hz; DOFBOT 10 Hz; NVIDIA/LeRobot 30 fps. — [Squint HTML](https://arxiv.org/html/2602.21203); [zero_shot_rgb_sim2real.md](https://github.com/StoneT2000/lerobot-sim2real/blob/main/docs/zero_shot_rgb_sim2real.md); [Dofbot_diffusion_policy](https://github.com/natsutan/Dofbot_diffusion_policy); [so101-tabletop-isaac](https://huggingface.co/lerobot/so101-tabletop-isaac)
- Latency: Genesis ACT names a 311 ms chunk-replan latency "that simulation hides"; nairb36 measured ~40 ms remote inference round trip with no evidence it caused failures. — [so101_act_cotrain](https://huggingface.co/omiya239532/so101_act_cotrain); [nairb36/so101-sim2real](https://github.com/nairb36/so101-sim2real)
- Proprioception noise: Squint adds σ = 5° Gaussian noise to joint positions. — [Squint HTML](https://arxiv.org/html/2602.21203)
- Actuator shaping as an alternative to modelling: wrap each joint in a 2-DoF feedforward–feedback controller so it tracks an idealised second-order reference model used in sim (2026-07-02); not tested on hobby servos. — [arXiv 2607.02205](https://arxiv.org/abs/2607.02205)
- Rendering: none of the found SO-101 pipelines report using RTX path tracing for data; the Isaac Lab Mimic page suggests the "RTX Minimal renderer for higher throughput" for camera-heavy jobs. — [Isaac Lab augmented imitation](https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/augmented_imitation.html)

### Inferences
- For our arm the cheapest high-value calibrations are: per-joint offset (a few degrees can flip grasps), gripper close angle/tip gap, wrist-camera pose/FOV by overlay, and matching the sim's position loop to the servo's (low stiffness + speed clamp) rather than a stiff PD.
- Since the M3 has no joint feedback, "joint state" in both sim and real should be the commanded joints (what LeRobot will record on the robot), possibly with Squint-style noise; feeding the sim's measured joints would create a train/deploy mismatch.

### Gaps
- No pipeline reported wrist-camera intrinsics matching procedures beyond FOV tuning; no RGB-D (depth-input) policy pipeline for low-cost arms was found — all real results above are RGB.
- No numeric DR ranges for physics (friction, mass, PD) were published by NVIDIA; Squint's ranges are not quantified in the text read.

## 4. Data generation: scripted/privileged expert vs teleop-in-sim vs Isaac Lab Mimic; throughput; parallel envs with cameras

### Takeaway
All three are used: teleop-in-sim with a leader arm (NVIDIA course, LeIsaac, ~70–100 demos), scripted/privileged state machines (LeIsaac datagen, DOFBOT IK expert, Genesis 50 episodes, sim-to-table 143 episodes, cube-to-tray), and Mimic-style augmentation from ~10–20 seeds (90.9 % generation success on SO-101; 1000 demos recommended for Franka stacking). Published throughput numbers for camera-rendered datagen are scarce; the camera-in-the-loop Isaac paths in the LeRobot ecosystem run 1 env per process.

### Cited Findings
- Teleop in sim: NVIDIA recommends ≥ 70 demos; its checkpoints used ~75. — [Strategy 1](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/09-strategy1-dr-teleop.html); [Sim evaluation](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/11-sim-evaluation.html)
- Scripted state machine: LeIsaac `generate.py --task LeIsaac-SO101-PickOrange-v0 --record --num_demos 50`, `--num_envs` usually 1, only successful demos counted, LeRobot recorder at 30 fps. — [LeIsaac state machine](https://lightwheelai.github.io/leisaac/docs/features/state_machine)
- Scripted IK expert on privileged pose: DOFBOT in Isaac Sim 6.0.1 with randomised box pose and an automatic success test. — [Dofbot_diffusion_policy](https://github.com/natsutan/Dofbot_diffusion_policy); Genesis: 50 scripted DR episodes. — [so101_act_cotrain](https://huggingface.co/omiya239532/so101_act_cotrain); Isaac Sim 6.1 + Isaac Lab 3.0 scripted expert, 143 episodes, one RTX PRO 6000. — [sim-to-table](https://github.com/ETohidi/physical-ai-hackathon-sim-to-table)
- Mimic: SO-101 `SO101-PickPlace-Mimic-v1` 20 seeds → 20/22 generated (90.9 %). — [so101-isaaclab](https://github.com/HYEONHEE5739/so101-isaaclab); MuJoCo MimicGen-style: 277 attempts → 200 kept; 90 % success for shifts < 6 cm, 79 % at 6–12 cm, 58 % > 12 cm; shifting before the final descent 71 % vs 38 % during it. — [so101-real2sim](https://github.com/biswajeetpandaisreal-arch/so101-real2sim)
- Isaac Lab Mimic (Franka stack): `--device cpu --num_envs 10 --generation_num_trials 1000`; "1000 demonstrations have been shown to provide good training results"; 10 envs "can be executed on a moderate laptop CPU"; BC results 62 % (1k) / 96.6 % (2k) in vanilla sim; no throughput numbers. — [Isaac Lab augmented imitation](https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/augmented_imitation.html)
- Throughput proxies: EnvHub Isaac SO-101 env ~80 ms/step at 30 fps with two 480×640 cameras on an RTX 3080 Laptop, `n_envs=1`, ~20 s start-up + shader compile on first run. — [so101-tabletop-isaac](https://huggingface.co/lerobot/so101-tabletop-isaac); ManiSkill RGB PPO with 1024 envs at 128×128 in ~8–10 GB. — [zero_shot_rgb_sim2real.md](https://github.com/StoneT2000/lerobot-sim2real/blob/main/docs/zero_shot_rgb_sim2real.md); Isaac Lab: the USD camera hits OOM above 48 parallel cameras, the Tiled camera scales to thousands of envs, RGB vs depth costs the same with Tiled rendering (benchmarked on one RTX PRO 6000, exact FPS only in a figure). — [Isaac Lab paper](https://arxiv.org/html/2511.04831)
- Isaac Lab 3.0.0-beta2.patch1 adds Isaac Sim 6.0.1 support; Tiled Rendering API renders all clones in one batched pass; Newton Warp renderer supports only rgb and depth (search summary). — [Isaac Lab 3.0 release notes](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/refs/release_notes.html); [Isaac Lab 3.0-beta2 camera docs](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/overview/core-concepts/sensors/camera.html)
- Dataset writing can dominate: so101-real2sim got ~100× faster writing by using collision meshes, no shadows and streaming video encoding instead of PNG frames. — [so101-real2sim](https://github.com/biswajeetpandaisreal-arch/so101-real2sim)
- Cosmos as a visual multiplier: NVIDIA adds 7 or 70 Cosmos-restyled episodes to 75 sim ones; the Berlin hackathon added 12. — [Strategy 3](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/14-strategy3-cosmos.html); [sim-to-table](https://github.com/ETohidi/physical-ai-hackathon-sim-to-table)

### Inferences
- Episodes/hour estimate (not measured by any source): at ~80 ms per rendered 30 Hz step with two cameras on a laptop GPU, a ~20 s episode is ~600 steps ≈ 50 s, i.e. ~70 episodes/hour per single-env process; recording at 10 Hz (render only every control step) and an RTX 5090 should give several hundred per hour per process. Several independent processes, or Isaac Lab tiled cameras with N envs, multiply this; VRAM is the limit, not compute, for 2 × 480×640 cameras.
- For us, the privileged scripted expert we already have maps 1:1 onto the LeIsaac/DOFBOT state-machine pattern; Mimic adds little when a full planner is available, except for cheap diversity around real seed demos.

### Gaps
- No primary source gives measured episodes/hour for camera-rendered SO-101 datagen in Isaac Lab, nor parallel-env counts with two cameras on 20–32 GB GPUs.

## 5. Reported results: zero-shot vs fine-tuned/co-trained, for small-object pick-and-place

### Takeaway
Real-robot numbers that exist: zero-shot RL 91.3 % (SO-101, Squint, 80 trials) and 91.6 % (Koch, 24 trials); zero-shot IL/VLA from teleop-in-sim 40 % end-to-end / 75 % pick (SO-101 GR00T N1.6, 20 trials), and 0 % → 80 % (10 trials) when one user matched the layout; co-training 85 % (17/20) with 50 sim + 10 real vs 25 % real-only (Genesis ACT). Everything else (LeIsaac, Mimic SO-101, NVIDIA's co-trained and Cosmos models) has sim numbers or none.

### Cited Findings
| Pipeline | Robot | Sim / real data | Real result | Source |
|---|---|---|---|---|
| ManiSkill3 visual PPO, green screen | Koch v1.1 | 15 M sim samples, 0 real | 22/24 = 91.6 % cube pick | [ManiSkill3 paper](https://arxiv.org/html/2410.00425v2) |
| Squint visual SAC, wrist cam only | SO-101 | RL in sim, 0 real | 73/80 = 91.3 % (lift cube 10/10, place cube 10/10, stack cube 8/10) | [Squint](https://arxiv.org/html/2602.21203) |
| Squint without colour jitter | SO-101 | same | 58/80 = 72.5 % | [Squint](https://arxiv.org/html/2602.21203) |
| ManiSkill3 PPO, unaligned scene ± randomisation | SO-101 | sim only | 0.28 → 0.44 (trials not given) | [arXiv 2511.09932](https://arxiv.org/html/2511.09932) |
| NVIDIA course GR00T N1.6, ~75 teleop-in-sim demos | SO-101 | sim only | 8/20 end-to-end (40 %), pick 75 %; sim 50 % | [nairb36](https://github.com/nairb36/so101-sim2real) |
| Same course checkpoint, user setup | SO-101 | sim only | 0 in ~150–200 episodes; 8/10 after matching layout | [NVIDIA forum](https://forums.developer.nvidia.com/t/physical-modelling-of-sim2real-so101-arm-project/371627) |
| Same, workshop (40 setups) | SO-101 (WowRobo) | sim only (as reported) | "about 60-70%" after one tweak | [NVIDIA forum](https://forums.developer.nvidia.com/t/physical-modelling-of-sim2real-so101-arm-project/371627) |
| Genesis ACT co-train | SO-101 | 50 scripted sim + 10 real (×3) | 17/20 = 85 % vs 5/20 real-only | [so101_act_cotrain](https://huggingface.co/omiya239532/so101_act_cotrain) |
| Isaac Sim 6.0.1 scripted IK → Diffusion Policy | Yahboom DOFBOT | sim only | fail raw; success after joint-offset + gripper fix (no rate) | [Dofbot_diffusion_policy](https://github.com/natsutan/Dofbot_diffusion_policy) |
| Isaac Sim 6.1 scripted + Cosmos → GR00T / ACT | SO-101 | 143 sim + 12 Cosmos + 30 real | GR00T sim-only erratic; ACT 1/2, 0/3 | [sim-to-table](https://github.com/ETohidi/physical-ai-hackathon-sim-to-table) |
| LeIsaac GR00T N1.5 (Seeed tutorial) | SO-101 | 3 teleop-in-sim sets | sim: failed to pick (no real test) | [Seeed wiki](https://wiki.seeedstudio.com/simulate_soarm101_by_leisaac/) |
| Isaac Lab Mimic SO-101 → ACT | SO-101 | 20 seeds → 20 generated | sim 34/50 = 68 %; no real | [so101-isaaclab](https://github.com/HYEONHEE5739/so101-isaaclab) |
| MuJoCo twin + MimicGen → ACT | SO-101 | 12 real + 200 generated | sim only (lifted 60 → 77 %); real open-loop replay 4/4 near, 0/4 at reach limit | [so101-real2sim](https://github.com/biswajeetpandaisreal-arch/so101-real2sim) |

- NVIDIA's own co-training mix is 5 real + 70–100 sim, with no published outcome. — [Strategy 2](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/13-strategy2-cotraining.html)

### Inferences
- With scripted sim data + IL, "a few real episodes" (5–10, oversampled to ~1/3 of frames) is the recurring recipe; the only controlled comparison (Genesis) shows a 3.4× gain over the same real data alone.
- Zero-shot IL from sim on these arms is fragile (0–40 % without careful layout/camera matching); zero-shot works best when the policy is trained with heavy visual randomisation on a digitally twinned camera view (Squint, ManiSkill3).

### Gaps
- No controlled sim-only vs co-trained vs real-only comparison with real trials exists for the Isaac Lab SO-101 pipelines (NVIDIA course, LeIsaac).
- Trial counts are small everywhere (10–24 per condition); no pipeline reports precision of placement in mm.

## 6. Practical pitfalls reported

### Takeaway
The recurring pitfalls are camera framing/pose mismatch, low object–background contrast, soft/lagging hobby servos modelled as stiff PD, per-unit joint offsets, too little gripper friction for round objects, lighting sensitivity, latency hidden by sim, and tooling breakage (LeRobot API/version churn, dataset-format mismatches, Windows build quirks).

### Cited Findings
- Camera angle too high/low and calibration were the main deviations at the 40-setup NVIDIA workshop; one user got 0 real placements until the layout matched the sim. — [NVIDIA forum](https://forums.developer.nvidia.com/t/physical-modelling-of-sim2real-so101-arm-project/371627)
- Pick is where sim→real drops (95 → 75 %); the real policy shows a strong spatial prior (front-right, rack slot 1). — [nairb36](https://github.com/nairb36/so101-sim2real)
- Gripper friction: cans tipped and the gripper "had insufficient friction"; 16×16 images trained in sim but did not transfer; small lighting changes degraded the policy. — [Squint HTML](https://arxiv.org/html/2602.21203)
- Vials rolling after being disturbed and dark lighting reduce success in sim. — [Sim evaluation](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/11-sim-evaluation.html)
- Servo dynamics: stiffness 16–62 vs 998 stock, 10°/frame clamp, lag up to 40°; kinematic twin accurate to ~1 cm still failed replays until actuators were identified. — [so101-real2sim](https://github.com/biswajeetpandaisreal-arch/so101-real2sim)
- Sim-only checkpoints reach the cube but don't complete the grasp: KP 120 vs servo P 16, unmodelled wrist-camera mass, 311 ms replan latency; unit conversion (degrees, gripper 0–100 vs radians) needed. — [so101_act_cotrain](https://huggingface.co/omiya239532/so101_act_cotrain)
- Hobby-servo backlash accumulates through the chain. — [Strategy 4](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/15-strategy4-sage.html)
- Per-unit joint offsets (≈ 3° and 1.6°) and gripper close angle decided real success on the DOFBOT. — [Dofbot_diffusion_policy](https://github.com/natsutan/Dofbot_diffusion_policy)
- RL can learn strange or forceful behaviour; ManiSkill recommends step-by-step confirmation first and watching for pressing into the table; over-training can make worse real policies; test several high-scoring checkpoints. — [zero_shot_rgb_sim2real.md](https://github.com/StoneT2000/lerobot-sim2real/blob/main/docs/zero_shot_rgb_sim2real.md)
- Tooling: LeRobot recorder keys `n`/`r`/`q` captured system-wide on Windows (cut demos); grasp detector assumed an open gripper at start. — [so101-real2sim](https://github.com/biswajeetpandaisreal-arch/so101-real2sim); GR00T loader rejected LeRobot v3.0 datasets; a starter targeted a non-existent Isaac Lab tag. — [sim-to-table](https://github.com/ETohidi/physical-ai-hackathon-sim-to-table); LeIsaac later commit broke recording, conversion script out of date with LeRobot API. — [Windows LeIsaac guide](https://hackmd.io/@asierarranz/rkg1tvT93gx); direct LeRobot recorder removed because its import vanished. — [so101-isaaclab](https://github.com/HYEONHEE5739/so101-isaaclab); LeRobot-Sim2Real pinned to an old LeRobot commit. — [lerobot-sim2real](https://github.com/StoneT2000/lerobot-sim2real)
- Isaac Lab on Windows 11: an `isaaclab.bat --install` "invalid directory name" failure with no confirmed fix (search summary). — [NVIDIA forum: Isaac Lab directory install issues](https://forums.developer.nvidia.com/t/isaac-lab-directory-install-issues/302818)

### Inferences
- Our known M3 issues (no joint feedback, soft bus servos, tip gap table, wrist cam close-look occlusion) are exactly the classes that broke the cited pipelines; the camera and the grasp moment deserve most of the calibration time.
- Paths with spaces (our "RoboARM 2026", "Sim Arm 2026") are a plausible trigger for Windows Isaac Lab install errors — unverified.

### Gaps
- No source quantified servo deadband effects on cheap arms; no source reported penetration/contact-parameter tuning for 2–4 cm cubes in PhysX.

## 7. Which pipeline is closest to our setup, and a minimal 3-day path

### Takeaway
No public pipeline targets the M3 Pro. The closest pieces are: the Yahboom DOFBOT pipeline (Isaac Sim 6.0.1, scripted IK expert, LeRobot, bus servos — closest robot + expert), the Genesis SO-101 co-training recipe (closest data recipe with a real ablation), so101-real2sim (Windows 11 + LeRobot 0.6.1 + actuator ID, closest tooling), Squint (wrist-camera-only zero-shot DR recipe) and NVIDIA/LeIsaac (DR values, green-screen DigitalTwin env, LeRobot recorder). The fastest path is our own Isaac Sim code + these recipes, not adopting LeIsaac wholesale.

### Cited Findings
- DOFBOT: scripted IK expert in Isaac Sim 6.0.1 → LeRobot 0.5.1 v3 dataset at 10 Hz → Diffusion Policy 50k steps; real success needed observation-offset correction. — [Dofbot_diffusion_policy](https://github.com/natsutan/Dofbot_diffusion_policy)
- Genesis: 50 scripted DR sim episodes pretrain (60k steps) → co-train with 10 real ×3 (33 % of frames), 20k steps, ~2.4 h on a consumer GPU, 3.9 GB → 85 % real vs 25 %. — [so101_act_cotrain](https://huggingface.co/omiya239532/so101_act_cotrain)
- so101-real2sim: LeRobot 0.6.1 on Windows 11; actuator ID by excitation; generation from real seeds; streaming video encoding. — [so101-real2sim](https://github.com/biswajeetpandaisreal-arch/so101-real2sim)
- Squint: wrist cam, 128×128, colour jitter essential, σ = 5° joint noise, 10 Hz sim. — [Squint HTML](https://arxiv.org/html/2602.21203)
- NVIDIA DR values and 5 real + 70–100 sim co-training structure. — [Strategy 1](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/09-strategy1-dr-teleop.html); [Strategy 2](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/13-strategy2-cotraining.html)
- ManiSkill camera-overlay alignment and green screen procedure. — [zero_shot_rgb_sim2real.md](https://github.com/StoneT2000/lerobot-sim2real/blob/main/docs/zero_shot_rgb_sim2real.md); LeIsaac DigitalTwin env as the Isaac Lab equivalent. — [LeIsaac DigitalTwin](https://lightwheelai.github.io/leisaac/docs/features/digitaltwin_env)
- LeIsaac version support stops at Isaac Sim 5.1 / Isaac Lab 2.3.0 (5.0 / 2.2.1 row matches hekate). — [LeIsaac installation](https://lightwheelai.github.io/leisaac/docs/getting_started/installation)

### Inferences
Minimal 3-day path (our own code; borrowed recipes in brackets):

- Day 1 — calibrate the twin (half a day each):
  - Joint offsets: command a set of poses on the real M3, photograph/measure (or use the touch probe) and fit per-joint offsets; apply them in the real driver or in `observation.state`, as DOFBOT had to [DOFBOT].
  - Servo response: send step/sine commands to each servo with a camera or the arm's own wrist view filming, measure lag and max speed (no feedback, so judge by video/AprilTag); in Isaac set low stiffness + a per-step joint-speed clamp so a sim replay of a real M3 episode's commands lands within ~1 cm of where the real gripper went [so101-real2sim; Genesis KP lesson].
  - Gripper: set close angle / tip gap from our measured gripper tip-gap table for the real M3 and give finger pads high friction; verify the 30 mm cube does not slip in sim [Squint friction lesson].
  - Wrist camera: set sim camera intrinsics from the Orbbec DCW2 factory intrinsics, mount pose from CAD/measure, then fix pose/FOV with an overlay tool — render the sim wrist view at the same commanded joints as a saved real frame and blend; iterate until table edges and cube line up [ManiSkill camera_alignment.py].
  - State convention: record commanded joints as `observation.state` in sim, exactly as the real recorder does (no feedback), plus σ ≈ 2–5° noise [Squint].
- Day 2 — generate data with the privileged expert:
  - Run our IK + grasp planner on true object poses in Isaac Sim; same episode cut as real M3 episodes (3 standing ticks before pregrasp → back at look pose), 10 Hz, same image size and camera key names as the real LeRobot 0.6.1 recorder; keep only successful episodes [LeIsaac state machine / DOFBOT].
  - DR per episode: object pose over the real reachable area, cube colour (contrast cases), table texture/colour, light exposure/colour temperature [NVIDIA ranges: exposure (−4, 3), 2500–9500 K, HDRI], camera pose ±2 cm / ±0.05 rad [NVIDIA], colour jitter [Squint], object size ±few mm, friction.
  - Target 300–500 sim episodes (Genesis used 50, NVIDIA 70–100; more is cheap), write video streams directly (no PNGs) [so101-real2sim]. If time allows, green-screen variant: real table photo as background [ManiSkill / LeIsaac DigitalTwin].
- Day 3 — train and test:
  - Three ACT runs with our existing LeRobot→TensorRT path: (a) real-only (our existing M3 episodes or 10–20 fresh ones), (b) sim-only, (c) sim pretrain → co-train with real oversampled to ~30 % of frames [Genesis].
  - Real eval: 20 trials on fixed, marked placements per model, logging pick / lift / place stages separately [nairb36 stage metrics]; the sim-only run doubles as a calibration check (if it reaches but misses grasps, revisit offsets/servo model, not the policy).

Catalogue of what is directly reusable:
| Item | Reuse for us | Effort |
|---|---|---|
| ManiSkill `camera_alignment.py` idea (overlay) | Port to Isaac wrist camera | small |
| NVIDIA DR config values (light, HDRI, camera pose) | Copy ranges into our generator | small |
| LeIsaac DigitalTwin / state-machine / LeRobot recorder | Patterns; code runs only on Isaac Sim ≤ 5.1 / Isaac Lab ≤ 2.3 | medium |
| Genesis co-train recipe (50 sim + 10 real ×3) | Data mix for ACT | small |
| so101-real2sim actuator-ID method | Servo model, adapted to no-feedback (video) | medium |
| DOFBOT offset correction | Per-joint offsets in state | small |
| Squint (ManiSkill) | Alternative: RL expert/policy with wrist cam if IL stalls | large (new sim) |

### Gaps
- Whether our Isaac Sim 5.0 grasp (7/7 lifts) holds with a soft, speed-clamped servo model is untested; no source covers a no-feedback arm.
- No source gives an M3 Pro / DCW2 camera asset or intrinsics workflow; the 3-day estimate is our inference, not a reported timeline from any pipeline.
