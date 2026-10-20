# Fine-motor / high-precision manipulation with learned policies on low-cost arms (2023-2026)

Scope note: all numbers below are REAL-ROBOT unless marked "(sim)". Dates are arXiv first-submission dates where known. Items marked "(search snippet only)" were seen in search-result summaries but their full text was not read; treat them as weaker evidence. Researched 2026-10-20.

Cost classes for orientation: SO-100/SO-101 ~$100-300 per arm (Feetech STS3215 hobby serial-bus servos, position control only); ALOHA/ALOHA 2 use ViperX 300 followers (Dynamixel, ~$5.6k each, "5-8 mm accuracy" per the ACT paper); AgileX Piper ~$2.2k material cost per the OpenEAI paper. Our Yahboom 5-DOF serial-bus-servo arms are closest in class to SO-101 (also 5 DOF + gripper, hobby bus servos, position control).

## 1. What are the most precise tasks demonstrated on low-cost arms, with what tolerances and success rates?

### Takeaway
The best-documented "mm-level" learned results on cheap arms are all on the ~$5k ALOHA/ViperX class, not on $100-300 hobby-servo arms: ACT battery slotting 96% (50 demos), ALOHA Unleashed gear insertion and robot-finger replacement described as "millimeter tolerance" at 75-95% for the first insertion (but ~4,000-5,000 demos per task), and velcro threading only 20%. On SO-100/SO-101 the published results are pick-place, stacking, sorting, pen-into-holder (45-80%) and test-tube-into-rack (25-30%); no SO-100/101 paper or report found demonstrates connector plugging, threading, or quantified mm-tolerance insertion.

### Cited Findings

**ALOHA (ViperX, Dynamixel), ACT paper (Zhao et al., arXiv 2304.13705, Apr 2023)**
- Six real tasks, 25 trials each, ACT subtask success: Slot Battery grasp 100 / place 100 / insert 96; Slide Ziploc grasp 92 / pinch 96 / open 88; Open Cup tip 100 / grasp 96 / open lid 84; Prep Tape grasp 96 / cut 92 / handover 72 / hang 64; Put On Shoe lift 100 / insert 92 / support 92 / secure 92; Thread Velcro lift 92 / grasp 40 / insert 20 — [ACT paper, arXiv 2304.13705v1](https://arxiv.org/html/2304.13705v1)
- Velcro geometry: the loop is 3 mm x 25 mm and the tie 2 mm x 10-25 mm; Thread Velcro (the tightest geometry) is the worst task at 20% insertion even with 100 demos (other tasks used 50). Battery slot tolerance is not given in the paper — [ACT paper](https://arxiv.org/html/2304.13705v1)
- Other stated tolerances: user study zip-tie hole 4 mm x 1.5 mm, cups with 2.5 mm clearance; sim insertion ~5 mm clearance (sim); sim cube transfer ~1 cm clearance (sim) — [ACT paper](https://arxiv.org/html/2304.13705v1)
- Abstract headline: "80-90% success" on six real tasks from "only 10 minutes worth of demonstrations" — [ACT abstract](https://arxiv.org/abs/2304.13705)
- All baselines (BC-ConvMLP, BeT, RT-1, VINN) reach 0% final-stage success on the real tasks; in sim bimanual insertion ACT got 32% (scripted data) / 20% (human data) (sim) — [ACT paper](https://arxiv.org/html/2304.13705v1)

**ALOHA 2, ALOHA Unleashed (Zhao et al., Google DeepMind, arXiv 2410.13126, Oct 2024)**
- 20 trials per task: GearInsert-1 (>=1 gear) 95%, GearInsert-2 75%, GearInsert-3 (all 3 gears) 40%; FingerReplace 75%; LaceEasy 70%, LaceMessy 40%; ShirtEasy 75%, ShirtMessy 70%; RandomKitchen Bowl 95% -> Bowl+Cup 65% -> +Fork 25% — [ALOHA Unleashed](https://arxiv.org/html/2410.13126)
- FingerReplace requires insertion with "millimeter tolerance"; GearInsert requires mm-level precision with a friction fit; both done from vision alone, without force-torque sensing, despite low-precision arms (no numeric clearance given) — [ALOHA Unleashed](https://arxiv.org/html/2410.13126)
- Demo counts: gears 4,005; finger replace 5,247; lacing 5,133; shirt 8,658; total ~26k real episodes collected on 10 robots in 2 buildings over 8 months by 35 operators — [ALOHA Unleashed](https://arxiv.org/html/2410.13126)

**Physical Intelligence, Real-Time Chunking (Black et al., arXiv 2506.07339, Jun 2025)**
- pi0.5 on a bimanual system of two 6-DoF arms with parallel-jaw grippers at 50 Hz; real tasks include "Plug ethernet" (reorient and plug in both ends of a cable, 6 steps) and "Light candle" (strike a match, light a candle); 10 trials per task/method, 480 episodes, ~28 h of robot time. Light candle is called the most precision-sensitive task; RTC had a large final-score advantage there and still lit the match with >300 ms delay. Per-task success rates are only in figures, arm model/cost not stated in the text read — [RTC paper](https://arxiv.org/html/2506.07339)

**Pi0 on ALOHA (openpi)**
- openpi ships a pi0-ALOHA-pen-uncap checkpoint ("uncaps a pen", trained on public data) alongside towel-folding and tupperware ones, but warns pi0 "was developed for our own robots" and "may or may not work for you"; no success numbers or recommended fine-tuning data size given — [openpi README](https://github.com/Physical-Intelligence/openpi)

**Mobile ALOHA (Fu et al., arXiv 2401.02117, Jan 2024)**
- 50 demos per task; co-training with static ALOHA data "can increase success rates by up to 90%"; tasks are household (saute shrimp, wall cabinet with heavy pots, call/enter elevator, rinse pan), not mm-precision — [Mobile ALOHA abstract](https://arxiv.org/abs/2401.02117)

**SO-100 / SO-101 (Feetech STS3215)**
- SmolVLA paper (HF, arXiv 2506.01844, Jun 2025), SO-100, 50 demos/task, fine-grained scoring: Pick-Place ACT 70 / pi0 100 / SmolVLA 75; Stacking 50 / 40 / 90; Sorting 25 / 45 / 70; averages 48.3 / 61.7 / 78.3. Number of eval trials not stated — [SmolVLA paper](https://arxiv.org/html/2506.01844)
- SmolVLA, SO-101 Pick-Place-Lego (small Lego brick into a transparent box, described as needing "precise grasping"): ACT 70 in-distribution / 40 OOD; SmolVLA 90 / 50 — [SmolVLA paper](https://arxiv.org/html/2506.01844)
- "Benchmarking VLAs on SO-101" (Yu & Qiu, Hiroshima Univ., arXiv 2606.08881, Jun 2026): 100 teleop demos per task, 20 trials per model-task. Precision Pen Placement ("insert it into the pen holder", tolerance not quantified): Wall-X 80, pi0.5 65, ACT 50, SmolVLA 45. Pen Transfer 70-95; Color Sorting 0-10; Multi-Object Packing 10-55; averages pi0.5 56.25, Wall-X 51.25, ACT 33.75, SmolVLA 32.5 — [SO-101 VLA benchmark](https://arxiv.org/html/2606.08881); summary [blog](https://mcobzarenco-fontaine-blog.static.hf.space/papers/so101-vla-benchmark.html)
- Community study "so101-tube-insertion" (FaithQin, trials Aug-Sep 2026, snapshot 2026-10-07): ACT on SO-101, 49 demos, seat a 50 mL test tube in a rack: 6/20 (vision + joint positions) vs 5/20 (plus servo load and current); 95% CIs ~9-54%; difference null (McNemar p=1.00) — [so101-tube-insertion](https://github.com/FaithQin/so101-tube-insertion)
- Sim-to-real RL "Squint" (arXiv 2602.21203, Feb 2026, RA-L 2026): 8 SO-101 tasks trained in ManiSkill3 in ~15 min on one RTX 3090, zero real data; real results 73/80 (reach 10/10 x2, lift cube 10/10, lift can 9/10, place 10/10 x2, stack cube 8/10, stack can 6/10). Reach success criterion is "within 2 cm above the target" — coarse, not mm-level — [Squint](https://arxiv.org/html/2602.21203)
- Community: a SmolVLA SO-101 model card reports an 89% grasp rate (72/81) on a held-out 81-target grid (grasp only, not full place) (search snippet only) — [dobri420/pick-cube-smolvla-so101](https://huggingface.co/dobri420/pick-cube-smolvla-so101); an ACT SO-101 reproduction reached 50% (10/20) (search snippet only) — [khushi02 repo](https://github.com/khushi02/so101-imitation-learning-ACT); a sim-expert-to-real ACT pipeline reports 68% on unseen layouts, with lateral grasp precision the main residual failure (search snippet only) — [aviadarn/so101-lerobot](https://github.com/aviadarn/so101-lerobot)

**AgileX Piper (~$2.2k)**
- OpenEAI-Platform (arXiv 2606.03392, Jun 2026): pi0 fine-tuned with matched data on three arms; averages ARX R5 0.71, Piper 0.64, OpenEAI-Arm 0.75; Piper per task Clean Table 0.86, Make Tea 0.40, Fold Towel 0.80, Fold T-shirt 0.50; material cost ARX R5 8.60 kUSD, Piper 2.16 kUSD, OpenEAI-Arm 0.79 kUSD (claimed <=0.05 mm EE error for their arm) — [OpenEAI-Platform](https://arxiv.org/html/2606.03392)
- Piper insertion: "Insert Tube" task with a nominal base-policy success of 75% (search snippet only) — [Self-Adaptive VLA, arXiv 2609.30092](https://arxiv.org/html/2609.30092v1); AsyncVLA reports 87.0% average real success over four Piper tasks incl. "precise grasping and object placement", 50 trials/task (search snippet only) — [AsyncVLA, arXiv 2511.14148](https://arxiv.org/pdf/2511.14148)

**Precise RL on industrial arms (reference point, not low-cost)**
- HIL-SERL (Luo et al., arXiv 2410.21845, Oct 2024): 100% success (100 trials) on RAM insertion, SSD assembly, USB grasp-insertion, cable clipping, IKEA panels, car dashboard, timing belt, Jenga whipping, object flipping; average imitation baseline 49.7%; e.g. RAM insertion BC 29% vs 100%, timing belt BC 2% vs 100%, USB BC 26% vs 100%; Diffusion Policy 27% on RAM insertion; average cycle time 9.6 s -> 5.4 s; "1 to 2.5 hours" of real training for most tasks. Tolerances in mm not given — [HIL-SERL paper](https://arxiv.org/html/2410.21845). The arm model was not named in the fetched text (see Gaps).

### Inferences
- The tightest quantified geometry with a learned policy on a cheap-ish arm is ACT's velcro (3 mm loop) and it is also the one that largely fails (20%); tasks that succeed at >=90% (battery slot, shoe) likely have self-aligning geometry, but the papers do not give clearances, so "mm-level" claims are mostly qualitative.
- Multi-stage precision compounds: per-step success ~80-95% becomes 40% by the third gear and 25% by the fork (ALOHA Unleashed); expect the same on any arm.
- On $100-300 SO-class arms, the published ceiling as of Oct 2026 is roughly "pen into a holder at 45-80% with 100 demos" and "test tube into a rack at 25-30% with 49 demos"; none of the classic fine-motor benchmarks (connector plugging, threading, sub-mm peg-in-hole) has a published SO-100/101 result.

### Gaps
- No source found with an SO-100/101/Koch/myCobot/xArm Lite result on USB/connector plugging, cable threading, or a peg-in-hole with a stated clearance.
- Exact battery-slot, gear and finger-socket clearances in ACT / ALOHA Unleashed are not published.
- RTC per-task success for "plug ethernet" exists only in figures; the arm type/cost of PI's bimanual robots was not stated in the text read.
- Koch v1.1, myCobot and xArm Lite: no quantitative fine-motor learned-policy results found in this pass.
- LeRobot hackathon (2025) results were not found with quantitative success rates.

## 2. What made the successful cases work (demos, data quality, cameras, control rate, chunking, RL, compliance, fingertips)?

### Takeaway
Three recipes appear: (a) 50-100 high-quality demos + action chunking + wrist cameras at high control rate (ACT: good for coarse-to-medium precision, fails on the finest task); (b) brute-force scale, thousands of demos per task + a diffusion head (ALOHA Unleashed: mm-tolerance insertions at 75-95% for the first step); (c) RL fine-tuning on top of ~20-30 demos with a learned reward and human interventions (HIL-SERL: ~100% on precise insertions, but shown on industrial arms with impedance control). Pretraining on community data helps cheap-arm VLAs (SmolVLA 51.7 -> 78.3), but it is pick-place-class data.

### Cited Findings
- ACT: 50 demos/task (100 for velcro), ~10-20 min of demo data per task; recording and control at 50 Hz; chunk size 100; four Logitech C922x webcams at 480x640 (the text read does not say where they are mounted; ALOHA is commonly built with two wrist + two fixed cameras, not re-verified here); ACT vs all baselines at 0% shows chunking + CVAE were decisive — [ACT paper](https://arxiv.org/html/2304.13705v1)
- ACT argues humans lack "industrial-grade proprioception" yet do delicate tasks by "learning from closed-loop visual feedback and actively compensating for errors"; the policy reacts to object positions rather than modelling contact precisely — [ACT paper](https://arxiv.org/html/2304.13705v1)
- ALOHA Unleashed recipe: 217M-param transformer with diffusion loss, 4 ResNet50 backbones on 4 RGB views, 50-action chunks (~1 s), 14-DoF joint-position actions at a 50 Hz target, replan ~every 1 s; trained 2M steps at batch 256 on 64 TPUv5e (~265 h); a diffusion head beat L1 regression (ACT-style) 70% vs 25% on ShirtMessy at similar size — [ALOHA Unleashed](https://arxiv.org/html/2410.13126)
- ALOHA Unleashed data came from 10 uncalibrated robots, 35 operators, 8 months — scale and diversity rather than calibration — [ALOHA Unleashed](https://arxiv.org/html/2410.13126)
- HIL-SERL recipe: ~20-30 offline demos; reward classifier from ~200 positive / 1000 negative frames (~10 trajectories, ~5 min); SpaceMouse interventions decreasing over training; wrist + side cameras cropped to 128x128 with pretrained ResNet-10; impedance controller with reference limiting for contact tasks; RL from scratch without demos/corrections = 0% — [HIL-SERL paper](https://arxiv.org/html/2410.21845)
- HIL-SERL's stated reason RL beats BC on precise tasks: RL learns from its own successes and failures and explores a wider state range than DAgger, while BC "often performs poorly on precise tasks like RAM insertion" — [HIL-SERL paper](https://arxiv.org/html/2410.21845)
- LeRobot's HIL-SERL port (SO-100/SO-101): default 10 fps control, 128x128 cropped images, actions in end-effector space with workspace bounds (example EE step 2 cm per axis), optional motor current in the observation "to detect contact events"; docs say some tasks "are nearly impossible to learn in joint space but become learnable" in EE space; recommended tasks are push-cube and pick-and-lift cube of 5-10 s; example records 15 demos; geared SO-101 leader makes interventions smoother than the gearless SO-100; docs claim "near-perfect task success" but give no SO-100 number — [LeRobot HIL-SERL docs](https://huggingface.co/docs/lerobot/hilserl)
- AutoSERL (arXiv 2607.01651, Jul 2026): from a single demonstration, 100% on insertion tasks across two (unnamed) platforms, matching HIL-SERL — [AutoSERL abstract](https://arxiv.org/abs/2607.01651)
- SmolVLA: 50 demos/task (10 per each of 5 start positions); top + wrist cameras on SO-100; chunk 50, 10 flow-matching steps; pretraining on 481 community datasets (~22.9k episodes, 10.6M frames) raised the SO-100 multi-task average from 51.7% to 78.3%; async inference ~30% faster (13.75 s -> 9.7 s per task) but sorting fell 70 -> 50 — [SmolVLA paper](https://arxiv.org/html/2506.01844)
- RTC: at 50 Hz with 50-step chunks, naive chunk switching creates "abrupt, out-of-distribution accelerations" at boundaries; temporal ensembling oscillated so much at +100/+200 ms latency that it triggered the robot's protective stop; RTC executed the same motion ~20% faster than synchronous inference — [RTC paper](https://arxiv.org/html/2506.07339)
- Squint (SO-101 sim-to-real RL): wrist camera only + proprioception, trained at 10 Hz in sim, run at 30 Hz on the real arm with actions scaled by 0.15; domain randomization of camera pose/FOV, lighting/colour, object size/friction, gripper speed and 5 deg proprioception noise; dropping colour jitter cut real success 73/80 -> 58/80 — [Squint](https://arxiv.org/html/2602.21203)
- Tactile/force sensing for tight insertion (not on cheap arms): ContactDP reports vision-only ~40-70% -> +force ~87-93% -> +tactile 95.8-100% on tight connector insertion (search snippet only) — [ContactDP, arXiv 2609.23800](https://arxiv.org/pdf/2609.23800); PolyTouch 86% vs 29% vision-only on wrench insertion (search snippet only) — [PolyTouch](https://www.alphaxiv.org/abs/2504.19341); but ManiFeel found little tactile benefit for USB and power-plug insertion (search snippet only) — [ManiFeel](https://arxiv.org/pdf/2505.18472)

### Inferences
- Demo count scales with required precision: ~50 demos suffices for cm-scale pick-place on SO-100; tasks with self-aligning mm features reached ~96% with 50 demos on ALOHA; truly mm-tolerance friction-fit insertions needed thousands of demos per task (ALOHA Unleashed) or online RL.
- The only learned route to ~100% on precise insertion in the literature (HIL-SERL) relies on compliant impedance control and EE-space actions; hobby position servos give no true torque mode, so the LeRobot port substitutes EE bounds, small steps and optional current readings — the compliance ingredient is the part least transferable to our arms.
- Wrist cameras are verified in HIL-SERL, SmolVLA SO-100 and Squint (and are standard on ALOHA builds, not re-verified here); the SO-101 Lego task in SmolVLA used top + side only and still reached 90% in-distribution, so a wrist camera helps but is not strictly required for cm-scale placement.

### Gaps
- No ablation found on SO-class arms showing how success scales from 50 -> 500 -> 5,000 demos for a precise task.
- No source quantified the effect of fingertip design (e.g. soft vs. hard pads) on cheap-arm precision tasks; ALOHA 2's gripper changes were not read in detail ([ALOHA 2 abstract](https://arxiv.org/abs/2405.02292) gives only general claims).
- No published HIL-SERL success rate on SO-100/101 for any task was found, despite LeRobot's support for it.

## 3. Failure modes and limits reported on cheap arms

### Takeaway
Reported failures on cheap arms are dominated by grasp instability and imprecise alignment, not by contact physics: in the SO-101 tube study 27/29 failures happened before the tube touched the rack. Hardware-level limits named explicitly are backlash, actuator noise, jitter, latency, limited repeatability, imperfect camera-to-action calibration, low payload, and gripper overheating (LeRobot halves gripper torque/current to avoid burnout). Calibration drift between sessions and multi-robot calibration differences are mentioned (ALOHA's several-mm build variance) but rarely quantified on SO-class arms.

### Cited Findings
- The SO-101 benchmark paper describes the arm as having "limited actuator precision, reduced control stability, lower payload capacity, and noisier motion execution", citing actuator noise, joint backlash, control latency, trajectory jitter, limited positional repeatability and imperfect calibration between visual observations and executed actions — [SO-101 VLA benchmark](https://arxiv.org/html/2606.08881)
- SO-101 benchmark failure taxonomy: grasp instability and repetition loops appear in ~91-100% of failed episodes for all policies; "state mismatch" (continuing after a failed grasp) from 98% (ACT) to 46% (pi0.5); precision misalignment 0-15% of failures; recovery rates pi0.5 30.77%, Wall-X 20.51%, ACT 6.45%, SmolVLA 3.23% — [SO-101 VLA benchmark](https://arxiv.org/html/2606.08881)
- SO-101 tube insertion failures (29): never grasped 18, lost from jaws 6, pecked 2, jammed at rack 2, other 1; a worn elbow servo "failed certification" mid-study and stopped a bench run — [so101-tube-insertion](https://github.com/FaithQin/so101-tube-insertion)
- LeRobot SO follower code: gripper Max_Torque_Limit 500 ("50% of max torque to avoid burnout"), Protection_Current 250 ("50% of max current to avoid burnout"), Overload_Torque 25% — [LeRobot so_follower.py](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/robots/so_follower/so_follower.py)
- LeRobot writes position PID gains P=16, I=0, D=32 to the STS3215 at connect; `max_relative_target` clips per-step motion "for safety purposes"; read retries exist because corrupted status packets "otherwise abort the control loop" — [LeRobot config_so_follower.py](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/robots/so_follower/config_so_follower.py)
- STS3215: 7.4 V ~19.5 kg.cm (12 V variant ~30 kg.cm), 1:345 metal gearbox, 12-bit magnetic encoder (vendor listing, search snippet only) — [OpenELAB STS3215](https://openelab.io/cs/products/feetech-sts3215-c001-servo); a third-party SO-101 spec page gives a ~500 g payload that "varies with pose and servo temperature" (search snippet only) — [roboticscenter.ai SO-101 specs](https://www.roboticscenter.ai/de/hardware/so-101/specs)
- No published mm repeatability/backlash figure for SO-100/STS3215 was found — [SO-ARM100 repo](https://github.com/TheRobotStudio/SO-ARM100) (search did not surface one)
- ALOHA Unleashed robots were uncalibrated, with base and camera mounting variance; measured base-segment distance ranged 731.5-736 mm across units — [ALOHA Unleashed](https://arxiv.org/html/2410.13126)
- ACT hardware limits: insufficient torque for heavy lifts, sealed bottles, tight marker caps; "fingernail"-type tasks hard; velcro failures from the gripper closing too early and missing the loop, worsened by low contrast and small size; candy unwrapping 0/10 in the first protocol — [ACT paper](https://arxiv.org/html/2304.13705v1)
- ALOHA Unleashed failures: under-represented start poses (finger upside down), shoe tipping, lace tangles; added stages degrade on small gears and thin objects (forks) — [ALOHA Unleashed](https://arxiv.org/html/2410.13126)
- Piper: failures "concentrated in grasp initiation and precise placement", attributed to its "distinct kinematic and actuation design" making "precise grasp alignment harder under the same fine-tuning budget" — [OpenEAI-Platform](https://arxiv.org/html/2606.03392)
- Chunk-boundary jerks and latency: synchronous inference pauses shift robot dynamics; temporal ensembling oscillations tripped protective stops — [RTC paper](https://arxiv.org/html/2506.07339)
- Squint on SO-101: can grasping hardest because of tipping and "insufficient gripper friction"; lighting changes degraded performance — [Squint](https://arxiv.org/html/2602.21203)
- SmolVLA community data: noisy/missing task labels and inconsistent camera naming required LLM relabelling and manual view mapping — [SmolVLA paper](https://arxiv.org/html/2506.01844)

### Inferences
- On cheap arms the binding constraint for "fine" tasks seems to be the approach and grasp (localization + jaw stability), which happen before any precise contact; improving grasp (fingertip friction, wrist camera, repeatable pre-grasp pose) probably matters more than adding force channels.
- The halved gripper torque that LeRobot needs for servo survival directly limits grip force, so friction-fit insertions (gears, plugs) that need push force are likely out of reach on stock SO-class grippers.

### Gaps
- No quantitative report found of session-to-session calibration drift or cross-unit policy transfer failure on SO-100/101 (only qualitative mentions).
- No source found quantifying STS3215 overheating during long policy rollouts (only the burnout-protection defaults).
- The GitHub issue tracker of LeRobot was not searched directly; search engines did not surface jitter/overheating threads.

## 4. Do closed-loop learned policies exceed the arm's open-loop accuracy? Evidence for and against

### Takeaway
Yes, with evidence on the ALOHA class: ViperX arms rated at 5-8 mm accuracy, uncalibrated and differing by several mm between units, achieved slot/insertion tasks the authors call mm-tolerance (96% battery insertion; 75-95% gear/finger insertion) using only visual feedback. But the evidence is qualitative (clearances not stated), and the same setups fail at the finest geometry (3 mm velcro loop: 20%). On SO-class arms there is no quantified evidence that a learned policy beats the arm's open-loop accuracy at mm level.

### Cited Findings
- ViperX 6-DoF follower: 750 g payload, "5-8 mm accuracy" (ACT paper's stated spec); yet Slot Battery insert 96%, Put On Shoe 92% — [ACT paper](https://arxiv.org/html/2304.13705v1)
- ALOHA Unleashed: gear and finger insertions "with millimeter tolerance" succeed from vision alone without F/T sensing "despite low-precision arms", on uncalibrated robots with several-mm build variance — [ALOHA Unleashed](https://arxiv.org/html/2410.13126)
- Counter-evidence on the same class: velcro (3 mm x 25 mm loop) insertion 20%, failures from mis-timed gripper closure and missing the loop — [ACT paper](https://arxiv.org/html/2304.13705v1)
- Counter-evidence on SO-101: tube-into-rack 25-30% with 49 demos; adding servo load/current did not help; most failures before contact — [so101-tube-insertion](https://github.com/FaithQin/so101-tube-insertion); pen-into-holder tops out at 80% with 100 demos — [SO-101 VLA benchmark](https://arxiv.org/html/2606.08881)
- Hardware matters at fixed data: pi0 with matched data averaged 0.64 on Piper vs 0.71 on ARX R5 and 0.75 on a more precise arm, with Piper failing in precise placement — [OpenEAI-Platform](https://arxiv.org/html/2606.03392)
- RL closes residual precision gaps on industrial arms (BC 26-29% -> 100% on USB/RAM insertion) — [HIL-SERL paper](https://arxiv.org/html/2410.21845); a sim-trained proprioceptive RL policy inserted pegs with 0.5-1.5 mm clearance per side in 73% of 237 real trials (hardware not identified; search snippet only) — [Blind Precise Assembly, arXiv 2610.23849](https://arxiv.org/html/2610.23849)

### Inferences
- Mechanism: a visual closed loop converts the arm's absolute (kinematic/calibration) error into a relative error between gripper and target seen in the wrist camera; what it cannot fix is resolution, backlash and dead-band of the servos at the final approach, nor missing compliance during contact. So a policy can beat open-loop accuracy (cm-level errors in our own logs, e.g. the M3's grips ~1.5 cm short per the project memory note m3pro-taught-aim) but not the servo's incremental resolution/backlash.
- Rough resolution bound (own arithmetic, not a source): a 12-bit encoder gives 360/4096 ~ 0.088 deg per step; at 0.3 m radius that is ~0.46 mm at the tip per step of one joint, before backlash and P-gain dead-band; so sub-mm placement is near or below the actuator floor on STS3215-class servos, while 2-5 mm is not.

### Gaps
- No study found that measures an SO-class arm's open-loop repeatability and then a learned policy's achieved placement error on the same arm.
- No published clearance numbers for ALOHA's battery slot, gears or finger socket, so "mm tolerance" cannot be pinned to 0.5 vs 2 vs 3 mm.

## 5. Honest assessment: is mm-level fine motor skill realistic on ~$100-300 serial-bus-servo arms with a learned policy?

### Takeaway
Partly. Realistic: 2-5 mm-class placement and slotting into chamfered or self-aligning fixtures, at roughly 50-90% with 50-200 good demos, wrist camera, action chunking and a fixed setup. Out of reach or unproven: sub-mm insertion, friction fits that need push force, connector plugging and threading, and >95% reliability on multi-step precision tasks. The precise successes in the literature used ~$5k arms with thousands of demos (ALOHA Unleashed) or RL with compliant industrial arms (HIL-SERL); no SO-class paper shows them.

### Cited Findings
- Best published SO-101 "precision" numbers: pen into holder 45-80% (100 demos, 4 policies) — [SO-101 VLA benchmark](https://arxiv.org/html/2606.08881); Lego into a transparent box 70-90% in-distribution, 40-50% OOD (50 demos) — [SmolVLA paper](https://arxiv.org/html/2506.01844); tube into rack 25-30% (49 demos) — [so101-tube-insertion](https://github.com/FaithQin/so101-tube-insertion)
- Sim-to-real RL on SO-101 reaches 91% on reach/lift/place/stack with ~15 min training and zero real data, but the precision bar is cm-level (reach within 2 cm) — [Squint](https://arxiv.org/html/2602.21203)
- ALOHA-class mm-tolerance insertion needed 4,005 (gears) and 5,247 (finger) demos for 95% / 75% first-step success, dropping to 40% for three gears — [ALOHA Unleashed](https://arxiv.org/html/2410.13126)
- With 50 demos on ALOHA, battery slotting 96% but 3 mm velcro threading 20% even with 100 demos — [ACT paper](https://arxiv.org/html/2304.13705v1)
- RL can lift precise insertions to ~100% in 1-2.5 h of real training, but in the published case with impedance-controlled industrial arms, EE-space actions, and ~20-30 demos — [HIL-SERL paper](https://arxiv.org/html/2410.21845); LeRobot's SO-100/101 port is documented for push/pick-lift tasks without published success numbers — [LeRobot HIL-SERL docs](https://huggingface.co/docs/lerobot/hilserl)
- Force/tactile sensing raises tight-insertion success substantially in some studies (ContactDP vision ~40-70% -> tactile ~96-100%, search snippet only) — [ContactDP](https://arxiv.org/pdf/2609.23800); TacDiffusion's own mm-level insertion success is ~50% (search snippet only) — [TacDiffusion](https://arxiv.org/abs/2409.11047); servo load/current on SO-101 did not help in the tube study — [so101-tube-insertion](https://github.com/FaithQin/so101-tube-insertion)

### Inferences
- Conditions under which mm-ish results are plausible on Yahboom/SO-class arms: (1) a wrist camera that sees the gripper tip and target at the final approach; (2) fixtures with chamfers/funnels so the last 1-2 mm are passive; (3) one fixed task and scene, 100-300 consistent demos from a skilled operator, same arm unit for data and deployment; (4) chunked policy at the highest stable control rate, with smooth chunk transitions (RTC-style or small chunks) to avoid boundary jerks; (5) higher-friction fingertips and a pre-grasp that centres the part; (6) optionally RL fine-tuning (HIL-SERL style, EE-space, small steps, workspace bounds) on a short sub-task like the final insertion, accepting this is unproven on hobby servos.
- What is probably out of reach on stock hobby-servo arms with a simple gripper: sub-mm or friction-fit insertions (gears, USB/RJ45 plugging), thread/zip-tie/velcro threading, tasks needing controlled push force or compliance, and long multi-step precise sequences at high reliability. Our arms have 5 DOF (no full 6-DoF orientation), which further limits approach angles for insertion; this is a geometric point, not a sourced finding.
- Expected-value judgement for the report writer: the cheapest big win is likely closing the loop visually at the final approach (wrist camera + policy or classic visual servoing) to beat the arm's cm-level open-loop errors and reach the few-mm regime; claims of "very precise training giving mm-level fine motor skills" on $100-300 arms are not supported by any published result found as of 2026-10-20.

### Gaps
- No direct evidence exists (found) for Yahboom serial-bus-servo arms specifically; all inferences transfer from SO-100/101 and ALOHA.
- No head-to-head data on whether more demos (e.g. 1,000+) on an SO-class arm would reach ALOHA-class insertion results; the ALOHA Unleashed scale has not been replicated on hobby servos.
- HIL-SERL hardware: the fetched paper text did not name the arm; it is widely described as Franka-based in secondary coverage, but that was not verified here.
