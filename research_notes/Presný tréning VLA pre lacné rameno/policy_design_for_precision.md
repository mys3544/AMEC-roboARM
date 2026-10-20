# Policy design choices for precise, low-error manipulation (VLA / visuomotor policies, 2024-2026)

Scope: which design choices of learned manipulation policies most affect positional precision, with evidence from
ablations and benchmark numbers, for a low-cost 5-DOF bus-servo arm with a wrist camera, a Jetson Orin NX 16 GB on
the robot, and an RTX 5090 (32 GB) / RTX A4500 (20 GB) for training and serving. Candidates: SmolVLA, X-VLA,
pi0 / pi0.5 (openpi), GR00T N1.x, ACT, Diffusion Policy; LeRobot as the stack. Research date: 2026-10-20.

Source-quality legend: items marked "(search summary)" were taken from a search engine's summary of the paper/page
and were not checked against the full text; items marked "(secondary)" come from a blog, wiki or mirror rather than
the authors. Everything else was read from the primary page (arXiv HTML/abstract, official docs, official blog).

General caveat: almost no paper reports positional error in millimetres. The evidence is success rate / progress
score on tasks of known tolerance (e.g. ACT's 3 mm velcro loop, ~5 mm sim insertion clearance), so "precision" below
is mostly inferred from success on tight-tolerance tasks.

---

## 1. Action representation: joint vs end-effector space; absolute vs delta / relative (to current state or chunk start)

### Takeaway
The best-controlled study to date (Feb 2026, 13,000+ real rollouts, 500+ models) finds **joint space + "chunk-wise
relative" actions (every action in the chunk is an offset from the robot state at the chunk start)** best in-domain,
beating absolute targets by ~8-15 points on average, while **step-wise deltas (each action relative to the previous
one) are worse** because they integrate prediction noise; EE space wins only when transferring across embodiments.
UMI (2024), LeRobot's pi-family processors, openpi and GR00T N1.6/N1.7 all moved to state-relative chunks; ACT (2023)
is the outlier that found absolute joint targets better than deltas.

### Cited Findings
- **"Demystifying Action Space Design for Robotic Manipulation Policies"** (Feng et al., arXiv 2602.23408, v1
  2026-02-26, v2 2026-04-23): 13,000+ real rollouts, 500+ trained models, four scenarios; "properly designing the
  policy to predict delta actions consistently improves performance"; joint space favours control stability,
  task (EE) space favours generalization — [arXiv abstract](https://arxiv.org/abs/2602.23408)
- Same paper, Table 1 overall averages (abs -> delta), verified in the HTML: regression head (ACT-style) EE 63.4 ->
  78.4, joint 71.2 -> 79.7; flow-matching head ("DP") EE 71.9 -> 82.9, joint 79.6 -> 88.0. Single-arm averages:
  ACT-EE 69.0 -> 89.6, ACT-joint 77.3 -> 88.0, DP-EE 74.0 -> 91.4, DP-joint 85.0 -> 95.9. Not uniform: e.g.
  multi-task Cube joint/ACT 94.4 abs vs 88.8 delta; Pick&Place joint/ACT 73.8 vs 72.8 — [arXiv HTML v2](https://arxiv.org/html/2602.23408v2)
- Same paper: the "delta" that wins is **chunk-wise** (relative to the state at chunk start); step-wise deltas lose
  "upwards of 10% on average" (figure values not in text). Proposition 4.1: step-wise decoding amplifies per-step
  prediction noise by about (2k+1)/pi for chunk length k (worst-case bound), while chunk-wise delta and absolute decode
  with O(1) noise — [arXiv HTML v2](https://arxiv.org/html/2602.23408v2)
- Same paper, setup: mainly a single-arm AgileX PiPER (6-DoF) at 30 Hz position control, dual PiPER for bimanual,
  AIRBOT for cross-embodiment, plus RoboTwin-2.0 sim (10 tasks, 50 demos); policies = FiLM ResNet-18 + 6-layer
  transformer with either MSE-regression ("ACT") or flow-matching ("DP") heads; transfer runs with pi0-LoRA —
  [blog review (secondary)](https://mcobzarenco-fontaine-blog.static.hf.space/papers/action-space-design.html)
- Same paper (secondary blog numbers, Tables 4-5): rel-joint vs abs-joint at 100/250/500 demos: 77.2 vs 64.4, 90.8 vs
  82.6, 93.5 vs 84.2 — the delta advantage does not close with more data; RoboTwin rel-qpos best (ACT 46.3, DP 48.0) vs
  absolute EE 26.7 / 26.0 — [blog review (secondary)](https://mcobzarenco-fontaine-blog.static.hf.space/papers/action-space-design.html)
- Same paper: joint space wins in-domain (esp. with flow-matching heads and more data); EE space wins under
  cross-embodiment co-training and pi0-LoRA transfer (direction only, numbers in a figure) — [blog review (secondary)](https://mcobzarenco-fontaine-blog.static.hf.space/papers/action-space-design.html); the reviewer of the HTML notes guideline 3 (task space for transfer) is stated more strongly than the visible evidence — [arXiv HTML v2](https://arxiv.org/html/2602.23408v2)
- Same paper: no millimetre precision metric, no gripper-representation detail, normalization of deltas only
  mentioned as "proper normalization" — [arXiv HTML v2](https://arxiv.org/html/2602.23408v2)
- **UMI** (Chi et al., Feb 2024) Cup Arrangement, 20 trials: relative trajectory 20/20 (100%), delta 16/20 (80%),
  absolute 5/20 (25%). Explanation: relative trajectories need no shared global frame; absolute actions suffer from
  calibration error; step deltas accumulate tracking error — [arXiv](https://arxiv.org/abs/2402.10329)
- **LeRobot docs** (Action Representations): LeRobot's default is **absolute joint** actions; "relative actions (used by
  OpenPI / pi0)" = offset from the state at prediction time for every action in the chunk; step "delta" actions
  accumulate error; pi0, pi0.5 and pi0_fast support relative actions via `RelativeActionsProcessorStep`, enabled with
  `--policy.use_relative_actions=true --policy.relative_exclude_joints='["gripper"]'`, after recomputing stats with
  `lerobot-edit-dataset --operation.type recompute_stats --operation.relative_action true --operation.chunk_size 50`;
  relative conversion runs before normalization; relative actions are compatible with RTC because each chunk
  references the state cached at the start of inference; FK/IK processor steps (`ForwardKinematicsJointsToEE`,
  `InverseKinematicsEEToJoints`) exist for EE-space with an SO-101 URDF — [LeRobot docs](https://huggingface.co/docs/lerobot/action_representations)
- **openpi**: the README itself does not document delta vs absolute; pi0-DROID is "faster inference than
  pi0-FAST-DROID, but may not follow language commands as well" — [openpi README](https://github.com/Physical-Intelligence/openpi). (LeRobot's docs attribute the relative-chunk convention to OpenPI/pi0 — [LeRobot docs](https://huggingface.co/docs/lerobot/action_representations).)
- **pi0 paper** does not state whether actions are absolute or delta / joint or EE; action vector padded to 18 dims;
  proprio = joint angles q_t — [arXiv HTML](https://arxiv.org/html/2410.24164)
- **GR00T**: N1.6 (released 2025-12-15) "predicts state-relative action chunks for most embodiments, rather than
  absolute joint angles or end-effector positions used in N1.5", claimed benefits: generalization across initial
  configurations, robustness to calibration differences — [aiwiki (secondary)](https://aiwiki.ai/wiki/groot_n1_6), [mintlify mirror of Isaac-GR00T docs (secondary)](https://mintlify.wiki/NVIDIA/Isaac-GR00T/whats-new); N1.7 uses a **relative EEF** action space (deltas from current pose), action horizon 16 -> 40, state/action dims 29 -> 132 — [Isaac-GR00T README](https://github.com/NVIDIA/Isaac-GR00T)
- **X-VLA** (Oct 2025) pretrains on a unified **EE pose** representation: xyz, absolute rotation in Rotate6D, binary
  gripper; MSE for position/rotation, BCE for gripper — [arXiv HTML](https://arxiv.org/html/2510.10274). In LeRobot,
  X-VLA has action modes `ee6d` (20-D), `joint` (14-D), `so101_bimanual` (5 joints + gripper per arm) and `auto`
  (pads the dataset's action dim to 20, loss only on real dims) — [LeRobot X-VLA docs](https://huggingface.co/docs/lerobot/xvla)
- **OpenVLA-OFT**: LIBERO uses delta EE pose actions; ALOHA uses absolute target joint angles (14-D) — [arXiv HTML](https://arxiv.org/html/2502.19645)
- **ACT** (2023): actions = absolute target joint positions of the *leader* arms; "We also noted degraded performance
  when using delta joint positions as actions instead of target joint positions" (no numbers) — [arXiv](https://arxiv.org/abs/2304.13705)
- **Diffusion Policy** (2023): position-control action space "consistently outperforms" velocity control for DP
  (BCRNN and BET got worse when switched to position); explanation: position control suffers less from compounding
  error, and DP handles the extra multimodality — [arXiv](https://arxiv.org/abs/2303.04137)
- **"Do You Need Proprioceptive States in Visuomotor Policies?"** (arXiv 2509.18644, Sep 2025): relative EEF action
  space performed best; absolute EEF and joint-angle variants scored 0 on spatial generalization tests — [arXiv HTML](https://arxiv.org/html/2509.18644)
- **BEHAVIOR Challenge 2025 1st place** (arXiv 2512.06951): predicts offsets from the current joint configuration
  and normalizes each chunk index separately because early chunk steps are close to the current state (search summary) — [arXiv](https://arxiv.org/pdf/2512.06951)
- **Delayed vs chunked policies**: the 2026 Lazzati/Levine chunking study ran its real Franka experiments with
  15 Hz *delta-joint-position* control — [alphaXiv](https://www.alphaxiv.org/abs/2608.02547)

### Inferences
- For a 5-DOF serial-servo arm with a known URDF, the evidence favours **joint space + chunk-wise relative actions,
  gripper absolute** (exactly LeRobot's `use_relative_actions` + `relative_exclude_joints=["gripper"]` path for pi0 /
  pi0.5). EE space mainly pays off for cross-embodiment transfer (e.g. sharing data between the X3 Plus and the M3 Pro,
  or starting from an EE-pretrained base like X-VLA/GR00T N1.7). A 5-DOF arm cannot realise arbitrary 6-D EE poses, so
  EE actions would need a projection step in IK — another argument for joint space on this hardware.
- Relative chunks make the policy insensitive to per-servo zero offsets / recalibration (the claimed GR00T N1.6
  benefit), which matters for cheap servos that get re-centred.
- Never use step-wise (cumulative) deltas with long chunks: noise grows with chunk length.
- SmolVLA and ACT in LeRobot default to absolute joints; the docs list relative-action support only for the pi
  family, so using relative actions with SmolVLA/ACT/X-VLA would need a custom processor (not verified whether the
  same processor step works unchanged for them).

### Gaps
- No study found that measures positional error (mm) as a function of action representation; all use success/progress.
- The chunk-wise-delta paper never swept the training chunk length and does not say how deltas were normalized.
- No ablation found specifically on low-cost hobby servo arms (SO-100/SO-101, Koch, Yahboom) comparing absolute vs relative.
- The ACT 2023 "delta joints were worse" remark has no numbers, and it compared step deltas (likely), not chunk-wise relative.

---

## 2. Action chunking and execution: chunk length, temporal ensembling, real-time chunking (RTC), asynchronous inference

### Takeaway
Chunking is the single largest gain over single-step BC (ACT: 1% -> 44%; OpenVLA-OFT: +14 points), but **executing too
much of a chunk open-loop costs precision and reactivity**: best results come from predicting ~1-2 s and executing
roughly 8-30 steps before replanning (SmolVLA: 10 of 50; Diffusion Policy: 8; delta actions peak at ~30 of 60 at
30 Hz). Temporal ensembling gives small gains for ACT (+3.3%) but hurt pi0 and caused oscillations under latency in
the RTC study; **RTC (inpainting the new chunk to agree with the frozen prefix) plus asynchronous inference** is the
current best way to hide latency without jerks, and it is built into LeRobot for pi0/pi0.5/SmolVLA.

### Cited Findings
- **ACT** chunk-size sweep (temporal ensembling off, avg of 4 sim settings): success rises from 1% at k=1 to 44% at
  k=100, then dips slightly at k=200/400 (near open-loop) "due to lack of reactive behavior and the difficulty in
  modeling long action sequences"; chunking also lifts BC-ConvMLP and VINN — [ACT paper](https://arxiv.org/abs/2304.13705)
- **ACT** temporal ensemble (exponential weights w_i = exp(-m*i) over predictions for the same timestep): +3.3% for
  ACT, +4% for BC-ConvMLP, -20% for VINN; authors say TE "mostly benefits parametric methods by smoothing out modeling
  errors"; both chunking and TE described as important for "precise and smooth motion" — [ACT paper](https://arxiv.org/abs/2304.13705)
- **pi0**: H=50; at 20 Hz (UR5e/Franka) inference every 0.8 s (16 actions executed), at 50 Hz every 0.5 s (25
  actions); chunks executed open-loop "since temporal ensembling hurt performance" — [pi0 arXiv HTML](https://arxiv.org/html/2410.24164)
- **SmolVLA** ablations (LIBERO avg %): chunk size n=1 50.0, 10 84.0, 30 78.5, 50 80.3, 100 74.5; actions executed
  before a new observation: 1 -> 80.3, 10 -> 82.8, 30 -> 70.8, 50 -> 51.8 — [SmolVLA arXiv HTML](https://arxiv.org/html/2506.01844)
- **SmolVLA** async vs sync on SO-100: success 78.3% sync vs 73.3% async (pick-place 75/80, stacking 90/90, sorting
  70/50); pick-place completion 13.75 s vs 9.70 s (~30% faster); cubes in fixed time 9 vs 19 — [SmolVLA arXiv HTML](https://arxiv.org/html/2506.01844)
- **OpenVLA-OFT**: parallel decoding + chunking raises LIBERO average 76.5% -> 90.2%; chunk K=8 for LIBERO, K=25 for
  ALOHA (no chunk-size sweep in text) — [OFT arXiv HTML](https://arxiv.org/html/2502.19645)
- **Diffusion Policy**: an 8-step action horizon was optimal for most tasks; longer horizons react too slowly; with
  position control DP keeps peak performance with latency up to 4 steps — [DP arXiv HTML](https://arxiv.org/html/2303.04137)
- **Action-space study (2026)**: training chunk k=60 (2 s at 30 Hz), execution window swept 15-60; delta actions peak
  near 30 executed steps and degrade with long open-loop execution, absolute keeps improving to 60 then saturates
  (figure; values not in text) — [blog review (secondary)](https://mcobzarenco-fontaine-blog.static.hf.space/papers/action-space-design.html), [arXiv HTML v2](https://arxiv.org/html/2602.23408v2)
- **Why chunking works** (Lazzati, Stachowicz, Chen, Metelli, Wagenmaker, Levine, arXiv 2608.02547, 2026-08-03): gains
  come from non-Markovian expressivity, reduced compounding error and implicit ensembling; LIBERO-90 avg success:
  Markovian 68.9%, AC(10) 89.2%, Delay(6) 94.0%, AC(10) with randomized-delay execution 93.6%; at 50-60 Hz delay
  alone did not reproduce chunking, grouping 5 steps into a sub-chunk helped — [alphaXiv](https://www.alphaxiv.org/abs/2608.02547)
- **ResiP** (Ankile et al., 2024): chunked BC (DP, predict 32, execute 8) behaves "more like trajectory planners" than
  closed-loop controllers; it cannot correct mid-chunk; chunk-based methods lose 19-26 points under random force
  perturbations vs 12 for the per-step residual — [arXiv HTML](https://arxiv.org/html/2407.16677v4)
- **RTC** (Black, Galliker, Levine, arXiv 2506.07339, Jun 2025): inference-time inpainting of the next chunk with
  soft masking; real pi0.5 bimanual tests at 50 Hz, H=50, 5 denoising steps, model latency 76 ms (baselines) / 97 ms
  (RTC), delays d~6 (LAN), ~11 (+100 ms), ~16 (+200 ms); RTC had best throughput at every delay, "no degradation"
  with injected delay, 20% faster than synchronous on the same motion; **both temporal-ensembling variants could not
  run at +100/+200 ms because oscillations triggered the protective stop**; TE performed poorly even at zero delay
  in simulation; BID needed 64 sampled chunks and 2.3x RTC's latency; the largest final-score gain was on the most
  precision-sensitive task (light candle with a match); match lighting still worked with >300 ms delay; hard masking
  is jerkier than soft — [RTC arXiv HTML](https://arxiv.org/html/2506.07339)
- **Training-time RTC** (Black, Ren, Equi, Levine, arXiv 2512.05964, 2025-12-05): simulates inference delay during
  training and conditions on the action prefix, removing inference-time overhead; beats inference-time RTC at higher
  delays in sim; on pi0.6 (box building, espresso) matches inference-time RTC at lower compute; used in the pi*0.6
  espresso demo (search summary) — [arXiv](https://arxiv.org/abs/2512.05964), [PI RTC page](https://pi.website/research/real_time_chunking)
- "Action-Prior Denoising for Smooth Real-Time Chunking" (arXiv 2605.25537, May 2026) proposes Soft RTC, arguing the
  binary prefix mask is too rigid (search summary) — [arXiv](https://arxiv.org/pdf/2605.25537)
- **LeRobot RTC**: supported for pi0, pi0.5, SmolVLA; `execution_horizon` typical 8-12, `max_guidance_weight` 10.0
  "optimal" for 10 flow steps, `prefix_attention_schedule` EXP recommended to start; `inference_delay` passed at
  runtime; run with `lerobot-rollout --inference.type=rtc`; async inference (no idle frames) and RTC (no
  discontinuities) are complementary, "use both together" — [LeRobot RTC docs](https://huggingface.co/docs/lerobot/rtc)
- **Soare, "Smooth-as-butter robot policies"** (2025-08-05; pi0 on ALOHA, 50 Hz, chunk 50, 80-90 ms on a 4090, delay
  9 steps): set guidance weight beta about equal to the number of denoising steps; beta=14 with n=10 gave erratic
  motion, beta=10 safe; lowering sigma_d to 0.2 clearly reduced transition discrepancy; single task, few episodes —
  [blog](https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html)
- **UMI**: inference-latency matching raised dynamic tossing from 69/120 (57.5%) to 105/120 (87.5%) — [UMI arXiv](https://arxiv.org/abs/2402.10329)
- **X-VLA in LeRobot** defaults: `chunk_size=32`, `n_action_steps=32` (whole chunk executed), `n_obs_steps=1`,
  10 denoising steps; the paper's pretraining uses 30 anchor points covering the next 4 s — [LeRobot X-VLA docs](https://huggingface.co/docs/lerobot/xvla), [X-VLA arXiv HTML](https://arxiv.org/html/2510.10274)
- **EagleVLA / "Jetson-PI"** (arXiv 2607.12659, Oct 2026): naive asynchronous execution 0/10 on a real folding
  subtask vs 8/10 for their foresight-corrected async; claims VLASH and RTC degrade more as delay grows on LIBERO
  (+14.8 points over VLASH; no RTC numbers given) — [alphaXiv](https://www.alphaxiv.org/abs/2607.12659)

### Inferences
- Practical setting for a cheap arm: train chunks of ~1-2 s (e.g. 30-50 steps at 30 Hz), execute ~8-15 steps (or use
  RTC with execution_horizon ~10) so the final approach is re-observed several times per second. Executing whole
  chunks (X-VLA's LeRobot default, SmolVLA's 50/50) is the configuration with the worst measured results.
- Temporal ensembling is an ACT-specific tool (it is cheap for ACT, which runs at hundreds of Hz); for slow flow-matching
  VLAs it fights latency and caused oscillation in the RTC study — prefer RTC.
- Async inference costs some success when the observation used for a chunk is stale (SmolVLA -5 points); RTC or
  latency matching (UMI) is what recovers it.

### Gaps
- No chunk-length sweep measured in millimetres of final placement error.
- RTC results are on pi0.5 with 50 Hz high-end bimanual arms; no published RTC study on hobby servo arms (Soare's blog is ALOHA).
- SmolVLA's sync/async real-robot comparison is a single run on 3 tasks; variance unknown.

---

## 3. Action decoders: discrete tokens (FAST / FAST+) vs flow matching vs diffusion vs regression (L1)

### Takeaway
Once actions are continuous and chunked, the head matters less than people think: **L1 regression ties diffusion on
LIBERO (95.3 vs 95.4) at 26x the throughput (OpenVLA-OFT)**, while discrete per-step binning is clearly worse (90.2) and
naive binning fails at 20-50 Hz. Flow matching beat regression in SmolVLA (80.25 vs 75.25) and in the 2026
action-space study (88.0 vs 79.7 on joint-delta), plausibly because demonstrations are multimodal. FAST tokens train
~5x faster and follow language better but decode ~7.5x slower (750 ms vs 100 ms per chunk on a 4090), which is a
liability for closed-loop precision.

### Cited Findings
- **OpenVLA-OFT** (Kim, Finn, Liang, arXiv 2502.19645, Feb 2025), LIBERO avg: OpenVLA autoregressive 76.5; + parallel
  decoding & chunking (discrete) 90.2; + continuous diffusion 95.4; + continuous **L1 regression 95.3**; L1 + wrist
  + proprio 97.1 (filtered data) — [OFT arXiv HTML](https://arxiv.org/html/2502.19645)
- OFT throughput (A100): L1 109.7 Hz / 73 ms vs diffusion 50 test steps 4.2 Hz / 1.9 s, 10 steps 19.3 Hz (91.0% on
  LIBERO-Long), 5 steps 35.1 Hz (90.0), 2 steps 80.3 Hz (85.7), **1 step 0.0%** — [OFT arXiv HTML](https://arxiv.org/html/2502.19645)
- OFT on ALOHA throughput/latency: OFT+ 77.9 Hz / 321 ms; RDT-1B 84.1 / 297 ms; Diffusion Policy 267.4 / 90 ms; pi0
  291.6 / 86 ms; ACT 432.8 / 58 ms — [OFT arXiv HTML](https://arxiv.org/html/2502.19645)
- **SmolVLA**: flow matching 80.25 vs L1 regression 75.25 (LIBERO avg) — [SmolVLA arXiv HTML](https://arxiv.org/html/2506.01844)
- **Action-space study (2026)**: flow-matching head beats MSE-regression head in all four cells (e.g. joint-delta 88.0
  vs 79.7; joint-abs 79.6 vs 71.2) — [arXiv HTML v2](https://arxiv.org/html/2602.23408v2)
- **ACT**: L1 loss instead of L2 "leads to more precise modeling of the action sequence"; the CVAE objective is crucial
  with human data (35.3% -> 2% without it) but irrelevant with scripted data (59 vs 58) — [ACT paper](https://arxiv.org/abs/2304.13705)
- **FAST** (Pertsch et al., arXiv 2501.09747, Jan 2025): DCT + BPE; tokens per 1-s chunk: DROID 15 Hz 105 -> 29,
  T-shirt folding 50 Hz 700 -> 53; naive binning made no progress on Table Bussing (20 Hz) or T-shirt folding (50 Hz);
  removing BPE hurt but still beat naive; pi0-FAST comparable to diffusion pi0 on small datasets (LIBERO, T-shirt),
  ~3x fewer steps on Table Bussing, matched generalist pi0 with ~5x fewer GPU hours, followed language better on DROID;
  **inference ~750 ms per chunk vs ~100 ms for diffusion pi0 on an RTX 4090**, flagged as a limitation for dynamic
  tasks — [FAST arXiv HTML](https://arxiv.org/html/2501.09747)
- **FAST+** universal tokenizer: trained on ~1M one-second chunks from diverse robots; ~2x fewer tokens on unseen
  datasets with policy performance close to dataset-specific FAST; DCT scale 10, BPE vocab 1024 — [FAST arXiv HTML](https://arxiv.org/html/2501.09747)
- **Knowledge Insulation** (Driess et al., arXiv 2505.23705, May 2025): "naively including such [continuous diffusion /
  flow] experts significantly harms both training speed and knowledge transfer"; proposes insulating the VLM backbone
  during VLA training ("Train Fast, Run Fast, Generalize Better") — [arXiv abstract](https://arxiv.org/abs/2505.23705)
- **VLA-Adapter** (arXiv 2509.09372, Sep 2025): 0.5B Qwen2.5 backbone, L1-loss policy (~97M params, Bridge Attention),
  LIBERO 97.8/99.2/97.2/95.0 avg **97.3** vs OFT 7B 97.1, pi0 94.2, SmolVLA 88.8; 219.2 Hz / 36.5 ms vs OFT 71.4 Hz —
  [arXiv HTML v2](https://arxiv.org/html/2509.09372v2)
- **RoboVLMs** (arXiv 2412.14058): continuous action space + separate policy head identified as the most effective
  VLA formulation (secondary summaries; tables not checked) — [arXiv HTML v4](https://arxiv.org/html/2412.14058v4)
- **Diffusion Policy** CNN vs transformer (max/avg success): Push-T visual CNN 0.91/0.84 vs transformer 0.78/0.66;
  ToolHang visual 0.95/0.73 vs 0.76/0.47; DDIM 10 inference steps ~0.1 s on an RTX 3080 — [DP arXiv HTML](https://arxiv.org/html/2303.04137)
- **X-VLA** uses flow matching (OT path), MSE for pose, BCE for gripper — [X-VLA arXiv HTML](https://arxiv.org/html/2510.10274)
- **GR00T N1.6/N1.7** action head: flow-matching DiT, 32 layers in N1.6, 16 in N1.7 — [Isaac-GR00T README](https://github.com/NVIDIA/Isaac-GR00T)

### Inferences
- For precise positioning on a small dataset, continuous heads (flow matching or L1) are the safe default; discrete
  autoregressive tokens (pi0-FAST) buy training speed/language following, not precision, and their latency hurts closed
  loop.
- Flow-matching heads need several integration steps: OFT's diffusion collapses at 1 step; SmolVLA/pi0 use 10, RTC
  real tests used 5. Cutting steps for Orin speed should be validated, not assumed.
- ACT (L1 regression + CVAE) remains a strong low-latency baseline that fits on any Orin; in the SmolVLA SO-100 test it
  scored lower (48.3% single-task) than SmolVLA (78.3% multitask pretrained).

### Gaps
- No direct FAST-vs-flow comparison on a tight-tolerance task with per-task numbers (FAST paper per-task numbers are only in figures).
- No paper isolates decoder type from model size/pretraining on a low-cost arm.

---

## 4. Control rate and closed-loop behaviour: frequency, short chunks, visual-servoing-like final approach, precise tasks

### Takeaway
High command rates (30-50 Hz) with frequent re-observation are standard for mm-level tasks; the decisive factor is
**closing the loop visually during the final approach**, which lets policies beat the arm's open-loop accuracy (ALOHA's
ViperX is only 5-8 mm accurate yet ACT slots batteries at 96%). Where imitation alone still fails on tight insertions
(RAM 29%, peg-in-hole 5%), the published fixes are per-step closed-loop corrections: residual RL (ResiP), on-robot RL
with human corrections (HIL-SERL, RECAP), or a coarse waypoint + wrist-camera dense phase (SPHINX).

### Cited Findings
- **ALOHA/ACT**: ViperX 6-DoF arms with "an accuracy of 5-8mm", ~$5,600 each; recording/control at 50 Hz; tasks:
  Slide Ziploc 88%, Slot Battery 96%, Open Cup 84%, Thread Velcro 20%, Prep Tape 64%, Put On Shoe 92% (25 trials);
  the velcro loop is 3 mm x 25 mm, tie 2 mm thick — "even a few millimeters of error during the first grasp will
  compound ... giving more than a 10mm deviation in the insertion phase"; velcro failures: early gripper close and
  "insertion is not precise enough", attributed to low-contrast perception; sim Bimanual Insertion clearance ~5 mm,
  Transfer Cube clearance ~1 cm — [ACT paper](https://arxiv.org/abs/2304.13705)
- **ACT user study**: mm-precision tasks (thread zip tie, unstack cups) took 33 s at 5 Hz vs 20 s at 50 Hz, and 16 s vs
  10 s; 5 Hz is 62% slower, p<0.001 — [ACT paper](https://arxiv.org/abs/2304.13705)
- **pi0**: up to 50 Hz for dexterous robots, 20 Hz for UR5e/Franka — [pi0 arXiv HTML](https://arxiv.org/html/2410.24164)
- **Diffusion Policy**: predicts at 10 Hz, linearly interpolated to 125 Hz for execution (Push-T); real Push-T 95%
  success, IoU 0.80 vs human 0.84 — [DP arXiv HTML](https://arxiv.org/html/2303.04137)
- **RTC** precise real tasks (pi0.5, 50 Hz): light candle with a match (no retries allowed), plug ethernet cable;
  largest final-score gain on candle lighting — [RTC arXiv HTML](https://arxiv.org/html/2506.07339)
- **HIL-SERL** (Luo, Xu, Wu, Levine, arXiv 2410.21845, v3 Mar 2025): BC/HG-DAgger -> HIL-SERL success: RAM insertion
  29% -> 100% (cycle 8.3 -> 4.8 s, 1.5 h training), SSD assembly 79 -> 100, USB grasp-insertion 26 -> 100, cable
  clipping 95 -> 100, IKEA top panel 35 -> 100, car dashboard 41 -> 100, timing belt 2 -> 100; average 49.7% -> 100%,
  cycle 9.6 -> 5.4 s; plain BC 12% on RAM insertion; actions = 6-D Cartesian twist to an impedance controller +
  discrete gripper; images 128x128 from wrist/side cameras; proprio relative to the episode's initial EE pose —
  [arXiv HTML v3](https://arxiv.org/html/2410.21845v3)
- **ResiP** (FurnitureBench, 2024): DP base -> residual RL, low randomness: one_leg 54 -> 98%, round_table 12 -> 94,
  lamp 7 -> 97, peg-in-hole 5 -> 99, mug-rack 26 -> 88, biman-insert 33 -> 93; residual runs every control step
  (~0.1 s) while the base plans every ~1 s; scaling BC to 10,000 demos lifted round_table only to 56% — [arXiv HTML v4](https://arxiv.org/html/2407.16677v4)
- **SPHINX** (ICLR 2025, Stanford/PI): waypoint policy on point clouds for long moves, then a dense policy on
  **wrist-camera images** for precise manipulation around a predicted salient point; 20-60 demos (search summary;
  headline percentage not verified) — [arXiv](https://arxiv.org/abs/2412.05426)
- **Action-space study**: 30 Hz position control on the PiPER — [blog review (secondary)](https://mcobzarenco-fontaine-blog.static.hf.space/papers/action-space-design.html)

### Inferences
- The ALOHA result is the key existence proof for cheap arms: closed-loop visual feedback compensates for 5-8 mm arm
  inaccuracy on tasks with ~3-5 mm clearance, but success then drops multiplicatively per precise stage (velcro 92% ->
  20%).
- For our arms: command at the servo bus's practical rate (target >= 20-30 Hz), re-observe every ~0.3-0.5 s during the
  final approach (short executed horizon or RTC), and keep the wrist camera's view of the gripper tips and target in
  frame at the moment of grasp.
- A coarse-to-fine split (classical planner / detector + IK for the approach, learned wrist-cam policy only for the last
  few cm) is supported by SPHINX and matches the existing hand-eye pipeline; it shrinks the region the policy must
  learn.

### Gaps
- No published control-rate sweep for learned policy *execution* (as opposed to teleoperation) on hobby servos.
- None of the precise-task papers report final positional error in mm.

---

## 5. Observations: wrist vs third-person cameras, resolution, depth / point clouds, proprioception (incl. commanded-only joints), history

### Takeaway
Wrist cameras are consistently helpful for close-range precision (OFT +wrist+proprio 95.3 -> 97.1; UMI fisheye 55% ->
100% vs rectified crop; removing the overhead camera even improved generalization in the state-free study); wide FoV
matters. Proprioception is a double-edged sword: it is needed for some coordination (UMI inter-gripper 30% -> 70%) but
lets policies memorise trajectories and kills spatial generalization (0/30 -> 30/30 when removed, with relative EE
actions + wide wrist cams). Explicit 3D (point clouds) helps under height/viewpoint/scale changes (DP3 85% vs 35% RGB;
GeoVLA 45° viewpoint 70% vs pi0 30%) but needs metric depth. Resolution increases above ~224-512 show little measured
benefit. History frames: mixed evidence; most current VLAs use a single frame.

### Cited Findings
**Cameras**
- **OpenVLA-OFT**: adding wrist image + proprio: 95.3 -> 97.1 avg LIBERO (modified data); doubles visual patches 256 ->
  512 and lowers throughput 109.7 -> 71.4 Hz; wrist and proprio were not ablated separately — [OFT arXiv HTML](https://arxiv.org/html/2502.19645)
- **UMI**: fisheye wrist camera 20/20 vs rectified 69° crop 11/20 (Cup Arrangement); digitally reflected side-mirror
  views 20/20 vs no mirrors 18/20 vs raw mirrors 17/20 — [UMI arXiv](https://arxiv.org/abs/2402.10329)
- **State-free policies** (arXiv 2509.18644): removing the overhead camera improved generalization in harder scenarios
  (100 cm table height: 0 with overhead vs 1.0 without); setup = dual wide-angle (~120°x120°) wrist cameras — [arXiv HTML](https://arxiv.org/html/2509.18644)
- **Fisheye camera study** (arXiv 2603.02139, 2026-03-02, CVPR 2026): wide FoV "significantly enhances spatial
  localization" but the benefit depends on scene feature richness; fisheye policies overfit in simple scenes,
  generalize better with diverse data; naive cross-camera transfer fails (scale overfitting), Random Scale Augmentation
  fixes much of it (no numbers in abstract) — [arXiv](https://arxiv.org/abs/2603.02139)
- SynthICL (arXiv 2606.08154): removing the wrist camera lowered simulated success 75.0% -> 60.3%, attributed to lost
  close-up feedback near the gripper (search summary) — [arXiv](https://arxiv.org/pdf/2606.08154)
- **SPHINX** uses the wrist camera for the precise phase because it captures close-up detail and is agnostic to wider
  scene changes (search summary) — [arXiv](https://arxiv.org/abs/2412.05426)
- **ACT** Thread Velcro failures blamed on low contrast between the black tie and background — [ACT paper](https://arxiv.org/abs/2304.13705)

**Resolution / tokens**
- **SmolVLA**: 512x512 global image, pixel shuffle, 64 visual tokens per frame, no tiling — [SmolVLA arXiv HTML](https://arxiv.org/html/2506.01844)
- **HIL-SERL**: 128x128 crops with a ResNet-10 still reach 100% on RAM/SSD/USB insertion — [arXiv HTML v3](https://arxiv.org/html/2410.21845v3)
- SPEAR-1: 224x224 vs 280x210 on SIMPLER WidowX 57.25% vs 58.3% (within noise); VLM4VLA: raising 224 -> 512 -> 768 did
  not meaningfully help on CALVIN when the vision encoder was fine-tuned; FurnitureVLA raised 224 -> 448 "to keep fine
  contact cues" without an ablation (all search summary) — [SPEAR-1](https://arxiv.org/abs/2511.17411), [VLM4VLA](https://arxiv.org/abs/2601.03309), [FurnitureVLA](https://arxiv.org/abs/2607.01212)

**Depth / 3D**
- **DP3** (Ze et al., 2024) real robot, 40 demos, 10 trials x 4 tasks: DP3 85% avg vs image Diffusion Policy 35% vs
  depth-image DP 20% (Roll-Up 90/0/40, Dumpling 70/30/20, Drill 80/70/10, Pour 100/40/10) (search summary of Table
  VIII) — [arXiv](https://arxiv.org/abs/2403.03954)
- **GeoVLA** (arXiv 2508.09071, Aug 2025): depth -> point cloud -> point embedding network -> 3D-enhanced action expert;
  LIBERO avg 97.7 (vs OFT 95.3, pi0 94.2 as reported there); real in-domain avg 86.3% vs CogACT 76.3%, pi0 57.5%;
  viewpoint 0/15/30/45°: 90/90/80/70 vs pi0 50/50/40/30; height and scale perturbations similarly favour GeoVLA; depth
  from a RealSense D435i third-person at ~0.8 m; no RGB-only ablation — [arXiv HTML v2](https://arxiv.org/html/2508.09071v2)
- **PointVLA** (arXiv 2503.07511): injects point-cloud features into a frozen pretrained action expert; reports gains
  over OpenVLA, DP, DexVLA and adaptation to unseen table heights (search summary; no numbers checked) — [arXiv](https://arxiv.org/abs/2503.07511)
- X-VLA + Depth-Anything-3 feature ablations exist as HF checkpoints (DA3-XVLA) but only training losses are posted,
  no success rates and no paper — [HF repo](https://huggingface.co/JackLiu0406/DA3-XVLA-roboreal-ablations)

**Proprioception**
- **State-free policies** (Sep 2025): height generalization (pi0, Pick Pen) state-based 0/30 at 72 and 90 cm vs
  state-free 30/30 and 29/30; horizontal: Fold Shirt 0.183 -> 0.834, Fetch Bottle 0.117 -> 0.784; abstract: height 0%
  -> 85%, horizontal 6% -> 64%; in-domain parity (LIBERO 0.938 with vs 0.945 without); same direction for ACT and DP;
  caveats: sensitive to background changes, an idle arm's view shift can cause motion; precision not measured —
  [arXiv HTML](https://arxiv.org/html/2509.18644)
- **UMI**: inter-gripper proprioception 6/20 -> 14/20 on bimanual cloth folding — [UMI arXiv](https://arxiv.org/abs/2402.10329)
- **SmolVLA**: state token in the VLM prefix 80.3 vs in the action-expert suffix 73.3 (cross-attention variant) — [SmolVLA arXiv HTML](https://arxiv.org/html/2506.01844)
- **ACT**: actions = *leader* joint positions, observations = measured *follower* joint positions; "It is important to
  use the leader joint positions instead of the follower's, because the amount of force applied is implicitly defined
  by the difference between them, through the low-level PID controller" — [ACT paper](https://arxiv.org/abs/2304.13705)
- **X-VLA** (LeRobot): `use_proprio=True` by default; custom action modes may zero gripper channels in proprio — [LeRobot X-VLA docs](https://huggingface.co/docs/lerobot/xvla)
- Copycat problem: imitation from observation histories can underperform imitation from the latest observation alone
  because the network copies past actions (Chuang et al., ECCV 2022) — [arXiv](https://arxiv.org/abs/2207.09705)

**History**
- X-VLA in LeRobot conditions on a single observation step (`n_obs_steps=1`) — [LeRobot X-VLA docs](https://huggingface.co/docs/lerobot/xvla); SmolVLA feeds the current frame(s) as 64 tokens each plus one state token (no history described in the extracted text) — [SmolVLA arXiv HTML](https://arxiv.org/html/2506.01844)
- RoboVLMs reports that adding observation history helps, especially on long-horizon CALVIN tasks, when aggregated in a
  separate policy head (secondary summaries) — [arXiv](https://arxiv.org/abs/2412.14058)
- Chunking already supplies the non-Markovian context that history would: actions in LIBERO-90 were best predicted from
  an observation ~10 steps back — [alphaXiv](https://www.alphaxiv.org/abs/2608.02547)

### Inferences
- For a single arm with a wrist camera: keep the wrist camera as the primary view (wide FoV, sharp, high contrast near the
  fingers), add a fixed overview only if the task needs global context; randomize scale/crop if cameras change
  (fisheye study's RSA).
- **Commanded-only joints**: when the servos report only the last command, the "state" equals the previous action. Then
  (a) proprio carries no information about tracking error, load sag or contact (ACT's force-via-leader/follower gap is
  lost), and (b) the policy can copy its own previous output (copycat shortcut). Options supported by the evidence:
  drop the state input and rely on relative actions + wrist cameras (state-free paper), or keep state but use
  chunk-wise relative actions anchored to that commanded state, which remains a consistent reference. Measured joints
  (where the bus servo supports position readback) are preferable as the reference for relative actions.
- Monocular pseudo-depth (Depth Anything) as a policy input is untested in the literature found; GeoVLA/DP3 used
  metric depth cameras. Point-cloud benefits mostly show up as robustness to height/viewpoint/scale shifts, which a
  fixed table setup may not need.

### Gaps
- No paper found that studies commanded vs measured joint positions as proprio input.
- No clean 224-vs-448 ablation on a tight-tolerance task; no study of image resolution on the wrist view specifically.
- No study combining monocular estimated depth with a VLA on a real arm with success-rate results.

---

## 6. Precision-oriented work 2025-2026 (models and training tricks) and reported precision

### Takeaway
The biggest precision jumps reported recently do not come from bigger VLAs but from **closed-loop post-training on
the robot's own rollouts** (HIL-SERL ~100% on insertions; RECAP more than doubles throughput and halves failures;
PLD residual RL -> ~99% LIBERO / 100% on real Franka+YAM tasks incl. GPU insertion; ConRFT 96.3% after 45-90 min) and
from fixing the action interface (continuous L1/flow heads, parallel chunk decoding, chunk-wise relative actions).
LIBERO is saturated (97-99% for OFT, VLA-Adapter, X-VLA, GeoVLA) and no longer discriminates precision.

### Cited Findings
- **pi*0.6 / RECAP** (Physical Intelligence, 2025-11-17): RL with Experience & Corrections via Advantage-conditioned
  Policies — value function + advantage-conditioned policy trained on demos, teleoperator corrections and autonomous
  experience; built on pi0.6 (5B-parameter VLM backbone + action expert); "more than doubles the throughput on some of
  the hardest tasks, and can decrease failure rates by 2x or more"; espresso sub-stages >90% success, ran 5:30-23:30;
  59 boxes assembled in a factory — [PI blog](https://www.pi.website/blog/pistar06)
- **OpenVLA-OFT** recipe (parallel decoding, chunking, continuous L1, FiLM): LIBERO 76.5 -> 97.1; removing FiLM drops
  language following on ALOHA to 33% (chance) — [OFT arXiv HTML](https://arxiv.org/html/2502.19645)
- **VLA-Adapter** 0.5B: LIBERO 97.3 (Pro 98.5), CALVIN ABC->D 4.42 (Pro 4.50) vs OFT 4.10; trainable in ~8 h on one
  consumer GPU (abstract); real tests on a 6-DOF Synria Alicia-D (numbers only in a figure) — [arXiv HTML v2](https://arxiv.org/html/2509.09372v2), [abstract](https://arxiv.org/abs/2509.09372)
- **X-VLA-0.9B** (Oct 2025): LIBERO 98.1, Simpler-WidowX 95.8, CALVIN 4.43, RoboTwin-2.0 70.0/39.0; cloth folding ~100%
  success, ~33 folds/h after 1,200 demos; PEFT with ~9M params: LIBERO 93% — [X-VLA arXiv HTML](https://arxiv.org/html/2510.10274)
- **HIL-SERL**: average 49.7% (BC/HG-DAgger) -> 100% over 13 tasks, 1-2.5 h of training each (6 h timing belt) —
  [arXiv HTML v3](https://arxiv.org/html/2410.21845v3)
- **PLD — "Self-Improving VLA Models with Data Generation via Residual RL"** (Xiao et al., NVIDIA/CMU/Berkeley/UT
  Austin, arXiv 2511.00091, ICLR 2026): frozen VLA + lightweight residual actors trained off-policy, hybrid rollouts,
  distilled back by SFT (works for flow-matching and autoregressive heads); ~99% LIBERO, >50% gains on SimplerEnv,
  100% on real Franka and YAM tasks, 1-hour continuous GPU insertion/unplugging without resets (search summary of
  abstracts/project page) — [arXiv](https://arxiv.org/abs/2511.00091)
- **ConRFT** (arXiv 2502.05450, Feb 2025): offline BC + Q-learning, then online consistency-policy fine-tuning of a VLA
  with human interventions; 96.3% average on 8 real contact-rich tasks within 45-90 min online, +144% success over
  supervised fine-tuning, 1.9x shorter episodes — [arXiv](https://arxiv.org/abs/2502.05450)
- **ResiP**: residual RL on top of a frozen chunked DP: peg-in-hole 5% -> 99%, one_leg 54% -> 98% (low randomness) —
  [arXiv HTML v4](https://arxiv.org/html/2407.16677v4)
- **GeoVLA**: 3D-aware action expert, LIBERO 97.7, strongest under viewpoint/height/scale shifts — [arXiv HTML v2](https://arxiv.org/html/2508.09071v2)
- **FAST+**: universal tokenizer, ~2x token reduction on unseen robots — [FAST arXiv HTML](https://arxiv.org/html/2501.09747)
- **GR00T N1.6 -> N1.7**: state-relative chunks, then relative EEF actions, longer horizon (40), new Cosmos-Reason2-2B
  backbone — [Isaac-GR00T README](https://github.com/NVIDIA/Isaac-GR00T)

### Inferences
- For a cheap arm, the highest-leverage precision work after picking a sane action interface is **on-robot correction
  data**: interventions during autonomous rollouts (HG-DAgger / RECAP-style) or a small residual RL policy on top of a
  frozen base (ResiP, PLD). These target exactly the "last few millimetres" failure mode.
- LIBERO numbers should not be used to choose between SmolVLA/X-VLA/VLA-Adapter for precision; the real-world SO-100
  results and own task tests matter more.

### Gaps
- SpatialVLA (ego-3D position encoding, adaptive action grids) was not checked in this pass; no numbers here.
- No 2025-2026 VLA paper found that reports millimetre placement error on a real arm.
- PLD real-world details (tasks, trial counts) not verified from full text.

---

## 7. Small models for a Jetson Orin NX 16 GB; does model size matter for precision?

### Takeaway
Model size is not the main driver of precision: 0.45-0.9B models match or beat 3-7B models on LIBERO (SmolVLA 87.3 vs
pi0 86.0; VLA-Adapter 0.5B 97.3 vs OFT 7B 97.1) and on SO-100 (SmolVLA 78.3% vs pi0 61.7%), though very small models
lose some (SmolVLA 0.24B 82.75). On-board speed is the real constraint: published Orin numbers are ~0.35-0.8 s per
SmolVLA chunk and 0.17-1.4 s for 3B models, so on an Orin NX 16 GB any VLA needs async + RTC, careful (mixed-precision,
not naive INT4) quantization, or off-board serving from the 5090.

### Cited Findings
**Size vs performance**
- **SmolVLA**: LIBERO avg 0.24B 82.75, 0.45B 87.3, 2.25B 88.75; pi0 3.3B 86.0; OpenVLA 7B 76.5; Meta-World 0.24B 56.95,
  0.45B 57.3, 2.25B 68.24, pi0 47.9; SO-100 (pick-place/stack/sort): SmolVLA 0.45B multitask 75/90/70 = 78.3, pi0 3.5B
  100/40/45 = 61.7, ACT single-task 70/50/25 = 48.3; SO-101 Lego pick-place 90% ID / 50% OOD vs ACT 70/40;
  pretraining matters: no pretraining single-task 40 -> multitask 51.7 -> pretrained multitask 78.3 — [SmolVLA arXiv HTML](https://arxiv.org/html/2506.01844)
- **SmolVLA** architecture: SmolVLM-2 first 16 LLM layers, ~100M action expert, ~450M total; layer-skip ablation: 8
  layers 75.0, 16 78.5, 24 79.5, 32 80.3 — [SmolVLA arXiv HTML](https://arxiv.org/html/2506.01844)
- **VLA-Adapter 0.5B** ~ OFT 7B on LIBERO (97.3 vs 97.1); with a frozen backbone VLA-Adapter 86.4 on LIBERO-Long vs
  SmolVLA 77.0 vs OFT 0.0 — [arXiv HTML v2](https://arxiv.org/html/2509.09372v2)
- **OFT**: removing the OpenVLA pretraining costs 5.2 points (appendix, as reported in the text) — [OFT arXiv HTML](https://arxiv.org/html/2502.19645)
- **Action-space study** policies are a ResNet-18 + 6-layer transformer (no VLM) yet reach 88-96 single-arm progress
  with the right action space — [blog review (secondary)](https://mcobzarenco-fontaine-blog.static.hf.space/papers/action-space-design.html), [arXiv HTML v2](https://arxiv.org/html/2602.23408v2)

**Edge latency (published numbers)**
- **SmolVLA on Jetson Orin Nano 8 GB** (vla.cpp, arXiv 2606.08094, Jun 2026, C++ runtime for 11 VLAs): median 353-457 ms
  per inference on LIBERO-Object (search summary) — [arXiv PDF](https://arxiv.org/pdf/2606.08094), [abstract](https://arxiv.org/abs/2606.08094)
- **SmolVLA on Jetson AGX Orin**, SO-ARM101: 806.7 ms baseline, 488.5 ms with adaptive flow steps (AdaVLA, arXiv
  2608.29208), success 87.5% vs 90% (search summary) — [arXiv PDF](https://arxiv.org/pdf/2608.29208)
- **SmolVLA desktop**: 99.2 ms on an RTX 4090 in the LeRobot paper's table (A100 278.8 ms, MPS 721.8 ms, CPU ~2,028 ms)
  (search summary) — [LeRobot paper](https://arxiv.org/abs/2602.22818)
- **SmolVLA ONNX pitfall** (arXiv 2609.14146, Sep 2026, RTX 2060): "FP16/INT8" ONNX exports cut p99 latency 1181 ->
  601/532 ms but LIBERO-Spatial fell 70% -> 41/40%; the artifacts were actually FP32 graphs — the drop came from a
  static 16-token language width; width 24 recovered success (75%) at about half latency — [arXiv](https://arxiv.org/abs/2609.14146)
- **pi0.5 on Jetson Orin** (variant unspecified): naive PyTorch 1,420.8 ms (0.70 Hz); EagleVLA (CUDA graphs, flow
  unrolling, 40M future-correction module, confidence-based VLM skipping) 374.5 ms total, 6.93 Hz, reaction 144 ms;
  Jetson Thor 457.9 ms -> 202.8 ms — [alphaXiv](https://www.alphaxiv.org/abs/2607.12659)
- **pi0 desktop**: 73 ms on board (RTX 4090, 3 cameras: 14 ms image encoders, 32 ms observation pass, 27 ms for 10 flow
  steps), 86 ms off-board incl. 13 ms network — [pi0 arXiv HTML](https://arxiv.org/html/2410.24164)
- **openpi** memory: inference >8 GB, LoRA fine-tuning >22.5 GB, full fine-tuning >70 GB — [openpi README](https://github.com/Physical-Intelligence/openpi)
- **GR00T N1.6-3B on Orin**, 4 denoising steps: TensorRT 173 ms (5.8 Hz), torch.compile 199 ms, eager 300 ms; Thor
  TensorRT 92 ms (mirror of NVIDIA docs; search summary) — [mintlify mirror (secondary)](https://mintlify.wiki/NVIDIA/Isaac-GR00T/deployment/optimization)
- **GR00T** requirements: inference 16 GB+ VRAM (lists Jetson AGX Thor/Orin); fine-tuning 40 GB+ per GPU recommended
  (N1.7) — [Isaac-GR00T README](https://github.com/NVIDIA/Isaac-GR00T); Seeed's N1.6 SO-101 guide says fine-tuning needs
  >48 GB VRAM and deploys on AGX Orin 64 GB (search summary) — [Seeed wiki](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.6_for_lerobot_so_arm_and_deploy_on_agx_orin/);
  N1.7 on Thor with NVFP4: 125 -> 39.9 ms, "no measurable loss in task success" (search summary) — [Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/)

**Quantization**
- FoldQuantVLA (arXiv 2609.24433): GR00T N1.7 on Orin, ALOHA tasks, 20 trials/arm: uniform W4A4 80% vs mixed (selected
  layers INT8) 92.5% vs W8A8 / BF16 87.5%; the INT8 layers cost ~1 ms on Orin (search summary) — [arXiv](https://arxiv.org/pdf/2609.24433)
- VLAQuantBench (arXiv 2609.25376): pi0.5 W4A4 on certain layer groups dropped LIBERO-Spatial to 7.0%; W4 or W4A8 stayed
  within 1.5 points (search summary) — [arXiv](https://arxiv.org/pdf/2609.25376)
- HoloQ-VLA (arXiv 2605.28803): full-stack GPTQ W4A16 on pi0.5 -> 16.0% on LIBERO, keeping DiT attention unquantized ->
  ~81-83%; their W4A4 98.0 vs FP16 97.1 (search summary) — [arXiv](https://arxiv.org/pdf/2605.28803)
- OpenVLA on AGX Orin 64 GB (Jetson AI Lab): FP16 95.3% accuracy / 840 ms, FP8 95.2% / 471 ms, INT4 90.1% / 336 ms
  (search summary; "accuracy" is likely an offline action-match metric, not task success) — [Jetson AI Lab](https://jetson-ai-lab.com/openvla.html)
- vla.cpp: ternary BitVLA completed 200/200 LIBERO-Object episodes on an 8 GB Orin Nano; ternary kernel 4.0-4.6x faster
  than a CUDA-core baseline on RTX 3060 / AGX Orin — [arXiv abstract](https://arxiv.org/abs/2606.08094)

**X-VLA training notes**: fine-tune with VLM unfrozen at 1/10 LR, `policy.dtype=bfloat16` to avoid OOM, 20k steps
suggested for a new embodiment (3k in the bimanual SO-101 example) — [LeRobot X-VLA docs](https://huggingface.co/docs/lerobot/xvla)

### Inferences
- On an Orin NX 16 GB (a smaller module than the AGX Orin used in most of the numbers above), expect SmolVLA at roughly
  0.4-1 s per chunk without optimization and 3B models (pi0.5, GR00T) to be marginal; this is why chunked execution with
  async + RTC (or off-board serving) is needed. Off-board serving from hekate's RTX 5090 over the lab LAN should give
  ~100 ms-class chunk latency (pi0: 73 ms compute + 13 ms network on a 4090), but adds a network dependency.
- Fine-tuning fit: SmolVLA (~450M) and X-VLA (0.9B, bf16) fit on the RTX A4500 20 GB / RTX 5090 32 GB; openpi LoRA
  (>22.5 GB) fits the 5090 only; full pi0 (>70 GB) and GR00T N1.6/N1.7 full fine-tuning (>40-48 GB) fit neither.
- If quantizing, keep the action expert / DiT attention at FP16/INT8 and quantize the vision-language backbone; validate
  closed-loop on the real task (the ONNX paper shows export details can silently change behaviour).
- ACT is by far the cheapest candidate (432.8 Hz throughput / 58 ms on the ALOHA setup in the OFT comparison, vs 77.9 Hz
  / 321 ms for OFT+ — [OFT arXiv HTML](https://arxiv.org/html/2502.19645)), so it is the most likely to run at control
  rate on the Orin itself; it is a sensible precision baseline before investing in a VLA (its Orin NX latency was not found).

### Gaps
- No published SmolVLA, X-VLA, pi0.5 or GR00T latency measured specifically on a Jetson **Orin NX 16 GB**.
- No study shows whether model size changes *positional accuracy* (mm) as opposed to success rates.
- vla.cpp per-model latencies and quantization effects on success are not in the abstract; full paper not read.
