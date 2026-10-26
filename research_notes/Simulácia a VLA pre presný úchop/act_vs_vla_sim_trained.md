# ACT vs current VLAs for a low-cost arm trained mostly on simulated demonstrations (state of October 2026)

Scope: model choice, data scale and compute for the Yahboom M3 Pro (5 joints + gripper on bus servos with no joint
feedback, Orbbec DaBai DCW2 RGB-D on the wrist at 640x480, Jetson Orin NX 16 GB). Our baseline is LeRobot 0.6.1 ACT
(resnet18, chunk 20, 52M) on the robot at 10 Hz via ONNX -> TensorRT FP16 at 6 ms per inference. Action-space design,
DAgger and residual RL are in the earlier notes (`research_notes/Presný tréning VLA pre lacné rameno/`) and are not
repeated here. Edge latency numbers already in `policy_design_for_precision.md` section 7 are only referenced; the
numbers below are new or more primary.

Labels used: **[REAL-LC]** = measured on a real low-cost arm (SO-100/SO-101 class); **[REAL-other]** = real robot,
not low-cost; **[SIM]** = simulation benchmark; **[CLAIM]** = vendor/author claim without a measurement in the source;
**[2nd]** = secondary source / search summary, not verified against the primary.

## 1. Candidates as of October 2026: size, licence, LeRobot integration

### Takeaway
The realistic field is ACT (52M), SmolVLA (0.45B), X-VLA (0.9B), pi0 / pi0.5 (~3B, openpi + LeRobot ports) and GR00T N1.7
(3B, now the default GR00T in LeRobot 0.6). Newer flagships (pi0.6/pi*0.6/pi0.7, GR00T N2) have no open weights, and I
found no SmolVLA2. Diffusion Policy and OpenVLA-OFT are not good fits (slow, and DP failed to converge on a cheap-arm
task where ACT worked).

### Cited Findings
**ACT**
- 52M parameters; ~5 ms average inference on an RTX 4090 vs ~370 ms for Diffusion Policy (~263M) in the LeRobot
  paper's latency table (Feb 2026; search summary) [2nd] — [LeRobot paper, arXiv 2602.22818](https://arxiv.org/pdf/2602.22818)
- On a bimanual SO-101 (Jetson Orin Nano Super 8 GB, Aug 2026), ACT trained on identical demos reached 19/20 while
  Diffusion Policy produced no usable policy (0/10) even at twice the training steps; authors attribute it to convergence
  cost, not an accuracy ceiling [REAL-LC] — [arXiv 2608.03938 abstract](https://arxiv.org/abs/2608.03938)

**SmolVLA (Hugging Face, Jun 2025)**
- 450M parameters (~100M action expert), SmolVLM-2 backbone with only the first 16 LLM layers; chunk n = 50; pretrained on
  LeRobot community data (~30k GPU-hours total project compute); "designed to be trained on a single GPU" [CLAIM];
  ~40% faster to train and 6x less memory than pi0 — [SmolVLA arXiv 2506.01844](https://arxiv.org/html/2506.01844)
- No SmolVLA2 release found as of Oct 2026 (search returned only the 2025 model) — [TechCrunch Jun 2025](https://techcrunch.com/2025/06/04/hugging-face-says-its-new-robotics-model-is-so-efficient-it-can-run-on-a-macbook)

**pi0 / pi0-FAST / pi0.5 (Physical Intelligence, openpi)**
- openpi ships pi0, pi0-FAST and pi0.5 base checkpoints ("10k+ hours of robot data") plus pi0.5-DROID / pi0.5-LIBERO;
  pi0.5 released Sept 2025, only its flow-matching head is supported; code Apache-2.0, repo also carries
  `LICENSE_GEMMA.txt` (weights licence not stated on the README) — [openpi README](https://github.com/Physical-Intelligence/openpi)
- openpi PyTorch path: pi0 and pi0.5 supported; NOT supported in PyTorch: pi0-FAST, mixed precision, FSDP, **LoRA** —
  [openpi README](https://github.com/Physical-Intelligence/openpi)
- pi0.6 / pi*0.6: no released weights; pi0.7 (16 Apr 2026) is the flagship, offered only through a Pi API beta waitlist
  since 13 Oct 2026 [2nd] — [physicalailab.io, Oct 2026](https://physicalailab.io/en/physical-intelligence-pi-models-openpi-api/).
  "OpenPIE-0.6" on HF is a third-party reimplementation, not a PI release — [exla-ai/openpie-0.6](https://huggingface.co/exla-ai/openpie-0.6)
- pi0, pi0_fast, pi05, xvla, wall_x, groot and eo1 are all listed as LeRobot v0.6.0 policies in a compute guide (Jul 2026) [2nd] —
  [npaka LeRobot v0.6.0 GPU guide](https://note.com/npaka/n/naf1aec4456cf?hl=en)

**GR00T N1.6 / N1.7 (NVIDIA)**
- N1.7 is GA on `main`; 3B params; backbone Cosmos-Reason2-2B (Qwen3-VL) replacing Eagle; DiT action head 32 -> 16 layers;
  action horizon 40 (was 16 in N1.6); state/action dims up to 132; code Apache 2.0, weights "NVIDIA Open Model License"
  (README also says "fully commercially licensable under Apache 2.0" — inconsistent); pretraining adds 20k h of EgoScale
  human video; "comparable performance to N1.6, with improved generalization and language-following" [CLAIM] —
  [Isaac-GR00T README](https://github.com/NVIDIA/Isaac-GR00T)
- N1.7 early access at GTC Mar 2026; GA reported 13 Jul 2026 [2nd] — [HF blog](https://huggingface.co/blog/nvidia/gr00t-n1-7), [Robohorizon](https://robohorizon.com/en-us/news/2026/07/nvidia-open-sources-gr00t-17-a-free-brain-for-any-humanoid/)
- GR00T N2: preview only, no downloadable checkpoint as of Aug 2026 [2nd] — [physicalailab.io](https://physicalailab.io/en/nvidia-groot-n2-vs-n17/)
- LeRobot v0.6.0 (7 Jul 2026) adds GR00T N1.7 (parity-tested against Isaac-GR00T, flash-attention optional), drops N1.5
  (pin lerobot==0.5.1), and adds MolmoAct2, EO-1, EVO1, Multitask DiT plus world-model policies; RTC via `lerobot-rollout`;
  MolmoAct2 inference ~12 GB bf16, LoRA fine-tune fits one 24 GB GPU — [LeRobot v0.6.0 blog](https://huggingface.co/blog/lerobot-release-v060)

**X-VLA (Oct 2025)**
- 0.9B (Florence-Large VL encoder + 24-block transformer, flow matching, per-embodiment soft prompts); pretrained on 290k
  episodes (DROID, RoboMind, AgiBot); LoRA tunes ~9M params (~1%); LIBERO 98.1%, Simpler-WidowX ~96%, RoboTwin-2.0 51.1
  [SIM]; real Soft-Fold cloth folding ~100% / 33 folds per hour with 1,200 trajectories on AgileX [REAL-other];
  paper licence CC BY-NC-ND 4.0, model licence not stated; RGB + proprioception only — [X-VLA arXiv 2510.10274](https://arxiv.org/html/2510.10274v1)

**OpenVLA / OFT**
- OpenVLA (7B) 838 ms per policy call on AGX Orin 64 GB FP16 TensorRT (vs 101-231 ms for GR00T/pi0.5/SmolVLA in the same
  table) [SIM, LIBERO] — [EdgeVLA-TRT](https://github.com/hungho77/EdgeVLA-TRT)

### Inferences
- For a 16 GB Orin NX and a 20 GB training card, only ACT, SmolVLA and (borderline) X-VLA are "native" fits; pi0.5 and
  GR00T are 5090-or-offboard models (details in sections 3-4).
- Diffusion Policy is dominated here: slower than ACT by ~70x at inference and harder to converge at hobby-scale budgets.
- OpenVLA-OFT (7B) is out of scope for this hardware; nothing in 2026 suggests it beats the 0.5-3B models on low-cost arms.

### Gaps
- Exact licence of pi0.5 weights (Gemma terms?) and X-VLA weights not confirmed from primary sources.
- I could not open the LeRobot v0.6.1 release notes to confirm whether pi05/xvla are still first-class there (they are in the
  v0.6.0 guide [2nd]).
- OpenVLA-OFT fine-tuning memory not re-verified this session.

## 2. Reported results on low-cost arms: success vs demos, sim-only and sim+real

### Takeaway
On real SO-100/SO-101 arms with 50-100 teleop demos per task, ACT lands at roughly 33-80% depending on the task, SmolVLA
is equal or modestly better (32-100%), and pi0.5 is the best measured (56% avg vs ACT 34% on a hard 4-task SO-101
benchmark, mainly via recovery). No controlled real-robot number exists yet for a sim-only policy on SO-101 cube picking;
the one documented GR00T N1.6 sim-only-style SO-101 run got 40% end-to-end with the gap concentrated at the grasp. The
strongest evidence for our plan is sim+real co-training: +36-38 points over real-only with ~10k task-aware sim demos and
20-50 real demos (Diffusion Policy, not a VLA).

### Cited Findings
**Real low-cost-arm head-to-heads**
- SO-101, 4 tasks, 100 teleop demos per task, 20 trials per model-task (Hiroshima Univ., Jun 2026) [REAL-LC]:
  ACT 75/0/10/50 (avg 33.75%), SmolVLA 70/5/10/45 (32.5%), Wall-X 95/0/30/80 (51.25%), pi0.5 95/10/55/65 (56.25%) for
  Pen Transfer / Color Sorting / Object Packing / Pen Placement; recovery rate ACT 6.45%, SmolVLA 3.23%, Wall-X 20.51%,
  pi0.5 30.77%; "execution instability" dominant failure for all; no training compute or latency reported —
  [arXiv 2606.08881](https://arxiv.org/html/2606.08881)
- SO-100, 50 demos per task (10 per start position x 5) (HF, Jun 2025) [REAL-LC]: Pick-Place / Stacking / Sorting:
  ACT single-task 70/50/25 (48.3%), pi0 3.5B multi-task 100/40/45 (61.7%), SmolVLA multi-task 75/90/70 (78.3%);
  SO-101 Pick-Place-Lego: SmolVLA 90% ID / 50% OOD vs ACT 70% / 40% (author-run) — [SmolVLA arXiv](https://arxiv.org/html/2506.01844)
- SO-101 cube pick-and-place, community, Mar 2026 [REAL-LC, 5 trials each]: SmolVLA dual-cam 5/5, SmolVLA wrist-only 4/5,
  ACT dual-cam 4/5 on 75 episodes; 50 episodes over a 30 cm workspace failed while 75 over 10 cm worked; 75 consistent
  episodes beat 81 mixed-strategy ones; colour generalisation 1/3 (placing failed for non-red cubes) —
  [ggando blog](https://ggando.com/blog/smolvla-so101/)

**Sim-only / sim+real on SO-101 (mostly 2026 community + NVIDIA course)**
- GR00T N1.6 on a real SO-101 (vials-to-rack, ~75 expert demos, checkpoint-10000, 20 trials) [REAL-LC]: pick 75%, transport
  given pick 100%, placement given arrival 53.3%, end-to-end 40%; the same policy in sim: pick 95%, end-to-end 50%; "the
  largest sim-to-real gap is at acquisition"; an early camera-orientation mismatch badly degraded runs until fixed; the README
  is inconsistent on whether the data was sim or teleop — [nairb36/so101-sim2real](https://github.com/nairb36/so101-sim2real)
- NVIDIA "SO-101 sim-to-real" course: GR00T N1.6-3B post-training, 20k steps, global batch 64, lr 1e-4, action horizon 16,
  50-200 demos per task advised; a sim-only set of 75 demos gives 50-70% **sim** success; co-training module = 5 real + 70-100
  sim episodes; **no real-robot success numbers published** in the course pages —
  [GR00T module](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/10-groot.html),
  [co-training module](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/13-strategy2-cotraining.html),
  [real evaluation](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/12-real-evaluation.html),
  [sim evaluation](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/11-sim-evaluation.html)
- NVIDIA staff checkpoint (8 Oct 2026): GR00T N1.7 sim-only 250-episode DR policy (20k steps) + 10 real episodes (3,447
  frames) for 5k more steps, 4x RTX PRO 6000, batch 64, relative arm actions, vision trainable / language frozen; **no
  evaluation reported** — [HF model card](https://huggingface.co/liorbenhorin-nv/so101_sim250_GR00T17_real10_ft5000_20261008)
- Sim-only SO-101 GR00T N1.6 policy vs a real-data policy: 66.3% episode-level agreement (MCC 0.36), self-reported [2nd] —
  [HF 5hadytru](https://huggingface.co/5hadytru/so101_bench_GR00T_N1.6-3B_WM_sim_real_corr)

**ACT scaling with scripted sim demos (SO-101 in MuJoCo, Sep 2026) [SIM]**
- Scripted waypoint expert (97-100% success after fixes); demo sweep at 40k steps: 25 demos 10%, 50 -> 34%, 100 -> 44%,
  200 -> 54%, 300 -> 50%; 600 episodes + warm start -> 88% / 92% (two seeds, 50 trials each); chunk 100 executed fully
  beat re-planning every 25/50 steps and temporal ensembling; "curated" subsets halved success (diversity > quality
  score); randomised wrist-camera FOV/pose hurt fine alignment; remaining failures are grasp-approach precision; 40k
  steps batch 8 = ~52 min on an RTX A4000 — [aviadarn/so101-lerobot](https://github.com/aviadarn/so101-lerobot)

**Co-training evidence (not low-cost arm, but the recipe we would copy)**
- Sim-and-Real Co-Training (NVIDIA/UT Austin, RSS 2025), Diffusion Policy: real-only avg 45.3% -> real + task-aware sim
  ("digital cousin") 81.1% -> real + DC + prior 83.2% over 6 real tasks (Panda: 50 real + ~10k sim per task; GR-1: 20 real
  + 1k sim); sim 10k -> 500 demos dropped C2SPnP 67 -> 53%; misaligned cameras 67 -> 56% and 95 -> 70%; ratio matters
  (CupPnP: 50% sim suboptimal, 99% sim best at 95%, 99.5-99.9% fell to 60%); co-training still wins at 400 real demos
  [REAL-other] — [arXiv 2503.24361](https://arxiv.org/html/2503.24361v2); same result in NVIDIA's R2D2 blog (89% vs 39% at
  400 demos) — [NVIDIA blog](https://developer.nvidia.com/blog/r2d2-training-generalist-robots-with-nvidia-research-workflows-and-world-foundation-models/)
- Zero real demos: a Cosmos-Policy world-action model trained on ~800 sim demos per task reached 35% real success on a
  Franka vs Diffusion Policy 5% (10 real demos) / 25% (50 real demos); cost 40 H100 x ~72 h [REAL-other] —
  [arXiv 2606.31101](https://arxiv.org/html/2606.31101v1)
- ACT trained only in sim transferred zero-shot at 35.0% mean on dexterous tasks (not a cube pick, not SO-101) [2nd] —
  [DEXTERA arXiv 2609.21045](https://arxiv.org/pdf/2609.21045)

### Inferences
- Our 50-real-episode ACT result (coarse motion, never reaching the cube on new scenes) is typical: published working
  setups either cover a small workspace densely (75 eps / 10 cm) or use 100+ demos per task; 50 episodes over a whole
  table is below the knee for ACT.
- Data scale for the sim stage: ACT's sim curve flattens near 200-300 scripted demos per task layout and keeps improving
  to ~600 with warm start; co-training gains keep coming up to ~10k sim demos. Plan for thousands of sim episodes, a
  ~90/10 sim:real sampling ratio as the starting point, and sweep it (the co-training paper shows a narrow optimum).
- Expect the sim-to-real loss to show up at the grasp (pick 95 -> 75% in the only SO-101 measurement), i.e. exactly our
  precision problem; camera pose match between sim and the real wrist camera is the cheapest big lever.
- A VLA beats ACT clearly only on multi-task / recovery-heavy benchmarks (pi0.5 +22 points); on a single cube pick both
  reach 80-100% in small-workspace community tests, so the VLA's gain for our single task family is uncertain.

### Gaps
- No published, controlled real-robot success rate for a sim-only or sim+real **ACT** or **SmolVLA** policy on SO-100/101
  cube picking.
- NVIDIA has not published real SO-101 numbers for its own sim-to-real course (sim-only vs co-trained).
- Few trials everywhere (5-20 per condition); confidence intervals are wide.

## 3. Fine-tuning requirements on our GPUs (A4500 20 GB, RTX 5090 32 GB shared, RTX 2070 Super 8 GB)

### Takeaway
ACT trains anywhere (2-6 GB). SmolVLA fits the A4500 and 5090 (10-16 GB) but not the 8 GB laptop at sensible batch sizes.
pi0.5 needs >22.5 GB for JAX LoRA (5090 only; full fine-tune >70 GB fits neither). GR00T N1.6/N1.7 officially wants
40-48 GB; with the LLM frozen a run used ~25 GB, so the 5090 is plausible and the A4500 is not. X-VLA is grouped with
the 24-40 GB policies by the LeRobot guide; at 0.9B in bf16 it may fit the A4500 with small batches, but that is untested.

### Cited Findings
- LeRobot v0.6.0 peak VRAM by policy (Jul 2026) [2nd]: act/vqbet/tdmpc 2-6 GB; diffusion 8-14 GB; smolvla 10-16 GB;
  pi0/pi0_fast/pi05/xvla/wall_x 24-40 GB ("can sometimes run on 24 GB with tighter settings"); groot/eo1 24-40 GB.
  Times for ~50 episodes / 45k frames at 640x480: ACT 30-60 min on RTX 4090/3090; SmolVLA batch 4 3-6 h on L4/A10G,
  batch 16 1-2 h on A100 40GB; pi0/pi05 batch 4 4-8 h on A100 40GB; freezing SmolVLA's vision encoder saves memory —
  [npaka guide](https://note.com/npaka/n/naf1aec4456cf?hl=en)
- SmolVLA and ACT on an RTX 3090 24 GB, 75 episodes, 20k steps, batch 64 (Mar 2026) [MEASURED]: SmolVLA dual-cam ~10.4 h,
  wrist-only ~10 h, ACT ~10.5 h; ACT at 640x480 with batch 64 ran out of memory and was resized to 224x224 —
  [ggando blog](https://ggando.com/blog/smolvla-so101/)
- SmolVLA paper fine-tuning budget: 100k steps (batch 64) in sim, **200k steps** for real-world tasks; pretraining 200k steps,
  batch 256, 4 GPUs — [SmolVLA arXiv](https://arxiv.org/html/2506.01844)
- openpi: inference >8 GB; LoRA fine-tune >22.5 GB (example RTX 4090); full >70 GB (A100 80GB / H100); `fsdp_devices`
  can shard across GPUs; no multi-node — [openpi README](https://github.com/Physical-Intelligence/openpi). The 22.5 GB
  figure is for the JAX path; a PyTorch LoRA proposal reports ~16 GB peak on one 4090 at batch 16 for 30k steps (not
  merged / not official) [2nd] — [openpi issue #842](https://github.com/Physical-Intelligence/openpi/issues/842)
- LeRobot pi05 on 24 GB: "Partly" — gradient checkpointing + bf16, batch size 1 in one walkthrough [2nd] —
  [ay-robots pi0.5 guide](https://www.ay-robots.com/blog/en/how-to-train-pi0-5-on-your-own-robot-data)
- GR00T N1.7: fine-tuning "1 or more GPUs with 40 GB+ VRAM recommended" (H100/L40; A6000 works but slower); example
  global batch 32; inference 16 GB+ — [Isaac-GR00T README](https://github.com/NVIDIA/Isaac-GR00T). Seeed's N1.6 SO-101
  guide (Jan 2026): ">48 GB" for fine-tuning, trained on an L20, batch 32 —
  [Seeed wiki](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.6_for_lerobot_so_arm_and_deploy_on_agx_orin/)
- GR00T N1.7 SO-101 sim pick-place repo: `--no-tune-llm --tune-visual --tune-projector --tune-diffusion-model`, 6000 steps
  at batch 8 = ~3-4 h on one L40S 48 GB; "the HF blog run used about 25 GB" [2nd within the repo] —
  [ddiddi/so101-sim-pickplace-gr00t](https://github.com/ddiddi/so101-sim-pickplace-gr00t)
- NVIDIA course GR00T N1.6: 20k steps, batch 64, "several hours" / 2-8 h depending on dataset size, 24 GB+ for full model
  training (course wording) — [NVIDIA course](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/10-groot.html)
- X-VLA in LeRobot: VLM unfrozen at 1/10 LR, `policy.dtype=bfloat16` to avoid OOM, ~20k steps for a new embodiment (from the
  earlier notes) — [LeRobot X-VLA docs](https://huggingface.co/docs/lerobot/xvla)

### Inferences
- Rough budgets for ~1-5k sim episodes (~100-300k frames at 10 Hz) on our cards (my estimates from the step rates above,
  not measured): ACT 50-100k steps = a few hours on the A4500 or 5090, and it even fits the 2070 Super; SmolVLA 20-50k
  steps at batch 32-64 = ~10-25 h on the A4500 (3090-class), roughly half on the 5090; pi0.5 LoRA 20-30k steps = 5090
  only, ~1 day, and blocks the shared card; GR00T N1.7 with frozen LLM, batch 8-16 = 5090 only, ~4-10 h per 5-10k steps.
  Training cost scales with steps x batch, not with episode count, so thousands of sim episodes mostly cost disk and
  data-loading, not GPU-hours.
- Because the 5090 is shared with another project (rule: ask before GPU-heavy work when <16 GB free), pi0.5/GR00T runs
  need explicit scheduling; ACT and SmolVLA can run on Dionysos' A4500 without touching hekate.
- Video decoding / image resizing at 640x480 dominates small-model training time (ACT took as long as SmolVLA in the
  3090 case); pre-resizing the sim dataset to 224-256 px is a cheap speed-up.

### Gaps
- No measured peak VRAM for GR00T N1.7 with `--no-tune-llm` at a given batch on a 32 GB card, and none for LeRobot pi05
  LoRA on 32 GB.
- No measured SmolVLA or X-VLA training throughput on an A4500 or 5090.

## 4. Inference: Jetson Orin NX 16 GB vs an offboard GPU; async inference and real-time chunking

### Takeaway
ACT is the only candidate that runs far faster than our 10 Hz control on the Orin NX (6 ms measured by us). SmolVLA and
X-VLA should be ~100-250 ms per chunk on an Orin NX with TensorRT/ONNX, i.e. usable with 50-action chunks and async
execution. pi0.5 and GR00T N1.7 measure 100-230 ms on the larger AGX Orin and need 8-16 GB of memory, so on a 16 GB
Orin NX that also runs the camera/ROS/bridge stack they are marginal; serving them from the 5090 (~25 ms compute) over
the lab network is the practical route. In LeRobot 0.6.1, RTC works only in the local `lerobot-rollout` loop; the async
gRPC server reportedly does not pass RTC inputs.

### Cited Findings
**Official GR00T N1.7 latency table (batch 1, horizon 40, 4 denoising steps, two 256x256 cameras, excluding RPC)** —
[Isaac-GR00T deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md):
- Jetson Orin (MAXN): eager 357.9 ms, torch.compile 224.3 ms, TensorRT 149.0 ms
- AGX Thor (MAXN): 106.8 / 100.5 / 76.8 ms, TensorRT + mixed NVFP4 35.6 ms
- RTX Pro 6000 Blackwell: 75.7 / 52.1 / 24.7 ms; H100: 22.9 ms TRT; L40: 35.0 ms TRT; DGX Spark 116.6 ms TRT
- README: "~40 Hz on RTX PRO 6000 ... ~28 Hz on Jetson AGX Thor with TensorRT"; inference needs 16 GB+ —
  [Isaac-GR00T README](https://github.com/NVIDIA/Isaac-GR00T)

**AGX Orin 64 GB, JetPack 6.2, TensorRT 10.3 FP16, median policy call (EdgeVLA-TRT, 2026) [SIM-LIBERO success]** —
[EdgeVLA-TRT](https://github.com/hungho77/EdgeVLA-TRT):
- GR00T N1.7 101 ms (97%), N1.6 120 ms (100%), N1.5 196 ms; pi0.5 231 ms (100%); SmolVLA 122 ms (72% vs 90% reported —
  port lost success); X-VLA 198 ms (98%); Evo-1 248 ms; MolmoAct2 413 ms; OpenVLA 838 ms; TurboVLA 32 ms

**Other Jetson numbers**
- SmolVLA FP16 PyTorch, batch 1, two 512x512 cameras: AGX Orin 67.4 ms (14.8 Hz), Xavier NX 312 ms [2nd] —
  [ApxInf PR #110](https://github.com/infinigence/ApxInf/pull/110); Orin Nano 8 GB split-ONNX FP16 189.9 ms (5.25 Hz),
  ~7 Hz with one camera, vs 1168 ms PyTorch FP32 [2nd] — [jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla)
- pi0.5 on Jetson Orin (Jetson-PI, Peking Univ., 14 Jul 2026): naive PyTorch 1420.8 ms (0.70 Hz) -> 412.9 ms total, 165 ms
  reaction, 6.06 Hz with CUDA-graph reuse + flow unrolling + a 40M "foresight" module (no TensorRT, no quantisation); Thor
  457.9 -> 309.5 ms; real robot (cloth folding, 15 Hz execution): RTX 4090 10/10, 7/10, 10/10 vs naive async on Orin 6/10,
  0/10, 5/10 vs Jetson-PI 10/10, 8/10, 9/10 [REAL-other] — [arXiv 2607.12659v1](https://arxiv.org/html/2607.12659v1)
  (the earlier notes quote 374.5 ms / 6.93 Hz from an alphaXiv summary of the same paper — versions differ)
- ACT on a Jetson Orin Nano Super 8 GB (bimanual SO-101, three cameras): 114.0 ms unconverted -> 17.9 ms TensorRT FP16 ->
  12.7 ms INT8 with 18-19/20 success preserved; INT8 calibration quantised only the ResNet18, none of the transformer
  layers [REAL-LC] — [arXiv 2608.03938](https://arxiv.org/abs/2608.03938)
- Orin NX 16 GB: 1024-core Ampere GPU with 32 Tensor Cores; 100 TOPS, 157 TOPS in JetPack 6.2 "Super"/MAXN SUPER mode
  (GPU clock 918 -> 1173 MHz) — [NVIDIA JetPack 6.2 blog](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules),
  [Connect Tech](https://connecttech.com/jetson-orin-nx-nano-super-mode/)

**Async inference / RTC in LeRobot 0.6.1 (blog dated 8 Oct 2026, claims not verified against the repo)** —
[particula.tech](https://particula.tech/blog/real-time-chunking-vs-async-inference-lerobot):
- Async = separate policy server over gRPC (insecure port, default localhost:8080); client merges overlapping chunks
  (`weighted_average` 0.3/0.7 default); inference memory ~14 GB for pi0, ~2 GB for SmolVLA (per the async docs)
- In v0.6.1 the async server calls `predict_action_chunk(observation)` only, so RTC's inference delay and previous chunk
  never reach the policy; PR 3112 "Distributed Real-Time Chunking" closed unmerged 2026-09-26; training-time RTC for pi0.5
  (PR 4056) merged 2026-08-21, after v0.6.1
- Local RTC: `lerobot-rollout --inference.type=rtc`, 30 fps loop, pi0/pi0.5/SmolVLA return 50-action chunks, ~20 executed
  per chunk; `inference_delay = ceil(max latency x fps)`; one slow call raises the delay until restart
- RTC paper (RTX 4090): 108.8 ms end-to-end with a 6.9 ms network hop (wired), 139.0 ms with a 21.2 ms hop (mobile); RTC
  adds ~21 ms over plain pi0.5 (97 vs 76 ms)
- LeRobot RTC docs cover pi0, pi0.5 and SmolVLA (flow-matching policies) — [LeRobot RTC docs](https://huggingface.co/docs/lerobot/en/rtc)
- SmolVLA async vs sync on SO-100: success 73.3% vs 78.3%, task time 9.7 s vs 13.75 s (~30% faster), 19 vs 9 cubes in fixed
  time [REAL-LC] — [SmolVLA arXiv](https://arxiv.org/html/2506.01844)
- Offboard example: GR00T N1.6 served from a Brev cloud GPU to a real SO-101 over ZMQ: round trip mean ~39.7 ms, P99 ~42 ms,
  one spike ~159 ms; latency not shown to cause the grasp failures [REAL-LC] — [nairb36/so101-sim2real](https://github.com/nairb36/so101-sim2real)

### Inferences
- The "Jetson Orin (MAXN)" rows above are most likely the AGX Orin; an Orin NX 16 GB has half the GPU cores (1024 vs 2048 on
  AGX Orin 64 GB — the AGX figure is from my knowledge, not verified this session) and about half the memory bandwidth, so
  expect roughly 1.5-2x the AGX latencies: SmolVLA ~180-250 ms, GR00T N1.7 ~150-300 ms, X-VLA ~300-400 ms, pi0.5
  ~350-450 ms per chunk with TensorRT. With 10 Hz control and 20-50-action chunks this is workable with async execution, but GR00T's "16 GB+" and
  pi0.5's ">8 GB" leave little of the Orin NX's shared 16 GB for the rest of the robot stack.
- Offboard over Wi-Fi: 5090 compute ~25 ms (GR00T TRT class) + Wi-Fi round trip 20-40 ms typical with 100-200 ms spikes ->
  ~50-70 ms per chunk, comfortably below one 10 Hz chunk period; chunked execution hides spikes, but the robot then depends
  on the network (and on hekate's GPU being free).
- Our own robot bridge already runs a 10 Hz policy job with TensorRT beside the arm; implementing RTC-style prefix
  conditioning ourselves is simpler than relying on LeRobot's async server, which (if the blog is right) cannot do RTC in
  0.6.1.
- Watch export fidelity: the EdgeVLA SmolVLA port lost 18 points on LIBERO, and the earlier notes record an ONNX export
  that silently changed SmolVLA's behaviour; validate any TensorRT VLA closed-loop, not by latency alone.

### Gaps
- No direct measurement of any VLA on a Jetson **Orin NX 16 GB** (all numbers are AGX Orin, Orin Nano, Thor or desktop).
- No measured Wi-Fi (as opposed to wired / internet) latency study for chunked VLA serving on a mobile manipulator.
- The LeRobot async/RTC limitation comes from one blog; not confirmed against the v0.6.1 source.

## 5. Precision: does a VLA grasp more precisely than ACT at the same data?

### Takeaway
There is no millimetre-level comparison of ACT vs any VLA on a low-cost arm. The available evidence says VLAs win on
recovery, multi-task and generalisation, not on raw positioning: in the SO-101 benchmark "precision misalignment" is a
minor share of failures for every model (ACT 15%, pi0.5 14%, SmolVLA 7%), while grasp instability (>90% of failed
trials) dominates for all, VLAs included.

### Cited Findings
- SO-101 failure analysis (share of failed trials, Jun 2026) [REAL-LC]: grasp instability ACT 94.3%, SmolVLA 92.6%, Wall-X
  100%, pi0.5 91.4%; precision misalignment ACT 15.1%, SmolVLA 7.4%, Wall-X 0%, pi0.5 14.3%; state mismatch ACT 98.1% vs
  pi0.5 45.7%; on the precision task (Pen Placement) ACT 50%, SmolVLA 45%, Wall-X 80%, pi0.5 65% —
  [arXiv 2606.08881](https://arxiv.org/html/2606.08881)
- In scripted-sim ACT on SO-101 the remaining failures after 600 demos are "grasp-approach precision", and placement success
  is flat from 50 demos on, so more of the same data has limited value [SIM] — [aviadarn/so101-lerobot](https://github.com/aviadarn/so101-lerobot)
- Community SO-101 cube test: failures were off-centre grasps where the cube slips and slight misjudgement of the bowl
  position, for both SmolVLA and ACT [REAL-LC] — [ggando blog](https://ggando.com/blog/smolvla-so101/)
- A thesis on low-cost hardware reports ACT keeps temporal coherence while Diffusion Policies achieve better precision on
  fine-grained tasks [2nd, search summary] — [FIT thesis](https://repository.fit.edu/etd/1539)
- Cross-arm comparisons on SO-101 are confounded by per-arm servo offset differences [2nd] —
  [VLA-REPLICA arXiv 2605.20774](https://arxiv.org/pdf/2605.20774)

### Inferences
- For a 30/40 mm cube in a gripper, the binding constraint is the grasp approach and the servo/hand-eye error (covered in the
  earlier notes), not the policy class; a VLA should not be expected to fix a 1-2 cm reach error that ACT has. Pretrained
  VLAs mainly add robustness to new scenes, recovery after a slip, and language.
- Model size is not the precision lever (earlier notes: 0.45-0.9B models match 3-7B on LIBERO); the lever is closed-loop
  replanning near the object plus data coverage of the final approach.

### Gaps
- No source reports grasp-position or placement error in mm for ACT vs a VLA on hobby servos.
- No study isolates "same data, same arm, single precise pick" ACT vs SmolVLA/pi0.5 with enough trials to separate them.

## 6. Depth input (RGB-D): which models take it, and does it help grasping?

### Takeaway
None of the mainstream candidates takes depth natively: X-VLA, GR00T N1.7, SmolVLA and openpi pi0/pi0.5 are RGB +
proprioception (+ language). Depth helps when injected as a separate 3D branch (point clouds / pointmaps): third-party
real-robot results show RGB-only pi0.5 at 37.5% vs PointVLA 46.3% vs a pointmap method 70.0%. For us depth is a likely
precision win, but it means modifying the model (easy for ACT, invasive for a 3B VLA).

### Cited Findings
- X-VLA inputs: multi-view RGB, language, proprioception; depth not mentioned — [X-VLA arXiv](https://arxiv.org/html/2510.10274v1)
- GR00T N1.7 README does not mention depth — [Isaac-GR00T README](https://github.com/NVIDIA/Isaac-GR00T)
- SeeR-VLA (Jul 2026) real-world table: RGB-only pi0.5 avg 37.5%, PointVLA 46.3%, robot-centric pointmaps 70.0%; peg-in-hole
  pi0.5 20% vs PointVLA 35% [REAL-other, 2nd] — [arXiv 2607.11498](https://arxiv.org/pdf/2607.11498)
- GeoVLA (depth -> point cloud, separate encoder): 86.3% avg over 8 real tasks, +28.8 points over pi0 [REAL-other, 2nd] —
  [arXiv 2508.09071](https://arxiv.org/html/2508.09071v2)
- QDepth-VLA uses depth only as auxiliary supervision (predict quantised depth), Piper arm, 50 demos per task —
  [arXiv 2510.14836](https://arxiv.org/pdf/2510.14836)
- A vendor fork of LeRobot (Lejurobot Kuavo docs) documents depth input for PI0/PI0.5 via a separate ViT or the PaliGemma
  backbone fused by concatenation or cross-attention (configuration reference, not a benchmark, not upstream LeRobot) —
  [Lejurobot depth support](https://www.kdc.docs.lejurobot.com/advanced/depth_support/)

### Inferences
- ACT's ResNet image inputs make depth a cheap experiment (an extra image stream or a 4th channel from the DCW2), and Isaac
  Sim renders perfect depth for the sim demos; the real DaBai DCW2 depth at 20-30 cm will be noisier than sim (and our
  earlier work found monocular metric depth unreliable at that range), so add depth noise/holes in sim if we try it.
- For a VLA, adding depth means retraining the input path on our data and losing part of the pretraining benefit; not a
  first step.

### Gaps
- No depth-vs-RGB ablation on a low-cost arm, and none for ACT specifically.
- No upstream LeRobot policy with native depth found.

## 7. Recommendation for our case (M3 Pro, Isaac Sim expert, Orin NX, October 2026)

### Takeaway
Do both, in order: **ACT first** on thousands of sim demos (+ a few dozen real, co-trained ~90/10), because it trains in
hours on the A4500, runs at 6 ms on the Orin NX, and the main risk (grasp precision and sim-to-real at acquisition) is
the same for every model. **VLA second**: SmolVLA if it must run on the robot and train on the A4500; pi0.5 (best
measured on SO-101) or GR00T N1.7 (best Isaac Sim / SO-101 sim-to-real tooling, LeRobot `groot`) if served offboard
from the 5090, chosen when language commands and multi-object generalisation become the goal.

### Cited Findings
- ACT and SmolVLA are close on real SO-101 (33.75 vs 32.5% avg; 4/5 vs 5/5 on a cube test); pi0.5 is ahead on a hard
  multi-task benchmark (56.25%) mainly through recovery (30.8% vs 6.5%) —
  [arXiv 2606.08881](https://arxiv.org/html/2606.08881), [ggando blog](https://ggando.com/blog/smolvla-so101/)
- SmolVLA beats pi0 on SO-100 multi-task (78.3 vs 61.7%) at 1/7 the size — [SmolVLA arXiv](https://arxiv.org/html/2506.01844)
- Co-training with task-aware sim demos adds ~36-38 points over real-only — [arXiv 2503.24361](https://arxiv.org/html/2503.24361v2)
- NVIDIA's SO-101 sim-to-real course and staff checkpoints already follow "sim-only GR00T -> +5-10 real episodes" —
  [NVIDIA course](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/13-strategy2-cotraining.html),
  [HF model card](https://huggingface.co/liorbenhorin-nv/so101_sim250_GR00T17_real10_ft5000_20261008)
- GR00T N1.7 TensorRT: 24.7 ms on an RTX Pro 6000, 149 ms on Jetson Orin; needs 16 GB+ for inference and 40 GB+ recommended
  for fine-tuning — [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), [deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md)

### Inferences
**Comparison for our constraints (estimates marked ~ are mine)**

| | ACT | SmolVLA | X-VLA | pi0.5 | GR00T N1.7 |
|---|---|---|---|---|---|
| Params | 52M | 450M | 0.9B | ~3B | 3B |
| Train VRAM | 2-6 GB (fits 2070S) | 10-16 GB (A4500) | 24-40 GB per guide; A4500 only with small batch (~untested) | LoRA >22.5 GB (5090 only) | 40-48 GB rec.; ~25 GB LLM frozen (5090 only) |
| Train time, ~50k steps | ~hours | ~10-25 h A4500 | ~ similar to SmolVLA x2 | ~1 day 5090 | ~4-10 h per 5-10k steps on 5090 |
| Orin NX latency | 6 ms (ours) | ~180-250 ms TRT | ~300-400 ms | ~350-450 ms, memory-tight | ~150-300 ms, but 16 GB+ = too tight |
| Offboard 5090 | not needed | ~ <50 ms | ~ <60 ms | ~70-100 ms | ~25 ms TRT |
| Language | none (needs a target input) | yes (English VLM) | yes | yes, best open-world | yes (Qwen3-VL, multilingual base) |
| Depth | easy to add | modify | modify | modify (fork exists) | modify |
| Real low-cost evidence | many, 33-80% | SO-100/101, 32-100% | none found | SO-101 best 56% | SO-101 40% e2e (one run) |
| Licence | Apache (LeRobot) | Apache (LeRobot) | unclear | code Apache, weights unclear | NVIDIA Open Model Licence |

- **Stage A (now): sim-trained ACT.** Thousands of scripted Isaac Sim episodes (diversity of cube pose/colour/lighting over
  the whole reachable table, camera pose matched to the real wrist camera), train on the A4500, select checkpoints by sim
  rollout success, then co-train or fine-tune with 30-50 real episodes. Expect the ACT result to tell us whether the
  precision problem is data coverage (sim fixes it) or arm/hand-eye error (no policy fixes it). Language at this stage:
  keep the existing Slovak -> Qwen -> JSON function path and give ACT the chosen target (detector box/mask as an extra input
  or a cropped view) rather than text.
- **Stage B: one VLA on the same sim+real dataset.**
  - If on-board at 10 Hz is a hard requirement: **SmolVLA** (LeRobot-native, RTC support, fits A4500, ~2 GB inference),
    with TensorRT export validated closed-loop. Feed it English commands produced by our local LLM from Slovak speech
    (SmolVLM2's language model is English-centric; this is an inference, not tested).
  - If offboard serving from hekate's 5090 over the lab Wi-Fi is acceptable: **pi0.5** has the best measured low-cost-arm
    result and recovery; **GR00T N1.7** has the NVIDIA SO-101 sim-to-real recipe (sim-only -> +10 real), LeRobot `groot`,
    the fastest TensorRT path and a Qwen3-VL backbone. Given our Isaac Sim pipeline, GR00T N1.7 is the more natural
    second step; pi0.5 is the stronger-evidence alternative. Both need the 5090 for training (schedule around the other
    project) and fail the "runs on the Orin NX" test.
  - X-VLA is attractive on paper (0.9B, LIBERO 98%) but has no low-cost-arm evidence and an unclear licence; keep as a
    fallback.
- Run every candidate on the identical sim + real split and the same real test protocol (>= 20 trials per condition), since
  published differences between ACT and SmolVLA are within a few trials.

### Gaps
- No published real result for any sim-trained model on a mobile manipulator with a wrist RGB-D camera and commanded-only
  joint state like ours; the stage-A experiment is itself the missing data point.
- No source compares Slovak vs English instruction following for any VLA.
