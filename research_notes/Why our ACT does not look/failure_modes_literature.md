# Failure modes that fit "goes one way wherever the cube is": what the literature measured (as of October 2026)

Scope: the published mechanisms behind a visuomotor policy that ignores where the object is -- copycat / causal
confusion from proprioception, proprioception suppressing vision during the target-dependent phases, memorised
trajectories, ACT's CVAE latent, an ambiguous target among several pieces, and a small image encoder trained without
augmentation -- each matched against what we measured on our own data (`our_data_diagnosis.md`). Action-space choices
(absolute vs relative, joint vs end effector) are in `research_notes/Presný tréning VLA pre lacné rameno/
policy_design_for_precision.md` section 1 and are not repeated. Web searches ran out partway (a shared budget); the
later checks were direct fetches of arXiv abstract and HTML pages.

Labels: **[REAL-LC]** real low-cost arm; **[REAL-other]** other real robot; **[SIM]** simulation; **[CLAIM]** claim
without a measurement in the source; **[2nd]** secondary source, not checked against the primary. **[OURS]** marks our
own measurements from `our_data_diagnosis.md` where they are set beside a paper's.

## 1. Copycat and causal confusion: the previous action as an input

### Takeaway
When the input contains information about the expert's previous action and consecutive actions are strongly
correlated, behaviour cloning learns to repeat the previous action, and the error appears exactly where the action
changes -- at a decision. That is our setup in its purest form: `observation.state(t) == action(t-1)` [OURS], and the
decision (which way to leave the look pose) lives in ~2 % of the frames [OURS]. Octo's authors saw proprioception make
policies "generally worse" and suspected the same mechanism.

### Evidence
- de Haan, Jayaraman, Levine, "Causal Confusion in Imitation Learning", NeurIPS 2019: access to more information can
  make a cloned policy worse ("causal misidentification"); the problem is common "especially when the imitator's inputs
  include history" (abstract; the brake-light example) [SIM] -- [arXiv 1905.11979](https://arxiv.org/abs/1905.11979)
- Wen, Lin, Darrell, Jayaraman, Gao, "Fighting Copycat Agents in Behavioral Cloning from Observation Histories",
  NeurIPS 2020: with temporally correlated expert actions and partial observations, the imitator copies the previous
  action; distribution shift makes it worse; fix: an adversarially learnt representation that removes previous-action
  information (abstract, no numbers there) [SIM] -- [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/1b113258af3968aaf3969ca67e744ff8-Abstract.html)
- Wen et al., "Keyframe-Focused Visual Imitation Learning", ICML 2021: a history-only "copycat" MLP marks the samples
  it predicts worst (expert changepoints); upweighting the top 10 % by x5 lifted CARLA success 33.0 -> 43.4 % over BC
  with history (BC without history 42.7 %), and image-MuJoCo returns by ~25-33 % (e.g. HalfCheetah 615 -> 819) [SIM]
  -- [ar5iv 2106.06452](https://ar5iv.labs.arxiv.org/html/2106.06452)
- Octo, Appendix E ("Things that did not work (yet)"): "Policies trained with propioceptive observations seemed
  generally worse ... We hypothesize this might be due to a causal confusion between the proprioceptive information and
  the target actions" [CLAIM: no numbers] -- [arXiv 2405.12213 HTML](https://arxiv.org/html/2405.12213v1)
- robomimic (Mandlekar et al. 2021): adding end-effector velocity and joint information to low-dimensional observations
  cost 49-88 % relative success; image agents were more tolerant (2-29 % drops); the authors suggest policies overfit to
  unneeded inputs [SIM] -- [arXiv 2108.03298 HTML](https://arxiv.org/html/2108.03298)

## 2. Proprioception suppresses vision exactly where the target must be found

### Takeaway
An ICLR 2026 study pinned the mechanism down: in the sub-phases where the motion changes and the target must be
localised, a vision+proprioception policy barely uses vision, because the proprioceptive signal lowers the training
loss faster and dominates the optimisation. Vision-only policies beat the concatenated ones on every task they report.
This is our measured picture: the heading at START is the target-dependent transition, and the state carries the rest
[OURS].

### Evidence
- Lu et al., "When would Vision-Proprioception Policies Fail in Robotic Manipulation?" (GAP), arXiv 2602.12032, ICLR
  2026 -- [arXiv HTML](https://arxiv.org/html/2602.12032):
  - "Vision-Proprioception policies perform 15.8 % worse than Vision-only policies" (Fig. 1) [SIM].
  - Sim, MLP head, 5 seeds x 100 rollouts: vision-only vs concatenation pick-place 91.8 vs 78.4, assembly 82.6 vs 74.6,
    bin-picking 63.2 vs 48.8, threading 43.6 vs 33.2; random proprio masking ("Mask", after RDT-1B) beats concatenation but is mixed against vision-only
    (86.4 / 89.2 / 60.4 / 47.2); GAP 94.2 / 94.2 / 70.4 / 53.0 [SIM].
  - Real, 20 trials each: vision-only vs concatenation vs GAP -- cube 14 / 11 / 17, press button 18 / 12 / 20,
    thermos into bag 11 / 7 / 16, lift lid and pour 9 / 5 / 15 [REAL-other].
  - Octo fine-tuned in sim: vision-only beats vision+proprio on all four tasks (e.g. disassemble 95 vs 82 %) [SIM].
  - The fix: estimate per timestep the probability of a motion-transition phase from proprioception (change-point
    detection + an LSTM) and scale the proprioception branch's gradient by lambda (1 - rho), lambda = 0.3, for the
    first half of training. Tested on MLP, diffusion, Octo and a modified ACT (dual-arm).

## 3. The state lets the policy memorise trajectories instead of generalising over space

### Takeaway
Dropping the state turned zero spatial generalisation into most of it for pi0, ACT and Diffusion Policy on real arms --
but only with actions relative to the end effector; with absolute joint (or absolute EE) actions even the state-free
policy did not generalise outside the trained heights and positions. In-domain all variants scored ~100 %. Our failure
is worse than theirs: ours fails inside the trained region, on layouts it has not seen.

### Evidence
- Zhao et al., "Do You Need Proprioceptive States in Visuomotor Policies?", arXiv 2509.18644 (Sept 2025) --
  [arXiv HTML v2](https://arxiv.org/html/2509.18644v2):
  - Table V (dual wide-angle wrist cameras): **ACT with state: height 0, horizontal 0.084; without: 0.933 / 0.517**;
    Diffusion Policy 0 / 0 -> 0.867 / 0.533 [REAL-other; trial counts per cell not stated for this table].
  - pi0, Pick Pen, 30 trials per cell: with state 30/30 in-domain, 0/30 at 72 cm and at 90 cm; without state (normal
    wrist camera) 30/30, 30/30, 17/30; noise on the state's height (+-5 cm) 30/30, 24/30, 11/30 [REAL-other].
  - Table VIII (pi0, height generalisation, with state): noise augmentation 0.633, more diverse data (75-84 cm tables)
    0.117, task-mixed training 0, LoRA 0 -- against 0.983 without the state [REAL-other].
  - Table III (state-free, Pick Pen): relative EEF 1.0 / 0.984 / 0.584 (in-domain / height / horizontal); absolute EEF,
    relative joint and absolute joint all 1.0 / 0 / 0 [REAL-other].
  - Limitations: vision-only policies may be sensitive to the background.
- Jung and Park, "Policy-Induced Hand Priors in Humanoid Dual-Arm Manipulation" (Aug 2026): the start pose induces a
  hand preference that differs by policy; broader start-pose coverage in training and targeted augmentation around a
  bad pose fix it (abstract: no numbers, policies not named) [SIM] -- [arXiv 2608.11769](https://arxiv.org/abs/2608.11769)

## 4. Large models memorise too: same trajectory when the object moves

### Takeaway
Our symptom -- one trajectory wherever the cube is -- is the documented failure of VLAs on fixed-layout benchmarks:
moved objects drop OpenVLA and pi0 to 0 %, and the trajectories stay "nearly identical". A bigger or pretrained model is
not by itself the cure; layout diversity in training and an evaluation on layouts never seen are.

### Evidence
- LIBERO-PRO (Zhou et al., arXiv 2510.03827, Oct 2025): under object-position perturbation, OpenVLA 0.98 -> 0.00 and
  pi0 0.92-0.98 -> 0.00 in all four suites, pi0.5 0.97-0.98 -> 0.08-0.38; "the model yields nearly identical
  trajectories" when objects are relocated; OpenVLA and pi0 collapse beyond ~0.2 units of displacement [SIM] --
  [arXiv HTML](https://arxiv.org/html/2510.03827v1)
- DemoGen (Xue et al., RSS 2025): Diffusion Policy / DP3 spatial generalisation is "concentrated near demonstrated
  placements"; "the effective range can be approximated by the union of the areas surrounding the demonstrated object
  placements"; pretrained 2D and 3D encoders helped but "none of these methods fundamentally resolve the spatial
  generalization problem" [SIM] -- [arXiv HTML 2502.16932](https://arxiv.org/html/2502.16932v1)

## 5. ACT's CVAE: the latent is supposed to absorb multimodality; in practice it often does nothing

### Takeaway
ACT trains a CVAE whose encoder sees the future actions and runs the decoder with z = 0 at test time. If the latent
carried "which way" during training, z = 0 at test time would produce the average heading. In our checkpoint it carries
nothing (KL 0.000 [OURS]), and an independent 2026 re-run of ACT's own ablation did not reproduce its benefit. The
averaging we see is plain L1 regression: with an input that cannot tell the targets apart, L1 returns their median.

### Evidence
- ACT: "At test time, we set z to be the mean of the prior distribution i.e. zero to deterministically decode"; without
  the CVAE objective, success with human data fell "from 35.3 % to 2 %"; with scripted data "almost no difference";
  beta = 10 [SIM] -- [arXiv 2304.13705 HTML](https://arxiv.org/html/2304.13705v1)
- Kang, "The Latent That Never Was: A Forensic Re-run of the CVAE Ablation in ACT" (Sept 2026): "The published drop
  does not reappear in our tests"; the sampled latent gave "little reconstruction benefit at every tested nonzero
  weight"; skipping the encoder raised training throughput [SIM] -- [arXiv 2609.16745](https://arxiv.org/abs/2609.16745)
- LeRobot 0.6.1 `modeling_act.py`: in eval mode `latent_sample = torch.zeros(...)` (read in the installed package).

## 6. An ambiguous target: several pieces in training, one in the evaluation

### Takeaway
The picker takes the nearest piece first (`Task.order`), so a training START frame shows 3-5 pieces and the action
follows one of them by a rule the policy must learn from pixels; the evaluation shows one cube. With a deterministic
L1 head an unlearnt rule yields the median over the pieces' directions -- a heading "between" them. Object-level
structure and an explicit 2D goal close most of this gap in simulation. A generative head would pick one piece instead
of the middle, but the evidence says multimodality is not why generative heads win.

### Evidence
- Li, Chapin, Chen, Peters, Kshirsagar, "More Structure, Not More Capacity" (July 2026), ManiSkill3 PickCube: frozen
  object-centric slots 55.0 +- 2.9 % vs dense DINO global feature 32.6 +- 1.5 %; 16x more patch tokens no better; an
  explicit 2D spatial goal plus native resolution 68.7 +- 4.2 %, near a 3D-oracle's 71.7 +- 4.1 %; grounding cut
  "Near-Miss" failures, not "No-Grasp" ones [SIM] -- [arXiv 2607.09825](https://arxiv.org/abs/2607.09825)
- Pan, Simchowitz et al., "Much Ado About Noising" (Dec 2025, rev. Feb 2026): generative control policies "do not
  succeed because they capture multimodal action distributions"; their edge is supervised iterative computation with
  some stochasticity; a two-step regression policy roughly matches flow policies [SIM] --
  [arXiv 2512.01809](https://arxiv.org/abs/2512.01809)
- Our data: the sim episodes draw 3-5 pieces (`m3_overnight.py`: `--pieces 3 + seed % 3`); only each clean-up's last
  attempt shows a single piece [OURS].

## 7. A small encoder, no augmentation, few scenes: memorisation by default

### Takeaway
Image policies trained without random shifts or crops overfit to pixel-exact scenes; every widely used recipe since
2020 turns augmentation on. LeRobot's ACT has it off by default, and our training commands never enable it.

### Evidence
- robomimic: removing pixel-shift randomisation cost 47 % and 35 % relative success (Square, Transport) [SIM]; on the
  real Can task 26.7 % without vs 73.3 % with [REAL-other] -- [arXiv 2108.03298 HTML](https://arxiv.org/html/2108.03298)
- Diffusion Policy: random crop in training, static centre crop at inference; crops 76x76 (sim, ~90 %) and 288x216
  of 320x240 (real) (Appendix A.3, Table 7) -- [arXiv 2303.04137 HTML](https://arxiv.org/html/2303.04137v5)
- Hansen et al. 2022: a shallow ConvNet from scratch with data augmentation is competitive with frozen pretrained
  representations (PVR, MVP, R3M) in sim and on a real robot (abstract) [SIM, REAL-other] --
  [arXiv 2212.05749](https://arxiv.org/abs/2212.05749)
- LeRobot 0.6.1 `ImageTransformsConfig`: `enable: bool = False`; when enabled, up to 3 of brightness 0.8-1.2, contrast
  0.8-1.2, saturation 0.5-1.5, hue +-0.05, sharpness 0.5-1.5, RandomAffine +-5 deg / +-5 % translate per frame (read in
  the installed package); our baseline's `train_config.json`: `image_transforms.enable: false`, `eval_split: 0.0`.
- ACT's paper mentions no augmentation; 50 demonstrations per task with four fixed cameras -- a different regime from
  one moving wrist camera and 13-42 layouts -- [arXiv 2304.13705 HTML](https://arxiv.org/html/2304.13705v1)

## 8. Which of these fit our symptoms

### Takeaway
All four main mechanisms are present and they compound; the measurements rank them.

| mechanism | fits? | our evidence [OURS] |
|---|---|---|
| copycat / state shortcut | yes, primary | state == previous action; a state-only lookup matches Friday's ACT (7.79 vs 7.79 deg); heading follows the state 0.83 vs picture 0.10 on unseen frames |
| memorised scenes (few layouts, no augmentation) | yes, primary | r 0.99 on its training START frames, slope 0.08 on unseen ones, shrinks on mirrored ones; 13 / 42 layouts; ~140 epochs |
| CVAE z = 0 averaging | no (as a separate cause) | KL 0.000; z = posterior gives the same chunk |
| ambiguous target among pieces | plausible, secondary | median-of-pieces heading is what L1 gives; untested: needs single-cube frames |
| ResNet18 cannot localise the cube | unknown | the 72-74 frame probe was underpowered; the ImageNet features are not the obvious bottleneck (section 7 of the fixes note) |
