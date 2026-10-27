# Data design for an ACT that looks, and the next training runs ranked

Scope: how the M3's datasets should be built so that the image is the only way to lower the loss at the moment the
policy chooses its way -- layouts against attempts, episodes per table cell, one piece or several, uniform placement,
start-pose jitter and DART noise in the simulator, held-out layouts -- and the next 2-3 training runs with what each
tests and how it passes on the sim grid and on the M3. Builds on `our_data_diagnosis.md` (our measurements),
`failure_modes_literature.md` and `fixes_with_effect_sizes.md` in this folder, and on the sim/co-training notes of
2026-10-26 (`research_notes/Simulácia a VLA pre presný úchop/`), which cover the sim-real ratio and the twin.
The co-trained 150k and the overnight dataset v2 / ACT v2 (`m3_overnight.py`) had not been read when this was written
(due Wednesday morning); the plan below starts by probing them.

Labels: **[REAL-LC]**, **[REAL-other]**, **[SIM]**, **[CLAIM]**, **[2nd]**; **[OURS]** our measurement; **[EST]** our
estimate.

## 1. Count layouts, not attempts

### Takeaway
What teaches a policy to find the object is the number of distinct scenes it must tell apart, not the number of
attempts: imitation performance follows the diversity of environments and objects, and per-scene demonstrations
saturate. Our 154 attempts are 42 scenes, the real-only set 13 [OURS] -- few enough to memorise, which is what the
baseline did (r 0.99 on its training START frames, slope 0.08 on unseen ones [OURS]).

### Evidence
- Lin et al., "Data Scaling Laws in Imitation Learning for Robotic Manipulation" (ICLR 2025): generalisation follows a
  rough power law in the number of environments and objects; "the diversity of environments and objects is far more
  important than the absolute number of demonstrations"; for 8 / 16 / 32 environment-object pairs performance plateaus
  at 400 / 800 / 1600 demonstrations -- ~50 per pair [REAL-other] -- [arXiv 2410.18647](https://arxiv.org/abs/2410.18647)
- DemoGen: the effective range is "the union of the areas surrounding the demonstrated object placements"; DP3 in the
  full workspace improved 37 % from 100 to 150 demonstrations and 6 % from 150 to 200 (Appendix B2) [SIM] --
  [arXiv 2502.16932 HTML](https://arxiv.org/html/2502.16932v1)
- Ours: 3-9 attempts per clean-up, one piece fewer each time; LeRobot's eval_split holds out the last episodes per task
  from layouts it trains on [OURS].

## 2. How many layouts per table cell

### Takeaway
No paper gives a density for a single wrist camera and a 2D policy, so this is an estimate: aim for >= 25-30 distinct
single-cube START layouts in the neighbourhood of each of the 18 grid cells (~500 layouts over J1 40-100), and as many
again over the rest of the drawn area (J1 30-135, 11-25 cm from J1's axis), uniform in bearing. That is ~1000 layouts,
about one night of three sim workers at the current ~3.3 min per clean-up if each clean-up yields 2-3 fresh layouts.
Check the count in `overnight.log` before training.

### Evidence
- The grid: 6 bearings (J1 40-100, ~12 deg = ~4 cm apart at 19 cm) x 3 reaches (16 / 19 / 22 cm) [OURS, NEXT.md].
- Lin et al.'s ~50 demonstrations per environment-object pair [REAL-other] and DemoGen's 150-200 demonstrations to
  saturate a 30 x 40 cm region with a 3D policy [SIM]; 2D image policies generalise spatially worse than DP3 in DemoGen's
  benchmark [SIM] -- hence more than 200 for our 2D ACT [EST].
- Sim throughput: ~3.3 min a 3-5 piece clean-up with an ACT training beside it (NEXT.md, 2026-10-26) [OURS].

## 3. One piece or several in an episode

### Takeaway
For the question "does it head for the cube", the START frame should show one candidate, or the target must be marked.
Several pieces with nearest-first make the target a rule the policy must infer from pixels; with an L1 head an unlearnt
rule yields the median heading between the pieces. Make about half the simulated episodes single-piece (or re-scatter
the remaining pieces after each pick so every attempt is a new layout), keep clean-ups for the skill, and mark the
target in the image when the clean-ups dominate (`fixes_with_effect_sizes.md` section 4).

### Evidence
- Explicit 2D goal: 55.0 -> 68.7 % with object-centric features in ManiSkill3 PickCube [SIM] --
  [arXiv 2607.09825](https://arxiv.org/abs/2607.09825); PEEK masks +32 points in sim, ACT 0.20 -> 0.40 on BRIDGE
  [SIM, REAL-other] -- [arXiv 2509.18282 HTML](https://arxiv.org/html/2509.18282v1)
- `shared/roboshared/task.py` `Task.order`: the nearest to the robot's origin first; `m3_overnight.py` draws 3-5 pieces
  an episode; only the last attempt of a clean-up shows one piece [OURS].

## 4. Uniform placement, and the box out of the way

### Takeaway
The real grips lean left (J1 median 76.5) and the sim's did too (mean 77) because the box's keep-clear emptied the
right; the per-episode random box (40177cb) widened the sim's picks to J1 39-131, mean 81. Draw the target's bearing
uniformly per 15 deg bin (rejection sampling) and its reach uniformly over 11-25 cm, and log the histogram with each
batch (`m3_sim_report.py placement`).

### Evidence
- Grip J1 bins (<40, 40-55, 55-70, 70-85, 85-100, 100-115, >115): real 8/13/11/15/13/11/9, sim 1/10/15/19/17/12/0
  (co-training set) [OURS]; NEXT.md, Tuesday: random box 52 of 64 picked, J1 39-131, mean 81 [OURS].

## 5. Start-pose jitter and DART noise in the simulator

### Takeaway
Two cheap changes in `m3_sim.py episode` attack the copycat loop directly. (a) Start half the simulated attempts from
the look pose plus jitter (e.g. J1 +-8 deg, J2-J4 +-4 deg) so a given state no longer implies a given heading. (b) DART:
perturb the executed command during the approach for 0.3-1 s at a time (sigma ~2-3 deg on J1-J4, not during the descent
and grip) and record the expert's corrective command from the perturbed pose as the label, so the data contain
departures from wrong headings back to the cube -- exactly what a policy that started the wrong way needs. The expert
plans on true poses already, so it can relabel.

### Evidence
- DART (Laskey et al. 2017): noise injected into the supervisor's control during collection; real grasping in clutter
  +62 % over behaviour cloning (Toyota HSR) [REAL-other]; up to 3x less computation than DAgger in sim [SIM] --
  [arXiv 1703.09327](https://arxiv.org/abs/1703.09327)
- Start-pose dependence and its cure by broader start-pose coverage and targeted augmentation (abstract, no numbers)
  [SIM] -- [arXiv 2608.11769](https://arxiv.org/abs/2608.11769)
- Our state at the decision: all real episodes start exactly at START, so a state-input policy keyed on START is what
  the data teach [OURS]. The perturbed ticks themselves should be flagged (the per-tick `verdict` field exists) and left
  out of the loss or kept short, so the policy does not learn to add noise.

## 6. The simulator's START is not the robot's START

### Takeaway
The robot card's joint zeros are compensated into the recorded commands, so 73 of 74 simulated attempts start off
START (J1 85-94, J6 69-78) while every real one starts exactly on it [OURS]. A policy that sees the state can tell sim
from real by it and keep two behaviours. Dropping the state input removes the issue; if the state stays, at least
know that the sim evaluation (which starts at START exactly) shows the policy the real data's signature.

## 7. Hold out layouts; pick checkpoints by looking, not by loss

### Takeaway
LeRobot's `eval_split` holds out the last episodes of each task, from layouts it trains on, and a state-only lookup
already matches Friday's ACT on it (7.79 vs 7.79 deg) [OURS]. Hold out whole recordings (~10 % of sim layouts and 2-3
real clean-ups) and judge checkpoints by the six-number probe (`our_data_diagnosis.md` section 10) and the sim grid's
heading slope. Print the state-only lookup's error beside every eval loss: a model near it is not using the camera.

### Evidence
- LeRobot 0.6.1 `datasets/factory.py`: "The last ceil(n_episodes * eval_split) episodes per task are held out"
  (installed package); `--dataset.episodes` selects the training episodes explicitly.
- The 26ft baseline: `eval_split 0.0`, 40k steps fine-tuned from a 100k run on mostly the same real images (~140
  passes) [OURS].

## 8. The next runs, ranked

### Takeaway
First probe what exists (the co-trained 150k and ACT v2) -- 30 min on the laptop. Then one run that removes both
shortcuts at once (no state input, augmentation, decision frames upweighted, v2's many layouts, held-out layouts),
with its single-variable twin (the same with the state) on the second GPU if the user frees the 5090. Then relative
actions; then the target mark. Each run keeps the Orin path (ONNX -> TensorRT FP16, ~6 ms).

| run | what changes | what it tests | pass (sim grid, 18 cells) | pass (M3) | if it fails |
|---|---|---|---|---|---|
| R0 probe | nothing: probe cotrain150k and ACT v2 on held-out START frames | is the state + memorisation still the story with more layouts? | -- (offline: slope >= 0.7, r >= 0.8, conflict b > a) | -- | go to R1 |
| R1 "look-first" | v2 data; no state input; image transforms on (colour defaults, affine <= 3 % / 3 deg); standing + first 1 s of each attempt duplicated x5; whole layouts held out | removing the copycat input and pixel memorisation makes it look | heading slope >= 0.7, r >= 0.8, median heading error <= 10 deg at 1.5 s; reached (2 cm) >= 12/18 (baseline 5/18) | 3 spots x 2 trials: heading within 10 deg of the cube's J1 at all three; closest approach <= 20 mm at 2 of 3 (baseline 39/71/18 mm) | jerky chunk starts -> R2; still blind -> R3 |
| R1b ablation | R1 with the state kept | the state's share of the failure (a P1 hypothesis) | report the same numbers | only if R1 passes | -- |
| R2 relative | R1 + chunk relative to the commanded pose (J6 absolute), stats recomputed, state added back in the ONNX | servo-like supervision on every frame; smooth chunk starts | R1's, plus first-step J1 error <= holding still's (1.4 deg) | R1's, plus a lift at >= 1 of 3 spots | keep R1's head, try GAP-style gradient scaling with the state |
| R3 target mark | R1 or R2 + the target drawn into each frame (projected 3D point; detector mask near the cube) | is "which piece" the remaining problem? | R1's with 2-3 pieces on the table: the marked one reached >= 80 % | marked cube among 3 pieces, heads for it 3/3 | VLA path (GR00T N1.7, relative actions, state handled the same way) |

### Notes on the table
- Training cost: 150k steps took ~3.6 h on the A4500 alone (11.5 steps/s); v2 is several times bigger, so 150k steps
  is fewer passes over it -- less memorisation, as intended [OURS, NEXT.md].
- Code still to write (none done here): constant or removed state in the launcher plus the two `batch[OBS_STATE].device`
  lines; export without the state input (R1) or with it only for the relative add-back (R2); a phase-duplication option
  in `m3_episodes.py build`; the held-out recording list; for R3 the projection of the pick record's target into each
  tick at build time and at run time in `m3policy.py`.
- Every pass criterion above is ours, set against the baseline's measured numbers, not taken from a paper.

## 9. Not found, not verified

### Takeaway
Four gaps remain; none blocks R0-R1, and each has a measurement that closes it on our side.

- No published layout density (episodes per cell) for a wrist-camera 2D policy; section 2 is an estimate.
- Mirror augmentation (flip the image, J1 -> 180 - J1, mirror J5) would double the layouts and cancel the left lean;
  no source was checked (the web-search budget ran out). The M3 is not symmetric (box side, right-front lidar), so at
  most for the approach phase.
- Whether a state-free ACT with absolute joint targets jumps at chunk boundaries on our arm: unmeasured; the
  state-free study's in-domain scores (1.0 for all action spaces) say it can work in-domain [REAL-other].
- DINOv2 / v3 latency on the Orin NX: not found; must be measured if an encoder swap is ever tried.
