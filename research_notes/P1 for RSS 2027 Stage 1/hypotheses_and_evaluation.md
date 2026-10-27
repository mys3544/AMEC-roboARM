# P1's hypotheses and evaluation protocol for the RSS 2027 Stage 1 abstract

Scope: which of P1's hypotheses H1-H6 (RESEARCH.md 9.6) our data can carry by ~1 December 2026, how to word the chosen
ones so that they can be falsified, and the evaluation protocol behind them: trial counts, Wilson intervals, sequential
tests, a written perturbation protocol for "cube moved mid-reach", an evaluator independent of the teacher, and how to
show that the simulator ranks checkpoints as the real M3 does. Our own state is taken from NEXT.md (27 October
morning): the sim expert writes LeRobot episodes (15 of 16 picks in the first batch; 134 of 189 simulated picks in the
box, 71 %; 52 of 64 with a box of its own), the real picker clears the table (32 of 34 attempts on 2026-10-23, 8 of 8 on
10-26, 25 picked over 8 clean-ups on 10-27), the real-only ACT heads one way wherever the cube is (sim grid: reached
5/18, lifted 1/18, in the box 0/18; real: nothing grasped in 4 trials), a co-trained ACT (74 sim + 80 real episodes) is
being evaluated, v2 runs tonight. Sample sizes and intervals marked "computed" are mine (a short Python script, normal
approximations; exact tests will differ slightly).

Labels as in the other notes: [REAL-LC], [REAL-other], [SIM], [CLAIM], [2nd].

## 1. Which hypotheses our data can carry by ~1 December

### Takeaway
Only two of H1-H6 can carry real numbers by 1 December: **H1 (parity)**, if topic 1's fixes and co-training make the
ACT grasp by ~13 November, and **H3 (the exchange rate)**, from a data-mix sweep we would train anyway. **H2 (the
student beats the teacher)** can carry its teacher half now -- the real picker under the perturbation protocol needs no
policy -- and its student half in simulation; it is the reason P1 exists and must be registered. H6 is not testable on
the real robot by December, and the RSI framing is crowded; make "zero human labels or demonstrations" a design
constraint, not a hypothesis. H4 (robot card) and H5 (gate) are separate papers; keep recording their data.
Add one validity hypothesis that the method needs: the simulator ranks checkpoints as the robot does.

### Cited findings
- RSS Stage 1 judges hypotheses and methodology; experiments may be preliminary or missing; falsification counts
  ([CfP](https://roboticsconference.org/information/cfp/); `rss2027_stage1_call.md`).
- Today's gap between sim and real behaviour: the real-only checkpoint goes to J1 ~93-98 on the robot and ~53-81 in the
  sim grid for the same cells; "the real and the sim agree that it fails, not on where it goes" (NEXT.md, 2026-10-27).
- Co-training on a near-identical setup moved a real SO-101 ACT from 6/21 to 19/21
  ([GitHub README](https://github.com/AadhavSivakumar/real2sim2real-so-arm101)) [REAL-LC]; earlier notes: Genesis ACT
  17/20 vs 5/20, NVIDIA/UT co-training 45.3 -> 83.2 % (`research_notes/Simulácia a VLA pre presný úchop/`).
- Physical self-improvement claims need an independent evaluator and isolated gains
  ([2610.18842](https://arxiv.org/abs/2610.18842), [2610.29677](https://arxiv.org/abs/2610.29677)).

### Inferences
| hypothesis | what it needs by 1 Dec | carried by 1 Dec? | role in the abstract |
|---|---|---|---|
| H1 parity | a student that grasps; static grid, teacher N=60, student N=40 | yes, if the ACT grasps by ~13 Nov | confirmatory (precondition for H2) |
| H2 student beats teacher | teacher under the protocol (no policy needed); student in sim with DAgger; student on the robot | teacher half yes; student half in sim; real student only if H1 goes | confirmatory, the headline |
| H3 exchange rate | 4-6 trainings of a data-mix sweep + a real grid each | yes, preliminary | exploratory, from the same sweep as HV |
| H4 robot card | sim cards (recorded); a camera-knock test on a working policy | sim only | out; IROS'27 |
| H5 hand-back gate | per-chunk verdicts (recorded); a working policy + hand-back | sim only | out; later |
| H6 no human in the loop | auto-reset + takeovers running unattended for nights | sim only | design constraint, not a test |
| HV (new) sim validity | >= 6 checkpoints with sim and real grids | yes, from the sweep | confirmatory: the method stands on it |

### Gaps
- Whether the co-trained ACT grasps at all is unknown until its evaluation (Wednesday 28 Oct).

## 2. The wording, falsifiable

### Takeaway
State each hypothesis with its population (cells, cube, perturbation), its metric, its N, its test, and three outcomes
decided in advance: supported, falsified, inconclusive. Freeze it in a dated commit before the runs (~5 November).

### Cited findings
- RSS values falsification "with meaningful insight" ([CfP](https://roboticsconference.org/information/cfp/)).
- A paired design with a same-seed re-run as a placebo separated a real corrector effect from run-to-run variation on
  frozen pi0.5 (43.2 % of failures succeeded on a plain re-run) ([2610.06921](https://arxiv.org/abs/2610.06921)) [SIM].
- Student below teacher on success but 2.9x faster on a real UR5e ([TRACE, 2609.38857](https://arxiv.org/abs/2609.38857))
  [REAL-other].

### Inferences
Proposed text (numbers to be fixed at the freeze):
- **H1, parity.** A student trained only on data the teacher produced -- privileged simulated episodes plus <= 100 real
  picker episodes, no human demonstration, label or reset in its training data -- reaches first-attempt grasp success
  within 10 points of the real picker on the 20-cell static grid (30 mm cube). Test: one-sided non-inferiority on the
  difference, paired by cell, alpha 0.05. Supported if the lower 95 % bound of (student - teacher) is above -10 points;
  falsified if the upper bound is below -10; otherwise inconclusive. Stage 1: N = 40 student / 60 teacher
  (preliminary); Stage 2: N = 120 each.
- **H2, the student beats its real teacher where the teacher cannot see.** When the cube is displaced 4 cm sideways
  after the teacher's last perception (trigger T2, section 4), a student trained with <= 3 rounds of simulated DAgger
  (the privileged expert relabels the student's states, displacements drawn in training) has a higher first-attempt
  success than the real picker. Test: one-sided two-proportion superiority, alpha 0.05, run sequentially (STEP) to
  N_max = 150 per arm; smallest effect of interest 15 points. Secondary: success within 60 s including the teacher's own
  retry, time to success, successes per minute; the same at T1 and at a 1 cm control. Falsified if the student's
  first-attempt success is not above the teacher's at N_max, or if it loses parity (H1) on still cubes.
- **HV, the simulator is a valid proxy.** Over >= 6 checkpoints spanning >= 30 points of real success, the sim grid
  ranks them as the real grid does: MMRV <= 0.10 and Spearman rho >= 0.8 (exact one-sided p < 0.05 needs rho >= 0.83 at
  n = 6, >= 0.64 at n = 8). Falsified if MMRV > 0.10 or rho < 0.5; checkpoint choice and DAgger then move to the robot.
- **H3 (exploratory).** The same sweep prices simulated teacher episodes in real ones: the number k of sim episodes
  that buys the real success of one real episode, with a bootstrap interval. Because both come from the same planner,
  k measures the domain gap without the demonstrator-style confound of human-real + scripted-sim co-training.
- **H6 as a constraint.** Every training example is produced, relabelled and judged by the teacher or the simulator;
  the audit is the episode log. Humans appear only in evaluation (audit, manual pushes) and are counted.

### Gaps
- The exact grid (18 cells today at 16/19/22 cm, J1 40-100) and the trigger distance are to be fixed from recorded
  attempts before the freeze.

## 3. Trial counts, intervals, sequential tests

### Takeaway
Wilson intervals on every rate. At 80 % success, 20 trials give [58, 92] % and 50 give [67, 89] %. Parity within 10
points at ~90 % success needs ~110 trials an arm; a 25-point gap is found with ~50 an arm, a 15-point gap needs
~110-130. Use STEP for two-policy comparisons and AnyRank (sequential, with the simulator as its prior) for ranking
several checkpoints; both control the error rate when stopping early.

### Cited findings
- Computed Wilson 95 % intervals: 16/20 [58, 92]; 40/50 [67, 89]; 48/60 [68, 88]; 18/60 [20, 43]; 30/60 [38, 62].
  Ours: teacher 32/34 [81, 98]; sim expert 134/189 [64, 77]; real-only ACT in the sim grid 0/18 [0, 18].
- Computed sample sizes (alpha 0.05, power 0.8): non-inferiority, margin 10 points, both at 90 %: 112 an arm; margin
  15: 50 an arm. One-sided superiority: 30 -> 55 %: 48 an arm; 20 -> 45 %: 43; 30 -> 45 %: 128; 20 -> 35 %: 109.
- **STEP** (RSS 2025, [2503.10966](https://arxiv.org/abs/2503.10966)): sequential comparison of two policies for the
  10-50-trial regime; decides in near-minimal trials with error control; up to 32 % fewer trials than baselines (v4;
  earlier versions said 37-40 %) [SIM / REAL-other].
- **AnyRank** (CoRL 2026, [2610.14603](https://arxiv.org/abs/2610.14603), full text read in part): ranks several
  policies sequentially (N-SCORE anytime-valid tests from the STEP authors + graphical multiple testing), allocates
  trials to the hard pairs, accepts priors from simulator rollouts or world models, and keeps family-wise error control
  even with a wrong prior; up to 50 % fewer hardware evaluations than Bonferroni-style methods on RoboArena, TRI-LBM and
  DROID data; default alpha 0.1; code at github.com/irom-princeton/multi_hypothesis_test.
- Blind, randomised trials are the standard set by TRI's LBM study ([2507.05331](https://arxiv.org/abs/2507.05331),
  abstract); RESEARCH.md section 6 lists the expected practice (20-30 trials a condition, 50 for a headline, fixed
  poses, randomised order, named generalisation axes, raw logs released).

### Inferences
- Budget (our estimates): a real trial with the policy (<= 25 s), a look before and after, and an automatic reset by
  the picker is ~60 s, so ~60 trials an hour. Stage 1 preliminaries: teacher static 60 + teacher perturbed 100 + six
  checkpoints x 40 + best student perturbed 50 = ~450 trials, ~8 h of robot time over three weeks. Stage 2: ~25 h more,
  which fits December-January.
- The automatic reset is the enabler: the picker already carries a held piece to (radius, J1) and lets go
  (`m3pick.carry_to`, `drop_at`). End each evaluation trial at the student's lift (+2 s), then the script carries the
  cube to the next cell; if the student did not grasp, the picker picks it where it lies. Count the trials a human had
  to reset.
- Interleave conditions (teacher, student, baseline) in randomised blocks over the same cells and the same
  perturbation draws; compare paired.

## 4. A written perturbation protocol for "moved mid-reach"

### Takeaway
Copy LIBERO-Dynamic's structure -- named families, speed tiers, a fixed phase, latency stated -- and adapt it to a
cube on the floor and a teacher with known phases. The trigger must be the same rule for teacher and student, defined
by geometry, and placed **after the teacher's last perception** (T2) for the headline. The base's own sidestep makes
the displacement automatic and repeatable; a smaller hand-pushed set shows the two agree.

### Cited findings
- LIBERO-Dynamic ([2610.24579](https://arxiv.org/abs/2610.24579), [full text](https://arxiv.org/html/2610.24579)):
  linear / Catmull-Rom / zigzag paths; 3, 6, 9 cm/s; motion phase fixed (scene advanced 40 control steps before the first
  observation); latency per model measured end-to-end and frozen; 1,800 episodes a configuration; pi0.5 96.9 -> 57.8 %
  static -> dynamic [SIM]; physical study 50 trials a condition [REAL-other]; no confidence intervals.
- AstroBench: cube-specific ACT 6/10 moving vs 8/10 static ([2610.31064](https://arxiv.org/abs/2610.31064), abstract).
- Our teacher's phases (m3pick.grip): pregrasp (move above the target) -> look.close (the last perception: re-aims by
  1-6 mm, once 25.6 mm) -> descend (one open-loop glide) -> close -> lift -> look.after. The gripper opens 61.5 mm, so a
  30 mm cube tolerates about +-15.75 mm sideways (reports/ "Simulácia a VLA pre presný úchop", our computation).
- A fixed-time trigger plus a scripted retreat is a baseline any learned corrector must beat
  ([2610.06921](https://arxiv.org/abs/2610.06921)) [SIM].

### Inferences
The protocol, to freeze before training:
1. **Object and cells.** The 30 mm picture cube on the printed sheet; the grid cells of the sim evaluation, the same in
   sim and real; the cube placed by the reset, its pose read by the look before the trial.
2. **Families.** P-base (primary, automatic): at the trigger the base sidesteps by Delta perpendicular to the arm's
   line to the cube, at ~5 cm/s; neither the teacher nor the student is told. P-push (secondary, manual, N = 20-30): a
   guided pusher slides the cube by Delta on a console beep. P-sim: a kinematic push in the simulator at the same
   timing and speed; also the base-sidestep form.
3. **Tiers.** Delta = 1 cm (control, inside the gripper's tolerance), 2 cm (edge), 4 cm (outside it); speed 5 cm/s
   (LIBERO-Dynamic's 3-9 cm/s range).
4. **Trigger.** T1: the first tick the commanded grip point is within r1 of the cube (before the teacher's close
   look). T2 (headline): within r2, where r2 is the median distance at the middle of the teacher's descent, measured on
   >= 20 recorded attempts before the freeze. One rule for both arms; the commanded pose is known to the bridge, so the
   trigger fires in the bridge, not by a human.
5. **Outcome.** Primary: first-attempt grasp (lifted >= 3 cm and gone from the floor at the after-look). Secondary:
   success within 60 s including retries, time to success, successes per minute, final displacement read by the
   after-look.
6. **Arms.** The real picker as it is (its next look round is its retry); the student; the student + fixed-time
   retreat-and-retry; the privileged sim expert (sim only, upper bound).
7. **N and order.** 50 per arm and condition at Stage 1 where run, up to 150 with STEP at Stage 2; randomised blocks;
   paired by cell and Delta.
8. **Latency.** State the student's chunk timing (chunk 20, 10 executed at 10 Hz) and the teacher's look and plan times;
   both run at their real speed (LIBERO-Dynamic's point: timing changes behaviour).
- A reviewer will say an open-loop descent is a strawman. The answer is the claim's scope -- the student exceeds its own
  teacher with no new engineering -- plus the T1 condition, where the teacher's close look can still react, and a
  sentence on visual servoing as the classical alternative.
- Base sidestep is ego-motion (the whole picture moves), a push is object motion. Report the two separately; P-base is
  also a real disturbance for a mobile manipulator (wheel slip, a bump).

### Gaps
- The base's sidestep accuracy over 2-4 cm at 5 cm/s is not measured (odometry drifts ~10 cm a full circle; a short
  sidestep should be far better). Measure it with the look before and after, and report the realised displacement.

## 5. Showing that the simulator ranks checkpoints as the robot does

### Takeaway
Run every checkpoint on the same cells in sim and on the M3, compute MMRV, Pearson and Spearman across checkpoints,
and add a behaviour-level check per cell (heading at 1.2 s, closest approach), because today the two agree only that the
baseline fails. Ranking is meaningful only across checkpoints whose real success differs by >= 20 points: at 40 real
trials each, two checkpoints 10 points apart are mis-ordered ~19 % of the time by noise alone.

### Cited findings
- **SIMPLER** ([2405.05941](https://arxiv.org/html/2405.05941)): RankViolation(i, j) = |R_i - R_j| x 1[(S_i < S_j) !=
  (R_i < R_j)], MMRV = (1/N) sum_i max_j RankViolation(i, j); MMRV weights each mis-ranking by its real gap, which Pearson
  does not; per task, then averaged. Visual matching MMRV 0.056 / Pearson 0.924 against validation action MSE 0.375 /
  0.308 [SIM vs REAL-other] (numbers from the earlier notes, `sim2real_gap_measurement_and_correction.md`).
- Earlier notes: a splat + physics twin r 0.90-0.94 (MMRV 0.08-0.17) across ACT, DP, pi0 and SmolVLA checkpoints, an
  Isaac Lab baseline 0.24 and 0.65 ([2511.04665](https://arxiv.org/html/2511.04665v2)); PolaRiS ~0.9 average, 0.81 worst;
  its fix for policies that never saw the sim: a short co-fine-tune with ~10 % in-domain sim data.
- New in October (abstracts via the API): YUBI-Real2Sim ([2610.17521](https://arxiv.org/abs/2610.17521)) reproduced the
  real ordering of pi0.5-based policies in 8 of 9 scene-task comparisons (3DGS + MuJoCo, fisheye wrist views) and found
  that view-quality scores do not yet predict ranking reliability; PhysCanvas ([2610.28613](https://arxiv.org/abs/2610.28613))
  world-model vs real success, mean task-wise Pearson 0.602 -> 0.885; ProWorld ([2610.15294](https://arxiv.org/abs/2610.15294))
  Pearson 0.95 / Spearman 0.94 over three policies and four tasks [CLAIM: trial counts not in the abstracts].
- Computed: exact one-sided Spearman critical values (no ties) 0.90 at n = 5, 0.83 at n = 6, 0.71 at n = 7, 0.64 at n = 8.
  Computed mis-ordering probability of two real estimates, true 60 vs 50 %: 26 % at 20 trials each, 19 % at 40, 14 % at
  60; true 60 vs 40 %: 10 %, 4 %, 1 %.

### Inferences
- Checkpoints: the data-mix sweep (real 20 / 80 x sim 0 / 1k / 4k, minus the meaningless corners), two training steps
  of one mix, and GR00T N1.7 if hekate is free -- 6-8 points spanning a range. All with >= 10 % sim in training, so no
  checkpoint is ranked on visuals it never saw.
- The confound: a checkpoint trained on more sim data looks better in sim for visual reasons alone. Report the
  correlation within one data mix (steps, seeds) as well as across mixes.
- Per-cell behaviour agreement (Pearson over cells of the J1 heading at 1.2 s and of the closest approach) is cheap --
  both evaluators already log them -- and it would have caught today's disagreement (real J1 ~95, sim ~65).
- Sim trials per checkpoint: 60 (three per cell); real: 40. Use AnyRank with the sim ranking as its prior to stop real
  trials early on pairs the sim already separates.

### Gaps
- No published sim-real ranking study for a cheap servo arm with a wrist RGB-D camera and command-only joint state was
  found (the closest, an SO-101 GR00T sim/real agreement of 66.3 %, is self-reported, earlier notes).

## 6. The referee, and an evaluator independent of the teacher

### Takeaway
Let the picker's verdicts label the training data, but report numbers from evaluators the loop cannot read or train
on: the simulator's true state in sim; on the robot, a human audit of every evaluation video, with the picker's
agreement with the audit reported. That is what 2610.18842 asks for, and it turns S2 (the referee study) into one
table rather than a claim at risk.

### Cited findings
- Score with "an independent evaluator that the improvement procedure cannot read, train on, or modify"; compare
  generations from the same start and budget ([2610.18842](https://arxiv.org/abs/2610.18842)).
- RoboJev-27B: 95.2 % balanced accuracy, AUROC 0.993 on held-out SO-101 episodes; open weights; ~0.26 s an episode on an
  H100 ([2610.33718](https://arxiv.org/abs/2610.33718)) [REAL-LC data].
- FailBench: 13 detectors, best 0.77 balanced accuracy, all biased to "success" (RESEARCH.md 2, S2).
- The M3 has no joint or gripper read-back (NEXT.md, 2026-10-23): its referee is the camera (the look after, the box).

### Inferences
- Real evaluation: picker verdict + human audit of 100 % of evaluation trials (~10 s a video), blind to the checkpoint's
  identity among students; Cohen's kappa between them. RoboJev as an optional third judge if the 5090 is free.
- Training: the picker's verdicts decide which attempts enter the dataset, as now.
