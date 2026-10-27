# P1 is still open for RSS 2027 only as "the student beats its own teacher": register three hypotheses by 5 November and measure the teacher now

P1's pipeline is no longer new. A community SO-101 project (undated, on Isaac Sim 6.1, so after 10 September) trained
ACT from a scripted Isaac Sim expert with DART noise and DAgger takeovers, 19/21 real against 6/21 real-only
([README](https://github.com/AadhavSivakumar/real2sim2real-so-arm101)), and SimPT (21 Oct) let a CuRobo planner make
VLA recovery data, real cable insertion 72 -> 96 % ([2610.29161](https://arxiv.org/abs/2610.29161)). What nobody has
shown is a student that beats its own classical teacher on a real robot: the closest, TRACE, has the student at 90 %
against the teacher's 95 %, but 2.9 times faster ([2609.38857](https://arxiv.org/abs/2609.38857)). RSS 2027 Stage 1
(4 Dec, 11:59 pm AoE, 5+1 pages) judges "the paper's hypothesis and methodology", accepts preliminary experiments and
counts a rigorous falsification as a contribution ([CfP](https://roboticsconference.org/information/cfp/)).
**Recommendation:** commit to three hypotheses -- **H1** parity with the picker within 10 points on a static grid,
**H2** the student beats the real picker when the cube is displaced 4 cm after the picker's last look, and **HV** the
simulator ranks at least six checkpoints as the M3 does (MMRV <= 0.10, Spearman >= 0.8). Report the sim-vs-real
exchange rate (H3) as exploratory from the same training sweep, and keep H4-H6 out. The teacher's half of H2 needs no
policy and can be measured this week. The student's half needs the ACT to grasp by **13 November, the go/no-go date**.
If it does not, the same registered protocol can still be submitted with the teacher's numbers, the simulated student
and a diagnosis.

Notes behind this report (sources, labels, computations): `research_notes/P1 for RSS 2027 Stage 1/`
(`scoop_check_since_oct21.md`, `rss2027_stage1_call.md`, `hypotheses_and_evaluation.md`). Others' numbers are theirs.
The web-search quota ran out on 27 October; the scoop check covers arXiv to 26 October.

## The pipeline is now commodity; the open claim is a student that beats its teacher

Between 21 and 27 October nothing appeared that does P1, but the ground around it filled further:

| work | what it does | what it does not do |
|---|---|---|
| Community SO-101 project ([README](https://github.com/AadhavSivakumar/real2sim2real-so-arm101), undated, Isaac Sim 6.1) | scripted state-machine expert in Isaac Sim, DART noise, DAgger takeovers by the expert, ACT co-trained with 40 teleop demos: 19/21 vs 6/21 real-only, operator-labelled | no grid, no CIs, no student-vs-teacher comparison; human teleop demos |
| SimPT ([2610.29161](https://arxiv.org/abs/2610.29161), 21 Oct) | CuRobo expert recovers from snapshots of the policy's states, in sim; real insertion 72 -> 96 % | the planner never acts on the real robot; no comparison with the planner |
| TILDE ([2610.25376](https://arxiv.org/abs/2610.25376), 18 Oct) | a learned reward model calls a learned teacher ("AutoDAgger"); up to 4.1x over BC on three real tasks | the teacher is a foundation model, not classical |
| ENPIRE ([2606.19980](https://arxiv.org/abs/2606.19980), Jun) | coding agents improve real policies with automatic resets and a detector-plus-rule verifier; 99 % pass@8 | judge partly learned; no classical teacher |
| TRACE ([2609.38857](https://arxiv.org/abs/2609.38857), 30 Sep) | privileged twin teacher + DAgger; real UR5e student 90 % vs teacher 95 %, 67 s vs 193 s | the student wins on time only |
| Backtrack ([2610.32941](https://arxiv.org/abs/2610.32941), 23 Oct) | failure head on a frozen VLA, then a scripted retrace and the same VLA again; LIBERO-Long failures -19 to -48 % | simulation only; the fallback is not a classical controller |
| LIBERO-Dynamic ([2610.24579](https://arxiv.org/abs/2610.24579), 17 Oct) | moving targets at 3/6/9 cm/s, fixed phase, latency-calibrated; pi0.5 96.9 -> 57.8 % | no teacher; no confidence intervals |
| RoboJev ([2610.33718](https://arxiv.org/abs/2610.33718), 24 Oct) | open 27B VLM success judge; 95.2 % balanced accuracy, AUROC 0.993 on held-out SO-101 episodes | a judge, not a teacher |

Four observations shape the abstract:
- **The pipeline is not the contribution.** "A privileged sim expert teaches ACT on a cheap arm, co-trained with real
  episodes" is done, and works (6/21 -> 19/21 on a near-identical setup). Our own real-only ACT failing where the data
  are thin is the same coverage failure that project reports.
- **Students rarely beat teachers on success.** Found only on another metric (TRACE, time), in simulation with improved
  demos ([2609.29020](https://arxiv.org/abs/2609.29020)), or for a skill the expert never needed (ASTEROID,
  [2610.30105](https://arxiv.org/abs/2610.30105)). One paper argues the opposite: keeping the expert and learning only its
  perception beats pixel imitation ([2609.34018](https://arxiv.org/abs/2609.34018)). H2 is plausible only where the
  *real* teacher cannot perceive and a *privileged* simulated teacher can -- the learning-by-cheating structure.
- **"Self-improvement" is crowded and now policed.** At least 15 robot RSI papers appeared in October. The survey
  [2610.18842](https://arxiv.org/abs/2610.18842) puts a fixed teacher with nightly retraining at L1 at best, and only if
  the reported numbers come from an evaluator the loop cannot read or train on. A review
  ([2610.29677](https://arxiv.org/abs/2610.29677)) finds that physical studies have not isolated gains from recursion.
  So P1 should not be sold as self-improvement.
- **VLM judges caught up on SO-101-like data.** RoboJev's 95.2 % makes FailBench's 0.77 an outdated argument for a
  physical referee. P1's referee is still free and fast, but the abstract should not rest on "VLM judges fail".

Still not found after these checks: a classical teacher that generates, relabels, judges and resets on the real robot;
a pre-registered student-vs-teacher battery; a manipulation policy conditioned on a measured fault vector; a learned
gate that hands control to a classical controller trained on that controller's verdicts.

## RSS 2027 Stage 1 rewards a registered plan, not finished results

The call, read on 27 October ([CfP](https://roboticsconference.org/information/cfp/)):
- **Format.** 6 pages: at most 5 of content and 1 of references. Only the PDF is reviewed: no appendix, website, video or
  supplementary material at Stage 1. Templates unmodified, PDF only, OpenReview.
- **Dates (AoE).** Abstract 4 Dec 2026 (Saturday 5 Dec, 12:59 in Slovakia). Stage 1 decision 5 Feb 2027, rebuttal
  12 Feb, Extension Charter 26 Feb, final 8-page paper 16 Apr, acceptance 30 Apr, conference 6-11 July, Athens.
- **What is judged.** Stage 1: hypothesis and methodology. The paper is "semi-complete"; experiments "may be missing or
  preliminary". Related work should cite only what is "directly tied" to the central problem. Stage 2 checks that the
  Charter -- the extensions the ACs require -- was carried out and that the work yielded insight. Negative results are
  not a reason to reject.
- **Rules that cause desk rejection.** Anonymity breaches: names, affiliations, acknowledgements, links to GitHub or
  YouTube, lab-specific references. Formatting changes. Hallucinated references, which are checked automatically. LLM use
  is allowed and its disclosure encouraged.
- **Practicalities.** Every author needs an OpenReview profile, and profiles without an institutional email can take up
  to two weeks to be moderated. The Stage 1 author list is "generally considered final".
- **Not yet published.** The review guidelines and the rebuttal template ("in due course"). Re-read the call in
  mid-November.

What this means for P1: the abstract is close to a registered report. Its product is the hypotheses, the protocol and
the analysis plan, plus enough preliminary numbers to show the pipeline runs and the protocol is measurable. Only seven
weeks separate the Charter from the final paper (26 Feb to 16 Apr). The full real-robot battery must therefore run in
December and January under the frozen protocol, not after the Charter arrives.

## Commit to parity, student-beats-teacher, and a simulator shown to rank like the robot

Five weeks carry three hypotheses, not six. Which of H1-H6 (RESEARCH.md 9.6) our data can carry by 1 December:

| hypothesis | carried by 1 Dec? | role |
|---|---|---|
| H1 parity | yes, if the ACT grasps by ~13 Nov | confirmatory; the precondition for H2 |
| H2 student beats teacher | teacher half now (no policy needed); student half in sim; real student only after H1's go | confirmatory; the headline |
| H3 exchange rate | yes, preliminary, from the data-mix sweep | exploratory |
| H4 robot card, H5 hand-back gate | simulation only | out; each is its own paper; keep recording their data |
| H6 no human in the loop | not on the robot by December | a design constraint, counted in the logs |
| **HV (new) the sim ranks checkpoints like the robot** | yes, from the same sweep | confirmatory; the method rests on it |

Proposed wording, to be frozen in a dated commit by **Thursday 5 November**:
- **H1, parity.** A student trained only on data the teacher produced reaches first-attempt grasp success within 10
  points of the real picker on the 20-cell static grid (30 mm cube). The data are privileged simulated episodes plus at
  most 100 real picker episodes, with no human demonstration, label or reset. Test: one-sided non-inferiority on the
  difference, paired by cell, alpha 0.05. Supported if the lower 95 % bound of (student - teacher) lies above -10 points;
  falsified if the upper bound lies below -10; otherwise inconclusive. N = 40-60 an arm at Stage 1 (preliminary), 120 at
  Stage 2 (~110 are needed at 90 % vs 90 % for 80 % power, computed).
- **H2, the student beats its real teacher where the teacher cannot see.** The cube is displaced 4 cm sideways after
  the picker's last perception (trigger T2 below). A student trained with at most 3 rounds of simulated DAgger has a
  higher first-attempt success than the real picker. In those rounds the privileged expert relabels the student's
  states, with displacements drawn in training. Test: one-sided superiority, alpha 0.05, sequential (STEP) up to 150
  an arm. The smallest effect of interest is 15 points; about 50 an arm find a 25-point gap, 110-130 a 15-point gap
  (computed). Secondary: success within 60 s including the picker's own retry, time to success, successes per minute
  (TRACE's lesson), and the same at T1 and at a 1 cm control. Falsified if the student is not ahead at N_max, or if it
  loses parity on still cubes.
- **HV, the simulator is a valid proxy.** Over at least 6 checkpoints spanning at least 30 points of real success, the
  sim grid ranks them as the real grid does: MMRV <= 0.10 and Spearman >= 0.8. Exact one-sided p < 0.05 needs
  Spearman >= 0.83 at n = 6 and >= 0.64 at n = 8 (computed). Falsified if MMRV > 0.10 or Spearman < 0.5; checkpoint
  choice and DAgger then move to the robot.
- **H3, exploratory.** The same sweep gives the number of simulated teacher episodes that buys the real success of one
  real teacher episode, with a bootstrap interval. Both come from the same planner, so the number measures the domain
  gap without the confound in published co-training, where human demos are mixed with scripted ones.

The story the three tell is one sentence: *the privileged simulated teacher can re-plan from any state because it
knows where the cube is; the real teacher cannot after its last look; a student taught by the first in simulation and
by the second on the robot inherits the first's reactivity and exceeds the second*. The student is expected to stay
below the privileged teacher. That is the honest frame, and it survives a reviewer who knows learning-by-cheating.

## The protocol: ~60 trials an arm, an automatic reset, a displacement after the teacher's last look, an evaluator outside the loop

**Numbers.** All rates come with Wilson 95 % intervals: at 80 % success, 20 trials give [58, 92] % and 50 give [67, 89] %.
Our teacher's 32 of 34 attempts (23 Oct) is [81, 98] %. Two-policy comparisons use STEP (RSS 2025,
[2503.10966](https://arxiv.org/abs/2503.10966)): sequential, error-controlled, up to 32 % fewer trials. Ranking several
checkpoints uses AnyRank (CoRL 2026, [2610.14603](https://arxiv.org/abs/2610.14603)). It takes the simulator's ranking as
a prior, keeps family-wise error control even when that prior is wrong, and needed up to 50 % fewer hardware evaluations
than Bonferroni-style methods; its code is released. Conditions run in randomised blocks over the same cells and the
same perturbation draws, and are compared paired. A plain same-seed re-run is the placebo: on frozen pi0.5, 43 % of
failures succeeded on a re-run alone ([2610.06921](https://arxiv.org/abs/2610.06921)).

**The automatic reset is the enabler.** The picker can already carry a held piece to a radius and J1 and let go
(`m3pick.carry_to`, `drop_at`). An evaluation trial ends at the student's lift (+2 s). The script then carries the cube to
the next cell; if the student did not grasp, the picker picks the cube where it lies. With looks before and after, a
trial takes ~60 s, so ~60 an hour. Stage 1 needs ~450 real trials (~8 h over three weeks); Stage 2 needs ~25 h more,
which fits December-January. Count every trial a human had to reset.

**The perturbation protocol**, modelled on LIBERO-Dynamic (named families, speed tiers, a fixed phase, latency stated)
and adapted to a cube on the floor and a teacher whose phases are known:
1. *Cells and cube.* The 30 mm picture cube on the printed sheet; the sim grid's cells, the same in sim and real.
2. *Families.* **P-base** (primary, automatic): at the trigger the base sidesteps by Delta, perpendicular to the arm's
   line to the cube, at ~5 cm/s; neither arm of the comparison is told. **P-push** (secondary, 20-30 trials): a guided
   pusher slides the cube by Delta on a console beep. **P-sim**: both forms in the simulator, plus speed tiers.
3. *Tiers.* Delta = 1 cm (a control inside the gripper's +-15.75 mm tolerance for a 30 mm cube), 2 cm (at the edge),
   4 cm (outside).
4. *Trigger, one rule for both arms, fired by the bridge.* T1: the commanded grip point first comes within r1 of the
   cube, before the picker's close look, which can still react. **T2 (headline):** within r2, the median distance at
   the middle of the picker's open-loop descent, measured on at least 20 recorded attempts before the freeze.
5. *Outcome.* Primary: first-attempt grasp (lifted at least 3 cm, gone from the floor at the after-look). Secondary:
   success within 60 s, time to success, realised displacement.
6. *Arms.* The real picker as it is; the student; the student plus a fixed-time retreat-and-retry; the privileged expert
   (sim only, the upper bound).
7. *Latency.* Both arms run at their real speed. State the student's chunk timing and the picker's look and plan times.

The base sidestep is ego-motion (the whole picture moves); a push moves only the cube. Report them separately. P-base is
also a real disturbance for a mobile manipulator, and it makes 50-150 trials a condition possible without a person at
the table.

**The evaluator sits outside the loop.** The picker's verdicts keep choosing which attempts enter the training data.
The reported numbers come from evaluators the loop cannot read: the simulator's true state in sim, and on the robot a
human audit of every evaluation video, blind to which student it is. The picker's agreement with the audit (Cohen's
kappa) is reported. RoboJev can be a third judge if the 5090 is free. This satisfies 2610.18842 and turns the referee
study (S2) into one table.

## The simulator earns the right to choose checkpoints only across six or more checkpoints that differ

Today the sim and the M3 agree only that the real-only ACT fails. On the robot it heads to J1 ~93-98 and in the sim to
~53-81 for the same cells (NEXT.md, 27 Oct). To test HV:
- **Checkpoints.** The data-mix sweep (real 20 / 80 x sim 0 / 1k / 4k), two training steps of one mix, and GR00T N1.7 if
  hekate is free: 6-8 checkpoints. All trained with at least 10 % sim data, PolaRiS's fix for policies that never saw
  the sim (earlier notes).
- **Metrics.** MMRV, as in SIMPLER: each policy's worst mis-ranking, weighted by its real gap, averaged. SIMPLER reached
  0.056 / Pearson 0.924 with visual matching against 0.375 / 0.308 for offline action error
  ([2405.05941](https://arxiv.org/html/2405.05941)). Also Pearson and Spearman with exact p-values, and **per-cell
  behaviour agreement**: J1 heading at 1.2 s and closest approach, sim against real. Both evaluators log these already,
  and they would have caught today's disagreement.
- **Noise.** At 40 real trials a checkpoint, two checkpoints whose true success is 10 points apart are mis-ordered ~19 %
  of the time; 20 points apart, ~4 % (computed). The real numbers can rank only checkpoints that differ by about 20
  points or more; MMRV discounts the rest.
- **The confound.** A checkpoint trained on more sim data looks better in sim for visual reasons alone. Report the
  correlation within one data mix as well as across mixes.
- **Bars from other work.** A splat twin reached r 0.90-0.94 across ACT, DP, pi0 and SmolVLA, while an Isaac Lab
  baseline reached 0.24 and 0.65 ([2511.04665](https://arxiv.org/html/2511.04665v2)). YUBI-Real2Sim reproduced 8 of 9
  real orderings ([2610.17521](https://arxiv.org/abs/2610.17521)). Nothing was found for a cheap servo arm with a wrist
  RGB-D camera and command-only joint state; that is itself a result worth one figure.

## The 5+1 pages

| section | pages | content |
|---|---|---|
| Title + abstract | 0.2 | e.g. "Can a student outgrow its classical teacher? Pre-registered tests on a low-cost mobile manipulator" |
| 1 Introduction | 0.7 | cheap arms and no human data; the privileged vs real teacher asymmetry; contributions: a zero-human-data loop, three registered hypotheses, a protocol, preliminary numbers |
| 2 Related work | 0.4 | only the directly tied: TRACE, SimPT, TILDE, VAMPS, HERO, Zero2Skill, DynamicVLA, LIBERO-Dynamic, SIMPLER/PolaRiS, 2610.06921, STEP/AnyRank |
| 3 Hypotheses | 0.6 | H1, H2, HV with metrics, N, tests and the three outcomes each; H3 named as exploratory |
| 4 System and method | 1.2 | the M3 Pro (no read-back, wrist RGB-D, Orin NX); the picker's phases and verdicts; the twin (board readings ~5 mm rms; servo gain 0.87-0.93, backlash 0.4-1.1 deg, lag 0.1-0.2 s -> a robot card per episode); the privileged expert and recorder; ACT, GR00T N1.7; DAgger in sim with displacements; the automatic reset. Fig. 1: the loop |
| 5 Evaluation protocol | 0.8 | grid, families, tiers, triggers, metrics, STEP / AnyRank / MMRV, the evaluator outside the loop, the robot-time budget. Table 1: conditions x N |
| 6 Preliminary results | 1.0 | teacher static and displaced, with intervals; sim expert yields; sim vs real grids for the checkpoints so far. Fig. 2: sim vs real success and per-cell heading; first DAgger round in sim |
| 7 Stage 2 plan and limitations | 0.3 | what the Charter would run; failure modes named |
| References | 1.0 | ~20, every one checked by hand |

Anonymity: name the commercial hardware, never the lab's machines, network or city; no links. One sentence discloses the
AI coding agent's role.

## Five weeks, one go/no-go on 13 November

| week | dates | work | output |
|---|---|---|---|
| 0 | Tue 27 Oct - Sun 1 Nov | Thursday meeting; topic 1's fixes into the next ACT run; the automatic reset; draft pre-registration; OpenReview profiles for every author; teacher on the static grid (N = 60, cubes placed by hand if the reset is not ready) | teacher static rate with CI |
| 1 | 2-8 Nov | **freeze H1/H2/HV (Thu 5 Nov, dated commit)**; teacher under P-base T2 x 4 cm (N = 50), T1 x 4 cm (30), 1 cm control (20); displacements in the sim expert; sweep trainings on Dionysos (no Isaac runs beside them) and hekate if free | the teacher half of H2 |
| 2 | 9-15 Nov | sim grids (60) and real grids (40) for each sweep checkpoint; **Fri 13 Nov go/no-go**: a checkpoint with >= 30 % first-attempt grasp on the real static grid (N >= 40) and heading within 10 deg of the cube on >= 80 % of cells | H1 preliminary, HV first points |
| 3 | 16-22 Nov | GO: sim DAgger rounds 1-2 with displacements; best student on the robot, static (40) and T2 x 4 cm (50); HV analysis. NO-GO: the fallback below | H2 preliminary |
| 4 | 23-29 Nov | write; two figures and one table from the logs; check every reference against its source; anonymity pass; the advisor reads it | full draft |
| 5 | 30 Nov - 4 Dec | buffer; **submit Thu 3 Dec** | submission |
| after | Dec - Jan | the full battery under the frozen protocol (H1 N = 120 an arm, H2 up to 150 with STEP, HV with 8 checkpoints) | ready before the Charter (26 Feb) |

**Fallback if the policy is still weak on 13 November.** RSS rejects weak hypotheses, not negative results:
- **A -- it grasps sometimes (10-30 %).** Keep H1, H2 and HV registered. Report H1 as inconclusive at this data scale,
  the teacher's envelope under displacement, the student's sim results after DAgger, and HV on the checkpoints we have.
  Add GR00T N1.7 as a second student for the Charter.
- **B -- it still does not head for the cube.** HV cannot be tested without spread, so submit only if the user accepts a
  thinner paper: the registered protocol, the teacher's envelope, the simulated student, and the diagnosis of why
  same-teacher data did not transfer (state shortcut, coverage). Otherwise hold P1 for CoRL 2027 (late May, date
  unverified) and give the five weeks to the policy itself. The user decides on 13 November.

## Decision table for our work

| hypothesis | data needed | by when | pass criterion | fallback |
|---|---|---|---|---|
| H1 parity | static grid: teacher 60; students 40 each (Stage 1); 120 each (Stage 2) | teacher 1 Nov; students 15 Nov; full Jan | lower 95 % bound of (student - teacher) > -10 points | report inconclusive or falsified with the diagnosis; GR00T N1.7 as a second student |
| H2 student beats teacher | teacher under P-base T2 x 4 cm (50), T1 (30), 1 cm (20); student in sim after DAgger 0-2 (60 a round); student real (50) | teacher 8 Nov; sim 22 Nov; real 22 Nov if go; full Jan | one-sided p < 0.05 by N_max 150 (STEP), effect of interest 15 points; Stage 1: estimate + Wilson CI | register it with the teacher half and the sim half; the real test goes to Dec-Jan and the Charter |
| HV sim ranks like the robot | >= 6 checkpoints x (60 sim + 40 real) + per-cell heading | 22 Nov | MMRV <= 0.10 and Spearman >= 0.8 | choose checkpoints on the robot with AnyRank; report the gap as a finding |
| H3 exchange rate (exploratory) | the same sweep | 22 Nov | none (estimate with a bootstrap CI) | leave it out of the abstract |
| H6 as a constraint | the episode logs | always | zero human demos, labels or resets in the training data, counted | -- |
| H4 robot card, H5 gate | keep recording cards and per-chunk verdicts | -- | not in Stage 1 | IROS'27 or later |

## Risks

1. **The ACT keeps ignoring the cube** (high). Topic 1 (the state shortcut, no augmentation, the one-heading average)
   must land in the next run. GR00T N1.7 on the 5090 is the second student. The go/no-go makes this risk explicit early.
2. **The sim does not rank like the robot** (medium-high). The sim's colours are lighter, one run in three renders
   black beside a training, and its grip is weaker at the far rim (71 % of simulated picks in the box). Mitigations: at
   least 10 % sim in every checkpoint, the per-cell behaviour check, the colour fix; if HV fails, choose checkpoints on
   the robot.
3. **"The teacher is a strawman"** (medium). An open-loop descent fails by design when the cube moves. Answer with the
   claim's scope (the student exceeds its own teacher with no new engineering), the T1 condition where the teacher can
   still react, its own retry counted, and one sentence on visual servoing as the classical alternative.
4. **Robot time and wear** (medium). ~450 trials before December, ~1,500 by February. The reset must work unattended;
   re-run the servo probe weekly (J2/J3 backlash drifts change the twin).
5. **Scoop** (medium-high). The pipeline is everywhere. An arXiv preprint after submission is allowed and timestamps
   priority at a cost in anonymity -- the user's call.
6. **Desk rejection** (low if checked). Anonymity (machine names, links), template changes, unverified references.
   Every 2026 arXiv ID gets opened before submission.
7. **Bandwidth** (medium). One student and an AI agent, with AGENT@ICSE (27 Nov) and HRI LBR (8 Dec) in the same weeks.
   RESEARCH.md section 7 already says to drop the quick wins first.
8. **Shared GPUs** (medium). hekate is shared and Dionysos crashed a training beside an Isaac evaluation on 27 Oct.
   Schedule trainings and evaluations, never both on Dionysos.
9. **The base sidestep's accuracy** (low-medium). Unmeasured over 2-4 cm. Read the realised displacement from the looks
   and report it.

## Conclusion

P1 survives the October flood only in a narrower form. The pipeline -- a privileged sim expert teaching a co-trained
policy on a cheap arm -- is done elsewhere and works. What is still open is whether a student taught by that loop
exceeds its own real teacher where the teacher is blind, and whether the simulator can be trusted to judge the student.
RSS 2027's Stage 1 asks for exactly that kind of falsifiable plan. Freeze three hypotheses by 5 November -- parity (H1),
student beats teacher under a displacement after the teacher's last look (H2), and the simulator ranking like the robot
(HV). Measure the teacher under the protocol now, and build the automatic reset first, because every number after it
depends on cheap trials. Decide on 13 November, from the real static grid, whether the abstract reports a student or
only its teacher. The robot card, the hand-back gate and "no human in the loop" stay out of these five pages.
