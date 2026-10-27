# Scoop check for P1, 21-27 October 2026 (with items the 21 October check missed)

Scope: what appeared, or was missed, around P1 ("a classical picker teaches a learned policy") since the scoop check in
RESEARCH.md 9.2 (1-21 October). Six neighbourhoods: experts that generate data (classical, scripted, privileged,
planner); DAgger with a non-human expert on real robots; self-improvement without human labels; students that beat
their teacher; policies conditioned on measured hardware parameters (the robot card); learned gates that hand
control to a classical controller. A seventh section covers moving-object benchmarks, because H2's protocol depends
on them. Method: the arXiv API sorted by submission date for cs.RO with keyword sets (DAgger; self-improvement /
autonomous / without human; co-training + sim; privileged / scripted expert; distillation + planner; demonstration
generation; reset-free / data flywheel; SO-100/101 / low-cost arm; embodiment descriptor / system identification /
hardware parameters; fallback / hybrid controller; camera pose conditioning; moving / dynamic objects; real-to-sim
evaluation; checkpoint selection; grasp + teacher), about 20 web searches, then the arXiv abstract pages (and two
HTML full texts) of everything load-bearing. Dates are arXiv v1 dates. Unless marked "full text", a claim rests on
the abstract only. The web-search quota ran out on 27 October; nothing after that was searched. "Not found" means not
found by these queries, which is not proof that nothing exists.

Labels: **[REAL-LC]** measured on a real low-cost arm; **[REAL-other]** measured on another real robot; **[SIM]**
simulation; **[CLAIM]** stated without a number in the source; **[2nd]** secondary source, not checked against the
primary. Others' numbers are theirs, not ours.

## 1. Experts that generate the data: classical, scripted, privileged, planner

### Takeaway
"A scripted expert in Isaac Sim teaches ACT on a cheap arm, co-trained with a few real episodes" is no longer a
contribution: a community SO-101 project does exactly that (Isaac Sim 6.1, a state-machine expert, DART noise,
DAgger takeovers, ACT 19/21 real against 6/21 real-only). SimPT (21 Oct) uses a CuRobo planner to make the
recovery data for VLA post-training. Still not found: a classical teacher that is at once the generator, the
relabeller, the referee and the reset on the real robot, and any paper that compares the student with its teacher.

### Cited findings
- **SimPT** ([2610.29161](https://arxiv.org/abs/2610.29161), 21 Oct): an automated sim framework for intervention-based
  VLA post-training. A progress graph on privileged sim state decides when to intervene; a CuRobo-based expert
  recovers from saved snapshots of the policy's states. Seven RoboTwin tasks, four post-training methods (RECAP,
  HG-DAgger, DPO+SFT, FlowPRO). Recovery attempts -75 %, recovery success 93.4 -> 98.9 % [SIM]; sim success 76.50 ->
  79.36 % [SIM]; real Ethernet-cable insertion 72 -> 96 % over 25 scenarios [REAL-other]; robot and VLA not named
  in the abstract. The planner acts in the simulator, not on the robot, and there is no student-vs-expert comparison.
- **Community SO-101 real2sim2real project** ([GitHub README](https://github.com/AadhavSivakumar/real2sim2real-so-arm101),
  undated; it uses Isaac Sim 6.1, released 10 Sep 2026): a phone-scan splat as the background in Isaac Sim 6.1 +
  Isaac Lab 3.0; a state-machine expert (pre-grasp, descend, grasp, rise, transport, release); per-episode domain
  randomisation; DART-style noise on executed actions with clean labels; DAgger-style takeovers in which a trained
  policy drives and the expert takes over on failure. 40 teleoperated real demos; real episodes upweighted x19 to
  ~25 % of frames; ACT (chunk 50, 30 executed) 80k steps on one H100. Real: co-trained **19/21**, real-only **6/21**,
  labelled by the operator, cube placed by hand, no fixed grid [REAL-LC]. Sim: 44/50 [SIM]. The real-only policy
  failed where real demos were thin (4 demos near the failing spot vs 336 sim episodes) -- the same coverage failure
  our real-only ACT shows. Not a paper; one operator's labels; no confidence intervals.
- **Real2Sem2Real** ([2610.28880](https://arxiv.org/abs/2610.28880), 20 Oct, abstract via the arXiv API): one human RGB
  video -> simulated robot demos with 3D semantic augmentation, 2,000+ episodes in ~2 h; a VLA trained on them
  "matches or exceeds" teleop-trained policies on two real robots [CLAIM].
- **TILDE** (re-checked, [2610.25376](https://arxiv.org/abs/2610.25376), 18 Oct): the teacher is a learned embodied
  foundation model; "AutoDAgger" lets a learned reward model trigger the teacher; recovery collection >2x faster than
  a human in the loop; up to 4.1x over behaviour cloning on three real tasks [REAL-other]; robot not named in the
  abstract; no student-vs-teacher number.
- **VAMPS** (re-checked, [2610.05331](https://arxiv.org/abs/2610.05331), 4 Oct): MPPI on a real Flexiv Rizon 10S
  records RGB and state; ACT is trained offline on that fixed set (pick-and-place, whiteboard erasing). The iterative
  "policy warm-starts the planner" loop runs only in simulation. The abstract gives no success numbers [CLAIM].
- **ENPIRE** ([2606.19980](https://arxiv.org/abs/2606.19980), 18 Jun 2026; missed by earlier checks;
  [project page](https://research.nvidia.com/labs/gear/enpire/)): coding agents improve real policies on a YAM arm
  with automatic resets and an automatic verifier (for zip-tie insertion: a detector, a segmenter and a geometric
  rule fused into a binary reward, "without human judgment"); 99 % pass@8, where pass@8 is up to 8 in-context retries
  within one rollout [REAL-other]. The judge is partly learned; nothing is distilled from a classical teacher.

### Inferences
- The pipeline (privileged sim expert -> co-trained ACT/VLA) is commodity by now. P1's contribution must sit in what
  the pipeline lets us test: whether the student exceeds its own real teacher, and whether the simulator can be
  trusted to pick checkpoints. Write that into the first paragraph of the abstract.
- The SO-101 project's numbers make our expectations concrete: co-training rescues the coverage failure (6/21 ->
  19/21) on a near-identical setup. If our co-trained ACT still "heads one way" after topic 1's fixes, the cause is
  more likely the training recipe (state shortcut, augmentation) than the idea of co-training.

### Gaps
- The SO-101 project's date, its exact policy inputs and its per-region numbers are not given in its README.
- SimPT's robot, VLA and real trial count are not in the abstract.

## 2. DAgger with a planner, scripted or privileged expert on real robots

### Takeaway
Of the DAgger papers on arXiv since 1 October, the real-manipulator ones use humans as experts (I3L, RoboPrompt,
PathTime-VLA, BlenDAgger, Reward-DAgger, EqC). The non-human experts relabel in simulation (SimPT, TRACE, CALM,
privileged noise distillation). The one real-robot student-vs-teacher number found, TRACE, has the student **below**
the teacher on success (90 vs 95 %) and **far above** it on time (67 vs 193 s). That is the shape H2 should expect.

### Cited findings
- The arXiv API search "DAgger" in cs.RO returned 115 results; the first 39 by date (to 19 Jun 2026) were read
  ([query](http://export.arxiv.org/api/query?search_query=abs:DAgger%20AND%20cat:cs.RO&sortBy=submittedDate&sortOrder=descending&max_results=40)).
- **TRACE** ([2609.38857](https://arxiv.org/abs/2609.38857), 30 Sep): a privileged teacher on a digital twin makes a
  fixed nominal rollout; a recurrent student corrects deviations from partial observations; BC then DAgger with teacher
  labels on the student's states. Sim, 511 scenes: student 90.7 %, nominal replay 43.4 %, closed-loop teacher 96.7 %;
  at a matched 26,373 labels, student-state supervision 87.8 % vs expert-only cloning 66.7 % [SIM]. Real UR5e:
  student 90.0 % vs teacher 95.0 %; 67.3 s vs 192.7 s; the teacher's 16.8 sensing retractions a trial avoided
  [REAL-other].
- **CALM** ([2609.29017](https://arxiv.org/abs/2609.29017), 24 Sep, abstract via the API): privileged sim experts and a
  stage router generate demos, DAgger refines a student; 76.2 % (762/1000) [SIM], 73.3 % (22/30) real [REAL-other].
- **Privileged noise distillation** ([2610.07745](https://arxiv.org/abs/2610.07745), 6 Oct, abstract via the API): a
  privileged sim teacher plus relabelled demos, BC and one DAgger round; 83.3 % vs 34.4 % at 5 mm noise, Franka
  zero-shot [SIM / REAL-other mixed in the abstract].
- **ASTEROID** ([2610.30105](https://arxiv.org/abs/2610.30105), 21 Oct): a privileged expert labels actions conditioned
  on the student's own on-policy history, so the student learns to explore, which the expert never needs; +40 % real
  success over standard distillation on vision-denied manipulation [REAL-other]; robot and trial count not given.
- Human experts, real robots: I3L ([2610.17748](https://arxiv.org/abs/2610.17748)), RoboPrompt
  ([2610.10534](https://arxiv.org/abs/2610.10534), +15.5 % for pi0.5 over 2-3 rounds), BlenDAgger
  ([2609.37599](https://arxiv.org/abs/2609.37599)), Reward-DAgger ([2610.04054](https://arxiv.org/abs/2610.04054)),
  EqC ([2610.18790](https://arxiv.org/abs/2610.18790)) -- abstracts via the API.

### Inferences
- "DAgger in simulation with a privileged planner" is now standard (SimPT, TRACE, CALM, our earlier notes' GSWorld and
  Diff-DAgger). Relabelling on the real robot with the real classical teacher is still not found, but for Stage 1 it is
  cheaper to state that relabelling happens in simulation and that the real teacher contributes demonstrations,
  takeovers and verdicts.
- TRACE gives P1 a precedent for "the student beats the teacher on throughput, not on success". H2 should report time
  to success and successes per minute beside first-attempt success.

### Gaps
- No real-robot DAgger paper with a classical expert relabelling on the hardware was found.

## 3. Self-improvement without human labels

### Takeaway
October brought a wave of "recursive self-improvement" (RSI) papers -- at least 15 robot ones since 1 October -- and
two position papers. Almost all use LLM/VLM agents or learned judges. Under the L0-L5 scale of
[2610.18842](https://arxiv.org/abs/2610.18842), P1's loop is **L1** at best, and only if it is scored by an evaluator
the loop cannot read or train on. RoboJev (24 Oct), an open VLM judge with 95.2 % balanced accuracy on held-out
SO-101 episodes, makes "VLM judges are weak" (FailBench's 0.77) an outdated argument for SO-101-like data.

### Cited findings
- **Who Improves the Robot?** ([2610.18842](https://arxiv.org/abs/2610.18842), 13 Oct; full text read in part): L0 = no
  retained change on its own, including "workflows whose retained changes depend on people intervening or labeling";
  L1 = capability-level updates on one task with a fixed improvement procedure; L2 adds updates to the procedure; L3/L4
  extend L1/L2 across tasks; L5 needs an independent evaluator to show each update gains more than the last. Its
  recommendations include scoring with "an independent evaluator that the improvement procedure cannot read, train on,
  or modify", comparing generations from the same start and budget, and recording whether feedback is real, simulated
  or imagined.
- **Toward Recursive Self-Improvement in Physical AI** ([2610.29677](https://arxiv.org/abs/2610.29677), 21 Oct): a review;
  it finds that "the reviewed physical studies have not isolated gains caused by recursion" (paraphrase of the
  abstract) and names conditions under which gains carry forward. No levels, no experiments.
- **RoboJev** ([2610.33718](https://arxiv.org/abs/2610.33718), 24 Oct): RoboJev-27B, a LoRA on an open 27B VLM, judges
  success from 10 frames, trained on 52,598 episode records; held-out SO-101: 95.2 % balanced accuracy, AUROC 0.993;
  unseen labs/robots: AUROC 0.890 on 459 episodes; ~0.26 s an episode on an H100; open weights [REAL-LC for the data
  it judges]. Where its training labels come from is not stated in the abstract.
- Learned or agentic judges on real arms (abstracts via the API): ReTrace ([2610.15814](https://arxiv.org/abs/2610.15814),
  Franka, plug insertion 0 -> 80 %, VLM reasoning over physical outcomes, no weight updates); PhysEvo
  ([2610.08995](https://arxiv.org/abs/2610.08995), PiPER 84 % over 25 trials); RoboRSI
  ([2610.12424](https://arxiv.org/abs/2610.12424), 104 rounds of household clean-up); AdaRSI
  ([2610.20474](https://arxiv.org/abs/2610.20474), GR00T N1.5 in RoboCasa365, 4.6 -> 13.8 % [SIM]); MobiAgent and
  Robo-COP as in RESEARCH.md 9.2.

### Inferences
- Do not sell P1 as "self-improvement" or "RSI": the field is crowded and its own reviewers now ask for isolated
  recursion gains. Sell it as a teacher-student question with a pre-registered test.
- P1's referee must be split: the picker's verdicts may label the training data, but the reported numbers need an
  evaluator outside the loop -- the simulator's true state in sim, and on the robot a human audit of every evaluation
  video (optionally RoboJev as a third judge). See `hypotheses_and_evaluation.md`, section 6.

### Gaps
- RoboJev's label source and licence are not given in the abstract; it needs a 5090-class GPU to run.

## 4. Students that beat their teacher

### Takeaway
No real-robot case was found where a student policy beats a classical teacher on success. Students win (a) on another
metric (TRACE: 2.9x faster), (b) in simulation when their demos were improved beyond the teacher's (outcome-sensitive
motion search), or (c) where they learn a capability the teacher does not need (ASTEROID). One September paper
argues the opposite direction: keeping the expert and learning only its perception beats pixel imitation. So H2 is
plausible only outside the real teacher's competence envelope, where a privileged simulated teacher is competent.

### Cited findings
- TRACE, above: student below teacher on success, above on time [REAL-other].
- **Outcome-Sensitive Motion Search** ([2609.29020](https://arxiv.org/abs/2609.29020), 24 Sep): teacher rollouts refined
  by local search and failed conditions repaired; the imitation policy "outperform[s] the privileged RL teacher" in
  catching success and impact [SIM, CLAIM: no numbers in the abstract].
- **Estimate, Don't Imitate** ([2609.34018](https://arxiv.org/abs/2609.34018), 27 Sep): keeping the state-based expert and
  learning a state estimator beats direct pixel-to-action imitation on five tasks; 76 % on a real Panda [REAL-other].
- OPTIMUS (CoRL 2023, [proceedings](https://proceedings.mlr.press/v229/dalal23a/dalal23a.pdf)) is the usual citation for
  planner-to-visuomotor distillation; its abstract does not compare student and expert.

### Inferences
- Phrase H2 as "the student exceeds its *real* teacher where the real teacher cannot perceive, because it learned from
  a *privileged* teacher that can" -- the learning-by-cheating structure. The student is expected to stay below the
  privileged simulated teacher; that is the honest frame and it survives review.

### Gaps
- No real-robot study found that measures student vs classical teacher on a pre-registered perturbation battery.

## 5. Policies conditioned on measured hardware parameters (the robot card)

### Takeaway
Still not found for manipulation. The nearest new items condition on structured or nominal descriptions (a humanoid
tracking policy, morphology-aware policies, posed camera views), not on a measured fault vector.

### Cited findings
- **MimicAll** ([2610.33508](https://arxiv.org/abs/2610.33508), 24 Oct): structured embodiment descriptor for
  cross-humanoid motion tracking; 99.4 % vs 42.9 % for a single-body specialist on randomised G1 bodies [SIM].
- **GaugeBench** ([2610.07597](https://arxiv.org/abs/2610.07597), 6 Oct): re-describing the same robot (axis direction,
  zero, ordering) drops morphology-aware policies' return from 4030.6 to 51.6; training across conventions keeps 80.6 %
  under axis reversal vs 3.6 % [SIM]. A warning that description-conditioned policies are brittle.
- **VersaCamVLA** ([2610.12451](https://arxiv.org/abs/2610.12451), 8 Oct): a VLA conditioned on posed RGB views
  (extrinsics); no numbers in the abstract [CLAIM].
- Backlash is randomised, not conditioned on, in sim-to-real hand work ([2601.02778](https://arxiv.org/abs/2601.02778)).
- Earlier (RESEARCH.md 9.2-9.3): the self-compensating VLA (2609.37334), MERL's embodiment-aware pi0.5 (2603.00182).

### Inferences
- The robot card stays open, but it is a second paper (IROS'27), not part of a 5-page Stage 1 abstract. Keep recording
  the card with every simulated episode (already done).

## 6. Learned gates that hand control to a classical controller

### Takeaway
Not found. The closest new item, Backtrack (23 Oct), puts a failure head on a frozen VLA, trained on outcome-labelled
rollouts, but its fallback is a scripted retrace followed by the same VLA, and it is simulation only. A fixed-time
trigger with a scripted retreat is as good as a learned corrector on a frozen pi0.5 (2610.06921). H5 stays open but
now has two baselines it must beat.

### Cited findings
- **Backtrack** ([2610.32941](https://arxiv.org/abs/2610.32941), 23 Oct): failure head of 0.48-0.88 % of the VLA's
  parameters reading its internal features; on a predicted failure, pause, retrace recent motion, hand back to the
  unchanged policy; LIBERO-Long failures -19 to -48 %, success up to +7.4 points; all twelve SmolVLA configurations on
  MetaWorld beat the plain policy [SIM]. SmolVLA, pi0.5, pi0-FAST. No AUROC or false-alarm rate in the abstract.
- **SkipVLA** ([2609.20648](https://arxiv.org/abs/2609.20648), 17 Sep): a classical planner does free-space motion and a
  VLA only the contact-rich skills, split by phase (not a learned failure gate); up to 2.5x faster at the same success
  on 13 LIBERO tasks and three pick-and-place tasks on a real YAM arm [SIM / REAL-other].
- **Does a learned corrector beat a simple retreat?** ([2610.06921](https://arxiv.org/abs/2610.06921), 2 Oct): frozen
  pi0.5 on four RoboTwin tasks, 3,888 paired episodes with a same-seed re-run as a placebo; +13.5 points (95 % interval
  about +9.4 to +17.7) on one task, nothing detectable on three; 43.2 % of failures succeed on a plain re-run; a
  fixed-time trigger plus a scripted return is not detectably different from the learned pipeline [SIM].
- A symbolic supervisor with a fail-safe backup plan for a learned dynamical system
  ([2610.19815](https://arxiv.org/abs/2610.19815), abstract via the API) [CLAIM: no numbers].

### Inferences
- If H5 is ever tested, the paired design with a same-seed re-run placebo from 2610.06921 is the template, and the
  baselines are: fixed-time hand-back, scripted retreat-and-retry, and a Backtrack-style head.

## 7. Moving-object benchmarks (for H2's protocol)

### Takeaway
LIBERO-Dynamic (17 Oct) is the protocol to copy: three trajectory families, speeds 3/6/9 cm/s, a fixed motion phase,
inference latency calibrated per model. pi0.5 falls from 96.9 % static to 57.8 % dynamic. It reports no confidence
intervals, which P1 can add.

### Cited findings
- **LIBERO-Dynamic** ([2610.24579](https://arxiv.org/abs/2610.24579), 17 Oct; full text read via
  [HTML](https://arxiv.org/html/2610.24579)): linear (three waypoints), Catmull-Rom (3-8 waypoints) and zigzag (5-8
  waypoints) paths; 0.03 / 0.06 / 0.09 m/s; motion starts at a fixed control step (demos: step 30; evaluation: the scene
  advances 40 steps before the first observation); the target rides an invisible platform through contact friction;
  latency measured end-to-end on an RTX 5090, median frozen and quantised to control steps at 20 Hz; 40 tasks x 3
  families x 5 paths x 3 speeds = 1,800 episodes a configuration. Static -> all dynamic: pi0.5 96.9 -> 57.8, FastWAM-Joint
  98.5 -> 40.6, VLA-JEPA 97.2 -> 35.7, pi0 94.2 -> 31.5, OpenVLA-OFT 97.1 -> 27.3 [SIM]. The best horizon K* is picked on
  the evaluation episodes (an oracle). Physical study: Franka FR3 at 15 Hz, 50 trials a condition [REAL-other]. No CIs,
  no significance tests.
- **AstroBench** ([2610.31064](https://arxiv.org/abs/2610.31064), 22 Oct, abstract via the API): a cube-specific ACT
  captured 6/10 moving vs 8/10 static targets; frozen pi0.5-DROID 4/200 moving vs 2/200 static.
- Others in the API list (abstracts): TrajWAM ([2610.25763](https://arxiv.org/abs/2610.25763)), MotionForge
  ([2609.25689](https://arxiv.org/abs/2609.25689), 40 tasks, 11 motion patterns [SIM]), DSDyn-VLA
  ([2609.39198](https://arxiv.org/abs/2609.39198)), SECD ([2609.23305](https://arxiv.org/abs/2609.23305), cup from a
  16 m/min conveyor 91 % [REAL-other]), Real-Time EXPO-FT (2609.18207, in RESEARCH.md).

### Inferences
- Our "moved mid-reach" battery should name its families (a single lateral step, not a continuous path, is enough for
  a cube on the floor), its displacement and speed tiers, and its trigger phase, the way LIBERO-Dynamic does, and add
  Wilson intervals and a sequential test.

## 8. What P1 still owns (27 October)

| claim | status | nearest |
|---|---|---|
| a privileged sim expert generates data for a cheap arm; co-trained ACT | taken | SO-101 community project; NVIDIA course; Genesis ACT (earlier notes) |
| a planner relabels / recovers the student's states in sim | taken | SimPT, TRACE, CALM, GSWorld |
| the real classical teacher is generator, referee and reset on the robot, zero human labels | open | ENPIRE (learned judge), TILDE (learned teacher) |
| the student beats its real teacher on a pre-registered perturbation battery | open | TRACE (time only), LIBERO-Dynamic (no teacher) |
| the simulator ranks checkpoints of a cheap-arm policy as the robot does | partly open | SIMPLER, PolaRiS, YUBI-Real2Sim (not cheap arms, not wrist RGB-D) |
| same-teacher sim-vs-real exchange rate | open, low novelty | co-training papers mix human real and scripted sim demos |
| robot card, hand-back gate | open | section 5, section 6 |
