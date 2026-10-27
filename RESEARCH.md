# Research and demo program -- what is worth a paper, what is worth showing

Written 2026-10-05 from seven parallel literature surveys: VLAs and self-generated data; open-set
perception and teaching new objects; 3D maps, splats and dynamic object memory; multi-robot search
and localization; voice, LLM agents and AI-built robots; precise picking on cheap arms; venues,
deadlines and demo craft. It re-checks [PLAN.md](PLAN.md)'s research program (tracks A-E, L,
S1-S5, swept on 2026-09-29) against what was published up to 5 October 2026, ranks what is worth
writing up, and plans what to show professors and scientists. Nothing in it has been run.

**Updated 2026-10-21:** [section 9](#9-october-update-2026-10-21-the-scoop-check-two-new-ideas-scan-to-skill)
adds the October scoop check, two new ideas for P1 (a measured robot card, a hand-back gate),
scan-to-skill from the robot's own scan, P1's revised hypotheses and four additions to the
simulator plan.

**Updated 2026-10-27:** [section 10](#10-what-still-needs-research-2026-10-27) ranks what still needs
research -- where our results contradict the plan, where a deadline forces a commitment, where a
choice is expensive to undo -- and what does not.

How sure: every paper below was checked against its arXiv abstract page, project page or official
call. The load-bearing ones were re-checked by hand on 2026-10-05: HERO, OpenRUA, HEROIC, Project
Fetch phase 2, the RSS 2027 call, the CoRL EEAI workshop and challenge, and ICSE SEIP. The surveys'
web-search quota ran out partway, so later checks went through arXiv abstracts. "Not found" means
not found there, which is not proof that nothing exists. "est." and "unverified" mean what they say.

## Bottom line

**What makes this lab unusual.** Reviewers rarely see these:
1. **A picker that scores itself.** The classical picker judges its own attempts (gripper
   readback, the object gone from the table) at ~21 s an attempt. That means free labels and
   large trial counts at no cost.
2. **Two different robots in one arena.** Their arms, cameras and readback differ. The arena is
   fenced and its map is checked to 1 cm.
3. **Everything is measured.** Every stage is timed, a browser console drives both robots, and
   voice works in three languages.
4. **The build itself is on record.** Dated commits record one student and an AI coding agent
   building the whole stack in 3.5 weeks.
5. **Nobody publishes on Yahboom robots.** An arXiv search for "Yahboom" finds one unrelated
   security paper, so these could become the reference platform. The cost: every paper must
   justify the hardware to reviewers who have never seen it.

**Six papers are worth writing** (table in section 1). In order:
1. P1, D: a picker that teaches a VLA.
2. P2, E: a map that knows who moved what.
3. P3, A: teach a new object and pick it within a minute.
4. P4, B with L and S5: two robots searching with arm cameras.
5. P5: the AI-agent co-development case study.
6. P6: Slovak/Czech voice on the robot, with a stop lane that bypasses every model.

**The deadline that matters most** is RSS 2027 Stage 1, **4 Dec 2026**. It is a 5+1-page extended
abstract. It wants a developed problem, hypotheses and method; preliminary experiments are fine.
"Rigorously falsifying a hypothesis" counts as a contribution, so a negative result is not a
reason to reject. That fits D exactly: whether the student beats the teacher is a falsifiable
hypothesis.

**The one change to PLAN.md I recommend.** D's novelty window is closing. HERO (Jul 2026),
Zero2Skill (Jul 2026) and DynamicVLA (NeurIPS 2026) each took part of it. Start D's data recorder
on the X3 now, in parallel with B on the M3, instead of after A. The X3's picker is stable, its
data loop can run unattended overnight, and the arena work runs on the M3 anyway. This changes the
agreed B -> A -> D order, so it is a question for you (section 8), not a decision.

**The show.** One continuous run: a visitor says a command in Slovak, both robots search the arena
as the fog of war drains, a new object is taught in 30 seconds, the learned policy chases a cube
the visitor pushes while the classical picker fails on screen beside it, and the map flags who
moved what (section 4).

## 1. The portfolio, ranked

| # | paper | track | still open in Oct 2026? | to first numbers | first venue | scoop risk | demo value |
|---|---|---|---|---|---|---|---|
| P1 | Teacher, relabeller, referee: a classical picker trains a VLA with no human labels | D (+S2) | partly: the generic claim is taken, the core below is open | 4-6 weeks | RSS'27 Stage 1, 4 Dec | medium-high | very high |
| P2 | Who moved it? Object permanence from the robots' own action log | E | yes, clearly | 4-6 weeks | IROS'27 (1 Mar) or RA-L | medium-high (6-12 months) | high |
| P3 | Show-and-pick: from "Teach" to the first pick in under a minute, on the robot | A | the combination is open | demo 1-2 weeks, paper ~6 | IROS'27 / RA-L | medium | very high |
| P4 | Where should the arm look? Two-robot search with measured detection models | B (+L, S5) | yes, on real robots | 6-10 weeks (needs AMCL + Nav2) | IROS'27 / RA-L | medium | high |
| P5 | Two robots in 25 days: an AI coding agent with hardware access as co-developer | -- | yes | 2-3 weeks of analysis | AGENT@ICSE, 13 Nov (was given as 27 Nov) | medium-high | medium (a talk) |
| P6 | "Stop" means stop: on-robot voice commanding in Slovak/Czech, cascaded vs speech-native | M3/S3 | yes | 2-4 weeks | HRI'27 LBR 8 Dec, demo 20 Nov | medium | high |

The smaller ones fold into these or wait:
- **S2, physical vs VLM success judges:** it becomes P1's referee.
- **S1 and S4, cheap RGB-D vs foundation depth for tiny objects:** an RA-L letter or a workshop.
- **L, the 180-degree mirror trap:** a section of P4, or ECMR alone.
- **S5, what localization error costs a grasp:** a section of P4.
- **C, watcher and fetcher:** later, with a local partner.
- **Millimetre picking on deadband servos:** a workshop paper plus a released tool.
- **A platform paper on the two robots:** later, in RA-M.

## 2. The papers

### P1 -- Teacher, relabeller, referee (track D)

**Claim.** A precise classical picker on a sub-1k-euro arm closes the whole VLA loop with zero
human labels. It generates the demonstrations, relabels the states the student visits (DAgger) and
judges success. After a few relabelling rounds the student beats its teacher where detect-then-move
cannot work: a cube moved mid-reach, or a new instruction.

**What is taken and what is open:**

| closest work | what it does | what it does not |
|---|---|---|
| HERO, "Practice Makes Policies" ([2607.26809](https://arxiv.org/abs/2607.26809), 29 Jul 2026) | zero human demos; a VLM picks primitives on Depth Anything 3 geometry; 664 episodes with automatic reverse-task resets (84.3 %); trains pi0.5 (Franka, four RealSense, per the survey) | no precise classical metrology, no DAgger relabelling, no human-equivalence curve, no cross-embodiment, not low-cost |
| Zero2Skill ([2607.14047](https://arxiv.org/abs/2607.14047), 15 Jul 2026) | an LLM agent drives SAM 3 + AnyGrasp + IK on a dual-arm PiPER with a VLM verifier; pi0.5 on 50 episodes reaches 80 %, equal to teleoperation | a human steps in when retries run out; never claims the student beats the generator |
| DynamicVLA ([2601.22153](https://arxiv.org/abs/2601.22153), NeurIPS 2026) | an automated demonstrator with tracked object state collects 2K real episodes (Franka, PiPER) | its demonstrator already handles moving objects, so student-beats-teacher is never tested |
| pi*0.6 Recap ([2511.14759](https://arxiv.org/abs/2511.14759)) | advantage-conditioned VLA from autonomous runs | success labels from human raters, human corrections |
| TiPToP ([2603.09971](https://arxiv.org/abs/2603.09971), Mar 2026) | a modular planner beats pi0.5-DROID with no robot data | nobody has distilled such a teacher into a VLA |
| Real-Time EXPO-FT ([2609.18207](https://arxiv.org/abs/2609.18207), 16 Sep 2026) | 42 -> 97 % on dynamic tasks, including dynamic picking, with 10 min of online data | the bar for "chase the cube" |

Every 2026 real-robot VLA DAgger paper found uses a human expert ([SOP](https://arxiv.org/abs/2601.03044),
[TimelyDAgger](https://arxiv.org/abs/2609.33157)). So does LeRobot v0.6's DAgger mode: the expert
is a human on a leader arm. Four things are still open:
- (a) a classical expert that is generator, relabeller and judge at once;
- (b) student beats teacher on a pre-registered battery of dynamic and novel cases;
- (c) the exchange rate between automatic and human demos on a cheap arm;
- (d) cross-embodiment trained on automatic data only (X3 -> M3).

**Hypotheses for the RSS Stage 1 abstract.** Each is falsifiable (revised on 2026-10-21 in
[section 9.6](#96-p1s-hypotheses-revised)):
- **H1, parity.** Trained only on the picker's data, the policy comes within 10 points of the
  picker on the 25-position grid.
- **H2, the student beats the teacher.** After k relabelling rounds the policy beats the picker on
  the moved-mid-reach battery, which the picker by construction fails or restarts.
- **H3, the exchange rate.** N automatic demos are worth M teleoperated ones, read off an
  iso-performance curve with diversity knobs: placement, noise, perturbations.
- **H4, transfer.** Pre-training on X3 automatic data cuts the M3 demos needed.

**What the survey forces into the design:**
- **A privileged teacher camera.** The X3's homography is valid only at its look poses, so the
  expert cannot re-perceive from the arbitrary states DAgger visits. Fix: a fixed overhead webcam
  above the X3's table, registered to the board with the same tool as `calibrate_table.py`. It
  knows where the cube is at all times; the student sees only the wrist camera and the joints. The
  same camera is ground truth and the demo's wide shot. This needs a purchase (section 8).
- **Noise injected while recording** (DART, [1703.09327](https://arxiv.org/abs/1703.09327)), so
  the data covers recovery states.
- **Hindsight instructions for free.** The picker knows the colour, size and position of what it
  picked, so "the blue one" and "the one nearest the box" are labelled automatically.
- **Model choice.**
  - Warning on SmolVLA: on MolmoAct2's multi-task SO-100/101 evaluation it scored 2.3 %, against
    pi0 at 45.3 % and MolmoAct2 at 56.7 % ([2605.02881](https://arxiv.org/abs/2605.02881)). Its
    ONNX export also dropped LIBERO-Spatial success from 70 % to 41 %
    ([2609.14146](https://arxiv.org/abs/2609.14146)).
  - So the main policies are served from the 5090 with LeRobot's async inference and real-time
    chunking ([docs](https://huggingface.co/docs/lerobot/rtc)):
    - pi0.5, LoRA through openpi (over 22.5 GB, so it fits a 5090);
    - X-VLA, 0.9B, already in LeRobot ([2510.10274](https://arxiv.org/abs/2510.10274)).
  - SmolVLA through [vla.cpp](https://arxiv.org/abs/2606.08094) is the on-robot variant.
  - Re-run closed-loop evaluation after every export step.
- **The scoring.** Gripper readback plus the table emptying, plus the overhead camera. A VLM judge
  and a learned reward model (Robometer-4B, in LeRobot v0.6) run beside them as alternative
  referees. That is study S2 inside P1.

**The S2 sub-result** is worth a workshop paper on its own:
- [FailBench](https://arxiv.org/abs/2609.03611) (3 Sep 2026): 13 detectors on 2,197 attempts, the
  best at 0.77 balanced accuracy, all biased towards "success". It used no non-visual signals and
  no mobile manipulation, and names both as gaps.
- The closest prior to S2 is the Physical Agentic Loop
  ([2604.07395](https://arxiv.org/abs/2604.07395)): a gripper watchdog, then a VLM check. The same
  group is close, so S2 is medium-high scoop risk.

**Experiment.**
- 25-position grid x {static, moved mid-reach, new instruction} x DAgger rounds 0-3, with Wilson
  intervals.
- An iso-performance curve: 50/100/200/400 automatic demos against 25/50/100 gamepad-teleoperated
  ones.
- X3 -> M3 with relative end-effector actions.

**Headline figure:** success against DAgger round for static, moved and new-instruction trials,
with the classical picker as a flat line.

**Effort:** 4-6 weeks to Stage 1-grade preliminary numbers:
- a LeRobot robot class for the X3 (~1 week);
- the recorder, overhead camera and auto-reset (~1 week);
- the first fine-tune (days);
- the first battery.

Two risks stay from PLAN.md: the servo bus at 10+ commands a second, and WiFi gaps (a watchdog
holds the arm).

**Demo:** "chase the cube". A visitor pushes the cube with a stick mid-reach. The policy re-aims;
the classical picker, on the same push, misses or restarts. Shown split-screen with the Timing card.

### P2 -- Who moved it? Object permanence from the robots' own action log (track E)

**Claim.** An object map whose state per object is existence, pose and provenance. The robots'
verified actions enter as calibrated observations:
- a grasp counts with a probability from gripper width, the lift and the depth change;
- a place counts at its logged pose.

Every change is attributed to the robot, to a person, or to a robot failure. Example of a failure:
the log says the cube went into the box, but the box is empty. Two robots share the map: the M3
scouts, the X3 picks without searching.

**Closest:**
- **[MM-3DSGU](https://arxiv.org/abs/2411.02938)** (CoRL'24 workshop): picks and places update a
  scene graph, but at room level, in Unity simulation only, with no attribution metrics. It calls
  itself a roadmap.
- **[Lost & Found](https://arxiv.org/abs/2411.19162)** (RA-L): uses the *human's* hand track to
  find what moved. The gripper log is the robot's version of that signal.
- **[POGS](https://arxiv.org/abs/2503.05189):** tracks objects moved by people and robots, but
  does not record who.
- **Perception-only memories** detect change only from what they see:
  [DovSG](https://arxiv.org/abs/2410.11989), [DynaMem](https://arxiv.org/abs/2411.04999),
  [DREAM](https://arxiv.org/abs/2606.00576) and [SuperMap](https://arxiv.org/html/2608.22896).
- **Pieces elsewhere:**
  - [RoboStream](https://arxiv.org/abs/2603.12939) keeps action-triggered changes in a VLM's token
    memory, not a map.
  - [CubifyGS](https://arxiv.org/abs/2606.28720) moves splat assets by rigid transforms but has no
    robot.

Not found anywhere: a real-robot object map that uses the verified action log as an observation and
reports robot / person / robot-failure attribution.

**Experiment.**
- ~100 episodes with scripted human interventions (move, remove, add, swap) and induced drops.
- Baselines: perception-only updates (DynaMem-style), log-odds existence (SuperMap), local
  re-mapping (DovSG).
- Metrics:
  - attribution F1;
  - time until the map is right again;
  - rescans avoided;
  - failures caught;
  - bytes between the robots;
  - ms and GB per update on the Orin.

**Headline figure:** a swimlane timeline (robot events, human events, the map's belief per object)
beside the attribution confusion matrix.

**Why it is cheap here:**
- M2.6 needs the object memory anyway.
- The bridges already log every pick phase (metrics, narration).
- The console already has an arena panel to draw it on.

**Scoop risk:** medium-high. The MM-3DSGU authors (QUT) have this on their roadmap, and the Carlone
and Pollefeys groups are adjacent ([ChronoGraph](https://arxiv.org/abs/2609.39665), 30 Sep 2026).
Expect a 6-12-month window.

**Demo:** the console's arena flags each change by colour: green = the robot, explained by its log;
red = a person; amber = the robot failed, its log contradicted. The browser twin goes with it: the
picked cube's splat rides with the gripper to the box. Spark draws several splat meshes, each with
its own transform. Splats are trained on the 5090; the Orin only sends poses and events.

### P3 -- Show-and-pick: from "Teach" to the first pick in under a minute (track A)

**Claim.** A three-tier teaching ladder, all on the robot:
1. a one-shot visual prompt;
2. an orbit-averaged prompt with negatives the robot mines itself;
3. a YOLO26 fine-tune kept only where a grasp confirmed the label, hot-swapped as INT8.

It runs on a TensorRT engine that takes the prompt embeddings as an *input*, so a taught object is
usable immediately with no rebuild. The reported metric is seconds from "Teach" to the first
successful pick.

**Closest:**
- **[Show, Don't Tell](https://arxiv.org/abs/2603.12751)** (Mar 2026, Spot): a 15 s human video
  becomes a fine-tuned detector in 4-7 min on 4 x T4. The views come from a person, there is no
  grasp check, and nothing runs on the robot.
- **[RoboFind](https://arxiv.org/abs/2609.20330)** (17 Sep 2026): a phone captures a multi-view
  reference bank. Success rose from 25 % to 85 % with no training and no grasping.
- **[Grasp-derived masks](https://arxiv.org/abs/2305.06305)** (2023): thousands of grasps, offline.
- **BOP's model-free track** sets the yardstick: a new object onboarded within 5 minutes
  ([2504.02812](https://arxiv.org/pdf/2504.02812)).
- **Engines that take prompts at run time:** YOLOE bakes its prompts into the export, and calling
  `set_classes` on an engine fails ([docs](https://docs.ultralytics.com/models/yoloe)). NanoOWL is
  the only edge stack found that changes its queries while it runs.

**A clean side paper: negatives.** The literature contradicts itself.
- [T-Rex-Omni](https://arxiv.org/abs/2511.08997) (AAAI'26): negative visual prompts help.
- [FSS-SAM3](https://arxiv.org/abs/2604.05433): negatives make SAM 3's masks collapse.

Nobody has tested negatives on a real-time detector with real look-alikes. Target a CVPR'27
workshop or BMVC.

**Experiment.**
- ~20 new objects, each with 2-3 look-alikes.
- Measure time to the first pick, and AP against 1/2/4/8/12 views.
- Ablations: with and without negatives, with and without grasp verification.
- Baselines: YOLOE text prompts, SAM 3 on the server, and a Show-Don't-Tell-style fine-tune.

**Headline figure:** pick success against seconds since "Teach" (log axis), the three tiers as
connected segments.

**Effort:**
- A1-A2, the demo: 1-2 weeks.
- A3, the engine with prompts as input: 1-2 weeks. This is the engineering contribution.
- A4, the grasp-verified fine-tune: 1-2 weeks, once the 5090 is available.

### P4 -- Where should the arm look? Two-robot search with measured detection models (track B, with L and S5)

**Claim.** A belief-based next-best-view planner. Its action is a base pose x arm-camera pose
pair. The two robots differ only through their *measured* detection models (probability of
detection by range, angle, size and camera). Views and picks are allocated by sequential auction,
compared with an LLM allocator and a hybrid on the same belief, with WiFi degraded on purpose.

**Closest:**
- **[HEROIC](https://arxiv.org/abs/2609.19803)** (17 Sep 2026). It has a heterogeneous team, rays
  as positive evidence and log-odds for misses, and a detection curve measured from 64k detections
  (chance below ~20 px, reliable by ~30 px). But it is air-ground, reported as simulation only, and
  has no auction baseline, no communication loss and no picking.
- **[DM3-Nav](https://arxiv.org/abs/2604.22014)** (IROS'26): two real robots, but navigation-level
  only, with no arm view and no detection model.
- **Single-robot planners:** [GenMOS](https://arxiv.org/abs/2303.03178) and
  [UniFField](https://arxiv.org/abs/2510.06754).
- **LLM allocation on two real mobile manipulators** ([2509.12838](https://arxiv.org/abs/2509.12838)),
  with no auction baseline.
- **Allocators under channel loss** ([2609.13711](https://arxiv.org/abs/2609.13711)): classical
  allocators only, in simulation.

Not found: a real-robot comparison of auction against LLM allocation, or any LLM allocator tested
under packet loss.

**What HEROIC's curve means here** (est.). With 640 px and an assumed ~69 deg field of view, a
30 mm cube crosses 20 px at ~0.7 m and 30 px at ~0.47 m. So:
- each close look clears only about a square metre;
- the arm-camera pose is the main decision;
- the DCW2's 1080p colour stream roughly triples the range for far looks.

**Experiment.**
- 20 seeded layouts x {lawnmower, frontier, belief next-best-view} x {1, 2 robots}.
- Ablation: measured vs *assumed* detection model. An over-confident miss model clears cells too
  fast and loses cubes.
- Allocation: auction vs LLM vs hybrid (the LLM decomposes, the auction prices), under `tc netem`
  Gilbert-Elliott loss.

**Headline figure:** probability that the target is still unfound against time, with belief maps at
0/30/60 s as insets.

**Sections that ride along:**
- **L, the mirror trap.** A symmetry-aware metric: probability mass on the mirror pose, and the
  time to resolve it after a kidnap. A lidar-only localizer, learned or not, stays bimodal in a
  symmetric box. Registering an *oblique* RGB-D view of the floor print to the known artwork breaks
  the tie. No prior found: [ground-texture work](https://arxiv.org/abs/2508.06177) uses close-range
  downward cameras, and [Move First, Commit Later](https://arxiv.org/abs/2607.17103) (Jul 2026)
  moves to resolve the ambiguity. Alone, L is incremental: ECMR'27 or a short IROS paper.
- **S5, what localization error costs a grasp.** Success against injected base offsets
  (0-10 cm, 0-15 deg), in three conditions: open loop, servo by passes, and servo plus the M3's
  sideways step. On two kinematically different robots, then replayed in the Isaac twin to see
  whether simulation predicts it. No such measurement was found for a low-cost mobile manipulator;
  the closest is [robust base placement for a Fanuc CRX](https://arxiv.org/abs/2511.15290).
  Low scoop risk, and the cleanest figure of the lot.

**Depends on** must-have 2 (AMCL, Nav2, the catalogue), which is the reason B came first in
PLAN.md: almost nothing in it is extra work.

### P5 -- Two robots in 25 days: an AI coding agent with hardware access as a co-developer

**Disclosure first.** I am the agent this study would be about, so weigh the recommendation with
that in mind. The novelty finding comes from the survey, not from me. Your supervisor may value it
less than the robotics papers.

**Claim.** The first longitudinal, commit-level study of an AI coding agent doing full systems
bring-up on real robots: drivers, calibration, perception, SLAM tuning, the UI and safety. The agent
works on two different robots over weeks, operates the hardware itself over SSH, runs
hardware-in-the-loop tests and keeps a persistent memory.

**Closest:**
- **Anthropic's Project Fetch.**
  - [Phase 1](https://www.anthropic.com/research/project-fetch-robot-dog) (Nov 2025): 8 staff,
    one day, one robot dog.
  - [Phase 2](https://www.anthropic.com/research/project-fetch-phase-two) (18 Jun 2026): Claude
    alone finished four tasks in 9 min 35 s, against 181 min for the human team with Claude and
    361 min without. It failed the closed-loop ball fetch.
- **[SPINE](https://arxiv.org/abs/2607.13049):** agentic debugging of teleoperation bring-up.
- **[OpenRUA](https://arxiv.org/abs/2610.02459)** (1 Oct 2026): Claude Code writes its own
  perception and control through ROS 2. 99.0 % on CaP-Bench, 87.0 % on LIBERO-PRO. This is the
  agent as a *runtime controller on benchmarks*, not as the engineer who builds the stack.
- **[ROSDevEval](https://link.springer.com/article/10.1007/s10664-026-10910-7)** (EMSE, Aug 2026):
  LLMs break ROS middleware contracts.

**What makes it rigorous:**
- time to each capability;
- defects classified by who found them (the agent, the person, or a hardware symptom) and by class:
  - hardware bias: the ~1 cm lidar range bias, the 55-75 ms clock skew;
  - integration;
  - logic;
- human interventions per session, coded with the
  [2605.29442](https://arxiv.org/abs/2605.29442) taxonomy;
- safety incidents and near-misses (the arm hitting the lidar, which led to the mesh-shell guard);
- the fix-commit ratio, tokens and cost;
- then a prospective, pre-registered replication: a second student adds one capability with and
  without the agent.

**The data exists.** It is in the git history, the dated `docs/STATUS.md` notes, the memory files
and the session transcripts. The transcripts must be scrubbed first: they hold host details and
the login flow. Threats to validity: one developer, the memory files as a confound, and the author
being the subject.

**Venue:**
- [AGENT@ICSE'27](https://conf.researchr.org/home/icse-2027/agent-2027), 13 Nov AoE (corrected 2026-10-27; 27 Nov was wrong): an IEEE-published
  5-page experience report.
- Then FSE'27 Industry (23 Jan), then a journal (EMSE or JSS).
- [ICSE SEIP](https://conf.researchr.org/track/icse-2027/icse-2027-seip) (23 Oct) only as an
  18-day sprint.

### P6 -- "Stop" means stop: on-robot voice commanding in Slovak and Czech

**Claim:**
- A model-free stop lane, measured from the onset of the word to zero wheel and joint velocity,
  with motors running.
- A comparison of cascaded speech recognition -> small LLM -> validator against a speech-native
  model (Gemma 4 E2B, audio straight to a tool call), on the Orin, in English, Slovak and Czech,
  scored by exact command match.

**Closest:**
- **SMaRTAban** (Int. J. Intelligent Robotics & Applications, 21 Sep 2026; Zelenay, Kočúr, Lukáč,
  Duchoň, Marko -- affiliation unverified, likely Slovak). Whisper + GPT-4o drive a quadruped in
  English and Slovak, in the cloud; remote transcription dominated latency. It is the cloud
  baseline, and the authors are possible collaborators.
- **Server benchmarks:** [MTVA-Bench](https://arxiv.org/abs/2609.20152),
  [CORTIS](https://arxiv.org/abs/2606.21453) and [From Text to Voice](https://arxiv.org/abs/2605.15104)
  run on servers and are English-centric. The last finds that audio errors fall mostly on argument
  values. That favours exact match over intent accuracy.
- **Constrained decoding for 0.6-4B models** ([2609.23742](https://arxiv.org/abs/2609.23742))
  makes every output valid JSON but leaves a semantic gap. Keep the validator.
- **No voice-stop latency to zero velocity has been published.**
  [SocialVLA](https://arxiv.org/abs/2610.02360) reports 336 ms from a *vision* trigger to a hold.

**Experiment.**
- 200+ stops per condition while the robots move.
- False stops per hour with a TV playing speech.
- Exact match, WER, p95 latency and energy per language and model.

**Headline figures:**
- a stop-latency CDF with a 0.5 s line;
- a Pareto plot of exact match against p95 latency.

**Already built:** push-to-talk, Parakeet v3 and the grammar parser, at 277-450 ms from letting go
to a command on the laptop.

**Venue:** [HRI'27](https://humanrobotinteraction.org/2027/submissions/) Videos & Demos (20 Nov)
and Late-Breaking Reports (8 Dec, archived in the ACM DL); later RA-L, or Interspeech 2027 (its
deadline is unverified).

### The smaller ones

- **S1 and S4, cheap depth vs foundation depth for tiny objects.**
  - The gap: no benchmark of low-cost RGB-D against predicted depth for 30-40 mm objects at
    20-50 cm, scored by recall and grasp success.
  - The X3's finding that a global affine fit of relative depth fails matches the literature: MOMA
    ([2506.17110](https://arxiv.org/abs/2506.17110)) needs scale-rotation-shift alignment, and
    AnchorD ([2605.02667](https://arxiv.org/abs/2605.02667)) anchors patch by patch.
  - Edges of foundation relative depth as an object detector were not found studied for tabletops.
  - Compare raw DCW2 depth, DA-V2-S edges, Depth Anything 3, LingBot-Depth 2.0 (completion trained
    on Orbbec data) and Fast-FoundationStereo (if the DCW2 exposes its IR pair).
  - Venue: an RA-L letter, or a workshop benchmark paper.
- **C, watcher and fetcher.** The X3's mast camera reads a pointing ray, the M3 fetches, and a
  conformal set decides when to ask back. Nearby work is all single-robot:
  [MIGU](https://arxiv.org/abs/2609.24995) (Sep 2026), [GIVE](https://arxiv.org/abs/2606.13435)
  and [LEGS-POMDP](https://arxiv.org/abs/2603.04705). A natural local partner: **Hajdúch & Kocur,
  Comenius University Bratislava** ([2603.15403](https://arxiv.org/abs/2603.15403): pointing with
  MoGe depth and a VLM, no robot). Venue: HRI LBR or RO-MAN.
- **Millimetre picking on deadband servos.** The pieces: approach from one side, a probe-measured
  reach offset, stationary fingers before the descent, servo by passes.
  - Prior work fixes backlash in hardware ([AhaRobot](https://arxiv.org/abs/2503.10070)) or models
    it ([NeuralActuator](https://arxiv.org/abs/2607.11734), RSS'26 outstanding systems paper).
  - Honest grade: a workshop paper or a mid-tier journal. The impact is in porting the recipe to an
    SO-101 and releasing it as a LeRobot tool.
- **A platform paper** (RA-M, later). Both robots as LeRobot robot classes, datasets as
  LeRobotDataset on the Hub, the console's `--fake` mode, a priced bill of materials,
  `CITATION.cff` and a DOI per release.

### Quick wins inside a month (optional)

- **The [CoRL'26 EEAI workshop](https://efficient-embodied-ai.github.io).**
  - Format: 4 pages, non-archival, **due 12 Oct**, held 12 Nov in Austin.
  - Two listed topics fit the X3's perception ladder: "fallback policies under stochastic response
    time" and "edge-cloud hybrid architectures for robot fleets". The ladder falls back when the
    server is gone (`DetectorOffline`), with a measured split between edge and server.
  - Cost: one week of measurement.
- **Its Embodied Efficiency Challenge.**
  - Task: compress OpenVLA-7B onto "Jetson Orin NX (16 GB) class hardware", scored on LIBERO; the
    organisers re-run the top entries. **Due 23 Oct.**
  - We own the exact target hardware, and the sprint teaches the VLA tooling P1 needs. But it
    takes 18 days and a LIBERO setup.
- **Not worth it:** ERF 2027's scientific track (12 Oct) wants a sustainability framing.

## 3. What is current -- the architectures to know (Oct 2026)

| area | use now | watch | not now |
|---|---|---|---|
| VLA | pi0.5 (openpi, LoRA on the 5090) and X-VLA (0.9B, in LeRobot) served from the 5090; SmolVLA through vla.cpp on the Orin; LeRobot async + real-time chunking | [GR00T N1.7](https://huggingface.co/blog/nvidia/gr00t-n1-7) (3B, 16 GB+, the NX is borderline); [MolmoAct2](https://arxiv.org/abs/2605.02881) (fully open); [DynamicVLA](https://arxiv.org/abs/2601.22153) (0.4B, moving objects); [Xiaomi-Robotics-0](https://arxiv.org/abs/2602.12684); [Jetson-PI](https://arxiv.org/abs/2607.12659) (pi0.5 at ~6 Hz on an Orin) | pi0.6 / pi0.7 and Gemini Robotics 2 (closed); 7B models on the robot |
| learning from its own data | DAgger with an algorithmic expert; DART noise; [STEP](https://arxiv.org/abs/2503.10966) sequential tests; Robometer-4B as a third judge | [Real-Time EXPO-FT](https://arxiv.org/abs/2609.18207); Recap-style advantage conditioning; world models for evaluation ([Ctrl-World](https://arxiv.org/abs/2510.10125), [WorldEval](https://arxiv.org/abs/2505.19017)) | full real-world RL (HIL-SERL) before the data loop works |
| detection | YOLO26 fine-tuned with INT8 QAT (post-training INT8 cost ~3 mAP on the Orin NX: 0.479 -> 0.449); YOLOE-26 visual prompts; SAM 3.1 as the offline labeller (a SAM-3-labelled YOLO reached 79.4 % mAP with no human labels, [2605.25860](https://arxiv.org/abs/2605.25860)) | T-Rex-Omni's negatives; [EfficientSAM3](https://github.com/SimonZeng7108/efficientsam3) (Orin NX in 6.5 GB, [2604.27128](https://arxiv.org/abs/2604.27128)); NanoOWL's runtime queries | Grounding DINO / DINO-X (cloud APIs) |
| depth | [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) Small (Apache-2.0; ~23 ms on an Orin NX with TensorRT per a community wrapper, variant unverified) as a drop-in for DA-V2-S; patch-wise affine anchoring to sparse metric points | [LingBot-Depth 2.0](https://github.com/robbyant/lingbot-depth) (trained on Orbbec data); PromptDA++; [Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo) | metric depth from one global affine fit (fails, as found) |
| 3D | RTAB-Map poses -> gsplat / Brush on the 5090 -> Spark in the console; [EC3R-SLAM](https://arxiv.org/abs/2510.02080) (9 FPS on an Orin NX) | [VGGT-SLAM 2.0](https://arxiv.org/abs/2601.19887) (code "upon publication"); MapAnything / Pi3X with the lidar poses as input; [O-SCD](https://arxiv.org/abs/2511.12370) change detection | VGGT-class models on the Orin (VGGT-SLAM "Failed" there); splat SLAM on the robot |
| object maps | an object memory with SigLIP 2 / DINOv3 embeddings; log-odds existence | DREAM, CubifyGS, ChronoGraph | full 3D scene graphs (the arena has a printed map instead) |
| search and localization | belief next-best-view with a measured detection model; sequential auction; Beluga AMCL | HEROIC; DM3-Nav; Move First, Commit Later | learned end-to-end navigation |
| speech and agents | Parakeet v3 + constrained decoding + a validator; a model-free stop lane | Gemma 4 E2B audio; [Gemini Robotics-ER 2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) as a server orchestrator; MCP-to-ROS with verify-then-act ([2608.29379](https://arxiv.org/abs/2608.29379)) | an LLM on any safety path: in [Mind the Refinement Gap](https://arxiv.org/abs/2610.02662), 12/12 plans that passed a temporal-logic monitor still executed unsafely |
| evaluation | Wilson intervals; blind randomised A/B; named generalisation axes; splat real-to-sim ([PolaRiS](https://arxiv.org/abs/2512.16881); careful assets reach r = 0.90 against real, generative ones 0.51, [2610.00731](https://arxiv.org/abs/2610.00731)) | RoboArena-style pairwise evaluation | offline action-prediction metrics as a proxy (they do not predict closed-loop success, [2609.13679](https://arxiv.org/abs/2609.13679)) |

## 4. The demo program

### 4.1 Ready now -- rehearse, count the successes, show

Nothing new needs building. Each piece needs only a week of rehearsal with its success rate logged:
- **Both robots on one page, with STOP ALL.** The real-robot stop time is still to be measured;
  the pretend robots confirm in 20 ms (M3) and ~0.8 s (X3).
- **The X3 clearing the table** with the Timing card open: 21.4 s an attempt (30.9 s before the
  first optimisation round), and a live success counter.
- **The M3 picking into the drop-off box its marker finds** (11-13 s a pick), with its narration.
  The story of the arm hitting the lidar, and the mesh-shell guard that followed, is a good safety
  story.
- **Voice in English, Slovak and Czech:** what was heard, the command as JSON, and 0.3-0.45 s from
  letting go to the command.
- **The arena's coloured 3D scan and floor print** in the browser, with the rigour story scientists
  respond to:
  - the map matches the fence to 1 cm;
  - getting there exposed two hardware faults, a ~1 cm lidar range bias and a 55-75 ms clock skew
    between the base board and the Jetson.

### 4.2 Within weeks

- **Teach in 30 seconds (P3, A1-A2).** "This is my cube" by push-to-talk; the arm orbits while a
  timer and an AP-against-views curve grow. Then it picks the taught object out of three
  look-alikes and shows the similarity margins.
- **The arena splat in the console,** both robots drawn at their slam_toolbox poses.
- **The fog of war (P4):** the belief drains on the mat, P_d-weighted, so far looks clear faintly
  and close looks fully.
- **The stop-lane readout (P6):** "Stop!" mid-motion with the latency on screen, while a TV plays
  speech to show there are no false stops.
- **The judge duel (S2):** a five-cube tower. Someone knocks it; the robot rebuilds it. After every
  placement the gripper/depth verdict and a VLM verdict appear side by side, with disagreements lit.

### 4.3 Within months

- **Chase the cube (P1):** split screen, the policy against the classical picker on identical
  pushes.
- **Who moved it (P2):** the colour-coded flags, and the cube's splat riding with the gripper.
- **Pull the WiFi (P4):** unplug the access point mid-search. Each robot keeps going on its local
  belief, and they merge when the link returns. Merges are idempotent: every observation carries
  an id, so a resent one counts once. The LLM allocator visibly stalls while the auction carries
  on.
- **Watcher and fetcher (C):** "put this there" with a point.

### 4.4 The north-star run (~10 minutes, uncut)

1. A visitor says, in Slovak: "Find the small cubes and put them in the box."
2. The command appears as JSON and the validator accepts it.
3. Both robots search; the fog of war drains over the mat.
4. The M3 finds; the precise X3 picks.
5. The visitor shows an object neither robot knows. Thirty seconds later the robot picks it from
   among look-alikes.
6. The visitor pushes a cube mid-reach. The policy follows it; the classical picker, replayed beside
   it, does not.
7. The visitor moves a cube by hand. The map flags it red, "a person moved this", and re-inspects
   it.
8. "Stop" halts everything, with the latency shown.

Every step is a paper's demo, so nothing is built only for show.

### 4.5 A lab visit, 12-15 minutes

| time | what |
|---|---|
| 0:00-1:00 | the robots are already picking in a loop when the visitor walks in; the success counter is visible |
| 1:00-3:00 | one slide: what the robots cost, the stack, the research question |
| 3:00-7:00 | the visitor gives two voice commands (in their language); the console shows every stage |
| 7:00-9:00 | the visitor disturbs the scene; the recovery, and one honest failure |
| 9:00-11:00 | both robots search for one small object; the beliefs converge; a fly-through of the splat |
| 11:00-12:30 | the numbers: trials with confidence intervals, and a baseline clip side by side |
| 12:30-15:00 | the visitor drives from the console; questions |

### 4.6 A 3-minute video

CoRL caps supplementary video at 3 min.

| time | what |
|---|---|
| 0:00-0:10 | the result first: one uncut hard success, labelled "autonomous, 1x" |
| 0:10-0:30 | the problem, and why cheap robots matter |
| 0:30-1:30 | the method, with overlays of the inner state |
| 1:30-2:20 | the generalisation montage (new objects, lighting, three languages), the baseline side by side, the success chart with intervals |
| 2:20-2:45 | failures and limits |
| 2:45-3:00 | the repo, data and bill of materials, with a QR code |

### 4.7 Rules

1. **Only show live what has measured at least 90 % this week,** and put the day's count on the
   console ("today: 14/15").
2. **Show the inner state:** the transcript, the JSON, the detections, the belief, the planned
   grasp, the map.
3. **Let the visitor command and disturb.** Their language, their choice of object, their push.
4. **Put a baseline beside every learned behaviour.**
5. **Show one failure on purpose, and the recovery.**
6. **Label every clip:** autonomous or teleoperated, the speed, cut or uncut.
7. **Keep it safe:** the fence, a named person on the e-stop, speed caps, hands out while the arm
   moves.
8. **Have a backup for every segment:** a recorded clip, charged batteries, a second network.

## 5. The calendar

Checked on official pages on 2026-10-05 unless marked.

| venue | type | deadline | event | fits | link |
|---|---|---|---|---|---|
| CoRL'26 EEAI workshop | 4 pp, non-archival | **12 Oct** | 12 Nov, Austin | the edge ladder, P6 | [site](https://efficient-embodied-ai.github.io) |
| EEAI Embodied Efficiency Challenge | OpenVLA-7B on an Orin NX 16 GB | **23 Oct** | 12 Nov | the VLA tooling for P1 | [site](https://efficient-embodied-ai.github.io) |
| ICSE'27 SEIP | 10+2 pp | **23 Oct** | 25 Apr-1 May 2027, Dublin | P5 (a sprint) | [SEIP](https://conf.researchr.org/track/icse-2027/icse-2027-seip) |
| RoboCup@Home 2027 (open league only) | intent 30 Oct; team paper + video 6 Dec | | 15-21 Jun 2027, Nuremberg | the whole stack (reach may limit) | [CfP](https://athome.robocup.org/call-for-participation-robocuphome-2027/) |
| CVPR'27 | main | 16 Nov | Jun 2027 | P3 only if public baselines are beaten | [dates](https://cvpr.thecvf.com/Conferences/2027/Dates) |
| HRI'27 Videos & Demos | video / demo | **20 Nov** | 8-12 Mar 2027, Santa Clara | P6, the console | [HRI](https://humanrobotinteraction.org/2027/submissions/) |
| AGENT'27 @ ICSE | 5 pp short or 8 pp full incl. refs, IEEE, single-anonymous | **13 Nov AoE** (corrected 2026-10-27) | 26 Apr 2027, Dublin | P5 | [CfP](https://conf.researchr.org/home/icse-2027/agent-2027) |
| **RSS'27 Stage 1** | 5+1 pp extended abstract | **4 Dec** (Stage 2: 16 Apr 2027) | 6-11 Jul 2027, Athens | **P1**, P2, P4 | [CfP](https://roboticsconference.org/information/cfp/) |
| HRI'27 Late-Breaking Reports | 2-4 pp, ACM DL | **8 Dec** | Mar 2027 | P6, C | [LBR](https://humanrobotinteraction.org/2027/late-breaking-reports/) |
| FSE'27 Industry | | 23 Jan 2027 | Jul 2027 | P5 | [dates](https://conf.researchr.org/dates/fse-2027) |
| **IROS'27** | main | **1 Mar 2027** | 26 Sep-1 Oct 2027, Florence | P2, P3, P4 | [RAS](https://www.ieee-ras.org/event/2027-ieee-rsj-international-conference-on-intelligent-robots-and-systems-iros-70525/) |
| CASE'27 | main | 1 Mar 2027 | 23-27 Aug 2027, Linz | millimetre picking | [site](https://2027.ieeecase.org/) |
| RA-L | rolling; transfer to IROS/CASE'27 until 30 Apr 2027 | any time | | P2-P4, S1/S4 | [RA-L](https://www.ieee-ras.org/publications/ra-l/) |
| ECMR'27 | main | spring 2027 (unverified) | Sep 2027 (unverified) | L, P4 | -- |
| CoRL'27 | main | late May 2027 (unverified) | Nov 2027 | P1 if RSS fails | [RLF](https://www.robot-learning.org/) |

Already passed:
- the main tracks of ICRA'27 (16 Sep), HRI'27 (18 Sep) and ICLR'27 (25 Sep);
- the CoRL'26 demo track;
- the NeurIPS'26 robot workshops.

ICRA'27 workshops are still to be announced.

## 6. How the numbers must look

What reviewers expect in 2026 (CoRL's checklist, [Kress-Gazit et al.](https://arxiv.org/abs/2409.09491),
TRI's blind trials):
- **Trial counts:** 20-30 trials per condition, 50 for a headline number.
- **Wilson 95 % intervals:** at 80 % success, 20 trials give [58, 92] % and 50 give [67, 89] %.
- **Separating two policies:** telling 60 % from 80 % at 80 % power needs ~80 trials each. At
  ~21 s an attempt with automatic scoring that is about half an hour of robot time per condition,
  plus resets. This is the X3's real advantage: use sequential tests (STEP) to stop early.
- **Fixed conditions:** randomise the order of conditions, and fix object poses on a printed grid.
- **Named generalisation axes:** new objects, lighting, clutter, robot (X3 / M3), language,
  operator.
- **Open data:** raw trial logs and uncut videos released; a limitations section that names the
  failure modes.
- **For cheap hardware:** validate on more than one platform. The two robots are an asset here;
  porting a result to an SO-101 is a cheap third. Report what the hardware and compute cost.

## 7. A suggested six months

| when | P1 (X3, overnight) | the arena (M3, then both) | quick wins |
|---|---|---|---|
| October | overhead camera; a LeRobot class for the X3; recorder + auto-reset; the first nights of data | AMCL + Nav2 + catalogue (M2); measure each robot's detection model | EEAI paper (12 Oct)?; scrub and mine the logs for P5 |
| November | first fine-tunes on the 5090; DAgger round 1; the dynamic battery fixed in writing | object memory + action-event log (P2); A1-A2 teach demo | HRI demo/video (20 Nov); AGENT@ICSE (13 Nov AoE) |
| 4 December | **RSS Stage 1 abstract**: H1-H4, method, preliminary numbers | | HRI LBR on voice (8 Dec) |
| Dec-Feb | rounds 2-3; the exchange-rate curve; X3 -> M3 | P2's ~100 episodes, or P4's 20 layouts x conditions | the lab-visit demo, rehearsed |
| 1 March | | **IROS'27**: P2 or P4, whichever has its numbers | |
| 16 April | **RSS Stage 2** if invited | | |

This is more than one person can do alone. If something must go, keep P1 and one of P2/P4, and
let the quick wins go first.

## 8. Decisions for you

**Decision 1 was answered on 2026-10-07: not D-first.** The order is A -> D, with B cut to what
the Slovak mission needs. It follows the user's showcase picks; see [PLAN.md](PLAN.md) v2.4. With D
a "maybe", the overhead webcam (decision 3) can wait. Decisions 2, 4 and 5 are still open.

1. **The order.** Start D's recorder on the X3 now, in parallel with B on the M3, aiming at RSS
   Stage 1 on 4 Dec? The agreed order was B -> A -> D. The reasons to change: D's window is closing
   (HERO, Zero2Skill, DynamicVLA), RSS's format suits it, and the X3's loop can run unattended
   overnight while the M3 does the arena work.
2. **The quick wins.** Which, if any:
   - the EEAI paper (12 Oct) or challenge (23 Oct);
   - the AGENT@ICSE experience report (13 Nov AoE, not 27 Nov);
   - HRI demo (20 Nov) and LBR (8 Dec).
3. **A purchase: one overhead USB webcam** above the X3's table, and later one above the arena. It
   is D's teacher sensor, ground truth for P4/L/S5, and the demo's wide shot.
4. **Collaborators.** Approach the Comenius pointing group (Hajdúch & Kocur) for C, and the
   SMaRTAban authors for P6?
5. **The 5090.** Access is still pending, and P1, A4 and the splats depend on it.

## 9. October update (2026-10-21): the scoop check, two new ideas, scan-to-skill

Written 2026-10-21 from three parallel scans, after the user asked whether JEPA and FiLM could
improve our models: latent world models and JEPA's alternatives; conditioning and adaptation in
place of FiLM; bigger directions and what appeared from 1 to 21 October. The abstracts of VeriVLA,
GaussianFactory, DUET-DINO, the self-compensating VLA, ZETA, Splat2Act, SimVLA, TILDE and VAMPS
were re-checked by hand; the others rest on the scans' checks of arXiv abstract pages. Nothing in
this section has been run.

Two things changed in our own plan since 5 October. The VLA is now taught in Isaac Sim first, the
M3 first and the X3 after ([NEXT.md](NEXT.md), from 26 October). And the 5090 has been in use since
12 October (the kit detector trains on hekate), so decision 5 in section 8 is settled.

### 9.1 Where JEPA and FiLM land

- **JEPA** (a model that predicts the *features* of hidden or future frames, not their pixels;
  I-JEPA, V-JEPA 2 and 2-AC, V-JEPA 2.1, RoboJEPA) brings nothing to today's perception models:
  - the kit detector: a JEPA encoder is an 80M+ transformer against a 3 ms budget, and the
    detector's misses (the wooden 40 mm cube) are gaps in its data;
  - identity: DINOv2 was 84/84 in leave-one-out; Find failed only on faces never shown at Teach;
  - Depth Anything already sits on a DINOv2 encoder.
- In the VLA, JEPA is an ingredient, not a claim: a FLARE-style loss that predicts the frame at
  the chunk's end during training (no cost at run time; GR00T N1.5 trains with it), and V-JEPA 2.1
  features added through a zero-initialised path ([PVI](https://arxiv.org/abs/2603.12772)). Both
  are ablations of SmolVLA.
- JEPA as a zero-shot planner is far below our picker: V-JEPA 2-AC 50-80 % pick-and-place on a
  Franka, [RoboJEPA](https://arxiv.org/abs/2610.10515)-8B 27 %, seconds a planning step.
- **FiLM** ([1709.07871](https://arxiv.org/abs/1709.07871)) is a conditioning layer: a small
  network turns a condition into a per-channel scale and shift. Fed a robot id it does what X-VLA's
  soft prompts already do. Its better version for us is the robot card (9.3).

### 9.2 The scoop check, 1-21 October

P1 is not scooped, but its neighbourhood filled up:

| paper | what it does | what it does not |
|---|---|---|
| TILDE ([2610.25376](https://arxiv.org/abs/2610.25376), 18 Oct) | "AutoDAgger": a reward model sees progress stall and calls the teacher, with no human watching; up to 4.1x over behaviour cloning on 3 real tasks | the teacher is a learned foundation model; no student-vs-teacher comparison |
| VAMPS ([2610.05331](https://arxiv.org/abs/2610.05331), 4 Oct) | an MPPI planner on a real Flexiv arm records its own data; ACT learns from it | offline on hardware, no DAgger there, no student-vs-teacher comparison |
| SimVLA ([2610.11248](https://arxiv.org/abs/2610.11248), 8 Oct) | a VLA from simulated data only (privileged-state skills), zero-shot to real homes; beats a policy on 50 real demos | no real-world loop |
| ReDOA ([2610.25691](https://arxiv.org/abs/2610.25691)) | a simulated state teacher runs on the real robot and is distilled into an image student, no human demos | non-prehensile tasks |
| Recova ([2610.01178](https://arxiv.org/abs/2610.01178)) | recoveries made in a digital twin, plus DAgger; real 23.8 -> 77.5 % | the twin, not the robot, makes the corrections |
| Reward-DAgger ([2610.04054](https://arxiv.org/abs/2610.04054)) | a progress reward model decides when to call the expert | the expert is a human |
| MobiAgent ([2610.03476](https://arxiv.org/abs/2610.03476)), Robo-COP ([2610.09228](https://arxiv.org/abs/2610.09228)) | real-robot self-improvement: 32.5 -> 57.5 % and 38.3 -> 50.0 % | learned judges |

**What still sets P1 apart:** a teacher that is not learned and is also the referee and the reset;
cheap servo arms; two embodiments; a student that beats its teacher.

**What P1 must now add:**
- a written perturbation protocol for "moved mid-reach" (speed, distance and timing fixed before
  training, N >= 50 a condition): moving targets are now benchmarked
  ([LIBERO-Dynamic](https://arxiv.org/abs/2610.24579), [TrajWAM](https://arxiv.org/abs/2610.25763));
- two cheap baselines, a scripted retreat-and-retry and a plain re-run:
  [2610.06921](https://arxiv.org/abs/2610.06921) found a scripted retreat as good as a learned
  corrector on a frozen pi0.5;
- its level on the L0-L5 scale of "Who Improves the Robot?"
  ([2610.18842](https://arxiv.org/abs/2610.18842)), which finds physical recursive self-improvement
  not yet shown.

**The other lanes:**
- **Verifiers are crowded.** [VeriVLA](https://arxiv.org/abs/2610.21985) (15 Oct): at critical
  states a JEPA world model ranks a flow VLA's candidate chunks; real completion 50.8 -> 65.5 %,
  +326 ms a decision on a 4090. [FARM](https://arxiv.org/abs/2609.11445): a 34K-parameter readout
  on a frozen VLA-JEPA (SO-101, PiPER, Franka). [DiVeR](https://arxiv.org/abs/2610.04933). All take
  success labels from people or VLM judges; the best VLM judge reaches 0.77 balanced accuracy
  ([FailBench](https://arxiv.org/abs/2609.03611)).
- **Phone scan -> twin -> VLA is taken.** [Splat2Act](https://arxiv.org/abs/2610.13945) (9 Oct): a
  phone scan -> a 3DGS twin with collision geometry -> a privileged simulated oracle -> pi0.5 on a
  Stretch 3; real 63.9 % against 66.7 % in the twin. [AutoRSR](https://arxiv.org/abs/2610.20336):
  73-87 % real. [GaussianFactory](https://arxiv.org/abs/2609.21112): demos from one scan, mostly
  kinematic; a diffusion policy 84.2 % on a real UR10e.
- **DINOv3 over V-JEPA 2 on wrist views.** [DUET-DINO](https://arxiv.org/abs/2609.10506) (Sep):
  V-JEPA 2's wrist-view predictions underestimate the change that fine actions cause; DINOv3 planned
  better. That matters for the X3, which has only a wrist camera.
- **Per-robot execution error.** The self-compensating VLA
  ([2609.37334](https://arxiv.org/abs/2609.37334), 29 Sep) adapts a VLA online from the gap between
  commanded and executed motion: more than 30 points on each of two arms of different ages. One arm
  type, no camera error, code not out.
- P2: [REMM](https://arxiv.org/abs/2610.12710), a revisable memory for a frozen VLA. P3:
  [NovaSplat](https://arxiv.org/abs/2610.13309), an object's 3DGS from a phone video for 6D pose.
  P5: [Agentic RSR](https://arxiv.org/abs/2610.10479), a coding agent rebuilds the scene and writes
  the policy. P6: [EarlyCommit](https://arxiv.org/abs/2610.20531), the target object chosen
  mid-sentence.

### 9.3 New idea: the robot card (FiLM done our way)

**Claim.** One policy for both arms, conditioned on about 20 *measured* numbers per robot, keeps
working when the hardware drifts: a knocked camera, a worn joint.

- **The card:** backlash per joint, the delay from command to motion, the hand-eye residual (the
  camera pose's error), the gripper's offset and the zero of its tip gap. The camera pose itself
  comes from the hand-eye calibration.
- **Measured** by a ~2-minute automatic probe: the arm runs a fixed set of moves while the camera
  tracks the gripper or the board. The M3 reports only the joints it was sent, so the camera is the
  sensor on both robots.
- **Trained** in Isaac Sim: each episode draws a card (backlash 0-6 mm a joint at 25 cm, a delay, a
  hand-eye error up to ~5 deg, a gripper offset) and the simulated arm behaves by it. Backlash is a
  dead band on the commanded joint targets before they reach the joint drives. The card enters the
  policy by FiLM on the action expert's layers or as one token; try both.
- **Actions** as relative end-effector chunks in a local frame as well as joint targets.
  [ZETA](https://arxiv.org/abs/2609.02546) on transfer to a new arm: local end-effector actions +15
  points, more varied training bodies +18, 5 % target-robot data +13.4.
- **R1:** with a 5 deg hand-eye yaw error added, the policy given its measured card picks >= 15
  points more than without one; a shuffled card loses the gain.
- **R2:** re-fitting only the card (or a soft prompt, ~1-4k parameters) on <= 50 automatically
  scored episodes recovers >= 80 % of a rank-4 LoRA's gain, within 10 minutes of a camera move.
- **Closest:** the self-compensating VLA (above: an online LoRA, one arm type, the error measured on
  the arm); MERL's embodiment-aware pi0.5 ([2603.00182](https://arxiv.org/abs/2603.00182): per-joint
  tokens and FiLM from static URDF facts, simulation only, no gain on SO101); HCP
  ([1811.09864](https://arxiv.org/abs/1811.09864): a hardware vector, simulation, 2018). Not found:
  a manipulation policy conditioned on a measured fault vector across different arms.
- **Demo:** knock the camera; the robot re-probes for two minutes and picks again. The C930e moved
  on 2026-10-21 and needed a refit by hand; this is that refit, automatic.
- **Where:** inside P1 as its transfer hypothesis (H4 in 9.6); an IROS'27 paper of its own if R1
  holds.

### 9.4 New idea: the hand-back gate (JEPA done our way)

**Claim.** The VLA does the fast, reactive approach; a small predictive model trained only on the
picker's own verdicts decides when to hand the last centimetres back to the millimetre-precise
picker. The hybrid beats both its parts.

- **The model:** an action-conditioned latent predictor with a progress head (will this chunk
  succeed?) on frozen features. Compare DINOv3 with V-JEPA 2.1 (DUET-DINO's finding, tested on a
  wrist-only arm, the X3, and an RGB-D arm, the M3). Optionally slots from the kit detector's masks
  plus DINOv2 identity: slot quality decides planning success
  ([Better Slots](https://arxiv.org/abs/2608.12078)).
- **The labels:** the picker's verdicts (gripper readback, the cube gone from the table) on every
  chunk; in the simulator, the expert's progress. No human, no VLM judge.
- **G1:** failure prediction with AUROC >= 0.85 on real attempts.
- **G2:** the hybrid gets more successes per minute than VLA-only, picker-only, a hand-back on a
  fixed timer and a scripted retreat, with <= 3 % needless hand-backs.
  [RESCUE](https://arxiv.org/abs/2610.13516) found in simulation that stop-and-reset tolerates only
  2-3 % false alarms.
- Every hand-back is a labelled hard state, so it goes into the next DAgger round.
- **Closest:** VeriVLA, FARM, DiVeR (they choose among the VLA's own chunks, with labels from people
  or VLMs); EqC ([2610.18790](https://arxiv.org/abs/2610.18790), an expert takes over when an energy
  gate fires); ReAP ([2610.16309](https://arxiv.org/abs/2610.16309)). Not found: a real-robot hybrid
  whose fallback is a classical picker and whose gate learns from that picker's verdicts.
- **Demo:** a live confidence gauge in the console; the VLA chases a pushed cube and hands the last
  centimetre to the picker.

### 9.5 Scan-to-skill from the robot's own scan (P3's long form)

- **The chain:** the M3 circles a new object (the scan exists: `tools/m3_scan.py` -> a splat on
  hekate); the splat and a collision mesh become a twin in Isaac Sim; the privileged expert makes
  demos on it; a policy is fine-tuned; the arm picks the object while the real picker judges.
- **Open after Splat2Act and AutoRSR:** no human at all (the robot scans), time-to-skill as the
  headline number, and the twin's predictions checked against hundreds of automatically refereed
  real trials.
- **S1:** from the M3 circling a never-seen object to a deployed learned grasp in under 4 hours with
  zero human input; the twin ranks >= 5 checkpoints in the same order as >= 100 refereed real
  trials (Spearman >= 0.8).
- **Caveat:** Isaac Sim renders splats (NuRec/3DGUT) but they are
  [visual only](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/assets/usd_assets_nurec.html):
  the collision mesh is ours to build, from the fused depth.
- **The story with P3:** taught in 30 seconds by DINOv2 and a classical pick; a learned skill for
  it by the next morning.
- **Effort** M-L; IROS'27 at the earliest.

### 9.6 P1's hypotheses, revised

The teacher now lives in the simulator first, and the M3 comes first. The set for the RSS Stage 1
abstract (4 Dec), each falsifiable and written down before the runs:
- **H1, parity** (as on 5 Oct): trained only on the teacher's data (simulated, then a few dozen real
  episodes), the policy comes within 10 points of the picker on the 25-position grid.
- **H2, the student beats the teacher,** on the written protocol: the cube pushed mid-reach at a
  fixed speed, distance and timing, N >= 50 a condition, against the picker, a scripted
  retreat-and-retry and a plain re-run. After k DAgger rounds the student wins by >= 15 points and
  stays within 5 points on still cubes.
- **H3, the exchange rate** (as on 5 Oct), plus the simulated teacher against the real one: N
  simulated demos are worth M real ones.
- **H4, transfer through the robot card:** with its measured card, an M3-trained policy needs fewer
  X3 episodes (R1 and R2 in 9.3).
- **H5, the gate:** the hybrid beats both its parts on successes per minute (G2 in 9.4).
- **H6, no human in the loop:** over <= 5 unattended nights the success curve climbs with no human
  labels, no human resets and no learned reward; the picker scatters the cube itself.

Slovak comes free: the picker's hindsight instructions are written in Slovak and English, so one
table reports both. On translated LIBERO, success drops 30-50 % outside English
([2606.11906](https://arxiv.org/abs/2606.11906)).

### 9.7 Four additions to Monday's simulator plan

Cheap from the first episode, costly to add later:
1. Each episode draws a robot card (backlash, delay, hand-eye pose, gripper offset), stored with the
   episode.
2. Actions recorded as relative end-effector chunks in a local frame as well as joint targets.
3. The expert's verdict and progress stored for every chunk: the gate's labels from the first
   night.
4. The mid-reach perturbation protocol written down before the first student is trained.

The FLARE-style loss and the V-JEPA 2.1 features (9.1) stay ablations of SmolVLA, not claims.

### 9.8 What not to do

- JEPA or RoboJEPA as a zero-shot planner (27-80 % pick-and-place on a Franka, seconds a step);
- a plain JEPA verifier for a VLA (VeriVLA, FARM, DiVeR);
- physics twins (PhysTwin-style) for rigid cubes: they pay off only for pushing or soft things;
- large video world-action models: Ctrl-World trained on nodes of eight A100s or H100s, and
  Cosmos-class models do not fit our GPUs;
- a learned scout-and-picker team: P4 stays classical;
- a Slovak VLA as a paper of its own: one table inside P1.

### 9.9 Decisions this raises

1. **P1's scope.** Fold the robot card (H4) and the gate (H5) into P1 for the RSS abstract, or keep
   P1 as on 5 October? Recommended: fold them in; each is a falsifiable hypothesis the others' data
   already serves.
2. **The overhead webcam** (decision 3) becomes necessary for H2 and H6 on the real X3: the
   relabeller must see the cube from any state the student reaches.
3. **Scan-to-skill:** after P3's teach demo, or not before IROS?

## 10. What still needs research (2026-10-27)

Written 2026-10-27 from PLAN.md, this file, NEXT.md up to the morning, the reports of 2026-10-20 and
10-26 (reports/, their notes in research_notes/) and the ACT's training command. Much is covered
already: VLA precision on cheap servos, sim-to-real and the Isaac versions, the October scoop check,
the venues. What is left is where our results contradict the plan, where a deadline forces a
commitment, or where a choice is expensive to undo. Nothing here has been researched yet; the user
chooses which to run.

| # | topic | kind | what it decides | needed by |
|---|---|---|---|---|
| 1 | why the ACT does not look at the cube | theory + practice | what changes in the next training run | now (meeting Thu 29 Oct) |
| 2 | P1 for the RSS Stage 1 abstract: competing papers since 21 Oct, hypotheses, evaluation | method + practice | the 2-3 hypotheses written down before the data | ~5 Nov (abstract 4 Dec) |
| 3 | Isaac Sim: grasp contacts and the wrist camera's render | practice | the sim's settings for the next night batches | this week |
| 4 | Slovak/Czech voice on the robot | practice | speech recognition, word boosting, wake word, speech out on the Orin | before stage 2; HRI 20 Nov / 8 Dec |
| 5 | is A3 still new (prior art) | theory + prior art | whether "the object's description as the engine's input" is a contribution | before A3; IROS 1 Mar |
| 6 | Healthcare Challenge 2027: platform and safety | strategy | what to buy or build (the lab applies) | before the application, 31 Jan |
| 7 | the object scan's geometry, by any means (not only splats) | theory + practice | how the scan gets its poses and its surface | before the scan's next round |
| 8 | the base's pose while it turns: lidar and odometry errors | theory + practice | time sync, de-skew, what localizes the base | before the arena (stage 3) and the scan |

1. **The ACT goes one way wherever the cube is** (J1 93-98 on the M3, 53-81 in the sim; NEXT.md
   2026-10-27), and the co-training data still lean left (J1 mean 77 real and sim). Causes that fit
   our setup: `observation.state` is the command and the actions absolute joint targets (the state
   shortcut, "copycat"; the 2026-10-20 report advised dropping the state or keeping it only as the
   anchor of relative actions -- not done yet); no image augmentation in the training command, a
   ResNet18 from ImageNet; clean-ups of several pieces taken nearest first (`task.order`) while the
   evaluation has one cube, and ACT ignores the task text and runs its latent at zero -- an unclear
   target can average out into one heading. Wanted: a 30-minute diagnostic that tells them apart
   (blank and shuffled image, zeroed state, saliency); fixes with published effect sizes (state
   dropout, crop and colour augmentation, frozen DINOv2/v3 or a spatial softmax, the picker's target
   as an input point or mask, a diffusion or flow head); the data's design (one piece or several an
   episode, episodes a table cell, DART noise).
2. **P1 / RSS'27 Stage 1.** About ten neighbours appeared in three weeks of October (9.2); H1-H6 (9.6)
   are too many for five weeks and must be written before the runs. Wanted: which 2-3 our data can
   carry (likely H1, H3, H6), what Stage 1 reviewers look for, and how to show the sim grid ranks
   checkpoints as the M3 does (SIMPLER-style rank correlation) -- today the two agree only that the
   real-only baseline fails, not where it goes.
3. **The twin where it costs data.** 29 % of simulated picks do not end in the box, gripped 12-15 mm
   over the piece's middle at the far rim, where the real M3 holds the same grips (a FINGER_SQUEEZE
   knob so far); PhysX's mimic joints do not hold; one run in three renders black beside a training;
   the sim's colours are lighter. Wanted: the PhysX settings that decide small parallel grasps
   (contact and rest offsets, friction combine, SDF or convex collision, solver iterations, drive
   force limits), what working SO-101 and Isaac Lab scenes use, what changes in 6.2. The 10-26 report
   compared versions, not contacts.
4. **Voice on the robot.** Parakeet v3 garbles Slovak (SOUNDS_LIKE rules: 79/79 on synthetic voices
   only); nothing runs on the robot yet; stage 2, HRI, P6 and the Healthcare Challenge's on-board rule
   all hang on it. Wanted: Parakeet v3 vs Canary-1B-v2 vs Slovak Whisper fine-tunes on an Orin NX, NeMo
   word boosting for the commands, a Slovak wake word, Slovak speech out, speech-native models that
   answer with a command.
5. **A3's novelty.** NanoOWL (OWL-ViT in TensorRT, class embeddings at run time) and the YOLO-World /
   YOLOE export paths may already do it; our own result says visual prompts do not reach single
   objects (DINOv2 4/4 and 84/84, YOLOE 1/4). A prior-art check before 1-2 weeks of graph surgery;
   P3 may be better framed as teach-to-pick time for one specific object.
6. **Healthcare Challenge 2027, if the lab commits.** The most expensive choice in sight (a Lite 6 on
   a lift column and a base, or a Stretch); all compute on board, a Jazzy controller, a single-fault
   safety concept, the scoring chapter withheld. Wanted: platforms and costs, safety standards and
   risk analyses, what won similar leagues, on-board compute (Orin AGX, Thor).
7. **The object scan's geometry.** PPISP (below) fixes colour only; the scan's faults are poses and
   coverage: the odometry drifts ~10 cm a circle, colour ICP moved cameras 0.7 cm median and 6.1 cm at
   most, the anchored fuse leaves the box ~0.7 cm too tall and a stop-5/6 reading ~6 cm off, the splat
   corrects cameras 3-6 mm (17 mm on the can), the tape's ghost from high up, the shiny lid. Wanted, by
   any means: multiway / pose-graph / bundle adjustment with depth and colour, the arena map or lidar
   pinning the base, feed-forward 3D models (MapAnything, VGGT, Pi3) for poses, surface methods (TSDF,
   2DGS / PGSR, neural SDF), view planning for coverage, shiny things, the colour (locked exposure,
   PPISP). Judged by the box's and the can's sizes against the ruler and by held-out views.
8. **The base's pose while it turns.** Known so far (M2, 2026-09-30): the base board's clock runs 55-75
   ms ahead of the Jetson, the lidars scan at ~7 Hz (270 deg each, two of them merged) and read ~1 cm
   long, Yahboom's EKF gives odom at ~6-7 Hz, carrying the robot doubles walls, RTAB-Map's grid came
   out +7 cm. Turning at 0.3-0.6 rad/s a 7 Hz scan sweeps 2.5-5 deg while it is taken, and 65 ms of
   clock offset is another 2-2.5 deg -- ~5 cm sideways at 1.2 m. Wanted: the error sources on cheap
   mecanum bases (scan motion distortion, timestamps, wheel slip in turns, IMU), the fixes (de-skew,
   clock sync or offset estimation, IMU-aided turns, scan-to-map matching, slam_toolbox / AMCL
   settings), and how to measure them (the fence, an overhead camera). Serves the scan, the arena and
   the Healthcare Challenge's navigation.

**The user's answers (2026-10-27):** research everything that brings better results or a more precise
approach -- topics 1-6, plus the scan's geometry by any means and the lidar's errors while the robot
turns (7, 8); the lab IS applying for the Healthcare Challenge; reports in English (reports/ and
research_notes/ as before).

**Not worth researching now:** AMCL/Nav2 in the arena (standard tools; our measurements decide);
depth (the duel settled it); the kit detector (its misses are data gaps); the M3's calibration (the
touches and the board, ~5 mm); the VLA's choice (decided 2026-10-26; again when ACT heads for the
cube); JEPA and world models (9.8); model export formats (a research started 2026-10-27, its notes
folder still empty). Later: P2's scoop check before December; how servo deadband meets action chunks
once the policy reaches the cube.

### 10.1 The results (2026-10-27, eight parallel research agents)

Each wrote a report in `reports/` (title = the topic) and 3-5 notes in `research_notes/<same name>/`. The
shared web-search quota (~200 searches for all eight) ran out partway, so later checks went to arXiv, GitHub
and official pages directly, and every "not found" needs a re-check before a submission. "Checked" below = a
claim re-read in its primary source or our code after the agent finished.

| # | report | bottom line | checked |
|---|---|---|---|
| 1 | Why our ACT does not look | two shortcuts beat looking: the state is the previous action (a lookup on the joints alone matches the ACT, 7.79 deg) and the heading is chosen in ~2 % of the frames; the real-only baseline recalls its 13 layouts; the CVAE latent is unused. Next run: no state, augmentation, departure frames x5, layouts held out | the state-free paper's ACT row (Table V: height 0 -> 0.933, horizontal 0.084 -> 0.517; relative actions, two wide wrist cameras) |
| 2 | P1 for RSS 2027 Stage 1 | still open only as "the student beats its own teacher"; freeze H1, H2 (cube moved 4 cm after the picker's last look) and HV (the sim ranks checkpoints as the robot) on 5 Nov; go/no-go 13 Nov at >= 30 % first-attempt grasps | the RSS call (4 Dec AoE, 5+1 pages, no links, OpenReview profiles up to two weeks); SimPT (2610.29161) |
| 3 | Isaac Sim grasping fidelity | the far-rim losses follow the URDF's short pads (< 13 mm down the piece: 8/31; >= 13: 129/158); sRGB paint fed as linear (2.5-5x too light); solve contacts after drives only from 5.1; drives 30-1000x too stiff for 120 Hz | the paint: `kit_synth.py` CUBE_PAINT -> `diffuse_color_constant` |
| 4 | Slovak voice on the robot | the garbled Slovak is Parakeet's language guess; KInIT's Slovak Parakeet + base Parakeet for English in sherpa-onnx; own stop/wake models; grammar before a small LLM; ~3-4 GB, ~1 s | KInIT's model card (CV24 7.92 vs 9.07 %, noisy set 6.62 vs 24.32 %, CC-BY-4.0, forgets the other languages) |
| 5 | Teach-to-pick prior art | prompts as engine inputs are shipped (NanoOWL, Isaac ROS Grounding DINO, YOLO-World exports); drop A3 as the claim; P3 = time to the first correct pick among same-shape twins, grasp-gated A4 | YOLOE's head in our Ultralytics 8.4.175 (`x[-1]` prompts; `fuse()` bakes them) |
| 6 | Healthcare Challenge 2027 platform | a two-robot team: our own AgileX Ranger Mini + lift column + Lite 6 + AGX Orin 64 GB + hard-wired E-stop chain (~EUR 19-25k), the M3 as guide; buy in November; Stretch 4 the fallback | the rule RB 8.1.1.2 as the notes quote it (not re-read in the PDF) |
| 7 | Object scan geometry | x, y and yaw are what is wrong; the printed floor measures them (16 mm on odometry -> 0.5-0.7 mm after a floor solve); the object improves only partly (lens distortion suspected); the box's +5 mm is not poses; the ghost is a pose jump | `camlock` exists in `tools/arena_map.py` |
| 8 | Base localization while turning | timing, not slip: the board stamps a scan when it sends it (~83 ms late, 167 ms spread), ~1.4 deg lag + smear at 0.3 rad/s; the EKF may get no IMU; a standing fence fit per stop for the scan | our launch: no imu_filter node; the second lidar's correction was tried in replays and left out on purpose (`lidars.yaml`) |

**The second round (same evening, three agents with a 40-search budget each; 7-24 searches used):**

| # | report | bottom line | checked |
|---|---|---|---|
| 9 | M3 joint read-back and board firmware | one STM32H743 board drives the arm servos itself (USART3, Feetech-SCS, a position read in Yahboom's code); factory firmware is a hex only; Yahboom's calibrate_arm.py makes the arm limp; read-only survey -> bench servo -> bus listening, reflash only with Yahboom's source | m3arm.py does say "no torque switch" (the user's power-cycle test, 2026-10-05) -- the contradiction is real, to test |
| 10 | Healthcare Challenge 2027 task approaches | every task from known methods; sub-skills score separately; Jazzy native on the AGX Orin, a Jazzy gateway for the M3, a small dispatcher (no Open-RMF); the arena GLB needs fixing for Isaac Sim | the F/T sensor: retailers list it as not compatible with the Lite 6, compatible with the 850 |
| 11 | P5 AI agent case study | AGENT 2027 is due **13 Nov AoE**, not 27 Nov; no prior report of an agent building a robot stack over weeks with hardware in the loop; our record in numbers; go for 5 pages if the supervisor agrees | the deadline and page limits on the official page |

**Decisions they raise:**
0. P5: write it for 13 Nov (the supervisor on Thursday), or move it to WSESE / FSE Industry.
1. The ACT's next run without the state input (R1), after the offline probe -- Wednesday.
2. P1's three hypotheses frozen by 5 Nov; OpenReview accounts now.
3. The Healthcare build: ask the organisers about robot-to-robot links, then order (November) or not.
4. A3 dropped as P3's claim; a DINOv2 crop engine instead.
5. The sim: calliper the real finger pads; fix the paint's colour space (also in the kit detector's frames).
6. The base: a read-only timing audit on the M3 before any localization change.

**Looked at 2026-10-27, PPISP** ([2601.18336](https://arxiv.org/abs/2601.18336), the user's find, for
the object scan): it fixes the camera's colour (exposure, white balance, vignetting, tone curve), not
the scan's faults, which are poses and coverage. Measured on our scans: exposure swings x0.89-1.20, the
corners 7-9 % darker; NEXT.md, "The scan's colours", has the numbers and the next steps.

## Method and caveats

Seven surveys ran in parallel on 2026-10-05, one per area above, each with the project's hardware
and plans as context and asked for a gap analysis with precise citations. Their web-search quota ran
out partway, so the later checks used arXiv abstract pages and the arXiv API. A claim that rests on
an abstract alone may miss what the paper body says. Model names inside 2026 abstracts were not
checked independently. Figures from other labs are their measurements, not ours. Re-check any
"not found" before claiming novelty in a submission: a two-hour related-work pass per paper, close
to its deadline.
