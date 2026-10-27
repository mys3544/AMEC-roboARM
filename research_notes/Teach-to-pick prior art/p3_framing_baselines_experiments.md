# P3's framing, baselines and experiments after the prior-art check (state 2026-10-27)

Scope: question (e) of RESEARCH.md section 10, topic 5 -- how to frame P3 ("Show-and-pick", track A) so that it is
publishable at IROS 2027 (deadline 1 March 2027) or RA-L, which baselines it must beat, which experiments carry it,
and what to build instead of A3's graph surgery. It draws on the three other notes in this folder
(`runtime_prompt_engines.md`, `instance_recognition_few_views.md`, `teach_and_fetch_robot_systems.md`) and on our own
results in PLAN.md track A. Nothing here has been run; numbers marked "est." are ours, not measured.

Labels: **[REAL-robot]**, **[BENCH]**, **[CLAIM]**, **[2nd]** as in the other notes; **est.** = our estimate.

## 1. What P3 can and cannot claim

### Takeaway
P3 cannot claim the engine with prompts as an input (NanoOWL, Isaac ROS Grounding DINO and YOLO-World exports do it),
nor DINOv2 matching for identity (CNOS, VAP, CNOS25 at BOP 2025). It can claim a measured, on-board path from "Teach"
to the first *correct* pick among same-shape look-alikes, with the views collected by the robot itself and the labels
for its later, faster detector admitted only by the robot's own grasps. The number is time-to-first-correct-pick and
the wrong-twin rate, not AP alone.

### The claim, as it could be written
"A mobile manipulator with an Orin NX learns a never-seen object in ~10 s by orbiting it with its arm camera and
depth, and picks it among same-shape look-alikes within a minute of Teach, with no training and no server. Grasp
outcomes then admit the robot's own frames into a per-object detector that replaces the gallery after N minutes. On
K objects in G look-alike groups, robot-collected views beat human photos, instance matching beats every
concept-level prompt (text, YOLOE visual prompts, SAM 3 exemplars, OWLv2 image queries), and the grasp gate halves
the wrong-twin rate of the fine-tuned detector."  (The last two clauses are hypotheses, to be tested.)

### Inferences
- The three contributions that survive the check: (1) robot-collected, coverage-aware views as the teaching act
  (static onboarding done by the arm, with depth masks for free); (2) the time ladder on board, teach in seconds ->
  gallery -> grasp-gated detector, with a measured crossover; (3) a twin benchmark (same shape, different print) with
  a wrong-twin metric. Our "YOLOE's visual prompts are category-level, DINOv2 crops instance-level" result becomes the
  motivating experiment, not the contribution.
- "Negatives" stays a side question: the evidence is mixed (T-Rex-Omni helps, FSS-SAM3 collapses, IDOW +0.16 AP); in
  P3 it is one ablation (margin rule vs explicit look-alike negatives), not a claim.

## 2. Baselines to beat

### Takeaway
Two groups. Concept-level baselines (expected to fail on twins, needed to show why): text prompts, YOLOE visual
prompts (single and multi-view), SAM 3 exemplars, OWLv2 image-guided. Instance-level baselines (the real
competitors): a VAP-style pipeline from human photos, CNOS25-style YOLOE + DINOv3 on a GPU server, and a
Show-Don't-Tell-style fine-tuned detector. Most run offline on recorded robot looks, so the expensive live picks are
spent only on the three or four that matter.

| # | baseline | group | where it runs | live picks? | what it tests |
|---|---|---|---|---|---|
| B1 | YOLOE-26 text prompt, a person's description ("white cube, fish print") | concept | Orin NX engine | yes | language cannot name twins |
| B2 | Grounding DINO with the same text | concept | Isaac ROS / 5090 | offline | the strongest text detector |
| B3 | YOLOE-26 visual prompt, 1 view and orbit-averaged (+ negatives) | concept | Orin NX | yes | our 1-of-4 result at scale |
| B4 | SAM 3.1 with an exemplar box | concept | 5090 | offline | concept-level by design ([SAM 3](https://arxiv.org/abs/2511.16719)) |
| B5 | OWLv2 image-guided one-shot | concept | 5090 | offline | the classic one-shot detector |
| B6 | VAP-style: 5 phone photos + Grounding DINO + DINOv2 [CLS] ([VAP](https://arxiv.org/html/2512.20014)) | instance | 5090 | offline + a few live | who collects the views |
| B7 | CNOS25-style: YOLOE proposals + DINOv3 ViT-L ([cnos25](https://github.com/vincirist/cnos25)) | instance | 5090 | offline | the off-board upper bound |
| B8 | IDOW-style fine-tuned DINOv2 ViT-S ([IDOW](https://arxiv.org/abs/2503.00359)) | instance | Orin NX | offline | is a better embedding worth it |
| B9 | YOLO26n fine-tuned on the orbit frames, no grasp gate ([Show, Don't Tell](https://arxiv.org/html/2603.12751)-style) | instance | train 5090, run Orin | yes | the gate's value |
| ours | depth (M3) or YOLOE prompt-free (X3) proposals + DINOv2 ViT-S gallery; then grasp-gated detector | instance | Orin NX | yes | -- |

### Inferences
- B6 vs ours isolates the one design choice no prior system made: the robot, not a person, collects the views.
  Expect robot views to win on the robot's own viewpoints and phone photos to win on faces the arm cannot reach
  (the M3's steep views stop at ~82 deg at 20 cm).
- B7 tells how much accuracy the Orin NX costs (ViT-S vs ViT-L, DINOv2 vs DINOv3). If the gap is small, that is a
  result in itself.

## 3. Experiments

### Takeaway
An offline identity study on recorded looks (cheap, all baselines, all ablations) plus a live study of ~250 picks
on the M3 (four arms x ~60 trials), with the X3 as a second embodiment for the gallery tier. The headline figure is
correct-pick success against seconds since Teach (log axis), each tier a segment, with the baselines as points.

### Design
- **Objects:** ~24 to teach in 6-8 twin groups of 3-4, plus ~10 never-taught distractors for open-set rejection:
  same shape different print (our 4 picture cubes; cans or boxes with different labels), same object different
  colour (markers, cups), near-identical with one small mark (a sticker -- the limit), and same print on different
  shapes (does the embedding read shape or texture?).
- **Teach conditions:** 1 view; 1 round of 4; 2 and 4 rounds turned a quarter (our 19-23 views); a person's 5 phone
  photos (B6); a scan's views (`teach_from_scan`). Record coverage (sides seen), not only count.
- **Offline identity:** every Find look recorded with ground-truth boxes and names (the depth boxes, checked by hand
  once); identity top-1 among the look-alikes, unknown rejection (AUROC, FPR at 95 % TPR), per baseline and ablation.
  Thousands of crops, no robot time.
- **Live picks:** seeded layouts, each target with its 2-3 twins and 2 unknowns; arms = ours, B1 or B3, B9, ours +
  grasp gate. Two-proportion power: 90 % vs 70 % success needs ~62 trials an arm, a wrong-twin rate of 5 % vs 25 %
  ~49 (alpha 0.05, power 0.8; est.). Report identity failures and grasp failures separately, as VAP did (50.9 % /
  49.1 % of its real failures).
- **Time:** teach (motion, embedding), search, pick, drop, each stage's median and IQR; per-stage latency on the
  Orin NX (p50/p99), memory.
- **Ablations:** views and coverage; negatives (none / margin rule / explicit look-alike negatives); backbone (DINOv2
  ViT-S/B, DINOv3 S/B, B8), descriptor (CLS, CLS + mean patch, best-3 vs mean); proposals (depth vs YOLOE prompt-free
  vs the kit detector); the grasp gate with injected label noise (a twin admitted as the target at 10/20/30 %).
- **Second embodiment:** the X3 (wrist camera, no depth) with YOLOE prompt-free proposals and the same gallery: does
  a gallery taught on the M3 transfer, and how many X3 views fix it?

### Inferences
- Effort, est.: offline set and baselines 1.5-2 weeks; live picks ~1 week of robot time (240-300 picks at ~1 min
  each with resets); A4 with the gate 1-2 weeks; writing 3 weeks. About 8-9 weeks of the 18 left before 1 March 2027.

## 4. What to build instead of A3

### Takeaway
A3 as "graph surgery on the YOLOE export" is 2-4 days of plumbing, not a contribution, and it does not move the
user-visible number. Build the identity engine and the benchmark instead.

1. **DINOv2 ViT-S/14 crop engine** (ONNX opset 17, FP16, static batch 8 and 16, 224 px) on both Orins, cosine vs
   PyTorch > 0.999 on our crops, latency p50/p99; the gallery stays a matrix outside the engine (`runtime_prompt_engines.md`
   section 3). 1-2 days.
2. **The twin benchmark**: objects, recorded looks with names, seeded layouts, the live protocol script; publish it.
   3-4 days.
3. **Baselines on the 5090** (B2, B4-B8), offline on the recorded looks. 3-4 days.
4. **A4 with the grasp gate**: the robot's own frames admitted by grasp outcome (X3 read-back; on the M3 the piece gone
   from the next look), YOLO26n fine-tuned on hekate, FP16 engine hot-swapped (INT8 buys nothing at our size, PLAN.md
   M5). 1-2 weeks, as planned.
5. Optional, for the X3's open-set proposals only: the YOLOE prompt-input export (unfused head, prompts as a second
   input). 2-4 days.

## 5. Venue and risks

### Takeaway
IROS 2027 fits a systems paper with this evaluation; RA-L (with the IROS presentation option, which in past years
shared the IROS deadline -- check the 2027 call) needs the full twin benchmark, the second embodiment and the grasp
gate result. The main risk is the "integration" critique, answered only by the comparisons in section 2.

### Risks
- **"It is CNOS/VAP on a robot."** Answer with B6 (who collects the views), the time ladder and its crossover, the
  wrong-twin metric, the released twin set. Without these the paper is a demo.
- **Scoop window ~6-12 months:** VAP (ICML 2026) can move to on-board compute; RoboFind (Sep 2026) and EgoAsk (Sep 2026)
  show active groups on personal objects; Show, Don't Tell can add a timed robot study. Re-check arXiv in February.
- **Small-N thresholds:** SAME = 0.65 and MARGIN = 0.05 were tuned on four cubes; calibrate on held-out objects or
  report the open-set curve.
- **Grasp failures drown identity:** the M3 grips ~1.5 cm short without its corrections; report identity and grasp
  separately, and use the taught-aim table.
- **Twins that are truly identical** have no right answer: the system must say "ambiguous" (ask), and the metric
  must count that as correct.
- **Licences:** YOLOE and YOLO26 AGPL-3.0, DINOv2 Apache-2.0; DINOv3's licence was not checked here -- check before
  shipping it on the robots.
- **Search coverage:** web searches ran out during this check; CoRL/IROS 2026 proceedings, Toyota HSR and RoboCup@Home
  teaching pipelines were not checked directly.
