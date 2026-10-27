# Prompts as engine inputs are already shipped; P3's contribution is the timed, on-board pick of a robot-taught object among look-alikes

A detector engine that takes the object's description as a run-time input is no longer a contribution, and A3's
"graph surgery on the YOLOE export" should not be the engineering claim of P3. NVIDIA's NanoOWL has kept OWL-ViT's
image encoder in TensorRT and matched text embeddings outside the engine since 2023-24; Isaac ROS 5.0 ships Grounding
DINO as a TensorRT engine whose text tokens are inputs, changed by a `set_prompt` service; community exports of
YOLO-World take the text embeddings as a graph input; and in Ultralytics' own YOLOE head the prompts are already a
tensor appended to the feature list, so the export wrapper is days of work, not weeks. More important, our own
measurement already decided what the "description" must be: YOLOE's visual prompts named 1 of 4 same-shape picture
cubes, DINOv2 crops named 4 of 4 (84/84 leave-one-out), and the DINOv2 gallery of taught views is already a run-time
input -- Teach to a never-seen cylinder put down took about 50 s with no rebuild of anything. That identity stage is
itself the field's standard ("propose, then match with DINOv2/v3": CNOS 2023, VAP at ICML 2026, the BOP 2025 winner
CNOS25 built from YOLOE plus DINOv3). What we did not find in any paper is the combination: views collected by the
robot itself, everything on a Jetson-class robot, same-shape look-alikes, a measured time from Teach to the first
*correct* pick, and labels for a faster per-object detector admitted only by the robot's own grasps. P3 should be
framed around that, with time-to-first-correct-pick and the wrong-twin rate as its numbers. Replace A3 by a DINOv2
crop engine (1-2 days), a released twin benchmark, and the baselines below; keep A4, with the grasp gate measured
against an ungated fine-tune.

The evidence is in `research_notes/Teach-to-pick prior art/` (four notes). Labels: [REAL-robot] measured on a real
robot, [BENCH] a benchmark or bench timing, [CLAIM] an author's or vendor's statement, [2nd] a secondary source. Web
searches ran out partway (a shared quota of about 200 for all agents; ~33 used here); every "not found" is bounded by
those searches and needs the usual re-check before submission.

## Engines with run-time prompts exist on Jetson, from NVIDIA itself

The design A3 aims at -- everything image-side in the engine, the query embeddings supplied at run time -- is how
every open-vocabulary stack on Jetson already works:

- **NanoOWL.** The TensorRT engine is OWL-ViT's image encoder only (outputs `image_class_embeds`, `logit_scale`,
  `pred_boxes`); the text encoder runs in PyTorch, and the class scores are an `einsum` in PyTorch's `decode()`
  ([owl_predictor.py](https://raw.githubusercontent.com/NVIDIA-AI-IOT/nanoowl/main/nanoowl/owl_predictor.py)). Text
  queries only, OWL-ViT only, no OWLv2. The README claims 95 FPS for B/32 on an AGX Orin [CLAIM]
  ([README](https://github.com/NVIDIA-AI-IOT/nanoowl)); a user measured 40-60 ms, and 400+ ms for OWLv2 [2nd]
  ([forum](https://forums.developer.nvidia.com/t/nanoowl-inference-takes-more-time-than-nanoowl-official-github-shows/293051)).
  Jetson Platform Services packages it as a service whose classes are set by a REST call
  ([JPS](https://docs.nvidia.com/jetson/jps/inference-services/zero_shot_detection.html)).
- **Grounding DINO.** TAO's TensorRT engine takes `input_ids`, `attention_mask`, `position_ids`, `token_type_ids` and
  `text_token_mask` beside the 544x960 image
  ([TAO](https://docs.nvidia.com/tao/tao-toolkit/latest/text/trtexec_integration/trtexec_grounding_dino.html)); Isaac ROS
  5.0 exposes a `set_prompt` service
  ([Isaac ROS](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_grounding_dino/index.html)).
  It is slow on an Orin (about 1 s an inference [2nd]).
- **YOLO-World.** The official export bakes the vocabulary and lists TensorRT as "coming soon"
  ([deploy.md](https://github.com/AILab-CVC/YOLO-World/blob/master/docs/deploy.md)), but community ONNX exports take
  `txt_feats` [batch, N, 512] as an input, N dynamic by a patched head [CLAIM]
  ([Instemic](https://huggingface.co/Instemic/yolo-world-onnx);
  [ibaiGorordo](https://github.com/ibaiGorordo/ONNX-YOLO-World-Open-Vocabulary-Object-Detection), N fixed).
- **YOLOE (ours).** Ultralytics bakes prompts into exports; a loaded export "can no longer accept new prompts"
  ([docs](https://docs.ultralytics.com/models/yoloe/)). But in our installed 8.4.175 the unfused head reads the
  prompts as `x[-1]`, sets `nc = x[-1].shape[1]` and scores by `einsum("bchw,bkc->bkhw", ...)`; only `fuse()` folds them
  into a 1x1 conv. A prompt-input export is the YOLO-World patch again: est. 2-4 days.

What A3 would buy is small. YOLOE in an engine runs in 5.36 ms against ~41 ms in PyTorch on the Orin NX [BENCH, ours,
PLAN.md M5], and a taught object would need no one-minute rebuild. But the PyTorch service already takes new prompts
without a rebuild, a pick look is not a video loop, and -- decisively -- YOLOE's prompts are not what names the
object. Our 2026-10-09 test: its visual prompts called all four picture cubes "cosmetics", its prompt embeddings
matched view to view named 1 of 4, DINOv2 4 of 4 [REAL-robot, ours]. A prompt-input YOLOE engine is worth building
only for the X3's open-set proposals (it has no depth), as a side task.

## The identity stage that works for us is the field's standard

Instance detection since 2023 is "class-agnostic proposals, then DINOv2 (now DINOv3) descriptors matched to
reference views". CNOS (SAM + DINOv2, +19.8 % AP on BOP's seven core sets) set it up
([2307.11067](https://arxiv.org/abs/2307.11067)); on HR-InsDet the off-the-shelf SAM + DINOv2 pair beat a trained
detector, 41.61 against 27.99 AP [BENCH]; IDOW (CVPR 2025) fine-tuned DINOv2 ViT-S/14 for instance matching and reached
57.01 AP with Grounding DINO proposals, 0.16-0.18 s an image on a desktop GPU, noting that "GroundingDINO is not
real-time" [BENCH] ([2503.00359](https://arxiv.org/html/2503.00359)). At BOP 2025, model-free 2D detection was won on
speed and openness by CNOS25: YOLOE-11L proposals, DINOv3 ViT-L/16 descriptors from 100 onboarding views, BOP-H3 AP
0.441 against 0.340 for FastSAM + DINOv2, 0.134 s an image on an RTX 4090 [BENCH; award per README]
([cnos25](https://github.com/vincirist/cnos25);
[leaderboard](https://bop.felk.cvut.cz/leaderboards/modelfree-detection-unseen-bop24/bop-h3/)). That is our A1-A2
architecture: YOLOE or the depth proposes, DINOv2 says which. NVIDIA ships the same pattern on Jetson for retail
products, which, like our cubes, often differ only in print: a detector, then a 2048-d embedding network as a
DeepStream secondary engine matched against a reference gallery; 0.77 accuracy with one reference image a class, 0.86
with 90; 59.8 images/s at batch 4 FP16 on an Orin NX [BENCH, CLAIM]
([model card](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/retail_object_recognition)).

Concept-level promptable models do not do this job by design. SAM 3 "detects, segments, and tracks objects ... based
on concept prompts" and returns "all matching object instances"; instance prompting is a separate mode
([2511.16719](https://arxiv.org/abs/2511.16719)). VAP's authors found frozen VLAs "collapse to category-level
recognition" ([2512.20014](https://arxiv.org/html/2512.20014)). Same-category look-alikes are costly when measured: in
PIN, a personalised-navigation benchmark, the best (DINOv2) agent fell from 23.13 to 14.84 success when same-category
distractors were present [BENCH] ([2410.18195](https://arxiv.org/html/2410.18195v2)). We found no benchmark of
same-shape, different-print twins and no published wrong-look-alike pick rate for a real arm. Our misses so far were
coverage, not the embedding: two news cubes taught from one side read 0.64 and 0.61 (bar 0.65); after four rounds
turned a quarter each (19-23 views) every cube was named right (0.80-0.85) [REAL-robot, ours].

## What nobody has measured: on-board, robot-taught, timed picks among same-shape twins

The closest systems, by what they lack against P3:

| work | what it does | on edge? | instance-level? | teach -> use | lacks vs us |
|---|---|---|---|---|---|
| [VAP, "Bring My Cup!"](https://arxiv.org/html/2512.20014) (ICML 2026) | ~5 user photos; Grounding DINO + DINOv2 [CLS] votes; frozen VLA on SO-101; 2 same-category look-alikes; pick-and-place 58.8 % vs 18.8 % [REAL-robot] | no (RTX A6000) | yes | "immediately", not timed; init 0.26 s | robot-collected views, on-board, timing, twins of the same shape, grasp-checked labels |
| [RoboFind](https://arxiv.org/abs/2609.20330) (Sep 2026) | phone-guided multi-view reference bank; quadruped search with a verification agent; 85 % vs 25 %, false success 75 % -> 5 % [REAL-robot] | not stated | yes | not timed | grasping, robot-collected views, edge |
| [Show, Don't Tell](https://arxiv.org/html/2603.12751) (Mar 2026) | 15 s human video -> auto-labelled set -> Faster R-CNN fine-tuned (3-4 min on 4 T4), used on a Spot; mAP 0.10-0.15 vs <= 0.08 text-prompted [REAL-robot, BENCH] | inference on robot runs, site not stated; training off-board | partly (clustering; humans sometimes "lock in" detections) | 4-7 min | seconds-scale use, robot-collected views, automatic grasp checks |
| [GOAT](https://arxiv.org/html/2311.06430) (2023) | image goals matched by SuperGLUE within the category; 86.4 % for image goals in 9 homes [REAL-robot] | not stated | yes | not timed | timing, twins, picking as the task |
| [iTeach](https://arxiv.org/abs/2410.09072) (2024, rev. Sep 2026) | MR-headset corrections, on-board fine-tune and redeploy; segmentation 26.1 -> 80.7; ~20 s an interaction [REAL-robot] | yes | no (class-agnostic masks) | ~20 s | identity |
| [iCub on-line detection](https://doi.org/10.1007/s10514-019-09894-9) (2020) | a person shows the object; kernel classifiers on CNN features; learns "within seconds" [2nd] | robot PC | object-level | seconds | foundation-model identity, look-alikes, picking |
| [BOP model-free](https://arxiv.org/html/2504.02812) / [CNOS25](https://github.com/vincirist/cnos25) | onboarding from a static orbit video, up to 5 min on one GPU; YOLOE + DINOv3 [BENCH] | no (RTX 4090) | yes | <= 5 min | robot, picks, edge |
| [Humanoid onboarding](https://arxiv.org/abs/2604.17258) (Apr 2026) | YOLOv8 + SAM 3D + FoundationPose on a G1; onboarding ~30 min [CLAIM] | not stated | per-object | ~30 min | speed, look-alikes |
| [Grasp segmentation](https://arxiv.org/abs/2305.06305) (2023), [Grasp2Vec](https://arxiv.org/pdf/1811.06964) (2018) | labels from thousands of grasps [REAL-robot] | -- | yes / instance grasping | offline | one new object taught in seconds |

Every part of P3 has prior art; the combination has none that we found. "Under a minute" alone is not new (iCub
claimed seconds in 2020). "Under a minute to the first correct pick, on board, among same-shape look-alikes, from views
the robot collected" is, as of these searches. The grasp-gated fine-tune (A4) is a small, defensible novelty: grasp
labels are old (Grasp2Vec), but no paper uses grasp outcomes to admit the frames that train a just-taught object's
detector.

## Frame P3 as teach-to-correct-pick time, with the robot collecting the views and grading the labels

**Recommended framing.** "Show it once, pick the right one: on-board instance teaching among look-alikes." A
mobile manipulator with an Orin NX learns a never-seen object in ~10 s by orbiting it with its arm camera and depth
(BOP's static onboarding, done by the arm, with masks for free), picks it among same-shape look-alikes within a
minute of Teach with no training and no server, and then lets grasp outcomes admit its own frames into a per-object
detector that takes over once it beats the gallery. The hypotheses to test, written down before the runs:
- **H1** robot-collected views (2-4 rounds) give a lower wrong-twin rate than 5 phone photos (VAP-style) on the robot's
  own viewpoints;
- **H2** every concept-level prompt (text, YOLOE visual, SAM 3 exemplar, OWLv2 image query) stays near chance among
  twins while the DINOv2 gallery stays above 90 % identity;
- **H3** the grasp gate halves the wrong-twin rate of the fine-tuned detector under realistic label noise;
- **H4** the median time from Teach to the first correct pick stays under 60 s on board, with each stage's share.

**Baselines to beat.** Concept-level (expected to fail on twins, needed to show why): YOLOE-26 text prompts with a
person's description (on the Orin), Grounding DINO with the same text, YOLOE-26 visual prompts single-view and
orbit-averaged with negatives, SAM 3.1 exemplar boxes, OWLv2 image-guided. Instance-level (the real competitors):
VAP-style human photos + Grounding DINO + DINOv2; CNOS25-style YOLOE + DINOv3 ViT-L on the 5090 as the off-board upper
bound; an IDOW-style fine-tuned DINOv2 ViT-S; a Show-Don't-Tell-style YOLO26n fine-tuned on the orbit frames without
the grasp gate. Most run offline on recorded robot looks; live picks go to ours, one concept-level baseline, the
ungated fine-tune and the gated one.

**What to build, in place of A3** (about two weeks in all, against A3's planned 1-2):
1. A DINOv2 ViT-S/14 crop engine on both Orins (ONNX opset 17, FP16, static batch 8/16, 224 px), checked against
   PyTorch (cosine > 0.999) and timed p50/p99; the gallery stays a matrix outside the engine -- adding an object is a
   row, as in NanoOWL's `decode()` or DeepStream's few-shot app. 1-2 days.
2. The twin benchmark: ~24 objects in 6-8 groups (same shape different print; same object different colour; one
   small mark; the same print on different shapes) plus ~10 never-taught distractors; recorded Find looks with names;
   seeded layouts; a live protocol script. Release it. 3-4 days.
3. The offline baselines on the 5090. 3-4 days.
4. A4 with the grasp gate (X3 gripper read-back; on the M3, the piece gone from the next look), YOLO26n fine-tuned
   on hekate, the FP16 engine hot-swapped (INT8 buys nothing at our size, PLAN.md M5). 1-2 weeks, as planned.
5. Only for the X3's open-set proposals: the YOLOE prompt-input export. 2-4 days, optional.

**Experiments.** An offline identity study on thousands of recorded crops (identity top-1 among twins; unknown
rejection AUROC; views and *coverage* 1/4/8/16/24 and 1-4 sides; negatives none / margin / explicit; DINOv2 ViT-S/B
vs DINOv3 S/B vs fine-tuned; CLS vs CLS + mean patch; best-3 vs mean; proposals from depth, YOLOE prompt-free or the
kit detector). A live study of ~250 picks on the M3: four arms of ~60 trials, enough to separate 90 % from 70 %
success or a 5 % from a 25 % wrong-twin rate (alpha 0.05, power 0.8, est.), identity failures reported apart from
grasp failures (VAP's real failures split 50.9 / 49.1 %). The X3 as a second embodiment for the gallery tier (wrist
camera, no depth). Time split into teach, search, pick and drop -- our first 50 s included 9 s searching for a
drop-off box that was not there. The headline figure: correct-pick success against seconds since Teach (log axis),
the tiers as segments, the baselines as points.

**Venue.** IROS 2027 (1 March 2027) fits this as a systems paper with the full comparison; about 8-9 of the 18
remaining weeks of work (est.). RA-L is realistic only with the released twin set, the second embodiment and H3 holding;
RA-L's IROS presentation option has in past years shared the IROS deadline -- check the 2027 call. The negatives
question stays one ablation: the literature is split (T-Rex-Omni: negative prompts help
[2511.08997](https://arxiv.org/abs/2511.08997); FSS-SAM3: they collapse SAM 3's masks
[2604.05433](https://arxiv.org/abs/2604.05433); IDOW: +0.16 AP from distractor negatives).

## Risks

- **"CNOS or VAP on a robot."** The likeliest review. Only the comparisons answer it: robot-collected vs human views
  (H1), the time ladder with a measured crossover (H4, A4), the wrong-twin metric and a released twin set. Without
  them P3 is a demo, not a paper.
- **Scoop, 6-12 months.** VAP's group can move on board; RoboFind (Sep 2026) and EgoAsk (Sep 2026) show active groups
  on personal objects; Show, Don't Tell can add a timed robot study. Re-check arXiv in February.
- **Thresholds tuned on four cubes.** SAME = 0.65 and MARGIN = 0.05 must be calibrated on held-out objects, or the
  open-set curve reported instead of one operating point.
- **Grasping drowns identity.** The M3 grips ~1.5 cm short without its taught-aim corrections; score identity and the
  grasp separately.
- **Identical twins have no right answer.** The system must say "ambiguous" and ask; the metric must count that as
  correct.
- **Coverage limits of the arm.** The M3's steepest view is ~82 deg at 20 cm; faces it cannot see stay untaught (the
  lying can at 0.47). Turning the object between rounds is a person's act; say so, or let the robot turn it.
- **Licences.** YOLOE and YOLO26 are AGPL-3.0, DINOv2 Apache-2.0; DINOv3's licence was not checked.
- **Search coverage.** Searches ran out before CoRL/IROS 2026 proceedings, Toyota HSR and RoboCup@Home teaching
  pipelines and Stretch work were checked directly.

## Conclusion

A3's premise -- an engine that takes the object's description as an input -- is already done on Jetson by NanoOWL,
Isaac ROS Grounding DINO and YOLO-World exports, and the YOLOE version is a small export wrapper. It also targets the
wrong description: the one that names our look-alikes is a DINOv2 gallery of taught views, which already works at run
time and is the field's standard (CNOS, VAP, CNOS25). Two corrections to RESEARCH.md P3 follow: NanoOWL is not the
only edge stack with run-time queries, and Show, Don't Tell does use its fine-tuned detector in on-robot runs (only
its training is off-board, 4-7 min in all). P3 stays publishable if it claims what no paper we found measures: a
robot that collects the views itself, runs everything on an Orin NX, and picks the right one of several same-shape
twins within a minute of Teach, then hands over to a detector trained only on frames its own grasps confirmed.
Decision recommended: drop A3 as a contribution, spend its time on the DINOv2 crop engine, the twin benchmark and
the baselines, keep A4 with the grasp gate, and aim at IROS 2027.
