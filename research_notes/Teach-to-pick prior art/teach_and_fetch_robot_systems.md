# Robots that are shown a new object and then fetch it: systems, teach times, grasp-checked labels (state 2026-10-27)

Scope: questions (c) and (d) of RESEARCH.md section 10, topic 5 -- robot systems (2018-2026) that learn a new object
from views or a demonstration and then find or pick it, any measured time from teaching to use, self-labelling by
grasping, and whether "teach to the first pick in under a minute, on the robot, among look-alikes, with
grasp-confirmed labels" is still open in October 2026. Our reference numbers (M3 Pro, Orin NX): Teach 10.0-10.5 s
(the wide look + 4 views + DINOv2 + home), about 50 s of robot time from Teach to a never-seen yellow cylinder put
down, "pick fish" picked among three same-shape picture cubes (PLAN.md track A, 2026-10-09 and 10-15).

Labels: **[REAL-robot]**, **[BENCH]**, **[CLAIM]**, **[2nd]** as in the other notes. Web searches ran out partway
(about 33 searches done, a shared quota), so every "not found" below is bounded by those searches and the papers
read; it needs the usual two-hour re-check before a submission.

## 1. Systems that fetch a specific object shown by images

### Takeaway
Instance-level fetching from reference images is published and recent: GOAT (2023, image goals in homes), VAP /
"Bring My Cup!" (ICML 2026, SO-101 arm, two same-category look-alikes an episode), RoboFind (September 2026, a phone
teaches a quadruped). All of them get their reference views from a person (photos, a phone, a video), run the
models on a server or do not say, and none reports the time from teaching to the first successful use.

### Cited Findings
- **GOAT** (2023): multimodal goals (category, language, an image of the instance), instance-aware memory; image
  goals matched to object views by SuperGLUE keypoints, only within the goal's category ("improves matching accuracy
  by 23 %"); success 86.4 % image, 68.2 % language, 94.3 % category goals, 83 % overall over 9 homes, 675 goals, 200+
  instances; pick-and-place shown as an application [REAL-robot] — [arXiv 2311.06430](https://arxiv.org/html/2311.06430)
- **OK-Robot** (2024): open-vocabulary pick-and-drop from text queries, no training; 58.5 % in 10 homes, 82 % in
  uncluttered ones [REAL-robot] — [arXiv 2401.12202](https://arxiv.org/abs/2401.12202); "under 10 minutes on average"
  from arriving in a new home to the first pick-and-drop [2nd: search summary of the PDF]
- **DynaMem** (2024): a dynamic spatio-semantic memory on Stretch SE3; 70 % pick-and-drop on non-stationary objects,
  >2x static baselines; text queries [REAL-robot, 2nd] — [arXiv 2411.04999](https://arxiv.org/pdf/2411.04999)
- **VAP, "Bring My Cup!"** (Dec 2025, ICML 2026): a training-free adapter for frozen VLAs; ~5 reference images an
  object as non-parametric memory; Grounding DINO proposes same-category candidates, DINOv2 [CLS] cosine votes, SAM 2
  tracks; two same-category look-alikes in every episode; real SO-101: selection 80.0 % vs 30.0 % (pi0.5) and 40-45 %
  (text baselines); pick-and-place 58.8 % vs 18.8 % (generic) and 27.5-31.2 % (soft/hard prompts), 20 trials x 8 tasks;
  one-time initialisation 0.26 s, per step 0.02 s tracking + 0.20 s VLA on an RTX A6000; a new object "can be used
  immediately after it is registered" -- no enrolment time reported; 50.9 % of real failures were per-view matches
  locking onto different instances [REAL-robot] — [arXiv 2512.20014](https://arxiv.org/html/2512.20014),
  [v3 appendix](https://arxiv.org/html/2512.20014v3)
- **RoboFind** (17 Sep 2026): a smartphone Target Teaching Agent builds a multi-view reference bank (AR, speech and
  haptic guidance, for blind and low-vision users); a quadruped searches and a Verification Agent checks candidates
  against the bank; 85.0 % vs 25.0 % for a first-stop baseline over 20 trials and ten targets; false success 75.0 % ->
  5.0 %; grasping and compute location not stated [REAL-robot] — [arXiv 2609.20330](https://arxiv.org/abs/2609.20330)
- **PIN** (NeurIPS 2024 D&B): personalised instance navigation from three reference images with same-category
  distractors, simulation only; best agent DINOv2-based, 14.84 success with distractors vs 23.13 without [BENCH] —
  [arXiv 2410.18195](https://arxiv.org/html/2410.18195v2)
- **EgoAsk** (15 Sep 2026): smart glasses share the user's view so a household robot can ask about personal objects
  during daily activity; an 18-person HCI study; recognition, robot and pick metrics not stated [CLAIM] —
  [arXiv 2609.16766](https://arxiv.org/abs/2609.16766)

### Inferences
- VAP is the paper a P3 reviewer will cite first. Its identity stage is ours (detector proposals + DINOv2 [CLS] of a
  few reference crops); its look-alikes are same-category objects; its compute is a server. What it lacks against us:
  robot-collected views, on-board compute, a timed teach, grasp-checked labels, and same-shape twins.
- RoboFind's "false success" metric is the right shape for ours: the rate at which the robot reports or picks the
  wrong look-alike.

## 2. Measured teach-to-use times: seconds existed before foundation models; minutes are typical now

### Takeaway
"Learn a new object in seconds on the robot" was claimed by IIT's iCub on-line detection (2019-2020, a person holds
the object, kernel classifiers on CNN features). Current published systems report minutes to tens of minutes from
showing to use: Show, Don't Tell 4-7 min (Spot), a humanoid onboarding pipeline ~30 min, OK-Robot under 10 min to the
first pick in a new home, BOP's model-free budget 5 min. iTeach re-trains its segmenter on board in ~20 s an
interaction, but it learns class-agnostic masks, not identities. Our ~50 s from Teach to the piece put down would be
the fastest published teach-to-pick for an instance among look-alikes we found -- if measured properly.

### Cited Findings
- iCub on-line object detection (Maiettini, Pasquale, Rosasco, Natale; Autonomous Robots 44, 2020): combines their
  2017/2018 pipelines (FALKON kernel classifiers on region features) so "a robot can learn to recognize new objects
  within seconds", taught by a human showing the object [2nd: abstract via search] —
  [DOI 10.1007/s10514-019-09894-9](https://doi.org/10.1007/s10514-019-09894-9); follow-ups with weak supervision
  from handheld to cluttered scenes — [arXiv 2012.14345](https://arxiv.org/abs/2012.14345)
- **Show, Don't Tell** (Akl et al., 19 authors, affiliation not checked, 13 Mar 2026): a ~15 s human video -> an auto-labelled
  dataset -> a Faster R-CNN R50 fine-tuned in ~3-4 min on 4 T4 GPUs; the whole pipeline 4-7 min; ~100 ms an image at
  inference (YOLO-World similar, Grounding DINO ~400 ms, Rex-Omni 1-2 s), used as the detector in on-robot runs on a
  Spot; mAP 0.10-0.15 vs 0.00-0.08 for text-prompted Grounding DINO / YOLO-World on in-house sets (but Grounding DINO
  with human prompts wins on Meccano, 0.19 vs 0.06); temporal clustering separates similar objects; "Humans are
  sometimes required to address confusion by removing spurious detections or 'locking in' good ones"; grasp misses
  confirmed by a human; where inference runs is not stated [REAL-robot, BENCH] —
  [arXiv 2603.12751](https://arxiv.org/html/2603.12751)
- **Rapid humanoid onboarding** (19 Apr 2026, Unitree G1): Roboflow-assisted labels for YOLOv8, SAM 3D meshes,
  FoundationPose tracking; onboarding a new object cut from 1-2 days to ~30 min; grasps at five positions, no success
  rate [REAL-robot, CLAIM] — [arXiv 2604.17258](https://arxiv.org/abs/2604.17258)
- **iTeach** (v1 Oct 2024, revised 29 Sep 2026): a mixed-reality headset lets a non-expert correct segmentation on
  site; ~20 s a teaching interaction, 150-300 labelled frames each; the model is fine-tuned on board and redeployed;
  segmentation 26.1 -> 80.7 on cluttered scenes; SceneReplica pick-and-place 72/100 [REAL-robot] —
  [arXiv 2410.09072](https://arxiv.org/abs/2410.09072)
- **BOP model-free onboarding**: up to 5 min of wall-clock time an object on one GPU [BENCH] —
  [arXiv 2504.02812](https://arxiv.org/html/2504.02812)
- **NovaSplat** (9 Oct 2026): a phone video -> 3DGS used for detection, coarse pose and refinement; onboarding time not
  in the abstract; no grasping reported [BENCH] — [arXiv 2610.13309](https://arxiv.org/abs/2610.13309)

### Inferences
- "Under a minute" alone is not a claim (iCub claimed seconds in 2020). "Under a minute *to the first correct pick*,
  on board, among same-shape look-alikes, from views the robot collected itself" is.
- The number must be split: teach (robot motion + embedding), search (finding it again), the pick itself, and the
  drop -- our 50 s had 9 s of searching for a box that was not there. Report the median and IQR over objects, with
  every stage's share.

## 3. Labels confirmed by grasping are old as a training signal, new as a gate on a taught detector

### Takeaway
Robots have labelled objects with their own grasps since 2018 (Grasp2Vec), up to thousands of grasps to train an
instance segmenter (2023). None of these teaches one new, named object and then uses grasp outcomes to admit or reject
the frames that fine-tune that object's detector. A4's "keep only labels a grasp confirmed" is a small, defensible
novelty if it is measured against an unfiltered fine-tune.

### Cited Findings
- Grasp2Vec (CoRL 2018): scene-before minus scene-after embedding = the grasped object's embedding; used for
  instance detection, localisation and goal-conditioned (instance) grasping, with the embedding similarity as reward
  [REAL-robot, 2nd] — [arXiv 1811.06964](https://arxiv.org/pdf/1811.06964)
- Self-Supervised Instance Segmentation by Grasping (Liu, Chen, Abbeel, 2023): a grasp segmentation model finds the
  grasped object from before/after images; cut-and-paste of thousands of grasps trains an instance segmenter; 5x lower
  error than image subtraction, better than a model on 10x the labelled data, grasp errors down >3x [REAL-robot] —
  [arXiv 2305.06305](https://arxiv.org/abs/2305.06305)
- Robot-supervised segmentation with in-hand objects (2019): fine-tuning on robot-made labels 38.1 -> 49.8 AP [2nd] —
  [arXiv 1904.00952](https://arxiv.org/pdf/1904.00952)
- Self-supervised unseen object instance segmentation via long-term robot interaction (pushing, 2023) [2nd] —
  [arXiv 2302.03793](https://arxiv.org/pdf/2302.03793); self-supervisory signals for object discovery from ego-motion
  and depth (2018) [2nd] — [arXiv 1806.03370](https://arxiv.org/html/1806.03370)

### Inferences
- The grasp check is cheap on our robots: gripper read-back on the X3, and on both "the piece is gone from the table"
  in the next look. On the M3 the servos report commands, not positions, so the visual check is the signal.
- The experiment that makes it count: inject label noise the way the open-set stage really makes it (a look-alike
  admitted as the target), fine-tune with and without the grasp gate, measure the wrong-twin rate of the result.

## 4. Is the full combination open (October 2026)?

### Takeaway
Yes for the combination, no for each part. Every part has prior art: instance matching by DINOv2 (CNOS, VAP),
robot-collected multi-view references (BOP static onboarding, VoxDet, GOAT's memory), seconds-scale on-robot learning
(iCub 2020, iTeach), fine-tuned bespoke detectors (Show, Don't Tell), grasp-made labels (Grasp2Vec, Liu 2023). We
found no paper that combines robot-collected views, on-board compute on a Jetson-class robot, same-shape look-alikes,
a timed teach-to-first-correct-pick, and grasp-gated self-training.

| system | views from | runs on edge? | instance-level? | teach -> use | look-alikes | grasp-checked labels |
|---|---|---|---|---|---|---|
| GOAT 2023 | user image | not stated (Stretch/Spot) | yes (SuperGLUE) | not timed | same-category | no |
| OK-Robot 2024 | text | no (server) | no | <10 min to first pick in a home [2nd] | no | no |
| iCub on-line det. 2020 | human holds it | robot's PC | object-level | "seconds" [2nd] | iCubWorld objects | no |
| iTeach 2024-26 | MR headset + robot | on-board fine-tune | class-agnostic | ~20 s an interaction | no | no |
| Show, Don't Tell 2026 | 15 s human video | inference on Spot runs; training 4 T4 | partly (clustering) | 4-7 min | similar objects, human help | human confirms misses |
| VAP 2026 | ~5 user photos | no (A6000) | yes (DINOv2) | "immediately", not timed | 2 same-category | no |
| RoboFind 2026 | phone, guided | not stated | yes (verification agent) | not timed | personal items | no (no grasp) |
| humanoid pipeline 2026 | photos + SAM 3D | not stated | per-object YOLO | ~30 min | no | no |
| ours A1-A2 | the arm's own orbit + depth masks | yes, Orin NX | yes (DINOv2, 84/84 LOO) | ~50 s to put down | same-shape twins | planned (A4) |

### Gaps
- Searches ran out before checking CoRL 2026 and IROS 2026 proceedings directly, Toyota HSR / RoboCup@Home teaching
  pipelines, and Stretch "teach an object" work; re-check those before claiming novelty.
- No paper read reports a wrong-twin (wrong look-alike picked) rate for a real arm.
