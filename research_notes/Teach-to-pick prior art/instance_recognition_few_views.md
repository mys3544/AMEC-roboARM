# Recognising one specific object from a few views: instance detection, matching and look-alikes (state 2026-10-27)

Scope: question (b) of RESEARCH.md section 10, topic 5 -- instance-level recognition from a few views (instance
detection benchmarks and methods, CNOS / SAM-6D / FoundPose-style template matching, DINOv2/DINOv3 matching,
personalised segmentation), how they treat same-shape look-alikes, and what that means for our A1-A2 result
(DINOv2 ViT-S/14 named 4 of 4 picture cubes, 84/84 leave-one-out; YOLOE's visual prompts 1 of 4). Robot systems that
use these methods are in `teach_and_fetch_robot_systems.md`; the run-time engines in `runtime_prompt_engines.md`.

Labels: **[REAL-robot]**, **[BENCH]**, **[CLAIM]**, **[2nd]** as in the other notes of this folder. Web searches ran
out partway; claims were checked on the primary page unless labelled [2nd].

## 1. The field's standard answer is "propose, then match with DINOv2/v3" -- exactly what A1-A2 do

### Takeaway
Since 2023 the strongest training-free way to find a specific object from reference views is: class-agnostic
proposals (SAM, FastSAM, Grounding DINO, or YOLOE) and DINOv2 (now DINOv3) descriptors matched against the reference
views. It won BOP's model-free 2D detection in 2025 (CNOS25: YOLOE + DINOv3) and is the best "off-the-shelf" baseline
on the instance detection benchmarks. Our A1-A2 pipeline (depth boxes + DINOv2 crops, best-3 views) is an instance
of it; the identity result is a replication, not a finding.

### Cited Findings
- CNOS (ICCV 2023 R6D workshop): SAM proposals + DINOv2 descriptors matched to CAD-rendered templates; +19.8 % AP over
  prior work on the seven BOP core datasets [BENCH] — [arXiv 2307.11067](https://arxiv.org/abs/2307.11067)
- HR-InsDet (NeurIPS 2023 D&B): 100 instances with multi-view captures; "the off-the-shelf class-agnostic segmentation
  model (SAM) and the self-supervised feature representation DINOv2 performs the best"; SAM + DINOv2 41.61 AP vs 27.99
  for a trained DINO detector; all methods poor on hard and small instances [BENCH, 2nd for the quote] —
  [arXiv 2310.19257](https://arxiv.org/html/2310.19257v1)
- IDOW (CVPR 2025): foundation models "have high recall but are not optimized for instance matching"; fine-tuning
  DINOv2 (default ViT-S/14, [CLS]) by metric learning with distractors as negatives and synthesised novel views: on
  HR-InsDet 57.01 AP (Grounding DINO proposals) vs 51.68 off-the-shelf, 48.75 vs 41.61 with SAM proposals; on
  RoboTools 59.4 vs 56.7, VoxDet 18.7; ViT-B/14 58.63; 24 references an instance on HR-InsDet, 100 on RoboTools;
  0.16-0.18 s an image (0.12 s of it Grounding DINO's proposals); "GroundingDINO is not real-time" [BENCH] —
  [arXiv 2503.00359](https://arxiv.org/abs/2503.00359), [HTML](https://arxiv.org/html/2503.00359)
- NIDS-Net: Grounding DINO + SAM proposals, instance embeddings as foreground-averaged DINOv2 patch tokens refined by a
  weight adapter; state of the art on four detection datasets and seven BOP core segmentation sets among RGB methods;
  checked on Fetch-robot images with a RealSense [BENCH, CLAIM] — [arXiv 2405.17859](https://arxiv.org/abs/2405.17859)
- VoxDet (2023): multi-view templates with poses -> a 3D template voxel, matched to query voxels; introduces RoboTools
  (20 instances, 24 cluttered scenes, 9k boxes) [BENCH] — [arXiv 2305.17220](https://arxiv.org/abs/2305.17220)
- CNOS25 (BOP 2025, Track 7): YOLOE-11L-seg proposals + DINOv3 ViT-L/16, 100 templates an object from the static
  onboarding sequence; BOP-H3 AP 0.441 vs 0.340 for FastSAM + DINOv2; 0.134 s an image on an RTX 4090; best fast and
  best open-source method [BENCH; award per README] — [vincirist/cnos25](https://github.com/vincirist/cnos25),
  [leaderboard](https://bop.felk.cvut.cz/leaderboards/modelfree-detection-unseen-bop24/bop-h3/)
- BOP 2024 model-free tasks: an object may be onboarded in "up to 5 minutes of the wall-clock time on a single
  computer with one GPU", from a static sequence (the camera moves round the still object) or a dynamic one (hands
  turn the object); best 2D detection then 31.9 AP (GFreeDet, 0.3 s an image with FastSAM) [BENCH] —
  [arXiv 2504.02812](https://arxiv.org/html/2504.02812). In 2025 the top BOP-H3 entry (3PT) reached 0.482 AP at 27 s an
  image — [leaderboard](https://bop.felk.cvut.cz/leaderboards/modelfree-detection-unseen-bop24/bop-h3/)
- NeMO (3DV 2026): a few RGB template views -> a sparse object point cloud (learned UDF) decoded with a query image
  for detection, segmentation and 6D pose; "quick object onboarding without retraining" [CLAIM] —
  [arXiv 2602.04343](https://arxiv.org/abs/2602.04343)
- Training-free reference-based instance segmentation ("No time to train!", 2025-26): memory bank of reference
  features, foundation-model correspondences; COCO-FSOD 36.8 nAP, PASCAL few-shot 71.2 nAP50 [BENCH] —
  [arXiv 2507.02798](https://arxiv.org/abs/2507.02798)

### Inferences
- BOP's "static onboarding" (the camera circles a still object) is what our A2 orbit does with an arm, and BOP's
  5-minute budget is the field's yardstick for "fast". Our Teach takes 10 s on an Orin NX with 4 views a round; the
  difference worth stating is on-board and seconds, not the matching method.
- IDOW is the cheapest upgrade with evidence: a DINOv2 ViT-S fine-tuned for instance metric learning on open data
  gave +5-7 AP over the frozen one. It is an ablation for P3, not a must.

## 2. Look-alikes: the methods know the problem, few benchmarks isolate it

### Takeaway
Same-category distractors are named as the hard case everywhere (PIN, VAP, IDOW, T-Rex-Omni), and they cost a lot
when measured (PIN: success 23.1 -> 14.8 when same-category distractors are present). But no benchmark we found
isolates *same-shape, different-print* twins like our picture cubes, and none reports a wrong-twin rate for picks.
Our "4 of 4 vs 1 of 4" result is a small but clean diagnostic: prompt embeddings of a real-time open-vocabulary
detector are category-level; DINOv2 crops are instance-level.

### Cited Findings
- PIN (NeurIPS 2024 D&B, Habitat + Objaverse): a target given by three reference images must be told from same- and
  other-category distractors; the best modular agent uses DINOv2: success 14.84 with same-category distractors vs
  23.13 without ("DINOv2 improves by 8.29") [BENCH] — [arXiv 2410.18195](https://arxiv.org/html/2410.18195v2)
- Bring My Cup! / VAP (ICML 2026): frozen VLAs "collapse to category-level recognition"; Grounding DINO proposes
  same-category candidates, DINOv2 [CLS] of ~5 reference crops votes; two same-category look-alikes in every episode;
  half of the real failures (50.9 %) were per-camera matches locking onto different instances, the other half failed
  manipulation [REAL-robot, SO-101] — [arXiv 2512.20014](https://arxiv.org/html/2512.20014),
  [v3](https://arxiv.org/html/2512.20014v3)
- IDOW samples SAM segments of random backgrounds as "universal negatives" and other instances as hard negatives; the
  distractor term adds only 54.10 vs 53.94 AP in its ablation [BENCH] — [HTML](https://arxiv.org/html/2503.00359)
- T-Rex-Omni (AAAI 2026): positive-only visual prompts are "vulnerable to visually similar but semantically
  different distractors"; negative visual prompts suppress them (training-free NNC module, NNH loss); 51.2 AP_r on
  LVIS-minival; no speed given [BENCH] — [arXiv 2511.08997](https://arxiv.org/abs/2511.08997)
- FSS-SAM3: in few-shot segmentation with a frozen SAM 3, negative prompts "weaken target representations" and can
  cause "prediction collapse" [BENCH] — [arXiv 2604.05433](https://arxiv.org/abs/2604.05433)
- ILIAS (CVPR 2025), instance-level retrieval at scale: domain-specific models "excel in that domain but fail on
  ILIAS"; a linear adaptation layer helps; "local descriptors in retrieval re-ranking are still a key ingredient",
  especially with clutter [BENCH] — [arXiv 2502.11748](https://arxiv.org/abs/2502.11748)
- NVIDIA retail embeddings (products often differ only in print): 0.7718 accuracy with 1 reference image a class,
  0.8605 with 90 (seen classes) [BENCH] — [model card](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/retail_object_recognition)
- Ours: two news cubes taught from one side stayed "unknown" (0.64 and 0.61 against the 0.65 bar, 2026-10-12);
  after four rounds turned a quarter each (19-23 views) every cube was named right (news 0.85 / 0.80, fish 0.85);
  a can taught standing from a scan and seen lying stayed unknown (0.47) [REAL-robot, ours] — PLAN.md track A

### Inferences
- Our misses were view coverage (an untaught face, a lying pose), not the embedding. That makes "which views does the
  robot need to collect" -- coverage, not count -- a measurable variable for P3 (AP against sides covered).
- A wrong-twin rate (the robot picks the look-alike) is the metric reviewers will ask for; RoboFind's "false success"
  (75 % -> 5 %, `teach_and_fetch_robot_systems.md`) is the closest published analogue.
- Negatives: the evidence is mixed (T-Rex-Omni helps, FSS-SAM3 collapses, IDOW +0.16 AP). On a DINOv2 gallery the
  cheap version is a margin rule against the look-alikes the robot has also seen (our MARGIN = 0.05); whether
  explicitly taught negatives beat the margin rule is an open, small experiment.

## 3. Concept-level promptable models (SAM 3, YOLOE, T-Rex) are not instance identifiers

### Takeaway
SAM 3's exemplar prompts and YOLOE's visual prompts define a *concept* and return every instance of it. SAM 3 keeps
instance-specific prompting in a separate mode. So "SAM 3 with an exemplar box" is a fair baseline that should fail
on same-category twins -- a useful negative row, not a competitor.

### Cited Findings
- SAM 3 "detects, segments, and tracks objects in images and videos based on concept prompts" (noun phrases, image
  exemplars or both) and returns masks "for all matching object instances"; 848M parameters; SAM 3.1 (Object
  Multiplex) released 2026-03-27 [CLAIM] — [arXiv 2511.16719](https://arxiv.org/abs/2511.16719),
  [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- The SAM 3 paper keeps instance-level edits from spilling to other instances through a hard switch from concept to
  instance mode and lists interleaving the two as future work [2nd: search summary of the PDF] —
  [arXiv 2511.16719](https://arxiv.org/pdf/2511.16719)
- SAM 3 on an Orin NX: ~0.95 s an image in TensorRT (ours, PLAN.M5); Embedl's quantised SAM 3 462 ms on an AGX Orin,
  validated only on TensorRT 10.1/10.3 [BENCH] — `research_notes/Model export formats for our hardware/vision_models.md` Q3
- PerSAM (ICLR 2024): one reference image + mask personalises SAM training-free; PerSAM-F tunes 2 parameters in ~10 s;
  PerSeg benchmark of 40 objects; no look-alike evaluation found [BENCH, 2nd] —
  [arXiv 2305.03048](https://arxiv.org/pdf/2305.03048)

### Inferences
- In P3 the baselines split cleanly into concept-level (text prompts, YOLOE visual prompts, SAM 3 exemplars, OWLv2
  image-guided) and instance-level (DINOv2/v3 gallery, IDOW-style fine-tuned DINOv2, a fine-tuned YOLO26 per object).
  The headline comparison is between the two groups on look-alike sets, then within the instance group on time.

### Gaps
- DINOv3 vs DINOv2 on instance retrieval with the same protocol: not found (searches ran out). CNOS25's gain over
  FastSAM+DINOv2 mixes the proposal model and the descriptor.
- No benchmark of same-shape, different-print twins was found; HR-InsDet, RoboTools and PIN mix shapes and categories.
- OWLv2's image-guided (one-shot) detection on look-alikes and on Jetson: not measured anywhere we looked.
