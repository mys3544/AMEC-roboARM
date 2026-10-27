# Engines that take the object's description at run time: who already does it (state 2026-10-27)

Scope: question (a) of RESEARCH.md section 10, topic 5 -- whether an inference engine (TensorRT / ONNX) that takes
the object descriptions (text or visual prompt embeddings) as a run-time INPUT is still new, which edge stacks already
do it, with what latency, and whether PLAN.md's A3 ("graph surgery on the YOLOE export", 1-2 weeks) is worth it or
should be replaced by a detector plus a DINOv2 identity head. Hardware: Jetson Orin NX 16 GB (JetPack 6.2, TensorRT
10.7) on both robots; our own numbers come from PLAN.md M5 and track A, and the code in `m3pro/tools/teach_service.py`.
Export mechanics in general are covered by the other session's `research_notes/Model export formats for our
hardware/vision_models.md` and are not repeated here.

Labels: **[REAL-robot]** measured on a real robot (ours or theirs); **[BENCH]** benchmark or bench timing;
**[CLAIM]** an author's or vendor's statement without a measurement we could check; **[2nd]** secondary source or
search summary, not verified against the primary. Web searches ran out partway through this session (a shared quota);
everything below was checked by fetching the primary page unless labelled [2nd].

## 1. Prompts as run-time inputs on Jetson exist since 2023-24, including NVIDIA's own stacks

### Takeaway
"The description as an engine input" is not new. NanoOWL (NVIDIA-AI-IOT) keeps the image encoder in TensorRT and
matches its class embeddings against text embeddings computed at run time; Jetson Platform Services wraps it as a
service whose classes change by a REST call; Isaac ROS 5.0 ships Grounding DINO as a TensorRT engine whose text
tokens are engine inputs, with a `set_prompt` service; community ONNX exports of YOLO-World take the text embeddings
(`txt_feats`) as a graph input. What none of them does is take a *visual* prompt for a *specific instance* -- and
our own measurement says YOLOE's visual prompts do not reach instance level anyway (section 3).

### Cited Findings
**NanoOWL (OWL-ViT in TensorRT)**
- The TensorRT engine covers only the image encoder: `export_image_encoder_onnx` exports input `image` and outputs
  `image_embeds`, `image_class_embeds`, `logit_shift`, `logit_scale`, `pred_boxes`. The text encoder runs in PyTorch /
  Hugging Face (`encode_text` -> `owlvit.text_model` -> `text_projection`), and the class scores are computed in
  PyTorch in `decode()` (normalised embeddings, `einsum`, shift and scale, sigmoid). Only text queries are supported
  (no image-conditioned queries) and only OWL-ViT checkpoints (B/32, B/16, L/14), no OWLv2 —
  [nanoowl/owl_predictor.py](https://raw.githubusercontent.com/NVIDIA-AI-IOT/nanoowl/main/nanoowl/owl_predictor.py)
- README table: OWL-ViT B/32 at 768 px 95 FPS and B/16 25 FPS on an AGX Orin (Orin Nano "TBD"), precision and
  method not stated [CLAIM] — [NanoOWL README](https://github.com/NVIDIA-AI-IOT/nanoowl)
- A user on an AGX Orin 64 GB (MAXN, jetson_clocks) measured ~40-60 ms per OWL-ViT B/32 inference (~55 ms at
  640x480, ~18 FPS) and 400+ ms for OWLv2 base/16 at 900 px; NVIDIA could not explain the gap beyond suggesting
  batch 32 (May-July 2024) [2nd: single user] —
  [NVIDIA forum](https://forums.developer.nvidia.com/t/nanoowl-inference-takes-more-time-than-nanoowl-official-github-shows/293051)
- Jetson Platform Services' zero-shot detection service credits NanoOWL; classes are set at run time by `POST
  /api/v1/detection/classes` with `objects`, `thresholds` and a stream `id`; no FPS figures; one stream; classes do
  not persist across reboots — [JPS zero-shot detection](https://docs.nvidia.com/jetson/jps/inference-services/zero_shot_detection.html)

**Grounding DINO (NVIDIA TAO, Isaac ROS)**
- TAO's TensorRT build has six inputs: `inputs` (3x544x960), `input_ids`, `attention_mask`, `position_ids`,
  `token_type_ids` (256 tokens each) and `text_token_mask` (256x256), batch 1-16 — the text is an engine input —
  [TAO trtexec with Grounding DINO](https://docs.nvidia.com/tao/tao-toolkit/latest/text/trtexec_integration/trtexec_grounding_dino.html)
- Isaac ROS 5.0 `isaac_ros_grounding_dino`: a `set_prompt` service "to set a new text prompt for object detection"
  and a `default_prompt` parameter; inference "requires significant GPU compute"; engine conversion "can take up to
  10-15 minutes" on a Jetson AGX Thor; no Orin FPS given (page updated 2026-09-21) —
  [Isaac ROS Grounding DINO](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_grounding_dino/index.html)
- A 2026 edge-inference benchmark reportedly measured Grounding DINO at ~1083 ms (TensorRT) vs ~1291 ms (PyTorch) per
  inference on a Jetson GPU [2nd: search summary; the abstract of
  [arXiv 2607.11356](https://arxiv.org/abs/2607.11356) only says TensorRT did not beat PyTorch for its transformer model]

**YOLO-World**
- Official deploy docs: ONNX export takes `--custom-text` (the vocabulary is fixed at export), `--without-nms`,
  `--without-bbox-decoder`; the TensorRT section reads "coming soon"; no image prompts —
  [YOLO-World deploy.md](https://github.com/AILab-CVC/YOLO-World/blob/master/docs/deploy.md)
- A third-party ONNX of YOLOv8-World v2 has inputs `images` and `txt_feats` [batch, num_classes, 512] (CLIP ViT-B/32
  text embeddings), num_classes dynamic by patching `WorldDetect.forward` to split by `text.shape[1]`; Ultralytics
  8.4.37, opset 18, onnxslim; no TensorRT or Jetson numbers [CLAIM] — [Instemic/yolo-world-onnx](https://huggingface.co/Instemic/yolo-world-onnx)
- Another repo exports "with the ability to accept the class embeddings as input" but with a FIXED class count
  ("Setting a dynamic number of classes gives an error") —
  [ibaiGorordo/ONNX-YOLO-World](https://github.com/ibaiGorordo/ONNX-YOLO-World-Open-Vocabulary-Object-Detection)

**YOLOE (what we run)**
- Ultralytics: classes set by `set_classes()` or `refer_image` "are baked into the exported weights"; "Once
  exported, the model can no longer accept new prompts"; the saved NPZ of prompt embeddings "is not an additional
  runtime input"; RepRTA "is folded away at export" — [Ultralytics YOLOE docs](https://docs.ultralytics.com/models/yoloe/)
- A user's question whether `set_classes` / `get_text_pe` work on a YOLOE TensorRT engine was closed as stale, not
  planned — [ultralytics#20155](https://github.com/ultralytics/ultralytics/issues/20155)
- A third-party package (onnx-predictor-yolo) claims support for YOLOE graphs "that expose a prompt input" [2nd: the
  PyPI page did not load] — [onnx-predictor-yolo](https://pypi.org/project/onnx-predictor-yolo/0.2.0/)
- THU-MIG's own repo: after re-parameterisation YOLOE "can be exported into the identical format as YOLOv8 / YOLO11";
  T4 TensorRT FPS with text / visual prompts e.g. YOLOE-11-S 301.2 / 73.3 [BENCH] —
  [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe)
- Ours: YOLOE-26s-seg with 3 text prompts baked in, FP16 480x640, 5.36 ms median on the M3's Orin NX (trtexec GPU
  compute), ~8x faster than the ~41 ms PyTorch service [BENCH, ours] — PLAN.md M5 table (2026-10-12)

**Other open-vocabulary detectors with a text path at run time**
- OmDet-Turbo: 100.2 FPS with TensorRT "and language cache" on an A100, ONNX export without post-processing; no edge
  numbers [CLAIM, 2nd] — [om-ai-lab/OmDet](https://github.com/om-ai-lab/OmDet)
- DOSOD (D-Robotics): YOLO-World-like, text embeddings mapped by an MLP adaptor into a joint space, runs like a
  closed-set detector at test time; DOSOD-S 1,582 FPS on an RTX 4090; an RDK X5 section exists (not read) [2nd] —
  [arXiv 2412.14680](https://arxiv.org/abs/2412.14680)

### Inferences
- RESEARCH.md P3 says "NanoOWL is the only edge stack found that changes its queries while it runs." That is out of
  date: NVIDIA's own Isaac ROS Grounding DINO and Jetson Platform Services do it too, and YOLO-World has community
  exports with the embeddings as a graph input. A reviewer will know at least NanoOWL and Isaac ROS.
- The pattern is always the same: everything image-side in the engine, the class/query embeddings as an input (or
  matched outside the engine). Nothing here is research; it is plumbing.

### Gaps
- No published Orin NX latency for Grounding DINO (Isaac ROS) or for any YOLO-World / YOLOE engine with a prompt
  input; no TensorRT build of the Instemic export was found.
- The onnx-predictor-yolo claim about YOLOE prompt-input exports could not be opened.

## 2. In Ultralytics' YOLOE head the prompt is already a tensor input; A3's "surgery" is a small export wrapper

### Takeaway
In Ultralytics 8.4.175 (our version), the unfused YOLOE head reads the prompt embeddings as the LAST element of its
input list, sets `nc` from that tensor's shape, and scores each anchor by an `einsum` against them. Export only
becomes static because `fuse()` folds the embeddings into a 1x1 conv. An export that skips the fuse and exposes the
embeddings as a second graph input is the same patch the community made for YOLO-World: days, not 1-2 weeks.

### Cited Findings
- `YOLOEDetect.forward(self, x)`: the docstring says `x` is the feature maps "followed by class prompt embeddings
  with shape (B, nc, embed)"; `forward_head` sets `self.nc = int(x[-1].shape[1])` and computes
  `contrastive_head[i](cls_head[i](x[i]), x[-1])`; `fuse(txt_feats)` folds them (`w = t @ w`) into the conv weights —
  [ultralytics head.py](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/nn/modules/head.py);
  checked in the local copy, ultralytics 8.4.175 (`C:\Users\Mys\.roboarm\train-gpu\Lib\site-packages\ultralytics\nn\modules\head.py`, line 1312-1314)
- `BNContrastiveHead.forward`: `x = self.norm(x)`, `w = F.normalize(w)`, `einsum("bchw,bkc->bkhw", x, w) *
  logit_scale.exp() + bias` — a batched matmul TensorRT handles natively; local `block.py` lines 812-826
- The visual prompt itself (SAVPE) needs the backbone features of the reference image and a mask
  (`get_vpe(x, vpe)`); in our pipeline it is computed once per Teach in the PyTorch service (`/embed`), so it never
  needs to be inside the per-frame engine — `m3pro/tools/teach_service.py`

### Inferences
- A3 as written (1-2 weeks of graph surgery as "the engineering contribution") overestimates both the work and the
  novelty. Expected work: wrap the unfused model so `forward(image, prompts)` appends `prompts` to the feature list,
  export ONNX with a dynamic (or fixed max, e.g. 32 with masking) class axis, check that YOLO26's end-to-end top-k
  head tolerates it, build on the Orin. Est. 2-4 days including validation against the baked engine.
- What it would buy: no 1-minute engine rebuild after a Teach, and YOLOE at ~5 ms instead of ~41 ms. Neither changes
  the user-visible number, because (i) the PyTorch service already accepts new prompts with no rebuild, and (ii) the
  identity that names the object is DINOv2's, which already takes the taught views at run time (section 3).

## 3. Our own result decides it: the run-time description that matters is the instance gallery, and it already exists

### Takeaway
The object's description that has to be a run-time input is the set of DINOv2 embeddings of its taught views (a
"gallery"), not YOLOE's prompt. That is already how A1-A2 work, and it is how NVIDIA ships few-shot retail
recognition on Jetson and how the fastest BOP 2025 model-free detector works (YOLOE proposals + DINOv3 matching).
Replace A3 with an engine for the identity head (DINOv2 ViT-S/14 crops, FP16, static batch) and keep the gallery
matching as a matmul outside it.

### Cited Findings
- Ours, 2026-10-09: YOLOE-26 visual prompts named every picture cube "cosmetics" (0.15-0.46); its prompt embeddings
  matched view to view named 1 of 4 (at 640 and 1280 px); DINOv2 ViT-S/14 on the crops named 4 of 4 (84/84
  leave-one-out), the right name 0.09-0.17 ahead [REAL-robot, ours] — PLAN.md track A; `teach_service.py` docstring
- Ours, 2026-10-15: Teach 10.0-10.5 s; a never-seen yellow cylinder found at 0.95 (next name 0.58), ~50 s of robot
  time from Teach to the piece put down, no rebuild of anything [REAL-robot, ours] — PLAN.md track A, NEXT.md
- NVIDIA TAO Retail Object Recognition: a ResNet-101 trunk + 2048-d embedder that "predicts their labels based on
  the embedding vectors in the reference space", deployed as a DeepStream secondary GIE behind a retail-item detector
  in a "deepstream-fewshot-learning-app"; "20-30 images/class for reference data is of the highest efficiency"; the
  embedding head 0.7718 with 1 reference image a class, 0.8605 with 90 (seen classes); Orin NX 16 GB FP16 batch 4
  59.8 images/s (trtexec) [BENCH, CLAIM] —
  [model card](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/retail_object_recognition),
  [MMS docs](https://docs.nvidia.com/mms/text/FSL_Retail_Object_Recognition.html)
- CNOS25: YOLOE (`yoloe-11l-seg`) proposals + DINOv3 ViT-L/16 descriptors, templates from the static onboarding
  sequences (100 a object), 0.134 s an image on an RTX 4090; BOP-H3 AP 0.441 vs 0.340 for the FastSAM+DINOv2
  baseline; README: winner of the Best Fast Method and Best Open Source Method awards, Track 7 (model-free 2D
  detection of unseen objects), ICCV 2025 [BENCH; award = CLAIM] — [vincirist/cnos25](https://github.com/vincirist/cnos25);
  leaderboard row (0.441, 0.134 s, 2025-10-08) on
  [BOP-H3 model-free detection](https://bop.felk.cvut.cz/leaderboards/modelfree-detection-unseen-bop24/bop-h3/)
- DINOv2 on Jetson: NVIDIA's NV-DINOv2 classification model ~31.55 FPS on an Orin NX at batch 16, 224x224 (backbone
  size not checked; probably ViT-L) [2nd] — [NGC NV-DINOv2](https://catalog.ngc.nvidia.com/orgs/nvaie/-/models/nv_dinov2_classification_model/);
  no published DINOv2 ViT-S/14 TensorRT latency on an Orin NX was found (the export-formats notes say the same)

### Inferences
- The engine that matters for P3's latency is the identity head: N crops (N <= 16) x ViT-S/14 at 224 px, FP16,
  static batch. Its output (N x 384) times the gallery (384 x M, M ~ 100 taught views) is microseconds on the CPU,
  so the gallery never needs to enter an engine: adding or forgetting an object is a row change, exactly as in
  NanoOWL's `decode()` or DeepStream's few-shot app.
- A single engine fusing detector + ROIAlign + DINOv2 is possible but not worth it: ROI features from the detector's
  own backbone are category-level (the same thing that failed in YOLOE's prompts), while DINOv2 on a full-resolution
  crop is what reached 84/84. Two engines (proposals, identity) and one matmul is the design.
- Proposals: on the M3 the depth already boxes every object; on the X3 (no depth) a prompt-free or generic-prompt
  YOLOE engine, or the kit detector, gives them. A prompt-input YOLOE export (section 2) is worth doing only for the
  X3's open-set proposals, and only as the 2-4-day task, not as a contribution.

### Gaps
- Our DINOv2 crop engine's latency and its FP16 agreement with PyTorch (cosine > 0.999 on our crops) are unmeasured.
- The NVIDIA retail card reports accuracy on seen classes only; its behaviour on new classes is implied, not shown.

## 4. Recommendation for A3

### Takeaway
Drop "A3 = prompt-as-input YOLOE engine" as a contribution. Spend 2-4 days on (1) a DINOv2 ViT-S/14 crop engine with
the gallery as a run-time matrix and a latency table on both Orins, and (2) optionally the YOLOE prompt-input export
for the X3's proposals. Put the saved week into the look-alike benchmark and the baselines that P3 needs
(`p3_framing_baselines_experiments.md`).

### Evidence summary
| stack | description as run-time input? | how | on Jetson | instance-level? |
|---|---|---|---|---|
| NanoOWL / JPS zero-shot | yes (text) | image encoder in TRT, text + decode in PyTorch; REST classes | AGX Orin 95 FPS B/32 [CLAIM], 40-60 ms user [2nd] | no (text categories) |
| Isaac ROS Grounding DINO (TAO) | yes (text tokens) | 5 text inputs to the TRT engine; `set_prompt` | ~1 s [2nd] | no |
| YOLO-World community ONNX | yes (`txt_feats`) | patched head, dynamic N | not reported | no |
| YOLOE (Ultralytics export) | no (baked) | `fuse()` folds prompts into a 1x1 conv | ours 5.36 ms, baked | no (ours: 1/4) |
| NVIDIA retail recognition | yes (reference gallery) | detector + embedding SGIE, gallery outside | Orin NX 59.8 img/s embedder [BENCH] | product-level |
| CNOS25 (BOP 2025) | yes (template descriptors) | YOLOE proposals + DINOv3 matching | RTX 4090 0.134 s | instance (object id) |
| ours, A1-A2 | yes (DINOv2 gallery) | depth boxes + DINOv2 crops, PyTorch | Orin NX, timing not split | yes (4/4, 84/84) |
