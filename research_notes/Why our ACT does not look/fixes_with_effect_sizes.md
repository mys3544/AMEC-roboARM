# Fixes that make a visuomotor policy look, with published effect sizes and their cost on our Orin

Scope: the changes to the input, the training and the model that the literature measured against "the policy ignores
where the object is", ranked by evidence and by what they cost us -- code in LeRobot 0.6.1 (the newest release, 3 Aug
2026, [PyPI](https://pypi.org/project/lerobot/)), training time on the A4500 / 5090, and inference on the M3's Orin NX,
where the ACT runs as ONNX -> TensorRT FP16 in 6 ms at 10 Hz. Relative vs absolute actions are covered in
`research_notes/Presný tréning VLA pre lacné rameno/policy_design_for_precision.md` section 1 and only summarised here.
Data design is in `data_design_and_next_runs.md`.

Labels: **[REAL-LC]**, **[REAL-other]**, **[SIM]**, **[CLAIM]**, **[2nd]** as in the other notes; **[OURS]** our own
measurement; **[EST]** our own estimate, not a measurement.

## 1. Remove the state from the network's input (or starve its gradient)

### Takeaway
The strongest single lever in the literature, measured on ACT itself: no state input took ACT from 0 to 0.933
(height) and from 0.084 to 0.517 (horizontal) on a real arm. Partial measures recover less: noising the state 0.633,
random masking above concatenation but mixed against vision-only, gradient scaling (GAP) above vision-only. For us it costs nothing at run time. LeRobot
0.6.1's ACT says the state is optional, but its code still reads the state's device in two places, so a dataset without
`observation.state` raises a KeyError -- a 2-line patch or a constant state is needed.

### Evidence
- State-free ACT and Diffusion Policy, real dual-arm, dual wide-angle wrist cameras: ACT 0 / 0.084 -> 0.933 / 0.517,
  DP 0 / 0 -> 0.867 / 0.533 (height / horizontal) [REAL-other] -- [arXiv 2509.18644 HTML](https://arxiv.org/html/2509.18644v2)
- Same paper, pi0 with state: +-5 cm noise on the state's height 0.633, more varied tables 0.117, without state 0.983
  [REAL-other].
- GAP (ICLR 2026): vision-only beats vision+proprio on every reported task; random proprio masking ("Mask") 86.4 /
  89.2 / 60.4 % on pick-place / assembly / bin-picking vs vision-only 91.8 / 82.6 / 63.2; GAP 94.2 / 94.2 / 70.4 [SIM];
  real cube 14 / 11 / 17 of 20 (vision-only / concat / GAP) [REAL-other] -- [arXiv 2602.12032 HTML](https://arxiv.org/html/2602.12032)
- LeRobot 0.6.1 `configuration_act.py` docstring: "May optionally work without an 'observation.state' key";
  `modeling_act.py` uses `batch[OBS_STATE].device` for the VAE padding mask and for the zero latent (read in the
  installed package) -- without a state key both raise. Fix in our launcher: take the device from the image tensor, or
  remove `observation.state` from `policy.input_features` and keep it in the dataset (needed as the anchor for relative
  actions, section 6).
- What it costs: nothing on the Orin (one token fewer in a 302-token encoder); the export drops the `state` input or
  keeps it only to add back relative actions.
- What it may break: the state-free paper's limitation is background sensitivity; with absolute joint targets the
  network must read the arm's pose from the wrist image, so the chunk boundaries may jump (we already see the first step
  of each chunk worse than holding still: J1 3.6 vs 1.4 deg, NEXT.md). Relative actions (section 6) remove that need.

## 2. Image augmentation: random shift / crop and colour

### Takeaway
Every modern image-BC recipe trains with random shifts or crops; robomimic measured 35-47 % relative loss in sim and 73
-> 27 % on a real arm without them. LeRobot ships the transforms but leaves them off. For a wrist camera with absolute
targets a shift is label noise on the bearing, so keep it small (<= 3 %, ~2 deg at START); colour jitter also narrows the
sim's lighter colours. Zero cost at run time.

### Evidence
- robomimic pixel-shift randomisation: -47 % / -35 % relative without it (Square / Transport) [SIM]; real Can 26.7 %
  without vs 73.3 % with [REAL-other] -- [arXiv 2108.03298 HTML](https://arxiv.org/html/2108.03298)
- Diffusion Policy trains with random crop (~90 % of the frame) and evaluates a centre crop [REAL-other recipe] --
  [arXiv 2303.04137 HTML](https://arxiv.org/html/2303.04137v5)
- Learning from scratch + augmentation is competitive with frozen pretrained encoders (abstract) [SIM, REAL-other] --
  [Hansen et al., arXiv 2212.05749](https://arxiv.org/abs/2212.05749)
- LeRobot 0.6.1: `--dataset.image_transforms.enable=true` turns on up to 3 of brightness / contrast (0.8-1.2),
  saturation (0.5-1.5), hue (+-0.05), sharpness (0.5-1.5) and RandomAffine (+-5 deg, +-5 % translate) per frame; also
  available by name: GaussianNoise, MotionBlur, JPEGCompression, GaussianPatchBrightness, RandomShadow, CoarseDropout,
  GammaCorrection, PlanckianJitter, and any torchvision v2 transform (read in the installed package).
- Our camera: fx 480.8 px at 640 px wide (NEXT.md) -> 1 deg of bearing ~ 8.4 px; a 5 % translate (32 px) is ~3.8 deg,
  3 % ~2.3 deg [EST]. The policy re-plans every second and the close look corrects, so a few degrees of noise on the
  departure is tolerable; on the final centimetres it is not -- one reason to prefer relative actions with shifts.

## 3. A different encoder: frozen foundation features are not the quick fix

### Takeaway
Frozen encoders lose to fine-tuned ones in policy learning, sometimes completely (frozen DINOv2 0.00 vs fine-tuned 0.90
on a real pour task). Fine-tuned DINOv2/v3 beat ResNets by up to ~10 points where measured. Since our failure is the
training signal (shortcut + memorisation) and not yet shown to be the features, an encoder swap is the third lever, not
the first. ACT already keeps the 15x20 feature grid with 2D position embeddings, so a spatial softmax adds nothing
structural to ACT (it matters for Diffusion Policy's global pooling).

### Evidence
- Lin et al., data scaling laws (ICLR 2025): "fine-tuning the DINOv2 ViT outperforms both ImageNet pre-trained ResNet
  and CLIP ViT"; Pour Water: fine-tuned DINOv2 ViT-L/14 0.90, LoRA 0.72, **frozen 0.00**, ViT-L from scratch 0.03
  [REAL-other] -- [arXiv 2410.18647 HTML](https://arxiv.org/html/2410.18647)
- Diffusion Policy Table 5 (robomimic Square, ResNet18): scratch 0.94, frozen 0.58, fine-tuned 0.92 [SIM]; real Push-T
  (Table 6): frozen ImageNet 0.15, frozen R3M 0.80, end-to-end 0.95 [REAL-other] --
  [arXiv 2303.04137 HTML](https://arxiv.org/html/2303.04137v5)
- DINOv3 in a diffusion policy: "up to a 10 % absolute increase" over ResNet-18 fine-tuned (Can); frozen DINOv3
  "competitive" (abstract) [SIM] -- [arXiv 2509.17684](https://arxiv.org/abs/2509.17684)
- Burns et al. (CoRL 2024): among ViTs, emergent segmentation predicts out-of-distribution success better than ImageNet
  accuracy; manipulation-specific pretraining (R3M-style) did not generalise better; on ALOHA it predicted success after
  50 demonstrations [SIM, REAL-other] -- [arXiv 2312.12444](https://arxiv.org/abs/2312.12444)
- Wang et al., "Feature Extractor or Decision Maker": end-to-end visual encoders take part in decisions; out-of-domain
  pretrained encoders averaged a 42 % drop against end-to-end policies [SIM] -- [arXiv 2409.20248](https://arxiv.org/abs/2409.20248)
- Spatial softmax: Diffusion Policy replaces ResNet's global average pooling with it "to maintain spatial information"
  (no ablation number) -- same DP paper; LeRobot's diffusion config default 32 keypoints.
- Orin cost: ResNet18 at 640x480 is ~11 GFLOPs (1.8 GFLOPs at 224x224 scaled by 6.1x the pixels) against ~4.6 GFLOPs
  for DINOv2 ViT-S/14 at 224x224 [EST]; no measured DINOv2 ViT-S latency on an Orin NX was found (searched NVIDIA's
  model card listing and forum results; the parallel export-formats research found none either and lists a ViT-S
  encoder at 30.75 ms FP16 on an Orin Nano, 364x490 input -- `research_notes/Model export formats for our hardware/
  vision_models.md`, Q2). A 224 input would shrink the 30 mm cube from ~30-40 px to ~10-15 px at the look pose (35-50 cm away, fx 480.8)
  [EST].

## 4. Tell the policy which piece: a mark on the image, a mask, or a goal token

### Takeaway
The picker already knows the target. Drawing it into the picture (PEEK-style) roughly doubled ACT's real success on
BRIDGE tasks and, in sim, masks alone added 32 points; an explicit 2D goal added ~14 points to object-centric features.
For us the cheapest mark is the target's 3D point (from the look) projected into every frame by the commanded pose and
the twin's camera model -- ~0 ms on the Orin and no change to the engine's inputs. A low-dimensional goal token
(ACT's `observation.environment_state`) is even simpler but opens a new shortcut: target + state -> trajectory without
looking.

### Evidence
- PEEK (Zhang et al., Sept 2025): a VLM predicts a path and task-relevant points; the path is drawn as a red polyline
  and a mask keeps only squares (8 % of the image side) around the points, everything else black; policies keep their
  architecture. ACT on WidowX/BRIDGE, 5 trials a task: basic 0.20 -> 0.40, clutter 0.00 -> 0.27, semantic 0.00 -> 0.07
  ("3.4x" overall); pi0 0.33 -> 0.67 / 0.20 -> 0.40 / 0.07 -> 0.40 [REAL-other]; sim ablation (3D-DA, cube stacking):
  neither 33.5, paths 52.8, masks 65.6, both 73.6 % [SIM]; VLM queried every 25-32 steps, 4-6 s per query on an RTX
  3090 -- [arXiv 2509.18282 HTML](https://arxiv.org/html/2509.18282v1)
- More Structure, Not More Capacity: object-centric slots 55.0 % vs dense 32.6 %; + explicit 2D goal and native
  resolution 68.7 % (oracle 71.7 %) [SIM] -- [arXiv 2607.09825](https://arxiv.org/abs/2607.09825)
- ARRO (Mirjalili et al., 2025-26): open-vocabulary segmentation masks out task-irrelevant pixels and overlays guides at
  training and inference, no extra training; "consistent performance gains" for Diffusion Policy, Octo, OpenVLA, pi0
  (abstract, no numbers) [CLAIM] -- [arXiv 2505.08627](https://arxiv.org/abs/2505.08627)
- Our ingredients: the target's x, y, z is in every pick record; the arm's commanded pose is in every tick; the twin's
  camera model holds to ~5 mm rms on the table (NEXT.md, 2026-10-26); the kit detector (YOLO26n-seg, 480x640) runs at
  3.00 ms FP16 on the Orin (PLAN.md M5; `research_notes/Model export formats for our hardware/vision_models.md`) if the
  mark must come from pixels near the cube.

## 5. A diffusion or flow head instead of ACT's L1 head

### Takeaway
Not now. A generative head would commit to one piece instead of the median heading, but the 2025 evidence says the
gain of generative heads is not multimodality; on small low-cost-arm datasets ACT has been the more reliable learner;
LeRobot's Diffusion Policy by default feeds two steps of state (more history -> a stronger copycat input) and runs a
large U-Net for many denoising steps. Revisit when the VLA (GR00T N1.7, flow head) is trained -- with the state handled as
in section 1.

### Evidence
- "Much Ado About Noising" (Pan, Simchowitz et al., 2025-26): generative control policies do not win by capturing
  multimodal distributions; a two-step regression policy roughly matches flow policies [SIM] --
  [arXiv 2512.01809](https://arxiv.org/abs/2512.01809)
- Bimanual SO-101, identical demos: ACT 19/20, Diffusion Policy 0/10 at twice the steps [REAL-LC] -- from the
  2026-10-26 notes (`research_notes/Simulácia a VLA pre presný úchop/act_vs_vla_sim_trained.md`, arXiv 2608.03938)
- SO-100 grab-to-bowl, 76 episodes: ACT 8/10; Diffusion Policy "unstable" (model card) [2nd] --
  [Hugging Face](https://huggingface.co/Two-Zero/act-so100-grab-to-bowl)
- LeRobot 0.6.1 diffusion defaults (installed package): `n_obs_steps 2`, `horizon 64`, ResNet18 ImageNet,
  `spatial_softmax_num_keypoints 32`, `crop_is_random True`, U-Net down dims 512/1024/2048.

## 6. Relative actions: the anchor stays, the input goes

### Takeaway
Chunks relative to the commanded pose make every frame's target depend on where the cube is relative to the gripper --
visual-servo supervision on all frames, not only at START -- and remove the absolute-pose recall that state-free
absolute targets would need. The state-free study only generalised with relative end-effector actions; in-domain all
action spaces worked. LeRobot 0.6.1 has `use_relative_actions` for pi0 / pi0.5 / pi0-FAST / GR00T only; for ACT the
existing `RelativeActionsProcessorStep` must be inserted by our launcher and the stats recomputed in relative space.

### Evidence
- State-free Pick Pen (Table III): relative EEF 1.0 / 0.984 / 0.584; absolute EEF, relative joint, absolute joint 1.0 /
  0 / 0 (in-domain / height / horizontal) [REAL-other] -- [arXiv 2509.18644 HTML](https://arxiv.org/html/2509.18644v2)
- Chunk-wise relative beat absolute by ~8-15 points in-domain; UMI 20/20 relative vs 5/20 absolute [REAL-other] --
  summarised in `policy_design_for_precision.md` section 1 (2026-10-20).
- LeRobot 0.6.1 (installed): `RelativeActionsProcessorStep` ("action -= state" for masked dims, caches the state for
  `AbsoluteActionsProcessorStep`), `compute_relative_action_stats` and `lerobot-edit-dataset ... relative_action`;
  `use_relative_actions` exists in the pi0, pi05, pi0_fast and groot configs, not in ACT's.
- Orin cost: adding the state back to the chunk inside the ONNX graph -- negligible.

## 7. Weight the decision frames

### Takeaway
The heading is decided in ~2 % of our frames [OURS]. Upweighting the decisive frames is cheap and measured: Wen et al.'s
keyframes x5 gave +10 points in CARLA; GAP's phase-scaled gradient gave +2-14 points over vision-only. On our data the
copycat-error rule does not find the departure (only 7.7 % of its top-10 % J1 errors are standing frames [OURS]), so
weight by phase: the standing frames and the first second of the departure, e.g. duplicated x5 as short extra episodes
at dataset build -- no change to LeRobot.

### Evidence
- Keyframe-focused imitation: top 10 % copycat error x5 -> CARLA 33.0 -> 43.4 % (BC with history) [SIM] --
  [ar5iv 2106.06452](https://ar5iv.labs.arxiv.org/html/2106.06452)
- GAP: gradient of the proprio branch scaled by 0.3 (1 - rho) in transition phases, first 50 of 100 epochs; real
  gains of +2 to +6 of 20 trials over vision-only [REAL-other] -- [arXiv 2602.12032 HTML](https://arxiv.org/html/2602.12032)

## 8. Summary table: what each fix bought elsewhere, and what it costs us

### Takeaway
The cheap fixes (no state input, augmentation, phase weighting, relative actions, a target mark) cost nothing or ~3 ms
on the Orin; the expensive ones (encoder swap, generative head) have the weakest case for our failure.

| fix | best measured effect elsewhere | label | our code | Orin cost |
|---|---|---|---|---|
| no state input | ACT 0.084 -> 0.517 horizontal, 0 -> 0.933 height | REAL-other | launcher patch (2 lines) + export | none |
| state noise / masking | 0 -> 0.633 (noise); masking > concatenation, mixed vs vision-only | REAL-other / SIM | launcher | none |
| GAP gradient scaling | +2..+6 of 20 real trials over vision-only | REAL-other | ~100 lines (phase estimator) | none |
| shift / crop augmentation | real Can 26.7 -> 73.3 % | REAL-other | a flag | none |
| colour augmentation | (sim-to-real colour gap; no isolated number found) | -- | a flag | none |
| fine-tuned DINOv2 / v3 | +up to 10 points; frozen 0.00 vs 0.90 | SIM / REAL-other | backbone swap, ACT code | unknown on NX; ViT-S 224 ~4.6 GFLOPs [EST] |
| target mark / mask on the image | ACT 0.20 -> 0.40; masks +32 points | REAL-other / SIM | projection in build + bridge | ~0 ms (projection) or 3 ms (detector) |
| 2D goal | 55.0 -> 68.7 % | SIM | env-state token | none |
| relative actions | rel EEF 0.584 vs abs 0 (OOD); UMI 20/20 vs 5/20 | REAL-other | processor + stats + export | none |
| phase upweighting | CARLA 33.0 -> 43.4 % | SIM | dataset build | none |
| diffusion / flow head | no multimodality gain; ACT more reliable on cheap arms | SIM / REAL-LC | new policy | tens to hundreds of ms [2nd] |
