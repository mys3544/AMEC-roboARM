# Fastest export and inference paths for our vision models (state 2026-10-27)

Scope: YOLO26 / YOLOE-26 (Ultralytics 8.4.x), ViTs (Depth Anything V2 / DA3, DINOv2, RF-DETR), SAM 2 / SAM 3 /
EfficientTAM / NanoSAM-style models, DLA, pre/post-processing, and newer alternatives. Hardware: 2x Jetson Orin NX
16 GB Super (sm_87, JetPack 6.2, CUDA 12.6, TensorRT 10.7, ORT 1.20), laptop RTX 2070 Super 8 GB (Turing sm_75),
hekate RTX 5090 32 GB (Blackwell sm_120), Dionysos RTX A4500 20 GB (Ampere sm_86). Our own numbers are from
[PLAN.md M5](C:\Users\Mys\Desktop\RoboARM 2026\PLAN.md) (table first filled 2026-10-12) and
[train/robot_bench.sh](C:\Users\Mys\Desktop\RoboARM 2026\train\robot_bench.sh).

## Q0. Toolchain state: which TensorRT / JetPack / ModelOpt to target (cross-cutting)

### Takeaway
Stay on JetPack 6.2 + TensorRT 10.7 on the robots for now: it is the documented last TensorRT with DLA on Jetson, and
JetPack 7.2 (TensorRT 10.16.2) shipped without working DLA on Orin. TensorRT 11.x (11.4.0 latest, Oct 2026) is
desktop-only, is strongly typed by default and removes `--fp16`/`--int8` implicit precision and the INT8 calibrator,
so desktop INT8/FP8 must go through ModelOpt Q/DQ ONNX.

### Cited Findings
- JetPack 7.2 released 2026-06-02: Jetson Linux 39.2, kernel 6.8, Ubuntu 24.04, CUDA 13.2.1, cuDNN 9.20.0, TensorRT
  10.16.2, VPI 4.1.3, DeepStream 9.1; supports "Jetson Orin Family" (Orin NX not named explicitly) — [NVIDIA JetPack 7.2 notes](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2.md)
- TensorRT 10.16.1 added Orin iGPU support through the ARM SBSA build "bundled in an upcoming JetPack 7.x release";
  "Orin DLA is not supported at this time, but support is planned for a future release" — [TensorRT 10.16.1 release notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-10/10.16.1.html)
- NVIDIA's 10.x -> 11.x Jetson migration guide: TensorRT 11.4.0 is not available for JetPack 6.x or 7.x ("stay on a
  10.x release that your JetPack version supports"); "use TensorRT 10.7 if DLA is required"; 11.x removed the
  IInt8Calibrator classes (implicit INT8 via `kINT8`/`--int8`) and "all precision-enabling builder flags" (kFP16,
  kBF16, kFP8 ...); replacements are strongly typed networks, ModelOpt AutoCast for mixed precision and ModelOpt Q/DQ
  for INT8; TensorRT 11.x requires CUDA 13.2 Update 2+ — [TensorRT 10.x->11.x Jetson migration](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html)
- TensorRT 11.4.0 (page updated 2026-10-05): "NVIDIA JetPack is not supported"; DLA only on Windows-on-ARM (beta, no
  GPU fallback, batch 1); `--stronglyTyped` has no effect since 11.0 (strong typing is the default) — [TensorRT 11.4.0 release notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.4.0.html)
- TensorRT 11.4.0 known issues relevant to us: strongly typed INT8+FP16 YOLOX-Tiny (640) "can produce numerically
  incorrect output on RTX 5080/5090" (passed in 11.3.0); some INT8 per-channel and strongly typed FP16 models on
  Blackwell may hit a CUDA launch failure; `--vc` version-compatible engines can segfault for FP16 ResNet-50/ViT on
  CUDA 13.2; engines built with 10.1-10.4 on Windows cannot be deserialized by 11.x — [TensorRT 11.4.0 release notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.4.0.html)
- TensorRT 11.4.0 supports compute capability SM 7.5 and higher (Turing is the minimum, not marked deprecated), Ampere
  8.0/8.6, Ada 8.9, Blackwell 10.x/12.0; Windows 10/11 x64; separate CUDA 12.9 and CUDA 13.4 builds, `tensorrt-cu12`
  pip wheels for CUDA 12.x — [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
- DLA on JetPack 7.2 (forum, Orin NX 16GB and AGX Orin, 2026-06-18): `trtexec --useDLACore=0 --allowGPUFallback`
  fails "DLACore 0 not available" with TensorRT 10.16.2.10; worked on JP 6.2 / TRT 10.3. NVIDIA staff: cause is the
  SBSA unified driver (06-22); a TRT 11.3.0 GA + JP 7.2.1 + CUDA 13.4.1 path with DLA was announced 09-10, but a user
  still had DLA compile failures on Orin NX with 7.2.1 (09-30) and docs still say 10.7 was the last DLA release; on
  10-08 staff said DLA "would be supported in JetPack 7.2.2, expected by the end of October" — [NVIDIA forum: DLA unavailable on JP 7.2](https://forums.developer.nvidia.com/t/dla-runtime-unavailable-on-jetpack-7-2-l4t-r39-2-cannot-create-dla-engine/373788)
- Seeed's JetPack 7.2 deep dive (2026-06-11) measured llama.cpp on AGX Orin: prompt 18.2 -> 25.8 tok/s, generation
  4.3 -> 5.5 tok/s, GPU clock 930 MHz -> 1.36 GHz, but with different hardware for the two runs and a new
  high-performance power mode; recommends a full reflash for 6.x -> 7.x and re-validating TensorRT engines and CUDA
  extensions — [Seeed JetPack 7.2 deep dive](https://wiki.seeedstudio.com/jetpack72_deep_dive_bk/)
- JetPack 7.2 also adds MAXN_SUPER for AGX Orin 32GB (200 -> 241 TOPS) — [NVIDIA JetPack 7.2 notes](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2.md)
- Ultralytics: on TensorRT 11+ INT8 and FP16 export go through NVIDIA ModelOpt; TensorRT 7-10 keep the legacy
  calibrator (since v8.4.63) — [Ultralytics TensorRT integration](https://docs.ultralytics.com/integrations/tensorrt/); [Ultralytics v8.4.63 release](https://community.ultralytics.com/t/new-release-ultralytics-v8-4-63/2051)
- ModelOpt ONNX PTQ example is built around the `nvcr.io/nvidia/tensorrt:26.02-py3` container (CUDA 13), ONNX
  Runtime 1.30.x; DLA Q/DQ route validated with TensorRT 10.7, direct DLA deployment with a pre-release 11.4.1.22
  build, "TensorRT 10.7 through 11.3 is not supported for this workflow" — [ModelOpt onnx_ptq example](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq)

### Inferences
- A JetPack 7.2 upgrade on the robots today buys newer TensorRT (10.16: better MHA/attention fusions from 10.14 on,
  see Q2) and possibly higher GPU clocks, but loses DLA and forces a full reflash plus a rebuild of every engine,
  plugin and CUDA-built wheel (the M3's ROS 2 stack, Isaac/ACT services). Not worth it before JetPack 7.2.2 has
  confirmed Orin NX DLA, unless a measured ViT speedup on 10.16 justifies it; test on one robot first.
- On the desktops, TensorRT 10.16 (still supports `--fp16`/`--int8`) is the least-friction path for comparing with
  the Orin's 10.7 numbers; TensorRT 11.4 is the forward path but needs every ONNX to carry its precision (ModelOpt
  AutoCast FP16, or Q/DQ for INT8/FP8).
- Engines are per GPU + TensorRT version: build on each of the four machines, never copy (also stated by Ultralytics,
  see Q1).

### Gaps
- No official NVIDIA statement found yet (as of 2026-10-27) that JetPack 7.2.2 has shipped or that Orin NX DLA works
  under it; the 10-08 forum post is the latest signal.
- No measured vision-model (CNN/ViT) speed comparison JetPack 6.2/TRT 10.7 vs JetPack 7.2/TRT 10.16 on an Orin NX was
  found; only Seeed's LLM numbers on AGX Orin.
- Whether a standalone TensorRT newer than 10.7 can be installed on JetPack 6.2 (CUDA 12.6) was not verified.

## Q1. Ultralytics YOLO26 / YOLOE-26 export: flags, Orin numbers, and why INT8 gives little or fails

### Takeaway
For our 3-ms YOLO26n-seg, FP16 TensorRT at 480x640 with the NMS-free head (`nms=False`) is the right default; the
published Orin NX INT8 gain for YOLO26n detect is ~0.6 ms at 640 with ~3 points of mAP50-95 lost. If INT8 is tried
again, do it as explicit Q/DQ (ModelOpt ONNX PTQ with the mask head excluded and `--autotune` for Q/DQ placement),
not the legacy implicit calibrator.

### Cited Findings
- Ultralytics export args (current docs): `quantize=16|8|32` replaces deprecated `half`/`int8`; `nms=None` exports raw
  predictions for external NMS, `nms=True` embeds NMS, `nms=False` selects the NMS-free head (output commonly
  `(batch, 300, 6)` with `max_det=300`); `simplify=True` default (onnxslim); `opset=None` = latest supported;
  `dynamic=False` default; `workspace=None` auto; `device=dla:0|dla:1` on Jetson; `data`/`split`/`fraction` for INT8
  calibration; the page notes "TensorRT 11 does not support DLA" and that its QAT benchmark figures were measured with
  TensorRT 10.16 at imgsz=640 — [Ultralytics export docs](https://docs.ultralytics.com/modes/export/)
- QAT: a checkpoint trained with `quantize=8` carries its own INT8 ranges, exported as Q/DQ nodes for `onnx` and
  `engine` with no calibration data; needs `compile=False` and ModelOpt to load — [Ultralytics export docs](https://docs.ultralytics.com/modes/export/)
- INT8 PTQ details: `MINMAX_CALIBRATION` default for GPU exports, `ENTROPY_CALIBRATION_2` for DLA; recommend >=500
  representative calibration images; small calibration batches can mis-scale; calibrate on the target device;
  Sigmoid layers kept at higher precision but confidences shift, so pick the threshold from the INT8 model's own F1
  curve — [Ultralytics TensorRT integration](https://docs.ultralytics.com/integrations/tensorrt/)
- Published Orin NX 16GB numbers (Ultralytics 8.4.33, 640 px, inference only, appear to be YOLO26n detect):
  TensorRT FP32 7.01 ms (mAP50-95 0.4770), FP16 4.13 ms (0.4789), INT8 3.49 ms (0.4489); Orin Nano Super FP16 4.57 /
  INT8 3.80 ms. Same page: "TensorRT 10.3.0 on JetPack 6 cannot build INT8 engines with nms=False or nms=True.
  Upgrading to 10.7.0 or later resolves this"; DLA via `model.export(format="engine", device="dla:0", quantize=16)`,
  DLA FP16/INT8 only, unsupported layers fall back to GPU and add latency; recommends `nvpmodel -m 0` and
  `jetson_clocks` — [Ultralytics NVIDIA Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- Previously collected (not re-verified this session): YOLO26s detect on Orin NX 6.4 ms FP16 / 4.8 ms INT8 — [PLAN.md M5 table](C:\Users\Mys\Desktop\RoboARM 2026\PLAN.md)
- Older Ultralytics Orin NX data (TensorRT 10.0.1, ultralytics 8.2.16, JetPack 6.0, batch 8, an older model):
  FP32 6.11 ms / FP16 3.18 / INT8 2.30 (mAP50-95 0.37/0.37/0.32) — [Ultralytics TensorRT integration](https://docs.ultralytics.com/integrations/tensorrt/)
- The DeepStream guide's Orin NX table (YOLO11, 640): FP32 8.64 / FP16 5.27 / INT8 4.54 ms, and it says its INT8
  section "does not work with TensorRT 10.x" (tested on TRT 8.x) — [Ultralytics DeepStream on Jetson](https://docs.ultralytics.com/guides/deepstream-nvidia-jetson)
- Relevant 8.4.x changes (all included in our 8.4.175): v8.4.63 TensorRT 11 support via ModelOpt; v8.4.72 fixed an
  INT8 engine-export crash on some RTX systems (calibration moved to CUDA with CPU fallback); v8.4.124 restored
  `dynamic=True` + `nms=True` runtime sizes; v8.4.128 scoped INT8 Sigmoid precision exclusions to detection heads for
  TRT 7-10 and removed it for TRT 11; v8.4.151 QAT TensorRT exports run unquantized layers in FP16 instead of FP32
  (TRT 10 and 11); v8.4.156 "TensorRT engines should export and run more efficiently while retaining
  accuracy-sensitive operations in higher precision" — [v8.4.63](https://community.ultralytics.com/t/new-release-ultralytics-v8-4-63/2051); [v8.4.72](https://community.ultralytics.com/t/new-release-ultralytics-v8-4-72/2078); [v8.4.124](https://newreleases.io/project/pypi/ultralytics/release/8.4.124); [v8.4.128](https://community.ultralytics.com/t/new-release-ultralytics-v8-4-128/2185); [v8.4.151](https://newreleases.io/project/pypi/ultralytics/release/8.4.151); [v8.4.156](https://newreleases.io/project/pypi/ultralytics/release/8.4.156)
- v8.4.100 (release dated 2026-07-18 on Zenodo): YOLO26 is the recommended model; YOLO26 segmentation is rejected for
  Hailo "because of its attention-based architecture"; downloaded engines must match GPU family / TensorRT / CUDA,
  export on the target device — [Ultralytics v8.4.100](https://community.ultralytics.com/t/new-release-ultralytics-v8-4-100/2129); [Zenodo v8.4.100](https://zenodo.org/records/14285281/latest)
- YOLOE: set classes (text) or `refer_image` + `visual_prompts`, then export; open-world modules are re-parameterized
  into a standard YOLO head, RepRTA folded away at export; exported models are static (classes baked in, a loaded
  export rejects `set_classes()`); YOLOE-26 needs ultralytics >= 8.4.0; the page gives no TensorRT or Jetson speeds
  — [Ultralytics YOLOE docs](https://docs.ultralytics.com/models/yoloe/)
- Anecdote: a YOLOE segmentation engine on Orin NX took ~1 s per 1280x720 frame end to end (no root cause) — [Ultralytics community: YOLOE slow on Jetson](https://community.ultralytics.com/t/yoloe-inference-very-slow-on-jetson-with-tensorrt/1443)
- Our numbers (TRT 10.7, trtexec GPU compute, CUDA graphs, clocks not locked): YOLO26n-seg 480x640 FP16 3.00 ms;
  QAT Q/DQ INT8 3.18 ms with 100 of 301 layers being Reformat; plain INT8 PTQ build fails (no INT8 kernel for
  `model.23/proto/cv3` even with layerPrecisions fp16); YOLO26s-seg 5.37; YOLOE-26s-seg 3 prompts 5.36 (vs ~41 ms
  PyTorch) — [PLAN.md M5](C:\Users\Mys\Desktop\RoboARM 2026\PLAN.md); [train/robot_bench.sh](C:\Users\Mys\Desktop\RoboARM 2026\train\robot_bench.sh)
- ModelOpt ONNX PTQ CLI: `python -m modelopt.onnx.quantization --onnx_path ... --quantize_mode int8|fp8|int4
  --calibration_data_path x.npy --calibration_method max|entropy --output_path ...`, plus `--calibrate_per_node`,
  `--trt_plugins`, `--autotune quick|default|extensive` ("Q/DQ placement optimization") and `--target_dla` — [ModelOpt onnx_ptq example](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq)
- On an A5000, "hard" INT8 calibration forced 94.5 % / 96.5 % of ops to FP32 for Tiny-ViT-5M / ViT-Tiny and was slower
  than "weak" INT8; ModelOpt gave the best transformer throughput — [arXiv 2609.13166 (NPU hardware evaluation)](https://arxiv.org/pdf/2609.13166)

### Inferences
- Why INT8 gains little at our size: at ~3 ms the engine is dominated by small, memory- and launch-bound layers; the
  convolutions INT8 speeds up are a minority, and every INT8 <-> FP16 boundary costs a Reformat (our 100/301);
  Sigmoid/SiLU-heavy heads and YOLO26's attention block stay in higher precision (Ultralytics keeps Sigmoid out of
  INT8 on TRT 7-10). The published YOLO26n detect gain (4.13 -> 3.49 ms, -16 %, at 640) is the ceiling for a
  best-case placement and costs ~3 mAP points; for seg, the proto/mask branch adds float work INT8 does not touch.
- Why plain PTQ fails and QAT builds: the legacy implicit calibrator lets TensorRT pick INT8 for every eligible
  layer, and 10.7 has no INT8 tactic for that proto conv shape even when pinned; explicit Q/DQ (QAT, or ModelOpt PTQ)
  only quantizes where Q/DQ nodes sit, so excluding the proto branch from quantization should build. ModelOpt's
  `--autotune` is the tool aimed at the Reformat problem.
- Since the network is 3.2 ms of an 8.3 ms end-to-end pick look (Q5), INT8 is the least valuable optimisation left;
  only worth one more attempt with ModelOpt autotune on TRT 10.7, and it should be accepted only if it beats FP16 by
  > 0.3 ms with mask mAP50 within 0.01.
- `nms=False` (NMS-free head) is the natural YOLO26 export; the Jetson guide's TRT 10.3 INT8 limitation is fixed in
  our 10.7.

### Gaps
- No published Orin numbers for YOLO26-seg (only detect) and none for YOLOE-26 engines.
- No Ultralytics issue found that reports our exact `proto/cv3` INT8 kernel failure; the cause above is inferred.
- The exact ModelOpt flags to exclude nodes (`--nodes_to_exclude`, `--op_types_to_quantize`, `--high_precision_dtype`)
  were not on the fetched example page; check `python -m modelopt.onnx.quantization --help` for the installed version.
- The Ultralytics QAT benchmark figures (TRT 10.16) mentioned on the export page were not extracted.

## Q2. ViTs (Depth Anything V2 / DA3, DINOv2, RF-DETR): attention fusion, FP16 vs INT8/FP8, fastest implementations, pitfalls

### Takeaway
FP16 TensorRT with static shapes and opset >= 17 (native LayerNormalization) is the fastest reliable ViT path on the
Orin; INT8 on ViT-S-class models gives ~0-5 % on Orin and has produced regressions and build failures. Speed comes
from resolution (multiples of 14) and newer TensorRT attention fusions, not from quantization. FP8 is a 5090-only
option (~1.4x on a CLIP image encoder on Ada).

### Cited Findings
- Orin Nano ViT-S + DPT (TRT 10.3, JP 6.2, ModelOpt 0.31-0.41, Jan-Feb 2026): FP16 ~41.8 ms, native `--fp16 --int8`
  ~39.7-40.4 ms, ModelOpt INT8 with `--stronglyTyped` ~114 ms (0.31/0.35) and build failure on 0.40/0.41 with
  `weightsPtr.h` "Assertion type() == expectedDataType<T>() failed"; ViT-S alone FP16 30.75 vs ModelOpt INT8 30.34
  ms. NVIDIA could not reproduce the 2.7x slowdown (FP16 36.7 vs INT8 35.4 ms, ~3 %), said the build works with an
  internal TensorRT, and concluded "the INT8 kernel gain is limited" for this model — [NVIDIA forum: ViT-S DPT INT8 on Orin Nano](https://forums.developer.nvidia.com/t/tensorrt-model-optimizer-int8-quantization-causes-2-7x-performance-regression-on-jetson-orin-nano-4gb-vit-s-dpt-architecture/357835)
- A DEIMv2 (DINOv3-based transformer detector) user on Orin NX reported FP16/BF16 580 QPS vs INT8 380 QPS (seen in a
  search summary, thread not read in full) — [NVIDIA forum: worse performance after quantization](https://forums.developer.nvidia.com/t/worse-performance-after-quantization-on-tensorrt/355549)
- TensorRT 10.14.1 added an `IAttention` API (fused attention kernel for transformers), automatic padding of MHA heads
  with misaligned head sizes so they still fuse, and fixed a ViT MHA regression on GB200 — [TensorRT 10.14.1 release notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-10/10.14.1.html)
- TensorRT 10.7 notes cover FP16 MHA with FP8 GEMMs for ViTs; FP8 MHA fusion needs head sizes that are multiples of
  16 — [TensorRT 10.7 release notes (PDF)](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1070/pdf/TensorRT-Release-Notes.pdf)
- LayerNorm FP16: TensorRT warns "Running layernorm after self-attention in FP16 may cause overflow"; remedies are an
  ONNX export at opset >= 17 (native LayerNormalization -> INormalizationLayer) or forcing LayerNorm to FP32 — [NVIDIA forum: LayerNorm FP16 overflows](https://forums.developer.nvidia.com/t/layernorm-fp16-overflows-trt-8-5-6-on-xavier-nx/345971); [Torch-TensorRT issue 2730](https://github.com/pytorch/TensorRT/issues/2730); [Torch-TensorRT PR 2509](https://github.com/pytorch/TensorRT/pull/2509)
- ONNX Runtime fixed ViT/BERT accuracy loss in TRT FP16 by forcing the LayerNorm `Pow` to FP32 — [ONNX Runtime commit](https://app.semanticdiff.com/gh/microsoft/onnxruntime/commit/2023836c2f97dd6fe417e0b6f2295efdd3346fb8)
- A DINOv3 user found `--layerPrecisions` keeping attention in FP32 did not stop NaNs (Myelin fusion can override
  per-layer constraints) — [NVIDIA forum: model outputting NaNs](https://forums.developer.nvidia.com/t/model-outputting-nans/360219)
- Depth Anything V2 on Jetson (arXiv 2603.01999, 2026): fine-tuned ViT-S DA V2, TensorRT BF16, 480x300, batch 4: AGX
  Orin 50.16 ms, Orin NX 98.63 ms (10.1 batches/s), Thor 19.02 ms per batch (JetPack 6.1) — [arXiv 2603.01999](https://arxiv.org/pdf/2603.01999)
- NVIDIA TAO DA V2 Large FP16 3x518x914 on AGX Orin: 3.705 FPS at batch 32 — [NGC TAO nvDepthAnythingV2](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/nvdepthanythingv2/deployable_relative_depthanythingv2_large_v1.0)
- spacewalk01/depth-anything-tensorrt (V1/V2; TRT 10.0 badge; last dated news 2024-08-20): 518x518, input sides
  divisible by 14; RTX 4090 FP16 incl. pre/post: S 3 ms, B 6 ms, L 12 ms; no Jetson numbers — [spacewalk01/depth-anything-tensorrt](https://github.com/spacewalk01/depth-anything-tensorrt)
- DA3 on Orin NX 16GB (GerdsenAI ROS 2 wrapper, 2026-02-05, JP 6.2, TRT 10.3): 153 ms PyTorch -> ~23 ms (43+ FPS)
  TensorRT, split as ~15 ms inference + ~8 ms shared-memory IPC; model size, resolution and precision not stated;
  TensorRT Python bindings broken in jetson-containers (issue #714) so it runs TRT on the host; first engine build
  15-20 min — [Open Robotics Discourse: GerdsenAI DA3 wrapper](https://discourse.openrobotics.org/t/release-gerdsenais-depth-anything-3-ros2-wrapper-with-real-time-tensorrt-for-jetson/52364); [GerdsenAI repo](https://github.com/GerdsenAI/GerdsenAI-Depth-Anything-3-ROS2-Wrapper)
- Community DA3 exports: Symphoenix/sylc_TRT (DA3-small ONNX at fixed grids plus a TensorRT engine cache),
  zerochocobo/DepthAnything3_ONNX (Small/Base depth branch at fixed resolution), litert-community DA3-Small FP16
  (896x504, LiteRT), Heliosoph DA3-Base / metric-large ONNX with fp16 siblings — [sylc_TRT](https://huggingface.co/Symphoenix/sylc_TRT); [zerochocobo DA3 ONNX](https://huggingface.co/zerochocobo/DepthAnything3_ONNX); [LiteRT DA3-Small](https://huggingface.co/litert-community/Depth-Anything-3-Small); [Heliosoph DA3-Base ONNX](https://huggingface.co/Heliosoph/da3-base-onnx)
- DINOv2 on Orin: a DINOv2-based FP16 pipeline at 640 px on AGX Orin ran 13.34 ms vs 25.60 ms FP32 end to end (1.92x)
  — [PixOOD real-time, arXiv 2607.28483](https://arxiv.org/pdf/2607.28483). Embedl publishes a quantized DINOv3 ViT-B/16
  TensorRT model (not read) — [embedl/dinov3-quantized-tensorrt](https://huggingface.co/embedl/dinov3-quantized-tensorrt)
- RF-DETR: Embedl RF-DETR Base on Orin Nano Super TRT FP16 ~14.83 ms (GPU compute, `--noDataTransfers`, CUDA graph,
  spin wait, clocks locked) — [embedl/rf-detr-base](https://huggingface.co/embedl/rf-detr-base). Roboflow's DeepStream
  guide on Orin NX: RF-DETR Nano ONNX with static batch 1 -> `trtexec` FP16, ~2 min build (JP 6.2, DS 7.1, TRT 10.3)
  — [Roboflow: RF-DETR in DeepStream](https://blog.roboflow.com/rf-detr-nvidia-deepstream/). README: Seg-S 4.4 ms at 384,
  Seg-2XL 21.8 ms at 768 on a T4 (TRT FP16, batch 1) — [RF-DETR README](https://cdn.jsdelivr.net/gh/roboflow/rf-detr@develop/README.md)
- Previously collected: DEIMv2 needs TensorRT 10.6+ for correct FP16 — [PLAN.md M5](C:\Users\Mys\Desktop\RoboARM 2026\PLAN.md)
- FP8 on desktop: ModelOpt -> ONNX -> TensorRT FP8 gave 1.39x (CLIP image encoder) over FP16 on an RTX 6000 Ada (per
  search summary; page not read in full) — [NVIDIA blog: FP8 checkpoints to TensorRT](https://developer.nvidia.com/blog/model-quantization-turn-fp8-checkpoints-into-high-performance-inference-engines-with-nvidia-tensorrt/).
  ModelOpt ONNX quantization supports FP8, INT8 and INT4 (not NVFP4) and claims "better support for vision
  transformers" — [ModelOpt ONNX quantization guide (PR preview)](https://nvidia.github.io/Model-Optimizer/pr-preview/pr-1993/guides/_onnx_quantization.html).
  FP4 on the 5090 has been shown for FLUX (ModelOpt + distillation QAT) — [NVIDIA blog: FP4 on RTX 50](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus)
- Our numbers: DA V2 Small FP16 ~23.6 ms on the Orin (~26 ms engine on the X3 per PLAN); DA3 S/B/L 46/82/208 ms for 3
  views after two export fixes (attention scale, camera-encoder qkv split); RF-DETR-Seg-Nano 8.0 ms at 312 — [PLAN.md M5](C:\Users\Mys\Desktop\RoboARM 2026\PLAN.md)

### Inferences
- Our DA V2 Small ~23.6 ms on Orin NX is in line with the 2603.01999 Orin NX figure (98.6 ms per batch of 4 at
  480x300 BF16, ~25 ms/image); there is no evidence of a much faster published DA V2-S implementation on Orin NX.
  Lower resolution is the main lever: ViT cost scales at least linearly with token count ((H/14)*(W/14)), so
  364x490 (910 tokens) is ~0.58x the tokens of 476x644 (1564 tokens).
- GerdsenAI's ~15 ms DA3 inference on Orin NX vs our 46 ms for 3 views of DA3-S (~15 ms/view) suggests they run
  DA3-Small single-view at a similar resolution; our numbers look close to the state of the art.
- TensorRT 10.14+ attention improvements (IAttention, head padding) are the main argument to try TRT 10.16 on Orin
  (JetPack 7.2) for the ViTs; worth a timed A/B on one robot before any fleet upgrade.
- INT8 for DA V2 / DINOv2 / RF-DETR on Orin: not worth the effort (0-5 % at best, failures common). FP8 / NVFP4 only
  exist on the 5090 (Ampere A4500 and Turing 2070 Super have no FP8 tensor cores — architecture knowledge, not
  verified in a source here).
- For FP16 correctness: export at opset 17+ (Ultralytics and torch.onnx both allow it), compare engine output with
  PyTorch FP32 on our frames, and if outputs drift, try ModelOpt AutoCast (keeps sensitive ops FP32) rather than
  `--layerPrecisions`, which Myelin may ignore.

### Gaps
- No published DINOv2 ViT-S/14 TensorRT latency on Orin NX found; we must measure.
- No ModelOpt INT8/FP8 accuracy result for DINOv2 or Depth Anything found.
- No source confirms which TensorRT version first fuses DINOv2-style attention (with its custom scale) into a single
  MHA kernel on sm_87; check `--profilingVerbosity=detailed --dumpLayerInfo` for `_gemm_mha_v2` / Myelin MHA nodes.
- No TensorRT-RTX or FP8 vision transformer benchmark on an RTX 5090 found.

## Q3. SAM 2 / SAM 3 / EfficientTAM / NanoSAM-style models

### Takeaway
On an Orin NX, only NanoSAM-class (CNN encoder) promptable segmenters are real-time (~8 ms on an AGX Orin); SAM 3 is
~0.5-1 s per image even quantized, so it stays an offline labeller on the 5090. EfficientSAM3 (distilled SAM 3,
2025) is the newest edge candidate but has no Jetson TensorRT numbers yet.

### Cited Findings
- Embedl SAM 3 (quantized): mixed INT8/FP16 Q/DQ ONNX (~3.1 GB external weights), built with `trtexec --fp16 --int8
  --builderOptimizationLevel=5` and a 4 GB pool, input 924 px "to enable TensorRT layer fusions"; AGX Orin 763 ms
  FP16 -> 462 ms (1.65x); L4 137 ms (torch.compile FP16) -> 104 ms; SA-Co/Gold cgF1 55.56 -> 53.77; validated only on
  TensorRT 10.1 and 10.3, "newer versions produce incorrect segmentation masks"; gated SAM licence plus Embedl
  community licence — [embedl/sam3](https://huggingface.co/embedl/sam3)
- Previously collected: SAM 3 ~0.95 s/image on an Orin NX in TensorRT — [PLAN.md M5](C:\Users\Mys\Desktop\RoboARM 2026\PLAN.md)
- EfficientSAM3 (arXiv 2511.15833, Nov 2025, Univ. of Bristol): progressive hierarchical distillation of SAM 3 into
  RepViT / TinyViT / EfficientViT students (nine variants), including a compact Perceiver memory for video — [EfficientSAM3 paper](https://arxiv.org/pdf/2511.15833v1); [EfficientSAM3 weights](https://huggingface.co/Simon7108528/EfficientSAM3). Repo last push reported as 2026-04-22 by an aggregator — [gittrend](https://gittrend.io/repo/SimonZeng7108/efficientsam3)
- NanoSAM (NVIDIA-AI-IOT): ResNet18 encoder distilled from MobileSAM; AGX Orin encoder 4.2 ms, full pipeline 8.1 ms;
  Orin Nano full 27 ms; MobileSAM 35 / 39 ms on AGX Orin; MobileSAM encoder must run FP32 (FP16 gave wrong results),
  decoders FP32; mIoU 0.706 vs MobileSAM 0.728; export: `trtexec --onnx=resnet18_image_encoder.onnx --fp16` and a
  decoder engine with `--minShapes/--optShapes/--maxShapes` on `point_coords`/`point_labels` — [NVIDIA-AI-IOT/nanosam](https://github.com/NVIDIA-AI-IOT/nanosam)
- EdgeTAM (CVPR 2025): 16 FPS on iPhone 15 Pro Max; memory attention, not only the encoder, is a latency bottleneck
  (2D spatial perceiver fix); DAVIS 2017 87.7 J&F — [HF Transformers EdgeTAM video docs](https://huggingface.co/docs/transformers/model_doc/edgetam_video).
  EfficientTAM ~10 FPS on iPhone 15 Pro Max (aggregator review) — [Liner review of EfficientTAM](https://liner.com/ko/review/efficient-track-anything)
- DART is a separate project that speeds SAM 3 multi-class inference with backbone sharing, detection-only inference
  and TensorRT (search summary only) — [On Efficient Variants of SAM survey](https://arxiv.org/pdf/2410.04960)

### Inferences
- Realistic on the Orin NX: NanoSAM-style (box-prompted refinement of YOLO boxes) at ~10-15 ms (Orin NX is between
  AGX Orin and Orin Nano); EdgeTAM / EfficientTAM / EfficientSAM3 small variants plausibly tens of ms, but unmeasured.
  SAM 3 is offline only.
- For label refinement on the 5090, run SAM 2 / SAM 3 in PyTorch (torch.compile, BF16/FP16); TensorRT for SAM 3 is
  fragile (Embedl's warning about mask errors on TRT > 10.3) and gives only ~1.3x on an L4, not worth it offline.
- Our kit detector already outputs masks, so a separate promptable segmenter on the robot is needed only for
  open-set objects (teach); there NanoSAM on YOLOE boxes is the cheap path.

### Gaps
- No SAM 2 / SAM 2.1 TensorRT Jetson numbers and no EdgeTAM/EfficientTAM/EfficientSAM3 Jetson TensorRT numbers found.
- DART's actual speedups were not read.

## Q4. DLA for the CNN detectors

### Takeaway
DLA is unlikely to pay for us: our detector is 3 ms on the GPU, YOLO26 contains attention/elementwise layers that
will fall back to the GPU, fallbacks add transfers and latency, and DLA only exists on TensorRT <= 10.7 on JetPack 6
today. It can still be worth one verbose build to see how much of YOLO26n stays on DLA, for running the detector
concurrently with Depth Anything on the GPU (throughput, not latency).

### Cited Findings
- trtexec DLA usage: `--useDLACore=0 --fp16 --allowGPUFallback` — [TensorRT: Working with DLA](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html)
- Ultralytics: DLA FP16 or INT8 only; unsupported layers fall back to GPU, adding latency; DLA mainly improves
  throughput and energy, not latency; `device=dla:0` export — [Ultralytics NVIDIA Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- Orin NX YOLOv7-Tiny: GPU-only 238 FPS beat GPU + 2 DLAs 225 FPS because of fallback layers and transfers; NVIDIA
  advised keeping whole models on GPU or DLA — [NVIDIA forum: how to use DLA correctly](https://forums.developer.nvidia.com/t/how-to-use-dla-correctly/322765)
- Add outputs falling back to GPU; DLA needs FP16/INT8 data formats — [NVIDIA forum: Add operations fell back to GPU](https://forums.developer.nvidia.com/t/output-tensor-of-add-operations-are-fellback-to-gpu/319208).
  NVIDIA advised static batch 1 for DLA profiling of YOLOv8 — [NVIDIA forum: DLA performance](https://forums.developer.nvidia.com/t/dla-performance/302939).
  A model that builds GPU-only failed with an internal error on DLA + fallback on Orin NX (TRT 8.5.2) — [NVIDIA forum: trtexec DLA internal error](https://forums.developer.nvidia.com/t/trtexec-failed-to-generate-engine-internal-error-with-dla/264892)
- YOLO26 segmentation has an "attention-based architecture" (rejected for Hailo for that reason) — [Ultralytics v8.4.100](https://community.ultralytics.com/t/new-release-ultralytics-v8-4-100/2129)
- ModelOpt can export an INT8 Q/DQ ONNX for DLA (`--target_dla`), legacy route validated with TensorRT 10.7 — [ModelOpt onnx_ptq example](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq)
- DLA availability: TRT 10.7 is the documented last DLA release on Jetson; JP 7.2 / TRT 10.16 has no Orin DLA; JP
  7.2.2 promised by end of October 2026 — see Q0 sources ([migration guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html), [forum](https://forums.developer.nvidia.com/t/dla-runtime-unavailable-on-jetpack-7-2-l4t-r39-2-cannot-create-dla-engine/373788))
- Our current view: "The DLA (two cores on the Orin NX 16 GB) adds throughput for CNNs, not latency; attention layers
  cannot use it" — [PLAN.md M5](C:\Users\Mys\Desktop\RoboARM 2026\PLAN.md)

### Inferences
- A DLA YOLO26n-seg would likely run several times slower than 3 ms (DLA clocks are lower than the GPU's and each
  GPU-fallback segment adds a DLA<->GPU hop); the only win is freeing ~3 ms of GPU per frame while Depth Anything
  (~24 ms) or the ACT policy runs. At our frame rates that GPU time is not the bottleneck, so DLA is low priority.
- If tried, the right test is concurrency: DA V2 on the GPU plus the detector on DLA at the same time, measuring
  both latencies, vs both on the GPU.

### Gaps
- No source lists which YOLO26 layers are DLA-capable; a `--verbose` build on TRT 10.7 ("Layers running on DLA/GPU")
  is the only way to know.
- No published DLA latency for any YOLO26 model.

## Q5. Pre/post-processing: GPU preprocessing, mask post-processing on GPU, zero-copy, share of latency

### Takeaway
On our robot the network is already the minority: 3.2 ms of an 8.3 ms end-to-end pick look, with 3.5 ms in mask
post-processing and 0.8 ms preprocessing (Ultralytics predict). Keeping masks on the GPU at proto resolution and
cropping only inside boxes is the biggest remaining win; JPEG decode on Orin via the NVJPG engine needs the Jetson
Multimedia API, not nvJPEG.

### Cited Findings
- Our M3 end to end (train/robot_e2e.py, 92 pick looks, Ultralytics predict): FP16 engine 8.3 ms/picture (p99 9.6):
  preprocess 0.8, inference 3.2, mask postprocess 3.5; PyTorch 32.6 ms — [PLAN.md M5](C:\Users\Mys\Desktop\RoboARM 2026\PLAN.md)
- Ultralytics 8.3.217 (2025-10-17): masks moved to uint8 throughout processing and plotting; NMS warm-up to smooth the
  first post-processing call — [Zenodo Ultralytics v8.3.217](https://zenodo.org/records/14285359/latest)
- nvJPEG (CUDA 13.x docs) lists hardware-accelerated decode for Ampere, Hopper, Ada, Blackwell and Jetson Thor — not
  Orin; nvJPEG input is host memory, output GPU memory — [nvJPEG docs](https://docs.nvidia.com/cuda/archive/13.2.1/nvjpeg/index.html)
- The Jetson platform has a dedicated NVJPEG unit; a multi-instance design using it gave up to 30.02 % speedup for
  large images (paper submitted 2026-08-27) — [Accelerating Data Preprocessing, arXiv 2608.27655](https://www.alphaxiv.org/abs/2608.27655)
- On AGX Orin, the Multimedia API `decodeToFd` cannot output to `NVBUF_MEM_SURFACE_ARRAY`, forcing an extra
  NvTransform copy before CUDA — [NVIDIA forum: decodeToFd performance](https://forums.developer.nvidia.com/t/decoder-to-fd-performance/377619)
- A DeepStream segmentation engine with ~20 ms GPU latency ran at 7 FPS with 10 % GPU utilisation: pipeline-bound —
  [NVIDIA forum: slow DeepStream segmentation](https://forums.developer.nvidia.com/t/deepstream-slow-framerate-7-fps-for-tensorrt-segmentation-engine-20ms-gpu-latency/175062)
- TIER IV's trt-yoloXP moves YOLOX-seg preprocessing to the GPU for embedded platforms — [tier4/trt-yoloXP](https://github.com/tier4/trt-yoloXP).
  On a TX2, moving per-pixel argmax to the GPU cut ENet end-to-end latency — [arXiv 2202.01421](https://arxiv.org/pdf/2202.01421)
- A 2026 Orin Nano Super study defines prediction latency as CUDA-synchronized preprocessing-to-detections excluding
  file read and decode, warning that mixed timing scopes confuse comparisons — [arXiv 2609.13551](https://arxiv.org/pdf/2609.13551)

### Inferences
- Mask post-processing (3.5 ms) costs more than the network (3.2 ms); the fixes are: compute `coeffs @ protos` and
  sigmoid on the GPU at 120x160 proto resolution, crop to boxes there, and upsample only the few kept masks (we
  have <= ~10 objects); avoid `retina_masks=True` and full-frame CPU resizing. A custom TensorRT/CUDA path (or
  CuPy/torch on the engine's output buffers) should bring post below ~1 ms.
- Preprocessing (0.8 ms) can drop further by letterbox/normalize on the GPU from a pinned buffer; our cameras are
  USB UVC (C930e on the X3, 848x480), so the frame arrives on the CPU anyway (MJPEG or YUYV); NVJPG decode via the
  Multimedia API or GStreamer `nvv4l2decoder`/`nvjpegdec` is an option only if CPU MJPEG decode shows up in a profile.
- True zero-copy (NVMM/argus) only applies to CSI cameras; for UVC, one pinned host-to-device copy of a 848x480 frame
  is well under 1 ms on Orin's unified memory.
- Always report three scopes: engine GPU compute (trtexec), engine + pre/post in-process, and camera-to-result.

### Gaps
- No published breakdown of YOLO26-seg end-to-end latency on Orin found; our own measurement is the reference.
- Whether VPI 3.x/4.x or nvImageCodec give a zero-copy JPEG path into CUDA on Orin was not confirmed.

## Q6. Newer, faster model alternatives (2025-2026) for instance segmentation and monocular depth on edge GPUs

### Takeaway
No 2025-2026 model clearly beats YOLO26n-seg for our three-shape kit at 3 ms, and none beats Depth Anything V2/3
Small for relative depth at Orin-friendly cost; the candidates to watch are RF-DETR-Seg (ICLR 2026), YOLO27
(previewed, no weights), EfficientSAM3 for promptable masks, and DA3-Small for depth.

### Cited Findings
- RF-DETR-Seg: Seg-S 4.4 ms at 384 and Seg-2XL 21.8 ms (49.9 COCO AP) at 768 on a T4, TRT FP16 batch 1 — [RF-DETR README](https://cdn.jsdelivr.net/gh/roboflow/rf-detr@develop/README.md); on our kit Seg-Nano at 312 was 8.0 ms and 0.683 mask mAP50 vs YOLO26n-seg 0.791 — [PLAN.md M5](C:\Users\Mys\Desktop\RoboARM 2026\PLAN.md)
- The Ultralytics export docs link an unreleased YOLO27 preview for planned export support — [Ultralytics export docs](https://docs.ultralytics.com/modes/export/); PLAN notes YOLO27 previewed 2026-09-13 without weights — [PLAN.md M5](C:\Users\Mys\Desktop\RoboARM 2026\PLAN.md)
- EfficientSAM3 (Nov 2025) — see Q3 — [EfficientSAM3 paper](https://arxiv.org/pdf/2511.15833v1)
- DA3 on Orin NX ~23 ms with TensorRT (Feb 2026) — [GerdsenAI DA3 wrapper](https://discourse.openrobotics.org/t/release-gerdsenais-depth-anything-3-ros2-wrapper-with-real-time-tensorrt-for-jetson/52364)
- Lightweight depth: RT-MonoDepth (ICIP 2024) 253 / 364 FPS on AGX Orin at 640x192, but KITTI-trained and compared only
  with older fast methods — [RT-MonoDepth, arXiv 2308.10569](https://arxiv.org/pdf/2308.10569)
- YOLO26n-depth: 3.80 ms on our Orin NX but flattens 30 mm pieces to ~6 mm (height error 23.9 mm vs DA V2 6.5 mm) —
  [PLAN.md M5](C:\Users\Mys\Desktop\RoboARM 2026\PLAN.md)
- Third-party YOLO26 vs YOLOv8 C++ TensorRT comparison on Orin Nano exists, but its YOLO26 INT8 figures were not
  retrievable — [Hackster: YOLOv8 vs v26 on Orin Nano](https://www.hackster.io/qwe018931/pushing-limits-yolov8-vs-v26-on-jetson-orin-nano-b89267)

### Inferences
- For instance segmentation the remaining gains are data (synthetic + real) and post-processing, not a new
  architecture; re-check when YOLO27 weights appear.
- For depth, the fastest real gain would be fine-tuning a small model (YOLO26n-depth or DA V2-S at lower
  resolution) on the M3's RGB-D, as PLAN already proposes, rather than a new off-the-shelf model.

### Gaps
- No 2025-2026 edge depth model with Jetson TensorRT numbers that beats DA V2-S on relative depth quality was found
  (searches returned older RT-MonoDepth / FastDepth). Video Depth Anything, MoGe-2, UniDepthV2, Metric3D v2 and
  Depth Pro were not checked for Jetson speed in this session.
- No YOLO27 release date or weights found.

## Benchmark plan for vision models

### Ground rules (all machines)
- Lock clocks before timing on the Orins: `sudo nvpmodel -m <MAXN_SUPER id>` and `sudo jetson_clocks` (our
  2026-10-12 table had unlocked clocks); stop or idle the robots' other GPU services and note which ran.
- trtexec as in train/robot_bench.sh: `--useCudaGraph --noDataTransfers --warmUp=500 --duration=10
  --timingCacheFile=timing.cache`, plus `--profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=x.json` once
  per engine to count Reformat layers and see whether attention fused into MHA kernels. Add
  `--builderOptimizationLevel=5` as a variant (Embedl used it for SAM 3) and record build time.
- Static shapes, batch 1 (DINOv2 crops: batch 8/16 fixed). Build every engine on the machine that runs it.
- Report three scopes: trtexec GPU compute; in-process engine + pre + post (train/robot_e2e.py); camera-to-result.
- Every speed row gets an accuracy row on the same engine.

### Orin NX (JetPack 6.2, TensorRT 10.7; keep DLA available)
1. YOLO26n-seg kit detector (baseline 3.00 ms FP16, mask mAP50 0.791):
   - FP16 engines, NMS-free vs raw head:
     `yolo export model=best.pt format=onnx imgsz=480,640 opset=17 simplify=True dynamic=False nms=False` and the same
     with `nms=None`; then `trtexec --onnx=... --fp16 ...` (also `yolo export ... format=engine quantize=16` to compare
     Ultralytics' own TensorRT build, which keeps sensitive ops in higher precision since 8.4.156).
   - INT8 attempt 2 (explicit PTQ): `python -m modelopt.onnx.quantization --onnx_path yolo26n.onnx --quantize_mode int8
     --calibration_method entropy --calibration_data_path kit_calib_500.npy --autotune default --output_path
     yolo26n.int8.onnx`, excluding the `model.23/proto/*` nodes (and ideally the whole seg branch) from quantization
     (check `--help` for the exclude flag of the installed ModelOpt); build with `--fp16 --int8`. Accept only if
     < 2.7 ms and mask mAP50 >= 0.78.
   - QAT engine rebuilt with `--builderOptimizationLevel=5` to see if Reformats drop.
   - Accuracy: `yolo val model=x.engine data=kit.yaml imgsz=480,640 batch=1` -> mask mAP50 / mAP50-95, plus recall on
     the 40 mm cube and 30 mm pieces; INT8: re-pick `conf` from its own F1 curve.
   - End to end: re-run robot_e2e.py after moving mask post-processing to the GPU (proto-resolution crop, upsample only
     kept masks); target < 5 ms total.
2. YOLOE-26s-seg (5.36 ms FP16): text-prompt and visual-prompt engines (`set_classes` / `refer_image` then
   `format=engine quantize=16 imgsz=480,640`); open-set recall on teach objects; INT8 not planned.
3. Depth Anything V2 Small (~23.6 ms FP16): ONNX opset 17 at three static sizes, all multiples of 14: 308x406,
   364x490, 476x644; `trtexec --fp16`; check the build log for the LayerNorm-overflow warning; compare depth against
   PyTorch FP32 (median abs rel diff) and run the duel metrics (m3pro/tools/m3_duel.py: height error median, % named,
   floor sd). INT8 skipped (Orin evidence: 0-5 %, failures).
4. DA3-S/B (46/82 ms for 3 views): single-view vs 3-view engines at 2 sizes; FP16 only; same depth checks; compare
   with GerdsenAI's ~15 ms DA3 inference.
5. DINOv2 ViT-S/14 crop identity: FP16, static batch 8 and 16 at 224x224; check cosine similarity vs PyTorch FP32
   (> 0.999) and the teach/find score (4/4 look-alikes).
6. RF-DETR-Seg-Nano: only re-run if YOLO26 regresses; FP16, `--builderOptimizationLevel=5`.
7. DLA probe (TRT 10.7 only): `trtexec --onnx=yolo26n.onnx --useDLACore=0 --fp16 --allowGPUFallback --verbose`;
   count layers on DLA vs GPU; then run it concurrently with the DA V2 engine on the GPU (two trtexec processes) and
   record both latencies vs both-on-GPU.
8. Optional promptable masks: NanoSAM encoder `trtexec --fp16` + decoder FP32 with point/box shape ranges, on YOLOE
   boxes; EfficientSAM3 smallest variant FP16 if its ONNX exports.
9. Optional JetPack 7.2.x A/B on ONE robot (only after confirming DLA in 7.2.2 or deciding DLA is not needed): rebuild
   items 1, 3, 4, 5 on TRT 10.16 and compare; requires full reflash and re-validating the robot stack.

### RTX 5090 (hekate, Blackwell; ask before GPU-heavy runs if < 16 GB VRAM free; Isaac Sim / training must not be disturbed)
- TensorRT 10.16 (for parity with the Orin flags) and TensorRT 11.4 (strongly typed: FP16 via ModelOpt AutoCast,
  INT8/FP8 via ModelOpt Q/DQ).
- DA V2 S/B/L and DA3 S/B/L: FP16 vs FP8 (`--quantize_mode fp8 --calibration_method max`, ~100-500 frames);
  BF16 as a fallback if FP16 overflows; check depth error vs FP32 and our duel metrics.
- YOLO26n/s-seg and YOLOE: FP16 vs INT8 Q/DQ; on TRT 11.4 verify INT8 outputs numerically (known issue: strongly typed
  INT8+FP16 YOLOX-Tiny incorrect on RTX 5090 in 11.4.0; passed in 11.3.0).
- SAM 3 for labels: PyTorch torch.compile BF16 baseline; Embedl INT8 TensorRT only on TRT 10.1/10.3 (newer TRT gives
  wrong masks per Embedl); measure label mIoU vs PyTorch on 50 kit frames.

### RTX A4500 (Dionysos, Ampere sm_86, shared lab PC: do not touch others' processes)
- FP16 and INT8 only (no FP8); same YOLO/DA engines as the 5090 to give an Ampere reference; TensorRT 10.16 or 11.4
  (both support sm_86).
### RTX 2070 Super laptop (Turing sm_75, 8 GB)
- FP16 and INT8 only; TensorRT 11.4 still supports SM 7.5 (minimum). For the console's local models, compare
  onnxruntime-gpu CUDA EP vs TensorRT EP vs a trtexec engine for YOLO26n-seg and DA V2-S; keep VRAM within 8 GB next
  to the local LLM.

### Pitfalls to watch
- TensorRT 11 removes `--fp16`/`--int8` and the INT8 calibrator: old scripts and Ultralytics' legacy path change
  behaviour; Ultralytics uses ModelOpt on TRT 11.
- INT8 Reformat overhead on small models; read `--dumpLayerInfo` before trusting a speed number.
- FP16 LayerNorm overflow in ViTs: opset >= 17, verify outputs; `--layerPrecisions` can be overridden by Myelin.
- DA3 export needs our two attention fixes (scale, camera-encoder qkv split); community DA3 ONNX exports may not have
  them, so diff their outputs against PyTorch.
- Input sizes for DINOv2-family ViTs must be multiples of 14.
- DLA: TRT 10.7 only on Orin today; GPU fallback can make it slower than GPU-only.
- Engines and calibration are device- and version-specific; never ship an engine between machines.
- Timing scope: trtexec GPU compute vs end-to-end; clocks locked vs not.
