# Speed now hides outside our engines

As of 27 October 2026, almost every model we use runs fastest on the path the robots already take. A static-shape ONNX is built on the target machine into a native TensorRT engine at FP16 (BF16 on Ampere and Blackwell where accuracy needs it) and replayed through a CUDA graph. Our Orin NX numbers already match or beat the published frontier: YOLO26n-seg runs in 3.00 ms where Ultralytics publishes 4.13 ms for YOLO26n at 640 px on the same board, Depth Anything V2 Small in ~23.6 ms against a published ~25 ms per image, and ACT in 5.9 ms against 8.7 ms in LeRobot's own export PR. The remaining milliseconds are outside the engines. Mask post-processing (3.5 ms) takes longer than the network (3.2 ms). The ACT service spends 12 ms around a 5.9 ms engine. The laptop's Qwen3.5 parser probably reprocesses its whole system prompt on every command.

The newest toolchain features pay off in only a few places:
- FP8 and NVFP4 on hekate's RTX 5090, for large transformer backbones (GR00T N1.7, pi0.5, LLM prefill).
- TensorRT Edge-LLM at INT4, for any LLM that runs on a robot.
- ModelOpt explicit Q/DQ, the only INT8 route that survives TensorRT 11. TensorRT 11's strongly typed workflow has no `--fp16`, no `--int8` and no calibrators, and our export scripts should adopt it now.

The robots should stay on JetPack 6.2 with TensorRT 10.7. JetPack 7.2.x would bring TensorRT 10.16, Edge-LLM and GR00T's official Orin path. It would cost a reflash, ROS 2 Humble and a rebuild of every engine, and it has no working Orin DLA until the promised 7.2.2 ships. No published result shows a speed gain for CNNs or small ViTs. VLAs belong on the 5090: GR00T at ~25 ms with TensorRT and pi0.5 at ~50 ms compiled, both estimates. SmolVLA, at ~150-190 ms, is the only VLA candidate to run on the robot. Many cross-device figures in the notes are extrapolations (AGX Orin to Orin NX, 4090 to 5090) or come from single unreviewed sources. They are tagged as such below, and the per-machine benchmark plans at the end turn each one into a measurement.

## Our Orin engines already sit at the published frontier

The robots run TensorRT 10.7 installed on top of JetPack 6.2, whose stock TensorRT is 10.3. Every figure below was measured that way, with trtexec GPU compute time, CUDA graphs and unlocked clocks ([PLAN.md M5](../PLAN.md); [train/robot_bench.sh](../train/robot_bench.sh)).

| Model on the Orin NX 16 GB Super | Ours, TRT 10.7 FP16 [M] | Best published comparable | Reading |
|---|---|---|---|
| YOLO26n-seg, 480x640 | 3.00 ms | YOLO26n at 640 (model unlabelled; likely detect): FP16 4.13 ms, INT8 3.49 ms with mAP50-95 0.4789 -> 0.4489 ([Ultralytics](https://docs.ultralytics.com/guides/nvidia-jetson)) | Ahead (smaller input); the most INT8 could save is ~0.5 ms |
| YOLO26n-seg, INT8 QAT | 3.18 ms, 100 of 301 layers are Reformat | ModelOpt 0.47 autotune keeps INT8 only if it is at least 1.02x faster ([ModelOpt releases](https://github.com/NVIDIA/TensorRT-Model-Optimizer/releases)) | Slower than FP16 because of unfused Q/DQ |
| YOLOE-26s-seg, 3 prompts baked in | 5.36 ms (PyTorch ~41 ms) | none published | — |
| Kit detector end to end | 8.3 ms: pre 0.8, infer 3.2, mask post 3.5 | none | Post-processing takes longer than the network |
| Depth Anything V2 Small | ~23.6 ms | ~25 ms per image (BF16, 480x300, batch 4) ([arXiv 2603.01999](https://arxiv.org/pdf/2603.01999)) | At the frontier; resolution is the lever |
| DA3 S/B/L, 3 views | 46 / 82 / 208 ms | ~15 ms of inference per view, model and resolution unstated ([GerdsenAI](https://discourse.openrobotics.org/t/release-gerdsenais-depth-anything-3-ros2-wrapper-with-real-time-tensorrt-for-jetson/52364)) | ~15 ms per view, a match |
| RF-DETR-Seg-N at 312 | 8.00 ms | RF-DETR Base 14.83 ms on an Orin Nano Super ([Embedl](https://huggingface.co/embedl/rf-detr-base)) | Fallback detector only (0.683 vs 0.791 mask mAP50) |
| ACT, chunk 20 | 5.9 ms engine, 12 ms service median ([NEXT.md](../NEXT.md)) | 8.7 ms FP16 at chunk 100 on an Orin NX ([LeRobot PR #4630](https://github.com/huggingface/lerobot/pull/4630)) | At the frontier; the code around the engine takes half the request |

Read across the rows, the table says the engine is no longer where our time goes.

On the kit detector, even a perfect INT8 engine would save less than fixing post-processing. Its best case is Ultralytics' 4.13 -> 3.49 ms, a 16 % cut, while mask post-processing alone takes **3.5 ms**. That step can move onto the GPU:
- compute `coeffs @ protos` and the sigmoid at the 120x160 proto resolution,
- crop to the boxes there,
- upsample only the few masks that are kept.

On the ACT path, a request takes **12 ms median for a 5.9 ms engine**, and the picture it acts on is already **47 ms old** ([NEXT.md](../NEXT.md)). JPEG, base64 and stdlib HTTP account for half the request, and the camera dominates the delay from observation to action. All the Orin numbers were taken with clocks unlocked ([PLAN.md M5](../PLAN.md)). A 3 ms engine fired once per 33 ms frame therefore also pays a DVFS ramp-up that `jetson_clocks` would remove; this is untested and cheap to test.

The laptop shows the same pattern. The Qwen3.5 4B parser's **646 ms median** is slower than the 535 ms of the pure-attention Qwen3 4B 2507 ([console/README.md](../console/README.md)). Qwen3.5 is hybrid-recurrent, so llama-server can only reuse its prompt prefix through context checkpoints, and the default minimum step between checkpoints is **8192 tokens** ([llama-server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)). Open issues report Qwen3.5 "forcing full prompt re-processing" ([llama.cpp #24587](https://github.com/ggml-org/llama.cpp/issues/24587)). A system prompt of a few thousand tokens is therefore probably processed again on every command. That is a hypothesis to confirm from the server log, not a measurement.

## FP16 TensorRT from a typed ONNX wins on every GPU we own

### On the Orin, native engines are about 3.4x faster than the alternatives

Ultralytics' Orin NX 16 GB table is the cleanest comparison: TensorRT FP16 runs in **4.13 ms**, against TorchScript 11.60, PyTorch 13.90, ONNX Runtime 14.18, NCNN 34.31 and OpenVINO 40.19 ms ([Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson)). The other runtimes add overhead or are poorly supported:
- **ONNX Runtime's TensorRT execution provider** builds the same engines but adds graph partitioning and builds each subgraph eagerly on the slow CPU. In one Orin Nano port that stalled session setup ([Jetson-ORB-SLAM3](https://arxiv.org/pdf/2608.17874)).
- **Torch-TensorRT's** Jetson docs cover only JetPack 6.2 ([Torch-TensorRT JetPack](https://docs.pytorch.org/TensorRT/v2.12.0/getting_started/jetpack.html)).
- **The TensorRT Python API** gives different latencies from trtexec on Orin, so the end-to-end path must be timed separately ([NVIDIA forum](https://forums.developer.nvidia.com/t/performance-discrepancy-python-api-vs-trtexec-on-jetson-agx-orin-board/258856)).
- **AOTInductor, ExecuTorch and TVM** have no current Orin benchmarks at all.

LLMs are the exception. On an Orin Nano Super, TensorRT Edge-LLM decoded a 2B VLM at **60 tok/s**, against 56 for vLLM and 38 for llama.cpp ([Hokwang Choi](https://hokwangchoi.com/blog/vlm-benchmarks/)).

### On Windows, classic TensorRT stays the throughput pick

NVIDIA itself recommends classic TensorRT for "the highest possible inference throughput" and positions TensorRT for RTX for end-user PCs ([NVIDIA TensorRT for RTX](https://developer.nvidia.com/tensorrt-rtx)).

TRT-RTX's appeal is start-up time: builds take 15-30 s, kernels are specialised just in time on the device, and CUDA graphs are built in. Its limits:
- No published head-to-head shows it matching classic TensorRT's peak speed.
- Turing needs its own device-specific engine, without FP32 GEMMs or 3D convolutions.
- It runs only under the WDDM driver model, not TCC ([TRT-RTX support matrix](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/support-matrix.html); [TRT-RTX 1.6 notes](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.6.html)).
- ONNX Runtime's built-in NvTensorRTRTX EP is deprecated in favour of a plugin you build from source. Both that EP and Windows ML's NVIDIA EP need Ampere or newer, which rules out the laptop ([ORT TRT-RTX EP](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html); [Windows ML EPs](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)).

The other Windows options:
- The ORT TensorRT EP in ORT 1.24 is pinned to TensorRT 10.14 ([ORT v1.24.1](https://github.com/microsoft/onnxruntime/releases/tag/v1.24.1)).
- DirectML is only maintained, with no new feature work ([Microsoft Learn](https://learn.microsoft.com/en-us/windows/ai/directml/dml)).
- `torch.compile` on Windows depends on the community `triton-windows` build, pinned to each PyTorch minor version ([triton-windows](https://github.com/triton-lang/triton-windows)).
- Torch-TensorRT 2.14 ships Windows wheels built on TensorRT 11.1, for models that resist ONNX export ([Torch-TensorRT releases](https://github.com/NVIDIA/Torch-TensorRT/releases)).

For VLAs the gap between eager PyTorch and TensorRT is large. GR00T N1.6 on an RTX 5090 drops from **58 ms eager to 31 ms** with only the action head in TensorRT ([GR00T TensorRT docs mirror](https://mintlify.wiki/NVIDIA/Isaac-GR00T/deployment/tensorrt)).

### Precision menu per machine

| Machine | GPU (architecture) | Precisions worth using | Not available or not worth it | Runtime pick |
|---|---|---|---|---|
| M3, X3 robots | Orin NX 16 GB Super (sm_87, 102.4 GB/s shared with the CPU) | FP16; INT8 for CNNs via explicit Q/DQ; INT4 weight-only for LLMs | FP8 and FP4 (no hardware) ([Edge-LLM matrix](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/support-matrix.html)); W8A8 GEMMs (2.73x slower than FP16 in one port) ([pi05-orin-edge](https://github.com/hous-lab/pi05-orin-edge)) | TRT 10.7 engines; Edge-LLM or llama.cpp for LLMs |
| Laptop | RTX 2070 Super 8 GB (sm_75, Turing) | FP16; INT8 | BF16 (Turing lacks it; general knowledge, not verified in the notes), FP8, FP4; the NvTensorRTRTX EP and Windows ML NVIDIA EP (Ampere+) | TensorRT (ORT TRT EP on 10.14, or trtexec 11.4); llama.cpp |
| Dionysos | RTX A4500 20 GB (sm_86, Ampere) | FP16, BF16, INT8 | FP8 needs SM89+, FP4 needs SM120 ([NVIDIA TRT-RTX blog](https://developer.nvidia.com/blog/run-high-performance-ai-applications-with-nvidia-tensorrt-for-rtx/)) | TRT 10.16 or 11.4 (CUDA 12 build); llama.cpp |
| hekate | RTX 5090 32 GB (sm_120, Blackwell) | FP16, BF16, INT8, FP8, NVFP4/MXFP4 | FP8/FP4 for small conv nets and ACT, which are launch-bound | TRT 11.4 or 10.16, with TRT-RTX to test; llama.cpp with NVFP4; WSL2 for vLLM and the GR00T scripts |

### Low precision pays only where weights and GEMMs dominate

On the RTX 5090 the gains from FP8 and FP4 are real for large models:
- **FLUX.1-dev:** NVIDIA measured **10,931 ms in FP16, 6,681 ms in FP8 and 3,853 ms in FP4** ([NVIDIA FP4 blog](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus/)).
- **Qwen3-8B in TensorRT-LLM:** 222.7 tok/s in NVFP4 against 170.0 in FP8 ([ComputingForGeeks](https://computingforgeeks.com/nvfp4-quantization-tensorrt-llm/)).
- **llama.cpp's native NVFP4 path:** prompt processing rose by 68 %, but decode stayed flat at ~73.6 tok/s, because decoding at batch 1 is limited by memory bandwidth ([InsiderLLM](https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/)).

Small networks behave differently. At 3 ms, YOLO26n-seg is limited by kernel launches and memory traffic, not arithmetic. VLAQuantBench found every quantized LLM kernel it tested slower than BF16 on an RTX 4090, even where memory use fell ([VLAQuantBench](https://arxiv.org/html/2609.25376)).

On the Orin, only one quantization recipe has been shown to speed up a large model: weight-only W8A16 on pi0.5's gate and up projections. W8A8 GEMMs ran **2.73x slower than FP16** there ([pi05-orin-edge](https://github.com/hous-lab/pi05-orin-edge)). W4A4 without careful calibration took pi0.5 and X-VLA to **0 % success** on LIBERO ([VLAQuantBench](https://arxiv.org/html/2609.25376)). ViTs fare no better under INT8:
- A ViT-S + DPT depth model ran 2.7x slower after ModelOpt INT8 on an Orin Nano. NVIDIA could not reproduce this, measured a ~3 % gain instead, and concluded that "the INT8 kernel gain is limited" ([NVIDIA forum](https://forums.developer.nvidia.com/t/tensorrt-model-optimizer-int8-quantization-causes-2-7x-performance-regression-on-jetson-orin-nano-4gb-vit-s-dpt-architecture/357835)).
- A DEIMv2 detector on an Orin NX fell from 580 to 380 queries per second ([NVIDIA forum](https://forums.developer.nvidia.com/t/worse-performance-after-quantization-on-tensorrt/355549)).

### TensorRT 11 changes how every engine is specified

What TensorRT 11.0 removed:
- weak typing and the FP16/INT8/FP8/INT4/FP4 builder flags;
- implicit INT8 calibration;
- in trtexec 11.x, the flags `--fp16 --int8 --bf16 --fp8 --int4 --best` and `--calib`.

The documented replacement is to run `python -m modelopt.onnx.autocast` on the ONNX, then build it with no precision flags ([TensorRT 11.0.0 notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.0.0.html); [trtexec migration](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-trtexec.html)).

Platform support differs:
- **Windows:** 11.4.0 ships CUDA 12.9 and 13.4 builds and still supports SM 7.5 ([TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)).
- **Jetson:** TensorRT 11 "requires CUDA 13.2 Update 2 or later", and 11.4.0 does not support JetPack at all ([Jetson migration guide](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/api/migration/tensorrt-10x-to-11x-jetson.html)).

Known 11.x issues that touch our GPUs ([TensorRT 11.4.0 notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.4.0.html); [11.0.0 notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.0.0.html)):
- A strongly typed INT8+FP16 YOLOX-Tiny gives wrong outputs on the RTX 5080/5090 in 11.4.0; it passed in 11.3.0.
- A strongly typed FP16 ViT-Base is 6-7 % slower on the RTX PRO 6000 Blackwell in 11.3 than in 11.2.1.
- Some strongly typed FP16 networks on a Blackwell Max-Q regressed by up to ~45 % in 11.0.

The practical consequence: each ONNX should carry its own precision. Use an AutoCast FP16 file and, where INT8 is wanted, a ModelOpt Q/DQ file. Build it with `--stronglyTyped` on TensorRT 10.7 today, and keep the weakly typed `--fp16` engine only for comparison.

### Engines stay per machine; the ONNX is the portable artifact

Default engines are tied to the GPU model, the exact TensorRT build and the OS. The portability options all cost something ([TensorRT engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)):
- `kAMPERE_PLUS` hardware-compatible engines exclude Turing and drop FP8/FP4 tactics, at some speed cost.
- `kSAME_COMPUTE_CAPABILITY` (TensorRT 10.9+) only helps identical GPUs.
- Version-compatible engines "can be slower".

With four different compute capabilities (7.5, 8.6, 8.7, 12.0), the durable contract is:
- one FP32 ONNX per model, plus typed derivatives of it;
- an engine built and cached on each machine;
- engine names that carry the TensorRT version, SM and precision (`model.trt107.sm87.fp16.engine`).

Avoid refit-enabled builds: TensorRT 10.16 documents that they regress convolution engines ([TensorRT 10.16.1 notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.16.1.html)).

## The matrix: what to run on each machine

Latency tags:
- **[M]** we measured it.
- **[P]** published for the same device class.
- **[P\*]** published by a single third-party or unreviewed source.
- **[X]** our extrapolation: AGX Orin to Orin NX at ~2x, 4090 or RTX Pro 6000 to 5090 by analogy, T4 to 2070 Super as the same Turing class.

A cell with no figure is unmeasured. Latency means engine GPU time unless the cell says end to end (e2e) or service. Robot cells assume JetPack 6.2, TensorRT 10.7 and ORT 1.20 at MAXN_SUPER.

### Vision: FP16 engines everywhere, FP8 only for large ViTs on hekate

| Model | Robots: Orin NX, TRT 10.7 | Laptop: RTX 2070 Super | hekate: RTX 5090 | Dionysos: RTX A4500 |
|---|---|---|---|---|
| YOLO26n/s-seg, YOLOE-26s (prompts baked in) | ONNX opset 17, `nms=False`, static 480x640 -> trtexec FP16 + CUDA graph: 3.00 / 5.37 / 5.36 ms [M], e2e 8.3 ms [M]. INT8 only as ModelOpt Q/DQ with autotune; saves at most ~0.5 ms | TensorRT FP16 (ORT TRT EP on 10.14, or trtexec 11.4 on an AutoCast ONNX); INT8 Q/DQ optional | TensorRT FP16 (10.16 weakly typed or 11.4 strongly typed); INT8 Q/DQ only after a numeric check (11.4 YOLOX INT8+FP16 bug on the 5090); FP8/FP4 not worth it | TensorRT FP16; INT8 Q/DQ optional |
| RF-DETR-Seg | FP16 only: Seg-N 8.00 ms at 312 [M]; fallback detector | TensorRT FP16; T4 (Turing) reference: Seg-S 4.4 ms at 384 [P] | TensorRT FP16/BF16 | TensorRT FP16/BF16 |
| Depth Anything V2 S (L for the depth duels) | ONNX opset >= 17, static sides that are multiples of 14, TensorRT FP16: S ~23.6 ms [M]. Lower the resolution: 364x490 has 0.58x the tokens of 476x644. No INT8 | TensorRT FP16 (no BF16); force LayerNorm to FP32 only if outputs drift | TensorRT FP16/BF16; try FP8 Q/DQ on L; RTX 4090 reference including pre/post at 518x518: S 3 / B 6 / L 12 ms [P] | TensorRT FP16/BF16; no FP8 |
| DA3 S/B/L | TensorRT FP16 with our two export fixes (attention scale, camera-encoder qkv split): 46 / 82 / 208 ms for 3 views [M]; single view ~15 ms [P\*] | TensorRT FP16 | TensorRT FP16/BF16; FP8 test on B and L | TensorRT FP16/BF16 |
| DINOv2-S/14 crop embedder | TensorRT FP16, static batch 8 or 16 at 224x224; no Orin NX figure published | TensorRT FP16 | TensorRT FP16/BF16 | TensorRT FP16/BF16 |
| SAM family | NanoSAM on YOLOE boxes (ResNet18 encoder in FP16, decoder in FP32): 8.1 ms on AGX Orin [P], ~10-15 ms here [X]; SAM 3 offline only (~0.95 s [M]) | not a target | SAM 2 / SAM 3 in PyTorch with torch.compile at BF16, for labelling; Embedl's SAM 3 TensorRT engine only on TRT 10.1/10.3 | same as hekate |

The evidence behind the vision cells:
- **Our INT8 failures follow a known pattern.** Q/DQ nodes placed where TensorRT cannot fuse them produce chains of reformat layers (our 100 of 301). Plain INT8 PTQ fails on TensorRT 10.7, which has no INT8 kernel for the mask head's proto conv ([PLAN.md M5](../PLAN.md)). ModelOpt's `--autotune` and node exclusion are the tools aimed at both problems ([ModelOpt ONNX PTQ](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/onnx_ptq/README.md)).
- **FP16 is safe for ViTs if exported correctly.** Export at opset 17 or later, so LayerNorm maps to a native normalization layer ([NVIDIA forum](https://forums.developer.nvidia.com/t/layernorm-fp16-overflows-trt-8-5-6-on-xavier-nx/345971)). Prefer ModelOpt AutoCast over `--layerPrecisions`, which Myelin fusion can override ([NVIDIA forum](https://forums.developer.nvidia.com/t/model-outputting-nans/360219)).
- **The TensorRT upgrade argument for ViTs is new attention fusion.** TensorRT 10.14 added an `IAttention` fused-attention API and automatic padding of misaligned MHA heads ([TensorRT 10.14.1 notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-10/10.14.1.html)). This is the only concrete reason to expect faster Depth Anything or DINOv2 engines from TensorRT 10.16.
- **SAM 3 on an edge GPU is offline-only.** Embedl's quantized SAM 3 takes 462 ms on an AGX Orin and produces wrong masks on TensorRT newer than 10.3 ([Embedl SAM 3](https://huggingface.co/embedl/sam3)). NanoSAM's 8.1 ms AGX Orin pipeline is the real-time option ([NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam)).

### Policies and VLAs: ACT and SmolVLA on board, everything larger on hekate

| Model | Robots | Laptop | hekate | Dionysos |
|---|---|---|---|---|
| ACT | ONNX (`dynamo=False`, opset 17) -> onnxslim -> TensorRT FP16 + CUDA graph: 5.9 ms engine, 12 ms service [M] | TensorRT FP16 for sim evaluation | same; FP8 does not help at this size | same |
| SmolVLA (450M) | split ONNX (9 graphs) -> ORT TRT EP or native TensorRT FP16 with IOBinding and a CUDA graph on the denoise step: ~150-190 ms [X from 189.9 ms on an Orin Nano, P\*]; untested on TRT 10.7 | possible but slow (RTX 2060: ONNX p99 601 ms; the TRT EP build did not fit in 6 GB [P]) | PyTorch with compile, or TensorRT; sub-100 ms round trip over a LAN on a 4090 [P] | same as hekate |
| GR00T N1.7 (3B) | not on board. JP 6.2 allows only the DiT in TensorRT: 216.5 ms on AGX Orin [P] -> ~400 ms here [X]. JP 7.2 full pipeline: 149 ms on AGX Orin [P] -> ~300 ms here [X] | no (needs 16 GB+) | `build_trt_pipeline.py` full pipeline at BF16 (ViT in FP32), under WSL2/Docker: ~25 ms [X from the RTX Pro 6000's 24.7 ms]; eager ~75 ms, compiled ~50 ms [X]; FlashRT FP8 14.0 ms, NVFP4 action head 8.64 ms [P\*] | same pipeline at BF16 as a fallback server; slower than the L40's 35 ms [X] |
| pi0.5 | not recommended. Best AGX Orin result 92 ms (W8A16 INT8 + fused kernels) [P] -> ~185 ms here [X]; BF16 ~440 ms [X] | no (needs more than 8 GB) | LeRobot `pi05` or openpi with torch.compile: ~50 ms [X from openpi JAX's 53.7 ms on a 4090]; Triton kernels + CUDA graph 27.3 ms on a 4090 [P]; FlashRT FP8 13.1 / FP4 8.75 ms [P\*]; RTC adds ~21 ms | BF16 with compile; at least the A6000's 128 ms uncompiled [X] |
| X-VLA (0.9B) | no Orin figure | no | LeRobot `xvla`: 35.7 Hz on a 4090 [P\*] | BF16 |

**Why the large VLAs stay off the robots.** NVIDIA calls Orin a "legacy edge option… only suitable for slow, non-reactive tasks". Its GR00T tables are measured on the 64 GB AGX Orin ([GR00T hardware recommendation](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md)). Our Orin NX has half the AGX's memory bandwidth (102.4 vs 204.8 GB/s) ([Turing Pi](https://docs.turingpi.com/docs/nvidia-jetson-orin-nx-intro-specs); [EagleVLA](https://arxiv.org/html/2607.12659)), so AGX numbers roughly double here. Edge-LLM's measured ratio supports that: Qwen3.5-2B decodes at 47.9 tok/s on the Orin NX and 83.3 tok/s on the AGX Orin ([Edge-LLM benchmarks](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/performance/performance-benchmarks.html)).

**The exporter recipe has converged.** The legacy TorchScript exporter (`dynamo=False`), static batch-1 shapes, simplification, the ViT kept in FP32, CUDA graphs on the action head, and a cosine check of at least 0.999 against PyTorch ([GR00T deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md)). LeRobot still has no official export ([LeRobot PR #4630](https://github.com/huggingface/lerobot/pull/4630)).

**The latency budget comes from chunking.** At 10 Hz, the inference delay in control ticks is d = ceil(latency / 100 ms):
- under ~100 ms including the network is d = 1 and effectively free;
- under 200 ms is d = 2, which is good;
- 300-500 ms works with RTC but reacts like a 2 Hz policy.

RTC adds ~21 ms on pi0.5 (76 -> 97 ms on a 4090) and was robust to an extra 100-200 ms of injected delay ([RTC paper](https://arxiv.org/html/2506.07339v2)). One secondary source says LeRobot 0.6.1's gRPC server cannot pass RTC's inputs ([particula.tech](https://particula.tech/blog/real-time-chunking-vs-async-inference-lerobot)), so a thin server of our own is the safer plan.

**Payloads should be JPEG resized on the robot.** Two raw 256x256 frames are ~393 KB per request, while JPEG is tens of KB. GR00T's stock ZMQ client sends raw `.npy` arrays. NVIDIA's VLA-Perf model concludes that "server-side inference, even with an RTX 4090 over WiFi, beats on-device Jetson Thor in most cases" ([VLA-Perf](https://arxiv.org/html/2602.18397)).

### Language and speech: fix caching and Slovak before chasing runtimes

| Model | Robots | Laptop | hekate | Dionysos |
|---|---|---|---|---|
| Qwen3.5-4B voice parser (current) | too slow on board: Edge-LLM INT4 gives 27.0 tok/s and 120 ms prefill [P, on JP 7.2] -> ~1 s per command [X] | llama-server (move b11617 -> b119xx, CUDA 12.4 build), Q4_K_M, json_schema grammar: 646 ms median [M]. Fix prefix reuse; try MTP | llama.cpp CUDA 13.4 build, Q4_K_M or NVFP4 GGUF (prefill +68 %, decode flat [P\*]); vLLM/SGLang under WSL2 only if batching is needed | llama.cpp Q4_K_M or Q8_0 |
| Distilled 0.8B/2B parser | JP 6.2: llama.cpp built for sm_87 (0.8B Q8_0, 2B Q4_K_M), roughly 0.6x Edge-LLM's decode [X]. Edge-LLM INT4 AWQ: 95.8 / 47.9 tok/s with 35.5 / 56.2 ms prefill [P, on JP 7.2] -> ~0.3 / ~0.6 s per command with a cached prefix [X] | llama.cpp: 2B Q4_K_M 364 ms [M] | — | — |
| Small VLM (Qwen3.5-2B) | Edge-LLM INT4 with an FP16 ViT: ViT 72.6 ms, 296-token prefill 170.6 ms, 48.3 tok/s [P, on JP 7.2] -> ~0.65 s per question [X]. On JP 6.2: llama.cpp with mmproj, or the vLLM Jetson container at W4A16 | llama.cpp with mmproj (Qwen3.5 vision on CUDA unverified) | vLLM under WSL2 with Qwen3.5-9B NVFP4 or Qwen3-VL-8B | vLLM at AWQ/W4A16 |
| Parakeet TDT 0.6B v3 | sherpa-onnx int8 on the CPU, if needed | sherpa-onnx int8 on the CPU, 4 threads: 154-300 ms for 1.4-3.5 s of audio [M]; fp16/fp32 ONNX on CUDA ~40-60 ms per 2.5 s [X from the T4's RTFx of 57] | not needed; Speech NIM under WSL2 (needs compute capability 8.0+ and 16 GB) | same |
| Canary-1B-v2 (+ KInIT's Slovak fine-tune) | too slow on the CPU (~4.5x Parakeet's cost) [X] | ONNX on CUDA (onnx-asr) with `source_lang` forced: ~0.1-0.15 s per 2.5 s [X]; 0.6-1 s on the CPU [X] | Speech NIM under WSL2 | same |

**Edge-LLM numbers.** The Orin NX figures are from NVIDIA's v0.10.0 dashboard, measured on JetPack 7.2 with CUDA 13.2 and TensorRT 10.16 ([Edge-LLM benchmarks](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/performance/performance-benchmarks.html)). Without prefix reuse, a 1,500-token system prompt would add ~0.6 s (0.8B), ~0.85 s (2B) and ~1.9 s (4B) per command on the Orin NX (our arithmetic from those prefill rates), so prefix reuse is mandatory on a robot.

**llama.cpp speed-ups to try.** llama.cpp now has in-tree MTP, EAGLE-3, DFlash and n-gram speculative decoding ([speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)). Qwen3.5 ships MTP heads ([Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B)), so `--spec-type draft-mtp` with an MTP-preserving GGUF is the decode accelerator to try. Its interaction with grammar-constrained output is undocumented.

**ASR: accuracy, not speed, is the problem.**
- **Parakeet v3** auto-detects language with no way to force it, which explains the Cyrillic outputs. Its FLEURS WER is cs 11.01 % and sk 8.82 % ([Parakeet v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)).
- **Canary-1B-v2** takes the source language as a prompt and reaches cs 7.86 % ([Canary-1B-v2](https://huggingface.co/nvidia/canary-1b-v2)).
- **KInIT's Slovak fine-tune** cuts Common Voice 24 Slovak WER from 14.00 % to **8.27 %** ([kinit/canary-1b-v2-sk](https://huggingface.co/kinit/canary-1b-v2-sk)).
- **Running Canary on the GPU** keeps it fast. onnx-asr measures Canary at RTFx 8 on a fast desktop CPU and 21 on a T4 ([onnx-asr](https://github.com/istupakov/onnx-asr)).
- **Speech NIM** needs compute capability 8.0+ and 16 GB, so it cannot run on the laptop or the Orin ([ASR NIM matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html)).

### Gaussian splats: faster loop on hekate, smaller files for the console

| Task | Robots | Laptop | hekate | Dionysos |
|---|---|---|---|---|
| Training | not worth it | no | gsplat 1.5.3 with our loop, 7000 steps ~1 min [M]. Target 20-40 s with fused SSIM, SelectiveAdam, `packed`, an MCMC cap and SH <= 1 [X]. gsplat main (2026 kernels, PyTorch >= 2.7) in a separate venv | possible but not needed (shared) |
| Delivery to the console | — | `.splat` (32 B per Gaussian, no SH) is fine at 150-300k Gaussians. For room scans, switch to SOG (~15-20x smaller than PLY) or SPZ v4 (~10x), written by splat-transform with `-H 0` | produces the files | — |
| Browser rendering | — | our WebGL2 viewer: move the sort into a Web Worker and pack each Gaussian into 32 B; Spark 2.2 (WebGL2) if SH or LoD is needed | — | — |
| Isaac Sim asset | — | — | Isaac Sim 6.0+: 3DGRUT `transcode --format lightfield` (OpenUSD ParticleField); `--format nurec` for an A/B; collider from the scan mesh | Isaac Sim 5.0.0: NuRec USDZ only |

**Training.** The newest gsplat on PyPI is still **1.5.3 (July 2025)**. Its 2026 kernel work lives only on the main branch, which requires PyTorch 2.7+ ([gsplat PyPI](https://pypi.org/project/gsplat/#history); [gsplat README](https://github.com/nerfstudio-project/gsplat)). For a 20-60-view object scan, our own per-step Python overhead is the cheaper target: a hand-written SSIM, six separate Adam optimizers, and SH work that the console then discards.

**Formats.** SOG and SPZ v4 are the compact formats with the widest reader support, and PlayCanvas' `splat-transform` writes both from our PLY ([PlayCanvas formats](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/); [splat-transform](https://github.com/playcanvas/splat-transform)). glTF `KHR_gaussian_splatting` is lossless and PLY-sized, so it is an interchange format, not a way to shrink files ([KHR README](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md)).

**Isaac Sim.** Isaac Sim 6.0 renders OpenUSD ParticleField natively. NuRec USDZ works from 5.0 to 6.0 and is slated for deprecation ([3DGRUT export README](https://github.com/nv-tlabs/3dgrut/blob/main/threedgrut/export/README.md)). One user found NuRec sharper and ParticleField flickery on the same PLY ([NVIDIA forum](https://forums.developer.nvidia.com/t/question-on-two-3dgs-rendering-pipelines-nurec-vs-particlefield-and-expected-quality-tradeoff-in-isaac-sim-6-0/373260)), which argues for an A/B on hekate's build.

## Stay on JetPack 6.2 until one of four triggers fires

The robots have three paths, plus a fourth that is not yet available. Only one of them is about speed.

| Path | What it buys | What it costs | Verdict |
|---|---|---|---|
| Stay: JetPack 6.2 (L4T 36.4.x) + the TensorRT 10.7 tarball | NVIDIA's recommended DLA stack ("Use TensorRT 10.7 if DLA is required"); the INT8 export fix that 10.3 lacks ([Ultralytics](https://docs.ultralytics.com/guides/nvidia-jetson)); ROS 2 Humble and both vendors' stacks; every current engine stays valid | CUDA 12.6 frozen while upstream ORT deprecates its CUDA 12 packages; no official Edge-LLM; Isaac-GR00T main says "JetPack 6.x is no longer supported"; none of the 10.14+ attention work; no route to TensorRT 11 | Default |
| Patch: JetPack 6.2.3 (L4T 36.5.2, 12 Aug 2026), by APT | Security and bug fixes; same CUDA 12.6, TRT 10.3 and cuDNN 9.3 ([NVIDIA forum](https://forums.developer.nvidia.com/t/jetpack-6-2-3-jetson-linux-36-5-2-is-now-live/379871)) | An `apt dist-upgrade` on a robot; afterwards, check that trtexec still reports 10.7 | Housekeeping, one robot first |
| Move: JetPack 7.2 / 7.2.1 (L4T 39.2.x) | Ubuntu 24.04, CUDA 13.2.1, cuDNN 9.20, TensorRT 10.16.2 ([JetPack 7.2 archive](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2)); the official Edge-LLM and GR00T Orin paths; required before any future TensorRT 11 for Jetson | A reflash with no in-place path, and no source on Yahboom carrier support; ROS 2 Humble -> Jazzy for both vendor stacks; no Orin DLA; every engine, JetPack 6 container and CUDA wheel rebuilt; Isaac ROS "coming soon"; no published CNN or ViT speed gain | Defer |
| TensorRT 11.x on Jetson | Strong typing; ModelOpt-only quantization | 11.4.0 "NVIDIA JetPack is not supported"; it needs JetPack 7.x and CUDA 13.2u2+, and it drops the precision flags, the calibrators and the cuBLAS/cuDNN tactic sources | Not available; prepare our exports now |

**DLA is the reason most often given for staying, and for us it is the weakest.** NVIDIA's migration guide says "Use TensorRT 10.7 if DLA is required" ([Jetson migration guide](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/api/migration/tensorrt-10x-to-11x-jetson.html)). On JetPack 7.2 and 7.2.1, Orin DLA builds fail with "Cannot create DLA engine". A user still failed on 7.2.1 with TensorRT 11 on 30 September, and on 8 October NVIDIA promised DLA in JetPack 7.2.2 "by the end of October" ([NVIDIA forum](https://forums.developer.nvidia.com/t/dla-runtime-unavailable-on-jetpack-7-2-l4t-r39-2-cannot-create-dla-engine/373788)). But we run nothing on the DLA. It is several times slower per frame than the GPU: on an Orin NX, YOLOv7-Tiny on the GPU alone beat GPU plus two DLAs ([NVIDIA forum](https://forums.developer.nvidia.com/t/how-to-use-dla-correctly/322765)). Running GPU and DLA together also slows both ([NVIDIA forum](https://forums.developer.nvidia.com/t/when-gpu-and-dla-are-used-at-the-same-time-the-time-consumption-increases-with-each-other/245332)).

**The real costs of JetPack 7.2 are elsewhere:**
- Ubuntu 24.04 means leaving ROS 2 Humble, which neither vendor's stack supports there ([PLAN.md](../PLAN.md), internal). A Humble container on a 24.04 host is a possible workaround that no note researched.
- No in-place upgrade path exists ([JetPack 7.2 announcement](https://forums.developer.nvidia.com/t/jetpack-7-2-jetson-software-goes-agentic-with-jetson-linux-39-2/372059)).
- No source covers Yahboom carrier support.
- JetPack 6.2 engines "should not be copied to JetPack 7.2" ([Seeed](https://wiki.seeedstudio.com/deploy_tensorrt_edge_llm_on_jetpack7.2/)).

**On the benefit side, there is no speed evidence.** No published A/B shows a CNN or small ViT running faster on TensorRT 10.16 than on 10.7 on an Orin. TensorRT 10.7 predates the ViT regression introduced around 10.9 and fixed in 10.13.2 ([10.13.2 notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.13.2.html)), so 10.16 restores nothing we lost. The plausible speed argument is the 10.14 attention work, which must be measured.

**The best case for moving is software, not kernels.** Edge-LLM is officially supported on Orin only with JetPack 7.2. Isaac-GR00T main no longer supports JetPack 6.x ([GR00T deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md)). Any TensorRT 11 for Jetson will need JetPack 7.x first. Neither on-robot use is needed today: the voice chain runs on the laptop, and GR00T on an Orin NX would take ~300 ms even with the full JetPack 7.2 pipeline.

**Four triggers should reopen the decision:**
1. JetPack 7.2.2 ships with Orin NX DLA verified, or we decide we never need DLA.
2. Yahboom support and both ROS 2 vendor stacks are confirmed on Ubuntu 24.04/Jazzy, or a Humble container proves out.
3. A concrete need for on-board Edge-LLM appears (the distilled parser or a VLM).
4. A timed A/B on a spare module shows TensorRT 10.16 cutting DA V2-S or DINOv2 by a margin that matters, say 20 %.

Until then, JetPack 6.2.3 is the one upgrade worth doing.

### The Windows machines can adopt TensorRT 11 and CUDA 13 piecemeal

Nothing released by October 2026 deprecates Turing. CUDA 13.0-13.4, TensorRT 11.4, TRT-RTX 1.6/1.7, cuDNN 9 and PyTorch's CUDA 13 wheels all keep sm_75; the cut fell on Volta and older ([CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html); [PyTorch notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)). The CUDA 12 window is closing all the same:
- PyTorch 2.14 is the last release with CUDA 12 wheels ([PyTorch 2.14 blog](https://pytorch.org/blog/pytorch-2-14-release-blog/)).
- ONNX Runtime's default GPU wheel has been CUDA 13 since 1.27 ([ORT CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)).
- On Windows the CUDA 13 path cannot come from pip, because `nvidia-cublas` 13.x ships no `win_amd64` wheel ([PyPI nvidia-cublas](https://pypi.org/project/nvidia-cublas/)).

What that means per machine:
- **Laptop.** Keep ORT 1.24.1 with CUDA 12 as the production path; its TensorRT EP on 10.14 still accepts weak typing. Add a standalone TensorRT 11.4 only for experiments. Move llama.cpp from b11617 to a current b119xx build, CUDA 12.4 zip first and the 13.4 zip as a test ([llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)).
- **hekate.** Its CUDA 12.8.1 toolkit matches the TensorRT 11.4 CUDA 12 wheel and the TRT-RTX CUDA 12.9 package without a toolkit change. TensorRT 10.16 alongside gives a flag-for-flag comparison with the robots. Every install there is a software change that CLAUDE.md says to ask about first.
- **Dionysos.** Driver 581.42 (R580) runs CUDA 13.0 natively and 13.x through minor-version compatibility, but features new in 13.4 need R615 or later. The CUDA 12 builds avoid asking the lab for a driver change. TRT-RTX fails under TCC, so check the A4500's driver model, read-only, before trying it ([TRT-RTX 1.6 notes](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.6.html)).

Turing is the oldest architecture left in CUDA 13. One personal wiki calls it the likely next one to be removed; that is speculation, not an NVIDIA statement ([Bommarito wiki](https://michaelbommarito.com/wiki/programming/tools/cuda-13/)).

## Ten actions ranked by value per hour

| Rank | Action | Machine | Expected gain | Cost and risk |
|---|---|---|---|---|
| 1 | Move YOLO mask post-processing to the GPU at proto resolution: crop inside boxes, upsample only the kept masks, no `retina_masks` | Both robots | 3.5 ms -> under ~1 ms; detector end to end 8.3 -> under ~5 ms [X] | About a day of code in our predict path; no OS change |
| 2 | Make Qwen3.5's static prompt prefix reusable: update llama.cpp, set `-cms 0` (or a small value), keep the names list and the command at the end; fall back to Qwen3 4B 2507 if checkpoints stay unreliable | Laptop | Plausibly most of the prompt-processing share of the 646 ms (a hypothesis) | Hours; verify with `timings.prompt_n` |
| 3 | Serve GR00T N1.7 (TensorRT full pipeline, BF16) and pi0.5 (torch.compile) from hekate through one thin persistent server: JPEG resized on the robot, TCP_NODELAY, stale requests dropped, RTC inputs accepted | hekate and robots | From ~300-500 ms on board [X] to ~25-80 ms plus 10-30 ms of network: d = 1 at 10 Hz [X] | Days; WSL2/Docker and a firewall rule on hekate need the user's go-ahead |
| 4 | Test Canary-1B-v2 with a forced `source_lang`, and KInIT's Slovak fine-tune, on the laptop GPU | Laptop | Slovak WER 14.00 -> 8.27 % (Common Voice 24) at ~0.1-0.15 s per command [X] | A day; VRAM next to the 4B LLM |
| 5 | Switch every export to a typed ONNX plus `--stronglyTyped`, ModelOpt Q/DQ for any INT8, per-machine engine names, no refit | All | No speed gain today; no breakage later at TensorRT 11 or JetPack 7 | Hours; benchmark typed against weak once |
| 6 | Trim the ACT service: a CUDA graph around `execute_async_v3`, keep-alive and TCP_NODELAY, smaller or GPU JPEG decode | M3 | 12 -> ~7 ms per request [X] | Hours |
| 7 | Time the engines with `jetson_clocks`, then decide a clock policy for tasks | Both robots | Unknown; removes DVFS ramp-up from per-frame latency | Minutes; power and heat on battery |
| 8 | Splat loop and delivery: fused SSIM, SelectiveAdam, `packed`, an MCMC cap, SH <= 1 for console splats; SOG/SPZ with `-H 0`; the worker sort and 32-byte packing in the viewer | hekate, laptop | ~1 min -> 20-40 s per fit [X]; files 2-5x smaller [X] | A day; hekate GPU (ask first) |
| 9 | One last INT8 attempt on YOLO26n-seg: ModelOpt `--autotune` on the Orin with the proto branch excluded; accept only under 2.7 ms with mask mAP50 >= 0.78 | Robots | At most ~0.5 ms | Half a day; whether ModelOpt installs on JetPack 6.2 is unverified |
| 10 | A/B of JetPack 7.2.x with TensorRT 10.16 on a spare module | Spare module | Unknown (`IAttention` for ViTs, Edge-LLM) | Days; only after a trigger fires |

Several popular options are not worth our time:
- INT8 for the ViTs on the Orin: flat to slower.
- FP8 or NVFP4 for small CNNs or ACT.
- W8A8 or W4A4 for VLAs on the Orin.
- DLA for latency. At most, a concurrency probe if a second always-on CNN ever appears.
- DirectML.
- Native-Windows TensorRT-LLM, deprecated there since 0.18 ([TensorRT-LLM notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html)).
- Triton Inference Server on Windows; its last Windows build was 24.03.
- Hardware- or version-compatible engines in production.
- Refit-enabled engines.
- Copying any engine between machines.

## Where the evidence conflicts, extrapolates or rests on one source

| Topic | What the sources say | How this report treats it |
|---|---|---|
| Edge-LLM on JetPack 6.2 | Sources disagree. The official support matrix (29 Sep 2026) lists Orin with JetPack 7.2 only ([matrix](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/support-matrix.html)). Edge-LLM 0.7.0 called JetPack 6.2.x "compatible", experimental ([0.7.0 overview](https://nvidia.github.io/TensorRT-Edge-LLM/0.7.0/overview.html)). Seeed's v0.9.1 guide says JetPack 6.2+ "remains compatible", yet builds Qwen3.5 with `-DCUDA_CTK_VERSION=13.2 -DENABLE_CUTE_DSL=ALL` ([Seeed](https://wiki.seeedstudio.com/deploy_tensorrt_edge_llm_on_jetpack7.2/)). One community run on an Orin Nano with JetPack 6.2.x and TensorRT 10.3 needed a CMA kernel change and an ONNX rewrite ([Hokwang Choi](https://hokwangchoi.com/blog/vlm-benchmarks/)) | Possible but unsupported on our robots. Because Qwen3.5 needs CuTe DSL kernels built against CUDA 13.2, a CUDA 12.6 build is doubtful (our inference). Every Orin NX Edge-LLM figure was measured on JetPack 7.2 |
| SmolVLA on Orin | 189.9 ms p50 on an Orin Nano 8 GB (JetPack 7.2, ORT + TRT EP, FP16, 9 graphs), from a single unreviewed upload ([eetmie](https://huggingface.co/eetmie/smolvla-base-onnx)), against "about 25 ms p50 on Orin Nano" in one blog ([jared-hpc](https://jared-hpc.com/posts/vla-physical-ai-tensorrt-tether/)) | Use ~190 ms. The 25 ms claim has not been reproduced and may cover only part of the pipeline. No Orin NX or JetPack 6.2 figure exists. A separate paper found "FP16/INT8" SmolVLA ONNX files that were actually FP32 ([arXiv 2609.14146](https://arxiv.org/abs/2609.14146)) |
| KHR_gaussian_splatting | The registry README says "Complete, Ratified" ([README](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md)). Secondary sources date the ratification commit to 2026-09-03, with no Khronos press release ([radiancefields](https://radiancefields.com/platforms/the-khronos-group)). Pages updated in July 2026 still say release candidate ([Khronos RC release](https://www.khronos.org/news/press/gltf-gaussian-splatting-press-release)) | Treat the base extension as ratified; the SPZ compression extension is still a draft |
| Orin DLA after JetPack 7.2 | An NVIDIA moderator said on 10 Sep that DLA works with JetPack 7.2.1 + TensorRT 11.3.0 GA. The TensorRT docs say 11.2.1 and earlier support only the Orin iGPU, and that 11.4.0 DLA is unsupported on JetPack ([support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)). A user still failed on 7.2.1 on 30 Sep. NVIDIA said on 8 Oct that DLA comes in 7.2.2 "by the end of October" ([forum](https://forums.developer.nvidia.com/t/dla-runtime-unavailable-on-jetpack-7-2-l4t-r39-2-cannot-create-dla-engine/373788)) | DLA requires JetPack 6.x with TensorRT 10.7 or older today; 7.2.2 had not appeared by 27 Oct |
| GR00T N1.7 on Orin | 149 ms for the full pipeline (README, 2 cameras, JetPack 7.2). 216.5 ms with only the DiT in TensorRT, plus "TensorRT 10.3 does not support the backbone engine" (HF card, 1 camera) ([card](https://huggingface.co/nvidia/GR00T-N1.7-3B)). 6.6 Hz with TensorRT (hardware page). All three are AGX Orin 64 GB | Orin NX ≈ 2x the AGX Orin time (half the bandwidth and half the GPU), supported by Edge-LLM's 47.9 vs 83.3 tok/s ratio. Still an extrapolation |
| VLA kernel numbers on the 5090 | FlashRT reports GR00T N1.7 at 14.0 ms FP8 and 8.64 ms with an NVFP4 action head, and pi0.5 at 13.10 / 8.75 ms. No accuracy check was published for GR00T, and all figures come from one author ([FlashRT](https://github.com/LiangSu8899/FlashRT)) | An upper bound on what custom kernels reach. Plan around NVIDIA's TensorRT pipeline: ~25 ms by analogy with the RTX Pro 6000 |
| ORT and CUDA 12 | The ORT docs list CUDA 12.8 builds for 1.27-1.30 ([docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)). An aggregated changelog quotes the 1.26 notes as saying CUDA 12 "will be removed in 1.27.0" ([Safety CLI](https://data.safetycli.com/packages/pypi/onnxruntime/changelog)). The 1.27 notes call the CUDA 12 packages deprecated ([newreleases](https://newreleases.io/project/github/microsoft/onnxruntime/release/v1.27.0)) | Treat ORT 1.30 as the last CUDA 12 line; pin versions |
| Bandwidth and labels | The Orin NX's 102.4 GB/s comes from a third-party spec page ([Turing Pi](https://docs.turingpi.com/docs/nvidia-jetson-orin-nx-intro-specs)). Ultralytics' Orin NX table does not name its model; a 5.3 MB `.pt` suggests YOLO26n detect at 640 ([Ultralytics](https://docs.ultralytics.com/guides/nvidia-jetson)) | Used as approximate |
| Dates and names | JetPack 7.2.1 is dated 11 or 12 Aug 2026, both from search summaries. DeepStream is 9.1 on the archive page and 8.0 in the forum post. llama.cpp is "v0.7.0" on Wikipedia, but GitHub shows only bNNNN tags ([Wikipedia](https://en.wikipedia.org/wiki/Llama.cpp)) | Immaterial to the decisions |

## One benchmark plan per machine

### Conventions that make the four plans comparable

**One CSV row per machine × model × runtime × precision**, with these columns:
- **Versions:** host, GPU, driver, CUDA runtime, and the TensorRT, TRT-RTX, ORT, PyTorch and llama.cpp versions.
- **Model:** model name, ONNX hash, input shape.
- **Build:** precision; strongly or weakly typed; runtime; CUDA graph on or off; clock state.
- **Neighbours:** other GPU processes and their VRAM.
- **Cold start:** build time, cold and with a timing cache; engine size; first-inference time (JIT, CUDA graph capture, lazy init).
- **Steady state:** p50/p90/p99; GPU compute versus enqueue time; latency with and without host-device transfers; end-to-end latency in the real process.
- **Resources:** peak memory; power (VDD_IN on the Orin, `power.draw` on Windows).
- **Accuracy:** the delta against the reference.

**Three timing scopes, always reported separately:**
- trtexec GPU compute;
- engine plus pre/post-processing in the real process;
- camera to result.

On Orin, trtexec and the Python API disagree ([NVIDIA forum](https://forums.developer.nvidia.com/t/performance-discrepancy-python-api-vs-trtexec-on-jetson-agx-orin-board/258856)). Report cold and warm runs apart. On shared machines, interleave configurations (A B C A B C) and report medians.

Every precision change gets an accuracy row on the same engine, against these gates:

| Family | Reference | Gate |
|---|---|---|
| Kit detector | FP16 engine on the kit val set | Mask mAP50 >= 0.78 (FP16 is 0.791); recall on the 40 mm cube and the 30 mm pieces; INT8 picks its confidence threshold from its own F1 curve |
| Depth models | PyTorch FP32 on saved close looks | Median abs-rel difference; `m3pro/tools/m3_duel.py` height error, % named, floor sd |
| DINOv2 | PyTorch FP32 embeddings | Cosine > 0.999; teach/find still 4/4 on the look-alikes |
| ACT | LeRobot at CUDA FP32 on `act_check.npz` | Per-joint max <= 0.5 deg (0.19 today) |
| VLAs | PyTorch with the same noise tensor | Mean <= 0.5 deg, max <= 2 deg, cosine >= 0.999; then the 20-cell sim grid and live trials scored by `m3_policy_trial.py` |
| LLM parser | `console/llm_cases.json` | 87/87 correct, valid JSON |
| ASR | Our own sk/cs/en recordings | WER/CER per language, count of wrong-script outputs, command accuracy through ASR -> LLM |
| Splats | Held-out views after pose alignment | PSNR and depth error in mm; the PSNR change when decoding each format back |

Ground rules for all four plans:
- Downloads (weights, binaries, containers) and any install on hekate or Dionysos need the user's go-ahead first.
- Engines are never copied between machines.
- Every ModelOpt or trtexec flag below should be checked with `--help` on the installed version first; the notes flag several as unverified.

### Orin NX robots: clocks, typed FP16, one INT8 retry, DLA probe, end to end

**Step 0, record the platform** before every campaign. `train/robot_bench.sh` calls `/usr/src/tensorrt/bin/trtexec`. The 10.7 tarball and the stock 10.3 deb install trtexec in different places ([jetson-containers PR #1750](https://github.com/dusty-nv/jetson-containers/pull/1750)), so the version line must say 10.7.

```bash
cat /etc/nv_tegra_release
dpkg -l | grep -E 'nvidia-jetpack|tensorrt|cudnn' | awk '{print $2,$3}'
T=/usr/src/tensorrt/bin/trtexec
$T --help | head -1                         # must show [TensorRT v1007xx], not v1003xx
sudo nvpmodel -q; sudo jetson_clocks --show
tegrastats --interval 100 --logfile tegra_$(date +%H%M).log &   # GPU MHz, EMC %, VDD_IN, temperatures
```

**Step 1, clocks.** Run every engine twice:
- **A:** MAXN_SUPER with clocks unlocked, as in the M5 table.
- **B:** pinned with `jetson_clocks`.

Record which robot services were up.

```bash
sudo jetson_clocks --store /tmp/clocks.conf
# ... run A ...
sudo jetson_clocks
# ... run B ...
sudo jetson_clocks --restore /tmp/clocks.conf
```

**Step 2, strongly typed FP16 baselines.** Models to build:
- YOLO26n-seg and YOLO26s-seg at 480x640
- YOLOE-26s
- DA V2-S at 308x406, 364x490 and 476x644
- DA3-S/B with 1 and 3 views
- DINOv2-S/14 at batch 8 and 16, 224x224
- the onnxslim'd ACT
- RF-DETR-Seg-N, only if YOLO regresses

Make the typed ONNX on the laptop, since ModelOpt is pure Python, then copy it to `~/roboarm/models/kit`. Run `onnxslim` on ViTs and policies first. Check the build log for the LayerNorm FP16-overflow warning, and the layer info for fused MHA nodes. Keep a weakly typed build for continuity with the 2026-10-12 table.

```bash
python -m modelopt.onnx.autocast --onnx_path m.onnx --output_path m_fp16.onnx        # on the laptop
$T --onnx=m_fp16.onnx --stronglyTyped --saveEngine=m.trt107.sm87.fp16.engine \
   --timingCacheFile=timing.cache --builderOptimizationLevel=5 \
   --profilingVerbosity=detailed --exportLayerInfo=m_fp16.layers.json --skipInference
$T --loadEngine=m.trt107.sm87.fp16.engine --useCudaGraph --noDataTransfers --useSpinWait \
   --warmUp=1000 --duration=15 --percentile=50,90,99 --exportTimes=m_fp16.times.json   # kernel time
$T --loadEngine=m.trt107.sm87.fp16.engine --useCudaGraph --warmUp=1000 --duration=15      # with H2D/D2H
$T --loadEngine=m.trt107.sm87.fp16.engine --separateProfileRun --dumpProfile --exportProfile=m_fp16.prof.json
$T --onnx=m.onnx --fp16 --saveEngine=m.trt107.sm87.weakfp16.engine --timingCacheFile=timing.cache \
   --useCudaGraph --noDataTransfers --warmUp=500 --duration=10                          # M5-style
```

**Step 3, one INT8 retry, YOLO26n-seg only.** Run ModelOpt on the Orin itself, so its autotune times the right TensorRT. Calibration data is 500 preprocessed kit frames saved as `kit_calib_500.npy`. Exclude `model.23/proto/*`, ideally the whole mask branch; the exclusion flag's name is unverified. Whether ModelOpt 0.47 installs on JetPack 6.2's Python 3.10 is also unverified. If it does not, place the Q/DQ nodes on the laptop without autotune.

Variants to run:
- `--calibration_method=max`;
- the Ultralytics QAT ONNX, rebuilt strongly typed and with `--builderOptimizationLevel=5`.

Accept an INT8 engine only if it runs under 2.7 ms and keeps mask mAP50 at or above 0.78.

```bash
python -m modelopt.onnx.quantization --onnx_path=yolo26n_seg.onnx --quantize_mode=int8 \
  --calibration_data_path=kit_calib_500.npy --calibration_method=entropy \
  --autotune=default --output_path=yolo26n_seg.int8.onnx
$T --onnx=yolo26n_seg.int8.onnx --stronglyTyped --saveEngine=yolo26n_seg.trt107.sm87.int8.engine \
   --timingCacheFile=timing.cache --profilingVerbosity=detailed --exportLayerInfo=int8.layers.json --skipInference
python3 -c "import json,sys; L=json.load(open(sys.argv[1]))['Layers']; \
print(len(L),'layers', sum('Reformat' in json.dumps(l) for l in L),'reformats')" int8.layers.json
```

**Step 4, DLA probe**, on TensorRT 10.7 only:
1. Do one verbose FP16 build to see how much of YOLO26n-seg stays on the DLA.
2. If most of it does, make an INT8 DLA ONNX with ModelOpt `--target_dla` (on 10.7 this needs the Q/DQ Translator and a matching calibration cache).
3. Run the concurrency test: Depth Anything V2-S on the GPU together with the detector on DLA0 and DLA1. Compare each process's latency with its solo run.

```bash
$T --onnx=yolo26n_seg.onnx --fp16 --useDLACore=0 --allowGPUFallback \
   --saveEngine=yolo26n_seg.trt107.dla.fp16.engine --verbose > dla_build.log 2>&1
grep -niE "layers running on (dla|gpu)" -A40 dla_build.log
$T --loadEngine=dav2s.trt107.sm87.fp16.engine --useCudaGraph --duration=15 > c_gpu.log 2>&1 &
$T --loadEngine=yolo26n_seg.trt107.dla.fp16.engine --useDLACore=0 --duration=15 > c_dla0.log 2>&1 &
$T --loadEngine=yolo26n_seg.trt107.dla.fp16.engine --useDLACore=1 --duration=15 > c_dla1.log 2>&1 &
wait; grep -H "GPU Compute Time" c_*.log
```

**Step 5, end to end, the number that counts.**
- **Kit detector:** re-run `train/robot_e2e.py` (the 92-look M5 measurement), timed per stage (preprocess, infer, mask post), before and after the GPU mask change. The target is under 5 ms.
- **ACT:** instrument `m3pro/tools/policy_service.py` per stage: base64 decode, `cv2.imdecode`, resize/normalize, H2D, engine, D2H, JSON encode. Compare with `$T --loadEngine=models/policy/act.engine --useCudaGraph --noDataTransfers --warmUp=2000 --iterations=2000 --percentile=50,90,99` and the same run without `--useCudaGraph`.
- **Loop:** in the bridge's policy job, record tick lateness, picture age and the 10.0 ticks/s rate.

**Step 6, optional extras:**
- **SmolVLA:** check that ORT 1.20 on the robot exposes TensorRT with `python3 -c "import onnxruntime as o; print(o.get_available_providers())"`. One Jetson AI Lab wheel shipped without it ([NVIDIA forum](https://forums.developer.nvidia.com/t/onnxruntime-installation-on-jetson-agx-orin-developer-kit/368180)). If it is missing, build the split graphs with trtexec.
- **LLM:** build llama.cpp for sm_87 and compare with Edge-LLM's Orin NX reference for the 2B model: 47.9 tok/s decode, 56.2 ms prefill.
- **ASR:** run sherpa-onnx int8 with 2, 4 and 6 threads while the robot stack runs.

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 && cmake --build build -j8 --config Release
./build/bin/llama-bench -m Qwen3.5-2B-Q4_K_M.gguf -ngl 99 -fa 1 -p 64,512 -n 32 -r 5 -o md
./build/bin/llama-bench -m Qwen3.5-0.8B-Q8_0.gguf -ngl 99 -fa 1 -p 64,512 -n 32 -r 5 -o md
```

**Pitfalls on the Orin:**
- Timing caches are valid only within one TensorRT version.
- `--noDataTransfers` with CUDA graphs is a best case; report it apart from host and end-to-end figures.
- `--dumpProfile` perturbs timing, so it belongs in a separate run.
- MAXN_SUPER throttles past the TDP: watch tegrastats clocks and temperatures on long runs and on battery.
- `--allowGPUFallback` hides layers that miss the DLA, so always read the build log.
- TensorRT 10.7 has no INT8 kernel for the YOLO26 proto conv, so that layer stays FP16.
- Ultralytics' own YOLO26 INT8 lost three mAP50-95 points on this board ([Ultralytics](https://docs.ultralytics.com/guides/nvidia-jetson)), so every INT8 engine is gated on the kit val set.
- Both robots agreed within 0.05 ms on engine timings ([PLAN.md M5](../PLAN.md)), so one robot is enough for engine runs. End-to-end runs need both, because their cameras differ.
- Measure with the robot stack both running and stopped, because ROS 2 and the vision services share the unified memory and the GPU.

### Laptop: ORT versus TensorRT, the parser's cache, and forced-language ASR

**Step 0, set up.** Run on mains with the High performance power plan, from a local folder that is not synced. Log clocks during every run. Locking clocks (`nvidia-smi -lgc <min>,<max>`, needs admin; undo with `-rgc`) is the user's call on this machine. Also record the VRAM used by the console's LLM, because the 8 GB budget is shared.

```powershell
nvidia-smi --query-gpu=driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used --format=csv -lms 200 > gpu_log.csv
```

**Step 1, vision runtimes** for YOLO26n-seg and DA V2-S. Compare:
- trtexec on TensorRT 11.4 with the AutoCast FP16 file;
- trtexec on TensorRT 10.14 with weak `--fp16`, the version ORT 1.24.1's TensorRT EP loads;
- the ORT TensorRT EP;
- the ORT CUDA EP, with and without a CUDA graph.

TensorRT 11 runs an FP32 ONNX at FP32 without any warning, so make sure the AutoCast file is the one being timed. TRT-RTX is optional here: it needs a Turing-specific engine and cannot be reached through ORT on Turing.

```powershell
python -m modelopt.onnx.autocast --onnx_path m.onnx --output_path m_fp16.onnx
& "$env:TRT114\bin\trtexec.exe" --onnx=m_fp16.onnx --saveEngine=m.trt114.sm75.fp16.engine --timingCacheFile=tc_sm75_114.cache --skipInference
& "$env:TRT114\bin\trtexec.exe" --loadEngine=m.trt114.sm75.fp16.engine --useCudaGraph --useSpinWait --noDataTransfers --warmUp=2000 --duration=20 --percentile=50,90,99 --exportTimes=times.json
& "$env:TRT1014\bin\trtexec.exe" --onnx=m.onnx --fp16 --saveEngine=m.trt1014.sm75.fp16.engine --timingCacheFile=tc_sm75_1014.cache --useCudaGraph --noDataTransfers --warmUp=2000 --duration=20
```

The same ORT harness runs on all three Windows GPUs. Invoke it as `python -I bench_ort.py m.onnx ref_input.npy`, with a real frame as input, never zeros. Run each configuration twice: once with an empty `trt_cache`, to record the engine build, and once warm.

```python
import sys, time, numpy as np, onnxruntime as ort
ort.preload_dlls()                                       # Windows DLL search, ORT >= 1.21
onnx, x_np = sys.argv[1], np.load(sys.argv[2]).astype(np.float32)
def bench(providers, n=1000, warm=100):
    t0 = time.perf_counter(); s = ort.InferenceSession(onnx, providers=providers)
    create_s = time.perf_counter() - t0                  # includes the TRT build on an empty cache
    io = s.io_binding()
    io.bind_ortvalue_input(s.get_inputs()[0].name, ort.OrtValue.ortvalue_from_numpy(x_np, "cuda", 0))
    for o in s.get_outputs():                            # fixed output buffers, needed for CUDA graphs
        shape = [d if isinstance(d, int) else 1 for d in o.shape]   # use real sizes for dynamic dims
        dt = np.float16 if "float16" in o.type else np.float32
        io.bind_ortvalue_output(o.name, ort.OrtValue.ortvalue_from_shape_and_type(shape, dt, "cuda", 0))
    t0 = time.perf_counter(); s.run_with_iobinding(io); first_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(warm): s.run_with_iobinding(io)
    lat = []
    for _ in range(n):
        t0 = time.perf_counter(); s.run_with_iobinding(io); lat.append((time.perf_counter() - t0) * 1e3)
    return dict(create_s=create_s, first_ms=first_ms, p50=np.percentile(lat, 50), p99=np.percentile(lat, 99))
trt = ("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
       "trt_engine_cache_path": "trt_cache", "trt_timing_cache_enable": True, "trt_cuda_graph_enable": True})
print("TRT EP       ", bench([trt, "CUDAExecutionProvider"]))
print("CUDA EP+graph", bench([("CUDAExecutionProvider", {"enable_cuda_graph": "1"})]))
print("CUDA EP      ", bench(["CUDAExecutionProvider"]))
```

**Step 2, the parser.**
1. Unpack a current llama.cpp build (CUDA 12.4 and 13.4 zips) next to b11617.
2. Run llama-bench for each model and quant in the matrix: Qwen3.5-4B, 2B and 0.8B in Q4_0, Q4_K_M, IQ4_XS, Q6_K and Q8_0, plus Qwen3-4B-Instruct-2507 Q4_K_M as the pure-attention control.
3. Run the end-to-end check that matters: the 87-case `console/llm_cases.json` bench against llama-server, started once with the default `-cms` and once with `-cms 0`.
4. After each command, grep the server log for "forcing full prompt re-processing" and check that `timings.prompt_n` in each response counts only the new tokens.
5. Add speculative decoding with json_schema still on, checking JSON validity and acceptance: `--spec-type draft-mtp --spec-draft-n-max 2` (then 3) with an MTP GGUF, and separately `--spec-type ngram-mod`.

Keep `-np 1`, and watch `predicted_n` so a template change does not silently re-enable thinking.

```powershell
.\llama-bench.exe -m Qwen3.5-4B-Q4_K_M.gguf -ngl 99 -fa 0,1 -p 64,512,1536 -n 32 -r 5 -o md
.\llama-server.exe -m Qwen3.5-4B-Q4_K_M.gguf -ngl 99 -c 4096 -np 1 --host 127.0.0.1 --port 8081 --jinja -cms 0 --ctx-checkpoints 32
```

**Step 3, ASR.**
- **Parakeet v3:** time the int8 export on the CPU with 4 threads as the baseline, against the fp32/fp16 export on CUDA. The int8 ONNX is a CPU format.
- **Canary-1B-v2:** run it through onnx-asr on CUDA with the language forced to sk, cs or en (check its README for the model id and the language argument).
- **KInIT's Slovak fine-tune:** run it through NeMo.
- **faster-whisper `large-v3-turbo`** with `compute_type="int8_float16", language="sk"`, only if Canary disappoints.

Score each model per language on our own 100-200 recorded commands (at least 3 speakers), including the 2026-10-21 failures.

```powershell
.\sherpa-onnx-offline.exe --provider=cpu --num-threads=4 --encoder=encoder.int8.onnx --decoder=decoder.int8.onnx --joiner=joiner.int8.onnx --tokens=tokens.txt --model-type=nemo_transducer cmd_001.wav cmd_002.wav
.\sherpa-onnx-offline.exe --provider=cuda --encoder=encoder.onnx --decoder=decoder.onnx --joiner=joiner.onnx --tokens=tokens.txt --model-type=nemo_transducer cmd_001.wav cmd_002.wav
```

**Step 4, splat formats and viewer.**
1. Convert one scan's PLY to each candidate format.
2. Decode each back to PLY and score held-out PSNR at the saved poses.
3. In Chrome, A/B our `cloud.js` against a worker-sorted, 32-byte-packed variant and against Spark 2.2 on the same scan. Record fetch-to-first-frame time, median and p95 frame time over a 10 s scripted orbit at 1920x1080, and the main-thread sort time.

```powershell
npm install -g @playcanvas/splat-transform
splat-transform NAME.ply -H 0 NAME_sh0.sog
splat-transform NAME.ply -H 0 NAME_sh0.spz
splat-transform NAME.ply NAME.compressed.ply
splat-transform NAME_sh0.sog NAME_from_sog.ply
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --disable-gpu-vsync --disable-frame-rate-limit
```

**Pitfalls on the laptop:**
- The display is attached to this GPU, so the compositor adds load and WDDM adds jitter.
- Antivirus scanning of freshly written engines inflates cold-start times.
- Laptop GPUs drop clocks between requests, so measure cold-after-idle separately.
- Delete `trt_cache` after any ORT, TensorRT or driver change.

### hekate: FP8 for big ViTs, VLA serving over WiFi, faster splats

**Step 0, preflight before every session.** CLAUDE.md applies:
- ask before anything GPU-heavy when less than ~16 GB of VRAM is free;
- never stop Isaac Sim or a job we did not start;
- do not lock clocks or change driver settings;
- ask before any software, firewall or scheduled-task change, including installing WSL2, Docker or a new TensorRT;
- for longer multi-step work, message the Remote Control session "hekate".

Interleave configurations over several rounds, because other tenants dominate p99.

```powershell
ssh hekate "powershell -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\daniel\Desktop\Sim Arm 2026\tools\status.ps1'"
ssh hekate "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,clocks.sm,temperature.gpu --format=csv; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"
```

**Step 1, vision.** Use the laptop's trtexec commands and ORT harness, with `sm120` in the engine names. Compare:
- weakly typed FP16 on TensorRT 10.16, for parity with the robots;
- strongly typed FP16 on TensorRT 11.4, which has known Blackwell FP16 regressions in 11.x.

Depth models:
- Make FP8 Q/DQ variants of DA V2-B/L and DA3-B/L, calibrated on 100-500 real close looks. Fall back to BF16 if FP16 overflows.
- Score every precision with the duel metrics.

YOLO:
- Compare FP16 with INT8 Q/DQ, checking INT8 outputs numerically because of the 11.4.0 YOLOX issue on the 5090.

TRT-RTX:
- CPU-only AOT against on-device AOT;
- cold first inference with an empty runtime cache against a warm one;
- steady state with CUDA graphs on and off.

```powershell
python -m modelopt.onnx.quantization --onnx_path dav2_l.onnx --quantize_mode fp8 --calibration_method max --calibration_data_path close_looks_256.npy --output_path dav2_l_fp8.onnx
tensorrt_rtx --onnx=m_fp16.onnx --saveEngine=m.rtx.engine --skipInference
tensorrt_rtx --onnx=m_fp16.onnx --useGPU --saveEngine=m.rtx_gpu.engine --skipInference
tensorrt_rtx --loadEngine=m.rtx.engine --warmUp=0 --iterations=1
tensorrt_rtx --loadEngine=m.rtx.engine --warmUp=2000 --duration=20
tensorrt_rtx --loadEngine=m.rtx.engine --warmUp=2000 --duration=20 --rtxCudaGraphStrategy=disable
```

**Step 2, policies.** These need the user's go-ahead first.
- **GR00T N1.7:** Isaac-GR00T's deployment scripts are Linux-only, so run them under WSL2/Docker. Use our checkpoint and LeRobot set to produce NVIDIA's three-way table (eager, torch.compile, TensorRT full pipeline), and verify the TensorRT actions against PyTorch with a fixed noise tensor.
- **pi0.5:** LeRobot `pi05`, which already runs natively on Windows here, eager and with `--use_torch_compile`, using a `triton-windows` build matched to the installed PyTorch minor. Add openpi's `serve_policy.py` if WSL2 is available.
- **FlashRT:** optional FP8/FP4 runs, only as an upper bound.

```bash
python scripts/deployment/build_trt_pipeline.py --model-path <our ckpt> --dataset-path <our LeRobot set> \
  --embodiment-tag NEW_EMBODIMENT --steps export,build,verify,benchmark --warmup 10 --num-iterations 20
python gr00t/eval/run_gr00t_server.py --dataset-path <our LeRobot set> --embodiment-tag NEW_EMBODIMENT \
  --execution-horizon 8 --host 0.0.0.0 --port 5555      # model-free replay server, for pure network tests
```

**Step 3, serving over the lab WiFi.** From each robot, send 3,000 requests at the real 10 Hz cadence, not back to back, with the camera streams on. Compare protocols with an identical payload (2x JPEG 256x256 at q85, 1x 224x224, and raw `.npy` for GR00T's stock client):
- our stdlib HTTP, with and without keep-alive and TCP_NODELAY;
- GR00T's ZMQ replay server;
- openpi's websocket;
- LeRobot gRPC.

Timing:
- Time on the robot's monotonic clock: t0 frame captured, t1 request sent, t2 reply received, t3 first action sent.
- Network time is (t2 - t1) minus the server's own logged time, so no clock sync is needed.
- Report p50/p95/p99/max and the count above 100 ms, against a `ping -i 0.1 -c 3000` baseline.

Loop:
- Record the inference delay d per chunk and the joint jump at each chunk boundary.
- Compare sync, async (request when ~10 actions remain), and async with RTC (`execution_horizon` 8-10, EXP schedule), with an extra 100 and 200 ms of injected delay.
- WSL2's NAT needs port forwarding or mirrored networking plus a Windows firewall rule. All of these are changes to ask about first.

**Step 4, LLM (optional).** Native llama.cpp CUDA 13.4: pp and tg for Qwen3.5-4B/9B in Q4_K_M and in NVFP4. Converting a GGUF to NVFP4 still needs an open PR or ModelOpt, so check what exists before downloading.

**Step 5, splats.** Run the baseline (`splat\splat.bat SPLAT_DIR --steps 7000 --name base`), then each variant on two scans, a can and a box, with 3 seeds:
- 3000/5000/7000 steps;
- SH 0, 1 or 3;
- fused SSIM;
- SelectiveAdam;
- `packed`;
- default, absgrad or MCMC strategy with cap 150k/300k.

Report the time to reach the final PSNR minus 0.2 dB, the final held-out PSNR after alignment, depth error, Gaussian count and peak VRAM. Exclude the first-use CUDA JIT build and image loading. Repeat the baseline in a separate venv with gsplat main. Run Isaac Sim A/Bs of ParticleField against NuRec only with an explicit go-ahead.

```powershell
python -m threedgrut.export.scripts.transcode NAME.ply -o NAME_pf.usdz --format lightfield --half
python -m threedgrut.export.scripts.transcode NAME.ply -o NAME_nurec.usdz --format nurec
```

**Pitfalls on hekate:**
- `nvidia-smi` inside WSL lacks utilisation and process queries, so sample from Windows.
- CUDA green contexts can reserve SMs for a policy server, but they need CUDA 13.1+ and do not isolate memory ([TensorRT 11.4.0 notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.4.0.html)).
- WSL2 overhead estimates range from ~10 % to "roughly halves" a 27B model on the 5090 ([runaihome](https://runaihome.com/blog/qwen36-27b-nvfp4-blackwell-2x-speed-guide-2026/)), so measure it before relying on WSL2 for anything latency-critical.
- The FP8 calibration set must be real frames, not zeros.
- Coordinate conventions differ between splat formats; check orientation visually after every conversion.

### Dionysos: the Ampere reference and fallback policy server

**Step 0, preflight**, read-only. The SSH shell is cmd.exe, not PowerShell. It is a shared lab PC, so:
- never touch others' processes (our own overnight sim jobs also run there);
- do not change the driver model, driver version or clocks;
- ask before GPU-heavy runs when less than ~16 GB of the 20 GB is free.

```bat
nvidia-smi -q | findstr /C:"Driver Model" /C:"Driver Version" /C:"CUDA Version"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,clocks.sm --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

**Step 1, the Ampere reference rows.** Build YOLO26n-seg, DA V2-S/L, DA3-B and DINOv2-S as FP16, BF16 and INT8 Q/DQ engines (no FP8 on Ampere). Use TensorRT 10.16 or the CUDA 12 build of 11.4, which avoids depending on R615+ driver features. Run the same ORT harness for the EP comparison. Try TRT-RTX only if the driver model reads WDDM.

**Step 2, fallback policy server.** These rows tell us whether Dionysos can stand in when hekate's GPU is busy:
- GR00T N1.7 through the same TensorRT full pipeline at BF16;
- pi0.5 at BF16 with torch.compile;
- the same 3,000-request network test from a robot. The network path to Dionysos differs from hekate's lab LAN, so its RTT and jitter must be measured, not assumed.

**Step 3, LLM and splats.** For the LLM, run llama.cpp llama-bench in Q4_K_M and Q8_0. For splats, run only a NuRec USDZ load check: its Isaac Sim 5.0.0 does not read ParticleField, which needs 6.0.

**Pitfalls on Dionysos:**
- Commands written for PowerShell break in cmd.exe.
- A busy neighbour can OOM a TensorRT build's workspace, so record free VRAM at build time.
- p99 will reflect other users more than the runtime; report medians over interleaved rounds.

## Conclusion

For our hardware, the optimisation frontier has moved from number formats to plumbing. On all four GPU types, FP16 TensorRT for our small models is within a few tenths of a millisecond of anything newer. Meanwhile, mask post-processing, request handling, prompt caching and the training loop cost several milliseconds to several hundred. The 2026 toolchain churn is real: TensorRT 11's strong typing, the CUDA 13 move on Windows, JetPack 7 on Orin. It is cheapest to absorb by making the typed ONNX the contract and treating each engine as a per-machine build artefact. Seen that way, the JetPack 7.2 question stops being a speed decision. It becomes a decision about ROS 2, the vendor stacks and Edge-LLM, and it can wait for its triggers.

The headline hardware features of 2026 pay off on exactly one machine, hekate's RTX 5090 with FP8 and NVFP4, and only for large transformers. That is the role the VLA plan already gives it, so the new number formats and our architecture point the same way: small models on the robots, large ones on hekate. The weakest part of this picture is evidence. Nearly every Windows cell in the matrix rests on vendor guidance rather than measurement, so the four benchmark plans are as much a deliverable of this research as its recommendations.
