# Fastest export formats and inference runtimes on Jetson Orin NX 16 GB Super (state as of 2026-10-27)

Context: both robots are Orin NX 16 GB Super dev kits at MAXN_SUPER on JetPack 6.2 / L4T R36.4.x. They run TensorRT 10.7 (JetPack 6.2 itself ships 10.3; 10.7 is NVIDIA's separate L4T tarball, see Q1). Our measured numbers come from PLAN.md M5 ("The table, first fill (2026-10-12)", C:\Users\Mys\Desktop\RoboARM 2026\PLAN.md lines 633-668): YOLO26n-seg FP16 3.00 ms, the INT8 QAT engine 3.18 ms (100 of its 301 layers are reformats), YOLO26s-seg 5.37 ms, YOLOE-26s 5.36 ms, RF-DETR-Seg-N 8.00 ms, YOLO26n-depth 3.80 ms. End to end, the kit detector takes 8.3 ms, of which mask postprocess is 3.5 ms and inference 3.2 ms.

Research was desk-only (web search and fetch, 2026-10-27). Nothing was run on the robots or the lab PCs.

## Q1. Newest JetPack/L4T for Orin NX; versions; upgrade path and risks; newer TensorRT on JetPack 6.2 without reflashing?

### Takeaway
There are two current lines for Orin NX.
- **JetPack 6.2.3 / L4T 36.5.2** (12 Aug 2026): Ubuntu 22.04, CUDA 12.6, TensorRT 10.3, cuDNN 9.3, the same stack we have. It is an APT upgrade.
- **JetPack 7.2 / L4T 39.2** (2 Jun 2026) and **7.2.1 / L4T 39.2.1** (Aug 2026): Ubuntu 24.04, kernel 6.8, CUDA 13.2.1, cuDNN 9.20, TensorRT 10.16.2. It needs a reflash, every engine has to be rebuilt, ROS 2 Humble is lost, and Orin DLA was broken at launch.

On JetPack 6.x the newest TensorRT is 10.7, NVIDIA's L4T tarball, which is what we already run. TensorRT 10.16+ and 11.x reach Orin only through the SBSA/CUDA 13 stack of JetPack 7.x. No published source shows TensorRT 10.16 running on a JetPack 6 host.

### Cited Findings
- **NVIDIA's JetPack archive** lists:
  - Orin-only releases: 6.2 = L4T 36.4.3, 6.2.1 = 36.4.4, 6.2.2 = 36.5.0, 6.2.3 = 36.5.2.
  - Thor-only releases: 7.0 = 38.2/38.2.1, 7.1 = 38.4.
  - Releases for both Thor and the "Jetson Orin Family": 7.2 = 39.2, 7.2.1 = 39.2.1.
  - [JetPack Archive](https://developer.nvidia.com/embedded/jetpack-archive)
- **JetPack 6.2 components:** CUDA 12.6.10, TensorRT 10.3.0, cuDNN 9.3.0, VPI 3.2, DLA 3.1, DLFW 24.0 — [JetPack 6.2 release notes](https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-62/release-notes/index.html). NVIDIA's 6.2.2 page repeats CUDA 12.6 / TRT 10.3 / cuDNN 9.3 — [JetPack 6.2.2 SDK page](https://developer.nvidia.com/embedded/jetpack-sdk-622) (via search snippet).
- **JetPack 6.2.3 / Jetson Linux 36.5.2**, posted 12 Aug 2026:
  - It is a "minor upgrade over JetPack 6.2.2" with "Fixes for known issues and security vulnerabilities", kernel 5.15 and Ubuntu 22.04. It "packages the same libraries, components and supports the same SDKs as JetPack 6.2.2".
  - APT upgrade: set r36.5 in `nvidia-l4t-apt-source.list`, then `apt update`, `apt dist-upgrade`, and `apt install --fix-broken -o Dpkg::Options::="--force-overwrite"`.
  - [NVIDIA forum announcement](https://forums.developer.nvidia.com/t/jetpack-6-2-3-jetson-linux-36-5-2-is-now-live/379871)
- **JetPack 6.2.2 / L4T 36.5.0:** one user's upgrade from 6.2.1 (L4T 36.4.7) moved the kernel from 5.15.148 to 5.15.185 — [Hokwang Choi blog, Apr 2026](https://hokwangchoi.com/blog/vlm-benchmarks/)
- **JetPack 7.2 components:**
  - Jetson Linux 39.2 is dated 6/02/2026.
  - Ubuntu 24.04, kernel 6.8, CUDA 13.2.1, cuDNN 9.20.0, TensorRT 10.16.2, VPI 4.1.3, Container Toolkit 1.19, Holoscan 3.9.0.
  - "Isaac ROS: Coming soon".
  - The release adds "support for the Jetson Orin product family within JetPack 7 releases".
  - [JetPack 7.2 archive page](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2)
  - The announcement post is dated 2 Jun 2026 — [NVIDIA forum, JetPack 7.2 announcement](https://forums.developer.nvidia.com/t/jetpack-7-2-jetson-software-goes-agentic-with-jetson-linux-39-2/372059)
  - Conflict: the archive page says DeepStream 9.1, the forum post says DeepStream 8.0.
- **JetPack 7.2 install:**
  - Install methods are an ISO written with balenaEtcher, SDK Manager, or manual flashing; then `sudo apt install nvidia-jetpack`.
  - The Orin Nano dev kit has no SD-card image: it is flashed from the unified ISO on a USB stick.
  - There is no in-place upgrade path from JetPack 6 on the page.
  - [NVIDIA forum, JetPack 7.2 announcement](https://forums.developer.nvidia.com/t/jetpack-7-2-jetson-software-goes-agentic-with-jetson-linux-39-2/372059)
- **JetPack 7.2.1 / L4T 39.2.1** dated 12 Aug 2026 (search-result summary of a forum listing) — [search result referencing NVIDIA forums](https://forums.developer.nvidia.com/t/jetpack-7-2-jetson-software-goes-agentic-with-jetson-linux-39-2/372059). It is also listed in the [JetPack Archive](https://developer.nvidia.com/embedded/jetpack-archive).
- **JetPack 7.2 on Orin, other notes:** it adds MAXN_SUPER for AGX Orin 32 GB (200 -> 241 TOPS); MIG is not supported on Orin — [RidgeRun JetPack 7.2 components](https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin/JetPack_7.2/Getting_Started/Components) (via search snippet); [JetPack 7.2 archive page](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2)
- **Engines are not portable across the upgrade:** an engine built on JetPack 6.2 "should not be copied to JetPack 7.2", it must be rebuilt — [Seeed: Deploy TensorRT Edge-LLM on JetPack 7.2](https://wiki.seeedstudio.com/deploy_tensorrt_edge_llm_on_jetpack7.2/)
- **Third-party carriers lag JetPack 7.2.** A Seeed forum thread asks whether its J401 carrier will get JetPack 7.2 support (title only) — [Seeed forum](https://forum.seeedstudio.com/t/nvidia-has-officially-announced-jetpack-7-2-june-1-2026-any-plans-for-j401-agx-orin-32gb-support/295471). Connect Tech announced JetPack 7.2 Yocto BSPs for its AGX Orin / Orin NX carriers — [Connect Tech](https://connecttech.com/jetpack-7-2-yocto/)
- **Project decision already recorded:** "Stay on JetPack 6.2 and ROS 2 Humble (agreed). JetPack 7.2 now supports Orin, but it moves to Ubuntu 24.04 and CUDA 13, where neither vendor's Humble stack runs" — C:\Users\Mys\Desktop\RoboARM 2026\PLAN.md line 156 (internal)
- **TensorRT 10.7 on JetPack 6.2 (our setup):**
  - jetson-containers PR #1750 (opened 11 Oct 2026, merged into `dev` 12 Oct 2026) adds `TENSORRT_VERSION=10.7` for L4T r36. It pulls `TensorRT-10.7.0.23.l4t.aarch64-gnu.cuda-12.6.tar.gz` and keeps cuDNN 9.3.0.
  - It was tested on an Orin Nano Super, L4T R36.4.3.
  - The default stays 10.3. The Python wheel install was broken until PR #1751.
  - [jetson-containers PR #1750](https://github.com/dusty-nv/jetson-containers/pull/1750)
- **NVIDIA's own TRT 10->11 Jetson migration guide:**
  - Jetson Orin on JetPack 6.x: "Upgrade to JetPack 7.x is required before a future TensorRT 11.x JetPack release."
  - JetPack 7.x: "Remain on a TensorRT 10.x release supported by the installed JetPack version."
  - On DLA: "Use TensorRT 10.7 if DLA is required."
  - [TensorRT 11.4.0 migration guide, Jetson](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/api/migration/tensorrt-10x-to-11x-jetson.html) (last updated 5 Oct 2026)
- **TensorRT 10.16.1 release notes:** "Support has been added in this TensorRT release for Orin iGPU while using the ARM SBSA build... This release will be bundled in an upcoming JetPack 7.x release", with a standalone early-access download. "Orin DLA is not supported at this time." — [TensorRT 10.16.1 release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.16.1.html) (page updated 23 May 2026)
- **NGC base images:** the newest `l4t-jetpack` tag listed is r36.4.0 (created 11/07/2024, about 10 GB, TensorRT 10). NVIDIA stopped publishing `l4t-base` after JetPack 6.1 and points to `l4t-cuda` instead — [NVIDIA forum](https://forums.developer.nvidia.com/t/l4t-base-36-4-x-image-missing-l4t-base-36-2-0-image-misconfigured-and-no-tensorflow-wheels-for-jp-6-1/325925); [NGC l4t-jetpack tags](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-jetpack/tags)
- **jetson-containers tags pair the r36.4 host with a CUDA 12.8 / Ubuntu 24.04 userspace**, e.g. `dustynv/llama_cpp:b5283-r36.4-cu128-24.04` and `dustynv/vllm:0.8.6-r36.4-cu128-24.04` — [ProventusNova LLM guide](https://proventusnova.com/blog/llm-inference-jetson-orin-llamacpp-ollama) (via search snippet)
- **Ultralytics' `latest-nvidia-arm64` image** (NVIDIA PyTorch 26.08, CUDA 13.4, TensorRT 11.2) is for PyTorch only, because "TensorRT 11.2.1 does not support JetPack". Jetson TensorRT exports should use the native TensorRT 10.x — [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson)

### Inferences
- **Newest TensorRT on our hosts:** without reflashing, the newest TensorRT we can run is 10.7, and we already have it. A JetPack 7.2 container on a JetPack 6.2 host would need the r39 SBSA driver / CUDA 13 under it. No source shows this working, and NVIDIA's docs tie TRT 10.16 on Orin to "an upcoming JetPack 7.x". The containers on r36.4 hosts top out at CUDA 12.8 userspace (forward compatibility), not CUDA 13. Treat "newer TRT on JetPack 6.2 via containers" as not available.
- **JetPack 6.2.3:** going 6.2 -> 6.2.3 is a security/bugfix APT upgrade with the same CUDA/TRT and no expected speed change. It is a low-risk housekeeping step.
- **JetPack 7.2.x would buy:**
  - TRT 10.16.2;
  - the official TensorRT Edge-LLM and GR00T N1.7 path (both list JetPack 7.2 for Orin; see Q5);
  - the route to TRT 11.
- **JetPack 7.2.x would cost:**
  - a reflash on Yahboom carriers (no source found on Yahboom's JetPack 7.2 support);
  - Ubuntu 24.04, so ROS 2 Humble -> Jazzy and both vendors' stacks (PLAN.md);
  - the DLA until JetPack 7.2.2 (Q3);
  - a rebuild of every engine.
- **Speed gain:** for our CNN engines there is no published evidence that 10.16 is faster than 10.7 on Orin. Expect a small change either way; measure before deciding (Benchmark plan).

### Gaps
- No primary NVIDIA page gives a JetPack 7.2.1 component list or date. The date 12 Aug 2026 comes from a search summary.
- No source on whether Yahboom ships or supports JetPack 7.2 images for the X3 Plus / M3 Pro carrier boards.
- No source confirms that ROS 2 Humble cannot run on JetPack 7.2. That claim is from our PLAN.md (Humble targets Ubuntu 22.04). A Humble container on a 24.04 host is a possible workaround that I did not research.
- No published A/B latency for the same CNN or ViT on Orin under TRT 10.3 / 10.7 vs 10.16.

## Q2. What changed in TensorRT 10.8 -> 10.16 and 11.x that matters on Orin

### Takeaway
The changes that matter on Orin are:
- **10.13.2** fixed a ViT regression of up to 10 % on Orin, relative to 10.9.
- **10.16** brought Orin iGPU into the SBSA build, which is what JetPack 7.2 ships.
- **TRT 11.x**, newest 11.4.0, removes weak typing, the precision flags and implicit INT8 (calibrators). It requires CUDA 13.2u2+ and does not yet support JetPack at all.

Orin sm_87 has no FP8 or FP4 path (confirmed by several NVIDIA sources), so the large FP8/NVFP4 features of 10.8-11.x do not apply. On Orin the precision menu stays FP16 / INT8 / INT4 (weight-only, for LLMs).

### Cited Findings
- **10.13.2:** "Fixed an up to 10% performance regression compared to TensorRT 10.9 for ViT on Orin GPUs." 10.13.2 also removed CUDA 11.x support — [TensorRT 10.13.2 release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.13.2.html)
- **10.4.0:** "For some Transformer models, including ViT, Swin-Transformer, and DETR, there was a performance drop in INT8 precision (including both explicit and implicit quantization) compared to FP16 precision. This issue has been fixed." — [TensorRT 10.4.0 release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.4.0.html) (via search snippet)
- **10.7.0:**
  - Its MHA work is FP8 MHA fusions on Ada (larger sequence lengths and head sizes) and better ViT performance with FP16 MHA plus FP8 GEMMs. Both are Ada features, not Orin ones.
  - Known Orin issue: BERT-Large INT8-QDQ engine size fluctuates by up to 60 MB from unstable tactic selection.
  - Refit-enabled engines have known performance gaps.
  - [TensorRT 10.7.0 release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.7.0.html)
- **10.16.1:**
  - Orin iGPU support in the ARM SBSA build, and no Orin DLA.
  - Strongly typed networks and explicit quantization are "recommended for migration ahead of TensorRT 11.0", which removes weakly typed APIs, implicit INT8, IPluginV2 and TREX.
  - FP8 convolutions are listed only for SM89/90/120/121.
  - `kREFIT` / `kREFIT_IDENTICAL` engines regress when convolutions sit in a branch or loop at FP16/INT8.
  - Static libraries are deprecated and will be removed in 11.0.
  - [TensorRT 10.16.1 release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.16.1.html)
- **TensorRT 11.x:**
  - "TensorRT 11.x removes all precision-enabling builder flags" (use ModelOpt AutoCast plus strongly typed networks instead).
  - "The `IInt8Calibrator` class and all subclasses have been removed" (use explicit Q/DQ via ModelOpt).
  - The tactic sources `kCUBLAS`, `kCUBLAS_LT` and `kCUDNN` are removed.
  - "TensorRT 11.x requires CUDA 13.2 Update 2 or later."
  - Engines built with 10.x run on the 11.x runtime; 8.x/9.x engines do not.
  - "NVIDIA JetPack is not supported in TensorRT 11.4.0."
  - [TensorRT 11.4.0 migration guide, Jetson](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/api/migration/tensorrt-10x-to-11x-jetson.html)
- **The TensorRT support matrix** (latest = 11.4.0, page updated 5 Oct 2026):
  - Footnote: "TensorRT 11.2.1 and earlier support only Orin iGPU."
  - DLA in 11.4.0 "is not supported for Linux SBSA, NVIDIA JetPack, or NVIDIA DriveOS deployments" (DLA beta only on Windows on ARM).
  - It supports compute capability 7.5+.
  - Wheels cover Python 3.10-3.14.
  - [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
- **FP8/FP4 on Orin:**
  - "Jetson Orin does not run FP8 or FP4 model engines"; Orin is "FP16, INT8, INT4 only" — [TensorRT Edge-LLM support matrix](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/support-matrix.html)
  - "FP8 and NVFP4 not available on Orin (sm_87)" — [Jetson AI Lab Edge-LLM tutorial](https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/)
  - NVFP4 requires SM100+ — [Edge-LLM performance page](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/performance/performance-benchmarks.html)
- **Ultralytics:** TensorRT 10.3.0 on JetPack 6 has a known issue that blocks INT8 engines with `nms=False` or `nms=True`; the fix is 10.7.0 or newer. On TensorRT 11+, Ultralytics does INT8 with ModelOpt explicit quantization — [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson); [Ultralytics TensorRT integration](https://docs.ultralytics.com/integrations/tensorrt) (via search snippet)

### Inferences
- **Our ViTs:** on TRT 10.7 they (DA V2 S, DA3, DINOv2, RF-DETR's backbone) sit before the 10.9 -> 10.13 regression window. Upgrading to 10.16 is not a known ViT speed-up on Orin, only "regression fixed". There is no published FP16 MHA fusion change for sm_87 in 10.8-10.16.
- **Prepare for TRT 11 now** even on 10.7: build with `--stronglyTyped` from ONNX that already carries its precision (ModelOpt AutoCast to FP16, explicit Q/DQ for INT8), and stop relying on `--fp16 --int8` flags and calibrators. That keeps the export path valid for TRT 11 on JetPack 7.x.
- **Refit:** we should not build engines refittable "for convenience". 10.16 documents that refit regresses conv engines.
- **Weight streaming:** it targets GEMM-heavy transformers that do not fit in memory (10.7 notes). It is irrelevant to our small models on a 16 GB unified-memory board.

### Gaps
- I found no release-note entry for 10.8-10.16 that names Myelin fixes for Squeeze/Unsqueeze or "mod_t" failures like the one our ACT export hit. Whether 10.16 builds the un-slimmed ACT ONNX is unknown.
- No source quantifies builder speed or CUDA-graph changes on Orin between 10.7 and 10.16.
- I did not read 10.8-10.15 release notes one by one (only 10.4, 10.7, 10.13.2, 10.15.1 headlines, 10.16.1). Smaller Orin items may be missing.

## Q3. DLA on Orin NX: layers, precisions, TRT status, real gains

### Takeaway
- **Hardware:** the Orin NX 16 GB has 2 DLA cores at 614 MHz, running FP16 or INT8 (INT8 is the useful one), CNN layers only.
- **Software:** on JetPack 6.x DLA works with TRT 10.3 and 10.7, and 10.7 is officially "the TRT to use if DLA is required". JetPack 7.2 / 7.2.1 cannot build DLA engines on Orin. NVIDIA says JetPack 7.2.2 (expected "by end of October" 2026, not seen yet) restores it.
- **Speed:** DLA is slower per frame than the GPU (2-5x, from third-party and forum numbers), and GPU+DLA concurrency slows both. It buys throughput and power, not latency.

### Cited Findings
- **Ultralytics:**
  - Orin NX 16 GB has 2 DLA cores at 614 MHz; Orin NX 8 GB has 1; AGX Orin has 2 at 1.6 GHz.
  - DLA "only works with FP16 or INT8".
  - Unsupported layers fall back to the GPU, so DLA targets throughput and energy, not latency.
  - "TensorRT 10.7 is the last release with DLA support."
  - Export: `model.export(format="engine", device="dla:0", quantize=16)`.
  - [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson)
- **JetPack 7.2 DLA failure (forum thread):**
  - Report of 18 Jun 2026: on Orin NX 16 GB and AGX Orin with JetPack 7.2 / TensorRT 10.16.2, DLA builds fail with "Cannot create DLA engine, 0 not available". The same commands worked on JetPack 6.2 / TRT 10.3.
  - NVIDIA moderator, 22 Jun: JetPack 7.2 uses an SBSA unified driver, so DLA is not available.
  - 10 Sep: TensorRT with DLA released as a standalone package. The moderator suggested JetPack 7.2.1 + TRT 11.3.0 GA, with conflicting CUDA 13.3/13.4 references.
  - 30 Sep: a user on Orin NX with 7.2.1 + TRT 11 still got DLA compile errors (code 8 / code 2).
  - 8 Oct: NVIDIA said "DLA would be supported in JetPack 7.2.2, expected to publish by the end of October".
  - [NVIDIA forum, DLA on JetPack 7.2](https://forums.developer.nvidia.com/t/dla-runtime-unavailable-on-jetpack-7-2-l4t-r39-2-cannot-create-dla-engine/373788)
- **Conflict on DLA in TRT 11:** the moderator's "TRT 11.3.0 GA with DLA" contradicts the TRT docs ("TensorRT 11.2.1 and earlier support only Orin iGPU"; 11.4.0 DLA unsupported on JetPack/SBSA). ModelOpt's README says strongly typed DLA Q/DQ needs 11.4+ ("a pre-release 11.4.1.22 build"). DLA support in the SBSA/JetPack 7 stack is still moving — [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html); [ModelOpt ONNX PTQ README](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/onnx_ptq/README.md)
- **A community reply recommends JetPack 6.1/6.2 for DLA** for now — [NVIDIA forum, JP 7.2 w/ Isaac ROS and DLAs](https://forums.developer.nvidia.com/t/jp-7-2-w-isaac-ros-and-dlas/374021) (via search snippet)
- **ProventusNova (third party, 18 May 2026, no cited sources):**
  - DLA runs FP16/INT8 (no FP32), fixed shapes only.
  - Not supported: custom plugins, dynamic shapes, LSTM/GRU, transformer attention (as of JetPack 6.2), most element-wise ops beyond add/mul.
  - Classic CNNs (YOLO, ResNet, MobileNet) are largely compatible.
  - YOLOv8m INT8 on AGX Orin: GPU < 5 ms vs DLA 15-25 ms.
  - Power: DLA 2-5 W per core vs GPU 10-25 W.
  - Its figures are partly illustrative and internally inconsistent on fallback thresholds.
  - [ProventusNova: TensorRT vs DLA](https://proventusnova.com/blog/tensorrt-vs-dla-jetson-orin/)
- **Forum measurements:**
  - YOLOv8s: DLA 44 FPS vs GPU 105 FPS, with higher latency — [forum 302939](https://forums.developer.nvidia.com/t/dla-performance/302939)
  - FP16 on DLA about 10x slower than the GPU for one model — [forum 240064](https://forums.developer.nvidia.com/t/getting-less-throughput-while-enabling-dlas-on-jetson-agx-orin/240064)
  - Running the GPU and DLA together raises the time of both (shared scheduling and memory) — [forum 245332](https://forums.developer.nvidia.com/t/when-gpu-and-dla-are-used-at-the-same-time-the-time-consumption-increases-with-each-other/245332)
  - DLA gaps widen under a concurrent GPU load (DriveOS) — [forum 343300](https://forums.developer.nvidia.com/t/dla-inference-latency-issue-on-orin-platform/343300)
  - For both cores, run two processes, `--useDLACore=0` and `--useDLACore=1` — [forum 280604](https://forums.developer.nvidia.com/t/the-throughput-is-too-slow-in-nvidia-jetson-agx-orin-dla/280604)
- **INT8 is DLA's native rate:** NVIDIA's cuDLA YOLOv5 QAT blog says DLA is optimized for INT8 convolution, well above its FP16 rate (search summary: "roughly 15x" the dense FP16 rate; not verified on the page) — [NVIDIA blog: YOLOv5 on Orin with cuDLA QAT](https://developer.nvidia.com/blog/deploying-yolov5-on-nvidia-jetson-orin-with-cudla-quantization-aware-training-to-inference/)
- **ModelOpt DLA flow:** `--target_dla` for INT8 Q/DQ export for DLA. On TensorRT 10.7 and earlier it needs the separate Q/DQ Translator plus a calibration cache matching the TRT version. Releases after 10.7 through 11.3 are not supported for this flow — [ModelOpt ONNX PTQ README](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/onnx_ptq/README.md)
- **10.16.1 known issue:** broadcasting element-wise layers on DLA with GPU fallback can lose accuracy when inputs use `kDLA_LINEAR` — [TensorRT 10.16.1 release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.16.1.html)

### Inferences
- **Latency:** for our 3 ms YOLO26n-seg the DLA will not cut latency; expect several times slower per frame. The mask head's proto conv, which has no INT8 GPU kernel on 10.7, and the NMS-free head are likely fallback points.
- **Where DLA could help:** a second, always-on CNN (for example YOLO26n-depth or a coarse detector) runs on the DLA at INT8 while the GPU serves the depth ViT or the ACT policy. Even then, check the GPU-side slowdown from shared DRAM bandwidth.
- **Our setup already fits:** JetPack 6.2 + TRT 10.7 is exactly NVIDIA's recommended DLA stack. Moving to JetPack 7.2 now would lose DLA until 7.2.2 is out and verified.

### Gaps
- No published YOLO26 (NMS-free) DLA layer-support or latency numbers on Orin NX.
- JetPack 7.2.2 has not appeared in searches as of 2026-10-27. Release status is unconfirmed.
- The DLA TOPS split for Orin NX 16 GB comes only from third parties (ProventusNova: 2 x 3.5 TOPS). There is no NVIDIA datasheet citation here.

## Q4. INT8 on Orin: best practice; why QAT engines get Reformat layers; measured INT8 vs FP16

### Takeaway
Current best practice:
1. Use explicit quantization (Q/DQ in ONNX) made by NVIDIA ModelOpt.
2. Build strongly typed.
3. Let ModelOpt 0.47's `--autotune` place Q/DQ by measured TensorRT latency. It keeps INT8 only where it beats FP16 by at least 1.02x, otherwise it writes back the no-Q/DQ model.

Reformat layers and slower INT8 engines come from Q/DQ nodes placed where TensorRT cannot fuse them, so INT8 and FP16 alternate around each conv or element-wise op. Measured gains on Orin are small for already-fast YOLO-class models (Ultralytics: YOLO26 FP16 4.13 -> INT8 3.49 ms on Orin NX, with a mAP drop). ViT-class models often regress: one Orin Nano ViT-S/DPT case was 2.7x slower; a DEIMv2 case on Orin NX went 580 -> 380 QPS.

### Cited Findings
- **Ultralytics Orin NX 16 GB, Ultralytics 8.4.33** (model not labelled on the page; the 5.3 MB .pt suggests YOLO26n):

  | Format | ms | mAP50-95 |
  |---|---|---|
  | TensorRT FP32 | 7.01 | 0.4770 |
  | TensorRT FP16 | 4.13 | 0.4789 |
  | TensorRT INT8 | 3.49 | 0.4489 |
  | PyTorch | 13.90 | |
  | ONNX | 14.18 | |
  | TorchScript | 11.60 | |

  [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson)
- **ModelOpt 0.47.0** (23 Sep; 0.48.0rc1 on 17 Oct; dates shown without a year, presumably 2026) adds Autotune to ONNX quantization. It "benchmarks Q/DQ placements in the requested runtime precision and keeps INT8/FP8 Q/DQ only if it meets a 1.02x TensorRT speedup by default", otherwise it saves the high-precision model without Q/DQ — [Model-Optimizer releases](https://github.com/NVIDIA/TensorRT-Model-Optimizer/releases)
- **ModelOpt 0.44.0** excludes small MatMul/Gemm from INT8, because small GEMMs "cannot efficiently use INT8/FP8 Tensor Cores and the added Q/DQ layers cause perf regressions in TensorRT" — [Model Optimizer changelog](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html) (via search snippet)
- **ModelOpt ONNX PTQ CLI:**
  - Command: `python -m modelopt.onnx.quantization --onnx_path=... --quantize_mode=<fp8|int8|int4> --calibration_data_path=calib.npy --calibration_method=<max|entropy|awq_clip|rtn_dq> --output_path=...`
  - Flags: `--autotune=<quick,default,extensive>`, `--calibrate_per_node`, `--trt_plugins`, `--target_dla`.
  - Use "at least 500" calibration samples for CNN and ViT models. INT8 needs opset 19+ (upgraded automatically).
  - Evaluation builds with `--engine_precision=stronglyTyped`.
  - [ModelOpt ONNX PTQ README](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/onnx_ptq/README.md)
- **ViT INT8 regression:** ModelOpt INT8 on Orin Nano 4 GB (ViT-S + DPT) ran 2.7x slower than FP16. Even plain trtexec `--fp16 --int8` gave only about 5 % over FP16 — [NVIDIA forum 357835](https://forums.developer.nvidia.com/t/tensorrt-model-optimizer-int8-quantization-causes-2-7x-performance-regression-on-jetson-orin-nano-4gb-vit-s-dpt-architecture/357835) (via search snippet)
- **DEIMv2 on Orin NX** (JetPack 6.2, TRT 10.7, ModelOpt `modelopt.torch.quantization` -> ONNX opset 17 -> `trtexec --stronglyTyped`): FP16/BF16 580 QPS, INT8 380 QPS. The only reply was an auto-generated LLM answer pointing at Q/DQ placement. No resolution — [NVIDIA forum 355549](https://forums.developer.nvidia.com/t/worse-performance-after-quantization-on-tensorrt/355549)
- **TensorRT explicit-quantization guidance:** quantize layers that do not commute with Q/DQ, and give non-weighted layers with INT8 inputs INT8 outputs too (quantize both inputs and outputs) — [TensorRT explicit quantization docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html) (via search snippet)
- **TRT 11** removes implicit INT8 and calibrators, so explicit Q/DQ is the only INT8 path going forward — [TensorRT 11.4.0 migration guide, Jetson](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/api/migration/tensorrt-10x-to-11x-jetson.html)
- **Ultralytics:** INT8 calibration "should use the target device for best results" — [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson)

### Inferences
- **Our QAT engine:** 100 reformats in 301 layers is the textbook sign of unfused Q/DQ. Ultralytics' `quantize=8` QAT export likely puts Q/DQ on conv inputs but not on the residual/concat/SiLU paths. TensorRT then converts between INT8 and FP16 at every boundary.
- **Fix to try:** re-quantize the FP16 ONNX with ModelOpt `--quantize_mode=int8 --calibration_method=entropy --autotune=default` on the Orin itself (autotune needs the target's TensorRT timings). Exclude the mask-proto conv that has no INT8 kernel, via `--nodes_to_exclude` / `--op_types_to_exclude`; these exact flag names were not on the README page I read, so check them. Then compare with `--dumpLayerInfo`.
- **Ceiling:** the possible gain is about 0.5 ms (Ultralytics' 4.13 -> 3.49 ms, 15 %) on a 3 ms engine. End to end, mask postprocess (3.5 ms) is the bigger target than INT8.
- **ViTs:** keep the depth models, DINOv2 and RF-DETR at FP16 on Orin. The evidence says INT8 is flat or slower for ViT-class models there.

### Gaps
- No published Orin numbers for YOLO26 INT8 QAT vs ModelOpt-autotuned PTQ.
- I did not verify the ModelOpt exclusion flag names or `--high_precision_dtype` on the fetched README.

## Q5. Alternative runtimes on Orin, and which is fastest per model class

### Takeaway
- **CNNs and ViTs:** native TensorRT engines built with trtexec on the device remain the fastest path. In Ultralytics' Orin NX table, TensorRT FP16 is about 3.4x ONNX Runtime and PyTorch.
- **Small LLMs and VLMs:** TensorRT Edge-LLM (INT4 AWQ) is the fastest published runtime. Orin NX 16 GB: Qwen3.5-0.8B 95.8 tok/s, 2B 47.9 tok/s, 4B 27.0 tok/s. On an Orin Nano it beat vLLM slightly and llama.cpp clearly (60 / 56 / 38 tok/s). Its official Orin support is on JetPack 7.2, although one community build ran it on JetPack 6.2.x with TRT 10.3.
- **ONNX Runtime with the TensorRT EP:** a convenience path, not a faster one.
- **Torch-TensorRT:** its Jetson docs cover JetPack 6.2 only.
- I found no Orin benchmarks for AOTInductor, ExecuTorch or TVM.

### Cited Findings
- **Ultralytics Orin NX 16 GB:** TensorRT FP16 4.13 ms vs ONNX 14.18, PyTorch 13.90, TorchScript 11.60, NCNN 34.31, OpenVINO 40.19, MNN 48.55, TF Lite 254.43 ms — [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson)
- **Ultralytics' onnxruntime-gpu wheels for Jetson:** 1.23.0 (Python 3.10, JetPack 6), 1.20.0 (Python 3.10, JetPack 6), 1.24.0 (Python 3.12, JetPack 7.2) — [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson)
- **Jetson AI Lab index:** it moved to `pypi.jetson-ai-lab.io/jp6/cu126` — [NVIDIA forum 344745](https://forums.developer.nvidia.com/t/getting-cuda-acceleration-for-onnx-on-jetson-orin-nano-with-jetpack-6-2/344745). One user's wheel from that index showed only CPU and Azure providers, with no TensorRT EP — [NVIDIA forum 368180](https://forums.developer.nvidia.com/t/onnxruntime-installation-on-jetson-agx-orin-developer-kit/368180)
- **Community ORT wheels:** v1.23.0 for JetPack 6.1-6.2.1 (Python 3.10, CUDA 12.6, TRT 10.3) — [guyin24/onnxruntime-gpu-for-jetson](https://github.com/guyin24/onnxruntime-gpu-for-jetson)
- **Upstream ONNX Runtime:**
  - Latest is v1.31.1 (26 Oct; v1.31.0 on 9 Oct; v1.30.0 on 10 Sep; dates without a year on GitHub, presumably 2026). The notes do not mention Jetson/L4T builds, and CUDA moved to a "CUDA Plugin EP" — [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)
  - 1.24.x: TensorRT EP upgraded to TRT 10.14, CUDA 13.0 support in the CUDA EP, "CUDA Graph enabled by default", Python 3.10 wheels dropped — [newreleases.io ORT 1.24.1](https://newreleases.io/project/pypi/onnxruntime/release/1.24.1) (aggregator; not checked against GitHub)
- **ORT TensorRT/CUDA EPs were impractical on an Orin Nano** in an ORB-SLAM3 GPU port: session setup stalled because per-subgraph engine builds run eagerly on the slow CPU — [Jetson-ORB-SLAM3 arXiv 2608.17874](https://arxiv.org/pdf/2608.17874) (via search snippet)
- **Torch-TensorRT's JetPack page** (docs v2.12.0) covers only JetPack 6.2 (CUDA 12.6, TRT 10.3, cuDNN 9.3). Build with `python setup.py bdist_wheel --jetpack` (example torch 2.8.0); torch/torchvision wheels from `pypi.jetson-ai-lab.io/jp6/cu126` — [Torch-TensorRT in JetPack](https://docs.pytorch.org/TensorRT/v2.12.0/getting_started/jetpack.html)
- **Eager PyTorch on Orin Nano Super** gains almost nothing in latency from FP16 (FP32 about 28.9-29.2 ms) — [arXiv 2609.13551](https://arxiv.org/pdf/2609.13551) (via search snippet)
- **Python API vs trtexec latency differ** on AGX Orin; the measurement method matters — [NVIDIA forum 258856](https://forums.developer.nvidia.com/t/performance-discrepancy-python-api-vs-trtexec-on-jetson-agx-orin-board/258856) (via search snippet)
- **TensorRT Edge-LLM support matrix** (updated 29 Sep 2026):
  - Jetson Orin: JetPack 7.2, CUDA 13.2, "FP16, INT8, INT4 only". JetPack 6.2 is not listed.
  - Wheels: CPython 3.10-3.12, aarch64.
  - [Edge-LLM support matrix](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/support-matrix.html)
  - Seeed's v0.9.1 guide says "JetPack 6.2+ remains compatible for FP16, INT8, and INT4 workflows" — [Seeed Edge-LLM guide](https://wiki.seeedstudio.com/deploy_tensorrt_edge_llm_on_jetpack7.2/). This conflicts with the official matrix.
- **Edge-LLM v0.10.0 benchmarks, Orin NX 16 GB** (JetPack 7.2, CUDA 13.2, TRT 10.16; batch 1; prompt 57-62 tokens; INT4 AWQ):

  | Model | Prefill | Decode | GPU mem |
  |---|---|---|---|
  | Qwen3.5-0.8B | 35.5 ms | 95.8 tok/s | 2.2 GB |
  | Qwen3.5-2B | 56.2 ms | 47.9 tok/s | 3.7 GB |
  | Qwen3.5-4B | 120.1 ms | 27.0 tok/s | 5.4 GB |
  | Qwen3-0.6B | | 124.9 tok/s | |
  | Qwen3-1.7B | | 59.0 tok/s | |
  | Qwen3-4B | | 32.2 tok/s | |
  | Qwen3-8B | | 18.4 tok/s | |

  For comparison, AGX Orin 64 GB: Qwen3.5-2B 83.3 tok/s, Qwen3.5-4B 47.2 tok/s — [Edge-LLM performance benchmarks](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/performance/performance-benchmarks.html)
- **Edge-LLM, other details:**
  - Model families: Qwen3, Qwen3.5/3.6, Qwen3-VL, InternVL3/3.5, Phi-4-MM, Nemotron-Nano, Alpamayo R1 (VLA).
  - Precisions on Orin: FP16 and INT4 AWQ.
  - Build: the C++ runtime is built from source on the device, and ONNX export runs on an x86 or Thor host (24 GB+ VRAM recommended).
  - [Jetson AI Lab Edge-LLM tutorial](https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/)
- **Three runtimes on an Orin Nano 8 GB Super** (JetPack 6.2.1 -> 6.2.2, Cosmos-Reason2-2B, a Qwen3-VL-2B architecture, W4A16):

  | Runtime | Decode | Text TTFT | Image TTFT | Peak memory |
  |---|---|---|---|---|
  | TRT Edge-LLM (built against TRT 10.3) | 60 tok/s | 29 ms | 420 ms | 4.3 GB |
  | vLLM | 56 tok/s | 61 ms | 75 ms | 6.9 GB |
  | llama.cpp Q4_K_M | 38 tok/s | 33 ms | 114 ms | not reported |

  Edge-LLM needed a CMA kernel change and an ONNX LM-head split — [Hokwang Choi, Apr 2026](https://hokwangchoi.com/blog/vlm-benchmarks/)
- **Older llama.cpp numbers on Orin NX 16 GB:** Llama 3.1 8B 18 tok/s, Qwen 2.5 7B 20 tok/s (setup unspecified) — [ProventusNova LLM guide](https://proventusnova.com/blog/llm-inference-jetson-orin-llamacpp-ollama) (via search snippet)
- **GR00T N1.7 (GA):**
  - Orin is "tested with JetPack 7.2" (CUDA 13.2, Python 3.12). Full-pipeline ONNX/TensorRT export is supported.
  - Inference needs a 16 GB+ GPU.
  - It runs at about 28 Hz on AGX Thor. There are no Orin or Orin NX numbers.
  - [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- **Platform SDKs:** JetPack 7.2 ships DeepStream 9.1 (forum post: 8.0) and Holoscan 3.9.0 — [JetPack 7.2 archive page](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2). gst-nvinfer DLA options are "Currently work in progress" on 7.2 (per a forum user citing the DeepStream docs) — [NVIDIA forum, DLA on JetPack 7.2](https://forums.developer.nvidia.com/t/dla-runtime-unavailable-on-jetpack-7-2-l4t-r39-2-cannot-create-dla-engine/373788)

### Inferences
- **CNNs** (YOLO26n/s-seg, YOLOE baked, YOLO26n-depth): trtexec- or Ultralytics-built TRT FP16 engines, as we do now. Alternative runtimes only add overhead.
- **ViTs** (DA V2 S, DA3, DINOv2, RF-DETR): TRT FP16 from ONNX. Gains lie in the export (static shapes, onnxslim, fewer views or smaller inputs), not in another runtime.
- **ACT policy** (5.9 ms): TRT FP16 after onnxslim, as now.
- **Voice LLM on the robot** (if moved off the laptop): Qwen3.5-2B INT4 at about 48 tok/s and 56 ms prefill on Orin NX via Edge-LLM is enough for single-function JSON calls. On JetPack 6.2, llama.cpp from jetson-containers is the simple path at about 35 % fewer tok/s (scaled from the Orin Nano comparison, an inference). Edge-LLM on JetPack 6.2 is possible but unsupported.
- **GR00T N1.7:** effectively requires JetPack 7.2, and its 16 GB+ requirement versus Orin NX's 16 GB shared with the CPU makes it marginal on our robots.
- **DeepStream / Holoscan:** they add pipeline value (zero-copy camera -> engine), not kernel speed. Not worth adopting for our ROS 2 / Python stack just for speed.

### Gaps
- No Orin benchmarks found for torch.compile / AOTInductor, ExecuTorch (CUDA backend), TVM / MLC on current versions, or Torch-TensorRT vs trtexec on the same model.
- No Edge-LLM numbers for Qwen3.5 on Orin NX under JetPack 6.2 (only the Orin Nano community run).
- No measured SmolVLA or GR00T latency on any Orin. No Parakeet-on-Orin runtime comparison.
- I could not confirm a jp6/cu126 onnxruntime-gpu wheel that includes the TensorRT EP on the Jetson AI Lab index.

## Q6. Power/clock settings and memory bandwidth effects

### Takeaway
- **Modes:** Super mode on Orin NX 16 GB (JetPack 6.2+) adds the 40 W and MAXN_SUPER modes and up to 70 % more AI TOPS (157 TOPS by vendor figures, GPU 918 -> 1173 MHz). MAXN_SUPER is uncapped and throttles clocks if power exceeds the TDP.
- **Clocks:** `nvpmodel -m 0` plus `jetson_clocks` pin the clocks. Our M5 table was measured without `jetson_clocks`, so its numbers carry DVFS ramp effects.
- **Bandwidth:** decode-bound LLMs are memory-bandwidth-bound at batch 1. I found no NVIDIA source for the Orin NX 16 GB bandwidth (102.4 GB/s is the commonly quoted figure).

### Cited Findings
- **Orin NX 16 GB modes:** MAXN SUPER and a new 40 W reference mode via JetPack 6.2 — [NVIDIA forum, Super mode post](https://forums.developer.nvidia.com/t/jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/320343). JetPack 6.2 power modes for Orin NX: 10 W, 15 W, 25 W, 40 W, MAXN SUPER. NVIDIA cites "up to a 70% increase in AI TOPS" for the Orin NX series and a 50 % memory-bandwidth boost only for Orin Nano — [JetPack 6.2 release notes](https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-62/release-notes/index.html)
- **Super mode is uncapped**, and the module throttles clocks if power exceeds the TDP budget — [Edge AI Vision (NVIDIA blog repost)](https://www.edge-ai-vision.com/2025/01/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/)
- **Vendor figures:** Orin NX 16 GB GPU up to 918 MHz (original) vs 1173 MHz (Super), 100 -> 157 TOPS — [Lanner](https://lannerinc.com/news-and-events/latest-news/lanner-s-edge-ai-computers-now-feature-jetpack-6-2-with-super-mode-for-enhanced-generative-ai-performance); 157 TOPS — [ThinkRobotics](https://thinkrobotics.com/ar/blogs/product-reviews-buying-guides/nvidia-jetson-orin-nx-module-review-specs-use-cases-price-and-how-it-compares)
- **Ultralytics recommends** `sudo nvpmodel -m 0`, `sudo jetson_clocks` and `jtop` (jetson-stats) before benchmarking — [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson)
- **GR00T's Thor number** was taken at "MAXN with jetson_clocks" — [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- **Batch-1 decode is memory-bandwidth-bound** on Orin, so runtimes differ by only about 10 ms per token — [Hokwang Choi, Apr 2026](https://hokwangchoi.com/blog/vlm-benchmarks/). That blog gives the Orin Nano 8 GB Super as 68 GB/s, which conflicts with NVIDIA's "50 % memory bandwidth boost" for Orin Nano in Super mode ([JetPack 6.2 release notes](https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-62/release-notes/index.html)).
- **Our own numbers:** clocks unlocked (no `jetson_clocks`), other GPU services up but idle; both robots agree within 0.05 ms — C:\Users\Mys\Desktop\RoboARM 2026\PLAN.md lines 633-636 (internal)

### Inferences
- **Clocks:** for a 3 ms engine fired once per 33 ms frame, DVFS idling between frames can inflate real (not trtexec back-to-back) latency. `jetson_clocks` or a fixed GPU minimum frequency is the cheapest "upgrade" to test. Weigh it against heat and power on a battery robot.
- **LLM decode:** compute does not scale it, bandwidth does. Edge-LLM's Orin NX vs AGX Orin ratio (Qwen3.5-2B: 47.9 vs 83.3 tok/s, about 0.58) is close to the commonly quoted 102 vs 205 GB/s ratio (about 0.5). This is consistent with bandwidth-bound decode, but neither bandwidth figure is verified here.
- **ViTs at high resolution** (DA3 multi-view, DA V2) are likely partly bandwidth-bound too, so INT4/INT8 weights help little there. Activations dominate.

### Gaps
- No NVIDIA primary source fetched for Orin NX 16 GB MAXN_SUPER clocks (GPU, CPU, DLA, EMC) or memory bandwidth (102.4 GB/s is unverified here).
- No published measurement of `jetson_clocks` on vs off for TRT latency on Orin NX.

## Benchmark plan for the Orin

Goal: one table per robot, the same models as M5 plus the ViTs and ACT. It should answer: (a) does INT8 via ModelOpt autotune beat FP16 on YOLO26n-seg; (b) what DLA gives as a second engine; (c) what clocks do; (d) later, the same on a spare JetPack 7.2.x board before any reflash of the robots.

**0. Record the platform each run:**
```bash
cat /etc/nv_tegra_release; dpkg -l | grep -E 'nvidia-jetpack|tensorrt|cudnn' | awk '{print $2,$3}'
/usr/src/tensorrt/bin/trtexec --version 2>/dev/null || trtexec --help | head -1   # 10.7 tarball path may differ
sudo nvpmodel -q; sudo jetson_clocks --show
```
Run `tegrastats --interval 100 --logfile tegra_<model>.log` in parallel during every run. It gives GPU% and frequency, EMC%, power rails (VDD_IN, VDD_CPU_GPU_CV, VDD_SOC) and temperatures.

**1. Clock conditions.** Run each engine twice: A = MAXN_SUPER, clocks unlocked (as M5); B = `sudo nvpmodel -m 0 && sudo jetson_clocks`. Restore after with `sudo jetson_clocks --restore` (store first with `--store`). Stop the robot's other GPU services, or record which ones were up.

**2. FP16 baseline** (strongly typed, ready for TRT 11). Export an FP16 ONNX (ModelOpt AutoCast or `onnxconverter_common.float16`, keeping I/O FP16), then:
```bash
trtexec --onnx=m_fp16.onnx --stronglyTyped --saveEngine=m_fp16.engine \
  --timingCacheFile=orin_trt107.cache --builderOptimizationLevel=5 \
  --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw \
  --profilingVerbosity=detailed --exportLayerInfo=m_fp16.layers.json --dumpLayerInfo
# timing: GPU compute only, CUDA graphs, no H2D/D2H, spin-wait
trtexec --loadEngine=m_fp16.engine --useCudaGraph --noDataTransfers --useSpinWait \
  --warmUp=1000 --duration=15 --percentile=50,90,99 --exportTimes=m_fp16.times.json
# per-layer profile in a separate run (profiling perturbs timing)
trtexec --loadEngine=m_fp16.engine --separateProfileRun --dumpProfile --exportProfile=m_fp16.prof.json
```
For comparison with the M5 table, also rebuild the old way: `trtexec --onnx=m.onnx --fp16` (weakly typed).

**3. INT8, explicit Q/DQ via ModelOpt** (on the Orin, so autotune times the right TensorRT):
```bash
python -m modelopt.onnx.quantization --onnx_path=m.onnx --quantize_mode=int8 \
  --calibration_data_path=kit_calib_500.npy --calibration_method=entropy \
  --autotune=default --output_path=m_int8.onnx
trtexec --onnx=m_int8.onnx --stronglyTyped --saveEngine=m_int8.engine \
  --timingCacheFile=orin_trt107.cache --profilingVerbosity=detailed --exportLayerInfo=m_int8.layers.json
```
Variants to run:
- `--calibration_method=max`;
- the mask-proto conv excluded (verify the exclusion flag name in `--help`);
- the Ultralytics QAT ONNX re-built strongly typed.

Count reformats with `grep -c -i reformat m_int8.layers.json`, or use Python json, filtering on the layer type and name. Record each engine's layer count and reformat count.

**4. DLA** (JetPack 6.2 + TRT 10.7 only; JetPack 7.2/7.2.1 cannot build DLA engines):
```bash
trtexec --onnx=m.onnx --fp16 --useDLACore=0 --allowGPUFallback --saveEngine=m_dla0_fp16.engine --verbose 2>&1 | tee dla_build.log
trtexec --onnx=m_int8_dla.onnx --int8 --fp16 --useDLACore=0 --allowGPUFallback --saveEngine=m_dla0_int8.engine \
  --inputIOFormats=int8:dla_hwc4 --verbose   # or fp16:chw16 for FP16 DLA input
```
Make the DLA INT8 ONNX with `--target_dla`; on 10.7 it needs the Q/DQ Translator and a matching calibration cache. Grep the build log for the layers that run on DLA versus fall back to the GPU. Then run the concurrency test, all three at once, for 15 s:
- the GPU engine (e.g. DA V2 S) with `--useCudaGraph`;
- the DLA0 engine;
- the DLA1 engine (`--useDLACore=1`).

Record each process's latency against its solo run.

**5. ViTs and ACT.** FP16 only, strongly typed. Run onnxslim first (`onnxslim in.onnx out.onnx`); a Myelin failure on raw ONNX is a data point too. Add `--useCudaGraph` timing and peak memory (tegrastats RAM, or `--memPoolSize` failures). For DA3 record 1 view and 3 views.

**6. End to end.** The Python path the robot uses (Ultralytics predict, or the TRT Python API with pinned or zero-copy buffers), timed per stage: preprocess / infer / postprocess (masks). It is the number that counts, and trtexec and the Python API differ on Orin.

**7. LLM (optional).** llama.cpp (jetson-containers r36.4 image) with Qwen3.5-0.8B/2B Q4_K_M via `llama-bench -p 64 -n 128`. Compare with the Edge-LLM v0.10.0 reference row for Orin NX (2B INT4 AWQ: 47.9 tok/s decode, 56.2 ms prefill / 62 tokens).

**What to record per row:** robot, JetPack/L4T, TRT build, model, input shape, ONNX hash, build flags (typed/weak, precision, opt level, DLA), engine size, build time, layer and reformat counts, GPU compute p50/p90/p99 (trtexec), host latency with transfers (drop `--noDataTransfers`), end-to-end p50/p99, clocks (A/B), power (VDD_IN mean) and energy per frame, peak RAM, accuracy on the kit val (mask mAP50 / mAP50-95) for every precision change.

**Pitfalls:**
- Engines are tied to the TRT version and device: rebuild after any JetPack or TRT change. JetPack 6.2 engines do not run on 7.2.
- Use the same timing cache only within one TRT version.
- `--noDataTransfers` plus CUDA graphs is the best case. Report it separately from host and end-to-end latency.
- Do not run `--dumpProfile` in the timed run.
- Warm-up: the first seconds under DVFS are slower. Use warm-up of 1 s or more and duration of 10 s or more.
- MAXN_SUPER throttles past TDP: watch tegrastats temperatures and clocks for long runs and battery sag.
- `--allowGPUFallback` hides DLA layer misses. Always read the build log.
- TRT 10.7 has no INT8 kernel for the YOLO26 proto conv: expect that layer to stay FP16.
- `--fp16 --int8` builds (weak typing) disappear in TRT 11. Keep the strongly typed variants as the reference so results carry over to JetPack 7.x.
- The 10.7 tarball's `trtexec` path differs from the deb install (jetson-containers PR #1750). Make sure the benchmark uses the intended binary (`trtexec --version`).
- Accuracy can drop silently with INT8: Ultralytics' Orin NX YOLO26 INT8 lost 3 mAP points (0.4789 -> 0.4489). Gate every INT8 engine on the kit val.
