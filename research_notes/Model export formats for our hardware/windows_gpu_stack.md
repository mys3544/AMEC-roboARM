# Fastest export formats and inference runtimes on our three Windows NVIDIA GPUs (state as of 2026-10-27)

Scope: RTX 2070 Super 8 GB (Turing, sm_75, laptop/desktop, driver 610), "hekate" RTX 5090 32 GB (consumer Blackwell, sm_120, shared), "Dionysos" RTX A4500 20 GB (Ampere, sm_86, driver 581.42, shared). Desk research only; nothing was run on the machines. Dates: "page updated" dates are page metadata, not release dates, unless stated. Facts taken from search-engine summaries rather than pages I opened are marked "(search summary)".

Version snapshot (October 2026), for the report writer:

| Component | Latest found | Date | Note |
|---|---|---|---|
| CUDA Toolkit | 13.4 Update 2 | 13.4 notes PDF dated 2026-10-12; 13.4.1 2026-09-11 | R615 driver branch for 13.4 features; 13.x minor-version compat on >= 580 |
| TensorRT | 11.4.0 | support matrix page updated 2026-10-05 | SM 7.5+; CUDA 12.9 and 13.4 builds |
| TensorRT for RTX | 1.7 listed as latest (1.6 notes read) | 1.6 page updated 2026-10-15 | CUDA 12.9 U1 and 13.4 packages |
| ONNX Runtime | 1.31.1 | 2026-10-26 (1.31.0: 2026-10-09) | GPU packages default to CUDA 13.0 since 1.27 |
| PyTorch | 2.14.x (2.14.0 GA 2026-09-02, 2.14.1 followed) | | last release with any CUDA 12.x wheels |
| Torch-TensorRT | 2.14.0 | 10 Sep (2026) | TensorRT 11.1, TRT-RTX 1.6, Windows wheels |
| TensorRT-LLM | 1.4 listed (1.3 ~2026-09) | | Windows deprecated since 0.18.0 |
| GR00T | N1.7 GA | | Linux-only deployment scripts |

## 1. CUDA 13.x, driver and framework support for Turing (sm_75), Ampere (sm_86), Blackwell (sm_120)

### Takeaway
Nothing released by October 2026 has dropped or deprecated Turing: CUDA 13.0-13.4, TensorRT 11.4, TensorRT-RTX 1.6/1.7, cuDNN 9 and PyTorch's CUDA 13 wheels all still include sm_75. Volta (sm_70) and everything older is the generation that was cut: from CUDA 13.0, TensorRT 10.5, and PyTorch's cu128+ builds. The real Turing restrictions are softer: TensorRT-RTX treats it as a second-class target, the ORT NvTensorRTRTX EP and Windows ML NVIDIA EP require Ampere+ (RTX 30xx+), and ORT/TensorRT hardware-compatible engines are Ampere+ only. The CUDA 12 to 13 move is happening across all stacks: ORT GPU packages default to CUDA 13 since 1.27, and PyTorch 2.14 is the last release with CUDA 12 wheels. Turing is fine on CUDA 13. The practical blocker on Windows is packaging: there are no Windows wheels for `nvidia-cublas` 13.x.

### Cited Findings
**CUDA / driver**
- CUDA 13.0 removed offline compilation for compute capability below 7.5 (Maxwell, Pascal, Volta) as "feature-complete" architectures. The R580 driver branch is the last to support them and is an LTS branch maintained for three years (search summary) — [CUDA 13.0 release notes PDF](https://docs.nvidia.com/cuda/archive/13.0.1/pdf/CUDA_Toolkit_Release_Notes.pdf); [NVIDIA blog: What's new in CUDA 13.0](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/)
- The current toolkit is CUDA 13.4 Update 2. Minor-version compatibility for 13.x needs driver >= 580 on Linux and Windows. New 13.4 features and platforms need an R615 or later driver. The driver has not been bundled with the Windows toolkit since 13.1. cuFFT 13.0 and cuSPARSE 13.0 dropped pre-Turing GPUs. The page has no Turing deprecation notice. The only announced CUDA 14.0 change is ARM64-SBSA moving to Armv8.2-A — [CUDA Toolkit release notes (current)](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- The 13.4 release-notes PDFs are dated Oct 12, 2026 (13.4) and Sep 11, 2026 (13.4.1) (search result titles) — [CUDA 13.4 notes PDF](https://docs.nvidia.com/cuda/pdf/CUDA_Toolkit_Release_Notes.pdf); [13.4.1 archive](https://docs.nvidia.com/cuda/archive/13.4.1/pdf/CUDA_Toolkit_Release_Notes.pdf)
- Speculation only, not an NVIDIA statement: a personal wiki says Turing is "likely next removal target" for a future CUDA 14.0 — [Bommarito wiki, CUDA 13](https://michaelbommarito.com/wiki/programming/tools/cuda-13/)

**TensorRT**
- TensorRT 11.4.0 is the latest release and "supports NVIDIA hardware with compute capability SM 7.5 or higher". The architecture filter lists 7.5 (T4). Platforms are Windows 10 x64, Windows 11 x64 and Windows Server 2022 x64 (plus Windows on Arm). There are two package builds: CUDA 12.9 ("compatible with CUDA 12.x versions only") and CUDA 13.4 ("13.x only"). The `tensorrt-cu12` pip wheels support CUDA 12.x. Python 3.8/3.9 bindings are deprecated — [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
- TensorRT 10.0 deprecated Volta and removed it in 10.5 (search summary) — [TensorRT 10.x release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes.html)
- The TensorRT 11.0 and 11.4 notes deprecate or remove no architecture. Windows 10 support "is deprecated and ends after October 2026" — [TensorRT 11.0.0 notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.0.0.html); [TensorRT 11.4.0 notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.4.0.html)

**TensorRT for RTX**
- TensorRT-RTX supports SM 7.5 or higher, with filters for 12.1, 12.0, 8.9, 8.6 and 7.5. By default, engines are portable across 8.6/8.9/12.0/12.1 but not Turing. Turing needs a device-specific engine ("deploy two engines"). On Turing it supports neither FP32 GEMMs nor 3D convolutions. The x86-64 packages come for CUDA 12.9 Update 1 and CUDA 13.4 — [TensorRT-RTX support matrix](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/support-matrix.html)
- TRT-RTX 1.6: a single engine that covers Turing and later "performs worse on Ampere and later". Only the WDDM driver is supported; TCC fails with "operation not supported" — [TensorRT-RTX 1.6 release notes](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.6.html)

**ONNX Runtime**
- Since ORT 1.27, the PyPI `onnxruntime-gpu` and NuGet GPU packages default to a CUDA 13.0 build. For 1.27.x-1.30.x, separate CUDA 12.x packages are built against 12.8. 1.21-1.26 defaulted to CUDA 12.8. All use cuDNN 9.x. `pip install onnxruntime-gpu[cuda,cudnn]` pulls the runtime DLLs, and `onnxruntime.preload_dlls()` (1.21+) loads them on Windows from torch's lib directory, then nvidia site-packages — [ORT CUDA EP docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- Conflicting statement: an aggregated changelog quotes the 1.26.0 notes as saying CUDA 12 support "will be removed in 1.27.0", with CUDA 13 continuing under a `gpu_cuda13` naming scheme (search summary). This contradicts the ORT doc table above, which lists CUDA 12 builds for 1.27-1.30 — [Safety CLI ORT changelog](https://data.safetycli.com/packages/pypi/onnxruntime/changelog) vs [ORT CUDA EP docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- ORT 1.24.1 (06 Feb 2026) added CUDA 13.0 support to the CUDA EP. Its TensorRT EP was upgraded to TensorRT 10.14 and dropped Python 3.10 wheels — [ORT v1.24.1 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.24.1)
- Our blocker is confirmed: `nvidia-cublas` 13.8.2.12 (released Oct 27, 2026) ships only manylinux x86_64/aarch64 wheels, with no `win_amd64` wheel — [PyPI nvidia-cublas](https://pypi.org/project/nvidia-cublas/)

**PyTorch**
- PyTorch 2.11 (released 23 Mar 2026): PyPI defaults to CUDA 13.0 on Linux. cu126 and cu128 stayed on download.pytorch.org. cu128/cu129 builds dropped Volta. CUDA 13.0 supports "Turing (SM 7.5) and newer" — [PyTorch v2.11.0 release](https://github.com/pytorch/pytorch/releases/tag/v2.11.0)
- PyTorch 2.11 matrix for Linux x86 and Windows (one combined row): 12.6.3 = 5.0-9.0 (includes 7.5); 12.8.1 = 7.5, 8.0, 8.6, 9.0, 10.0, 12.0; 13.0.0 = 7.5, 8.0, 8.6, 9.0, 10.0, 12.0 — [RFC #172663, CUDA matrix for 2.11](https://github.com/pytorch/pytorch/issues/172663)
- PyTorch 2.12 (post of 1 Apr 2026): CUDA 12.8 deprecated with no prebuilt binary. 13.0.x stable on PyPI. 13.2.x experimental. 12.6.3 kept as "legacy" for Maxwell/Pascal/Volta. 13.0 and 13.2 include Turing 7.5, Ampere 8.0/8.6 and Blackwell 10.0/12.0 — [dev-discuss: Introducing CUDA 13.2 and deprecating CUDA 12.8 (2.12)](https://dev-discuss.pytorch.org/t/introducing-cuda-13-2-and-deprecating-cuda-12-8-release-2-12/3337)
- 2.12.1 reached GA — [dev-discuss: 2.12.1 GA](https://dev-discuss.pytorch.org/t/pytorch-2-12-1-general-availability/3402). 2.13 released 8 Jul 2026 — [2.13 key dates](https://dev-discuss.pytorch.org/t/pytorch-release-2-13-key-dates/3390); [PyTorch 2.13 blog](https://pytorch.org/blog/pytorch-2-13-release-blog/)
- PyTorch 2.14.0 GA was 2 Sep 2026, and 2.14.1 followed. 2.14 is the last line with CUDA 12.x wheels. From 2.15, CUDA 12.6 is build-from-source only (drops Maxwell/Pascal/Volta). Nightlies dropped CUDA 13.0 from the week of 28 Sep 2026 and moved to 13.2 (stable/default) and 13.4 (prototype) (search summary) — [PyTorch 2.14 blog](https://pytorch.org/blog/pytorch-2-14-release-blog/); [dev-discuss notice: CUDA 12.6 wheels end at 2.15](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)

**cuDNN**
- The cuDNN 9.0, 9.7 and 9.10.x support matrices still list compute capability 7.5 (Turing). The 9.10 matrices add Blackwell 10.0/12.0 and footnote Volta/Pascal/Maxwell (search summary) — [cuDNN 9.10.1 support matrix](https://docs.nvidia.com/deeplearning/backend/v9.10.1/reference/support-matrix.html); [cuDNN 9.0.0 support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.0.0/reference/support-matrix.html)

**Ampere+ only items that exclude Turing**
- ORT TensorRT EP `trt_engine_hw_compatible` (sm80+ engines): "Turing and earlier GPUs and Jetson Orin are not eligible" — [ORT TensorRT EP docs](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- ORT NvTensorRTRTX EP: "RTX GPUs based on Ampere and later" — [ORT TensorRT RTX EP docs](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html). Windows ML NvTensorRtRtx EP: "GeForce RTX 30XX and above", driver 32.0.15.5585+ and CUDA 12.5 — [Windows ML execution providers (page dated 2026-10-05)](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)

### Inferences
- **RTX 2070 Super, latest versions that still work:** CUDA 13.4 (any CUDA 13 runtime needs driver >= 580; laptop driver 610 is fine; 13.4-only features want R615+); TensorRT 11.4 (both cu12 and cu13 builds); TensorRT-RTX 1.6/1.7 (Turing-specific engine; FP32-GEMM/3D-conv gaps); ORT 1.31 CUDA EP and TensorRT EP (but not NvTensorRTRTX EP / Windows ML NVIDIA EP); PyTorch 2.14 cu130/cu132 (2.11 RFC lists 7.5 for the Windows rows); cuDNN 9.x. Turing is not "unsupported" anywhere we would use. Turing is also the oldest generation left in CUDA 13, so it is the obvious next one to go. Plan for a CUDA 14 / TensorRT 12 cut, but no such notice exists today.
- **Staying on CUDA 12 vs moving to 13:** The CUDA 12 window is closing. PyTorch 2.15 has no CUDA 12 wheels, and ORT's default GPU wheel has been CUDA 13 since 1.27. On Windows the CUDA 13 path cannot be satisfied from pip, because `nvidia-cublas` 13.x has no Windows wheel. Practical options are: (a) install the CUDA 13.x toolkit (cuBLAS/cuDNN DLLs) with NVIDIA's Windows installer and put its `bin` on PATH, or point `preload_dlls(directory=...)` at it; (b) let torch cu130 supply the DLLs, since `preload_dlls()` searches torch's lib dir first — not verified that torch's Windows cu130 wheel bundles cuBLAS 13; (c) stay on ORT <= 1.30 with its CUDA 12.8 build. The laptop's current ORT 1.24.1 + CUDA 12 choice is still valid but frozen.
- **llama.cpp on Turing:** CUDA 13.x keeps sm_75, so a CUDA 13 llama.cpp build is not unsupported on the 2070 Super. The cuda-12.4 build is the conservative choice, but no source shows a Turing speed or stability penalty for CUDA 13. This has not been tested.
- **Dionysos (driver 581.42, R580):** can run CUDA 13.0 apps natively and 13.x apps by minor-version compatibility (>= 580). TensorRT 11.4's cu13 build is "built with CUDA 13.4" and TRT-RTX 1.6's runtime cache checks the driver version. A driver at or above R615 (and >= 595.97 for Isaac Lab 3.0) is the safe recommendation. It must be asked first, because the server is shared.
- **hekate (CUDA 12.8.1 toolkit):** the TensorRT 11.4 cu12 wheel (built with 12.9, "12.x only"; some features need 12.8+ or 12.9+) and the TRT-RTX CUDA 12.9 U1 package are the matching choices without a toolkit change.

### Gaps
- I did not find the precision-by-architecture grid of the TensorRT 11.4 support matrix; it is rendered by interactive filters and was not extractable.
- I did not find a cuDNN support matrix newer than 9.10.1 in search, so cuDNN 9.1x-latest Turing status is unverified (no deprecation notice found).
- I did not verify the exact driver branch for the laptop's "driver 610" (likely R610, which would correspond to a CUDA 13.3-era driver).
- It is unclear whether ORT 1.31 still publishes a CUDA 12 variant; the docs only list CUDA 12 builds for 1.27-1.30.

## 2. Precisions per GPU, and which runtimes exploit FP8 / NVFP4 / MXFP4 on the RTX 5090 on Windows

### Takeaway
On the RTX 5090 (sm_120), FP8 and FP4 tensor-core paths are real and exposed on native Windows through classic TensorRT (FP4 since 10.8, via ModelOpt-quantized ONNX with Q/DQ), TensorRT-RTX (FP4 GEMM needs SM120, FP8 needs SM89+), the ORT TensorRT EP (NVFP4 custom ops since 1.24), the ORT CUDA EP (MXFP4/NVFP4 MoE kernels on SM120 Windows in 1.30/1.31), and llama.cpp (NVFP4/MXFP4 GGUF). TensorRT-LLM, vLLM and SGLang are not native-Windows options. Measured gains: FP4 vs FP8 is about 1.7x end-to-end for FLUX in TensorRT, and about 1.3x LLM decode in TRT-LLM (OS not stated). TensorRT 11 changed the workflow: precision is no longer a builder flag. The ONNX itself must carry FP16/BF16 types or Q/DQ nodes (ModelOpt AutoCast / PTQ), and implicit INT8 calibration is gone. That matters for all three GPUs, including INT8 on Turing.

### Cited Findings
**Hardware precision gates (vendor statements)**
- In TensorRT-RTX, FP4 requires Blackwell (SM120) or newer and FP8 requires Ada (SM89) or newer. The API is strongly typed: "the precision of operations in the model graph cannot be changed during optimization" — [NVIDIA blog: Run high-performance AI apps with TensorRT for RTX (12 Jun 2025)](https://developer.nvidia.com/blog/run-high-performance-ai-applications-with-nvidia-tensorrt-for-rtx/)
- TRT-RTX lists precisions FP32, FP16, BF16, INT8, INT4 WoQ GEMM, FP8 and FP4 GEMM. FP8 conv/matmul need 4-byte alignment. FP8 deconvolution is not supported — [TRT-RTX support matrix](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/support-matrix.html)
- A TensorRT support-matrix footnote says FP4 runs in "hardware emulation mode" on some hardware, where it "does not accelerate FP4 linear operations". The hardware is not named — [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
- The TensorRT `kSAME_COMPUTE_CAPABILITY` hardware-compatibility level "allows usage of new hardware features like FP8 and FP4"; `kAMPERE_PLUS` engines exclude such tactics — [TensorRT engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)

**TensorRT 11 precision workflow change (affects all three GPUs)**
- TensorRT 11.0 removed weak typing (`ILayer::setPrecision`, `ITensor::setType`, and BuilderFlags kFP16/kINT8/kFP8/kINT4/kFP4) and implicit INT8 quantization (`IInt8*Calibrator`, `setDynamicRange`). Use explicit Q/DQ instead. cuDNN-dependent plugins are unsupported — [TensorRT 11.0.0 notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.0.0.html)
- trtexec in 11.x drops `--fp16 --int8 --bf16 --fp8 --int4 --best` and `--calib`. `--stronglyTyped` is accepted but has no effect. The documented replacement is `python -m modelopt.onnx.autocast --onnx_path model.onnx --output_path model_fp16.onnx`, then `trtexec --onnx=model_fp16.onnx --saveEngine=engine.plan` — [Migrating trtexec usage 10.x to 11.x](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-trtexec.html)
- ModelOpt's docs have an "ONNX Quantization - Windows" section (a Linux ONNX path is newer, beta), plus Triton kernels for FP8 (E4M3) and NVFP4 (E2M1) fake quantization (search summary) — [NVIDIA Model Optimizer docs](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.kernels.quantization.common.html)

**Measured FP4 / FP8 numbers on the RTX 5090**
- TensorRT 10.8 + ModelOpt 0.27, ONNX 1.18 (opset 23 needed for FP4 export), FLUX on RTX 5090, end-to-end: FLUX.1-dev, 30 steps: FP16 10,931 ms, FP8 6,681 ms, FP4 3,853 ms. FLUX.1-schnell, 4 steps: 4,427 / 913 / 591 ms. FC layers reach "up to 3.1x" FP8 throughput in FP4. VRAM: FP16 39.3 GB, FP8 24.6 GB, FP4 21.7 GB (default mode). FP4 export uses a custom TRT_FP4QDQ op that becomes a two-step DQ pattern. Attention stays FP8 and norms higher. QAT/SVDQuant recovered quality (ImageReward BF16 1.118, FP4 PTQ 1.096, FP4 QAT 1.119) — [NVIDIA blog: TensorRT unlocks FP4 image generation on RTX 50 (14 May 2025)](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus/)
- TensorRT-LLM, Qwen3-8B, one RTX 5090: single-user decode was 222.7 tok/s in NVFP4 vs 170.0 tok/s in FP8 (about +30%), for about -1.5 GSM8K points. The OS is not clearly stated (TRT-LLM is Linux/WSL) — [ComputingForGeeks: NVFP4 vs FP8 on RTX 5090](https://computingforgeeks.com/nvfp4-quantization-tensorrt-llm/)
- ComfyUI on RTX 5090 (secondary blog): FLUX Dev 7.73 it/s NVFP4 vs 4.21 it/s FP8. NVFP4 "can run up to 2x slower than FP8" without the right PyTorch/CUDA 13 setup, i.e. when the fallback path is used — [runaihome ComfyUI NVFP4 guide](https://runaihome.com/blog/comfyui-nvfp4-rtx-speed-guide-2026/)
- A video model (LTX-2.3) on RTX 5090 took 278 s in NVFP4 vs 540 s in FP8, with visible quality problems (search summary, single blog) — [Zenn: RTX 5090 NVFP4 tested](https://zenn.dev/toki_mwc/articles/rtx5090-nvfp4-quantization-reality?locale=en)

**Runtime coverage of FP8/FP4 on Windows**
- ORT 1.24.1: the TensorRT EP (TensorRT 10.14) gained "NVFP4 custom ops", and the TensorRT RTX EP gained BF16 — [ORT v1.24.1 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.24.1)
- ORT CUDA Plugin EP 0.2.0 / ORT 1.30: FP4 QMoE on by default in CUDA builds "with Windows support for Blackwell SM120". The FP8 DeepGEMM path is off on Windows. ORT 1.31.0 (09 Oct 2026) / CUDA Plugin EP 0.3.0: "MXFP4 on Blackwell consumer GPUs", a fixed MXFP4 W4A16 QMoE on SM120/121, NVFP4 QMoE updates, and opt-in GEMM auto-tune only on SM90+ — [ORT releases page](https://github.com/microsoft/onnxruntime/releases)
- NVIDIA says Windows ML's NvTensorRtRtx EP "takes advantage of FP8 and FP4" tensor cores. Measured: "up to 50% faster throughput" than prior DirectML on an RTX 5090. CUDA graphs give about 30% on LLMs — [NVIDIA blog: Windows ML on RTX PCs (23 Sep 2025)](https://developer.nvidia.com/blog/deploy-ai-models-faster-with-windows-ml-on-rtx-pcs/)
- Torch-TensorRT: 2.10 `run_llm` compiles pre-quantized and on-device-quantized FP8/NVFP4 models. 2.13 adds FP8 attention (Hopper+). 2.14 adds native FP8 for TensorRT's IAttention layer — [Torch-TensorRT releases](https://github.com/NVIDIA/Torch-TensorRT/releases)
- TensorRT-LLM: 0.19.0 added "FP8 support for SM120". 0.17.0 added GeForce RTX 50 "through WSL" for limited models. 1.0 fixed FP4 on sm120 (Qwen3) and MoE GEMM illegal memory access on SM120. No release explicitly lists NVFP4 for SM120 — [TensorRT-LLM release notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html)
- llama.cpp: build b8967 reportedly added native NVFP4 MMQ kernels for sm_120. On an RTX 5090-class card, pp512 went from about 3,295 to 5,547 t/s (+68%), while decode (tg128) stayed flat at about 73.6 t/s (secondary blog, date unclear) — [insiderllm: FP4 in llama.cpp](https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/). Native MXFP4 for Blackwell started as experimental PR #17906. The compile arch target matters (`121a-real` vs `120f`) — [NVIDIA forum thread on the llama.cpp MXFP4 PR](https://forums.developer.nvidia.com/t/llama-cpp-experimental-native-mxfp4-support-for-blackwell-pr/355639)

**Known precision-related issues on our GPUs (TensorRT 11.x)**
- 11.4 known issues: an INT8+FP16 YOLOX-Tiny network gives accuracy errors on RTX 5080 and RTX 5090. An FP16 GridSample network fails on RTX 5080 (Windows). A strongly typed FP16 ViT-Base is 6-7% slower on RTX PRO 6000 Blackwell (CC 12.0) in 11.3 than in 11.2.1 — [TensorRT 11.4.0 notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.4.0.html)
- 11.0 known issues: strongly typed FP16 networks on RTX PRO 6000 Blackwell Max-Q (sm120, aarch64) regress by up to about 45% GPU time on some networks. EfficientNet/RegNet regress 5-10% on RTX PRO 6000 Blackwell. FP8 convolutions with kernels larger than 32 (e.g. 7x7) on SM89/90/120/121 fall back to FP16/FP32 — [TensorRT 11.0.0 notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.0.0.html)

### Inferences
- **Per-GPU precision menu (hardware facts from general knowledge; I could not extract the per-arch grid, see Gaps):**
  - Turing (sm_75): FP16 tensor cores and INT8 (Q/DQ via ModelOpt in TRT 11). No BF16, FP8 or FP4 acceleration. In TRT-RTX, avoid FP32 GEMM layers on Turing (unsupported).
  - Ampere (sm_86): FP16/BF16/TF32, INT8, and 2:4 structured sparsity. No FP8/FP4 acceleration; FP8 needs SM89+ per NVIDIA.
  - Blackwell consumer (sm_120): adds FP8 and FP4 (NVFP4, MXFP4) on top of the Ampere set.
- **What to use for each model class on the RTX 5090:**
  - ViTs and policies (DINOv2, Depth Anything, SAM 2, ACT/DiT action heads) at batch 1: FP16/BF16 with TensorRT (or TRT-RTX) is the baseline. FP8 helps mainly GEMM-heavy large models. NVFP4 pays off when weights and GEMMs dominate (big transformers, LLM prefill, diffusion), not for small conv nets or tiny ACT models, where launch overhead and memory-bound ops dominate.
  - LLM decode at batch 1 is bandwidth-bound: FP4 halves bytes vs FP8, which matches the roughly 30% decode gain measured in TRT-LLM.
  - llama.cpp's NVFP4 gain is in prefill, not decode. For our Qwen3.5 4B voice LLM, that means faster prompt processing, not faster token output.
- **INT8 on Turing in TensorRT 11:** our old `--int8` + calibration-cache workflow (if any) no longer exists. YOLO-class INT8 on the 2070 Super needs ModelOpt ONNX PTQ to Q/DQ, or staying on TensorRT 10.x. The 11.4 note about YOLOX INT8+FP16 accuracy errors on RTX 5090 means INT8 YOLO accuracy must be checked per GPU.
- **NVFP4 on Windows has no turnkey "LLM server" path** like TRT-LLM/vLLM. Native Windows coverage of FP4 is TensorRT/TRT-RTX engines (diffusion/ViT/custom), ORT CUDA EP MoE kernels, and llama.cpp GGUF.

### Gaps
- I found no source that cleanly benchmarks NVFP4 vs FP8 vs FP16 on native Windows with classic TensorRT for ViT- or policy-sized models; all FP4 numbers are diffusion or LLM.
- I found no measured TRT-RTX FP4 vs classic TensorRT FP4 speed comparison.
- The TensorRT 11 precision-per-compute-capability table (BF16 on 7.5, sparsity support in 11.x) was not extractable. BF16-not-on-Turing and 2:4 sparsity on Ampere are from general knowledge and were not verified in this session.
- The ModelOpt CLI flags for FP8/NVFP4 ONNX PTQ on Windows were not verified against current ModelOpt docs.

## 3. TensorRT for RTX, Windows ML, and ONNX Runtime execution providers on Windows

### Takeaway
TensorRT-RTX is NVIDIA's client-side TensorRT: a sub-200 MB SDK where the AOT step needs no GPU, kernels are JIT-specialized on the device in seconds, a runtime cache keeps them, and CUDA graphs are built in. It is at v1.6/1.7 with CUDA 12.9 and 13.4 packages. NVIDIA itself positions classic TensorRT as the choice for "highest possible inference throughput" and TRT-RTX for end-user PCs, and I found no head-to-head speed numbers. In ORT the picture is in flux. The built-in NvTensorRTRTX EP is deprecated in favour of NVIDIA's standalone plugin EP (EP ABI, source build). The CUDA EP is being released as a plugin EP (CUDA Plugin EP 0.3.0 matched to ORT 1.31). The TensorRT EP is still built in (TensorRT 10.14 in 1.24). DirectML is in "sustained engineering" (maintenance, not formally deprecated), and Windows ML is Microsoft's successor; its NVIDIA EP is NvTensorRtRtx, RTX 30xx and above only. For our servers (Python, LAN serving), classic TensorRT or the ORT TensorRT EP remains the fastest-known route on every GPU. TRT-RTX/Windows ML is an app-distribution technology that is attractive for Ampere and Blackwell, but not usable for the 2070 Super through ORT.

### Cited Findings
**TensorRT-RTX**
- The SDK is under 200 MB. Phase 1 (AOT) optimizes and serializes and "does not require a GPU". Phase 2 (JIT) specializes for the installed GPU, compiling shape-specialized kernels in a background thread while fallback kernels serve early requests. Example: FLUX.1-dev took about 270 ms for step 1 vs about 100 ms for later steps. Windows ships as a zip — [NVIDIA blog: TensorRT for RTX (12 Jun 2025)](https://developer.nvidia.com/blog/run-high-performance-ai-applications-with-nvidia-tensorrt-for-rtx/)
- Build times are "typically within 15 to 30s". 1.1 improved compile time by 1.5x on average (search summary). The porting guide says classic TensorRT autotunes on the target device and its runtime libraries are "around 1 GB". NVIDIA: use TensorRT for datacenter GPUs when you need "the highest possible inference throughput", and TRT-RTX for end-user PCs — [NVIDIA TensorRT for RTX page](https://developer.nvidia.com/tensorrt-rtx); [TRT-RTX porting guide](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/porting.html); [TRT-RTX 1.1 notes](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.1.html)
- Version history (search summary of the release notes):
  - 1.2: CUDA 13.0, built-in CUDA graphs with dynamic shapes.
  - 1.3: thread-safe multi-GPU, improved FP8 on Blackwell.
  - 1.4: CUDA 13.2 and PyPI availability.
  - 1.5: RoiAlign, GEMV with dynamic shapes.
  - 1.6: CUDA 13.4.
  - Sources: [1.4 notes](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.4.html); [1.5 notes](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.5.html); [1.3 notes](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.3.html)
- TRT-RTX 1.6 (navigation lists 1.7 as latest):
  - **Features:** deferred weights loading; runtime-cache deserialization about 20-30% faster; lower CPU overhead and GPU memory for static shapes.
  - **AOT advice:** CPU-only AOT "can reduce performance for some models". On-device AOT (`--useGPU` or `kCURRENT`) may be faster, so measure both.
  - **Runtime cache:** must match GPU SKU, TRT-RTX version and CiG state, with driver >= the cached one. Caches over 100 MB add overhead.
  - **Engines are "not forward-compatible"**: rebuild on runtime upgrade.
  - **Unsupported layers:** NonMaxSuppression, NonZero, Multinomial.
  - **Windows:** WDDM only, not TCC. `cudaMallocAsync` can fail in some environments; a `cudaMalloc` allocator workaround disables CUDA graphs. Some concat-then-conv patterns are slower on Windows. The timing cache is deprecated (no effect).
  - Source: [TRT-RTX 1.6 notes](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.6.html)
- Torch-TensorRT ships a separate `torch-tensorrt-rtx` package since 2.11 (TRT-RTX 1.3), with wheels for 2.12-2.14 (2.14 targets TRT-RTX 1.6) — [Torch-TensorRT releases](https://github.com/NVIDIA/Torch-TensorRT/releases)

**ORT execution providers**
- TensorRT EP:
  - The docs' version table is stale: it stops at "main, 1.21, 1.22 = TensorRT 10.9, CUDA 12.0-12.8", and says only CUDA 12 GPU packages have been released since 1.22.
  - Options: engine cache (`trt_engine_cache_enable/path/prefix`), EPContext dump (`trt_dump_ep_context_model`, embed_mode 0 recommended), timing cache (efficientnet-lite4 build 34.6 s to 7.7 s; yolov4 108.6 s to 9.4 s), `trt_cuda_graph_enable`, and `trt_engine_hw_compatible` (Ampere+, "may have lower throughput and/or higher latency").
  - Engines are not portable across model, ORT, TRT or hardware changes.
  - Source: [ORT TensorRT EP docs](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- ORT 1.24.1 (06 Feb 2026):
  - TensorRT EP moved to TensorRT 10.14, with automatic plugin loading and NVFP4 custom ops.
  - TensorRT RTX EP gained runtime caching, CUDA graph support, BF16 and memory-mapped engines.
  - The notes also list "CUDA Graph: enabled by default". The extracted text does not say for which EP.
  - Source: [ORT v1.24.1](https://github.com/microsoft/onnxruntime/releases/tag/v1.24.1)
- NvTensorRTRTX EP:
  - The built-in EP "is deprecated" ("We strongly recommend using the standalone EP ABI plugin"), replaced by NVIDIA/TensorRT-RTX-EP-ABI. The plugin EP ABI arrived in ORT 1.23.
  - It is "currently" build-from-source only; PyPI/NuGet are "coming soon".
  - Supports Ampere+ RTX.
  - Options include `enable_cuda_graph` (default true), `nv_runtime_cache_path`, `nv_profile_{min,opt,max}_shapes` and `nv_multi_profile_enable`.
  - The docs give no performance comparison vs the TRT or CUDA EPs.
  - Source: [ORT TensorRT RTX EP docs](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html)
- The TensorRT-RTX-EP-ABI test matrix: EP 0.5 pairs with ORT 1.25.0-1.30.0 and TRT-RTX 1.7. EP 0.3 = ORT 1.24.4-1.25.2. EP 0.1 = ORT 1.24.0 only. It covers Ampere+ on Windows and Linux (Linux from 0.3, Windows-on-Arm from 0.4), and has a `tools/benchmark` directory but no published numbers — [NVIDIA/TensorRT-RTX-EP-ABI](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI)
- CUDA EP:
  - CUDA graphs are a "Preview" feature (`enable_cuda_graph`). Restrictions: no If/Loop/Scan, the whole graph must be on the CUDA EP, fixed shapes and addresses, IO binding required, no multi-threaded Run on one session, and the first Run captures (high latency).
  - Source: [ORT CUDA EP docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- ORT 1.31.0 (09 Oct 2026) is "matched with CUDA Plugin EP 0.3.0" (minimum core 1.24.4). The notes recommend building core and plugin from the same ORT revision — [ORT releases](https://github.com/microsoft/onnxruntime/releases)

**DirectML / Windows ML**
- DirectML is in "sustained engineering": still supported, but new feature work moved to Windows ML for ONNX Runtime deployments on Windows. ORT's install page says "For new Windows projects, consider WinML instead". `onnxruntime-directml` 1.24.4 shipped 17 Mar 2026 (search summary) — [Microsoft Learn: Introduction to DirectML](https://learn.microsoft.com/en-us/windows/ai/directml/dml); [ORT install](https://onnxruntime.ai/docs/install/); [PyPI onnxruntime-directml](https://pypi.org/project/onnxruntime-directml/)
- Windows ML includes the CPU EP and "DirectML (legacy)". Downloadable EPs (Windows 11 24H2, build 26100+, via `ExecutionProviderCatalog`) include NvTensorRtRtx (NVIDIA), MIGraphX, OpenVINO, QNN, VitisAI and WebGPU (experimental) for Windows ML 2.x. The NVIDIA EP needs "GeForce RTX 30XX and above" — [Windows ML execution providers (2026-10-05)](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)
- NVIDIA reports Windows ML with the TRT-RTX EP at up to 50% higher throughput than DirectML on RTX 5090. CUDA graphs give about 30% on LLMs. The EP is under 200 MB and auto-downloaded. With IO binding, SD 3.5 copy time dropped about 75x — [NVIDIA blog: Windows ML on RTX (23 Sep 2025)](https://developer.nvidia.com/blog/deploy-ai-models-faster-with-windows-ml-on-rtx-pcs/)

### Inferences
- **Maturity ranking for our Python services, as of Oct 2026:**
  1. Classic TensorRT 10.x/11.x is the most mature and documented, and NVIDIA's "max throughput" pick.
  2. The ORT TensorRT EP is mature and gives ORT-API convenience with CPU/CUDA fallback for unsupported ops. Its engines are classic TRT engines, so speed should be close to trtexec, minus partitioning overhead. It is pinned to the TensorRT major/minor ORT was built with (10.14 for 1.24), so you cannot drop TensorRT 11 under ORT 1.24.
  3. ORT CUDA EP (cuDNN/cuBLAS), with CUDA graphs: a simpler and slower fallback.
  4. TRT-RTX / NvTensorRTRTX plugin EP: newest and the fastest-moving API. It has good startup (JIT seconds instead of minutes of autotune) and built-in CUDA graphs, but no published peak-speed parity data, and the ORT plugin needs a source build.
- **For the 2070 Super:** the NvTensorRTRTX EP and Windows ML NVIDIA EP are out (Ampere+ / RTX 30xx+). Raw TRT-RTX is possible with a Turing-specific engine, but with FP32-GEMM/3D-conv gaps. Classic TensorRT or the ORT TRT EP is the choice.
- **For Dionysos (RTX A4500):** TRT-RTX requires WDDM. Some RTX professional boards on Windows can be switched to TCC. If the A4500 runs in TCC, TRT-RTX and WSL GPU access fail. Check the driver model before testing (read-only `nvidia-smi -q`), and do not change it, since that is a system setting on a shared server.
- **DirectML has no reason to exist in our stack:** every GPU we have runs CUDA; DML is the slow legacy path.

### Gaps
- I found no published head-to-head of TRT-RTX vs classic TensorRT latency/throughput on the same model and GPU, so this must be measured (see the benchmark plan).
- The exact current release dates of TRT-RTX 1.6 and 1.7 were not stated on the pages read.
- Whether the ORT CUDA EP inside the `onnxruntime-gpu` 1.31 wheel is now only the plugin (requiring a separate install) or still built in was not confirmed. The release text only says core is "matched with CUDA Plugin EP 0.3.0".

## 4. PyTorch-side options on Windows: torch.compile/Triton, AOTInductor, Torch-TensorRT, CUDA graphs

### Takeaway
On native Windows, `torch.compile` with Inductor GPU codegen needs the community `triton-windows` fork, now also hosted under the triton-lang GitHub org. It is pinned per PyTorch minor (PyTorch 2.11 to Triton 3.6). PyTorch does not install Triton on Windows itself and does not officially guarantee it. Torch-TensorRT is the most "official" PyTorch-native fast path on Windows: Windows x86-64 wheels plus a C++ zip, Dynamo frontend, 2.14.0 on TensorRT 11.1 with CUDA 12.6/13.0/13.2. It also ships a TRT-RTX variant. For small transformer policies at batch 1, CUDA graphs are what matter, whether via `torch.compile(mode="reduce-overhead")`, TensorRT CUDA graphs or TRT-RTX's built-in graphs. NVIDIA's own GR00T numbers show TensorRT about 1.9x faster than eager end-to-end on an RTX 5090. No source compared torch.compile vs TensorRT for ACT-size policies on Windows; this needs our benchmark.

### Cited Findings
- `triton-windows` lists PyTorch 2.11 with Triton 3.6 and says each PyTorch minor "is only guaranteed to work with a specific Triton minor". The fork's README claims `triton.jit` and `torch.compile` "just work" on Windows 10/11 (project's own claim). Pin `triton-windows<3.7` for 3.6 (search summary) — [triton-lang/triton-windows](https://github.com/triton-lang/triton-windows); [woct0rdho/triton-windows releases](https://github.com/woct0rdho/triton-windows/releases)
- A third-party guide says official Triton wheels are Linux-only and "torch doesn't install Triton at all on Windows" (stated for 2.14), so the matching triton-windows must be installed by hand (search summary, secondary) — [Medium compatibility guide](https://medium.com/@vici0549/the-definitive-guide-to-pytorch-cuda-and-flash-attention-compatibility-ebec1161ec10)
- The PyTorch 2.11 release notes have no Windows-specific torch.compile/Inductor notes. They list AOTInductor fixes (mixed-device constants, `cpp_kernel_name` public, `mm_type_out` fallback), CUDA-graph partitioning improvements, and Blackwell FP8 items (`_scaled_mm` layout check removed on Blackwell; FP8 blockwise scaling in Inductor Triton templates) — [PyTorch v2.11.0](https://github.com/pytorch/pytorch/releases/tag/v2.11.0)
- PyTorch 2.13 added a CuTeDSL "Native DSL" Inductor backend alongside Triton (faster compilation) — [PyTorch 2.13 blog](https://pytorch.org/blog/pytorch-2-13-release-blog/). 2.14: Inductor gets NVGEMM (CUTLASS kernels generated with CuTeDSL) (search summary) — [PyTorch 2.14 blog](https://pytorch.org/blog/pytorch-2-14-release-blog/)
- Torch-TensorRT versions and targets:

  | Version | Date | PyTorch | TensorRT | CUDA | TRT-RTX |
  |---|---|---|---|---|---|
  | 2.14.0 | 10 Sep | 2.14 | 11.1 | 12.6/13.0/13.2 | 1.6 |
  | 2.13.0 | 28 Jul | 2.13 | 11.0 | 13.0/13.2 | |
  | 2.12.x | | 2.12 | 10.16 | 13.0 | |
  | 2.11.0 | 07 Apr | 2.11 | 10.15 | 12.6/12.8/12.9/13.0 | |

  Windows x86-64 wheels exist for all of these, plus a Windows C++ zip for 2.14. The runtime uses C++ when available and Python otherwise. The FX frontend leaves binary releases in H1 2026. Windows ExecuTorch support is planned for 2.15 — [Torch-TensorRT releases](https://github.com/NVIDIA/Torch-TensorRT/releases)
- Torch-TensorRT first made Windows official in v2.3.0 (Dynamo only, Python runtime at the time); the PyPI page lists "Windows/GPU: Supported (Dynamo only)" (search summary) — [Torch-TensorRT Windows getting started](https://pytorch.org/TensorRT/getting_started/getting_started_with_windows.html); [PyPI torch-tensorrt](https://pypi.org/project/torch-tensorrt)
- GR00T N1.6-3B on RTX 5090, 4 denoising steps:

  | Mode | Data | Backbone | Action head | End-to-end |
  |---|---|---|---|---|
  | TensorRT (action head only) | 2 ms | 18 ms | 11 ms | 31 ms, 32.1 Hz |
  | PyTorch eager | | | | 58 ms, 17.3 Hz |

  Only the DiT action head is optimized with torch.compile or TensorRT, and the backbone stays eager (mirror of the repo docs, not official domain) — [mintlify mirror of Isaac-GR00T TensorRT docs](https://mintlify.wiki/NVIDIA/Isaac-GR00T/deployment/tensorrt)
- GR00T N1.7 (GA) adds "full-pipeline ONNX and TensorRT export". Batch-1 LIBERO with TensorRT runs about 40 Hz on RTX PRO 6000 Blackwell Server and about 28 Hz on Jetson AGX Thor. Deployment scripts are Linux-only (dGPU uses CUDA 12.8). It has a ZMQ server/client — [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- ORT CUDA-graph restrictions are listed in section 3; TRT-RTX enables whole-graph CUDA capture by default — [TRT-RTX benchmarking](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/performance/benchmarking.html)

### Inferences
- **For transformer policies (ACT, SmolVLA action expert, pi0.5/GR00T action heads):**
  - TensorRT (FP16/BF16, with CUDA graph) is the likely winner on all three GPUs. That matches what we already do on the M3 with TRT 10.7.
  - `torch.compile(mode="max-autotune")` plus CUDA graphs ("reduce-overhead") should get close for GEMM-bound parts, but on Windows it depends on an unofficial Triton build, and each PyTorch upgrade needs a matching Triton minor.
  - AOTInductor (`torch._inductor.aoti_compile_and_package`) removes Python from the serving loop but carries the same Triton dependency for GPU kernels. It is not shown to beat TensorRT.
- **For VLAs with a big VLM backbone (GR00T, pi0.5):** the backbone dominates, e.g. 18 of 31 ms for GR00T on the 5090. Export the whole pipeline (N1.7 now does) or use Torch-TensorRT on the backbone too. FP8 on the 5090 is the main lever; on the A4500 and 2070 Super it is FP16/BF16 only.
- **Torch-TensorRT on Windows** is the lowest-friction way to TensorRT for models that are hard to export to ONNX (dynamic control flow in policies). It needs TensorRT 11.x (2.13+), so strong typing applies: cast the model to FP16/BF16 in PyTorch before compiling.

### Gaps
- I found no benchmark of `torch.compile` (triton-windows) vs TensorRT on native Windows for any model; nor did I verify whether triton-windows needs MSVC Build Tools or bundles its own C compiler.
- I found no source for SmolVLA/pi0.5/ACT TensorRT numbers on RTX 5090/A4500/2070 Super; the GR00T N1.6 table comes from a third-party mirror of NVIDIA docs.

## 5. Serving policies, VLAs and LLMs from a Windows box to robots over the LAN

### Takeaway
There is no first-class NVIDIA or community LLM-serving stack on native Windows in 2026:
- TensorRT-LLM has been deprecated on Windows since 0.18 and supports RTX 50 only via WSL.
- vLLM has no official Windows support; the WSL2 route, Docker Model Runner (WSL2) and unofficial forks exist.
- Triton Inference Server's Windows build was a limited beta whose last Windows release seen is 24.03 (2024).

The pragmatic answers:
- **Policies/VLAs:** a small custom native-Windows server (Python + TensorRT/ORT, or GR00T's own ZMQ server pattern) that keeps the engine and CUDA graph warm and sends compact messages over TCP. That is the pattern we already use with ACT on the M3, and GR00T uses with ZMQ.
- **LLMs/VLMs:** llama.cpp `llama-server` natively, with CUDA and NVFP4/MXFP4 on the 5090. Use WSL2 + vLLM/SGLang/TRT-LLM only when batching or throughput features are needed.

WSL2 GPU overhead is described by NVIDIA as "close to near-native" with documented feature limits. Third-party measurements are anecdotal, ranging from about 10% to "roughly halves" depending on workload.

### Cited Findings
- TensorRT-LLM: "Windows platform support is deprecated as of v0.18.0. All Windows-related code and functionality will be completely removed in future releases." 0.17.0 added GeForce RTX 50 "through WSL". The latest notes list 1.4 (1.3 references 2026-09) — [TensorRT-LLM release notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html); [TRT-LLM 0.18.2 overview](https://nvidia.github.io/TensorRT-LLM/0.18.2/overview.html)
- vLLM: per a May 2026 secondary write-up, the official docs list only Linux platforms, there is "no public roadmap" for Windows, and WSL2 is the recommended route (search summary) — [fazm.ai: vLLM Windows 2026](https://fazm.ai/t/vllm-windows-support-2026). Docker Model Runner gained vLLM on Windows (11 Dec 2025) via Docker Desktop/WSL2 — [Docker blog](https://www.docker.com/blog/docker-model-runner-vllm-windows). An unofficial native fork with Windows wheels exists — [SystemPanic/vllm-windows releases](https://github.com/SystemPanic/vllm-windows/releases)
- Triton Inference Server: the first Windows beta came in 2.33.0 (NGC 23.04), with limited features and no CUDA shared memory or metrics. 24.02 added a basic Python backend. The 2.45.0 notes say there is no Windows release for 24.04 and the latest is 24.03 (search summary) — [Triton v2.33.0](https://github.com/triton-inference-server/server/releases/tag/v2.33.0); [Triton releases](https://github.com/triton-inference-server/server/releases/)
- CUDA on WSL2 (guide updated 12 Oct 2026, covers CUDA 13.4): overhead can be "close to near-native".
  - **Limits:** pinned system memory is limited; no full Unified Memory; "Concurrent CPU/GPU access is not supported"; NVML/nvidia-smi lacks utilization and process queries; no OpenGL-CUDA interop.
  - **Setup:** WDDM GeForce/Quadro only, with no TCC. Install only the Windows driver; do not use the `cuda`/`cuda-drivers` metapackages inside WSL.
  - Source: [NVIDIA CUDA on WSL user guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- Anecdotal WSL2 numbers (weak sources):
  - Ollama native Windows vs WSL2 differs by 10-13% tok/s — [insiderllm WSL2 guide](https://insiderllm.com/guides/wsl2-local-ai-windows-guide/)
  - llama.cpp 7B on RTX 4070 Ti: about 68 tok/s native Linux vs about 58 WSL2, with different CUDA versions — [kunalganglani WSL2 vs Linux](https://www.kunalganglani.com/blog/wsl2-vs-native-linux-development)
  - "WSL can roughly halve RTX 5090 speed on a 27B model", with Windows-native llama.cpp between WSL and Linux — [runaihome Qwen3.6 NVFP4 guide](https://runaihome.com/blog/qwen36-27b-nvfp4-blackwell-2x-speed-guide-2026/)
  - About 33% slower GPU work in WSL2 vs Ubuntu — [MSEndpointMgr](https://msendpointmgr.com/2025/12/15/managing-native-linux-vs-wsl-1-2-a-technical-comparison-for-windows-admins/)
- GR00T ships a server/client mode where `Gr00tPolicy` runs on a GPU server and the client connects over ZMQ — [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- TensorRT 11.4 documents CUDA green contexts (CUDA 13.1+) to partition SMs and work queues between workloads, with no memory partitioning or MIG-style isolation — [TensorRT 11.4.0 notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.4.0.html)

### Inferences
- **Lowest LAN latency for a policy:** run natively on Windows, one process per model. Keep the engine and execution context alive and use CUDA graphs and pinned buffers, with IO binding for ORT. Send JPEG/compact tensors over a persistent TCP/ZMQ socket rather than per-request HTTP. A batch-1 policy needs no batching server, so Triton's absence on Windows costs nothing.
- **Serving on shared GPUs (hekate with Isaac Sim + other jobs, Dionysos):**
  - Latency jitter comes from other tenants, not the runtime. Green contexts could reserve SMs for the policy server but do not isolate memory, and require CUDA 13.1+ on the server.
  - On hekate, Isaac Sim (Windows-native) and the server would share the GPU anyway. WSL2 adds a second driver model path with no benefit for a batch-1 TensorRT server.
- **LLM (Qwen3.5 family):**
  - On the 5090, run llama.cpp natively with an NVFP4/MXFP4 or Q4 GGUF; that is the no-WSL path that exploits Blackwell FP4.
  - Use WSL2 + vLLM/SGLang (or TRT-LLM) only if continuous batching or multi-client throughput is needed, and measure the WSL penalty first, since sources disagree widely.
  - Dionysos (Ampere) has no FP4/FP8, so llama.cpp Q4/Q8 GGUF or ORT GenAI INT4 would be the native options. ORT GenAI was not researched here.

### Gaps
- I found no controlled native-Windows vs WSL2 CUDA benchmark for TensorRT or ORT; all WSL numbers are anecdotal LLM measurements.
- Triton Inference Server's Windows status after 24.03 (2024) was not found; it is likely discontinued, but this is unconfirmed.
- I did not check whether SGLang has any Windows story (assumed Linux/WSL only).

## 6. Engine portability: hardware-compatible and version-compatible engines, ONNX as the artifact

### Takeaway
TensorRT plans stay non-portable by default: they are tied to GPU model/compute capability, exact TensorRT version (major.minor.patch.build) and OS. Escape hatches exist and cost speed:
- `kAMPERE_PLUS` hardware compatibility: Ampere+ only, excludes Turing, no FP8/FP4 tactics.
- `kSAME_COMPUTE_CAPABILITY` (10.9+): same-CC only, but allows FP8/FP4.
- Version-compatible engines with an embedded lean runtime: TRT 10 to 11 forward-compatible, and 11.0 to 11.x.
- Cross-OS builds (`setRuntimePlatform(kWINDOWS_AMD64)`).

TensorRT-RTX makes portability the default (one engine for sm_86/89/120/121, JIT-specialized on device), but not for Turing, and its engines are not forward-compatible across TRT-RTX versions. ONNX (with FP16/Q-DQ types baked in, since TRT 11 is strongly typed) is the portable artifact. Build per GPU and per runtime version, and cache.

### Cited Findings
- Default engines run only on the device type they were built on.
  - **`kAMPERE_PLUS`:** checks CC >= 8.0 and excludes non-portable tactics, which can cost throughput/latency.
  - **`kSAME_COMPUTE_CAPABILITY`:** same CC as the build GPU, "generally performs better than kAMPERE_PLUS", and allows FP8 and FP4.
  - **Version compatibility (`kVERSION_COMPATIBLE`):** embeds the lean runtime unless `kEXCLUDE_LEAN_RUNTIME` is set. TRT 10 VC engines run on TRT 11, TRT 8 VC on TRT 10, and 11.0 VC on 11.1+. Other cross-major combinations are unsupported. VC engines "can be slower".
  - **Cross-OS:** engines run only on the build platform unless built with `setRuntimePlatform` (for example `kWINDOWS_AMD64`), which needs a Windows builder resource library and cannot run on the build host.
  - **Practical advice:** build on the smaller-memory GPU and only deserialize trusted plans (plans contain executable code).
  - Source: [TensorRT engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)
- `kSAME_COMPUTE_CAPABILITY` arrived in TensorRT 10.9. Without VC, a plan fails to deserialize if major/minor/patch/build differ (search summary) — [TensorRT 10.9.0 release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.9.0/getting-started/release-notes.html)
- Windows-specific: engines built with TensorRT 10.1.0 may fail to deserialize with 11.0.0 on Windows (error 126); rebuild with 10.2+ — [TensorRT 11.0.0 notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.0.0.html). 11.4 lists a Windows issue with version-compatible engines from 10.1-10.4 that use the RoiAlign plugin — [TensorRT 11.4.0 notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.4.0.html)
- ORT TensorRT EP:
  - Cached engines/profiles must be deleted after any model, ORT, TRT or hardware change.
  - `trt_engine_hw_compatible` makes `sm80+` engines (not Turing/Orin).
  - EPContext models (`_ctx.onnx`) wrap a prebuilt engine, so `nvinfer_builder_resource` is not needed at runtime.
  - Source: [ORT TensorRT EP docs](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- TRT-RTX:
  - Default portability across CC 8.6/8.9/12.0/12.1; Turing needs its own engine — [TRT-RTX support matrix](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/support-matrix.html)
  - "Engines are not forward-compatible" across TRT-RTX upgrades. The runtime cache is keyed on GPU SKU, version, CiG state and driver — [TRT-RTX 1.6 notes](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.6.html)
  - Building the AOT stage for narrower CC targets cut compile time by 16-28% — [NVIDIA TRT-RTX blog](https://developer.nvidia.com/blog/run-high-performance-ai-applications-with-nvidia-tensorrt-for-rtx/)
- TensorRT 11.4 introduces TensorRT-Model-Connect (public preview, separate OSS project), a path from a Hugging Face/local checkpoint to TensorRT inference without an ONNX export step — [TensorRT 11.4.0 notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.4.0.html)
- FP4 ONNX export needs ONNX 1.18 / opset 23 (TRT_FP4QDQ to a two-step DQ pattern) — [NVIDIA FP4 image-generation blog](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus/)

### Inferences
- **Recommended artifact policy for our three GPUs:**
  - Keep one FP32 ONNX source of truth plus derived typed ONNX files: `*_fp16.onnx` from ModelOpt AutoCast, `*_int8qdq.onnx` and `*_fp8qdq.onnx` / `*_nvfp4.onnx` for the 5090 only.
  - Build a plan per GPU (sm_75, sm_86, sm_120) and per TensorRT version, named e.g. `model.<trtver>.<sm>.<prec>.plan`, and keep the ORT EP caches per machine.
  - Hardware-compatible engines are useless for the mixed fleet (Turing excluded, Ampere+ costs speed). Same-CC engines would only help if we had two RTX 5090s.
- **The Jetson Orin robots (TRT 10.7 per our memory notes) cannot share plans with the Windows PCs** (different OS, arch and version). ONNX remains the hand-off between hekate/laptop training and robot builds.
- **Version-compatible engines** would let a TRT 10.x-built plan run under TRT 11 (e.g. ORT 1.24 TRT 10.14 plans on a future TRT 11 runtime), at some speed cost. Better to rebuild, since build time with a timing cache is seconds to minutes.

### Gaps
- I found no quantified slowdown for `kAMPERE_PLUS` or version-compatible engines in NVIDIA docs; only "may be lower/slower".

## 7. Peak runtime + precision per GPU (synthesis of sections 1-6)

### Takeaway
The best-supported fast path on all three GPUs is classic TensorRT (11.4, or 10.14 under ORT 1.24) with a typed ONNX and CUDA graphs:
- **RTX 2070 Super:** FP16 (INT8 Q/DQ for CNN detectors).
- **RTX A4500:** FP16/BF16 (INT8 Q/DQ).
- **RTX 5090:** FP16/BF16 baseline, then FP8 and NVFP4 where GEMMs dominate.

TRT-RTX should be benchmarked on the A4500 and 5090 as a near-equal with much faster build/start. LLMs on Windows: llama.cpp natively (NVFP4/MXFP4 on the 5090).

### Cited Findings
- NVIDIA: classic TensorRT for "highest possible inference throughput"; TRT-RTX for lighter end-user deployment — [NVIDIA TensorRT for RTX page](https://developer.nvidia.com/tensorrt-rtx)
- FP4 vs FP8 vs FP16 measured on the RTX 5090 with TensorRT (FLUX.1-dev, 30 steps): 3,853 / 6,681 / 10,931 ms — [NVIDIA FP4 blog](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus/)
- TensorRT on the GR00T N1.6 action head on the RTX 5090: E2E 58 ms eager to 31 ms — [mintlify mirror of GR00T docs](https://mintlify.wiki/NVIDIA/Isaac-GR00T/deployment/tensorrt)
- Turing exclusions: no NvTensorRTRTX EP / Windows ML NVIDIA EP (Ampere+, RTX 30xx+), and no ORT hardware-compatible engines — [ORT TRT-RTX EP docs](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html); [Windows ML EPs](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers); [ORT TensorRT EP docs](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)

### Inferences
| Model class | RTX 2070 Super (sm_75, 8 GB) | RTX A4500 (sm_86, 20 GB) | RTX 5090 (sm_120, 32 GB) |
|---|---|---|---|
| YOLO-class det/seg | TensorRT FP16; INT8 via ModelOpt Q/DQ if accuracy holds | TensorRT FP16 or INT8 Q/DQ | TensorRT FP16 (INT8 small gain; check the 11.4 YOLOX INT8 accuracy issue); FP8/FP4 not worthwhile for small conv nets |
| ViTs (DA V2/3, DINOv2, SAM 2 encoder) | TensorRT FP16 (FP32 accumulate where accuracy needs it); no BF16 | TensorRT FP16/BF16 | TensorRT FP16/BF16; try FP8 Q/DQ on large (L/G) variants |
| ACT / small policies | TensorRT FP16 + CUDA graph | same | same; FP8 unlikely to help at this size |
| VLAs (GR00T N1.7, pi0.5, SmolVLA) | VRAM-limited (GR00T asks 16 GB+) | TensorRT FP16/BF16 full-pipeline export | TensorRT BF16, then FP8 backbone; NVFP4 for the LLM backbone if accuracy holds |
| LLM/VLM (Qwen3.5) | llama.cpp CUDA, Q4 GGUF | llama.cpp Q4/Q8 | llama.cpp NVFP4/MXFP4 GGUF native; WSL2 + vLLM/TRT-LLM for batching |
| ASR (Parakeet) | ORT CUDA/TRT EP FP16 | same | same |
- The 2070 Super's ORT 1.24.1 + CUDA 12 stack is still current enough. Its TRT EP is pinned to TensorRT 10.14, which still has `--fp16`/implicit INT8, a convenience lost with TensorRT 11.

### Gaps
- Every cell in the table above is an expectation from vendor guidance; there are no measured numbers for our models on these GPUs. The benchmark plan below fills this.

## Benchmark plan for the Windows GPUs

### Takeaway
Use one fixed-shape FP32 ONNX per model and derive typed variants. On each GPU, compare:
1. trtexec (classic TensorRT)
2. the ORT TensorRT EP
3. the ORT CUDA EP (with and without CUDA graph)
4. tensorrt_rtx (Ampere/Blackwell; Turing with a Turing-specific engine)
5. torch.compile (triton-windows) and Torch-TensorRT

Record first-run (build/JIT/capture) separately from steady state, with p50/p99 latency, GPU compute vs host enqueue time, VRAM, clocks and an accuracy delta. On hekate and Dionysos, check free VRAM and other users' processes first (CLAUDE.md: ask before GPU-heavy work when less than ~16 GB is free). Do not lock clocks or change driver settings there without asking.

### Cited Findings (tool behaviour the plan relies on)
- tensorrt_rtx CLI (`tensorrt_rtx.exe`):
  - `--saveEngine --skipInference`, `--loadEngine`, `--profilingVerbosity=detailed --dumpLayerInfo --dumpProfile`, `--rtxCudaGraphStrategy=disable` (CUDA graphs on by default), `--noSpinWait`, `--includeDataTransfers`, `--loadInputs`, `--warmUp`, `--duration`, `--iterations`.
  - Defaults: at least 200 ms warmup, then at least 10 iterations or 3 s. Random inputs by default.
  - Locking clocks: `nvidia-smi -lgc`/`-rgc`; supported clocks: `nvidia-smi -q -d SUPPORTED_CLOCKS`; throttling starts around 85 C, monitor with `nvidia-smi dmon -s pcu`.
  - Report Throughput, GPU Compute Time (excludes transfers) and Enqueue Time (host overhead).
  - Source: [TRT-RTX benchmarking guide](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/performance/benchmarking.html)
- TRT 11 trtexec: no precision flags; convert with `python -m modelopt.onnx.autocast --onnx_path model.onnx --output_path model_fp16.onnx` first — [trtexec 10.x to 11.x migration](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-trtexec.html)
- TRT-RTX: CPU-only AOT may be slower than on-device AOT (`--useGPU` / `kCURRENT`), so measure both. The runtime cache removes JIT on later loads — [TRT-RTX 1.6 notes](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.6.html); [NVIDIA TRT-RTX blog](https://developer.nvidia.com/blog/run-high-performance-ai-applications-with-nvidia-tensorrt-for-rtx/)
- ORT: the CUDA EP CUDA graph needs IO binding and fixed addresses, and its first Run captures. The TRT EP has engine and timing caches and `trt_cuda_graph_enable`. `preload_dlls()` handles Windows DLL search — [ORT CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html); [ORT TensorRT EP](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- On WSL, nvidia-smi lacks utilization/process queries — [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

### Inferences (the plan itself; commands use flags documented above, plus standard trtexec/ORT flags from general knowledge, so verify each with `--help` on the installed version)

**0. Pick the models.** Use fixed batch-1 shapes:
- YOLO26n-seg kit detector, 640x640
- Depth Anything V2 S and L at the close-look resolution
- DINOv2-S/B crop embedder
- SAM 2 image encoder
- our ACT (the slim ONNX already used on the M3)
- optionally the GR00T N1.7 or SmolVLA exported pipeline on hekate

Export FP32 ONNX (opset 17+; opset 23 for FP4) and keep fixed reference inputs (`.npy`/`.bin` of real frames, never zeros) to measure accuracy.

**1. Make the typed ONNX variants (once, on the laptop; ModelOpt is pure Python).**
```powershell
python -m modelopt.onnx.autocast --onnx_path m.onnx --output_path m_fp16.onnx          # FP16 for TRT 11 / TRT-RTX
# INT8 / FP8 explicit Q/DQ (check flag names with: python -m modelopt.onnx.quantization --help)
python -m modelopt.onnx.quantization --onnx_path m.onnx --quantize_mode int8 --calibration_data calib.npy --output_path m_int8.onnx
python -m modelopt.onnx.quantization --onnx_path m.onnx --quantize_mode fp8  --calibration_data calib.npy --output_path m_fp8.onnx   # 5090 only
```
NVFP4 is produced through the ModelOpt torch path and exported to ONNX opset 23, for the 5090 only.

**2. Classic TensorRT (trtexec), each GPU.** TensorRT 11.4 for the standalone runs. TensorRT 10.14 when matching ORT 1.24.1's TRT EP; on 10.x use `--fp16`/`--int8` on the FP32 ONNX instead of AutoCast.
```powershell
trtexec --onnx=m_fp16.onnx --shapes=images:1x3x640x640 --saveEngine=m.trt114.sm75.fp16.plan `
  --timingCacheFile=tc_sm75.cache --skipInference
trtexec --loadEngine=m.trt114.sm75.fp16.plan --shapes=images:1x3x640x640 `
  --useCudaGraph --useSpinWait --noDataTransfers --warmUp=2000 --duration=20 `
  --percentile=50,90,99 --exportTimes=times.json --separateProfileRun --dumpProfile
# portability cost (Ampere/Blackwell only):
trtexec --onnx=m_fp16.onnx --hardwareCompatibilityLevel=ampere+ --saveEngine=m.hwc.plan
```
Record:
- build time (with and without a timing cache)
- engine size
- GPU Compute Time p50/p90/p99
- Enqueue time
- throughput
- the same run with `--noDataTransfers` removed (realistic H2D/D2H)

**3. TensorRT-RTX (A4500, 5090; 2070 Super only with a Turing-specific engine, which needs the compute-capability flag; see `tensorrt_rtx --help`).**
```powershell
tensorrt_rtx --onnx=m_fp16.onnx --shapes=images:1x3x640x640 --saveEngine=m.rtx.plan --skipInference   # CPU-only AOT
tensorrt_rtx --onnx=m_fp16.onnx --shapes=images:1x3x640x640 --useGPU --saveEngine=m.rtx_gpu.plan --skipInference  # on-device AOT
tensorrt_rtx --loadEngine=m.rtx.plan --shapes=images:1x3x640x640 --warmUp=0 --iterations=1     # cold: JIT cost
tensorrt_rtx --loadEngine=m.rtx.plan --shapes=images:1x3x640x640 --warmUp=2000 --duration=20  # steady state
```
- Record the cold first-inference time with an empty runtime cache vs a populated one; check `--help` for the runtime-cache flag name.
- Compare both AOT modes.
- Repeat with `--rtxCudaGraphStrategy=disable` to see the CUDA-graph contribution.
- On Dionysos, first confirm the board is in WDDM (`nvidia-smi -q`, "Driver Model"), because TRT-RTX fails on TCC.

**4. ONNX Runtime EPs (Python harness; the same script on all three GPUs).**
```python
import time, numpy as np, onnxruntime as ort
ort.preload_dlls()                                   # Windows DLL search (ORT >= 1.21)
x_np = np.load("ref_input.npy").astype(np.float32)   # real frame, fixed shape
def bench(providers, n=1000, warm=100):
    t0 = time.perf_counter()
    s = ort.InferenceSession("m.onnx", providers=providers)
    t_create = time.perf_counter() - t0               # includes TRT engine build if cache empty
    io = s.io_binding()
    x = ort.OrtValue.ortvalue_from_numpy(x_np, "cuda", 0)
    io.bind_ortvalue_input(s.get_inputs()[0].name, x)
    outs = []
    for o in s.get_outputs():                         # pre-allocated outputs: fixed addresses for CUDA graphs
        shp = [d if isinstance(d, int) else 1 for d in o.shape]
        dt = np.float16 if "float16" in o.type else np.float32
        v = ort.OrtValue.ortvalue_from_shape_and_type(shp, dt, "cuda", 0)
        io.bind_ortvalue_output(o.name, v); outs.append(v)
    t0 = time.perf_counter(); s.run_with_iobinding(io); t_first = time.perf_counter() - t0
    for _ in range(warm): s.run_with_iobinding(io)
    lat = []
    for _ in range(n):
        t0 = time.perf_counter(); s.run_with_iobinding(io); lat.append(time.perf_counter() - t0)
    lat = np.array(lat) * 1e3
    return dict(create_s=t_create, first_ms=t_first*1e3, p50=np.percentile(lat,50), p99=np.percentile(lat,99))

trt = ("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
       "trt_engine_cache_path": "trt_cache", "trt_timing_cache_enable": True, "trt_cuda_graph_enable": True})
cuda_g = ("CUDAExecutionProvider", {"enable_cuda_graph": "1"})
print("TRT EP ", bench([trt, "CUDAExecutionProvider"]))
print("CUDA EP+graph", bench([cuda_g]))
print("CUDA EP", bench(["CUDAExecutionProvider"]))
# Ampere/Blackwell only, if the TensorRT-RTX plugin EP is built and registered:
# ort.register_execution_provider_library("NvTensorRTRTX", r"path\to\plugin.dll")  # see TensorRT-RTX-EP-ABI docs
```
- Run each configuration twice: the first run has an empty engine cache (records build time), the second a warm cache.
- On ORT 1.24.1 the TRT EP needs TensorRT 10.14 DLLs (e.g. `pip install tensorrt-cu12==10.14.*`); a TensorRT 11 install will not load. For outputs with dynamic dims, use real output shapes instead of the `1` placeholder.
- `onnxruntime_perf_test` gives the same numbers, e.g. `onnxruntime_perf_test -e tensorrt -I -r 1000 -i "trt_fp16_enable|true trt_cuda_graph_enable|true" m.onnx`. It is not in the pip wheel (build from source), so the Python harness is the default.

**5. PyTorch: torch.compile and Torch-TensorRT.** Install the matching `triton-windows` minor for the installed torch first.
```python
import torch, torch_tensorrt
m = load_model().eval().cuda().half(); x = torch.from_numpy(x_np).cuda().half()
def t(fn, n=1000, warm=50):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    with torch.inference_mode():
        for _ in range(warm): fn(x)
        torch.cuda.synchronize(); ts = []
        for _ in range(n):
            s.record(); fn(x); e.record(); e.synchronize(); ts.append(s.elapsed_time(e))
    return sorted(ts)[n//2], sorted(ts)[int(n*.99)]
print("eager", t(m))
mc = torch.compile(m, mode="max-autotune")            # includes CUDA graphs ("max-autotune" uses cudagraphs; also try "reduce-overhead")
print("compile", t(mc))                               # time the FIRST call separately: compile/autotune cost
mt = torch_tensorrt.compile(m, ir="dynamo", inputs=[x])  # TRT 11: precision follows the (FP16) model dtype
print("torch-trt", t(mt))
```

**6. What to record for every run** (one CSV row per GPU x model x runtime x precision):
- **Versions:** hostname, GPU, driver, CUDA runtime, TRT/TRT-RTX/ORT/torch versions.
- **Setup:** precision, input shape, CUDA graph on/off.
- **Cold start:** build/compile time, first-inference latency (JIT/capture), engine/cache size.
- **Steady state:** p50/p90/p99 latency, GPU compute time vs enqueue time (trtexec/tensorrt_rtx), throughput.
- **Resources:** peak VRAM (`nvidia-smi --query-gpu=memory.used,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv -lms 200` during the run), and other GPU processes present (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`).
- **Accuracy vs FP32 reference:**
  - detector: mAP50 on the kit val set
  - depth: abs-rel / mm error on saved close looks
  - DINOv2: cosine to FP32 embedding
  - ACT: action MSE on recorded episodes

**7. Pitfalls**
- **Clocks/boost:** GeForce boost and thermals vary run to run. On the laptop, lock clocks (`nvidia-smi -lgc`, needs admin) or at least log clocks and temperature and use the High performance power plan. On hekate/Dionysos, do not lock clocks (a system change on shared machines; ask first). Instead interleave configurations (A B C A B C) over several rounds and report medians.
- **Shared VRAM and SMs:** Isaac Sim and others' training on hekate, and other users on Dionysos, change latency and can OOM the engine build (TensorRT's builder workspace). Check free VRAM first, and ask before heavy runs when less than ~16 GB is free. Note concurrent processes in every row. Expect p99 to be dominated by neighbours, not the runtime.
- **First-run costs:**
  - ORT TRT EP builds the engine in Session creation (minutes for big ViTs without a timing cache).
  - TRT-RTX JIT-compiles on first inference (fallback kernels first; FLUX step 1 took about 270 ms vs about 100 ms later) unless the runtime cache is warm.
  - torch.compile autotunes on the first call.
  - CUDA graph capture happens on the first Run.
  - Always report cold and warm separately.
- **TRT 11 strong typing:** an FP32 ONNX in trtexec 11 runs in FP32 (slow) with no warning about `--fp16` absent. Make sure the FP16 file is the one benchmarked. Watch for FP16 overflow in ViTs (Depth Anything, DINOv2): compare against FP32 outputs.
- **Engine/version pinning:** delete ORT `trt_cache` after any TRT/ORT/driver change. TRT-RTX caches are keyed on driver version too.
- **Windows specifics:** WDDM scheduling and Hardware-Accelerated GPU Scheduling can add jitter, and a display attached to the GPU (the laptop) adds compositor load. Antivirus scanning of freshly written engine/cache files can inflate cold-start times. Run from a local disk, not a synced folder.
- **Transfers:** report both `--noDataTransfers` (kernel speed) and end-to-end with H2D/D2H (what the robot sees). Use pinned host buffers and IO binding in the ORT harness.
- **WSL:** if WSL2 is tried for vLLM/TRT-LLM, nvidia-smi inside WSL lacks utilization data, so sample from the Windows side.

### Gaps
- Exact flag names for the TRT-RTX runtime cache and Turing-target build, the ModelOpt ONNX PTQ CLI, and the plugin-EP registration call were not verified against the installed versions; check each with `--help` or the docs before running.
- This plan has not been executed on any of the three machines (desk research only).
