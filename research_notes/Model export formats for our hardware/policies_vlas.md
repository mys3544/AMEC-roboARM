# Fastest export and inference paths for robot policies and VLAs (state as of 2026-10-27)

Context for the reader (from the repo, not the web): the M3 runs ACT (LeRobot 0.6.1, chunk 20, a new
chunk every 10 ticks at 10 Hz) as ONNX -> onnxslim -> TensorRT 10.7 FP16 on the Orin NX 16 GB Super:
5.9 ms per inference, max 0.19 deg off LeRobot's own chunks on held-out frames, 12 ms median for a full
request through the local stdlib HTTP service (JPEG + base64 + HTTP included), picture 47 ms old
(median), loop 10.0 ticks/s with 0 late (NEXT.md; m3pro/tools/policy_service.py). TRT 10.7 refused the
raw export (Myelin "should never be called on mod_t tensors" on Squeeze/Unsqueeze tokens) until
onnxslim. Plans: GR00T N1.7 (LeRobot `groot`) served from the 5090, pi0.5 second, SmolVLA if a VLA must
run on board, maybe X-VLA.

Hardware facts used throughout:
- Orin NX 16 GB Super mode: 1024 CUDA cores, 32 Tensor Cores, GPU up to 1173 MHz, 157 TOPS sparse / 78 dense INT8 — [NVIDIA JetPack 6.2 Super mode blog](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules) (the table's column alignment was garbled in the fetched text; double-check). Memory: 16 GB 128-bit LPDDR5, 102.4 GB/s, shared by CPU and GPU — [Turing Pi Orin NX specs](https://docs.turingpi.com/docs/nvidia-jetson-orin-nx-intro-specs). For LLMs, Super mode gains only about 1.1x on the 16 GB NX (Llama 3.1 8B: 20.5 -> 22.8 tok/s) — [NVIDIA blog](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules).
- "Jetson Orin" in NVIDIA's GR00T tables is the 64 GB AGX Orin ("Orin, 64 GB shared") — [GR00T hardware recommendation](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md). AGX Orin bandwidth is 204.8 GB/s — [EagleVLA paper, Table 2](https://arxiv.org/html/2607.12659).
- **Inference used everywhere below:** the Orin NX has about half the AGX Orin 64 GB's memory bandwidth (102.4 vs 204.8 GB/s) and, as I recall, half its GPU (1024 vs 2048 CUDA cores; I did not re-verify the AGX core count). So **AGX Orin numbers should be roughly doubled for our Orin NX.** That is an estimate, not a measurement.
- Precision support by GPU (general CUDA knowledge, not re-verified in this session): the Orin NX (sm_87) and the A4500 (sm_86) are Ampere, so FP16/BF16/INT8 but no FP8 or FP4 tensor cores. The RTX 2070 Super (sm_75, Turing) has FP16 and INT8 but no BF16 tensor cores. The RTX 5090 (sm_120, Blackwell) has FP8 and NVFP4. Consistent with this, FlashRT ships only INT8/BF16 paths for sm_87 and FP8/FP4 paths for the 5090 — [FlashRT README](https://github.com/LiangSu8899/FlashRT).

## 1. NVIDIA GR00T N1.x (N1.5 / N1.6 / N1.7): official deployment path, Orin vs RTX 5090, server

### Takeaway
The official path for N1.7 is Isaac-GR00T's `scripts/deployment/build_trt_pipeline.py`. It exports ONNX, builds TensorRT engines for the whole model (ViT + LLM backbone + DiT action head, the "full_pipeline" mode), verifies the result and benchmarks it, in BF16 with the ViT kept in FP32. FP8 and mixed NVFP4 are offered only on Thor. Measured by NVIDIA, a full policy call takes 149 ms on AGX Orin (TensorRT, JetPack 7.2) against about 23–25 ms on H100 or RTX Pro 6000 Blackwell. On our Orin NX on JetPack 6.2, only the older DiT-only TensorRT mode works and would take about 300–450 ms (estimated), so N1.7 belongs on the 5090. There the TensorRT number is about 25 ms (by analogy with the RTX Pro 6000), and third-party kernels report 9–14 ms. The stock server is ZeroMQ + msgpack.

### Cited Findings
**Export tooling and modes (N1.7, Isaac-GR00T main):**
- `build_trt_pipeline.py --model-path <ckpt> --dataset-path <data> --embodiment-tag <TAG> --flat-e2e` runs `--steps export,build,verify,benchmark`. Component scripts: `trt/export_onnx_n1d7.py`, `trt/build_tensorrt_engine.py`, `trt/verify_n1d7_trt.py` (PyTorch vs TRT action check), `trt/trt_model_forward.py` (CUDA graphs on the action head), and `trt/n1d7_optimization_config.py` with `n1d7_calibration.py` (FP8/NVFP4 calibration) — [Isaac-GR00T deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md).
- There are three modes. `dit_only` (the legacy N1.6 default: only the DiT in TRT), `action_head` (state/action encoders + DiT + decoder in TRT), and `full_pipeline` (backbone ViT + LLM and action head in TRT). `full_pipeline` is the recommended N1.7 default "because the backbone dominates inference time" — [deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md).
- `--precision` supports only `bf16`. The ViT runs in FP32 for accuracy. Thor adds `--quantization fp8|mixed_nvfp4` (needs ModelOpt and calibration data) — [deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md).
- Pitfalls the README lists:
  - Engines are architecture-specific.
  - Batch is static.
  - `--n-envs 1` (the ViT engine has static shapes).
  - Use the legacy ONNX exporter (`dynamo=False`), because the dynamo exporter can break dynamic sequence lengths.
  - Builds take 2–5 min.
  - TRT 10.16.1.11 is pinned on Spark because 10.15.1 fails to compile the optimized action encoder at batch 2.
  - On aarch64, use plain `python`, not `uv run`.

  Source: [deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md).
- Accuracy: verification expects cosine >= 0.999. LIBERO (H100, 20 episodes, one task): PyTorch 20/20, TRT full pipeline 19/20, "within simulation noise" — [deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md).

**Platform support / versions:**
- Isaac-GR00T main now requires **JetPack 7.2 / Jetson Linux 39.2 on Orin, and "JetPack 6.x is no longer supported"**. Stack: Python 3.12, CUDA 13.2, PyTorch 2.13.0, flash-attn rebuilt for sm_87 — [deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md).
- JetPack 7.2 is the first JetPack 7.x for Orin. It ships Jetson Linux 39.2, Ubuntu 24.04, CUDA 13.2.1, cuDNN 9.20 and TensorRT 10.16.2; a 7.2.1 patch (L4T 39.2.1) is reportedly dated 2026-08-11 (unconfirmed by NVIDIA) — [RidgeRun JetPack 7.2 components](https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin/JetPack_7.2/Getting_Started/Components), [NVIDIA JetPack 7.2 archive](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2). Engines built on JP 6.2 must be rebuilt on 7.2 — [Seeed JetPack 7.2 guide](https://www.seeedstudio.com/blog/2026/09/04/jetpack-7-2-full-guide-deploy-on-recomputer-jetson/).
- NVIDIA calls Orin a "legacy edge option… only suitable for slow, non-reactive tasks". Thor (12.4 Hz with TRT) is the recommended robot-mounted target. Inference needs a GPU with 16 GB+; default fine-tuning needs 40 GB+ and peaks at about 35 GB — [hardware recommendation](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md).

**Measured latency, N1.7.** Official, median full policy call, batch 1, horizon 40, 4 denoising steps, two 256x256 cameras, preprocessing and decoding included, RPC excluded — [deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md):

| Device | Eager | torch.compile | TRT full pipeline | TRT + mixed NVFP4 |
|---|---|---|---|---|
| H100 | 101.5 | 53.4 | 22.9 | — |
| RTX Pro 6000 Blackwell | 75.7 | 52.1 | 24.7 | — |
| L40 | 123.7 | 66.8 | 35.0 | — |
| DGX Spark | 200.8 | 183.1 | 116.6 | — |
| AGX Thor | 106.8 | 100.5 | 76.8 | 35.6 |
| Jetson (AGX) Orin | 357.9 | 224.3 | 149.0 | — |

- The older HF model card (1 camera) gives Orin eager 342.8 ms, torch.compile 217.0 ms and TRT **DiT-only 216.5 ms**, with the note that "TensorRT 10.3 does not support the backbone engine" on Orin. It also lists Thor TRT 93.8 ms and H100 TRT 27.9 ms — [nvidia/GR00T-N1.7-3B card](https://huggingface.co/nvidia/GR00T-N1.7-3B). The hardware page gives Orin 2.9 Hz eager and 6.6 Hz TRT — [hardware recommendation](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md). The sources differ because they come from different releases and camera counts.
- N1.6 on Orin: one NVIDIA page says 300 ms eager and 173 ms TRT. Another says 2.9 Hz eager and 4.6 Hz TRT, both DiT-only — [GR00T TensorRT docs mirror](https://nvidia-isaac-gr00t.mintlify.app/deployment/tensorrt). A user measured about 338 ms in PyTorch on AGX Orin 64 GB against NVIDIA's 220 ms chart, and suspected that CPU-side transforms and the tokenizer inside `get_action` added the difference — [Isaac-GR00T issue #410](https://github.com/NVIDIA/Isaac-GR00T/issues/410).

**Model facts:**
- N1.7 is 3B parameters in BF16. The VLM backbone is Cosmos-Reason2-2B; the action head is a flow-matching DiT with AdaLN and per-embodiment MLP encoders/decoders — [model card](https://huggingface.co/nvidia/GR00T-N1.7-3B). The deployment README describes it as "Qwen3-VL-based… 24 ViT blocks, 16 LLM layers, 32-layer DiT" — [deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md). As far as I know Cosmos-Reason2 is Qwen3-VL-based, so the two descriptions are probably consistent; not verified.
- License conflict: the HF card says NVIDIA Open Model License — [model card](https://huggingface.co/nvidia/GR00T-N1.7-3B); a third-party summary says Apache 2.0 — [letsdatascience](https://letsdatascience.com/news/nvidia-brings-isaac-gr00t-17-to-lerobot-c971cae7). Trust the card.
- GR00T 1.7 came to LeRobot (the `groot` policy) around 2026-07-06 — [NVIDIA forum announcement](https://forums.developer.nvidia.com/t/nvidia-and-hugging-face-bring-new-models-and-frameworks-to-lerobot-for-the-open-robotics-community/376499), [letsdatascience](https://letsdatascience.com/news/nvidia-brings-isaac-gr00t-17-to-lerobot-c971cae7). A course using LeRobot `groot` suggests training chunk_size 40, starting with n_action_steps 20, RTC execution_horizon 20 and queue_threshold <= 5 — [Seeed reBot course ch. 22](https://wiki.seeedstudio.com/robotics/rebot_physical_ai_course_chapter_22/).

**Blackwell / RTX 5090 (third-party, unverified):**
- FlashRT, an Apache-2.0 hand-written CUDA kernel engine (news item dated Oct 2026), reports on RTX 5090:
  - GR00T N1.6: 12.53 ms (T=16) and 13.08 ms (T=50).
  - GR00T N1.7: **14.0 ms FP8, 8.64 ms with an NVFP4 action head** (2 views).
  - Thor: N1.6 41–45 ms.

  Sources: [FlashRT README](https://github.com/LiangSu8899/FlashRT), [NVIDIA forum post, 2026-05-02](https://forums.developer.nvidia.com/t/real-time-inference-on-thor-rtx-pi0-5-gr00t-n1-6-1-7-thor-23-hz-rtx-5090-50-80hz/368788). No accuracy check was posted for GR00T.
- rMuscle (a cross-execution cache; arXiv 2609.19104, v3 2026-10-21) reports GR00T N1.6 at 51.0 Hz on RTX 4090 and 17.5 Hz on Thor — [search snippet of arXiv 2609.19104](https://arxiv.org/pdf/2609.19104). The abstract only claims 1.29–1.42x — [arXiv abs](https://arxiv.org/abs/2609.19104).

**Server (Isaac-GR00T):**
- `gr00t/eval/run_gr00t_server.py --model-path <ckpt> --embodiment-tag <tag> --device cuda:0 --host 0.0.0.0 --port 5555`. ZeroMQ (Nagle off by default) with MessagePack; numpy arrays go as `.npy` bytes, so images are raw, not JPEG.
- The client is `PolicyClient(host, port, timeout_ms=15000, strict)` with `ping/get_action/reset/get_modality_config`, and there is an optional `api_token`. A model-free replay server (`--dataset-path … --execution-horizon 8`) exists, which is useful for pure network tests.
- Source: [GR00T server-client docs mirror](https://mintlify.wiki/NVIDIA/Isaac-GR00T/deployment/server-client), [Isaac Lab Arena GR00T quickstart](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/running_a_real_policy/gr00t.html). No official latency figure exists for the ZMQ hop.

### Inferences
- Our Orin NX (JP 6.2, TRT 10.7) can only run N1.7 the legacy way: PyTorch backbone plus DiT-only TRT. That is about 216 ms on AGX Orin, so roughly 400+ ms on the NX. Even full_pipeline TRT on JP 7.2 would be about 300 ms (2 x 149). At 10 Hz that is an inference delay of d = 3–5 ticks, which async/RTC can carry, but NVIDIA itself labels it "slow, non-reactive". **Serve N1.7 from hekate's 5090.**
- On the 5090, expect about 75 ms eager, about 50 ms torch.compile and about 25 ms TRT full pipeline, by analogy with the RTX Pro 6000 Blackwell (I believe both are GB202; not verified). Then d = 1 tick at 10 Hz, even with 10–30 ms of WiFi on top. FP8 or NVFP4 on the 5090 is possible via FlashRT, but NVIDIA's own recipe exposes quantization only for Thor. Gain about 10 ms, which does not matter at 10 Hz.
- Moving the robots to JetPack 7.2 means Ubuntu 24.04 (so a ROS 2 Jazzy migration) and new Yahboom carrier support. Both are unknown. It is a big change made only to run GR00T on board, and nothing above suggests the result would be fast enough.
- Our Windows servers: Isaac-GR00T is documented for Linux only (Ubuntu 22.04+ for dGPU), so on hekate and Dionysos it probably means WSL2 or Docker-in-WSL2. That needs a port-forwarding or mirrored-network decision for robot access; see question 6.

### Gaps
- No official RTX 5090 (consumer) or RTX A4500 number for N1.6/N1.7; the A4500 (Ampere, 20 GB) fits inference memory-wise (16 GB+ requirement) but is likely slower than the L40's 35 ms TRT (unverified).
- No measured GR00T number on any Orin NX 16 GB; no peak-memory figure for N1.7 inference on Jetson.
- Whether TRT 10.7 (our JetPack 6.2) can build the N1.7 backbone engine is not stated (only "TRT 10.3 does not support it").
- N1.5 numbers were not collected (superseded twice).
- No measured ZeroMQ round-trip figures.

## 2. Physical Intelligence pi0 / pi0.5 / pi0-FAST: fastest inference, and can it run on Orin NX 16 GB

### Takeaway
On a desktop GPU, pi0 and pi0.5 run at 13–30 ms per chunk with optimized kernels: Dexmal's Triton/CUDA-graph realtime-vla takes 27 ms on a 4090 for 2 views, and FlashRT's FP8/FP4 on the 5090 takes 9–13 ms. openpi JAX takes about 54 ms and naive PyTorch about 106 ms. On Jetson, all published numbers are AGX Orin 64 GB or Thor; the best AGX Orin result is 92 ms (pi0.5, W8A16 INT8 + fused kernels + CUDA graph, FlashRT port). pi0.5 at FP16 (about 6.5 GB) should fit the NX's memory, but expect about 200 ms with heroic kernels and 400+ ms in BF16 PyTorch (estimates). So serve pi0.5 from the 5090 over websocket, with RTC.

### Cited Findings
- "Running VLAs at Real-time Speed" (Dexmal/StepFun, arXiv 2510.26742, 2025-10-30), pi0, RTX 4090, 10 flow steps, chunk 63:

  | | 1 view | 2 views | 3 views |
  |---|---|---|---|
  | Naive PyTorch | 105.0 ms | 106.5 ms | 113.9 ms |
  | openpi JAX | 43.8 ms | 53.7 ms | 67.6 ms |
  | Optimized | **20.0 ms** | **27.3 ms** | **36.8 ms** |

  CUDA graphs alone gave about 2x. Graph simplification (folding norm affine params and the time encoder, fusing QKV + RoPE) saved 7–8 ms. Triton GEMM tuning saved about 1.5 ms, fused gated FFN about 1.7 ms, fused scalar ops about 4 ms. The 480 Hz "full streaming" idea is only a sketch. Code: github.com/Dexmal/realtime-vla — [arXiv 2510.26742](https://arxiv.org/html/2510.26742v1).
- openpi added PyTorch support (pi0 and pi0.5, not pi0-FAST) in Sept 2025. With torch.compile, inference speed is "comparable between JAX and PyTorch". Inference needs > 8 GB (RTX 4090 example); LoRA fine-tuning > 22.5 GB; tested only on Ubuntu 22.04; remote serving over websocket on port 8000 — [openpi README](https://github.com/Physical-Intelligence/openpi).
- RTC paper (PI, NeurIPS 2025): pi0.5, RTX 4090, BF16, **5 denoising steps: 76 ms vanilla, 97 ms with RTC guidance** (+21 ms). Bidirectional decoding with N=16 takes 115–223 ms — [arXiv 2506.07339v2](https://arxiv.org/html/2506.07339v2).
- EagleVLA (arXiv 2607.12659, CoRL 2026 spotlight, v7 2026-10-23), pi0.5 latency:
  - 76.4 ms on RTX 4090, 128.2 ms on A6000, 457.9 ms on Thor.
  - **Orin: 2289.4 ms (30 W) and 1420.8 ms (50 W) naive PyTorch.**
  - Their llama.cpp-based engine with CUDA graph reuse, GPU-resident buffers and flow unrolling gets Orin to 271.1 ms total and 10.3 Hz control (97 ms "reaction time") via async scheduling.
  - The 30/50 W power modes suggest AGX Orin.

  Source: [arXiv 2607.12659 HTML](https://arxiv.org/html/2607.12659). An earlier version (titled "Jetson-PI") reported 0.7 -> 6.06 Hz — [besthub summary](https://www.besthub.dev/articles/jetson-pi-enables-real-time-vla-on-robots-boosting-jetson-orin-control-frequency-8-66-ab80c293bddc).
- hous-lab/pi05-orin-edge, Jetson **AGX Orin 64 GB**, L4T 36.5, CUDA 12.6, torch 2.8 (FlashRT port):
  - Model p50 92.4 ms (locked clocks: p99 93.9 ms).
  - Client p50 113–150 ms over the Jetson USB link.
  - BF16 client p50 219–222 ms.
  - Method: W8A16 channel-split INT8 on the gate/up projections plus a fused GEGLU kernel. W8A8 was dropped as **2.73x slower than FP16 GEMM on Orin**.
  - LIBERO-10: 92.2% (INT8+fused) vs 91.8% (BF16) vs 92.4% openpi reference over 500 episodes.
  - Server is `server_flashrt_official_ws.py` speaking the openpi protocol.

  Source: [hous-lab/pi05-orin-edge](https://github.com/hous-lab/pi05-orin-edge).
- Jetson AI Lab "OpenPi pi0.5 on Thor" (JetPack 7.2), pipeline JAX -> PyTorch -> ONNX (ModelOpt FP8/NVFP4) -> trtexec (10–30 min build) -> CUDA-graph replay:
  - PyTorch BF16: about 132 ms total.
  - TRT FP8: about 54 ms.
  - TRT FP8 + NVFP4: about 49 ms.
  - Cosine vs PyTorch: 0.9995 (FP8), 0.9946 (FP8+NVFP4).
  - Orin is not covered.

  Source: [Jetson AI Lab tutorial](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor).
- FlashRT (Apache-2.0):
  - pi0.5 on RTX 5090: **13.10 ms FP8, 8.75 ms FP4** (2 views).
  - pi0 on RTX 5090: 21.16 ms FP8.
  - pi0.5 on Thor: 40.71 ms OpenPI FP8, 20.33 ms NVFP4 + FA4.
  - pi0-FAST: 2.39 ms/token on the 5090, 8.1 ms/token on Thor.
  - LIBERO-10: 93.32% NVFP4 on the 5090 (5,000 episodes) vs 94.29% reference.
  - Orin (sm_87) is a "community port", pi0.5 INT8/BF16 only.
  - API: `flash_rt.load_model(checkpoint=…, config="pi05", framework="torch")`.

  Sources: [FlashRT README](https://github.com/LiangSu8899/FlashRT), [NVIDIA forum](https://forums.developer.nvidia.com/t/real-time-inference-on-thor-rtx-pi0-5-gr00t-n1-6-1-7-thor-23-hz-rtx-5090-50-80hz/368788). The same author reports an MLIR-TRT pi0.5 at about 70 ms (device not named) — [forum](https://forums.developer.nvidia.com/t/real-time-inference-on-thor-rtx-pi0-5-gr00t-n1-6-1-7-thor-23-hz-rtx-5090-50-80hz/368788).
- Memory, from a third-party blog (2026-05-22): pi0 12.5 GB and pi0.5 FP32 13 GB, i.e. "does not fit" the NX in FP32. A SnapFlow-distilled pi0.5 student (1 step) is 6.5 GB in FP16 and scored 96.7% vs the teacher's 93.3% on LIBERO-10 with n=30 — [jared-hpc.com](https://jared-hpc.com/posts/vla-physical-ai-tensorrt-tether/). The blog lists "Pi0 INT8" as fitting the NX.
- pi0/pi0.5 depend on CUDA 12.x; Thor's CUDA 13 caused problems (NVIDIA advised JP 7.0/7.1) — [NVIDIA forum thread](https://forums.developer.nvidia.com/t/how-to-run-a-pi0-or-pi0-5-model-on-agx-thor-device/362389).
- VLA-Perf (NVIDIA Research, arXiv 2602.18397, 2026-02-20) is a roofline model calibrated on Dexmal's pi0: real systems reached 73–83% of the roofline. Modeled pi0: RTX 4090 31.1 ms, Thor 52.6 ms. Their conclusion: "server-side inference, even with an RTX 4090 over WiFi, beats on-device Jetson Thor in most cases" — [arXiv 2602.18397](https://arxiv.org/html/2602.18397).

### Inferences
- **Can pi0.5 run on the Orin NX at all?** Memory-wise yes in FP16/BF16 (about 6.5–7 GB), but not alongside much else: the NX has about 14 GB usable and our robots also run YOLOE, Depth Anything TRT, the kit detector and ROS 2. Latency-wise, halving the AGX Orin results gives about 185 ms (FlashRT INT8, the best case), about 440 ms (BF16 PyTorch client) and about 540 ms (EagleVLA). At 10 Hz that is d = 2–6 ticks: possible with RTC, but slow to react, and it needs a community kernel stack pinned to L4T 36.5 / CUDA 12.6 / torch 2.8.
- **Fastest practical path for us:** openpi or LeRobot `pi05` on the 5090. Turn on torch.compile (LeRobot's `--use_torch_compile`) first; that should land near openpi JAX (about 50 ms) and gives d = 1 at 10 Hz. Only go to FlashRT FP8/FP4 (about 9–13 ms) if reaction time matters; at 10 Hz it does not.
- pi0-FAST is autoregressive: latency scales with the token count, it is not in openpi PyTorch, and RTC (inpainting) does not apply. Low priority for us.

### Gaps
- No pi0/pi0.5 measurement on any Orin NX; no measured peak memory for pi0.5 BF16 inference on Jetson.
- I did not find LeRobot's own `pi05` latency (PyTorch port) on any GPU; openpi PyTorch numbers were not published, only "comparable to JAX with torch.compile".
- Tokens per chunk for pi0-FAST on our 6-DoF chunk were not found.

## 3. SmolVLA and other small on-board VLAs (X-VLA, VLA-Adapter, TinyVLA, NORA, MiniVLA, EdgeVLA): size, export, Jetson numbers

### Takeaway
SmolVLA (450M; about 1.6–2 GB in memory) is the only VLA in our list that comfortably fits on the Orin NX next to our other models. The only Jetson measurement I trust is about 190 ms per chunk on an Orin Nano 8 GB: ORT + TensorRT EP FP16 over 9 split ONNX graphs, JetPack 7.2, about 90 ms of it vision. Expect about the same or slightly better on our NX, i.e. 10 Hz control only with async/RTC (d = 2). There is no official SmolVLA ONNX/TRT export, and naive exports have silently changed closed-loop behaviour. X-VLA (0.9B) is in LeRobot but has no Jetson Orin number (19.7 Hz on Thor, third-party). For VLA-Adapter, TinyVLA and NORA I found no Jetson numbers.

### Cited Findings
- SmolVLA paper (arXiv 2506.01844): 450M parameters, designed to run on consumer GPUs or even CPUs; an async inference stack — [HF papers page](https://huggingface.co/papers/2506.01844). Async gave about 2x faster task completion at comparable success, and sub-100 ms round trip on a LAN with an RTX 4090 — [HF async blog, 2025-07-10](https://huggingface.co/blog/async-robot-inference).
- Split ONNX bundle (eetmie/smolvla-base-onnx), from `lerobot/smolvla_base` (LeRobot 0.5.1, opset 17), on **Jetson Orin Nano 8 GB**, JetPack R39.2.1, MAXN_SUPER, ORT 1.24.0 + TRT 10.16.2:
  - 9 graphs: 7 on TRT, 2 on CPU. FP16, IOBinding.
  - p50 **189.9 ms (5.25 Hz)**, p95 190.9 ms; vision 91.2 ms, denoise 59.3 ms; 1,871 MB resident.
  - vs LeRobot FP32: cosine >= 0.99934, action error median 0.10% / max 2.05% of range (synthetic scene; the author warns real frames may show about 3x more FP16 drift).

  Source: [HF model page](https://huggingface.co/eetmie/smolvla-base-onnx). Unreviewed single source.
- Conflict: a blog claims SmolVLA FP16 ORT-TRT at "about 25 ms p50 on Orin Nano" (about 20 ms INT8) and SmolVLA monolithic at 19.5 ms with ORT-TRT FP16 vs 108 ms with ORT CUDA on an A10G. It also reports CUDA graphs on `expert_denoise` at 3.0–3.8x on Ampere — [jared-hpc.com, 2026-05-22](https://jared-hpc.com/posts/vla-physical-ai-tensorrt-tether/). The 25 ms Orin Nano figure contradicts the 190 ms bundle measurement and is not reproduced elsewhere; treat it as unverified, perhaps a partial pipeline.
- "When Faster VLA Deployment Changes Closed-Loop Behavior" (arXiv 2609.14146, 2026-09-12), SmolVLA, LeRobot 0.6.1, RTX 2060:
  - p99: PyTorch+AMP 1181 ms vs ONNX 601 ms ("FP16") / 532 ms ("INT8").
  - But the "FP16"/"INT8" artifacts were byte-identical FP32 graphs.
  - LIBERO-Spatial success fell from 70% to 41% under ONNX, traced mostly to a 16-token language width; width 24 restored 75%.
  - The TensorRT EP build did not finish within 6 GB.

  Source: [arXiv 2609.14146](https://arxiv.org/abs/2609.14146).
- Arm's learning path exports SmolVLA to one ONNX graph (`scripts/export_onnx.py`, 6 preprocessed inputs including tokenized language and images) and validates it against PyTorch — [Arm learning path](https://learn.arm.com/learning-paths/cross-platform/smolvla-onnx-conversion/convert-and-validate-onnx/).
- SmolVLA inference memory: about 2 GB, against about 14 GB for PI0 (as cited by a third-party blog) — [particula.tech, 2026-10-08](https://particula.tech/blog/real-time-chunking-vs-async-inference-lerobot).
- X-VLA:
  - 0.9B with per-embodiment soft prompts, in LeRobot via `pip install "lerobot[xvla]"`, checkpoints `lerobot/xvla-base`, `-libero`, `-folding` — [lerobot/xvla-base card](https://huggingface.co/lerobot/xvla-base).
  - rMuscle reports X-VLA at 35.7 Hz on RTX 4090 and 19.7 Hz on Thor — [arXiv 2609.19104 (search snippet)](https://arxiv.org/pdf/2609.19104).
  - FlashRT supports X-VLA on RTX SM120/121, torch only — [FlashRT](https://github.com/LiangSu8899/FlashRT).
  - X-VLA tolerates W4 and W8A8 (see question 7).
- Other small edge VLAs:
  - LiteVLA-Edge: AGX Orin, 150.5 ms mean (about 6.6 Hz), 4-bit GGUF after FP32 fine-tuning — [arXiv 2603.03380](https://arxiv.org/html/2603.03380).
  - EdgeVLA targets high-end edge GPUs such as AGX Orin (no Hz given) — [same paper](https://arxiv.org/html/2603.03380).
  - MiniVLA reports ONNX/TRT export of its vision encoder for Orin Nano (no Hz in what I saw) — [MiniVLA listing](https://app.alphaneural.io/assets/xintaozhen/MiniVLA).
  - MolmoAct-7B profiled on Orin/Thor: up to 75% of latency is memory-bound action generation — [arXiv 2603.02271](https://arxiv.org/abs/2603.02271).
- Recommendation from the Jetson blog: optimize the denoise loop (it runs every call) before the VLM, use CUDA graphs on the expert, export off-Jetson on Python 3.12 (LeRobot needs 3.12 while JetPack 6 ships 3.10), pin `numpy<2` on aarch64 — [jared-hpc.com](https://jared-hpc.com/posts/vla-physical-ai-tensorrt-tether/).

### Inferences
- The Orin NX has the same 102 GB/s bandwidth as the Orin Nano Super and a slightly higher GPU clock (as I recall: 1024 cores at about 1.17 vs about 1.02 GHz; not re-verified). So SmolVLA on our NX via the split-ONNX TRT route should be about 150–190 ms. That is d = 2 at 10 Hz: workable with RTC or async, not synchronous.
- The 91 ms vision stage is the target: our ACT already sees one 640x480-derived image. A smaller input, fewer views, or caching the VLM prefix across the denoise steps (LeRobot already caches the KV prefix) are the levers. Reaching 10 Hz synchronous would need about 60 ms total; no source shows that on Orin-class hardware.
- JetPack 6.2 (TRT 10.7) is not the stack used for the 190 ms result (that was JP 7.2, TRT 10.16). Our TRT 10.7 Myelin trouble may recur on SmolVLA's attention graphs. The split-graph approach, plus onnxslim and the legacy exporter, is the known-good recipe.

### Gaps
- No SmolVLA measurement on an Orin NX; none on JetPack 6.2 / TRT 10.7.
- No Jetson numbers found for VLA-Adapter, TinyVLA, NORA; X-VLA has no Orin number.
- I could not confirm whether LeRobot `groot` or `xvla` support RTC in 0.6.1 (the RTC docs list only pi0, pi0.5 and SmolVLA).

## 4. ACT / Diffusion Policy / flow-matching heads: ONNX/TensorRT best practice; LeRobot's own export support

### Takeaway
LeRobot (latest release 0.6.1, 2026-08-03 per a secondary source) still has **no official ONNX/TensorRT export**. It sits on the 0.7.0 roadmap as TRAIN-09, an "edge deployment export guide". An open PR (#4630) adds an ACT export example, and its reviewer measured ACT on an Orin NX at **8.7 ms TRT FP16, 22.3 ms FP32, 57 ms PyTorch eager** (chunk 100). That matches our 5.9 ms (chunk 20). The industry recipe has converged: legacy TorchScript exporter (`dynamo=False`), opset 17, static batch-1 shapes, simplify (onnxslim / graph folding), keep sensitive parts (ViT, norms) in FP32/BF16, CUDA graphs around the engine, and verify per action dimension against PyTorch on CUDA. For Diffusion Policy I found no Orin numbers.

### Cited Findings
- LeRobot export status:
  - Feature request #3146 ("ONNX / TensorRT export for trained policies", proposed `lerobot-export` CLI) is open — [issue #3146](https://github.com/huggingface/lerobot/issues/3146).
  - The 0.7.0 community roadmap (opened 2026-06-17, still open, no date) lists TRAIN-09 "Publish and verify a guide for exporting policies to edge deployment targets", unassigned. It also lists RUN-03 (sync / local async / GPU async), RUN-05 (RTC snapshot atomicity), RUN-06 (training-time RTC, cloud inference) and TRAIN-04 (torch.compile) — [issue #3832](https://github.com/huggingface/lerobot/issues/3832).
  - Related open PRs: #3493 (modular export to ONNX/OpenVINO) and #3521 (ONNX export CLI) — [PR #4630 discussion](https://github.com/huggingface/lerobot/pull/4630).
  - A fork RFC warns that hand exports can fail silently (camera order, normalization) and plans TRT via ONNX or ExecuTorch — [shoumikhin/lerobot RFC #4](https://github.com/shoumikhin/lerobot/issues/4).
- PR #4630 (opened 2026-09-14, open, review 2026-10-16), ACT export:
  - One state feature plus N cameras, flat inputs, output `action_chunk (B,100,6)`, opset 17, TorchScript exporter (`dynamo=False`; dynamo needs a stricter input structure). Temporal ensembling is left outside the graph.
  - Verified with ORT on random inputs: max abs diff about 1e-6 on CPU, 6.3e-4 against torch on CUDA.
  - Reviewer on a **Jetson Orin NX: fp32 engine 22.3 ms, fp16 8.7 ms, PyTorch eager 57 ms; vs torch CUDA max abs diff 1.7e-3 (fp32) / 6.1e-3 (fp16)**.
  - Reviewer notes: `trtexec` is not on PATH on JetPack; dynamic batch needs `--minShapes/--optShapes/--maxShapes`; engines are tied to GPU + TRT version; two bugs block the script as written.

  Source: [PR #4630](https://github.com/huggingface/lerobot/pull/4630).
- LeRobot rollout has `--use_torch_compile` (default false) and `--fps` (default 30); RTC and sync backends via `--inference.type` — [LeRobot inference docs](https://huggingface.co/docs/lerobot/inference).
- Isaac-GR00T guidance that generalizes: legacy exporter (`dynamo=False`), static shapes, ViT in FP32, CUDA graphs on the action head, cosine >= 0.999 verification, rebuild per GPU — [Isaac-GR00T deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md).
- TensorRT 10.7.0 introduced dedicated `ISqueezeLayer`/`IUnsqueezeLayer` (squeeze/unsqueeze with shape-tensor axes, previously IShuffleLayer) — [TensorRT 10.7.0 release notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-10/10.7.0.html). I found no public report of the exact "mod_t" Myelin error.
- Version sensitivity: TRT 10.15.1 fails on GR00T's optimized action encoder where 10.16.1.11 works — [GR00T deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md). A TRT 10.14 regression (InstanceNorm, dynamic shape) crashed in a Myelin kernel — [NVIDIA forum](https://forums.developer.nvidia.com/t/trt-10-14-dynamic-shape-instancenormalization-crashes-with-cuda-720-in-myl-inor-kernel/368018).
- CUDA graphs: about 2x on pi0 by removing launch overhead — [arXiv 2510.26742](https://arxiv.org/html/2510.26742v1). 3.0–3.8x on SmolVLA's denoise loop on Ampere — [jared-hpc.com](https://jared-hpc.com/posts/vla-physical-ai-tensorrt-tether/). Graph reuse took pi0.5 on Orin from 1420.8 to 412.9 ms — [EagleVLA](https://arxiv.org/html/2607.12659). TRT engines replay via CUDA graph in the Thor pi0.5 tutorial — [Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor).
- ONNX export of Diffusion Policy has failed for at least one user (onnx issue #6848) — [onnx/onnx #6848](https://github.com/onnx/onnx/issues/6848).
- ORT TensorRT EP vs ORT CUDA EP for SmolVLA on an A10G: 19.5 vs 108 ms (5.5x) — [jared-hpc.com](https://jared-hpc.com/posts/vla-physical-ai-tensorrt-tether/). Jetson engine cache: about 1–2 s start when cached, 2–5 min rebuild after a JetPack upgrade — [same](https://jared-hpc.com/posts/vla-physical-ai-tensorrt-tether/).

### Inferences
- Our ACT path is already the best known one; there is nothing official to switch to. Cheap remaining wins:
  - Wrap `execute_async_v3` in a CUDA graph. ACT at about 6 ms is probably partly launch-bound; trtexec's `--useCudaGraph` shows the gain.
  - Move JPEG decode and resize to GPU or to a smaller input. Our 12 ms request vs 5.9 ms engine means about 6 ms is outside the engine.
  - Keep the build static (batch 1) and onnxslim'd.
- The ACT deviation we see (0.19 deg max vs LeRobot) is in line with the PR's 6.1e-3 abs diff in normalized units. Report error in degrees per joint, not cosine: cosine 0.999 can hide a 1–2 deg wrist error.
- Upgrading TRT (JetPack 7.2 ships 10.16.2) may remove the onnxslim dependency, but the engine and numerics must be re-verified. Not worth it for ACT alone.
- For flow-matching heads (SmolVLA, pi0.5, GR00T DiT), export the denoise step, not the loop: one graph called N times inside a CUDA graph, with a fixed noise tensor as an explicit input so that PyTorch-vs-TRT comparisons are deterministic.

### Gaps
- No Diffusion Policy (UNet or transformer) TRT numbers on Orin found.
- Whether TRT 10.8+ fixes the "mod_t" Myelin error on unslimmed ACT exports is unknown (not reported publicly).
- PR #4630's normalization placement (inside or outside the graph) was not visible.

## 5. Real-time chunking and async execution: how they hide latency; the budget a 10 Hz chunked policy needs

### Takeaway
Async inference removes idle time: the next chunk is computed while the current one executes. RTC (PI, NeurIPS 2025) additionally inpaints the start of the new chunk so it continues the actions already committed during the inference delay. It costs about +21 ms per call on pi0.5 (76 -> 97 ms on a 4090) and was "completely robust" to +100 and +200 ms of injected delay on real robots. For a 10 Hz chunked policy the hard limit is d = ceil(latency / 100 ms) < H − s (prefix length < the overlap region). The practical aim is d <= 2 (latency under about 200 ms, including the network) to stay reactive.

In LeRobot 0.6.1, RTC (`lerobot-rollout --inference.type=rtc`) runs only in-process with the GPU. The gRPC async server does not pass RTC's inputs, so remote-GPU serving there means async without RTC (unconfirmed except by one blog). ACT is not flow-matching, so RTC does not apply to it as-is.

### Cited Findings
- RTC definition: inference delay d is the number of controller timesteps between receiving an observation and the chunk becoming available. RTC freezes the actions guaranteed to execute during d and inpaints the rest with guidance. It works with any diffusion- or flow-based policy without retraining — [arXiv 2506.07339](https://arxiv.org/abs/2506.07339).
- RTC real-world setup and results — [arXiv 2506.07339v2 HTML](https://arxiv.org/html/2506.07339v2):
  - pi0.5, H=50, controller dt=20 ms (50 Hz), 5 denoising steps, β clipped at 5 (with n=5, any β >= 4.25 affects only the first step).
  - The soft mask is 1 for i<d, exponential decay for d<=i<H−s, and 0 beyond.
  - **LAN remote inference added 10–20 ms, giving d about 6.** Injected +100 / +200 ms gave d about 11 / 16.
  - RTC had the best throughput at all delays and no degradation with delay. Synchronous degraded linearly. Temporal ensembling oscillated badly enough to trigger protective stops at +100/+200 ms.
  - 480 episodes, about 28 h on the robot.
- Training-time RTC: "Training-Time Action Conditioning for Efficient Real-Time Chunking" (Black, Ren, Equi, Levine; arXiv 2512.05964, 2025-12-05). Simulate the delay in training and condition on the action prefix, which removes the inference-time guidance overhead. It matched inference-time RTC on pi0.6 real tasks and beat it in simulation at high delays — [arXiv 2512.05964](https://arxiv.org/pdf/2512.05964). A follow-up "Soft RTC" (arXiv 2605.25537, May 2026) relaxes the binary prefix mask — [arXiv 2605.25537](https://arxiv.org/pdf/2605.25537).
- VLASH (MIT Han lab et al., arXiv 2512.01031, 2025-11-30) conditions the policy on the future state, estimated by rolling the current state through the previous chunk. It claims up to 2.03x speedup and 17.4x lower reaction latency vs sync "while fully preserving accuracy" — [arXiv 2512.01031](https://arxiv.org/pdf/2512.01031). Code: github.com/mit-han-lab/vlash.
- EagleVLA (Orin-focused, CoRL 2026) adds a future-correction module. On pi0.5 LIBERO averaged over delays 1–9: sync 97.3/99.6/96.7/93.5, VLASH 74.4/86.1/84.2/81.3, RTC 92.6/96.6/94.1/86.4, EagleVLA+Sched 97.4/98.6/96.8/92.5. Real robot at 15 Hz: naive async on Orin 6/10, 0/10, 5/10 vs EagleVLA 10/10, 8/10, 9/10 — [arXiv 2607.12659](https://arxiv.org/html/2607.12659).
- Bidirectional decoding (BID) takes 2.3x RTC's latency on pi0.5 and underperformed RTC in simulation — [arXiv 2506.07339v2](https://arxiv.org/html/2506.07339v2).
- LeRobot RTC:
  - Supports pi0, pi0.5 and SmolVLA.
  - Parameters: `execution_horizon` (typ. 8–12), `max_guidance_weight` (10.0 "optimal" for 10-step flow matching), `prefix_attention_schedule` (LINEAR/EXP/ONES/ZEROS, EXP recommended), and `inference_delay` passed per call.
  - Offline check: `examples/rtc/eval_dataset.py`.

  Source: [LeRobot RTC docs](https://huggingface.co/docs/lerobot/en/rtc). Rollout flags: `--inference.rtc.*`, `--inference.queue_threshold` (default 30) — [LeRobot inference docs](https://huggingface.co/docs/lerobot/inference).
- LeRobot RTC caveats (one secondary source, 2026-10-08, not verified against code) — [particula.tech](https://particula.tech/blog/real-time-chunking-vs-async-inference-lerobot):
  - The v0.6.1 gRPC async server calls `predict_action_chunk` with the observation only, so RTC cannot be used with remote serving. PR 3112 closed unmerged 2026-09-26; PR 4836 is open.
  - The delay is `ceil(max_latency × fps)` from a running max that never resets after warm-up, so one slow call inflates d until restart.
  - The code default schedule is LINEAR although the docs say EXP.
  - Training-time RTC (PR 4056) merged to main on 2026-08-21 but is in no release.
  - The async server binds an insecure port 8080 without auth; default aggregate is `weighted_average` 0.3/0.7.
- LeRobot async — [HF async blog](https://huggingface.co/blog/async-robot-inference):
  - The client sends a new observation when the queue falls below `chunk_size_threshold` g × chunk (recommended 0.5; 0.7 found good).
  - The server keeps up when inference_time <= 1/fps.
  - Overlaps are combined by an `aggregate_fn` (replace or weighted blend).
  - Observations are streamed because they exceed gRPC's 4 MB limit.
- GR00T: "action chunking lets a ~10 Hz inference rate sustain ~30 FPS execution"; Thor's 12.4 Hz is called the recommended 10+ Hz minimum — [GR00T hardware recommendation](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md).
- Related 2026 work not read in detail: "Real-Time Execution with Autoregressive Policies" (arXiv 2606.13355) and "Realtime-VLA FLASH: speculative inference for diffusion-based VLAs" (arXiv 2605.13778) — [search listing](https://arxiv.org/pdf/2606.13355), [arXiv 2605.13778](https://arxiv.org/pdf/2605.13778).

### Inferences
- **Budget at our 10 Hz (dt = 100 ms), chunk H = 20, a new chunk every s = 10 ticks.**
  - Sync (what the M3 does now): the arm stalls for the whole latency at each chunk boundary. That is fine for ACT (12 ms), costs about 12% of the time at 120 ms, and is unacceptable at 300 ms.
  - Async (request when about 10 actions remain): any latency under 10 ticks (1 s) never starves the queue. But the new chunk is conditioned on an observation that is d ticks old, and the first d actions of the new chunk are "in the past". So reaction time ≈ latency + up to s·dt.
  - Practical targets: **network + inference under about 100 ms gives d = 1 and is effectively free; under 200 ms (d = 2) is good; 300–500 ms (d = 3–5) works with RTC but reacts like a 2 Hz policy.**
  - At our scale that puts GR00T or pi0.5 on the 5090 (25–80 ms plus 10–30 ms WiFi) at d = 1, SmolVLA on the NX (about 150–190 ms) at d = 2, and GR00T/pi0.5 on the NX (300–500+ ms) at d = 3–6.
- RTC needs H − s > d. With H = 20, s = 10 we have 10 ticks of room. For RTC, use a longer overlap: request when 10 actions remain and blend over `execution_horizon` 8–10, the ticks after d.
- RTC needs the policy in-process, or our own server that accepts `prev_chunk_left_over` and `inference_delay`. LeRobot 0.6.1's gRPC server does not, so for GR00T or pi0.5 on hekate we would write a thin server: ours or GR00T's ZMQ, which calls `predict_action_chunk(obs, inference_delay=d, prev_chunk_left_over=…)`.
- ACT is not a flow model, so RTC guidance does not apply. VLASH-style state roll-forward, or training-time action-prefix conditioning (the 2512.05964 idea), could be applied to ACT in principle. Both are untested by anyone for ACT as far as I found.

### Gaps
- Not verified in the LeRobot source: the claim that 0.6.1's gRPC server cannot do RTC, and whether `groot` supports RTC in 0.6.1.
- No study gives an explicit "maximum acceptable d" for pick-and-place; RTC's paper shows no degradation up to d about 16 at 50 Hz (about 320 ms) on its tasks.

## 6. Network serving from a Windows GPU box to a Jetson: protocols, measured RTT on WiFi vs Ethernet, image compression

### Takeaway
The three ecosystem protocols are all fine at 10 Hz: openpi uses websocket, Isaac-GR00T uses ZeroMQ + msgpack (raw npy arrays), and LeRobot uses gRPC streaming. Published overheads: a policy server added 0.3 ms on localhost and reached up to 220 Hz over 1 GbE with two JPEG 224² cameras (VLAgents). PI measured 10–20 ms for LAN remote inference. On an Orin over USB networking, FlashRT's client saw 20–35 ms over the model time. HF reports sub-100 ms round trip for SmolVLA on a 4090 over a LAN. The dominant variable on WiFi is jitter (p99/max), not the mean. Resize on the robot first (224–256 px), then JPEG: raw 2 x 256² is about 393 KB per request against tens of KB as JPEG. GR00T's stock client sends raw arrays.

### Cited Findings
- openpi: `serve_policy.py` on port 8000; the client `WebsocketClientPolicy(host, port)`, `client.infer(obs)["actions"]`. Resize on the client with `image_tools.resize_with_pad(img, 224, 224)` + `convert_to_uint8` "to minimize bandwidth and latency"; query every N steps and execute the chunk open-loop — [openpi remote_inference.md](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md).
- Isaac-GR00T: ZeroMQ (Nagle disabled), MessagePack, numpy as `.npy` bytes, port 5555, optional `api_token`, client `timeout_ms` 15000 (30000 suggested for slow networks) — [GR00T server-client docs](https://mintlify.wiki/NVIDIA/Isaac-GR00T/deployment/server-client).
- LeRobot async: gRPC with streamed observations (they exceed the 4 MB unary limit); "gRPC about 5x faster than a comparable REST API"; sub-100 ms round trip on a LAN with SmolVLA on an RTX 4090 — [HF async blog](https://huggingface.co/blog/async-robot-inference). The server binds an insecure port without credentials (default 8080) — [particula.tech](https://particula.tech/blog/real-time-chunking-vs-async-inference-lerobot).
- VLAgents (UTN Nuremberg, arXiv 2601.11250, 2026-01-16): RPyC, shared memory on the same host, JPEG over TCP otherwise; 0.3 ms added on localhost; up to 220 Hz over 1 Gbit Ethernet with two 224x224 cameras (transport only); "about 3x faster than other common policy servers" (openpi websocket, LeRobot gRPC, OpenVLA FastAPI were among those compared) — [arXiv 2601.11250](https://arxiv.org/html/2601.11250). WiFi was not tested.
- PI's RTC experiments: LAN remote inference adds 10–20 ms — [arXiv 2506.07339v2](https://arxiv.org/html/2506.07339v2).
- FlashRT pi0.5 on AGX Orin: model p50 92 ms, client p50 113–150 ms. "USB transport and client glue add about 20–35 ms" (openpi protocol over the Jetson USB network) — [hous-lab/pi05-orin-edge](https://github.com/hous-lab/pi05-orin-edge).
- RoboPocket: round trip under 150 ms over standard WiFi with persistent server-side state — [arXiv 2603.05504 (search snippet)](https://arxiv.org/pdf/2603.05504).
- VLA-Perf's network model (not measurements):
  - WiFi 6: 3.5 ms base, 560/800 Mbps up/down. WiFi 7: 2.5 ms. 1 GbE: 0.1 ms.
  - Modeled B100 end-to-end: 3.8 ms on 1 GbE vs 8.4 ms on WiFi 7.
  - Async gives a 2.6x throughput gain over sync on WiFi 7.
  - Splitting VLM on the server and action expert on the device is "usually slower than device-only and always slower than server-only".

  Source: [arXiv 2602.18397](https://arxiv.org/html/2602.18397).
- OpenTau recommends JPEG encoding with tuned quality for gRPC clients — [OpenTau deployment docs](https://opentau.readthedocs.io/en/latest/tutorials/deployment.html). ReferTrack streams frames over websocket to a GPU server and drops superseded frames to avoid stale commands under jitter — [arXiv 2607.20061 (search snippet)](https://arxiv.org/pdf/2607.20061).
- A course on N1.7 + LeRobot frames latency as T_e2e ≈ T_capture + T_pack + T_net + T_infer + T_unpack + T_actuate, and suggests compressing observations; it gives no numbers — [Seeed reBot ch. 22](https://wiki.seeedstudio.com/robotics/rebot_physical_ai_course_chapter_22/).
- openpi and Isaac-GR00T document Linux only (Ubuntu 22.04 for openpi; "Linux" for GR00T) — [openpi README](https://github.com/Physical-Intelligence/openpi), [GR00T N1.7 card](https://huggingface.co/nvidia/GR00T-N1.7-3B).

### Inferences
- Payload arithmetic (mine):
  - One 224x224x3 uint8 frame is 150.5 KB raw; 256x256x3 is 196.6 KB; 640x480x3 is 921.6 KB.
  - At a realistic lab-WiFi 50–150 Mbit/s uplink (an assumed range, not measured), 2 x 256² raw takes 21–63 ms to send; JPEG q85 at about 15–25 KB per frame (typical, not measured here) takes 2–8 ms.
  - Conclusion: **send JPEG, resized on the robot**. If we keep GR00T's stock ZMQ client, encode JPEG ourselves and decode on the server through a custom endpoint, rather than shipping `.npy`.
- Protocol choice matters less than (a) a persistent connection (no per-request TCP/HTTP setup), (b) TCP_NODELAY (ZeroMQ sets it; our stdlib `http.server` does not by default), (c) dropping stale requests, and (d) timeouts with a safe fallback (hold the pose). Reusing the vendor protocol (ZMQ for GR00T, websocket for openpi) saves code; a single thin server of our own lets us add RTC inputs.
- Windows servers (hekate, Dionysos): the servers will most likely run under WSL2 or Docker. WSL2's default NAT networking needs port forwarding (or "mirrored" networking) and a Windows firewall rule for robots to reach port 5555/8000. Both count as firewall or software changes, which CLAUDE.md says to ask about first. A native-Windows PyTorch server (LeRobot already trains natively on Windows here) avoids that, at the cost of flash-attn and Triton support. Unverified for `groot`.
- Measure WiFi round trips at the real 10 Hz cadence, while the robot streams camera MJPEG and while other lab traffic is present. Jetson WiFi power-save is a common source of 100+ ms spikes (general experience, not sourced here).

### Gaps
- No published measurement of ZeroMQ vs websocket vs gRPC round trip over WiFi with JPEG observations; the numbers above are LAN/Ethernet or modeled.
- No source on the latency added by WSL2 networking for a robot policy server.

## 7. Quantization of VLAs (INT8/FP8/NVFP4/4-bit weight-only): measured success-rate impact

### Takeaway
Weight-only 4/8-bit and W8A8 are near-lossless for pi0.5 and X-VLA in closed loop. W4A4 without careful calibration collapses success to 0% for pi0.5, X-VLA and OpenVLA-OFT. FP8 and NVFP4 TensorRT on Thor or Blackwell keep pi0.5 within about 1 point on LIBERO (cosine 0.995–0.9995). On the Orin (Ampere) the only real option is W8A16 weight-only INT8, which saved time on AGX Orin with no measurable loss. W8A8 was 2.7x slower than FP16 there, and quantized LLM kernels were slower than BF16 on a 4090. Quantize for memory, verify for speed, and always judge in closed loop.

### Cited Findings
- VLAQuantBench (arXiv 2609.25376, 2026-09-21), closed-loop LIBERO Spatial/Object/Goal/Long, RTN quantizer:

  | Model | Baseline | W4 | W8A8 | W4A4 | W4A8 |
  |---|---|---|---|---|---|
  | pi0.5 | 99/99/94/95 | 97.5/99/96.5/94 | 98.5/99/97/93.5 | **0 on all four** | 99/100/96.5/96.5 |
  | pi0 | 78/87.5/84.5/48.5 | 69/90.5/79/41 | 74.5/88.5/81/45 | 61/74/58/31 | — |
  | X-VLA | 96.8/97.9/97.4/97.9 | 95.8/97.9/96.8/95.8 | 96.3/97.9/96.3/98.4 | 0 | — |
  | OpenVLA-OFT | 100/97.5/96/92.5 | — | 36/2.5/13/64 | 0 | — |

  OpenVLA-OFT's W8A8 is recovered by keeping one fc2 projection in high precision. Sensitivity ordering differs per model (for pi0.5 the LLM is most sensitive; for X-VLA, the action head). On a Franka, W8/W8A8 stayed within 6 pp of BF16 and W3 hurt a lot. **"All tested quantized LLM kernels used less memory than BF16 but were slower on an RTX 4090"** — [arXiv 2609.25376](https://arxiv.org/html/2609.25376).
- OpenVLA (7B), BridgeData real robot: bf16 71.3%, int8 58.1% (attributed to slower 1.2 Hz inference, not precision), int4 71.9% at 7.0 GB vs 16.8 GB — [OpenVLA paper](https://arxiv.org/pdf/2406.09246) (as summarized in search results).
- QVLA (arXiv 2602.03782): on OpenVLA, SmoothQuant W8A8 gives 75.8% and W4A4 63.2% vs 76.5% FP; QVLA's channel-aware W4A4 gives 76.0% — [arXiv 2602.03782](https://arxiv.org/html/2602.03782v1).
- HoloQ-VLA (arXiv 2605.28803): W4A4 pi0.5 98.0% vs FP16 97.1% on LIBERO, vs 82.0% for the best prior W4A4 — [arXiv 2605.28803](https://arxiv.org/pdf/2605.28803).
- ActQuant (arXiv 2605.24011): OpenVLA-OFT at 3.0 bits/weight keeps 95.0%, while QVLA drops to 37.4% (0% at 2.5 bpw) — [arXiv 2605.24011](https://arxiv.org/pdf/2605.24011).
- pi0.5 TRT on Thor: FP8 cosine about 0.9995, FP8+NVFP4 cosine 0.9946 (guidance: if consistently below 0.99, fall back to FP8-only) — [Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor). FlashRT pi0.5 NVFP4 on the 5090: LIBERO-10 93.32% (5,000 episodes) vs 94.29% reference; Thor cosine 0.999905 (FP8) and 0.999707 (NVFP4) — [FlashRT](https://github.com/LiangSu8899/FlashRT).
- pi0.5 W8A16 channel-split INT8 on AGX Orin: LIBERO-10 92.2–92.4% vs BF16 91.8% (n=500); W8A8 2.73x slower than FP16 GEMM on Orin — [hous-lab/pi05-orin-edge](https://github.com/hous-lab/pi05-orin-edge).
- GR00T N1.7: TRT full pipeline BF16 19/20 vs PyTorch 20/20 (n=20, H100); NVFP4 is only on Thor (35.6 ms vs 76.8 ms BF16 TRT) with no published success numbers — [GR00T deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md).
- SmolVLA "FP16/INT8" ONNX artifacts turned out to be FP32. The success change came from the token width, not precision — [arXiv 2609.14146](https://arxiv.org/abs/2609.14146).
- LiteVLA-Edge uses 4-bit GGUF on AGX Orin (150.5 ms) — [arXiv 2603.03380](https://arxiv.org/html/2603.03380). BitVLA (1-bit) reports that post-training INT4 hurts OpenVLA more than OpenVLA-OFT — [arXiv 2506.07530](https://arxiv.org/html/2506.07530v2).

### Inferences
- For us:
  - On the 5090, FP8 or NVFP4 is the only place low precision is both supported and fast. Gains are about 10 ms on a 25 ms call, irrelevant at 10 Hz, so keep BF16 until latency matters.
  - On the Orin NX, quantization is mostly about fitting next to our other models. W8A16 weight-only on the LLM's big GEMMs is the one recipe shown to help on Orin. Do not use W8A8 or W4A4 there.
  - ACT (about 80M parameters by my recollection of the ACT paper) gains nothing from INT8 at 6 ms. Stay on FP16 and keep checking in degrees.
- Precision verdicts must come from our closed-loop sim grid plus real picks: offline cosine or LIBERO transfers poorly across models (the VLAQuantBench conclusion).

### Gaps
- No closed-loop quantization study for GR00T N1.6/N1.7 or SmolVLA found.
- No INT8 TensorRT (PTQ) numbers for any VLA on Orin NX.

## Benchmark plan for policies

### Takeaway
Measure four things separately and then together:
- (1) engine latency on the target GPU;
- (2) service latency, i.e. what the robot's request sees (encode, transport, decode, preprocess, engine, postprocess);
- (3) the 10 Hz loop: tick lateness, picture age, inference delay d, chunk-boundary jumps;
- (4) accuracy vs LeRobot PyTorch, in degrees per joint, offline and in closed loop.

Do it for ACT on the Orin NX (baseline already partly measured), SmolVLA on the Orin NX, and GR00T N1.7 / pi0.5 on hekate's 5090 (and Dionysos's A4500 as a fallback), served over the lab WiFi. Desk-research note: nothing below has been run; CLAUDE.md rules apply (ask before GPU-heavy work on hekate with under 16 GB VRAM free, before any firewall or software change there, and never touch others' jobs on hekate or Dionysos).

### What to measure, where, and how

**A. Engine latency (pure GPU time)**
- *Orin NX (M3, X3):*
  - Record the power mode first: `sudo nvpmodel -q`, then `sudo jetson_clocks` (record `jetson_clocks --show`), and run `tegrastats --interval 100` alongside for GPU freq, EMC and RAM.
  - ACT: `/usr/src/tensorrt/bin/trtexec --loadEngine=models/policy/act.engine --useCudaGraph --noDataTransfers --warmUp=2000 --iterations=2000 --avgRuns=100 --percentile=50,90,99`. Run once without `--useCudaGraph` to see the launch-overhead share. Use `--dumpProfile --separateProfileRun` for per-layer times. trtexec is not on PATH on JetPack — [PR #4630 review](https://github.com/huggingface/lerobot/pull/4630).
  - SmolVLA: per-graph and total (vision, prefix, denoise x 10) in ORT-TRT EP or native TRT, matching the split-graph recipe — [eetmie/smolvla-base-onnx](https://huggingface.co/eetmie/smolvla-base-onnx).
  - Report p50/p99/max; the clock state (MAXN_SUPER vs 25 W) changes results.
- *hekate 5090 / Dionysos A4500 (GR00T N1.7):* `python scripts/deployment/build_trt_pipeline.py --model-path <our ckpt> --dataset-path <our LeRobot set> --embodiment-tag NEW_EMBODIMENT --steps export,build,verify,benchmark --warmup 10 --num-iterations 20`, plus `--inference-mode pytorch` and torch.compile for the three-way table (eager / compile / TRT) that NVIDIA publishes — [deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md). Linux/WSL2 likely required.
- *pi0.5:* LeRobot `pi05` eager vs `--use_torch_compile`, and openpi `serve_policy.py` (JAX and PyTorch). Optionally FlashRT FP8/FP4 on the 5090.
- Use the same inputs as deployment: 1 wrist camera at the trained resolution, our 6-DoF state, chunk length as trained (ACT 20; GR00T 40 or 16; pi0.5 50 / 10 steps).

**B. Service latency (what one robot request costs)**
- On the robot (ACT, local service): instrument `policy_service.py` per stage — base64 decode, `cv2.imdecode`, resize/normalize, H2D, engine, D2H, JSON encode. Today 12 ms total vs 5.9 ms engine.
- Remote (5090): time on the robot's monotonic clock — t0 frame captured, t1 request sent, t2 reply received, t3 first action of the chunk sent. Server-side, log receive / preprocess / infer / send so network time = (t2 − t1) − server time; no clock sync needed.
- Protocols to compare with an identical payload (2 x JPEG 256² at q85 and 1 x 224²; raw `.npy` for GR00T's stock client):
  - our stdlib HTTP (with and without TCP_NODELAY / keep-alive);
  - GR00T ZMQ via the model-free replay server: `run_gr00t_server.py --dataset-path <data> --embodiment-tag <tag> --execution-horizon 8` — [GR00T server docs](https://mintlify.wiki/NVIDIA/Isaac-GR00T/deployment/server-client);
  - openpi websocket (`scripts/serve_policy.py`);
  - LeRobot gRPC (`python -m lerobot.async_inference.policy_server` / `robot_client`; verify the exact 0.6.1 flags).
- Over WiFi (IISLab-AMEC) and, if a cable is possible, Ethernet. 3,000 requests at the real 10 Hz cadence, not back-to-back, with the camera MJPEG streams on. Report p50/p95/p99/max and the count above 100 ms. Baseline with `ping -i 0.1 -c 3000`.

**C. Loop / chunk timing (on the robot, in the bridge's policy job)**
- Per tick: scheduled vs actual send time (lateness), picture age at inference, queue length.
- Per chunk: the inference delay d = ceil((t2 − t0)/100 ms), and the joint-space jump between the last executed action of the old chunk and the first used action of the new one (deg, per joint).
- Per run: commands per second (should be 10.0) and late ticks.
- Compare modes: sync (today), async (request when about 10 actions remain), async + RTC for flow policies (`execution_horizon` 8–10, EXP schedule, `max_guidance_weight` 10 for 10 steps / about 5 for 5 steps). Inject delay (+100, +200 ms) as the RTC paper did — [arXiv 2506.07339v2](https://arxiv.org/html/2506.07339v2).
- Offline RTC sanity check on our dataset before the robot: `python examples/rtc/eval_dataset.py --policy.path=<ckpt> --dataset.repo_id=<our set> --rtc.execution_horizon=10 --rtc.max_guidance_weight=10.0 --device=cuda` — [LeRobot RTC docs](https://huggingface.co/docs/lerobot/en/rtc).

**D. Accuracy vs PyTorch**
- Offline: for each engine and precision (FP32/FP16/BF16; INT8 W8A16; FP8/NVFP4 on the 5090), held-out frames as in `act_check.npz`. Report per-joint max and mean abs error in degrees over the whole chunk and for the first 10 actions, plus the worst frame.
- For flow-matching models, feed the same noise tensor to PyTorch and the engine; otherwise the comparison measures sampling noise.
- Gates (proposed): ACT max <= 0.5 deg (today 0.19); VLAs: mean <= 0.5 deg, max <= 2 deg, and cosine >= 0.999 as NVIDIA uses.
- Closed loop: the 20-cell sim grid (m3_sim / policy_service once per chunk) for each engine variant vs PyTorch, then live trials scored by `m3_policy_trial.py`.

**E. Memory and endurance**
- Orin: peak RAM via tegrastats/jtop with the full robot stack running (vision service, depth TRT, kit detector, ROS 2) plus the policy. Engine file size and load time (cached engine about 1–2 s; rebuild 2–5 min).
- A 10-minute continuous run at 10 Hz: watch for thermal throttling (GPU clock in tegrastats) and latency drift.

### Pitfalls to design around
- Warm-up: torch.compile, cuDNN autotune, TRT lazy init and the first ZMQ/gRPC connection are slow. LeRobot's large-VLA server loads the model on first client connect (1–2 min) — [async docs mirror](https://www.mintlify.com/huggingface/lerobot/inference/async-inference). LeRobot's RTC delay uses a running max that one slow call inflates — [particula.tech](https://particula.tech/blog/real-time-chunking-vs-async-inference-lerobot). Discard warm-up, and restart between configs.
- Timing GPU code without `torch.cuda.synchronize()` or CUDA events measures the launch, not the work.
- DVFS: Jetson results without `jetson_clocks` vary a lot; NVIDIA's numbers are MAXN. Our NX's Super mode is "MAXN_SUPER" and throttles above TDP — [NVIDIA blog](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules).
- Mislabelled precision and silent export changes: check what the engine actually runs (per-layer precision in `--dumpLayerInfo`), and check the language-token width, camera order and normalization — [arXiv 2609.14146](https://arxiv.org/abs/2609.14146), [lerobot RFC](https://github.com/shoumikhin/lerobot/issues/4).
- Static shapes and exporter: batch 1, legacy exporter `dynamo=False`, onnxslim before TRT 10.7. Engines are tied to GPU + TRT version, so rebuild after any JetPack upgrade — [GR00T deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md).
- Reference on CPU vs CUDA: the PR found 1e-6 vs 6e-4 differences just from the reference device — [PR #4630](https://github.com/huggingface/lerobot/pull/4630). Fix the reference (CUDA FP32) and say so.
- WiFi: measure p99/max at the real cadence; consider Jetson WiFi power-save; drop stale requests rather than queueing them.
- The M3 reports the joints it was sent, not measured ones (NEXT.md), so "state" in a request is the commanded pose. Keep it the same in training and serving.
- Windows hosts: WSL2 port forwarding or mirrored networking and Windows firewall rules are needed for robots to reach the server. Ask first (CLAUDE.md). Shared GPUs: check free VRAM and others' jobs before every heavy run.
- Do not compare across papers' "Orin": NVIDIA's tables are AGX Orin 64 GB; halve for the NX as a first guess, then measure.
