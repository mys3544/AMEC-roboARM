# Fastest formats and runtimes for small LLMs, small VLMs and speech recognition on our hardware (state 2026-10-27)

Scope: the voice-command chain (Parakeet TDT 0.6B v3 -> Qwen3.5 4B -> one JSON function call), a distilled 0.8B/2B parser on the robots, and a later small VLM. Machines: 2x Jetson Orin NX 16 GB Super (sm_87, 102 GB/s, JetPack 6.2 / CUDA 12.6 / TRT 10.7), laptop RTX 2070 Super 8 GB (sm_75, Windows 11), hekate RTX 5090 32 GB (sm_120, Windows 11, shared), Dionysos RTX A4500 20 GB (sm_86, Windows, shared). Desk research only; nothing was run on any machine.

Our own baseline, for reference (local files, not web):
- console/llm.py starts `llama-server.exe -m <gguf> -ngl 99 -c 4096 -np 1 --host 127.0.0.1 --port 8081 --jinja` from llama.cpp b11617 (Windows, CUDA 12.4), and asks `/v1/chat/completions` with `temperature 0`, `max_tokens 256`, `response_format` = json_schema (turned into a grammar by llama.cpp) and `chat_template_kwargs.enable_thinking=false` — [console/llm.py](C:\Users\Mys\Desktop\RoboARM 2026\console\llm.py)
- Bench 2026-10-20 on the 2070 Super, Q4_K_M GGUFs (unsloth), 45 sentences sk/cs/en: Qwen3.5 4B 45/45, median 646 ms (worst 767 ms, 2.7 GB); Qwen3 4B Instruct 2507 44/45, 535 ms; Gemma 4 E4B 43/45, 1067 ms; Qwen3.5 2B 42/45, 364 ms (1.3 GB); Gemma 4 E2B 37/45, 541 ms. After prompt rules + SOUNDS_LIKE: 4B 71/71, median 732 ms — [console/README.md](C:\Users\Mys\Desktop\RoboARM 2026\console\README.md)
- ASR: sherpa-onnx, `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8`, CPU, 4 threads, push-to-talk (offline recognition of the whole utterance). 4 threads: 1.4 s audio 154 ms, 1.6 s 177 ms, 2.5 s 207 ms, 3.5 s 300 ms (RTF ~0.08-0.11); 1 thread 327/377/513 ms; 2 threads 186/215/295 ms; 8 threads no better than 4 — [console/README.md](C:\Users\Mys\Desktop\RoboARM 2026\console\README.md), [console/voice.py](C:\Users\Mys\Desktop\RoboARM 2026\console\voice.py)
- Field problem 2026-10-21: Parakeet wrote spoken Slovak as Russian in Cyrillic or as Polish-looking text; "the recogniser itself has no Slovak switch" — [console/README.md](C:\Users\Mys\Desktop\RoboARM 2026\console\README.md)

## 1. llama.cpp in late 2026: quant types, CUDA backend, Turing and CUDA 12 vs 13, Jetson, best quant for 0.8-4B, prompt caching

### Takeaway
llama.cpp is at build b11928 (27 Oct 2026; we run b11617, ~300 builds older), ships Windows CUDA 12.4 and CUDA 13.4 binaries, has NVFP4 (with a native Blackwell sm_120 MMQ path since b8967, April 2026) and in-tree MTP/EAGLE-3/DFlash/DSpark speculative decoding. NVFP4/MXFP4 give no speed advantage on Turing or Orin (no FP4 hardware); Q4_K_M/Q4_0/Q8_0 stay the right choices there. The single most important llama.cpp issue for us is that Qwen3.5 is a hybrid recurrent (Gated DeltaNet) model, for which llama-server's prefix cache relies on "context checkpoints" with a default minimum step of 8192 tokens; there are open reports of forced full prompt re-processing with Qwen3.5, so our cached-system-prompt assumption must be verified.

### Cited Findings
**Releases and builds**
- Latest tags on 27 Oct 2026 are b11913-b11928 (e.g. b11928 12:30 UTC adds MiniCPM-OCR scoring model; b11926 GLM5-next). Tags are build numbers (bNNNN); no semver tags such as v0.7.0 appear on the GitHub releases page — [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)
- Windows CUDA binaries in each release: `llama-bNNNN-bin-win-cuda-12.4-x64.zip` (+ `cudart-llama-bin-win-cuda-12.4-x64.zip`) and `llama-bNNNN-bin-win-cuda-13.4-x64.zip` / `-arm64.zip` (+ matching cudart); Linux ships CUDA 12.8 and 13.4 builds; Windows also gets Vulkan, SYCL, ROCm 10.0 and OpenVINO builds — [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)
- Conflict: Wikipedia lists a "stable release 0.7.0 / 15 October 2026" and a third-party guide says v0.6.0 shipped 5 Oct 2026; the GitHub releases page shows only bNNNN tags. Treat the semver numbers as unverified — [Wikipedia: Llama.cpp](https://en.wikipedia.org/wiki/Llama.cpp); [tech-insider guide](https://tech-insider.org/llama-cpp-tutorial-2026/); contradicted by [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)
- A release-history aggregator lists build b11888 on 25 Oct 2026 — [freedom.tech llama.cpp history](https://freedom.tech/project/llama-cpp/)

**FP4 quant types (NVFP4, MXFP4)**
- `GGML_TYPE_NVFP4 = 40`; CUDA NVFP4 dp4a kernel PR #20644 merged 26 Mar 2026; generic NVFP4 MMQ kernel PR #21074 merged 1 Apr 2026; Vulkan NVFP4 PRs #21455/#21539 (10/14 Apr); SYCL #21227 — [InsiderLLM FP4 guide](https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/)
- Build b8967 (29 Apr 2026) enables native NVFP4 MMQ for sm_120 (Blackwell). One first-day r/LocalLLaMA test (Qwen3.6-27B-NVFP4, RTX 5090, b8966 -> b8967): pp512 3,295 -> 5,547 t/s (+68%), pp512 at depth 32768 2,515 -> 3,587 t/s (+43%), tg128 73.7 -> 73.6 t/s (unchanged; decode is bandwidth-bound) — [InsiderLLM FP4 guide](https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/) (secondary source)
- NVFP4 is 4.5 bits/weight vs Q4_K_M ~4.83 bpw, so it saves only ~7% size (page's 30 Sep correction); no native FP4 path on sm_89 or earlier; Ampere gets memory savings only; Turing/Orin not mentioned; no independent NVFP4-vs-Q4_K_M quality comparison yet, early reports suggest small models may suffer more; HF->NVFP4 GGUF conversion needs the still-open PR #21095 or NVIDIA ModelOpt (0.43.0 `NVFP4_DEFAULT_CFG`) — [InsiderLLM FP4 guide](https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/)
- MXFP4 (gpt-oss native format) is listed in the llama.cpp README hot topics; ik_llama.cpp has its own MXFP4 type. A Feb 2026 issue reported a CUDA build failure in PTX assembly when compiling MXFP4 template instances for sm_120 — [search summary of llama.cpp README / issue #19662](https://github.com/ggml-org/llama.cpp/issues/19662) (may be outdated)

**Gated DeltaNet (Qwen3.5) kernels**
- PR #20340 enabled a chunked fused Gated DeltaNet (GDN) path (with a Metal kernel in #20361); PR #20443 transposed the CUDA GDN state indexing for coalesced reads (the fused kernel had been reading the state column-wise on row-major storage) — [semanticdiff commit d28961d8](https://app.semanticdiff.com/gh/ngxson/llama.cpp/commit/d28961d81e73e32b295d0ad638f3ff14676aeeda); [semanticdiff commit e30f1fdf](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/commit/e30f1fdf74ea9238ff562901aa974c75aab6619b). No CUDA prompt-processing numbers for these were found.

**Speculative decoding / MTP (in mainline, docs/speculative.md)**
- `--spec-type` values: `none` (default), `draft-simple` (separate draft model, `-md`), `draft-eagle3`, `draft-dflash`, `draft-dspark` (Qwen3-backbone drafts only), `draft-mtp` (the model's own Multi-Token-Prediction heads), and n-gram variants `ngram-cache`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod` (no draft model); types can be combined; `--spec-default` enables ngram-mod; greedy sampling needed for exact output matching — [llama.cpp docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- Server defaults: `--spec-draft-n-max 3`, `--spec-draft-n-min 0`, `--spec-draft-p-split 0.10`, `--spec-draft-sampling greedy` — [llama-server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- `draft-mtp` needs a GGUF that keeps the MTP tensors; such a Qwen3.5-4B Q4_0 GGUF exists and self-speculates with `--spec-type draft-mtp` (its card also says MTP was on an `mtp-exp` branch at the time of writing — likely now superseded by mainline) — [shujunyi/Qwen3.5-4B-MTP-Q4_0](https://huggingface.co/shujunyi/Qwen3.5-4B-MTP-Q4_0); an `unsloth/Qwen3.5-2B-MTP-GGUF` (Q4_K_XL) is referenced in [llama.cpp issue #24587](https://github.com/ggml-org/llama.cpp/issues/24587)
- Reported MTP speedups (large models, not ours): Qwen3.6 27B on RTX 3090 38 -> 65 t/s with `--spec-type mtp --spec-draft-n-max 3` — [DataCamp tutorial](https://datacamp.com/tutorial/multi-token-prediction-llama-cpp); "Qwen3.8 27B" Q4_K_M on Jetson AGX Thor ~13 -> ~35 t/s with `-ngl all -fa on --spec-type draft-mtp --spec-draft-n-max 3` — [Jetson AI Lab speculative decoding tutorial](https://www.jetson-ai-lab.com/tutorials/speculative-decoding)

**Server flags relevant to our latency (current README)**
- Prompt cache on by default (`--cache-prompt`, request field `cache_prompt: true`); `--cache-reuse N` default 0 (off); `-cram/--cache-ram` default 8192 MiB host-RAM prompt cache; `--cache-idle-slots` on — [llama-server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- Hybrid/SWA models: `-ctxcp/--ctx-checkpoints N` default 32 per slot; `-cms/--checkpoint-min-step N` default **8192 tokens** (0 = no minimum); `--swa-full` default false — [llama-server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- `-fa/--flash-attn` default `auto`; `--jinja` now on by default; `-np/--parallel` default -1 (auto) with unified KV (`-kvu`) on when slots are auto; `--warmup` on; `--slot-save-path` for saving/restoring slot state; `--json-schema`/`--grammar` CLI flags and `json_schema`/`grammar` request fields; llguidance and CUDA graphs are not mentioned in the README — [llama-server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- Open issues on Qwen3.5 (hybrid recurrent) in llama-server: logs show "forcing full prompt re-processing due to lack of cache data" (with `unsloth/Qwen3.5-2B-MTP-GGUF:Q4_K_XL`), checkpoints erased on the request that tried to use them, and checkpoint invalidation when the middle of the prompt changes length; raising `--ctx-checkpoints` / lowering `--checkpoint-min-step` helped partly; a successful restore was seen when prompts were nearly identical — [issue #24587](https://github.com/ggml-org/llama.cpp/issues/24587), [issue #24714](https://github.com/ggml-org/llama.cpp/issues/24714), [issue #22450](https://github.com/ggml-org/llama.cpp/issues/22450); a V2EX thread reports a merged fix restoring checkpoints for DeltaNet/Mamba hybrids and a later regression (unverified) — [V2EX](https://www.v2ex.com/t/1219800)

**Turing (2070 Super) and CUDA 12 vs 13**
- CUDA 13.x drops Maxwell, Pascal and Volta (compute capability < 7.5); Turing sm_75 is retained (PyTorch keeps Turing in its CUDA 13 builds; PyTorch 2.14 is the last with CUDA 12.x wheels) — [PyTorch dev-discuss notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432); [localaimaster on CUDA 13](https://localaimaster.com/blog/pascal-gpus-after-cuda-13)
- Flash attention works on Turing in llama.cpp (anecdotal user reports; Pascal lacked the f16 FA device code) — [llama.cpp issue #7055](https://github.com/ggml-org/llama.cpp/issues/7055)

**Jetson / Orin**
- The Orin Nano Super benchmark below ran llama.cpp with GGUF Q4_K_M on JetPack 6.2.2 (L4T 36.5.0) — i.e. llama.cpp builds and runs on JetPack 6.2 CUDA 12.6 — [hokwangchoi/jetson-orin-nano-benchmarks](https://github.com/hokwangchoi/jetson-orin-nano-benchmarks)
- Jetson AI Lab uses llama.cpp with `-hf ...:Q4_K_M -ngl all -fa on` on Jetson (Thor) for MTP and DFlash demos — [Jetson AI Lab speculative decoding](https://www.jetson-ai-lab.com/tutorials/speculative-decoding)

### Inferences
- Upgrade the laptop from b11617 to a current build (b119xx). Keep the CUDA 12.4 zip as the safe default for the 2070 Super; the CUDA 13.4 zip should also run on Turing (CUDA 13 keeps sm_75), but needs a driver new enough for CUDA 13.x and it is not verified that the release binary includes sm_75 code — test both.
- Our bench hints that prompt caching may not be working for Qwen3.5: Qwen3 4B 2507 (pure attention, same size class) was faster (535 ms) than Qwen3.5 4B (646 ms). With `--checkpoint-min-step 8192` by default and a system prompt of a few thousand tokens at most, llama-server may be reprocessing the whole system prompt on each command. First check the server log for "forcing full prompt re-processing", then try `-cms 0` (or a small value, e.g. 256) and confirm `timings.prompt_n` drops to the new tokens only. Keep every variable part (taught names list, the command) at the very end of the prompt so the static prefix is byte-identical.
- Quant choice at batch 1 (decode is memory-bandwidth-bound: 2070S 448 GB/s, Orin NX 102 GB/s): 4B -> Q4_K_M (or Q4_0, which has the simplest/fastest CUDA dp4a/MMQ path; compare both); 2B -> Q4_K_M vs Q6_K vs Q8_0 (2B Q8_0 ~2.3 GB still fast on the 2070S; on Orin each extra GB costs ~10 ms per token); 0.8B -> Q8_0 (tiny models lose more from 4-bit, and file size is <1 GB). NVFP4/MXFP4 bring nothing on Turing or Orin (no FP4 tensor cores; NVFP4 is only ~7% smaller than Q4_K_M) — reserve them for the 5090.
- Theoretical decode ceilings (file size / bandwidth): Qwen3.5 4B Q4_K_M 2.7 GB -> ~165 t/s on the 2070S, ~38 t/s on Orin NX; 2B 1.3 GB -> ~340 t/s / ~78 t/s. Real numbers land ~60-75% of this.
- MTP (`--spec-type draft-mtp`) is the most promising decode accelerator for a fixed JSON output with Qwen3.5 (the family is trained with MTP heads, see Q3), but it needs an MTP-preserving GGUF and its interaction with grammar-constrained sampling and with hybrid-model checkpoints must be measured; `ngram-mod` self-speculation is a zero-cost thing to try for repetitive JSON keys.
- On Orin, build llama.cpp from source with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87` (standard practice; not re-verified in this pass).

### Gaps
- No published llama-bench numbers for Qwen3.5 0.8B/2B/4B on an RTX 2070 (Super) or on Orin NX 16 GB were found; must be measured.
- Whether the official Windows CUDA 13.4 binaries include sm_75 SASS/PTX, and the minimum driver version for them, was not verified.
- Whether CUDA graphs are enabled for hybrid (GDN) models and what the GDN kernels' CUDA prefill speed is on sm_75/sm_87 was not found.
- Exact status of the hybrid-model checkpoint fix (which build, which defaults) is unverified; the referenced issues are open/active.
- Whether grammar/JSON-schema constrained sampling and `draft-mtp` work together in llama-server was not documented.

## 2. Alternatives to llama.cpp at batch 1 (TensorRT-LLM, TensorRT Edge-LLM, NIM, vLLM/SGLang, MLC, ExLlama, ORT GenAI, Ollama) and constrained output

### Takeaway
On the Orin NX the fastest documented path for Qwen3.5 0.8B/2B/4B is NVIDIA TensorRT Edge-LLM (C++ runtime, INT4 AWQ): Qwen3.5-2B decodes at 47.9 t/s and Qwen3.5-0.8B at 95.8 t/s on Orin NX 16 GB, and on an Orin Nano Super Edge-LLM decoded a 2B model ~1.6x faster than llama.cpp (60 vs 38 t/s). But Edge-LLM is officially supported on Orin only with JetPack 7.2 (CUDA 13.2, TRT 10.16; JetPack 6.2 is "compatible/experimental"), and Qwen3.5 needs CuTe DSL kernels. vLLM (NVIDIA's Jetson Orin container, W4A16) is the second option on Orin. On the Windows laptop (Turing) llama.cpp remains the practical choice: TensorRT-LLM lists Linux only and Ampere+, Speech/LLM NIMs need CC 8.0+, vLLM has Turing problems. On the 5090, vLLM/SGLang (WSL2/Docker) with NVFP4 or llama.cpp with native NVFP4 are both viable.

### Cited Findings
**TensorRT Edge-LLM (Jetson)**
- Docs "last updated 29 Sep 2026" (commit e649dad); versions 0.7.0 -> 0.8.0 -> 0.9.0/0.9.1 -> 0.10.0/0.10.1 -> 0.11.0 exist. Officially supported: Jetson Thor, DRIVE Thor, DGX Spark (GB10) and Jetson Orin; Orin "supports FP16, INT8, and INT4 runtime precisions only"; C++ runtime with no Python dependencies (experimental Python API); features: EAGLE3, MTP, DFlash, DSpark speculative decoding, paged attention, LoRA, streaming, KV cache reuse, FP8 KV cache, an experimental OpenAI-compatible server; a "Guided Decoding" feature page exists — [Edge-LLM 0.11.0 overview](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/overview.html)
- In 0.7.0, Orin was only "compatible" on JetPack 6.2.x with experimental status; 0.8.0/0.9.x list Orin as officially tested on JetPack 7.2 — [Edge-LLM 0.7.0 overview](https://nvidia.github.io/TensorRT-Edge-LLM/0.7.0/overview.html); [Edge-LLM 0.9.1 overview](https://nvidia.github.io/TensorRT-Edge-LLM/0.9.1/overview.html)
- Seeed guide (updated 31 Jul 2026, Edge-LLM v0.9.1): JetPack 7.2 is "the officially supported TensorRT Edge-LLM path for Jetson Orin", JetPack 6.2 "Compatible"; build with `-DEMBEDDED_TARGET=jetson-orin -DCUDA_CTK_VERSION=13.2 -DENABLE_CUTE_DSL=ALL` because "current Qwen3.5 and other supported model paths require them"; export (`tensorrt-edgellm-export`) runs on an x86 GPU host, then `llm_build --maxBatchSize 1 --maxInputLen 1024 --maxKVCacheCapacity 4096` and `llm_inference` on the Jetson; do not reuse a JetPack 6.2 engine on 7.2 — [Seeed wiki](https://wiki.seeedstudio.com/deploy_tensorrt_edge_llm_on_jetpack7.2/)
- JetPack 7.2 = Jetson Linux 39.2, CUDA 13.2.1, TensorRT 10.16.2 (a 7.2.1 point release exists); it adds the Orin family to JetPack 7; a June 2026 guide notes JetPack 6 / CUDA 12.6 containers and PyTorch wheels do not run properly on it — [NVIDIA JetPack 7.2 archive](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2); [RidgeRun JetPack 7.2 components](https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin/JetPack_7.2/Getting_Started/Components); [NVIDIA forum guide, June 2026](https://forums.developer.nvidia.com/t/setting-up-the-nvidia-jetson-orin-nano-super-dev-kit-on-jetpack-7-2-a-practical-guide-june-2026/372490)
- Edge-LLM supported models include Qwen3.5 0.8B/2B/4B/9B/27B (dense) and 35B-A3B/122B-A10B (MoE), Qwen3.6 27B/35B-A3B, Qwen3.8 27B, Qwen3 0.6B-14B, Llama 3.x 1-8B, Gemma 4, Nemotron-H, plus ASR (Qwen3-ASR 0.6B/1.7B, Nemotron-3.5-ASR streaming 0.6B) and Qwen3-TTS; Orin must not use FP8/MXFP8/FP4/NVFP4 checkpoints; Orin Nano may need `--externalize-weights` for INT4 builds; DFlash drafts listed for Qwen3.5 4B/9B/27B/35B-A3B; MTP listed only for Gemma 4 "assistant" checkpoints — [Edge-LLM supported models](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/getting_started/supported-models.html)
- **Orin NX 16 GB, Edge-LLM 0.10.0, JetPack 7.2 / CUDA 13.2 / TRT 10.16, batch 1, INT4 AWQ** (runtime dashboard; prefill length 57-62 tokens):

  | Model (LLM) | Prefill ms (tokens) | Prefill t/s | Decode t/s |
  |---|---|---|---|
  | Qwen3-0.6B | 17.6 (61) | 3,478 | 124.9 |
  | Qwen3-1.7B | 39.5 (61) | 1,551 | 59.0 |
  | Qwen3-4B-Instruct-2507 | 78.4 (57) | 730 | 32.2 |
  | Qwen3.5-0.8B | 35.5 (62) | 1,741 | 95.8 |
  | Qwen3.5-2B | 56.2 (62) | 1,100 | 47.9 |
  | Qwen3.5-4B | 120.1 (62) | 514 | 27.0 |

  With 2048-token prefill (`llm_bench`): Qwen3.5-0.8B 841 ms (2,436 t/s), decode 92.8; Qwen3.5-2B 1,161 ms (1,764 t/s), 47.0; Qwen3.5-4B 2,558 ms (801 t/s), 26.6; Qwen3-1.7B 674 ms (3,039 t/s), 52.2. AGX Orin 64 GB decodes Qwen3.5-2B at 83.3 t/s, Thor at 123.6 t/s. No speculative-decoding rows for Orin NX — [Edge-LLM 0.11.0 performance benchmarks](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/performance/performance-benchmarks.html)
- Edge-LLM with Qwen3.5-4B and the experimental OpenAI-compatible server has been reported on Jetson Thor (no Orin report found) — [NVIDIA forum](https://forums.developer.nvidia.com/t/tensorrt-edge-llm-on-jetson-thor-openai-compatible-server-and-streaming-client/370353)

**llama.cpp vs vLLM vs Edge-LLM on an Orin (closest head-to-head found)**
- Jetson Orin Nano 8 GB Super (MAXN_SUPER, JetPack 6.2.2), Cosmos-Reason2-2B, medians of 5 runs: llama.cpp GGUF Q4_K_M ctx 4096 — text TTFT 33 ms, image TTFT 114 ms, TPOT 26 ms, 38 t/s; vLLM (Embedl port) W4A16 AWQ ctx 1024 — 61 ms, 75 ms, 17 ms, 56 t/s, 6.9 GB peak; Edge-LLM W4A16 AWQ — 29 ms, 420 ms (visual engine adds ~208 ms), 17 ms, 60 t/s, 4.3 GB. Edge-LLM needed a `cma=950M` kernel parameter and an ONNX rewrite splitting the LM-head MatMul; JetPack 6.2.1 blocked vLLM and Edge-LLM, 6.2.2 unblocked vLLM — [hokwangchoi/jetson-orin-nano-benchmarks](https://github.com/hokwangchoi/jetson-orin-nano-benchmarks)

**vLLM / SGLang**
- Jetson AI Lab serves Qwen3.5 with the image `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` (`vllm serve` with reasoning and tool-call parsers); Qwen3.5 9B uses W4A16 on Orin, NVFP4 on Thor — [Jetson AI Lab Qwen3.5 27B](https://www.jetson-ai-lab.com/models/qwen3-5-27b); [Jetson AI Lab Qwen3.5 9B](https://www.jetson-ai-lab.com/models/qwen3-5-9b)
- Jetson AI Lab Qwen3.5 4B page lists Orin NX 16 GB, vLLM container only, checkpoint `RedHatAI/Qwen3.5-4B-quantized.w4a16` for Orin (NVFP4 `AxionML/Qwen3.5-4B-NVFP4` for Thor); its benchmark section has "No data available" — [Jetson AI Lab Qwen3.5 4B](https://www.jetson-ai-lab.com/models/qwen3-5-4b)
- Community vLLM 0.17.0 wheel for Orin (SM 8.7): official wheels ship Marlin kernels for SM 8.0/8.6/8.9/9.0 but not 8.7; on AGX Orin 64 GB Qwen3.5-35B-A3B GPTQ-Int4 ~31 t/s vs llama.cpp ~22.5 t/s — [thehighnotes/vllm-jetson-orin](https://huggingface.co/thehighnotes/vllm-jetson-orin)
- Qwen's model card recommends SGLang and vLLM from main/nightly for Qwen3.5 (at release) — [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- vLLM on Turing: a Sep 2026 forum report of a shared-memory limit failure (98304 > 65536) for head_dim=512 attention on compute capability 7.5; Red Hat's official vLLM containers/CI cover Ampere and Hopper but not Volta/Turing — [vLLM forum, hardware support](https://discuss.vllm.ai/c/hardware-support/5.md); [Red Hat AIPCC-1149](https://issues.redhat.com/browse/AIPCC-1149)
- Jetson AI Lab speculative decoding with vLLM: MTP via `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`; Gemma 4 E2B with `google/gemma-4-E2B-it-assistant` drafter; DFlash (`"method":"dflash"`, 15 tokens) and DSpark (vLLM only); benchmarked with AIPerf 0.11.0 + SPEED-Bench at concurrency 1 — [Jetson AI Lab speculative decoding](https://www.jetson-ai-lab.com/tutorials/speculative-decoding)

**TensorRT-LLM (desktop/server)**
- Support matrix: Linux x86_64 and aarch64 only (Windows and Jetson not mentioned); GPUs Ampere, Ada, Hopper, Blackwell (Turing not listed = community support at best); Ampere precisions FP16/BF16/INT8/INT4 (no FP8); Qwen3/Qwen3-MoE listed, Qwen3.5/Qwen3-Next (Gated DeltaNet) not listed (page dated 15 Sep 2025, so possibly stale) — [TensorRT-LLM support matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)
- Native Windows TensorRT-LLM was announced as a beta in 2023 for RTX 30/40 with 8 GB+; no 2026 status found — [NVIDIA blog (2023)](https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/)
- TensorRT-LLM guided decoding: XGrammar (and llguidance) backends; JSON, JSON Schema, regex, EBNF; `guided_decoding_backend: xgrammar` in trtllm-serve config; schema via OpenAI `response_format` — [TensorRT-LLM guided decoding](https://nvidia.github.io/TensorRT-LLM/latest/features/guided-decoding.html)

**Other runtimes**
- ExLlamaV3 (EXL3): tensor/expert-parallel for consumer setups; Qwen3.8-27B EXL3 quants built with exllamav3 1.4.2 (cu128, torch 2.9.0) on an RTX 4070 under WSL2; no source on Turing support or Qwen3.5 small models — [exllamav3 GitHub](https://github.com/turboderp-org/exllamav3); [Qwen3.8-27B-exl3](https://huggingface.co/billmccartney/Qwen3.8-27B-exl3)
- ONNX Runtime GenAI model builder lists "Qwen" generically; no evidence it supports Qwen3.5's Gated DeltaNet; a LiteRT-LM conversion of Qwen3.5-0.8B reports its 0.14 engine cannot bind the linear-attention conv/recurrent state buffers — [ORT GenAI model builder](https://www.mintlify.com/microsoft/onnxruntime-genai/guides/model-builder); [litert-community/Qwen3.5-0.8B](https://huggingface.co/litert-community/Qwen3.5-0.8B)
- MLC-LLM: no 2026 release with Qwen3.5 found; the different project pymllm (UbiquitousLearning/mllm) claims Qwen3/Qwen3-VL/Qwen3.5 on Jetson Orin with BF16, W4A16, W8A8 (experimental, Mar 2026) — [mllm README](https://raw.githubusercontent.com/UbiquitousLearning/mllm/main/README.md)
- Ollama v0.40.9 / v0.40.11 (late Oct 2026) pulled in llama.cpp updates — [Releasebot Ollama notes](https://releasebot.io/updates/ollama) (aggregator; low confidence)
- NVIDIA Speech/LLM NIMs: the Speech NIM requires compute capability 8.0+ and 16 GB VRAM (RTX 40xx/50xx, A-series, DGX Spark; WSL2 on RTX 40xx+), no Jetson listed — [NVIDIA ASR NIM support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html)

### Inferences
Per-machine pick for the text parser (batch 1, short output):

| Machine | Fastest documented | Practical now | Notes |
|---|---|---|---|
| Laptop 2070S (Turing, Windows) | llama.cpp CUDA (no NVIDIA TRT-LLM/NIM path for Turing+Windows) | llama.cpp latest b119xx, CUDA 12.4 build, Q4_K_M/Q8_0, GBNF/json_schema | Fix the prompt cache first (Q1); try MTP |
| Orin NX (JetPack 6.2) | Edge-LLM INT4 AWQ (Qwen3.5-2B 47.9 t/s, 0.8B 95.8 t/s on JP 7.2) | llama.cpp built for sm_87 (works on JP 6.2, ~35% slower decode than Edge-LLM on the Orin Nano test) or vLLM Jetson container W4A16 | Edge-LLM officially wants JetPack 7.2: an OS upgrade of a robot is a decision for the user (rebuild of all TRT 10.7 engines, ROS 2 stack) |
| hekate 5090 (Blackwell, Windows) | vLLM/SGLang (WSL2/Docker) with NVFP4, or llama.cpp with native NVFP4 MMQ | llama.cpp Windows CUDA 13.4 build | Shared GPU (Isaac Sim, training): ask first |
| Dionysos A4500 (Ampere, Windows) | vLLM/SGLang in WSL2 (AWQ/GPTQ INT4, no FP8) | llama.cpp CUDA build | Shared lab PC |

- Edge-LLM on Orin NX with a cached system prompt, for a ~30-token command and ~25 JSON output tokens: 0.8B ~35 ms + 25/95.8 s ~ 0.3 s; 2B ~56 ms + 0.52 s ~ 0.6 s; 4B ~120 ms + 0.93 s ~ 1.05 s. Without prefix reuse, a 1,500-token system prompt adds ~0.6 s (0.8B), ~0.85 s (2B), ~1.9 s (4B) on Orin NX. So on the robot the distilled parser should be 0.8B or 2B, the output JSON should be short, and KV/prefix reuse is mandatory.
- Constrained output on the robot: llama.cpp (GBNF + `json_schema`) is the only runtime here whose grammar support is confirmed by source in this pass; Edge-LLM has a Guided Decoding page whose mechanism is unverified. Since the parser is a fine-tuned 0.8B/2B, a fallback "parse + validate + retry/grammar fallback" (what console/llm.py already does with the regex grammar) keeps it safe even without runtime-level constraints.
- Shorter output = proportionally less latency on every runtime (decode dominates on Orin): abbreviated keys and enum values in the schema are cheap wins.

### Gaps
- No source confirmed Edge-LLM's guided decoding implementation (XGrammar?) or whether its KV cache reuse works for Qwen3.5 hybrid layers.
- No Orin NX numbers for vLLM or llama.cpp on Qwen3.5 small models; no Edge-LLM numbers on JetPack 6.2.
- vLLM/SGLang current constrained-decoding backends (XGrammar, llguidance, outlines) and ExLlamaV3/MLC/ORT-GenAI grammar support were not re-verified in this pass (from prior knowledge: vLLM and SGLang support XGrammar and llguidance; llama.cpp can optionally build with llguidance).
- TensorRT-LLM's current Windows status (beyond the 2023 beta) and whether its 2026 versions support Qwen3.5 were not found.
- NVIDIA NIM for LLMs on Jetson Orin: no source found.

## 3. Qwen3.5 family facts (sizes, architecture, quantized releases, VL)

### Takeaway
Qwen3.5 small models (0.8B, 2B, 4B, 9B; released 2 Mar 2026, Apache 2.0) are dense, natively multimodal (early-fusion vision) hybrid models: three Gated DeltaNet (linear-attention, recurrent-state) layers per one gated full-attention layer, 262k native context, a 248k vocabulary and MTP heads. The hybrid layers are why many runtimes (ORT GenAI, LiteRT, older TRT-LLM) lag, and why prefix caching is special in llama.cpp; KV cache is tiny. No official Qwen-org FP8/AWQ/GPTQ/GGUF releases were found for the small sizes; RedHat (W4A16, W8A8, FP8-dynamic), unsloth (GGUF incl. MTP variants) and NVIDIA/AxionML (NVFP4) supply them.

### Cited Findings
- Qwen3.5 small series 0.8B, 2B, 4B, 9B released together on 2 Mar 2026; base versions also released; 0.8B/2B "small, fast, and ideal for edge devices", 4B "a surprisingly powerful multimodal base for lightweight agents" — [GIGAZINE](https://gigazine.net/gsc_news/en/20260303-qwen-3-5-small)
- Qwen3.5-2B: 24 layers, hidden 2048, layout 6 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN)); GDN 16 heads for V and 16 for QK, head dim 128; Gated Attention 8 Q heads, 2 KV heads, head dim 256, RoPE dim 64; FFN 6144; vocabulary 248,320 (padded), tied LM head; MTP "trained with multi-steps"; 262,144 native context; vision encoder; early-fusion training on multimodal tokens; recommended runtimes SGLang/vLLM (main/nightly), Transformers main, KTransformers; no official quantized variants listed (~278 community quantizations) — [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- Qwen3.5-4B: 32-layer hybrid stack of Gated DeltaNet and Gated Attention blocks — [Azure AI catalog FW-Qwen3.5-4B](https://ai.azure.com/catalog/models/FW-Qwen3.5-4B)
- Context extensible to 1,010,000 tokens with YaRN; 0.8B handles text, image, video, thinking mode and tool calling — [tinyweights write-up](https://tinyweights.dev/posts/qwen3-5-tiny-multimodal-thinking-model/); [InsiderLLM small models guide](https://insiderllm.com/guides/qwen-3-5-small-models-9b-beats-30b/)
- Dense vs MoE: some secondary sources say "Gated Delta Network + MoE" for the small series; NVIDIA's Edge-LLM support list classes 0.8B-27B as dense and 35B-A3B / 122B-A10B as MoE — [GIGAZINE](https://gigazine.net/gsc_news/en/20260303-qwen-3-5-small); resolved by [Edge-LLM supported models](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/getting_started/supported-models.html)
- NVIDIA's Megatron-Bridge docs treat Qwen3.5 small models (0.8B, 2B, 4B) as vision-language models ("Qwen 3.5 / 3.6") — [NeMo Megatron-Bridge Qwen3.5-VL](https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/vlm/qwen35-vl.html)
- Later generations exist only in larger sizes in the sources found: Qwen3.6 27B / 35B-A3B, Qwen3.8 27B (Edge-LLM list); no Qwen3.6/3.8 small (0.8-4B) release found — [Edge-LLM supported models](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/getting_started/supported-models.html)
- Quantized checkpoints: RedHatAI Qwen3.5-4B FP8-dynamic, W4A16 and W8A8 via llm-compressor (examples cover dense VL and MoE variants) — [vLLM llm-compressor Qwen3.5](https://docs.vllm.ai/projects/llm-compressor/en/latest/key-models/qwen3.5/); `RedHatAI/Qwen3.5-4B-quantized.w4a16` (Orin) and `AxionML/Qwen3.5-4B-NVFP4` (Thor) — [Jetson AI Lab Qwen3.5 4B](https://www.jetson-ai-lab.com/models/qwen3-5-4b); a community FP8 W8A8 of Qwen3.5-4B (vision kept BF16) lost ScreenSpot-v2 56.5% -> 51.6% — [Shashwat42/Qwen3.5-4B-FP8](https://huggingface.co/Shashwat42/Qwen3.5-4B-FP8)
- GGUFs with MTP tensors: `shujunyi/Qwen3.5-4B-MTP-Q4_0` (warns requantizing from an 8-bit source is lossier than from BF16) — [HF card](https://huggingface.co/shujunyi/Qwen3.5-4B-MTP-Q4_0); `unsloth/Qwen3.5-2B-MTP-GGUF` — [llama.cpp issue #24587](https://github.com/ggml-org/llama.cpp/issues/24587)

### Inferences
- KV cache is negligible: for 2B only 6 of 24 layers keep KV (2 KV heads x 256 dim) ~12 KB/token in FP16 -> ~50 MB at 4k context; the 18 GDN layers hold a fixed recurrent state. Context size is not a memory concern on the Orin; cache-reuse mechanics are.
- The tied 248k-vocab LM head is a big share of a 0.8B model's per-token bytes (on the order of 0.25 B params), so decode on 0.8B gains less from 4-bit body quantization than expected; keep the LM head/embedding at 8-bit or higher when distilling/quantizing.
- Because the same small checkpoints are VLMs, a distilled 2B parser fine-tuned from Qwen3.5-2B could keep its vision tower and later double as the "which object" VLM (one model in memory on the robot) — at the cost of making the fine-tune multimodal-aware.
- Quantize the distilled model from BF16 directly (llama-quantize from an F16/BF16 GGUF; AWQ via ModelOpt/llm-compressor for Edge-LLM/vLLM), and keep the MTP head if MTP speculative decoding is wanted.

### Gaps
- The official Qwen org pages for 0.8B/4B were not opened; exact 0.8B/4B hidden sizes and the vision encoder size were not confirmed.
- No official Qwen-published GGUF/AWQ/GPTQ/FP8 for Qwen3.5 small sizes was found (may exist in the 21-item collection).

## 4. Small VLMs runnable on Orin NX 16 GB: formats and measured latency

### Takeaway
Qwen3.5 0.8B/2B/4B (native VLMs) and Qwen3-VL 2B/4B have official Orin NX 16 GB numbers in TensorRT Edge-LLM (INT4 AWQ LLM + FP16 ViT): Qwen3.5-2B VLM ViT 72.6 ms, ~296-token prefill 171 ms, decode 48 t/s; Qwen3.5-0.8B VLM ViT 25 ms, decode 95 t/s. Cosmos-Reason2-2B on the similar-bandwidth Orin Nano Super gives image TTFT 75 ms (vLLM) / 114 ms (llama.cpp) / 420 ms (Edge-LLM). For "which object" questions on the robot, a Qwen3.5-2B VLM is the best-documented choice; Gemma 4 E2B/E4B, InternVL3.5 and Cosmos-Reason2 are supported in Edge-LLM but lack Orin NX numbers. Larger VLMs belong on the 5090.

### Cited Findings
- Orin NX 16 GB, Edge-LLM 0.10.0, JetPack 7.2, batch 1, INT4 AWQ / FP16 vision (runtime dashboard): Qwen3.5-0.8B VLM prefill 296 tokens 121.0 ms, ViT 25.1 ms, decode 95.3 t/s; Qwen3.5-2B VLM 170.6 ms, ViT 72.6 ms, 48.3 t/s; Qwen3.5-4B VLM 371.1 ms, ViT 73.3 ms, 27.0 t/s; Qwen3-VL-2B 93.5 ms (292 tokens), ViT 77.1 ms, 58.5 t/s; Qwen3-VL-4B 217.2 ms, ViT 77.4 ms, 31.7 t/s; Qwen3-VL-8B 400.5 ms, ViT 110.5 ms, 18.4 t/s — [Edge-LLM performance benchmarks](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/performance/performance-benchmarks.html)
- Edge-LLM VLM list: Qwen2.5-VL 3B/7B, Qwen3-VL 2B/4B/8B, Cosmos-Reason2 2B/8B, InternVL3/3.5 1B-14B, Phi-4-Multimodal, Gemma 4 (E2B, E4B, 12B, 26B-A4B, 31B; image and audio input on some variants), Qwen3.5/3.6 VLM (original, non-pre-quantized VLM checkpoints) — [Edge-LLM supported models](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/getting_started/supported-models.html); [Edge-LLM 0.9.0 supported models](https://nvidia.github.io/TensorRT-Edge-LLM/0.9.0/user_guide/getting_started/supported-models.html)
- Cosmos-Reason2-2B on Orin Nano Super (JetPack 6.2.2): image TTFT llama.cpp Q4_K_M 114 ms, vLLM AWQ 75 ms, Edge-LLM 420 ms; decode 38 / 56 / 60 t/s; first image request in llama.cpp and vLLM takes 0.5-3 s (lazy ViT compile) — [hokwangchoi/jetson-orin-nano-benchmarks](https://github.com/hokwangchoi/jetson-orin-nano-benchmarks)
- Jetson AI Lab Cosmos Reason2 tutorial lists AGX Thor, AGX Orin 64/32 GB and Orin Nano Super (2B only with vLLM due to memory) — [Jetson AI Lab Cosmos Reason2](https://www.jetson-ai-lab.com/tutorials/cosmos-reason2-vlm); a community report runs Cosmos-Reason2-2B on an 8 GB Orin Nano — [NVIDIA forum](https://forums.developer.nvidia.com/t/cosmos-reason2-2b-running-on-jetson-orin-nano/361074)
- Jetson AI Lab benchmark table: SmolVLM 2B 8.1 (Orin Nano) and 12.9 (Orin Nano Super), units not given in the snippet; Orin NX numbers "being collated" — [Jetson AI Lab benchmarks](https://jetson-ai-lab.com/benchmarks.html)
- Qwen3.5 9B VLM on Orin via vLLM container with a W4A16 checkpoint — [Jetson AI Lab Qwen3.5 9B](https://www.jetson-ai-lab.com/models/qwen3-5-9b)
- Gemma 4 E2B has a paired MTP "assistant" drafter (`google/gemma-4-E2B-it-assistant`) and a QAT W4A16 checkpoint (`google/gemma-4-E2B-it-qat-w4a16-ct`) used with vLLM on Jetson — [Jetson AI Lab speculative decoding](https://www.jetson-ai-lab.com/tutorials/speculative-decoding)
- Our laptop text-parsing bench: Gemma 4 E2B 37/45 (541 ms), E4B 43/45 (1067 ms) vs Qwen3.5 2B 42/45 (364 ms), 4B 45/45 (646 ms) — [console/README.md](C:\Users\Mys\Desktop\RoboARM 2026\console\README.md)

### Inferences
- One camera frame + short question on Orin NX with Qwen3.5-2B VLM (Edge-LLM): ~73 ms ViT + ~170 ms prefill + 20 answer tokens at 48 t/s (~420 ms) ~ 0.65 s; 0.8B ~0.4 s. With llama.cpp (Q4_K_M + mmproj) expect roughly 1.3-1.6x slower decode based on the Orin Nano Cosmos test.
- The image token count drives prefill: downscale the wrist/mast frame to the model's minimum useful resolution (the benchmark's ~230 image tokens is a good target) before asking.
- Run the VLM in the same process/model as the parser only if memory is tight; Orin NX 16 GB shares memory with the robot stack (depth TensorRT, YOLO, ROS 2), and the Edge-LLM INT4 2B footprint (~4.3 GB measured for Cosmos-Reason2-2B) is modest.
- On the 5090: any 7-9B VLM (Qwen3.5 9B NVFP4, Qwen3-VL 8B) via vLLM is far faster and better; latency then is network + ~100 ms.

### Gaps
- No Orin NX latency found for SmolVLM2, Moondream (2/3), Gemma 4 E2B/E4B, InternVL3.5 1-2B, or Nemotron Nano VL.
- llama.cpp mmproj support/speed for Qwen3.5 vision on CUDA (sm_87, sm_75) not verified.
- NVIDIA Cosmos Reason2 latency on Orin NX specifically not found.

## 5. ASR: fastest Parakeet TDT 0.6B v3 paths, RTF on CPU / 2070S / Orin, and better Slovak/Czech models

### Takeaway
Parakeet TDT 0.6B v3 int8 on CPU is already fast for push-to-talk (our 154-300 ms for 1.4-3.5 s); GPU paths are 1.6x (T4 CUDA) to ~9x (RTX 5070 Ti TensorRT) faster than a fast desktop CPU in onnx-asr's tests. Its weakness is accuracy and script control for Slovak/Czech: Parakeet v3 auto-detects language with no forcing (our Cyrillic outputs). Canary-1B-v2 (same Granary data, takes the source language as a prompt) is clearly better on Czech FLEURS (7.86% vs 11.01%) and a Slovak fine-tune by KInIT (kinit/canary-1b-v2-sk) cuts Common Voice 24 Slovak WER from 14.00% to 8.27%, at ~4.5x the CPU cost of Parakeet. NVIDIA's new Nemotron-3.5-ASR streaming 0.6B (June 2026) lists Czech and Slovak but is much worse on them (~20-27% FLEURS WER). Speech NIM does not run on Turing or Jetson.

### Cited Findings
**Parakeet TDT 0.6B v3**
- Released on HF 14 Aug 2025; 25 European languages with automatic language detection; FLEURS WER cs 11.01%, sk 8.82%, en 4.85% (greedy, no LM, punctuation/case removed); Open ASR Leaderboard English average 6.34%, RTFx 3,332.74 (hardware unstated); NeMo 2.4; streaming supported via a NeMo chunked script — [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3); report [arXiv 2509.14128](https://arxiv.org/abs/2509.14128)
- sherpa-onnx int8 export: encoder.int8.onnx 622 MB, decoder 12 MB, joiner 6.1 MB; with 2 threads RTF 0.325 on a 3.8 s clip, 0.028 on a 334 s file (CPU not stated; `provider="cpu"`) — [sherpa-onnx NeMo transducer models](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html)
- onnx-asr RTFx (audio duration / processing time) for NeMo Parakeet v2/v3: Ryzen 9800X3D CPU 36, Cortex-A53 1.0, T4 CUDA 57, RTX 5070 Ti TensorRT 320; Canary v2: 8 / N/A / 21 / 36; supports Parakeet, Canary v1/v2, GigaAM, Whisper, VAD for long audio; providers CUDA, TensorRT, CoreML, DirectML, ROCm, WebGPU — [istupakov/onnx-asr](https://github.com/istupakov/onnx-asr)
- ONNX quantization study of Parakeet TDT-0.6B-v3 on CUDA, batch 16: int8 RTFx 97.10 (1.02 GB), int4 112.94 (higher WER) — [arXiv 2604.14493](https://arxiv.org/pdf/2604.14493)
- sherpa-onnx int8 Parakeet v3 16.5x real-time on an M5 MacBook Air — [BenchLM](https://benchlm.ai/models/parakeet-tdt-0-6b-v3) (secondary)
- Whisper Large V3 vs Parakeet V3 on FLEURS: cs 11.3 vs 11.0, sk 8.4 vs 8.8; English Open ASR average Parakeet v3 6.32% vs Whisper Large V3 Turbo 7.83% — [Whisper Notes blog](https://whispernotes.app/blog/parakeet-v3-default-mac-model) (vendor blog)

**sherpa-onnx / ONNX Runtime state**
- sherpa-onnx v2.0.1 (26 Oct 2026; Canary greedy-search fix), v2.0.0 (23 Oct; Qwen3-ASR examples, Canary language-ID handling), v1.13.8 (10 Sep; onnxruntime 1.28.2, OpenVINO NPU), v1.13.5 (11 Aug; "Build libs and Python wheels for CUDA 13", Nemotron 3.5 ASR streaming ONNX export, onnxruntime 1.27.1 on Linux aarch64, Qualcomm QNN) — [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases)
- onnxruntime v1.27.0: "CUDA 12 packages are deprecated, please move to CUDA 13 ASAP"; v1.28.0 makes cuDNN and cuFFT optional at runtime for the CUDA EP — [ORT v1.27.0](https://newreleases.io/project/github/microsoft/onnxruntime/release/v1.27.0); [ORT 1.28.0](https://newreleases.io/project/pypi/onnxruntime/release/1.28.0)
- sherpa-onnx's CUDA EP library has cuDNN as a hard dependency in LocalAI's build (July 2026); an older (2024) Windows report failed on cuDNN 8 vs 9 mismatch — [LocalAI PR #11145](https://github.com/mudler/LocalAI/pull/11145); [sherpa-onnx issue #878](https://github.com/k2-fsa/sherpa-onnx/issues/878)

**Better Slovak/Czech candidates**
- Canary-1B-v2 (978M params, HF 14 Aug 2025): ASR + translation for 25 European languages; FLEURS cs 7.86%, FLEURS-25 average 8.40%, MLS-6 7.27%, Open ASR Leaderboard mean 7.15%, RTFx 749; "outperforms Whisper-large-v3 on English ASR while being 10x faster" — [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2); [arXiv 2509.14128](https://arxiv.org/abs/2509.14128)
- KInIT Slovak fine-tune `kinit/canary-1b-v2-sk` (CC-BY-4.0, NeMo checkpoint): Common Voice 24 test WER 8.27% vs 14.00% for base canary-1b-v2 — [kinit/canary-1b-v2-sk](https://huggingface.co/kinit/canary-1b-v2-sk)
- A DirectML-oriented ONNX export of canary-1b-v2 exists (encoder fp + int8, decoder from istupakov's export) — [Masterx/canary-1b-v2-onnx](https://huggingface.co/Masterx/canary-1b-v2-onnx)
- A Slovak Whisper large-v3 fine-tune reports FLEURS sk 9.2 -> 5.5 (base model ambiguous on the card) — [NaiveNeuron/whisper-large-v3-sk](https://huggingface.co/NaiveNeuron/whisper-large-v3-sk/blob/main/README.md)
- Nemotron-3.5-ASR streaming 0.6B (HF 4 Jun 2026, OpenMDW-1.1, cache-aware FastConformer-RNNT, chunks 80/160/320/560/1120 ms, language prompt or `auto`, punctuation and case): 40 locales; cs and sk are "broad-coverage" tier; FLEURS cs 24.18 (80 ms) -> 20.41% (1.12 s) with LangID, sk 25.08 -> 21.28%; en 9.43 -> 7.91%; NeMo 26.06, Transformers >= 5.13.0, NeMo-Speech.cpp GGUF; hardware list includes Jetson and Turing — [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b); sub-100 ms time-to-final per [Together AI](https://together.ai/models/nvidia-nemotron-35-asr)
- Qwen3-ASR 0.6B and 1.7B (Apache 2.0): 30 languages + 22 Chinese dialects, Czech listed; Slovak not confirmed — [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B); [Qwen3-ASR mirror README](https://gitee.com/ahsh14/Qwen3-ASR); supported in sherpa-onnx v2.0.0 examples and in TensorRT Edge-LLM — [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases); [Edge-LLM supported models](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/getting_started/supported-models.html)
- Open ASR Leaderboard paper (arXiv 2510.06961, revised Mar 2026) covers 86 systems over 12 datasets; its multilingual track lists de/fr/it/es/pt (no cs/sk) — [arXiv 2510.06961](https://arxiv.org/abs/2510.06961)

**NIM / Riva**
- Speech NIM: Parakeet-tdt-0.6b-v3 (`type=multi`, offline only), Canary 1b multilingual (offline), Whisper Large v3 (offline), Nemotron ASR Streaming (`type=multi`, 40 locales, streaming only); requires compute capability >= 8.0 and >= 16 GB VRAM; RTX 40xx/50xx, DGX Spark; Windows 11 via WSL2 (driver 570+, Ubuntu 24.04+, Podman recommended); Jetson and Turing not listed; page updated 7 Oct 2026 — [NVIDIA ASR NIM support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html)

### Inferences
- Speed is not our ASR problem; Slovak accuracy and script are. Test Canary-1B-v2 with `source_lang=sk|cs|en` (forcing the language removes the Cyrillic/Polish outputs), and the KInIT Slovak fine-tune, before investing in GPU paths. Language choice could come from the console (a per-user/per-session setting) or from a quick Parakeet pass.
- Speed estimates (onnx-asr ratios; our laptop CPU is slower than a 9800X3D): Canary v2 on the laptop CPU for a 2.5 s command ~0.6-1 s (too slow for comfort), on the 2070S CUDA ~0.1-0.15 s; Parakeet v3 on the 2070S CUDA ~40-60 ms for 2.5 s. Both fit next to a 4B Q4 LLM in 8 GB VRAM.
- Use fp32/fp16 ONNX exports on the CUDA/TensorRT providers, not the int8 (dynamic-quantized MatMulInteger) export, which is a CPU format and may be slow or fall back on the CUDA EP (verify in benchmark).
- On the Orin, ASR on its 8 Cortex-A78AE cores in int8 is plausible (RK3588 A76 cores ran Parakeet v2 int8 at RTF 0.088-0.22, per the sherpa-onnx page), but it competes with the robot stack; GPU onnxruntime on JetPack 6.2 means CUDA 12.6 wheels while upstream ORT is deprecating CUDA 12 — another reason JetPack 7.2 would simplify things later.
- Nemotron-3.5-ASR streaming is attractive for latency (finals ~100 ms after speech) but its cs/sk WER (~20%+) is roughly double Parakeet v3's; not worth it for Slovak commands.
- Whisper large-v3(-turbo) is competitive on Slovak (8.4) and supports language forcing, but is slower than Parakeet/Canary; only worth testing as faster-whisper int8_float16 on the 2070S if Canary disappoints.

### Gaps
- No measured Parakeet v3 RTF on an RTX 2070 (Super), on Orin NX CPU or GPU, or with sherpa-onnx's CUDA/TensorRT providers on Windows.
- Canary-1B-v2 Slovak FLEURS WER and English FLEURS WER not visible on the card (truncated); Whisper large-v3-turbo cs/sk FLEURS not found.
- Whether sherpa-onnx ships a Canary-1B-v2 export (it does support "Canary" per release notes; which size not confirmed) and whether its Windows GPU wheels are CUDA 12 or 13 was not confirmed.
- No WER for any model on short robot commands (our domain) — needs our own test set.
- "Parakeet v4" — no evidence it exists; NVIDIA's 2026 additions found are Nemotron-3.5-ASR streaming and NIM support for v3.

## 6. Voice pipeline latency tricks (streaming ASR, VAD, warm models, prompt cache, speculative decoding)

### Takeaway
For push-to-talk with ~2-3 s commands, the remaining latency is ~0.2-0.3 s ASR (CPU) + ~0.65 s LLM. The largest LLM win is likely making the system-prompt prefix actually cached for the hybrid Qwen3.5 (checkpoint settings), then shortening the JSON output and trying MTP; for ASR, moving Parakeet to the GPU or choosing a language-forced model matters more for accuracy than for speed. Streaming ASR only pays off for hands-free use, and the multilingual streaming model is weak on Slovak.

### Cited Findings
- llama-server: prompt cache on by default; context checkpoints for hybrid models (`--ctx-checkpoints 32`, `--checkpoint-min-step 8192`); `--cache-ram` host prompt cache; `--slot-save-path` to persist slot state; `--warmup` default on; MTP/n-gram speculative types — [llama-server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md); [docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- Qwen3.5 hybrid cache problems: full re-processing warnings, checkpoint invalidation when the prompt middle changes — [issue #24587](https://github.com/ggml-org/llama.cpp/issues/24587); [issue #24714](https://github.com/ggml-org/llama.cpp/issues/24714)
- Edge-LLM lists KV cache reuse and MTP/EAGLE3/DFlash/DSpark — [Edge-LLM overview](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/overview.html)
- Nemotron-3.5-ASR streaming chunk sizes 80-1120 ms, sub-100 ms time-to-final — [HF card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b); [Together AI](https://together.ai/models/nvidia-nemotron-35-asr)
- onnx-asr and sherpa-onnx both support VAD-segmented recognition (sherpa-onnx Flutter VAD+ASR examples in v1.13.6) — [onnx-asr](https://github.com/istupakov/onnx-asr); [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases)
- First image requests in llama.cpp/vLLM on Orin take 0.5-3 s (lazy compile), i.e. warm-up matters — [jetson-orin-nano-benchmarks](https://github.com/hokwangchoi/jetson-orin-nano-benchmarks)
- Our console already warms the LLM with one request ("domov") at start and loads sherpa-onnx once on a helper thread — [console/llm.py](C:\Users\Mys\Desktop\RoboARM 2026\console\llm.py); [console/voice.py](C:\Users\Mys\Desktop\RoboARM 2026\console\voice.py)

### Inferences
- Order of work: (1) confirm prefix reuse with Qwen3.5 (log + `timings.prompt_n`), set `-cms 0`/small, keep the static prompt first and the names list + command last; if hybrid checkpoints stay unreliable, Qwen3 4B 2507 (pure attention, 44/45, 535 ms) is the fallback with reliable KV prefix reuse; (2) shorten the output JSON (short keys/enums, no optional whitespace; `max_tokens` close to the longest valid answer); (3) try `--spec-type draft-mtp` with an MTP GGUF and `ngram-mod`; (4) move Parakeet to the GPU or switch to Canary for sk/cs.
- Keep the GPU from dropping to idle clocks between commands on the laptop (the first request after idle pays clock ramp-up); measure cold vs warm.
- Optional: start LLM prefill of the static part before the user finishes speaking is already what a cached prefix gives; with streaming ASR one could also pre-feed partial text, but llama-server would need checkpoint restore per partial — not worth it at our latencies.
- For hands-free operation, sherpa-onnx's Silero VAD + offline Parakeet/Canary on each segment is simpler and more accurate for sk/cs than a streaming multilingual model.

### Gaps
- No measurements of how much time llama-server spends in grammar sampling per token for our schema.
- No public numbers for MTP acceptance on short JSON outputs with Qwen3.5 small models.

## Benchmark plan for language and speech

Rules: desk plan only, nothing has been run. hekate and Dionysos are shared — per CLAUDE.md, check free VRAM (hekate status.ps1) and ask before anything GPU-heavy, never stop others' jobs. Robots: run with the robot stack in its normal state and also with it stopped, to see contention.

**Test sets (make once, reuse everywhere)**
- Text: `console/llm_cases.json` (71 sentences sk/cs/en with expected calls) + a held-out set of new phrasings.
- Audio: record the same command sentences spoken by >= 3 people (sk, cs, en), 16 kHz mono WAV, on the robots' real microphone and the laptop's, with reference transcripts. ~100-200 utterances. Include the 2026-10-21 failure cases (valec, uprac, kocka...).

**What to record**
- LLM: prompt-processing t/s at 64/512/1536/2048 tokens; generation t/s at 32 tokens; end-to-end ms per command through llama-server (`timings.prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms` from the response), cold (first after load / after 60 s idle) and warm; whether `prompt_n` equals only the new tokens (cache hit) or the whole prompt; accuracy on llm_cases.json; VRAM/RAM; on Orin also power mode and tegrastats.
- ASR: RTF (processing time / audio duration) per utterance length bucket (1-2 s, 2-3 s, 3-5 s), cold first-call time, WER and CER per language (normalised: lowercase, no punctuation; also a diacritics-insensitive CER), count of wrong-script outputs (Cyrillic/Polish), and end-to-end command accuracy ASR -> LLM (the metric that matters).

**Laptop (RTX 2070 Super, Windows)**
- Upgrade test: unpack current `llama-b119xx-bin-win-cuda-12.4-x64.zip` and `...-cuda-13.4-x64.zip` beside b11617.
- `llama-bench.exe -m Qwen3.5-4B-Q4_K_M.gguf -ngl 99 -fa 0,1 -p 64,512,1536 -n 32 -r 5 -o md` (and `-d 1536` for decode at depth, if the build's `--help` lists `-d/--n-depth`); repeat for Qwen3.5-2B and 0.8B in Q4_0, Q4_K_M, IQ4_XS, Q6_K, Q8_0, and Qwen3 4B 2507 Q4_K_M as the pure-attention control.
- Prompt cache check: `llama-server.exe -m <gguf> -ngl 99 -c 4096 -np 1 --port 8081 -cms 0 --ctx-checkpoints 32` (vs default `-cms`), send the same command twice and two different commands; grep the server log for "forcing full prompt re-processing"; compare `timings.prompt_n`.
- Speculative: same server plus `--spec-type draft-mtp --spec-draft-n-max 2` / `3` with an MTP GGUF (`unsloth/Qwen3.5-2B-MTP-GGUF`, a 4B MTP GGUF), and separately `--spec-type ngram-mod`; record acceptance and ms, with `response_format` json_schema on (check that grammar + speculation still give valid JSON).
- ASR: sherpa-onnx `sherpa-onnx-offline --provider=cuda --encoder=... --decoder=... --joiner=... --tokens=... --model-type=nemo_transducer <wavs>` with the fp32/fp16 Parakeet v3 export (and int8 on CPU with `--num-threads=4` as baseline); onnx-asr with CUDA/TensorRT providers for Parakeet v3 and Canary-1B-v2 (language forced to sk/cs/en; check onnx-asr README for the model ids and language argument); NeMo (WSL2 or Windows Python) for `kinit/canary-1b-v2-sk`; optionally faster-whisper `large-v3-turbo` with `compute_type="int8_float16", language="sk"`. Record VRAM with the LLM loaded at the same time (8 GB budget).

**Orin NX (JetPack 6.2, no OS change without the user's approval)**
- `sudo nvpmodel -q` (record mode; use MAXN SUPER if that is the robot's normal mode) and `sudo jetson_clocks` for repeatable numbers; log `tegrastats --interval 500`.
- llama.cpp: build `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 && cmake --build build -j8 --config Release`; run the same llama-bench matrix for Qwen3.5-0.8B/2B (Q4_K_M, Q8_0) and the distilled parser when it exists; then llama-server end-to-end with the console's real prompt and json_schema.
- vLLM: `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` (check it supports JetPack 6.2 / L4T 36.x before pulling — ask before downloading; the Orin Nano report needed JetPack 6.2.2) with `RedHatAI/Qwen3.5-2B` W4A16 (or self-quantised), `--max-model-len 4096`, prefix caching on, `response_format` json_schema; `vllm bench serve --max-concurrency 1` or AIPerf 0.11.0 `--concurrency 1`.
- Edge-LLM: only as an experiment on JetPack 6.2 ("compatible/experimental"); export/quantize INT4 AWQ on an x86 GPU host (laptop or Dionysos), then on the Orin `llm_build --maxBatchSize 1 --maxInputLen 2048 --maxKVCacheCapacity 4096` and `llm_bench --mode prefill` / decode; compare to NVIDIA's Orin NX table (JetPack 7.2). A JetPack 7.2 upgrade (CUDA 13.2, TRT 10.16; all our TRT 10.7 engines and JP6 containers/wheels rebuilt) is a separate decision.
- ASR on Orin: sherpa-onnx int8 CPU with 2/4/6 threads while the robot stack runs; GPU via a JetPack-matched onnxruntime-gpu build if CPU contention is visible.
- VLM (later): Qwen3.5-2B/0.8B with one wrist-camera frame at ~230-300 image tokens: ViT ms, prefill ms, decode t/s, first-image cold time; llama.cpp (GGUF + mmproj) vs vLLM container.

**hekate RTX 5090 / Dionysos A4500 (shared; ask first)**
- llama.cpp Windows CUDA 13.4: llama-bench for Qwen3.5-4B/9B in Q4_K_M vs NVFP4 (5090 only; native sm_120 path) — pp and tg.
- vLLM/SGLang in WSL2 or Docker for a served VLM (Qwen3.5 9B NVFP4 on the 5090; AWQ/W4A16 on the A4500, no FP8 on Ampere): TTFT at concurrency 1 from the laptop over the LAN (network round trip included).
- Speech NIM (Parakeet v3 `type=multi`, Canary) via WSL2 on the 5090 only if a served ASR is wanted (needs 16 GB VRAM; check free VRAM first).

**Pitfalls**
- llama-bench measures neither grammar sampling nor prefix-cache reuse; for our use case the llama-server end-to-end number is the one that counts, and for Qwen3.5 it can differ a lot from llama-bench because of hybrid checkpoints.
- `-np` auto (default -1) creates several slots; keep `-np 1` so the cached prefix is always in the slot that answers.
- Thinking must stay off (`enable_thinking: false`); a template change in a new build can silently re-enable it (watch `predicted_n`).
- Different GGUF sources quantise differently (unsloth "UD" dynamic quants vs plain); compare like with like, and quantise the distilled model from BF16.
- int8 ONNX ASR models are CPU-oriented; benchmark the fp16/fp32 export on CUDA/TensorRT, and include the TensorRT provider's one-time engine build in "cold" time (cache the engine).
- CUDA 13 builds need a newer driver; ORT is deprecating CUDA 12 packages while JetPack 6.2 is CUDA 12.6 — pin versions on the robots.
- Laptop power: run on mains with the high-performance power plan; laptop GPUs downclock on battery and between requests (measure cold-after-idle separately).
- Jetson: unified memory is shared with the robot's TensorRT/ROS processes — measure with the stack running; record nvpmodel mode; first VLM image request is slow (lazy compile), so warm up after load.
- WER on short commands is noisy: report per-language counts and command-level accuracy, not only WER.
