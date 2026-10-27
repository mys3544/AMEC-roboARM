# The M3's Orin can hear Slovak: a Slovak-tuned Parakeet, a trained stop-word model and a grammar-first parser fit in 3-4 GB and about a second

Compute is not what stands between us and Slovak voice on the robot. The first published Jetson measurement puts Parakeet TDT 0.6B v3 at 0.33 s median per clip on an 8 GB Orin Nano, and an int8 Parakeet of the same size runs at ~0.09 real-time factor on four ARM cores. What failed on 21 October is **language identification**: Parakeet v3 decides the language itself, has no "Slovak" switch, and wrote our Slovak as Russian or Polish. Three fixes follow, in order. First, a recogniser that only knows Slovak: KInIT's Parakeet fine-tune (Common Voice 24: 7.92 % WER vs 9.07 % base; KInIT's noise-augmented set: 6.62 % vs 24.32 %), with the base model kept for English because every Slovak fine-tune forgets the other languages. Second, keep the int8 encoder only if our own recordings show it costs nothing; this month it was shown to drop whole sentences on longer audio. Third, boost the ~60 command words, with tools that work but are a few weeks old. The stop lane must not wait for any of this. No product spots "stoj" or "stůj" (Porcupine trains 9 languages, not Slovak or Czech), so we train small openWakeWord-style models on synthetic plus real Slovak speech and recordings of the robot's own noise. The 4B parser that answers in 0.65 s on the laptop would need ~2 s on the Orin. On the robot the grammar goes first and a distilled 0.8B second, with a compact output. Speech-native models such as Gemma 4 E2B publish no Slovak or Czech audio results, so they belong in P6's comparison, not in the stack. Speech out is pre-rendered phrases plus Piper, with Supertonic 3 on trial. The microphone is the largest unknown and costs an hour to measure. With a test set recorded this week, the stack can be on the M3 for the HRI demo on 20 November, and the study's numbers can be in the late-breaking report on 8 December.

Labels: **[MEASURED-ORIN]** measured on a Jetson Orin; **[MEASURED-other]** measured elsewhere or by the authors; **[CLAIM]** a vendor or author statement; **[2nd]** a secondary source. Details, more numbers and all gaps are in the four notes in `research_notes/Slovak voice on the robot/`: `asr_slovak_czech_on_orin.md`, `wake_stop_vad_microphones.md`, `parser_and_speech_native.md` and `tts_slovak_czech.md`.

## Slovak fails on language identification, not on compute

The published accuracy ranks the models clearly. On FLEURS (read sentences), Canary-1B-v2 is the best open model for both our languages. Parakeet v3 and Whisper large-v3 are level, about 50 % worse. The 2026 streaming and LLM-style recognisers are two to three times worse, or lack Slovak entirely. Slovak fine-tunes close most of the gap, above all on noisy speech:

| model | FLEURS sk | FLEURS cs | FLEURS en | CV24 sk (base -> Slovak fine-tune) | KInIT noisy set sk (base -> fine-tune) |
|---|---|---|---|---|---|
| Parakeet TDT 0.6B v3 (600M) | 8.82 | 11.01 | 4.85 | 9.07 -> **7.92** | 24.32 -> **6.62** |
| Canary-1B-v2 (978M) | **5.74** | **7.86** | 4.50 | 14.00 -> 8.27 | 14.39 -> 6.37 |
| Whisper large-v3 (1.55B) | 8.40 | 11.33 | 4.25 | 19.70 -> 7.68 | 16.22 -> 5.67 |
| Whisper large-v3-turbo (809M) | 10.7 | -- | -- | 29.23 -> 9.29 | 18.35 -> 5.60 |
| KInIT whisper-small-sk (0.2B) | -- | -- | -- | -- | 47.90 -> 8.64 |
| Nemotron-3.5-ASR streaming 0.6B, 1.12 s chunks | 21.28 | 20.41 | 7.91 | -- | -- |
| Qwen3-ASR-0.6B | no Slovak | 21.80 (group average) | -- | -- | -- |

WER in %, all [MEASURED-other]. Sources: the Canary/Parakeet report's Table 11 ([arXiv 2509.14128](https://arxiv.org/html/2509.14128)), KInIT's cards ([parakeet-sk](https://huggingface.co/kinit/parakeet-tdt-0.6b-v3-sk), [canary-sk](https://huggingface.co/kinit/canary-1b-v2-sk), [whisper-large-v3-sk](https://huggingface.co/kinit/whisper-large-v3-sk), [turbo-sk](https://huggingface.co/kinit/whisper-large-v3-turbo-sk)), SloPalSpeech for turbo on FLEURS ([arXiv 2509.19270](https://arxiv.org/html/2509.19270v1)), and the cards of [Nemotron-3.5-ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) and [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B). KInIT's noisy set is not public.

Four points decide the design.

**The fine-tunes are Slovak-only.** Every KInIT card warns that fine-tuning "significantly degrades performance on other languages". Canary's fine-tune also loses its translation. So one model cannot serve all three languages after fine-tuning, and nothing is published on how a Slovak-tuned model treats Czech. Our command stems are shared between the two languages (`kock`/`kostk`, `zober`/`vezmi`, `uprac`/`uklid`), so a Slovak-flavoured transcript of Czech speech may still parse. That is an inference to test, not a fact. No Czech fine-tune of Parakeet or Canary was found. Base Canary with `source_lang="cs"` is the best published Czech model.

**Parakeet is the right size for the Orin.** KInIT's Parakeet and Canary fine-tunes are equally accurate (6.62 % vs 6.37 % on their noisy set), and the card says Parakeet is the faster of the two. On a CPU, Canary costs ~4.5x as much (onnx-asr RTFx 8 vs 36 on a desktop CPU, [onnx-asr](https://github.com/istupakov/onnx-asr)). Parakeet also has the better noise curve: on LibriSpeech with MUSAN noise at -5 dB it scores 12.21 % against Canary's 19.38 %. The first Jetson numbers come from an 8 GB Orin Nano at 25 W on JetPack 7.2.1, with English clips [MEASURED-ORIN] ([Turing Pi, 26 Oct 2026](https://turingpi.com/whisper-vs-parakeet-jetson-orin-nano-benchmarks/)):

- Parakeet v3 under NeMo FP32: real-time factor 0.045, P50 0.33 s and P95 0.52 s per clip.
- That run's process held 6.35 GiB of memory (the PyTorch stack). The int8 ONNX files total 640 MB.
- Whisper large-v3-turbo: real-time factor 0.22, 1.74 s per clip. It is five times slower than Parakeet on the same board.

On ARM cores, sherpa-onnx runs the int8 Parakeet v2 (the same encoder size as v3) at RTF 0.088 with 4 threads on an RK3588 [MEASURED-other] ([sherpa-onnx](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html)). We estimate ~0.25-0.35 s on the Orin's CPU for a 2.5 s command, and ~0.1 s on its GPU. These are our estimates.

**The int8 encoder needs a check.** A sherpa-onnx pull request merged today (27 Oct 2026) measured the Parakeet v3 int8 export at 17.2 % WER on concatenated clips with pauses, against 5.9 % for the fp32 export, and blamed the int8 encoder. On single utterances int8 greedy scored 2.21 %, with no fp32 figure beside it [MEASURED-other] ([PR #4084](https://github.com/k2-fsa/sherpa-onnx/pull/4084)). Our console runs the int8 export. Our commands are short, so the effect is probably small, but it has to be measured on our Slovak recordings before int8 goes on the robot.

**Language routing.** With a Slovak model and an English one, something must choose between them, and there are three ways:

- a setting (the event page already has an SK / EN switch);
- the wake phrase ("Ahoj robot" against "Hey robot");
- run both recognisers and keep the transcript the grammar parses, with the session language breaking ties. Two Parakeets cost twice the time and memory, ~0.2 s on the GPU.

The first two are deterministic and free; the third needs no habits from the speaker.

## A Slovak-only recogniser first, boosting second, SOUNDS_LIKE last

Boosting the command words at decode time is now possible in all three toolchains. It does not fix a wrong language, and the tools are young:

- **NeMo GPU phrase boosting** (GPU-PB, "TurboBias", ASRU 2025) supports CTC, TDT and Canary without retraining ([NeMo docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/asr_customization/word_boosting.html)). On 108M English models, key-phrase F-score rose by 8-10 points with greedy decoding and by 17-23 points with beam search, at a 2-5 % speed cost. Precision fell (97 % -> 90 %) [MEASURED-other] ([arXiv 2508.07014](https://arxiv.org/html/2508.07014)). One user found it "did not really work" for Parakeet TDT [2nd] ([NeMo #14500](https://github.com/NVIDIA-NeMo/NeMo/issues/14500)). No result exists for any non-English model.
- **sherpa-onnx hotwords** need `modified_beam_search`. With Parakeet v3 that mode returned "Yeah." or empty text on about a third of requests in March ([issue #3267](https://github.com/k2-fsa/sherpa-onnx/issues/3267)). The fix (per-frame beams, a `length_reward`, 0.5 recommended for v3) was merged today and is not yet in a release ([PR #4084](https://github.com/k2-fsa/sherpa-onnx/pull/4084)).
- **Whisper:** faster-whisper takes `hotwords` and `initial_prompt` ([code](https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/faster_whisper/transcribe.py)). whisper.cpp can hold decoding to a GBNF grammar of the commands [2nd] ([openHAB docs](https://openhab.org/addons/voice/whisperstt)). A grammar-held KInIT whisper-small-sk or medium-sk is one experiment worth a day. On the Orin Nano, Whisper small ran at RTF 0.09.

Our `SOUNDS_LIKE` regexes stay as the last, deterministic layer. Measure boosting by command exact match, not WER. Never boost "stop" in the command recogniser as a substitute for the stop lane.

## No product hears "stoj": the stop lane is a keyword model we train

The stop lane (PLAN.md M3) must run always, never through the recogniser or the LLM, and halt the robot within 0.5 s of the word, with the motors running. Today's stop is text matching on Parakeet's output after push-to-talk, so it waits for both. The options, as of October 2026:

- **Porcupine** trains en, fr, de, it, ja, ko, zh, pt and es. "Czech, Slovak, and Polish are not included" ([Picovoice docs](https://picovoice.ai/docs/model-api/porcupine/)).
- **openWakeWord** uses a frozen shared speech embedding plus a tiny classifier per word ([repo](https://github.com/dscripka/openWakeWord)):
  - one Raspberry Pi 3 core runs 15-20 models;
  - code Apache-2.0; its pretrained models are non-commercial (we train our own);
  - documented "English only"; last release February 2024;
  - targets <5 % false rejects and <0.5 false accepts per hour, measured in the deployment environment.
- **microWakeWord** is a streaming MixConv model run every 30 ms, trained on Piper samples, with a Linux runtime ([repo](https://github.com/kahrendt/microWakeWord), [pymicro-wakeword](https://github.com/OHF-Voice/pymicro-wakeword)). It is likewise documented for English.
- **sherpa-onnx's keyword spotter** is open-vocabulary, but its models are Chinese and English only ([docs](https://k2-fsa.github.io/sherpa/onnx/kws/index.html)).

So we train our own: one openWakeWord classifier each for `stop` (said in all three languages), `stoj`/`stojte`, `stůj`/`stujte` and `zastav`. They share the embedding pass, so the stop lane costs about one stream of CPU. Synthetic speech alone is not enough. Studies of TTS-trained keyword spotting find that real negatives from the deployment environment and a modest number of real positives close most of the gap: a 46 % false-reject rate fell to 18 % with real negatives added [2nd] ([Park et al. 2024](https://www.isca-archive.org/syndata4genai_2024/park24_syndata4genai.pdf); [Synth4Kws](https://arxiv.org/abs/2407.16840)). Slovak TTS voices are few (one Piper voice, Supertonic 3's presets, OmniVoice cloning for research use). The recipe:

- 5-10k synthetic positives per word, perturbed and mixed with the robot's own noise at 0-20 dB SNR;
- Slovak and Czech near-misses ("stojí", "postoj", "stopa", "zastávka", "stôl");
- hours of real negatives recorded through the M3's microphone: motors, lab talk, Slovak and Czech radio;
- 20-50 real positives from lab members.

Tune the threshold for recall: a false stop costs seconds, a missed stop can cost a collision. The parser's stop rule stays as a second detector, and the hardware e-stop stays as the safety function. A voice stop is a convenience on top of it, never instead of it.

## Wake word and end of speech: push-to-talk first, a trained wake phrase second

Push-to-talk (the console and a physical button on the robot) has zero endpoint delay and zero false wakes. It should carry the demo and the study's scripted part. For hands-free use, the same openWakeWord training gives a wake phrase per language family, which can also pick the recogniser (previous sections). End of speech is Silero VAD: MIT, ~2 MB, trained on 6,000+ languages, under 1 ms per 30 ms chunk on one CPU thread [CLAIM] ([silero-vad](https://github.com/snakers4/silero-vad)). It is already inside sherpa-onnx and openWakeWord. The cost is the silence it waits for: Yahboom waits 1.5 s, our plan 0.5 s. Endpoint in two steps. After ~300 ms of silence, run the recogniser and the grammar. If they yield one complete, valid command, act; otherwise wait out the full 0.5-0.7 s. This trims ~0.2-0.3 s from simple commands without cutting off a slow "zober... červenú kocku".

## The laptop's 4B parser would take about two seconds on the Orin

On the Orin a small LLM's time is decode time. Our schema makes the model write all seven fields with their nulls, ~40-50 tokens. NVIDIA's Orin NX numbers (TensorRT Edge-LLM, INT4, JetPack 7.2) are [MEASURED-ORIN] ([Edge-LLM benchmarks](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/performance/performance-benchmarks.html)):

| Qwen3.5 | decode | prefill, 2,048 tokens |
|---|---|---|
| 0.8B | 95.8 t/s | 0.84 s |
| 2B | 47.9 t/s | 1.16 s |
| 4B | 27.0 t/s | 2.56 s |

That puts the 4B at ~1.7 s per command with Edge-LLM and the 0.8B at ~0.5 s; with llama.cpp, which we would run on JetPack 6.2, the 4B lands nearer 2.5-3 s. llama.cpp decodes at roughly 60 % of Edge-LLM's speed (Orin Nano Super, a 2B model: 38 vs 60 t/s, [benchmarks](https://github.com/hokwangchoi/jetson-orin-nano-benchmarks)). On an Orin NX, Gemma 4 E4B decodes ~13 t/s in llama.cpp [MEASURED-ORIN] ([forum](https://forums.developer.nvidia.com/t/llama-cpp-mtp-lifted-llm-performance-of-jetson-orin/372493)). The prompt prefix must be cached, and Qwen3.5's hybrid layers make that unreliable in llama-server at default settings (`Model export formats.../language_speech.md` section 1).

So on the robot:

1. stop words -> halt;
2. the grammar (`voice.parse`), which answers in under a millisecond and handles most simple commands;
3. only when it cannot, a Qwen3.5 0.8B (2B if 0.8B falls short), LoRA-distilled from paraphrases a large model writes on hekate, labelled by our 4B, and trained on *recogniser-shaped* text: TTS out, through the chosen recogniser with noise, back in. It emits a compact command of ~10-20 tokens, e.g. `pick m3 cube red -` held by a GBNF grammar (~0.1-0.2 s at Edge-LLM speeds);
4. the validator, with read-back and a 2 s cancel window for multi-step plans.

The 4B stays on the laptop as the reference. Training on recogniser-shaped text is also how `SOUNDS_LIKE` stops growing by hand.

## Speech-native models are a comparison, not the stack

No open speech-native model publishes Slovak or Czech speech accuracy and runs comfortably on the Orin:

- **Gemma 4 E2B** (Apache-2.0, a 305M audio encoder, 30 s clips) evaluates ASR on 12 languages without Czech or Slovak ([technical report](https://arxiv.org/html/2607.02770v1)). Its llama.cpp audio path had a transcription bug and no server endpoint after it was merged [2nd] ([#21820](https://github.com/ggml-org/llama.cpp/issues/21820), [#21868](https://github.com/ggml-org/llama.cpp/issues/21868)). It was our weakest text parser (37/45).
- **Ultravox v0.5 1B** (Whisper-turbo encoder, frozen Llama-3.2-1B, MIT) inherits Whisper's languages but reports no ASR accuracy ([card](https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_2-1b)).
- **Qwen3.5-Omni** recognises Czech but is API-only [2nd].
- **Voxtral 3** shows CS and SK in its FLEURS chart, is an API product, and calls tools through a separate LLM ([Mistral](https://mistral.ai/news/voxtral-3/)).

P6's comparison stays new because of the language. Score the cascade against Gemma 4 E2B going from audio to a tool call, by exact command match and p95 latency per language, on the same recordings. Run Gemma first in Transformers on the laptop or hekate, then on the Orin only if a robot moves to JetPack 7.2 (Edge-LLM lists Gemma 4 E2B audio input) or llama.cpp's audio path proves fixed.

## Speech out: pre-rendered phrases, Piper for the rest, Supertonic 3 on trial

- **Piper** has one Slovak voice: `sk_SK-lili` medium, a single speaker at 22 kHz, fine-tuned from an English voice ([card](https://huggingface.co/rhasspy/piper-voices/raw/main/sk/sk_SK/lili/medium/MODEL_CARD)). For Czech it has two, `jirka` and `kasandra`.
- **Supertonic 3** (~99M ONNX, 31 languages including sk and cs, OpenRAIL-M, already in sherpa-onnx) is the one new model likely to beat lili on a CPU. An independent 4-core CPU test measured real-time factor 0.165 at 2 steps ("robotic") and 0.313 at 5 steps ("clear") [MEASURED-other] ([HeyNEO](https://heyneo.com/blog/kokoro-tts-vs-supertonic-3-tts); [repo](https://github.com/supertone-inc/supertonic)).
- **OmniVoice** (600+ languages, voice cloning) has non-commercial weights ([card](https://huggingface.co/k2-fsa/OmniVoice)).
- The 2026 TTS releases from Mistral do not list Slovak or Czech; Qwen3-TTS's language list was not checked.

Most of what the robot says is templated. Pre-render the fixed phrases with the best voice available, store them as WAVs, and synthesize on the robot only names and other new text. Build Slovak number and case agreement ("štyri kocky", "päť kociek") from tables, not from the LLM. Until the module's echo cancellation is verified, gate the wake word while the robot speaks. The stop lane stays live.

## The microphone may decide more than the model

Yahboom describes the M3's module only in marketing terms: a MEMS microphone, a cavity speaker, far-field pickup, echo cancellation and noise reduction [CLAIM]. Microphone count, SNR and whether its processing can be bypassed are not published. Its sister boards are rated at 5 m in quiet and 1 m in noise [CLAIM]. Recognisers degrade gracefully down to ~5 dB SNR (Parakeet averages 8.2 % against 6.3 % clean) and break near 0 to -5 dB (11.7 % and 19.9 %) [MEASURED-other] ([card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)). Wheels and servos next to the microphone can reach that.

Neural enhancement in front of a recogniser can lower accuracy: DeepFilterNet2 output scored 74.5 % word accuracy against 79.5 % for the noisy input [2nd]. Beamforming with a learned noise model cut WER by up to 55 % on a mobile robot after under two minutes of calibration [MEASURED-other] ([arXiv 2303.00829](https://arxiv.org/abs/2303.00829)). A ReSpeaker XVF3800 (4 mics, on-chip echo cancellation and beamforming, USB) is the off-the-shelf upgrade [CLAIM]. A wireless lavalier is the demo's safe choice.

The first hour on the robot answers most of this: run `arecord -l`, record the channel count, and measure SNR at 1 m and 2 m with the base driving and the arm moving.

## The recommended on-robot stack

Memory and latency figures are our estimates unless labelled; "measure" marks what the first week must confirm. Everything except the LLM (and the recogniser, if int8 must go) runs on the CPU, leaving the GPU to perception and the ACT.

| stage | model / runtime | memory | latency | why | fallback |
|---|---|---|---|---|---|
| audio in | Yahboom module via ALSA, 16 kHz mono; push-to-talk button + console | -- | -- | already fitted | lavalier or ReSpeaker XVF3800 |
| stop lane | own openWakeWord models: stop, stoj, stůj, zastav; CPU, always on; calls the halt service directly | <0.1 GB | detect ~0.1-0.3 s after the word (measure); target <0.5 s to zero velocity | no product speaks Slovak; shares one embedding pass; never waits for a model | parser's stop rule; console STOP; hardware e-stop |
| wake (hands-free) | own openWakeWord "Ahoj robot" / "Hey robot" + Silero VAD gate | shared | ~0.1-0.3 s | same training; the phrase picks the language | push-to-talk |
| end of speech | Silero VAD in sherpa-onnx; two-step endpoint 0.3 / 0.5-0.7 s | ~10 MB | 0.3-0.7 s of silence | standard, <1 ms a chunk [CLAIM] | push-to-talk release |
| ASR Slovak + Czech | KInIT parakeet-tdt-0.6b-v3-sk -> ONNX, sherpa-onnx; int8 on 4 CPU threads if it costs nothing on our set, else fp16 on the GPU | ~0.7-1 GB int8 / ~1.3 GB fp16 | ~0.25-0.35 s CPU, ~0.1 s GPU for 2.5 s | removes the wrong-language outputs; noise-trained | base Canary-1B-v2 with source_lang; KInIT whisper-small/medium-sk |
| ASR English | base parakeet-tdt-0.6b-v3 (current) | ~0.7-1.3 GB | as above | fine-tunes forget English | Canary (en) |
| boosting | sherpa-onnx hotwords, master after PR #4084, `length_reward` 0.5; ~60 command forms | -- | beam search instead of greedy (measure) | lifts command words | `SOUNDS_LIKE` |
| parser, fast | `voice.parse` grammar (stop first) | 0 | <1 ms | deterministic, already tested | -- |
| parser, LLM | Qwen3.5-0.8B LoRA-distilled, llama.cpp CUDA sm_87, Q8_0, compact grammar-held output, cached prefix | ~1-1.2 GB | ~0.2-0.5 s | 4B ~2 s on the Orin | 2B; grammar only; laptop 4B over the network |
| validator | rules: known objects, robot free, box seen; read-back + 2 s cancel | 0 | <1 ms | small models leave a semantic gap | ask back |
| speech out | pre-rendered WAVs + Piper sk_SK-lili / cs_CZ-jirka in sherpa-onnx on CPU | ~0.1-0.2 GB | ~0.05 s cached, ~0.2-0.4 s synthesized (measure) | fast, known; most phrases fixed | Supertonic 3 if the A/B favours it |

Total: ~3 GB with int8 recognisers on the CPU, ~4 GB with fp16 on the GPU. From the end of speech to a command: ~0.4-0.6 s by the grammar and ~0.7-1.3 s by the LLM. That meets PLAN.md's "simple commands under 1 s".

## Build order, 27 October to 8 December

1. **This week (to 2 Nov): the test set and the bake-off, on the laptop.**
   - Survey the M3's audio device and measure the SNR with motors running.
   - Record ~10 lab speakers through the robot's module and a close-talk reference at once.
   - Run each candidate on the same files: Parakeet v3 int8 against fp32, KInIT Parakeet-sk exported to ONNX, Canary-1B-v2 with a forced language, and a KInIT Whisper-sk. Score by command exact match through the grammar and the 4B.
   - Start generating the synthetic stop and wake data.
2. **Week 2 (3-9 Nov): on the M3.**
   - A small voice service on the robot speaking the console's `/api/voice` contract, so the console sends its recordings there: one code path, as PLAN.md M3 asks.
   - Silero VAD and the stop-lane models v0, wired to the halt service.
   - First stop-latency runs; check that the ACT's 10 Hz loop keeps time while the recogniser runs.
3. **Week 3 (10-16 Nov).** Language routing, sherpa-onnx built from master for hotwords, the pre-rendered phrase set plus Piper, and the M3.1 sentence end to end. Ask before using hekate's GPU for the parser's distillation (CLAUDE.md).
4. **By 20 Nov: the HRI demo video**, everything on board with the laptop's network unplugged.
5. **To 8 Dec.** The study (next section), the 0.8B parser if it beats the grammar's coverage, the Gemma 4 E2B arm on the same recordings, and the late-breaking report.

## Evaluation plan: real speakers, motors running

The plan serves the demo (20 Nov), the late-breaking report (8 Dec) and later P6.

**Participants and recordings.**
- At least 14 native speakers: 8 Slovak, 6 Czech, mixed genders, recruited beyond the team. Each also speaks English.
- Each speaker gives, in their own language:
  - 40 scripted commands, balanced over pick by colour and size, clean up, sort, look, find, home, fetch, taught names and the two-robot sentences;
  - 10 elicited commands (shown a scene, "tell the robot");
  - 10 stops and 5 non-commands.
- In English each speaker gives 30 scripted commands.
- Each speaker records in two conditions: motors off, and motors running (the base on a scripted path, the arm sweeping).
- The robot's module and a close-talk reference record together at 1.5 m.
- That is about 1,000 Slovak, 750 Czech and 840 English utterances. Two people transcribe a 10 % overlap.
- Recording people needs consent and GDPR handling, and possibly the faculty's ethics approval for an HRI study. Ask about that this week; it can take longer than the engineering.

**Commands.** Each pipeline runs on the same recordings: the laptop baseline (Parakeet int8 + 4B), the on-robot stack (routed Parakeet + grammar + 0.8B), Canary with a forced language, and Gemma 4 E2B speech-native. The measures:

- exact command match: every field equal to the reference; the primary metric;
- action accuracy;
- WER and CER, normalised, plus a diacritic-insensitive CER;
- the wrong-language rate (Cyrillic or Polish output);
- false commands on non-commands;
- p50 and p95 latency from the end of speech to the command, and from the end of speech to the first motion.

Report Wilson 95 % intervals. Compare pipelines with McNemar's test on paired utterances and with a bootstrap over speakers, since utterances cluster by speaker. The headline figure is P6's Pareto plot of exact match against p95 latency, one panel per language.

**Stops.**
- The robot runs a scripted motion while a stop word is played at a random moment.
- At least 200 trials per condition (quiet with motors running; TV speech at ~60 dBA with motors running), using real speakers' recorded stops replayed from a loudspeaker at a fixed level. At least 50 live trials check that replay matches live speech.
- Log the audio on the Jetson clock, the detector's trigger, the halt call and the end of motion.
- Ground truth for the end of motion is 240 fps video synced by a clap or LED on a subset. The base's odometry (6-7 Hz, its clock 55-75 ms off) is too coarse for it.
- Report a CDF from word onset and from word end to zero velocity, with a 0.5 s line, and the miss rate. With 0 misses in 200 trials, the 95 % upper bound is 1.5 % (the rule of three).

**False wakes and false stops per hour.** Record at least 10 hours per condition through the robot's microphone: lab working hours, Slovak and Czech radio and TV overnight, the robot's own spoken phrases, and motors running. With zero events in 10 h, the 95 % bound is 0.3 per hour, under openWakeWord's 0.5 target.

**For the demo:** one Slovak sentence moving both robots, "Stoj!" mid-motion with its latency on screen, a Czech and an English command, and the network unplugged. **For the late-breaking report:** the exact-match and latency table per language and pipeline, the stop CDF and the false stops per hour. As far as this search found, they would be the first published numbers for on-board Slovak and Czech commanding (SMaRTAban, RESEARCH.md P6, ran in the cloud).

## Risks

- **Slovak quality on real speakers is unknown.** All 79/79 so far are synthetic voices, and Czech through the Slovak model is untested. *Mitigation:* the bake-off before any integration; keep Canary with a forced language as the fallback.
- **The tools are weeks old.** The sherpa-onnx beam-search fix merged today and is unreleased. GPU-PB on TDT is unverified. Qwen3.5 prefix caching in llama.cpp is fragile. Gemma 4's audio path is buggy. *Mitigation:* the stack works without any of them (greedy decoding, the grammar, `SOUNDS_LIKE`), and each is an add-on behind a measurement.
- **Contention.** The recogniser shares the CPU with ROS 2 and the bridge, the GPU with the ACT's 10 Hz loop, and 16 GB of memory with all of it. *Mitigation:* read `tegrastats` first; log ACT loop jitter with the recogniser running.
- **Stop words trained on few Slovak voices** may miss real speakers or fire on "stojí" and "postoj". *Mitigation:* real negatives, near-misses, real positives, the parser's stop rule as a second detector.
- **The microphone's processing is unknown**, and the robot's own speech can wake it. *Mitigation:* the one-hour survey; gate the wake word during playback; a lavalier for the demo.
- **JetPack 6.2 against 7.2.** Edge-LLM is official only on 7.2, and onnxruntime is deprecating CUDA 12. A JetPack upgrade means rebuilding every TensorRT engine and is the user's decision. *Mitigation:* plan on llama.cpp and pinned onnxruntime builds on 6.2.
- **Licences.** Parakeet, Canary and KInIT's NeMo models are CC-BY-4.0 (attribution). Supertonic 3 is OpenRAIL-M. OmniVoice and openWakeWord's pretrained models are non-commercial. The Piper runtime is GPL; sherpa-onnx runs Piper voices without it.
- **Safety framing.** A voice stop is a convenience, not a safety function; the hardware e-stop stays. This matters for the Healthcare Challenge, where English (possibly German, unverified) matters more than Slovak. The base models cover both.

## The evidence is thin exactly where we work

Most numbers are the authors' or vendors' own, on read speech. The few Orin measurements (an 8 GB Orin Nano, NVIDIA's Edge-LLM table, a forum report, a Piper write-up) cover no Slovak, no Czech and no running robot. The web-search quota ran out near the end, and Yahboom's pages answered HTTP 429, so the module's specification comes from search summaries. The notes list every gap and where it was looked for.

## Conclusion

Slovak voice on the M3 is buildable from parts that exist today, and what failed so far is not compute. The Orin has it: a Parakeet-class recogniser costs a few tenths of a second and about a gigabyte. What failed in October was a recogniser guessing the language, and Slovak fine-tunes trained on noise fix that, at the price of routing English to a second model. The stop lane cannot be bought in Slovak and must be trained, with real negatives from the robot itself. The laptop's 4B parser does not transfer to the Orin; the grammar plus a distilled 0.8B does. The order of work follows from this: record real speakers this week and let command exact match pick the recogniser, put the recogniser and stop lane on the M3 next week, and add the wake word, parser and speech out in time for the demo on 20 November. The study that follows gives the late-breaking report its first on-robot numbers for Slovak and Czech commands, stop latency and false stops per hour.
