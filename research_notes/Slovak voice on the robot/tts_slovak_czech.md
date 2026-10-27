# Speech out in Slovak and Czech on the robot (state of 27 October 2026)

Scope: the robot's spoken answers from the M3's own speaker: short acknowledgements ("Beriem červenú kocku"), read-backs of a plan with a cancel window, and the M3.1 summary ("Hotovo: M3 štyri kocky, X3 tri"), in Slovak, Czech and English, on the Orin NX's CPU without disturbing perception. PLAN.md names Piper with `sk_SK-lili` ("one mediocre Slovak voice is the only one that runs on the robot today") and `cs_CZ`; Yahboom ships Piper as its local fallback. Desk research only.

Labels: **[MEASURED-ORIN]**, **[MEASURED-other]**, **[CLAIM]**, **[2nd]** as in `asr_slovak_czech_on_orin.md`.

## 1. What exists for Slovak and Czech on a small device

### Takeaway
Three open options cover Slovak and Czech on a CPU: Piper (one Slovak voice, two Czech voices; VITS, tiny, fast, the known quantity), Supertonic 3 (2026, a news item dates it to May; ~99M parameters in ONNX, 31 languages including sk and cs, preset voices, OpenRAIL-M, already wired into sherpa-onnx) and OmniVoice (0.6B, 600+ languages with zero-shot cloning, but non-commercial weights and GPU-oriented). Supertonic 3 is the only new model that plausibly beats Piper's Slovak on quality at a CPU budget, and it is the one to A/B. The big 2026 TTS releases from Mistral and Qwen do not list Slovak or Czech, or were not verified to.

### Evidence
- **Piper Slovak:** `sk_SK-lili` medium, 1 speaker, 22,050 Hz, dataset CC0 (OHF-Voice voice-datasets), "fine-tuned from the U.S. English lessac voice at medium quality" — [MODEL_CARD](https://huggingface.co/rhasspy/piper-voices/raw/main/sk/sk_SK/lili/medium/MODEL_CARD); it is the only Slovak voice — [piper-voices sk_SK](https://huggingface.co/rhasspy/piper-voices/tree/main/sk/sk_SK)
- **Piper Czech:** `cs_CZ-jirka` (low, medium) and `cs_CZ-kasandra` — [piper-voices cs_CZ](https://huggingface.co/rhasspy/piper-voices/tree/main/cs/cs_CZ), [jirka](https://huggingface.co/rhasspy/piper-voices/tree/main/cs/cs_CZ/jirka)
- **Piper runtime:** the maintained Piper is `OHF-Voice/piper1-gpl` (GPL) — PLAN.md sources; sherpa-onnx runs Piper/VITS voices and in v2.0.0 (23 Oct 2026) removed its espeak-ng and piper-phonemize dependencies — [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases)
- **Piper on an Orin Nano** (JetPack 6.x, ROS 2, `en_US-amy-low`): "first audible sound ≈ 800-900 ms" after warm-up, ~4 s for the first sentence before warm-up; CUDA flag "silently ignored" without a working CUDA provider [MEASURED-ORIN, blog, whole ROS pipeline] — [thomasthelliez.com](https://thomasthelliez.com/blog/running-piper-tts-on-nvidia-jetson-orin-nano-with-low-latency/); a Piper voice at ~0.10 RTF on an RK3588 board [2nd] — [Turing Pi RK3588](https://turingpi.com/whisper-cpp-piper-tts-arm64-turing-pi-rk3588/)
- **Supertonic 3:** ~99M parameters across the public ONNX assets; 31 languages "ar, bg, hr, cs, da, nl, en, et, fi, fr, de, el, hi, hu, id, it, ja, ko, lv, lt, pl, pt, ro, ru, sk, sl, es, sv, tr, uk, vi" (+ `na` for unknown); 44.1 kHz output; preset voice styles; model OpenRAIL-M, code MIT; runtimes Python (ONNX Runtime), C++, Rust, Go, Java, C#, Swift, Node, browser [CLAIM] — [supertone-inc/supertonic](https://github.com/supertone-inc/supertonic), [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3); an independent 4-core CPU benchmark: mean RTF 0.165 at 2 diffusion steps ("robotic") and 0.313 at 5 steps ("clear") [MEASURED-other] — [HeyNEO benchmark](https://heyneo.com/blog/kokoro-tts-vs-supertonic-3-tts) (via search summary, [2nd] for the wording); vendor: RTF ~0.012-0.023 on an Apple M4 Pro CPU at 2 steps [CLAIM]; sherpa-onnx supports Supertonic (diacritics fix in v1.13.5, `na` tag in v2.0.0) — [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases)
- **OmniVoice** (k2-fsa): 0.6B, diffusion-LM-style, "over 600 languages" (646 in metadata; Slovak and Czech not named explicitly), zero-shot voice cloning and voice design; code Apache-2.0, weights CC-BY-NC; "RTF as low as 0.025" with no hardware named [CLAIM] — [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)
- **Not covering us:** Voxtral TTS 3 language library shows Polish, Hungarian, Ukrainian and others but not Slovak or Czech [CLAIM] — [Mistral: Voxtral 3](https://mistral.ai/news/voxtral-3/); Qwen3-TTS is supported in TensorRT Edge-LLM — [Edge-LLM supported models](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/getting_started/supported-models.html) — its language list was not checked here (the search budget ran out)

### Inferences
- Most of what the robot says is templated. Pre-render every fixed phrase (acknowledgements, refusals, "Neviem, zopakuj", the read-back frames, numbers 0-20, object and colour words in the right case) offline with the best voice we can get, including a large model on hekate or a cloned lab voice for research use, and store WAVs on the robot: zero synthesis latency, best quality, and the same voice in the paper's video. Synthesize on the robot only what is genuinely new (taught names).
- On the Orin's CPU, Piper medium should synthesize a short sentence in well under 0.3 s and start playing sooner; Supertonic 3 at 5 steps would be ~0.3x real time on 4 cores (≈0.5 s for a 1.5 s sentence) unless run on the GPU. Neither needs the GPU.
- Slovak and Czech number and case agreement ("štyri kocky", "päť kociek") is a text problem: generate the sentence with the grammar's own tables, not with the LLM.

### Gaps
- No listening test or MOS of `sk_SK-lili`, `cs_CZ-jirka` or Supertonic 3's Slovak was found (looked: Piper's voice cards and issues via search, Supertonic's README and HF card, two TTS benchmark blogs).
- No Orin NX measurement of Piper or Supertonic was found; the Orin Nano number above is end-to-end in ROS 2 with an English low-quality voice.
- Meta's MMS-TTS (VITS per language, includes Slovak and Czech, non-commercial) was not re-checked in this pass.

## 2. Playing it without hearing itself

### Takeaway
Speech out interacts with speech in: the robot's own voice reaches its microphone. Unless the voice module's echo cancellation is verified, the wake-word detector and the recogniser should be gated while the robot speaks, and the stop lane should stay live but with a re-armed threshold right after playback ends. The stop lane must never be muted for longer than a word.

### Evidence
- Yahboom describes echo cancellation on the M3's module [CLAIM] (see `wake_stop_vad_microphones.md` section 4); a community document says the older CSK4002 board has "no AEC" [2nd] — [yahboom-mcp VOICE_AUDIO.md](https://github.com/sandraschi/yahboom-mcp/blob/main/docs/hardware/VOICE_AUDIO.md)
- The Orin Nano Piper write-up used small PulseAudio buffers (`pacat_latency_msec` 60, `pacat_process_time_msec` 30) and a ~150-300 ms drop window after interrupts [MEASURED-ORIN, blog] — [thomasthelliez.com](https://thomasthelliez.com/blog/running-piper-tts-on-nvidia-jetson-orin-nano-with-low-latency/)

### Inferences
- Test once: play the robot's own phrases (all of them) for an hour through its speaker with the wake word and stop lane listening, and count false triggers. This is part of the false-wakes-per-hour measurement in the evaluation plan.
- Keep the "spoken summary from the console" path (M3.1) for the laptop demo; move it to the M3's speaker when the stop-lane false triggers from its own voice are at zero.
