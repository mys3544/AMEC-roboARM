# Wake word, stop word, end of speech and the microphone on a moving robot (state of 27 October 2026)

Scope: everything in front of the recogniser on the M3 Pro: a Slovak/Czech/English **stop lane** that never waits for a model (PLAN.md M3: "stop / stoj / stuj" straight to the halt service, under 0.5 s from the end of the word to zero velocity, motors running), a **wake word** for hands-free commands, voice activity detection to end an utterance, and the **microphone**: Yahboom's voice module and what motor noise does. Today the console's stop is text matching on the recogniser's output (`voice.parse`: any word containing "stop", or "halt", "freez", "abort", "zastav", "zastan", "stoj", "stuj"), so it waits for push-to-talk and Parakeet ([console/voice.py](../../console/voice.py)). Desk research only.

Labels: **[MEASURED-ORIN]**, **[MEASURED-other]**, **[CLAIM]**, **[2nd]** as in `asr_slovak_czech_on_orin.md`.

## 1. Off-the-shelf wake-word engines do not speak Slovak or Czech; we train our own

### Takeaway
Picovoice Porcupine, the polished commercial engine, offers nine training languages and none of them is Slovak, Czech or Polish. openWakeWord and microWakeWord are open, train from synthetic speech and run on Linux, but both are documented for English only, and openWakeWord's pretrained models are non-commercial (we would train our own anyway). sherpa-onnx's open-vocabulary keyword spotter has Chinese and English models only. So every route to a Slovak wake or stop word goes through training a small model ourselves: openWakeWord (one shared frozen embedding plus a tiny classifier per word, so a stop model and a wake model cost almost nothing to run together) is the easiest; microWakeWord is the leaner alternative.

### Evidence
- **Porcupine:** model-training languages en, fr, de, it, ja, ko, zh, pt, es — "Czech, Slovak, and Polish are not included" [CLAIM] — [Porcupine Model API](https://picovoice.ai/docs/model-api/porcupine/); v2.2 release notes once named Polish and Russian among added languages, which the current docs do not list [2nd] — [Porcupine releases](https://github.com/Picovoice/porcupine/releases)
- **openWakeWord:** fixed ONNX mel front end -> frozen shared Google speech-embedding backbone -> small fully connected or 2-layer RNN classifier; "15-20 openWakeWord models simultaneously in real-time" on one Raspberry Pi 3 core; code Apache-2.0, pretrained models CC BY-NC-SA 4.0; positives from open-source TTS, "several thousand" examples minimum; ~30,000 h of negative speech, noise and music for the included models; targets <5 % false rejects and <0.5 false accepts per hour, to be measured on audio from the deployment environment; optional Silero VAD gate (`vad_threshold`) and Speex noise suppression (x86 and Arm64 Linux); "only English is supported"; last release v0.6.0 (Feb 2024) [CLAIM] — [openWakeWord](https://github.com/dscripka/openWakeWord)
- A 2026 training/validation toolkit for openWakeWord gates a model at ≤0.5 false accepts/hour at ≥80 % recall on ≥1 h of real room audio, with ~20 real recordings of the phrase and near-miss negatives; one robot's model reached 0.00 FA/h over 75 min at 91 % recall (English); it warns macOS scored near zero where Linux scored 0.89+ FA/h [MEASURED-other, anecdotal] — [openwakeword-colab-toolkit](https://github.com/ubaldexela-cloud/openwakeword-colab-toolkit)
- **microWakeWord:** MixConv streaming model, features every 10 ms, inference every 30 ms, positives from the Piper sample generator, negatives as pre-computed spectrogram features, TensorFlow Lite for Microcontrollers [CLAIM] — [kahrendt/microWakeWord](https://github.com/kahrendt/microWakeWord); a Linux Python runtime exists (Apache-2.0, `okay_nabu` example) — [pymicro-wakeword](https://github.com/OHF-Voice/pymicro-wakeword); community notebooks train Spanish words with Piper and multilingual TTS ensembles [2nd] — [Home Assistant community](https://community.home-assistant.io/t/colab-notebooks-to-train-your-custom-openwakeword-or-microwakeword/1026635), [microWakeWord-Trainer-AppleSilicon](https://github.com/TaterTotterson/microWakeWord-Trainer-AppleSilicon)
- **sherpa-onnx keyword spotting:** "a tiny ASR system" that only decodes given keywords; per-keyword boost (`:1.5`) and trigger threshold (`#0.35`); pretrained models zh-en 3M, Chinese 3.3M and English (GigaSpeech) 3.3M only — [sherpa-onnx KWS](https://k2-fsa.github.io/sherpa/onnx/kws/index.html)
- **Yahboom's own engine:** wake word "Hello yahboom", Chinese and English only (PLAN.md M3, from Yahboom's voice chapters)

### Inferences
- Train one openWakeWord classifier per phrase: `stop` (English, also said in Slovak and Czech), `stoj`/`stojte`, `stůj`/`stujte`, `zastav`; and a wake phrase per language family if wanted ("Ahoj robot", "Hey robot"). All share the embedding pass, so the stop lane and the wake word together cost one stream's CPU.
- The English "STOP" could also be a sherpa-onnx English KWS keyword as a second, independent detector; Slovak/Czech spellings through English BPE tokens would be a hack and are not recommended.

### Gaps
- No published Slovak or Czech wake-word model, or false-accept figure for any non-English openWakeWord/microWakeWord model (looked: the two repositories, Home Assistant community threads, sherpa-onnx docs, Picovoice docs).
- The language coverage of openWakeWord's frozen Google embedding is not stated in its README.
- End-to-end detection latency (end of the word to the score crossing the threshold) is not published for either engine; it must be measured.

## 2. Synthetic Slovak speech is enough to start, not to finish

### Takeaway
Keyword spotters trained only on text-to-speech miss many real utterances; what closes most of the gap is real *negative* audio from the deployment environment and a modest amount of real positives. For Slovak we have few TTS voices (one Piper voice, Supertonic 3's presets, OmniVoice cloning for research use), so diversity must come from augmentation, Czech/Polish voices reading Slovak spellings, voice cloning of lab members, and 20-50 real recordings per phrase from as many people as we can get.

### Evidence
- Synth4Kws (Interspeech SynData4GenAI 2024, English single words): more TTS phrase diversity and more utterances consistently improved EER and AUC; with 50k real utterances as the base, adding TTS improved EER by 30.1 % and AUC by 46.7 % [MEASURED-other] — [arXiv 2407.16840](https://arxiv.org/abs/2407.16840)
- Park et al. (same workshop, 2024): TTS-only training gave a 46.47 % false-reject rate; adding real negatives (ambient noise, non-target speech) cut it to 17.94 %; 2,000 real utterances from 100 speakers plus TTS came within ~3x of a 3.8M-utterance real-data baseline [2nd, search summary of the PDF] — [Park et al. 2024](https://www.isca-archive.org/syndata4genai_2024/park24_syndata4genai.pdf)
- TTS-generated confusable words reduced false alarms sharply in an older wake-word study [2nd] — [arXiv 2011.01460](https://arxiv.org/html/2011.01460); LLM-generated confusables + TTS improved c-AUC by 11.3 % on Speech Commands [2nd] — [arXiv 2505.22995](https://arxiv.org/pdf/2505.22995)
- Slovak/Czech TTS voices available: Piper `sk_SK-lili` (one speaker), `cs_CZ-jirka`, `cs_CZ-kasandra`; Supertonic 3 (sk and cs among 31 languages, preset voices); OmniVoice (zero-shot cloning, 600+ languages, CC-BY-NC) — see `tts_slovak_czech.md`

### Inferences
- Recipe: ~5-10k synthetic positives per phrase (all voices, speed/pitch perturbation, room impulse responses, robot-noise mixing at 0-20 dB SNR), near-miss negatives in Slovak and Czech ("stôl", "stojí", "postoj", "stopa", "zastávka", "stuha", "Stano"), plus real negatives: hours of the robot's own motors, lab talk, and Slovak/Czech radio or TV recorded through the M3's microphone. Then 20-50 real positives from lab members for validation, more for fine-tuning if recall is short.
- Asymmetric costs: a false stop costs a few seconds, a missed stop can cost a collision. Set the stop threshold for recall (e.g. ≥98 % on real recordings) and accept more false stops per hour than for the wake word; report both.

### Gaps
- No study of TTS-trained keyword spotting for Slovak or Czech, or any West Slavic language, was found.

## 3. End of speech: Silero VAD, and where the 0.5 s goes

### Takeaway
Silero VAD (MIT, ~2 MB, trained on 6,000+ languages, <1 ms per 30+ ms chunk on one CPU thread) is the standard choice and is already inside sherpa-onnx and openWakeWord; the cost of VAD is not compute but the silence it waits for. With hands-free use, the end-of-speech timeout (Yahboom 1.5 s, our plan 0.5 s) is the largest single latency item after the recogniser, so it is worth ending early when the grammar already holds a complete command.

### Evidence
- Silero VAD: "One audio chunk (30+ ms) takes less than 1ms to be processed on a single CPU thread", ONNX "may even run up to 4-5x faster", JIT model ~2 MB, 8 and 16 kHz, trained on over 6,000 languages, MIT (one badge says CC BY-NC 4.0, an inconsistency on the page) [CLAIM] — [silero-vad](https://github.com/snakers4/silero-vad); v6 chunk 512 samples (32 ms) at 16 kHz, ~0.5 ms per chunk on Apple Silicon in a third-party port [MEASURED-other] — [Silero-VAD-v6.2.1-MLX](https://huggingface.co/aufklarer/Silero-VAD-v6.2.1-MLX)
- sherpa-onnx ships VAD + ASR examples and Ten-VAD; v2.0.0 fixed a 64-bit sample-position crash on long streams and made the microphone/ALSA examples feed one window at a time — [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases)
- Yahboom waits 1.5 s of silence; our plan 0.5 s; budget estimate wake 0.2 + end of speech 0.5 + recognition 0.2-0.4 + parser 0.3-1 + first audio 0.3 s — [PLAN.md](../../PLAN.md) M3

### Inferences
- Two-step endpointing: after 250-300 ms of silence run the recogniser and the grammar; if the grammar returns a complete, valid command, act; otherwise wait to the full 0.5-0.7 s. This trims ~0.2-0.3 s on the simple commands without cutting off "zober... červenú kocku".
- Keep push-to-talk (console and a physical button) as the primary mode for the demo and the study's scripted part: it has zero endpoint delay and zero false wakes.

## 4. The microphone and the motors

### Takeaway
What Yahboom fitted on the M3 is only described in marketing terms (a MEMS microphone, a cavity speaker, far-field pickup, echo cancellation, noise reduction); Yahboom's other voice boards are command-word chips (CSK4002, CI1302) rated at 5 m in quiet but 1 m in noise. Published recognisers degrade gracefully down to ~5 dB SNR and fall apart near 0 to -5 dB; a robot's own motors next to its microphone can reach that. Neural speech enhancement in front of a recogniser can lower accuracy, so it should not be added without a measurement; array beamforming with a learned noise model cut WER by up to 55 % on a mobile robot. The cheapest large win for a demo is a close-talk (lavalier) microphone; the honest study condition is the robot's own microphone with the motors running.

### Evidence
- M3 Pro voice module: "a high-sensitivity MEMS microphone and a cavity speaker, which can clearly pick up voice and has functions such as far-field pickup, echo cancellation, voice broadcast, and environmental noise reduction" [CLAIM, via search summary of Yahboom's blog; the page itself returned HTTP 429 on 27 Oct] — [Yahboom M3 Pro unboxing](https://www.yahboom.store/blogs/news/unboxing-and-reviewing-rosmaster-m3-pro-ros2-robot-with-multimodal-ai-large-model); it appears as the system's audio input and output, sensitivity set in GNOME (PLAN.md M3, from Yahboom's course)
- Other Yahboom voice boards: CSK4002 module with two microphones, 360° pickup to 5 m, 95 % wake rate, 85 preset commands [CLAIM] — [RobotShop listing](https://www.robotshop.com/products/yahboom-yahboom-intelligent-voice-speech-recognition-module-w-5v-power-supply); a community driver doc says the CSK4002 board talks USB-CDC serial (CH340/CP2102) with "no AEC" [2nd] — [yahboom-mcp VOICE_AUDIO.md](https://github.com/sandraschi/yahboom-mcp/blob/main/docs/hardware/VOICE_AUDIO.md); CI1302 module: neural noise reduction and echo cancellation, ~120 custom Chinese/English command words [CLAIM, 2nd] — [Yahboom product](https://category.yahboom.net/products/voice_module_asr-tts), [ORIWHIZ listing](https://oriwhiz.com/products/yahboom-intelligent-voice-interaction-module-supports-custom-command-words-ci1302-chip); a reseller listing of the YB-MAE01 board: "maximum recognition distance in quiet is 5 m; ... in a noisy environment 1 m" [CLAIM, 2nd] — [eBay listing](https://www.ebay.com/itm/357171676239); the X3's course documents the CI1302 board (x3plus/PLAN.md sources, chapter 16151)
- Noise robustness of Parakeet v3 (MUSAN, average over benchmarks): 6.34 % clean, 8.23 % at 5 dB, 11.66 % at 0 dB, 19.88 % at -5 dB [MEASURED-other] — [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- Enhancement can hurt: in a 2024 URGENT Challenge table DeepFilterNet2 output had lower word accuracy (74.54 %) than the noisy input (79.53 %) [2nd] — [arXiv 2505.23515](https://arxiv.org/pdf/2505.23515); GTCRN is 23.7K parameters, 39.6 MMAC/s [2nd]
- Ego-noise: PCA-selected noise covariance + MVDR beamformer on a mobile robot, under two minutes of noise calibration, up to 55 % lower WER and up to 10 dB SDR [MEASURED-other] — [arXiv 2303.00829](https://arxiv.org/abs/2303.00829) (the search summary adds: 16-microphone array on a Clearpath Jackal, Whisper small, small-room WER 62.6 % -> 30.9 % [2nd]); a 16-mic array on a 7-DoF arm moved towards the speaker improved SI-SDR and WER [MEASURED-other] — [arXiv 2602.17818](https://arxiv.org/abs/2602.17818)
- Add-on array: ReSpeaker XVF3800, circular 4-mic, XMOS XVF3800 with echo cancellation, multi-beam forming, de-reverberation, direction of arrival, noise suppression, 60 dB AGC, 5 m pickup, USB or I2S firmware [CLAIM] — [Seeed wiki](https://wiki.seeedstudio.com/respeaker_xvf3800_introduction/)
- KInIT trained their Slovak models with 75 % noise-augmented samples (phone noise, background speech, noise) [MEASURED-other] — [kinit/parakeet-tdt-0.6b-v3-sk](https://huggingface.co/kinit/parakeet-tdt-0.6b-v3-sk)

### Inferences
- First measurement on the robot (an hour): `arecord -l`, `lsusb`, the device's channel count and rates (does it give raw channels or one processed channel?), and SNR of a spoken sentence at 1 m and 2 m with the base driving and the arm moving (record speech-only and noise-only, compute the ratio). If SNR at 1 m is above ~10 dB, the module is fine; if near 0 dB, plan a close-talk or array microphone.
- Mute or gate the wake word while the robot speaks unless the module's echo cancellation is verified; otherwise the robot's own "Hotovo" can wake it.
- The arm's servos sit centimetres from an arm-mounted microphone; where the module is mounted on the M3 matters and should be photographed and noted in the paper.

### Gaps
- Microphone count, SNR, sample rates, whether echo cancellation runs on the module and whether it can be bypassed: not found (Yahboom's product and course pages returned HTTP 429 or JS shells to the fetcher; the GitHub repo's voice folder has no hardware notes). The robot survey answers it in minutes.
- No published measurement of mecanum-base or bus-servo noise levels at a robot's microphone.
