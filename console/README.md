# The console -- both robots on one page

One page on this laptop for the X3 Plus and the M3 Pro: each robot alone or both side by side,
a STOP on each and STOP ALL always on screen (Space or Esc too). Plan v2.3, M1, first version.

```bash
.venv/Scripts/python console/serve.py --fake    # pretend robots, no hardware
.venv/Scripts/python console/serve.py           # the real ones (robots.json)
```

then open http://localhost:8099/ (`.claude/launch.json` has both as `console-fake` and `console`).

**Where the robots are** (`locate.py`, since 2026-10-20): as it starts, serve.py finds each robot by
its name (`m3pro.local`, `x3plus.local`: mDNS), else at its `robots.json` address, and with `--scan`
by looking round this laptop's networks for the bridges' port 8761 (only on our router or hotspot);
the page gets the addresses found. `python console/locate.py [--scan]` says where each robot is.
The event page for visitors is http://localhost:8099/demo.html (below).

- **The robots** are listed in `robots.json`: the X3's panel, on the robot since 2026-10-06
  (`docker compose --profile web up -d web` in `~/roboarm`, port 8080; the laptop copy,
  launch config `webapp-robot`, still works against the bridge instead), its arm card on the
  same port, and the M3's drive page (`m3pro/tools/arena_map.py start`, port 8095).
  `robots.fake.json` points at `x3plus/tools/webapp.py --sim` and `m3pro/arena/fake_m3.py`, which
  `--fake` starts: the M3's real drive logic on the saved arena map, with a ray-cast lidar.
- **X3 card:** the wrist camera, arm / job / camera state, Pick one, Clear the table, Home;
  everything else on its own page (the "own page" link).
- **M3 card:** the live map with the fence fit, the robot, its scan and the loop; hold the pad
  or W/A/S/D (sideways on A/D), Q/E to turn; the auto loop. The robot's own limits apply: at
  most 0.15 m/s, stopped 0.4 s after the page stops repeating a command; the loop also stops
  35 cm short of anything in its path, driving by hand has no distance limit.
- **M3 arm card** (`m3arm` in `robots.json`: the M3's bridge, the service `roboarm-m3bridge`
  on the robot, which runs the pick itself, `m3pro/roboarm/m3pick.py`): the arm camera live with the
  last look inset (what was found, marked; click to swap), the job and its step, picked and missed,
  the last attempt, what is in view and whether it is in reach; Pick nearest / a cube / a block,
  Clean up all in reach (to the drop-off, one by one), Home; where each step's time went (the
  robot's `/api/metrics`) and the robot's own narration of the job.
- **X3 arm card** (`x3arm`): the same card on the X3's panel, which serves the M3's `/pick/*`
  (`x3plus/roboarm/web/session.py`): its jobs are the panel's one-click pick and clear the
  table; no driving yet (the box is hidden), no drop-off box yet (drops at its fixed pose).
- **STOP ALL** lets go of any held key, stops every robot in `robots.json` -- shown or not --
  and logs how long each took to confirm (pretend robots: M3 20 ms, X3 about 0.8 s).
- **3D scan:** the M3's coloured 3D scans of the arena, as `m3pro/tools/arena_map.py
  rtab-export` leaves them in `m3pro/out/arena/rtab/` (on this laptop, so no robot needed); pick
  one from the list. Drag turns, right-drag or Shift-drag moves, the wheel (or a pinch) zooms,
  a double-click or "reset view" goes back to the view from above. Camera colour or height colour (0 to 45 cm above the floor, blue to red), and the point
  size (a 7 MB export, 463k points, is on screen in 0.1-0.2 s). "Floor print" shows the
  export's floor print instead (`NAME_floor.png`: the floor from above, 1 cm a pixel), turned a
  quarter when it stands upright so that it lies as the 3D view starts, the same way round.
- **Pop out** opens one robot, or the 3D scan, alone in its own window (`/?solo=m3`,
  `/?solo=cloud`), e.g. for a second monitor; the layout is remembered in the browser.

How it is built: `serve.py` (standard library) serves the page and passes JSON through as
`/robot/<id>/<path>` -- the X3's panel sends no CORS headers; camera streams and maps come
straight from the robots. The scans come from `serve.py` too: `/clouds.json` lists the PLY
files in the export folder, each with its floor print, and `/clouds/<name>` is one of those
(that folder, `.ply` and `.png` only).
`cloud.js` draws them with WebGL2 and no library (the one PLY `m3pro/arena/ortho.py` writes:
binary little-endian, float x y z and uchar red green blue; any other is refused), loaded only
when the 3D view opens. No framework, no build step. It listens on localhost only (IPv4 and IPv6: on Windows `localhost`
is tried as `::1` first, and without it every connection waited 2 s). Not yet: dockview panels
(drag, split, float), the arena panel with both robots on the mat, SSE instead of polling.

`python -m pytest console` runs `test_serve.py` and `test_voice.py`.

## The event page (demo.html)

http://localhost:8099/demo.html (2026-10-20, for the event out of the lab on 2026-10-22): each arm
in `robots.json` side by side -- its camera live with the last look inset, what it is doing in one
sentence ("Beriem 30 mm kocku", from the job's step and its log), news under it ("Zobral som ... za
13 s", "Stôl je uprataný"), the job's counts; big buttons: Clean up, Look, Home, and on the M3
Teach / Find / Pick it by name. **H** hides every button (what visitors see), **SK / EN** in the
bar, Talk (hold V) and STOP as here (Space, Esc). The same /robot/<id>/ JSON; nothing on the
robots changed. Tried on `--fake` and on both robots: a clean-up each, 8 picked, 0 missed.

## Voice

Hold **Talk** in the bar, or hold **V**, say a command, let go. Speech recognition runs on this
laptop (no cloud): NVIDIA's Parakeet TDT 0.6B v3, int8 ONNX, through sherpa-onnx on the CPU. It
knows 25 European languages, English, Slovak and Czech among them. Once:

```bash
uv sync                                         # from the repo root: sherpa-onnx is locked
.venv/Scripts/python console/voice.py fetch     # the model: 487 MB from sherpa-onnx's GitHub
                                                # releases, unpacked to 670 MB in console/models/
```

(sherpa-onnx 1.13.8 and sherpa-onnx-core are the console's own dependencies,
console/pyproject.toml, in the workspace's lock.) `console/models/` is gitignored (`models/` anywhere). Without sherpa-onnx or the
model, the console runs as before, the Voice box says which is missing and the line that fixes
it, and typed commands still work.

- **The commands without the language model** (`voice.py`'s `parse`, a small grammar; with the
  model, below, it still finds every stop). The M3 unless the X3 is named ("X3", "X three", "Plus").
  - *stop*, *halt*, *freeze*; Slovak *stoj*, Czech *stůj*, both *zastav* -> STOP ALL. It wins
    whenever it is said, whatever else is ("please stop the robot", "don't stop"), and stops
    every robot, whichever was named.
  - *pick up [the] [colour] [size] cube | block*: "pick up the red cube", "X3, pick up the 40 mm
    blue cube", "grab the small yellow cube" (small 30 mm, big 40 mm; "thirty millimetre",
    "3 cm"), "pick up the wooden block"; *Vezmi červenou kostku*, *Zober zelenú kocku*.
  - *clean up*, *clear the table / arena*, *pick them all*; *upratať*, *ukliď*.
  - *go home*, *home*; *domov*, *domů*.
  - anything else, two commands at once, or both robots named: not a command, nothing done.
- **What is done** (`console.js` hands `window.consoleActions` to `voice.js`): stop is the bar's
  STOP ALL. The X3's pick / clean up / home are its card's Pick one / Clear the table / Home (it
  picks what it finds first: a colour or size is shown, not passed on). The M3's go to its arm's
  pick service, robots.json id and kind `m3arm` (the M3's bridge, port 8761), and the X3's go to
  `x3arm` the same way when it is listed (else to its panel card's buttons): `pick/start` with
  `{"kind", "colour", "size_mm"}`, `pick/cleanup`, `pick/home`, and STOP ALL sends `pick/stop`;
  the page polls `pick/state`. Until that entry is in `robots.json`, an M3 command says "M3 arm
  not connected".
- **The page** shows the state, what was heard, the command (as JSON), what was done, and the
  time from letting go to the command; the Log keeps each one. The microphone opens on the
  first press (the browser asks once) and stays open while the page is; 0.3 s from before each
  press is kept, so a word begun a moment early is not cut; a press is sent after 15 s at most.
  Not in a popped-out window.
- **The wire:** `POST /api/voice` with a WAV (16-bit PCM mono, any rate; the page sends the
  microphone's own, and sherpa-onnx resamples to 16 kHz) answers `{"text", "intent", "audio_s",
  "asr_ms"}`; with `Content-Type: text/plain` the text is parsed only. `GET /api/voice` says ready / loading / why not, and starts the load: the
  page asks as it opens, so the model is in memory (serve.py then holds 0.8 GB) by the first word.
- **Measured** on this laptop (i7-10875H, 2026-10-01): the model loads in 2.8 s (3.3 s with the
  pretend robots running). Recognition with 4 threads: "Stop." (1.4 s of audio) 154 ms, "Go
  home." (1.6 s) 177 ms, "Pick up the red cube." (2.5 s) 207 ms, 3.5 s 300 ms; 1 thread 327 /
  377 / 513 ms, 2 threads 186 / 215 / 295 ms, 8 threads no better than 4. In the page, letting go
  to the command: 277 ms for a 2.5 s recording, 380-450 ms for 2.7-4.3 s, with both pretend
  robots on the same laptop; the page's own part (then also a resample to 16 kHz; WAV, POST,
  parse) is 15-20 ms of it. Seven commands spoken by Windows' own voices, all heard and parsed
  right; no real voice yet.
- `python console/voice.py parse pick up the red cube` prints the command in a sentence.

### The language model (since 2026-10-20)

What was said goes to a small LLM on this laptop's GPU (`llm.py`): llama.cpp's `llama-server`
(release b11617, Windows, CUDA 12.4) with Qwen3.5 4B (Q4_K_M), on 127.0.0.1:8081. It picks one
function and fills its arguments as JSON held to a schema (llama.cpp's grammar: only those
functions and values): *pick* (kind, colour, size), *pick_named* (a thing the M3 was taught, by
the name it means -- "podaj mi rybu" -> fish; the names come from the M3's `/pick/taught`),
*clean_up* (only cubes or blocks if said), *sort* (the robots cannot yet), *look*, *find*,
*home*, *stop*, *unknown*; robot m3, x3 or both. A stop never goes to it: the grammar finds
it first, at once. Not answered in 6 s, or garbled: the grammar's command. serve.py starts
the server with the console (`--llm off`, or another model), and asks it once at the start --
the first answer after a start takes ~15 s (CUDA readies itself), every next ~0.6 s. The reply
says `"by": "llm"` with `llm_ms`, or `"by": "grammar"`.

Files, outside the repo, in `~/.roboarm/llm` (ROBOARM_LLM moves it): `bin/` both zips of the
llama.cpp release unpacked (`llama-b11617-bin-win-cuda-12.4-x64.zip`, `cudart-llama-bin-win-cuda-
12.4-x64.zip`, GitHub ggml-org/llama.cpp), `models/` the GGUFs (unsloth's on Hugging Face).
`python console/llm.py bench` runs each model there on `llm_cases.json` (45 sentences, Slovak,
Czech, English, some without accents as the recogniser may write them); 2026-10-20 on the RTX
2070 Super, the prompt tuned once on the 2B's mistakes (general rules, not the sentences):

| model (Q4_K_M) | right of 45 | median | worst | file |
|---|---|---|---|---|
| **Qwen3.5 4B** (default) | **45** | 646 ms | 767 ms | 2.7 GB |
| Qwen3 4B Instruct 2507 | 44 | 535 ms | 593 ms | 2.5 GB |
| Gemma 4 E4B | 43 | 1067 ms | 3020 ms | 5.0 GB |
| Qwen3.5 2B | 42 | 364 ms | 431 ms | 1.3 GB |
| Gemma 4 E2B | 37 | 541 ms | 678 ms | 3.1 GB |

2026-10-21, spoken on the robots: English came through clean, Slovak did not -- Parakeet wrote it as
Russian in Cyrillic ("Икс три убрать ствол") or Polish-looking ("Plus zober kotku nie kwadrę"), and
the model took misheard cubes for taught things ("velkou kotku" -> the yellow cylinder, on the M3
though "Plus" was said). 14 such sentences went into `llm_cases.json` (59 now) and the prompt got
rules for them (Cyrillic and Polish spellings read as Slovak, a cube or block word never a taught
thing, a thing not taught is unknown, unsure is unknown): Qwen3.5 4B 53/59 -> **59/59**, median
704 ms, the 7 taught names given. The recogniser itself has no Slovak switch. A second round the same
day: "valec" heard as "Valence", "balet", "Vallet", "uprac" as "o prata", "u pracovstvo" -- told
in the prompt, the 4B model began picking cubes for them (64/71), so llm.SOUNDS_LIKE puts those
words back before it reads the sentence: **71/71** (12 more sentences), median 732 ms.
