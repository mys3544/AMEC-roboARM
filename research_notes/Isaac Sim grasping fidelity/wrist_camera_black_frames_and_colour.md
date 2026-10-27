# The simulated DCW2 wrist camera: black frames and colours lighter than the real ones (state of October 2026)

Scope: the RGB render of the M3's wrist camera in Isaac Sim 5.0 on Dionysos. Dionysos runs Windows 11 Education
with an RTX A4500 on driver 581.42 and is shared with other users. The render setup is
`isaac_m3.py`'s `_camera` / `_picture`, called with `SimulationApp({"headless": True})`, the RTX real-time renderer
and a Replicator `rgb` annotator at 640x480.

Two problems:
- the first renders after new MDL materials are black, so the script now renders 60 warm-up frames and retries until
  a frame is not black;
- about one run in three is black all through while an ACT training uses the same GPU, and `remote()` reruns it.

Separately, the sim's colours are lighter than the real camera's. Depth rendering, the camera's pose and hand-eye are
out of scope. The 2026-10-26 notes cover them (`sim2real_gap_measurement_and_correction.md`). Labels as in the
other notes.

## 1. What our launch actually runs: DLSS on, multi-GPU on, sync loads on

### Takeaway
`SimulationApp({"headless": True})` in 5.0 gives the following defaults:
- `RaytracedLighting`;
- `anti_aliasing = 3` (DLSS);
- `multi_gpu = True`;
- `sync_loads = True` (material and USD loads already synchronous);
- the 5.0 base experience adds `renderer.asyncInit = true`, `rtx.hydra.mdlMaterialWarmup = true` and DLSS
  `execMode = 3` (Auto).

DLSS renders a 640x480 camera at a lower internal resolution. Switch to `anti_aliasing = 4` (DLAA in NVIDIA's own
scenes) for native resolution, and to `multi_gpu = False` on one-GPU machines.

### Evidence
- From `DEFAULT_LAUNCHER_CONFIG` in Isaac Sim 5.0's `simulation_app.py` [CLAIM: source read] —
  [simulation_app.py v5.0.0](https://github.com/isaac-sim/IsaacSim/blob/v5.0.0/source/extensions/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py):
  - `"multi_gpu": True`, `"sync_loads": True`, `"renderer": "RaytracedLighting"`, `"anti_aliasing": 3`,
    `"extra_args": []`;
  - "anti_aliasing (int): Antialiasing mode, 0: Disabled, 1: TAA, 2: FXAA, 3: DLSS, 4:RTXAA";
  - sync loads become `--/rtx/materialDb/syncLoads`, `--/rtx/hydra/materialSyncLoads` and
    `--/omni.kit.plugin/syncUsdLoads`;
  - the experience file is looked for under `EXP_PATH` (`omni.isaac.sim.python.kit`, then
    `isaacsim.exp.base.python.kit`, then `isaacsim.exp.base.kit`).
- From the 5.0 base experience [CLAIM] —
  [isaacsim.exp.base.kit v5.0.0](https://github.com/isaac-sim/IsaacSim/blob/v5.0.0/source/apps/isaacsim.exp.base.kit),
  [isaacsim.exp.base.python.kit](https://github.com/isaac-sim/IsaacSim/blob/v5.0.0/source/apps/isaacsim.exp.base.python.kit):
  - `renderer.asyncInit = true # Don't block while renderer inits`;
  - `rtx.hydra.mdlMaterialWarmup = true`, `rtx.post.dlss.execMode = 3`;
  - `omni.replicator.asyncRendering = false`, `app.asyncRendering = false`;
  - `vulkan = true` ("off by default on Windows" without it).
- LeRobot's SO-101 Isaac scene sets `/rtx/post/aa/op = 4` because "the default DLSS renders 640x480 cameras at
  320x240" [CLAIM] — [so101-tabletop-isaac env.py](https://huggingface.co/lerobot/so101-tabletop-isaac/blob/main/env.py).
  NVIDIA recommends DLSS Quality (`/rtx/post/dlss/execMode = 2`) for synthetic data: below about 600x600 the
  Performance mode "may cause ... transparent or incorrectly rendered edges" [CLAIM] —
  [6.0 known issues](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/known_issues.html)
- Dionysos has a machine-wide `EXP_PATH` pointing into another user's folder (NEXT.md, 2026-10-26). 5.0's
  `python.bat` runs fine, but the Kit log should confirm which `.kit` was loaded.

## 2. Black frames: documented causes and fixes, and what fits our "one run in three"

### Takeaway
No source ties black renders to another CUDA process on the same GPU. The documented causes fall into three groups.
**Our pattern** (whole runs black, only beside a training, on Windows 11 with an R580 driver) fits the Windows-11
driver issue and render-timing settings best. In order of cost:
1. Launch with Isaac Lab's synchronous-render settings (`renderer.waitIdle`, `hydraEngine.waitIdle`,
   `checkForHydraRenderComplete`), the throttling flag, DLAA and `multi_gpu = False`.
2. Keep the black detector and rerun.
3. Do not render beside a training (already the rule since 2026-10-27).
4. A driver of 595 or newer is NVIDIA's fix for Windows 11 black images. That is a shared-machine change, so ask the
   user first. Alternatively, render on hekate.

The documented causes:
- **materials not loaded yet:** rt_subframes >= 2, warm-up renders;
- **async rendering or frame skipping:** throttling and async flags, an extra capture step on Windows;
- **driver, multi-GPU or renderer bugs:** a driver >= 595 on Windows 11, `--vulkan`, multi-GPU off.

### Evidence
- Isaac Sim 6.0 known issues [CLAIM] —
  [6.0 known issues](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/known_issues.html):
  - "On Windows 11, the viewport may flicker and SDG pipelines may write black images. To resolve this, update the
    NVIDIA display driver to version 595 or higher."
  - "When running SDG pipelines with Replicator in standalone mode on Windows, the first frame may be skipped by
    writers or the data might be missing in annotators. ... add an extra capture call (rep.orchestrator.step())".
  - Both 5.1 and 6.0 [CLAIM] —
    [5.1 known issues](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/known_issues.html):
    - "If you see a black screen when running on Windows, use the --vulkan command-line argument";
    - "If randomized materials are not loaded on time for synthetic data generation the rt_subframes must be set to
      be at least 2";
    - "On some multi GPU systems when creating a render product the main viewport will go black".
- Replicator troubleshooting [CLAIM] —
  [Replicator troubleshooting (5.1)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/troubleshooting.html):
  - frames "may be skipped due to the isaacsim.core.throttling extension toggling /app/asyncRendering=True by default
    when the timeline is stopped"; fix `--/exts/isaacsim.core.throttling/enable_async=false`;
  - "If the captured images are written as black, try starting Isaac Sim once with the --reset-user".
- Isaac Lab's headless camera experience for Isaac Sim 5.1 adds [CLAIM] —
  [isaaclab.python.headless.rendering.kit v2.3.0](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/apps/isaaclab.python.headless.rendering.kit):
  - `app.updateOrder.checkForHydraRenderComplete = 1000`, `app.renderer.waitIdle=true`,
    `app.hydraEngine.waitIdle=true` ("Avoids frame offset issue");
  - `omni.replicator.asyncRendering = false` ("Force synchronous rendering");
  - `renderer.multiGpu.maxGpuCount=1` ("Avoids unnecessary GPU context initialization");
  - `app.vulkan = true` ("avoids torch+cu12 error on windows").
- Field reports:
  - Isaac Sim 5.1, Windows 11: `rep.orchestrator` "sometimes returns all-zero RGBA frames", in headless and windowed
    runs, unresolved by May 2026. Staff suggestions: do not call `world.reset()` or `world.stop()` during capture;
    call `wait_until_complete()` once at the end; retry with `omni.kit.app.get_app().update()` while the frame is
    zeros [2nd: summarised] —
    [NVIDIA forum 360288](https://forums.developer.nvidia.com/t/rep-orchestrator-returning-all-rgba-channels-0/360288).
  - Isaac Sim 5.1: black tiled cameras in RTX real-time mode, confirmed by NVIDIA and "fixed in 6.0"; path tracing
    avoided it [2nd] — [IsaacSim #367](https://github.com/isaac-sim/IsaacSim/issues/367).
  - Isaac Sim 6.0.1, Linux: RTX 3090s (Ampere) "render solid black with no errors" while 5090s render. The maintainer
    blamed a non-validated R570 driver and advised 595.58 [2nd] —
    [discussion #875](https://github.com/isaac-sim/IsaacSim/discussions/875).
  - Isaac Sim 6.0.1: the RT 2.0 viewport is sometimes black after load and some MDL materials never render, a
    regression from 5.1. Maintainer: `--/renderer/multiGpu/enabled=false` and driver 595 [2nd] —
    [discussion #758](https://github.com/isaac-sim/IsaacSim/discussions/758).
- Ours (NEXT.md, `isaac_m3.py`):
  - black first renders after new MDL materials; 60 warm-ups; `_picture` re-renders up to 20 x 30 times;
  - one run in three black all through beside a training, "the logs alike", whether the camera was made before or
    after the reset;
  - the 2026-10-27 lesson: "no Isaac run on Dionysos while it trains" (a loader crash).
- Not found: an NVIDIA statement or issue linking black frames to a concurrent CUDA training process. Searched: the
  Isaac Sim 5.1/6.0 known issues, Replicator troubleshooting, IsaacSim GitHub issues and discussions, and NVIDIA
  forum results before the search quota ran out.

### Inference
Pass these through `SimulationApp`'s `extra_args`:
- `--/app/renderer/waitIdle=true`, `--/app/hydraEngine/waitIdle=true`,
  `--/app/updateOrder/checkForHydraRenderComplete=1000`;
- `--/exts/isaacsim.core.throttling/enable_async=false`;
- and set `anti_aliasing: 4`, `multi_gpu: False`.

`--/renderer/asyncInit=false` is a further candidate, because a renderer still initialising while the GPU is busy
could leave a render product unbound. That is a hypothesis, not documented. Measure the black-run rate over 10 runs
beside a training before and after; the baseline is about 1 in 3.

The "frame offset" settings matter beyond black frames too. If the annotator returns the previous frame, the
recorded picture lags the recorded state by one tick at 10 Hz. Check one episode by stepping the arm and seeing in
which tick the picture moves.

## 3. How much lighter: measured on our own overlay

### Takeaway
On the 2026-10-26 wide look (`out/sim/look_20261026_131823.jpg`, real | sim | blend):
- **brightness:** the sim's paper is 213 against the real's 145-184;
- **colour:** the green cube's top is (134,176,135) against (28,58,26), saturation 0.24 against 0.55;
- **contrast:** the black tape is 45 against about 2; the frame's luma contrast (std) is 18 against 47.

Two different faults:
1. the paint colours were taken from photos (sRGB) and fed to OmniPBR as linear albedo, so they are 2.5-5x too
   light;
2. the real camera's exposure and tone curve (auto-exposure on a white sheet, a black level, a steeper curve) are
   not modelled.

### Evidence
- Patch means (ours: 640x480 halves of the overlay, 12-30 px patches) [SIM vs REAL-LC]:

  | patch | real DCW2 | sim |
  |---|---|---|
  | paper, middle | (145,143,148) | (213,213,213) |
  | paper, far | (184,182,189) | (216,216,216) |
  | green cube top | (28,58,26) | (134,176,135) |
  | black tape | (1,5,2) | (45,45,45) |
  | whole frame mean / luma std | (136,135,141) / 46.6 | (211,211,211) / 18.2 |

- Our colour constants are photo values:
  - `CUBE_PAINT` "paint as measured on them (the user's photos)", e.g. green (0.23, 0.46, 0.23); `TABLE_GREY` 0.74;
    in `train/synth/kit_synth.py` and `m3_sim.py`;
  - `isaac_m3.py` passes them straight to OmniPBR's `diffuse_color_constant` or to `displayColor`.
- OmniPBR's texture input marks 8-bit RGB textures as sRGB by default ("auto"). The colour constant is a plain
  `color` [CLAIM] — [OmniPBR](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/OmniPBR.html).
  LeRobot's Isaac scene converts every colour: "An sRGB display color ... in the linear space Isaac Sim shades in",
  `c**2.2` [CLAIM] — [so101-tabletop-isaac env.py](https://huggingface.co/lerobot/so101-tabletop-isaac/blob/main/env.py).
- Arithmetic (ours): 0.46^2.2 = 0.18 and 0.23^2.2 = 0.040. With the sim's own paper as the reference, the green
  top would render at about (55-65, 105-115, 55-65) instead of (134,176,135), and its saturation would roughly
  double. A remaining factor of about 1.6 in brightness is the exposure.
- The real camera's own variation, measured on our object scans (RESEARCH.md, PPISP note): exposure swings
  x0.89-1.20 and the corners are 7-9 % darker [REAL-LC].

## 4. Matching and randomising the colour

### Takeaway
1. Fix the colour space (section 3).
2. Match exposure and tone with a curve fitted on 3-5 paired looks, applied to the rendered frame in the recorder.
   This is the published method and it does not depend on the renderer's settings.
3. Randomise around the fit, within what the real camera actually does: exposure x0.85-1.25, small white-balance
   and contrast changes, 0-10 % vignetting.
4. Add colour jitter in training as well; the ACT command has no image augmentation today.

Leave the RTX tonemapper at its default on 5.0, where exposure is global. On 6.x a per-camera exposure attribute
exists.

### Evidence
- Fitting a degree-2 polynomial RGB transform between renders and real captures, with real white balance and
  exposure fixed, raised sim-real policy correlation. Without it, r dropped from 0.944 to 0.805, 0.901 to 0.714, and
  0.915 to 0.529 on three tasks [SIM vs REAL-other, from the prior note] —
  [arXiv 2511.04665](https://arxiv.org/html/2511.04665v2); `sim2real_gap_measurement_and_correction.md`.
- Squint (SO-101, wrist camera only): real 73/80 with colour jitter, 58/80 without [REAL-LC, from the prior note] —
  [arXiv 2602.21203](https://arxiv.org/html/2602.21203)
- RTX post-processing has tone-mapping operators (Clamp, Linear, Reinhard, Modified Reinhard, HejlHableAlu,
  HableUc2, ACES, Iray), camera exposure / film ISO / f-stop / white point, a histogram-based auto exposure (white
  point scale, adaptation speed, EV clamping), colour correction and colour grading [CLAIM] —
  [RTX post-processing](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx_post-processing.html)
- Per-camera exposure [CLAIM]:
  - Kit 108.1: "Camera exposure can now be controlled per-camera, as opposed as a global setting" —
    [RTX 108.1 notes](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer-release-notes/108_1.html);
  - via `OmniRtxCameraExposureAPI_1` (`exposure` in stops, `exposure:fStop`, `exposure:iso`,
    `exposure:responsivity`, `exposure:time`), which "supersede those in the Render Settings ... Tone Mapping" —
    [RTX cameras](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html);
  - Isaac Sim 5.0/5.1 are on Kit 107.3, so this is 6.x only —
    [5.1 release notes](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/release_notes.html).
- Isaac Sim 6.0 models a camera ISP pipeline: black level, white-balance gains, a 3x3 colour matrix, Bayer encoding,
  shot and Gaussian noise, an ISP program. But "The sample ISP program ... is only available on Linux x86_64"
  [CLAIM] — [6.0 camera sensors](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/sensors/isaacsim_sensors_camera.html)
- NVIDIA's SO-101 course randomises sky exposure (−4, 3) EV and colour temperature 2500-9500 K [CLAIM, from the
  prior note] — `open_sim2real_pipelines.md`.

### Inference: the recipe
1. Convert every colour constant: `c ** 2.2`, or the exact sRGB EOTF, in `m3_sim.py` / `isaac_m3.py`. That covers
   CUBE_PAINT, BLOCK_PAINT, SIDES40, TABLE_GREY, the tape and the box. Textures need nothing.
2. Render 3-5 looks the robot also took (the 2026-10-26 overlays). Sample matched patches: paper, tape, desk grey,
   each paint, a picture face, the box. Fit per-channel `real = a + b·sim + c·sim²` on linearised values, and apply it
   in `_save_jpeg` before encoding.
3. Per episode, draw exposure x U(0.85, 1.25), white-balance gains x U(0.95, 1.05) per channel, contrast ±10 % and
   vignetting 0-10 % at the corners. Store them in the robot card like the other randomisations.
4. Done when the patches agree to within about 20 levels per channel and the luma std of a wide look is within
   20 %.

## 5. Shader warm-up, and the cost of the fixes

### Takeaway
Keep a warm-up, but tie it to "picture not black and stable". The shader cache makes later runs fast. DLAA and the
wait-idle settings slow rendering somewhat; measure seconds per picture, about 0.3 s now.

### Evidence
- "the first time an application switches to motion raytracing, shaders will need to be recompiled causing a
  potentially multi-minute pause ... The compiled shaders are then cached" [CLAIM] —
  [RTX 109.0 notes](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer-release-notes/109_0.html)
- LeRobot's scene renders up to 100 warm-up captures until each camera returns data, else it raises [CLAIM] —
  [env.py](https://huggingface.co/lerobot/so101-tabletop-isaac/blob/main/env.py)
- 5.0's base experience already sets `rtx.hydra.mdlMaterialWarmup = true` [CLAIM] —
  [isaacsim.exp.base.kit v5.0.0](https://github.com/isaac-sim/IsaacSim/blob/v5.0.0/source/apps/isaacsim.exp.base.kit)

## Gaps
- The exact carb paths of the tone-mapping settings in Kit 107.3 were not verified, except `/rtx/post/aa/op` and
  `/rtx/post/dlss/execMode`. Read them from the Render Settings panel, or dump `carb.settings` under `/rtx/post` on
  Dionysos.
- That the colour constants are linear is inferred from MDL's `color` type and LeRobot's conversion, not stated on
  an NVIDIA page. A quick check: a texture of sRGB 128 next to a constant of 0.5 renders differently if so.
- No measurement of whether the remaining one-in-three black runs disappear with the settings above. Only a
  10-run test will tell.
