# Coverage, shiny and dark surfaces, colour, and where each step runs (state on 2026-10-27)

Scope: the object scan's faults that are not the cameras' poses -- what the arm can see from the circle
and how to see more (next-best-view, a second ring), the see-through "ghost" over the box, the depth
camera's holes on shiny and dark surfaces, the colour swings between views, and what each fix costs on
the Orin NX, the laptop (RTX 2070 Super) or hekate (RTX 5090) within ~5 minutes a scan. The poses are in
`poses_and_registration.md`, the surface methods in `surfaces.md`. Labels: [REAL] our capture, [BENCH]
a published benchmark, [CLAIM] an author's claim not checked further, [2nd] from a secondary source or a
search summary. The web-search budget ran out partway through this pass; later checks used primary pages
(arXiv abstracts, GitHub READMEs) only.

## 1. Coverage: the circle sees the thing through a narrow band; a closer second ring is the cheap fix

### Takeaway
The gap is geometric, not a planning problem: from STOP_R 0.38 m the arm sees a 15 cm thing through a
38-56 deg elevation band, and from 0.20 m through 67-89 deg, so "higher looks" mean a closer stop, not a
higher arm. A fixed second ring -- 4-6 stops at ~0.28-0.30 m with the arm looking down, the views chosen
from the first ring's TSDF (surface voxels seen by fewer than ~3 views or only from below) -- costs
~40-60 s and is all the next-best-view a 2-minute scan needs. The NBV literature (FisherRF, ActiveGS,
ObjSplat) solves open-ended exploration; on a known circle round one object its gain over a fixed
two-ring plan is small and unmeasured for our case.

### Evidence
- [REAL] The DCW2 sits on arm4 before J5, looks within the arm's upright plane, sees nothing nearer
  than ~0.19 m and reaches ~0.25 m from the shoulder; from one spot a thing is seen through a ~20 deg
  cone: 67-89 deg up at 20 cm, 38-56 deg at 40 cm (a 15 cm tall thing). -- `m3pro/roboarm/m3scan.py`
  module docstring; the robot stands 0.38 m from the thing, its back ~0.55 m out (`m3pro/tools/m3_scan.py`).
- [REAL] Scans so far: 12 stops x 2-3 views, 24-36 views, ~90-116 s; "the scan only looks from low
  down" (NEXT.md, "Splat ghost": a second, higher look at each stop, +~1 min).
- [CLAIM] FisherRF selects views by expected information gain from the Fisher information of the
  radiance field's parameters; "state-of-the-art results on multiple tasks, including view selection".
  -- [arXiv 2311.17874](https://arxiv.org/abs/2311.17874)
- [CLAIM] ObjSplat (Jan 2026, rev. Jun 2026): object-centric Gaussian surfels, a next-best-path planner
  that looks several steps ahead on a spatial graph, back-face and occlusion-aware view scoring; less
  scan time and path length than prior methods (no numbers on the abstract page).
  -- [arXiv 2601.06997](https://arxiv.org/abs/2601.06997)
- [2nd] ActiveGS (RA-L 2025, Jin et al.) adds confidence to a Gaussian splat map and plans views to
  low-confidence regions (scene level); "next-best-sense" extends FisherRF to depth uncertainty and touch
  on a real robot; Go Fetch (2020) uses four predefined wrist-camera views round a target; Omni-Scan
  (2025) argues that wrist cameras "significantly limit coverage" because of arm kinematics and moves
  the object instead. -- search summaries of
  [arXiv 2412.17769](https://arxiv.org/pdf/2412.17769v2), [Go Fetch](https://arxiv.org/pdf/2004.00899),
  [Omni-Scan](https://arxiv.org/pdf/2508.00354v1)

### Inferences
- Plan the second ring from the first ring's own TSDF: for each candidate stop on a 0.28-0.30 m circle and
  each arm view the IK allows (`stop_views` already samples the cone), count the surface voxels that
  view would see that have < 3 observations; take the best 4-6. That is a greedy NBV in ~50 lines on the
  laptop, and it needs good poses first (a misregistered first ring makes the TSDF's "unseen" voxels wrong).
- A closer ring brings the bumper within ~18 cm of a box's middle (box radius 9 cm): keep the existing
  clearance check (`clear_of`) and the chassis guard; a turntable stays the fallback for tall things.

## 2. The ghost over the box is mostly pose error; SH extrapolation and free space are the rest

### Takeaway
In the box splat's turntable (24 frames at 45 deg) the see-through frames (13-15) also show the yellow
floor line twice, a few cm apart -- two registrations of the same floor -- and they sit in the sector where
the box's per-stop re-read jumped 59 mm between stops 4 and 5. The splat's pose refinement moves cameras
only 3-6 mm, so a cm-level misplacement survives as a second, semi-transparent layer that the low
training views can hide but a 45 deg view cannot. Fix the poses first (`poses_and_registration.md`); then
two cheap checks: render the turntable with SH degree 0 (the console's `.splat` already keeps DC colour
only) to see how much is view-dependent colour extrapolated to unseen angles, and drop Gaussians that sit
in space the depth camera measured as empty.

### Evidence
- [REAL] `m3pro/out/scan/20261012_172202/splat_turntable.jpg` (rendered with SH degree 3:
  `train/splat/train_splat.py`, `render(params, view, k, 3)` in the turntable loop): frames 13-15 show a
  doubled yellow L and the box partly transparent; the other 21 frames are clean.
- [REAL] The box scan's per-stop re-read of the thing's centre moved 10, 23, 20, 9, **59**, 10, 8, 15, 14,
  21, 16 mm from stop to stop (`scan.json` "centres"); the can's 7-33 mm. The splat's camera corrections:
  box 0.3-7.5 mm, can 2.5-17.3 mm (`splat_poses.json` "moved_mm").
- [REAL] Our floor check (`poses_and_registration.md`, section 2) measures the views' disagreement on the
  floor directly.

### Inferences
- Free-space culling is the TSDF's own logic (space in front of a measured depth is empty) applied to the
  splat: for each trained view, a Gaussian whose centre projects inside the image at a depth more than
  ~1 cm in front of the measured depth is in seen-empty space. A few lines in `train_splat.py` after
  training (or as a periodic prune). No paper was checked for this exact rule; it is an inference.
- SH degree 3 from 24-36 views on one circle extrapolates badly above the circle; for console splats
  (DC only) train with SH 0-1 and compare held-out PSNR.

## 3. Shiny and dark surfaces: more depth frames and more angles first, a learned depth refiner second

### Takeaway
The DCW2 is a structured-light stereo camera: where the projected IR does not come back (a can's
polished lid, glossy black print) it leaves holes or wrong depth. Cheapest first: median 3-5 depth frames
per view (planned in PLAN.md M4 A step 2, not done), extra views of the lid from the second ring, and
masking saturated highlights in the colour. Learned refiners that take RGB + raw depth exist (Camera
Depth Models, Prompt Depth Anything, Marigold-DC) but none is trained for an Orbbec camera; FoundationStereo
works on IR stereo pairs, but whether the DCW2's driver gives both IR images was not established. For a
demo object, a matte scanning spray is the practical answer.

### Evidence
- [REAL] Our numbers: the DCW2 reads a 30 mm cube's height to ~0.5 mm (task brief); the best monocular
  depth on our close looks is 5.2-6.9 mm median (DA V2 Large, NEXT.md "Depth duel") -- a learned
  monocular depth is ~10x worse than the DCW2 where the DCW2 reads. The can's lid was "left open" in the
  fuse and the splat (memory, NEXT.md).
- [REAL] The DCW2 per Orbbec's sheet: binocular structured light, depth 640x400 at 15 fps, 0.2-3 m
  (PLAN.md M4); the bridge starts the driver with `enable_ir:=false` (`m3pro/tools/m3bridge.py`), so an
  IR stream exists; whether it is one or both cameras' was not found.
- [BENCH] Camera Depth Models ("Manipulation as in Simulation", ICLR 2026): RGB + raw depth -> denoised
  metric depth, trained on simulated noise of a specific camera; zero-shot on Hammer AbsRel 0.0708 ->
  0.0312 (D435 split), 0.0475 -> 0.0229 (L515); 0.151 s per frame on an RTX 4090; released for D435,
  L515, D405, Azure Kinect, ZED 2i -- not Orbbec; fixes "many" glass/metal errors qualitatively but fails
  on a metal plane. -- [arXiv 2509.02530](https://arxiv.org/html/2509.02530v1); model cards on
  Hugging Face list CC-BY-NC-4.0 [2nd]
- [CLAIM] FoundationStereo (CVPR 2025 oral, NVIDIA): zero-shot stereo, handles textureless / reflective /
  translucent objects (qualitative); README: "tested it on monochrome or IR stereo images (e.g. from
  RealSense D4XX series) and it works well too"; Jetson support 2025-07-03; a real-time model
  (Fast-FoundationStereo) 2025-12-15; TensorRT FP16 ~6x on an RTX 3090.
  -- [GitHub NVlabs/FoundationStereo](https://github.com/NVlabs/FoundationStereo)
- [CLAIM] Prompt Depth Anything (CVPR 2025): a low-res metric depth (LiDAR) as the prompt of a depth
  foundation model, up to 4K output; SOTA on ARKitScenes and ScanNet++. -- [arXiv 2412.14015](https://arxiv.org/abs/2412.14015)
- [CLAIM] Marigold-DC (ICCV 2025): zero-shot depth completion, a monocular-depth diffusion model guided
  by sparse measurements at test time. -- [arXiv 2412.13389](https://arxiv.org/abs/2412.13389)
- [CLAIM] ASGrasp (2024): raw IR + RGB from an active stereo camera into a learned stereo network for
  transparent objects. -- [arXiv 2405.05648](https://arxiv.org/html/2405.05648v2)

### Inferences
- For our scan the lid is small and seen from many views; the splat with the depth loss already fills it
  photometrically where the depth is missing (the loss uses only valid depth). The open lid in the TSDF is
  then a TSDF-only problem: carve the TSDF mesh's lid from the splat's rendered depth, or take the
  surface from the splat (`surfaces.md`).
- Do not retrain or adopt a depth network for this before the poses are fixed: it would not move the box's
  size, which is a pose and measurement problem.

## 4. Colour: hold the exposure during the scan, then model what is left

### Takeaway
Measured on our two scans, the DCW2's auto exposure swings x0.89-1.20 between views and the white balance
B/G 0.89-1.18; the corners are 7-9 % darker. The exposure can already be held through the Orbbec
driver's services (`arena_map.py camlock`); the white balance cannot be read on the DCW2. Holding the
exposure helps twice: the colours, and every photometric registration (coloured ICP, ECC, the splat's
pose refinement) that assumes a point looks the same from every view. What is left -- white balance and
vignetting -- is a per-view gain (already fitted by `scan_photometry.py`) applied before the TSDF's
colours; PPISP or a bilateral grid only for the splat's colours; Open3D's colour map optimisation for a
sharp TSDF mesh texture.

### Evidence
- [REAL] Exposure x0.89-1.20 (box) / x0.93-1.14 (can), B/G 0.89-1.18 / 0.90-1.08, corners 9 / 7 % darker;
  per-view gains take the same floor cell's spread only 8.2 -> 6.9 % / 10.1 -> 9.2 %; the rest sits on the
  print and tape edges (8.9 % against 4.5 % on plain floor) -- misregistration.
  -- `m3pro/tools/scan_photometry.py` docstring; NEXT.md "The scan's colours".
- [REAL] `m3pro/tools/arena_map.py camlock`: reads `get_color_exposure`, turns
  `set_color_auto_exposure` off and sets `set_color_exposure` to the settled value; white balance:
  "Property is not supported" on the DCW2, left on auto; restarting the camera restores auto.
- [2nd] OrbbecSDK v1.10.35 added "color auto exposure priority" for Gemini UW, DaBai DCW and DaBai DCW2
  (search summary of the [OrbbecSDK releases](https://github.com/orbbec/OrbbecSDK/releases)).
- [BENCH] Coloured ICP's colour term compares a point's colour with its match's tangent-plane colour
  function (Park et al. 2017): it presumes consistent colour across scans.
  -- [Open3D coloured ICP](https://www.open3d.org/docs/latest/tutorial/pipelines/colored_pointcloud_registration.html)
- [BENCH] Open3D colour map optimisation (Zhou & Koltun, SIGGRAPH 2014): mesh + RGB-D images + camera
  trajectory; rigid (6-DoF per camera) or non-rigid (plus local image warps); in the tutorial the rigid
  residual falls 21,628 -> 9,387 in 100 iterations on 33 frames.
  -- [Open3D colour map optimisation](https://www.open3d.org/docs/latest/tutorial/pipelines/color_map_optimization.html)
- [CLAIM] PPISP (CVPR 2026, NVIDIA): per-frame exposure and colour, per-camera vignetting and tone
  curve behind the splat's render; held-out PSNR on 3DGS 22.6 -> 25.4 (BilaRF data), 23.1 -> 24.4 (T&T);
  geometry not measured. -- [arXiv 2601.18336](https://arxiv.org/abs/2601.18336); NEXT.md. gsplat's main
  branch lists PPISP (Jan 2026) as an alternative to bilateral grids, unreleased on PyPI
  (`research_notes/Model export formats for our hardware/gaussian_splats.md`, section 1).

### Inferences
- Order: (1) camlock at the scan's start (the scan job already has the bridge; the services are the
  driver's), (2) rerun `scan_photometry.py` -- the exposure spread should fall to ~1; (3) per-view gains in
  the fuse; (4) only then PPISP, if the splat's colours still matter for the demo.
- Auto white balance cannot be held on the DCW2, so per-view colour gains stay in the pipeline.

## 5. What runs where within ~5 minutes a scan

### Takeaway
Everything that fixes the geometry is light: floor or tag registration and the pose solve are OpenCV and
NumPy (seconds on the laptop, could run on the Orin), the TSDF is Open3D on the CPU (tens of seconds on the
laptop for 24-36 views at 1.5 mm). Only the appearance and surface refinements (splat, 2DGS/PGSR, a
learned depth refiner) want the 5090, and they fit in 1-5 minutes there. Feed-forward 3D models also fit
(seconds) but do not give mm poses (`poses_and_registration.md`, section 4).

### Evidence
- [REAL] Our floor registration over 24 views (ECC on 1 mm orthophotos, neighbouring stops only) took a
  few minutes in plain Python on the laptop CPU; SIFT-only pairs take seconds (`poses_and_registration.md`).
- [REAL] The splat: gsplat 1.5.3, 7000 steps, 53-61 s on the 5090 for the box (24 views)
  (`m3pro/out/scan/20261012_172202/splat_train.log`); the whole `splat_cycle.py` ~2 min.
- [BENCH] VGGT backbone: 0.31 s for 20 frames, 1.04 s for 50 frames, 5.6-11.4 GB, on one H100 at 336x518
  -- [arXiv 2503.11651](https://arxiv.org/html/2503.11651).
- [BENCH] DTU training times on the surface methods' own GPUs: 2DGS 19.2 min, PGSR 40 min, GOF 1 h, QGS
  48 min, GSPrior 42 min, Neuralangelo >> 12 h (GSPrior's Table 2) -- for 49-64 images at DTU's resolution;
  our 24-36 views at 640x480 are several times smaller (inference).
  -- [arXiv 2603.19682](https://arxiv.org/html/2603.19682v2)

### Gaps
- No measured Open3D TSDF time on the Orin NX at 1.5 mm for 36 views (NEXT.md: "The fusion on the robot
  (Open3D's aarch64 wheel)" is still open).
- Whether the DCW2 driver can stream both IR images (needed for FoundationStereo) -- check with
  `ros2 launch orbbec_camera dabai_dcw2.launch.py --show-args` on the robot.
