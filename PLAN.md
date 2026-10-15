# Plan v2.4 -- two robots, one arena: search, pick, voice, 3D, and research worth showing

Written 2026-09-29; v2.1 the same day with the user's answers about the arena, the voice module, the
servers, the object kit and JetPack, and with the three things they asked to have explained: why
Nav2, whether 3D scanning would localize better, and how tracks D and A work. v2.2, also the same
day: the floor is Yahboom's Autopilot Track Map (not the sand-table map), the landmarks are ~30 cm
tall, an Isaac Sim twin of the arena already exists, the research order B -> A -> D is agreed, and
**nothing is built until the user says "let's start"**. Supersedes [m3pro/PLAN.md](m3pro/PLAN.md)
(v1), whose M3 bring-up and depth detector are the M3's part of M2.

**v2.3, 2026-09-30 -- started.** The user said "let's start", and: the first version of
localization does NOT use the Isaac Sim twin (it comes back later); the arena is mapped on the
real robots instead, in the levels of M2.1a (slam_toolbox 2D -> RTAB-Map coloured 3D -> Gaussian
splats and a neural SLAM comparison on the 5090). First thing to show: the arena's slam_toolbox
map from the M3, checked against the fence's known 3.2 x 2.8 m.

**v2.4, 2026-10-07 -- re-ordered by the showcase the user picked.** After the research sweep of
2026-10-05 ([RESEARCH.md](RESEARCH.md): six papers ranked, thirty showcase builds on a shared
menu), the user marked what they want to show:
- **Want:** voice in English, Slovak and Czech; the M3 tidying into its box; teach a new object in
  30 seconds; real against predicted depth (the depth duel); an object scan with the arm; a "stop"
  that never waits for a model; one Slovak sentence for both robots.
- **Maybe:** track D's learned policy (the night shift, chasing a moved cube, instructions like "the
  one nearest the box"); A4's retrain-and-swap; splats; the scout and the cube riding along in the
  twin (B, E); a live AI-agent demo.
- **Nothing from track B's search** (the fog of war, the kidnap test, pulling the WiFi).

Agreed the same day:
- **The research order is A -> D** (was B -> A -> D). B shrinks to what the Slovak mission needs:
  localization, Nav2, a simple split of the work. Its belief-search planner and study L become
  optional, later.
- **One orbit, two demos:** A2 moves into stage 1 beside the object scan (M4 A). The same arm
  orbit, with depth masks, gives both the 3D scan and the 30-second teach.
- **The depth duel (S1) gets a stage.** It falls out of the X3 plan's depth table
  ([x3plus/PLAN.md](x3plus/PLAN.md), its stage 2), with the M3's stereo depth shown beside it.
- **Voice comes before the arena:** the stop lane and the voice pipeline on the robot first.
- **The Slovak mission starts box-sharing (M3.1).** Both robots clear their own areas into the
  M3's box, taking turns, and neither drives across the arena. The arena-wide and free-form
  versions follow.
- **The live AI-agent demo** (P5 in RESEARCH.md) is added as optional.

The X3's own plan, [x3plus/PLAN.md](x3plus/PLAN.md) (2026-10-06: the M3's abilities from the wrist
camera and a depth network), sits under this one.

Inputs:
- the user's priorities (next section) and their answers: no fixed demo date; big models on the
  lab's servers, and on the robot only a small local model that turns a spoken command into
  settings ("pick only small cubes", "pick up the big cube"); English required, Slovak/Czech a
  welcome addition; the arena, voice module, servers and kit as described in M2, M3 and the
  architecture; stay on JetPack 6.2;
- a research sweep, six parallel surveys: detection models and export, 3D mapping and scanning,
  voice and language, arm interaction, arena navigation and multi-robot search, the operator
  console; and Yahboom's course and shop (the track map, the kit, the voice stack). Figures quoted from
  them are other people's measurements unless marked "est."; each is re-measured here before
  anything relies on it. Sources are at the end.

What has run since: *Where things stand*, below.

## What we are building

Must-haves, in the user's order:
1. **One application for both robots:** a switchable robot menu, or flexible windows to use both
   at once.
2. **Search and pick in the arena:** each robot finds objects we define inside the fenced arena,
   localizing on the fence with its lidar, and picks them.
3. **Voice commands**, at least on the M3 Pro. English required; Slovak/Czech a bonus.
4. **A 3D scan with coloured point clouds** from the M3's new depth camera.
5. **The newest YOLO and similar models**, benchmarked here and exported for the fastest inference.

Across all of it: current research, innovative and good to watch -- not Yahboom's demos redone.

## Where things stand (2026-10-07)

| part | built | not yet |
|---|---|---|
| M1 console | one page for both robots; STOP per robot and STOP ALL; pop-outs; pretend robots (`--fake`); the 3D scan view with the floor print; voice push-to-talk; arm cards for the M3 and, since 10-06, the X3 | dockview panels, the arena panel, SSE; STOP ALL timed on the real robots |
| M2 arena | slam_toolbox map within ~1 cm of the fence (3.200 x 2.785 m, `arena_v4`; on the way: lidar range biases of ~1 cm and a 55-75 ms clock skew between the base board and the Jetson); RTAB-Map coloured 3D scan; the floor orthophoto; a camera-down survey planner | AMCL, Nav2, the catalogue as data, search; walls still 4-6 cm thick |
| M2.6 M3 pick | hand-eye; IK with sag; objects from the depth floor; a second close look with the camera turned onto a target the first look cut; the careful grip by the bumper; the mesh-shell guard, 1.0 cm off the lidars; a slide or a drive-up when out of reach; the drop into the box its marker finds. 11-13 s a pick | the 40 mm cube right in front of the right lidar (needs the slide) |
| X3 ([x3plus/PLAN.md](x3plus/PLAN.md)) | the pick runs on the robot behind the M3's `/pick/*` (first run 20.0 s, landed 2.3 mm); lens and lens-on-arm models with a sag fit; Depth Anything anchored to the floor; heights from two views by parallax | the base under ROS (X6), the box (X5), the arena (X7) |
| M3 voice | console push-to-talk: Parakeet v3 on the laptop and a grammar for EN/SK/CS (pick by colour or size, clean up, home; stop words -> STOP ALL); 277-450 ms from letting go to the command; synthetic voices only | anything on the robot: the module, a wake word, the stop lane, the LLM parser, the validator, spoken answers |
| M4 3D | the arena scan (B) | the object scan (A), splats (C) |
| M5 models | YOLOE-26 served on the X3 (PyTorch); Depth Anything V2 S as a TensorRT engine; DA3 and MoGe-2/3 measured on the X3's views | the kit detector, INT8, the benchmark table |
| research | the sweep and the ranked papers (RESEARCH.md) | no track started |

Waiting on: access to the 5090 and dionysus.

## In one page

- **Three tiers.** The robots do everything that must be fast or safe: control, localization,
  perception engines, the stop word, a small command parser. The lab servers (the RTX 5090 in the
  lab, dionysus with the RTX 4500 in the other building) do everything big: labelling, training,
  Gaussian splats, larger language and vision models. They are shared with other people, so no
  must-have depends on them at run time. The laptop runs the console.
- **Console (M1).** One page for both robots: dockable panels that switch to one robot, split side
  by side, or pop out onto a second monitor; the arena drawn on the mat's own image; STOP ALL
  always on screen.
- **Arena (M2).** Yahboom's Autopilot Track Map inside a 3.2 x 2.8 m, 30 cm white PVC fence: the
  fence and the three ~30 cm landmarks are the lidar map, and the landmarks break the rectangle's
  180-degree symmetry for the lidar itself -- no markers needed. AMCL localizes each robot on it. A
  3D scan would not localize better than the lidar here (plain white walls); the camera adds the
  printed track as a second, independent fix. The planned neural localizer and the camera cue are
  compared in study L. Nav2 drives (why: M2.3); the arm camera finishes each approach; search is a
  belief map both robots share and drain. The map comes from slam_toolbox on the real M3 (v2.3);
  the existing Isaac Sim twin comes back later for ground truth, synthetic scans and B's trials.
- **Voice (M3).** Yahboom's module is fitted. A stop lane that never waits for a model, and a
  command lane: wake word -> Parakeet v3 speech recognition (English, Slovak, Czech) -> a small
  local model that only fills in the command's settings -> a validator -> the skill -> a spoken
  answer. Places are the track's printed features ("bring the red cube to the parking bay"). One
  Slovak sentence can give both robots a job at once, sharing the M3's box (M3.1, v2.4).
- **3D scan (M4).** The M3's arm orbits an object, stopping at each view; coloured mesh and point
  cloud in the console in about two minutes (est.). The arena is scanned while driving -- which
  also gives an orthophoto of the floor print for localization. Upgrade: Gaussian splats trained
  on the 5090, viewed in the browser.
- **Models (M5).** The kit is three shapes, so the main detector is a small YOLO26 fine-tuned on our
  own frames -- labelled for free by the M3's depth -- and exported as an INT8 engine. YOLOE-26 as
  a TensorRT engine stays as the open-set baseline (the same step took YOLO26s from 20.4 to 6.4 ms).
- **Research (order v2.4).** A first: teach an object by showing it. A1-A2 are part of the first
  show; A3-A4 make the paper. Then D: the classical picker trains a vision-language-action model,
  the user's "maybe". E and C later. B is cut to what the Slovak mission needs; its belief-search
  planner and study L are optional.
- **Order (v2.4).** Foundations -> a first show (the object scan and the teach from one orbit, the
  depth duel, the model table) -> voice (the stop lane, the robot's own pipeline, the box-sharing
  Slovak mission) -> the arena (localization, Nav2, search and pick; the mission arena-wide) ->
  research tracks.

## Architecture

```
laptop: console at http://localhost:8090 (the microphone works: localhost is a secure context)
   |  per robot, direct: HTTP commands, SSE state, MJPEG (later WebRTC), point clouds
   +------------------------------------+---------------------------------------+
   X3 Plus (Orin NX, JetPack 6.2)                     M3 Pro (Orin NX, JetPack 6.2)
   bridge: arm + base (+ /odom, /cmd_vel)             m3bridge (rclpy): arm over /arm6_joints
   C930e wrist camera, mast RGB-D, RPLidar            DaBai DCW2 on the arm, 2 x T-mini Plus
   ROS 2 Humble on its own domain                     ROS 2 Humble, domain 30, voice module
   +------------------------------------+---------------------------------------+
                                        |  jobs over HTTP, never on a safety path
      RTX 5090 server (in the lab, IISLab network)  |  dionysus, RTX 4500 (other building, VPN)
      labelling | detector and policy training | policy serving during experiments |
      Gaussian splats | larger language/vision models | distilling the command parser
```

Rules:
- A server or the WiFi going away degrades a robot, never endangers it: the robot falls back the
  way the detector ladder already does (`DetectorOffline`).
- **Shared servers.** The 5090 sits on the robots' network: batch jobs, plus online services during
  experiment sessions (D's policy server, a large vision-language model). dionysus is reached by
  static IP or VPN: batch jobs only (training, labelling, splats). Because other people use both:
  everything we run is a container with an explicit GPU and memory limit; a job checks the GPU is
  free (`nvidia-smi`) and checkpoints so it can pause; GPU time is agreed with the other users;
  our datasets and models live in our own directory.
- ROS 2 stays inside each robot, each on its own domain (the M3's vendor stack uses 30). Robots
  and console talk HTTP and Server-Sent Events. A ROS view only for debugging (foxglove_bridge +
  Lichtblick in a pop-out). If the robots ever share ROS topics: one zenoh-bridge-ros2dds per robot
  with an allow-list (poses, detections, beliefs -- never TF, scans or images).
- Stay on JetPack 6.2 and ROS 2 Humble (agreed). JetPack 7.2 now supports Orin, but it moves to
  Ubuntu 24.04 and CUDA 13, where neither vendor's Humble stack runs; Humble is supported until
  May 2027. New models run in containers or on the servers.
- Code stays separated: the X3's in `x3plus/` (moved from the root 2026-10-06; its own plan is [x3plus/PLAN.md](x3plus/PLAN.md)), the M3's in `m3pro/`, the shared console in a new
  top-level `console/` (agreed), with small additions to both robots' web servers.

## The digital twin (Isaac Sim) -- later, not in the first version (v2.3)

Deferred on 2026-09-30: the first version maps and localizes on the real robots (M2.1a). Until
the twin is back, its jobs are done without it: the map by slam_toolbox, checked against the known
fence; synthetic scans for the neural localizer by ray-casting that map in Python; B's many trials
in a small 2D Python simulator; Nav2 tuned on the robots; the console against a fake robot
backend; study L's ground truth static only (the track's printed points).

The lab already has the arena in Isaac Sim: the scanned map, the identical fence and the landmark
objects. That changes how every arena-side piece is developed -- in the twin first, on the robots
second:
- **The arena map for AMCL** comes from the twin's geometry (fence + landmarks, exact), not from a
  tape measure; one slam_toolbox run on the real robot checks it.
- **Localization (study L):** exact ground truth in the twin; unlimited synthetic lidar scans for the
  neural localizer, rendered with the same fence and landmarks; the real runs then measure the
  sim-to-real gap.
- **Search (track B):** the belief planner, the auctions and the detection model tried over hundreds
  of seeded layouts before 20 real runs.
- **Navigation:** Nav2 and AMCL settings (MPPI, the keep-out circles, the collision monitor) tuned in
  the twin, including two robots meeting.
- **Perception before the M3 is up:** rendered RGB-D of the kit's pieces for the depth detector, the
  3D scan pipeline and the detector's far range bins.
- **The console:** a simulated robot answers the same HTTP API as a real one, so the console can
  drive the twin -- develop and demonstrate without the hardware, and switch between twin and
  robot per panel.
To check at the start: the Isaac Sim version and where it runs; whether the X3 and M3 are modelled
(URDF -> USD) with their lidars at the right heights and their cameras; the ROS 2 bridge (Humble).

## M1 -- The console: switch between the robots, or use both at once

**Stack** (all vendored into the repo, no build step, no framework):

| piece | version (Sep 2026) | why |
|---|---|---|
| dockview-core | 8.3.x, MIT, ~85 KB gzip, 0 dependencies | tabs, drag-to-split, floating groups, pop-out windows, layout to/from JSON -- all in the free tier. Pin the MIT release: a paid tier appeared in August 2026 |
| custom elements + Canvas 2D | -- | every panel is a `<robot-camera robot="m3">`-style element: no framework |
| three.js | r186, MIT | point clouds and meshes; loaded only when a 3D panel opens |
| Spark | 2.2, MIT | Gaussian splats in the same three.js scene (M4 upgrade) |
| MediaMTX | 1.21, MIT | later: WebRTC video from the Jetson's hardware H.264 encoder |
| foxglove_bridge + Lichtblick | 3.5 / 1.29 | ROS debugging views in a pop-out, no code of ours |

Rejected: golden-layout (no release since 2022), FlexLayout (React only), Lumino (11 packages),
Foxglove (closed since 2024; embedding needs a paid or academic plan), Potree and gsplat.js
(dormant), Rerun's web viewer (~52 MB).

**Design:**
- `console/robots.json`: each robot's id, name, colour, API and video URLs. Each backend answers
  `GET /api/robot` with what it can do (arm, base, depth, voice), so panels adapt.
- **Switch or both.** "Focus X3" / "Focus M3" retargets every panel at once; the "Both" preset is two
  columns; anything can be dragged, split, floated or popped out onto a second monitor. Layouts
  save themselves (localStorage) and named presets live in git as JSON.
- **Always visible:** a STOP per robot and STOP ALL; each robot's colour on every panel header;
  every command names its robot; one driver per robot at a time.
- **Panels:** camera; "how it sees" (detections); manual control; jobs; timing; log; the arena
  (the mat's image with both robots, paths, the object memory and the search belief drawn on it);
  3D viewer; catalogue (with Teach); voice (push-to-talk, transcript, parsed command, Run).
- **Backends:** both robots' web servers gain CORS headers, `/api/robot` and an SSE stream
  (`/api/events`, 5-10 Hz) replacing polling. Point clouds go as int16 millimetres + RGB, zlib'd
  with the standard library and unpacked in the browser with `DecompressionStream`.
- **Pitfalls found:** the microphone only works in a secure context, so the console runs on the
  laptop at `localhost`, not from a robot's address; browsers keep ~6 connections per host and each
  MJPEG view holds one, so few video views per robot, never both robots through one proxy.
- **Upgrade:** WebRTC through MediaMTX from `nvv4l2h264enc` (baseline profile, published over RTSP --
  JetPack 6 ships GStreamer 1.20, too old for WHIP); ~5-8 % CPU at 1080p30 measured by RidgeRun on
  an Orin NX. MJPEG is not slower on a LAN; what it costs is bandwidth and connections.

Exit: both robots driven from one page; focus switch and side-by-side work; STOP ALL halts both in
under 0.5 s; a layout survives a reload.

**First version, 2026-09-30** (`console/`, see its README): plain page, no dockview yet -- X3 /
M3 / Both, a card per robot (X3: camera, state, Pick one / Clear the table / Home; M3: live map,
drive pad and keys, auto loop), STOP per robot and STOP ALL (Space/Esc), pop-out per robot,
layout remembered. `serve.py` passes JSON through (the X3 panel has no CORS). `--fake` runs a
pretend X3 (`tools/webapp.py --sim`) and M3 (`m3pro/arena/fake_m3.py`: the real drive logic,
now `m3pro/arena/drive_core.py`, on the saved map with a ray-cast lidar). Checked on the pretend
robots: driving, the auto loop, the layout surviving a reload; STOP ALL confirmed by the M3 in
20 ms and the X3 simulator in ~0.8 s (to measure on the real X3). Not on the real robots yet.

## M2 -- The arena: localize on the fence, search, pick what we define

**M2.1 The arena as it is.** Yahboom's Autopilot Track Map (a UV-printed canvas, 3.2 x 2.8 m) on the
floor; a custom fence around it, 3.2 x 2.8 m, 30 cm tall, 5 mm white matte PVC, no markings; three
distinguishable 3D-printed landmarks, ~30 cm tall, at fixed positions; the whole arena modelled in
Isaac Sim (above).
- White matte PVC is an ideal lidar target (bright, diffuse). Every lidar plane must sit below the
  30 cm top edge of the fence AND of the landmarks, with margin: measure the M3's two T-mini Plus and
  the X3's RPLidar (it sits on the top deck: check it first).
- 3.2 x 2.8 is nearly square, but a lidar that sees all four walls tells 3.2 from 2.8, so a
  90-degree mistake cannot happen; a 180-degree turn fits a bare rectangle perfectly. The landmarks
  stand in the lidar planes, so the lidar itself rules that out (unless they sit symmetrically --
  the twin will show) -- no markers on the fence.
- The floor print is known artwork: dark grey roads inside a yellow outer loop, four green islands
  outlined in yellow (a wide one, a small square one, a tall one, and a U-shaped parking bay marked
  "P"), zebra crossings at the junctions, white turn arrows. The layout is asymmetric and
  high-contrast: a second, independent fix for the camera. Its features become named places for the
  catalogue, the console and voice (the parking bay, the islands, the crossings).
- The map (v2.3): slam_toolbox on the M3's two merged lidars, a PGM + YAML at 2 cm (the T-mini
  Plus is good to about 2 cm; at 2 cm the arena is 160 x 140 cells), checked against the fence's
  known 3.2 x 2.8 m. A second layer is the track print, scaled to metres and aligned to the fence --
  from RTAB-Map's top-down view of the floor (M4 B) -- for camera localization and as the
  console's arena.

**M2.1a Mapping, in levels (v2.3).**
1. *2D lidar SLAM -- the map AMCL and Nav2 use.* slam_toolbox (installed on both robots, Yahboom's
   launch on the M3) with our own settings for a small arena. Checked by fitting the fence: four
   walls, 3.2 x 2.8 m, square corners, the three landmarks as blobs. Its localization mode (scan
   matching against the saved pose graph) is a contender in study L next to AMCL. Both robots
   localize on this one map: the common frame for track B.
2. *Lidar + RGB-D SLAM -- the coloured 3D arena.* RTAB-Map on the M3 (Yahboom's launch: wheel
   odometry, the merged lidars, the DCW2 with the arm in the mapping pose [90, 180, 5, 0, 90, 0]);
   multi-session; exports the cloud, the mesh, the top-down image of the print and the camera
   poses for splats (M4 B).
3. *GPU mapping (cuVSLAM, nvblox)* -- only if a live GPU mesh is wanted: the DCW2 is untested with
   cuVSLAM's RGB-D mode and the JetPack 6 line of Isaac ROS lags behind.
4. *Neural SLAM -- the research comparison, on the 5090 from recorded data.* A Gaussian splat of
   the arena from RTAB-Map's poses (M4 C), and one neural SLAM system (GS-ICP SLAM with RGB-D, or
   VGGT-SLAM 2.0 from RGB only, e.g. the X3's wrist camera) against it.

What the M3 gives (surveyed 2026-09-30): the two T-mini Plus lidars come from the micro-ROS board
as `/scan0` (`laser0_frame`, left rear) and `/scan1` (`laser1_frame`, right front), each 270 deg
in 500 points at ~7.1 Hz, 0.05-12 m, mounted 14.5 cm above the floor -- well under the 30 cm
fence; `/odom_raw` at ~11 Hz and `/imu/data_raw` at ~24 Hz. Yahboom's EKF turns those into
`/odom` + the `odom -> base_footprint` transform, `ira_laser_tools` merges the two scans in
`base_link` (1 deg steps, 4 m cut-off, by default) and `yahboom_laser_filter` blanks returns
under 0.18 m (the chassis) into `/scan`. Nothing publishes the lidars' static transforms unless
`m3pro_bringup/static_tf.launch.py` runs. ROS 2 Humble, domain 30, Cyclone DDS.

**First maps, 2026-09-30** (`m3pro/arena/`, driven by `m3pro/tools/arena_map.py`; maps in
`m3pro/arena/maps/`). slam_toolbox at 2 cm on the merged lidars, the robot driven by our drive
page (`arena/drive_server.py`: by hand from a browser, or a slow loop round the fitted fence that
detours round the landmarks; lidar stop 35 cm ahead). Fence fit (`arena/fence_fit.py`):

| map | driven | fence (nominal 3.2 x 2.8 m) | corners | walls: seen, rms |
|---|---|---|---|---|
| arena_v0 | ~0.5 m | 3.213 x 2.801 m | 89.7-90.0 deg | 79-98 %, 1.1-1.5 cm |
| arena_v1 | ~2.5 m | 3.225 x 2.817 m | 89.3-89.8 deg | 100 %, 1.3-1.6 cm |
| arena_v2 | + one full loop (~3 min) | 3.232 x 2.823 m | 89.5-90.0 deg | 100 %, 1.6-2.1 cm |

All three landmarks found (21-30 cm across; closed outlines after the loop). The size creeping
up and the walls thickening with distance driven point at drift the scan matching does not fully
remove; slam_toolbox drops many scans ("queue is full": Yahboom's EKF publishes odom -> base at
only ~6 Hz). Next: the fence's true inside size by tape, then tune (EKF rate, scan queue, match
settings) until the map stops growing with distance.

**Tuning, 2026-09-30.** The user measured the fence: 3.2 x 2.8 m. One auto run was recorded
(`arena_map.py record loop1`, 128 s: 20 s standing, then half the loop) and replayed through
configurations side by side on the robot (`arena_map.py tune`, one ROS domain each, the fence
fitted every 20 s). What was found:
- **The lidars read long.** Standing still, with no SLAM, the two lidars alone see the fence
  1.5 and 2.3 cm too big: range biases of 0.77 and 1.15 cm (`lidar_calib.py`). That was most
  of the "growth": the first map was made close to one wall, where it shows least.
- **The robot board's clock runs 55-75 ms ahead of the Jetson's** (scans +55 ms, odometry +72,
  IMU +74 against arrival), so the EKF, which waits for the host clock to reach a stamp, always
  trails the scans, and slam_toolbox (queue of one scan) drops most of them. Keeping them
  (`scan_queue_size` 10) wrecked the map; the travel threshold (10 cm) picks the key scans
  anyway, so the drops cost little.
- **Chosen (now the launch defaults):** each lidar's own scan to slam_toolbox (`scan_relay.py`,
  biases taken off), EKF at 30 Hz, Yahboom's lidar mounting (the ICP correction changed
  nothing). In replay: 3.198 x 2.799 m at the end and within 1 cm of 3.2 x 2.8 all the way
  (Yahboom's settings: +1.5 to +2.4 cm and creeping up); reproduced in two separate replays.
- **Live, same day (after charging):** the per-lidar setting drops nearly every scan on the robot
  (a scan every 70 ms, the late odometry: one sensor registered, the map never grew), so the
  default went back to Yahboom's merged scan -- now merging the bias-corrected scans. A full live
  run (loop4, merged raw) replayed through settings: bias-corrected + Yahboom's EKF rate (7 Hz) +
  20 cm between key scans ends at 3.200 x 2.785 m, its long side moving 0.6 cm over the loop
  (Yahboom's settings: 3.219 x 2.805, +1.1 cm; EKF 30 Hz grows more). Now the default; within
  the tape's and the 2 cm cells' precision. To confirm live on a charged battery. The auto loop
  stopped twice "blocked" near the end of a live run; the cause is now logged.
- **Before that:** the robot's base board stopped publishing after the tuning runs
  (battery 10.39 V and falling; the agent restarted, the board did not reconnect). Walls are
  still 4-6 cm thick in every configuration -- the next thing, after a live run.

**M2.2 Where am I -- and would 3D scanning localize better?** Localization and driving are separate
parts: localization publishes where the robot is (one transform, `map -> odom`) and Nav2 only reads
it. AMCL, the neural network you plan, or a camera method are interchangeable without touching the
driving. The cues this arena offers:

| cue | sensor | what it gives here | limits |
|---|---|---|---|
| the fence's shape | 2D lidars, 360 deg, 6-12 Hz | continuous pose, ~1-3 cm and ~1 deg (est.) | a bare rectangle also fits a 180-degree turn |
| the three landmarks (~30 cm) | lidar (they stand in its plane); camera by appearance | the symmetry broken; a pose after the robot is picked up and moved | can be hidden behind the other robot |
| the track print | RGB-D camera looking at the floor | an absolute pose with no markers: depth measures the floor plane exactly, so the camera view becomes a metric top-down image to match against the print | the arm must point the camera at the floor; long plain road stretches carry little |
| a 3D scan of the fence | RGB-D camera | the same wall line the lidar already sees | narrower (91 deg), slower (15 fps), shorter (3 m) |
| a learned lidar model | 2D lidars | a pose from a single scan, anywhere | needs data; less exact than scan matching |

Our reading: in this arena a 3D scan does not localize better than the lidar -- the walls are plain
white planes, so depth adds nothing to the line the lidar already sees, and it sees less of it. With
~30 cm landmarks the lidar also breaks the symmetry by itself. The camera earns its place where the
lidar is blind: the printed track, as an independent absolute fix. The neural network fits best as a
global initializer that scan matching then refines, and it can be trained on unlimited synthetic
scans from the Isaac Sim twin (fence + landmarks + measured noise), then fine-tuned on real ones.
Which combination wins is a measurement: study L.

The baseline: nav2_amcl or Beluga AMCL (a drop-in, Humble release June 2026) with `OmniMotionModel`,
`likelihood_field_prob` with beam skipping on, `sigma_hit` 0.05-0.1, `update_min_d` 0.05 /
`update_min_a` 0.1, and no random-particle recovery (it invites jumps to the mirror pose). Start
from known poses -- the track's printed corners and crossings are free reference points. A
floor-print fix resets `/initialpose` when it disagrees by more than 0.3 m or 20 degrees.

**M2.3 Driving -- why Nav2, and what else we considered.** Nav2 is ROS 2's standard navigation stack:
it plans a path, follows it smoothly, avoids what the lidars see (the other robot included), stops
before a collision, and recovers when stuck (back off, wait, re-plan). Why it here:
- Yahboom already installed and configured it on the M3 -- their navigation runs out of the box, so
  the M3 can drive in the arena on day one;
- it drives a mecanum base properly: the MPPI controller's omni model moves sideways and diagonally
  with smooth velocities;
- keep-out zones: the cubes are below every lidar, so known ones become no-go circles our code
  republishes -- except the target, for its own approach;
- an independent safety layer (the collision monitor) slows and stops near obstacles whatever the
  planner does;
- it is standard: our effort goes into search and manipulation, and the research sits on a base
  everyone recognizes.

Considered instead: a ~200-line holonomic go-to-pose controller of our own -- enough for a 3.2 x 2.8
m arena and fully transparent, but we would re-write obstacle avoidance, the other robot, recoveries
and safety stops; learned end-to-end navigation -- a research project of its own, not needed here;
Yahboom's road-network navigation -- limited to drawn roads, while search needs free-space goals.
So: Nav2 behind our own `drive_to(pose)` skill. If it becomes a burden (the X3 has no Nav2 setup in
our software yet), the small controller replaces it without touching anything above.
Settings: Smac 2D planner, MPPI with the Omni model (TwirlingCritic on, PreferForward off), a
velocity smoother, the collision monitor; the last ~10 cm by the arm camera's visual servo. On the X3
the bridge owns the serial port, so it also publishes `/odom` + TF and takes `/cmd_vel`.

**M2.4 The catalogue -- what we define.** Yahboom's kit (sizes from Yahboom's chapters; a calliper
check is pending):

| object | size | colours / marks | gripper (Yahboom's width table) |
|---|---|---|---|
| small cube | 30 x 30 x 30 mm | red, green, blue, yellow; some carry AprilTags | 134 deg |
| big cube | 40 x 40 x 40 mm | AprilTags | 115 deg |
| block (two cubes glued) | 30 x 30 x 60 mm | as the cubes | 134 deg, across a 30 mm side |

Per entry: names in EN/SK/CS, size class ("small" = 30, "big" = 40, "block"), colour, grasp width and
angle, and -- for track A -- visual-prompt embeddings. The catalogue feeds the detector, the voice
parser (names and size words) and the grasp planner. Named places are the track's features (the
parking bay, the four islands, the crossings).
A fixed three-shape kit simplifies detection: the M3's depth tells 30 from 40 mm and finds the 60 mm
block directly, colour comes from HSV, and a small detector fine-tuned on our own frames beats any
open-vocabulary model on speed and accuracy (M5). The AprilTags on the marked cubes double as
ground truth for measuring detection error.

**M2.5 Search.** A belief grid over the arena (5 cm cells), updated by detections AND by
non-detections through a MEASURED detection model: the probability of seeing an object by range,
viewing angle, size and camera. The whole arena is within 3 m, but a 30 mm cube at 2.5 m is only
~6 px wide in the 640-px colour stream (the DCW2's 1080p stream: ~28 px at 1.5 m; est.), so where to look
from, and how close, still decides the time to find. Candidate views pair a base pose with a camera
pose (the M3 tilts its arm; the X3's mast RGB-D looks far, its wrist camera near) and are scored by
the probability mass they would see per second of travel. Coarse to fine: see at 1-2.5 m -> drive to
a staging pose -> close re-detection -> pick. The console shows the belief draining over the mat
("fog of war"). Both robots share one belief (track B).

**M2.6 Pick and drop.** The X3: its current pipeline, driven to a staging pose first. The M3: the v1
plan's bring-up and depth detector ([m3pro/PLAN.md](m3pro/PLAN.md): the survey, `m3bridge`, the arm
over `/arm6_joints`, its kinematics and hand-eye calibration, the height map, look-then-move). Drop
zones are named places on the track. A text-queryable object memory (each object's SigLIP 2 or DINOv3
embedding next to its position) answers "where did you see the red cube?".

Exit: from a known start, each robot finds three catalogue objects placed at random and delivers
them to a named zone; recorded: success, time to find, time to pick, localization error.

## M3 -- Voice, on the M3 first

**What Yahboom does:** their AI voice module (MEMS microphone, cavity speaker, echo cancellation) is
fitted on our M3. It appears as the system's audio input and output (its sensitivity is set in
GNOME's Sound settings). Wake word "Hello yahboom"; cloud speech recognition (Aliyun, iFlytek) with
SenseVoiceSmall and Piper as the local fallbacks -- so local speech already runs on this Orin; cloud
LLMs through a Dify server answering with a JSON action list parsed by a regex; Chinese and English
only. We keep the idea of actions as a fixed list and drop the rest.

**Two lanes:**
- **Stop lane**, always on, never through a model: a keyword spotter for "stop / stoj / stuj"
  (sherpa-onnx) calls the halt service directly. Target under 0.5 s from the end of the word to zero
  velocity, measured while the robot is talking and its motors run. The hardware e-stop stays.
- **Command lane:**
  1. wake word: our own model (microWakeWord or openWakeWord trained on Piper-generated samples;
     openWakeWord's pretrained models are non-commercial), or the module's, if it does it on-chip;
  2. Silero VAD, 0.5 s of silence ends the command (Yahboom waits 1.5 s);
  3. speech recognition: Parakeet-TDT-0.6B-v3 through sherpa-onnx on the robot -- 25 European
     languages including Slovak (8.8 % WER on FLEURS) and Czech (11.0 %), word timestamps (for C).
     SenseVoiceSmall, which Yahboom ships, has no Slovak or Czech;
  4. understanding: a grammar fast path for the simple commands (under 1 s); otherwise the small
     local model, constrained by a JSON grammar to fill in the command's settings, nothing else;
  5. a validator: known objects, inside the arena, robot free; ambiguous -> it asks back;
     a multi-step plan is read back with a 2 s window to cancel;
  6. the skill; a spoken answer with Piper (English voices; Slovak `sk_SK-lili`, Czech `cs_CZ` --
     one mediocre Slovak voice is the only one that runs on the robot today).
- **The command, as settings** -- what the small model fills in:

```
"Pick only the small cubes"         -> {"action":"pick","filter":{"class":"cube","size":"small","count":"all"}}
"Pick up the big cube"              -> {"action":"pick","filter":{"class":"cube","size":"big","count":1}}
"M3, bring the red cube to the parking bay"
                                    -> {"robot":"m3","action":"bring","filter":{"class":"cube","colour":"red","count":1},"to":"parking"}
"Zober vsetky modre kocky"          -> {"action":"pick","filter":{"class":"cube","colour":"blue","count":"all"}}
```

- **The small model, distilled.** A large open model on the 5090 writes thousands of paraphrases in
  English, Slovak and Czech with their JSON, driven by the catalogue's names and the mat's places; a
  LoRA fine-tune turns Qwen3.5-0.8B or -2B (Apache-2.0, tool calling, Feb-Mar 2026) into the parser;
  it runs in llama.cpp on the Orin. Measured against the large model: exact-match rate, latency.
  Gemma 4 E2B (April 2026, takes audio directly) is the alternative and the subject of study S3.
- **Bigger questions go to the servers**, never needed for the basic commands: multi-step requests,
  "what is in the parking bay?" with a vision-language model. No cloud needed.
- **Audio:** the survey finds the module's devices (`arecord -l`, `aplay -l`) and whether its wake
  word runs on the module or in Yahboom's software. Push-to-talk in the console from day one, for
  both robots: the console sends the recording to the robot's own speech pipeline, one code path.
- Latency budget (est.): wake 0.2 s + end of speech 0.5 s + recognition 0.2-0.4 s + parser 0.3-1 s
  + first audio 0.3 s -> about 1.5-2.5 s to a spoken acknowledgement; the fast path under 1 s.

Exit: 50 English commands from 5 speakers with the motors running: at least 90 % parsed exactly,
the stop lane under 0.5 s, simple commands under 1 s; then Slovak and Czech with the same test.

### M3.1 -- One Slovak sentence, two robots (v2.4)

The user's pick from the showcase menu. It comes in three versions, each needing more than the one
before; v1 needs nothing the robots cannot already do.

**v1 -- box-sharing.** "Zober vsetky male kocky a daj ich do boxu" ("pick up all the small cubes
and put them in the box"). Both robots clear the small cubes from their own areas into the M3's box
(ArUco id 10, inside 15 x 11 cm), one at a time, and leave the big ones.

- **The station.** The box stands where both arms reach it. For the X3 that is in front of or to the
  right of its arm (J1 10-180, x3plus/PLAN.md X5). The M3 drives back to its job's start before
  every drop, as it does today. Neither robot crosses the arena: the M3 works the area round its
  start, with the moves it already has (slide, drive up); the X3 its own work area.
- **The command.** The console's grammar learns mission words. With both robots named, or "spolu" /
  "together", a command becomes one mission for both instead of being refused (today `voice.py`
  answers "both robots named"):
  `{"robots":["m3","x3"],"action":"pick","filter":{"class":"cube","size":"small","count":"all"},"to":"box"}`.
  The validator then checks that:
  - both robots are free;
  - each one sees the box;
  - the filter names known pieces.

  The plan is read back, with a 2 s window to cancel. No server model is needed: this is the
  grammar or the small parser.
- **Turns at the box.** The console's mission runner holds a lease on the box. A robot asks before
  a drop, drops, then hands the lease back. If the console or the WiFi goes away, a robot holding a
  piece waits at its carry height and never drops blind; STOP ALL works as always. The X3 already
  runs the M3's `/pick/*` interface, so both robots are driven the same way.
- **The answer.** Each robot's progress shows on its arm card. A spoken summary in Slovak through
  Piper (`sk_SK-lili`) comes from the console for now and from the M3's speaker once its voice
  pipeline runs: "Hotovo: M3 styri kocky, X3 tri" ("Done: M3 four cubes, X3 three").
- **Measured.** Small cubes boxed and big cubes left; seconds from the end of the sentence to the
  first motion and to the last drop; clashes at the box (target 0); words heard against words said.
- **Needs:**
  - X3 plan X5, the box; the X3 does not need to drive for v1;
  - the grammar's mission words;
  - the lease in `console/serve.py`;
  - Piper on the laptop.
- **Exit:** said by three speakers, the robots clear both areas into the one box with no clash at
  it, and the summary is spoken in Slovak.

**v2 -- arena-wide.** The same sentence, with the M3 searching the whole arena. Needs M2:
localization, Nav2, the catalogue, search and pick. The X3 joins the search only with x3plus/PLAN.md
X7.

**v3 -- free-form.** Any mission in Slovak, Czech or English. A large model on the 5090 splits it
into steps for each robot, the console draws the plan as a graph and fills it in as the robots work,
and the validator stays between the model and the motors. Needs the servers; B's simple work split
(its auction) is the fallback when the model does not answer.

## M4 -- 3D scan with coloured point clouds (the M3's camera)

The camera, per Orbbec's own sheet: depth 640x400 at **15 fps** in every mode (Yahboom says 30;
we measure), 0.2-3 m (5 m in high-power mode), depth FOV 91 x 62 deg, hardware depth-to-colour
alignment, colour up to 1080p at 15 fps, USB 2.0.

**A. Object scan with the arm -- the headline demo.**
1. Hand-eye calibration once (ChArUco board, `cv2.calibrateHandEye`; it is also M2.6's).
2. The arm orbits the object on 2-3 rings of 8-12 views at 0.25-0.35 m and STOPS for each view
   (the M3 cannot read its joints back, so nothing is captured while moving); exposure and white
   balance locked; 3-5 depth frames averaged per view.
3. Forward-kinematics poses as the start; crop to a box, remove the floor (RANSAC); multi-scale
   coloured ICP (Park 2017) against the growing model; TSDF fusion at 1-2 mm; marching cubes.
4. Export: the point cloud as PLY, the mesh as GLB; the console's 3D panel shows both.
- Speed (est.): 1-2 min of capture, set by the arm, then 10-60 s on the Orin's CPU with the plain
  Open3D wheel -- CUDA is not needed at this size (Open3D's aarch64 wheels are CPU-only;
  jetson-containers builds it with CUDA if ever needed).
- The kit's pieces are small for a scan (a 30 mm cube is about 35-40 depth pixels across at 0.25 m;
  est.): scan the landmarks, a pile of cubes, or a bigger textured object for the demo.
- v2: the mecanum base circles the object while the arm keeps the camera on it -- all 360 deg, with
  ICP closing the loop.

**B. The arena while driving.** Yahboom's RTAB-Map launch (wheel odometry + lidar + DCW2) with the
arm in a fixed survey pose; `Reg/Strategy` ICP with the lidar and `RGBD/NeighborLinkRefining=true`
so the fence corrects mecanum slip; a few minutes at 0.2 m/s or less for this small arena (est.);
`rtabmap-export --cloud` (PLY) or `--mesh --texture`, and `--poses_camera` for the splats. The same
scan, seen from above, is an orthophoto of the floor print: the reference image for M2.2's camera
localization if Yahboom's print file is not at hand. nvblox (Isaac ROS 3.2, the JetPack 6 line) is
the optional live GPU mesh.

**C. The upgrade people remember: Gaussian splats.** The robot logs keyframes (colour, aligned
depth, poses from A or B) as a nerfstudio `transforms.json` with depth; the 5090 trains a splat with
gsplat (Apache-2.0; its 2D-Gaussian mode gives sharper surfaces) or Brush, seeded from the TSDF
cloud, in minutes (est.); converted to SOG/SPZ, it is shown by Spark in the console together with
the mesh, the arena and the robots. A live option, Photo-SLAM in RGB-D mode (17.9-28 fps on an AGX
Orin; GPL-3.0; documented for JetPack 5), is an experiment, not the plan. Feed-forward 3D models
(VGGT, MapAnything, Pi3X, Depth Anything 3) add little where the depth and the poses are good; where
they may help -- black or shiny objects, closer than 0.2 m, the X3's monocular camera -- they run on
the servers (study S4).

*Done 2026-10-12, the object first:* `m3_scan.py fuse --export` writes the scan's views, depths and
corrected cameras; `train/splat/train_splat.py` (gsplat 1.5.3 built for Windows on hekate) fits it with
the depth camera's depth held to the render (weight 0.5: held-out depth 64 -> 6 mm off); the `.splat`
it writes, cropped to the starting points' cylinder (0.4 of the Gaussians only painted the room), is
drawn by the console's own WebGL viewer (cloud.js: EWA quads, a counting sort, 5-10 ms for 0.2 M
Gaussians) beside the clouds -- no Spark needed yet. The can: 206,625 Gaussians, 6.6 MB.
All of it is one command, `tools/splat_cycle.py` (scan -> console, ~2 min); the Jetson box: held-out
PSNR 22.7 dB, depth 3.2 mm off; where the scan's poses disagree, a doubled edge. Pose refinement (each
camera's move and turn fitted with the splat, rate 1e-4; held-out views aligned to the splat before
scoring) on the same exports: the box 24.4 -> 25.5 dB, the can 20.3 -> 21.5 dB; the cameras moved a
median 3-6 mm and 0.5-1 deg (the can's up to 17 mm). The can's print and lid now read from all round.

Exit: press Scan -> within about two minutes a coloured mesh and point cloud of the object in the
console; an arena point cloud; one Gaussian splat of the arena with the robots in it.

## M5 -- The newest YOLO and friends, exported for speed

**Where things stand (Sep 2026).** Ultralytics 8.4.165 (28 Sep). YOLO26 (Jan 2026): NMS-free
head, 7 tasks including metric depth; YOLOE-26 takes text prompts, example images (visual prompts)
or no prompt; export now has `quantize=16|8` and training `quantize=8` (INT8 quantization-aware
training); 8.4.157 made TensorRT engines up to 20 % faster. YOLO27 was previewed on 13 Sep with no
weights -- on the watch list. TensorRT 10.3 on the hosts, 10.4 in our container.

**Labels for free.** On the M3 the depth detector outlines every object on the floor, and the
catalogue's sizes and colours name it: hundreds of labelled frames in an afternoon, from both
cameras' viewpoints, at every range the search needs. SAM 3 on the 5090 tightens the masks; the
tagged cubes give exact positions. The X3's frames are labelled the same way where the M3 has seen
the same scene, or by the X3's own ranging.

**The shortlist** (latencies: Orin NX TensorRT, measured by others unless "est."):

| # | model | why | Orin NX | licence | defines objects by |
|---|---|---|---|---|---|
| 1 | YOLO26n/s-seg, fine-tuned on our kit | the main detector: three shapes, colour from HSV; INT8 by quantization-aware training | detect n 4.1 / 3.5 ms, s 6.4 / 4.8 ms (FP16 / INT8) | AGPL-3.0 | fixed classes |
| 2 | YOLOE-26s/m-seg | what we run today; the open-set baseline; its prompts bake into a normal engine | est. 9-12 ms FP16 (41 ms in PyTorch today) | AGPL-3.0 | text, example images, none |
| 3 | RF-DETR-N/S and Seg-N (ICLR 2026) | transformer, fine-tunes well on small data | N 8.3 ms, Seg-N 18.1 ms with pre/post (TensorRT 10.3) | Apache-2.0 | fixed classes |
| 4 | DINOv2 ViT-S matching on object proposals (CNOS-style) | a new object from a few photos, no training (track A) | est. 5-15 ms a batch | Apache-2.0 | example images |
| 5 | OWLv2 (image-guided) / NanoOWL | one-example baseline for track A | NanoOWL 95 fps on an AGX Orin | Apache-2.0 | text; OWLv2 also an image |
| 6 | SAM 3 / 3.1, on the 5090 | the best labeller | 0.95 s an image on an Orin NX even in TensorRT: offline only | SAM licence, gated | text + example boxes |
| 7 | RT-DETRv4-S / D-FINE-S | Apache transformer baselines | est. 9-10 ms | Apache-2.0 | fixed classes |
| 8 | YOLO26n-depth | the X3 has no depth camera: 5.9x faster than Depth Anything V2 Small per Ultralytics | measure | AGPL-3.0 | -- |

DEIMv2 (DINOv3-based, state of the art) needs TensorRT 10.6+ for correct FP16 -- servers only.
Grounding DINO 1.5 Edge, DINO-X and T-Rex2 are cloud APIs: not benchmarkable on the robot.

**Export recipes:**
- Ultralytics: `yolo export model=best.pt format=engine quantize=16 imgsz=480,640` (the camera's
  4:3, a quarter less work than 640x640); INT8 with `quantize=8 data=kit.yaml` and 500+
  calibration frames, or train with `quantize=8` and skip calibration. TensorRT 10.3 cannot build an
  NMS-free INT8 engine: export falls back to the NMS head -- measure both.
- YOLOE with visual prompts: `refer_image` + `visual_prompts` -> save the prompt embeddings -> export;
  after export the classes are fixed. `get_visual_pe` / `set_classes(names, embeddings)` allow
  averaging several views and cutting the prompt-free model's 4,585-name vocabulary to ours.
- RF-DETR, RT-DETRv4, D-FINE: ONNX -> `trtexec --fp16`. No INT8 for vision transformers here: ModelOpt
  INT8 made a ViT-S 2.7x SLOWER on an Orin Nano with TensorRT 10.3 (the same warning holds for Depth
  Anything).
- Any other CNN: ModelOpt explicit INT8 (ONNX PTQ, entropy calibration) -> `trtexec --fp16 --int8`.
- Always: build on the robot inside the container that serves it; static shapes; a timing cache
  (rebuilds in about a minute instead of six); CUDA graphs when measuring latency. The DLA (two cores
  on the Orin NX 16 GB) adds throughput for CNNs, not latency; attention layers cannot use it.

**Benchmark protocol.** The test sets of both robots, binned by range (0.3, 1, 1.5, 2.5 m -- the
search needs the far bins); per model: mAP on the kit, recall on the 30 mm pieces, colour accuracy,
false positives on the track's printed arrows and crossings, tags and shadows, p50/p99 latency engine-only and
end-to-end, memory, and energy per frame (tegrastats). One table, re-run whenever a model or an
export changes.

Exit: the table for models 1-3 and 8 on both robots; the kit detector served as an INT8 engine.

**The table, first fill (2026-10-12).** Speed: TensorRT 10.7 engines built on each robot (trtexec,
GPU compute time only, CUDA graphs, 10 s after 0.5 s of warm-up), clocks not locked (no
jetson_clocks), the robots' other GPU services up but idle. Both robots are Orin NX 16 GB at
MAXN_SUPER and agree within 0.05 ms. Accuracy: mask mAP50 (mAP50-95) on the kit's real val -- 14 of
the M3's pick looks from 8 sessions, only three of them 40 mm cubes, so read it as a direction.

| # | model | input | precision | M3 ms median / p99 | X3 ms median / p99 | kit mask mAP50 (50-95) |
|---|---|---|---|---|---|---|
| 1 | YOLO26n-seg, real + Isaac Sim synthetic, fine-tuned on real | 480x640 | FP16 | 3.00 / 3.06 | 3.01 / 3.03 | **0.791** (0.537) |
| 1 | the same, INT8 by quantization-aware training | 480x640 | INT8 Q/DQ | 3.18 / 3.21 | 3.18 / 3.20 | 0.796 (0.538) |
| 1 | YOLO26n-seg, real pictures only | 480x640 | FP16 | (as above) | (as above) | 0.653 (0.532) |
| 1 | YOLO26s-seg, real + synthetic, fine-tuned on real | 480x640 | FP16 | 5.37 / 6.56 | 5.39 / 6.54 | 0.697 (0.505) |
| 2 | YOLOE-26s-seg, 3 text prompts baked in | 480x640 | FP16 | 5.36 / 6.40 | 5.39 / 6.57 | open-set: not on the kit yet |
| 3 | RF-DETR-Seg-Nano, real pictures only | 312x312 | FP16 | 8.00 / 9.20 | 7.94 / 8.00 | 0.683 (0.537) |
| 8 | YOLO26n-depth | 480x640 | FP16 | 3.80 / 4.89 | 3.80 / 4.96 | depth: the duel's to measure (S1) |

What it says so far:
- **YOLO26n-seg in FP16 is the kit detector**: 3 ms, the best accuracy, and synthetic pictures from
  the robot's own camera poses lifted it most (0.653 -> 0.791; the 40 mm cube 0.343 -> 0.672). A
  synthetic camera anywhere 18-55 cm away had made it lose the 40 mm cube: size in the picture is
  what tells the two cubes apart from the robot's few poses.
- **INT8 buys nothing here, so the exit's "INT8 engine" is moot for now**: the QAT engine keeps the
  accuracy but runs 0.18 ms slower -- of its 301 layers 100 are reformats between INT8 and float
  around the convolutions -- and a plain INT8 build fails (TensorRT 10.7 has no INT8 kernel for the
  mask head's proto conv, even pinned to FP16). At 3 ms against a 33 ms frame there is nothing to win.
- **YOLOE baked into an engine is ~8x its PyTorch speed** (5.4 ms against ~41 ms served today): the
  open-set teach could run that way. **RF-DETR** is no better on the kit at 2.7x the time.
- **YOLO26n-depth at 3.8 ms** against Depth Anything V2 Small's ~26 ms engine on the X3 -- but in the
  depth duel (S1; m3pro/tools/m3_duel.py, 69 pictures, 192 pieces) it **flattens the 30 mm pieces to
  ~6 mm**: anchored as Depth Anything is, height error 23.9 mm median and 8 % named (DA 6.5 mm, 54 %;
  the DCW2 0.5, 100 %), the floor about as flat (8-10 mm sd); as it reads, its scale is off by ~0.5 m.
  As trained it cannot stand in for Depth Anything on the X3; fine-tuned on the M3's own RGB-D (the
  DCW2 as the label) it might -- a study of its own.
- **End to end on the M3** (train/robot_e2e.py, its 92 pick looks, Ultralytics' predict): the kit
  detector as an FP16 engine 8.3 ms a picture (p99 9.6: preprocess 0.8, inference 3.2, the masks'
  postprocess 3.5), in PyTorch 32.6 ms -- the masks, not the network, are now the bigger share.
- **On the X3's camera (2026-10-15) the M3-trained detector is half as good, and the X3's own
  pictures fix most of it** -- but a model for both is worse on the M3. The X3's looks labelled
  without hands (x3plus/tools/kit_dataset_x3.py: each piece a survey found projected through the
  wrist camera's model; train/refine_x3_sam.py: SAM 2 redraws the outlines, and drops pictures
  where a mask disagrees or the detector sees a piece unlabelled), 680 of 1,085 kept; a fresh
  round of 71 held out as the X3 test (train/eval_kit.py, mask mAP50):

  | model | M3 val (14) | X3 val (71) | X3 test (71) | X3 test cube30 / cube40 / block |
  |---|---|---|---|---|
  | `n_cycle`: M3 + synthetic (the 0.791 above) | **0.791** | 0.302 | 0.246 | 0.333 / 0.002 / 0.403 |
  | `n_20261015`: + the X3's pictures | 0.560 | **0.656** | **0.497** | 0.589 / 0.452 / 0.449 |

  The M3's loss is mostly its 40 mm cube (0.672 -> 0.056, three in its val) under 15x more X3
  pictures. Proposed, not yet decided: a model per robot for now -- the X3 `n_20261015`, the M3
  `n_cycle` until its own 40 mm and block frames are in.
- Not yet: the kit detector served by a robot's vision service, energy (tegrastats), locked clocks,
  and a val with tens of 40 mm cubes on the M3.

## The research program -- gaps worth a paper, demos worth watching

Each track: a gap checked against 2024-26 work (sources at the end), doable on these robots with the
lab servers, measurable, and built on the must-haves so no work is wasted.

**Order since v2.4 (2026-10-07), from the user's showcase picks:**
- **A first.** A1-A2 come in stage 1 on the object scan's orbit; A3-A4 make the paper.
- **Then D**, the user's "maybe".
- **B** is cut to what the Slovak mission needs: localization, Nav2, and its auction as the
  fallback work split. Its belief-search planner, the fog of war, and study **L** are optional.
- **E** and **C** later.

Before v2.4 the order was B -> A -> D. B came first then because it grows out of must-have 2.

| track | the gap | the demo | builds on | effort, risk |
|---|---|---|---|---|
| B Two robots, one belief | search planning that decides where an ARM camera looks, for small objects, shared by two different robots | the fog of war drains over the mat as the M3 tilts its camera and the X3 sweeps; they split the picks | M2, M1 | M, med |
| A Show-and-pick | multi-view + negative visual prompts compiled into an edge engine; labels checked by the robot's own grasps; time from "Teach" to first pick | "Teach": the arm circles a new object, and a minute later the robot picks it out from among look-alikes | M5, M2, M4 | S (A1-A2) to M (A3-A4), low-med |
| D A picker that teaches a VLA | a vision-language-action policy trained only on demos our classical picker generates, corrects and scores by itself | an overnight time-lapse of the X3 teaching itself; next day the policy follows a cube moved mid-reach, which detect-then-move cannot | the X3 as it is | M, med |
| E A map that knows what it did | object memory (and splats) updated from the robot's own action log -- telling what the robot moved from what a person moved | the picked cube's splat rides with the gripper to the drop; a cube a person moved is flagged and re-inspected | M4, M2 | M, med |
| C "Bring me that" | pointing + words grounded in a live object memory on edge robots, one robot watching the person while the other fetches, asking back only when unsure | "take this and put it there" among identical cubes, in English or Slovak | M3, M2 | M-L, med-high |
| L Which localization wins | lidar geometry vs a learned lidar model vs the printed floor seen in RGB-D, alone and fused, in a small symmetric arena | a robot lifted and turned 180 degrees re-localizes in seconds; live error against ground truth | M2 | S-M, low |

**B -- Two robots, one belief (optional since v2.4; only its localization, Nav2 and auction are
needed, by M3.1 v2-v3).** The M2.5 belief with a measured detection model; the next
view chosen over base pose AND arm tilt, within what the 5-DOF arm can reach; different sensors
(M3: a tiltable RGB-D; X3: mast RGB-D far, wrist camera near); sequential auctions for views and
picks, against an LLM allocator on the servers. Simulation first: hundreds of seeded trials in the
Isaac Sim twin (both robots, the measured detection model), then 20 real ones. Why it is the
easiest start: every piece (localization, the detection model, the belief, the arena panel) is
needed for must-have 2 anyway; the research part is the planner and the sharing. Evaluation: 20
seeded layouts x {lawnmower, frontier of the seen area, belief next-best-view} x {1, 2 robots} with
the kit's pieces: time to find, success within 180 s, path length, pieces knocked; allocation: picks
a minute, idle time, reassignments, with WiFi degraded.

**A -- Show-and-pick: how complex it is.** In four levels; A1-A2 give the demo, A3-A4 the paper.
Since v2.4, A2 shares the object scan's orbit (M4 A): one capture gives both the 3D scan and the
taught object.
- *A1, a few days -- "show it once".* 1-3 snapshots of a new object; YOLOE turns the boxed object into
  a visual prompt (a vector that describes it) and detects it from then on. Built into Ultralytics;
  the engine is rebuilt with the new class (about a minute with a timing cache).
- *A2, about a week -- "the robot looks for itself".* The M3's arm circles the object (8-12 views),
  the depth cuts it out of every view (masks for free), the views' prompts are averaged and
  look-alikes are added as negatives. Measured: accuracy against the number of views.
- *A3, 1-2 weeks -- no rebuild at all.* An engine that takes the object descriptions as an INPUT, so
  a new object is usable the moment it is taught. Ultralytics cannot do this for TensorRT engines
  today; it needs surgery on the exported graph -- this is the contribution.
- *A4, 1-2 weeks -- from prompt to trained model.* The orbit frames become a training set; a small
  YOLO26 is fine-tuned on the 5090 in minutes, keeping only labels a successful grasp confirms, and
  the new INT8 engine is hot-swapped on the robot.
Evaluation: AP and pick success for 1/3/5/10 views, with and without negatives, against text
prompts, DINOv2 matching, SAM 3 and the fine-tuned model; seconds from Teach to first pick. For the
kit itself A is not needed (M5 trains on it directly); it is for everything new.

**First live result, 2026-10-09 (A1-A2 on the M3; `m3pro/roboarm/m3teach.py`,
`m3pro/tools/teach_service.py`, `m3pro/tools/m3_show.py`).** Four white picture cubes (apricot,
news, cosmetics, fish -- same shape, different prints) each taught from 5-6 views (the wide look
turned on J1, two from above; the depth boxes the thing in each). YOLOE-26's visual prompts found
the cubes but named every one "cosmetics" (0.15-0.46); matched view to view, YOLOE's prompt
embeddings named 1 of 4 right (640 and 1280 px), DINOv2 ViT-S/14 on the crops 4 of 4, the right
name 0.09-0.17 ahead. So the identity is DINOv2's (the user's choice), YOLOE stays for detection
and A3: the four cubes then named right live (0.67-0.80), two blocks and the rest unknown
(<= 0.54), and "pick fish" picked it out of its look-alikes into the box in 20 s. For the paper:
visual prompts on a real-time detector do not reach instance level for same-shape look-alikes;
the prompt-as-input engine (A3) needs an instance-level stage beside it.

**Views from every side, and the first timed teach, 2026-10-15.** Taught from one side only, both
news cubes stayed unknown on 2026-10-12 (0.64 and 0.61, the bar 0.65). Teach again with the same
name adds views: four rounds each, the cube turned a quarter between them (4 views a round), took
the cubes to 19-23 views (news 19, fish 22, cosmetics 20, apricot 23). Find with the Jetson box,
the can and both news cubes and the fish among them then named every cube right -- news 0.85 and
0.80, fish 0.85 -- and the box 0.76; the can stayed unknown (0.47), lying and mostly out of the
picture (taught standing, from a scan). **Teach to first pick** on a thing the robot had never seen
(a yellow cylinder, ~3 cm): Teach 10 s (every Teach that day 10.0-10.5 s: the wide look, 4 views,
DINOv2, home), then Pick by name: found at 0.95, the next-best name 0.58 behind; picked and put
down 29 s after it was found -- 9 s of that a search for the drop-off box that was not there, 19 s
the pick. About 50 s of robot time from Teach to the piece put down; the demo's "teach in 30 s"
holds for the teach itself.

**D -- A picker that teaches a VLA: how it works.** (After A since v2.4, the user's "maybe";
RESEARCH.md P1 has the competing 2026 work and the design changes it forces, e.g. an overhead
teacher camera.)
- *What a VLA is.* A vision-language-action model is one neural network that looks at the camera
  image, the joint angles and an instruction ("pick up the red cube") and outputs the next arm
  motions directly -- about a second of joint targets at a time, many times a second. No separate
  detector, IK or grasp planner: it learns the whole mapping from examples. Examples: SmolVLA (450M
  parameters, Hugging Face), pi0.5, GR00T.
- *The usual bottleneck* is data: people must demonstrate each task 50-200 times by remote control.
- *Our twist:* the X3 already has a classical picker that works (detector -> IK -> grasp, 1-4 mm).
  It produces the demonstrations itself:
  1. **Record.** The X3 runs its normal picks while we record, 10-30 times a second, the wrist
     image, the joint angles, the commanded joint targets (the "actions") and the instruction text.
  2. **Reset by itself.** After each drop it places a cube at a random reachable spot, so the next
     episode starts on its own. Overnight: hundreds of episodes with nobody present.
  3. **Train.** The dataset goes to the 5090; SmolVLA is fine-tuned from its pretrained weights
     (trained on SO-100/101 arms, which have our arm's layout: base yaw, three pitch joints, wrist
     roll) -- a few hours.
  4. **Run.** The policy runs on the 5090 or the laptop as a policy server; the robot streams its
     image and joints and receives action chunks over WiFi (LeRobot's asynchronous inference); a
     watchdog holds the arm if chunks stop coming. Later on the Orin itself (SmolVLA needs ~2 GB).
  5. **Correct.** When the policy drifts, the classical picker -- which can compute the right move
     from any arm position -- supplies the corrected action; the corrections join the dataset and
     the policy is retrained (DAgger). The robot improves itself.
  6. **Score.** Success is checked automatically (the cube is gone from the table, the gripper's
     read-back shows it closed on something), so success rates need no manual labelling.
- *What it can show:* whether a small VLA trained only on self-generated data matches the classical
  picker; what it does that the classical one cannot -- follow a cube moved during the reach,
  handle a new instruction ("the blue one") -- and how many automatic demos replace human ones.
  Fully autonomous data, correction and scoring on a low-cost arm is the gap.
- *Effort and risks:* 3-5 weeks to first numbers: a LeRobot plugin for the arm (~1 week), the
  recorder and reset loop (~1 week), training (days), evaluation. Risks: the X3's servo bus must take
  10+ commands a second; WiFi hiccups (the watchdog); the first policy may pick worse than the
  classical one -- that is a result, not a failure. The X3 first because it already picks reliably
  and reads its joints back; the M3 cannot yet.
- *Evaluation:* a 25-position grid x 2 trials with confidence intervals; seconds per pick; success
  when the cube is moved mid-reach; 50 vs 200 automatic demos, + noise, 50 hand-guided demos (the
  X3 can go torque-off), + correction rounds; X3 -> M3 transfer.

**E -- A map that knows what it did.** Systems today detect moved objects from what they see
(DynaMem, DovSG, DREAM, SuperMap) or re-optimise splats after passive change detection; none uses
the arm's own action log (grasp succeeded, released at this pose) as an object-permanence prior on a
16 GB edge device, or separates robot-caused from human-caused change. The two robots share the
object map with the fence as the common frame (the M3 scouts, the X3 picks without searching).
Evaluation: 30+ pick/place cycles with injected human moves: change precision/recall and latency,
object error in mm against the tagged cubes, text-query accuracy over time, ms and GB per update,
against re-mapping from scratch.

**C -- "Bring me that".** MediaPipe hands and pose on the CPU + depth give the eye-to-fingertip ray;
Parakeet's word timestamps tie "this" and "there" to the gesture; the ray is cast into the shared
object memory (the X3's mast camera can watch the person while the M3 fetches); a question is asked
only when a calibrated uncertainty says so (KnowNo-style); the piece is put down where pointed, or
handed over into an open palm. Recent work (MIGU, Sep 2026; LEGS-POMDP, HRI 2026; GIVE, Jun 2026) is
single-robot and mostly simulated or GPU-heavy. Evaluation: accuracy against the angle between
candidates (10/20/30 deg) and distance (1-3 m), questions asked, success, time; language only /
gesture only / both / both + questions; 8-10 participants.

**L -- Which localization wins, in a small symmetric arena.** AMCL on the fence and landmarks (the
baseline), your neural lidar localizer (trained on synthetic scans from the Isaac Sim twin, then real
ones), the track print through the RGB-D camera, and fusions of them. Ground truth: exact in the
twin; on the real robots, static at the track's printed corners and crossings; a camera above the
arena would add moving ground truth later, if wanted. Metrics: position and heading error, time to
recover after being lifted and turned (the 180-degree trap), how often the wrong mirror pose is
chosen, compute per update, and the gap between twin and real. It answers "would 3D scanning be
better?" with numbers, and gives the neural localizer a proper evaluation.

**Smaller studies** (each a short paper or a thesis chapter):
- S1 real vs predicted depth for finding small objects: the M3's stereo depth against the X3's
  YOLO26n-depth / Depth Anything V2 -- recall on the kit, false positives, latency, energy.
  Scheduled in stage 1 since v2.4 as "the depth duel". Its X3 half is x3plus/PLAN.md's depth table,
  and the first rows are in: the networks flatten 30-40 mm pieces, while parallax from two views
  gives real heights. The demo shows the M3's stereo depth beside it.
  **The M3 half, 2026-10-09** (`m3pro/tools/m3_duel.py`: Depth Anything V2 Small on the very
  pictures the DCW2 took, anchored to the floor as the X3 does; 64 pick looks, 171 whole 30 mm
  pieces the DCW2 named): on the WIDE look the DCW2 reads heights to 0.5 mm (median; p90 0.8),
  Depth Anything to 5.2 mm (p90 12) on the arm model's floor -- 68 % of the pieces named right --
  and 3.6 mm (p90 10, 77 %) on the DCW2's own floor; the table flat to 1.0 mm sd against 6-9.
  On the CLOSE look (~20 cm) Depth Anything fails: 15.5 mm median, 18-27 % named, though the floor
  fit says its scale is pinned and the look is no steeper (16 deg from straight down, the wide 20)
  -- the cause is open. Unlike the X3's views, its median height is right (30 for 30): it
  scatters rather than flattens. Picture panels: `m3pro/out/duel/panel_*.jpg`.
- S2 physical checks vs VLM judges for grasp and stacking success (a five-cube tower; someone knocks
  the top cube off; the robot notices and rebuilds). VLM judges lean towards "success" (FailBench,
  Sep 2026: 0.77 balanced accuracy at best); depth height change and gripper read-back are free. The
  checker becomes D's referee.
- S3 cascaded (Parakeet -> small LLM) vs speech-native (Gemma 4 E4B, audio straight to a tool call) on
  the robot, English and Slovak, motors running: exact command match, word error rate, p95 latency,
  energy, false wakes an hour.
- S4 do 3D foundation models help a cheap close-range RGB-D camera on dark, shiny and tiny objects
  (MapAnything, Pi3X, Depth Anything 3 against the DCW2's TSDF; Chamfer distance, completeness, time)?
- S5 how much localization error costs grasping: pick success across the arena against AMCL error,
  with and without the visual servo over the last metre.

**Watch list, not now:** YOLO27 (no weights); JetPack 7.2 with TensorRT Edge-LLM and Isaac ROS 4.6
(Ubuntu 24.04 -- breaks the vendor stacks); GR00T N1.7 (16 GB for inference, JetPack 7.2); pi0.7 (no
open weights); SAM 3D Objects (32 GB); VGGT-class models on the Orin (memory bandwidth); Nav2's vector
object server (Lyrical only); open-vocabulary 3D scene graphs at full scale (built for rooms full of
semantics; our arena has a printed map instead).

## Order -- by value, with a demo at the end of each stage

v2.4 order:

| stage | work | the demo | state 2026-10-07 |
|---|---|---|---|
| 0 Foundations | the M3's bring-up (v1 plan W0: survey, m3bridge, arm, kinematics, hand-eye); console v1 with both robots; the X3's base into ROS (now x3plus/PLAN.md X6) | both robots driven from one page | done, except the X3's base |
| 1 First show | the object scan from the M3's arm orbit (M4 A), and from the same orbit the teach (A1-A2); the depth duel (S1: x3plus/PLAN.md's depth table + the M3's stereo depth beside it); the model table (M5, x3plus/PLAN.md X9) | a coloured 3D scan in the console; "teach in 30 s", picked out from look-alikes; real vs predicted depth; the speed table | the X3's depth side running |
| 2 Voice | the stop lane on the robot; the robot's own pipeline (the module, wake word, VAD, Parakeet, parser, validator, Piper); the real-voice test; the box-sharing Slovak mission (M3.1 v1) | "Stop!" with its latency on screen; "Zober vsetky male kocky a daj ich do boxu" -- both robots, one box | console push-to-talk built |
| 3 The arena | AMCL, Nav2, catalogue, search and pick (M2) for the M3, then the X3 (x3plus/PLAN.md X7); the mission arena-wide (M3.1 v2) | "find and pick these three"; the Slovak mission over the whole arena | map and 3D scan done |
| 4 Research | A3-A4 (the paper); D on the X3 if chosen; free-form missions on the servers (M3.1 v3); optional: B's belief search and L, E, C, the live AI-agent demo | each track's demo and its numbers | not started |

Stage 1 is about two weeks (est.), stage 2 two to three, stage 3 a few weeks; the research runs
alongside once its stage is in place. The showcase menu's own total for the picked Wants is 8-15
weeks of one person's work.

Until v2.4 the order was: 0 foundations -> 1 the object scan, model table and A1 -> 2 the arena
with B and L -> 3 voice -> 4 B, A2-A4, D, then E and C.

## Questions

**Answered 2026-09-29:**
- the arena: Yahboom's Autopilot Track Map, 3.2 x 2.8 m, inside a 30 cm white matte PVC fence
  without markings, with three fixed 3D-printed landmarks ~30 cm tall -- all already modelled in
  Isaac Sim (the scanned map, the identical fence, the objects);
- Yahboom's voice module is fitted;
- the servers: the RTX 5090 in the lab on IISLab, dionysus with the RTX 4500 in another building by
  static IP or VPN, both shared; a colleague runs them (the user will arrange access);
- `console/` and the backend changes approved; the X3's bridge may drive the base and Nav2 may run on
  it (reasons in M2.3);
- the research order B -> A -> D is agreed (D and A explained above) -- replaced on 2026-10-07 by
  A -> D;
- the catalogue is the robot's kit (small cubes, big cubes, the two-cube block);
- stay on JetPack 6.2;
- **nothing is built until the user says "let's start".**

**Answered 2026-09-30:** "let's start" -- given; the Isaac Sim twin is left out of the first version
of localization (used later); mapping as recommended (M2.1a); the arena map is the first thing to
show.

**Answered 2026-10-06** (x3plus/PLAN.md): the X3 uses only its arm camera; its pick runs on the
robot; table first, then the arena mat; it shares the M3's box.

**Answered 2026-10-07:**
- the showcase picks, as listed under v2.4;
- the research order A -> D, with B cut to what the Slovak mission needs;
- the Slovak mission starts box-sharing (M3.1 v1);
- the live AI-agent demo is optional.

**Open, before stage 2:**
- **Where the shared station stands.** The box must be within both arms' reach. Proposed: the X3
  parked on the mat by the fence, with its work area and the box in front of or right of its arm,
  and the M3 working the area next to it.
- **RESEARCH.md's remaining decisions:**
  - which quick-win venues, if any;
  - whether to contact the two Slovak groups;
  - the overhead webcam (it was for D, which is now a "maybe", so it can wait).

**Waiting on:** server access from the colleague (Docker on the 5090, dionysus over VPN, a directory
for datasets, agreed GPU time).

**To check at the start, no need to ask:** the lidars' heights against the 30 cm fence and
landmarks; the Isaac Sim version, where it runs, and whether both robots are modelled with their
lidars and cameras; the M3 survey of v1 W0.1 (it needs either the user at the robot or an SSH key
for this laptop on the M3).

## Sources

Collected 2026-09-29. Yahboom's course chapters are listed in [m3pro/PLAN.md](m3pro/PLAN.md); the
arena and kit pages used for v2.1-v2.2:
[Autopilot Track Map (3.2 x 2.8 m)](https://category.yahboom.net/collections/a-robotics-map/products/autopilot-track-map) |
[colour block sorting, 30 mm blocks](https://www.yahboom.net/build?id=15844) |
[shape sorting, 30x30x60 mm block](https://www.yahboom.net/build?id=15847)

Console:
[dockview](https://github.com/dockview/dockview/releases) ([licence](https://dockview.dev/docs/overview/licence)) |
[three.js](https://github.com/mrdoob/three.js/releases) |
[Spark](https://github.com/sparkjsdev/spark) |
[MediaMTX](https://github.com/bluenviron/mediamtx) |
[Lichtblick](https://github.com/lichtblick-suite/lichtblick) |
[foxglove_bridge](https://index.ros.org/p/foxglove_bridge/) |
[secure contexts](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts/features_restricted_to_secure_contexts) |
[connections per host](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Connection_management_in_HTTP_1.x) |
[Orin NX H.264 (RidgeRun)](https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin_NX/GStreamer_Pipelines/H264)

Arena, localization, search, multi-robot:
[REP 2000](https://github.com/ros-infrastructure/rep/blob/master/rep-2000.rst) |
[nav2 AMCL](https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/others/configuring_amcl/) |
[beam skipping (Ekumen)](https://ekumenlabs.com/blog/posts/software-archaeology/) |
[Beluga AMCL](https://index.ros.org/p/beluga_amcl/) |
[MPPI](https://index.ros.org/p/nav2_mppi_controller/) |
[STVL](https://index.ros.org/r/spatio_temporal_voxel_layer/) |
[zenoh-plugin-ros2dds](https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds) |
[CLAP](https://arxiv.org/abs/2509.08495) |
[ground-texture localization (Micro-GPS)](https://arxiv.org/abs/1710.10687) |
[VLFM](https://github.com/bdaiinstitute/vlfm) |
[OneMap](https://github.com/KTH-RPL/OneMap) |
[OK-Robot](https://arxiv.org/abs/2401.12202) |
[DynaMem](https://arxiv.org/abs/2411.04999) |
[ActPerMoMa](https://arxiv.org/abs/2310.00433) |
[NATS](https://arxiv.org/abs/2011.04825) |
[two real mobile manipulators, LLM allocation](https://arxiv.org/abs/2509.12838) |
[COHERENT](https://arxiv.org/abs/2409.15146)

Voice and language:
[Yahboom voice chapter](https://www.yahboom.net/build?id=15786) ([core code](https://www.yahboom.net/build?id=17709), [module tests](https://www.yahboom.net/build?id=17714)) |
[Parakeet-TDT-0.6B-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
[sherpa-onnx](https://k2-fsa.github.io/sherpa/onnx/index.html) |
[Silero VAD](https://github.com/snakers4/silero-vad) |
[microWakeWord](https://github.com/kahrendt/microWakeWord) |
[openWakeWord](https://github.com/dscripka/openWakeWord) |
[Piper](https://github.com/OHF-Voice/piper1-gpl) ([Slovak voice](https://huggingface.co/rhasspy/piper-voices/raw/main/sk/sk_SK/lili/medium/MODEL_CARD)) |
[Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-4B) |
[Gemma 4](https://ai.google.dev/gemma/docs/core/model_card_4) ([on Jetson](https://www.jetson-ai-lab.com/tutorials/gemma4-on-jetson/)) |
[TensorRT Edge-LLM benchmarks](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/performance/performance-benchmarks.html) |
[KnowNo](https://arxiv.org/abs/2307.01928) |
[RoboGuard](https://arxiv.org/abs/2503.07885)

3D:
[DaBai DCW2 (Orbbec)](https://www.orbbec.com.cn/index/Product/info.html?cate=38&id=55) |
[coloured ICP](https://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html) |
[Open3D on ARM](https://www.open3d.org/docs/release/arm.html) |
[jetson-containers](https://github.com/dusty-nv/jetson-containers) |
[RTAB-Map parameters](https://raw.githubusercontent.com/introlab/rtabmap/master/corelib/include/rtabmap/core/Parameters.h) |
[nvblox with Orbbec on JetPack 6.2](https://wiki.seeedstudio.com/deploy_nvblox_jetson_agx_orin/) |
[gsplat](https://docs.gsplat.studio/main/apis/rasterization.html) |
[Brush](https://github.com/ArthurBrussee/brush) |
[nerfstudio data](https://docs.nerf.studio/quickstart/data_conventions.html) |
[splat-transform](https://github.com/playcanvas/splat-transform) |
[Photo-SLAM](https://github.com/HuajianUP/Photo-SLAM) |
[MapAnything](https://github.com/facebookresearch/map-anything) |
[Depth Anything 3](https://github.com/bytedance-seed/depth-anything-3) |
[VGGT](https://github.com/facebookresearch/vggt) |
[RayFronts](https://arxiv.org/abs/2504.06994) |
[DovSG](https://arxiv.org/abs/2410.11989) |
[DREAM](https://arxiv.org/abs/2606.00576) |
[POGS](https://arxiv.org/abs/2503.05189) |
[RoboEXP](https://arxiv.org/abs/2402.15487) |
[ROMAN](https://arxiv.org/abs/2410.08262) |
[Inria 3DGS licence](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md)

Models and export:
[ultralytics](https://pypi.org/project/ultralytics/) ([releases](https://github.com/ultralytics/ultralytics/releases)) |
[YOLO26 paper](https://arxiv.org/abs/2606.03748) |
[YOLOE](https://docs.ultralytics.com/models/yoloe) |
[Jetson guide (measured latencies)](https://docs.ultralytics.com/guides/nvidia-jetson/) |
[depth task](https://docs.ultralytics.com/tasks/depth/) |
[export settings](https://docs.ultralytics.com/usage/cfg/) |
[YOLO27](https://docs.ultralytics.com/models/yolo27/) |
[RF-DETR](https://github.com/roboflow/rf-detr) ([C++/TensorRT](https://github.com/infracv/rf-detr-cpp)) |
[DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) |
[RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) |
[D-FINE](https://github.com/Peterande/D-FINE) |
[SAM 3](https://github.com/facebookresearch/sam3) ([on Orin NX](https://github.com/dataplayer12/SAM3-TensorRT)) |
[EfficientSAM3](https://github.com/SimonZeng7108/efficientsam3) |
[OWLv2](https://huggingface.co/docs/transformers/en/model_doc/owlv2) |
[NanoOWL](https://github.com/NVIDIA-AI-IOT/nanoowl) |
[CNOS](https://github.com/nv-nguyen/cnos) |
[ModelOpt ONNX PTQ](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq) |
[ViT INT8 regression on Orin](https://forums.developer.nvidia.com/t/tensorrt-model-optimizer-int8-quantization-causes-2-7x-performance-regression-on-jetson-orin-nano-4gb-vit-s-dpt-architecture/357835) |
[JetPack 7.2](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2) |
[TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html) |
[Ultralytics licence](https://www.ultralytics.com/license)

Manipulation and interaction:
[SmolVLA](https://huggingface.co/docs/lerobot/smolvla) |
[LeRobot async inference](https://huggingface.co/docs/lerobot/async) |
[LeRobot 0.6.0](https://huggingface.co/blog/lerobot-release-v060) |
[integrating a robot](https://github.com/huggingface/lerobot/blob/main/docs/source/integrate_hardware.mdx) |
[LeKiwi](https://huggingface.co/docs/lerobot/lekiwi) |
[openpi](https://github.com/Physical-Intelligence/openpi) |
[MolmoAct2](https://arxiv.org/html/2605.02881v1) |
[vla.cpp on Orin](https://arxiv.org/html/2606.08094) |
[DemoGen](https://arxiv.org/abs/2502.16932) |
[GraspGen](https://github.com/NVlabs/GraspGen) |
[GraspGen-X](https://arxiv.org/abs/2606.00998) |
[FailBench](https://arxiv.org/abs/2609.03611) |
[MIGU](https://arxiv.org/abs/2609.24995) |
[LEGS-POMDP](https://arxiv.org/abs/2603.04705) |
[GIVE](https://arxiv.org/abs/2606.13435) |
[MediaPipe hands](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) |
[VPG](https://arxiv.org/abs/1803.09956)
