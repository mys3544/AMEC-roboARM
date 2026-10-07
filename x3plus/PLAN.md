# X3 Plus: the M3 Pro's abilities from one camera and a depth network -- the plan

Written 2026-10-06. The user asked for the X3 Plus to do everything the M3 Pro does, but with
the Depth Anything model on its wrist camera instead of a real 3D camera, and for every model to
run as a fast engine (TensorRT). It sits under the two-robot plan ([../PLAN.md](../PLAN.md) v2.3:
arena, voice, models, research) and fills in its X3 half. Nothing in it has been built yet,
except the folder move (X0). The user answered its four questions the same day (below).

Sources: our code on both robots (read 2026-10-06), the project notes, and Yahboom's X3 Plus
course (links at the end). Numbers marked *est.* are estimates, everything else was measured on
our robots or is quoted with its source.

## Questions -- answered 2026-10-06

1. **The mast camera: not used.** The user: it is unusable; use only the camera on the arm. (It
   is Yahboom's Astra Pro Plus, fixed level, blind to the arm's workspace.) Everything in this
   plan, the arena search included, comes from the wrist camera and Depth Anything.
2. **The pick runs on the robot**, behind the M3's `/pick/*` interface (X1). The panel stays for
   manual control, calibration and the debug views.
3. **Table first, then the arena mat.** The table has the ground truth (the 2 mm homography,
   tags, the arm's own placements); the printed mat follows once the numbers hold.
4. **Drop-off: the M3's box**, shared by both robots (black, inside 15 x 11 cm, ArUco
   DICT_4X4_50 id 10 inside the lid). With no box found, the X3 drops at its fixed `DROP_POSE`,
   as the M3 drops on its line. One box for two robots means they take turns at it (track B).

Assumed unless you say otherwise:
- The X3 gets its own copies of the M3's perception and planning pieces (in `x3plus/`), like
  `m3pro/` got its own copy of the X3's code. No shared package between the robots.
- The X3's bridge owns the robot's serial board (arm, wheels, IMU) and publishes `/odom` and
  takes `/cmd_vel` for ROS -- already agreed in PLAN.md M2.3. Yahboom's ROS driver must then
  not run on the X3 (its factory autostart is already disabled).
- About two hours of your time at the robot for the recorded test set (X2.2): placing the kit
  objects, checking a ruler now and then.

## In one page

- **The idea: make the wrist camera an RGB-D camera.** All of the M3's pick is geometry on
  metric depth: the floor plane, everything 6-100 mm above it, heights that name the kit pieces.
  The X3 gets the same input -- colour, depth in millimetres, the camera matrix, the pose --
  from a `/rgbd` endpoint, so the M3's code runs on it almost unchanged. The depth is built from
  three things the X3 has: the floor, known exactly from the arm's pose; Depth Anything's
  relative depth, anchored to that floor in every frame; and a second look from a known pose for
  heights that do not depend on the network's scale.
- **Why the anchoring should work now when it failed in September.** The global fit failed
  (30 mm cubes measured 70-120 mm) because the survey look points almost straight down: the
  whole floor is at nearly one distance, so scale and offset cannot be told apart. The M3 looks
  obliquely (its floor runs from 12 to 45 cm ahead). An oblique look spreads the floor over a
  wide range of distances, which pins the scale; a smooth correction term takes Depth Anything's
  low "hills"; the raised regions the edge method already finds are left out of the fit.
- **Measured before trusted.** A recorded test set with ground truth (tags, placements by the
  arm, known sizes) scores every depth source and model in one table: height error per kit
  piece, centre error, naming accuracy, latency. Target: heights within 5 mm (30 from 40 mm) on
  95 % of pieces, centres within 3 mm at the calibrated looks and 5 mm elsewhere.
- **Then the M3's abilities, one by one:** the pick as a service on the robot (X1), looks from
  any pose with a close look before the grasp (X3), the colour-edge turn for square grips, the
  two-finger and body collision model (X4), the box drop with search and put-back (X5), driving
  up to things out of reach (X6), the arena (X7), a 3D scan (X8).
- **Models.** Depth Anything V2 Small stays as a TensorRT FP16 engine and gets a smaller,
  faster input for close looks. YOLOE becomes an engine too (it still runs in PyTorch). Metric
  depth models (YOLO26-depth, MoGe-2, Depth Anything V2 metric, Depth Anything 3) compete on
  the test set, each exported the same way. One export tool, engines built on the robot in the
  container that serves them, one benchmark table (X9).
- **What the X3 keeps that the M3 lacks:** joint read-back, gripper feedback (`grasped()`), an
  arm that repeats to 1-4 mm (the M3's is about 10x worse), a 2 mm calibrated table view, the tag
  rung and the continuous ride scan. The plan builds on these rather than copying the M3's
  workarounds for not having them.

## Results so far (2026-10-06, stage 1 and the start of X2)

What was measured on the robot, and what it changed in the plan:

- **X1 done.** The panel runs on the robot and serves the M3's `/pick/*`; the console's arm card
  drives both robots. First live pick through it: 20.0 s, landed 2.3 mm from the aim.
- **The lens (X2.1).** fx 506.9 +- 1.4 px from 49 hand-held tilted views (`camera_model.py
  intrinsics`); the config's 416.5 came from down-looking views that cannot pin it.
- **The lens on the arm (X2.1).** `roboarm/lens.py`: mount 85 / 41 / +4 mm (the ruler: 85 / 39 / 0),
  turned ~6 deg. The arm's own pose is the limit: 8.5 px (~3 mm) with the joints read to 0.08 deg
  (`Arm.read_fine`); a gravity-sag term (J2/J3 give 2-3.5 deg at full lever) brings it to 6.0 px
  (7.6 on views left out of the fit; `lens.sagged`).
  So FK places the camera to ~1 deg / 2.5-3 mm -- the calibrated looks stay the precise path.
- **Depth anchored to the floor (X2.3) works for the TABLE, not for the PIECES.** On an angled look
  the scale is pinned (+-0.03 %) and the table comes out flat to +-2 mm; looking straight down it
  cannot be pinned (the September failure, explained). But Depth Anything V2 S reads the 30/40 mm
  pieces as ~15 mm.
- **Better networks (X2.5), same views, fitted to the table** (true 40 / 30 / 30 mm):

  | model | red 40 | yellow 30 | blue 30 | per picture (RTX 2070S) |
  |---|---|---|---|---|
  | Depth Anything V2 Small | 15 | 15 | 15 | 26 ms (robot, TensorRT) |
  | Depth Anything 3 Metric-Large | 14-16 | 6-12 | 12-15 | 214 ms |
  | MoGe-2 S / B / L | 15-35 | 9-38 | 16-47 | 0.5-1.2 s |
  | Depth Anything 3 Large, 3 views + our poses | 33-34 | 19-25 | 22-27 | 259 ms |
  | the same on 2x crops | 31-42 | 30, 30, 56 | 26-35 | ~1 s |
  | MoGe-3 Large (July 2026, fine detail) | 27-33 | 26-37 | 17-29 | ~2 min here (Triton kernels compiled on Windows) |

  Every network flattens small pieces in a wide picture; their own metric scale is useless at
  20-60 cm (heights 0.1-0.7 m off). Fitting to the table is always needed.
- **Heights from geometry (X2.4) work.** `roboarm/parallax.py`: two base yaws of the same look, the
  table's map measured from the board (taking out the arm's rotation error, 2.6-7 px), the height
  that makes the views agree. Real: red 40 -> 42, yellow 30 -> 27, blue 30 -> 30 mm (means over
  views), from the pieces' edges.
- **2026-10-07: the models on the robot, as engines (X9).** `~/roboarm/.venv-ml` (`tools/ml_env.sh`)
  runs PyTorch, DA3 and Ultralytics beside JetPack's TensorRT 10.7. DA3 Small / Base / Large, 3 views
  + poses at 280x504, FP16: **46 / 82 / 208 ms for the 3 views** (PyTorch FP16 on the Orin: 152 /
  145 / 289 ms per picture), within 0.12-0.4 % (median) of PyTorch after two export fixes in
  `tools/export_da3.py` (the attention scale; the camera encoder's q k v split). They must be fed
  the poses the way DA3's API normalises them (`da3_pose.normalised`); raw metres read every piece
  ~12 mm. Heights as before: Large 33 / 19 / 27.
- **What each model sees** (`out/depth_test/20261006_112651/what_each_model_sees.jpg`; the pieces'
  places re-measured from the survey picture -- the blue block lay at -63 deg, yellow at -30;
  sizes came back at 38.5x38.3, 29.9x29.1, 64.8x30.3 mm):

  | model (TensorRT FP16, Orin) | the pieces, 4 looks | |
  |---|---|---|
  | YOLO26n/s-seg (COCO), 28 / 33 ms | none, ever | a "dining table" |
  | YOLOE-26s prompt-free, 65 ms | all 3 from above (IoU 0.84-0.93), 2 of 3 in each angled look | named "envelope", "recreational vehicle", "extrude" at 0.26-0.83; ~100 other masks |
  | YOLOE-26s, text ("cube", "block", "wooden block", "toy block", "box"), 37 ms | 1 a look at most, mostly yellow | no false masks |
  | parallax, read as the 90th percentile over each piece's silhouette | 39/39/42, 28/30/29, 29/30/28 mm | all 9 within 2 mm |

  A second layout the same afternoon (`out/depth_test/20261007_130616`, pieces 4 cm nearer, turned):
  the same picture. Parallax red 40/40/42, blue 31/28/29 mm; the yellow cube sat below the angled
  looks' frame in two of the three, and parallax needs two views of a thing. DA3-Large red 35-36,
  blue 22-28; YOLOE prompt-free all 3 from above again ("gift", "extrude", "envelope").
  A third (`20261007_131444`, turned again, no hands): red 39/39/38 -- but the blue block, lying
  end-on with its near end below the frame, read 46/60/38: only the far edge of its plain top was
  in shot, against the board behind it. **Parallax needs a piece's outline in two views**; the
  pick's centring look gives that, and the kit's sizes reject a 60. The angled look sees the table
  from 150 mm out (the survey up to 262 mm): a test layout wants centres at 190-230 mm.

**So the X2 design changes:** a network finds and outlines things (and anchors the table); heights
come from parallax between looks the arm takes anyway (the base turns of the ride, the close look);
the kit's sizes snap the result. MoGe-3 flattens them too (its table is the flattest: 0.7-3 mm).

## The two robots side by side

| | M3 Pro | X3 Plus |
|---|---|---|
| camera for picking | Orbbec DaBai DCW2 RGB-D on arm4, 48 mm above the arm | Logitech C930e (mono, 848x480 MJPG) on arm_link4, 85 mm from J4, 39 mm above the axis; does not roll with J5 |
| depth | measured, 16-bit mm, from ~0.19 m (closer reads nothing) | none at the wrist; Depth Anything V2 Small: relative, any range (no near limit) |
| other cameras | none | Astra Pro Plus on the mast, 140 mm behind the arm base, fixed level: not used (user, 2026-10-06) |
| arm | 5 joints + gripper, no read-back, ~25-27 cm reach, J1 0-180 | 5 joints + gripper, read-back +-1-2 deg, grasp ring 90-253 mm (planner to 232), J1 10-180 |
| arm accuracy | position 1-5 mm, tilt 1.8 deg spread: needs a close re-look | 1-4 mm open loop; heavy camera makes J2-J4 stop up to 2 deg short (repeatable approach fixes it) |
| body near the arm | deck, two lidar heads at the front corners | mast behind (J2 stalls on it at 121 deg), base plate ~60 mm out, lidar on the deck |
| lidar | two T-mini Plus, 270 deg each, 14.5 cm up | one 360 deg lidar (Yahboom: YDLIDAR 4ROS; height unmeasured) |
| base | mecanum, driven over ROS (`/cmd_vel`, `/odom_raw`) by the micro-ROS board | mecanum, same serial board as the arm (`set_car_motion`), not used by our code yet |
| software | ROS 2 Humble in the desktop session; our m3bridge (rclpy) | ROS 2 Humble installed, unused; our bridge on Rosmaster_Lib |

Yahboom's own X3 Plus software (their course) is ROS 1 in Docker: MoveIt, MediaPipe gesture
control, colour-block transport between points marked in rviz, voice-driven grips, garbage
sorting with YOLOv11, line following that clears obstacles off the line. Their arm demos grip a
block handed into the gripper or recognised in front of the camera; none measures depth at the
gripper. Our robot runs ROS 2 Humble with Yahboom's ROS 2 workspace instead
(`~/yahboomcar_ros2_ws`). Their YOLOv11 TensorRT recipe is the plain
`yolo export format=engine` that `tools/export_engine.py` already does.

## What the M3 does that the X3 does not -- the gap list

| ability | M3 Pro today | X3 Plus today | work |
|---|---|---|---|
| the job runs on the robot | `Picker` in the bridge, `/pick/start|cleanup|home|stop|settings`, `/pick/state|log|frame.jpg`, `/api/metrics` | `Session` in the panel on the laptop; the bridge only moves the arm and forwards vision | X1 |
| console arm card | buttons, speed slider + 13 settings, Timing table, narration, live view + last look, battery | Pick one / Clear the table / Home through the panel | X1 |
| 3D from the camera | floor plane (RANSAC), `level()` corrects the camera's tilt and height, objects 6-100 mm above the floor, kit named by height | one homography per calibrated look, heights from the cube assumption, Depth Anything as edges only | X2, X3 |
| look from any pose | close look before every grasp, look-again within +-15/30 deg | three calibrated looks x any J1 yaw; elsewhere positions are flagged meaningless | X2, X3 |
| blocks and heights | 30/40 mm cubes, 30x30x60 lying or standing, by measured height | top-face rectangles since 09-24; height from the short side | X3 |
| object turn | colour edges on the top face: 0-2 deg repeatability | `minAreaRect` of the outline | X3 |
| grip choice | wrist turns square-first, both fingers modelled, careful grip by the bumper, every glide checked with the real wrist and opening | `roll_for` +-45 from rest, reach-correction passes | X4 |
| collision model | chassis steps + mesh shell + lidar room + floor clearance + the held object | the mast (planar check) and the table only; no base plate, no camera-vs-object check | X4 |
| drop-off | ArUco box: search over 5 J1 looks, re-check before each drop, drops spread inside, put-back if lost | fixed `DROP_POSE` (J1 180) + random spread | X5 |
| driving | nudge forward/sideways (<= 10 cm a move), slide right for clearance, drive back before the drop | none | X6 |
| arena | slam_toolbox map `arena_v4`, drive page, auto loop, survey planner, fence fit | none | X7 |
| 3D scan | RTAB-Map with the DCW2 (parked: quality not good enough yet) | none | X8 |
| record of every look | colour + depth + pose + K per look in `out/pick` | sweep pictures only, overwritten per station | X2.2 |
| speed | 11-15 s a pick, wide look to wide look | 21.4 s per attempt (clear the table), 17-20 s pick-drop-return | X1, X3 |

## X0 -- The folder (done 2026-10-06)

The X3's code moved from the repo root into `x3plus/`, as `m3pro/` has its own: `roboarm/`,
`tools/`, `tests/`, `vision_service/`, `docker/`, `cameras/`, `docs/`, `compose.yaml`,
`pyproject.toml`, `uv.lock`, `.dockerignore`, its README, and the local `data/`, `out/`,
`vendor/`. `x3plus/` is its own uv project; on the robot nothing changes (`sync_robot.py`
pushes the same 63 files to the same `~/roboarm` paths). Updated: the console's pretend X3 and
notes, `.claude/launch.json`, the M3's cross-references, PLAN.md's code-layout line. The
console's `voice` dependency group stays in `x3plus/pyproject.toml`, where it already was.
Checked: 321 X3 tests, 104 console tests, the M3 suite, ruff, the simulator from the new path.

## X1 -- The pick runs on the robot, behind the M3's interface

1. Run the panel's `web` service on the robot instead of `bridge` + laptop panel (it owns the arm
   and camera itself; detection talks to the vision container on localhost). Measure per-pick
   time against the 21.4 s baseline: the WiFi hop goes away from every detection (*est.* 150-200
   ms a look).
2. Add the M3's routes on top of the `Session` jobs: `/pick/start {kind, at, drive}` -> the
   one-click pick, `/pick/cleanup` -> clear the table, `/pick/home`, `/pick/stop`,
   `/pick/settings` (speeds and pauses as one editable list, saved to `out/pick/settings.json`),
   `/pick/state` in the M3's shape (job, step, found, picked, missed, battery, box), `/pick/log`,
   `/pick/frame.jpg`. The console's arm card then works for the X3 with a new entry in
   `robots.json` (kind `x3arm`), and voice routes the same way.
3. Save every look like the M3 does (colour, read-back pose, K, later the depth) -- the raw
   material for X2, track D and study S1.

Exit: the console picks with the X3 from the same card as the M3's; per-pick time measured.

## X2 -- Depth from one camera (the heart of the plan)

**X2.1 A proper camera model.** Everything below needs the camera as a 3D instrument, not a
homography per look:
- Intrinsics of the C930e at 848x480 from a ChArUco board held at many TILTS (from down-looking
  views alone the focal length came out anywhere from 416 to 501: ill-conditioned).
- The lens on arm_link4 refined by PnP over many poses, using READ-BACK joints (the M3 cannot do
  this; its hand-eye fit was degenerate). Checks: the J1-axis circle must land on (0,0); the
  three calibrated looks' homographies must agree with the model to ~2 mm; a touch test.
- Output: for any pose, the camera's position and orientation in the base frame, and for any
  floor pixel its metric depth (ray meets the plane z = 0).

**X2.2 A recorded test set with ground truth** (`tools/record_rgbd.py`): per frame the colour
picture, the read-back pose, K and the look's name. Scenes: the kit (30 and 40 mm cubes, blocks
lying and standing, tagged and plain, white and black) alone and in clusters, on the white table
and on the arena mat. Looks: the survey (straight down), a new oblique WIDE look (the M3's kind:
floor ~12-45 cm ahead), side looks, close looks 10-15 cm off, frames from the ride. Ground truth:
the 2 mm homography at the calibrated looks, tag poses, the arm's own placements (it puts a cube
down at a commanded point to about 2 mm), and the known sizes.

**X2.3 Depth Anything anchored to the floor.** Depth Anything gives inverse depth up to an
unknown scale and offset: `d = a/Z + b`. On floor pixels Z is known (X2.1), so fit `a`, `b` and a
smooth low-order warp `w(u,v)` robustly (Huber/RANSAC) on the floor, leaving out the raised
regions the edge method finds; then `Z = a / (d - b - w)` everywhere. The fit is only as good
as the spread of floor distances in the picture:

| look | floor distances in the picture | 1/Z spread | scale |
|---|---|---|---|
| survey (tool pitch ~175, straight down) | lens height to ~1.2x it | ~20 %, and symmetric | indistinguishable from the offset: the September failure |
| oblique WIDE look (as the M3's) | ~15 to ~45 cm | ~3x | well determined (*est.*: to test first) |

So the first experiment is cheap: one oblique look, the existing engine, the floor fit, heights
of 30/40/60 mm pieces against truth.

**X2.4 Heights from a second look.** The close look the M3 takes before every grasp is also a
stereo pair with the wide look, and the X3 knows both poses from read-back. For a candidate
height h, the plane z = h maps one view into the other exactly; sweep h over 0-80 mm in 1 mm
steps and keep the h at which the two views' top-face outlines (Depth Anything's masks, which are
crisp even on plain tops) overlap best. This gives metric height independent of the network's
scale and checks X2.3 on every pick.

**X2.5 Metric models, scored on the same set** (all TensorRT, X9): YOLO26n/s-depth (Ultralytics,
metric, AGPL-3.0, trained at 768 px), MoGe-2 ViT-S (35 M parameters, metric point map), Depth
Anything V2 Metric-Indoor-Small, Depth Anything 3 Small (80 M, Apache-2.0, multi-view and can take
known camera poses). A metric model replaces the anchored scale only where it wins on the table
of X2.2. They were trained on rooms and streets, not on a cube 20 cm away, so the anchor is
expected to stay; a fine-tune can change that:
- **The M3 teaches the X3 to see depth.** The M3 records RGB-D frames of the same kit on the
  same mat at every look already (`out/pick`: colour + 16-bit depth + pose). Those, plus the
  X3's own frames labelled by geometry (exact floor + the catalogue's boxes where picks
  confirmed them), fine-tune YOLO26-depth or Depth Anything V2 Small on the 5090. The cameras
  differ (DCW2 colour vs C930e): train at one canonical focal length. This is study S1 of
  PLAN.md turned into a method, and a paper-sized result if it closes the gap.

**X2.6 `/rgbd` on the X3.** Colour JPEG, 16-bit depth PNG in mm, K, the read-back pose and a
confidence mask (floor from geometry = sure; anchored network depth = scored; no depth = 0), in
the M3's exact format, so tools and tests can be shared.

Exit: the X2.2 table with heights within 5 mm on 95 % of kit pieces at the oblique and close
looks, and `/rgbd` serving it on the robot within the time of one look.

## X3 -- The M3's perception on the X3

Port, from `m3pro/roboarm/m3vision.py` and `m3pick.py`:
- `objects()`: points 6-100 mm above the floor, 4 mm cells, footprint rectangle grown on the
  camera's side when cut, height as the 95th percentile, `kit_item()` by height within 7 mm.
- whole vs cut, the merging of looks (3 cm, 6 cm for cut things), nearest-first choice.
- `colour_turn` / `refine_turns`: pure colour, needs only the outline and the camera -- ports
  as is and replaces `minAreaRect` for the turn.
- the WIDE look (oblique, sees past the reach for drive-up targets, up to ~45 cm) next to the
  existing calibrated looks, which stay as the precise path where they apply.
- the close look before the grasp and look-again when cut, from read-back.
- `level()` does not port: on the M3 the measured floor corrects the arm's tilt, but the X3's
  floor comes FROM the arm's pose. The X3 instead relies on read-back + repeatable approach
  (1-2 mm), the calibrated looks, and on the arena the printed mat itself (known artwork: its
  lines give the camera's pose by PnP where they are in view).

Exit: from the wide look and one close look, kit pieces found, named and turned anywhere in reach;
clear-the-table on the table with blocks standing and lying.

## X4 -- Grasp planning and the body

- **The X3's body model**, measured by ruler and photos (the M3's lesson: the arm's own camera
  cannot scan the body): the mast, the base plate (~60 mm out), the lidar, the wheels and
  battery, the front edge. Boxes, kept apart from the arm's own column.
- **The arm's shell:** capsules round the measured links (the X3's URDF is wrong in its lengths,
  so its meshes are not used as they are), the C930e as a box on arm_link4 -- checked against
  the body AND against tall objects (unchecked today: `grasp.py` notes it).
- From the M3: both fingers in the check when the gripper is open, wrist choices square-first
  then off-square slack, the careful grip near the body, every glide checked as it will be driven
  with its wrist and opening, the held object kept clear on the way to the drop.
- Keep the X3's own strengths: the reach-correction passes (fewer: X2.4 gives height up front),
  `grasped()` as the "picked" check, the raw-pulse path for J2 below 0 at the far rim.

Exit: no planned path that the model calls clear touches anything in a 50-pick run; picks close
to the robot's front that today are refused.

## X5 -- The box, and the rest of the job

The M3's `box_look` / `box_search` / `see_box` / `box_way` / `put_back`, on the M3's own box
(marker id 10, 64 mm; inside 15 x 11 cm, walls 3.8 cm): find it at job start from side looks over
J1 (the X3 covers J1 10-180: front and right, so the box sits in front of or right of the arm),
re-check it from where it was found before each drop, spread drops inside it away from what is
already there, carry at a set height, and put the object back where it was picked if the box is
gone. The marker needs no depth (PnP on its known side). Fallback without a box: the fixed
`DROP_POSE`. Shared with the M3: when both robots work, one drops at a time (track B decides who).

Exit: 10 picks into the box in a row, with the box moved once mid-job.

## X6 -- The wheels

- The bridge drives the base over the same serial board (`set_car_motion`, encoder speeds, the
  IMU) under the one serial lock the arm already uses. Check first: report rates and how the arm
  and base share the 115200-baud line; a powered USB hub (the CH340 dropped off USB once next to
  the C930e).
- The M3's moves: nudge forward, sideways only if needed, at most 10 cm a move and 5 cm/s, keep a
  4 cm gap, slide sideways for clearance, drive back to the job's start before every drop
  (the box does not move with the robot).
- Then a small rclpy node in the bridge publishes `/odom` + TF and takes `/cmd_vel`, so
  slam_toolbox, AMCL and Nav2 can run (PLAN.md M2.3).

Exit: a piece 35 cm out is driven up to and picked; odometry checked against a tape measure.

## X7 -- The arena

- The X3 localizes on the M3's arena map (`m3pro/arena/maps/arena_v4`): same arena, one common
  frame for track B. AMCL vs slam_toolbox localization, as study L plans.
- `drive_core.py`, `fence_fit.py` and the drive page are plain Python: copy them with the X3's
  footprint, stop distances and camera constants. Check first that the lidar plane is under the
  30 cm fence and what the mast and arm hide from it.
- Search from the wrist camera only: the arm raised into far-looking poses (the M3 tilts its
  camera the same way) for the coarse pass, then wide looks from staging poses on the printed
  mat. How far a 30 mm cube is still found is measured, not assumed (PLAN.md M2.5's detection
  model): it sets how many staging poses the search needs.
- Exit as PLAN.md M2.6: from a known start, find three kit pieces placed at random and deliver
  them to a named place.

## X8 -- 3D from the X3

- **Object scan:** the arm circles an object at known poses (read-back); anchored depth per view
  fused into a TSDF on the robot (Open3D); a coloured cloud in the console's 3D view.
- **Feed-forward 3D on the 5090:** Depth Anything 3 or MapAnything with the known poses as input
  -- study S4 of PLAN.md, which names the X3's monocular camera as the case where they may help.
- **Arena scan** from the wrist camera while driving: the anchored depth and the poses from
  odometry into RTAB-Map as RGB-D, or the frames to the 5090 for a monocular reconstruction --
  after the object scan works.

## X9 -- Models as fast engines

| model | job | today | target | expected on the Orin NX |
|---|---|---|---|---|
| Depth Anything V2 Small | relative depth: edges, then anchored metric | TensorRT FP16, 518x910, 26 ms (23.6 ms trtexec at 518x686); called over the WiFi | the same engine on the robot in the pick's loop; a second 364x644 engine for close looks; CUDA graph | 26 ms; ~12-15 ms at 364x644 (*est.*: half the patches) |
| YOLOE-26s-seg-pf | open-set masks, names | PyTorch `.pt`, 41 ms (244 ms mean per request during a job) | TensorRT FP16; the kit's prompts baked in for track A | 9-12 ms (*est.*, PLAN.md M5) |
| YOLO26n/s-seg on our kit | the main detector (PLAN.md M5 #1) | -- | INT8 by quantization-aware training | 3.5-4.8 ms (others' numbers) |
| YOLO26n/s-depth | metric depth candidate -- **measured 2026-10-07, zero-shot useless here**: all five sizes on the 9 angled looks put the table at 0.39-0.47 m and read the pieces -12..+20 mm when anchored (table +-5-8 mm, worse than V2 S); only a fine-tune on our own frames could change that | PyTorch FP16 on the Orin: 40-77 ms (n..x) at 768 | TensorRT FP16 if a fine-tune wins | n 2.7 ms on a T4 (Ultralytics) |
| MoGe-2 ViT-S | metric point map candidate | -- | ONNX -> TensorRT FP16 (a ROS 2 TensorRT node exists to learn from) | measure |
| Depth Anything V2 Metric-Indoor-Small | metric candidate | -- | ONNX -> TensorRT FP16, as V2 Small | ~26 ms |
| Depth Anything 3 Small | multi-view depth with known poses | -- | on the 5090 first; an engine only if it earns a place | measure |
| Parakeet TDT 0.6B v3 (console) | voice | int8 ONNX on the laptop CPU | stays | 0.3-0.45 s |

Rules (from PLAN.md M5 and what we learnt on the X3):
- One tool: `tools/export_engine.py` grows an ONNX path (`torch.onnx.export`, opset 17,
  `dynamo=False`, then `trtexec --fp16` with a timing cache) next to its Ultralytics path, and
  loads YOLOE with the YOLOE class (today it uses `YOLO`, so YOLOE was never exported).
- Build on the robot with the host's TensorRT 10.7, which the vision service runs on since
  2026-10-07 (uv venvs + systemd, robot/; the containers and their 10.4 are retired; engines do
  not move between TensorRT versions). Static shapes. Engines are not in git; the ONNX
  files are in `models/`, and setup builds the engines.
- Every export is checked against its PyTorch model on 20 test-set frames (Depth Anything's
  engine: correlation 0.99999) and timed engine-only and end to end, p50/p99, memory, energy
  (tegrastats). One table, re-run on every change.
- FP16 for the vision transformers (Depth Anything, MoGe, DA3): INT8 made a ViT-S 2.7x SLOWER on
  an Orin with TensorRT 10.3. INT8 for the CNNs (YOLO). The DLA only for CNN throughput.

## Order, with a demo at the end of each stage

| stage | work | demo |
|---|---|---|
| 1 | X1 (pick on the robot, the M3's interface) + X2.1 camera model + X2.2 test set | the X3 picking from the console's arm card; the time per pick without the WiFi hop |
| 2 | X2.3-X2.6 + X9's first rows | the depth table: heights of every kit piece from one oblique look; engines and timings |
| 3 | X3 + X4 | blocks standing and lying, picked from any look, close to the robot's front |
| 4 | X5 + X6 | into the box; a piece driven up to and picked |
| 5 | X7 + X8 | "find and pick three" in the arena; an object scan in the console |

Stages 1-2 are about two weeks of work (*est.*), the rest about a week each; track D's recorder
(RESEARCH.md) can start in stage 1, since X1.3 saves every look anyway.

## Risks

- **Depth Anything on the printed mat.** Painted lines, arrows and zebra crossings are flat, but
  the network may still draw edges there; the test set covers the mat for this reason.
- **The floor anchor depends on the pose.** Read-back is good to 1-2 deg; at 20 cm, 1 deg is 3.5
  mm. The calibrated looks stay the precise path, X2.4 cross-checks every pick.
- **A heavier duty cycle for the arm.** More looks mean more servo heat and battery with the
  heavy C930e; time and battery are logged per job.
- **One serial line for arm, wheels and IMU** at 115200 baud: measure before relying on it.
- **Licences.** YOLO26/YOLOE are AGPL-3.0 (fine for research, matters for a public release);
  Depth Anything V2 Small and Depth Anything 3 Small/Base are Apache-2.0, DA3 Large/Giant
  non-commercial; MoGe's code is MIT (check the weights' terms).

## To check at the start (no need to ask)

- the lidar's model (Yahboom says YDLIDAR 4ROS; the robot's workspace has the Slamtec driver)
  and its height against the 30 cm fence;
- the base: `set_car_motion` and the encoder/IMU report rates over the shared serial line;
- the C930e's view at the oblique WIDE look: where the fingers and base plate appear in it;
- Yahboom's ROS 2 X3 Plus driver in `~/yahboomcar_ros2_ws`, to be sure nothing starts it.

## Sources

Yahboom's ROSMASTER X3 Plus course, https://www.yahboom.net/study/ROSMASTER-X3-PLUS (chapters load
through `https://www.yahboom.net/build?id=<id>`, as for the M3's course):
- [How to use the Astra camera](https://www.yahboom.net/public/upload/upload-html/1743490117/1.%20How%20to%20use%20the%20Astra%20camera.html) (8303)
- [RTAB-Map 3D mapping navigation](https://www.yahboom.net/public/upload/upload-html/1743490314/7.%20RTAB-Map%203D%20mapping%20navigation.html) (8309)
- [Lidar basic, 4ROS](https://www.yahboom.net/public/upload/upload-html/1743490481/Lidar%20basic-4ROS-EAI.html) (8312)
- [Robot URDF model](https://www.yahboom.net/public/upload/upload-html/1743490080/8.%20Robot%20URDF%20model.html) (8302)
- [Recognizes color block and handling](https://www.yahboom.net/public/upload/upload-html/1743493491/Recognizes%20color%20block%20and%20handling.html) (8359)
- [Voice control robotic arm grip objects](https://www.yahboom.net/public/upload/upload-html/1743493628/4.Voice%20control%20robotic%20arm%20grip%20objects.html) (8364)
- [Voice control robotic arm garbage sorting](https://www.yahboom.net/public/upload/upload-html/1743493647/5.%20Voice%20control%20robotic%20arm%20garbage%20sorting.html) (8365)
- [YOLOv11 model conversion + TensorRT](https://www.yahboom.net/public/upload/upload-html/1743491439/YOLOv11Model%20conversion%20%2B%20tensorrt%20accele.html) (14320)
- [Introduction of voice module](https://www.yahboom.net/public/upload/upload-html/1762850854/1.%20Introduction%20of%20voice%20module.html) (16151, CI1302 offline command words)

Models and parts:
- [Ultralytics depth estimation (YOLO26-depth)](https://docs.ultralytics.com/tasks/depth) (metric, 768 px, trains on RGB + depth PNG in mm)
- [MoGe / MoGe-2](https://github.com/microsoft/moge) and [ros2-moge-trt](https://github.com/ika-rwth-aachen/ros2-moge-trt)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (model sizes and licences per its README)
