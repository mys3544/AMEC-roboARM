# M3 Pro: detection with the depth camera, TensorRT, 3D maps -- the plan

**Superseded by [../PLAN.md](../PLAN.md) (v2, 2026-09-29): two robots, the arena, voice, the 3D scan
and the research tracks.** This v1 stays as the M3's bring-up and depth-detector detail (v2's M2.6),
and its Sources list the Yahboom course chapters. One correction from v2's research: Orbbec's own
sheet gives the DCW2's depth at 15 fps, not 30.

Written 2026-09-29, before any of it is built. It answers four questions: can the M3's depth
camera make detection better, faster and more reliable; what is in Yahboom's M3 Pro course
(the transformer models, the 3D map); can a 3D map serve localization, navigation or picking;
and which of our models should become TensorRT engines.

Nothing here has run on the M3 yet. Facts marked *Yahboom* come from Yahboom's course pages
and the code they list (links at the end); step 0.1 checks them on the robot before anything
depends on them.

## In one page

**Better.** On the X3 Plus every hard problem came from reading heights off a flat picture: a
table homography per look, tag magnification, far-corner cube ranging, three calibrated looks,
and a relative-depth network used only for its edges. The M3's Orbbec DaBai DCW2 sits on the
arm and *measures* depth (640x400 at 15 fps per Orbbec, usable from about 0.2 m). So detection becomes
geometry first -- the table plane, then everything standing on it, in millimetres -- and neural
networks only name things and cover what depth cannot see: flat things, black or shiny tops,
and the last 15-20 cm of a descent.

**Faster.** That geometry is a plane fit and a connected-components pass over one depth image,
on the order of 10 ms on the CPU, with no network in the critical path. The rest is plumbing.
On the X3 the neural model took 41 ms on its own, but a request averaged 244 ms during a job
(p90 382, the live view queueing on the same GPU), and a laptop -> bridge -> vision round trip
was 180-250 ms over WiFi. So perception moves onto the robot, next to the camera. The DaBai sees
about 40 x 30 cm from 30 cm up (the X3's C930e saw 30 x 17 cm from its survey pose), so one to
three looks should replace the 7- to 21-station ring. Whatever networks remain become TensorRT
engines built on the M3.

**More reliable.** Before anything is tuned, a recorded RGB-D test set with measured ground
truth, so every change is scored offline -- objects found, ghosts, millimetres, milliseconds --
before it touches the robot. The X3's lesson stands: judge a pick by where the fingers land.

**3D maps, two scales.** For picking, a height map of the table fused over a few looks (W1).
For moving between work areas, Yahboom's 2D lidar map (slam_toolbox, Nav2, AprilTag
relocalization) first, then a 3D obstacle layer from the camera (RTAB-Map, or NVIDIA's nvblox)
for what lidars at wheel height cannot see. Navigation comes after picking works: the camera
is on the arm and cannot look ahead and down at once.

**Transformers.** Yahboom's "Transformer Model Tracking" is a single-object tracker (an ONNX
model) plus a point-cloud cut that gives the tracked object's 3D centre; their language
features call cloud LLMs. The transformers worth having here are open-vocabulary detectors (to
name objects by text) and big models used OFFLINE as teachers that label our frames, so a
small fast model can be trained on the lab's objects and served as an engine.

**TensorRT.** Build every engine on the M3, in the container that serves it (an engine is
welded to its TensorRT build and GPU). FP16 by default; INT8 only for CNNs and only when the
test set shows it costs nothing; the DLA cores cannot run attention layers.

## What the M3 Pro gives us

| | | from |
|---|---|---|
| Camera | Orbbec DaBai DCW2, binocular structured light: depth up to 640x400 at 15 fps (Orbbec's sheet; Yahboom says 30), 0.2-3 m (5 m in high-power mode), error < 1 % at 1 m, depth FOV 91 x 62 deg, depth-to-colour (D2C) alignment in hardware, USB 2.0 | Orbbec, Yahboom |
| Where | On the arm, looking along the tool (eye-in-hand). Yahboom's code puts it 100 mm behind the tool frame and 48 mm above it (`EndToCamMat`); colour intrinsics fx = fy = 477.6, cx 319.4, cy 238.6 at 640x480, so the colour view is about 68 x 53 deg | Yahboom code; mounting to confirm |
| Driver | `ros2 launch orbbec_camera dabai_dcw2.launch.py` -> `/camera/color/image_raw`, `/camera/depth/image_raw` (16-bit mm), `/camera/color/camera_info`, `/camera/depth/points`, `/camera/depth_registered/points`. The depth sensor is an OpenNI-protocol device (no `/dev/video` node, 2026-09-28); Orbbec's Python SDK v2 only takes it after a firmware upgrade to UVC, so we use the ROS 2 driver already on the robot | Yahboom, Orbbec |
| Arm | Bus servos on the micro-ROS board. Commands are ROS 2 topics: `/arm6_joints` (`arm_msgs/ArmJoints`: joint1..joint6 in degrees + `time` in ms) and `/arm_joint` for one servo. Torque-free at power-up until the first command. The course shows no servo read-back topic: its demos remember what they commanded | Yahboom; read-back to check |
| Kinematics | `ros2 run arm_kin kin_srv` -> service `/get_kinemarics` (`kin_name` "fk" or "ik"), KDL on the `yahboom_M3Pro_description` URDF. All joints at 90 = arm straight up, tool 552 mm above `base_link`. Offsets in `arm_kin/param/offset_value.yaml` | Yahboom |
| Gripper | Servo 6 stalls and can burn if told to close past the object. Yahboom's table, object width -> servo angle: 0 cm 180, 2 cm 152, 3 cm 134, 4 cm 115, 5 cm 95, 6 cm 57 | Yahboom |
| Base | Mecanum: `/cmd_vel`, `/odom`, IMU. Two T-mini Plus lidars (`/scan0` left rear, `/scan1` right front, 0.05-12 m) merged into `/scan_multi`, the chassis filtered out -> `/scan` | Yahboom |
| Their stack | `~/yahboomcar_ws` (M3Pro_demo, arm_kin, M3Pro_KCF, yahboom_yolov8, the Orbbec driver) and `~/M3Pro_ws` (m3pro_bringup, M3Pro_navigation, slam_mapping, apriltag_localization, smart_tracker, multi_brains); models in `~/MODELS` | Yahboom |

**How Yahboom's grasping works, and why we do not copy it.** It reads depth at five pixels
around a box centre, maps that point through the nominal `EndToCamMat` and one x/y/z offset
fitted on a single 40 mm block, asks `/get_kinemarics` for the joints, and drives the base
until the object is 240-260 mm from `base_link`. It is the right place to learn the frames and
topics. It is not the precision we reached on the X3 (fingertips 1-4 mm from the aim), which
came from measuring each error separately: the camera, the reach, the tool growing as it closes.

**What our copy assumes that is no longer true.** [roboarm/arm.py](roboarm/arm.py) talks
Rosmaster_Lib on `/dev/ttyUSB0`; on the M3 that port is the micro-ROS board, held by its agent,
and must never be opened. [roboarm/workspace.py](roboarm/workspace.py) and the three calibrated
looks assume a monocular camera and a table homography. [roboarm/kinematics.py](roboarm/kinematics.py)
holds the X3's link lengths. [vision_service/depth.py](vision_service/depth.py) guesses depth
from one RGB frame.

## W0 -- Bring-up: our software on the M3 (first; everything depends on it)

Goal: the panel drives the M3's arm, shows its colour and depth, and a point clicked in the
depth image is touched by the fingertip.

**0.1 Survey, read-only, about an hour.** With the agent and the camera running, record:
- the camera's topics, rates, encodings and sizes (`ros2 topic hz`, `ros2 topic echo --once
  /camera/color/camera_info` and the depth one); whether D2C is on (`ros2 param list
  /camera/camera`, look for align/registration) and whether depth arrives at 640x400 or aligned
  at 640x480 (Yahboom's code resizes it, which suggests the former);
- the arm's topics and types (`ros2 topic list -t`, `ros2 interface show arm_msgs/msg/ArmJoints`):
  is there any servo angle, gripper state or battery voltage coming back?
- where the DaBai physically sits, and what it sees at Yahboom's "recognition" pose
  [90, 150, 12, 20, 90, 0] and "mapping" pose [90, 180, 5, 0, 90, 0];
- the URDF's joint origins, then a ruler on the real arm (the X3's URDF had one link 2x too long);
- the ML stack: `dpkg -l | grep -E "nvinfer|cudnn"`, `nvpmodel -q`, the torch / tensorrt /
  ultralytics versions, `docker images`, the CDI spec, free disk;
- what the joystick autostart publishes (it must not fight our commands);
- `~/MODELS/tracker/` (the transformer tracker's ONNX).

**0.2 `m3bridge`: ROS 2 at the edge, our code unchanged behind it.** A small rclpy process on
the host (system Python 3.10 with `/opt/ros/humble` and Yahboom's workspaces sourced, so
`arm_msgs` imports) that serves the contract of [roboarm/web/bridge.py](roboarm/web/bridge.py):
`/arm/call` with the same `ALLOWED` methods, `/arm/interrupt`, `/stream.mjpg` -- plus
`GET /rgbd`: colour, aligned depth (16-bit PNG), intrinsics and the joint angles at capture.
Our core stays ROS-free and the panel keeps working.

**0.3 `M3Arm` behind `/arm/call`.** `move_to` -> one `ArmJoints` with `time` = the largest
joint change / speed; `glide` = one long `time`; `interrupt` = re-send the pose reached by then
(interpolated in time);
the gripper only through the width table, never past it. Without read-back: `read()` returns
the commanded pose and says so, `grasped()` becomes "the object is gone from the table" (a
re-look), `battery()` reads whatever the board publishes. A simulated `M3Arm` for the tests.
If the camera loop of 1.5 turns out not to be enough, servo read-back could be added to the
micro-ROS firmware (Yahboom's control-board course shows how to build and flash it) -- a
last resort, because a bad flash stops the robot.

**0.4 Kinematics.** The M3 arm's measured geometry into [roboarm/kinematics.py](roboarm/kinematics.py)
(if it is still base yaw + three pitches + wrist roll, the closed-form IK carries over); our FK
checked against `/get_kinemarics` fk on a dozen poses; new `SAFE_LIMITS`, with the chassis, the
lidars and the camera cable as the things to keep clear of.

**0.5 Camera.** D2C on, checked on a cube's rim (colour edge and depth edge within 1-2 px);
intrinsics from `camera_info`; the driver's depth filters and exposure settled.

**0.6 Hand-eye, measured.** The lab's A3 ChArUco board from 15-25 arm poses ->
`cv2.calibrateHandEye` (eye-in-hand) -> the tool-to-camera transform, compared with Yahboom's
nominal one. The same views can also fit each joint's zero (the board pose FK predicts against
the one PnP measures) -- worth doing if the touch test misses. Then the touch test
([tools/touch_probe.py](tools/touch_probe.py), re-aimed): click a point in the depth image, drive
the closed fingertip there, measure the miss at three places across the workspace.

Exit: touch test within 5 mm at all three places; the `m3pro` tests green with the simulated arm.

## W1 -- Detection from depth: better and more reliable

**1.1 Record the test set first** (a `tools/record_rgbd.py`): colour, aligned depth,
intrinsics and the joints at capture, into `data/eval/<scene>/`. About 25 scenes from the survey
pose: each lab cube alone (20/30/40/55 mm; red, blue, yellow, wooden, white, black, tagged), the
30 x 30 x 60 block flat and on end, two cubes touching, a cube at the frame edge, the drop tray,
the empty table, flat things that must NOT become objects (the face-up ChArUco board, paper, a
tag lying flat), and one shiny or transparent thing as a stress test. Ground truth: positions
from a printed grid or the touch probe, sizes by calliper.

**1.2 The geometry detector** (`roboarm/rgbd.py`, numpy + OpenCV, testable offline like the rest):
- depth -> a height image: back-project with the intrinsics, move to the base frame (FK at
  capture x hand-eye), fit the table plane (RANSAC, seeded with the known table height), keep
  each pixel's height above the plane;
- objects = connected regions higher than about 6-8 mm (at least 5 sigma of the plane's
  residual) and big enough, found in image space (fast);
- per object: the top face (pixels within ~3 mm of the region's top), its minimum-area
  rectangle in base x/y -> centre, heading, width, length; height measured, not inferred;
  `clipped` when it touches the frame or a depth hole;
- out as the `Target` of [roboarm/detect.py](roboarm/detect.py) (`x`, `y`, `width_m`,
  `length_m`, `angle_deg`, `height_m`, `clipped`), so `grasp.plan()`, the roll choice and the
  Session keep working.

**1.3 Where depth is blind, the RGB rungs stay -- gated.** Black, shiny or transparent tops
leave holes: a colour or YOLOE silhouette there is a *candidate*, promoted only when a depth
step borders it or a second look confirms it. Things under ~6 mm (tags, paper) come only from
RGB, are marked flat and are never grasped. Labels (a colour name, YOLOE's class, a tag id)
attach to depth objects by overlap. The X3's depth gate, made the default.

**1.4 A height map instead of a ring.** A 2 mm grid over the workspace in base coordinates;
each look writes what it saw and clears what it now sees empty, so a picked object vanishes.
The object list comes from the map, so two looks agree by construction. One to three looks per
table instead of 7-21 stations.

**1.5 Pick with it.** Survey look -> choose -> move the camera 25-30 cm above the target (depth
still valid) -> measure again: this is the closed loop, because the error is measured relative
to a camera bolted to the gripper, which covers what joint read-back could not -> descend open
loop for the last 15-20 cm where depth goes blind -> close to the width table's angle for the
measured width -> lift -> re-look: the object must be gone from the table.

**1.6 Score it.** Offline on the test set, then CLEAR THE TABLE on the robot with
[roboarm/metrics.py](roboarm/metrics.py).

Exit (proposed, to adjust after W0): offline, at least 98 % of graspable objects found, zero
ghosts, centre and size within 3 mm, at most 30 ms a frame on the Orin; live, 19 of 20 held and
under 15 s an attempt (the X3: 21.4 s).

What depth gets wrong, and the answer to each:

| failure | what it looks like | answer |
|---|---|---|
| closer than 0.15-0.2 m | zeros | measure at 25-30 cm, descend blind (1.5) |
| black, shiny, transparent tops | holes, spikes | RGB candidates confirmed by a second look (1.3); a depth-completion model later (W3) |
| flying pixels on rims | a halo of wrong heights | the top face from the region's interior; erode the rim |
| D2C off or misaligned | size and centre biased one way | check in 0.5; software registration from the extrinsics if needed |
| flat things (< 6 mm) | not separable from the table | RGB only, never grasped |
| sunlight, another projector | noisy patches | median over 2-3 frames; not facing a window |

## W2 -- Faster: TensorRT and the pipeline around it

**2.1 Measure first.** Per stage with roboarm.metrics, per model with `trtexec`, load and power
with `tegrastats` or `jtop`. Demo runs in MAXN_SUPER with `jetson_clocks` (check what the M3 is
set to).

**2.2 What we run, and what to do with each:**

| model | now (X3) | on the M3 | expected |
|---|---|---|---|
| YOLOE-26s-seg-pf (neural rung) | PyTorch through Ultralytics; 41 ms model, 244 ms a request in a job | FP16 engine ([tools/export_engine.py](tools/export_engine.py)), first checking the prompt-free head exports cleanly; at the camera's 4:3, 640x480 rather than a 640x640 letterbox (a quarter less work); for names, a text-prompted YOLOE with the classes baked in before export; an INT8 trial | model 1.5-2x (yolo26n went 27.7 -> 18.8 ms); the request mostly from 2.4 |
| Depth Anything V2 Small | FP16 engine, 26 ms | only if W1 needs a hole-filler; rebuild at 518x686 (4:3) or smaller | none, or retire it |
| Yahboom's transformer tracker (`super_track*.onnx`) | not used | if W3 wants tracking: `trtexec --fp16`, against onnxruntime | measure |
| new models from W3 (open vocabulary, segmentation, a fine-tuned small detector) | -- | FP16 engines from ONNX; NanoOWL and NanoSAM already ship as TensorRT | measure |
| AprilTag / ArUco, colour | CPU OpenCV | stay on the CPU | -- |

**2.3 Build rules.** On the M3, inside the vision container, with its TensorRT rather than the
host's (JetPack 6.2 ships 10.3, the X3's host had 10.7, the dustynv base 10.4); static shapes; a
timing cache so a rebuild takes a minute, not six; one engine per camera and input size, named so.

**2.4 The pipeline, where most of the time went.** Perception on the robot: frames never cross
WiFi, the panel gets results and a preview. No JPEG round trip between processes on the same
box. Pre-processing on the GPU. Our own decoder for YOLO26's NMS-free head (1 x 300 x 6)
instead of Ultralytics' Python post-processing, masks only for the boxes kept. Depth geometry
on the CPU while the GPU runs the network. One TensorRT runner for every engine, generalised
from [vision_service/depth.py](vision_service/depth.py), which already matches the CPU model to
0.1 % at 26 ms.

**2.5 INT8 and the DLA, as experiments.** INT8 for the CNNs, calibrated on 300-500 of our
recorded frames, kept only if W1's scores do not move. ViTs stay FP16 (Orin has no FP8, and INT8
hurts them). The Orin NX 16 GB's two DLA cores run CNN layers at INT8/FP16 with static shapes
but no attention: worth it only when two networks fight over the GPU.

**2.6 Considered, not first.** ONNX Runtime with the TensorRT execution provider (easiest to
wire, caches engines, a little slower); Torch-TensorRT (models without a clean ONNX); DeepStream
or Triton (built for many streams, we have one camera); Isaac ROS NITROS (zero-copy GPU hand-off
between ROS nodes; release 3.2 is the JetPack 6.2 line, 4.x targets Thor and JetPack 7).

Exit: a table per model -- ms before and after, the same scores on the test set -- and
end-to-end ms per frame on the robot.

## W3 -- Transformer models: where they earn their place

**What Yahboom ships.** `super_tracker` (package `smart_tracker`, a .deb) runs an ONNX
transformer tracker on the colour image: draw a box, it follows it and re-acquires it after a
loss; `pcl_segment` cuts that box out of the point cloud and publishes the object's 3D centre as
TF `base_link -> roi_object`. Their "AI large model" chapters (`multi_brains`) send speech or
text to cloud LLMs through Dify; a vision model answers with a pixel box and
`grasp_obj(x1, y1, x2, y2)` hands it to a KCF tracker and the depth. YOLOv8 runs as a TensorRT
engine to make the base follow a chosen object.

**Candidates**, each scored on the recorded set (found, named right, ms on the Orin):
- *Name objects by text* ("the red cube", "the wooden one"): NanoOWL (OWL-ViT built for Jetson,
  TensorRT, a ROS 2 node exists) against YOLOE with the vocabulary baked in (text prompts found
  nothing on the tagged cube on 2026-09-11 -- retest on the DaBai's picture), and a small VLM
  (Florence-2, Moondream) as the slow fallback.
- *Cut out an odd-shaped object from a box or a point:* NanoSAM; EfficientSAM3 (SAM 3
  distilled; a community TensorRT port exists).
- *Track while moving:* Yahboom's tracker (inspect the ONNX, build an engine) or a SAM 2-family
  tracker -- only for moving targets or visual servoing.
- *Sharper metric depth:* Prompt Depth Anything takes low-resolution metric depth plus RGB and
  returns sharp metric depth (ViT-S is the small one) -- only if W1 shows holes or rims limit it.

**Recommended use: teachers, not runners.** Offline, a big model (SAM 3, an open-vocabulary
detector) and the depth geometry label our recorded frames -- depth says where the objects
are, the big model what they are and where their edges run -- and a small model (YOLO26-seg,
or RF-DETR-Seg Nano/Small) is fine-tuned on the lab's objects and served as an FP16 or INT8
engine. Big models teach; small ones run at camera rate.

**Language, later.** Instead of asking a VLM for pixel boxes (Yahboom's way), give a language
model our object list -- labels, positions, sizes -- and let it choose. Choosing among measured
objects is reliable where drawing boxes is not.

Exit: the scored table and one decision -- which, if any, joins the pipeline.

## W4 -- 3D maps, localization and navigation

**Table scale** is W1's height map. A full 3D volume (Open3D TSDF, or nvblox on the GPU) only
if the arm needs collision checks or objects get stacked.

**Room scale, in three steps.** Work areas have to be on the surface the robot drives on, or on
a low platform: with every joint at 90 the arm's tool is only 552 mm above `base_link`, so a lab
table is out of reach from the floor.
1. Yahboom's stack as shipped: map the lab with slam_toolbox on the merged lidars; Nav2 with
   slam_toolbox localization; AprilTag relocalization (36h11 tags, 48 mm, recorded in the map
   frame). Named places in a YAML: each work area's approach pose, the drop-off.
2. 3D obstacles: the lidars at wheel height miss table tops, chair seats and overhangs; the
   camera does not. RTAB-Map (RGB-D + lidar + wheel odometry, Yahboom's launch line) in
   localization mode first -- it is installed and documented -- or Isaac ROS 3.2 nvblox (a GPU
   TSDF/ESDF feeding a Nav2 costmap layer) if RTAB-Map is too slow.
3. The mobile pick: drive to a work area -> square up (an AprilTag at the area, or its edge from
   depth) -> W1 -> pick -> drive to the drop-off -> place -> home. Navigation only has to be good
   to "area in view and in reach", because the arm camera re-finds everything relative to the
   robot. The mecanum base can sidestep to bring a far object into the arm's sweet spot
   (Yahboom holds objects 250 mm from `base_link`).

The camera is on the arm: for step 2 it faces forward while driving (Yahboom's mapping pose)
and turns down at the work area.

Exit: from a random start, drive to work area A, clear it, come back -- several times in a row.

## W5 -- Reliability, all the way through

- Offline replay tests of the recorded set in [tests/](tests/), no robot needed, so a change that
  loses an object fails before it reaches the robot.
- Every failed pick saves its RGB-D frames, joints and detections under `out/failures/` (the
  X3's `out/sweep_*.jpg` were overwritten by the next sweep).
- Watchdogs: camera frames stale for 0.5 s, or the agent gone -> stop the arm and the base.
- The gripper never closes past the width table; a close that ends wide open is "missed".
- Living with Yahboom's autostart: never open `/dev/myserial`, stay on ROS_DOMAIN_ID 30 (the
  agent's), and keep the joystick node from publishing while we drive.

## Order

| step | work | done when |
|---|---|---|
| 1 | W0.1 survey | the "not verified" list below is answered |
| 2 | W0.2-0.6 bring-up | the panel moves the arm and shows colour + depth; touch test within 5 mm |
| 3 | W1.1 test set | ~25 scenes with ground truth in `data/eval/` |
| 4 | W1.2-1.4 depth detector, offline | the score table |
| 5 | W1.5-1.6 and W2 | a 20-pick run: held, s an attempt, ms a frame |
| 6 | W3 | the model table and the decision |
| 7 | W4 | a mobile clear-the-table |

Steps 1-5 are the core: a day each for 1 and 3, a few working days each for 2 and 4. W3 and W4
are open-ended and start once step 5 holds.

## Decisions for you

1. **Where perception runs.** On the robot, the laptop only showing the page (recommended:
   frames never cross WiFi), or on the laptop as with the X3.
2. **The ROS 2 boundary.** `m3bridge` in front of our ROS-free code (recommended now), or our
   code rewritten as ROS 2 nodes (tidier once navigation is in; a much bigger job).
3. **IK.** The M3's measured geometry in our closed-form IK, with `/get_kinemarics` as a
   cross-check (recommended), or Yahboom's service as the IK.
4. **The camera while driving.** Lidar-only navigation (the camera stays free for the table), or
   camera forward for a 3D obstacle layer.
5. **Which transformer first:** naming objects by text, tracking, or language commands.

## Not verified yet

- The DaBai's exact mounting, and whether `dabai_dcw2.launch.py` turns D2C on.
- Whether any servo angle, gripper state or battery voltage reaches ROS.
- The M3 arm's link lengths and joint limits (URDF against a ruler).
- TensorRT, torch and Ultralytics versions on the M3, and whether Docker with the GPU (CDI) is
  set up as on the X3.
- What the joystick autostart publishes.

## Sources

Yahboom's ROSMASTER M3 Pro course, https://www.yahboom.net/study/ROSMASTER-M3PRO. The page
loads chapters with `build(id)`: `https://www.yahboom.net/build?id=<id>` returns JSON whose
content is an iframe, and the iframe's HTML is the chapter. Read on 2026-09-29:

- [Dabai_DCW2 camera introduction](https://www.yahboom.net/public/upload/upload-html/1755249144/1.Dabai_DCW2%20camera%20introduction.html) (15818)
- [Depth camera distance measurement](https://www.yahboom.net/public/upload/upload-html/1755249209/3.Depth%20camera%20distance%20measurement.html) (15820)
- [Wood block volume measurement](https://www.yahboom.net/public/upload/upload-html/1755249232/4.Wood%20block%20volume%20measurement.html) (15821)
- [YOLOv8 object detection](https://www.yahboom.net/public/upload/upload-html/1756883862/7.YOLOv8%20object%20detection.html) (16155)
- [Deep learning object tracking](https://www.yahboom.net/public/upload/upload-html/1756883881/8.Deep%20learning%20object%20tracking.html) (16156, YOLOv8 + TensorRT)
- [Transformer Model Tracking Algorithm](https://www.yahboom.net/public/upload/upload-html/1781334856/9.%20Transformer%20Model%20Tracking%20Algorithm.html) (19247)
- [Robotic arm solution](https://www.yahboom.net/public/upload/upload-html/1776250646/1.Robotic%20arm%20solution.html) (15839, FK/IK service)
- [Robotic Arm Calibration](https://www.yahboom.net/public/upload/upload-html/1755228720/3.%20Robotic%20Arm%20Calibration.html) (15767)
- [3D tracking machine code](https://www.yahboom.net/public/upload/upload-html/1776251986/14.3D%20tracking%20machine%20code.html) (15853, `EndToCamMat`, intrinsics)
- [Desktop tracking and gripping machine code](https://www.yahboom.net/public/upload/upload-html/1776252024/13.Desktop%20tracking%20and%20gripping%20machine%20code.html) (15852)
- [KCF tracking and gripping objects](https://www.yahboom.net/public/upload/upload-html/1776251839/10.KCF%20tracking%20and%20gripping%20objects.html) (15848)
- [Subscribe to the bus servo topic](https://www.yahboom.net/public/upload/upload-html/1789722518/5.Subscribe%20to%20the%20bus%20servo%20topic.html) (20522)
- [Precautions for DIY Robotic Arm](https://www.yahboom.net/public/upload/upload-html/1776249699/Precautions%20for%20DIY%20Robotic%20Arm.html) (17534, the gripper table)
- [Dual Lidar fusion](https://www.yahboom.net/public/upload/upload-html/1755248108/2.Dual%20Lidar%20fusion.html) (15803)
- [RTAB-Map mapping](https://www.yahboom.net/public/upload/upload-html/1755248860/12.RTAB-Map%20mapping.html) (15813) and [navigation](https://www.yahboom.net/public/upload/upload-html/1755248928/13.RTAB-Map%20navigation.html) (15814)
- [Navigation via Pose-Constrained Visual Relocalization](https://www.yahboom.net/public/upload/upload-html/1786001246/Navigation%20via%20Pose-Constrained%20Visual%20Relocalization.html) (19248)
- [Embodied intelligent robot system architecture](https://www.yahboom.net/public/upload/upload-html/1781336830/3.Embodied%20intelligent%20robot%20system%20architecture.html) (17708, multi_brains)
- [Multimodal visual understanding + robotic arm grab](https://www.yahboom.net/public/upload/upload-html/1776251348/Multimodal%20Large%20Model+robotic%20arm%20gripping.html) (15782)

Elsewhere:
- Isaac ROS [release notes](https://nvidia-isaac-ros.github.io/releases/index.html) and
  [nvblox](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_nvblox/index.html);
  [3.2 is the Orin NX / JetPack 6.2 release, 4.0 needs Thor](https://forums.developer.nvidia.com/t/isaac-ros-version-4-jetpack-v6-2-isaac-ros-argus/352333)
- [JetPack 6.2 release notes](https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-62/release-notes/index.html) (TensorRT 10.3, CUDA 12.6)
- [Orbbec SDK v2](https://github.com/orbbec/OrbbecSDK_v2) (DaBai DCW2 after the UVC firmware upgrade) and [OrbbecSDK_ROS2](https://github.com/orbbec/OrbbecSDK_ROS2)
- [NanoOWL](https://github.com/NVIDIA-AI-IOT/nanoowl), [ROS2-NanoOWL](https://github.com/NVIDIA-AI-IOT/ROS2-NanoOWL)
- [YOLOE: classes baked in before a TensorRT export](https://docs.ultralytics.com/models/yoloe)
- [RF-DETR](https://github.com/roboflow/rf-detr) (detection and segmentation, TensorRT)
- [Prompt Depth Anything](https://github.com/DepthAnything/PromptDA)
- [EfficientSAM3](https://github.com/SimonZeng7108/efficientsam3)
