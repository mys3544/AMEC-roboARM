# Tasks 2-4, transport, OR cleaning with the tray, surgical assistance: approaches on the Lite 6 manipulator (state of 27 October 2026)

Scope: what tasks 2 (clinical object transport), 3 (room preparation and cleaning) and 4 (surgical assistance) ask
in rulebook V1.0, and how to solve each sub-skill in software on the planned mobile manipulator (AgileX Ranger Mini
3.0 + 24 V lift column + UFACTORY Lite 6 + Jetson AGX Orin 64 GB; a UFACTORY 850 after selection), reusing our
pickers, detectors and twins. Covered: recognising surgical instruments, tubes and packages; grasping them; placing on
a trolley and in a tray layout; pushing the trolley; wiping without a force sensor; cabinets and shelves; sterile
presentation, opening packaging, waste, dropped items; robot-to-human handover. Not covered: the hardware choice
(see `reports/Healthcare Challenge 2027 platform.md`). Labels: **[RULE]** rulebook V1.0 / FAQ; **[DOC]** official
documentation or a paper's own page, opened; **[REAL]** measured on our robots; **[CLAIM]** vendor or author claim;
**[2nd]** search summary or secondary page; **[OURS]** our design inference; **[KNOW]** background knowledge not
re-checked (never load-bearing).

## 1. What the three tasks ask, and the scoring levers inside them

### Takeaway
Each task is split into sub-skills that are scored separately ("strong performance in one area is rewarded even if
other areas are not completed"), so a robot that does the cheap sub-skills reliably beats one that attempts all of
them badly. Two levers recur: fixtures and pre-opened materials raise raw points but lower RPL, and the sterile rules
(no hovering over cleaned areas, the robot never enters sterile areas, only non-sterile parts touched) are judged by
physicians.

### Evidence
- Task 2 [RULE, RB 2.2]: sub-skill 1 "identify and manipulate small clinical objects, such as surgical instruments or
  bio samples, from a designated area ... place these objects onto a mobile trolley without causing damage or
  contamination"; sub-skill 2 "transport larger clinical objects, such as transport trolleys ... pushing or pulling";
  sub-skill 2.1 patient transport in a wheelchair or bed for a bonus; "Each sub-skill is scored independently";
  "fully autonomous - teleoperation is not permitted"; no dynamic obstacles in 2027.
- Task 3 [RULE, RB 2.3]: device disinfection of "a designated area ... according to a predefined cleaning procedure";
  "must be able to handle such situations without relying on UV light or other non-mechanical methods. This includes
  e.g. wiping, scrubbing"; delicate object positioning from a trolley to a surgical tray "following a predefined
  layout"; "not allowed to hover over or re-enter areas that have already been cleaned"; the task ends when the room
  is disinfected, the tray is correct and the robot is parked.
- Task 4 [RULE, RB 2.4]: instruction understanding ("processing verbal commands, resolving ambiguities, and confirming
  the requested action before execution"); material identification "by appearance or labeling" and retrieval;
  sterile handover and waste disposal; technical challenge "open packaging without contaminating the sterile
  contents"; dropped or unavailable instruments handled "according to hygiene protocols"; bonus for bringing a
  contaminated reusable instrument to reprocessing; "the robot is considered non-sterile, it is not allowed to enter
  or cross sterile areas".
- RPL rises with "handling materials that are neither specially adapted nor pre-opened" and standard furniture
  instead of "a dedicated magazine"; fixtures "lower the Robot Power Level" [RULE, RB 4.2.2].
- Storage may include hinged, sliding and roller-shutter doors, cabinet doors, drawers and open shelves [RULE,
  RB 5.1.4]. Teams bring "one set of all objects used in the competition", except arena equipment (trolley, bed,
  shelves) [RULE, RB 1.4.4].
- Arena geometry from the organisers' model: counters 0.92 m, desks 0.99-1.03 m, storage cabinets 1.92 m tall and
  0.65 m deep, bin 0.61 m [2nd for us, measured in `research_notes/Healthcare Challenge 2027 platform/rules_and_task_demands.md`].
- The Lite 6: 600 g payload, 440 mm reach, built-in collision detection [DOC, via the platform notes].

### Inferences [OURS]
- Because the team brings the objects, every instrument, tube and package can be scanned, taught and rendered
  synthetically before the event; "unseen objects" matter only for the technology bonus.
- Order of value per effort: task 4 sub-skill 1 (pure speech) -> task 2 sub-skill 1 and task 3 tray (pick and place,
  our strength) -> task 3 wiping (passive tool) -> task 4 retrieval from open shelves and handover -> trolley
  pushing -> packaging -> drawers and cabinet doors -> patient transport (skip in 2027).

## 2. Recognising instruments, tubes and packages

### Takeaway
Use the recipe that already works for our kit: a closed-set YOLO26 segmenter trained on real frames plus Isaac Sim
renders of our own object set (we bring it), DINOv2 identity for look-alikes, and text reading for packaged items
("catheter type B"). Public instrument datasets are small (4 classes; 31 classes not public) and serve only for
pre-training. A VLM is the slow second opinion, not the main path.

### Evidence
- Labeled Surgical Tools and Images (Lavado, da Silva, Caramelo, 2018): 2,620 images, 4 classes (curved mayo scissor,
  scalpel no. 4, straight dissection clamp, straight mayo scissor), boxes with an occlusion flag, ODbL [DOC,
  [Dataset Ninja](https://datasetninja.com/labeled-surgical-tools-and-images),
  [Kaggle](https://www.kaggle.com/datasets/dilavado/labeled-surgical-tools)].
- PWISeg: ceiling-camera, tray and OR images with COCO boxes and keypoints, 1,788/200/185 images [2nd,
  [arXiv 2311.09819](https://arxiv.org/pdf/2311.09819)]. SurgCount-HD: dense tray counting, one class
  ("circular instrument handle") [2nd, [arXiv 2602.11024](https://arxiv.org/pdf/2602.11024)].
- Autonomous tray assembly (2026): a Staubli TX2-60L, "A custom dataset comprising 31 surgical instruments and 6,975
  annotated images", YOLO12 plus a ResNet fine classifier, a dual electromagnetic gripper; public release not stated
  [DOC, [arXiv 2602.01679](https://arxiv.org/abs/2602.01679)].
- LLM scrub nurse (Ng et al., Advanced Intelligent Systems 2025): instruments detected with oriented boxes at 97.6 %
  AP, speech commands, an LLM triggering robot actions through tool use, 94 % standalone success, latency under 1 s
  [DOC, abstract via [Crossref, doi 10.1002/aisy.202500483](https://doi.org/10.1002/aisy.202500483)].
- RoboNurse-VLA (2024): SAM 2 plus Llama 2, grasping and handing over instruments on voice command [DOC,
  [arXiv 2409.19590](https://arxiv.org/abs/2409.19590)].
- Our stack: YOLO26n-seg + Isaac Sim synthetic kit 0.79 mask mAP50, FP16 3.0 ms on both Orins; synthetic frames help
  only when the render camera matches the robot's looks; DINOv2 crops 4/4 vs YOLOE visual prompts 1/4 on look-alike
  printed cubes [REAL, memory `kit-detector`, `m3pro-teach`].
- VLMs listed for Jetson include Cosmos Reason 2 (2B, 8B), Nemotron Nano 12B VL and Gemma 4 (vLLM or llama.cpp); no
  speeds on the page [DOC, [Jetson AI Lab models](https://www.jetson-ai-lab.com/models/)].

### Inferences [OURS]
- Pipeline: YOLO26 segmentation (class = instrument type) -> oriented box from the mask (PCA) for the grasp axis ->
  DINOv2 identity when two items look alike (e.g. two clamp sizes) -> for packages, OCR of the label inside the
  package mask (a small OCR engine; a VLM only to resolve a mismatch). Train on our own items in the arena twin's
  lighting, the way `train/synth/kit_synth.py` did for the kit.
- Flat steel instruments on a steel trolley are the hard case for depth; segment in RGB, use depth only for the
  surface plane.

## 3. Grasping: fingers for tubes and boxes, a magnet for flat instruments

### Takeaway
Flat instruments lying on a surface are hard to pinch; the published tray-assembly robot uses electromagnets. A
combined gripper (the Lite 6's fingers plus a small electro-permanent magnet pad on the palm) picks scissors and
clamps flat and tubes upright, inside the 600 g payload. Grasp quality is checked from the gripper and the camera, and
a failed grasp is retried once from a new look, as the @Home winners do.

### Evidence
- The tray-assembly system's "custom dual electromagnetic gripper" [DOC, arXiv 2602.01679].
- @Home winners verify grasps from gripper encoders and hand over to a human after repeated failures [DOC,
  arXiv 2412.14989].
- The xArm SDK has `close_lite6_gripper`/`open_lite6_gripper` ("can only be used on Lite6 series robotic arms"),
  joint torque and current read-back and payload identification by current (`iden_tcp_load`, "only available on the
  lite6 model manipulator") [DOC, [xArm Python SDK API](https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/doc/api/xarm_api.md)].
- Our pickers land 1-5 mm on kit pieces (2.3 mm on the X3's first robot-side run; M3 8/8 into the box after the
  fingertip refit) [REAL, PLAN.md and memory `m3pro-sim-expert`].
- Surgical steel: many ring-handled instruments are martensitic stainless (magnetic) [KNOW]; check our own set with a
  magnet before relying on it.

### Inferences [OURS]
- Payload identification by current gives a cheap "something in the hand" check after each grasp (weight before vs
  after), beside the camera check.
- A reconfiguration between tasks (e.g. another tool) counts as a separate robot with its own risk analysis
  (RB 6.4.3); one combined tool avoids that.

## 4. The trolley: the base pushes it, the arm only loads it

### Takeaway
Trolley forces (tens to over 100 N) are far beyond the Lite 6, so the base must do the pushing through a coupling on
the robot. Nav2's docking server supports non-charging docks and is the place for a "trolley dock" plugin (detect the
trolley's legs or handle, approach, latch); after latching, the robot publishes the combined footprint and drives
slowly. NVIDIA's healthcare workflows have a push-cart task, but only for a humanoid.

### Evidence
- Docking server: docks with "non-charging infrastructure such as static locations (ex. conveyers) or dynamic
  locations (ex. pallets)", plugin dock types, a database of docks, AprilTag or custom detectors, `dock_backwards`,
  undocking action [DOC, [nav2_docking README, Jazzy](https://api.nav2.org/nav2-jazzy/html/md_nav2_docking_README.html)].
- Push forces: ~230 N initial and ~130 N sustained for loaded carts in a caregiver table; interior doors capped at
  22 N in US accessibility codes [2nd, in the platform notes].
- Isaac for Healthcare `locomanip_push_cart`: "Walk to and push a cart", Unitree G1, GR00T N1.6 [DOC,
  [i4h-workflows](https://github.com/isaac-for-healthcare/i4h-workflows)].
- Ranger Mini ROS 2 driver: `cmd_vel` only, Ranger Mini V1-V3, a `humble` branch; motion modes are not documented
  [DOC, [ranger_ros2](https://github.com/agilexrobotics/ranger_ros2)].
- The trolley is arena equipment brought by the organisers; its model is not published [RULE, RB 1.4.4].

### Inferences [OURS]
- Coupling: a passive two-point hook on the robot's rear that slides under the trolley's handle bar plus a small
  latch, so the trolley yaws with the robot (rigid pair). Pulling (robot ahead, trolley behind) gives the patient
  pathway a clear view ahead; pushing hides the robot's front lidar. Decide on the real trolley.
- Sub-skill 1 does not need the push: load the trolley with the arm wherever it stands, score sub-skill 1, then try
  sub-skill 2.

## 5. Wiping without a force sensor

### Takeaway
The Lite 6 has no force/torque sensor and UFACTORY's sensor is listed as not compatible with it, so wiping is
position-controlled with a passive spring-loaded pad, and contact is checked from the arm's estimated joint torques
and from the camera. Coverage is a boustrophedon over the designated area, ordered "far to near, then away from the
cleaned side", so that no part of the robot passes over a cleaned strip. With the 850 after selection, the UFACTORY
force/torque sensor allows real force control.

### Evidence
- "Mechanical" cleaning is required: "wiping, scrubbing, or using specialized cleaning tools" [RULE, RB 2.3.3.1];
  "not allowed to hover over or re-enter areas that have already been cleaned" [RULE, RB 2.3.4].
- xArm SDK: `get_joints_torque`, servo `currents`, `set_collision_sensitivity` (0-5), modes 0 position, 1 servo, 2
  joint teaching, 4 joint velocity, 5 Cartesian velocity, 6-7 online trajectory; F/T functions require "the Six-axis
  Force Torque Sensor ... (the third party is not currently supported)" [DOC, xArm Python SDK API].
- UFACTORY's 6-axis F/T sensor: several listings say "Not compatible with UFACTORY Lite 6"; the 850 version is for the
  850 and new xArm 5/6/7; one listing contradicts (mentions "5 Lite") [2nd,
  [RobotShop UK](https://uk.robotshop.com/products/ufactory-6-axis-force-torque-sensor-xarm-567),
  [RobotShop JP](https://jp.robotshop.com/en/products/ufactory-6-axis-force-torque-sensor-xarm)].
- Compliant scrubbing research: SCCRUB (2025) scrubs with a soft arm under open-loop force control and removes ~99.7 %
  of contamination [2nd, [arXiv 2507.06053](https://arxiv.org/abs/2507.06053)]; a scrubbing robot models tool contact
  as mass-spring-damper [2nd, NSF PAR paper]. No paper on spring-pad wiping with a position-controlled cobot was found
  (one search).
- Wiping force: one scrubbing robot caps normal force at ~12 N; no published target for OR wiping [2nd, platform
  notes].
- The Operating Room's cleanliness is set with `SetCleanliness` (DIRTY=0, CLEAN=1) [RULE, RB 8.1.3.5].

### Inferences [OURS]
- Tool: a flat pad (e.g. 120 x 60 mm, disposable wipe clipped on) on a 15-20 mm spring travel; command the pad 8-10 mm
  below the measured surface (depth plane fit) so the spring gives a roughly constant 5-15 N; calibrate the spring and
  check contact by the change in the estimated joint torques against a free-air pass.
- Order for a table the robot faces: strips parallel to the robot's front, starting at the far edge and moving
  towards the robot; the arm is always over not-yet-cleaned surface. For areas wider than the reach: columns cleaned
  one after the other with the base crabbing sideways away from the cleaned column (Ranger's crab mode, if the driver
  exposes it). Lift the pad only over dirty area; the return path to the next strip runs over dirty area too.
- Evidence of cleaning for referees and physicians: a before/after image of the area with the wiped strips drawn on
  it, and the cleaning procedure (e.g. two passes with overlap) stated on the screen. Then `SetCleanliness(CLEAN)`.
- Which "designated area" (the OR table, the C-arm, the floor) is unknown: ask.

## 6. The tray in a predefined layout

### Takeaway
This is our pick-and-place with a template: register the tray (its rectangle by segmentation and the depth plane),
place each instrument at its slot pose, and verify by a final look. Precision of 5-10 mm suffices for a tray layout;
our pickers are better than that. The arm must not pass over instruments already placed (contamination); the
placement order is "far slots first".

### Evidence
- "placed in the correct positions on the tray, following a predefined layout, while avoiding any damage or
  contamination" [RULE, RB 2.3.3.2]; instruments come "from a mobile trolley, e.g. as a continuation from Task 2"
  [RULE, RB 2.3.2].
- Isaac for Healthcare `scissor_pick_and_place`: "Pick up scissors and place them in a tray", SO-ARM 101, GR00T N1.5 or
  N1.7, also rule-based [DOC, i4h-workflows]; `locomanip_tray_pick_and_place`: "Move a surgical tray from a shelf to a
  cart" (Unitree G1) [DOC].
- The tray-assembly system packs 31 instrument types into sterile trays and reports fewer tool-to-tool collisions than
  human-packed trays [DOC, arXiv 2602.01679].

### Inferences [OURS]
- Layout as data: a JSON of slot poses in the tray frame, taught once by placing each instrument by hand and taking
  one look; the same file drives the sim and the robot.
- A good first VLA showcase for the technology bonus: the sim expert (our picker with true poses) generates tray
  episodes in the arena twin, as it does for the M3 today; the classical placer stays the competition path.

## 7. Surgical assistance: understand, confirm, fetch, present, dispose

### Takeaway
Sub-skill 1 is pure software and should be near-perfect: the voice pipeline turns the request into one function call
(`fetch(item="catheter", variant="B")`), the robot repeats it and waits for "yes". Retrieval starts with open shelves
and light cabinet doors (hook the handle, pull with the base); heavy drawers wait for the 850. Sterile presentation
with one arm needs a peel aid; a clip on the robot itself keeps it part of the robot rather than an arena fixture.
Waste goes into the 0.61 m bin. Dropped items: say so, fetch a replacement, take the dropped reusable item to
reprocessing for the bonus.

### Evidence
- "processing verbal commands, resolving ambiguities, and confirming the requested action before execution" [RULE,
  RB 2.4.3.1]. In the final: "The operator gives the robot an acoustic command requesting catheter type B. The robot
  retrieves the catheter from storage, hands it to the operator, disposes of the packaging, and returns to its waiting
  position" [RULE, RB 3.2.1].
- Opening packaging: "using specialized grippers or tools to manipulate sterile items" [RULE, RB 2.4.3.4]. A 2026
  gripper for separating and opening sealed sterile pouches found the normal force the most sensitive variable and
  needed two copies of the gripper, one per flap, to open seals robustly; nurses "open up to 240 bags per shift"
  [DOC, [arXiv 2603.10890](https://arxiv.org/abs/2603.10890), ICRA 2026]. CSIC offers the gripper for licensing
  [2nd, [CSIC](https://www.csic.es/en/innovation-and-transfer/technological-offer/ict-artificial-intelligence-robotics-health/gripper-grasping-and-opening-sterile-medical-pouches)].
- Fixtures must be declared and lower RPL; pre-opened materials lower RPL [RULE, RB 4.1.1, 4.2.2, 6.4.4].
- NimbRo opened doors "to the inside and outside with two arms" and used VLMs to judge cupboard door state [DOC,
  GRC 2026 abstract via platform notes].
- Shelves: cabinets are 1.92 m tall; the top ~0.4 m is out of reach for a compact robot [2nd, platform notes].
- Our voice parser: grammar first, then a small LLM to one JSON function [REAL on laptop; plan in the voice report].

### Inferences [OURS]
- Confirmation dialogue with the ambiguity rule: if the item table has more than one match ("catheter" -> A, B), ask
  "Type A or type B?"; never act on an unconfirmed request; the operator can say "stop" at any time (stop lane).
- Peel aid on the robot: a spring clip on the robot's deck holds one flap; the arm pulls the other flap back and
  presents the opened pouch with the item facing the operator; the operator takes the sterile item without touching
  the outside. Declare it in the TDP as part of the robot. Fallback: present the closed package, which scores less.
- Waste: drop into the bin from 0.1 m above its opening (bin top 0.61 m). Dropped item on the floor: detect it with
  the head camera, say "The instrument is contaminated, I will bring a replacement", fetch one, then pick the dropped
  one (if reusable, per a single-use/reusable table) and carry it to the reprocessing location.

## 8. Handover to a person without a force sensor

### Takeaway
Present the item at a fixed, rehearsed pose outside the sterile field, slow and low-sensitivity-safe, and release on
a pull. Without a wrist sensor, detect the pull from the arm's estimated joint torques (a change against the hold
value) confirmed by vision (a hand on the item, from a hand-keypoint model) or by a spoken "got it". The published
release controllers fuse torque and vision; vision-only release exists too.

### Evidence
- Release controller fusing a CNN on joint-torque data with RGB-D finger detection: 98 % accuracy in preliminary real
  experiments, but on a Kinova with joint torque sensors [2nd, [arXiv 2210.15085](https://arxiv.org/pdf/2210.15085)].
- GoferBot treats release as vision only (human reach and grasp from the same RGB camera) and notes force-based
  release suffers from induced forces [2nd, [arXiv 2304.08840](https://arxiv.org/pdf/2304.08840)].
- Hand keypoints: 26,768 images with 21 keypoints per hand, CC BY-NC-SA 4.0 images; train a YOLO pose model on it
  (no pretrained hand model listed) [DOC, [Ultralytics hand-keypoints](https://docs.ultralytics.com/datasets/pose/hand-keypoints/)].
- xArm SDK `get_joints_torque` and `currents` are reported in the rich report stream; `set_collision_sensitivity`
  sets the vendor collision detection [DOC, xArm Python SDK API].
- ISO 10218-1/-2:2025 now contains the former ISO/TS 15066 collaborative limits [2nd, platform safety notes].

### Inferences [OURS]
- Release rule: hand keypoints inside the item's box for > 0.3 s AND (torque deviation over threshold OR "got it"),
  else keep holding; after 10 s ask "Please take the catheter". Calibrate the torque threshold with the item in hand
  at the handover pose; the Lite 6's estimate is noisy, so it is a confirmation, not the only trigger.
- Present slowly (< 0.1 m/s near the hand) and keep the vendor's collision detection on. These numbers go into the
  risk analysis.

## 9. What typically fails in these tasks

### Takeaway
Grasp failures on flat steel, slips on placement, forces the arm does not have, and slow perception under the
clock. Count repeated trials of each sub-skill; show the counts in the application.

### Inferences [OURS]
| failure | guard |
|---|---|
| flat instrument not grasped | magnet pad; retry from a new look; ask for help last (scores less, still scores) |
| wrong item fetched | DINOv2 identity + label OCR + spoken confirmation of what was found ("I have catheter type B") |
| cabinet door or drawer too stiff | open shelves first; hook + base pull; 850 after selection |
| wiping misses area or touches cleaned strip | far-to-near order, overlap 20 %, before/after image |
| trolley swings when pushed | rigid two-point coupling, 0.3 m/s, footprint swap |
| handover never released | vision + torque + voice triggers, timeout and a spoken prompt |
