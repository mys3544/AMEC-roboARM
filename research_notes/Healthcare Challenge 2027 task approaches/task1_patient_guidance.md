# Task 1, patient guidance to the CT: what it asks and how to solve it on a Jetson (state of 27 October 2026)

Scope: what task 1 of the Healthcare Challenge @ RoboCup 2027 asks precisely (rulebook V1.0, 27.08.2026, read in full
from `C:\Users\Mys\Desktop\Siemens sutaz\2026_09_01_Rulebook_released_V1.pdf`; the official FAQ and overview read in a
browser on 2026-10-27), and the software approach per sub-step: finding the patient, keeping her identity, leading
her at a reduced-mobility pace, doors, speech in the right languages and the CT workflow through the Hospital
Controller. The platform choice (the M3 Pro with a 1.1 m mast as guide, the self-built manipulator as fallback) is
taken from `reports/Healthcare Challenge 2027 platform.md` and not redone. Labels: **[RULE]** rulebook V1.0 or the
official FAQ; **[DOC]** official documentation or a paper's own page, opened; **[REAL]** measured on our robots
(project memory and reports); **[CLAIM]** a vendor's or author's statement not checked by us; **[2nd]** a search
summary or secondary page; **[OURS]** our design inference; **[KNOW]** background knowledge not re-checked this
session (never load-bearing). "RB x.y" is a rulebook section.

## 1. What task 1 asks, and what is still unknown

### Takeaway
Seven sub-steps, all of them navigation, perception and dialogue; none needs an arm. The run ends when the robot is
back at its parking position. No adverse events in 2027. The rulebook does not say which language the patient
speaks, and the scoring chapter is withheld, so the weights of the seven steps are unknown. Two things that look like
details are scored: the pathways (patient pathway while accompanying, staff pathway when alone) and the door method
(robot, patient or controller; physical opening ranks higher).

### Evidence
- Sub-steps [RULE, RB 2.1.4]: (1) reception and greeting: "detect the arriving patient, approach her, introduce itself
  as the guiding system, and confirm the patient's concern (e.g. appointment and name)"; (2) guidance to the CT, "deal
  with doors along the way, and maintain an appropriate distance and pace (the patient may be of reduced mobility)";
  (3) instruction at the scanner, "recognize when the patient is ready"; (4) patient preparation (e.g. take off shoes
  for an ankle scan); (5) scan initiation through the Hospital Controller: "ensure the table is in the correct
  position, start the scan, and wait for completion"; (6) escort back outside; (7) return to parking, "which marks the
  end of the run and stops the time".
- "For the current competition, teams should not expect such situations to occur" (questions, anxiety, scanner
  errors, collapse) [RULE, RB 2.1.4].
- Pathways: "the robot uses the patient pathway while accompanying the patient and the staff pathway while moving on
  its own"; disregarding them "affects both the task score and the real-world readiness" [RULE, RB 6.3.1]. They are
  "marked in the arena and announced during the setup days" [RULE, RB 6.3.1].
- Doors: robot, patient or data communication; "The chosen method and its complexity are reflected in the scoring"
  [RULE, RB 6.4.9]; "Performing the action physically is generally more demanding and is therefore rewarded with a
  higher score" [RULE, RB 8.1.1.1].
- Contact: the robot may not initiate contact; a patient may lean on it, and then it "must be designed to safely bear
  the applied forces" [RULE, RB 6.1.4]. The platform report decided to declare no physical support in 2027.
- External actors raise RPL: "Relying on external, non-instructed persons demonstrates that the system generalises
  beyond the team's own voices, phrasing, and habits" [RULE, RB 4.2.2].
- A medical panel of physicians rates "the appropriateness of the robot's actions in a clinical context" on equal
  footing with the technical referees [RULE, RB 6.1.5].
- Language: the rulebook names English only for the TDP and web site [RULE, RB 1.4.2]; the FAQ says nothing about the
  patient's language (read 2026-10-27, [FAQ](https://www.robocup2027.event.siemens-healthineers.com/blocks/faq)). The
  scenario names are "Jessy Smith" (task 1) and "Marco Mustermann" (final) [RULE, RB 2.1.1, 3.2.1]. Not found anywhere
  else (rulebook full text searched for "English", "German", "language", "speech").
- The FAQ: "Each challenge is divided into subtasks, including easier tasks that allow teams to earn points" [RULE,
  FAQ].

### Inferences [OURS]
- Task 1 is the cheapest task to score: it needs no new hardware beyond the M3's mast and E-stop, and every sub-step
  is software we can build and test in the lab and in the twin.
- The likely actors are organisers (Siemens Healthineers, Erlangen) or other teams. Plan for English and German, and
  ask the organisers which one (question list in the report).

## 2. The step machine and who runs it

### Takeaway
Run task 1 as one explicit, logged state machine on the M3 (it holds the Hospital Controller order, the dialogue,
navigation goals and the CT calls). Every state publishes its plan as the intent string. This is also the
"explainability" item of the technology bonus and the robustness lesson of the @Home winners.

### Evidence
- @Home winners run behaviour as hierarchical state machines and show perception results on a screen "helped users
  and debugging" [DOC, NimbRo 2024, [arXiv 2412.14989](https://arxiv.org/html/2412.14989v1); summarised in
  `research_notes/Healthcare Challenge 2027 platform/what_wins_similar_leagues.md`].
- The robot must publish an intent `std_msgs/String` "describing the robot's current intention or reasoning,
  intended for spectator displays" at 1-2 Hz [RULE, RB 4.1.4.1].
- T bonus lists "explainability and introspection of the robot's decisions, which is of particular value in a medical
  context" [RULE, RB 4.2.4].

### Inferences [OURS]
- States: WAIT_AT_PARKING -> DETECT_ARRIVAL -> APPROACH -> GREET_AND_VERIFY -> GUIDE(leg i, door j) -> AT_CT_EXPLAIN
  -> PREPARE (shoes) -> POSITION (table at BOTTOM, patient lies down, ready check) -> SCAN (robot steps back, talks)
  -> RETRACT_AND_HELP_UP -> ESCORT_OUT -> FAREWELL -> RETURN_TO_PARKING (staff pathway). Each state has a timeout, a
  retry and a spoken fallback; the intent string is generated from the state and its arguments (template, not LLM),
  e.g. "Guiding Jessy Smith to CT, door Hallway_1 opening via controller, patient 1.3 m behind".
- The same machine runs against the mock Hospital Controller in the Isaac Sim twin and on the real robot.

## 3. Finding and keeping the patient: detection, tracking, re-identification on a Jetson

### Takeaway
Use three cues and fuse them: an RGB person detector with a tracker that carries an appearance embedding (BoT-SORT
with ReID in Ultralytics, or DeepStream's NvDCF with ReIdentificationNet), legs from the 2D lidars (DR-SPAAM), and a
short "enrolment" of the patient's look at the greeting. A guide drives in front, so it must see behind it: on the M3
the arm can turn the wrist camera backwards while driving, and its two lidars cover the rear. The @Home winners lost
points exactly here (a lost track), so the fallback must be spoken ("Please follow me, I am here").

### Evidence
- Ultralytics trackers: BoT-SORT, ByteTrack, OC-SORT, Deep OC-SORT, FastTracker, TrackTrack; BoT-SORT has optional
  ReID: "ReID is disabled by default to minimize overhead. Enable it by setting `with_reid: True`"; `model: auto`
  "uses native YOLO features when available"; `track_buffer`, `proximity_thresh`, `appearance_thresh` tune it [DOC,
  [Ultralytics tracking](https://docs.ultralytics.com/modes/track/)].
- DeepStream: NvDCF re-associates objects after partial or full occlusion; ReIdentificationNet generates person
  embeddings; on an Orin NX GPU PeopleNet ResNet34 INT8 runs ~286 FPS with NvDCF (no ReID row in the table); several
  forum threads report ReID crashes and engine rebuilds on DeepStream 7-8 [2nd, search summary of
  [DS 7.1 performance](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_Performance.html),
  [ReIdentificationNet](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/reidentificationnet) and NVIDIA forum
  threads].
- 2D lidar person detection: DROW3 and DR-SPAAM are "real-time person detectors using 2D LiDARs mounted at ankle or
  knee height"; 8.8-24.9 FPS on a Jetson AGX Xavier; a ROS 1 example node, a ROS 2 port by another contributor [DOC,
  [2D_lidar_person_detection](https://github.com/VisualComputingInstitute/2D_lidar_person_detection)].
- NimbRo 2025-2026 fused RGB-D detections with 2D-lidar leg tracks; in 2024 it lost the operator "while avoiding a
  hard-to-see chair" [DOC, NimbRo GRC 2026 abstract and arXiv 2412.14989, via the platform notes].
- A hospital guide robot (Freiburg, 2017) adapted its velocity to the perceived category of person (mobility aids
  detected by a deep detector) [2nd, [arXiv 1708.00674](https://arxiv.org/pdf/1708.00674)].
- Our stack: YOLO26 kit detector 3.0 ms FP16 on the Orin NX; DINOv2 crops identified 4/4 taught cubes where YOLOE
  visual prompts got 1/4 [REAL, project memory `kit-detector`, `m3pro-teach`]. The M3 has two T-mini Plus lidars and
  the DaBai DCW2 depth camera on its arm [REAL, PLAN.md Architecture].

### Inferences [OURS]
- Recommended on the M3 (Orin NX 16 GB): YOLO26n person detection at 15-30 Hz + BoT-SORT with ReID (`model: auto`
  first; a dedicated ReID encoder only if look-alike actors appear) + DR-SPAAM on the merged rear scan. The patient
  track is "locked" at the greeting (embedding gallery from 5-10 frames while she stands in front). Re-acquisition
  after loss: the same gallery, then the leg track nearest to the last position, then speech.
- DeepStream is faster but brittle with ReID in forum reports; Ultralytics keeps our existing export path (ONNX ->
  TensorRT). Start there.
- The wrist camera as a "head": the M3's arm can point the DCW2 backwards over the mast while the base drives; the arm
  must then stay inside the chassis guard and below the mast (our mesh-shell guard already checks the arm against the
  body [REAL, `m3pro-chassis-guard`]).

## 4. Leading at the patient's pace, on the right pathway

### Takeaway
Nav2 has no official social-navigation layer in 2025-2026. What exists officially and suffices for a static arena
with one patient: the route server (graph routes with operations at nodes and edges, in Jazzy), the speed limit
input, keepout and speed filters, and the collision monitor. Leading is our own small node: it sets the speed limit
from the measured gap to the patient and stops and turns towards her when she falls behind. Social costmap layers
from research (Nav2CAN, learned social costs) and HuNavSim are for testing, not for the run.

### Evidence
- No official people-aware Nav2 layer was found in 2025-2026; research add-ons: a learned social cost map published
  as an OccupancyGrid and used as a local costmap (2024), Nav2CAN costmap plugins for proxemics and F-formations
  (Aalborg 2023), and HuNavSim's social layer baseline [2nd, search summary;
  [arXiv 2407.10547](https://arxiv.org/pdf/2407.10547), [Nav2CAN](https://vbn.aau.dk/ws/files/561057816/Nav2CAN_conference_paper.pdf),
  [HuNavSim](https://arxiv.org/pdf/2305.01303)].
- HuNavSim: human agents with six reactions to a robot, a social force model, social-navigation metrics; wrappers for
  Gazebo, Webots and Isaac Sim; "Tested in ROS2 Humble" [DOC, [hunav_sim](https://github.com/robotics-upo/hunav_sim)].
- Nav2 Following Server (`opennav_following`, documented for Jazzy): follows "a detected object or specific frame",
  `desired_distance` default 1.0 m, optional search by rotating [DOC,
  [configuring following server](https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/others/configuring_following_server/)].
  It follows; it does not lead.
- Nav2 Route Server in Jazzy: GeoJSON graphs, edge scorers, operations such as `AdjustSpeedLimit`,
  `ReroutingService` and `TriggerEvent` [2nd for defaults, search summary of the
  [route server page](https://docs.nav2.org/configuration/packages/configuring-route-server.html); [DOC] package
  listed for Jazzy at [docs.ros.org](https://docs.ros.org/en/jazzy/p/nav2_route)]. Graphs can be drawn in RViz with
  the Nav2 Route Tool on a slam_toolbox map [2nd].
- Walking speeds a guide must cover: a guide robot study adapted online to users walking at about 0.2, 0.7 and 1.1 m/s
  over 25-60 m [2nd, [proactive control paper](https://www.manoonpong.com/ProactiveControl/main_body_final.pdf)].
- Our state: slam_toolbox maps within ~1 cm of the fence; AMCL and Nav2 not yet running on our robots [REAL,
  PLAN.md "Where things stand"].

### Inferences [OURS]
- Two route graphs, "patient" and "staff", drawn on the site map during the setup days from the marked pathways
  (RB 6.3.1). The M3 runs Humble, whose Nav2 has no route server; there the same effect comes from a waypoint list per
  pathway followed with `NavigateThroughPoses` plus a keepout mask for the other pathway. On the Jazzy manipulator the
  route server does it natively.
- Guide control law: target gap 1.0-1.5 m (front of patient to back of robot); speed limit = clamp(k x (2.0 m - gap),
  0, 0.6 m/s); stop when the gap exceeds 2.5 m, turn the screen towards her and speak; resume when she is within
  1.5 m. Publish it on Nav2's speed limit input so the planner and the collision monitor stay in charge of safety.
- Navigation must exist first: AMCL + Nav2 on the M3 is a prerequisite for task 1 and is not done (PLAN.md M2).

## 5. Doors: through the controller first, the patient second, the robot never with the M3

### Takeaway
On the M3, open doors through the Hospital Controller (`/hospital/doors/operate`) and wait for the OPEN state; ask the
patient as the fallback. Physical opening ranks higher but needs force the M3 does not have. Bind each door to a node
or edge of the route graph so that the request is issued automatically on approach.

### Evidence
- OperateDoor: goal `{device_id, command}` with COMMAND_OPEN=0 / COMMAND_CLOSE=1; goals are rejected for unknown
  doors, a door already moving, or one already in the requested end state; DoorState reports CLOSED, OPENING, OPEN,
  CLOSING, OBSTRUCTED and `percentage_open` [RULE, RB 8.1.3.2].
- "Not every door in the arena is necessarily physically present; some may be simulated or represented symbolically"
  [RULE, RB 6.4.9].
- Nav2's `TriggerEvent` route operation: "meant to be named accordingly to the event of query in the parameter file
  (e.g. OpenDoor, CallElevator)"; it calls a `std_srvs/Trigger` service fixed at launch or named in the node's or
  edge's metadata [DOC, [TriggerEvent API, Jazzy](https://api.nav2.org/nav2-jazzy/html/classnav2__route_1_1TriggerEvent.html)].
- Door leaves in the organisers' model are 0.95-0.97 m; the smallest door the robot must pass is 0.75 m [RULE,
  RB 4.1.2.1; GLB measurement in the platform notes].

### Inferences [OURS]
- On Jazzy: a `TriggerEvent` per door calls a tiny service in our gateway that sends OperateDoor and returns once
  `percentage_open` is 100 (or OBSTRUCTED -> stop and ask). On Humble (M3): the same logic as a state of the task
  machine before each door waypoint.
- Close the door behind us only if the scenario requires it (unknown; ask).

## 6. Speech: English and German, closed questions, a screen beside it

### Takeaway
Use the voice stack we already have, moved on board: Parakeet TDT 0.6B v3 (it covers English and German at ~5 % WER on
FLEURS and detects the language itself), a grammar-first parser with the small LLM as second stage, and pre-rendered
plus Piper speech out. Keep the dialogue made of closed questions the robot asks ("Are you Jessy Smith?"), show every
sentence on the mast screen with large yes/no buttons, and use the LLM only to understand free answers. Identity check
with two identifiers is what clinicians expect.

### Evidence
- Parakeet TDT 0.6B v3: 25 European languages including English and German; FLEURS WER English 4.85 %, German 5.04 %;
  "automatically detects the language of the audio" [DOC,
  [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)].
- Our voice research: Parakeet v3 0.33 s median per clip on an 8 GB Orin Nano; the 4B parser that answers in 0.65 s on
  the laptop would need ~2 s on the Orin NX; on the robot "the grammar goes first and a distilled 0.8B second";
  speech out pre-rendered phrases plus Piper [2nd/REAL mix, `reports/Slovak voice on the robot.md`]. Grammar + rules
  hit 79/79 commands on the laptop [REAL, memory `local-voice-llm`].
- NimbRo lost a command because "a quiet operator could not be heard over band noise"; touchscreen fallbacks helped
  [DOC, arXiv 2412.14989].
- Two patient identifiers (name plus date of birth) are standard patient-safety practice in hospitals [KNOW].

### Inferences [OURS]
- Greeting script: "Hello, I am <name>, the guide for tonight's CT appointments. Are you Jessy Smith?" -> yes ->
  "Please tell me your date of birth" (checked against the appointment list) -> "Your CT of the right ankle is ready.
  Please follow me; I will walk slowly." Each line on the screen, each answer also by button.
- Language: start in English; switch to German when the first answer is German (Parakeet's language tag). Two Piper
  voices (en, de) pre-render all fixed phrases [KNOW for Piper's German voices].
- The patient is not wearing our microphone: a directional microphone array on the mast at ~1.1 m, robot speaks
  facing her, and the screen covers the rest. Measure the word error at 1 m with the motors running (the voice report
  already plans this measurement).

## 7. The CT workflow through the controller

### Takeaway
The scan is three controller actions and two observations. The robot lowers the table, asks the patient to lie down,
checks she is ready by voice and by pose, leaves the beam area while `radiating` is true, then retracts, lowers and
helps her off by speech.

### Evidence
- MoveTable: COMMAND_UP=0 / COMMAND_DOWN=1, feedback CtTableState (BOTTOM, MOVING_UP, MOVING_DOWN, TOP, PAUSED,
  `percentage_done`). RunScan: COMMAND_START=0 moves the table to TOP and then in and out of the gantry;
  COMMAND_STOP=1 halts; COMMAND_RETRACT=2 moves the tabletop out after a stop; CtScannerState has `radiating`
  [RULE, RB 8.1.3.4].
- "Instruct the patient to position herself on the examination table; recognize when the patient is ready"
  [RULE, RB 2.1.4].

### Inferences [OURS]
- Sequence: MoveTable DOWN (to BOTTOM) -> "Please take off your shoes and lie on your back, feet towards the
  scanner" -> ready check (YOLO pose: shoulders and hips at table height and roughly horizontal, plus a spoken "yes")
  -> robot moves to a waiting point outside the gantry area -> RunScan START, speak "Please keep still" -> wait for
  the action result -> RETRACT if needed -> MoveTable DOWN -> "You may sit up slowly and put your shoes on".
- Whether the robot may stay in the room during the simulated radiation is not stated; leaving it is the clinically
  correct behaviour and costs seconds.

## 8. What typically fails, and the cheapest guard against each

### Takeaway
The failures that cost @Home teams points are operational: lost tracks, noise, timeouts, late robots. Each needs a
designed fallback, tested by counting repeated runs.

### Evidence
- NimbRo 2024: operator lost behind a chair; time ran out in two tasks; arm hardware defect; quiet operator in band
  noise [DOC, arXiv 2412.14989].

### Inferences [OURS]
| failure | guard |
|---|---|
| patient track lost behind the robot | rear view (arm camera turned back) + lidar legs + ReID gallery; stop and speak after 2 s |
| speech not understood | closed questions, screen buttons, repeat once, then proceed with the screen only |
| door does not open (controller rejects, OBSTRUCTED) | re-query `get_states`, then ask the patient |
| localisation drift in a new arena | map on setup days; AMCL with the walls only below 1.4 m; route graph re-drawn on site |
| the run's clock | no waits without a timeout; park even if a middle step failed (step 7 stops the time) |
