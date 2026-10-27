# Every Healthcare Challenge task can be built from our pick-and-place stack, Nav2 and an on-board voice dialogue; the application should show the sub-skills on real robots and the final in a twin of the official arena driven by the robot's own Jetson

**Bottom line.** None of the four tasks needs a method we do not know. Each is a chain of sub-skills that are scored
separately, and most of them are navigation, dialogue or table-top pick-and-place, which we already do to 1-5 mm. The
real gaps are elsewhere. Nav2 does not run on our robots yet. The voice stack is not on board yet. And three things
need force the Lite 6 lacks: trolley pushing (the base does it), wiping (a passive spring pad) and stiff drawers
(after the 850 upgrade). Build in this order:

1. Run the new manipulator natively on ROS 2 Jazzy (JetPack 7.2 now supports the AGX Orin), so that it speaks to the
   Jazzy Hospital Controller without a bridge.
2. Give the Humble M3 a small Jazzy gateway that talks to its own stack over HTTP, never over cross-distro DDS.
3. Coordinate the two robots with a small dispatcher, plus a fallback that uses only the controller's own latched
   states, in case the rules forbid robot-to-robot links.

For the 31 January application, film task 1 on the M3 and every pick-and-place sub-skill on the real manipulator,
with success counts. Show the whole final in Isaac Sim: a twin of the organisers' arena with both robots, driven by
the same AGX Orin that will drive the real robot.

Written 2026-10-27 for the lab's planning. It builds on `reports/Healthcare Challenge 2027 platform.md`, whose
hardware choice it takes as given: an AgileX Ranger Mini 3.0, a 24 V lift column, our Lite 6, an AGX Orin 64 GB and
a hard-wired E-stop chain, with the M3 Pro as guide. Evidence and links are in five notes in
`research_notes/Healthcare Challenge 2027 task approaches/`:

- `task1_patient_guidance.md`
- `tasks2to4_manipulation.md`
- `fleet_and_hospital_controller.md`
- `simulation_demo.md`
- `comparable_work.md`

Labels in the tables: [RULE] rulebook V1.0 or FAQ, [DOC] documentation, [REAL] measured on our robots, [2nd]
secondary. "RB x.y" is a rulebook section.

## 1. What decides the approach

Five facts shape every choice below.

1. **Sub-skills are scored separately.** Task 2: "Each sub-skill is scored independently" [RULE, RB 2.2.3]. The FAQ:
   "Each challenge is divided into subtasks, including easier tasks that allow teams to earn points". A robot that
   does the cheap sub-skills reliably beats one that attempts everything badly. Our order of value per effort:
   1. task 1 entirely;
   2. task 4's instruction understanding (pure software);
   3. the Hospital Controller calls (doors, CT, lab, OR);
   4. task 2's pick-to-trolley and task 3's tray layout (our strength);
   5. wiping;
   6. shelf retrieval and handover;
   7. trolley pushing;
   8. packaging;
   9. drawers;
   10. patient transport (skip in 2027).
2. **The multiplier punishes help.** Fixtures and pre-opened materials raise raw points but lower RPL. External
   actors, standard furniture and fleet coordination raise it (RB 4.2.2). Physicians score clinical appropriateness on
   equal footing with the referees (RB 6.1.5). How the robot speaks, waits, checks identity and respects sterile zones
   is therefore worth as much as speed.
3. **All computing is on board, and the controller is the only channel during a run** (RB 4.1.4.1, 8.1.1.2).
   Speech, the LLM, perception, Nav2 and the policies run on the AGX Orin and on the M3's Orin NX.
4. **Much is still unknown.** The scoring chapter, the Hospital Controller's code, the object list, the trolley model,
   the patient's language and the rule on robot-to-robot links are all unpublished (section 9).
5. **Our starting point** [REAL]:
   - Done: slam_toolbox maps; pickers landing 1-5 mm; a YOLO26 detector at 3.0 ms on the Orin NX; DINOv2 identity;
     ACT at 10 Hz on the Jetson; an M3 twin within ~5 mm with a privileged expert; voice on the laptop (Parakeet plus a
     grammar-constrained 4B LLM, 79/79 commands).
   - Not yet done: AMCL and Nav2 on a robot, voice on a robot, anything on the manipulator.

The software on each robot has the same shape:

```
   Hospital Controller (Jazzy, domain 0, arena WiFi)
          |  ExecuteTask, device actions/services, latched states
   gateway (Jazzy): ExecuteTask server, pose 20 Hz, intent 1 Hz, device clients, state cache, log
          |  in-process (manipulator, Jazzy)  or  HTTP on 127.0.0.1 (M3, Humble)
   task machine: one explicit state machine per task, every state with timeout, retry, spoken fallback
     |-- navigation: AMCL + Nav2 (route graphs for the patient and staff pathways, speed limit, docking)
     |-- perception: YOLO26 seg + DINOv2 identity + label OCR; person tracks (BoT-SORT ReID + lidar legs)
     |-- dialogue: mic array -> Parakeet v3 (en/de) -> grammar -> small LLM -> one function call; Piper TTS; screen
     |-- manipulation: xArm SDK / MoveIt 2 (Lite 6), pickers, wiping planner, handover controller
   safety (independent of all of the above): E-stop chain, MCU watchdog, speed and zone limits
```

## 2. Task 1: patient guidance to the CT, on the M3 with a mast

The M3 with a 1.1 m mast (screen, microphone array, speaker) runs task 1, which keeps the manipulator free in the
final. The manipulator with a head is the fallback.

| sub-step | method / component | runs on | reuse from our stack | risk |
|---|---|---|---|---|
| detect the arriving patient, approach | YOLO26n person detection + BoT-SORT with ReID (Ultralytics, `with_reid: True`) [DOC]; approach to 1.2 m facing her | M3 Orin NX | YOLO26 TensorRT export path [REAL] | crowd behind 1.4 m walls: restrict to the entrance zone |
| greet, verify name and appointment | closed questions ("Are you Jessy Smith?", date of birth) spoken and on screen with yes/no buttons; Parakeet v3 (en 4.85 %, de 5.04 % FLEURS WER, automatic language detection) [DOC]; grammar first, small LLM second | M3 | console voice pipeline [REAL on laptop] | language unspecified (ask); hall noise: close mic array, screen fallback |
| guide at a reduced-mobility pace, keep distance | own guide node sets Nav2's speed limit from the gap to the patient (target 1.0-1.5 m, max 0.6 m/s, stop above 2.5 m and speak); rear view = the arm camera turned backwards + DR-SPAAM leg detection on the lidars [DOC] | M3 | slam_toolbox map [REAL]; arm camera; mesh-shell guard | AMCL and Nav2 not yet running on our robots; track loss (NimbRo's failure) |
| patient pathway vs staff pathway | waypoint lists per pathway + keepout mask on Humble; route graphs on Jazzy (Nav2 Route Server) [DOC] | M3 | - | pathways announced only on setup days: draw them on site |
| doors | OperateDoor via the controller, wait for OPEN, OBSTRUCTED -> stop and ask; fallback: ask the patient [RULE, RB 8.1.3.2] | M3 + gateway | - | physical opening scores higher but needs force the M3 lacks |
| instruct at the scanner, shoes, ready check | MoveTable DOWN; speech and screen; ready = YOLO pose (body horizontal at table height) + spoken yes | M3 | - | pose model on a person lying down: test |
| run the scan | RunScan START, step out of the gantry area, "please keep still", wait for the result, RETRACT, MoveTable DOWN [RULE, RB 8.1.3.4] | gateway | - | whether to leave the room during `radiating` (ask) |
| escort out, farewell, park | same guide node; parking via the staff pathway; the time stops at parking | M3 | - | none beyond navigation |

Two design points matter most:

- **The guide must see behind itself.** On the M3 the arm can point the wrist camera backwards over the mast while the
  base drives, inside the existing chassis guard.
- **The identity check uses two identifiers.** Name and date of birth is what clinicians do; the physicians will
  recognise it.

The task machine's state is the intent string, e.g. "Guiding Jessy Smith to CT, door Hallway_1 opening via
controller, patient 1.3 m behind". That string is also the explainability item of the technology bonus.

## 3. Task 2: object transport

| sub-step | method / component | runs on | reuse | risk |
|---|---|---|---|---|
| find instruments, tubes, supplies | YOLO26 segmentation trained on our own object set (we bring it, RB 1.4.4), real frames + Isaac Sim renders; DINOv2 for look-alikes; label OCR for packages | AGX Orin | kit detector pipeline and `kit_synth.py` [REAL]; DINOv2 identity [REAL] | flat steel on steel trolleys: segment in RGB, depth only for the plane |
| grasp | Lite 6 fingers for tubes and boxes, a small magnet pad for flat instruments (the published tray packer uses electromagnets [DOC]); grasp check by camera + payload identified from motor current (`iden_tcp_load`, Lite 6 only [DOC]) | Lite 6 + Orin | pickers, hand-eye, IK [REAL] | 600 g payload incl. tool; non-magnetic instruments |
| place on the trolley | top-surface plane from depth, free-space placement grid, verify by a look | Orin | drop-into-box logic [REAL] | trolley model unknown |
| couple to the trolley | Nav2 docking server with a custom non-charging dock plugin (it supports pallets and conveyors [DOC]); detect the trolley's legs/handle; passive two-point hook + latch on the robot | Orin + base | - | coupling hardware untested; Ranger driver modes undocumented |
| push or pull between rooms | footprint swap (robot + trolley), 0.3 m/s, doors via the controller | Orin | - | forces 100+ N (base, not arm); swing |
| patient transport (bonus) | not in 2027 | - | - | safety case too heavy |

Sub-skill 1 does not depend on sub-skill 2. Load the trolley wherever it stands, score that, then try the push.

## 4. Task 3: OR cleaning and the tray

| sub-step | method / component | runs on | reuse | risk |
|---|---|---|---|---|
| find the designated area | surface segmentation + plane fit; the area's outline from the order or the room model | Orin | depth floor fitting [REAL] | "designated area" undefined (ask) |
| wipe mechanically | passive spring pad (15-20 mm travel), commanded 8-10 mm below the surface for ~5-15 N; contact confirmed by the change in `get_joints_torque` against a free-air pass [DOC]; no F/T sensor on the Lite 6 (UFACTORY's sensor listed as incompatible [2nd]) | Lite 6 | - | uneven force; no published removal rate for this method: measure our own |
| never hover over cleaned parts | boustrophedon strips ordered far edge to near edge; wider areas column by column, the base crabbing away from the cleaned column; pad lifted only over dirty area | Orin + base | coverage logic of the M3's camera-down survey planner [REAL] | arm links over cleaned strips if the order is wrong: check in the twin |
| set the OR to CLEAN | `SetCleanliness(CLEAN)` after a before/after image | gateway | - | none |
| instruments to the tray layout | tray registration (rectangle + plane), slot poses from a JSON taught once, far slots first, final verification look | Lite 6 + Orin | pickers at 1-5 mm [REAL] | tray model and layout unknown (ask) |
| park | Nav2 | Orin | - | - |

After selection, the 850 with UFACTORY's force/torque sensor (compatible with the 850 [2nd]) turns wiping into real
force control.

## 5. Task 4: surgical assistance

| sub-step | method / component | runs on | reuse | risk |
|---|---|---|---|---|
| understand and confirm | the robot's own microphone array (a worn wireless microphone could count as another channel to the robot, RB 8.1.1.2); Parakeet -> grammar -> small LLM -> `fetch(item, variant)`; repeat and wait for "yes"; ask when more than one item matches; "stop" lane always on | Orin | voice pipeline [REAL on laptop]; stop-word plan (voice report) | the 4B LLM ~2 s on the Orin NX (est.), faster on the AGX; external operators |
| identify in storage | detector + DINOv2 + label OCR; spoken confirmation of what was found ("I have catheter type B") | Orin | as task 2 | look-alike packages: OCR decides |
| retrieve | open shelves first; light cabinet doors by hook + base pull; drawers after the 850; MoveIt 2 with the Lite 6 configs from xarm_ros2 [DOC] for collision-aware reaching into shelves | Lite 6 + base | mesh-shell guard idea | 1.92 m cabinets: top shelf out of reach; stiff doors |
| open packaging, present sterile side out | a spring clip on the robot's own deck holds one flap, the arm peels the other (the only published pouch opener needs two grippers [DOC]); present with the item facing the operator | Lite 6 | - | peel force near payload; declared as robot hardware, not a fixture |
| hand over | fixed handover pose outside the sterile field, slow; release when hand keypoints are on the item AND (torque deviation OR "got it"); 10 s timeout and a prompt | Lite 6 + Orin | - | Lite 6 torque estimate noisy: vision is the main trigger |
| dispose of packaging | drop into the 0.61 m bin | Lite 6 + base | drop logic [REAL] | - |
| dropped or contaminated item | detect on the floor, say so, fetch a replacement; reusable items to reprocessing for the bonus (single-use/reusable table) | Orin | detector | floor pick with the column down |

The two recent robotic scrub nurses fit this plan. One combined oriented-box detection, speech and an LLM calling
robot tools (94 % success, under 1 s latency). The other used SAM 2 and Llama 2. Both are fixed arms. Ours adds the
base, storage and packaging (`comparable_work.md`).

## 6. The final, and how two robots share it

The final splits cleanly by place and time:

| phase | M3 (guide, courier) | manipulator |
|---|---|---|
| arrival | greets, verifies, guides to the preparation room | parked |
| blood sample | the physician puts the tube in the M3's caddy (a scenario actor, allowed) | parked |
| CT | guides, instructs, starts the scan | parked |
| during the scan | carries the sample to the lab, `AnalyzeBlood` | watches `/hospital/lab/state` |
| abnormal values (cTnT > 14 ng/L etc., RB 8.1.3.3) | back to the CT: retract, help the patient off, hand over to staff | CLEAN_OR: wipe, tray, `SetCleanliness(CLEAN)`; checks `xray_active` before entering |
| intervention | parks | waits at the designated position |
| catheter type B | parked | confirm, fetch, peel, present, release, dispose, park (time stops) |

The two parallel phases (sample while scanning, OR while the patient is with staff) are what a fleet buys in the final.

## 7. The fleet and the Hospital Controller interface

**Distribution.**
- The manipulator runs JetPack 7.2 and ROS 2 Jazzy natively. JetPack 7.2 "adds support for the Jetson Orin product
  family" on Ubuntu 24.04 [2nd], and Isaac ROS 4.6 added Orin and JetPack 7.2 [DOC].
- Two ports are needed. xarm_ros2 has a `jazzy` branch [DOC]. AgileX's `ranger_ros2` shows only `humble` [DOC], so
  we port it (est. 2-4 days).
- The TensorRT engines must be rebuilt for TensorRT 10.16.

The M3 stays on JetPack 6.2 and Humble with its vendor stack. Its gateway runs in a `ros:jazzy` container with host
networking and talks to `m3bridge` over HTTP on 127.0.0.1.

We avoid cross-distro DDS everywhere:

- The ROS community says Humble and Jazzy are not compatible [2nd].
- rmw_zenoh keys carry type hashes from Jazzy on [2nd].
- The zenoh DDS bridge documents no cross-distro guarantee [DOC].

**Gateway (one per robot, the only process on domain 0).** Its parts:

- the ExecuteTask server, which maps each order to a job and feeds the job's phases back;
- the pose publisher at 20 Hz, transformed from our map into the arena frame with a fit made on setup day;
- the intent publisher at 1 Hz;
- device clients with timeouts;
- a cache of the latched states, served to the robot over HTTP;
- a log of every order, call and state change.

A mock controller written from RB 8.1.3 takes about a day and is used until the organisers release theirs. Use the
same RMW as the controller PC (Jazzy's default Fast DDS, unless they say otherwise). Test discovery on a busy 2.4 GHz
access point. Nothing safety-related depends on the link: on link loss the robot finishes or holds its step.

**Fleet.** Do not build on Open-RMF for 2027:

- It was "initially developed for the healthcare industry" [DOC], but it needs a central server talking to every robot
  during the run.
- Its Python fleet adapter (free_fleet) is Jazzy-only, has no releases and was tested mostly with Cyclone DDS [DOC].
- Our two robots work in separate zones, so traffic negotiation buys little.

Instead, a small dispatcher on the manipulator: a capability table, assignment by estimated time, and a mutex per door
and narrow corridor. If RB 8.1.1.2 forbids robot-to-robot links, each robot starts its phases from controller events
it can already see:

- the manipulator from the lab's DONE state with abnormal values;
- the M3's sample run from the CT scan leaving IDLE.

That still meets the rulebook's definition of autonomous fleet management, which is "without hands-on interaction by
the team" (RB 6.4.3). Open-RMF remains an option for 2028, or to show its dashboard in the sim.

## 8. The simulation demo and the application video

**The arena file** [measured, `simulation_demo.md`]:

- A Blender glTF 2.0 export: 90 nodes, 71 meshes, ~315k triangles, metres, Y-up.
- No colliders, joints or animations.
- The arena centre sits at x 34.6, z -28.9 in the file's coordinates, and most meshes are scaled 0.01.
- The walls are 3.0 m high in the model, against ~1.4 m in reality.
- Every hinged door leaf has its node origin on a vertical edge, which is where the hinge goes.
- The CT table is already a nine-part chain.

**Recipe** (about one week):

1. GLB to glTF or USD. Isaac Sim 6.0 lists `.gltf`, not `.glb`, and its Metrics Assembler converts units and up axis
   [DOC].
2. Re-root the scene at the arena centre and cut the walls to 1.4 m.
3. Static triangle-mesh colliders.
4. Revolute joints for the doors, prismatic joints for the sliding doors and the CT table.
5. Nav2 map from the occupancy-map tool.
6. A variant scene with one 0.75 m door, the narrowest the rulebook allows.

**Robots.**

- The manipulator: the Ranger Mini, a prismatic column and the Lite 6 from xarm_ros2.
- The M3: our existing twin.
- The patient: a scripted animated character, or HuNavSim's Isaac Sim wrapper [DOC]. omni.anim.people was deprecated
  after Isaac Sim 5.0 [2nd]; check its 6.0 successor.
- Props: NVIDIA's i4h SimReady catalogue (instruments, trays, carts; licensed for non-commercial R&D) [DOC], plus our
  own scans.
- Isaac Sim's bundled `hospital.usd` serves as a second, unseen hospital, to show that navigation generalises [DOC].

**Isaac for Healthcare.** Its hospital workflows (push a cart, tray to cart, scissors into a tray) run on a G1
humanoid and an SO-ARM 101 with GR00T policies, on Linux with 16-48 GB of VRAM [DOC]. They are references and
assets, not our pipeline.

**In the loop.**

- Isaac Sim 6.0 on hekate publishes the robots' sensors with its internal Jazzy libraries, which need a manual
  override [DOC].
- Nav2, the task machines, the gateways and the mock controller run on the AGX Orin under JetPack 7.2: hardware in
  the loop.
- Doors and the CT move in the sim from `JointState`, as the organisers' own `joint_bridge_node` does [RULE].
- Installing Pixi or changing the firewall on hekate is a software change. Ask first (CLAUDE.md).

**Publishing.** The arena is licensed through pre-registration. Publish the scene-building script, our robot models
and the code, not the converted arena.

**Storyboard (2:30-2:45, 1080p, every simulated or sped-up clip labelled, no AI enhancement):**

| time | content | real or sim |
|---|---|---|
| 0:00-0:10 | team, both robots, the arena twin | real + sim |
| 0:10-0:40 | task 1 with an external actor: greet, two identifiers on screen, guide through a door opened via the mock controller, CT on the controller panel; overlay: person track, gap, intent | real (lab mock-up) |
| 0:40-1:05 | task 2: instrument and tube onto the trolley, couple, push through a door | real (push x2, labelled) |
| 1:05-1:25 | task 3: far-to-near wiping with the coverage drawn; three instruments into the tray layout | real |
| 1:25-1:50 | task 4: "Catheter type B, please" -> confirmation -> shelf -> peel -> release on pull -> bin; transcript and the function call on screen | real |
| 1:50-2:20 | the final in the organisers' arena, both robots, phase timeline and controller states | sim |
| 2:20-2:35 | E-stop, lidar unplugged, planner killed -> safe stops | real |
| 2:35-2:45 | success counts, links to code and twin, #Healthineers #HCC@RC27 | text |

## 9. Build order: November to 31 January, then to June

Effort is in person-weeks (pw), estimated. The platform report's hardware dates and its 20 December gate apply. Voice
on the M3 shares the voice report's dates (HRI demo 20 November, late-breaking report 8 December).

| when | software work | effort | done when |
|---|---|---|---|
| 27 Oct-9 Nov | email the organisers (section 10 questions); `hospital_interfaces` + mock controller in a Jazzy container; GLB -> USD with colliders and door joints (on hekate, ask first) | 1.5 pw | `ros2 action send_goal` opens a door in the twin |
| 10-30 Nov | **AMCL + Nav2 on the M3** (prerequisite for everything); the M3's gateway; task 1 machine with person tracking, guide node and dialogue (voice on board per the voice report); M3 in the twin | 4 pw | the M3 guides a person through a lab door, 10 runs counted |
| 1-20 Dec | manipulator bring-up: JetPack 7.2, Jazzy, xarm_ros2, ranger_ros2 port, Nav2 with route graphs, gateway; manipulator URDF/USD; tray and pick-to-trolley on the real arm (table first, then on the base) | 5 pw | **gate 20 Dec**: drives with the arm, picks to a trolley, E-stop chain works |
| 21 Dec-6 Jan | (holidays) sim: task machines for tasks 2-4 against the twin; the final scripted with both robots; the sim expert generating tray episodes | 2 pw | the final runs end to end in the twin |
| 7-24 Jan | lab mock-up (0.75 m door, 0.92 m counter, trolley, shelf); wiping with the spring pad; handover; trolley coupling; filming with counts; safety fault-injection clips | 4 pw | every row of the storyboard filmed |
| 25-31 Jan | TDP, web site, risk analyses, declared room preparations, submit by 29 Jan | 1.5 pw | submitted |
| Feb-15 Mar | robustness (repeat counts on each sub-skill); integrate the released Hospital Controller; label OCR; peel clip; drawer and door strategies; external-actor dialogue tests in English and German | ongoing | counts improve; nothing new filmed |
| 15 Mar-mid May | after selection: the 850 + F/T sensor (force-controlled wiping, drawers, doors); the VLA layer for the technology bonus (tray or shelf, trained in the twin); full rehearsals of the final; spares | ongoing | hardware freeze mid-May |
| June | test slots: map, pathways and the arena-frame fit on setup days | - | - |

**What must be real in the application, and what can be simulated.**

- **Real:**
  - task 1 on the M3 with an external person;
  - pick to trolley; the tray layout; wiping; the handover;
  - the gateway against the mock controller;
  - E-stop and fault injection;
  - success counts.

  The FAQ gives "additional consideration" to a working prototype. The real clips are what make the application
  credible.
- **Simulated (labelled as such):**
  - the full final in the official arena;
  - the two-robot coordination;
  - doors and the CT driven by the controller;
  - the unseen hospital;
  - hundreds of runs for statistics.
- **Fallback if the 20 December gate fails:** the M3 real, the Lite 6 on a table stand real (pick, tray, wipe,
  handover), and the manipulator simulated. That is weaker, but still eligible under the FAQ.

## 10. Risks

| risk | likelihood | effect | mitigation |
|---|---|---|---|
| Nav2 not yet running on any of our robots | certain today | blocks tasks 1-4 | first item of November; the twin for tuning |
| manipulator bring-up slips (Jazzy port, coupling, column) | high | weak video | 20 Dec gate; table-stand fallback; the twin |
| speech in a noisy hall, unknown language | high | task 1 and 4 points | closed questions, screen buttons, mic array, en + de; ask the language |
| patient track lost while leading | medium | task 1 stalls | rear camera + lidar legs + ReID gallery + spoken re-acquisition |
| robot-to-robot links forbidden | medium | no direct fleet link | event-driven coordination via controller states |
| Hospital Controller released late or different | medium | gateway rework | build against the mock from RB 8.1.3; isolate it in the gateway |
| Lite 6 too weak for doors, drawers, peel force | high | task 4 partial | open shelves first, base pulls, robot-mounted peel clip; 850 after selection |
| wiping not accepted as "disinfection" by physicians | medium | task 3 points | stated procedure, overlap, before/after image, removal test with a tracer |
| arena data licence forbids publishing the twin | medium | less SF/T | publish scripts and robot models only; read the terms |
| GPU time on hekate (someone's job) | medium | slow sim | Dionysos as second machine; ask before heavy runs |
| team capacity (RSS, HRI demo, voice report) | high | everything late | the build order above; cut patient transport, drawers, VLA before 31 Jan |

## 11. Questions to add to the organisers' email

These come on top of the platform report's list:

1. Topic names, QoS and `frame_id` for the pose and the intent.
2. How several robots of one team are addressed, and who receives HEART_ATTACK.
3. The order type that starts task 2, and whether the sample goes into the analyser physically.
4. Whether our robots may talk to each other during a run.
5. The controller's RMW and discovery settings.
6. The door ids, and which doors are physical.
7. The patient's and the operator's language.
8. What "designated area" means in task 3; the tray layout and the tray model.
9. Whether the robot may stay in the CT room while `radiating` is true.

## Conclusion

The Healthcare Challenge rewards a team that does many sub-skills reliably and safely, more than one that solves a
single task brilliantly. Our software already covers the hardest perception and precision parts: detectors trained
on synthetic data, identity from a few views, millimetre picks, policies on the Jetson and twins of our robots. Three
things are missing: navigation on the robots, the voice stack on board, and an arm with force.

The approach:

- one explicit task machine per robot, with every state spoken, shown and logged;
- Nav2 with pathway graphs and a guide node that sets the speed limit from the patient's gap;
- the kit recipe applied to instruments;
- a spring pad for wiping, a magnet for flat steel, and a peel clip on the robot;
- a Jazzy-native manipulator, and a Jazzy gateway on the Humble M3;
- a small dispatcher, with a controller-state fallback if direct links are forbidden.

The application should show these sub-skills real, with counts, and the final in a twin of the official arena driven
by the robot's own Jetson. Start this week with the organisers' questions, the mock controller and the arena
conversion. Start November with Nav2 on the M3, because everything else stands on it.

## Method and caveats

- **Sources.**
  - Rulebook V1.0, read in full (text extracted locally).
  - The official FAQ and overview, read in a browser on 2026-10-27: no newer rulebook or controller visible, and no
    patient language stated.
  - The arena GLB, measured with a stdlib script.
  - Primary pages of Nav2, Isaac ROS, Isaac Sim 6.0, Isaac for Healthcare, xArm SDK, xarm_ros2, ranger_ros2,
    free_fleet, Open-RMF, Ultralytics, Parakeet and arXiv abstracts.
- **Search budget.** 24 web searches. Several facts rest on search summaries [2nd]:
  - JetPack 7.2 details;
  - Isaac ROS 4.6 being Jazzy;
  - Humble/Jazzy incompatibility;
  - the UFACTORY F/T sensor's Lite 6 incompatibility;
  - the route server's defaults;
  - the Isaac Sim people-simulation deprecation.
- **Blocked pages.** docs.ros.org was behind a bot check, so the static-peer discovery setting is from background
  knowledge.
- **Effort figures** are estimates, not measurements.
- **Unknowns** that could change this plan:
  - the scoring chapter;
  - the Hospital Controller's code;
  - the object list and trolley model;
  - the language;
  - the rule on robot-to-robot links.
