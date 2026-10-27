# Healthcare Challenge @ RoboCup 2027: what the rules and the tasks demand (state of 27 October 2026)

Scope: what the rulebook V1.0 (27.08.2026, `C:\Users\Mys\Desktop\Siemens sutaz\2026_09_01_Rulebook_released_V1.pdf`,
73 pages, read in full) and the official site (read in a browser on 2026-10-27; the site is JavaScript-rendered, so
plain fetches return an empty page) require from a team and its robots, what each task physically demands, and what is
still unknown. Arena dimensions come from the organisers' GLB (`2026_09_01_RoboCup_HCC_Arena_V1 1.glb`, Blender glTF,
metres, Y-up), measured by me as world-space bounding boxes of its nodes. Platforms, safety and compute are in the
sibling notes (`platforms_compute_costs.md`, `safety_concept_standards.md`); comparable leagues in
`what_wins_similar_leagues.md`. Rulebook references are given as "RB section".

Labels: **[RB]** = rulebook V1.0; **[SITE]** = official site / FAQ, read 2026-10-27; **[MAIL]** = the organiser's
email to the lab of 2026-10-02 as recorded in the project memory (not a public source); **[GLB]** = my measurement of
the arena model; **[2nd]** = secondary source or search summary.

## 1. Where things stand on 27 October 2026

### Takeaway
V1.0 is still the only rulebook. The scoring chapter (RB 7) is withheld, the qualification "details" are promised for a
later version, and the Hospital Controller's source code is not public yet. The dates are firm: application
**31.01.2027**, selection notice **15.03.2027**, competition in the RoboCup week **15-21.06.2027** in Nuremberg (the
Robotics Institute Germany gives the challenge itself as 17-20 June).

### Cited Findings
- Timetable on the official overview: 01.09.2026 rulebook release; 31.01.2027 application deadline; 15.03.2027 "All
  applicants will be informed whether their team has been selected"; 15.-21.06.2027 competition [SITE] —
  [overview](https://www.robocup2027.event.siemens-healthineers.com/blocks/overview)
- Pre-registration is needed to download the rulebook and the 3D environment "due to the licensing terms of the 3D
  data"; it does not commit the team and "does not replace the official RoboCup 2027 registration" [SITE] —
  [FAQ](https://www.robocup2027.event.siemens-healthineers.com/blocks/faq). The download area itself needs a login; I did
  not log in, so **a newer file there (V1.1) cannot be excluded** — check it.
- "Updated versions of the dataset, Rulebook, FAQ, simulation environment, or supporting documents may be released ...
  Teams are responsible for using the latest officially published version" [SITE] — FAQ.
- RB 7: "This chapter is intentionally withheld ... to prevent teams from being constrained or biased in their design";
  only the ranking logic is given: tasks 1-4 summed, top 3 to the final, final score reset to zero [RB 7.1].
- RB 8.1.1: "The Hospital Controller will be released as source code with an upcoming version of this rulebook." A web
  search for `hospital_interfaces` / `hospital_controller` found nothing public (2026-10-27).
- RIG: "Nuremberg, Germany – June 17 – 20, 2027", EUR 170,000 total prize money; NVIDIA provides "developer hardware,
  computing resources, and technical support" [2nd] —
  [RIG announcement](https://robotics-institute-germany.de/healthcare-challengerobocup-2027-benchmarking-physical-ai-for-healthcare/),
  [RIG challenge page](https://robotics-institute-germany.de/healthcare-challenge-robocup-2027/)
- The RoboCup 2027 site gives 17-21 June at the Nuremberg Exhibition Center, no fees or deadlines yet —
  [2027.robocup.org](https://2027.robocup.org); RoboCup Germany gives 15-21 June —
  [robocup.de](https://robocup.de/en/events/world-cup-2027)

### Gaps
- RoboCup 2027 registration fees: not found (2027.robocup.org, robocup.de, RoboCup 2026 pages); budget from past
  editions and ask.

## 2. What the application must contain and how teams are chosen

### Takeaway
The qualification material is a **Team Description Paper** (Overleaf template, English, published in the RoboCup
Symposium proceedings if accepted), a **promotional video** (no AI generation or enhancement), a **team web site**, a
**risk analysis per robot**, the **declared room preparations**, and optionally designs or drawings. No prototype is
required, but the FAQ says a working robot or advanced prototype "may receive additional consideration", and simulation
demonstrations are "strongly encouraged". Selection asks for feasibility, a realistic plan and confidence that a working
robot exists by June 2027.

### Cited Findings
- Participation steps: pre-registration; "submission of qualification material, including a team description paper, a
  promotional video (AI-generation or enhancement of video is prohibited) and possibly additional material like designs
  or drawings"; final registration via RoboCup (qualified teams only); safety assessment; competition [RB 1.3.2].
- Evaluation criteria: TDP; relevant scientific contribution/publications; professional quality of robot and software;
  novelty of approach; relevance to healthcare and medical robotics; performance in previous competitions; promo video;
  team web site [RB 1.4.1].
- TDP must hold: team name, members incl. leader, web-site link, contact; body: hardware with an image of the robot(s),
  "safety systems used on the robot, including emergency stop procedure", functional and software architectures,
  innovative technology, "the need to adapt objects or the environment", applicability to medical tasks; English;
  Overleaf template [RB 1.4.2]. Multi-institution teams "explicitly welcome" — describe how the team is organised.
- Video: show "autonomous navigation, picking, and placing", ideally what the robot sees and its current plan; 1080p
  16:9; audio not required; hashtags #Healthineers and #HCC@RC27; it is replayed at the event [RB 1.4.3].
- Risk analysis "as part of their application ... for every robot intended for deployment"; it should "highlight novel
  aspects of the system and describe the team's safety concept" [RB 6.4.8]. A reconfigured robot counts as a separate
  robot with its own risk analysis [RB 6.4.3].
- "All planned room preparations must be declared as part of the team's application, specifying the type, location, and
  purpose of each modification" [RB 6.4.4].
- FAQ: "A physical robot or hardware prototype is not required during the application phase. However, the application
  must convincingly demonstrate that the proposed robot ... can be developed and operational by June 2027. Teams that
  already have a working robot or an advanced hardware prototype may receive additional consideration"; simulation
  demonstrations "are therefore strongly encouraged"; the FAQ names "Gazebo or similar simulation tools" [SITE].
- FAQ selection list: "The technical feasibility of the proposed solution. A realistic development plan. Confidence that
  a functional robot will be available by June 2027. Relevant simulation results, concepts, or prototype
  demonstrations" [SITE].
- FAQ: all four challenges count; partial solutions score; simulation-only teams cannot compete; no framework mandated;
  no team-size limit; one person in one team only; teams made only of Siemens Healthineers staff are not eligible [SITE].
- Costs: Siemens Healthineers covers no travel, accommodation, shipping, insurance, visa or RoboCup fees [SITE].
  Prizes 100k/50k/20k EUR paid by the European RoboCup Association as a donation to the university, 75 % in 2027 and
  25 % after taking part in 2028 [SITE]. Every admitted team receives the same NVIDIA package [SITE].
- Questions go to robocup@siemens-healthineers.com before admission; no official forum; hashtag #HCCatRC2027 [SITE].

### Inferences
- "Performance in previous competitions" is a criterion we score zero on; publications, open source and the quality of
  the video and TDP must carry it.
- The lab's earlier email (2026-10-02) already established that compact ROS 2 robots "can certainly be competitive",
  fleets, reconfiguration between tasks and declared fixtures are fine, sponsored hardware is allowed, and no Jazzy
  migration is needed [MAIL].

## 3. The score formula and which levers we control

### Takeaway
Score per task = (P x M + T) x SF with M = 1 + RPL + MGL [RB 4.2.1]. M multiplies every task, so a robot that is
general (few fixtures, standard furniture, external people's voices, fleet coordination, deterministic safety control)
and medically mature (single-fault safety demonstrated, cleanable design, a 60601/13485 concept) gains on all four
tasks at once. T rewards demonstrated generalisation (VLA, LLM commands, map-free operation, sim-to-real with a
published simulation, open source, explainability). SF is granted once the team "demonstrably" shares knowledge.

### Cited Findings
- RPL raisers: standard hospital furniture instead of a dedicated magazine; materials not adapted or pre-opened;
  working without environment adaptation / in an unknown arena; "real-time and deterministic control of
  safety-relevant functions"; "fleet management and the coordination of several cooperating robots"; spoken commands
  from external persons [RB 4.2.2]. Fixtures "may raise the raw task points, but they lower the Robot Power Level"
  [RB 4.2.2].
- MGL raisers: cleanability and hygienic design; single-fault safety and fail-safe behaviour; clinical extensibility;
  conformity concept for IEC 60601 and ISO 13485 — "actual certification is not required; a concept ... is sufficient"
  [RB 4.2.3]. A 3D-printed part can get the full cleanability score with a written manufacturing concept [RB 4.2.3.1].
- Single-fault safety is **mandatory**; a sound concept earns the base contribution, a demonstrated implementation
  "earns more"; "Redundant hardware such as dual processors is explicitly not required" [RB 4.2.3.2] (detail in
  `safety_concept_standards.md`).
- T examples: VLA or other foundation models generalising to unseen objects; LLM-based command understanding beyond
  fixed sets; map-free or zero-shot operation; "sim-to-real approaches accompanied by a published simulation";
  reusable open source; explainability and introspection [RB 4.2.4].
- SF examples: open-source release with documentation, preprints, FAIR datasets or simulation environments, workshops,
  evidence of reuse; "A single well-documented open-source release or a clearly written technical report is valued
  equally to a peer-reviewed publication" [RB 4.2.5].
- Dual-track rating: a medical panel of physicians and the technical referees, "on equal footing" [RB 6.1.5]. Gradual
  autonomy: help, hints and environment changes are allowed but lower the score [RB 6.1.1-6.1.3]. Doors: by the robot,
  by the patient, or by data (Hospital Controller); the method is reflected in the score, physical opening ranks higher
  [RB 6.4.9, 8.1.1.1].

### Inferences
- Several T and SF items are things the lab already does: ACT/VLA policies trained in Isaac Sim twins (sim-to-real), a
  local LLM that turns speech into one function call (natural-language commands), the console's live view and logs
  (explainability), an open repository. Publishing the arena twin with our robot models would serve T, SF and the
  application video at once.
- A second robot that is coordinated autonomously is worth more than its task contribution, because "fleet
  management" raises RPL, which multiplies every task.

## 4. Hard constraints on the robots

### Takeaway
Any platform must pass a 0.75 m x 1.95 m door in transfer configuration, never tip, run on its own battery at no more
than 120 V DC with no cable power during runs, carry a hardware emergency stop that kills motors and hazardous
functions, talk to the Hospital Controller over the arena's 2.4 GHz WiFi, and do all computing on board.

### Cited Findings
- Door: "In its smallest variant, such a door is 1.95 m high and 0.75 m wide"; no clearance behind doors is guaranteed
  [RB 4.1.2.1]. Floor load at most 500 kg/m2; the robot "cannot tip over, even in extreme poses"; no rails [RB 4.1.2.2].
- Self-stabilising robots (humanoids) over 50 kg need a full safety concept with an FMEA [RB 4.1.2.4]; hydraulics over
  750 ml need a leak test [RB 4.1.2.3].
- "Only internal voltages up to a maximum of 120 V Direct Current (DC) may be used"; own energy source; no cable-bound
  supply except hot-plug DC charging; an on-board AC charger must not be used in competition and needs an electrician's
  protocol in the team area [RB 4.1.3.1].
- E-stop: "an emergency-stop device that deactivates the motors and any hazardous functions"; for commercial platforms
  without one, waived if the manufacturer supplies a CE declaration of conformity or equivalent [RB 4.1.3.2].
- Interfaces: `geometry_msgs/PoseStamped` at 10-30 Hz, `std_msgs/String` intent at 1-2 Hz, accept and execute tasks 1-4
  and the finale [RB 4.1.4.1]. The provided ROS 2 PC runs Ubuntu 24.04 native, ROS 2 Jazzy, `ROS_DOMAIN_ID=0`, 2.4 GHz
  WiFi; it "must not be used as an edge server, computational resource, or for any team-specific processing tasks";
  "failures or delays caused by issues in communication bridges ... shall be considered as equivalent to any other robot
  malfunction" [RB 4.1.4.1].
- During a run the Hospital Controller is "the only interface that team members may use to interact with their robot";
  "no remote operation, external network connection, or other technical communication channel with the robot is
  permitted" [RB 8.1.1.2].
- Up to 5 active mobile robots and 10 stationary robots or machines per team [RB 6.4.3].

### Inferences
- The 120 V DC and no-cable rules exclude arms whose controller needs mains AC (unless an inverter inside the robot is
  accepted, which would put 230 V AC inside the robot — I would not rely on that). Prefer arms with native DC supply.
- "Other technical communication channel with the robot" may also cover robot-to-robot links inside our own fleet (for
  example an access point on a robot). This decides whether a fleet can coordinate at all and must be asked.

## 5. The Hospital Controller and the Humble/Jazzy question

### Takeaway
The robot serves the `ExecuteTask` action (the controller is the client) and calls the device actions and services
(doors, CT table and scan, lab analysis, OR cleanliness). Our robots run Humble on JetPack 6; the organisers say a bridge
is fine, but cross-distribution ROS 2 is not guaranteed, so the safest design is one small Jazzy gateway process per
robot that alone touches domain 0 on the arena WiFi, with the robot's own ROS graph kept local.

### Cited Findings
- Device interface: `/hospital/doors/operate` (action OperateDoor; states CLOSED, OPENING, OPEN, CLOSING, OBSTRUCTED,
  percent open), `/hospital/lab/analyze` (AnalyzeBlood; READY, ANALYSING, DONE; cTnT/cTnI/CK-MB), `/hospital/ct/table/move`
  and `/hospital/ct/scan/run` (MoveTable, RunScan with START/STOP/RETRACT), `/hospital/operating_room/set_cleanliness`
  and `get_state` (CLEAN/DIRTY, `xray_active` = do not enter), `/hospital/execute_task` (ExecuteTask) [RB 8.1.3].
- ExecuteTask "is served by the robot itself"; order types CLEAN_OR, COLLECT_LAB, GUIDE_CT, HEART_ATTACK,
  SURGICAL_ASSIST; feedback phases QUEUED, NAVIGATING, EXECUTING, DONE; goal fields `value`, `target_location`, `task_id`
  [RB 8.1.3.6]. Key states are latched (transient local) [RB 8.1.3.1].
- The organisers' simulated robots "teleport between fixed room waypoints" and publish pose and TF — a template for our
  mock [RB 8.1.4.2].
- "technically there are no guarantees about inter-distribution compatibility" (ROS Discourse) [2nd] —
  [discourse.openrobotics.org](https://discourse.openrobotics.org/t/incompatability-between-distributions/43747); type
  hashes were introduced from Iron on and services/actions are composed of several generated types (REP-2016) [2nd] —
  [REP-2016](https://github.com/ros-infrastructure/rep/blob/eabe46a6aa7f039aca95e8f2f65ff92e6c1a0466/rep-2016.rst),
  [ROSCon 2023 talk](https://roscon.ros.org/2023/talks/ROS_2_Types_On-the-wire_Type_Descriptions_and_Hashing_in_Iron_and_onwards.pdf).
- rmw_zenoh embeds type hashes in its keys, so Humble and Jazzy nodes see each other but do not exchange data over it
  [2nd] — [Stereolabs zenoh guide](https://www.stereolabs.com/docs/integrations/ros-2/using-zenoh-as-middleware).

### Inferences
- The ExecuteTask list has no explicit order type for task 2 (object transport); COLLECT_LAB is the nearest. Ask.
- In the printed ExecuteTask definition the comments for result and feedback look swapped (the feedback is
  `current_phase`, `progress`, `task_id`). Build against the released `hospital_interfaces` package, not the PDF.
- Until the code is released we can write our own `hospital_interfaces` and a mock controller from RB 8.1.3 in a Jazzy
  container (about a day of work) and test the gateway against it.

## 6. What each task physically demands

### Takeaway
Task 1 needs navigation, people tracking and dialogue but no manipulation (doors can go through the controller or the
patient). Tasks 2-4 need an arm working at table height (0.9-1.03 m), shelves up to 1.92 m, a bin at 0.61 m, a base
that can push a trolley, and wiping with contact force. A table-height mobile manipulator is the only single machine
that covers 2, 3 and 4.

### Cited Findings (rulebook)
| task | subtasks | physical and interaction demands |
|---|---|---|
| 1 Patient guidance to CT [RB 2.1] | greet and confirm name/appointment; guide through doors at a pace for reduced mobility; instruct positioning on the CT table; instruct to take off shoes; operate the scanner through the controller; escort out; return to parking | people detection and following, speech dialogue (likely with an external actor for RPL), doors (robot / patient / controller), patient pathway [RB 6.3.1]; no adverse events in 2027 |
| 2 Object transport [RB 2.2] | pick small clinical objects (instruments, samples, supplies) and place them onto a mobile trolley; push or pull a trolley between rooms; optional patient transport in a wheelchair or bed for a bonus | arm payload for small objects, placement onto a trolley top, trolley coupling and pushing force, doors; fully autonomous, no teleoperation [RB 2.2.5]; no dynamic obstacles in 2027 |
| 3 OR cleaning + tray [RB 2.3] | mechanically disinfect a designated area ("wiping, scrubbing", no UV-only); never hover over or re-enter cleaned areas; move instruments from the trolley to a tray in a predefined layout; return to parking | contact force control, coverage planning, precise placement, set OR to CLEAN via the controller |
| 4 Surgical assistance [RB 2.4] | understand and confirm verbal requests; find the item by look or label in storage; present it so that only non-sterile parts are touched; dispose of packaging; open packaging (technical challenge); handle dropped or contaminated items; reusable item to reprocessing for a bonus | cabinets, drawers and shelves of several kinds [RB 5.1.4]; the robot must not enter sterile areas [RB 2.4.4] |

- Contact: robots must not initiate contact with humans unless the task asks; a patient may lean on the robot during
  guidance and the robot must "safely bear the applied forces without causing harm or loss of stability" [RB 6.1.4].
- Teams must bring "one set of all objects used in the competition" except arena equipment (trolley, bed, shelves)
  [RB 1.4.4] — we need the object list.

### Cited Findings (forces, secondary)
- Accessibility codes cap interior door opening force at 5 lbf (22.2 N); other swinging doors up to 30 lbf (133 N) to
  set in motion and 15 lbf (67 N) to open fully (US model codes) [2nd] —
  [idighardware.com](https://idighardware.com/2009/02/door-opening-force/). European EN 12217 classes: not found.
- Trolley studies measured initial and sustained push forces on 100-300 kg trolleys; a caregiver table lists 51 lb
  initial and 30 lb sustained (~230 N and ~130 N) [2nd] —
  [Duke push/pull guidance](https://www.safety.duke.edu/sites/default/files/Pushin%20pulling%20moving%20equipment%20on%20wheels.pdf),
  Bennett et al. via [Rhodes University](https://commons.ru.ac.za/vital/access/Person/Bennett,%20Anthea%20I).
- Wiping force: no published target found; one scrubbing robot caps its normal force at about 12 N [2nd] —
  [IMECE poster](https://imece.secure-platform.com/a/solicitations/261/sessiongallery/21466/application/172734).

### Inferences
- A light instrument trolley needs tens of newtons to start; a loaded one or a bed more than 100 N. That is a job for a
  base of 50+ kg with a fixed coupling, not for an arm with 0.6 kg payload.
- Wiping at 5-15 N (my estimate from the above) through an arm needs either a wrist force sensor, joint torque sensing
  or a passive compliant tool (spring-loaded pad) — the passive pad is the cheap route.
- Opening real cabinet doors and drawers at 20-60 N exceeds the Lite 6 (0.6 kg payload); a 3-5 kg arm or a
  hook-and-base-pull strategy is needed.

## 7. The final and the multi-robot rule

### Takeaway
The final chains all four tasks without pause or human intervention: guide the patient, carry the blood sample to the
lab during the scan, clean and prepare the OR, wait, then fetch "catheter type B", hand it over and dispose of the
packaging. A fleet is allowed only if every transition is autonomous. Reaching the final needs a top-3 sum over tasks
1-4.

### Cited Findings
- No exchange of robots or reconfiguration by the team during the final; interventions defined as anything that
  "directly affects the robot's hardware, software, or configuration" [RB 3.1]. Physical support of the patient "is
  permitted and may result in additional points" [RB 3.2.1]. Full completion is not expected ("a north star")
  [RB 3.2.2]. Parc fermé applies [RB 6.4.1].
- Multi-bot in the finals: "only permitted if no manual intervention by the operator is required for the transition
  between configurations. Autonomous fleet management must be demonstrated" [RB 6.4.3].
- Ward management dispatches HEART_ATTACK as one order [RB 8.1.2.5].

### Inferences
- Two robots can work in parallel in the final (sample to the lab while the other prepares the OR), which may save
  time — if robot-to-robot communication is allowed (section 4).

## 8. The arena from the GLB

### Takeaway
The model is a 14 x 11 m walled area (16 x 13 m floor) with 10 hinged doors of 0.95-0.97 m and two sliding doors of 1.5
and 2.0 m; work surfaces are at 0.9-1.03 m, storage cabinets are 1.92 m tall and 0.65 m deep, the bin top is at 0.61 m.
The real arena will differ (venue-dependent, 1.4 m walls, maybe half-height).

### Cited Findings [GLB]
- Floor 16.0 x 13.0 m; walls 14.0 x 11.0 m, modelled 3.0 m high (rulebook: ~1.4 m real walls, opaque, reflective edge
  protectors possible, no glass, corridors ~2.5 m [RB 5.1.1-5.1.2]).
- Doors: `Door_01`...`Door_10` leaves 0.95-0.97 m wide, 2.23 m high; `SlidingDoor_01` 1.50 m, `SlidingDoor_02` 2.00 m.
- Storage cabinets 1.08 and 1.30 m wide, 1.92 m high, 0.65 m deep; lab base cabinets 0.92 m high (the counter), wall
  cabinets from 1.60 to 2.32 m; a desk top at 0.99-1.03 m; bin 0.61 m high; bed 2.26 x 1.16 m, 1.04 m high; reception
  desk 1.12 m; CT (NAEOTOM Alpha) gantry ~2.0 m; a mobile C-arm (CIARTIC Move) and an Artis table in the OR.
- Rooms named in the model: Hall, Lab, OR, Storage, Office, Changing room, Reception; the rulebook figure adds scanning
  room, phlebotomy room, entrance and waiting area, clinical corridor [RB 5.1.3, Fig. 5.2].
- An empty helper node `stretch_helper` sits at 0.92 m height — probably a modelling aid, not a robot.

### Inferences
- An arm whose shoulder can travel between ~0.8 and ~1.3 m reaches the counters, desks, trolley tops and the middle
  cabinet shelves; the top 0.4 m of the 1.92 m cabinets stays out of reach for most compact platforms.
- 1.4 m walls mean cameras above that height see the hall and the crowd; lidars should sit low and localisation should
  not trust features above the walls.

## 9. Open questions for the organisers (robocup@siemens-healthineers.com)

1. Is there a rulebook newer than V1.0 in the download area, and when will chapter 7 and the qualification details be
   published? Is there a page limit for the TDP?
2. When will the Hospital Controller source be released? Can teams get a pre-release for testing?
3. Which ExecuteTask order type starts task 2? Must the sample be physically put into the lab analyser?
4. Does RB 8.1.1.2 forbid communication between our own robots (own WiFi or access point on a robot) during a run?
   Is a 5 GHz link of our own allowed?
5. The object list for "one set of all objects" (instruments, sample tubes, packaging, tray and its layout, trolley
   model); the door types and their opening forces; the language of the patient dialogue.
6. Does a stationary arm placed by the team count as a "fixture" (lower RPL) or as a robot?
7. What does the NVIDIA package contain and when does it arrive?
8. Is the floor level (thresholds, ramps, cable covers)?
