# Build one mobile manipulator from an AgileX base, a lift column and our Lite 6, keep the M3 as patient guide, and buy the bigger arm only after selection

**Bottom line.** The lab should apply to the Healthcare Challenge @ RoboCup 2027 as a two-robot team: a self-built
mobile manipulator that attempts all four tasks and the final (AgileX Ranger Mini 3.0 base, a 24 V lift column, the
UFACTORY Lite 6 we own, a Jetson AGX Orin 64 GB, a hard-wired E-stop chain with an independent watchdog), and the M3 Pro
as patient guide and sample courier, which also earns the rulebook's fleet bonus. Stage 1 costs about EUR 19-25k
excl. VAT (EUR 14-20k on the cheaper Tracer 2.0 base). Buy it in November: the base and the safety parts stay useful to
the lab if we are not selected, and the FAQ gives "additional consideration" to teams with a working prototype, so the
January video should show the real robot. Upgrade the arm after the 15 March selection to a UFACTORY 850 (~EUR 9-11k,
same software) or, if sponsored, a Kinova Gen3. A Hello Robot Stretch 4 (~EUR 40-45k landed) is the fallback if our
build is not driving with its arm by 20 December. The application should be won by the safety concept, an Isaac Sim
twin of the arena, on-board foundation models and evidence of repeated runs. Our hardware is not what will win it.

Written 2026-10-27 for the lab's decision. Evidence and links are in the four notes in
`research_notes/Healthcare Challenge 2027 platform/`: `rules_and_task_demands.md`, `platforms_compute_costs.md`,
`safety_concept_standards.md`, `what_wins_similar_leagues.md`. "RB x.y" refers to the rulebook V1.0 (27.08.2026).
Prices were read on 2026-10-27 and are labelled list, reseller or est. (estimate); none is a quote to us yet.

## 1. The rules reward a general, safe, well-documented robot, and the scoring chapter is still withheld

V1.0 is still the only public rulebook. Its scoring chapter is withheld "to prevent teams from being constrained ...
in their design" (RB 7), the qualification details are promised for a later version, and the Hospital Controller's
source has not been released (RB 8.1.1). The download area needs a login. I did not log in, so the lab should check
there for a V1.1. The official timetable, read on 2026-10-27, gives the application deadline 31.01.2027, the selection
notice 15.03.2027 and the competition 15-21.06.2027 in Nuremberg ([overview](https://www.robocup2027.event.siemens-healthineers.com/blocks/overview)).
The Robotics Institute Germany gives the challenge days as 17-20 June.

What we know of the scoring already decides the design. Each task scores (P x M + T) x SF with M = 1 + RPL + MGL
(RB 4.2.1). M multiplies all four tasks:

- **Robot Power Level (RPL)** rises with generality: standard furniture instead of magazines, unadapted materials, an
  unknown arena, "real-time and deterministic control of safety-relevant functions", "fleet management and the
  coordination of several cooperating robots", and commands from external people. Fixtures raise raw points but lower
  RPL (RB 4.2.2).
- **Medical Grade Level (MGL)** rises with cleanable design, single-fault safety (mandatory as a concept, more points if
  demonstrated) and a concept for IEC 60601 and ISO 13485. Certification is not required (RB 4.2.3).
- **Technology bonus (T)** rewards demonstrated generalisation: VLAs or foundation models on unseen objects, LLM
  command understanding, map-free operation, "sim-to-real approaches accompanied by a published simulation", open
  source, explainability (RB 4.2.4).
- **Scientific community factor (SF)** is granted once the team demonstrably shares its work; one documented
  open-source release counts as much as a paper (RB 4.2.5).

A panel of physicians rates clinical appropriateness on equal footing with the technical referees (RB 6.1.5).

The hard constraints are few but strict. The robot must pass a 0.75 m x 1.95 m door and never tip over. It uses at most
120 V DC from its own battery, with no cable power during runs. A hardware E-stop must deactivate motors and hazardous
functions; this is waived only for commercial robots with a CE declaration. The robot talks to the Hospital Controller
(ROS 2 Jazzy, domain 0, 2.4 GHz WiFi): it serves `ExecuteTask` and publishes `PoseStamped` at 10-30 Hz and an intent
string at 1-2 Hz. During a run nothing else may connect to it: "no remote operation, external network connection, or
other technical communication channel" (RB 4.1, 8.1.1.2). All computing therefore runs on the robots. Bridge failures
count as robot failures. The organisers told us on 2026-10-02 that Humble is fine behind a bridge. Cross-distribution
ROS 2 is not guaranteed, though, so each robot should get one small Jazzy gateway that alone touches domain 0, while
its own graph stays local.

## 2. Tasks 2-4 and the final need one mobile manipulator at table height; task 1 needs none

| task | what it physically demands | who covers it in our plan |
|---|---|---|
| 1 Patient guidance to the CT | find and greet the patient, confirm identity, guide at a reduced-mobility pace through doors (by the robot, the patient or the controller), instruct (shoes off, lie on the table), run the CT through the controller, escort out, park; patient pathway | M3 Pro with a 1.1 m mast (screen, microphone, speaker), or the manipulator's head |
| 2 Object transport | pick instruments, sample tubes and supplies and place them on a mobile trolley; push or pull the trolley between rooms (tens to >100 N); optional patient transport in a wheelchair or bed for a bonus | manipulator: arm for the objects, base + coupling for the trolley; no patient transport in 2027 |
| 3 OR cleaning + tray | wipe a designated area mechanically without passing over cleaned parts; lay instruments onto a tray in a set layout; set the OR to CLEAN | manipulator: compliant wiping pad, coverage plan, sideways base motion along the table |
| 4 Surgical assistance | understand and confirm spoken requests; find items in cabinets, drawers and shelves; present them sterile-side-out; dispose of packaging; open packaging; deal with dropped items | manipulator: voice + LLM, shelves up to ~1.5 m, bin at 0.61 m; packaging opening probably needs a declared fixture |
| Final | all four chained without intervention or reconfiguration; fleets only with autonomous transitions | manipulator alone, or with the M3 carrying the sample in parallel |

The arena model gives the working geometry (measured from the organisers' GLB): 14 x 11 m, doors of 0.95-0.97 m
(sliding 1.5 and 2.0 m), counters at 0.92 m, desks at 0.99-1.03 m, storage cabinets 1.92 m tall and 0.65 m deep, the
bin at 0.61 m and a 1.04 m bed. The real walls are ~1.4 m high, with reflective edge protectors and no glass (RB 5.1).
An arm whose shoulder travels between ~0.8 and ~1.3 m reaches counters, trolley tops and the middle shelves. The top of
the 1.92 m cabinets stays out of reach for any compact robot. Opening real cabinets and pushing trolleys takes 20-130 N
(door codes cap interior doors at 22 N; push studies give 100-230 N for loaded carts). That exceeds the Lite 6's 0.6 kg
payload, so it has to come from the base or from a stronger arm.

## 3. A self-built AgileX + lift + UFACTORY robot beats buying for this lab, but only narrowly

| option | T1 | T2 | T3 | T4 | final | cost (excl. VAT) | lead time | main risk |
|---|---|---|---|---|---|---|---|---|
| A. Owned robots only: M3 guide, R2 couriers, Lite 6 on a cart placed in a room | yes (M3) | objects at a station only, no trolley | only if the station stands at the OR table | handover from a station only | barely (no repositioning allowed) | EUR 1-3k est. | 2-4 weeks | uncompetitive; fixtures cut RPL; no E-stops today |
| **B. Ranger Mini 3.0 + 24 V column + Lite 6 + AGX Orin, with the M3 (recommended stage 1)** | yes | yes (light objects, trolley via coupling) | yes (passive pad, light force) | yes (light drawers, shelves ≤1.5 m) | yes, one robot | **EUR 19-25k** (base EUR 12,480 reseller list) | parts 3-6 weeks, integration 8-10 weeks | integration workload; Lite 6 payload and reach |
| C. B with a UFACTORY 850 (5 kg, 850 mm, DC box) (recommended stage 2) | yes | yes | yes, firmer wiping | yes, real drawers and doors | yes | +EUR 9-11k (USD 8,399-8,999 list + DC box) | "on demand", weeks | heavier arm on a column: stability interlock |
| D. B with a Kinova Gen3 (4 kg, 902 mm, joint torque sensing) | yes | yes | yes, force-controlled | yes, safest handover | yes | EUR 45-60k est. (USD 33,500 reseller) | unknown | price; only with a sponsor |
| E. B on a Tracer 2.0 (24 V, 0.61 m wide, differential) | yes | yes | yes, the arm must cover lateral motion | yes | yes | EUR 14-20k (base EUR 7,500 reseller list) | 3 weeks | 0.61 m in 0.75 m doors; "flat ground only" |
| F. Hello Robot Stretch 4 + Orin NX + gripper + dock, with the M3 | yes (add a tablet) | small objects yes, trolley doubtful | yes | yes up to ~1.2 m | yes | USD 35,935 list, ~EUR 40-45k landed est. | 2-4 weeks | no CE; runstop acceptance; 2.5 kg at reach; little hardware novelty |
| G. PAL TIAGo Pro | yes | yes | yes | yes, two 7-DoF torque-sensing arms | yes | quote only; third parties guess USD 85-95k | unknown | budget |
| H. Galaxea R1 Lite | yes | yes | yes | yes, 3.5 kg arms, torso to 1.7 m | yes | USD 39,999 list | unknown | EU support, safety documentation |

The case for **B then C**:

- **Coverage.** It covers every task and lets a single robot attempt the final.
- **Cost.** It costs less than half a Stretch 4 bundle.
- **Fits our stack.** It keeps everything on 24/48 V DC and Jetson TensorRT, which we already run.
- **Arm upgrade without new code.** The Lite 6 code (xArm SDK, `xarm_ros2`) carries over to the 850.
- **Safety we can prove.** We control the whole safety architecture, which is what MGL and RPL score: a hard-wired
  E-stop chain and an independent watchdog that we can demonstrate at inspection.
- **Base choice.** The Ranger Mini 3.0 is 0.50 m wide and has crab and spin modes, so it passes 0.75 m doors with
  margin. It can slide along an OR table while the arm wipes and dock sideways to a trolley handle. Its battery is
  48 V LiFePO4, 24 Ah (one reseller's figure; verify).
- **It matches the winners.** The 2025 and 2026 @Home winner, Tidyboy, built exactly this kind of robot: a compact
  round omni base plus one long-reach arm, chosen because "teams struggle to let their robot simply move around the
  environment".
- **Precedent in hospitals.** Diligent's Moxi puts a Kinova arm on a base for hospital work.

The case against B is integration. A base, a column, an arm, a head, sensors and a safety chain take one engineer
8-10 weeks to make dependable, beside the RSS paper and the other October topics. That is why the plan has a decision
gate on 20 December (section 7).

Why not buy: **Stretch 4** is the strongest turnkey option. It is light and backdrivable, its compact 43 cm round base
passes doors easily, it runs 8 h and carries the same Orin NX as our M3. It costs USD 29,950 list and ships in 2-4
weeks. Against it:

- It carries 2.5 kg at 55 cm reach and its 120 cm lift range reaches only about table and middle-shelf height.
- Pushing a hospital trolley with a 46 kg three-wheel base is unproven.
- It is sold "only certified for laboratory and research use", so the CE waiver for the E-stop does not apply. The
  Safety Committee would have to accept its runstop.

It remains the fallback if the build slips, and a good buy only if a sponsor pays. **TIAGo Pro** is the most capable
and safest platform (two 7-DoF torque-sensing arms, brakes, omni base, ROS 2 with real-time kernel), but it is priced
on request and third-party guesses are around USD 90k. Ask PAL for a loan or sponsorship, not a quote. **Using only our
robots (A)** fails the trolley, most OR work and, realistically, the final. The M3 has no joint read-back and the R2s
are Ackermann cars. A stationary Lite 6 is a fixture that lowers RPL on every task.

Keep the **M3 Pro** in the team. Guiding the patient needs speech and following, not manipulation. The M3 can carry the
sample to the lab in parallel during the final. And a second robot coordinated autonomously raises RPL, which
multiplies all four tasks. It needs a hardware E-stop and a mast with a screen, microphone and speaker at ~1.1 m
(EUR 200-600 est.): a knee-high robot leading a patient with reduced mobility is a tripping hazard. Leave the R2s and
the X3 out.

## 4. The safety concept is a scored deliverable: hard-wired E-stops, a watchdog and fault-injection demos

**What is required.** The rulebook requires four things:

- a hardware E-stop;
- a single-fault-safety concept that names the safety functions, gives at least one credible single fault per
  function and shows the safe state and how it is reached;
- a risk analysis for every robot and every reconfigured variant, submitted with the application;
- passing the Safety Committee's assessment before the first slot (RB 4.1.3.2, 4.2.3.2, 6.4.3, 6.4.7, 6.4.8).

The same items score: MGL for single-fault safety ("a demonstrated implementation earns more"), RPL for deterministic
safety control. Dual processors are "explicitly not required"; a software or architectural solution suffices.

**Outline for both robots:**

1. **E-stop chain (stop category 0, or 1 for the base).** ISO 13850 and IEC 60204-1 allow only category 0 or 1 for
   emergency stops. Two latching mushroom buttons on the manipulator, plus the base's own, feed a two-channel safety
   relay. The relay opens the base motor power and the column power and triggers the arm's E-stop input; the Jetson
   stays powered to log. Reset is manual and separate. The M3 gets a mushroom button in its motor and servo supply.
   Today it has none, which would fail RB 4.1.3.2.
2. **Wireless E-stop as a second channel, never the only one.** PLd/SIL 2 products exist: FORT Robotics (PLd Cat 3),
   Anderson Controls' 2.4 GHz mesh (PL d, SIL 2, CE, one transmitter to up to 10 machines, 9-30 V DC receiver),
   Cattron and efa. They are quote-only. Use an EU band (2.4 GHz or 868 MHz), never 900 MHz US units, and expect a
   crowded 2.4 GHz band where the arena WiFi also runs. Loss of the radio heartbeat must open the chain.
3. **Independent watchdog.** A microcontroller expects a 20-50 Hz heartbeat from the main computer. On a timeout it
   drops the base enable, stops the arm and holds the column. This is our "real-time and deterministic" safety control
   and the demonstrable single-fault path when the computer hangs.
4. **Single-fault table per function.** Each row gives the fault, the safe state and the path to it:
   - E-stop: a welded relay contact → the second channel opens.
   - Heartbeat: the planner hangs → the watchdog stops the robot.
   - Lidar: dropout or a frozen scan → a stale-timestamp check triggers a protective stop, with a bumper or depth camera
     as the second sensor.
   - Speed near people: the speed is capped near detected people and when localisation is uncertain.
   - Arm collision: vendor collision detection plus force and speed limits during handover.
   - Tip-over: column up → base speed limit, and an IMU tilt stop.
   - Network: WiFi loss stops nothing safety-related; the gateway holds the robot or finishes the action safely.
   - Battery: the BMS and fuses cut power.
   - Column pinch: the column's own end switches and safety nut.

   The full table is in the safety note, section 4.
5. **Fault-injection demo at the inspection and in the video.** Pull the lidar cable, kill the planner, cut WiFi,
   press the wireless stop; the robot reaches its safe state each time.
6. **Risk analysis per robot (ISO 12100 loop).** Write 20-30 rows per robot and configuration: hazard, situation,
   people exposed (patient actor, operator actor, team, referees, the public behind 1.4 m walls), severity x
   probability on a declared scale, measures (inherent → protective → information), residual risk, verification. Seed
   the list from ISO 10218-2 Annex A and add the mobile hazards: collision, tipping, column pinch, trolley run-away,
   battery, sharp instruments, the patient leaning on the robot. The RB lets patients lean; we should declare that we
   do not offer physical support in 2027.
7. **Standards to cite.** These are cited as design intent; we claim no conformity:
   - ISO 12100 (risk assessment).
   - ISO 13482 (service robots). Edition 2, "Robotics — Safety requirements for service robots", is at the FDIS stage;
     publication not confirmed.
   - ISO 10218-1/-2:2025 for the arm; it now contains the former ISO/TS 15066 collaborative limits.
   - ISO 13849-1 (performance levels), IEC 60204-1 and ISO 13850 (stops and E-stops).
   - ISO 3691-4:2023 for the base's personnel detection.
   - For MGL: IEC 60601-1 (single fault safe), ISO 14971, ISO 13485 and IEC 62304.
   - Note that the EU Machinery Regulation 2023/1230 applies from 20 January 2027.
8. **Cheap MGL items.**
   - A manufacturing concept for cleanability: printed covers now, seamless injection-moulded or stainless parts in
     series. The rulebook gives this very example (RB 4.2.3.1).
   - A two-page concept of how the design and our development process would map onto IEC 60601-1, ISO 13485 and
     ISO 14971.

## 5. All computing rides on the robots: one AGX-class Jetson on the manipulator, the Orin NX on the M3

The ban on external networks forces everything onto the robots: speech recognition, the LLM that turns speech into a
function, detection and segmentation, a manipulation policy, Nav2 and SLAM. The two @Home winners could not have run
here as they did: NimbRo called GPT-4o over a 5G router, and Tidyboy used a backpack laptop and Google's speech API.
We already run the kit detector (3.0 ms) and ACT (5.9 ms, a true 10 Hz) under TensorRT on the M3's Orin NX. Our 4B LLM
answers in ~0.65 s on the laptop but has not yet been tried on a Jetson.

NVIDIA raised Jetson prices in July 2026:

| product | old (USD) | new (USD) | power |
|---|---|---|---|
| AGX Orin 64 GB dev kit | 1,999 | 3,499 | 15-60 W |
| AGX Thor dev kit | 3,499 | 5,499 | 40-130 W, 128 GB |
| Orin NX 16 GB module | 599 | 999 | |

Both dev kits were out of stock at NVIDIA's own store on 21 October. Every admitted team gets the same NVIDIA hardware
package, whose contents are unpublished.

**Recommendation:** put one Jetson AGX Orin 64 GB in the manipulator now. It holds speech, a 4B LLM, perception, a
policy and Nav2 with room to spare. Plan for Thor only if the NVIDIA package brings one; it is the right board for
multi-billion-parameter VLAs, and JetPack 7 would allow Jazzy natively (not re-checked). The M3 keeps its Orin NX for
its own voice and navigation (research topic 4).

Power while working is roughly 250-500 W (est.). The Ranger Mini's ~1.15 kWh then lasts 2.5-4 h; DC hot-plug charging
between runs is allowed. IGX Thor has a functional-safety island and NVIDIA mentions IEC 60601 support through partners.
It is attractive for the MGL story but has no public price; it is a 2028 consideration.

## 6. Winners of similar leagues build one capable robot, run foundation models with fallbacks and carry spares

The open-platform league of RoboCup@Home is the closest benchmark. Tidyboy won 2025 and 2026; NimbRo won 2024 and came
second in 2025 and third in 2026. In 2027 @Home becomes one open league.

**Hardware.** Both winners run one strong mobile manipulator per robot:

- NimbRo uses a TIAGo++ raised by 200 mm, with two 7-DoF arms and an RTX A4500 box on its back.
- Tidyboy uses a 30 kg custom robot: a 501 mm omni base and a Kinova Gen3 reaching the floor to 1.5 m without a lift.

Both kept a spare robot or platform.

**Software.** Both combine closed-set detectors retrained on site with open-vocabulary models. An LLM maps free speech
onto 3-15 robot functions, and behaviour runs as hierarchical state machines. Grasps are verified from gripper
encoders, with a handover after repeated failures. A touchscreen is the fallback when speech fails.

**What lost them points.** Robustness, not capability: a lost person track, time-outs, an arm hardware defect, a quiet
voice in a noisy hall, robots arriving late.

**Healthcare-specific benchmarks.** METRICS HEART-MET averaged technical and social-acceptance scores. The HCC's
physician panel suggests the same weight on how the robot explains, waits and speaks.

**Lessons for us:**

- Build one capable robot, not many weak ones.
- Bring spares of anything that ends a run.
- Add a screen fallback for the patient dialogue and a close microphone.
- Show reliability as counts of repeated runs, not single clips.
- Drive the robot to Nuremberg ourselves.

## 7. The application can be strong by 31 January if the real base drives in the video

**Checklist (from RB 1.3.2, 1.4, 6.4.4, 6.4.8 and the FAQ):**

| item | content | note |
|---|---|---|
| Pre-registration | team in the Participant Management portal; check the download area for a newer rulebook | done? (we hold V1.0) |
| Team Description Paper | Overleaf template, English; team, members, leader, web link, contact; hardware with images; safety systems and E-stop procedure; functional and software architectures; innovative technology; environment adaptations needed; medical relevance; contributing institutions | page limit unknown; @Home uses 8 pages LNAI + 2-page annex |
| Promotional video | autonomous navigation, picking, placing; show what the robot sees and its plan; 1080p 16:9; no AI generation or enhancement; #Healthineers #HCC@RC27 | mix real clips with the Isaac Sim final; label speed-ups |
| Team web site | English, for a broad audience; robots, approach, publications, open-source links | the lab's GitHub plus a page |
| Risk analysis per robot | manipulator (each configuration) and M3 | section 4, item 6 |
| Safety concept | single-fault table, E-stop chain, watchdog, fault-injection evidence | scores MGL even on paper |
| Declared room preparations | type, location and purpose of every marker, label or fixture | as few as possible (RPL) |
| Development plan | timeline and budget to June 2027, risks, funding of travel | the FAQ's selection criteria |
| Simulation results | the arena GLB in Isaac Sim with both robots doing tasks 1-4 and the final | also serves T ("published simulation") and SF |
| Evidence of competence | publications, open-source releases, the public demo of 2026-10-22, success counts | offsets "previous competitions" = none |
| Medical partner | a clinician adviser (e.g. from a Košice university hospital or medical faculty) | strengthens the physician-panel side; not required |

**Timeline to 31 January:**

- **27 Oct-9 Nov: decide and ask.**
  - Agree the budget.
  - Email the organisers the questions in section 8.
  - Request quotes (AgileX via Generation Robots, LINAK, UFACTORY, a wireless E-stop vendor, Hello Robot, PAL,
    Kinova).
  - Convert the GLB to USD.
  - Write `hospital_interfaces` and a mock Hospital Controller from RB 8.1.3 in a Jazzy container.
- **10-30 Nov: order and model.**
  - Order the base, column, E-stop parts and AGX Orin.
  - Build the Isaac Sim arena with the manipulator (AgileX URDF + column + Lite 6 URDF) and the M3.
  - Fit the M3's E-stop and mast.
  - Put voice on the robots (topic 4).
  - Draft the safety concept and both risk analyses.
- **1-20 Dec: assemble.**
  - Frame, power, E-stop chain, watchdog.
  - ROS 2 bring-up: base driver, `xarm_ros2`, Nav2, the Jazzy gateway against the mock controller.
  - Scripted sim runs of all tasks.
- **Decision gate, 20 December.** Is the manipulator driving with its arm mounted and the E-stop chain working? If not,
  submit on the simulation, the M3 and a Lite 6 stand, and put a Stretch 4 purchase on the table for after selection.
- **7-24 Jan: film and write.**
  - Build a lab mock-up: 0.75 and 0.95 m doors, a 0.92 m counter, a trolley, a cabinet.
  - Film real clips: door pass, pick to trolley, trolley push, wipe, handover, E-stop and fault injection, with success
    counts.
  - Film the full simulated final.
  - Write the TDP and the web site.
- **25-31 Jan: review and submit**, aiming for 29 January.

**After submission:**

- **To selection (15 March):** keep developing; pursue sponsors; prepare the arm order so it goes out the day we are
  selected.
- **From 15 March:** final registration with RoboCup; arm upgrade and integration (4-6 weeks); a full-size rehearsal of
  all tasks and the final; spares; hardware freeze in mid-May; van to Nuremberg.

## 8. Questions to ask the organisers and vendors now

**Organisers** (robocup@siemens-healthineers.com):

1. Is there a rulebook newer than V1.0, and when come chapter 7 and the qualification details? Is there a TDP page
   limit?
2. When will the Hospital Controller code be released, and can teams have a pre-release?
3. Which `ExecuteTask` type starts task 2? None of the five matches object transport. Must the sample be put into the
   analyser physically?
4. Does RB 8.1.1.2 forbid our own robots from talking to each other during a run (an access point on a robot, a 5 GHz
   link)? The fleet plan depends on this.
5. What is the object list we must bring (instruments, tubes, packaging, tray layout, trolley model)? What door types
   and opening forces should we expect, and in which language does the patient speak?
6. Does a team-placed stationary arm count as a fixture?
7. What does the NVIDIA package contain, and when does it arrive?
8. Is the floor level?
9. What are the RoboCup 2027 registration fees?

**Vendors:**

- **AgileX / Generation Robots:** stock and lead time, the Ranger Mini 3.0 battery voltage, E-stop wiring and an
  external stop input, CE declaration, the ROS 2 Humble driver, an academic price.
- **UFACTORY:** the Lite 6 E-stop input and its 24 V draw on battery, the 850's weight and DC box input range, an
  education discount, sponsorship.
- **LINAK or Progressive Automations:** a 24 V column quote; medical documentation for the LC3.
- **Hello Robot:** EU landed price, CE plans, whether the Stretch 4 runstop removes motor power, trolley-pushing
  capability, academic loan.
- **PAL Robotics:** a TIAGo or TIAGo Pro loan or event rental.
- **Kinova:** academic price or loan for a Gen3.
- **Wireless E-stop vendors:** EU band, price, heartbeat timeout.

**Sponsors.** Siemens Healthineers gives no hardware, but "strong partnerships and external support are welcome" (FAQ).
Ask TUKE and the faculty for travel money, and regional companies and the vendors above for sponsorship.

## 9. Risks and how to cut them

| risk | likelihood | effect | mitigation |
|---|---|---|---|
| The build takes longer than planned (team also on RSS, VLA, voice) | high | weak video, weak prototype | order in November; decision gate on 20 Dec; Stretch 4 fallback |
| The Lite 6 is too weak for drawers, wiping, doors | high | tasks 3-4 partial | passive compliant tools; doors via the controller; 850 upgrade after selection |
| Scoring chapter changes the incentives | medium | redesign | modular robot; keep fixtures removable and declared |
| Robot-to-robot communication not allowed during runs | medium | no fleet bonus, the M3 idle in the final | ask now; the manipulator can do task 1 alone with its head |
| Humble/Jazzy bridge failure on the arena WiFi | medium | counts as robot failure | one Jazzy gateway per robot, tested against the mock and the released controller; nothing safety-related depends on WiFi |
| 2.4 GHz congestion at RoboCup | high | lost orders, a wireless E-stop tripping | all autonomy on board; latched states; an EU-band wireless stop with a tested timeout |
| Safety assessment fails on site | low-medium | missed slots, no reschedule | send the safety concept early; fault-injection video; inspection rehearsal |
| Jetson shortage or price | medium | late compute | buy one AGX Orin now; NVIDIA package for the rest |
| Speech in a noisy hall, external actors | high | task 1 and 4 points | close microphone, screen fallback, confirmation dialogue, English first |
| Costs not covered (travel, transport, fees) | certain | budget | budget EUR 6-12k (est.) beside the hardware; drive the robot |
| No clinical partner | medium | physician panel unimpressed | approach a Košice clinician in November |

## Conclusion

Apply, and build. The rulebook rewards generality, demonstrable safety, on-board intelligence and shared knowledge.
Those are the lab's strengths in software, simulation and Jetson deployment. What the lab lacks is a single robot that
can reach a table, push a trolley and open a cabinet. A self-built manipulator on an AgileX Ranger Mini 3.0 with a 24 V
lift column and our Lite 6 closes that gap for about EUR 19-25k excl. VAT. A UFACTORY 850 after selection adds the
force the Lite 6 lacks. The M3 Pro, with a hardware E-stop and a mast, guides the patient and makes the team a fleet.

Spend the next three months on four things:

- the robot driving with its arm by 20 December;
- an Isaac Sim twin of the organisers' arena with both robots;
- a safety concept with an independent watchdog that we can break on camera;
- a TDP and video that show repeated, counted runs.

Ask the organisers now about robot-to-robot communication, the object list and the missing task-2 order. Ask the
vendors for quotes and sponsorship. Keep a Stretch 4 purchase as the fallback if the build slips past the December
gate.

## Method and caveats

- **Sources.** I read the rulebook V1.0 in full from the lab's copy and measured the arena GLB's bounding boxes. I read
  the official overview and FAQ in a browser on 2026-10-27; the site is JavaScript-rendered and its download area
  needs a login. Vendor pages and papers were opened where linked in the notes; items marked [2nd] there rest on search
  summaries.
- **Search budget.** The shared web-search budget ran out near the end of the session. Three items rest on one reseller
  page each: the Ranger Mini 3.0 battery voltage, the xArm DC box input range and the AgileX E-stop wiring. The
  standard editions marked [KNOW] were not re-checked.
- **Prices.** No price here is a quote to us. EUR/USD conversions and landed costs are estimates.
- **Unknowns.** The scoring chapter, the Hospital Controller code and the NVIDIA package may change parts of this memo.
