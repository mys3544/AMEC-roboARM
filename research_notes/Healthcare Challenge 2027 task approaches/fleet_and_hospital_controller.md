# The Hospital Controller interface, Humble-to-Jazzy bridging and fleet coordination (state of 27 October 2026)

Scope: how our robots talk to the organisers' Hospital Controller (ROS 2 Jazzy, domain 0, 2.4 GHz WiFi), what works
today for connecting our ROS 2 Humble robots to it, and how two robots (the self-built manipulator and the M3 Pro)
can coordinate as a fleet for the RPL bonus, with and without a direct robot-to-robot link. Sources: rulebook V1.0
chapter 8 (read in full), ROS, Nav2, Open-RMF, zenoh, NVIDIA Jetson and Isaac ROS pages read on 2026-10-27, and our own
architecture (PLAN.md). Labels: **[RULE]** rulebook V1.0 / FAQ; **[DOC]** official documentation, opened; **[REAL]**
our robots; **[CLAIM]** vendor claim; **[2nd]** search summary or secondary page; **[OURS]** our design inference;
**[KNOW]** background knowledge not re-checked (never load-bearing).

## 1. What the controller exposes and what the robot must serve

### Takeaway
The controller owns the devices (doors, lab, CT, OR) as actions and services under `/hospital`, with latched state
topics; the robot serves the `ExecuteTask` action, publishes its pose (10-30 Hz) and an intent string (1-2 Hz). Four
details are unspecified and must be asked: the topic names for pose and intent, the pose's frame, how several
robots of one team are addressed, and which order type starts task 2.

### Evidence
- Interface table [RULE, RB 8.1.3.1]: `/hospital/doors/operate` (action OperateDoor), `/hospital/doors/state(s)`,
  `/hospital/doors/get_states`; `/hospital/lab/analyze` (AnalyzeBlood), `/hospital/lab/state`;
  `/hospital/ct/table/move` (MoveTable), `/hospital/ct/table/state`, `/hospital/ct/scan/run` (RunScan),
  `/hospital/ct/scan/state`; `/hospital/operating_room/set_cleanliness`, `/get_state`, `/state`;
  `/hospital/execute_task` (ExecuteTask). "key states latched (transient local durability)"; "the package
  hospital_interfaces provides all message, service and action definitions".
- ExecuteTask "is served by the robot itself (RobotServerNode)"; goal: `uint8 value` (CLEAN_OR=0, COLLECT_LAB=1,
  GUIDE_CT=2, HEART_ATTACK=3, SURGICAL_ASSIST=4), `string target_location`, `string task_id`; result `bool success`,
  `string message`; feedback `uint8 current_phase` (PHASE_QUEUED=5 ... PHASE_DONE=8), `float32 progress`,
  `string task_id` [RULE, RB 8.1.3.6]. The printed comments are attached to the wrong blocks; the order goal ---
  result --- feedback is the standard one. Build against the released package.
- Minimal interfaces: "Robot Pose geometry_msgs/msg/PoseStamped (10 <= FPS <= 30)", "Robot Intent std_msgs/msg/String
  ... (1 <= FPS <= 2)", and "Perform Task 1-4" and "Perform Finale" [RULE, RB 4.1.4.1]. No topic names are given.
- The organisers' simulated robots "act as an action server for the ExecuteTask action", "teleport between fixed room
  waypoints", publish a PoseStamped and a TF frame, keep an internal queue; "Multiple instances run concurrently";
  adding a robot means editing a ROBOTS list and the activity logger's robot ids [RULE, RB 8.1.4.2]. The RViz panel
  can "send tasks to the robots and monitor their progress" [RULE].
- Lab results: abnormal means cTnT > 14 ng/L, cTnI > 20 ng/L, CKMB > 5 %; set by a controller parameter
  `cardiac_abnormal` [RULE, RB 8.1.3.3]. OR `xray_active` "indicates that simulated radiation is active and that the
  room must not be entered" [RULE, RB 8.1.3.5].
- Controller PC: Ubuntu 24.04, ROS 2 Jazzy, `ROS_DOMAIN_ID=0`, 2.4 GHz WiFi; "must not be used as an edge server";
  bridge failures count as robot failures [RULE, RB 4.1.4.1]. "The Hospital Controller will be released as source
  code with an upcoming version of this rulebook" [RULE, RB 8.1.1]. No newer rulebook or controller release was
  visible on the public overview and FAQ pages on 2026-10-27 [RULE, read in a browser].
- The ward order list has no order for object transport; COLLECT_LAB is "collect laboratory results" [RULE,
  RB 8.1.2.5].

### Inferences [OURS]
- Our mock controller is a day's work: the five device nodes as Python stubs with the timings and states of RB 8.1.3,
  the ExecuteTask client as a CLI, and a fake RViz panel replaced by `ros2 action send_goal`. Swap in the real
  package the day it is released.
- The pose's frame: the controller's 3D scene comes from `hospital_description`; our SLAM map has its own origin. A
  fixed transform from our map to the arena frame, fitted on setup day from 3-4 surveyed points (door centres),
  makes our PoseStamped land in the right room on the spectators' display.

## 2. Humble and Jazzy: what works today

### Takeaway
Do not let Humble and Jazzy nodes talk DDS to each other: the ROS community says distributions are not compatible
(type descriptions changed message handling), rmw_zenoh embeds type hashes from Jazzy on, and the zenoh DDS bridge
documents no cross-distribution guarantee. Two clean options exist. (a) Run the new manipulator natively on Jazzy:
JetPack 7.2 brought the AGX Orin to Ubuntu 24.04, and Isaac ROS 4.6 supports it. (b) On the M3, which stays on Humble,
run a small Jazzy gateway in a container that talks to the robot's own stack over HTTP (our bridges already speak
HTTP and SSE), never over DDS. Either way, only one process per robot touches domain 0.

### Evidence
- "distributions are not compatible unfortunately", mainly because serialization changed with the type description
  feature; another reply: do not mix nodes between ROS distros [2nd, search summary of
  [ROS Discourse](https://discourse.openrobotics.org/t/incompatability-between-distributions/43747)].
- rmw_zenoh: "From ROS 2 Jazzy onward, rmw_zenoh topic key expressions include the message type hash (RIHS01 ...).
  This prevents interoperability with ROS 2 Humble and earlier" [2nd, search summary of
  [PX4 zenoh docs](https://docs.px4.io/main/en/middleware/zenoh.html)].
- zenoh-plugin-ros2dds README: no statement on Humble/Jazzy interoperability; "tested with
  RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"; a per-robot namespace prefixes "ALL topics/services/actions" [DOC,
  [zenoh-plugin-ros2dds](https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds)].
- JetPack 7.2: Jetson Linux 39.2, Ubuntu 24.04 root file system, "adds support for the Jetson Orin product family
  within JetPack 7 releases"; CUDA 13.2.1, TensorRT 10.16.2 on Orin [2nd, search summary of
  [NVIDIA JetPack 7.2 archive](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2) and
  [NVIDIA forum](https://forums.developer.nvidia.com/t/jetpack-7-2-jetson-software-goes-agentic-with-jetson-linux-39-2/372056)].
- Isaac ROS 4.6.0 (18 Aug 2026): "Added support for Jetson Orin", "Added support for JetPack 7.2"; Isaac ROS 5.0.0
  (21 Sep 2026): "Migrated to ROS 2 Lyrical Luth" [DOC, [Isaac ROS releases](https://nvidia-isaac-ros.github.io/releases/index.html)].
  Isaac ROS 4.6 on Orin with JetPack 7.2 is a Jazzy setup [2nd,
  [openelab](https://openelab.io/blogs/learn/ros-2-lyrical-vs-jazzy)].
- xarm_ros2 has `humble` and `jazzy` branches and Lite 6 MoveIt configs [DOC,
  [xarm_ros2](https://github.com/xArm-Developer/xarm_ros2)]; ranger_ros2 shows only a `humble` branch [DOC,
  [ranger_ros2](https://github.com/agilexrobotics/ranger_ros2)].
- Our robots: JetPack 6.2, Humble, the M3's vendor stack on domain 30; robots and console talk HTTP and SSE; "If the
  robots ever share ROS topics: one zenoh-bridge-ros2dds per robot with an allow-list" [REAL, PLAN.md Architecture].
- The organisers told us "no need to migrate to Jazzy; Docker, bridges and gateways are fine" [MAIL of 2026-10-02 as
  recorded in project memory].

### Inferences [OURS]
- Manipulator: flash JetPack 7.2, install Jazzy, build xarm_ros2 (jazzy) and port ranger_ros2 (a small CAN node over
  AgileX's ugv_sdk; budget 2-4 days). Rebuild our TensorRT engines (TensorRT 10.16 instead of 10.7). The gateway is
  then an ordinary node in the robot's graph, with a DDS router or a second participant on domain 0 limited to the
  controller's topics.
- M3: a `ros:jazzy` container (Ubuntu 24.04, no GPU needed) with host networking, `ROS_DOMAIN_ID=0`, the same RMW as
  the controller (Jazzy's default Fast DDS unless the organisers say otherwise), talking to `m3bridge` over
  127.0.0.1 HTTP. Nothing in it is on a safety path: a lost WiFi link holds the current step or finishes it safely.
- Discovery over the arena WiFi: multicast may be filtered; Jazzy's static-peer setting (`ROS_STATIC_PEERS`) or a
  Fast DDS discovery server are the fallbacks [KNOW; docs.ros.org was behind a bot check for our fetcher]. Test on a
  2.4 GHz access point with other traffic before June.

## 3. The gateway, one per robot

### Takeaway
A single node owns everything on domain 0: the ExecuteTask server, the pose and intent publishers, the clients for the
devices, a cache of the latched states and a log. It turns controller orders into jobs for the robot's own task
machine and turns the task machine's state into feedback and intent.

### Evidence
- "Teams are only permitted to communicate with this system via the provided interfaces" [RULE, RB 4.1.4.1]; "the
  Hospital Controller is also the only interface that team members may use to interact with their robot during a
  task" [RULE, RB 8.1.1.2].
- A brief window in the room preparation is "provided solely for establishing necessary connections (e.g., network,
  Hospital Controller interface)" [RULE, RB 6.4.5].

### Inferences [OURS]
| part | content |
|---|---|
| ExecuteTask server | accepts one order at a time; maps value -> job (GUIDE_CT -> task 1, CLEAN_OR -> task 3, SURGICAL_ASSIST -> task 4, COLLECT_LAB -> sample run, HEART_ATTACK -> final); feedback phases from the job's state; result with a short message |
| pose publisher | 20 Hz, TF map -> base_link, transformed into the arena frame |
| intent publisher | 1 Hz, the task machine's current state as one sentence |
| device clients | OperateDoor, MoveTable, RunScan, AnalyzeBlood, SetCleanliness, GetOperatingRoomState; every call with a timeout |
| state cache | doors, CT, lab, OR (latched topics), exposed to the robot over HTTP |
| health | heartbeat to the robot; on link loss: finish the current motion, hold, retry |
| log | every order, call and state change with time, for the TDP's success counts and the referees |

## 4. Fleet coordination: a small dispatcher, not Open-RMF, and a fallback without links

### Takeaway
Open-RMF was "initially developed for the healthcare industry" and has a Jazzy release, but it needs a central server
talking to every robot during the run, its Python fleet adapter (free_fleet) is Jazzy-only and tested mostly with
Cyclone DDS, and our two robots work in separate zones where traffic negotiation buys little. Build a small dispatcher
on the manipulator instead (capability table, assignment by estimated time, a mutex per door and narrow corridor). If
the organisers forbid robot-to-robot links (RB 8.1.1.2), the robots can still cooperate through the controller's
shared, latched states, each reacting to events the controller publishes. That still meets "autonomous fleet
management", which the rulebook defines as no hands-on interaction.

### Evidence
- RPL rises with "fleet management and the coordination of several cooperating robots" [RULE, RB 4.2.2]. In the
  final, several robots are allowed "only ... if no manual intervention by the operator is required for the
  transition between configurations. Autonomous fleet management must be demonstrated, meaning all robots must operate
  without hands-on interaction by the team during the run" [RULE, RB 6.4.3].
- "no remote operation, external network connection, or other technical communication channel with the robot is
  permitted while a run is in progress" [RULE, RB 8.1.1.2] - whether this covers links between our own robots is the
  open question the platform report already raised.
- Open-RMF: "a free, open source, modular software system that enables sharing and interoperability between multiple
  fleets of robots" and doors, elevators and building systems; "initially developed for the healthcare industry" [DOC,
  [open-rmf.org](https://www.open-rmf.org/)]; a Jazzy release in August 2024 [2nd,
  [Open Robotics blog](https://www.openrobotics.org/blog/2024/8/18/open-rmf-gets-jazzy)].
- free_fleet: Python Open-RMF fleet adapter, "uses zenoh as a communication layer between each robot and the fleet
  adapter", zenoh-bridge-ros2dds 1.5.0 on the robot, supports "ROS 2 Jazzy" on Ubuntu 24.04, mostly tested with
  rmw-cyclonedds-cpp, other RMWs "have shown varying results", no releases yet [DOC,
  [free_fleet](https://github.com/open-rmf/free_fleet)].
- The organisers' ward management dispatches HEART_ATTACK as one order [RULE, RB 8.1.2.5].

### Inferences [OURS]
- With links allowed: the manipulator's gateway receives HEART_ATTACK and runs the dispatcher; the M3 gets jobs over a
  team link (HTTP over our own access point, or zenoh between the two gateways). Door and corridor mutexes prevent a
  meeting in a doorway.
- Without links: zones are fixed by the final's story (M3: entrance, preparation, CT, lab; manipulator: storage, OR),
  and each robot starts its phases on controller events: the M3 on the order (or on the patient's arrival), the
  manipulator on `/hospital/lab/state` = DONE with abnormal values (then CLEAN_OR and the tray), the M3's sample run on
  `/hospital/ct/scan/state` leaving IDLE. Whether both robots may receive the HEART_ATTACK order, or one robot must own
  it, must be asked.
- Open-RMF stays an option for 2028 or for the sim demo (its dashboard is good to show), not a 2027 dependency.

## 5. The final's choreography for two robots

### Takeaway
The final splits cleanly by place and time: the M3 guides the patient, takes the sample to the lab during the scan
and starts the analysis; the manipulator cleans and prepares the OR when the lab reports abnormal values, waits, then
fetches catheter type B, hands it over and disposes of the packaging. The parallel phases are what a fleet buys in
the final.

### Evidence
- Final storyline: welcome; physician takes a blood sample in the preparation room; guide to CT; "During the scanning
  process, the robot transports the blood sample to the laboratory"; abnormal values; "The robot cleans and prepares
  the operating room"; waiting position during the intervention; catheter type B on an acoustic command; handover,
  packaging disposal, return; "The time is stopped when the robot reaches the designated parking position" [RULE,
  RB 3.2.1]. Physical support of the patient "is permitted and may result in additional points" [RULE, RB 3.2.1].
- No exchange of robots or reconfiguration during the final; parc ferme applies [RULE, RB 3.1, 6.4.1].

### Inferences [OURS]
| phase | M3 (guide) | manipulator |
|---|---|---|
| arrival | greets, verifies, guides to the preparation room | waits at parking |
| blood sample | waits; physician puts the tube in the M3's caddy | waits |
| CT | guides to CT, instructs, starts the scan | waits (or starts cleaning if the order allows) |
| during the scan | carries the sample to the lab, `AnalyzeBlood` | reads `/hospital/lab/state` |
| abnormal values | returns to CT, retracts, helps the patient off, hands over to staff | `CLEAN_OR`: wipes, lays the tray, `SetCleanliness(CLEAN)` |
| intervention | parks | waits at the designated waiting position |
| catheter B | parked | understands and confirms, fetches, presents, disposes, returns, parks (time stops) |

## 6. Questions for the organisers that this note adds

### Takeaway
Seven interface questions decide the gateway's design and should go in the same email as the platform questions.

### Inferences [OURS]
1. Topic names and QoS for the robot's pose and intent; the pose's `frame_id` and how it relates to the arena model.
2. How several robots of one team are addressed (one `ExecuteTask` per robot under a namespace?), and whether a
   team's fleet may receive HEART_ATTACK once or per robot.
3. The order type for task 2; whether the sample must be put into the analyser physically.
4. Whether our robots may talk to each other during a run, and over which network.
5. The RMW implementation and any discovery settings on the controller PC; whether multicast works on the arena WiFi.
6. Door device ids and whether doors close automatically; which doors are physical.
7. Whether the robot may stay in the CT room while `radiating` is true, and what "designated area" means in task 3.
