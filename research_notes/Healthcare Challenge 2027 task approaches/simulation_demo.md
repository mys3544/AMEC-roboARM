# The simulation demo: the organisers' arena in Isaac Sim, NVIDIA's healthcare tools and the application video (state of 27 October 2026)

Scope: how to turn the organisers' arena model (`C:\Users\Mys\Desktop\Siemens sutaz\2026_09_01_RoboCup_HCC_Arena_V1 1.glb`)
into an Isaac Sim scene in which our two robots run the real task software against a mock Hospital Controller; which
assets and tools NVIDIA offers for hospitals (Isaac for Healthcare, SimReady assets, Isaac Sim's own hospital,
people simulation); how the ROS 2 side connects on our Windows GPU PCs; and what a 2-3 minute application video should
show. I inspected the GLB's JSON chunk with a stdlib script (node tree, meshes, materials, accessor bounds; no geometry
was modified). Labels: **[RULE]** rulebook V1.0 / FAQ; **[DOC]** official documentation, opened; **[GLB]** my
measurement of the arena file; **[REAL]** our robots and servers (project memory); **[2nd]** search summary or
secondary page; **[OURS]** our design inference; **[KNOW]** background knowledge not re-checked.

## 1. What the application needs from simulation

### Takeaway
Simulation is "strongly encouraged" and judged as "relevant simulation results"; it also earns the technology bonus
when sim-to-real comes "accompanied by a published simulation", and the community factor when shared. One constraint
matters for publishing: the arena model is licensed through pre-registration, so we publish our conversion scripts,
robot models and task code, not the converted arena.

### Evidence
- FAQ: the dataset includes "a 3D simulation environment, enabling teams to develop and present concepts using Gazebo
  or similar simulation tools before physical hardware is available. Simulation-based demonstrations are therefore
  strongly encouraged"; selection weighs "Relevant simulation results, concepts, or prototype demonstrations"; "proposals
  based solely on simulations are also eligible, provided they credibly demonstrate that a functional robot will be
  available by June 2027"; a working prototype "may receive additional consideration" [RULE,
  [FAQ](https://www.robocup2027.event.siemens-healthineers.com/blocks/faq), read 2026-10-27].
- "Pre-registration is required due to the licensing terms of the 3D data" [RULE, FAQ]; use of the simulation "is
  subject to the associated Terms and Conditions included with the download" [RULE, RB 5.1.5].
- T bonus: "sim-to-real approaches accompanied by a published simulation" [RULE, RB 4.2.4]; SF: "providing reusable
  datasets, simulation environments, or benchmarks following ... FAIR data principles" [RULE, RB 4.2.5].
- Video: "clearly demonstrate the robot's ability to perform the tasks ... such as autonomous navigation, picking,
  and placing"; desired: "seeing what the robot sees, and the plan currently followed"; 1080p 16:9; "AI-generation or
  enhancement of video is prohibited" [RULE, RB 1.4.3, 1.3.2].

### Inferences [OURS]
- Read the terms in the download area before publishing anything derived from the GLB. Our own robots' USDs, the
  mock controller and the scene-building script are ours to publish.

## 2. What the arena file contains

### Takeaway
A clean Blender glTF 2.0 export: 90 nodes, 71 meshes, ~315k triangles, two JPEG textures, metres, Y-up, no colliders,
no joints and no animations. Three surprises: the arena sits ~45 m from the file's origin (centre near x 34.6, z
-28.9), most meshes carry a 0.01 scale (centimetre geometry), and the walls are modelled 3.0 m high while the real
ones are ~1.4 m. Each hinged door leaf is one mesh whose node origin lies on a vertical edge, which is where the
hinge goes; the CT's patient table is already a nine-part chain.

### Evidence [GLB]
- Header: generator "Khronos glTF Blender I/O v5.0.21", glTF 2.0; extensions used `KHR_materials_emissive_strength`,
  `KHR_materials_specular`, none required; 1 scene, 90 nodes, 71 meshes, 90 materials, 2 textures (JPEG, 1.18 and
  1.05 MB), 0 animations, 0 skins, 0 cameras; ~315,000 triangles; 16.3 MB.
- No node or mesh name contains "coll"; the file has no physics.
- World bounds (accessor min/max through the node transforms): floor 16.0 x 13.0 m from x 26.55 to 42.55 and z -35.44
  to -22.44; walls 14.0 x 11.0 m, 3.0 m high; ceiling at 3.0 m. Room groups (CT, Storage, OR, Hall, Lab,
  Changing_room, Office, Reception, Rooms) carry the translation (34.552, -0.01, -28.938).
- Doors: `Door_01`...`Door_10` leaves 0.95 m wide, 2.23 m high, 0.17 m thick; `Door_01`'s node origin is at
  x 34.202, the leaf spans x 33.25-34.20, so the origin is on one vertical edge (same for `Door_09`). `SlidingDoor_01`
  1.50 m and `SlidingDoor_02` 2.00 m are scaled boxes.
- CT: `NAEOTOM_Alpha_Model` 1.46 x 1.99 x 2.49 m; the table `MPT_Vitus__2749_` is a chain of nodes `0_001` -> `1` ->
  ... -> `8`, each with its own mesh. OR: `Artis_Basic_Table`, `CIARTIC_Move` (C-arm, five-part chain),
  `CIARTIC_Trolly_Monitor_System`. Storage: two cabinets 1.08 x 1.92 x 0.65 m. `Bin` 0.61 m high. `Bed` 2.26 x 1.16 x
  1.04 m. An empty `stretch_helper` node at 0.92 m in the lab.
- The rulebook: walls ~1.4 m, opaque, corridors ~2.5 m, "figures ... should be considered illustrative examples
  rather than final layouts" [RULE, RB 5.1.1-5.1.2].

### Inferences [OURS]
- Re-root the scene at the arena centre and Z-up; cut the walls to 1.4 m so the robots' cameras see over them as they
  will on site (this changes what the person detector and localisation see).
- Doors become revolute joints about the vertical axis through each door node's origin; sliding doors prismatic;
  the CT table two prismatic joints (up/down, in/out) on the chain's links.

## 3. Converting the GLB into a usable Isaac Sim stage

### Takeaway
Isaac Sim 6.0's asset converter lists `.gltf` (not `.glb`) and its Metrics Assembler converts units and up axis on
load, but glTF arrives as visual geometry only. A scripted pipeline is safer than the import dialog: unpack the GLB
to glTF (or export USD from Blender), import, re-root, then apply static triangle-mesh colliders to walls, floor and
furniture, joints to doors and the CT table, and generate the Nav2 map from the colliders with Isaac Sim's occupancy
map tool. Budget a week, mostly for doors and checking.

### Evidence
- Isaac Sim 6.0 importer formats: `.fbx`, `.obj`, `.gltf` [DOC,
  [formats](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/importer_exporter/formats.html)]. "By default, Isaac Sim
  has enabled the Metrics Assembler, which automatically converts the asset scale for the distance unit, mass unit,
  and Up Axis" [DOC, [OpenUSD fundamentals](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/omniverse_usd/open_usd.html)].
  World up axis +Z [2nd, search summary of the 6.0 conventions page]. The 6.0.1 release notes say the converter,
  importer and exporter "received major upgrades" [2nd].
- glTF/GLB imports have "visual geometry but no USDPhysics schemas"; apply collision afterwards; convex hull for static
  obstacles, convex decomposition for dynamic graspable objects [2nd,
  [SimReady guide](https://simready.artlabs.ai/resources/isaac-sim-asset-requirements-and-best-practices)].
- Occupancy Map Generator (`isaacsim.asset.gen.omap`): uses "physics collision geometry in the Stage", saves a ROS
  map YAML and image for map_server; GUI workflow, no documented headless API [2nd, search summary of the
  [occupancy map page](https://docs.isaacsim.omniverse.nvidia.com/latest/digital_twin/ext_isaacsim_asset_generator_occupancy_map.html)].
- Our sim work runs Isaac Sim 5.0 on Dionysos (RTX A4500 20 GB, shared) and 6.0 on hekate (RTX 5090, someone's job
  must not be disturbed); USD stages are built by scripts run through `tools/isaac_exec.py` [REAL, CLAUDE.md, memory
  `dionysos-server`, `hekate-gpu-pc`, `m3pro-sim-expert`].

### Inferences [OURS]
1. Blender (or gltf-transform) on the laptop: GLB -> glTF + bin + textures, or straight to USD; keep metres.
2. Isaac Sim script: reference the asset, check `metersPerUnit` = 1 and up axis Z after the Metrics Assembler,
   translate by (-34.55, 0, +28.94) in the original axes so the arena centre is the origin.
3. Colliders: `UsdPhysics.CollisionAPI` on every static mesh with triangle-mesh approximation (static colliders may
   be concave); convex decomposition only for objects the robot moves (trolley, tray, instruments).
4. Joints: a revolute joint and a position drive per door leaf at its node origin; prismatic joints on the CT table;
   the OR table as static.
5. Nav2 map: occupancy map with Z bounds at the lidars' height (Ranger's lidar ~0.2-0.4 m, the M3's T-mini ~0.2 m).
6. Check against the rulebook: the 0.75 m minimum door is narrower than the model's 0.95 m doors; add one 0.75 m
   door in a variant scene to prove the robot passes.

## 4. Robots and objects in the scene

### Takeaway
All three robot models exist as open URDFs (Lite 6 in xarm_ros2, the Ranger Mini from AgileX, our M3); the lift
column is one prismatic joint we add. Hospital props come from NVIDIA's i4h SimReady catalogue (instruments, trays,
carts; non-commercial R&D licence) and from our own scans of the objects we bring.

### Evidence
- xarm_ros2: Lite 6 kinematics, `lite6_moveit_fake/realmove/gazebo` launch files, `add_gripper=true` for the Lite 6
  gripper; humble and jazzy branches [DOC, [xarm_ros2](https://github.com/xArm-Developer/xarm_ros2)].
- i4h asset catalogue (archived 2 Sep 2026, moved to i4h-digital-twin): surgical robots, anatomy, "Medical Equipment",
  "Hospital Environment: Models of hospital rooms, operating theaters, and medical facilities", and Lightwheel
  SimReady "deformable medical and surgical props, including instruments, devices, carts, trays, and scene layouts";
  Lightwheel assets "non-commercial, research and development use only" [DOC,
  [i4h-asset-catalog](https://github.com/isaac-for-healthcare/i4h-asset-catalog),
  [i4h-digital-twin](https://github.com/isaac-for-healthcare/i4h-digital-twin)].
- Our object scans become meshes and Gaussian splats; the kit's synthetic renders trained the detector [REAL, memory
  `m3pro-object-scan`, `kit-detector`].
- Our M3 twin: URDF -> USD, a privileged expert, a 10 Hz recorder, a twin within ~5 mm rms of the real arm on a ChArUco
  board [REAL, memory `m3pro-sim-expert`].

### Inferences [OURS]
- The manipulator's URDF = Ranger Mini + a prismatic column (stroke 0.4-0.7 m) + the Lite 6 + head (camera, screen,
  microphone). Write it once; it serves the sim, MoveIt and the risk analysis drawings.
- The M3's sim expert pattern (our real picker told the true poses) carries over to the manipulator for tray and
  trolley episodes.

## 5. NVIDIA's healthcare tools: what Isaac for Healthcare offers us

### Takeaway
Isaac for Healthcare's "Hospital Automation Robotics" workflows are the closest public match to tasks 2-3 (push a
cart, move a surgical tray, scissors into a tray), but they target a Unitree G1 humanoid and an SO-ARM 101 with GR00T
policies, need Linux with 16-48 GB of VRAM, and publish no checkpoints in the README. Use them as references and their
assets as props, not as our pipeline. Isaac Sim's bundled `hospital.usd` is a free second, unseen hospital for
showing that navigation generalises.

### Evidence
- i4h-workflows: laparoscopic (dVRK PSM, STAR), ultrasound, endoluminal, and hospital automation: `assemble_trocar`,
  `locomanip_push_cart` ("Walk to and push a cart", G1, GR00T N1.6), `locomanip_tray_pick_and_place` ("Move a surgical
  tray from a shelf to a cart", G1, GR00T N1.6), `scissor_pick_and_place` ("Pick up scissors and place them in a
  tray", SO-ARM 101, GR00T N1.5 or N1.7, also rule-based), tablecloth spreading via XR hand tracking; Isaac Sim 6.0;
  "Ubuntu 22.04 or 24.04 on x86_64 or aarch64"; "16 GB VRAM minimum for Isaac Sim; 48 GB recommended for i4h
  workflows"; no releases [DOC, [i4h-workflows](https://github.com/isaac-for-healthcare/i4h-workflows) and its
  README].
- Isaac Sim environments include `hospital.usd`, "A hospital environment, with multiple rooms and spaces" [DOC,
  [environment assets](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/assets/usd_assets_environments.html)].

### Inferences [OURS]
- Our GPU PCs run Windows; the i4h workflows state Ubuntu. Running them would need a Linux machine; the assets (USD)
  work in our Windows Isaac Sim.
- Saying "our tray task runs in NVIDIA's healthcare assets" is cheap credibility for the application; NVIDIA is the
  hardware sponsor of the challenge.

## 6. People in the scene

### Takeaway
A walking patient for the guidance demo is available through Isaac Sim's animated people (omni.anim.people / Isaac
Replicator Agent, with a navmesh) or through HuNavSim's Isaac Sim wrapper (social-force agents with reactions to the
robot). Both need scripting to make the "patient" follow our robot; a simple scripted follower (go to a point 1.2 m
behind the robot each second) is enough for the video.

### Evidence
- Isaac Sim 5.0 docs: Omni.Anim.People "is being deprecated in the next release"; move to Isaac Replicator Agent
  (IRA), which builds on it; navmesh-based navigation needs a NavMesh built for the stage [2nd, search summary of the
  [5.0 page](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/action_and_event_data_generation/ext_replicator-agent/ext_omni_anim_people.html)
  and an NVIDIA forum thread].
- HuNavSim: wrappers for Gazebo, Webots and Isaac Sim (`Hunav_isaac_wrapper`, v2.0), six reactions to the robot,
  social-force motion, "Tested in ROS2 Humble" [DOC, [hunav_sim](https://github.com/robotics-upo/hunav_sim)].

### Inferences [OURS]
- Check what replaced omni.anim.people in Isaac Sim 6.0 before building on it (not verified this session).

## 7. ROS 2 in the loop on our Windows GPU PCs

### Takeaway
Isaac Sim 6.0 on Windows ships internal ROS 2 libraries (Humble by default, Jazzy by a manual override) and can also
use a native Jazzy installed with Pixi. The simplest loop: Isaac Sim on hekate publishes the robots' sensors with the
internal Jazzy libraries over the lab LAN; our task stack, Nav2, the gateway and the mock controller run on the AGX
Orin itself (JetPack 7.2, Jazzy), so the application can say the twin was driven by the computer that drives the
robot. Doors and the CT move in the sim from `sensor_msgs/JointState`, which avoids custom message types inside Isaac
Sim, exactly as the organisers' own `joint_bridge_node` does for RViz.

### Evidence
- "In Windows, Isaac Sim automatically loads the internal ROS 2 Humble libraries, if no other ROS libraries are
  sourced"; to use Jazzy, set `ROS_DISTRO=jazzy`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` and the Jazzy library folder on
  `PATH`; "The Pixi workspace installs ROS 2 Jazzy natively on Windows via RoboStack conda packages"; "Pixi-based
  native ROS 2 installation is only supported with ROS 2 Jazzy"; WSL for Humble is "(Deprecated). Custom ROS
  Interfaces are not supported" [DOC,
  [ROS 2 installation, other platforms, 6.0](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_ros_other_platforms.html)].
- The organisers' `hospital_arena` package "converts the telemetry of the devices into a JointState message to animate
  the URDF 3D scene" [RULE, RB 8.1.4.2].
- hekate: Windows 11, RTX 5090, Isaac Sim 6.0; "Ask before ... any firewall, SSH, scheduled-task or software change
  there" [REAL, CLAUDE.md].

### Inferences [OURS]
- Before the Orin arrives: the gateway, mock controller and task machine run in a Jazzy container on a Linux PC or in
  Isaac Sim's own Python (scripted, no Nav2) for early videos. Installing Pixi on hekate is a software change: ask.
- DDS between Windows and Linux on the lab LAN usually works with the same RMW and domain; firewall rules on hekate
  may block it (ask before changing them).

## 8. The application video: a storyboard

### Takeaway
2:30-2:45 at 1080p: real robots for every sub-skill we can show (with success counts), the Isaac Sim twin for the
full final and for the fleet, a split screen with the robot's view and its plan everywhere, the E-stop and a fault
injection, and an honest label on every simulated or sped-up clip.

### Evidence
- Required and desired content [RULE, RB 1.4.3]; no AI generation or enhancement [RULE, RB 1.3.2].
- @Home 2027's call asks for a video of at most 10 minutes showing at least five integrated abilities [DOC, via the
  platform notes]; it is the nearest template for format conventions.

### Inferences [OURS]
| time | shot | real or sim | overlay |
|---|---|---|---|
| 0:00-0:10 | team, both robots, the arena twin from above | real + sim | title, #Healthineers #HCC@RC27 |
| 0:10-0:40 | task 1: the M3 greets an external actor, verifies name and birth date on its screen, leads through a door opened via the (mock) controller, CT steps on the controller's panel | real (lab mock-up) | robot camera with person track and gap, intent string, controller panel |
| 0:40-1:05 | task 2: picks an instrument and a tube onto the trolley, couples, pushes the trolley through a door | real, push sped up x2 (labelled) | detections, grasp pose, footprint swap |
| 1:05-1:25 | task 3: wipes a table section far-to-near with the coverage drawn; lays three instruments in the tray layout | real | coverage map, slot targets |
| 1:25-1:50 | task 4: "Catheter type B, please" -> "Catheter type B, correct?" -> "Yes" -> fetch from a shelf, peel, present, release on pull, dispose | real | transcript, function call JSON, hand keypoints |
| 1:50-2:20 | the final, both robots in the organisers' arena: guide, sample to lab, OR clean and tray, catheter | sim (labelled) | phase timeline, controller states, fleet assignments |
| 2:20-2:35 | safety: E-stop press, lidar unplugged -> protective stop, planner killed -> watchdog stop | real | safety state machine |
| 2:35-2:45 | counts (e.g. "pick to trolley 18/20") and links to code and the twin | text | - |
