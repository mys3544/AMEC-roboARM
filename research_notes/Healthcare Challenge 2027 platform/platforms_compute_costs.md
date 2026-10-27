# Healthcare Challenge 2027: platforms, on-board compute and costs (state of 27 October 2026)

Scope: which robot hardware can cover the four tasks and the final under the rulebook's constraints (≤120 V DC,
own battery, 0.75 m doors, all compute on board; see `rules_and_task_demands.md`), what it costs and how fast it comes.
Covered: turnkey mobile manipulators (Hello Robot Stretch 4, PAL TIAGo Pro, Galaxea R1 Lite, AgileX Cobot Magic), arms
for a build (UFACTORY Lite 6 / xArm 6 / 850, Kinova Gen3, AgileX PiPER), bases (AgileX Tracer 2.0, Ranger Mini 3.0,
Scout Mini), lift columns, and Jetson compute after NVIDIA's July 2026 price rise. What the lab owns: 2x Yahboom
ROSMASTER R2, the X3 Plus, the M3 Pro (Orin NX 16 GB) and a UFACTORY Lite 6.

Price labels: **list** = vendor's own published price; **reseller** = a shop's published price; **quoted** = a quote to
us (none yet); **est.** = my estimate. All prices were read on 2026-10-27 unless a date is given. USD and EUR are kept
as published; conversions are estimates. Slovak VAT (23 % since 2025, not re-checked this session) and import duty on
US goods come on top of excl.-VAT figures. Other labels: **[2nd]** = search summary, not opened; **[CLAIM]** = vendor
claim. The shared web-search budget ran out near the end of the session; items that needed one more search are marked
"verify".

## 1. Constraints that filter the platforms

### Takeaway
The rules and the arena reduce the choice to: a base no wider than ~0.6 m that can push a trolley; an arm working
between ~0.6 and ~1.5 m height with ≥0.5 kg payload (more for drawers, doors and wiping); DC power only; and an on-board
GPU computer. Arms whose controller needs mains AC are out.

### Cited Findings
- Door 0.75 m x 1.95 m minimum; ≤120 V DC; own battery; no cable power; E-stop; no external compute during a run
  (RB 4.1.2-4.1.4, 8.1.1.2; detail in `rules_and_task_demands.md` sections 4-6).
- Work heights in the arena model: counters 0.92 m, desks 0.99-1.03 m, cabinets to 1.92 m, bin 0.61 m (same note,
  section 8).

### Inferences
- A trolley needs tens to >100 N of push force; that comes from the base and a coupling, not from a light arm.
- One machine that does tasks 2-4 also does task 1 if it carries a head (screen, microphones, speaker) at ~1.2-1.5 m;
  then a single robot can attempt the whole final.

## 2. Turnkey mobile manipulators

### Takeaway
Stretch 4 is the only turnkey option with a public price under USD 30k and a lead time of weeks; it is light, safe by
design and runs a Jetson Orin NX like our M3, but carries 2.5 kg at reach, cannot push heavy trolleys or beds, reaches
only ~1.2 m high, and is sold for "laboratory and research use" without CE. TIAGo Pro is the most capable but priced
on request (third parties guess USD 85-95k). Galaxea R1 Lite (USD 39,999) and AgileX Cobot Magic (from EUR 52,000)
are data-collection platforms with less mature safety and EU support.

### Cited Findings
**Hello Robot Stretch 4** (announced 12 May 2026 [2nd] —
[Robotics & Automation News](https://roboticsandautomationnews.com/2026/05/12/hello-robot-unveils-stretch-4-a-simply-useful-robot-that-puts-people-first/101401/))
- List USD 29,950; "typically ships within 2-4 weeks from receipt of purchase order or pre-payment"; options: docking
  station USD 1,495, parallel gripper USD 1,495, "NVIDIA Jetson Orin NX" 16 GB USD 2,995, spare battery USD 1,495,
  3-year warranty USD 2,495; "currently only certified for laboratory and research use"; Stretch 3 upgrades not
  offered; ships "to most countries internationally"; no European distributor listed —
  [hello-robot.com/purchase](https://hello-robot.com/purchase/)
- Datasheet (Rev5, as launched): 9 DOF (3-wheel omni base, lift, telescoping arm, 3-axis wrist); reach 55 cm + 6 cm
  wrist; payload 2.5 kg arm extended, 4 kg retracted; 46 kg (33 kg without ballast); 160 cm tall, 43 cm diameter
  (product page: 45 cm); lift range 120 cm; top speed 60 cm/s; 20 mm threshold; dual Hesai JT128 hemispherical lidars,
  head and wrist cameras; Intel NUC 15 (Core Ultra 5 225H, 32 GB) plus optional Orin NX 16 GB; force limiting via motor
  current, 100 Hz watchdogs, IMU tilt avoidance; runstop button in the head; 512 Wh LiFePO4; IP20; 10-30 °C; "not yet
  certified for compliance with FCC Class A" —
  [datasheet PDF](https://hello-robot.com/wp-content/uploads/2026/05/HelloRobot-DataSheet-Stretch-4-Rev5_AsLaunched.pdf),
  [product page](https://hello-robot.com/stretch-4/)
- Runstop behaviour: the Stretch 4 quick start says it cuts power to all motors; older docs describe it as a pause
  that makes joints backdrivable; no safety rating is given [2nd] —
  [Stretch 4 safety guide](https://docs.hello-robot.com/stretch-4-safety-guide/),
  [Stretch 4 quick start](https://docs.hello-robot.com/stretch-4-quick-start-guide)
- Stretch 3 (USD 24,950 in our 2026-10-20 note) is no longer on the purchase page.

**PAL Robotics TIAGo Pro** — [pal-robotics.com](https://pal-robotics.com/robot/tiago-pro/)
- Two 7-DoF arms with torque sensing in every actuator, 3 kg each, 6 arm brakes; torso lift 35 cm; mecanum omni base;
  Intel i5/i7, NVIDIA Jetson optional; RGB-D head, 4-mic array, two 360° laser scanners, 10" touchscreen head; ROS 2 on
  Ubuntu LTS with PREEMPT-RT; emergency stop button; price "Request a quote"; no lead time given.
- Third-party price guesses: USD 85-95k (Robofy), USD 90k "not verified" (humanoid.guide), EUR 29-59k (likely the
  older TIAGo) [2nd] — [humanoid.guide](https://humanoid.guide/product/tiago-pro/),
  [Robofy](https://robofy.world/compare/pal-robotics-tiago-vs-pal-robotics-tiago-pro)

**Galaxea R1 Lite** — store list USD 39,999 (bundles to ~USD 45k); 6-DoF arms, 70 cm reach, 3.5 kg; torso to 1.7 m
[2nd] — [galaxea-dynamics.com](https://galaxea-dynamics.com/collections/all-products)

**AgileX Cobot Magic** (Mobile-ALOHA-style teleoperation platform) — reseller from EUR 52,000 excl. VAT, out of stock —
[Generation Robots](https://www.generationrobots.com/en/274_agilex)

### Inferences
- Stretch 4 bundle for this challenge (robot + Orin NX + parallel gripper + dock): USD 35,935 list, ~EUR 31-33k before
  shipping, duty and VAT (est.), so roughly EUR 40-45k landed (est.).
- Stretch's runstop would need to be shown to "deactivate the motors" to the Safety Committee; without a CE declaration
  the waiver in RB 4.1.3.2 does not apply.
- Trolley pushing with a 46 kg three-wheel omni base is untested as far as I found; this is the main functional doubt.

## 3. Arms for a self-built mobile manipulator

### Takeaway
The owned Lite 6 (0.6 kg, 440 mm) is enough for instruments, sample tubes and packages and for the application demos,
but weak for drawers, doors and wiping. The UFACTORY 850 (5 kg, 850 mm, ~USD 8.4-9k list) or xArm 6 (5 kg, 700 mm)
keep the same SDK and run from a DC control box; a Kinova Gen3 (4 kg, 902 mm, joint torque sensing, integrated
controller) is the premium choice at ~USD 33.5k.

### Cited Findings
- **Lite 6**: list USD 2,999; 600 g payload, 440 mm reach, ±0.5 mm, 8 kg, built-in control box, Python/ROS/ROS 2/C++,
  collision detection — [ufactory.cc](https://www.ufactory.cc/lite-6-collaborative-robot/). Resellers: USD 4,186 (Knox
  Labs), EUR 4,516 kit with grippers (RobotShop EU) [2nd].
- **xArm 6**: UFACTORY's comparison table USD 8,399; 5 kg, 700 mm, ±0.1 mm, 12.5 kg, separate control box [2nd];
  reseller EUR 8,399 (Top3DShop, 2-3 weeks), EUR 9,679 incl. tax (RobotShop EU) [2nd] —
  [RobotShop EU](https://eu.robotshop.com/products/xarm-6-dof-robotic-arm)
- **UFACTORY 850**: USD 8,399-8,999 in UFACTORY's own tables; 5 kg, 850 mm, ±0.02 mm; sold with a DC control box
  (RobotShop USD 10,259 incl. US tariffs, "On Demand") [2nd] —
  [RobotShop 850 DC](https://www.robotshop.com/products/ufactory-850-6-dof-robotic-arm-850mm-w-dc-control-box-2m-cables)
- **xArm DC control box**: newer listings say "24–72 VDC input for battery & mobile use", 24 VDC output 672 W max,
  2.6 kg; an older UFACTORY page says 24 V only — verify [2nd] —
  [RobotShop](https://www.robotshop.com/products/xarm-dc-control-box)
- **Kinova Gen3 7-DoF**: RobotShop US USD 33,500 incl. tariffs; guides put it at USD 30-35k, up to ~50k configured
  [2nd] — [RobotShop](https://www.robotshop.com/products/kinova-gen3-robot-7dof-l53),
  [grabarobot price guide](https://grabarobot.com/blog/kinova-robot-arm-price-guide-2026/). Tidyboy (RoboCup@Home
  winner) used it: 902 mm, 4 kg, "lack of a separate controller box", but "high price tag and limited joint range" —
  [Tidyboy-OPL TDP](https://athome.robocup.org/wp-content/uploads/OPL-Tidyboy-OPL-TDP.pdf)
- **AgileX PiPER** 6-axis: reseller EUR 1,700 excl. VAT, in stock — [Generation Robots](https://www.generationrobots.com/en/274_agilex)
- Franka-class arms with an AC control unit: excluded by the 120 V DC rule unless the vendor offers a DC variant (not
  checked).

### Inferences
- Code written for the Lite 6 (xArm Python SDK, `xarm_ros2`) carries over to the 850/xArm 6, so starting on the Lite 6
  and upgrading after selection costs little software.
- For wiping, a passive spring-loaded pad on the Lite 6 is a cheap first answer; force-controlled wiping argues for the
  Kinova (torque sensing) or a wrist F/T sensor.

## 4. Bases

### Takeaway
AgileX bases are the practical choice: Tracer 2.0 (24 V, 0.61 m wide, 150 kg payload, EUR 7,500 excl. VAT) is the
cheapest; Ranger Mini 3.0 (0.50 m wide, crab/spin modes, suspension, 100-120 kg, from EUR 12,480 excl. VAT) passes
doors and aligns sideways to tables and trolleys more easily. Both were out of stock at the EU reseller with 3-6 week
delivery. The Scout Mini is too small for a lift column and an arm.

### Cited Findings
- **Tracer 2.0**: EUR 7,500 excl. / 9,000 incl. VAT, "delivery within 3 weeks" (out of stock); 702 x 610 x 169 mm,
  54-56 kg, 150 kg payload, 2.0 m/s, 24 V 30 Ah lithium, differential drive, 10 mm obstacle, "flat ground only", ROS
  support — [Generation Robots](https://www.generationrobots.com/en/404310-autonomous-mobile-robot-tracer-20-agv.html).
  US resellers USD 5,850-9,500 [2nd].
- **Ranger Mini 3.0**: from EUR 12,480 excl. / 14,976 incl. VAT (out of stock) —
  [Generation Robots](https://www.generationrobots.com/en/274_agilex); RobotLAB: 720 x 500 x 345 mm, 100 kg (some
  distributors 120 kg), "48V lithium iron phosphate, 24Ah", 7-8 h, spin / traverse / diagonal / Ackermann modes,
  2 m/s, CAN, ROS 1/ROS 2 SDK, USD 10,800 or 11,900 (page inconsistent) —
  [RobotLAB](https://www.robotlab.com/store/ranger-mini-for-research/). Battery voltage from one reseller only — verify.
- **Scout Mini**: EUR 6,112.60 excl. VAT (out of stock, 4-6 weeks) — Generation Robots.
- Clearpath (Rockwell) bases: not checked this session.

### Inferences
- Ranger Mini's traverse mode lets the robot slide along an OR table while the arm wipes a strip, and dock sideways to a
  trolley handle; with a differential Tracer the arm must cover that motion.
- A 48 V base feeds the xArm/850 DC box directly (if 24-72 V is confirmed) and the Lite 6 through a 48-to-24 V DC-DC
  converter; the Tracer's 24 V matches the Lite 6 directly.

## 5. Lift columns

### Takeaway
A 24 V DC electric column of 400-700 mm stroke puts the arm's shoulder anywhere from ~0.6 to ~1.3 m on an AgileX deck.
LINAK's LC3 is a 24 V column sold for operating tables and X-ray equipment (useful for the MGL story); the
cobot-specific Ewellix LIFTKIT costs ~USD 6.4k and its UR/TM versions take mains AC. Prices for 24 V columns were not
found; budget EUR 0.8-2.5k (est.) and ask for quotes.

### Cited Findings
- LINAK LC3: 24 V DC, 4,000-6,000 N compression, strokes 200/400-700 mm, IPX4 (IPX6 option), safety nut and end
  switches; positioned for "operating tables and X-ray equipment"; no IEC 60601-1 statement found [2nd] —
  [Delta (LINAK partner)](https://www.delta-emea.com/D-bot/ecosystem/partners/accessories-linak-elevate-lc3.html),
  [LINAK](https://linak.com/segments/medline-careline/tech-trends/quiet-and-powerful-lifting-columns-for-medical-equipment/fastsafe-powerful)
- Ewellix LIFTKIT for UR: USD 6,411.63 (700 mm) / 6,579.52 (900 mm) on Vention, 1,500 N push; LIFTKIT-TM main input
  120/230 VAC [2nd] — [Vention](https://vention.io/parts/telescopic-lift-column-for-ur-700mm-stroke-1991)
- Progressive Automations: 24 V three-stage column LG-07-3RM-32-440-HS-24V (32" stroke, 440 lb), price not shown; the
  USD 352 FLT-11 runs on 120 VAC [2nd] — [progressiveautomations.com](https://progressiveautomations.com/products/flt-11)

### Inferences
- Stability: an 8 kg arm at 1.3 m on a 55 kg base is benign for tipping when the base stands; the risk is braking or
  turning with the column up. An interlock (column high → base speed limit) belongs in the safety concept.

## 6. On-board compute

### Takeaway
NVIDIA raised Jetson prices in July 2026: the AGX Orin 64 GB developer kit is now USD 3,499 (was 1,999) and the AGX Thor
kit USD 5,499 (was 3,499), both often out of stock. One AGX Orin 64 GB holds our whole stack (speech, a 4B LLM, the
detector, an ACT/VLA policy, Nav2); Thor (40-130 W, 128 GB) is the choice for multi-billion-parameter VLAs. Since every
admitted team gets an NVIDIA hardware package, buy at most one AGX Orin before selection and plan Thor from the package.

### Cited Findings
- July 2026 price table (USD, modules at 1k units): AGX Thor dev kit 3,499 → 5,499; T5000 2,999 → 4,999; T4000
  1,999 → 2,999; AGX Orin dev kit 1,999 → 3,499; AGX Orin 64 GB module 1,599 → 2,999; Orin NX 16 GB module 599 → 999 —
  [CNX Software, 2026-07-22](https://cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101)
- Power: Thor T5000 40-130 W, T4000 40-70 W, AGX Orin 15-60 W; Thor and AGX Orin kits out of stock on NVIDIA's
  marketplace on 2026-10-21 (Newegg USD 5,499 for Thor) —
  [physicalailab.io, checked 2026-10-21](https://physicalailab.io/en/jetson-agx-thor-price-vs-orin/)
- Jetson T3000/T2000 announced for Q1 2027 availability [2nd] —
  [NVIDIA blog](https://blogs.nvidia.com/blog/jetson-thor-robotics-edge-ai-agent/)
- IGX Thor: functional safety island designed to ISO 26262 / IEC 61508; NVIDIA says partners support "accelerating
  medical certifications (IEC 60601, 62304)"; 10-year lifecycle; no price [2nd] —
  [IGX Thor datasheet](https://developer.download.nvidia.com/assets/igx/robotics-datasheet-igx-thor-module-board-kit-nvidia-us-4845700-web.pdf),
  [developer.nvidia.com/igx](https://developer.nvidia.com/igx)
- FAQ: "Every team admitted to the Challenge receives the same package and the same access, provided directly by
  NVIDIA" — [FAQ](https://www.robocup2027.event.siemens-healthineers.com/blocks/faq). Contents not published.
- Our own numbers (project memory): YOLO26n kit detector FP16 3.0 ms on the Orin NX; ACT 5.9 ms TensorRT FP16 on the
  M3's Orin NX at a true 10 Hz; Qwen3.5 4B Q4 answers in ~0.65 s on the laptop's RTX 2070 Super (not yet on a Jetson).
  Model export details: `research_notes/Model export formats for our hardware/jetson_orin_stack.md`.

### Inferences
- Workload sizing for the mobile manipulator (est.): ASR (Parakeet-class, ~1-2 GB), a 4B LLM at Q4 (~3 GB), TTS
  (CPU), YOLO-class detection and segmentation, one manipulation policy, Nav2 and SLAM on the CPU. That fits an AGX Orin
  64 GB (and probably a 32 GB) with room; the M3 keeps its Orin NX 16 GB for its own voice and navigation (research
  topic 4).
- JetPack 7 (Thor) is built on Ubuntu 24.04, which would allow native Jazzy for the controller-facing node (from
  knowledge; not re-checked this session). On JetPack 6 (Ubuntu 22.04, our robots) the Jazzy gateway runs in a
  container.
- Power budget (est.): AGX Orin 15-60 W or Thor 40-130 W; arm 50-200 W while moving; base 100-300 W while driving;
  sensors and head ~30-50 W; total ~250-500 W while working. Ranger Mini 48 V x 24 Ah ≈ 1.15 kWh → ~2.5-4 h; Tracer
  24 V x 30 Ah = 720 Wh → ~1.5-3 h. DC hot-plug charging is allowed between runs (RB 4.1.3.1).

## 7. Cost roll-ups (excl. VAT, est. unless marked)

### Takeaway
A self-built mobile manipulator with our Lite 6 costs about EUR 19-25k on a Ranger Mini (EUR 14-20k on a Tracer);
upgrading to a UFACTORY 850 adds ~EUR 9-11k; a Kinova version lands near EUR 50-60k. A Stretch 4 bundle is about
EUR 40-45k landed. Using only the robots we own costs EUR 1-3k but cannot attempt trolleys, the final or most
manipulation in the OR.

### Inferences (component estimates; list prices where marked)
| item | Ranger build | Tracer build | note |
|---|---|---|---|
| base | 12,480 (reseller list) | 7,500 (reseller list) | out of stock, 3-6 weeks |
| 24 V lift column + controller | 800-2,500 | 800-2,500 | quote needed |
| frame, mounts, wipeable covers | 800-1,500 | 800-1,500 | aluminium profile + printed covers |
| E-stop chain (buttons, safety relay, contactor), wireless E-stop | 600-2,500 | 600-2,500 | wireless PLd units are quote-only |
| Jetson AGX Orin 64 GB dev kit | ~3,300 (USD 3,499 list) | same | or NVIDIA package |
| 2D lidar(s), RGB-D camera, head (screen, mic array, speaker) | 900-2,500 | 900-2,500 | the AgileX bases ship without lidar (verify) |
| DC-DC converters, fuses, cabling | 200-400 | 200-400 | |
| arm | 0 (Lite 6 owned) | 0 | upgrade: 850 + DC box ~9-11k |
| **total stage 1** | **~19-25k** | **~14-20k** | |

- M3 upgrades for task 1 (mushroom E-stop in the motor supply, a ~1.1 m mast with screen, mic and speaker): EUR 200-600.
- Event costs (not covered by the organisers): travel and a week's stay for ~5 people, van transport Košice-Nuremberg,
  RoboCup fees (amounts not found) — budget EUR 6-12k (est.).

## 8. Gaps
- No quotes yet from any vendor; TIAGo / TIAGo Pro price unknown; Clearpath not checked.
- Ranger Mini 3.0 battery voltage and the xArm DC box input range each rest on reseller pages.
- Whether Stretch 4 can push a hospital trolley; whether its runstop satisfies RB 4.1.3.2.
- Contents and timing of the NVIDIA package.
