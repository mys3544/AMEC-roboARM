# Healthcare Challenge 2027: the safety concept, emergency stops, risk analyses and the standards to cite (state of 27 October 2026)

Scope: what the rulebook requires on safety (hardware E-stop, a single-fault-safety concept, a risk analysis per robot
and per configuration, the on-site safety assessment), how to meet it with our robots (a self-built mobile manipulator
on an AgileX base with a UFACTORY arm, and the Yahboom M3 Pro as patient guide), which standards teams should cite and
their current editions, and what wireless E-stops exist. It is a design outline for the TDP and the risk analyses, not
a certification plan: the rulebook says the assessment "does not constitute a certification" (RB 6.4.7).

Labels: **[RB]** = rulebook V1.0 (27.08.2026); **[2nd]** = secondary source or search summary; **[KNOW]** = from my
background knowledge, not re-checked this session (the shared search budget ran out); **[OURS]** = proposal for our
robots.

## 1. What the rulebook requires

### Takeaway
Four things are mandatory: an emergency-stop device that deactivates motors and hazardous functions; a single-fault
safety concept covering every safety-relevant function; a risk analysis in the application for every robot (and every
configuration); and passing the Safety Committee's on-site assessment before the first slot. Safety also scores:
single-fault safety raises MGL (more if demonstrated), "real-time and deterministic control of safety-relevant
functions" raises RPL.

### Cited Findings [RB]
- E-stop "that deactivates the motors and any hazardous functions"; waived for commercial platforms only with a CE
  declaration of conformity (RB 4.1.3.2).
- Single-fault safety "means that the system remains in a safe state after any single fault (single fault tolerance,
  cf. IEC 60601-1). Addressing this aspect is mandatory". A sufficient concept must "identify the safety-relevant
  functions", "describe at least one credible single-fault scenario per function (e.g. encoder failure, communication
  loss, sensor dropout)", and "explain the intended safe state and how it is reached (e.g. controlled stop, fallback
  mode, redundant path)". A demonstration may be "software architecture diagrams" or showing "in software that the
  system transitions to a safe state upon a simulated fault"; "Redundant hardware such as dual processors is explicitly
  not required" (RB 4.2.3.2).
- Risk analysis per robot in the application, highlighting novel aspects and the safety concept (RB 6.4.8); each
  reconfigured variant is evaluated separately with its own risk analysis (RB 6.4.3).
- Safety assessment before the first performance slot; failing teams may fix issues but get no extra slots; the
  committee "may provide recommendations" (RB 6.4.7). The TDP must describe "safety systems used on the robot, including
  emergency stop procedure" (RB 1.4.2).
- Robots must not initiate contact with humans; a patient may lean on the robot and it must bear the force safely
  (RB 6.1.4). Safety shoes for everyone near the robot; no loose clothing or open long hair (RB 1.4.4, 6.4.6). Robots
  over 50 kg that self-stabilise need an FMEA (RB 4.1.2.4). No tipping in any pose (RB 4.1.2.2).
- Each team "is solely responsible for the safety of its members and the safe operation of its robot(s)" (RB 10.2).

## 2. The emergency stop: categories, wiring and what to cut

### Takeaway
Use a hard-wired, latching E-stop chain (red mushroom on yellow, reachable on every robot) that removes power from the
base drives, the arm and the lift column through a two-channel safety relay; the computers stay powered so the robot
can log and report. ISO 13850 and IEC 60204-1 allow only stop category 0 or 1 for emergency stops; for our light
machines category 0 (immediate power removal, the arm's brakes hold) is simplest, a category 1 stop on the base avoids
skidding with the column up.

### Cited Findings
- Stop categories: 0 = uncontrolled stop by immediate removal of power; 1 = controlled stop, power removed once
  stopped; 2 = controlled stop with power kept; "ISO 13850 limits the selection of stop category to Category 0 or 1 and
  excludes Category 2" and IEC 60204-1 does the same [2nd] —
  [machinerysafety101](https://machinerysafety101.com/series/emergency-stop/),
  [Kollmorgen stop functions](https://www.kollmorgen.com/en-us/developer-network/stop-and-emergency-stop-function)
- Stretch 4 has a runstop button in its head that, per its quick start, cuts power to all motors; older Stretch docs
  call it a pause and warn not to rely on software stops for safety [2nd] —
  [Stretch 4 safety guide](https://docs.hello-robot.com/stretch-4-safety-guide/)
- UFACTORY arms ship with an E-stop input on the control box (product behaviour) [KNOW]; AgileX bases carry an E-stop
  button on the chassis [KNOW] — verify both wirings with the vendors.

### Inferences [OURS]
- Mobile manipulator: two mushroom E-stops (one on each side at ~1 m) plus the base's own; a safety relay (e.g. a
  PNOZ- or 3SK1-class two-channel relay, EUR 150-300 est.) opens the contactors for base motor power and column power and
  drives the arm's E-stop input. Reset is manual and deliberate (a separate reset button), never automatic.
- M3 Pro: add a mushroom E-stop in series with the motor-driver and servo supply (the Jetson stays on). Today it has no
  hardware E-stop at all, which would fail RB 4.1.3.2.
- R2s and the X3 are not proposed for the competition; if used, the same retrofit applies.

## 3. Wireless emergency stops

### Takeaway
A wireless E-stop is a useful second channel (a team member can stop a robot from across the arena) but must not
replace the on-robot buttons. PLd/SIL 2 units exist (FORT Robotics, Anderson Controls, Cattron, efa) at quote-only
prices; in the EU use a 2.4 GHz or 868/869 MHz product, not the 900 MHz US band, and expect a crowded 2.4 GHz band at
RoboCup (the arena WiFi is 2.4 GHz too). The receiver must fail safe: loss of the radio heartbeat opens the chain.

### Cited Findings [2nd]
- FORT Robotics wireless E-stop "Designed to meet ISO 13849 PLd Cat 3", SIL 2 (IEC 61508); handheld unit paired with a
  receiver; price not shown — [Clearpath store](https://store.clearpathrobotics.com/products/wireless-e-stop),
  [New Eagle](https://store.neweagle.net/shop/products/autonomous-remote-safety/fort-robotics-foundation-wireless-e-stop/)
- Anderson Controls 2.4 GHz mesh E-stop: "SIL 2 based on the IEC61508" and "PL d based on EN ISO 13849-1", CE; DC
  receiver 9-30 VDC with two independent 5 A outputs (one auto-resets, one needs a power cycle); up to 10 machines;
  transmitters sold separately; no prices — [andersoncontrol.com](https://andersoncontrol.com/shop/wireless-e-stop-systems/)
- Cattron Safe-D-Stop (PL d) and efa Controls (PLd, SIL2) also exist —
  [wireless-emergency-stop.com](https://wireless-emergency-stop.com/wireless-e-stops/safe-d-stop),
  [efa Controls](https://efa-controls.com/en/wireless-e-stop-a-multi-operator-emergency-stop/)

### Inferences
- One transmitter for the whole fleet (mesh, 1-to-many) suits a two-robot team.
- Budget a few hundred to ~EUR 2k (est.); ask vendors for EU-band units and a heartbeat timeout figure.

## 4. Single-fault safety: definition and a concept per safety function

### Takeaway
IEC 60601-1 calls equipment "single fault safe" when it stays free of unacceptable risk when one means of risk
reduction is defective or one abnormal condition is present. For us that means: list the safety functions, give each a
credible single fault, and show that the robot then reaches a defined safe state by a path that does not depend on the
failed element. A software watchdog chain plus the hardware E-stop meets the minimum; an independent microcontroller
that drops the motor-enable lines when heartbeats stop is a cheap "demonstrated implementation".

### Cited Findings
- IEC 60601-1: single fault condition (3.116) = "one means of reducing risk is defective, or one abnormal condition is
  present"; single fault safe (3.117) = free of unacceptable risk during the expected service life under a single
  fault; ISH1 (2021): essential performance must also be kept under single fault conditions [2nd] —
  [Johner Institute](https://johner-institute.com/articles/risk-management-iso-14971/functional-safety/),
  [Johner, essential performance](https://blog.johner-institute.com/regulatory-affairs/essential-performance-of-medical-devices/)
- RB 4.2.3.2 minimum content (section 1 above).

### Inferences [OURS] — concept table for the mobile manipulator
| safety function | credible single fault | safe state | how it is reached |
|---|---|---|---|
| SF1 emergency stop | one relay contact welds | motors unpowered | two-channel safety relay; second channel still opens; relay self-test at reset |
| SF2 motion supervision (heartbeat) | main computer or ROS node hangs | base and arm stopped (cat 1/2), then power off | independent MCU expects a 20-50 Hz heartbeat; on timeout it drops the base enable, triggers the arm stop input and the column hold |
| SF3 obstacle and person stop | lidar dropout or frozen scan | base stopped | scan freshness and plausibility check (timestamp, range variance); stale → protective stop; bumper or depth camera as second sensor |
| SF4 speed and separation near people | localisation wrong | slow mode | speed capped by the low-level controller near detected people and whenever localisation covariance is high |
| SF5 arm collision | joint position read wrong / collision missed | arm stopped | vendor collision detection + current/torque limit; reduced speed and force limits when a person is within reach |
| SF6 tip-over | column up while braking or turning | column lowered, base slow | interlock: column above threshold → base speed limit; IMU tilt watchdog → stop |
| SF7 communication with the Hospital Controller lost | WiFi drop | finish or pause the current action safely, hold position | gateway watchdog; never depends on the network for any safety function |
| SF8 battery | cell overheat or short | power cut | BMS + fuses; charging only when attended |
| SF9 lift column pinch | end switch fails | column stops | column's own end switches + safety nut (LINAK LC3) + software limits |

- M3 Pro (patient guide): SF1 (new hardware E-stop), SF2 (heartbeat on its Orin to the base board), SF3 (two lidars;
  stale scan → stop), SF4 (speed capped at walking pace near the patient), SF7; plus a specific hazard: tripping over a
  knee-high robot — a mast with a flag and screen at ~1.1 m and a rule to lead at 1-1.5 m distance. The arm stays
  folded while guiding. The M3's arm servos give no joint read-back (project memory), so the arm must never move near a
  person.

## 5. The risk analysis per robot

### Takeaway
Write each robot's risk analysis as an ISO 12100 loop (limits of the machine → hazard identification → risk estimation
→ evaluation → reduction → residual risk), with the hazard list seeded from ISO 10218-2 Annex A plus the mobile hazards
(collision while driving, tipping, battery, pinch at the column, trolley run-away, the patient leaning on the robot),
scored on a declared severity x probability matrix. One table per robot and per configuration, about 20-30 rows each.

### Cited Findings [2nd]
- ISO 12100 loop and a 5x5 severity-by-probability matrix as a common practice; the standard does not prescribe the
  scale — [theresarobotforthat.com](https://theresarobotforthat.com/blog/iso-12100-risk-assessment-template/)
- A filled-in robot-cell template with typical hazards based on ISO 10218-2 Annex A —
  [Automata risk assessment guidance](https://automata.tech/wp-content/uploads/2020/05/Automata-RiskAssessmentGuidance.pdf)
- A 2026 FMEA-based risk assessment template for collaborative robots —
  [CiteDrive / DOI 10.1177/1748006x261461714](https://www.citedrive.com/en/discovery/a-failure-mode-and-effects-analysis-fmea-risk-assessment-template-for-collaborative-robot/)

### Inferences [OURS] — columns and first rows
Columns: id; life-cycle phase (transport, setup, run, charging, maintenance); task; hazard (ISO 12100 type:
mechanical, electrical, thermal, ergonomic, environment); hazardous situation; who is exposed (patient actor,
operator actor, team, referee, public behind the 1.4 m walls); severity S1-S4; probability P1-P4; risk; measures
(inherent design → protective devices → information); residual risk; verification (test, video, inspection).
First rows for the mobile manipulator: base drives into a person (lidar + speed limit + bumper); arm strikes the
operator during handover (low speed, force limit, handover pose outside the face zone, the operator takes the item);
pinch between column stages (covers, end switches); trolley rolls away on release (couple/brake procedure); tipping with
the column up (interlock, IMU); battery fault while charging (BMS, fuse, attended charging); sharp instruments carried
(gripper holds by the handle, tray carries blades); trip over cables (none: no cable power allowed); patient leaning on
the robot (static stability test with a declared force; refuse physical support in 2027).

## 6. Which standards to cite, and their status in October 2026

### Takeaway
Cite ISO 12100 (risk assessment), ISO 13482 (service robots; edition 2 at FDIS stage as "Robotics — Safety requirements
for service robots"), ISO 10218-1/-2:2025 (the arm; ISO/TS 15066 is now inside it), ISO 13849-1 (performance levels of
safety functions), IEC 60204-1 and ISO 13850 (electrical equipment and E-stop), ISO 3691-4:2023 (personnel detection
on driverless mobile platforms, borrowed for the base), and for the medical grade IEC 60601-1 (single fault safety),
ISO 14971 (medical risk management), ISO 13485 (quality management) and IEC 62304 (software life cycle) — as a concept
of how the design would comply, not as claimed conformity. Mention that the EU Machinery Regulation 2023/1230 applies
from 20 January 2027.

### Cited Findings
- ISO 13482: the ISO page for edition 2 (ISO/FDIS 13482, "Robotics — Safety requirements for service robots") lists it
  at the FDIS approval stage, to replace ISO 13482:2014; the CEN draft is titled with "(ISO/FDIS 13482:2026)"; scope
  widened to personal and professional service robots, person-carrier clauses removed [2nd] —
  [ISO 83498](https://www.iso.org/standard/83498.html),
  [iTeh FprEN ISO 13482](https://standards.iteh.ai/catalog/standards/cen/dffe4c3c-bed0-4d01-86a8-8582e7c5624a/fpren-iso-13482),
  [IEEE Spectrum](https://spectrum.ieee.org/domestic-humanoid-robot-safety-standards). Publication not confirmed.
- ISO 10218-1:2025 and -2:2025 published February-March 2025; functional safety requirements made explicit; collaborative
  requirements formerly in ISO/TS 15066 integrated; cybersecurity added [2nd] —
  [A3 FAQ](https://www.automate.org/robotics/blogs/updated-iso-10218-faq),
  [TÜV SÜD](https://www.tuvsud.com/ja-jp/press-and-media/2025/march/tuv-sud-announcement-iso-10218-1-2025-and-iso-10218-2-2025-released)
- ISO 3691-4:2023 (second edition): personnel detection by pressure-sensitive bumpers or electro-sensitive devices
  covering the full width in the travel direction; stop before reaching a stationary person; zone concept [2nd] —
  [Pilz](https://www.pilz.com/en-INT/company/news/articles/238928),
  [ANSI preview](https://webstore.ansi.org/preview-pages/ISO/preview_ISO+3691-4-2023.pdf)
- EU Machinery Regulation (EU) 2023/1230 applies from 20 January 2027 and repeals Directive 2006/42/EC on that date
  (corrigendum of 4 July 2023 fixed the dates) [2nd] —
  [BAuA](https://www.baua.de/EN/Topics/Work-design/Machine-and-operational-safety/Machinery/Machinery-Directive-Revision)
- ISO 13849-1:2023, ISO 12100:2010, ISO 13850:2015, IEC 60204-1, ISO 14971:2019, ISO 13485:2016, IEC 62304,
  IEC 80601-2-77/-78 (medical robots for surgery / rehabilitation) — editions [KNOW], not re-checked this session.
- The rulebook itself names IEC 60601 and ISO 13485 as examples for MGL and IEC 60601-1 for single fault tolerance
  (RB 4.2.3, 4.2.3.2).

### Inferences
- Cite ISO 13482 as "ISO 13482:2014 (edition 2 at FDIS, 2026)" and design to the new structure where possible; it is
  the closest standard to a hospital service robot.
- For the arm, ISO 10218-1/-2:2025 with its integrated power-and-force-limiting values is the reference for handover
  speeds and forces; we claim no conformity, only design intent.

## 7. The medical-grade extras that cost little

### Takeaway
Three cheap MGL items: a manufacturing concept for cleanability (printed covers now, injection-moulded or stainless
seamless parts in series, wipe-disinfectable, no dirt traps), a short IEC 60601-1 / ISO 13485 / ISO 14971 concept
(how the design and our development process would map to them), and a live fault-injection demo at the inspection
(pull the lidar cable, kill the planner, cut WiFi → robot reaches its safe state).

### Cited Findings [RB]
- Manufacturing concept clause: "printed in Fused Deposition Modeling (FDM) for the prototype; manufactured in series as
  an injection-moulded or electro-polished stainless-steel part" is the rulebook's own example (RB 4.2.3.1).
- Demonstrations in software of the transition to a safe state count as implementation (RB 4.2.3.2).

## 8. Gaps
- No standard text was read (all paywalled); clause numbers come from secondary sources.
- Wireless E-stop prices, EU bands and heartbeat timeouts need quotes.
- Whether the Safety Committee accepts Stretch's runstop or a vendor's chassis E-stop as the RB 4.1.3.2 device.
- Whether the Safety Committee wants anything beyond RB 4.2.3.2 (for example a declared maximum speed near the patient);
  the next rulebook version should say.
