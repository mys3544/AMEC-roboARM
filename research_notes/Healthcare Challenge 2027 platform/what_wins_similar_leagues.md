# What wins leagues like the Healthcare Challenge: RoboCup@Home 2024-2026, HEART-MET, euROBIN (state of 27 October 2026)

Scope: what the winners of the closest competitions built and did, read from their own papers and pages, and what
transfers to the Healthcare Challenge 2027, whose rules forbid any external computing or network during a run (see
`rules_and_task_demands.md`). Covered: RoboCup@Home Open Platform League (OPL) 2024-2026 (NimbRo@Home, Tidyboy), the
league's 2027 format, the METRICS HEART-MET healthcare benchmarks (2022-2023), the euROBIN coopetition (2024), and the
one commercial hospital mobile manipulator with a public design (Diligent Moxi). Not covered: the official @Home 2026
score sheet (not found), DSPL 2025 results (not found), ERL consumer-robot results after 2023.

Labels: **[PRIMARY]** = the team's own paper or page, opened; **[2nd]** = search summary, not opened.

## 1. Who won, 2024-2026

### Takeaway
Two teams dominate the open-platform @Home league: Tidyboy (Pusan and Seoul National Universities) won 2025 and 2026,
NimbRo@Home (University of Bonn) won 2024 and placed second in 2025 and third in 2026. In 2027 @Home is a single open
league; standard platforms are gone. 24 teams competed in @Home 2026.

### Cited Findings
- 2024 OPL (Eindhoven): NimbRo 8,852 points, Tidyboy-OPL 7,495, SocRob@Home 6,901; NimbRo was second after stage 1 and
  led after stage 2 [PRIMARY] — [arXiv 2412.14989](https://arxiv.org/html/2412.14989v1)
- 2025 OPL (Salvador): Tidyboy first, NimbRo second "despite much delayed arrival of our robots" [PRIMARY] —
  [NimbRo@Home page](https://www.ais.uni-bonn.de/nimbro/@Home/),
  [GRC 2026 abstract](https://www.ais.uni-bonn.de/papers/GRC_2026_Memmesheimer_NimbRo_Home.pdf); AIhub round-up [2nd]
- 2026 (Incheon): normalised final scores Tidyboy 100 %, LAR@Home (Portugal) 77 %, NimbRo 74 % [PRIMARY] — NimbRo page.
  eR@sers (Tamagawa) placed 8 of 24 with a Toyota HSR and a Unitree G1 and won the Best Humanoid Award [PRIMARY] —
  [erasers.jp](https://erasers.jp/2026/07/06/results-of-robocup-2026-incheon/)
- 2027 @Home: Open Platform League only, "no separate standard platform league exists"; qualification material due
  6 Dec 2026: a video of at most 10 minutes showing at least 5 integrated abilities, a TDP of 8 pages (Springer LNAI)
  plus a 2-page annex listing robot photos, third-party software and "external computing devices", and a web site with
  publications of the last 5 years [PRIMARY] —
  [@Home 2027 call](https://athome.robocup.org/call-for-participation-robocuphome-2027/)

### Inferences
- The Healthcare Challenge's qualification material mirrors @Home's (TDP, video, web site, previous results), so the
  @Home 2027 format is a good template where the HCC rulebook is silent (page count, video length).

## 2. The winners' hardware

### Takeaway
Both winners use one capable mobile manipulator per robot, not fleets of small robots: NimbRo a modified TIAGo++ (omni
base, lifting torso raised by 200 mm, two 7-DoF arms, an RTX A4500 box on its back), Tidyboy its own 30 kg robot (a
501 mm round three-omniwheel base and one Kinova Gen3 arm reaching from the floor to 1.5 m without a lift). Both kept a
second identical robot or a second platform as a spare.

### Cited Findings [PRIMARY]
- NimbRo: "modified PAL Robotics TIAGo++ Omni robot ... dual 7-DoF arms, an Orbbec Gemini 2 RGB-D camera, an Ouster
  OSDome 3D LiDAR, a wide-FoV camera ..., and an onboard Zotac ZBOX QTG7A4500 GPU workstation"; Zoom AM7 microphone,
  Bose speaker, front screen; silicone-coated custom grippers; "we extended the height of the robot by 200 mm" — GRC 2026
  abstract. 2024 paper: two SICK TiM571 lidars, a 10-inch touchscreen, a Teltonika router with WiFi and 5G fallback, "a
  second, similarly configured robot was kept for redundancy" — arXiv 2412.14989.
- Tidyboy Lucio: "a circular base with three large Omni wheels ... 501mm, which is 16% larger than the HSR"; chose it
  because "teams struggle to let their robot simply move around the environment, which may include smaller-than-typical
  door openings" and because omni drive makes cramped planning easier; Kinova Gen3 7-DoF, 902 mm, 4 kg, Robotiq 2F-140;
  "Thanks to the long reach of the arm, Lucio does not require a torso lift joint to reach the objects from the ground
  to the 1.5 meter high"; downsides of the Kinova: "a high price tag and limited joint range"; 24 V Li-ion; 30 kg;
  Intel NUC + NVIDIA Jetson; an Isaac Sim model of the robot — [Tidyboy-OPL TDP](https://athome.robocup.org/wp-content/uploads/OPL-Tidyboy-OPL-TDP.pdf)
  (the PDF carries no year; it describes the 2023-2024 robot).
- Hospital practice: Diligent's Moxi, the best-known hospital mobile manipulator, integrates a Kinova (Gen2) arm on a
  mobile base [2nd] — [IEEE Spectrum](https://spectrum.ieee.org/mobile-dexterous-robots-a-key-element-against-covid19)

### Inferences
- A compact round or omni base and one long-reach arm is the proven recipe for doors and cramped rooms; a lift column
  substitutes for arm length when the arm is short (our Lite 6).
- No winner relied on a fleet of small robots for manipulation tasks.

## 3. The winners' software

### Takeaway
Winning stacks combine closed-set detectors trained on site with open-vocabulary models, use an LLM to turn free speech
into a handful of robot functions, run behaviour as hierarchical state machines, verify grasps from gripper encoders,
and fall back to a touchscreen when speech fails. Tidyboy adds imitation learning for arm motions and its own LLM
fine-tuned on generated command data.

### Cited Findings [PRIMARY]
- NimbRo 2024: open-vocabulary mmGrounding-DINO + NanoSAM; closed-set YOLO and MaskDINO fine-tuned per task on a
  63,516-instance, 104-class dataset in 10-30 min on an RTX 4090; Faster Whisper, NVIDIA Maxine denoising, Coqui TTS;
  GPT-4o over 5G with function calling over ~3-15 robot functions, which also rejects infeasible commands; YOLOv8 pose,
  face embeddings and re-identification for the operator; grasp failures detected from gripper encoders, after repeated
  failures a human handover; "touchscreen fallbacks and visual display of perception results helped users and debugging"
  — [arXiv 2412.14989](https://arxiv.org/html/2412.14989v1)
- NimbRo 2025-2026: a two-stage on-site labelling loop (pretrained proposals, refinement with CLIP/ViT/Theia ensembles,
  Segment Anything + CVAT) so detectors can be reconfigured during the competition; person tracking fusing RGB-D
  detections with 2D-lidar leg tracks; SLAM Toolbox + AMCL; MPC for omni motion; VLMs judging scene state (e.g. door
  state of cupboards); door opening to the inside and outside with two arms — GRC 2026 abstract.
- Tidyboy: one software framework shared between HSR and Lucio except the arm controller; motion-primitive arm
  planning; imitation-learned bimanual arm motions; an LLM trained on a generator built from the GPSR/EGPSR command
  grammar, "training within a one-hour timeframe on a single Nvidia RTX 3090" — Tidyboy-OPL TDP.

### Inferences
- Our pieces map onto this recipe: kit detector (closed set, trained on synthetic Isaac Sim data), DINOv2 identity
  for taught objects, a local 4B LLM producing one function as JSON, ACT policies on the Orin, the console's live view
  (the "visual display" fallback).

## 4. What made the difference in practice

### Takeaway
Robustness and operations decide rankings: NimbRo lost points to a person-tracking loss, time-outs and one arm hardware
defect, and still won through stage-2 and final consistency; both top teams carry spares; NimbRo's wins came with
watchdogs on every sensor and network fallbacks.

### Cited Findings [PRIMARY]
- NimbRo 2024 failures: operator lost while avoiding a hard-to-see chair; time ran out in Receptionist and GPSR; a
  hardware defect blocked the arm in Storing Groceries; a quiet operator could not be heard over band noise. Robustness
  came from "hybrid Wi-Fi/5G connectivity, sensor watchdogs, and promptable perception for fast adaptation" —
  [arXiv 2412.14989](https://arxiv.org/html/2412.14989v1)
- NimbRo 2025 reached the final "despite much delayed arrival of our robots" — logistics are a real risk —
  [NimbRo@Home page](https://www.ais.uni-bonn.de/nimbro/@Home/)

### Inferences
- Bring a spare of anything that ends a run (computer, gripper, battery); ship early or drive (Košice-Nuremberg is a
  day's drive).
- Plan for noise: a directional or close microphone and a screen fallback for the patient dialogue.

## 5. What does not transfer to the Healthcare Challenge

### Takeaway
Both @Home winners leaned on external computing (NimbRo: GPT-4o over 5G; Tidyboy: a backpack laptop with an RTX 2080
and Google Cloud speech). The Healthcare Challenge forbids any external network or communication channel during a run,
so everything — speech, LLM, perception, policies — must run on the robot. This favours teams that already deploy
models on Jetsons, which we do.

### Cited Findings
- NimbRo: cloud LLM calls "via the 5G router" — arXiv 2412.14989; Tidyboy: "Backpack Laptop: Intel i7 CPU, 32GB RAM,
  NVIDIA 2080 GPU"; "Speech recognition: Google Cloud API" — Tidyboy-OPL TDP [PRIMARY].
- HCC: "no remote operation, external network connection, or other technical communication channel with the robot is
  permitted while a run is in progress" (rulebook V1.0, 8.1.1.2).

### Inferences
- The HCC also adds what @Home lacks: a physician panel scoring clinical appropriateness on equal footing with the
  referees, sterile-field rules, and a safety assessment with a written concept. Domestic-league experience is not
  enough; a clinical advisor matters.

## 6. Healthcare benchmarks before this challenge

### Takeaway
The METRICS HEART-MET campaigns (2021-2023) benchmarked assistive robots in healthcare with functional tests and a
score that averaged technical performance and social acceptance; field winners were long-standing @Home teams
(b-it-bots, RoboNotts). The euROBIN coopetition (Nancy, Nov 2024) rewarded teleoperation-plus-autonomy in its service
league. Neither ran hospital logistics or OR tasks at the HCC's scale; the HCC is the first of its kind.

### Cited Findings [2nd]
- HEART-MET physically assistive robot challenge at ICRA 2023: RoboNotts first with 7.917; "The overall score is the
  average of the technical and social acceptance score" —
  [metricsproject.eu](https://metricsproject.eu/healthcare/results-heart-met-2023-physically-assistive-robot-challenge/)
- HEART-MET field campaign 2023 (Florence): b-it-bots led object detection, person detection and object sorting —
  [metricsproject.eu](https://metricsproject.eu/healthcare/heart-met-field-campaign-2023-results/)
- euROBIN coopetition 2024: personal/service robots league won by Alter-Ego (University of Pisa) "for their developments
  of hybrid non-line-of-sight teleoperation and autonomy" —
  [winners announcement](https://hcr.ijs.si/wp-content/uploads/2025/03/Coopetition-Winners-announcement-PR-for-28_11_2024_final-1.pdf)

### Inferences
- Social acceptance was scored separately in HEART-MET; the HCC's medical panel and patient-facing task 1 suggest
  similar weight on how the robot speaks, waits and explains.

## 7. Implications for our application

### Takeaway
To look like a winner on paper: one capable, compact mobile manipulator with a spare plan; on-board foundation models
(local LLM, open-vocabulary perception) with closed-set fallbacks; grasp verification and watchdogs; a screen fallback
for speech; an Isaac Sim twin of the arena with our robot; and evidence of reliability (repeated runs with success
counts) rather than single lucky clips.

### Inferences
- The video should show continuous shots and say when footage is sped up (the @Home rule), show what the robot sees and
  its plan (HCC rule), and show the E-stop.
- Our missing "previous competitions" criterion can be offset by the event of 2026-10-22 (a public demo), publications
  and open-source releases; a joint team with a clinical partner adds the medical side.
