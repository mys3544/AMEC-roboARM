# What comparable teams and papers did for these tasks, 2017-2026 (state of 27 October 2026)

Scope: prior work that maps onto the Healthcare Challenge's four tasks, read for what transfers to our robots: the
RoboCup@Home winners (already covered in `research_notes/Healthcare Challenge 2027 platform/what_wins_similar_leagues.md`,
only summarised here), robotic scrub nurses, instrument tray assembly, compliant cleaning, sterile pouch opening,
hospital guide robots, hospital mobile manipulators and NVIDIA's healthcare workflows. Sources were opened where
marked [DOC]; most hospital-robot items rest on abstracts or search summaries [2nd]. No Healthcare Challenge team has
published anything yet (the challenge is new; searched "Healthcare Challenge RoboCup 2027" within the platform
research and the official pages). Labels: **[DOC]** the paper's or project's own page, opened; **[2nd]** search
summary or secondary page; **[CLAIM]** vendor statement; **[OURS]** our inference.

## 1. RoboCup@Home winners: one capable robot, foundation models with fallbacks

### Takeaway
The closest league's winners (NimbRo, Tidyboy) combine closed-set detectors with open-vocabulary models, an LLM over a
handful of robot functions, hierarchical state machines, grasp verification and screen fallbacks, and lose points to
operations rather than capability. Everything they ran off-board must run on board here.

### Evidence
- NimbRo: open-vocabulary detection plus per-task closed-set detectors trained on site; Faster Whisper; GPT-4o over 5G
  with function calling; YOLOv8 pose, face embeddings and re-identification; gripper-encoder grasp checks; touchscreen
  fallbacks; 2025-2026 person tracking fusing RGB-D with lidar leg tracks and door opening with two arms [DOC,
  [arXiv 2412.14989](https://arxiv.org/html/2412.14989v1), [GRC 2026 abstract](https://www.ais.uni-bonn.de/papers/GRC_2026_Memmesheimer_NimbRo_Home.pdf)].
- Tidyboy: a 501 mm omni base and a Kinova Gen3; an LLM trained on the command grammar's generator "within a one-hour
  timeframe on a single Nvidia RTX 3090"; Google Cloud speech and a backpack laptop [DOC,
  [Tidyboy-OPL TDP](https://athome.robocup.org/wp-content/uploads/OPL-Tidyboy-OPL-TDP.pdf)].

### Inferences [OURS]
- Our grammar-first parser with a small LLM behind it is the on-board version of both teams' language stacks.

## 2. Robotic scrub nurses: voice, instrument detection, handover

### Takeaway
Two recent systems do most of task 4's sub-skills on a fixed arm: one with oriented-box instrument detection, speech
and an LLM calling robot tools (94 % success, under 1 s latency), one with SAM 2 and Llama 2. Neither moves, opens
packaging or works with shelves; the handover release is not described in the abstracts.

### Evidence
- Ng, Ma, Heng, Chiu, Li, "Large Language Model-Embedded Intelligent Robotic Scrub Nurse with Multimodal Input",
  Advanced Intelligent Systems 2025: grasping, transfer, sorting and counting; oriented-box detection at 97.6 % AP;
  speech commands; the LLM "uses tool use to trigger robot actions"; 94 % standalone success; latency under 1 s [DOC,
  Crossref abstract, [doi 10.1002/aisy.202500483](https://doi.org/10.1002/aisy.202500483)].
- RoboNurse-VLA (2024): "integrating the Segment Anything Model 2 (SAM 2)" and "the Llama 2 language model" for
  grasping and handing over instruments on voice command [DOC, [arXiv 2409.19590](https://arxiv.org/abs/2409.19590)].
- Older: Gestonurse mapped gestures and speech to instruments on a tray; a 2021 voice-controlled cobot nurse [2nd,
  search summary].

### Inferences [OURS]
- Both are fixed arms at a table; the competition's version adds the base, storage and sterile packaging. Our edge is
  the mobile part and the on-board compute, not the detector.

## 3. Instrument trays: detection datasets and an autonomous tray packer

### Takeaway
An autonomous tray packer (2026) used YOLO12 plus a fine classifier on a 31-instrument dataset and an electromagnetic
gripper. Public instrument datasets are small. The pattern to copy: detector for the type, a second model for the fine
identity, a magnet for flat steel.

### Evidence
- "fully automated robotic system that sorts and structurally packs surgical instruments into sterile trays"; "31
  surgical instruments and 6,975 annotated images"; Staubli TX2-60L; YOLO12 + ResNet classifier; dual electromagnetic
  gripper; public release not stated [DOC, [arXiv 2602.01679](https://arxiv.org/abs/2602.01679)].
- Lavado et al. 2018: 2,620 images, 4 instrument classes on trays, ODbL [DOC,
  [Dataset Ninja](https://datasetninja.com/labeled-surgical-tools-and-images)]. PWISeg and SurgCount-HD [2nd].

## 4. Cleaning by contact

### Takeaway
Mechanical cleaning research uses compliance, either in the arm (soft or low-stiffness robots) or in the tool; no
paper found does spring-pad wiping with a position-controlled cobot, so we should measure our own removal rate.

### Evidence
- SCCRUB (2025): a soft tendon-driven arm scrubbing under learned open-loop force control removed ~99.7 % of
  contamination vs 47.1 % for wetting and rinsing [2nd, [arXiv 2507.06053](https://arxiv.org/abs/2507.06053)].
- A related scrubber uses low-stiffness joints and hybrid force-position control, modelling contact as
  mass-spring-damper [2nd, NSF PAR].

### Inferences [OURS]
- A removal test with a visible tracer (e.g. a marker smear or fluorescent gel under UV light for the photo only) gives
  us a number for the TDP and the physicians.

## 5. Opening sterile pouches

### Takeaway
The only robotic pouch opener found (ICRA 2026) needs a purpose-built gripper with a dented roller and two copies of
it, one per flap. A one-armed robot needs a second "hand": a clip on its own body is the cheapest.

### Evidence
- "A gripper for flap separation and opening of sealed bags" (Foix, Oriol, Torras, Borras; ICRA 2026): "the most
  sensitive variable affecting performance being the normal force applied"; "When two copies of the gripper grasp both
  flaps, the system withstands the forces needed to open the seals robustly"; nurses open up to 240 bags per shift
  [DOC, [arXiv 2603.10890](https://arxiv.org/abs/2603.10890)]. CSIC licenses the design [2nd].

## 6. Guide robots in hospitals

### Takeaway
Guide robots adapt their speed to the person (0.2-1.1 m/s measured in one care-home study) and detect mobility aids;
recent work is sparse. The pace adaptation is the part physicians will judge.

### Evidence
- Proactive guide robot adapting online to users walking at about 0.2, 0.7 and 1.1 m/s over 25-60 m, tested with an
  elderly resident in a Danish care centre (paper template dated 2020) [2nd,
  [manoonpong.com](https://www.manoonpong.com/ProactiveControl/main_body_final.pdf)].
- Freiburg (2017): a hospital robot detecting people and their mobility aids and adapting its velocity to the category
  [2nd, [arXiv 1708.00674](https://arxiv.org/pdf/1708.00674)].
- Patient-robot co-navigation with deep RL using gait velocity from leg detection (2023) [2nd,
  [Applied Sciences](https://doaj.org/article/b62ed78e096f429eaa51e31d7c1c12f8)].
- Gap: no 2024-2026 study on follower distance for elderly hospital patients was found (one search).

## 7. Hospital mobile manipulators

### Takeaway
Real deployments are delivery-focused (Moxi); research nursing manipulators (ARNA, IONA) were tested in simulated
hospitals. None does OR preparation or sterile handover; the competition's tasks are ahead of deployed practice.

### Evidence
- Moxi: deployed in "over 25 hospital facilities" [CLAIM, [Diligent Robotics](https://www.diligentrobots.com/blog/welcoming-diligent-robotics-bringing-our-autonomy-platform-into-hospitals)];
  a Kinova arm on a mobile base [2nd, platform notes].
- ARNA: omnidirectional base + 7-DoF arm for nursing tasks, tested in a simulated hospital with nursing students [2nd,
  [NSF PAR](https://par.nsf.gov/servlets/purl/10284608)]. IONA (IROS 2023): shared-autonomy nursing robot, 9-person
  study in a virtual hospital testbed [2nd, [WPI](https://users.wpi.edu/~zli11/papers/C2023_IROS_Boguslavskii_IONA.pdf)].

## 8. NVIDIA's healthcare workflows

### Takeaway
NVIDIA's hospital-automation examples (push a cart, tray to cart, scissors into a tray) show that the challenge's
sponsor thinks of these tasks as GR00T-policy tasks on humanoids and small arms. A team that shows the same tasks with
a classical, verifiable pipeline and a VLA as a research layer speaks both languages.

### Evidence
- i4h-workflows hospital automation: `locomanip_push_cart`, `locomanip_tray_pick_and_place` (Unitree G1, GR00T N1.6),
  `scissor_pick_and_place` (SO-ARM 101, GR00T N1.5/N1.7 or rule-based), `assemble_trocar` [DOC,
  [i4h-workflows](https://github.com/isaac-for-healthcare/i4h-workflows)].
- The challenge's NVIDIA "Physical AI Support Package" goes to every admitted team; contents unpublished [RULE, FAQ].

## 9. Gaps

- No published approach to the Healthcare Challenge itself (new competition).
- No 2024-2026 study of guide distance for elderly patients; no paper on sensorless (motor-current) handover release;
  no spring-pad wiping paper with a cobot (one search each).
- The scrub-nurse papers were read as abstracts only.
