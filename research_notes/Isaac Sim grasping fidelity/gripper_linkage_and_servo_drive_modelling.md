# Modelling the M3's four-bar gripper and its hobby-servo squeeze in PhysX (state of October 2026)

Scope: how to represent Yahboom's parallel gripper (one servo, two cranks, two parallel idler links, fingertip pads)
as an Isaac Sim articulation, and how to set its drive so that the squeeze resembles a position-controlled bus servo
closing a few degrees past the piece. The starting point is `m3pro/sim/isaac_m3.py` as of 2026-10-27:
- `FINGER_DRIVE` (4 N·m/deg, 0.2, 1.5 N·m) on `rlink1_Joint`;
- the trial `FINGER_SQUEEZE` (12, 0.6, 4);
- the other five linkage joints driven every step by `LINK_DRIVE` (200 N·m/deg, 2, 50 N·m) to their share of the
  finger's actual angle, because PhysX mimic joints "did not hold" after import.

Contacts and friction are in `physx_contacts_for_small_parallel_grasps.md`. Labels as there: [REAL-LC], [REAL-other],
[SIM], [CLAIM], [2nd]; "ours" = computed for this note from the repo's URDF and code.

## 1. Why the imported mimic joints failed: the importer looks the reference joint up before it exists

### Takeaway
The 5.0-era URDF importer sets a mimic joint's reference by searching the stage for a joint of that name, at the
moment it creates the mimic joint. If the reference joint has not been created yet, the relationship is silently
left empty, and so are the mimic joint's limits. The left side's joints (`llink*`) reference the right crank
`rlink1_Joint`. If the importer reaches them first, they end up without a reference. That matches what we saw on
2026-10-23. When setting mimic joints by hand, the gearing must be −multiplier (PhysX's equation is
q_A + G·q_B + γ = 0), and the follower joints must carry no drive of their own.

### Evidence
- `configureMimicAPI` in the importer [CLAIM: source read] —
  [UrdfImporter.cpp](https://github.com/isaac-sim/urdf-importer-extension/blob/main/source/extensions/isaacsim.asset.importer.urdf/plugins/import/UrdfImporter.cpp):
  - applies `PhysxMimicJointAPI` with gearing `-joint.mimic.multiplier` and offset `joint.mimic.offset`;
  - authors `naturalFrequency` / `dampingRatio`;
  - then calls `FindPrimByNameAndType(stage, joint.mimic.joint, UsdPhysicsJoint)`;
  - only `if (source_prim)` does it add the reference target and the limits (the source's range ±20 %);
  - there is no retry or error. That repository is marked deprecated and holds the 4.5/5.0-era code; Isaac Sim 5.0
    ships `isaacsim.asset.importer.urdf`, presumably the same logic (not checked byte for byte).
- PhysX mimic joint [CLAIM] —
  [Articulations, Mimic Joints](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html):
  - "q_A + G*q_B + γ = 0";
  - "Each dof may be in any sub-tree of the articulation";
  - "each dof of a mimic joint must be capable of motion";
  - compliance via `naturalFrequency` / `dampingRatio` (default −1 = rigid).
- NVIDIA's 2F-85 rigging tutorial [CLAIM] —
  [Rig Closed-Loop Structures (5.1)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/rig_closed_loop_structures.html):
  - the mimic follower gets gearing −1 and the reference `finger_joint`;
  - "All drive features are copied over from the reference joint and having an authored joint drive would
    negatively impact the drive outcome".
  - The Isaac Sim known issues list "Some grippers with parallel mechanism (that is, Robotiq 2F-85 and 2F-C2) have
    links that do not move with rest of the gripper" —
    [5.1 known issues](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/known_issues.html).
- Yahboom's URDF mimic tags (ours, `M3Pro.urdf`):
  - multipliers: `rlink2` +1, `llink1` +1, `llink2` −1, `rlink3` −1, `llink3` +1, all on `rlink1_Joint`;
  - the mimic joints are `continuous` (no limits);
  - `rlink3`/`llink3` are dangling idlers: the loop to the pads is not closed in the URDF.

### Inference
To try the real mimic joints again on 5.0, author after import, per follower:
- `PhysxMimicJointAPI:rotZ` (the axis token does not matter for a revolute joint, per the tutorial);
- `physxMimicJoint:rotZ:referenceJoint` -> `rlink1_Joint`;
- `gearing = −multiplier`, `offset = 0`;
- finite limits;
- **remove** the `DriveAPI` the importer gave it.

Verify on a bench that the pads stay parallel within 1 deg under squeeze. If it jitters, add compliance (a natural
frequency near the simulation's, damping ratio 1) or armature (section 3) before going back to drives.

## 2. Closing the four-bar: optional, and what NVIDIA does

### Takeaway
A tree with mimic joints is the standard simplification. NVIDIA's own 2F-85 model "does not use loop-closing joints"
in its tuning example. A closed loop is possible by excluding one joint from the articulation, or with the 5.x URDF
importer's `<loop_joint>` tag, but loops need smaller timesteps. Our pads already translate correctly through the
±1 mimic shares. A loop adds nothing that matters for a 30 mm grasp.

### Evidence
- "Six revolute joints ... It is a simplified version that does not use loop-closing joints ... The other five
  joints are all mimic joints of joint J0" [CLAIM] —
  [Gripper tuning example: Robotiq 2F-85 (107.3)](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/gripper_tuning_example.html)
- To break a loop: "choose one joint to exclude from the Articulation ... a maximal coordinate joint ... it is the
  joint that accumulates the most error"; pick one "with no limits, no resistance, and no drive" [CLAIM] —
  [Rig Closed-Loop Structures](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/rig_closed_loop_structures.html)
- The 5.x URDF importer accepts `<loop_joint name type="spherical">` with two link frames [CLAIM] —
  [URDF importer 5.1](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html)
- "closed articulation loops often requires higher simulation frequency than the default 60Hz" [CLAIM] —
  [Articulation Stability Guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html)
- Our pads' convex-hull faces keep a constant 1.4/2.0 deg opening at every finger angle. So the ±1 shares hold the
  pads at a fixed orientation, as a parallelogram does (ours).

## 3. The drives are 30-1000 times too stiff for 120 Hz; NVIDIA's gripper recipe fixes this

### Takeaway
NVIDIA's rule is to keep (drive natural frequency) x (timestep) near 1, "particularly if the driven joint
participates in a mimic joint or ... a gripping action". Our finger drive sits at 30-137, the linkage drives at
213-1026, and the linkage's 50 N·m force limit can spin a 5e-6 kg·m^2 link to thousands of rad/s in one step.
Because PhysX drives are implicit, more stiffness can even give *less* force. NVIDIA's recipe for a gripper:
- max force = the actuator's real torque;
- stiffness = max force / the target overshoot at which full force is wanted;
- damping for a damping ratio of about 1;
- armature = rotor inertia x gear ratio^2 to bring the natural frequency down.

For a hobby servo, the armature is far larger than the URDF's link inertias.

### Evidence
- NVIDIA's stability rules [CLAIM] —
  [Articulation Stability Guide (107.3)](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html):
  - "naturalFrequency = sqrt(stiffness / inertia)";
  - "Avoid simulation timestep and drive natural frequency combinations whose product is much greater than 1. This
    is particularly true if the driven joint participates in a mimic joint or participates in a gripping action";
  - "Avoid maximum drive forces that would generate very large accelerations if applied to the smallest body mass
    or inertia";
  - "armature should be set to JG2" (rotor inertia x gear ratio^2);
  - set targets near the current state or rate-limit them, since far targets create large forces.
- The 2F-85 tuning example [CLAIM] —
  [Gripper tuning example](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/gripper_tuning_example.html):
  - max drive force from the spec (5 kg at μ 0.3 -> 83 N a finger -> 3.3 N·m at a 0.04 m lever, rounded to 4 N·m,
    times six joints driven through mimics = 24 N·m on J0);
  - stiffness so the max force is reached at a 5 deg (0.087 rad) target delta: 275 N·m/rad instead of the asset's
    5700, natural frequency 4.0e4 -> 9.0e3 rad/s;
  - with 5e-3 kg·m^2 armature, 234 rad/s, which "could work with a simulation frequency of 60";
  - damping from damping ratio 1: d = 2·sqrt(s·i) ≈ 0.06;
  - "One unit of stiffness/damping in USD is equivalent to 180/π units ... in PhysX";
  - implicit spring: F = s·dx / (dt²·nf² + 1), so "increasing stiffness beyond a certain point will not result in
    the drive providing more force; instead, the drive will provide less force".
- PhysX: "the drives are implicit"; very large gains are stable alone but "not necessarily ... in conjunction with
  other hard constraints such as mimic joints, joint position limits or joint velocity limits" [CLAIM] —
  [Articulations](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html)
- **Our numbers** (ours: URDF inertias about each joint axis at a 30 mm opening; USD stiffness x 180/π):

  | drive | stiffness (PhysX units) | inertia about the axis | nat. frequency | nf·dt at 120 Hz | per 32-iteration substep |
  |---|---|---|---|---|---|
  | finger `rlink1` (rlink1 + rlink2 subtree) | 229 N·m/rad | 1.74e-5 kg·m^2 | 3,626 rad/s | 30 | 0.94 |
  | finger, rlink1 alone | 229 | 8.5e-7 | 16,387 | 137 | 4.3 |
  | FINGER_SQUEEZE, subtree | 688 | 1.74e-5 | 6,280 | 52 | 1.6 |
  | LINK_DRIVE on `rlink2` (pad) | 11,459 | 4.9e-6 | 48,171 | 401 | 12.5 |
  | LINK_DRIVE on `llink1` + `llink2` | 11,459 | 1.75e-5 | 25,583 | 213 | 6.7 |
  | LINK_DRIVE on `rlink3` (idler) | 11,459 | 7.6e-7 | 123,109 | 1026 | 32 |
  | finger subtree + armature 1e-3 | 229 | 1.0e-3 | 475 | 4.0 | 0.12 |

  LINK_DRIVE's 50 N·m on the pad link's 4.9e-6 kg·m^2 is 1e7 rad/s^2, about 85,000 rad/s gained in one 1/120 s
  step. The guide warns against exactly this ("rotations approaching 360 deg/second in a single simulation step").
- **Servo references from working SO-101 sims** (STS3215 bus servo) [CLAIM: authors' configs] —
  [soarm101-isaac docs](https://github.com/Jbruslind/soarm101-isaac/blob/main/docs/ISAAC_SIM_ENVIRONMENTS.md),
  [so101.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/so101.py):
  - Isaac Lab's SysID SO-101 USD gripper: k = 1.19 N·m/deg (68 per rad), d = 0.049 per deg, max force 3.35 N·m,
    with SysID joint friction and armature in the USD;
  - Isaac Lab's high-PD variant: k 17.8 N·m/rad, d 0.6, effort 10;
  - soarm101-isaac's IK task held a 4 cm cube "with only 1 N m" [SIM];
  - soarm101-isaac commands the jaw 0.14 rad past the measured contact angle, with an action clamp that keeps the
    target within 0.05 rad of the jaw, "so the squeeze stays at about 4 N m or less" [SIM].
- Isaac Lab's `SO101_CFG` now resolves "actuator gains, friction, armature, and limits" from the SysID USD [CLAIM] —
  [isaaclab_assets CHANGELOG 0.6.6](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/docs/CHANGELOG.rst).
  The armature values themselves sit in a binary USD that this note did not open.

### Inference: a servo-like finger for the M3
- **Max force:** the J6 servo's stall torque at the crank. Its model is not in the repo. Keep 1.5 N·m until
  measured, and measure the real squeeze directly (report, protocol B). Raising it to 4 N·m (FINGER_SQUEEZE) only
  helps if the drive actually saturates, and adds penetration if the pads are already pressing in. Log the finger's
  applied or measured joint effort to know.
- **Stiffness:** full torque at the planner's overshoot (2.4-2.6 deg ≈ 0.044 rad).
  - s ≈ 1.5 / 0.044 ≈ 34 N·m/rad = **0.6 N·m/deg** in USD.
  - Same requested torque as today, but 7x softer, which matches how a bus servo with a P loop behaves near its
    target.
- **Armature** on `rlink1`/`llink1`: about 1e-3 kg·m^2 (`physxJoint:armature`). A micro-servo rotor of 1e-8 to
  1e-7 kg·m^2 at a gear ratio of 200-300 gives 4e-4 to 9e-3; this is an assumption, not a datasheet value. Then
  nf ≈ 184 rad/s and nf·dt ≈ 1.5 at 120 Hz.
- **Damping:** d = 2·sqrt(34 x 1e-3) ≈ 0.37 N·m·s/rad ≈ **0.0065 N·m·s/deg**.
- Check that the gripper still closes at the real speed (J6 capped at 150 deg/s in the picker).
- **Linkage:** proper mimic joints (section 1). If drives stay: 20 N·m/deg, max force about 5 N·m, armature
  1e-4 kg·m^2 on each follower, and test that the pads do not tilt under squeeze.

## 4. Hobby-servo specifics: what to model and what not

### Takeaway
Model three things: the torque cap (stall torque), the speed cap, and the soft P loop near the target. Backlash and
delay are already in the robot card. PhysX 5.6 adds a drive performance envelope (speed-torque trade-off) and joint
friction. Isaac Sim 5.0 exposes armature and joint friction for sure; whether the envelope reaches USD in 5.0 was
not checked. None of these is needed for the grasp itself. A first-order lag on the arm's targets is, because a
stiff drive follows the board's straight-line sweeps with step changes in velocity, and the carry is where 13 of the
27 lifted-then-lost picks went.

### Evidence
- The PhysX 5.6.1 performance envelope: |effort| <= maxEffort − velocityDependentResistance·|v| and
  |v| <= maxActuatorVelocity − speedEffortGradient·|effort|. Joint friction is Coulomb (static and dynamic) plus
  viscous [CLAIM] — [Articulations](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html)
- "Large forces can also be produced by setting position and velocity drive targets far from the current joint state.
  Consider ... a rate-limiter on the drive targets" [CLAIM] —
  [Articulation Stability Guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html)
- SO-101 fastener pick in Isaac Lab: at 1 cm/step end-effector speed, 55/64 placed and "7 of 9 failures: the bolt
  slipped out of the jaws during carry". At 0.6 cm/step, 62/64 and 127/128 [SIM] —
  [soarm101-isaac docs](https://github.com/Jbruslind/soarm101-isaac/blob/main/docs/ISAAC_SIM_ENVIRONMENTS.md)
- Earlier cheap-arm twins: identified servo stiffness 16-62 vs 998 stock, lag up to 40 deg on fast moves;
  replay success 9/16 stock vs 12/16 identified [SIM, from the prior note] —
  `research_notes/Simulácia a VLA pre presný úchop/open_sim2real_pipelines.md`
- Ours:
  - `ARM_DRIVE` = 40 N·m/deg (2,292 per rad), max 3 N·m, `ARM_MAX_DPS` 150 deg/s;
  - the servo probe of 2026-10-23: lag 0.10-0.20 s, 15 deg moves turning 0.87-0.93 of the command (NEXT.md)
    [REAL-LC];
  - the sim plays the board's sweeps as straight lines, so the velocity steps at every key.

### Inference
Low-pass the arm targets in `isaac_m3.py` with a first-order filter of about 0.05-0.1 s. That is servo-like and
would also model part of the card's "gain". Alternatively cap acceleration. Compare the lifted-then-lost counts on
the same seeds.

## 5. Isaac Sim 6.x importer: mimic is now written as Newton's schema

### Takeaway
When the M3 is re-imported in 6.x, mimic tags become `NewtonMimicAPI` (`newton:mimicJoint`, `mimicCoef1` = scale,
`mimicCoef0` = offset), and "no equivalent PhysxMimicJointAPI is authored". Whether the PhysX backend honours it for
our gripper must be checked on the imported asset. The docs say "the runtime consumes NewtonMimicAPI directly". Our
own drive-the-linkage code would keep working either way.

### Evidence
- [URDF importer 6.0](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html)
  [CLAIM]:
  - mimic -> `NewtonMimicAPI`;
  - articulation root gets `NewtonArticulationRootAPI`, "the PhysX-specific PhysxArticulationAPI is no longer
    authored";
  - armature maps to `PhysxJointAPI.armature`.
- Isaac Lab 3.x's new Franka asset uses "a driven leader finger with a passive mimic follower" (`panda_finger2_passive`,
  stiffness 0) [CLAIM] —
  [isaaclab_assets CHANGELOG](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/docs/CHANGELOG.rst),
  [franka.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/franka.py)
- Isaac Sim 5.1/6.0 known issue: "failed to find internal joint object for PhysxMimicJointAPI" errors for robots
  with mimic joints "can be ignored" [CLAIM] —
  [6.0 known issues](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/known_issues.html)

## Gaps
- The J6 servo's model, stall torque and gear ratio are not in the repo. All torque numbers above are placeholders
  until the real squeeze is measured.
- Whether the drive performance envelope and mimic compliance attributes are exposed in Isaac Sim 5.0's USD schema
  was not checked on the install. The 107.3 docs describe mimic compliance; the envelope is documented for PhysX
  5.6.1.
- The overnight FINGER_SQUEEZE batch (2026-10-27/28) was not available to this note.
