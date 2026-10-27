# PhysX contacts for small parallel grasps: why the M3 twin drops far-rim pieces (state of October 2026)

Scope: the contact side of the M3 Pro twin in Isaac Sim 5.0 on Dionysos (`m3pro/sim/isaac_m3.py`, played by
`m3pro/tools/m3_sim.py` / `m3_expert.py`). The question is which geometry and PhysX settings decide whether a 30 mm
cube, a 40 mm cube or a 30x30x60 mm block stays between the two fingers. Isaac Sim versions, drivers and
installation are in `research_notes/Simulácia a VLA pre presný úchop/isaac_sim_versions_side_by_side.md` and are
not repeated. The gripper's linkage and drive tuning are in `gripper_linkage_and_servo_drive_modelling.md`. What
other projects use is in `working_setups_and_what_changes_in_6x.md`. This note starts from our own data:
189 simulated picks (out/sim, read-only, re-analysed 2026-10-27) and the URDF's finger meshes
(out/sim/yahboom_M3Pro_description).

Labels: **[REAL-LC]** measured on a real low-cost arm (ours or SO-100/101 class); **[REAL-other]** real robot, not
low-cost; **[SIM]** simulation measurement; **[CLAIM]** vendor or author statement, documented default or advice
without a measurement in the source; **[2nd]** secondary source, not checked against the primary. "Ours" marks an
analysis made for this note from files in the repo.

## 1. Our 189 picks: the pads' overlap with the piece decides the outcome

### Takeaway
Success follows how far the URDF finger pads reach down the piece's side. With less than 10 mm of vertical overlap,
0 of 8 picks reached the box. With 10-13 mm, 8 of 23 did. With 13 mm or more, 129 of 158 did (82 %). At the far rim
(pitch 40-60 deg) the median overlap is only 12.9 mm. The URDF's pads end 12.9 mm past the Gripping point at a
30 mm opening, and 9.4 mm at 40 mm. The planner's model assumes 15 mm, and the touch test's fit puts the real shut
tips about 7 mm beyond the URDF's. Friction makes no difference because the force is not what is missing; the
contact is.

### Evidence
- **Outcome by overlap** (ours: `episoderesult_*.json` + `out/sim_episodes/*/episode.json`, the same 189 picks as
  `m3_sim_report.py misses`). Overlap = half the piece's height − (grip height over the middle − pad reach x
  sin(pitch)). Pad reach was taken from the URDF meshes: 12.9 mm for 30 mm pieces, 9.4 mm for the 40 mm cube. [SIM]

  | vertical overlap of the pads | picks | in the box | lost after lifting | never lifted |
  |---|---|---|---|---|
  | < 10 mm | 8 | 0 (0 %) | 1 | 7 |
  | 10-13 mm | 23 | 8 (35 %) | 6 | 9 |
  | >= 13 mm | 158 | 129 (82 %) | 20 | 9 |

  Median overlap: 18.5 mm when the pick went in, 15.1 mm when the piece was lifted and lost, 11.5 mm when it never
  rose. [SIM]
- **By pitch** (ours) [SIM]:

  | pitch | picks | in the box | median grip over middle | median overlap | picks under 13 mm overlap | ... with +7 mm reach |
  |---|---|---|---|---|---|---|
  | 40-60 | 42 | 55 % | 13.1 mm | 12.9 mm | 22 | 1 |
  | 61-79 | 60 | 70 % | 11.5 mm | 15.7 mm | 7 | 1 |
  | 80-90 | 87 | 83 % | 9.2 mm | 18.7 mm | 2 | 1 |

- **When the piece goes** (ours; a piece counts as gone when it moves 10 mm against the grip point after the
  fingers shut) [SIM]:
  - All 25 "never lifted" left within 0.25-0.66 s of the shut (median 0.36 s), in the lift step.
  - Of the 27 lifted-then-lost, 14 went in the lift (0.5-1.7 s) and 13 in the carry to the box (1.2-2.3 s).
  - The carry losses sit mostly at steep pitch (75-90 deg) with 12 mm grips, so the carry is a second, separate
    problem (section 5 and the report).
- **The URDF's fingers** (ours: forward kinematics of `M3Pro.urdf` with its own mimic multipliers, meshes
  `rlink2.STL` / `llink2.STL`, in Yahboom's `Gripping` frame).
  - The pad's inner face runs from about 7 mm behind the Gripping point to 12.9 mm past it at a 30 mm opening. It
    reaches 9.4 mm past at 40 mm and 16.8 mm when shut to 4.4 mm.
  - The fingers are 6 mm thick, so the contact is a strip at most 6 mm wide.
  - The convex hulls' inward faces open 1.4 deg (right) and 2.0 deg (left) toward the tips, the same at every
    opening. So the first contact is the pad's root end or the piece's top edge.
- **The models' fingertips:**
  - `m3kinematics.FINGERS` = 15 mm past the Gripping point.
  - The fingertip touches of 2026-10-26 fit "the shut tips 9 mm past FINGERS", about 24 mm, where the URDF's shut
    tips reach 16.8 mm. That is about 7 mm longer. It is one fitted parameter that trades against J2 (NEXT.md), not
    a caliper measurement. [REAL-LC]
  - The real M3 held the same high grips: 8 of 8 on 2026-10-26, 24 cm out too. [REAL-LC]
- **Friction does not matter** (NEXT.md; friction 0.5-0.8 gives 67/78/66 %) [SIM], as expected if the grip is
  limited by where and how the contact forms, not by the friction coefficient.

### Inference
Before any solver setting, make the sim's pads as long and as wide as the real rubber pads. Measure them first:
calipers on the shut gripper, from a reference such as the crank pin, against `rlink2.STL`. If the 7 mm holds,
then:
- 21 of the 22 far-rim grips now under 13 mm overlap move into the band where 82 % succeed.
- The 25 "never lifted" would mostly disappear.
- This is the largest single lever on the far-rim rate.

The planner needs no change: it already aims for the real fingers.

## 2. Friction patches, anchors and torsion: the knobs PhysX actually has

### Takeaway
PhysX 5 has one friction model: patch friction. Each contact patch gets at most two friction anchors. Contacts
closer than the scene's `frictionCorrelationDistance` (default 25 mm) are merged into one anchor point. A single
anchor cannot resist the piece turning about the contact normal, which is the grasp axis. Our pads are a 6 mm strip
on a 30 mm face, so at the default the patch very likely collapses to one anchor. Isaac Lab's lift and stack tasks
set 6.25 mm. Torsional friction exists only as a per-shape `torsionalPatchRadius` / `minTorsionalPatchRadius`. It
works only with TGS and only when a patch has a single anchor. Use both: the smaller correlation distance first,
then a small minimum torsional radius.

### Evidence
- Patch friction, from the PhysX 5.6.1 docs [CLAIM]:
  - "PhysX computes friction per contact patch. Up to two contact points lying in the contact patch area are
    selected as friction anchors".
  - Normal impulses of all points "are accumulated and distributed equally among the friction anchors".
  - TGS applies friction in all position and velocity iterations. PGS applies it in the last 3 position iterations.
  - [Rigid Body Dynamics](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/RigidBodyDynamics.html)
- `PxFrictionType` is deprecated: "only the patch friction model is supported now" [CLAIM] —
  [PxFrictionType](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/structPxFrictionType.html)
- Scene friction parameters [CLAIM] —
  [PxSceneDesc](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/classPxSceneDesc.html):
  - `frictionCorrelationDistance` "used to decide whether contacts are close enough to be merged into a single
    friction anchor point"; default 0.025 x length scale.
  - `frictionOffsetThreshold`: contact points separated by more than it get no friction; default 0.04.
  - Isaac Lab's `PhysxCfg` has the same defaults (0.025 m, 0.04 m) —
    [physx_manager_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_physx/isaaclab_physx/physics/physx_manager_cfg.py).
- Torsional friction [CLAIM] —
  [PxShape](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/classPxShape.html):
  - `setTorsionalPatchRadius`: friction "proportional to the penetration depth"; "Will only be active, if the
    friction patch has a single anchor point only"; "Only supported in combination with solver type
    PxSolverType::eTGS".
  - `setMinTorsionalPatchRadius` > 0 applies "some torsional friction ... regardless of the value of
    torsionalPatchRadius or the amount of penetration".
  - Both default to 0.
  - In USD they are `physxCollision:torsionalPatchRadius` / `minTorsionalPatchRadius` (Isaac Lab's
    `CollisionPropertiesCfg` exposes both) —
    [schemas_cfg.py v2.3.0](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/sim/schemas/schemas_cfg.py).
- Isaac Lab's grasp tasks set `friction_correlation_distance = 0.00625` [CLAIM]:
  - in the Franka lift task —
    [lift_env_cfg.py v2.3.0](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py);
  - in the stack tasks' PhysX preset on `main` (Isaac Lab 3.x), together with `bounce_threshold_velocity=0.01` and
    `solve_articulation_contact_last=True` —
    [stack_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/contrib/stack/stack_env_cfg.py).
- Our scene sets none of these. `isaac_m3.py` applies only `solverType TGS` to the scene and leaves correlation
  distance, offset threshold and torsion at their defaults (read 2026-10-27).

### Inference
- On `/physicsScene`:
  - `physxScene:frictionCorrelationDistance = 0.00625`;
  - `physxScene:bounceThreshold = 0.01` (harmless with restitution 0).
- On the pads and pieces, `physxCollision:minTorsionalPatchRadius` about 1-2 mm. That is a fraction of the pad's
  6 mm width; the value is ours, not a published number.
- Check it in a grip bench (the report's protocol): the piece should stop turning about the grasp axis when held
  12-15 mm above its middle.

## 3. Contact and rest offsets, depenetration, CCD and mass

### Takeaway
These are second-order for us:
- Keep the pieces' 2 mm contact offset (the same as Isaac Lab's lift objects) and give the pads an explicit 1-2 mm
  with rest offset 0. The importer authors no offsets.
- Cap the pieces' `maxDepenetrationVelocity` at about 1-5 m/s. Squeezed light bodies are otherwise shot out by
  depenetration impulses.
- Use the pieces' real masses: every piece is 15 g today, the 40 mm cube and the block are heavier.
- CCD is not needed at our speeds.

### Evidence
- Contact offset: shapes closer than the sum of their contact offsets generate contacts; the default is
  0.02 x length scale. Rest offset: shapes "come to rest at a distance equal to the sum of their restOffset values",
  default 0 [CLAIM] — [PxShape](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/classPxShape.html).
  The guide's advice: rest offset 0 when colliders match the visuals, contact offset "slightly larger" [CLAIM] —
  [Advanced Collision Detection](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/AdvancedCollisionDetection.html)
- Isaac Lab's current lift task gives grasp objects `contact_offset=0.002`, 16 position / 0 velocity iterations,
  friction randomised 0.5-1.0, at `sim.dt = 1/120` [CLAIM] —
  [core/lift/lift_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/core/lift/lift_env_cfg.py).
  The stacking cubes use `max_depenetration_velocity=5.0` [CLAIM] —
  [stack_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/contrib/stack/stack_env_cfg.py).
  Isaac Lab's SO-101 asset sets `max_depenetration_velocity=1.0` [CLAIM] —
  [so101.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/so101.py)
- An Isaac Sim 5.0 user's grasped can "pops, shoots upward, or gets ejected" with `max_depenetration_velocity` at
  10000. The maintainer blamed "large depenetration impulses on a light, thin object" and said to tune mass/inertia,
  contact offsets, material and solver together. No numbers were given and the thread is unresolved [2nd: forum
  thread read through a summariser] —
  [IsaacSim discussion #372](https://github.com/isaac-sim/IsaacSim/discussions/372)
- NVIDIA's collision guide lists, in order [CLAIM] —
  [Collision Behavior Guide (107.3)](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/collision_guide.html):
  - increase the contact offset;
  - speculative CCD;
  - "Make objects heavier, within reasonable limits" (very light bodies and big mass ratios hurt convergence);
  - reduce the maximum depenetration velocity;
  - more position iterations;
  - friction (rarely above ~2.0);
  - a smaller timestep only "if all other adjustments don't lead to good results".
- The URDF importer authors no contact or rest offsets. It sets the collision approximation (convex hull or
  decomposition) and drives only [CLAIM: source read] —
  [UrdfImporter.cpp](https://github.com/isaac-sim/urdf-importer-extension/blob/main/source/extensions/isaacsim.asset.importer.urdf/plugins/import/UrdfImporter.cpp).
  That repository is marked deprecated and is the 4.5/5.0-era code.
- The URDF's gripper links weigh 2.7-6.1 g with inertias of 7.6e-7 to 1.7e-5 kg m^2 about their joints (ours, from
  `M3Pro.urdf`). The pieces are 15 g in every sim episode (`m3_sim.py`). A wooden 40 mm cube is closer to 35-45 g at
  wood densities of 0.55-0.7 g/cm^3 (ours, arithmetic; weigh them).

### Inference
"Flew off in the carry" (NEXT.md) fits depenetration: stiff drives push the pads into the piece, and the
unbounded depenetration velocity throws it out. Cap it at 1 m/s on the pieces, as the SO-101 asset does, and watch
penetration in the grip bench.

## 4. Solver: TGS, iterations, timestep, and the articulation's constraint order

### Takeaway
Keep TGS. Velocity iterations can drop to 0-1. The 32 position iterations on the arm already act as substeps.
Raising physics from 120 to 240 Hz is cheap next to rendering, and both NVIDIA and LeRobot's SO-101 scene do it for
small grasps. The most effective solver change, `solveArticulationContactLast`, needs Isaac Sim 5.1 or 6.x; 5.0
does not have it. Under the default order, the articulation's drives are solved after dynamic contact and "win",
so the pads push into the piece.

### Evidence
- From the PhysX 5.6.1 docs [CLAIM] —
  [Rigid Body Dynamics](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/RigidBodyDynamics.html):
  - TGS "is generally the recommended solver"; "The number of substeps is equal to the number of position
    iterations".
  - "it is often sufficient to run TGS without any velocity iterations"; the default is 4 position / 1 velocity.
  - "If you find a need to use a setting higher than 30, you may wish to reconsider the configuration of your
    simulation".
- The articulation constraint order is: link dynamic contact, mimic joint, joint drive, position limit, velocity
  limit, static constraint, static contact. "the constraint that runs last will 'win'". With hard drive, limit and
  contact, "adding more iterations will make no qualitative difference". The cure is less stiffness ("reducing the
  drive stiffness, damping and maximum force or ... the timestep") [CLAIM] —
  [Articulations](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html)
- `physxScene:solveArticulationContactLast = True` reorders the solver: "maximum joint velocity is enforced last"
  and "dynamic contact is resolved immediately before static contact". This "can lead to a reduction in persistent
  contact depth ... stronger contact normal forces and stronger friction forces that aid gripping strength", at "a
  measurable performance penalty" [CLAIM] —
  [Articulation Stability Guide (107.3)](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html)
- New in Isaac Sim **5.1**: "solving articulation collision contacts last to improve on gripper penetration issues"
  (omni.physx 107.3.18 -> 107.3.26) [CLAIM] —
  [5.1.0 release notes](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/release_notes.html).
  Isaac Lab: "only available with Isaac Sim 5.1" [CLAIM] —
  [physx_manager_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_physx/isaaclab_physx/physics/physx_manager_cfg.py)
- Effect on an SO-101 cube pick in Isaac Sim 6.2 / Isaac Lab 3.0 [SIM] —
  [soarm101-isaac docs](https://github.com/Jbruslind/soarm101-isaac/blob/main/docs/ISAAC_SIM_ENVIRONMENTS.md),
  [issue #28](https://github.com/Jbruslind/soarm101-isaac/issues/28):
  - placed 1/10 with the flag off, 20/20 with it on (with wrist yaw alignment and joint-limit-aware IK);
  - Isaac Lab's changelog 8.1.9 (2026-07-14): "the moving jaw tunneled into grasped objects because the
    articulation position drive was resolved after the contacts" —
    [isaaclab_tasks CHANGELOG](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/docs/CHANGELOG.rst).
- Timestep [CLAIM unless marked]:
  - NVIDIA's 2F-85 rigging tutorial: a 2.5 kg payload "slips and fails" until the scene runs "at least 80 steps per
    second"; "Contacts are very sensitive to time-stepping" —
    [Rig Closed-Loop Structures (5.1)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/rig_closed_loop_structures.html).
  - LeRobot's SO-101 scene with 3 cm, 10 g wooden cubes runs CPU PhysX at `PHYSICS_HZ = 240` (8 substeps per
    30 fps step) — [so101-tabletop-isaac env.py](https://huggingface.co/lerobot/so101-tabletop-isaac/blob/main/env.py).
  - Ours: 120 Hz, robot 32 / 4 iterations, pieces 16 / 4.
- Known limitations [CLAIM] —
  [Known Physics Limitations (107.3)](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/current_limitations.html):
  - since 105.1, velocity iterations above 4 are no longer silently turned into position iterations;
  - joint forces read high with nonzero TGS velocity iterations ("Force sensing: Use TGS with zero velocity
    iterations").

### Inference
On 5.0: 240 Hz, robot 32 / 1, pieces 16 / 1. The contact-last flag is the first thing to test once a 5.1+ or 6.x
install is at hand. hekate already has 6.0, and 6.0.1 runs on Dionysos's driver per the 2026-10-26 notes. It is
the documented fix for exactly "pads squeeze the piece out".

## 5. Collision shapes: hulls, decompositions, SDF

### Takeaway
Use primitives where the real part is simple: box colliders for the pieces and for the rubber pads. SDF is for
concave dynamic meshes such as nuts and bolts and is not needed here. The URDF finger's convex hull is acceptable at
30-40 mm openings, because the pad is the innermost feature there. Its length and width, not its convexity, are the
problem (section 1).

### Evidence
- The URDF importer offers "Convex Hull" (one hull a link) or "Convex Decomposition"; our import uses hulls
  (`cfg.convex_decomp = False`) [CLAIM] —
  [URDF importer 5.1](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html)
- NVIDIA's rigging tutorial: convex hull "is a good balance", but "for grippers, you often want the fingertips to
  have a collision mesh that closely follows the contour of fingertips' geometry" [CLAIM] —
  [Rig Closed-Loop Structures](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/rig_closed_loop_structures.html)
- GPU-compatible convex hulls are limited to 64 vertices and faces [CLAIM] —
  [Known Physics Limitations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/current_limitations.html).
  SDF needs a resolution of about 250 and is meant for detailed concave colliders [CLAIM] —
  [Collision Behavior Guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/collision_guide.html)
- Working SO-101 setups [CLAIM] —
  [soarm101-isaac](https://github.com/Jbruslind/soarm101-isaac/blob/main/docs/ISAAC_SIM_ENVIRONMENTS.md),
  [isaaclab_assets CHANGELOG](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/docs/CHANGELOG.rst):
  - the jaws use convex decomposition (soarm101-isaac) or "hand-made convex pieces per jaw" (Isaac Lab's SysID
    asset);
  - Isaac Lab fixed `SO101_CFG` "running convex decomposition instead of using the asset's authored convex hulls".
- Ours: the picture cubes are `UsdGeom.Mesh` + `convexHull` (8 vertices); the plain ones are `UsdGeom.Cube` box
  primitives. Either is fine for a box.

## 6. Friction coefficients and combine mode

### Takeaway
They are not our bottleneck: 0.5-0.8 on the pieces makes no difference. If the pad material should dominate, set
its combine mode to `max`: the pair uses the higher of the two modes.

### Evidence
- "The effective combine mode for the pair is maximum(material0.combineMode, material1.combineMode)", with the order
  average < min < multiply < max [CLAIM] —
  [PxCombineMode](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/structPxCombineMode.html).
  NVIDIA's 2F-85 tutorial gives fingertips friction 0.8 with combine mode Max [CLAIM] —
  [Rig Closed-Loop Structures](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/rig_closed_loop_structures.html)
- Compliant contacts (PhysX >= 5.1, negative restitution = spring stiffness) are "particularly useful when dealing
  with contact configurations that become easily unstable" [CLAIM] —
  [Rigid Body Dynamics](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/RigidBodyDynamics.html).
  Isaac Lab exposes `compliant_contact_stiffness/damping` [CLAIM] —
  [physics_materials_cfg.py v2.3.0](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/sim/spawners/materials/physics_materials_cfg.py).
  Possible model for the rubber pads later; untested by us.
- Ours: rubber 1.2/1.0 on `rlink2`/`llink2`, wood 0.5-0.8, both with the default (average) combine mode.

## Gaps
- No published contact-parameter study for 2-4 cm cubes in PhysX was found. I looked in the PhysX 5.6.1 docs,
  Omni Physics 107.3 guides, Isaac Lab configs and changelogs, the SO-101 projects above and the prior notes. The
  web-search quota ran out partway, so the later checks used known primary URLs only.
- The real M3's pad dimensions are not in the repo. The 7 mm reach difference is a fit, not a measurement.
- Whether 5.0's PhysX applies `minTorsionalPatchRadius` in our case (one anchor per patch) needs a grip-bench check.
