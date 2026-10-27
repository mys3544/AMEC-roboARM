# How working Isaac setups grasp small objects, and what 6.x changes for our twin (state of October 2026)

Scope: the settings of setups that are known to grasp 2-5 cm objects in Isaac Sim / Isaac Lab, set beside ours:
- Isaac Lab's Franka lift and stack tasks (2.3 and current `main`);
- Isaac Lab's SO-101 tasks and assets;
- a community SO-101 project on Isaac Sim 6.2 with measured success rates;
- LeRobot's official SO-101 Isaac Sim scene;
- NVIDIA's Robotiq 2F-85 tuning example.

It also lists which of the knobs exist in Isaac Sim 5.0 (Dionysos), 5.1 and 6.x, and what else 6.x changes for the
twin. The version and driver question itself is in
`research_notes/Simulácia a VLA pre presný úchop/isaac_sim_versions_side_by_side.md`. Labels as in the other notes.

## 1. Side by side: what the working setups set

### Takeaway
Every working small-object setup differs from ours in the same places:
- a smaller friction correlation distance (6.25 mm) or a higher physics rate (100-240 Hz);
- object depenetration capped (1-5 m/s);
- 0-1 velocity iterations;
- gripper effort at or below the real actuator's torque;
- on 5.1+/6.x, `solve_articulation_contact_last`.

None of them drives a linkage with 50 N·m stiff drives. Our 120 Hz and 16-32 position iterations are in range.

### Evidence

| setup | physics rate | solver / iterations | scene contact settings | object | gripper drive | source |
|---|---|---|---|---|---|---|
| Isaac Lab 2.3 Franka lift (Isaac Sim 5.x) | 100 Hz, decimation 2 | TGS; robot 8/0; cube 16/1 | friction corr. 6.25 mm, bounce 0.01 | DexCube x0.8, max depenetration 5 m/s | prismatic, k 2e3, d 1e2, effort 200 N; close target 0 (full) | [lift_env_cfg](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py), [joint_pos cfg](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/joint_pos_env_cfg.py), [franka.py](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_assets/isaaclab_assets/robots/franka.py) |
| Isaac Lab main, stack (Franka, SO-101) | 100 Hz, decimation 5 | TGS; cubes 16/1 | friction corr. 6.25 mm, bounce 0.01, **contact last** | max depenetration 5 m/s | per robot | [stack_env_cfg](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/contrib/stack/stack_env_cfg.py) |
| Isaac Lab main, core lift | 120 Hz, decimation 4 | objects 16/0 | defaults | 5 cm cube 0.2 kg, contact offset 2 mm, friction 0.5-1.0 random | per robot | [core/lift](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/core/lift/lift_env_cfg.py) |
| Isaac Lab SO-101 (SysID asset) | task's | robot 8/0 | task's | max depenetration 1 m/s on links | USD SysID: 68 N·m/rad, max 3.35 N·m; high-PD variant 17.8 N·m/rad, 10 N·m | [so101.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/so101.py), [soarm101-isaac doc](https://github.com/Jbruslind/soarm101-isaac/blob/main/docs/ISAAC_SIM_ENVIRONMENTS.md) |
| soarm101-isaac (SO-101, Isaac Sim 6.2 / Lab 3.0) | 30 Hz policy | robot 8/0 | friction corr. 6.25 mm, bounce 0.01, **contact last** | 3 cm cube, friction 1.0 | k 80, d 4, effort 10 N·m; jaw target 0.14 rad past contact, clamped to 0.05 rad ahead | [doc](https://github.com/Jbruslind/soarm101-isaac/blob/main/docs/ISAAC_SIM_ENVIRONMENTS.md) |
| LeRobot so101-tabletop-isaac (Isaac Sim 6.2) | **240 Hz** CPU PhysX, 8 substeps / 30 fps | asset's | defaults | 3 cm, 10 g wooden cubes, friction 1.0 | asset's (NVIDIA RobotStudio SO-101) | [env.py](https://huggingface.co/lerobot/so101-tabletop-isaac/blob/main/env.py) |
| NVIDIA 2F-85 example | 250 Hz (example), >= 80 Hz needed for a 2.5 kg payload | TGS, up to 64 position iterations | - | cylinder | max force from spec, full force at a 5 deg target delta, armature suggested | [gripper tuning](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/gripper_tuning_example.html), [rig tutorial](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/rig_closed_loop_structures.html) |
| **ours (Isaac Sim 5.0)** | 120 Hz | TGS; robot 32/4; pieces 16/4 | defaults (corr. 25 mm) | 15 g every piece, contact offset 2 mm, depenetration unset | finger 4 N·m/deg (229/rad), max 1.5 N·m (trial 12/deg, 4 N·m); linkage 200 N·m/deg, 50 N·m | `m3pro/sim/isaac_m3.py`, `m3_sim.py` |

All entries are [CLAIM] (configs as published), except "ours".

## 2. Measured sim grasp results in these setups

### Takeaway
The only measured, controlled A/B results for small parallel grasps in Isaac are from soarm101-isaac, all in
simulation:
- contact-last on vs off: 20/20 vs 1/10;
- wrist aligned to the cube faces: 20/20 vs 11/20;
- slower carry: 62/64 vs 55/64.

No published study calibrates *grasp holding* in Isaac against the real arm. The real-vs-sim numbers that exist are
end-to-end policy successes (prior notes).

### Evidence
- soarm101-isaac [SIM] —
  [doc](https://github.com/Jbruslind/soarm101-isaac/blob/main/docs/ISAAC_SIM_ENVIRONMENTS.md),
  [issue #28](https://github.com/Jbruslind/soarm101-isaac/issues/28):
  - before the fixes, 0/5 placed;
  - after: 20/20 headless, 5/5 with cameras; with `solve_articulation_contact_last=False`, 1/10;
  - without wrist yaw aligned to the cube faces, 11/20: "they catch its edges and the jaw stalls at 0.31-0.38 rad";
  - fastener carry: 1 cm/step gave 55/64, "7 of 9 failures: the bolt slipped out of the jaws during carry";
    0.6 cm/step gave 62/64.
- Isaac Lab moved the SO-101 stack tasks back to PhysX because under Newton MJWarp "the gripper ... penetrates the
  cubes" and grasps did not hold [CLAIM] —
  [isaaclab_tasks CHANGELOG](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/docs/CHANGELOG.rst)
- Real-vs-sim results for cheap arms (prior notes, not re-checked here):
  - a Genesis-pretrained ACT for SO-101 reached the cube in sim but "did not complete grasps" from sim alone; with 10
    real episodes it reached 17/20 real vs 5/20 [REAL-LC];
  - so101-real2sim: twin replay 9/16 with stock actuators vs 12/16 identified [SIM];
  - Squint: 73/80 real [REAL-LC].
  - `research_notes/Simulácia a VLA pre presný úchop/open_sim2real_pipelines.md`
- Not found: a "grasp held in sim vs held in real" calibration for PhysX. I looked in Isaac Lab's configs and
  changelogs, the SO-101 projects above, LeRobot's Isaac env README, NVIDIA's 2F-85 guides and the prior notes;
  web search ended early on quota.

## 3. Which knob exists in which version

### Takeaway
On Dionysos's 5.0 we can set everything except contact-last. Contact-last needs 5.1 or 6.x. hekate has 6.0, and
6.0.1 runs on Dionysos's driver per the 2026-10-26 notes. Per-camera exposure and the ISP model are 6.x only.

### Evidence

| knob | 5.0 | 5.1 | 6.x | source |
|---|---|---|---|---|
| friction correlation distance, offset threshold, bounce threshold | yes | yes | yes | [PxSceneDesc](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/classPxSceneDesc.html) |
| torsional / min torsional patch radius (TGS only) | yes (schema since 2023) | yes | yes | [PxShape](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/classPxShape.html), [forum, 2023.1.1](https://forums.developer.nvidia.com/t/geometryprimview-uses-out-of-date-usdphysics-api-name/280412) |
| joint armature, joint friction | yes | yes | yes | [Articulations](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html) |
| mimic joints with compliance (natural frequency, damping ratio) | yes (107.3 guide; importer authors them) | yes | importer writes NewtonMimicAPI instead | [stability guide 107.3](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html), [URDF importer 6.0](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html) |
| `solveArticulationContactLast` | **no** | yes (new in 5.1) | yes | [5.1 release notes](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/release_notes.html) |
| per-camera exposure (`OmniRtxCameraExposureAPI_1`) | no (Kit 107.3) | no | yes (Kit >= 108.1) | [RTX 108.1](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer-release-notes/108_1.html), [cameras](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html) |
| camera ISP pipeline (WB, CCM, noise) | no | no | 6.0, sample ISP Linux-only | [6.0 camera sensors](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/sensors/isaacsim_sensors_camera.html) |
| multitick rendering (cameras on physics time) | no | no | 6.0 | [6.0 release notes](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/release_notes.html) |

Isaac Sim 5.1 is marked "no longer supported" on its own docs [CLAIM] —
[5.1 rig tutorial page](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/rig_closed_loop_structures.html).

## 4. What else 6.x changes for the twin

### Takeaway
Moving the twin to 6.x is more than a reinstall:
- the URDF must be re-imported with the rewritten importer: mimic as Newton schema, a new articulation root API, a
  new import API;
- the default renderer becomes RTX Real-Time 2.0, so the colours change and the colour fit must be redone;
- the camera API moves to `isaacsim.sensors.experimental.rtx`.

The gains are contact-last, per-camera exposure and multitick rendering. Stay on PhysX: Newton's gripper contact is
"still being tuned".

### Evidence
- Isaac Sim 6.0.0 [CLAIM] — [6.0 release notes](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/release_notes.html):
  - Kit 110.1.1;
  - "URDF and MJCF importers apply Newton schemas to imported assets";
  - multitick rendering ("Cameras ... can be scheduled and rendered at rates and offsets driven by physics simulation
    time");
  - `isaacsim.sensors.camera` deprecated in favour of `isaacsim.sensors.experimental.rtx`;
  - `URDFImportFromROS2Node` deprecated: "Use RobotDefinitionReader and URDFImporter instead";
  - experimental Newton backend.
- RTX [CLAIM]:
  - from Kit 108.0, "RTX - Real-Time 2.0 is Now the Default Render Mode" —
    [RTX 108.0](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer-release-notes/108_0.html);
  - the Kit 109.0.2 notes list "Scenes appeared significantly darker when using RTX Real-Time 2.0 compared to RTX
    Interactive mode" —
    [RTX 109.0.2](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer-release-notes/109_0_2.html).
- 6.0.1 black-render field reports on Ampere and multi-GPU, fixed by driver 595 or by disabling multi-GPU [2nd] —
  [#875](https://github.com/isaac-sim/IsaacSim/discussions/875), [#758](https://github.com/isaac-sim/IsaacSim/discussions/758).
  Dionysos's A4500 is Ampere.
- Isaac Lab SO-101 stack tasks default to PhysX: Newton MJWarp's "gripper contact response is still being tuned for
  this robot" [CLAIM] —
  [isaaclab_tasks CHANGELOG](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/docs/CHANGELOG.rst)
- LeRobot's SO-101 Isaac env (built on 6.2, updated 2026-10-27) is a compact template for the 6.x Python API:
  - `SimulationManager.setup_simulation(dt=...)`, `enable_gpu_dynamics(False)`;
  - a paused timeline with physics stepped explicitly;
  - captures via `rep.orchestrator.step(delta_time=0.0, pause_timeline=False)` after
    `physx.update_transformations`;
  - sRGB -> linear colours, DLAA.
  - [CLAIM] — [env.py](https://huggingface.co/lerobot/so101-tabletop-isaac/blob/main/env.py)

### Inference
Port in this order:
1. test contact-last on hekate's 6.0 or a side-by-side 6.0.1 on Dionysos with the M3 USD re-imported (the grip
   bench only);
2. if it helps, move the expert and recorder;
3. redo the colour fit under RT 2.0.

Until then, apply all 5.0-compatible settings on Dionysos. Ask the user before installing on Dionysos or using
hekate's GPU, as CLAUDE.md requires.

## Gaps
- The SO-101 SysID asset's armature and friction values sit in a binary USD that was not read here. They would give
  a measured hobby-servo armature to compare with our 1e-3 kg·m^2 assumption.
- No source compares grasp holding between 5.0 and 6.x on the same asset.
