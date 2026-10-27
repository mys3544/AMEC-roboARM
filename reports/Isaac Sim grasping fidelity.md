# The twin drops far-rim picks because its finger pads are too short, not because of friction: lengthen the pads first, then fix PhysX's friction patches, drive stiffness and render settings

**Bottom line.** In our 189 simulated picks, one thing decides whether a piece reaches the box: how far the URDF's
finger pads reach down the piece's side.
- **Overlap and success:** under 13 mm of vertical overlap, 8 of 31 picks succeed; at 13 mm or more, 129 of 158
  (82 %).
- **The far rim:** at pitch 40-60 deg, half the grips have less than 13 mm.
- **Why:** the URDF's pads end 12.9 mm past the grip point at a 30 mm opening, and only 9.4 mm at 40 mm. The planner
  assumes 15 mm, and the fingertip touches of 26 October fit the real tips about 7 mm longer than the URDF's.

So the first fix is geometry. Caliper the real rubber pads and give the twin box colliders of that size. If the
7 mm holds, 21 of the 22 shallow far-rim grips move into the 82 % band.

The contact model also needs to behave more like rubber on wood:
- a 6.25 mm friction correlation distance, as in Isaac Lab's grasp tasks;
- a small minimum torsional patch radius;
- depenetration capped at 1 m/s;
- 240 Hz physics;
- a finger drive that is 7x softer, with servo-like armature;
- proper mimic joints instead of the 50 N·m drives that now hold the linkage.

The best documented fix for grippers that push objects out, solving contacts last, exists only from Isaac Sim 5.1,
so test it on 6.0.

For the camera, the paint colours are photo (sRGB) values used as linear albedo, which makes them 2.5-5x too light.
Convert them, fit one exposure and tone curve, and randomise around it. Make rendering synchronous and
native-resolution with launch flags. The one-in-three black runs beside a training have no documented cause except
Windows 11 with a pre-595 driver, so keep the detector and the rerun.

The settings below can be tried in one afternoon on Dionysos's 5.0.

Sources and the full evidence are in `research_notes/Isaac Sim grasping fidelity/` (four notes). Numbers are labelled
there: [SIM] simulation, [REAL-LC] our or another low-cost arm, [CLAIM] vendor or author statement, [2nd] secondary.

## 1. Success follows the pads' overlap with the piece, and the URDF's pads are short

The 2026-10-27 summary said "the grip's height decides". Re-reading the same 189 picks (`out/sim/episoderesult_*`,
`out/sim_episodes/*`) with the URDF's finger meshes makes the mechanism visible. What matters is the overlap: how far
the pads reach down the piece's side when the fingers shut. It combines the grip height, the pitch and the pad's
length. [SIM]

| vertical overlap of the pads with the piece | picks | in the box | lost after lifting | never lifted |
|---|---|---|---|---|
| under 10 mm | 8 | 0 % | 1 | 7 |
| 10-13 mm | 23 | 35 % | 6 | 9 |
| 13 mm or more | 158 | 82 % | 20 | 9 |

| pitch | picks | in the box | median overlap | grips under 13 mm | ... if the pads were 7 mm longer |
|---|---|---|---|---|---|
| 40-60 deg (far rim) | 42 | 55 % | 12.9 mm | 22 | 1 |
| 61-79 deg | 60 | 70 % | 15.7 mm | 7 | 1 |
| 80-90 deg | 87 | 83 % | 18.7 mm | 2 | 1 |

The URDF's geometry, from forward kinematics of `M3Pro.urdf` with its own mimic shares:
- **Reach:** the inner face of each pad runs from about 7 mm behind Yahboom's `Gripping` point to 12.9 mm past it
  when the fingers shut on a 30 mm piece, and to 9.4 mm past at 40 mm.
- **Width:** the fingers are 6 mm thick, so the contact is a strip at most 6 mm wide.
- **Angle:** the convex hulls' inner faces open 1.4-2.0 deg toward the tips. The first contact is therefore at the
  pad's root or the piece's top edge.

The robot's model assumes the tips reach 15 mm past the Gripping point (`m3kinematics.FINGERS`). The touch test fit
the real shut tips 9 mm beyond that: about 24 mm, against the URDF's 16.8 mm when shut. The real M3 held the same
high grips, 8 of 8 on 26 October, including 24 cm out. [REAL-LC]

The timing agrees with a geometric cause:
- all 25 "never lifted" pieces left within 0.25-0.66 s of the shut, as the arm began to rise, gripped at 14.7 mm
  over their middle (median), that is at the top edge of a 30 mm cube;
- friction 0.5-0.8 made no difference.

The fingers squeezed; they squeezed the wrong few millimetres.

The 7 mm comes from one fitted parameter that trades against J2, not from a caliper. So the first step is to measure
the real pads: length past the crank pin, width, and how far the rubber stands proud of the plastic. Compare them with
`rlink2.STL`. If they are longer, add a box collider of the real pad's size to `rlink2` and `llink2`, with the rubber
material. The planner needs no change: it already aims for the real fingers.

## 2. The contact model cannot resist turning, and the drives win against the contacts

Two PhysX facts matter for a narrow pad on a 30 mm face.

**Friction anchors.** PhysX 5 has only patch friction: "Up to two contact points lying in the contact patch area are
selected as friction anchors". Contacts closer than the scene's `frictionCorrelationDistance` are merged into one
anchor point. The default is 25 mm. A 6 x 13-20 mm strip almost certainly ends up with a single anchor. A single
anchor cannot resist the piece turning about the contact normal, which in a parallel grasp is the grasp axis.

PhysX has torsional friction (`torsionalPatchRadius`, `minTorsionalPatchRadius` on each shape), with two limits:
- it "Will only be active, if the friction patch has a single anchor point only";
- it is "Only supported in combination with solver type PxSolverType::eTGS".

Isaac Lab's Franka lift task and its current stack tasks set the correlation distance to 6.25 mm; we use the
default. [CLAIM]

**Constraint order.** Within an articulation PhysX solves, in order:
1. dynamic contact;
2. mimic joints;
3. drives;
4. limits;
5. static contact.

"the constraint that runs last will 'win'". A stiff finger drive therefore undoes each iteration's contact
correction and the pad ends up inside the piece. NVIDIA added a scene flag, `physxScene:solveArticulationContactLast`,
that moves dynamic contact to the end. It "can lead to a reduction in persistent contact depth ... stronger contact
normal forces and stronger friction forces that aid gripping strength". It is **new in Isaac Sim 5.1**, so 5.0 does
not have it. [CLAIM]

On an SO-101 picking 3 cm cubes in Isaac Sim 6.2, it took placing from 1/10 to 20/20, together with two planner
fixes. Isaac Lab now enables it for every stack task, with the comment "Let object contacts stall position-driven
grippers before they tunnel through a grasp". [SIM, CLAIM]

## 3. The gripper's drives are 30-1000 times too stiff for 120 Hz

NVIDIA's articulation guide gives a rule: keep (drive natural frequency) x (timestep) near 1, "particularly if the
driven joint participates in a mimic joint or ... a gripping action". PhysX drives are implicit:
F = s·dx / (dt²·nf² + 1). Past a point, more stiffness gives *less* force, and stiff drives fight contacts. Ours,
computed from the URDF's inertias about each joint axis:

| drive | stiffness | natural frequency | nf·dt at 120 Hz |
|---|---|---|---|
| finger (`rlink1`, crank + pad) | 4 N·m/deg = 229 N·m/rad | 3,600 rad/s | 30 (137 for the crank alone) |
| FINGER_SQUEEZE trial | 12 N·m/deg | 6,300 rad/s | 52 |
| linkage on the pad (`rlink2`) | 200 N·m/deg | 48,000 rad/s | 401 |
| linkage on the idler (`rlink3`) | 200 N·m/deg | 123,000 rad/s | 1026 |
| finger + 1e-3 kg·m^2 armature | 4 N·m/deg | 475 rad/s | 4 |

The linkage drives' 50 N·m limit applied to a 5e-6 kg·m^2 pad link can add about 85,000 rad/s in one 1/120 s step.
The guide warns against exactly this.

NVIDIA's 2F-85 gripper example gives the recipe:
- **max force** = the actuator's real torque;
- **stiffness** = max force / the target overshoot at which full force is wanted;
- **damping** for a damping ratio of 1;
- **armature** = rotor inertia x gear ratio^2, to bring the natural frequency down. For a hobby servo this is far
  more than the URDF's 1e-7 to 1e-5 kg·m^2 links.

Our planner closes about 2.5 deg (0.044 rad) past the piece, as the real servo does. At today's 1.5 N·m that gives
**0.6 N·m/deg stiffness, armature 1e-3 kg·m^2, damping 0.0065 N·m·s/deg**: the same requested torque, 7x softer,
well conditioned. For scale, the system-identified SO-101 bus servo in Isaac Lab has 68 N·m/rad (1.19 N·m/deg) with a
3.35 N·m limit, and one SO-101 sim held a 4 cm cube with 1 N·m. [CLAIM, SIM]

The FINGER_SQUEEZE trial (12 N·m/deg, 4 N·m) moves the other way: stiffer, with 2.7x the force cap. It helps only if
the delivered force is too low. Nobody has measured the delivered force. Log the finger's measured joint effort and
the contact force in the sim, and the real squeeze with a scale (section 7, B).

Why the linkage is driven at all: the 5.0-era URDF importer looks a mimic joint's reference up by name *at the moment
it creates the mimic joint* (`FindPrimByNameAndType`). If the reference does not exist yet, it silently leaves both
the reference and the limits empty. Our left-side joints reference the right crank `rlink1_Joint`. That fits "the
importer left the left side's without a reference joint".

Set by hand, a mimic joint needs:
- gearing = **−multiplier** (PhysX: q_A + G·q_B + γ = 0);
- finite limits;
- **no drive of its own**. NVIDIA: "having an authored joint drive would negatively impact the drive outcome".

That is the clean replacement for the 200 N·m/deg drives. NVIDIA's own 2F-85 model is a tree of six revolute joints,
five of them mimics, with no loop closure.

## 4. Half the lifted-then-lost picks go in the carry, where the twin's arm jerks

Of the 27 picks lifted and then lost, 14 went during the lift and 13 during the carry to the box, 1.2-2.3 s after the
shut. The carry losses sit mostly at steep pitch (75-90 deg) with about 12 mm grips. Overlap does not explain them.
[SIM]

The twin plays the board's sweeps as straight lines, and the arm drive (40 N·m/deg) tracks them almost exactly. The
joint velocity therefore jumps at every key, so the held piece sees short, hard accelerations. The real servos lag
0.10-0.20 s and turn 0.87-0.93 of a 15 deg command (the servo probe), which smooths the same sweep. [REAL-LC]

NVIDIA's guide suggests "a rate-limiter on the drive targets". In a community SO-101 fastener task built on Isaac
Lab (soarm101-isaac), a faster carry (1 cm/step instead of 0.6) dropped placing from 62/64 to 55/64, "7 of 9
failures: the bolt slipped out of the jaws during carry". [SIM]

A first-order filter of 0.05-0.1 s on the arm targets in `isaac_m3.py` is servo-like and costs nothing.

## 5. Fixes ranked by expected effect on the far-rim grasps

The expected effects are inferences from the numbers above, not measurements.

1. **Pad geometry as the real pads (measure first).** It addresses the 25 never-lifted and part of the lift losses.
   At the far rim, 22 of 42 grips sit under 13 mm overlap; with pads 7 mm longer, 1 would. Expected: far-rim success
   from 55 % toward the 80 % the twin already reaches with enough overlap.
2. **`solveArticulationContactLast`.** It needs 5.1 or 6.x. It is the documented fix for pads squeezing pieces out
   (1/10 -> 20/20 on an SO-101 in sim). Expected: fewer "flew off" and lift losses. Test it on 6.0 with the grip
   bench before porting the pipeline.
3. **Friction correlation distance 6.25 mm and a small minimum torsional patch radius on pads and pieces.** They give
   the narrow pad a way to resist turning. Expected: fewer pieces swinging out during lift and carry.
4. **Drive conditioning:** a softer finger drive with armature, mimic joints (or linkage drives at 20 N·m/deg and
   5 N·m with armature), and `maxDepenetrationVelocity` 1 m/s on the pieces. Expected: less jitter and penetration;
   no more ejections.
5. **240 Hz physics** (LeRobot's SO-101 Isaac scene for 3 cm cubes; NVIDIA: "Contacts are very sensitive to
   time-stepping"). Expected: smaller, cheaper gains on top of 2-4.
6. **Servo-like smoothing of the arm targets.** It addresses the 13 carry losses, mostly at steep pitch, so it matters
   less for the far rim but more for the totals.
7. **Real piece masses** (all are 15 g today) and friction. Small: friction already shows no effect. Heavier pieces
   make the twin slightly harder, which is the honest direction.

## 6. The camera: black runs are a timing or driver issue; the light colours are a colour-space bug

**Black frames.** Our launch is `SimulationApp({"headless": True})`, which in 5.0 means:
- DLSS anti-aliasing (`anti_aliasing = 3`), multi-GPU on, synchronous material loads on;
- from the base experience, `renderer.asyncInit = true`.

The documented causes of black output on Windows:
- **materials not loaded yet:** `rt_subframes` >= 2;
- **async rendering or skipped frames:** the throttling extension toggles async rendering
  (`--/exts/isaacsim.core.throttling/enable_async=false`); a missing first frame on Windows needs an extra
  `rep.orchestrator.step()`;
- **driver and multi-GPU bugs:** "On Windows 11 ... SDG pipelines may write black images. To resolve this, update the
  NVIDIA display driver to version 595 or higher" (6.0 known issues). Dionysos is Windows 11 on 581.42.

Isaac Lab's headless camera app adds `app.renderer.waitIdle`, `app.hydraEngine.waitIdle` and
`app.updateOrder.checkForHydraRenderComplete = 1000`, "Avoids frame offset issue", and limits rendering to one GPU.
A Windows 11 forum thread on 5.1 reports intermittent all-zero frames, unresolved. Staff advised not calling
`world.reset()` during capture and retrying with `app.update()`. [CLAIM, 2nd]

I found no source tying black frames to a concurrent CUDA training. The practical course:
- launch with the synchronous settings plus `anti_aliasing = 4` (DLAA: LeRobot's Isaac scene uses it because "the
  default DLSS renders 640x480 cameras at 320x240") and `multi_gpu = False`;
- keep the black detector and rerun;
- keep renders and trainings apart, as the night plan already does;
- leave the driver question to the user, because the machine is shared.

Check one side effect: if the annotator returns the previous frame, every recorded picture lags its state by one tick.

**Colours.** Measured on the 26 October overlay of the wide look [SIM vs REAL-LC]:

| patch | real DCW2 | twin |
|---|---|---|
| paper | 145-184 | 213-216 |
| green cube top | (28,58,26) | (134,176,135) |
| black tape | about 2 | 45 |
| luma contrast (std) | 47 | 18 |

Two faults:
1. **Colour space:** `CUBE_PAINT` and the other colours are photo values ("paint as measured on them (the user's
   photos)") passed as linear albedo. LeRobot's Isaac scene converts every colour with `c**2.2`. For green
   (0.23, 0.46, 0.23), the linear albedo should be (0.04, 0.18, 0.04), which makes the cube about 2x darker and twice
   as saturated.
2. **The camera itself:** the real camera's auto-exposure makes white paper mid-grey, and its tone curve crushes
   blacks.

Fit one per-channel curve on 3-5 paired looks and apply it to the rendered frame in the recorder; this is the
published method, and it raised sim-real correlation strongly in the prior notes' sources. Then randomise around the
fit within what the real camera does: exposure x0.85-1.25 (our scans: x0.89-1.20), white balance ±5 %, contrast
±10 %, corners 0-10 % darker (ours: 7-9 %). Add colour jitter in training too. One SO-101 study saw 73/80 real with
it and 58/80 without. [REAL-LC]

Per-camera exposure (`OmniRtxCameraExposureAPI_1`) and an ISP model exist only from Kit 108.1, that is Isaac Sim 6.x.
6.x also defaults to RT 2.0, so the fit must be redone after a move. [CLAIM]

## 7. Settings to try in one afternoon (Isaac Sim 5.0 on Dionysos, nothing else running on the GPU)

| setting (where) | value | why | expected effect | how to check |
|---|---|---|---|---|
| Pad colliders: a box on `rlink2` and on `llink2`, rubber material | the real pad's size and position, from calipers (provisional: the URDF pad extended 7 mm toward the tip) | the URDF's pads reach 12.9 mm past the grip point; the real tips are fitted about 7 mm longer | far-rim "never lifted" mostly gone | bench A at pitch 50, grip 15 mm over the middle: lifts? |
| `physxScene:frictionCorrelationDistance` | 0.00625 | two friction anchors on a narrow pad (Isaac Lab lift and stack) | less turning of held pieces | bench A: piece's tilt against the fingers during the lift |
| `physxScene:bounceThreshold` | 0.01 | as Isaac Lab's grasp tasks | none expected (restitution 0) | none |
| `physxCollision:minTorsionalPatchRadius` (pads, pieces) | 0.0015 m | torsion where a patch keeps one anchor (TGS only) | as above | as above, A/B against it off |
| `physxCollision:contactOffset` / `restOffset` (pads) | 0.002 / 0.0 | explicit, the same as the pieces | less pre-contact jitter | penetration in bench A < 0.5 mm |
| `physxRigidBody:maxDepenetrationVelocity` (pieces) | 1.0 m/s | squeezed light pieces are shot out by depenetration | no "flew off" | count ejections in the batch |
| piece masses | weighed: about 16 g (30 mm), 35-45 g (40 mm cube), about 30 g (block) | all are 15 g today | slightly harder, more honest | none |
| physics rate (`World(physics_dt=...)`) | 1/240 | contacts are timestep-sensitive (LeRobot SO-101 scene: 240 Hz) | small extra margin | seconds per episode (expect a few % more) |
| velocity iterations (robot, pieces) | 1 (now 4) | PhysX: TGS often needs none; joint forces read high with them | none on the grip; cleaner effort readings | effort logs |
| finger drive (`rlink1_Joint`) | 0.6 N·m/deg, 0.0065 N·m·s/deg, max force 1.5 N·m (then the measured stall torque) | full torque at the planner's 2.5 deg overshoot, as a servo P loop | same squeeze, no stiffness fight | bench A: finger effort ≈ max force when shut; closing time as the real |
| `physxJoint:armature` (`rlink1_Joint`, `llink1_Joint`) | 1e-3 kg·m^2 | reflected servo inertia; nf·dt 30 -> about 1.5 | no pad jitter | pad tilt and contact force steady over 2 s |
| linkage | mimic joints: reference `rlink1_Joint`, gearing = −multiplier, no DriveAPI, limits ±70 deg; fallback LINK_DRIVE 20 N·m/deg, max 5 N·m, armature 1e-4 | the 200 N·m/deg, 50 N·m drives fight the contacts | the same, with fewer hard constraints | pads parallel within 1 deg under squeeze |
| arm targets | first-order filter, time constant 0.08 s (or an acceleration cap) | the board's sweeps tracked stiffly jerk the held piece | fewer carry losses | lost-in-carry count on the same seeds |
| render launch (`SimulationApp` config) | `anti_aliasing: 4`, `multi_gpu: False`, `extra_args`: `--/app/renderer/waitIdle=true`, `--/app/hydraEngine/waitIdle=true`, `--/app/updateOrder/checkForHydraRenderComplete=1000`, `--/exts/isaacsim.core.throttling/enable_async=false` | synchronous, native resolution, one GPU (Isaac Lab, LeRobot) | sharper frames, no stale frames, maybe fewer black runs | 10 runs beside a training: black count (baseline about 1 in 3); seconds per picture |
| paint colours (`m3_sim.py`, `isaac_m3.py`) | `c ** 2.2` on every colour constant | photo colours used as linear albedo | the green top from 176 to about 110, saturation about doubled | the overlay patches |
| exposure and tone (recorder, after rendering) | per-channel quadratic fit on 3-5 paired looks, then exposure x0.85-1.25, WB ±5 %, contrast ±10 %, vignette 0-10 % per episode, stored in the robot card | the DCW2's auto-exposure and tone curve | patches within about 20 levels; contrast within 20 % | overlay patches; DINOv2 distance between real and sim looks (optional) |

Later, on 6.0 (hekate, or a side-by-side 6.0.1 on Dionysos; ask the user first): `physxScene:solveArticulationContactLast
= True` on the re-imported M3. The 6.x importer writes mimic as `NewtonMimicAPI`; check that the PhysX backend obeys
it.

## 8. Test protocol

Run everything on fixed seeds and layouts, so each change is a paired A/B.

**A. Grip bench in the twin (minutes per variant).** Use `m3_sim.py grip` (a cube picked as the M3 picks it) on a grid:
- 3 spots: 17, 21 and 25 cm from J1's axis;
- 3 pieces: 30 mm cube, 40 mm cube, block;
- 2 grip heights: as planned, and planned + 4 mm.

Log per pick:
- the overlap;
- whether it lifts and holds;
- the piece's tilt against the fingertips during the lift;
- the finger's measured joint effort, and the contact force from a `PhysxContactReportAPI` on the piece;
- penetration.

Then a grip-strength probe without new code: repeat the lift with `--mass` 15, 50, 100 and 200 g and note the mass
at which it slips. Order the variants as in the table: geometry first, each next one on top.

**B. The same on the real M3 (half an hour, the user with the arm).**
- Caliper the pads: length past the crank pin, width, rubber thickness. Compare with `rlink2.STL`.
- With the arm holding a 30 mm cube at pitch 50 and 90, at grip heights about 9 and 15 mm over the middle, pull the
  cube out along the fingers with a thread and a luggage or kitchen scale. Read the force at which it slips.
- Do the same in the twin: a ramped force, or the mass at which it slips.

The twin should hold within the real's range, not far above it.

**C. Far-rim batch.** 40 simulated picks drawn only from the far rim (pitch 40-60, more than 23 cm out), baseline
against the winning variant, with the same seeds. Report:
- in the box, never lifted, lost in the lift, lost in the carry;
- Wilson intervals;
- McNemar on the discordant pairs.

Success means the far-rim rate is within about 10 points of the middle-of-the-table rate. A rate near 100 % with
real-looking grips is suspicious: the twin should still fail where the real arm fails.

**D. Render.** 10 runs of `m3_sim.py episode` beside an ACT training, before and after the launch settings. Report
the black runs, seconds per picture, and one frame-lag check: step the arm and see in which tick the picture moves.

**E. Colour.** Before and after the colour fixes, on 3-5 paired looks: mean RGB of paper, tape, desk grey, each paint
and a picture face. Report luma std. Target: patches within about 20 levels, contrast within 20 %.

## 9. Risks

- **The 7 mm is a fit, not a measurement.** If the real pads are not longer, lengthening the twin's pads makes the
  twin optimistic. Measure before changing (B).
- **Mixed physics in one dataset.** The expert's episodes would come from two physics versions. Store a sim-settings
  version beside the robot card in `episode.json` and let `m3_episodes build` filter on it, as it does for
  `twin_sag`.
- **Softer drives and mimic joints can let the pads tilt and wedge.** That was the 23 October failure with targets
  instead of positions. Check pad parallelism in A before a batch.
- **Torsional patch radius may do nothing** once the smaller correlation distance creates two anchors, since it acts
  only on single-anchor patches. Keep it only if A shows an effect.
- **Contact-last means moving to 6.x:**
  - a new URDF importer (mimic as Newton schema, new API);
  - RT 2.0 colours;
  - field reports of black renders on Ampere with non-595 drivers.
  Test it with the bench before porting the pipeline.
- **Rendering costs:** DLAA and the wait-idle settings slow rendering. The episode budget is about 3.3 min now;
  measure seconds per picture.
- **Colour fitting:** a fit on 3-5 frames under one light overfits; the randomisation and training-time jitter must
  cover the rest. Do not also tune the renderer's tonemapper toward the same target, or the correction is applied
  twice.
- **The driver:** a newer driver on Dionysos affects the machine's other users and the 5.0 install. It is the user's
  decision.
- **Unknown source of the black runs:** the remaining one-in-three may have no fix in 5.0. The detector and rerun
  stay.

## Conclusion

The twin loses far-rim picks mainly because its fingers are shorter than the real ones. Success tracks how far the
URDF's pads reach down the piece: 82 % with 13 mm or more of overlap, 26 % below. The far rim is where the planner
grips highest and the pads reach least. Measure the real pads and give the twin colliders of that size; that is the
largest lever.

Then bring the contact and the gripper into the range NVIDIA and working SO-101 setups use:
- a 6.25 mm friction correlation distance;
- a small torsional radius;
- capped depenetration;
- 240 Hz;
- a servo-like finger drive with armature;
- mimic joints instead of 50 N·m linkage drives;
- smoothed arm targets for the carry.

Plan the move to Isaac Sim 6.x for `solveArticulationContactLast`, the documented cure for grippers that push pieces
out.

On the camera:
- convert the photo colours to linear;
- fit and randomise the camera's exposure and tone curve;
- launch with synchronous, native-resolution rendering;
- keep the black-frame detector, because the one-in-three black runs beside a training have no documented cause
  except Windows 11 with a pre-595 driver.

Everything except contact-last and per-camera exposure works in 5.0 and can be tried in one afternoon with the grip
bench, before the next night batch.
