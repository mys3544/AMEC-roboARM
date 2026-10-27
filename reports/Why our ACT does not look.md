# Our ACT does not look because it can win without looking: it copies its own last command and memorises its 42 training scenes

**Bottom line.** The M3's ACT heads one way wherever the cube is because its training rewards two shortcuts, not
because ACT or a ResNet18 cannot see a 30 mm cube. First, `observation.state` is exactly the previous action, so a
nearest-neighbour lookup on the six joint numbers alone -- no camera -- scores **7.79 deg** on the held-out chunks where
Friday's ACT scored 7.79 and the weekend's 6.05 (holding still: 13.29); only the ~2 % of frames standing at the look
pose need the picture to know which way to go. Second, the 154 co-training attempts are only **42 scenes**, and the real-only
baseline saw its 13 scenes ~140 times: it reproduces the heading on its own training START frames (r 0.99) but
is blind on unseen ones (slope 0.08 on simulated frames; on mirrored real frames it shrinks to the middle), and once
moving it steers by its state (weight 0.83 against 0.10 for the picture). ACT's CVAE plays no part (KL 0.000). The
published fix with the largest measured effect on ACT itself is to remove the state input -- 0.084 -> 0.517 horizontal
and 0 -> 0.933 height generalisation on a real arm ([Zhao et al. 2025](https://arxiv.org/html/2509.18644v2)) -- with
image augmentation (robomimic: 26.7 -> 73.3 % on a real arm, [arXiv 2108.03298](https://arxiv.org/html/2108.03298))
and many more distinct layouts ([Lin et al. 2024](https://arxiv.org/abs/2410.18647)). **Do next:** probe the co-trained
150k and ACT v2 with the 30-minute offline test before any robot time; then train **R1** (dataset v2, no state input,
augmentation, the decision frames upweighted, whole layouts held out) beside its with-state twin; relative actions next,
a target mark third. All keep the 6 ms TensorRT path. Our measurements and the code of the probe:
`research_notes/Why our ACT does not look/our_data_diagnosis.md`; the literature in the three other notes there.

Labels used below: [OURS] measured on our data and checkpoints on 2026-10-27; [REAL-LC] real low-cost arm;
[REAL-other] other real robot; [SIM] simulation; [CLAIM] claim without a measurement; [2nd] secondary source.

## 1. The state is the previous action, and a lookup on it matches our ACT

`m3_episodes.py` records the state as the command sent and the action as the next tick's command, so
`state(t) == action(t-1)` to the last bit in all 154 episodes [OURS]. That is the textbook setting of the
**copycat problem**: an imitator that sees the expert's previous action learns to continue it, and fails exactly where
the action changes ([Wen et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/1b113258af3968aaf3969ca67e744ff8-Abstract.html);
[de Haan et al., NeurIPS 2019](https://arxiv.org/abs/1905.11979)). ALOHA's ACT fed the follower arm's *measured* joints
([ACT paper](https://arxiv.org/html/2304.13705v1)); ours feeds the command itself, with no lag and no noise.

How much of the task the state alone solves, on the same held-out frames NEXT.md used to rank checkpoints [OURS]:

| predictor (m3_pick_20261023b, 8 held-out episodes, 1037 frames) | mean error over the 2 s chunk | J1 one second ahead |
|---|---|---|
| hold still | 13.29 deg | 13.6 deg |
| **10-nearest-neighbour lookup on J1..J6, no image** | **7.79** | 11.2 |
| Friday's ACT, 60k (39 episodes) | 7.79 | 11.4 |
| weekend ACT, 100k (50 episodes) | 6.05 | 8.5 |

Three quarters of the 100k's gain over holding still is available without the camera. The same lookup scores 7.88 deg
on the co-training set and 8.39 on the real 10-26 set (5-fold by episode) [OURS]. The held-out loss we chose
checkpoints by has mostly measured the state shortcut.

## 2. The heading is decided in 2 % of the frames, so the loss gives the camera little reason to learn it

Where the state cannot predict the heading [OURS, co-training set]:

- Frames standing at the look pose before the departure: 363 of 19,344 (**1.9 %**). There the lookup's J1 one second
  ahead is 22.2 deg off; in the first second of motion 7.6 deg -- the departure itself encodes the target. The grip's
  J1 is predicted from the commanded J1 with r^2 0.84 two ticks after leaving and 0.97 after eight.
- From START, the median of what follows in the data (the L1 optimum for a policy that does not look) is J1 90 -> 66 in
  one second (co-training set) and 90 -> 76 in two (real 10-26 set). The real-only 10-23c 20k/40k went to J1 ~62-64 from START
  on fresh frames on 2026-10-26.

An ICLR 2026 study found the same mechanism in general: in the sub-phases where the target must be localised, the
vision+proprioception policy hardly uses vision, because proprioception lowers the loss faster; vision-only beat the
concatenated policy on every task, e.g. real cube 14 vs 11 of 20, thermos 11 vs 7 of 20 [REAL-other], pick-place 91.8 vs
78.4 % [SIM] ([GAP, arXiv 2602.12032](https://arxiv.org/html/2602.12032)). Octo's authors saw proprioception make
policies "generally worse" and suspected causal confusion [CLAIM] ([Octo, App. E](https://arxiv.org/html/2405.12213v1)).

## 3. The baseline recalls its training scenes; on new ones it goes a default way and then follows itself

Counterfactual inputs to the real-only baseline (`m3_act_20261026ft` 40k, its ONNX), heading = J1 two seconds ahead
[OURS]:

| START frames | heading sd (truth) | r with the grip's J1 | slope |
|---|---|---|---|
| its own real training frames (all 72 were trained on) | 27.0 (27.4) | +0.99 | 0.96 |
| the same frames mirrored left-right | 14.0 | does not follow the mirrored cube (corr -0.41) | -- |
| unseen simulated frames (74) | 7.5 (21.4) | +0.21 | 0.08 |

A grey picture sends it to J1 58, a black one to 64 -- the default heading; the robot's 93-98 and the sim grid's 53-81
are that default shifted by the scene's overall look. Half a second into a departure, given the state of one attempt
and the picture of another, its heading follows the state with weight **0.83** and the picture with **0.10** on unseen
frames; shifting the START state by 10 deg shifts the heading by 6-10 deg [OURS]. The method is the cue-conflict test
of [Fang et al., Sept 2026](https://arxiv.org/abs/2609.16815).

Why it memorised: the real-only set is **72 attempts from 13 layouts** (a clean-up gives 3-9 attempts in nearly the
same scene), the co-training set 154 from 42; the baseline saw these images ~140 times (100k steps on 10-23c, then 40k
more); no augmentation was on; LeRobot's `eval_split` holds out the last episodes of each task from layouts it trains on
[OURS]. The same failure, at scale: under moved objects OpenVLA and pi0 fall from 0.82-0.99 to 0.00 on LIBERO and their
trajectories stay "nearly identical" [SIM] ([LIBERO-PRO](https://arxiv.org/html/2510.03827v1)); a 3D diffusion policy's
effective range is "the union of the areas surrounding the demonstrated object placements" [SIM]
([DemoGen](https://arxiv.org/html/2502.16932v1)).

## 4. The CVAE is not the culprit; the ambiguous target and the encoder are second-order

- **CVAE, z = 0:** the posterior has collapsed -- KL 0.000 on every START frame, and decoding with the posterior mean
  gives the same chunk as z = 0 (12.70 / 12.70 deg on sim frames) [OURS]. An independent re-run of ACT's own CVAE
  ablation (Sept 2026) could not reproduce its 35 -> 2 % drop and found "little reconstruction benefit" [SIM]
  ([Kang, arXiv 2609.16745](https://arxiv.org/abs/2609.16745)). Our model is a plain L1 regressor; its "averaging" is
  the L1 median of whatever the input cannot tell apart.
- **Several pieces in training, one in the test:** the picker takes the nearest piece first; a sim START frame shows
  3-5 pieces. An unlearnt rule gives a median heading between them. Plausible but untested; an explicit 2D goal added
  ~14 points in sim ([More Structure, Not More Capacity](https://arxiv.org/abs/2607.09825): 55.0 -> 68.7 %) [SIM].
- **The encoder:** a quick linear probe on 72-74 START frames was underpowered (leave-one-layout-out R^2 <= 0.27)
  [OURS]. The literature says frozen foundation features are not a quick fix: frozen DINOv2 scored 0.00 against 0.90
  fine-tuned on a real pour task ([Lin et al.](https://arxiv.org/html/2410.18647)) [REAL-other]; fine-tuned DINOv3 adds
  up to 10 points over ResNet-18 [SIM] ([arXiv 2509.17684](https://arxiv.org/abs/2509.17684)). ACT keeps the 15x20
  feature grid with 2D position embeddings, so a spatial softmax adds nothing structural to it.
- **A diffusion or flow head:** generative heads do not win through multimodality ([Much Ado About Noising](https://arxiv.org/abs/2512.01809))
  [SIM], and on cheap arms ACT has been the more reliable learner (SO-101: ACT 19/20, DP 0/10 [REAL-LC], from the
  2026-10-26 notes). Not now.

## 5. Removing the state is the largest measured fix; augmentation and layouts are the other two

| fix | measured elsewhere | label | cost on the Orin |
|---|---|---|---|
| no state input | ACT 0 / 0.084 -> 0.933 / 0.517 (height / horizontal); DP 0 / 0 -> 0.867 / 0.533; pi0 height 0/30 -> 30/30 | REAL-other | none |
| noise on the state | 0 -> 0.633 (pi0 height) vs 0.983 with no state | REAL-other | none |
| random proprio masking | above concatenation, mixed against vision-only (86.4 vs 91.8 % pick-place; 89.2 vs 82.6 % assembly) | SIM | none |
| GAP gradient scaling | +2..+6 of 20 real trials over vision-only | REAL-other | none |
| pixel shift / crop | real Can 26.7 -> 73.3 %; -35..-47 % relative without it in sim | REAL-other, SIM | none |
| decision-frame upweighting | CARLA 33.0 -> 43.4 % (keyframes x5) | SIM | none |
| relative (chunk) actions | state-free rel EEF 0.584 vs abs 0 horizontal; UMI 20/20 vs 5/20 | REAL-other | none |
| target drawn into the image | ACT 0.20 -> 0.40 (PEEK, BRIDGE); masks +32 points | REAL-other, SIM | ~0 ms (projection) or 3.0 ms (detector) |
| more distinct layouts | power law in scenes; ~50 demos per scene saturate | REAL-other | none |
| fine-tuned DINOv2/v3 | up to +10 points; frozen 0.00 | SIM, REAL-other | unmeasured on NX |
| diffusion / flow head | no multimodality gain | SIM | tens to hundreds of ms |

Sources: [state-free](https://arxiv.org/html/2509.18644v2), [GAP](https://arxiv.org/html/2602.12032),
[robomimic](https://arxiv.org/html/2108.03298), [keyframes](https://ar5iv.labs.arxiv.org/html/2106.06452),
[PEEK](https://arxiv.org/html/2509.18282v1), [Lin et al.](https://arxiv.org/abs/2410.18647); relative actions and UMI in
`research_notes/Presný tréning VLA pre lacné rameno/policy_design_for_precision.md`. Two caveats from the state-free
study: with absolute joint (or absolute EEF) actions even its state-free policy did not generalise outside the trained
heights and positions, while in-domain every action space scored 1.0; and vision-only policies may be sensitive to the
background [REAL-other]. Our grid lies inside the trained region, so R1 can keep absolute targets; R2 adds relative ones.

What LeRobot 0.6.1 (the newest release, [PyPI](https://pypi.org/project/lerobot/)) gives us: image transforms exist but
are off by default (`--dataset.image_transforms.enable=true`: brightness, contrast, saturation, hue, sharpness and
RandomAffine +-5 deg / +-5 %); ACT's docstring calls the state optional, but `modeling_act.py` reads
`batch[OBS_STATE].device` twice, so a dataset without the state raises -- a 2-line patch in our launcher, or remove the
key from `policy.input_features`; `RelativeActionsProcessorStep` exists but `use_relative_actions` is wired only for
pi0 / pi0.5 / pi0-FAST / GR00T, so ACT needs it inserted by our launcher (read in the installed package).

## 6. The recipe for our robots

| step | what | why | pass criterion | fallback |
|---|---|---|---|---|
| 0. probe (Wed morning, laptop, 30 min) | the six-number probe (`our_data_diagnosis.md` section 10) on the co-trained 150k and ACT v2, on START frames from layouts neither trained on (new sim seeds; mirrored real frames) | v2 has hundreds of new sim layouts; it may look already -- or still copy | heading slope >= 0.7 and r >= 0.8 on unseen layouts; conflict test picture weight b > state weight a; mirrored heading follows (corr >= 0.6) | fails -> step 2 |
| 1. sim grid + M3 (only for a checkpoint that passed 0) | `m3_sim.py evaluate` (18 cells) and the baseline's three spots on the M3 | the probe is offline; the grid and the robot are the real test | grid: slope >= 0.7, median heading error <= 10 deg at 1.5 s, reached (2 cm) >= 12/18 (baseline 5/18); M3: heading within 10 deg at all three spots, closest <= 20 mm at 2 of 3 (baseline 39 / 71 / 18 mm) | go to step 2 |
| 2. R1 "look-first" (Dionysos A4500, ~3.6 h at 150k) | dataset v2; **no state input**; image transforms on (colour defaults, affine <= 3 % / 3 deg); standing frames + first 1 s of each attempt duplicated x5 at build; ~10 % of sim layouts + 2-3 real clean-ups held out whole | removes the copycat input, fights pixel memorisation, gives the decision frames weight | as steps 0 and 1 | jerky chunk starts -> step 4; blind -> step 5 |
| 3. R1b (5090 on hekate, if the user frees it) | R1 with the state kept | isolates the state's share; a P1 hypothesis | same numbers reported, no pass needed | -- |
| 4. R2 relative | R1 + chunk relative to the commanded pose (J6 absolute), stats recomputed in relative space, the state added back inside the ONNX | supervision depends on the cube's place relative to the gripper in every frame; no absolute-pose recall | step 1's, plus first-step J1 error <= holding still's (1.4 deg); a lift at >= 1 of 3 M3 spots | GAP-style gradient scaling with the state instead |
| 5. R3 target mark | R1/R2 + the target drawn into each frame: the pick record's 3D point projected by the commanded pose and the twin's camera (~0 ms), or the detector's mask near the cube (3.0 ms) | resolves "which piece" with several pieces on the table | marked cube among 3 pieces reached >= 80 % in the grid; heads for it 3/3 on the M3 | VLA path (GR00T N1.7 with relative actions, the state handled as in R1) |
| data, from tonight | sim: half the episodes single-piece or re-scattered after each pick; bearing uniform per 15 deg bin; start jitter (J1 +-8, J2-J4 +-4 deg) on half; DART bursts (sigma 2-3 deg, 0.3-1 s, approach only) with the expert's corrective labels; count layouts, aim ~1000 | layouts, not attempts, teach looking; corrections teach recovery from a wrong start | `m3_sim_report.py placement`: flat J1 histogram; >= 25 single-cube START layouts near each grid cell | fewer layouts -> train longer on fewer steps per layout, more augmentation |

The pass thresholds are ours, set against the baseline's measured numbers; none is taken from a paper. Code still to
write (none done in this research): the launcher patch (state removal, relative-action step), the export without the
state or with the add-back, a phase-duplication and a held-out-layout option in `m3_episodes.py build`, start jitter and
DART bursts in `m3_sim.py episode`, and for R3 the projection at build time and in `m3policy.py`.

## 7. Risks

- **State-free with absolute targets may jump at chunk boundaries.** The network must read the arm's pose from the wrist
  picture; the first step of each chunk is already worse than holding still (J1 3.6 vs 1.4 deg, NEXT.md). R2 removes
  the need; the guard and max_dps clamp catch the rest.
- **Augmentation costs millimetres.** A 3 % shift is ~19 px, ~2.3 deg of bearing at START (fx 480.8) [OURS estimate];
  the close look and the 1 s re-plan correct it, the final centimetres do not. Keep shifts small, or crop-and-centre at
  test time as Diffusion Policy does.
- **Vision-only policies react to the background** (the state-free study's limitation). The lab table, tape and box
  are stable; the event tables were not.
- **A target mark is a new shortcut.** The policy may follow the mark rather than the cube; a projected mark carries the
  twin's error (~5 mm rms on the table; tips were 12 mm short before SAG J2 1.1). Use the detector's mask for the last
  centimetres if the projected mark drifts.
- **The sim and the robot disagree on where a blind policy goes** (sim 53-81, robot 93-98): rank checkpoints by the
  offline probe on real held-out layouts as well as by the sim grid.
- **Our probe is small.** 72-74 START frames, one checkpoint; the "unseen" set for the real-only baseline was the sim
  domain, where a domain gap alone could blind it -- the mirrored real frames and the cue-conflict test point the same
  way, but rerun the probe on held-out real layouts for v2.
- **Running Isaac on Dionysos while it trains killed a training once** (NEXT.md, Tuesday): keep the grid evaluation
  after training, or on hekate.

## 8. What this does not settle

- Whether ACT v2 (hundreds of new sim layouts) already looks: unknown until step 0.
- Whether ResNet18 features hold the cube's place well enough: needs a probe on hundreds of single-cube sim frames with
  ground truth (the 72-74 frame probe was inconclusive).
- No published layout density for a wrist-camera 2D policy; the ~25-30 layouts per cell is our estimate from DemoGen
  [SIM] and Lin et al. [REAL-other].
- Mirror augmentation (flip + J1 -> 180 - J1) could double layouts and cancel the left lean; not checked against any
  source (the shared web-search budget ran out), and the M3 is not symmetric (box, right-front lidar).
- DINOv2 latency on the Orin NX: not found by this or the export-formats research; measure before any encoder swap.

## Conclusion

The ACT failed to head for the cube because two shortcuts lowered its loss further than looking did: its input contains
its own previous command, which predicts the next 2 s nearly as well as the whole network did (7.79 deg either way), and
its 13-42 scenes are few enough to memorise, which it did (r 0.99 on training scenes, slope 0.08 on new ones). The
heading is a decision made in 2 % of the frames, and nothing in the training made the camera responsible for it. The
CVAE is not involved. The measured remedies are cheap and keep the 6 ms engine: drop the state input (ACT 0.084 ->
0.517 on a real arm), augment the images, upweight the departure, and train on many more distinct layouts with held-out
layouts for selection. Probe the co-trained 150k and ACT v2 first -- 30 minutes on the laptop -- and only put a
checkpoint on the grid and the M3 when its heading follows the cube on scenes it never saw. Then R1 with its with-state
twin, R2 with relative actions, R3 with the target marked.
