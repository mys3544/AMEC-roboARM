# Why our ACT does not look: measurements on our own data and checkpoints (2026-10-27)

Scope: what our datasets and the real-only baseline checkpoint say about the failure, measured on the laptop on
2026-10-27, read-only, with LeRobot's own venv (`~/.roboarm/lerobot`, LeRobot 0.6.1, onnxruntime on the CPU, the RTX
2070 Super for the torch probes). Datasets: `m3pro/out/lerobot/m3_pick_20261023b`, `..._20261023c`, `m3_pick_20261026`
(the baseline's training set, 72 real attempts) and `m3_cotrain_20261026` (154 attempts: 80 real + 74 sim). Checkpoint:
`train/runs/act/m3_act_20261026ft/checkpoints/040000` (its exported `act_onnx/act.onnx`), the real-only baseline that
heads to J1 93-98 on the M3 and 53-81 in the sim grid wherever the cube is (NEXT.md, Tuesday). The co-trained 150k and
ACT v2 are on Dionysos, not on the laptop, and were not probed. The scripts lived in the session's scratch folder, not in
the repo; section 10 gives the procedure and the core code so they can be rerun on any checkpoint.

Labels: **[OURS]** = measured here on our data; the literature's numbers are in the other notes of this folder with
their own labels ([REAL-LC], [REAL-other], [SIM], [CLAIM], [2nd]).

## 1. observation.state is the previous action, exactly

### Takeaway
In every episode `observation.state(t) == action(t-1)` to the last bit. Our policy is fed its own previous command
as an input -- precisely the "observation history containing the previous action" setting in which Wen et al. (2020)
found behaviour cloning learns to copy instead of to look. ALOHA's ACT fed the follower arm's *measured* joints, which
lag the leader's command; ours has no lag and no noise at all.

### Evidence
- `max |state(t) - action(t-1)|` over all 154 episodes of m3_cotrain_20261026: **0.0 deg** [OURS]. By construction:
  `m3_episodes.frames()` sets the action to the next tick's commanded pose, and the state is the commanded pose.
- The ACT paper's observation is "the current joint positions of follower robots", its action "the absolute joint
  positions" of the leader arms ([ACT, arXiv 2304.13705 HTML](https://arxiv.org/html/2304.13705v1)) -- a measured
  state, not the command.

## 2. A lookup on the six joint numbers scores like Friday's ACT

### Takeaway
A 10-nearest-neighbour lookup that sees no picture at all -- the median of the 2 s chunks that followed the 10 training
frames with the closest J1..J6 -- reaches **7.79 deg** mean error over the chunk on the same 8 held-out episodes where
Friday's 60k ACT scored 7.79 and the weekend's 100k 6.05 (holding still: 13.29). Three quarters of the improvement the
held-out loss rewards is available without the camera. The eval loss we chose checkpoints by mostly measures the
state shortcut.

### Evidence [OURS]
| dataset, split | frames | state-only 10-NN | hold still | ACT (NEXT.md) |
|---|---|---|---|---|
| m3_pick_20261023b, LeRobot's held-out 8 episodes | 1037 | **7.79** | 13.29 | 60k (Friday): 7.79; 100k: 6.05 |
| m3_pick_20261023c, held-out 8 | 990 | 8.56 | 12.86 | -- |
| m3_pick_20261026 (real), 5-fold by episode | 9545 | 8.39 | 13.57 | -- |
| m3_cotrain_20261026, 5-fold by episode | 19344 | 7.88 | 14.92 | -- |

- J1 one second ahead on m3_pick_20261023b's held-out frames: lookup 11.2 deg, Friday's 60k 11.4, the 100k 8.5,
  holding still 13.6 (mean error over the chunk, deg, all six joints; the ACT numbers from NEXT.md, 2026-10-26).
- (13.29 - 7.79) / (13.29 - 6.05) = 0.76 of the 100k's gain over holding still.

## 3. The heading is decided in about 2 % of the frames

### Takeaway
The only frames where the state cannot tell which way J1 will go are the few standing at the look pose before the
departure: 363 of 19,344 frames (1.9 %) in the co-training set. There the lookup's J1 one second ahead is 22 deg off;
from the first tick of motion it is 7.6 deg off, because the departure direction already encodes the target. The L1
loss therefore asks the image for the heading in ~2 % of its samples, and the median of what follows START in the data
is a fixed heading: J1 ~66 (co-training set) or ~76-80 (real 10-26 set) -- the "default way out" a blind policy takes.

### Evidence [OURS]
- Frames before the first move (> 0.5 deg on any joint): median 3 an episode, 363 of 19,344 (1.9 %), co-training set;
  261 of 9545 (2.7 %) in m3_pick_20261026. Episodes run 89-290 frames (median 121).
- Linear fit of the grip's J1 from the commanded J1 k ticks after the departure (154 episodes): r^2 0.62 at k=0, 0.76
  at 1, 0.84 at 2, 0.92 at 5, 0.97 at 8; median |J1 - grip J1| 15, 13, 10, 2.5, 1.0 deg.
- State-only lookup, J1 one second ahead, by phase (co-training set, 5-fold): standing at START 22.2 deg; first second
  of the departure 7.6; 1-3 s 7.8; later (close look, grasp, carry, drop, return) 10.5.
- From START exactly, the 30 nearest training frames' J1 path (median, the L1 optimum): 90 -> 66 deg in 1 s (co-training
  set); 90 -> 76 in 2 s (real 10-26 set). The real-only 20k/40k on 2026-10-26 went to J1 ~62-64 from START on fresh
  frames, close to the first figure (NEXT.md, Monday item 2).
- Wen et al.'s keyframe rule (upweight the samples a history-only predictor gets most wrong) does **not** pick these
  frames out on our data: of the 10 % of frames with the largest J1 copycat error, 7.7 % are the standing frames and
  82 % come after the grasp (when the carry to the box starts is a timing question). The departure has to be
  upweighted by phase, not by copycat error (see `fixes_with_effect_sizes.md` section 7).

## 4. The baseline recalls its training scenes and is blind on new ones

### Takeaway
On the START frames it was trained on, the real-only baseline's heading two seconds ahead matches the demonstration
almost exactly (r 0.99). On the same frames mirrored left-right -- the same pieces, a scene it has not seen -- its
output shrinks towards the middle and does not follow the mirrored cube. On simulated START frames it never saw, the
heading barely depends on where the target is (slope 0.08). It has learnt "which of my training scenes is this", not
"where is the cube".

### Evidence [OURS] (ONNX of m3_act_20261026ft/040000, the state as recorded, J1 of the chunk's last step = +2 s)
| START frames | n | heading sd (truth sd) | r with grip J1 | slope | median error |
|---|---|---|---|---|---|
| own real training frames (eval_split 0.0: all 72 trained on) | 72 | 27.0 (27.4) | +0.99 | 0.96 | ~1 deg (1.4 / 0.7 on two subsets of 62 / 10) |
| the same frames mirrored left-right | 72 | 14.0 | +0.40 (vs the unmirrored target) | 0.20 | -- |
| unseen simulated frames (74 sim attempts) | 74 | 7.5 (21.4) | +0.21 | 0.08 | -- |

- Mirrored real frames: correlation of the mirrored heading with the mirrored target (180 - own heading) **-0.41**: it
  does not turn the other way when the scene does. On sim frames the same correlation is -0.64.
- A grey image at START: J1 59 at +1 s, 58 at +2 s; black: 65 / 64. The sim grid's 53-81 and the robot's 93-98 are
  this "default heading" shifted by the scene's overall look.
- The baseline was fine-tuned for 40k steps (batch 8: ~34 epochs over 9545 frames) from the 10-23c 100k
  (~107 epochs over 7481 frames, mostly the same real images): ~140 passes over ~13 layouts.

## 5. Once moving, it steers by its own last command

### Takeaway
Shifting the START state by 10 deg of J1 shifts the predicted heading by 6 deg (real frames) to 10 deg (sim frames).
Half a second into a departure, with the state of one attempt and the picture of another, the heading follows the
state's target with weight 0.83 and the picture's with 0.10 on unseen (sim) frames. This is the copycat loop: whatever
way the first chunk goes, the state then says "this is a trajectory towards there" and the policy continues it.

### Evidence [OURS]
- START state J1 -10 / +10 deg, own image: heading moves -5.8 / +6.0 deg on real frames (sd 1.6-1.7), -9.7 / +10.3 on
  sim frames (sd 0.8-1.0).
- Cue conflict 0.5 s after the first move (5 ticks), heading at +2 s regressed on the two attempts' grip J1: sim
  frames a (state) **+0.83**, b (picture) **+0.10** (440 pairs); real training frames a +0.60, b +0.40 (421 pairs --
  the picture there is a memorised key, section 4). With matched inputs the heading at that moment correlates 0.96
  (sim) / 1.00 (real) with the grip J1, and so does the state alone (0.99 / 0.97).
- The method is the cue-conflict intervention of Fang et al. (Sept 2026), who pair the task state of one stage with
  the robot configuration of another and find ACT, RISE, Diffusion Policy and BC-RNN following the configuration in
  the approach phases ([arXiv 2609.16815](https://arxiv.org/abs/2609.16815); numbers only in a figure).

## 6. The CVAE latent is unused: z = 0 changes nothing

### Takeaway
The CVAE encoder's posterior has collapsed onto the prior: KL 0.000 on every START frame, and decoding with the
posterior mean (given the demonstrated chunk) gives the same chunk as z = 0. "ACT averages because it runs its latent at
zero" is therefore not a separate cause in our checkpoint -- the model is a plain L1 regressor, and L1 returns the
median of whatever the input cannot tell apart.

### Evidence [OURS]
- m3_act_20261026ft/040000, START frames: KL per frame (sum over 32 dims) median **0.000** on real and sim frames;
  mean chunk error z = 0 vs z = posterior mean: 12.70 / 12.70 deg (sim), 1.26 / 1.26 deg (real training frames).
- Consistent with a 2026 re-run of ACT's CVAE ablation that could not reproduce the published 35 % -> 2 % drop and found
  "little reconstruction benefit" from the latent ([Kang, arXiv 2609.16745](https://arxiv.org/abs/2609.16745)) [SIM].

## 7. 154 attempts are 42 scenes

### Takeaway
A clean-up gives one recording and 3-9 attempts in nearly the same scene (one piece fewer each time). The real-only
set is **72 attempts from 13 layouts**, the co-training set 154 from 42. The image-to-target mapping has to be learnt
from a few dozen distinct tables, which a 52M model memorises. LeRobot's eval_split holds out the last episodes of each
task -- from layouts it trains on -- so the held-out loss flatters memorisation.

### Evidence [OURS]
- Distinct recordings (layouts): m3_pick_20261023c 58 attempts / 10; m3_pick_20261026 72 / 13; m3_cotrain_20261026
  154 / 42; attempts per layout median 5, 5, 3, max 9.
- LeRobot 0.6.1 `datasets/factory.py`: "The last ceil(n_episodes * eval_split) episodes per task are held out".
- Grip J1 bins (<40, 40-55, 55-70, 70-85, 85-100, 100-115, >115) -- real 8/13/11/15/13/11/9, sim 1/10/15/19/17/12/0:
  the 18-cell grid's J1 40-100 holds ~110 attempts, so coverage of the evaluated region is not thin in attempts; it is
  thin in layouts.

## 8. The state also tells sim from real

### Takeaway
All 72 real episodes start exactly at START (90/105/10/0/90/76); only 1 of 74 simulated ones does, because the robot
card's joint zeros are compensated into the recorded commands (sim starts J1 85-94, J6 69-78). A policy that takes the
state can tell the domain from it, which works against co-training. The sim evaluation starts at START exactly -- the
real data's signature.

### Evidence [OURS]
- First-frame states, co-training set: 81 of 154 within +-1 deg of START on all joints (all real ones plus 1 sim);
  sim first-frame J1 85-94, J6 69-78.

## 9. A linear probe on 72-74 frames could not tell whether the features hold the cube's place

### Takeaway
Inconclusive, and a warning about splits: ridge regression from ResNet18 layer-4 features (pooled 5x5) of the START
frame to the grip's J1 looked perfect with leave-one-frame-out (R^2 0.99) and failed with leave-one-layout-out
(R^2 <= 0.27). The first number leaked through near-identical frames of the same layout. A useful probe needs hundreds of
single-cube frames with ground truth -- the sim grid's.

### Evidence [OURS]
- Leave-one-recording-out (27 sim layouts / 13 real): ImageNet ResNet18 R^2 +0.27 (sim), -0.05..+0.01 (real); the
  checkpoint's own fine-tuned backbone -12.6..+0.07 (sim), -1.2..+0.07 (real); shuffled labels -0.2..-8.9.

## 10. The 30-minute diagnostic (any checkpoint, laptop, no robot)

### Takeaway
Six numbers separate "looks", "memorises", "copies the state" and "dead latent". Run them on START frames from layouts
the checkpoint did not train on, before any sim-grid or robot time. A checkpoint that fails them will not head for the
cube on the robot either.

### Procedure (what was run above; ~2 min per checkpoint on the CPU for 74 frames, 30 min including setup)
1. **Frames**: for each attempt in a set of held-out layouts, the START frame (`ticks[0]` of `m3_attempts.jsonl`) and
   the frame 5 ticks after the first move; the grip's J1 from the pick record as the target.
2. **Heading test**: the chunk's J1 at +2 s with the own image and state; report sd, r and slope against the grip J1.
   Pass: slope >= 0.7 and r >= 0.8 on unseen layouts (baseline: 0.08 / 0.21 on sim).
3. **Blank / mirror**: grey and black image at START (the default heading); the image flipped left-right: pass when
   corr(mirrored heading, 180 - own heading) >= 0.6 (baseline: -0.41 real, -0.64 sim). The M3 is nearly symmetric
   about J1 90 at the look pose; the box and the right lidar are not, so treat this as a coarse test.
4. **State sensitivity**: START +-10 deg on J1, own image: the heading should move < 3 deg (baseline 6-10).
5. **Cue conflict** at +5 ticks: state of attempt i, picture of attempt j, regress the heading on both targets:
   pass when b (picture) >= 0.6 and > a (state) (baseline sim: a 0.83, b 0.10).
6. **Latent**: KL of the CVAE posterior on START frames; if ~0, drop the CVAE question (`use_vae=false` changes nothing
   but the training time).

Core of the counterfactual probe (onnxruntime on the exported `act.onnx`: inputs `image` 1x3x480x640 RGB 0..1 and
`state` 1x6 degrees, output `chunk` 1x20x6):

```python
sess = ort.InferenceSession("act.onnx", providers=["CPUExecutionProvider"])
run = lambda img, st: sess.run(None, {"image": img, "state": st[None]})[0][0]   # 20 x 6, degrees
img = lambda p: np.asarray(Image.open(p).convert("RGB"), np.float32).transpose(2, 0, 1)[None] / 255
heading = [run(img(f), s0)[19, 0] for f, s0 in start_frames]            # J1 at +2 s
mirror  = [run(img(f)[..., ::-1].copy(), s0)[19, 0] for f, s0 in start_frames]
conflict = [(grip[i], grip[j], run(img(mid[j]), mid_state[i])[19, 0]) for i, j in pairs]
a, b, c = np.linalg.lstsq(np.c_[X_i, X_j, np.ones(len(Y))], Y, rcond=None)[0]   # follows state / picture
```
