# Next -- the list (as of 2026-10-23, Friday evening)

## Friday 2026-10-23: back in the lab after the event

- [x] **Both robots back on IISLab-AMEC** (booted there: the M3 keeps the network it starts on), found
      by name, bridges and the M3's detector up, batteries 11.4 / 11.6 V, settings as before the event.
- [x] **A check clean-up on each:** the M3 **5 of 5, 0 missed**, all into the box (4 cubes out on its
      left out of reach). The X3 **5 picked, 2 missed**, every drop on the line out on the right: **its
      box was not seen** ("marker 10 not seen all round") -- the box nearer the arm (centre <= ~28 cm
      out), marker to the arm. Its fingertip lands 8.1 mm short (median of 7 reach passes; 8.8 mm on
      2026-10-21, 67 of them) and -5 mm left (1.1 then): the trip did not move its camera much.
      Its log has no pick on 2026-10-22 -- the X3 not shown, or its clock wrong that day?
      **The X3's box moved nearer** (found at J1 180, x 0.116 y -0.248, 27 cm from the base): **8 of 8,
      0 missed, all into the box**, 8-11 s a pick (22-24 s with a survey); the fingertip still
      5-12 mm short by its reach passes, the grips good. Left: a block lying 25.4 cm out (past the X3's
      reach) -- the job's last line then says "no kit cube or block in view (1 left ...)".
- [x] **The M3's joints: none come back.** The board publishes no servo angle (`ros2 topic list`:
      /arm6_joints and /arm_joint in only); Yahboom's /Curjoints is their apps' own echo of what they
      sent. For the VLA the state is the last command (relative actions, as reports/ suggested), and
      the camera is the only judge of where the arm is -- hence the probe below.
- [x] **The M3's servos measured by its camera** (`m3pro/tools/m3_servo_probe.py`, run on the robot;
      the ChArUco board flat in the pick zone, seen from J1 88, J2 105-115, J3 0-25, J4 0; one joint
      turning turns the camera by its angle, so J1-J4 each alone; the PnP scale checked: board 312 mm
      under the camera against the depth's 309). Results in m3pro/out/servo/ (laptop and robot):

      | joint | backlash | repeat sd | gain, static (+-4..12 deg) | turned of a 15 deg sweep | lag | there after the commanded end |
      |---|---|---|---|---|---|---|
      | J1 | 0.36-0.38 deg (1.6 mm at 25 cm) | 0.01-0.02 deg | 0.91-0.94 | 13.1-13.3 (0.88) | 100-130 ms | 0.12-0.22 s, overshoot <= 0.26 deg |
      | J2 | 0.47 deg (2.0 mm) | 0.09 deg | 0.92 | 13.0-13.1 (0.87) | 125-195 ms | 0.14-0.83 s, overshoot 0.4-0.9 deg (faster more) |
      | J3 | 0.86-1.09 deg (3.7-4.8 mm) | 0.03-0.17 deg | 1.06 at J3 4, 1.15 at 25 | 13.8-14.2 (0.93) | 105-190 ms | 0.1-1.5 s (slow at 60-90 deg/s) |
      | J4 | 0.57 deg (2.5 mm) | 0.05 deg | 0.91 | 7.1-7.9 of 8 | 140-160 ms | 0.1-0.3 s, overshoot <= 0.15 deg |

      What it means: **the joints turn ~10 % less than commanded** (J3's small-swing gain is pose
      dependent: gravity), backlash 0.4-1.1 deg, repeatable to ~0.1 deg, ~0.1-0.2 s behind the
      command (the camera's own latency in it: its stamps wander -25..+108 ms against the robot's
      clock), J2/J3 ringing up to ~1 s after fast moves. **For the simulator:** position drives
      ~0.15 s behind, gains 0.85-1.0 and backlash 0.3-1.2 deg randomised per episode. **For the
      picker** (not done): J1's 0.9 alone puts a piece 30 deg off straight ahead ~1 cm sideways at
      20 cm -- part of what the taught aim corrects; a per-joint gain in m3arm needs each joint's
      true zero too (measured against something fixed, not by a turn). J5/J6 are past the camera.
- [x] **Every M3 job kept as an episode** (`m3pro/roboarm/m3record.py`, in the bridge, deployed):
      10 times a second while a job runs the colour picture, the pose sent, the odometry and the
      step, under the robot's ~/roboarm/out/episodes/<stamp>_<job>/ (ticks.jsonl, episode.json with
      the log and each pick's record; the stamps in the robot's clock, Asia/Shanghai). A look job:
      75 ticks in 7.8 s, every one with its picture, 5.2 MB (the oldest go past 20 GB). **The first
      recorded clean-up** (20261023_155503_clean_up): 7 picked, 0 missed, 1152 ticks in 122 s (dt
      median 0.103 s, p99 0.149, max 0.37), every tick a picture (median 18 ms old), 50 MB, the 7
      pick records in episode.json; the steps cut it into attempts (look.wide -> pregrasp ->
      look.close -> descend -> close -> lift -> drop -> look.after), the pictures right for each
      (a contact sheet checked by eye). Into LeRobot: below.
- [x] **Why that clean-up left a cube and the red block** (the user): both just past the arm's reach --
      J1's axis is 3.3 cm ahead of the robot's middle, and at the grip's height (18 mm) the arm reaches
      23.2 cm from it; the cube (x 0.187 y +0.184) was 24.0 cm out, the block (x 0.267 y -0.072) 24.4.
      The cube was said to be "too close to the robot's body": its only pose, a grip at its very top at
      pitch 50, had the arm 16 cm over the chassis's front-left corner (18 needed). Now why_not says
      **"at the edge of the arm's reach ... -- the robot N cm closer would reach it"** for any piece
      whose middle is past the reach, either side (m3pick.closer_for: the least drive forward with a
      plan; the cube 3 cm, the block 1 cm), "(may drive closer lets the base drive up)" when driving is
      off. So: **M3 pieces within ~22 cm of J1's axis** -- ~25 cm ahead of the robot's middle straight
      out, less to the sides -- or "may drive closer" on.
- [x] **The recordings as a LeRobot dataset** (`m3pro/tools/m3_episodes.py`): `fetch` copies the
      robot's finished episodes to m3pro/out/episodes/ (scp, the laptop's key); `build OUT`, run with
      LeRobot's own python (`~/.roboarm/lerobot`: uv venv, `lerobot[dataset]==0.6.1`, CPU torch 2.11,
      outside the workspace), makes one episode per pick attempt -- from its pregrasp to the end of
      the look after it, matched in order to the job's pick records (a job where they do not match is
      left out, not guessed); the ticks taken at 10 fps by the nearest; observation.images.wrist (AV1
      video, 640 x 480), observation.state = J1..J6 as sent (servo degrees), action = the next frame's
      state, task "pick up the 30 mm cube and put it in the box" (the drop from the log: box or line);
      missed attempts out unless --keep-failed; OUT/m3_attempts.jsonl keeps each one's pick record.
      The recorder keeps "started_unix" since (deployed): the log's times line up with the ticks;
      before, the robot's local time (Asia/Shanghai) stands in. **Today's clean-up:** 7 episodes, 939
      frames (12-18 s each), 7.3 MB from 50 MB of JPEGs (1 min 20 s on the laptop's CPU); loaded back
      through LeRobotDataset: action = next state exactly, the pictures 0.7 % off the JPEGs (mean abs),
      the tasks right. On Windows torchcodec does not load (its FFmpeg DLLs): LeRobot falls back to PyAV.
- [x] **Five rounds with the 40 mm cube on the M3** (the user setting the table, wood up twice, blue,
      yellow, green in a sort): 34 attempts, 32 picked, every one recorded (rounds 1-4 clean-ups 8/9,
      9/10, 5/5, 5/5; the sort 5/5, 2 blocks onto the line). The served detector missed the cube close
      up wood up and painted alike, and called it a block blue up. **A labelling fault found on the
      way** (kit_dataset.not_kit): anything yellow was dropped as the yellow cylinder -- the 40 mm cube
      yellow up and a 30 mm cube with a yellow top too; now only a round yellow blob is the cylinder
      (yellow_square: 4 corners, filling its rectangle; 18 labels back, the M3 test set unchanged).
      The LeRobot set rebuilt from all of today's: **39 episodes, 5473 frames, 42 MB** (25 x "the 30 mm
      cube", 6 x the 40 mm cube, 6 x the block into the box, 2 x the block onto the line).
- [x] **The detector fine-tuned on today's looks** (M3 set 268 train / 63 val, 40 epochs on the
      laptop's GPU, ~7 min each), judged on the M3's fixed test set (40 looks, 25 with the 40 mm cube;
      "found" = cube40 at conf 0.25 with box IoU 0.5):

      | model | mask mAP50 all / wide / close | cube40 mAP50 | 40 mm cube found wide, close |
      |---|---|---|---|
      | `lap_20261016_m3` (served until today) | 0.840 / 0.951 / 0.729 | 0.845 | 13/13, 8/12 |
      | `dio_20261019_mix` | 0.861 / 0.979 / 0.760 | 0.888 | 11/13, 10/12 |
      | `lap_20261023_m3` (from n_20261015b) | 0.855 / 0.972 / 0.737 | 0.887 | 13/13, 10/12 |
      | **`lap_20261023_mixm3`** (from dio_20261019_mix) | **0.865 / 0.980 / 0.754** | **0.915** | 12/13, **11/12** |

      **The M3 serves `lap_20261023_mixm3`** (2026-10-23): ~/roboarm/models/kit/yolo26n_kit.pt and its
      FP16 engine built there (7.5 min); the one before kept as yolo26n_kit_20261016m3.{pt,onnx,engine}
      (back: copy those over yolo26n_kit.*, `systemctl --user restart roboarm-vision`). Worse only on
      blocks (0.839 against 0.886). Its engine on two of today's looks: the 40 mm cube wood up 0.85,
      yellow up 0.67 (seen in training: a check that it runs, no more). Still the second opinion: the
      depth decides. Judge it live on the next clean-ups ("kit detector:" lines).
- [x] **The new detector live** (a clean-up, 5 of 5, the 40 mm cube wood up in the middle): it saw the
      cube wide (0.83) and close (0.81). The log's "it missed cube40" that round was the depth's own
      "40 mm cube" at the picture's right edge -- the box's rim -- rightly not called: the "agrees on"
      count undersells it.
- [x] **The M3's gripper measured** (the user, a ruler between the rubber fingertips' inner faces ~1 cm
      from their ends; also as m3pro/data/m3_gripper_gap.json, on the laptop only: data/ is not in git):

      | J6 | 0 | 30 | 60 | 90 | 120 | 150 | 165 | 170 |
      |---|---|---|---|---|---|---|---|---|
      | gap, mm | 61.5 | 61.5 | 56.5 | 49 | 33 | 15 | 4 | 0 (touching) |
      | Yahboom's table | ~67* | ~67* | 59 | 51 | 37 | 21 | 11 | 7 |

      (* m3kinematics.finger_span's carried-on line.) Open stop from J6 0 to 30; ~2 mm play between
      the fingers (at 150: 15 free, 13 squeezed). Narrower than the table everywhere: the picker's
      close (the table's J6 for the width) goes **~5 mm past touch on a 30 mm piece** (J6 134: ~25 mm)
      and ~4 on the 40 mm cube (J6 115: ~36 mm) -- a firm grip, the servo pushing; the widest is 61.5
      mm, not ~67 (the clearance checks think the fingers wider: on the safe side). Nothing changed in
      the picker (it grips well); for the simulator, the measured table.
- [x] The event itself (2026-10-22): it went well (the user, 2026-10-23).
- [ ] **ACT on the 39 real episodes, the laptop's GPU** (2026-10-23, a test that our data trains -- not
      a policy for the robot yet): LeRobot's venv got CUDA torch (2.11+cu128 from uv's cache) and
      `lerobot[training]`; the RTX 2070 Super runs ACT (52M, resnet18) at ~7.8 steps/s, batch 8, AMP;
      samples come at ~18 ms each (AV1 through PyAV: torchcodec's DLLs do not load on Windows).
      lerobot-train's checkpoints/last symlink needs Developer Mode on Windows (WinError 1314):
      `train/lerobot_train_win.py` writes checkpoints/last.txt instead. The run: chunk 20 (2 s), 10
      executed, 60k steps, 10 % of the episodes held out (eval loss every 2000), checkpoints every
      10k -> train/runs/act/m3_act_20261023 (+ .log). Smoke run: eval loss 0.646 -> 0.584 in 200 steps.
      **Moved to Dionysos at 20k** (the user: as soon as possible): C:\Users\Hercules\roboarm-lerobot (uv venv,
      torch 2.11+cu128, lerobot[training] 0.6.1; the same launcher), resumed from the laptop's 20k checkpoint
      (its train_config.json's paths changed; workers 6, eval every 5000) at ~10.4 steps/s against the
      laptop's ~6; over an SSH session held open from the laptop. Eval loss 0.297 (14k), 0.291 (20k), 0.287 (25k).
      **The 20k checkpoint on the 6 held-out episodes** (`m3pro/tools/m3_policy.py eval`; mean |error|, deg,
      J1-J6): a second ahead 11.3 / 10.7 / 10.3 / 8.1 / 7.9 / 6.1 against holding still's 12.4 / 25.7 / 16.3 /
      15.0 / 5.3 / 10.0 (the whole 2 s chunk 8.6 against 13.6) -- it has learnt the motion, coarsely: 10 deg
      of J1 is ~4 cm at 22 cm. **The robot client** (`m3_policy.py run CHECKPOINT [--max-dps 40] [--dry]`):
      from the episodes' start (the wide look, J6 76), 10 Hz, the bridge's MJPEG stream (/rgbd took 57-532 ms
      over the WiFi), the bridge's guard on every pose (floor 8 mm), steps capped at max-dps/10, J6 <= 150,
      STOP by Space/Esc, the console's STOP or Ctrl-C. Dry run against the M3: 9.4 steps/s, the picture 10 ms
      old (median), inference ~2 ms a step (a new chunk every 10). Not yet run moving.

- [x] **ACT on the real M3** (the 30k checkpoint, `m3_policy.py run`, 2026-10-23 afternoon, a 30 mm cube in
      the middle, the box on the right): two runs, 25 and 40 s. Both went through the whole pick's shape --
      down to the table, the gripper shut to J6 ~134 (the picker's value for 30 mm), up, round to J1 ~170-180
      (the box), down towards it -- and **missed the cube** both times (the user). The loop ran at ~5.5-6.5
      steps/s, not 10 (each move waits for the bridge over the WiFi; some steps 1.9 s). Run 2 ended on an
      HTTP 400 from the bridge at J1 ~166 that the client then did not read (now it does: the reason is said).
      Held-out error over the 2 s chunk by checkpoint: 20k 8.61, 30k 8.28, 50k 8.01, **60k (the end) 7.85 deg**
      (holding still 13.63); a second ahead at 60k J1-J6 10.3 / 10.0 / 9.7 / 7.7 / 7.4 / 4.8; eval loss 0.267,
      flattening. The checkpoints on the laptop: train/runs/act/m3_act_20261023{,_dio}/.
      A third run with the 50k (14:27): turned to J1 ~110, right of the cube in the middle, and was stopped
      after 8.6 s by the bridge's guard -- "the arm would reach 0.8 cm above the floor (x +0.19 m)": it went
      lower than the client's 8 mm floor (the picker's first try; it goes to 5 mm where nothing else plans).
- [x] **Three more rounds** (the user's new layouts): A 3/3 (pieces turned, on the left), B 8 counted (cubes
      4-5 cm apart, a block standing) -- **one of them shoved 4.3 cm out of reach and counted "picked"**: the
      picker looks only where the piece was (MATCH_M 3 cm) and its look after saw only the middle; C a sort,
      4/4 (2 blocks onto the line). m3_episodes now marks such an attempt a miss (a piece of the kind
      turning up within 8 cm of the target where no look before had one): the only one of the day.
      The dataset rebuilt from all of the day's: m3pro/out/lerobot/m3_pick_20261023b, 58 episodes, 7955 frames
      (33 the 30 mm cube, 10 the 40 mm, 11 a block into the box, 4 a block onto the line).
- [x] **The weekend's ACT run on Dionysos** (detached: weekend_act.ps1 creates it through WMI, so no SSH
      session holds it): the new dataset, 100k steps, batch 8, eval every 5k, checkpoints every 20k ->
      C:\Users\Hercules\roboarm-lerobot\m3_act_20261023b (+ .log there). Started 14:19: 50 episodes to train,
      8 held out; ~10 steps/s alone on the GPU, ~3 h. The day's run (39 episodes) had eval loss 0.275 at 45k.
      **Finished** (2 h 48 min, ~11 steps/s): eval loss 0.275 (5k), 0.227 (30k), 0.218 (60k), 0.211 (75k),
      **0.210 (100k)** -- flat from ~75k. The 100k's pretrained_model and the log on the laptop:
      train/runs/act/m3_act_20261023b/checkpoints/100000, train/runs/act/m3_act_20261023b.log.
      **Judged (2026-10-26) on its 8 held-out episodes** -- all from the evening's rounds, none seen by Friday's
      run either, so a fair duel (`m3_policy.py eval CKPT --dataset out/lerobot/m3_pick_20261023b`, 1037 frames):

      | checkpoint | whole 2 s chunk | 1 s ahead J1-J6 (deg) | next step J1-J6 |
      |---|---|---|---|
      | hold still | 13.29 | 13.6 / 26.0 / 13.3 / 15.7 / 4.8 / 9.9 | 1.4 / 2.7 / 1.6 / 1.8 / 0.4 / 0.9 |
      | Friday's 60k (39 episodes) | 7.79 | 11.4 / 9.6 / 8.4 / 7.5 / 7.4 / 4.0 | 5.8 / 3.3 / 2.8 / 3.3 / 3.9 / 2.0 |
      | **weekend 100k (50 episodes)** | **6.05** | **8.5 / 6.9 / 7.4 / 5.9 / 5.7 / 3.1** | 3.6 / 2.9 / 2.2 / 2.3 / 1.8 / 1.5 |

      Better on every joint (J1 a second ahead 11.4 -> 8.5 deg: ~3.3 cm at 22 cm, still coarse). Its first step
      is still worse than holding still (J1 3.6 against 1.4): the first move of each chunk jumps.

## Monday 2026-10-26

1. hekate (back Friday evening): its status script (CLAUDE.md) -- reachable, Isaac Sim's task, free VRAM.
   **Back, at a new address: 192.168.73.11** (DHCP; .88 timed out, the user gave the new one; ~/.ssh/config
   and CLAUDE.md say so). Status 2026-10-26 09:45: up 36 min, the 5090 idle (31.6 of 32.6 GB free), no Isaac
   Sim or Python running (the log's "Loaded: yes" is from 10-12), screen locked; the script says no one is
   logged on, yet AMEC-PC's apps run -- the scheduled tasks need a logged-on user, check before
   `schtasks /run /tn IsaacSimStreaming`. Its Remote Control session "hekate" is offline.
2. The weekend's ACT: `m3_policy.py eval` on its held-out episodes against today's 60k (**done**, above: 100k
   6.05 deg against 7.79); then on the M3
   again -- first the loop at a true 10 Hz (send without waiting for each move: the bridge's move_to
   blocks; or run the policy beside the bridge on the robot).
   **The loop on the robot (2026-10-26, the user chose it over sending without waiting from the laptop):**
   - the bridge's policy job, `roboarm/m3policy.py` (POST /pick/policy {seconds, max_dps, dry}; a Picker job,
     so the console's STOP, busy, the log and the recorder work as for a pick): a fixed 10 Hz clock, the newest
     colour picture (<= 0.5 s old) and the believed pose -- as the recorder kept them --, the service asked
     once a chunk (10 of 20), each step held to max_dps / 10 a joint (J6 <= 150) and sent as one command over
     0.1 s NOT waited for (`M3Arm.step_to`, the guard on every pose: 1.8 ms on the Orin); STOP, a refusal or a
     stale picture ends it, the arm holds; a line a tick in out/policy/run_STAMP.jsonl;
   - the model: `tools/policy_service.py` on 127.0.0.1:8763 (.venv-ml, unit roboarm-m3policy, installed, not
     enabled), ACT as ONNX -> TensorRT FP16 built there (139 s), checked against LeRobot's chunks on 4 held-out
     frames: **max 0.19 deg, 5.9 ms an inference**. TensorRT 10.7 could not build the graph as exported
     (Myelin "should never be called on mod_t tensors"): `export` slims it (onnxslim) and then it builds;
   - from the laptop: `m3_policy.py export CKPT` (ONNX with the normalisation inside; ONNX Runtime = LeRobot
     to 3e-5 deg) and `m3_policy.py robot CKPT [--seconds 30] [--max-dps 100] [--dry]` (exports and uploads
     when new, builds and checks there, starts the service, runs the job, shows its log, Space/Esc/Ctrl-C =
     STOP, the run's log back into m3pro/out/policy/). The LeRobot venv got onnx, onnxruntime, onnxslim;
   - **max_dps 100 by default: the recordings' own speed.** The picker moves J1-J5 at up to 100 deg/s (10 deg
     a tick; 90th percentile J1 70, J2 90 deg/s) and J6 160: Friday's 40 deg/s cap ran the policy at under
     half its training speed in the fast parts -- a second mismatch beside the 5 Hz;
   - **dry run on the M3, 15 s: 150 ticks, 10.0 a second, 0 late** (ticks 0.099-0.101 s apart), the service
     answered in 12 ms (median, JPEG + HTTP included), the pictures 47 ms old (median, max 91). Deployed by scp
     (the robot's copies matched git HEAD first), the bridge restarted.
   - **The first moving run (100k, 30 s, 100 deg/s; a 30 mm cube green up in the middle, the box on the right,
     pliers top left -- 17:35):** the loop perfect (300 ticks, 10.0 a second, 0 late, the policy 12 ms), but
     **the arm never left START**: wanted START +-1 deg the whole run. Not the engine or the loop: LeRobot's
     own PyTorch on today's frames (from the run's recorded episode, 20261026_173550_policy) from START moves
     0.6 deg in a second with the 100k, 1.0-1.2 with Friday's 50k/60k, 3.8 with Friday's 30k -- while on the
     recordings' own first frames every checkpoint swings 57-72 deg towards the piece (memorised).
     **The cause is in the episodes:** of the 743 training frames at START (<= 1 deg) only 25 move within a
     second, 718 stay -- each episode ends with the whole look after the drop (6-64 frames standing at START,
     m3_episodes cuts "to the end of the look after") and starts already on its way (0-1 frames at START). L1
     learns the median: at START, stay. On Friday the less trained 30k still crept off it.
     **Fixed in m3_episodes (2026-10-26):** an episode now runs from LEAD = 3 ticks before its pregrasp (only
     ticks the arm stood still in, after the episode before) to its arrival back at the look pose (the look
     after's first tick there); the pick checks (drop, pushed away) still use the whole attempt. Rebuilt as
     **m3pro/out/lerobot/m3_pick_20261023c**: the same 58 attempts in the same order, 7481 frames (7955), at
     START now **223 move within a second, 39 stay** (33 of them an episode's last frame) -- was 25 / 718.
     **Training on Dionysos** (started 2026-10-26 ~10:53 its clock, detached: act_20261023c.ps1 -> .cmd, the
     weekend's settings: 100k steps, batch 8, eval every 5k, checkpoints every 20k, 50 train / 8 held out) ->
     C:\Users\Hercules\roboarm-lerobot\m3_act_20261023c (+ .log there). Then: each checkpoint from START on
     today's frames (m3pro/out/policy/frames/000072.jpg, 000150.jpg: a fresh scene), then on the M3.
     Eval loss 0.274 (10k), 0.264 (15k), 0.261 (20k) -- not comparable with b's (its easy standing frames gone).
     **The 20k leaves START:** on today's frames 26 deg in its first second (J1 90 -> 64, J2 105 -> 80, J4 0 ->
     20), where b's 100k moved 0.6 -- the cut was it. But it heads ~30 deg left of the cube (the clean-up
     gripped it at J1 94): the recordings' mean way out, or the pliers on the left -- to be seen on the M3,
     where it looks again every chunk.
     **The 20k on the M3** (18:31, pliers off the table, the 30 mm cube in the middle; its engine 0.63 deg from
     LeRobot, 6.6 ms): it left START at once and kept turning left and down -- J1 90 -> 69 -> 56 -> 47 -> 37, J2
     105 -> 56 -- until the guard refused it after 3.0 s ("16 cm above the floor over the chassis", J1 37); the
     arm held. 9.9 ticks a second, 0 late. Not towards the cube (J1 94): **the recordings lean left** -- the
     grips' J1 median 75.5 (quartiles 54-98), 55 % of the departures go left, 6 of 58 targets in the middle
     (x 0.20-0.26, y +-3 cm). Next: the later checkpoints from START on today's frames (do they head for J1
     ~94?), and more episodes with pieces in the middle and on the right (clean-ups; or the simulator).
     **The 40k** (eval loss 0.250): the same on today's frames -- J1 to ~62 in its first second, the cube at 94.
     More steps on these data do not turn it: the placements must change (the simulator's uniform ones, or
     more real rounds in the middle and on the right).
3. **The picker's "picked" check** (shared roboshared.task.pick): a piece shoved off is not picked --
   look round (both sides) after an attempt and compare with the look before (as m3_episodes does now).
   Both robots, tests, deploy. **Done 2026-10-26:** where the gripper cannot tell (the M3; the X3 when its
   read-back has no word), picked = nothing left on the spot AND not `task.shoved()`: a piece of the target's
   kind (a cube of its size, +-7 mm) turning up 3-8 cm (MATCH_M-PUSH_M) off the spot that nothing of the look
   before accounts for (within 3 cm, 6 cm of one seen cut short; the target left out) -> "NOT picked: shoved
   N cm off its spot", the record "pushed_to" as m3_episodes marks it. A look round only when it would see
   some of the 8 cm ring the middle look after does not (`round_sees_more`: the M3's by m3teach.in_view on
   LOOK and WIDE_SIDES -- out at the sides yes, straight ahead and by the bumper no; a look round is 6.7 s
   against the middle's 2.3 s; the X3 never, its look after is its map's); it then serves the next pick too.
   Also fixed on the way: the look after was compared with the grip's spot in the base frame from before the
   drop's drive back (only matters after a drive). Tests: shared 31, X3 358, M3 153; deployed to both
   (their copies matched git first), both bridges restarted. **Live on the M3** (a clean-up, the cube straight
   ahead at x 0.238 y -0.016): picked into the box in 18 s, the look after the middle one only (no look round
   there), "picked" with no pushed_to -- the new path runs clean.
   **More episodes for the middle and the right** (the user's layouts, 2026-10-26 afternoon): round 1 7 of 8
   (156 s) -- **a real shove caught**: "NOT picked: shoved 6.5 cm off its spot" (a 30 mm cube pushed from
   0.233/-0.034 to 0.213/-0.092, the arm carried nothing to the box; the next pick took it); round 2 7 of 13
   (273 s, misses mostly "still there": the cube far left at 0.12/+0.19, a block twice) -- and **a false
   shove**: a cube lying at the side since the job began, which the pick's planning look (the middle one,
   the last pick's look after) had not seen, turned up in the look round after it as "new". Fixed: the
   check now takes everything this job has seen (Task.seen, in the job's start frame; the target's own
   readings out) -- a test repeats round 2's case; shared 32, X3 358, M3 153; deployed to both, bridges restarted.
4. **The simulator** (m3pro/sim, Dionysos): the DCW2 on arm4 rendering at the hand-eye pose, the lab's
   table and pieces (kit_synth.py's), then the expert's loop -- the picker on the true poses -- and the
   recorder writing the same LeRobot format: thousands of episodes, the real 58 to fine-tune.
5. The X3: re-teach its grips with the camera where it is now (still open).

## This week, to the meeting on Thursday 2026-10-29: sim expert -> ACT that heads for the cube

From the research of 2026-10-26 (reports/Simulácia a VLA pre presný úchop.md; its notes in
research_notes/ of the same name; the earlier precision report beside it). The sim's first job is the
table's **coverage**, not millimetres: the real data lean left and the ACT heads ~30 deg off. Sim-only
imitation reaches the cube but loses the grasp on real arms (0-40 %); sim + a few real episodes
co-trained reach 80-85 % (Genesis ACT 17/20 vs 5/20 real-only; NVIDIA 45 -> 83 %).

**Decisions (the user, 2026-10-26):**
- Isaac Sim: the week stays on **5.0 on Dionysos** (the grip works there). Standard later: 6.2 + Isaac
  Lab 3.0 GA, which needs driver >= 595.97 on Windows -- **ask the user before any driver change on
  Dionysos** (581.42 now; 6.0.1 runs on it unchanged). Side by side is safe (own folder or uv venv,
  `--portable-root`, never rename an extracted install, no machine-wide ISAACSIM_ASSET_ROOT).
- hekate: the **Hungarica LLM may be paused** when we need the 5090 (Isaac Lab 3.0 + 6.2 in a uv venv,
  GR00T N1.7, two ACT runs at once) -- say what was stopped and how it restarts.
- **Every simulated episode carries (RESEARCH.md 9.7):** the **robot card** it was drawn with (backlash
  per joint, command-to-motion delay, servo gain, hand-eye pose error, gripper offset), stored with the
  episode; and the **expert's verdict and progress for every chunk** (the hand-back gate's labels from
  the first night). Relative end-effector actions beside the joint targets: cheap to add too.
- Models: ACT now (trains anywhere, 6 ms on the Orin); the first VLA GR00T N1.7 (LeRobot `groot`, on
  the 5090, served over the network), pi0.5 second, SmolVLA if it must run on board.

**Must show on Thursday:** (1) the sim expert writing episodes in our LeRobot format (the same keys,
10 fps, 640x480, state = the command sent, action = the next tick, the same cut); (2) the target
placements, real vs sim, as a histogram; (3) the twin's gap in mm (fingertip at closure, RMS path,
ms delay); (4) a co-trained ACT that heads for the cube on a 20-cell grid in sim and on the M3.
Stretch: first real grasps of the co-trained ACT; GR00T N1.7 training on hekate.

| when | what | where |
|---|---|---|
| Mon | 73 real episodes -> dataset; ACT fine-tuned on it (real-only baseline); Dionysos inventory, read-only | laptop, Dionysos |
| Tue | DCW2 on arm4 at the hand-eye pose (intrinsics, 640x480), table and pieces from kit_synth.py, an overlay on 3-5 real poses; a servo emulator (delay ~0.15 s, gain ~0.9, backlash 0.4-1.1 deg, the speed cap) and the grip re-tested 7/7 | Dionysos 5.0 |
| Tue | the expert loop: m3pick's plan on true poses, the real look sequence, placements uniform to ~22 cm from J1's axis, small action noise; the LeRobot recorder with the robot card and per-chunk verdicts; >= 500 episodes overnight | Dionysos |
| Wed | the gap: 5-10 real episodes replayed in sim and on the arm over the ChArUco board; ACT sim-only and co-trained (90 % sim, real rebalanced by placement); an eval loop in sim (policy_service once a chunk, 20-cell grid) | M3, Dionysos, hekate |
| Wed eve | guarded runs on the M3: does it head for J1 ~94 with the cube in the middle? | M3 |
| Thu | 10-20 paired trials on the grid against the picker: mm at closure, failure taxonomy, Wilson intervals | M3 |

Fallback if the sim camera is not ready Tuesday evening: the expert's video, the real-only ACT on 73
episodes and the measured gap.

**Dionysos inventory (read-only, 2026-10-26):** a MACHINE-wide `EXP_PATH=C:\Users\mImrich\Isaac_Sim_Projects\scripts`
(another user's) -- Isaac Sim looks for its .kit apps there, so a 6.x venv must set its own EXP_PATH (5.0's
python.bat runs fine as it is); LongPathsEnabled = 1; uv in C:\Users\Hercules\.local\bin (Python 3.12 via uv;
py has 3.11 and 3.13); no Kit data dirs under Hercules yet; 5.0's clear_caches.bat asks before each step but
its shader and texture caches are shared by every version (only Kit\107.3 is 5.0's own); VRAM 7.5 of 20 GB
in use with our ACT run.

**The wrist camera in the sim (done Monday afternoon, ahead of plan):** `m3_sim.py look [--pose] [--cube x,y,yaw]
[--real FRAME] [--table JSON]` -- the DCW2 as a USD camera on the URDF's DCW2 link at its optical frame (K 480.8 /
480.9, 640x480; the hand-eye fit of 2026-09-30 stays unused, degenerate), the arm held at command + SAG as the
robot's model has it. **The simulated camera sits 0.6 mm from the model's, its axis the same to 0.004**; the cube
lands on the model's projected pixel. The lab table from kit_synth.py's pieces: paper (textures/paper_faint.png:
paper.png's print at 30 %, as faint as on the real sheet), the grey desk, tape at the far corners (the real top-left
tape within ~20 px), dome 250 + a sun from the far left (shadows towards the robot, right, as in the real frame); the
lab's green picture cube textured (paint top, pictures round). Gaps seen on the first overlay (out/sim/look_*.jpg,
real | sim | blend at START): **the sim shows the fingertips at the picture's bottom, the real one does not** (the
real camera tilted or set off the URDF's -- to measure on the ChArUco board); the cube ~20 px right / 10 px low in
the real frame (its place from the grip target, not ground truth); the sim's green lighter. Two traps: the first
renders after new MDL materials are black (60 warm-up renders now), and **one run in three renders black all
through** with the ACT training on the same GPU -- `remote()` reruns it (up to 3).

**The sim expert and recorder (done Monday afternoon):** `tools/m3_expert.py` -- the real m3pick.Picker told the
table (its looks give the pieces the camera at that pose would see, whole or cut; the box when its middle is in the
picture), its arm an M3Arm whose board is a recorder on a clock of its own: a clean-up of a drawn layout plans in
~15 s as the board's commands (pose, ms) with the job's steps, log and pick records. `m3_sim.py episode [--seed]
[--pieces] [--card drawn|zero]` -- layout (pieces uniform over 11-25 cm from J1's axis, x >= the bumper + 3 cm, 6 cm
apart, clear of the box; 30 mm picture cubes 60 %, the 40 mm cube 20 %, blocks 20 %), the box near where the real
one stands (marker 10 on its lid), the paper and light drawn, a **robot card** (delay 0.10-0.20 s, gain 0.86-1.0
round 90, zero +-1.5 deg, backlash 0.3-1.2 deg, gripper zero, the camera's mount +-2 cm / 3 deg); the expert
**compensates the card's joint part** (the arm lands as planned; the commands recorded carry it -- an open-loop plan
under the card dropped 3 of 3, compensated it picks), the sim plays the board's sweeps the card's delay late and
records 10 Hz JPEGs -> out/sim_episodes/<stamp>_s<seed>_clean_up/ in m3record's layout, ticks with `progress` and
`verdict` per tick, episode.json with `robot_card` and each attempt judged by where the simulated piece ended
(in the box and low). **First batch: 15 of 16 picks into the box**, ~3.3 min an episode (~0.3 s a picture) with
an ACT training beside it. `m3_episodes.py build OUT --episodes out/sim_episodes` builds from it unchanged (18
attempts from 8 episodes). **Night run from 14:10:** `tools/m3_sim_night.sh W now|wait` x3 (workers 2 and 3 after
the ACT fine-tune; seeds 100000*W on), logs out/sim/night_w*.log; stop: `touch m3pro/out/sim/STOP`. The laptop must
stay awake (lid open, on power).

**The closed-loop evaluation in the sim (done Monday afternoon, ahead of plan):** `m3_sim.py evaluate CKPT
[--trials 20] [--seconds 25]` -- the checkpoint copied to Dionysos (roboarm-sim/policies/), `tools/policy_http.py`
in LeRobot's venv there on the CPU (the robot's policy-service API on 127.0.0.1:8764, LeRobot's own pipeline),
`isaac_m3.py`'s run_policy: per trial the arm at START and a 30 mm cube on a grid cell, then 10 Hz: the picture,
the pose sent, a chunk every 10 ticks, steps held to max_dps/10 and swept over the tick as the board does; per
cell closest approach of the tool point, lift, in the box. The grid: 18 cells at 16/19/22 cm, J1 40-100 (the box
takes the right). **First run, the real-only fine-tune's 10k on 3 middle cells: closest 20 / 33 / 71 mm, nothing
lifted, J1 after 1.2 s 62-73 where the cubes were at 80-100** -- the left lean shows in the sim too. ~100 s a
20 s trial with two night workers and a training on the GPU.

**The twin's gap, measured (2026-10-26 14:25, `tools/m3_twin_gap.py`, the board flat in the pick zone):** 16 poses
round the servo probe's look (J1 78/88/98, J2 108-122, J3 0-12, J4 0-10), 12 saw the board (2 refused by the guard
over the chassis, 2 did not see it). Through the twin's model (command + SAG, the URDF's DCW2 mount) the board
lands **2.5 mm apart median, 7.4 mm at most (rms 3.9) on the table, 1.2 deg turned (median), its corners
reprojected 5.5 px** (median); the same look at the start and the end 1.4 mm apart (repeatability). The card's
camera turn (sd 1.5 deg) fits; its shift (sd 8 mm) is wider than the 4 mm rms -- kept wide as randomisation.
What this does not see: an offset common to every pose (absorbed into the board's mean -- the grips' ~1.5 cm
short without the taught aim is one), and the low grip poses (all 12 are looks, the camera 25-30 cm up).
out/twin/20261026_142531.json.
**Low, as the grips (`m3_twin_gap.py --low <that run>`, 14:31):** 8 poses from m3kinematics.reach (the grip
point 8-11 cm over the board, 65/75 deg down, the camera 19-22 cm up; the board then shows ~3 x 3 squares, 6-10
corners, PnP 0.1-0.4 px), 7 seen. Against the high looks' board: **on the table 3-14 mm (median ~5), but +7 to
+17 mm in height (median ~+11), 1.9-4.0 deg** -- where the LOOK in the same run read +3 mm and 0.4 deg (and 7 mm
along x: the board moved a little between the runs, or backlash from below; most low poses share that x).
~~So the twin's constant SAG leaves ~1 cm in the low shapes~~ -- **corrected the same afternoon:** that
compared runs between which the board had moved ~7 mm (the LOOK repeated in a run showed it). **The sag fit
(`tools/m3_sag_fit.py`, 38 readings from 3 runs -- 12 high, 26 low incl. a third run of 29 low poses, 18 seen --
each run its own board pose, 5-fold cross-validation, the held-out poses' board middle against their run's):
today's constant SAG 2.5 mm median on the table, 1.7 mm median in height, 4.8 mm rms 3D; refitted constant 5.5,
+J4 5.5, slopes centred on today's 4.4, a camera-mount correction (6 numbers) 4.6, a gravity lever model 3.7 --
the last only through J2 -10..-18 deg and J4 +9 deg, offsets the arm could not hide: it trades the joints
against the camera's mount, which a camera alone cannot tell apart, and would move the gripper by centimetres.
Nothing adopted, neither on the robot nor in the sim (the user asked for both; the evidence says the twin's
model already holds to ~5 mm rms, about what the readings repeat to).** What would measure the gripper's own sag:
the fingertips on known points (tools/m3_touch.py's touch test), not the camera. Two traps found on the way: from
19-22 cm the detector finds 6-7 corners (MIN_CORNERS 8 -> 6), and **the DCW2's driver stalled for three poses
-- /rgbd handed out one picture for all three** (40-62 mm "errors"): the tool now takes only a picture newer
than the last one. out/twin/20261026_143116.json (+ each pose's picture).

**The fingertip touches (2026-10-26 ~15:00-15:40, `tools/m3_touch.py`, the user with a ruler):** the look found 24
corners (camera 310 mm over the floor by depth, 313 by the board -- and 7 mm lower, 1.5 deg steeper than the
model has it at the look); `go N` now aims the FINGERTIPS (not the grip point) --up over corner N, the fingers as
steep as the arm reaches (90 near, 75-80 middle, 55 far) and shut to 5 mm (`--gap`) -- slanted and open, nobody
could tell where the arm pointed. Ten readings (the slanted ones by the finger line extended with a ruler, the
point under the tips worked out), out/touch/touches.jsonl: **the tips sit 12 mm short, 4 mm right and 19 mm low
of the model (rms 13.8 mm), nearly alike in every pose** -- reach 16-27 cm, J1 67-111, fingers 55-90 deg. The
short part is the taught aim's (m3aim: grips taught 8-22 mm further out, +5 left, 2026-10-09) -- measured
again, independently. Open 40 mm vs shut 5 mm at one pose: the shut tips 6 mm lower, 3 mm nearer (the fingers
swing), so most of the 19 mm is the arm. Fits (scratch: touch + camera readings together): **J2 2.9 deg
further down than SAG has it (SAG J2 4.0 -> 1.1) and the shut tips 9 mm longer: 5.3 mm held out, the camera's
twin readings unchanged at 4.4 mm, and it foretells the look's camera 6 mm low (the depth: 7)** -- three
measurements agreeing. Trading J2 against J3 (-10/+9) fits the touches as well but spoils the camera's (8.8);
a plain xyz bias 4.5 (no physics). J1 can't take the 4 mm sideways (no SAG on a mirrored joint; within the
readings' noise). **In the sim now** (m3_sim.TWIN_SAG {2: -2.9}): every card's J2 zero centres on it, so the
expert's recorded commands are what the REAL arm needs (as the taught aim makes the picker's), and `evaluate`'s
arm sits as the real one; `m3_episodes build` leaves out sim episodes made before (card without twin_sag) -- the
~20 of this afternoon. **Then in the robot's model too (the user's: "zapracuj opravu J2 do m3kinematics.SAG,
prepočítaj tabuľku zamierenia m3aim a otestujeme to živým clean-upom"): SAG J2 4.0 -> 1.1, TWIN_SAG back to
{}**; m3aim's table re-derived -- the same taught poses and seen places through the new model (the wide look's
camera 3 mm further out, 6 mm lower): its offsets 8-22 mm out became -3..+10, the left ones stay +1..+10 (the J2
fit can't take them). What it changes on the robot: the aim (horizontal) about the same, the descent ~9 mm
higher for the real tips (the shut tips' 9 mm not taken: FINGERS stays, fingertips_mm stays 8) -- the grips sit
higher on the piece; if they slip, fingertips_mm lower (a setting, live). J2 alone tilted the model's camera
2.9 deg steeper at the look (the depth: 1.5) and 37.4 at the survey (camcheck 34.9): **CAMERA_TILT 1.5 deg**
turns the DCW2 back up on arm4 (35.9; the board's readings 4.9 mm against 4.4). Eight tests moved with the
geometry (45b1bb9; the wide look's far edge is now 43.3 cm, not 45; the fused scan box 4.8 mm off -- SIGHT_PUSH
was fitted on the old camera, refit it on the next real scan). **Live, 2026-10-26 evening (deployed 45b1bb9):**
the first clean-up gripped the 30 mm cube and put it back -- the box stood 4 cm behind J1's axis (x -0.006),
every spot in it past J1 180 (the old model the same); the box 5 cm forward: **picked into the box, 18 s, the
close look moved the aim 5.9 mm, grip at pitch 80, J5 76, the look after empty**. One pick: more over the
table (left, right, far, the 40 mm cube, a block) before calling it.

## Tuesday and Wednesday (2026-10-20/21), for Thursday's event out of the lab (2026-10-22)

The user's three for these two days (asked 2026-10-19), and first thing on Tuesday morning
(Tuesday's done, and Wednesday's -- Thursday's checklist below):

- [x] **Tuesday morning: the M3's detectors judged fairly, on looks no model has seen** (2026-10-20).
      2 Clean up rounds (wood up 7 picked, painted up 6), their 40 looks (20 wide, 20 close; 25 with
      the 40 mm cube) are **the M3's fixed test set**: m3pro/out/kit_test (tools/kit_test_m3.py
      add/build; kit_dataset.py leaves them out of training), train/eval_kit.py `--test` per part.
      | model | mask mAP50 all / wide / close | 40 mm cube found (conf 0.25) wood wide, wood close, painted wide, painted close |
      |---|---|---|
      | `lap_20261016_m3` (served) | 0.840 / 0.951 / 0.729 | 9/9, 3/6, 4/4, 6/6 = 22/25 |
      | `dio_20261019_mix` | 0.861 / 0.979 / 0.760 | 7/9, 6/6, 4/4, 6/6 = 23/25 |
      | `dio_20261019_m3` | 0.835 / 0.975 / 0.717 | 7/9, 6/6, 2/4 (as other kinds), 6/6 = 21/25 |
      **A tie: nothing deployed**, the M3 keeps `lap_20261016_m3`. The served model misses the wooden
      face up close, the synthetic ones far off -- one cube either way out of 25. The blind spot left
      is the WOOD face at close range. The test set's own faults (all models alike): in 2 close looks
      the depth's outlines sit half a cube off the pieces (155819, 160227), pieces already in the box
      and cut by the picture's edge are labelled -- hence ~12 "missed" 30 mm cubes up close for every
      model. Worth cleaning (drop labels on the edge and inside the box) before the next comparison.

- [x] **A new UI: the event page** (2026-10-20, b1a073c): `http://localhost:8099/demo.html` beside
      the console. Asked first: both audiences (us driving, visitors watching), the laptop's own
      screen, a new page. Each arm's camera big with the last look inset, what it does in one sentence
      ("Beriem 30 mm kocku"), news under it ("Zobral som ... za 13 s", "Stôl je uprataný"), the job's
      counts; big buttons Uprac stôl / Pozri sa / Domov / Roztrieď, on the M3 Nauč sa / Nájdi / Zober;
      **H** hides every button (the visitors' view), SK / EN, Talk (hold V), STOP (Space, Esc). Tried
      on --fake and on both robots: a clean-up each, 8 picked, 0 missed.
- [x] **Voice commands with a local LLM** (2026-10-20, fb7e7e3; the user: local, on the laptop):
      Parakeet -> Qwen3.5 4B (llama.cpp b11617, CUDA 12.4, the laptop's RTX 2070 Super; files in
      ~/.roboarm/llm) -> one function as JSON held to a schema: pick, pick_named (the M3's taught
      names), clean_up (cubes or blocks only), sort, look, find, home, stop, unknown, on the M3, the
      X3 or both. A stop never waits for it (the grammar, at once); no answer in 6 s: the grammar's.
      Five models on 45 Slovak, Czech and English sentences: Qwen3.5 4B 45/45 at 0.65 s (console/
      README.md has the table). serve.py starts and warms it and prints each command ("voice: ...";
      2026-10-20 "roztrieď" printed to a cp1252 pipe failed the request -- fixed, with a test).
  - [x] **Spoken, on the robots** (2026-10-21, three rounds): English came through clean; Slovak
        did not -- Parakeet wrote it as Russian in Cyrillic ("Икс три убрать ствол"), Polish-looking
        ("kotku nie kwadrę") or French/English ("Plus Zober Valence", "um ban ballet"), and the model
        took misheard cubes for taught things on the wrong robot ("velkou kotku" -> the M3's yellow
        cylinder). Now: prompt rules (Cyrillic/Polish spellings read as Slovak, a cube or block word
        never a taught thing, a thing not taught or a sentence unsure is unknown, "valec" alone the
        cylinder), and llm.SOUNDS_LIKE putting the robots' mangled words back ("valec", "uprac",
        "stôl") before the model reads them -- told in the prompt instead, the 4B began picking cubes
        for them. llm_cases.json 45 -> 79 sentences (all today's misheard), Qwen3.5 4B **79/79**,
        median 644 ms. Round three: no wrong robot moved; 6 of 13 right, the rest nothing (unread).
        The recogniser has no Slovak switch: at the event, short sentences, English as the fallback.
        Later: Whisper forced to Slovak, or Parakeet's hotwords (modified beam search).
- [x] **Sorting** (2026-10-20): roboshared.task's `sort` job -- a clean-up whose blocks go onto the
      line out on the right (J1 180), the cubes into the box; refused, the arm home, without a box or
      with the box within 4 cm of the line (LINE_CLEAR_M). POST /pick/sort; the event page's
      Roztrieď, voice "roztrieď" (by colour refused: the robots read no colours). Tests on the
      pretend robot; on --fake the X3 sim (no box) refuses as it should.
  - [x] **Deployed to both robots** (2026-10-21; the robots' tests pass, the X3's but 3 of its old
        retired panel's, test_web, as on the robot before) and the bridges restarted.
  - [x] **Sorted on the M3** (2026-10-21), the box in front at J1 135. The first try refused, as it
        should: the box stood at y -0.22, on the line. Then 4 rounds: 3 cubes in, both blocks left (no
        grip); after the fixes below, 3 + 1 block on the line (2 misses: a cube shoved, a block stood up);
        the last **5 of 5, 0 missed**: 3 cubes into the box, 2 blocks onto the line, 216-248 mm out.
  - [x] **The user's three from watching it** (2026-10-21, the M3; tests on the laptop, deployed):
        - **Drops slipped off the box's edge**: the spots keep 4 cm from its side walls and 4.5 from its
          front (were 2.5), the lid's side as it was (4.5; the open lid is a ramp in) -- roboshared.box,
          both robots.
        - **The wrist turned over at a carry pose (J2 130), before and after each pick**: before, the
          open fingers swung over the front deck as J5 went 90 -> 270 -- now closed to J6 150, turned
          where the hand is, opened (m3pick.turn_way); after, with the box at J1 45/135 what is held came
          over the deck's corner -- now round an arc 20-26 cm up, 17-19 cm out (VIEW_ARCS).
        - **The floor limit** (fingertips over the table): 8 -> 5 mm (the setting's default and the
          M3's saved value): the two blocks left had no square grip at 8 or 6, both had one at 5.
        - **"The blocks are not too close"**: the one by the front-left corner was refused by the front
          deck's 17.5 cm box across the full width; only the tyre is there (URDF wheels x 0.078,
          y +-0.098, 4 cm round) -- the front corners keep 1 cm over the tyres now (CHASSIS by y, the
          user's yes); picked in the last round.
  - [x] **The M3's floor and line drop, after watching** (2026-10-21, the user's): at 5 mm for every
        grip the fingers touched the table by small cubes and shoved the robot -> 8 mm first,
        "fingertips_least_mm" 5 only where nothing plans at 8 (Picker.plan/floors; the log says so);
        blocks bounced off the table from DROP_UP's 11 cm -> let go with what is held LINE_OVER 2 cm
        over the table where the arm reaches that low (19-21 cm out), else the lowest of 5/6/8/11 cm.
        Then 5 of 5 again, no grip needed the 5 mm.
  - [x] **Sorted on the X3** (2026-10-21) -- after its camera: the first two tries picked 1 of 16, the
        fingertips landing where aimed (2-3 mm by read-back) and the cube still there. **The wrist
        camera had moved** (the user's guess): refitted the primary look (calibrate_table.py, the
        board still taped where it was): the view had shifted ~3 cm towards the robot. The user then
        set the camera to a place they can find again and it was fitted once more: 28 corners, 3.5 mm
        worst, within 4-10 mm of the 2026-09-24 fit; verify: the fingertip 7 mm short and 7 mm right of
        the corner (by eye). The old fits are kept on the robot (data/cameras/c930e/backup/). The taught
        aim (2026-10-15's 9 grips, through the old camera) made it worse there (-5 mm out, +3 left), so
        **the X3 aims by the camera alone now, 7 mm further out** (its saved settings: taught_aim 0,
        reach_offset_mm 7; the code's defaults unchanged). Then **4 of 4, 0 missed**: 3 cubes into
        the box, a block onto the line (263 mm out). Its box read 12 cm out and 25 cm right at first
        (refused: on the line) -- moved straighter in front, found at J1 135.
  - [x] **The X3's first tries 1 cm short** (the user, watching, at 7 mm): its reach offset set to
        17 mm; the check clean-up (2026-10-21): **9 picked, 1 missed** (picked the second time), one
        left out of reach (15 cm out, 20 cm left), 9-17 s a pick. All drops went to 2 spots 2 cm apart:
        the box's centre was ~33 cm from the X3's base, its reach's edge, where 3 of the 8 drop spots
        (4 cm margins) are in reach -- **the X3's box ~5 cm nearer** (centre <= ~28 cm out).
  - [x] **The M3's aim "a little off"** (the user: gets the job done): checked with a clean-up at the
        day's end, 6 of 6, the user watching: "the grips were fine" -- left as it is (no trims).
  - [ ] The X3, later: re-teach its grips (tools/x3_teach.py) with the camera where it is now, and
        refit the lens on the arm (lens_on_arm.json, camera_model.py handeye -- its capture still
        drives the old panel's /api/arm/move) -- the box look uses it.
- **The network at the event** (2026-10-20):
  - [x] **Decided 2026-10-21: the phone's hotspot**, the router stays home (its profile not made; the
        line below kept for another day). **Our router** (the user: our router, the phone only if we must). 2026-10-21 the user was
        unsure about changing the router's settings -- **none need to change**: the DHCP reservations
        (M3 dc:4a:9e:f3:51:b7 -> .66, X3 dc:4a:9e:37:21:35 -> .210) only kept the lab addresses, and
        the console finds the robots by name now (below), whatever addresses the router hands out.
        What is left is on the robots: a profile for its SSID on both (the user types the password;
        autoconnect-priority 5, below IISLab-AMEC's 10). Try in the lab: a robot `nmcli con up` on
        it, the laptop on it, the console, back. Ready to paste on each robot (`ssh jetson@m3pro.local`,
        `ssh jetson@x3plus.local` first -- PowerShell eats quotes in a one-line remote command; the
        WiFi is wlP1p1s0 on both; the M3's sudo asks its password, the X3's does not), the SSID put in:
        `read -rsp "router password: " P; echo; sudo nmcli connection add type wifi con-name event-router ifname wlP1p1s0 ssid "SSID" wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$P" connection.autoconnect yes connection.autoconnect-priority 5; unset P`
  - [x] Fallback, the user's Android hotspot: profile `hotspot-zabka` ("Zakrič Žabka", WPA-PSK)
        on both robots, autoconnect, priority 1 (checked). On the phone: WPA2, 2.4 GHz, no auto-off.
  - [x] **Renamed**: the M3 is `m3pro`, the X3 `x3plus` (both were `yahboom`); the laptop resolves
        m3pro.local and x3plus.local. Their desktop sessions predate the rename: a new X window may
        fail until a reboot (the trip reboots them anyway).
  - [x] **The console finds the robots** (console/locate.py): as serve.py starts, each robot by
        its name, else at its robots.json address, and with `--scan` round the laptop's networks
        (only on ours); `python console/locate.py` says where each is. **Tried in the lab**
        (2026-10-21): both by name, lookups 0.0-0.2 s -- but serve.py's very first lookup of the X3
        missed once (its robots.json address took over), so the name is now tried twice, 4 s each.
  - Skipped (the user, 2026-10-21): an Ethernet fallback (static profiles on eno1).
- [ ] **The fetch, a demo for Thursday** (the user, 2026-10-21 at lunch: "the robot moves towards the
      cube, picks it up, comes back with it and drops it into the box"). Built that day from what the
      M3 had (a pick may drive up to what is out of reach, and drives back before the drop); the event
      page sent every pick without driving. Now: the M3's **Prines** button (its teach row: with a
      name typed, that taught thing; empty, the nearest cube or block) and voice **fetch** ("M3,
      prines kocku", "prines rybu", "fetch the red cube"; the grammar knows fetch/bring/prines/dones;
      always the M3, the X3 does not drive). m3teach.pick_named takes `drive`, POST /pick/named
      {"drive": true}. In the pretend world: a cube 42 cm out -> 10 + 10 cm forward, picked, 20 cm
      back, into the box at the front-left (test_m3pick). The bench 87 sentences, 84 right, the 3
      misses safe (a pick where a fetch was meant: no driving). Deployed; **not tried on the robot**.
  - [x] **On the real M3** (2026-10-21, five tries): **it works** -- the last: the box found at J1 180
        (x 0.039 y -0.249), 10 cm forward, the 30 mm cube picked (close look 2.2 mm off), 10 cm back,
        dropped into the box; 24 s. The two before drove 18 cm + 9 cm left and 10 + 6 cm, picked and
        came back (28-31 s) but found no box: it stood ~20 cm in front of the robot's middle, the
        marker cut by the box look's top edge (the user measured "ahead" from the bumper, 12 cm
        further). One before that picked a block in reach on the right instead: **a pick takes what
        is in reach first** -- only the far piece on the table. **The setup that works:** the box
        beside the robot, level with its front wheels, ~25 cm right of its middle, marker to the arm;
        the lane clear; the cube 35-45 cm ahead.
  - [ ] Later, if wanted: Prines goes for the nearest piece out of reach even with others in reach
        (and a named fetch takes only that thing: Task.pick tries every piece in reach, nearest `at`
        first); a second box look tilted out for a box further forward; the log saying why it does
        not drive (2026-10-21: the box too near, and only "out of reach" in the log).
  - [x] First try (2026-10-21): no drive -- the table's
        clutter by the bumper (a cable grommet, a phone) was in the lane, and the box at the
        front-left was never seen (**the M3's left looks see its own deck**). So: the box **on the
        right, beside the robot** (~10 cm ahead, **its centre >= 24 cm right** of the robot's middle,
        marker to the arm: J1 160-180); since then the drive also keeps its footprint DRIVE_GAP 3 cm
        from the box (Picker.approach: its black walls give the depth nothing); **a clear lane** the
        robot's width + 3 cm each side from the bumper past the cube; one cube **35-45 cm** ahead
        (past the arm's ~27 cm, inside the wide look's 45 cm); the table >= 60 cm deep in front (no
        edge sensing: odometry only, 10 cm a move at 5 cm/s, 3 moves at most, then back). STOP /
        Space stops the base too.
  - The show, in order (each a minute or two): **Pozri sa** -> **Uprac stôl** (in reach) -> a cube put
    out of reach, **Prines** ("Podchádzam bližšie" ... "Vraciam sa na štart") -> a visitor's thing:
    **Nauč sa** (its name typed, ~50 s), put it 35-45 cm out, **Prines** with the name ("prines
    <name>") -> **Roztrieď** (cubes into the box, blocks onto the line: the box then back in front, the
    right side clear) -> on the X3 meanwhile Uprac stôl / Roztrieď.
- [x] **The dress rehearsal on the phone's hotspot** (2026-10-21, 14:40-16:20, in the lab):
  - **Cold start** (both robots off and on by the switch): every service up by itself, the X3's bridge
    after 114 s, the M3's after 151 s. The M3's arm reads "not connected" until the first command when
    Yahboom's micro-ROS agent (desktop autostart) comes up after the bridge: the first button connects it.
  - **The hotspot**: both robots joined it (the X3 in 4 s), the laptop too (Windows went back to
    IISLab-AMEC once by itself: connect again); the console found both **by name** on it (mDNS works
    on the Android hotspot: laptop 10.31.21.150, M3 .191, X3 .81); both cameras streamed fine on 2.4 GHz.
  - **The M3 must start on the network it works on**: moved from IISLab to the hotspot while running,
    its camera stopped (the bridge restarted the driver every 21 s, no frames) and its arm too -- ROS 2
    (CycloneDDS) keeps the address it started with. A bridge restart brought the camera back, not the
    arm (the agent's); started on the hotspot (IISLab-AMEC's autoconnect off for the test, on again
    after) everything worked. So at the event: hotspot on BEFORE the robots; if the WiFi drops and the
    M3 comes back with another address, power it off and on.
  - **The show**: M3 look 11 s; clean-up 2 of 2, 44 s; sort **7 of 7** (5 cubes into the box, 2 blocks
    onto the line), 2 min 1 s; **fetch: 30 mm cube 34 cm out, 10 cm forward and 2 right, back, into the
    box, 33 s**. Fetch failures on the way: the box's middle 21.8 cm right of the robot's -> no drive
    (DRIVE_GAP from the box at any distance), only "out of reach" in the log; pieces in reach taken
    first; a piece in reach that no grip reaches ends a fetch ("Refused: no way down with the fingers
    turned"). X3 clean-ups 6 of 6 (the 6th a phantom, below), 6 of 8, 7 of 8 after the refit (below).
    Not rehearsed: Teach + fetch by name. Voice once ("m3 pozri sa" right; "Pôrissa" unknown).
  - **Fixed and deployed that afternoon** (tests: M3 134, X3 358, shared 27):
    - **The M3's box looks every 22.5 deg** (m3pick.BOX_LOOKS, were 45): the box beside the robot's right
      had its marker cut by the picture's right edge at J1 135 and its left edge at 180 (640 px, 67 deg;
      the marker a quarter of the picture) -- "marker 10 not seen all round". Checked live: found.
    - **The zone round the box 4.5 cm** (roboshared.task BOX_NEAR, was 2 cm past the 1 cm wall): the X3
      picked a "30 mm cube" 3.6 cm from the box's inside -- air (the black walls' relief in the depth).
    - **The X3's survey turn smooth**: each picture's yaw from the commanded setpoint less the servos'
      0.1 s trail, not read back -- the read held the serial line the setpoint stream needs, and the
      stream caught up on its clock: jumps of 10-20 deg now and then (the user: "it looks bad"). Smooth
      now (the user), the look again still 1-11 mm off the search's.
    - **The X3's near and outer looks refitted** (calibrate_table.py with the board; since the morning's
      camera refit only the primary was calibrated, so nothing under ~11 cm was seen): from yaw 0 only,
      0.7 mm worst each (pooled over 5 yaws 7.2 and 16.8 mm: the board against J1's axis); verify (near):
      fingertip 7 mm short (the arm's own reach) and 4-5 mm left. Then cubes at 9.5-12 cm found and picked
      (two at the first go, one at the second). Fits kept in data/cameras/c930e/backup/.
- [ ] **Thursday's checklist** (updated with the rehearsal):
  - **The network is the phone's hotspot** (the user, 2026-10-21): "Zakrič Žabka" (WPA2, 2.4 GHz, no
    auto-off), which both robots know (hotspot-zabka, autoconnect, priority 1 -- IISLab-AMEC, 10, is
    not there). **The phone's hotspot on first, then the robots** (the M3 keeps the address it starts
    with), the laptop on it too; the console finds the robots by name (`serve.py --scan` if not). The
    router is left at home.
  - Pack: both robots and their chargers; the box (marker 10); the kit (30 and 40 mm cubes, the
    blocks, the wooden 40 mm cube) and the taught things (fish, apricot, news, cosmetics, the yellow
    cylinder, the monster can, the jetson box); the laptop and its charger, a mouse; the phone
    (the hotspot) and its charger. **The X3's A3 ChArUco board and tape** -- if its wrist
    camera is knocked in transport (it had moved on 2026-10-21) the camera fit needs it. Pad the X3's
    wrist camera for the trip. No Ethernet cable or switch: that fallback was skipped (2026-10-21).
  - On site, in order: the phone's hotspot on -> both robots on (their services start at boot,
    ~2 min) -> the laptop on the hotspot -> `.venv\Scripts\python console\serve.py` (finds the robots by name,
    starts the LLM, ~20 s) -> Chrome at http://localhost:8099/demo.html (F5 after any console
    restart) -> Pozri sa on each (the M3's arm card may say "not connected" until then: the first
    button connects it), then Uprac stôl on each with 3-4 pieces -> H for the visitors' view.
  - Each table, its marker facing the arm: for a **sort** (Roztrieď) **the box in front** (J1 90-120,
    ~15-25 cm straight ahead) and **the right side clear** -- the blocks go onto the line out there, a
    box there is refused; for the M3's **fetch** (Prines) **the box beside it on the right**, level
    with its front wheels, **its middle 25-26 cm right of the robot's** (21.8 cm: no drive), the lane
    ahead clear, **nothing else in the arm's reach** (it takes that first), one cube 35-40 cm ahead.
    Never on the M3's left: it cannot see there. M3 pieces 15-25 cm out, not right against the robot.
    The box near the arm (15-25 cm): further out the marker on the lid can stand out of the box look's
    top edge (seen once, 2026-10-21).
  - **The X3's aim**: its camera was refitted 2026-10-21 and it aims by the camera alone (taught aim
    0, reach offset 17 mm, its saved settings; 9 of 10 picked with it). Its box no further than ~28 cm
    from its base (its centre), or the drops pile up on the few spots it reaches. Its near and outer
    looks refitted in the afternoon: X3 pieces from ~9 cm out, best 12-25 cm (the look again finds
    nothing under ~11 cm, so a miss there is picked at the second go). First clean-up misses with the fingertip landing where
    aimed -> the camera moved: look at its mark; quick fix the console's "reach offset" setting
    (mm further out), proper fix x3plus/tools/calibrate_table.py with the board (bridge stopped).
  - **The M3**: fingertips 8 mm over the table, 5 only where nothing else grips (console settings).
  - **Voice**: English is the sure one; Slovak in short sentences, the robot first ("Plus, uprac
    stôl"); "stop" / "stoj" always stops both at once (no model in the way); unread -> nothing moves.
  - When something fails: a robot not found -> `serve.py --scan`; still not -> power it off and on
    with the hotspot on (it joins within a minute); voice not understood -> the typed line or the
    buttons; the LLM down -> the grammar answers by itself; the M3's board silent -> power off and
    on (2026-10-15); a camera stalled -> the bridge restarts its driver, else power off and on; **the
    M3's camera and arm dead after the WiFi dropped** -> power the M3 off and on (the hotspot on);
    **a fetch that does not drive** -> the box too near the robot's side, or a piece within reach.
  - Batteries: the M3 refuses under 10.5 V (the page warns under 10.8), the X3 under 10.0 (warns
    under 10.5) -- charge between showings.

## After the event (from Monday 2026-10-26): a VLA taught in Isaac Sim -- the M3 first, then the X3

The user (2026-10-20): "VLA in Isaac Sim -- can we simulate an expert there and train from it?" Yes:
PLAN.md's track D (a picker that teaches a VLA, RESEARCH.md P1), with the teacher in the simulator.
There the expert knows every piece's true pose, so it needs no detector or depth: perfect demos,
automatic resets, thousands of episodes a night, and DAgger for free (from any state the student
reaches, the expert's next move). **The M3 first** (Yahboom's URDF matches it), **the X3 too**, after.
**Where:** Dionysos (Isaac Sim 5.0, A4500 20 GB), the laptop's GPU and, from Monday, hekate again (back
Friday evening 2026-10-23: its 5090 and Isaac Sim 6.0).

- [ ] **The M3's arm in Isaac Sim** (~1 week): Yahboom's URDF (yahboom_M3Pro_description/urdf/
      M3Pro.urdf, which m3pro/roboarm/m3kinematics.py follows, with the hand-eye calibration's
      corrections) -> USD by the URDF importer; the base fixed (table-top only), the arm's joints and
      the gripper as drives held to the servos' speeds; the DCW2 at m3pro/data/m3_handeye.json's pose
      (colour, depth too); the table and the lab's pieces from train/synth/kit_synth.py. **First check
      the grip physics** (friction, finger contact): a 30 mm cube gripped must lift, carry and drop --
      the hard part.
  - [x] **Started 2026-10-23 on Dionysos (Isaac Sim 5.0): the grip physics works.** `m3pro/sim/isaac_m3.py`
        (Isaac's python: `import` Yahboom's URDF -> m3pro.usd, base fixed; `play` a plan on a table, 120 Hz,
        pictures at its marks) and `m3pro/tools/m3_sim.py` on the laptop (`setup`; `grip --x --y --yaw --size
        --mass --friction [--close]`: the picker's own plan_grasp, without the real arm's SAG, servo degrees ->
        URDF radians, J6 -> the finger joint through the gripper measured today; runs it there, ~35 s, summary +
        a picture sheet in m3pro/out/sim/). The tool point lands where m3kinematics says (200 vs 202 mm).
        What it took: the finger's URDF range opens 49.7 mm, the real one 61.5 (widened to -1.0..0.6 rad);
        the linkage's joints driven stiffly each step to their share of where the finger IS -- PhysX's mimic
        joints did not hold (the importer left the left side's without a reference; set by hand the links hung
        free), and following the finger's target wedged the cube out; the arm's joints capped at 150 deg/s (a
        target far ahead flung the cube); the plan rises to the look's height before turning (swung round low,
        the cube hit the robot's front-right corner). **Then 7 of 7 picked, carried and let go:** 30 mm cubes
        at friction 0.3-0.8, at (0.16, +0.13), (0.24, -0.08), turned 20-60 deg, the 40 mm cube turned 30.
        Next: the camera (the DCW2 on arm4) and the lab's table and pieces (kit_synth.py's), then the expert's
        loop and the recorder.
- [ ] **The expert:** m3pick's own plan (IK, approach, grip, drop into the box) fed the simulator's
      true poses instead of the depth's; for DAgger, its next action from any state.
- [ ] **Recorder and resets:** a LeRobotDataset per episode (the wrist image, the joints, the action =
      joint targets at 10-30 Hz, the instruction -- "zober modrú kocku" and in English); the pieces
      scattered anew after each; 1-5k episodes on Dionysos (Isaac Sim 5.0) or hekate (6.0).
- [ ] **Train and judge in the simulator:** SmolVLA or X-VLA through LeRobot (SmolVLA fits Dionysos's
      A4500 20 GB; pi0.5 with LoRA only on the 5090); success rate, unseen scenes, "the blue one", a
      cube moved mid-reach (what detect-then-move cannot do).
- [ ] **On the real M3:** a few dozen real episodes from the classical picker to fine-tune; the policy
      served from the laptop or the 5090 (LeRobot's async inference), a watchdog holding the arm when
      action chunks stop. Check first: the M3 reports the joints it was sent, not measured ones
      (why PLAN.md D chose the X3) -- fine for the actions, to be seen for the state. **Checked
      2026-10-23:** nothing comes back at all; the servos measured by the camera instead (top).
- [ ] **Then the X3:** its URDF's forearm is over twice too long (x3plus/roboarm/kinematics.py) -> its
      model from config.py's measured links; the same expert (x3pick) and recorder. One policy for both
      arms is a research question of its own (cross-embodiment).
- [ ] **Later: "podaj mi nástroj" -- small and thin things** (the user, 2026-10-20: could a VLA grip a
      scalpel, or things smaller than the cubes?). After the first VLA steps with the cubes work. What
      decides it is the thing's shape and the arm's floor, not the VLA:
      - small things standing >= ~1 cm off the table are in reach (a 15-20 mm cube, a Lego brick, a
        marker, a screw; an SO-101 picked a Lego brick at 70-90 % where it was trained);
      - flat and thin, lying on the table (a scalpel handle 2-3 mm, a coin), is very hard: the
        fingertips must reach 1-2 mm over the table, our floor is ~1-2 mm with the camera's loop, the
        servos' backlash 3-6 mm a joint, the M3 1.5 cm off without its corrections;
      - so: **the tools in a foam stand** (handles 2-3 cm up) -- first by Teach + Find on the M3 (DINOv2
        tells them apart already), then as a VLA task; later a contact grip (soft fingertips, down until
        the table, close along it) learned from demos; thinner, longer fingers with rubber -- measure how
        far our grippers close first; matte tape on shiny metal (the depth sees no 2-3 mm thing at all);
      - **never a real blade** at the event or near people: a handle without its blade or a printed one.
- **The risk is sim-to-real:** images (the kit detector's lesson: synthetic pictures helped only once
  they looked like our table and pieces), contact physics, cheap servos' dynamics. Hence domain
  randomisation plus a real fine-tune -- and "a teacher in the simulator vs a teacher on the robot"
  as the paper's experiment (RESEARCH.md P1).
- **Time:** the first numbers in the simulator in 2-3 weeks, on the real M3 in 4-6.

## Next week (from Monday 2026-10-19): the M3's detector and the 40 mm cube

The one 40 mm cube is the kit detector's blind spot on the M3: wood up it is missed, side up it is
called a block. **What it is** (the user's photos, 2026-10-19): bare end-grain wood on TWO opposite
faces, its four sides painted yellow, green, red and blue (in that order round it) -- not "wood with one
blue face" as noted on 2026-10-16. The live rounds: missed in
most close looks; 8 wide looks of it: found 6 times. Fine-tuning on 8 more wide looks gained nothing
(`lap_20261016b_m3` 0.795 vs 0.802 on the M3 val, cube40 0.68 vs 0.69) -- the M3 keeps
`lap_20261016_m3`. The likely cause: **the synthetic set never had a wooden cube** --
train/synth/kit_synth.py paints every piece one jittered colour (COLOURS), so the 40 mm cube was
always a plain coloured block.

- [x] **Dionysos, the second GPU server** (2026-10-19): `ssh Hercules@147.232.40.154` (key; the user
      installed OpenSSH and widened its firewall rule to every profile), RTX A4500 20 GB, Isaac Sim
      5.0.0; a shared lab PC -- never touch the other users' processes.
- [x] **kit_synth.py renders the lab's cube** (2026-10-19): a mesh with a material per face -- the
      wood from the user's photos (train/synth/textures/wood_a/b.png), the four painted sides -- 70 %
      of the 40 mm cubes, 60 % of them wood up; the yellow cylinder about, unlabelled. Runs on
      Isaac Sim 5.0 too (a USD sun) and headless over SSH (`train/synth/run_standalone.py`). Each
      picture after 30 dropped renders: the previous picture's pieces showed through as ghosts.
- [x] **Synthetic pictures with the real 40 mm cube, two rounds** (2026-10-19, Dionysos):
      1. 2,000 of the arena mat with the painted cube -> real + synthetic, then real only, on the
         laptop's GPU: M3 val 0.762 / 0.790 (cube40 0.635 / 0.650) against the served
         `lap_20261016_m3`'s 0.802 (0.693). Why it did not help: **the M3 works on a table, not the
         mat** -- white paper printed faintly with a tag board on a grey desk, its 30 mm cubes white
         picture cubes (the picture beside a real look: scratchpad, sent to the user).
      2. kit_synth.py now renders the lab table in 60 % of the frames (textures/paper.png, tape,
         a power strip, cables, brighter light) and the lab's own pieces in 80 % (the four picture
         cubes -- 15 pictures and 5 tags cut from the user's photos -- the red block with a tag on
         its end and the blue one; 72ee213). 2,000 more, then trained ON DIONYSOS (20 min for both
         stages): `dio_20261019_mix` 0.784 (cube40 0.672), `dio_20261019_m3` 0.759 (0.643) against
         0.802 (0.693).
      **The M3 val cannot decide this:** 40 pictures, the 40 mm cube in 8, and the served model saw
      part of them in training (the scene split moved as frames came in). Hence Tuesday morning's
      test above.
- [x] **Dionysos trains too** (2026-10-19): C:\Users\Hercules\roboarm-train\.venv (torch 2.11 +
      cu128, ultralytics 8.4.175 -- the laptop's pins) sees the A4500 over SSH; train_kit.py,
      eval_kit.py, synth/synth_to_yolo.py and the M3 set (datasets\m3b) there; the whole cycle
      (render, labels, mix, two fine-tunes, comparison) runs there, only weights come back.
- [ ] **Real pictures of it from the clean-ups** (no server needed): a few Clean up rounds with the
      40 mm cube on the table, turned between rounds, wood and blue up -- each round keeps its wide and
      close looks, labelled by the depth (kit_dataset.py, the box and the cylinder now left out).
      Then on the laptop's GPU (~6 min): `C:\Users\Mys\.roboarm\train-gpu\Scripts\python.exe
      train/train_kit.py --data <set>/kit.yaml --model train/runs/kit/n_20261015b/weights/best.pt
      --epochs 40 --name ...`, then train/eval_kit.py on the M3 val; build the engine on the M3 only
      if it gains (models/kit/yolo26n_kit.pt, `.venv-ml/bin/python tools/export_engine.py`, ~7 min).
- [ ] Judge it LIVE, not only on val: a clean-up, the "kit detector:" lines -- does it see the cube
      wood up, in close looks?
- [x] The look job's fix (look_over) on the X3 too: deployed with the sort, 2026-10-21.


Stage 1 (PLAN.md, "First show") has four demos: a coloured 3D scan in the console, "teach in 30 s,
picked out from look-alikes", real vs predicted depth, and the speed table. The scan (also as a
Gaussian splat) and the speed table (PLAN.md M5) are done; the depth duel has its table and its
pictures (3 below); the teach from look-alikes is done too (1 below: every cube named, Teach 10 s,
Teach to the piece put down ~50 s). The kit detector is one model on both robots since 2026-10-16
(2 below: the X3 picks with it, the M3 logs it beside its depth -- its score still to read), and the
close look's depth fit is fixed as far as the networks allow (3). The robot work (you) and the
laptop/hekate work (me) run side by side.

## 0. The M3's camera -- solved: it was the conversion beside it

On 2026-10-15 the DCW2 kept going silent -- colour and depth stop, the driver process stays up, not a
word in its log, no USB event in the kernel log. It had been frozen since 2026-10-13 09:55 while
/health still said 22 fps; after a restart it stopped again at 09:52 and 10:03, and in bursts after
10:18.

- Done: the bridge restarts the driver by itself after 5 s without frames (a line in the pick log;
  a job that meets a stall says "try again in half a minute"), and starts it without the IR stream
  and the point clouds (bdbbc83, deployed on the M3).
- **The cause: `~/edge-vlm` on the robot (not ours).** `build_all_gguf.sh` converted VLMs to GGUF
  there from 09:53 to 11:16 (Qwen 3.5 2B/4B/9B, LFM2.5-VL 3B, MiniCPM-V 4.7 1B done; Gemma 4 E4B
  failed while writing its 15 GB BF16 -- not the disk, 621 GB free). The watchdog restarted the
  driver 26 times while it ran, the last at 11:17:02 -- and not once in the 1.5 h after, the robot
  idle. The script builds on the robot on purpose (K-quants differ between x86 and ARM): run such
  jobs while the camera is not needed, e.g. overnight. (Copying model files onto the M3 -- the
  edge-vlm relay, hekate -> laptop -> M3 by `scp -3`, until ~12:27 -- did not stop it.)
- [x] **/health reports a frozen camera** (merged 746ab04 + 18a976b, deployed on both robots
      2026-10-16): the M3's newest colour and depth age and an error once either is older than
      STALE_S, on the console's arm card too; on both robots the stream's fps falls to 0 when frames
      stop (it held the last rate).
- **The M3's board went silent at ~12:33** (no node /YB_Node topics: battery, odometry, arm), after
  a failed Teach had turned the arm towards the left front tyre (see 1). Restarting the micro-ROS
  agent (`~/start_agent.sh`'s command, in a desktop terminal) and the board's reset button brought
  back battery and wheels but NOT the arm's /arm6_joints subscription; only switching the robot off
  and on did (12:55: arm connected, camera fresh). Next time: straight to the power switch.

## 1. Teach from look-alikes -- done 2026-10-15

It works on a table: Teach, Find and Pick by name move the arm only (leave "may drive closer" off,
no Scan). Teach takes the thing nearest the camera: until 2cf84bc (deployed 2026-10-15) that was
the left front tyre when the cube stood 20 cm out -- "6 cm tall at x 0.130 y +0.119", 0 views, the
arm left turned. A failed Teach still does not send the arm home: press Home.

- [x] **Every side taught:** 4 rounds per cube, turned a quarter between (Teach with the same name
      adds views) -- news 19 views, fish 22, cosmetics 20, apricot 23 (from 5-8 on one side).
- [x] **Find** with the Jetson box, the can, both news cubes and the fish: every cube named right
      (news 0.85 and 0.80 -- unknown at 0.64/0.61 on 2026-10-12 -- fish 0.85), the box 0.76; the can
      unknown (0.47), lying and mostly out of the picture.
- [x] **The headline number** (PLAN.md track A): a yellow cylinder never seen before -- Teach 10 s
      (every Teach 10.0-10.5 s), Pick by name found it at 0.95 (the next name 0.58 behind), picked
      and put down 29 s later; ~50 s of robot time in all.
- [ ] **Faster for the demo:** skip the drop-off box's search when there is none (9 s of the 29);
      Find's closer look at far unknowns (the tape at the paper's edge) costs a few seconds more.
- [ ] Optional: the can standing and whole in the picture -- named? (taught standing, from a scan)


## 2. The kit detector -- the X3's pictures in, then onto the robots

The detector's validation had only 3 pictures with a 40 mm cube and the block is barely judged.
(YOLO26s + synthetic, trained again on 2026-10-13: 0.697, the same as on 2026-10-12 -- YOLO26n stays.)
On 2026-10-15 it went through the X3's own vision-service code unchanged (right names and outlines)
but on the X3's pictures it took the tag board's squares for cubes and blocks (0.3-0.6): it had
never seen that camera or the board.

- [x] **The X3's pictures as training data** (fbf8639): `x3plus/tools/kit_dataset_x3.py` fetches the
      looks the X3 keeps and labels them by survey -- each kit piece a survey saw whole is projected
      through the wrist camera's model into every look of it; on hekate `train/refine_x3_sam.py`
      redraws the outlines with SAM 2 and takes out pictures where a mask disagrees or the current
      detector sees an unlabelled piece. `kit_cycle.py --x3` runs it all. 185 new X3 looks on
      2026-10-15 (empty tag board; pieces on paper; pieces on the board; a standing block) besides
      900 from 2026-10-06/08 (the user's tagged cubes and blocks are kit pieces too). In the run
      below 680 of 1,085 kept, 261 out on a mask, 144 on an unlabelled piece. The laptop's key is
      on the X3 now.
- [x] **The run `n_20261015`** (`kit_cycle.py --synth --x3`; its mix stage, then finished by hand
      after kit_cycle stopped on printing -- fixed in 4bf8405), judged by `train/eval_kit.py` on each
      robot's val and on **an X3 test round held out** (round 5, 71 pictures, x3plus/out/looks_test):

      | mask mAP50 | M3 val (14) | X3 val (71) | X3 test (71) |
      |---|---|---|---|
      | `n_cycle` (M3 + synthetic) | **0.791** | 0.302 | 0.246 |
      | `n_20261015` (+ the X3's) | 0.560 | **0.656** | **0.497** |

      Twice as good on the X3; on the M3 it loses its 40 mm cube (0.672 -> 0.056, three in val)
      under 15x more X3 pictures. PLAN.md M5 has the per-class numbers.
- [x] **Decided** (2026-10-16): one model for both robots, `n_20261015b` (below).
- [x] **`n_20261015` on the tag board** (the morning's 25 check pictures, never trained on, through
      the X3 service's own detector.py): not one false cube on the board any more (the old model had
      7-9 finds on the busiest); the red and yellow cubes cube40, the blue one a block. A few weak
      ones left off the board -- a post-it, something on the keyboard, empty paper (0.38-0.49);
      none kit-sized, so the X3's ranging drops them. Picture: x3plus/out/kit_detector_on_x3_new.jpg.
- [x] **The X3 serves it** (2026-10-15 evening): `yolo26n_kit_x3.pt` (= n_20261015's best.pt) in
      ~/roboarm/models/kit/ on the X3, its 640x640 FP16 engine built there (~7 min,
      `.venv-ml/bin/python tools/export_engine.py --model models/kit/yolo26n_kit_x3.pt --imgsz 640`);
      `ROBOARM_VISION_MODEL=/home/jetson/roboarm/models/kit/yolo26n_kit_x3.pt` in
      ~/roboarm/robot.local.env (empty before: the default was YOLOE-26s-pf); roboarm-vision
      restarted: /health says yolo26n_kit_x3.engine, cube30/cube40/block; a live frame through the
      bridge's /vision/detect came back cube40 0.52 with its outline. **Back to YOLOE:** delete
      those lines in robot.local.env, `systemctl --user restart roboarm-vision`.
- [x] **A Clean up round on the X3 with it** (13:52-13:55): 11 pieces picked into the box (a 40 mm
      cube, eight 30 mm, two blocks), 8-11 s each, 2 misses on one cube close in (x 0.10, taken at the
      third go), the table clear at the end in 2 min 48 s. The X3's battery 11.2 V after: charge it.
- [x] **M3 frames** (2026-10-15, 5 clean-up rounds): 60 new looks -- 40 mm cubes seen 33 times,
      blocks lying 34, standing a few more. The M3 set: 117 train / 26 val pictures (40 mm cubes
      70 / 8, from 16 / 3). The yellow cylinder's pick looks set aside (m3pro/out/duel/frames_not_kit:
      its depth names it a 30 mm cube). Fetched by key now (`m3_duel.py fetch`, c558563).
- [x] **The run `n_20261015b`** (M3's new frames + X3 + synthetic): mask mAP50 on the NEW M3 val
      (26 pictures) 0.634 (`n_20261015` 0.563; its 40 mm cube 0.11 -> 0.28), X3 val 0.666, X3 test
      0.471 (0.497) -- the X3's the same within the noise, the M3's best: one model for both.
- [x] **The X3 serves `n_20261015b`** (2026-10-16): ~/roboarm/models/kit/yolo26n_kit.pt and its FP16
      engine (built there, 7 min), robot.local.env points at it (robot.local.env.20261016 is the
      file before; `yolo26n_kit_x3.*` kept): /health yolo26n_kit.engine, a live frame -> cube30, block.
- [x] **The M3 serves it as a second opinion** (2026-10-16): its own `roboarm-vision` user unit
      (m3pro/robot/roboarm-vision.service, .venv-ml, 127.0.0.1:8760, the X3's vision_service code
      brought into m3pro/), the same yolo26n_kit.pt + an engine built on the M3 (its older
      yolo26n_kit.* now *_20261012.*). `roboarm/m3kit.py` asks it at every look (~55 ms) and the pick
      logs "wide look, kit detector: agrees on n of m; depth X, detector Y 0.55; only it saw ...";
      the look's .json keeps its answer ("detector"). The depth stays in charge.
- [x] **Its score, 3 Clean up rounds** (2026-10-16; 2, 2 and 10 picked -- rounds 1-2 ended on a full
      box): the detector is right where the depth fails (crumpled paper and the box's rim read as
      cubes, a cube cut by the picture's edge, the standing red block) and wrong on the one 40 mm
      cube (blue on one face, bare wood on the rest: wood up it is not seen at all, blue up it is a
      block) and the tag cube (a block). Both call the yellow cylinder a 30 mm cube. Round 3's two
      "40 mm cube" picks cannot both be it -- the depth read another piece 40 mm tall (the tag cube?).
      Pictures: the looks' .json keep both answers.
- [x] **The M3's labels cleaned** (`m3pro/tools/kit_dataset.py` not_kit()): 123 of its 694 depth
      labels were the black box's wall or lid named a cube, or the yellow cylinder -- now background.
- [x] **One model or one per robot?** (2026-10-16, trained on the laptop's GPU -- hekate was busy
      with edge-vlm -- as fine-tunes of `n_20261015b`, 40 epochs on the cleaned labels):

      | mask mAP50 | M3 val (36) | X3 val (71) | X3 test (71) |
      |---|---|---|---|
      | `n_20261015b` | 0.649 | **0.666** | 0.475 |
      | `lap_20261016_shared` (M3 + X3) | 0.756 | 0.629 | **0.495** |
      | `lap_20261016_m3` (M3 only) | **0.820** | 0.473 | 0.255 |

      A shared base, fine-tuned per robot where that robot gains: the M3 serves `lap_20261016_m3`
      (its 40 mm cube 0.51 -> 0.77; only 6 in val), the X3 keeps `n_20261015b` (the new shared one
      is no better there). The laptop trains in `C:\Users\Mys\.roboarm\train-gpu` (torch 2.11 +
      cu128, ultralytics 8.4.175 -- train/'s pins, outside the repo).
- [x] **The box read wrong** (2026-10-16, why rounds 1-2 stopped early): the marker pose is fine (IPPE's
      other solution always 5 px worse), but the box look's camera was levelled by whatever plane the
      depth saw most of -- at J1 135 the room's floor and chairs past the table's edge are in view -- and
      put the box 10 cm off, out of reach (the arm model's camera: where it stood). And a box standing
      still reads +-1.5 cm and +-3 deg: "the box has moved" (1.5 cm, 5 deg) before nearly every drop.
      Now: a levelled camera more than 3 cm or 5 deg off the arm model's is not trusted (m3pick.level_trusted;
      usual 12 mm, 1.7 deg); the box is the mean of the job's readings of it, moved only past 3 cm / 8 deg
      (roboshared box.merged); every box look kept as out/pick/boxview_* (picture, depth, both cameras).
      Deployed 2026-10-16 on the M3 (with roboshared serve.py: the look job came along) and the X3
      (box.py, task.py). **Checked on the M3:** a clean-up, 5 picked, 0 missed, 5 drops into the box
      and not one "moved"; 4 of its 8 box looks fell back to the arm model's camera (all while holding
      a piece); it stopped on 3 pieces out of reach (28 cm out, 41 cm aside, one at the right lidar).
- [ ] **The M3's own detector, live:** in that round it still missed the 40 mm cube in most looks
      (once cube40 0.57) -- its better val numbers (6 cube40s in val) did not show; the cube is one, so
      its pictures stay few. 2026-10-16: 8 looks of it (`/pick/look`, nothing picked: near, far, left,
      right, wood up and blue up) -- the depth named it in all 8, the detector in 6 (once missed far
      right, once a 30 mm cube 0.82); fine-tuned again with them as `lap_20261016b_m3`.
- [x] **The look job on the M3** (2026-10-16): it failed at once ("Picker.look() missing 'tag'") -- the
      shared job was named look(), as the M3's own looks are, and the M3's won. Now look_over() (the
      job is still "look"); a test on the M3's side; deployed on the M3.

## 3. Me, while the robot work runs (laptop + hekate)

- [x] **Depth duel, the big models** (2026-10-13; train/depth/ on hekate, `m3pro/tools/m3_duel.py run`
      scores them): DA V2 Large takes the close look from 15.5 to 9.6 mm (named 19 -> 33 %), still
      over the 7 mm bar; wide 4.8 mm (2.7 mm and 95 % on the DCW2's floor). The metric models (Depth
      Pro, DA V2 Metric Indoor) read 2-4x too far at 20-30 cm.
- [x] **The duel's pictures** (2026-10-15): `m3_duel.py run --panels 16 --versus da2l` -> colour |
      DCW2 | DA V2 Small | DA V2 Large in m3pro/out/duel/; for the demo panel_wide_5 and panel_close_0.
- [x] **Why the close look fails** (2026-10-15/16, d14f9d0, `m3pro/tools/duel_floorfit.py`): about
      half was the floor fit. A third of a close look's predicted floor is no floor (the black box,
      tape, in 4 of 31 the gripper's finger before the lens -- `a` then 4-9x off) and the fit kept 98 %
      of it; the quadratic warp made every look worse. `monodepth.anchor` now: no warp, and up to 3
      fits more without what reads over 10 mm. Height error, median / p90, named within 7 mm:

      | | DA V2 Small close | DA V2 Large close | DA V2 Large wide |
      |---|---|---|---|
      | 2026-10-09 (tuned on), before | 15.5 / 32.9, 19 % | 9.6 / 25.5, 33 % | 4.8 / 8.9, 78 % |
      | 2026-10-09, after | 11.7 / 26.3, 34 % | 5.2 / 17.5, 61 % | 3.5 / 7.2, 88 % |
      | 2026-10-15 (held out), before | 13.2 / 33.7, 32 % | 8.0 / 18.5, 43 % | 5.5 / 13.1, 68 % |
      | 2026-10-15, after | 10.9 / 32.7, 38 % | 6.9 / 19.0, 52 % | 5.6 / 10.8, 71 % |

      The rest is the network: anchored on the DCW2's own floor (the best any anchor can do) DA Small
      close stays ~9-10 mm, so DA Small will not make 7 mm close; Large just does. monodepth.anchor
      is the duel's only (the X3's live depth rung finds raised regions by edges): no robot changed.
- [x] **DA3 in the duel** (2026-10-16, on the laptop's GPU, ~350 ms a picture; `train/depth/predict_depth.py
      --models da3l,da3mono`, one picture each, 1/depth anchored as DA V2's disparity): clearly worse --
      close 13.4 / 16.8 mm (DA3 Large, 10-09 / 10-15) against DA V2 Large's 5.2 / 6.9; wide 8.4 against
      3.5-5.6. Worse with the DCW2's own floor too: its relief, not the fit (vs the DCW2: DA V2 Large
      correlates 0.957, DA3 Large 0.879, DA3 Mono 0.948). Untried: its depth anchored as depth (Z = s x + t),
      several views at once (it is made for that; the X3's engines take 3). DA V2 Large stays the best.
- [x] The duel's summary in PLAN.md S1 (2026-10-16): the table, the cause, what it means for the demo.
- [x] **The laptop's GPU** (2026-10-16): the root env has onnxruntime-gpu (CUDA 12 from nvidia-*
      wheels, `m3_duel.onnx_session`), the RTX 2070 Super runs DA V2 Small in 52 ms against the CPU's
      741 -- the duel's maps in seconds, not the hour of 2026-10-15. The test suites run no model:
      the GPU does not touch them.
- [ ] **The scan's cloud from the splat's cameras:** the splat now corrects each camera (median 3-6 mm,
      up to 17 mm on the can; `splat_poses.json` beside each scan). Fusing with those may fix the box
      coming out ~0.7 cm too tall and its stop-5/6 reading ~6 cm off.

## 4. If time is left

- [x] **The X3's reach far out** (2026-10-15, 74e398f): taught by hand as on the M3 --
      `x3plus/tools/x3_teach.py` (keys steer the open fingers over a piece, Enter saves), nine grips:
      the X3 reaches FURTHER than its model, 1 mm at 14.5 cm, 11-14 mm at 25 cm, a few mm left.
      roboarm/x3aim.py aims back by a line in the reach (8.7 mm rms against 12.5 raw); the console's
      "taught aim" setting: 1 on (now), 0 the old way. First clean-up with it: 11 picked, 0 missed.
      More grips taught LOW (fingers beside the piece, ~10-15 mm up) would sharpen it: most of the
      nine were saved 40-49 mm up, where the arm bends otherwise. Beyond ~25 cm nothing reaches.
- [ ] **Splat ghost:** from high up a see-through copy of the floor tape floats over the box -- the
      scan only looks from low down. Try a second, higher look at each stop (+~1 min a scan).
- [ ] **Your lab objects** (2026-10-09 list): the 3 electrical-tape rolls (measure their outside
      diameter), the screwdriver, the yellow cylinder -- Scan with a name teaches them, Splat shows
      them; picking them needs a grip beyond the kit's sizes (a tape roll is at the ~65 mm limit).
- [ ] The fusion on the robot (Open3D's aarch64 wheel), so a scan needs no laptop.

## hekate (the 5090)

- **Out of the lab and offline, back Friday evening (2026-10-23)** (the user). Until then GPU work
  runs on Dionysos and the laptop's GPU; what needs hekate waits: the splat (`tools/splat_cycle.py`,
  gsplat is built there -- so the two splat items in 3 and 4 above), pi0.5, Isaac Sim 6.0.
- [ ] **Monday 2026-10-26, first:** `ssh hekate` and its status script (CLAUDE.md) -- does it answer at
  its old address, is Isaac Sim's task still there, how much VRAM the other job leaves.
- **SSH works without Tailscale** (the user, 2026-10-20): `ssh hekate` over the lab LAN; nothing left
  to wait on from Rene.

## Carried over from the plan

- Where the shared station stands -- the box within both arms' reach (before stage 2, the Slovak
  mission).
- RESEARCH.md's open choices: quick-win venues, contacting the two Slovak groups, the overhead webcam.

## Done on 2026-10-13, 2026-10-15 and 2026-10-16

- **2026-10-13**, only hekate's jobs over lunch (the robot part waited): the depth duel's big models
  (f5d5e2b); YOLO26s + synthetic again, a repeat of 2026-10-12; `kit_cycle.py` reads hekate's output
  as UTF-8 (94daf9c).
- **2026-10-15:** the M3's camera found frozen since 2026-10-13; the bridge's driver watchdog, IR and
  point clouds off, deployed (bdbbc83) -- the stalls were the GGUF conversion beside it; the duel's
  pictures (44eed8b); news taught 2 more rounds; the console on the current code (Splat button); the
  kit detector checked on the X3's pictures, 185 new X3 looks in 4 rounds, the X3 labelling by
  projection + SAM 2 and `kit_cycle.py --x3` (fbf8639); the training sets go to hekate as one tar;
  the M3's board hang (power cycle); Teach no longer takes the left front tyre (2cf84bc, deployed);
  an X3 test round held out and the per-robot evaluation (4bf8405): `n_20261015` 0.497 against
  0.246 on the X3 test, 0.560 against 0.791 on the M3 val. Teach finished: every side of the four
  cubes, Find names them all, a new thing taught in 10 s and picked by name ~50 s after. The X3 serves
  the kit detector (11 picked in its first clean-up); 60 M3 frames with 40 mm cubes and blocks; a
  thing no plan reaches is planned once a job (9183510: on the M3 plan 6.4 s a pick with one such
  block on the table, 0.4 s without), deployed on both robots -- the X3's shared code brought up
  from 2026-10-08 with it (7 picked, 2 out of reach, the cylinder missed twice).
- **2026-10-16:** the frozen-camera /health merged (746ab04, 18a976b) and deployed on both robots; the
  floor fit without the warp and with refits (d14f9d0), checked on the held-out 2026-10-15 looks with
  DA V2 Large maps from hekate; the laptop's GPU in the root env (onnxruntime-gpu); `n_20261015b` on
  both robots -- the X3 picks with it, the M3 logs it beside its depth.
