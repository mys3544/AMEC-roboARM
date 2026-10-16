# Next -- the list (as of 2026-10-15, evening)

Stage 1 (PLAN.md, "First show") has four demos: a coloured 3D scan in the console, "teach in 30 s,
picked out from look-alikes", real vs predicted depth, and the speed table. The scan (also as a
Gaussian splat) and the speed table (PLAN.md M5) are done; the depth duel has its table and its
pictures (3 below); the teach from look-alikes is done too (1 below: every cube named, Teach 10 s,
Teach to the piece put down ~50 s). Left for stage 1: the kit detector -- trained with the X3's
pictures too, twice as good on the X3, worse on the M3 (2 below) -- and the close look's depth
(3). The robot work (you) and the laptop/hekate work (me) run side by side.

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
- [ ] A separate session makes /health report a frozen camera ("Report a frozen M3 camera in the
      bridge's /health"): merge it -- it touches m3pro/tools/m3bridge.py, as bdbbc83 did.
- Before robot work: GET http://192.168.73.66:8761/rgbd twice -- the same `stamp` is a frozen camera.
- **The M3's board went silent at ~12:33** (no node /YB_Node topics: battery, odometry, arm), after
  a failed Teach had turned the arm towards the left front tyre (see 1). Restarting the micro-ROS
  agent (`~/start_agent.sh`'s command, in a desktop terminal) and the board's reset button brought
  back battery and wheels but NOT the arm's /arm6_joints subscription; only switching the robot off
  and on did (12:55: arm connected, camera fresh). Next time: st## 1. Teach from look-alikes -- done 2026-10-15

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

each with Scan + its name).

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
- [ ] **Decide:** a model per robot for now (proposed: the X3 `n_20261015`, the M3 `n_cycle`), or
      one model with the M3's pictures weighted up -- after the M3 frames below.
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
- [ ] **The run `n_20261015b`** (M3's new frames + X3 + synthetic, started ~14:15): judge it with
      `train/eval_kit.py` against `n_cycle` and `n_20261015` on the NEW M3 val (26) -- the old 0.791
      was on 14 pictures -- and on the X3 test; if it holds both, one model for both robots.
- [ ] **The M3** (agreed 2026-10-15) after its frames, as a vision service of its own whose answer
      the pick logs beside the depth's at every look (the depth stays in charge; the log is the score).

## 3. Me, while the robot work runs (laptop + hekate)

- [x] **Depth duel, the big models** (2026-10-13; train/depth/ on hekate, `m3pro/tools/m3_duel.py run`
      scores them): DA V2 Large takes the close look from 15.5 to 9.6 mm (named 19 -> 33 %), still
      over the 7 mm bar; wide 4.8 mm (2.7 mm and 95 % on the DCW2's floor). The metric models (Depth
      Pro, DA V2 Metric Indoor) read 2-4x too far at 20-30 cm.
- [x] **The duel's pictures** (2026-10-15): `m3_duel.py run --panels 16 --versus da2l` -> colour |
      DCW2 | DA V2 Small | DA V2 Large in m3pro/out/duel/; for the demo panel_wide_5 and panel_close_0.
- [ ] **Why the close look fails:** in the pictures both networks fail alike where a large thing (the
      black box, the tape rolls) fills the close view -- the floor fit, not only the network. Try the
      fit on floor pixels away from large raised things; DA3 (not tried yet); then the duel's summary
      in PLAN.md S1.
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

## Waiting on others

- **Rene: SSH to hekate over Tailscale.** `tailscale ping` answers, port 22 times out, hekate's
  firewall allows the Tailscale range -- so shields-up or the tailnet ACL on Rene's side (Tailscale
  there runs as AMEC-PC). Meanwhile `Host hekate` in `~/.ssh/config` is the lab LAN address
  192.168.73.87; switch it back to `hekate.tail39f820.ts.net` once fixed.

## Carried over from the plan

- Where the shared station stands -- the box within both arms' reach (before stage 2, the Slovak
  mission).
- RESEARCH.md's open choices: quick-win venues, contacting the two Slovak groups, the overhead webcam.

## Done on 2026-10-13 and 2026-10-15

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
