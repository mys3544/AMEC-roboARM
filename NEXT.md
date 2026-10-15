# Next -- the list (as of 2026-10-15, afternoon)

Stage 1 (PLAN.md, "First show") has four demos: a coloured 3D scan in the console, "teach in 30 s,
picked out from look-alikes", real vs predicted depth, and the speed table. The scan (also as a
Gaussian splat) and the speed table (PLAN.md M5) are done; the depth duel has its table and its
pictures (3 below). Left: the teach from look-alikes -- started 2026-10-15, stopped by the M3's
camera (0 below), which holds again -- and the kit detector, now trained with the X3's pictures
too (2 below). The robot work (you) and the laptop/hekate work (me) run side by side.

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

## 1. Teach from look-alikes -- you at the robot, me at the console (~30 min left)

The picture cubes (news, fish, cosmetics, apricot) were taught on 2026-10-09 from one side only
(5-8 views), so Find does not know their other faces: on 2026-10-12 both news cubes stayed unknown
(0.64 and 0.61, under the 0.65 bar). No code needed -- Teach again with the same name ADDS views.
It works on a table: Teach, Find and Pick by name move the arm only (leave "may drive closer" off,
no Scan). A round takes ~5 s (4 views, the arm only turns).

- [ ] **news:** 2 of 4 rounds done 2026-10-15 -- 13 views now (5 old + 8; the new ones agree
      0.95-0.96); the 3rd failed on the camera. Two more, the second news cube for one of them.
- [ ] **fish, cosmetics, apricot:** alone ~20-25 cm before the bumper, name in the field, **Teach**;
      turn it a quarter on the spot, **Teach** again with the same name; 4 rounds each (a 5th,
      tipped over, is a bonus).
- [ ] **Find** with the box, the can, both news cubes and the fish together: all named?
- [ ] **The headline number:** a new thing (one never taught) -- Teach, then Pick it by name; I time
      Teach -> first pick from the log. The demo says "teach in 30 s".
- [ ] If a cube is still unknown: I look at its scores and views (out/teach/NAME/) and we decide
      (more views, a lower bar, or a scan-teach with Scan + its name).

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
- [ ] **The run `n_20261015`** (`kit_cycle.py --synth --x3 --no-fetch`, started ~12:40): compare it
      with the best so far (mask mAP50 **0.791**; 40 mm cube 0.672) on the M3's val and the X3's
      val SEPARATELY (the set's val mixes both now), and look at it on the tag board again.
- [ ] **M3 frames** with 40 mm cubes and the 30x30x60 block out (~15 looks each) -- still to do.
- [ ] **Onto the robots** (agreed 2026-10-15): the X3 once the X3 val says it no longer sees cubes in
      the board -- its vision service takes it as is (ROBOARM_VISION_MODEL, a 640x640 engine built
      on the robot); the M3 after the M3 frames, as a vision service of its own whose answer the
      pick logs beside the depth's at every look (the depth stays in charge; the log is the score).

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
  projection + SAM 2 and `kit_cycle.py --x3` (fbf8639); the training sets go to hekate as one tar.
