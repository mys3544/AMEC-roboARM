# Tomorrow, 2026-10-13 -- the list

Stage 1 (PLAN.md, "First show") has four demos: a coloured 3D scan in the console, "teach in 30 s,
picked out from look-alikes", real vs predicted depth, and the speed table. The scan (now also as a
Gaussian splat) and the speed table (PLAN.md M5) are done. The list below goes for the other two,
and feeds the kit detector on the way. The robot work (you) and the laptop/hekate work (me) run side
by side.

## 0. Before we start (~5 min)

- [ ] **M3 charged** -- it was charging on 2026-10-12 after running low (the arm refuses below 10.5 V).
- [ ] **Restart the console** -- the one started 2026-10-12 at 15:15 runs the old code (no Splat
      button): Ctrl-C in its terminal, then `.venv/Scripts/python console/serve.py` from the repo root.
- [ ] I check hekate (`tools\status.ps1`: GPU free, nobody training) and the M3's bridge and camera.

## 1. Teach from look-alikes -- you at the robot, me at the console (~45 min)

The picture cubes (news, fish, cosmetics, apricot) were taught on 2026-10-09 from one side only
(5-8 views), so Find does not know their other faces: on 2026-10-12 both news cubes stayed unknown
(0.64 and 0.61, under the 0.65 bar). No code needed -- Teach again with the same name ADDS views.

- [ ] For each of the 4 cubes: alone ~20-25 cm before the bumper, name in the field, **Teach**; turn
      it a quarter on the spot, **Teach** again with the same name; 4 rounds (a 5th, tipped over, is
      a bonus). ~1 min a round, ~20 min in all.
- [ ] **Find** with the box, the can, both news cubes and the fish together: all named?
- [ ] **The headline number:** a new thing (one never taught) -- Teach, then Pick it by name; I time
      Teach -> first pick from the log. The demo says "teach in 30 s".
- [ ] If a cube is still unknown: I look at its scores and views (out/teach/NAME/) and we decide
      (more views, a lower bar, or a scan-teach with Scan + its name).

## 2. Frames for the kit detector -- you (~30 min), then hekate trains on its own (~1 h)

The detector's validation had only 3 pictures with a 40 mm cube and the block is barely judged.

- [ ] Pick runs with **40 mm cubes** and **the 30x30x60 block** out (each pick look keeps its labelled
      frame): ~15 looks each, cubes in different places and turns.
- [ ] I run `uv run python tools/kit_cycle.py --synth --int8` and compare it with today's best
      (YOLO26n + Isaac Sim pictures, then real: mask mAP50 **0.791**; 40 mm cube 0.672).

## 3. Me, while the robot work runs (laptop + hekate)

- [ ] **Depth duel** (stage 1's "real vs predicted depth"): why Depth Anything reads heights 15.5 mm
      off on the close look though its floor fit holds (PLAN.md S1) -- from the saved looks if they
      hold enough, else a few new looks on the robot; then the duel's table and picture for the demo.
- [ ] **The scan's cloud from the splat's cameras:** the splat now corrects each camera (median 3-6 mm,
      up to 17 mm on the can; `splat_poses.json` beside each scan). Fusing with those may fix the box
      coming out ~0.7 cm too tall and its stop-5/6 reading ~6 cm off.
- [ ] **The kit detector on the robots** (FP16 engine, 3.0 ms on both Orins) in their vision services
      -- once step 2 says which model.

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

## Done on 2026-10-12

- **hekate** set up over the lab LAN: key login, uv, `train/` as its training project; VS 2022 Build
  Tools and CUDA 12.8 for gsplat.
- **The object scan works live:** re-read at every stop, anchored fusion, size correction only where
  the top is unseen. Can 16.7 cm x 31 mm (33 true); Jetson box 16.7 x 11.5 x 4.6 cm (16.4 x 11 x 4).
- **Console:** Scan, Fuse last scan, Teach from last scan, **Splat last scan**, Show in 3D; the 3D view
  draws Gaussian splats; typing in a field no longer drives the base.
- **Teach from a scan** (name + Scan); Find scores the best 3 views and looks closer at unknowns; the
  bridge refuses a frozen camera.
- **Kit detector:** YOLO26n-seg with Isaac Sim pictures (camera at the robot's looks) 0.791 mask
  mAP50; `tools/kit_cycle.py` does the whole cycle; the model table on both robots (PLAN.md M5).
- **Gaussian splats on hekate:** depth supervision (held-out depth 64 -> 6 mm), cropped to the scan,
  pose refinement (+1 dB on both); `tools/splat_cycle.py` or the console button, ~2 min a scan.
  In the console: `splat_box_20261012_172202`, `splat_can_20261012_165048`.
