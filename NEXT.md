# What is left for later -- from the 2026-10-09 session, to pick up on Monday 2026-10-12

Things set aside on the day, in the order they unblock each other. The background is in PLAN.md
(v2.4, stage 1) and the commits of 2026-10-09 on branch `m3pro`.

## Waiting on you (setup or hardware first)

1. **hekate, setup phase 2.** On hekate, in `C:\Users\daniel\Desktop\Sim Arm 2026`:
   `powershell -ExecutionPolicy Bypass -File .\setup\setup-pc.ps1 -LaptopPublicKey "<this laptop's
   ~/.ssh/id_ed25519.pub>" -RemoteControlAtLogon`, then say **"ready"**. Claude then: `ssh hekate
   hostname` (must print HEKATE, no password), copies `tools\isaac_exec.py` here and merges
   `laptop\CLAUDE-laptop-snippet.md` into this project's CLAUDE.md, and runs the snippet's status
   command. Isaac Sim is not started, and nothing on hekate is stopped or changed without asking.
   The laptop's `~/.ssh/config` already has `Host hekate` (the Tailscale name
   `hekate.tail39f820.ts.net`).
2. **The first live object scan.** The M3 on the floor -- the arena mat if possible, its print
   helps line the views up -- the charging cable OFF (it drives), ~1.2 m clear all round, one object
   ~38 cm straight ahead of its front (matte and colourful, up to ~15 cm tall; a 20 cm one needs
   `--radius 0.42`). Then `m3pro/tools/m3_scan.py start`, `fetch`, `fuse`: the cloud shows in the
   console's 3D view. Built and tested in simulation only so far.

## Then

3. **The kit detector and the model table** (stage 1, M5), on hekate: YOLO26n/s trained on the
   M3's labelled frames (every pick look keeps `*_rgb.jpg` + depth + outlines since 2026-10-09; ~90
   so far -- a few more clean-up runs give more), INT8 engines, timed on both robots.
4. **Teach your lab objects:** the 3 electrical-tape rolls (measure their outside diameter), the
   screwdriver, the yellow cylinder. The pick first needs a grip for things that are not the kit's
   sizes: a tape roll is at the gripper's ~65 mm limit (or pinched by its wall), the screwdriver
   by its handle.
5. **Object scan, after its first run:** a Scan button in the console; the fusion on the robot
   (Open3D's aarch64 wheel); the teach taken from the scan's own views.
6. **Teach follow-ups:** re-teach "news" from another angle (one news cube was named only 0.04
   ahead of the next name, under the 0.05 needed); measure the seconds from Teach to the first pick
   (the paper's headline number); set the unknown threshold on more scenes.
7. **Depth duel, the open question:** why Depth Anything reads heights 15.5 mm off on the close
   look though its floor fit holds (PLAN.md S1).
8. *Optional:* a few more taught grips with blocks and in the corners (`m3pro/tools/m3_teach.py`)
   to firm up the aim table (`m3pro/roboarm/m3aim.py`).

## Carried over from the plan (not from this day)

- Where the shared station stands -- the box within both arms' reach (before stage 2, the Slovak
  mission).
- RESEARCH.md's open choices: quick-win venues, contacting the two Slovak groups, the overhead webcam.
- Branch `m3pro` is 12 commits ahead of GitHub (not pushed).
