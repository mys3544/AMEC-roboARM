# What is left for later -- from the 2026-10-12 session

Things set aside at the end of the day, roughly in the order they unblock each other. The background
is in PLAN.md (v2.4, stage 1) and the commits of 2026-10-12 on branch `m3pro` (all pushed).

## Done on 2026-10-12

- **hekate** set up (phase 2): `ssh hekate` works with the key, uv 0.13 under `daniel`, a training
  project in `Sim Arm 2026\train` (this repo's `train/`). Over the lab LAN for now -- see below.
- **The kit detector, first pass:** `m3pro/tools/kit_dataset.py` -> YOLO26n/s-seg trained on hekate
  (`train/train_kit.py`): mAP50 0.94 on the 30 mm cube; the 40 mm cube (19 examples) and the block
  cannot be judged yet. ONNX at 480x640 in `train/runs/kit/` (not in git).
- **The object scan works live:** a first look that holds a 20 cm thing, the thing read again at
  every stop (the wheels drift 1-3 cm a stop, ~10 cm a circle), the views anchored on it when fused,
  a size correction only for views that do not see the thing's top. A 500 ml can: 16.7 cm tall, 31 mm
  round (33 true); the Jetson box: 16.7 x 11.5 x 4.6 cm (16.4 x 11 x 4 true).
- **Console:** Scan, Fuse last scan, Show in 3D, Teach from last scan on the M3 arm card; typing in
  a field no longer drives the base or STOPs ALL.
- **Teach from a scan:** a name + Scan teaches the thing from its 24-36 views all round; Find scores
  a name by its best 3 views. 'jetson box' and 'monster can' taught; Find names the box (0.78-0.79).
- **Find looks closer** at what the wide look cannot name; **the bridge refuses a frozen camera**
  (the DCW2 driver stopped once; `systemctl --user restart roboarm-m3bridge` brings it back).

## Next

1. **Charge the M3** -- it ended the day at 10.72 V (the arm refuses below 10.5).
2. **Re-teach the picture cubes from all sides.** Find with the box, the can and the two news cubes:
   the box named, both news cubes unknown even after the closer look (0.64 with fish 0.01 behind;
   0.61, news first but under 0.65). They were taught on 2026-10-09 from one side (5-8 views), and
   Find sees other faces. Either a scan of a cube (3 cm is near the scan's lower limit: try it) or a
   standing teach that looks at it from several sides. Then Find again with the same layout, and the
   seconds from Teach to the first pick (the paper's headline number).
3. **More kit frames, then the detector again:** a few pick runs with 40 mm cubes and blocks out
   (every pick look keeps its labelled frame), then `uv run python tools/kit_cycle.py --int8` from the
   repo root. Afternoon results (val: 14 real pictures from 8 sessions, only 3 of them 40 mm cubes),
   mask mAP50: YOLO26n real only 0.653; + 1500 Isaac Sim pictures with the camera at the robot's
   looks, then fine-tuned on real, **0.791** (40 mm cube 0.343 -> 0.672; `n-synth2-ft`); INT8 QAT
   costs ~0.014; RF-DETR-Seg-Nano 0.683. The speed half of the model table is in PLAN.md M5 (both
   robots: YOLO26n-seg FP16 3.0 ms, INT8 QAT 3.2 -- slower, 100 reformats --, YOLO26s 5.4, YOLOE-26s
   engine 5.4, RF-DETR-Seg-N 8.0, YOLO26n-depth 3.8; `train/robot_bench.sh`). Next: a 40 mm cube
   validation worth the name, synthetic pictures in `kit_cycle.py`, the kit detector served by the
   robots' vision services (FP16 engine), end-to-end times, YOLO26n-depth in the depth duel.
4. **Scan loose ends:** the Jetson box reads ~6 cm off at one stop (stop 5-6, both times -- one side
   of it?); the box comes out ~0.7 cm too tall; the can's lid comes out open (shiny, no depth); the
   fusion still runs on the laptop (Open3D's aarch64 wheel would put it on the robot).
5. **Teach your lab objects** (from the 2026-10-09 list): the 3 electrical-tape rolls (measure their
   outside diameter), the screwdriver, the yellow cylinder -- scanned and taught now; the pick first
   needs a grip for things that are not the kit's sizes (a tape roll is at the gripper's ~65 mm limit).
6. **Depth duel, the open question:** why Depth Anything reads heights 15.5 mm off on the close look
   though its floor fit holds (PLAN.md S1).

## Waiting on others

- **Rene: SSH to hekate over Tailscale.** `tailscale ping` answers, port 22 times out, hekate's
  firewall allows the Tailscale range -- so shields-up or the tailnet ACL on Rene's side (Tailscale
  there runs as AMEC-PC). Meanwhile `Host hekate` in `~/.ssh/config` is the lab LAN address
  192.168.73.87; switch it back to `hekate.tail39f820.ts.net` once fixed.

## Carried over from the plan

- Where the shared station stands -- the box within both arms' reach (before stage 2, the Slovak
  mission).
- RESEARCH.md's open choices: quick-win venues, contacting the two Slovak groups, the overhead webcam.
