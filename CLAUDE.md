# RoboARM 2026

Plans: PLAN.md (both robots), x3plus/PLAN.md, m3pro/PLAN.md. What is left from the last session: NEXT.md.

## GPU PC "hekate" (Windows 11, RTX 5090)

- Reached as `ssh hekate`: `Host hekate` in `~/.ssh/config` is its lab LAN address 192.168.73.11
  since 2026-10-26 (DHCP: it was .87, then .88 -- if ssh times out, ask the user for the new one; works
  only on IISLab-AMEC; the host key is kept under the old .87). No Tailscale needed (2026-10-20).
- Run a command there: `ssh hekate "<PowerShell command>"` (the remote shell is Windows PowerShell).
- Project folder there: `C:\Users\daniel\Desktop\Sim Arm 2026` (it has its own CLAUDE.md with the rules for that machine).
- Status (GPU, Isaac Sim, ports, log), read-only:
  `ssh hekate "powershell -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\daniel\Desktop\Sim Arm 2026\tools\status.ps1'"`
- Start Isaac Sim (headless, streaming, Python server): `ssh hekate "schtasks /run /tn IsaacSimStreaming"`
  Never launch Isaac Sim directly over SSH: it would die with the SSH session and has no desktop session for the GPU.
  It is ready when status shows `Loaded: yes`.
  To stop it: `post_quit()` through isaac_exec hung for minutes in the streaming app (2026-10-12);
  `ssh hekate "schtasks /end /tn IsaacSimStreaming"`, then stop the kit.exe that task started (check
  its start time -- never another instance's).
- Synthetic kit pictures: `train/synth/kit_synth.py` inside the running sim (see its docstring).
- Run code inside the running sim: `python tools/isaac_exec.py --via hekate <script.py>` or `--via hekate -c "<code>"`.
  It opens and closes the SSH tunnel itself and talks to port 8230 on the PC.
- Daniel watches the sim with the Isaac Sim WebRTC Streaming Client connected to hekate's LAN address.
- For longer work on the PC (several steps, editing files there), message the Remote Control session "hekate"
  instead of driving everything over SSH.
- Our training there: `train/` (its own uv project, CUDA torch; NOT a workspace member -- never
  `uv run` inside train/ on this laptop, it installs hekate's torch here) is copied to
  `Sim Arm 2026\train` with its lock. The kit detector's whole cycle, from the repo root:
  `uv run python tools/kit_cycle.py [--size n|s] [--int8]` (fetch the M3's looks, build the set,
  train on hekate, ONNX and weights back to train/runs/kit/).
  An object scan's Gaussian splat: `uv run python tools/splat_cycle.py [STAMP] [--name THING]` (the
  newest scan in m3pro/out/scan -> fuse --export -> train/splat/ on hekate, ~1 min -> the .splat into
  the console's 3D view).
- Training runs on that GPU. Do not stop, kill or restart anything there that you did not start.
  Ask before starting anything GPU-heavy when less than ~16 GB VRAM is free, and before any firewall,
  SSH, scheduled-task or software change there.
