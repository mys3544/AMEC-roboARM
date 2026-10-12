# RoboARM 2026

Plans: PLAN.md (both robots), x3plus/PLAN.md, m3pro/PLAN.md. What is left from the last session: NEXT.md.

## GPU PC "hekate" (Windows 11, RTX 5090)

- Reached as `ssh hekate`. For now `Host hekate` in `~/.ssh/config` points at its lab LAN address
  192.168.73.87 (works only on IISLab-AMEC): port 22 over Tailscale (`hekate.tail39f820.ts.net`,
  shared in from Rene's tailnet) times out on Rene's side, shields-up or ACL (2026-10-12). Switch the
  HostName back to the Tailscale name once Rene has fixed that.
- Run a command there: `ssh hekate "<PowerShell command>"` (the remote shell is Windows PowerShell).
- Project folder there: `C:\Users\daniel\Desktop\Sim Arm 2026` (it has its own CLAUDE.md with the rules for that machine).
- Status (GPU, Isaac Sim, ports, log), read-only:
  `ssh hekate "powershell -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\daniel\Desktop\Sim Arm 2026\tools\status.ps1'"`
- Start Isaac Sim (headless, streaming, Python server): `ssh hekate "schtasks /run /tn IsaacSimStreaming"`
  Never launch Isaac Sim directly over SSH: it would die with the SSH session and has no desktop session for the GPU.
  It is ready when status shows `Loaded: yes`.
- Run code inside the running sim: `python tools/isaac_exec.py --via hekate <script.py>` or `--via hekate -c "<code>"`.
  It opens and closes the SSH tunnel itself and talks to port 8230 on the PC.
- Daniel watches the sim with the Isaac Sim WebRTC Streaming Client connected to hekate (Tailscale IP, or the LAN address while Tailscale is blocked).
- For longer work on the PC (several steps, editing files there), message the Remote Control session "hekate"
  instead of driving everything over SSH.
- Our training there: `train/` (its own uv project, CUDA torch; NOT a workspace member -- never
  `uv run` inside train/ on this laptop, it installs hekate's torch here) is copied to
  `Sim Arm 2026\train` with its lock. The kit detector's whole cycle, from the repo root:
  `uv run python tools/kit_cycle.py [--size n|s] [--int8]` (fetch the M3's looks, build the set,
  train on hekate, ONNX and weights back to train/runs/kit/).
- Training runs on that GPU. Do not stop, kill or restart anything there that you did not start.
  Ask before starting anything GPU-heavy when less than ~16 GB VRAM is free, and before any firewall,
  SSH, scheduled-task or software change there.
