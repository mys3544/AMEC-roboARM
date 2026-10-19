"""kit_synth.py without a running Isaac Sim: its own headless SimulationApp, then generate(N) -- for a
machine where nothing is kept running for isaac_exec.py (Dionysos's Isaac Sim 5.0, 2026-10-19).

    C:\\isaac-sim-standalone-5.0.0-windows-x86_64\\python.bat run_standalone.py N [--seed S] [--out DIR]
"""

import argparse
import asyncio
import os
import sys
import time

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("n", type=int)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "synth_raw"))
a = ap.parse_args()

from isaacsim import SimulationApp  # before any omni import

began = time.perf_counter()
app = SimulationApp({"headless": True})
print(f"Isaac Sim up in {time.perf_counter() - began:.0f} s", flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kit_synth

began = time.perf_counter()
task = asyncio.ensure_future(kit_synth.generate(a.n, seed=a.seed, out=a.out))
while not task.done():   # the generator's awaits run as the app updates
    app.update()
task.result()
print(f"{a.n} frames in {time.perf_counter() - began:.0f} s -> {a.out}", flush=True)
app.close()
