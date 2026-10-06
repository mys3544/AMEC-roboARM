# RoboARM 2026

Two Yahboom robots in the lab, each with its own folder, and one console for both.

| folder | what | start here |
|---|---|---|
| [`x3plus/`](x3plus/) | **ROSMASTER X3 Plus** (192.168.73.210): wrist camera + Depth Anything V2, panel, bridge, vision container | [x3plus/README.md](x3plus/README.md), [x3plus/PLAN.md](x3plus/PLAN.md), [x3plus/docs/STATUS.md](x3plus/docs/STATUS.md) |
| [`m3pro/`](m3pro/) | **ROSMASTER M3 Pro** (192.168.73.66): DaBai DCW2 RGB-D on the arm, pick service, arena mapping | [m3pro/README.md](m3pro/README.md) |
| [`console/`](console/) | one page on the laptop for both robots, voice push-to-talk | [console/README.md](console/README.md) |

Plans for both robots: [PLAN.md](PLAN.md) (v2.3, the arena, voice, models) and
[RESEARCH.md](RESEARCH.md) (paper-worthy tracks).

Each robot folder is its own uv project with its own `pyproject.toml`, `uv.lock`, `tests/`,
`compose.yaml` and `tools/sync_robot.py`. Both are copied to `~/roboarm` on their robot, so the
layout on the robots does not depend on this repo's. Run a robot's commands from its folder:

```bash
cd x3plus
uv sync --frozen
uv run python -m pytest -q
uv run python tools/webapp.py --sim --port 8090
```

On the lab laptop one shared `.venv` at the top serves all three (`../.venv/Scripts/python`
from a robot folder, `.venv/Scripts/python console/serve.py` from here).
