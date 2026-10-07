# RoboARM 2026

Two Yahboom robots in the lab, each with its own folder, and one console for both.

| folder | what | start here |
|---|---|---|
| [`x3plus/`](x3plus/) | **ROSMASTER X3 Plus** (192.168.73.210): wrist camera + Depth Anything V2, panel, bridge, vision container | [x3plus/README.md](x3plus/README.md), [x3plus/PLAN.md](x3plus/PLAN.md), [x3plus/docs/STATUS.md](x3plus/docs/STATUS.md) |
| [`m3pro/`](m3pro/) | **ROSMASTER M3 Pro** (192.168.73.66): DaBai DCW2 RGB-D on the arm, pick service, arena mapping | [m3pro/README.md](m3pro/README.md) |
| [`console/`](console/) | one page on the laptop for both robots, voice push-to-talk | [console/README.md](console/README.md) |

Plans for both robots: [PLAN.md](PLAN.md) (v2.3, the arena, voice, models) and
[RESEARCH.md](RESEARCH.md) (paper-worthy tracks).

## Python

All of it is one **uv workspace** ([pyproject.toml](pyproject.toml)): the members `console`,
`x3plus` (project `roboarm-x3`) and `m3pro` (`roboarm-m3`) share one `uv.lock`. On the laptop one
`.venv` at the top holds all three, with pytest, ruff and paramiko (the robot tools' SSH):

```bash
uv sync --frozen                                  # here, at the top: the one .venv
cd x3plus && uv run python -m pytest -q           # each folder's tests, from that folder
uv run python tools/webapp.py --sim --port 8090
uv run python tools/sync_robot.py                 # push to the robot, with its pins
```

Each robot folder is still copied flat to `~/roboarm` on its robot, so the robots' layout does not
depend on this repo's. With it goes `requirements.txt`: that robot's pins exported from the lock
(`uv export --package roboarm-x3`); the robot installs exactly those into its own uv venv. On the
robots (Linux, aarch64) the lock keeps numpy 1.x: ROS 2 Humble's message modules and JetPack's
torch and TensorRT, which come from the system and not from PyPI, were built against it.
