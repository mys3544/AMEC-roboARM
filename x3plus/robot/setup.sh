#!/usr/bin/env bash
# The X3 Plus's Python and services, on the robot, from ~/roboarm (tools/sync_robot.py puts
# everything there, with requirements.txt: this robot's pins from the workspace's uv.lock).
#
#   bash robot/setup.sh          # first time, and after the pins or the units change
#
# 1. ~/roboarm/.venv, a uv venv on JetPack's Python 3.10 that also sees the system's packages
#    (ROS 2 Humble's rclpy for driving, which PyPI does not have), with exactly the pins.
# 2. ~/roboarm/.venv-ml for the vision service: tools/ml_env.sh (JetPack's torch and TensorRT).
# 3. The systemd user units roboarm-vision and roboarm-x3bridge, enabled, so they start at boot (the
#    old panel, roboarm-web, disabled: the console is the one app, 2026-10-07).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

if ! grep -qs "include-system-site-packages = true" .venv/pyvenv.cfg; then
    rm -rf .venv
    uv venv .venv --python /usr/bin/python3.10 --system-site-packages --prompt roboarm-x3
fi
uv pip sync --python .venv/bin/python requirements.txt
bash tools/ml_env.sh

mkdir -p "$HOME/.config/systemd/user" out
cp robot/roboarm-*.service "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable roboarm-vision.service roboarm-x3bridge.service
systemctl --user disable roboarm-web.service 2>/dev/null || true
if ! cmp -s robot/jetson-clocks.service /etc/systemd/system/jetson-clocks.service; then
    echo "the clocks at maximum at every boot, once: sudo cp robot/jetson-clocks.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now jetson-clocks"
fi
if ! loginctl show-user "$USER" -p Linger | grep -q yes; then
    echo "the services start at boot only with lingering on; once: sudo loginctl enable-linger $USER"
fi
echo "start or restart them with: systemctl --user restart roboarm-vision roboarm-x3bridge"
