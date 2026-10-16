#!/usr/bin/env bash
# The M3 Pro's Python and service, on the robot, from ~/roboarm (tools/sync_robot.py puts
# everything there, with requirements.txt: this robot's pins from the workspace's uv.lock).
#
#   bash robot/setup.sh          # first time, and after the pins or the unit change
#
# 1. ~/roboarm/.venv, a uv venv on the system's Python 3.10 that also sees the system's packages
#    (ROS 2 Humble's rclpy and Yahboom's arm_msgs, which PyPI does not have), with exactly the pins.
# 2. The systemd user unit roboarm-m3bridge, enabled, so it starts at boot.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

if ! grep -qs "include-system-site-packages = true" .venv/pyvenv.cfg; then
    rm -rf .venv
    uv venv .venv --python /usr/bin/python3.10 --system-site-packages --prompt roboarm-m3
fi
uv pip sync --python .venv/bin/python requirements.txt

mkdir -p "$HOME/.config/systemd/user" out
cp robot/roboarm-*.service "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable roboarm-m3bridge.service
systemctl --user enable roboarm-m3teach.service   # the teach's YOLOE, from .venv-ml (tools/ml_env.sh first)
systemctl --user enable roboarm-vision.service    # the kit detector's second opinion (m3kit), from .venv-ml
if ! cmp -s robot/jetson-clocks.service /etc/systemd/system/jetson-clocks.service; then
    echo "the clocks at maximum at every boot, once: sudo cp robot/jetson-clocks.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now jetson-clocks"
fi
if ! loginctl show-user "$USER" -p Linger | grep -q yes; then
    echo "the service starts at boot only with lingering on; once: sudo loginctl enable-linger $USER"
fi
echo "start or restart it with: systemctl --user restart roboarm-m3bridge"
