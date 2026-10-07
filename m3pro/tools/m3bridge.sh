#!/usr/bin/env bash
# The M3's bridge (tools/m3bridge.py) on the robot, detached, its log in ~/roboarm/logs/m3bridge.log.
# By hand: the usual way is the service, `systemctl --user start roboarm-m3bridge` (robot/), which
# runs it with --camera. This script is for a bridge WITHOUT the camera, beside the mapping stack.
#
#   bash tools/m3bridge.sh start [--camera]   # --camera: the DCW2's driver too (not with the mapping stack)
#   bash tools/m3bridge.sh status
#   bash tools/m3bridge.sh stop               # the arm holds where it is
#
# setsid first on the line: the bridge leads its own process group (the camera driver in it),
# which stop signals as a whole.
cd "$(dirname "$0")/.." || exit 1
PIDFILE=logs/m3bridge.pid
ROS="source robot/ros.sh"
PY=.venv/bin/python   # robot/setup.sh: the uv venv that also sees ROS 2's packages
[ -x $PY ] || PY=python3
case "$1" in
  start)
    shift
    mkdir -p logs
    if [ -f $PIDFILE ] && kill -0 -- "-$(cat $PIDFILE)" 2>/dev/null; then echo "already running"; exit 1; fi
    if systemctl --user -q is-active roboarm-m3bridge 2>/dev/null; then echo "the roboarm-m3bridge service runs it already"; exit 1; fi
    setsid nohup bash -c "$ROS && exec $PY tools/m3bridge.py $*" > logs/m3bridge.log 2>&1 < /dev/null &
    echo $! > $PIDFILE
    sleep 10
    tail -n 3 logs/m3bridge.log
    ;;
  stop)
    [ -f $PIDFILE ] || { echo "not running"; exit 0; }
    kill -INT -- "-$(cat $PIDFILE)" 2>/dev/null
    for _ in 1 2 3 4 5 6; do kill -0 -- "-$(cat $PIDFILE)" 2>/dev/null || break; sleep 1; done
    kill -TERM -- "-$(cat $PIDFILE)" 2>/dev/null
    rm -f $PIDFILE
    tail -n 2 logs/m3bridge.log
    ;;
  status)
    if [ -f $PIDFILE ] && kill -0 -- "-$(cat $PIDFILE)" 2>/dev/null; then echo "running (group $(cat $PIDFILE))"; else echo "not running"; fi
    curl -s -m 3 localhost:8761/health; echo
    ;;
  *) echo "usage: bash tools/m3bridge.sh start [--camera] | status | stop"; exit 2 ;;
esac
