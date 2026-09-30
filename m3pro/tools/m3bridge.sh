#!/usr/bin/env bash
# The M3's bridge (tools/m3bridge.py) on the robot, detached, its log in ~/roboarm/logs/m3bridge.log.
#
#   bash tools/m3bridge.sh start [--camera]   # --camera: the DCW2's driver too (not with the mapping stack)
#   bash tools/m3bridge.sh status
#   bash tools/m3bridge.sh stop               # the arm holds where it is
#
# setsid first on the line: the bridge leads its own process group (the camera driver in it),
# which stop signals as a whole.
cd "$(dirname "$0")/.." || exit 1
PIDFILE=logs/m3bridge.pid
ROS="source /opt/ros/humble/setup.bash && source ~/yahboomcar_ws/install/setup.bash && source ~/M3Pro_ws/install/setup.bash && export ROS_DOMAIN_ID=30 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
case "$1" in
  start)
    shift
    mkdir -p logs
    if [ -f $PIDFILE ] && kill -0 -- "-$(cat $PIDFILE)" 2>/dev/null; then echo "already running"; exit 1; fi
    setsid nohup bash -c "$ROS && exec python3 tools/m3bridge.py $*" > logs/m3bridge.log 2>&1 < /dev/null &
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
