#!/bin/bash
# Replay a recorded run (tools/arena_map.py record) through the mapping stack with the given
# launch arguments, on a ROS domain of its own -- so several run side by side and none of them
# sees the live robot. The map is saved and the fence fitted every 20 s (OUT/t20, t40, ...: does
# the map grow as the robot drives?) and at the end (OUT/map, OUT/fit.json):
#
#     replay.sh BAG OUT DOMAIN [name:=value ...]      (tools/arena_map.py tune runs it)
# (no `set -u`: ROS's setup.bash reads variables that are not set)
BAG=$1 OUT=$2
export ROS_DOMAIN_ID=$3 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
shift 3
source /opt/ros/humble/setup.bash
source ~/yahboomcar_ws/install/setup.bash
source ~/M3Pro_ws/install/setup.bash
HERE=$(dirname "$(readlink -f "$0")")
mkdir -p "$OUT"

snapshot() {   # snapshot NAME: the map as it stands, and its fence fit
    timeout 30 ros2 run nav2_map_server map_saver_cli --free 0.196 --occ 0.65 -f "$OUT/$1" \
        > "$OUT/$1.saver.log" 2>&1 \
        && python3 "$HERE/fence_fit.py" "$OUT/$1.yaml" > "$OUT/$1.fit.json" 2> /dev/null
}

# setsid: the stack leads its own process group, ended as one below
setsid ros2 launch "$HERE/arena_slam.launch.py" sim:=true "$@" > "$OUT/launch.log" 2>&1 &
STACK=$!
sleep 8
ros2 bag play "$BAG" --clock --topics /scan0 /scan1 /odom_raw /imu/data_raw > "$OUT/play.log" 2>&1 &
PLAY=$!
T=0
while kill -0 $PLAY 2> /dev/null; do
    sleep 20
    T=$((T + 20))
    kill -0 $PLAY 2> /dev/null && snapshot "t$T"
done
sleep 8   # the last scans through slam_toolbox
snapshot map
python3 "$HERE/fence_fit.py" "$OUT/map.yaml" --png "$OUT/map_fit.png" > /dev/null
cp "$OUT/map.fit.json" "$OUT/fit.json"
kill -INT -- -$STACK
sleep 6
kill -TERM -- -$STACK 2> /dev/null
grep -c "dropping message" "$OUT/launch.log" > "$OUT/dropped.txt"
echo "done: $OUT"
