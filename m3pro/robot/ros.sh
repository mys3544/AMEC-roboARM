# ROS 2 Humble and Yahboom's workspaces, for every process of ours that talks ROS on the M3
# (robot/roboarm-m3bridge.service, tools/m3bridge.sh). Sourced, not run. LF line ends.
source /opt/ros/humble/setup.bash
source ~/yahboomcar_ws/install/setup.bash
source ~/M3Pro_ws/install/setup.bash
export ROS_DOMAIN_ID=30 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
