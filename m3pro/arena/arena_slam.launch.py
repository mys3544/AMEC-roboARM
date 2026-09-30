"""Map the arena with the M3 Pro's two lidars: slam_toolbox on Yahboom's odometry and scan merge,
and the drive page (drive_server.py) to move the robot while it maps.

    ros2 launch ~/arena/arena_slam.launch.py      (tools/arena_map.py start does this)

Yahboom's own launch (slam_mapping slam_toolbox.launch.py) leaves out the lidars' static
transforms, merges at 1 degree, cuts at 4 m and maps at 5 cm; this one keeps their EKF and
chassis filter and changes those for a 3.2 x 2.8 m arena (merge_params.yaml, slam_params.yaml).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))


def include(package, name):
    path = os.path.join(get_package_share_directory(package), "launch", name)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path))


def generate_launch_description():
    return LaunchDescription([
        # base_footprint -> base_link -> laser0_frame, laser1_frame, imu_frame
        include("m3pro_bringup", "static_tf.launch.py"),
        # /odom_raw + /imu/data_raw -> /odom and odom -> base_footprint
        include("ekf_bringup", "ekf.launch.py"),
        # /scan0 + /scan1 -> /scan_multi in base_link
        Node(package="ira_laser_tools", executable="laserscan_multi_merger",
             name="laserscan_multi_merger", parameters=[os.path.join(HERE, "merge_params.yaml")]),
        # /scan_multi -> /scan, returns under 0.18 m (the chassis) blanked
        Node(package="yahboom_laser_filter", executable="laser_filter_node",
             name="laser_filter_node", parameters=[{"angle_min": -180.0, "angle_max": 180.0}]),
        Node(package="slam_toolbox", executable="async_slam_toolbox_node", name="slam_toolbox",
             parameters=[os.path.join(HERE, "slam_params.yaml")], output="screen"),
        # the browser page for driving it by hand or on a loop, http://<robot>:8095/
        ExecuteProcess(cmd=["python3", os.path.join(HERE, "drive_server.py")], output="screen"),
    ])
