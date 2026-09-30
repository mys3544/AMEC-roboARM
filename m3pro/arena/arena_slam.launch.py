"""Map the arena with the M3 Pro's two lidars: slam_toolbox on Yahboom's odometry filter, and
the drive page (drive_server.py) to move the robot while it maps.

    ros2 launch ~/arena/arena_slam.launch.py [name:=value ...]   (tools/arena_map.py start)

Arguments -- the same file maps live and replays a recorded run (replay.sh), so what is tuned on
a recording is what runs on the robot. The defaults are what tuning on 2026-09-30 chose (the
map within 1 cm of the fence all through a replay; Yahboom's settings: 2 cm too big, growing):
    sim:=true          use the recording's clock; no drive page
    ekf_hz:=N          the odometry filter's rate (30; Yahboom's is 7)
    ekf_predict:=true  the filter stamps odom -> base_footprint at the current time, predicted,
                       not at its last measurement -- a scan newer than that waits for a
                       transform, and slam_toolbox drops it when the next scan comes
    scans:=lidars      slam_toolbox takes each lidar's own scan (scan_relay.py, which also takes
                       off each lidar's range bias); scans:=merged, the merged /scan instead
    lidars:=FILE       where the lidars sit and their range biases (lidars.yaml)
    slam:=FILE         slam_toolbox's settings (slam_params.yaml)
    slam_set:="k=v;k=v"  some of them overridden, e.g. "resolution=0.01" (scan_queue_size=10
                       keeps the scans the late odometry makes it drop -- and wrecked the map)

Yahboom's own launch (slam_mapping slam_toolbox.launch.py) leaves out the lidars' static
transforms, merges at 1 degree, cuts at 4 m and maps at 5 cm. The merged, filtered /scan is made
here in any case: the drive page's safety stop reads it.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = {"sim": "false", "ekf_hz": "30", "ekf_predict": "false", "scans": "lidars",
            "lidars": os.path.join(HERE, "lidars.yaml"),
            "slam": os.path.join(HERE, "slam_params.yaml"), "slam_set": ""}


def static_tf(parent, child, x=0.0, y=0.0, z=0.0, yaw_deg=0.0):
    return Node(package="tf2_ros", executable="static_transform_publisher",
                name=f"tf_{child}", output="log",
                arguments=["--x", str(x), "--y", str(y), "--z", str(z),
                           "--yaw", str(yaw_deg * 3.141592653589793 / 180),
                           "--frame-id", parent, "--child-frame-id", child])


def nodes(context):
    arg = {k: LaunchConfiguration(k).perform(context) for k in DEFAULTS}
    sim = arg["sim"] == "true"
    clock = {"use_sim_time": sim}
    with open(arg["lidars"]) as f:
        lidars = yaml.safe_load(f)
    ekf_yaml = os.path.join(get_package_share_directory("ekf_bringup"), "params",
                            "yahboom_M3Pro_ekf.yaml")
    slam_set = {k: yaml.safe_load(v) for k, v in
                (pair.split("=", 1) for pair in arg["slam_set"].split(";") if pair)}
    out = [
        static_tf("base_footprint", "base_link", z=0.02),
        static_tf("base_link", "imu_frame", x=0.06, z=0.08),
        *[static_tf("base_link", frame, *pose) for frame, pose in lidars.items()
          if frame.endswith("_frame")],
        # /odom_raw + /imu/data_raw -> /odom and odom -> base_footprint
        Node(package="robot_localization", executable="ekf_node", name="ekf_filter_node",
             parameters=[ekf_yaml, {**clock, "frequency": float(arg["ekf_hz"]),
                                    "predict_to_current_time": arg["ekf_predict"] == "true"}],
             remappings=[("/odometry/filtered", "/odom")]),
        # /scan0 + /scan1 -> /scan_multi in base_link
        Node(package="ira_laser_tools", executable="laserscan_multi_merger",
             name="laserscan_multi_merger",
             parameters=[os.path.join(HERE, "merge_params.yaml"), clock]),
        # /scan_multi -> /scan, returns under 0.18 m (the chassis) blanked
        Node(package="yahboom_laser_filter", executable="laser_filter_node",
             name="laser_filter_node",
             parameters=[{"angle_min": -180.0, "angle_max": 180.0}, clock]),
        Node(package="slam_toolbox", executable="async_slam_toolbox_node", name="slam_toolbox",
             parameters=[arg["slam"], {**clock, "scan_topic":
                                       "/scan_lidars" if arg["scans"] == "lidars" else "/scan"},
                         slam_set],
             output="screen"),
    ]
    if arg["scans"] == "lidars":
        out.append(ExecuteProcess(cmd=["python3", os.path.join(HERE, "scan_relay.py"),
                                       arg["lidars"], "--ros-args",
                                       "-p", f"use_sim_time:={arg['sim']}"],
                                  output="screen"))
    if not sim:   # the browser page for driving it by hand or on a loop, http://<robot>:8095/
        out.append(ExecuteProcess(cmd=["python3", os.path.join(HERE, "drive_server.py")],
                                  output="screen"))
    return out


def generate_launch_description():
    return LaunchDescription([*[DeclareLaunchArgument(k, default_value=v)
                                for k, v in DEFAULTS.items()],
                              OpaqueFunction(function=nodes)])
