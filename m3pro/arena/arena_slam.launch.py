"""Map the arena with the M3 Pro's two lidars: slam_toolbox on Yahboom's odometry filter, and
the drive page (drive_server.py) to move the robot while it maps.

    ros2 launch ~/arena/arena_slam.launch.py [name:=value ...]   (tools/arena_map.py start)

Arguments -- the same file maps live and replays a recorded run (replay.sh), so what is tuned on
a recording is what runs on the robot. The defaults are what tuning on 2026-09-30 chose, on the
live run loop4 replayed: Yahboom's merge of the two lidars with each one's range bias taken off,
Yahboom's EKF rate, and slam_params.yaml's 20 cm between key scans -- the fence 3.200 x 2.785 m
at the end, its long side moving 0.6 cm over the loop (Yahboom's settings: +1.9 cm, +1.1):
    sim:=true          use the recording's clock; no drive page
    motion:=false      bench mode: the drive page shows everything and refuses to move (on a
                       table the lidars cannot see the edge)
    ekf_hz:=N          the odometry filter's rate (7, Yahboom's; 30 let the map grow more)
    ekf_predict:=true  the filter stamps odom -> base_footprint at the current time, predicted,
                       not at its last measurement -- a scan newer than that waits for a
                       transform, and slam_toolbox drops it when the next scan comes
    scans:=lidars      slam_toolbox takes each lidar's own scan (scan_relay.py, which also takes
                       off each lidar's range bias); scans:=merged, the merged /scan instead.
                       Live, lidars drops nearly every scan (scan_relay.py says why)
    bias:=false        merge the lidars' raw scans (Yahboom's), not the bias-corrected ones
    lidars:=FILE       where the lidars sit and their range biases (lidars.yaml)
    slam:=FILE         slam_toolbox's settings (slam_params.yaml)
    slam_set:="k=v;k=v"  some of them overridden, e.g. "resolution=0.01" (scan_queue_size=10
                       keeps the scans the late odometry makes it drop -- and wrecked the map)
    mapper:=rtabmap    RTAB-Map instead of slam_toolbox: the coloured 3D arena (plan M2.1a
                       level 2). Starts the DaBai (depth aligned to colour); the lidar /scan
                       aligns the key frames (ICP), the camera colours them. The arm must be in
                       the pose camera:= describes before it starts (arena_map.py arm mapping)
    camera:="x,y,z,yaw,pitch"  base_link -> camera_link in m and degrees, for the arm's mapping
                       pose [90, 180, 5, 0, 90, 0]: Yahboom's 0.05,0,0.38,0,0 with the height
                       and tilt camera_check.py measured from the floor (0.385 m up, 0.7 deg
                       above level, 2026-09-30)
    rtab_db:=FILE      RTAB-Map's database (~/arena/rtab/arena.db), started afresh

Yahboom's own launch (slam_mapping slam_toolbox.launch.py) leaves out the lidars' static
transforms, merges at 1 degree, cuts at 4 m and maps at 5 cm. The merged, filtered /scan is made
here in any case: the drive page's safety stop reads it.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = {"sim": "false", "motion": "true", "ekf_hz": "7", "ekf_predict": "false",
            "scans": "merged", "bias": "true",
            "lidars": os.path.join(HERE, "lidars.yaml"),
            "slam": os.path.join(HERE, "slam_params.yaml"), "slam_set": "",
            "mapper": "slam_toolbox", "camera": "0.05,0,0.365,0,-0.7",
            "rtab_db": os.path.join(os.path.dirname(HERE), "arena", "rtab", "arena.db")}
# RTAB-Map for the arena: the lidar aligns key frames (ICP, 2D), wheel odometry through the EKF,
# a key frame every 5 cm or 3 deg, the 2D grid from the lidar at 2 cm.
RTAB_ARGS = ("--delete_db_on_start --Reg/Strategy 1 --Reg/Force3DoF true"
             " --RGBD/NeighborLinkRefining true --RGBD/ProximityBySpace true"
             " --RGBD/LinearUpdate 0.05 --RGBD/AngularUpdate 0.05"
             " --Grid/FromDepth false --Grid/CellSize 0.02"
             " --Icp/VoxelSize 0.02 --Icp/MaxCorrespondenceDistance 0.1 --Icp/PointToPlane false")


def static_tf(parent, child, x=0.0, y=0.0, z=0.0, yaw_deg=0.0, pitch_deg=0.0):
    rad = 3.141592653589793 / 180
    return Node(package="tf2_ros", executable="static_transform_publisher",
                name=f"tf_{child}", output="log",
                arguments=["--x", str(x), "--y", str(y), "--z", str(z),
                           "--yaw", str(yaw_deg * rad), "--pitch", str(pitch_deg * rad),
                           "--frame-id", parent, "--child-frame-id", child])


def include(package, launch, **arguments):
    path = os.path.join(get_package_share_directory(package), "launch", launch)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path),
                                    launch_arguments=arguments.items())


def rtabmap(arg, sim):
    """The DaBai, where it sits, and RTAB-Map on it, the lidar and the EKF's odometry."""
    x, y, z, yaw, pitch = (float(v) for v in arg["camera"].split(","))
    os.makedirs(os.path.dirname(os.path.expanduser(arg["rtab_db"])), exist_ok=True)
    return [
        include("orbbec_camera", "dabai_dcw2.launch.py", depth_registration="true",
                enable_point_cloud="false", enable_colored_point_cloud="false"),
        static_tf("base_link", "camera_link", x, y, z, yaw, pitch),
        include("rtabmap_launch", "rtabmap.launch.py",
                rgb_topic="/camera/color/image_raw", depth_topic="/camera/depth/image_raw",
                camera_info_topic="/camera/color/camera_info", odom_topic="/odom",
                frame_id="base_footprint", map_topic="/map", use_sim_time=str(sim).lower(),
                rviz="false", rtabmap_viz="false", approx_sync="true",
                approx_sync_max_interval="0.1", qos="2", visual_odometry="false",
                icp_odometry="false", subscribe_scan="true", scan_topic="/scan",
                topic_queue_size="30", queue_size="30", wait_for_transform="0.3",
                # the camera's launch sets log_level "none" for everything included after it,
                # which rcl rejects -- RTAB-Map died on it
                log_level="info",
                database_path=os.path.expanduser(arg["rtab_db"]), rtabmap_args=RTAB_ARGS),
    ]


def nodes(context):
    arg = {k: LaunchConfiguration(k).perform(context) for k in DEFAULTS}
    sim, bias = arg["sim"] == "true", arg["bias"] == "true"
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
        # /scan0 + /scan1 (or with bias, scan_relay.py's /scan0_c + /scan1_c) -> /scan_multi
        Node(package="ira_laser_tools", executable="laserscan_multi_merger",
             name="laserscan_multi_merger",
             parameters=[os.path.join(HERE, "merge_params.yaml"), clock,
                         {"laserscan_topics": "/scan0_c /scan1_c"} if bias else {}]),
        # /scan_multi -> /scan, returns under 0.18 m (the chassis) blanked
        Node(package="yahboom_laser_filter", executable="laser_filter_node",
             name="laser_filter_node",
             parameters=[{"angle_min": -180.0, "angle_max": 180.0}, clock]),
    ]
    if arg["mapper"] == "rtabmap":
        out += rtabmap(arg, sim)
    else:
        out.append(Node(package="slam_toolbox", executable="async_slam_toolbox_node",
                        name="slam_toolbox", output="screen",
                        parameters=[arg["slam"], slam_set, {**clock, "scan_topic": "/scan_lidars"
                                    if arg["scans"] == "lidars" else "/scan"}]))
    if arg["scans"] == "lidars" or bias:
        out.append(ExecuteProcess(cmd=["python3", os.path.join(HERE, "scan_relay.py"),
                                       arg["lidars"], "--ros-args",
                                       "-p", f"use_sim_time:={arg['sim']}"],
                                  output="screen"))
    if not sim:   # the browser page for driving it by hand or on a loop, http://<robot>:8095/
        bench = [] if arg["motion"] == "true" else ["--no-motion"]
        out.append(ExecuteProcess(cmd=["python3", os.path.join(HERE, "drive_server.py"), *bench],
                                  output="screen"))
    return out


def generate_launch_description():
    return LaunchDescription([*[DeclareLaunchArgument(k, default_value=v)
                                for k, v in DEFAULTS.items()],
                              OpaqueFunction(function=nodes)])
