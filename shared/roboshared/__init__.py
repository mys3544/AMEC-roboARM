"""What both robots do the same way (the repo's shared/, README.md):

    task     the jobs -- pick once, clean up, home, drop -- their flow, settings and what the
             console watches; a robot's subclass is its body (m3pro/roboarm/m3pick.py)
    box      the drop-off box: its marker (ArUco 4x4 id 10) and the inside from it
    metrics  where the time goes

ROS-free and robot-free: each robot's own package (x3plus/roboarm, m3pro/roboarm) supplies the arm,
the camera, the wheels and how it finds things. On the laptop it is installed in the workspace's
.venv; each robot's tools/sync_robot.py copies it to ~/roboarm/roboshared.
"""
