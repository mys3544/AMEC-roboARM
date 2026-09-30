"""Map the arena with the M3 Pro: start slam_toolbox, look at the map as it grows, save it.

    python tools/arena_map.py start        # upload arena/, start the mapping stack on the robot
    python tools/arena_map.py snap         # the map so far + the fence fit -> out/arena/
    python tools/arena_map.py save NAME    # map + fence fit -> arena/maps/, pose graph on the robot
    python tools/arena_map.py drive        # restart only the drive page (new code), map kept
    python tools/arena_map.py status
    python tools/arena_map.py stop

Drive the robot meanwhile from http://192.168.73.66:8095/ (arena/drive_server.py, part of the
stack: by hand, or "Auto" loops the arena slowly) or with Yahboom's gamepad. The stack runs
detached on the robot, in its own process group, logging to ~/arena/logs/, so a dropped SSH
session does not stop it; `stop` ends that group and nothing else.
Runs with the SYSTEM Python 3.12 (paramiko); the password comes from ROBOARM_SSH_PASSWORD or is
asked for.
"""

import getpass
import json
import os
import pathlib
import shlex
import sys
import time

import paramiko

HOST, USER = "192.168.73.66", "jetson"
HERE = pathlib.Path(__file__).resolve().parents[1]
ARENA = HERE / "arena"
REMOTE = f"/home/{USER}/arena"
ROS = ("source /opt/ros/humble/setup.bash && source ~/yahboomcar_ws/install/setup.bash"
       " && source ~/M3Pro_ws/install/setup.bash"
       " && export ROS_DOMAIN_ID=30 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp")
SAVER = "ros2 run nav2_map_server map_saver_cli --free 0.196 --occ 0.65 -f"   # Yahboom's thresholds
PGID = f"{REMOTE}/slam.pgid"


def connect():
    password = os.environ.get("ROBOARM_SSH_PASSWORD") or getpass.getpass(f"{USER}@{HOST} password: ")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=password, look_for_keys=False, timeout=15)
    return client


def sh(client, command, timeout=60):
    """Run `command` under bash on the robot; returns (exit code, stdout + stderr)."""
    _in, out, err = client.exec_command(f"bash -c {shlex.quote(command)}", timeout=timeout)
    text = out.read().decode(errors="replace") + err.read().decode(errors="replace")
    return out.channel.recv_exit_status(), text


def running(client):
    code, _ = sh(client, f"test -f {PGID} && kill -0 -- -$(cat {PGID}) 2>/dev/null")
    return code == 0


def start(client):
    if running(client):
        print("already mapping; `status` shows it, `stop` ends it")
        return 1
    # [x]: so that pgrep does not find this very command line
    _c, text = sh(client, "pgrep -af 'slam_[t]oolbox|laserscan_multi_[m]erger|ekf_[n]ode' || true")
    if text.strip():
        print(f"another SLAM stack is running on the robot -- stop it first:\n{text}")
        return 1
    sftp = client.open_sftp()
    sh(client, f"mkdir -p {REMOTE}/logs {REMOTE}/snaps {REMOTE}/maps")
    for pattern in ("*.py", "*.yaml", "*.html"):
        for path in sorted(ARENA.glob(pattern)):
            sftp.put(str(path), f"{REMOTE}/{path.name}")
    log = f"{REMOTE}/logs/slam_{time.strftime('%Y%m%d_%H%M%S')}.log"
    # setsid: the launch leads its own process group, whose id `stop` signals. Nothing may go
    # in front of it on this line: `a && b &` backgrounds a subshell, whose pid is not the group's
    # and which keeps the SSH channel open.
    sh(client, f"setsid nohup bash -c '{ROS} && exec ros2 launch {REMOTE}/arena_slam.launch.py'"
               f" > {log} 2>&1 < /dev/null & echo $! > {PGID}")
    print(f"started; log {log}; waiting 12 s for the stack")
    time.sleep(12)
    return status(client)


def status(client):
    if not running(client):
        print("not mapping")
        return 1
    _c, text = sh(client, f"ps -o pid,etime,args --sid $(cat {PGID}) | cut -c1-140;"
                          " curl -s -m 3 localhost:8095/api/state; echo;"
                          f" {ROS} && for t in /scan /odom /map; do echo \"-- $t\";"
                          f" timeout 6 ros2 topic hz $t 2>&1 | grep -m1 'average rate' || echo none;"
                          f" done; echo '-- log'; tail -n 8 $(ls -t {REMOTE}/logs/*.log | head -1)",
                  timeout=60)
    print(text)
    return 0


def save_map(client, stem):
    """map_saver_cli + fence_fit on the robot: `stem`.pgm/.yaml, `stem`_fit.png/.json."""
    client.open_sftp().put(str(ARENA / "fence_fit.py"), f"{REMOTE}/fence_fit.py")
    code, text = sh(client, f"{ROS} && timeout 25 {SAVER} {stem}"
                            f" && python3 {REMOTE}/fence_fit.py {stem}.yaml --png {stem}_fit.png"
                            f" > {stem}_fit.json", timeout=60)
    if code:
        print(text)
        raise SystemExit("saving the map failed (is the stack mapping? `status`)")


def fetch(client, stem, dest, name):
    dest.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    for suffix in (".pgm", ".yaml", "_fit.png", "_fit.json"):
        sftp.get(f"{stem}{suffix}", str(dest / f"{name}{suffix}"))
    summary(json.loads((dest / f"{name}_fit.json").read_text()))
    print(f"-> {dest / (name + '_fit.png')}")


def summary(fit):
    fence = fit["fence"]
    off = " / ".join("not seen" if e is None else f"{e:+.1f} cm" for e in fence["error_cm"])
    print(f"fence {fence['size_m'][0]:.3f} x {fence['size_m'][1]:.3f} m (off by {off}),"
          f" corners {fence['corner_angles_deg']} deg")
    for i, wall in enumerate(fence["walls"]):
        print(f"  wall {i}: {wall['length_m']:.3f} m, rms {wall['rms_cm']:.1f} cm,"
              f" seen {wall['seen'] * 100:.0f} %")
    for i, blob in enumerate(fit["blobs"], 1):
        print(f"  blob {i}: at {blob['arena_xy_m']} m in the arena, {blob['size_cm']} cm,"
              f" {blob['cells']} cells")


def snap(client):
    save_map(client, f"{REMOTE}/snaps/snap")
    fetch(client, f"{REMOTE}/snaps/snap", HERE / "out" / "arena",
          f"snap_{time.strftime('%Y%m%d_%H%M%S')}")
    return 0


def save(client, name):
    stem = f"{REMOTE}/maps/{name}"
    save_map(client, stem)
    code, text = sh(client, f"{ROS} && ros2 service call /slam_toolbox/serialize_map"
                            f" slam_toolbox/srv/SerializePoseGraph \"{{filename: '{stem}'}}\"",
                    timeout=60)
    print(f"pose graph on the robot: {stem}.posegraph/.data" if code == 0 else text)
    fetch(client, stem, ARENA / "maps", name)
    return 0


DRIVE = f"python3 {REMOTE}/drive_server.py"


def drive(client):
    """Restart only the drive page with the local drive_server.py, drive.html and fence_fit.py;
    the map being built is kept. A drive server started this way runs outside the launch's
    process group; `stop` ends it too."""
    if not running(client):
        print("not mapping: `start` starts the drive page with the rest")
        return 1
    sftp = client.open_sftp()
    for name in ("drive_server.py", "drive.html", "fence_fit.py"):
        sftp.put(str(ARENA / name), f"{REMOTE}/{name}")
    end_drive(client)
    log = f"{REMOTE}/logs/drive_{time.strftime('%Y%m%d_%H%M%S')}.log"
    sh(client, f"setsid nohup bash -c '{ROS} && exec {DRIVE}' > {log} 2>&1 < /dev/null &")
    time.sleep(4)
    _c, text = sh(client, "curl -s -m 3 localhost:8095/api/state")
    print(f"drive page restarted, log {log}\n{text}")
    return 0


def end_drive(client):
    """SIGINT (it publishes a zero on the way out), then TERM if it lingers."""
    sh(client, f"pkill -INT -f '^{DRIVE}'; for i in 1 2 3 4 5; do"
               f" pgrep -f '^{DRIVE}' > /dev/null || exit 0; sleep 1; done;"
               f" pkill -TERM -f '^{DRIVE}'", timeout=30)


def stop(client):
    end_drive(client)
    if not running(client):
        print("not mapping")
        return 0
    sh(client, f"kill -INT -- -$(cat {PGID})")
    for _ in range(10):
        time.sleep(1)
        if not running(client):
            print("stopped")
            return 0
    sh(client, f"kill -TERM -- -$(cat {PGID})")
    time.sleep(2)
    print("stopped (TERM)" if not running(client) else "still running -- look at `status`")
    return 0


def main():
    commands = {"start": start, "status": status, "snap": snap, "drive": drive, "stop": stop}
    if len(sys.argv) == 3 and sys.argv[1] == "save":
        return save(connect(), sys.argv[2])
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        print(__doc__)
        return 2
    return commands[sys.argv[1]](connect())


if __name__ == "__main__":
    sys.exit(main())
