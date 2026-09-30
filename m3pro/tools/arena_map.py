"""Map the arena with the M3 Pro: start slam_toolbox, look at the map as it grows, save it.

    python tools/arena_map.py start        # upload arena/, start the mapping stack on the robot
    python tools/arena_map.py snap         # the map so far + the fence fit -> out/arena/
    python tools/arena_map.py save NAME    # map + fence fit -> arena/maps/, pose graph on the robot
    python tools/arena_map.py drive        # restart only the drive page (new code), map kept
    python tools/arena_map.py record NAME  # a replayable log of one auto loop, for tuning
    python tools/arena_map.py calib NAME   # the two lidars' mounting, from its standing start
    python tools/arena_map.py tune NAME [CONFIG ...]   # replay it through settings, compare
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
    sh(client, f"mkdir -p {REMOTE}/logs {REMOTE}/snaps {REMOTE}/maps")
    upload(client)
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


RAW = "/scan0 /scan1 /odom_raw /imu/data_raw"   # what a replay feeds the stack


def record(client, name):
    """A replayable log for tuning, ~/arena/bags/NAME: 20 s standing (the lidars' mounting is
    calibrated from it), the auto loop, 10 s standing. The raw topics, plus what the live stack
    made of them (/scan, /odom, /tf) for comparison."""
    bag = f"{REMOTE}/bags/{name}"
    if not running(client):
        print("not mapping: `start` first (the loop is planned on the live map)")
        return 1
    if sh(client, f"test -e {bag}")[0] == 0:
        print(f"{bag} exists already")
        return 1
    sh(client, f"mkdir -p {REMOTE}/bags; setsid nohup bash -c '{ROS} && exec ros2 bag record"
               f" -o {bag} {RAW} /cmd_vel /battery /scan /odom /tf /tf_static'"
               f" > {REMOTE}/logs/bag_{name}.log 2>&1 < /dev/null & echo $! > {REMOTE}/bag.pgid")
    print("recording; standing still for 20 s")
    time.sleep(20)
    _c, text = sh(client, "curl -s -m 5 -X POST -H 'Content-Type: application/json'"
                          " -d '{\"on\": true}' localhost:8095/api/auto")
    print(f"auto loop: {text}")
    try:
        started = time.monotonic()
        while time.monotonic() - started < 480:
            time.sleep(10)
            state = json.loads(sh(client, "curl -s -m 5 localhost:8095/api/state")[1])
            # the live map's fence as the robot drives: does it grow?
            save_map(client, f"{REMOTE}/snaps/rec")
            fence = json.loads(sh(client, f"cat {REMOTE}/snaps/rec_fit.json")[1])["fence"]
            print(f"  {time.monotonic() - started:4.0f} s {state['mode']} {state['auto']}"
                  f" fence {fence['size_m'][0]:.3f} x {fence['size_m'][1]:.3f}"
                  f" thick {[w['thick_cm'] for w in fence['walls']]} {state['note']}")
            if state["mode"] == "idle" and state["auto"] is None:
                break
        print("standing still for 10 s")
        time.sleep(10)
    finally:
        sh(client, "curl -s -m 5 -X POST localhost:8095/api/stop")
        sh(client, f"kill -INT -- -$(cat {REMOTE}/bag.pgid); sleep 3")
    print(sh(client, f"{ROS} && ros2 bag info {bag}")[1])
    return 0


CONFIGS = {   # what `tune` tries by default: arena_slam.launch.py arguments
    "live": [],                                   # what `start` runs now
    "yahboom": ["scans:=merged", "ekf_hz:=7"],    # Yahboom's scan merge and EKF rate
    "merged30": ["scans:=merged"],
}


def upload(client):
    sftp = client.open_sftp()
    for pattern in ("*.py", "*.yaml", "*.html", "*.sh"):
        for path in sorted(ARENA.glob(pattern)):
            sftp.put(str(path), f"{REMOTE}/{path.name}")


def calib(client, name):
    """lidar_calib.py on a recording's first 15 s; the corrected mounting -> ~/arena/lidars_cal.yaml."""
    upload(client)
    _c, text = sh(client, f"{ROS} && cd {REMOTE} && python3 lidar_calib.py bags/{name}"
                          " --write lidars_cal.yaml && cat lidars_cal.yaml", timeout=600)
    print(text)
    return 0


def tune(client, name, wanted):
    """Replay recording NAME through configurations side by side (replay.sh, one ROS domain
    each) and compare their maps. Configurations: the names in CONFIGS (all by default), or
    LABEL=arg:=value,arg:=value."""
    configs = {}
    for item in wanted or CONFIGS:
        label, _eq, args = item.partition("=")
        configs[label] = args.split(",") if args else CONFIGS[label]
    upload(client)
    bag = f"{REMOTE}/bags/{name}"
    info = sh(client, f"{ROS} && ros2 bag info {bag}")[1]
    seconds = float(info.split("Duration:")[1].split("s")[0])
    for i, (label, args) in enumerate(configs.items()):
        out = f"{REMOTE}/tune/{name}/{label}"
        sh(client, f"rm -rf {out}; mkdir -p {out}; setsid nohup bash {REMOTE}/replay.sh {bag}"
                   f" {out} {41 + i} {' '.join(shlex.quote(a) for a in args)}"
                   f" > {out}.log 2>&1 < /dev/null &")
    print(f"replaying {seconds:.0f} s through {len(configs)} configurations at once")
    time.sleep(seconds + 20)
    logs = " ".join(f"{REMOTE}/tune/{name}/{label}.log" for label in configs)
    for _ in range(20):
        done = sh(client, f"grep -l '^done' {logs} | wc -l")[1].strip()
        if int(done) >= len(configs):
            break
        time.sleep(10)
    dest = HERE / "out" / "arena" / "tune" / name
    dest.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    print(f"{'config':14} {'fence m':>15} {'off cm':>12} {'corners deg':>24} {'walls thick cm':>22}"
          f" {'dropped':>7}")
    for label in configs:
        out = f"{REMOTE}/tune/{name}/{label}"
        try:
            fit = json.loads(sh(client, f"cat {out}/fit.json")[1])
            sftp.get(f"{out}/map_fit.png", str(dest / f"{label}.png"))
        except (ValueError, OSError):
            print(f"{label:14} failed -- {out}.log, {out}/launch.log")
            continue
        fence = fit["fence"]
        dropped = sh(client, f"cat {out}/dropped.txt")[1].strip()
        corners = " ".join(f"{a:5.1f}" for a in fence["corner_angles_deg"])
        thick = " ".join(f"{w['thick_cm']:4.1f}" for w in fence["walls"])
        print(f"{label:14} {fence['size_m'][0]:.3f} x {fence['size_m'][1]:.3f}"
              f" {fence['error_cm'][0]:+5.1f} {fence['error_cm'][1]:+5.1f}"
              f" {corners:>24} {thick:>22} {dropped:>7}")
    print("\nthe fence as the replay went on (every 20 s), cm off 3.2 / 2.8:")
    for label in configs:
        out = f"{REMOTE}/tune/{name}/{label}"
        fits = sh(client, f"for f in $(ls {out}/t*.fit.json | sort -V); do echo $(basename $f)"
                          f" $(tr -d '\\n ' < $f); done")[1].splitlines()
        steps = []
        for line in fits:
            step, _sp, text = line.partition(" ")
            try:
                off = json.loads(text)["fence"]["error_cm"]
                steps.append(f"{step.split('.')[0]} " + "/".join(
                    "  -  " if e is None else f"{e:+.1f}" for e in off))
            except ValueError:
                steps.append(f"{step.split('.')[0]} (no fit)")
        print(f"{label:14} " + "   ".join(steps))
    print(f"-> {dest}")
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
    for name in ("drive_server.py", "drive_core.py", "drive.html", "fence_fit.py"):
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
    if len(sys.argv) == 3 and sys.argv[1] in ("save", "record", "calib"):
        return {"save": save, "record": record, "calib": calib}[sys.argv[1]](connect(), sys.argv[2])
    if len(sys.argv) >= 3 and sys.argv[1] == "tune":
        return tune(connect(), sys.argv[2], sys.argv[3:])
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        print(__doc__)
        return 2
    return commands[sys.argv[1]](connect())


if __name__ == "__main__":
    sys.exit(main())
