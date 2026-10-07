"""Push changed project files to the robot over SFTP, then run the given commands.

    uv run python tools/sync_robot.py                       # sync only
    uv run python tools/sync_robot.py "uname -a"            # sync, then run each CMD in ~/roboarm

Runs in the repo's uv workspace (paramiko is in its `laptop` group). The robot gets this robot's
pins from the workspace's uv.lock as requirements.txt (`uv export --package roboarm-m3`, with the
dev group for tests on the robot), installed there with `uv pip sync requirements.txt`.
The password comes from ROBOARM_SSH_PASSWORD, or is asked for. Files are compared
by sha256 and only the changed ones are uploaded; nothing is ever deleted remotely.
The tree is CRLF on both ends, so nothing is converted.
"""

import getpass
import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile

import paramiko

HOST, USER = "192.168.73.66", "jetson"   # the M3 Pro; the X3 Plus is ../../x3plus/tools/sync_robot.py
LOCAL = pathlib.Path(__file__).resolve().parents[1]
REMOTE = "/home/jetson/roboarm"
MEMBER = "roboarm-m3"   # this robot's project in the workspace (../../pyproject.toml)

PATTERNS = ["roboarm/**/*.py", "roboarm/*.npz", "tools/*.py", "tests/*.py",
            "vision_service/*.py", "pyproject.toml",
            # its venv and systemd unit (robot/setup.sh)
            "robot/*",
            ".dockerignore", ".gitignore",
            # the robot's desktop setup, applied on the host by tools/display_setup.sh
            "tools/*.sh", "tools/*.conf", "tools/*.xml"]


def pins(folder: pathlib.Path) -> pathlib.Path:
    """This robot's pins from the workspace lock, as a requirements file in `folder`."""
    out = folder / "requirements.txt"
    subprocess.run(["uv", "export", "--frozen", "--quiet", "--no-header", "--package", MEMBER, "--no-emit-workspace",
                    "--no-default-groups", "--group", "dev", "--output-file", str(out)],
                   cwd=LOCAL, check=True)
    return out


def local_files(extra: dict[str, pathlib.Path]) -> dict[str, pathlib.Path]:
    out = {}
    for path in (LOCAL.parent / "shared" / "roboshared").glob("**/*.py"):   # what both robots share
        if "__pycache__" not in path.parts:
            out["roboshared/" + path.relative_to(LOCAL.parent / "shared" / "roboshared").as_posix()] = path
    for pattern in PATTERNS:
        for path in LOCAL.glob(pattern):
            if path.is_file() and "__pycache__" not in path.parts:
                out[path.relative_to(LOCAL).as_posix()] = path
    return out | extra


def run(client, command: str, echo: bool = True):
    _stdin, stdout, stderr = client.exec_command(f"cd {REMOTE} && {command}", get_pty=False)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if echo:
        print(f"$ {command}\n{out}{err}[exit {code}]")
    return code, out, err


def main() -> int:
    sys.stdout.reconfigure(errors="replace")   # the robot's output may not fit a Windows console
    password = os.environ.get("ROBOARM_SSH_PASSWORD") or getpass.getpass(f"{USER}@{HOST} password: ")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=password, look_for_keys=False, timeout=15)
    tmp = tempfile.TemporaryDirectory()
    paths = local_files({"requirements.txt": pins(pathlib.Path(tmp.name))})
    files = {rel: hashlib.sha256(path.read_bytes()).hexdigest() for rel, path in paths.items()}
    # run() works inside REMOTE, so it must exist first (the M3 had none until 2026-09-30)
    client.exec_command(f"mkdir -p {REMOTE}")[1].channel.recv_exit_status()
    _code, out, _err = run(client, "sha256sum " + " ".join(sorted(files)) + " 2>/dev/null",
                           echo=False)
    remote = {}
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            remote[parts[1].strip()] = parts[0]
    changed = [rel for rel, digest in sorted(files.items()) if remote.get(rel) != digest]
    sftp = client.open_sftp()
    for rel in changed:
        parent = f"{REMOTE}/{rel.rsplit('/', 1)[0]}" if "/" in rel else REMOTE
        run(client, f"mkdir -p {parent}", echo=False)
        sftp.put(str(paths[rel]), f"{REMOTE}/{rel}")
        print("uploaded", rel)
    sftp.close()
    print(f"{len(changed)} file(s) uploaded, {len(files) - len(changed)} unchanged")
    for command in sys.argv[1:]:
        run(client, command)
    client.close()
    tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
