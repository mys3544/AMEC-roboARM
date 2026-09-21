"""Push changed project files to the robot over SFTP, then run the given commands.

    python tools/sync_robot.py                       # sync only
    python tools/sync_robot.py "docker compose ps"   # sync, then run each CMD in ~/roboarm

Runs with the SYSTEM Python 3.12 (it has paramiko); the project venv does not.
The password comes from ROBOARM_SSH_PASSWORD, or is asked for. Files are compared
by sha256 and only the changed ones are uploaded; nothing is ever deleted remotely.
The tree is CRLF on both ends, so nothing is converted.
"""

import getpass
import hashlib
import os
import pathlib
import sys

import paramiko

HOST, USER = "192.168.73.210", "jetson"
LOCAL = pathlib.Path(__file__).resolve().parents[1]
REMOTE = "/home/jetson/roboarm"

PATTERNS = ["roboarm/**/*.py", "roboarm/web/static/*.html", "tools/*.py", "tests/*.py",
            "vision_service/*.py", "docker/*", "docs/*.md", "compose.yaml", "pyproject.toml",
            "uv.lock", ".dockerignore", ".gitignore",
            # the robot's desktop setup, applied on the host by tools/display_setup.sh
            "tools/*.sh", "tools/*.conf", "tools/*.xml"]


def local_files() -> dict[str, str]:
    out = {}
    for pattern in PATTERNS:
        for path in LOCAL.glob(pattern):
            if path.is_file() and "__pycache__" not in path.parts:
                rel = path.relative_to(LOCAL).as_posix()
                out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def run(client, command: str, echo: bool = True):
    _stdin, stdout, stderr = client.exec_command(f"cd {REMOTE} && {command}", get_pty=False)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if echo:
        print(f"$ {command}\n{out}{err}[exit {code}]")
    return code, out, err


def main() -> int:
    password = os.environ.get("ROBOARM_SSH_PASSWORD") or getpass.getpass(f"{USER}@{HOST} password: ")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=password, look_for_keys=False, timeout=15)
    files = local_files()
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
        sftp.put(str(LOCAL / rel), f"{REMOTE}/{rel}")
        print("uploaded", rel)
    sftp.close()
    print(f"{len(changed)} file(s) uploaded, {len(files) - len(changed)} unchanged")
    for command in sys.argv[1:]:
        run(client, command)
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
