#!/usr/bin/env python3
"""Run Python code inside the running Isaac Sim on the GPU PC (hekate).

Talks to Isaac Sim's isaacsim.code_editor.python_server, which listens only on
127.0.0.1 of the PC. From the laptop, pass --via to go through an SSH tunnel
that this script opens and closes for you:

    python3 isaac_exec.py --via hekate scripts/build_scene.py
    python3 isaac_exec.py --via hekate -c "import sys; print(sys.version)"
    cat scene.py | python3 isaac_exec.py --via hekate

On the PC itself, leave out --via (or use tools/isaac_exec.ps1, which needs no Python).

What you get back: whatever the code printed, the value of a final bare
expression as "=> value", and on error the traceback. Python globals persist
between calls, and top-level await works.
Exit codes: 0 ok, 1 the code raised, 2 bad usage, 3 Isaac Sim not reachable.
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time

DEFAULT_PORT = 8230  # ISAAC_PY_PORT in config.bat on the PC
START_HINT = 'start it with: ssh {host} "schtasks /run /tn IsaacSimStreaming"'


class Unreachable(Exception):
    pass


def free_local_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def open_tunnel(host, local_port, remote_port):
    cmd = [
        "ssh", "-N",
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-L", f"{local_port}:127.0.0.1:{remote_port}",
        host,
    ]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise Unreachable("the 'ssh' command was not found on this machine")


def close_tunnel(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def connect(port, tunnel, deadline):
    """Connect to 127.0.0.1:port, retrying while an SSH tunnel is still starting."""
    while True:
        if tunnel is not None and tunnel.poll() is not None:
            err = tunnel.stderr.read().decode(errors="replace").strip()
            raise Unreachable(f"SSH tunnel failed: {err or 'ssh exited with code %d' % tunnel.returncode}")
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=10)
        except ConnectionRefusedError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.25)


def run(code, port, timeout, tunnel=None, connect_wait=0.0):
    sock = connect(port, tunnel, time.monotonic() + connect_wait)
    with sock:
        sock.settimeout(timeout)
        sock.sendall(code.encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)  # the server runs the code only after end-of-input
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser(
        description="Run Python code inside the running Isaac Sim.",
        epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="Python file to run ('-' = read stdin)")
    ap.add_argument("-c", "--code", help="run this code string instead of a file")
    ap.add_argument("--via", metavar="SSH_HOST", default=os.environ.get("ISAAC_SSH_HOST"),
                    help="tunnel through this SSH host, e.g. hekate (default: $ISAAC_SSH_HOST)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("ISAAC_PY_PORT", DEFAULT_PORT)),
                    help=f"Python server port on the PC (default {DEFAULT_PORT})")
    ap.add_argument("--timeout", type=float, default=600,
                    help="seconds to wait for the code to finish (default 600)")
    ap.add_argument("--json", action="store_true", help="print the raw JSON reply")
    a = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass

    if a.code is not None:
        code = a.code
    elif a.file and a.file != "-":
        with open(a.file, encoding="utf-8") as f:
            code = f.read()
    elif a.file == "-" or not sys.stdin.isatty():
        code = sys.stdin.read()
    else:
        code = ""
    if not code.strip():
        ap.print_usage(sys.stderr)
        print("error: no code given (pass a file, -c CODE, or pipe code in)", file=sys.stderr)
        return 2

    where = f"{a.via}:127.0.0.1:{a.port}" if a.via else f"127.0.0.1:{a.port}"
    hint = START_HINT.format(host=a.via or "hekate")
    tunnel = None
    try:
        if a.via:
            local_port = free_local_port()
            tunnel = open_tunnel(a.via, local_port, a.port)
            raw = run(code, local_port, a.timeout, tunnel, connect_wait=20)
        else:
            raw = run(code, a.port, a.timeout)
    except Unreachable as e:
        print(f"Cannot reach Isaac Sim ({where}): {e}", file=sys.stderr)
        return 3
    except ConnectionRefusedError:
        print(f"Nothing is listening on {where}. Is the Isaac Sim streaming instance running? "
              f"If not, {hint}", file=sys.stderr)
        return 3
    except socket.timeout:
        print(f"No reply within {a.timeout:g} s from {where}. The code may still be running "
              "inside Isaac Sim (raise --timeout for long jobs).", file=sys.stderr)
        return 3
    except (ConnectionResetError, BrokenPipeError) as e:
        raw = ""
        if not a.via:
            print(f"Connection to {where} dropped: {e}", file=sys.stderr)
            return 3
    finally:
        close_tunnel(tunnel)

    if not raw.strip():
        print(f"No answer from Isaac Sim's Python server at {where}. "
              f"Is the streaming instance running? If not, {hint}", file=sys.stderr)
        return 3

    if a.json:
        print(raw)
        return 0 if re.search(r'"status"\s*:\s*"ok"', raw) else 1

    res = json.loads(raw)
    out = res.get("output") or ""
    if out:
        sys.stdout.write(out if out.endswith("\n") else out + "\n")
    if res.get("result") is not None:  # None (e.g. a trailing print()) is not worth showing
        r = res["result"]
        print("=> " + (r if isinstance(r, str) else
                       json.dumps(r) if isinstance(r, (list, dict)) else repr(r)))
    if res.get("status") != "ok":
        strip_ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
        print(f"{res.get('ename', 'Error')}: {res.get('evalue', '')}", file=sys.stderr)
        tb = res.get("traceback") or []
        if tb:
            print(strip_ansi.sub("", "".join(tb) if isinstance(tb, list) else str(tb)),
                  file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
