"""Make the Yahboom M3 Pro join the lab WiFi (IISLab-AMEC) at boot instead of its own hotspot.

    python m3pro/tools/lab_wifi.py

Until this has run the robot is reachable only on its hotspot (ROSMASTER, 192.168.8.88), so
the laptop hops over as in open_hotspot.py, whose helpers this uses:

1. join ROSMASTER and leave the robot a NetworkManager keyfile for IISLab-AMEC, with the
   passphrase from this laptop's saved profile (never printed),
2. probe: the robot joins IISLab-AMEC for a moment, logs the address it got, and comes back
   to the hotspot -- a wrong passphrase costs nothing,
3. switch: the robot makes IISLab-AMEC its boot network, stops starting the hotspot and moves
   over; it puts the hotspot back unless a client confirms within three minutes,
4. the laptop goes back to IISLab-AMEC and confirms over SSH at the address from step 2.

lab_wifi.sh is the robot's half. Runs with the SYSTEM Python 3.12 (it has
paramiko); the SSH password comes from ROBOARM_SSH_PASSWORD, or is asked for.
"""

import getpass
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET

from open_hotspot import (
    HOME,
    INTERNET,
    ROBOT,
    USER,
    Keeper,
    answers,
    as_root,
    go_home,
    reach,
    ssh,
    upload,
)

LAB = HOME
SCRIPT = pathlib.Path(__file__).with_name("lab_wifi.sh")
REMOTE = "/tmp/m3pro_lab_wifi.sh"
# lab_wifi.sh installs NEW, and in switch waits CONFIRM_WAIT s for CONFIRM
NEW = "/tmp/m3pro-lab.nmconnection"
CONFIRM, CONFIRM_WAIT = "/tmp/m3pro-lab-wifi.confirm", 180
LOG = "/var/log/m3pro-lab-wifi.log"
WAYS_IN = ["open", "with passphrase"]   # ways to join ROSMASTER, most likely first


def lab_passphrase() -> str:
    """IISLab-AMEC's passphrase, from this laptop's saved WLAN profile."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["netsh", "wlan", "export", "profile", f"name={LAB}", "key=clear",
                        f"folder={tmp}"], check=True, capture_output=True)
        key = ET.parse(next(pathlib.Path(tmp).glob("*.xml"))).getroot().find(".//{*}sharedKey")
    if key is None or key.findtext("{*}protected") != "false":
        sys.exit(f"the laptop's {LAB} profile has no readable passphrase")
    return key.findtext("{*}keyMaterial")


def keyfile(passphrase: str) -> str:
    """NetworkManager keyfile for the lab WiFi; switch turns its autoconnect on."""
    psk = passphrase.replace("\\", "\\\\")
    psk = "\\s" + psk[1:] if psk.startswith(" ") else psk
    return (f"[connection]\nid={LAB}\nuuid={uuid.uuid5(uuid.NAMESPACE_DNS, LAB)}\ntype=wifi\n"
            "autoconnect=false\n\n"
            f"[wifi]\nmode=infrastructure\nssid={LAB}\ncloned-mac-address=permanent\n\n"
            f"[wifi-security]\nkey-mgmt=wpa-psk\npsk={psk}\n\n"
            "[ipv4]\nmethod=auto\nmay-fail=false\n\n"
            "[ipv6]\naddr-gen-mode=stable-privacy\nmethod=auto\n")


def last_run(client) -> str:
    """What the robot logged in its latest probe or switch."""
    with client.open_sftp() as sftp, sftp.open(LOG) as f:
        text = f.read().decode(errors="replace")
    return "== " + text.rsplit("== ", 1)[-1].rstrip()


def start(client, password: str, mode: str) -> int:
    """Start probe or switch on the robot as a detached unit, since both leave the hotspot."""
    return as_root(client, password,
                   f"systemd-run --unit=m3pro-lab-wifi-{mode} --collect /bin/bash {REMOTE} {mode}")


def left_hotspot(password: str) -> bool:
    """Wait until the robot takes its radio off the hotspot; if it never does, say why."""
    started = time.monotonic()
    while answers((ROBOT, 22)):
        if time.monotonic() - started > 40:
            client = ssh(password)
            print(last_run(client))
            client.close()
            return False
        time.sleep(1)
    return True


def switch_to_lab(keeper: Keeper, password: str, passphrase: str) -> int:
    if not reach(keeper, WAYS_IN, time.monotonic() + 120):
        print("ROSMASTER did not answer: is the robot on and in range?")
        return 1
    client = ssh(password)
    upload(client, SCRIPT, REMOTE)
    with client.open_sftp() as sftp, sftp.open(NEW, "w") as f:
        f.chmod(0o600)   # before the passphrase goes in
        f.write(keyfile(passphrase).encode())
    code = start(client, password, "probe")
    client.close()
    if code or not left_hotspot(password):
        return 1

    started = time.monotonic()
    if not reach(keeper, WAYS_IN, started + 150):
        print("the robot did not come back to its hotspot after the probe")
        return 1
    client = ssh(password)
    probe = last_run(client)
    print(probe, flush=True)
    found = re.search(r"lab address ([\d.]+)", probe)
    if not found:
        print(f"the robot could not join {LAB}; nothing changed")
        client.close()
        return 1
    lab_ip = found.group(1)
    code = start(client, password, "switch")
    client.close()
    if code or not left_hotspot(password):
        return 1

    # The robot is moving to the lab WiFi and waits CONFIRM_WAIT s for us there.
    started = time.monotonic()
    go_home(keeper)
    while time.monotonic() - started < CONFIRM_WAIT - 15:
        if answers((lab_ip, 22)):
            client = ssh(password, lab_ip)
            client.exec_command(f"touch {CONFIRM}")[1].channel.recv_exit_status()
            time.sleep(3)   # the robot looks for CONFIRM once a second
            print(last_run(client))
            client.close()
            print(f"the robot is on {LAB} at {lab_ip}")
            return 0
        time.sleep(2)

    print(f"no answer from {lab_ip}; the robot puts its hotspot back after {CONFIRM_WAIT} s")
    time.sleep(max(0.0, started + CONFIRM_WAIT + 30 - time.monotonic()))
    if reach(keeper, WAYS_IN, time.monotonic() + 90):
        client = ssh(password)
        print(last_run(client))
        client.close()
    return 1


def main() -> int:
    password = os.environ.get("ROBOARM_SSH_PASSWORD") or getpass.getpass(f"{USER}@{ROBOT} password: ")
    passphrase = lab_passphrase()
    keeper = Keeper()
    keeper.start()
    try:
        return switch_to_lab(keeper, password, passphrase)
    finally:
        keeper.hold(None)
        if not answers(INTERNET, 3):
            go_home(keeper)


if __name__ == "__main__":
    sys.exit(main())
