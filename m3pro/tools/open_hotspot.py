"""Take the password off the Yahboom M3 Pro's own hotspot, so ROSMASTER becomes an open network.

    python m3pro/tools/open_hotspot.py            # remove the password
    python m3pro/tools/open_hotspot.py --check    # only show it; exit 0 when it is open

The M3 Pro is reachable only on its own hotspot (ROSMASTER, 192.168.8.88), which has no
internet, so anything that itself needs the internet -- Claude Code, for one -- has to do
the whole job in one run that ends back on the lab WiFi. The run:

1. joins ROSMASTER with the factory passphrase,
2. starts open_hotspot.sh on the robot as a detached systemd unit: it restarts
   the hotspot without a password, which drops this very SSH session and WiFi link, and
   puts the password back by itself unless a client confirms within two minutes,
3. joins ROSMASTER again, now as an open network, and confirms over SSH,
4. goes back to IISLab-AMEC and waits until the internet answers.

Networks are joined with one-off WLAN profiles, so the laptop's saved profiles are left
alone, and through wlanapi.dll because `netsh wlan connect` needs the Windows Location
permission on this box (as in tools/wifi_hop.ps1). Windows drifts back to the network
with internet, so a keeper thread re-joins while the robot is being worked on.

Runs with the SYSTEM Python 3.12 (it has paramiko). The SSH password comes from
ROBOARM_SSH_PASSWORD, or is asked for before the laptop leaves its network.
"""

import argparse
import ctypes
import getpass
import os
import pathlib
import socket
import sys
import threading
import time
from ctypes import wintypes
from xml.sax.saxutils import escape

import paramiko

ROBOT, USER = "192.168.8.88", "jetson"
SSID, PASSPHRASE = "ROSMASTER", "12345678"   # the M3 Pro's factory hotspot
HOME = "IISLab-AMEC"                          # saved profile on this laptop, has internet
INTERNET = ("api.anthropic.com", 443)
SCRIPT = pathlib.Path(__file__).with_name("open_hotspot.sh")
REMOTE = "/tmp/m3pro_open_hotspot.sh"
# open_hotspot.sh waits CONFIRM_WAIT s for CONFIRM before it puts the password back
CONFIRM, CONFIRM_WAIT = "/tmp/m3pro-open-hotspot.confirm", 120
LOG = "/var/log/m3pro-open-hotspot.log"

wlanapi = ctypes.WinDLL("wlanapi.dll")


class GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8)]


class InterfaceInfo(ctypes.Structure):         # WLAN_INTERFACE_INFO
    _fields_ = [("guid", GUID), ("description", ctypes.c_wchar * 256), ("state", ctypes.c_int)]


class InterfaceList(ctypes.Structure):         # WLAN_INTERFACE_INFO_LIST, first entry only
    _fields_ = [("count", wintypes.DWORD), ("index", wintypes.DWORD), ("items", InterfaceInfo * 1)]


class ConnectionParameters(ctypes.Structure):  # WLAN_CONNECTION_PARAMETERS
    _fields_ = [("mode", ctypes.c_int), ("profile", wintypes.LPCWSTR), ("ssid", ctypes.c_void_p),
                ("bssids", ctypes.c_void_p), ("bss_type", ctypes.c_int), ("flags", wintypes.DWORD)]


def profile_xml(passphrase: str | None) -> str:
    """A WLAN profile for ROSMASTER: WPA2 with the passphrase, or open when it is None."""
    if passphrase is None:
        auth, key = "<authentication>open</authentication><encryption>none</encryption>", ""
    else:
        auth = "<authentication>WPA2PSK</authentication><encryption>AES</encryption>"
        key = ("<sharedKey><keyType>passPhrase</keyType><protected>false</protected>"
               f"<keyMaterial>{escape(passphrase)}</keyMaterial></sharedKey>")
    return ('<?xml version="1.0"?>'
            '<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">'
            f"<name>{SSID}</name><SSIDConfig><SSID><name>{SSID}</name></SSID></SSIDConfig>"
            "<connectionType>ESS</connectionType><connectionMode>manual</connectionMode>"
            f"<MSM><security><authEncryption>{auth}<useOneX>false</useOneX></authEncryption>"
            f"{key}</security></MSM></WLANProfile>")


PROFILES = {"with passphrase": profile_xml(PASSPHRASE), "open": profile_xml(None)}


def wlan_connect(profile: str) -> int:
    """Ask Windows to join a network. `profile` is a saved profile's name, or a whole
    profile XML that is used once and never saved. Returns 0 when the request is taken."""
    handle, version = wintypes.HANDLE(), wintypes.DWORD()
    rc = wlanapi.WlanOpenHandle(2, None, ctypes.byref(version), ctypes.byref(handle))
    if rc:
        return rc
    try:
        interfaces = ctypes.POINTER(InterfaceList)()
        rc = wlanapi.WlanEnumInterfaces(handle, None, ctypes.byref(interfaces))
        if rc:
            return rc
        guid = GUID.from_buffer_copy(interfaces.contents.items[0].guid)
        wlanapi.WlanFreeMemory(interfaces)
        temporary = profile.startswith("<")
        params = ConnectionParameters(1 if temporary else 0, profile, None, None, 1, 0)
        return wlanapi.WlanConnect(handle, ctypes.byref(guid), ctypes.byref(params), None)
    finally:
        wlanapi.WlanCloseHandle(handle, None)


def answers(address: tuple[str, int], timeout: float = 2.0) -> bool:
    try:
        socket.create_connection(address, timeout=timeout).close()
        return True
    except OSError:
        return False


class Keeper(threading.Thread):
    """Holds the laptop on the profile it is given: Windows drifts back to a network
    that has internet, and the robot restarts its hotspot in the middle of the job."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.profile = None

    def hold(self, profile: str | None):
        with self.lock:   # once this returns, no re-join to the old profile is in flight
            self.profile = profile

    def run(self):
        misses = 0
        while True:
            time.sleep(3)
            ok = answers((ROBOT, 22))
            with self.lock:
                if self.profile is None or ok:
                    misses = 0
                elif (misses := misses + 1) >= 3:
                    wlan_connect(self.profile)
                    misses = 0


def join(keeper: Keeper, way: str, timeout: float) -> bool:
    """Join ROSMASTER one way and wait until the robot's SSH port answers."""
    print(f"joining {SSID} ({way})", flush=True)
    keeper.hold(PROFILES[way])
    start = time.monotonic()
    rc = wlan_connect(PROFILES[way])
    if rc:
        print(f"  WlanConnect error {rc}")
        time.sleep(min(timeout, 5))
        return False
    while time.monotonic() - start < timeout:
        if answers((ROBOT, 22)):
            print(f"  robot answers after {time.monotonic() - start:.0f} s", flush=True)
            return True
        time.sleep(1)
    print(f"  no answer in {timeout:.0f} s", flush=True)
    return False


def reach(keeper: Keeper, ways: list[str], until: float) -> str | None:
    """Try the ways of joining ROSMASTER in turn until one reaches the robot or time is up."""
    while time.monotonic() < until:
        for way in ways:
            left = until - time.monotonic()
            if left > 0 and join(keeper, way, min(30, left)):
                return way
    return None


def go_home(keeper: Keeper) -> bool:
    keeper.hold(None)
    print(f"back to {HOME}", flush=True)
    for _ in range(3):
        wlan_connect(HOME)
        for _ in range(10):
            if answers(INTERNET, 3):
                print("  internet is back")
                return True
            time.sleep(1)
    print(f"  STILL NO INTERNET: join {HOME} by hand")
    return False


def ssh(password: str, host: str = ROBOT) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=USER, password=password, look_for_keys=False,
                   allow_agent=False, timeout=15)
    return client


def as_root(client: paramiko.SSHClient, password: str, command: str) -> int:
    """Run a command on the robot with sudo, print what it says, return its exit code."""
    stdin, stdout, _ = client.exec_command(f"sudo -S -p '' {command} 2>&1", timeout=90)
    stdin.write(password + "\n")
    stdin.flush()
    print(stdout.read().decode(errors="replace").rstrip(), flush=True)
    return stdout.channel.recv_exit_status()


def upload(client: paramiko.SSHClient, local: pathlib.Path, remote: str):
    sftp = client.open_sftp()
    with sftp.open(remote, "wb") as f:
        f.write(local.read_bytes().replace(b"\r\n", b"\n"))
    sftp.close()


def open_hotspot(keeper: Keeper, password: str, check_only: bool) -> int:
    way = reach(keeper, ["with passphrase", "open"], time.monotonic() + 120)
    if way is None:
        print(f"{SSID} did not answer: is the robot on and in range?")
        return 1
    client = ssh(password)
    upload(client, SCRIPT, REMOTE)
    if check_only or way == "open":
        code = as_root(client, password, f"bash {REMOTE} check")
        client.close()
        return code
    code = as_root(client, password,
                   f"systemd-run --unit=m3pro-open-hotspot --collect /bin/bash {REMOTE} open")
    client.close()
    if code:
        return code

    # The robot now restarts its hotspot without a password, which drops this link, and
    # waits CONFIRM_WAIT s for a client to come back on the open network.
    started = time.monotonic()
    keeper.hold(PROFILES["open"])
    while answers((ROBOT, 22)):
        if time.monotonic() - started > 40:
            print("the hotspot never restarted")
            client = ssh(password)
            as_root(client, password, f"tail -n 20 {LOG}")
            client.close()
            return 1
        time.sleep(1)
    print(f"the hotspot went down after {time.monotonic() - started:.0f} s", flush=True)
    if reach(keeper, ["open"], started + CONFIRM_WAIT - 10):
        client = ssh(password)
        client.exec_command(f"touch {CONFIRM}")[1].channel.recv_exit_status()
        time.sleep(3)   # the robot looks for CONFIRM once a second
        as_root(client, password, f"tail -n 20 {LOG}")
        code = as_root(client, password, f"bash {REMOTE} check")
        client.close()
        return code

    print(f"could not join {SSID} without a password; the robot puts it back after "
          f"{CONFIRM_WAIT} s, waiting for that")
    time.sleep(max(0.0, started + CONFIRM_WAIT + 40 - time.monotonic()))
    if reach(keeper, ["with passphrase", "open"], time.monotonic() + 90):
        client = ssh(password)
        as_root(client, password, f"tail -n 20 {LOG}")
        client.close()
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove the password from the M3 Pro's hotspot.")
    parser.add_argument("--check", action="store_true", help="only show how the hotspot is set up")
    args = parser.parse_args()
    password = os.environ.get("ROBOARM_SSH_PASSWORD") or getpass.getpass(f"{USER}@{ROBOT} password: ")
    keeper = Keeper()
    keeper.start()
    try:
        return open_hotspot(keeper, password, args.check)
    finally:
        go_home(keeper)


if __name__ == "__main__":
    sys.exit(main())
