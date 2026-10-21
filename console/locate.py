"""Where the robots are on this network (2026-10-20, for the event out of the lab: our router, or the
phone's hotspot, which hands out addresses of its own choosing). robots.json gives each robot's lab
address; this finds the address it has now, each robot told by its bridge's /health (the M3's camera
reports colour_age_s, the X3's bridge its calibration):

    1. its name, by mDNS -- the robots were renamed m3pro and x3plus on 2026-10-20 (both had been
       "yahboom"); Windows resolves NAME.local;
    2. its robots.json address, if that robot answers there;
    3. only when asked (--scan): a scan of this laptop's networks (each a /24) for the bridges' port
       -- on our router or the phone's hotspot, not on a network that is not ours.

    python console/locate.py [--scan]     # where each robot is, and how it was found

serve.py does it as it starts (not with --fake; its --scan too): the page gets the addresses found
in /robots.json. A robot found nowhere keeps its robots.json address. Standard library only.
"""

import concurrent.futures
import ipaddress
import json
import pathlib
import socket
import sys
import threading
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
NAMES = {"m3": "m3pro.local", "x3": "x3plus.local"}   # the robots' hostnames since 2026-10-20
BRIDGE_PORT = 8761   # both robots' bridges: /health, /pick/*


def robot_of(kind):
    """robots.json's kind -> the robot it is on: "m3" (m3, m3arm) or "x3" (x3, x3arm)."""
    return "m3" if kind.startswith("m3") else "x3"


def who(health):
    """Which robot a bridge's /health is from: "m3", "x3" or None."""
    if not isinstance(health, dict):
        return None
    if "calibration" in health:
        return "x3"
    if "colour_age_s" in (health.get("camera") or {}):
        return "m3"
    return None


def health(host, timeout=0.6):
    try:
        with urllib.request.urlopen(f"http://{host}:{BRIDGE_PORT}/health", timeout=timeout) as reply:
            return json.loads(reply.read())
    except (OSError, ValueError):
        return None


def resolve(name, timeout=2.0):
    """`name`'s IPv4 address, or None (getaddrinfo has no timeout of its own: mDNS may take seconds;
    a daemon thread, so a lookup still hanging does not hold the program at its end)."""
    got = []

    def look():
        try:
            got.append(socket.getaddrinfo(name, None, socket.AF_INET)[0][4][0])
        except OSError:
            pass   # unknown here
    worker = threading.Thread(target=look, daemon=True)
    worker.start()
    worker.join(timeout)
    return got[0] if got else None


def networks():
    """This laptop's IPv4 networks, as /24s (not loopback, link-local or Tailscale's)."""
    own = {a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    skip = [ipaddress.ip_network(n) for n in ("127.0.0.0/8", "169.254.0.0/16", "100.64.0.0/10")]
    return sorted({ipaddress.ip_network(f"{a}/24", strict=False) for a in own
                   if not any(ipaddress.ip_address(a) in n for n in skip)}, key=str)


def scan(nets=None, timeout=0.4):
    """{robot: address} of the bridges answering on these networks (this laptop's): a few seconds."""
    hosts = [str(h) for net in (networks() if nets is None else nets) for h in net.hosts()]
    found = {}
    with concurrent.futures.ThreadPoolExecutor(64) as pool:
        for host, answer in zip(hosts, pool.map(lambda h: health(h, timeout), hosts), strict=True):
            if who(answer):
                found.setdefault(who(answer), host)
    return found


def locate(robots, scan_too=False, say=print):
    """robots.json's list with each url's host where that robot answers now (`scan_too`: scanned for
    when neither its name nor its robots.json address answers)."""
    labs = {robot_of(spec["kind"]): urllib.parse.urlsplit(spec["url"]).hostname for spec in robots}

    def where(robot):   # by its name, or at its robots.json address: (address, how), or None
        for _ in range(2):   # 2026-10-21 the X3's name missed once as serve.py started, then never (a cold lookup)
            named = resolve(NAMES[robot], 4.0)
            if named and who(health(named, 1.5)) == robot:
                return named, f"by its name {NAMES[robot]}"
        if who(health(labs[robot])) == robot:
            return labs[robot], "at its robots.json address"
        return None
    with concurrent.futures.ThreadPoolExecutor(len(labs) or 1) as pool:   # both at once: ~3 s if neither is here
        found = {robot: at for robot, at in zip(labs, pool.map(where, labs), strict=True) if at}
    wanted = set(labs)
    if scan_too and wanted - set(found):
        for robot, host in scan().items():
            if robot in wanted and robot not in found:
                found[robot] = (host, "by a scan of this laptop's networks")
    for robot in sorted(wanted):
        say(f"{robot}: {found[robot][0]} ({found[robot][1]})" if robot in found
            else f"{robot}: not found -- its robots.json address kept{'' if scan_too else ' (--scan looks round this network)'}")
    located = []
    for spec in robots:
        robot, url = robot_of(spec["kind"]), urllib.parse.urlsplit(spec["url"])
        if robot in found:
            host = found[robot][0] if url.port is None else f"{found[robot][0]}:{url.port}"
            spec = {**spec, "url": url._replace(netloc=host).geturl()}
        located.append(spec)
    return located


if __name__ == "__main__":
    for spec in locate(json.loads((HERE / "robots.json").read_text()), scan_too="--scan" in sys.argv):
        print(f"  {spec['id']:6s} {spec['url']}")
    sys.exit(0)
