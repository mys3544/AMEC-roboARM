"""Show the M3 a thing, then have it find or pick that thing by name (roboarm/m3teach.py on the
robot; its YOLOE in tools/teach_service.py). From the laptop, following the robot's log:

    python tools/m3_show.py teach apricot   # the thing standing ALONE in front: the arm looks round it
    python tools/m3_show.py find            # names every thing on the table in front
    python tools/m3_show.py pick apricot    # finds it and picks it into the box
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_handeye as he

ROUTES = {"teach": "/pick/teach", "find": "/pick/find", "pick": "/pick/named"}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ROUTES or (sys.argv[1] != "find" and len(sys.argv) < 3):
        print(__doc__)
        return 2
    reply = he.http(ROUTES[sys.argv[1]], {"name": " ".join(sys.argv[2:])})
    if "error" in reply:
        raise SystemExit(f"not started: {reply['error']}")
    since = len(he.http("/pick/log?since=0")["lines"])
    while True:
        time.sleep(1.0)
        lines = he.http(f"/pick/log?since={since}")["lines"]
        for line in lines:
            print(line["text"], flush=True)
        since += len(lines)
        state = he.http("/pick/state")
        if not state["busy"]:
            print("error:" if state["error"] else "done.", state["error"] or "")
            return 0 if not state["error"] else 1


if __name__ == "__main__":
    sys.exit(main())
