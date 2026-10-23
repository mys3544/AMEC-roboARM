"""How the M3's servos really move, for the simulator's joint drives (NEXT.md, "After the event").
The arm reads none of its servos back (roboarm.m3arm), so the camera on its forearm watches the lab's
ChArUco board lying still on the table: every change in the camera's pose is the arm's. One joint
turning turns the camera by the same angle whatever the lever, so joints 1-4 are each measured on
their own (5 and 6 are past the camera).

    python tools/m3_servo_probe.py static  [--joints 1 2 3 4] [--delta 8] [--repeats 3]
        each joint brought to the look pose from above and from below, again and again: the
        backlash (the two sides apart), the repeatability (the spread on one side) and the gain
        (the angle turned for 2 x delta commanded) -- in degrees and in mm at 25 cm
    python tools/m3_servo_probe.py dynamic [--joints 1 2 3 4] [--delta 15] [--speeds 15 30 60 90]
        each joint swept by --delta at each speed while the camera watches: how far behind the
        command it runs, when it is there, how far it overshoots

Run ON THE ROBOT (`ssh jetson@192.168.73.66`, then in ~/roboarm `.venv/bin/python tools/...`): the
pictures' capture stamps and the command's time are then one clock, and /rgbd is local. The board
as for tools/m3_touch.py, flat in the pick zone, seen from LOOK; no job running.
Each run -> out/servo/<stamp>_<kind>.json, every reading in it.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import m3_handeye as he

from roboarm import m3arm

OUT = ROOT / "out" / "servo"
# the board as for tools/m3_touch.py, seen from higher than its look pose (88, 105, 25, 0), which leaves
# the board at the picture's bottom edge: here 20 of its 24 corners, round the middle (2026-10-23)
LOOK = {"1": 88, "2": 115, "3": 0, "4": 0, "5": 90}
SETTLE_S = 1.0     # still before a static reading: the servos are there, the picture sharp
SLOW_DPS = 20      # between readings
LEVER_M = 0.25     # an angle as a miss, this far out
MIN_CORNERS = 8


def reading() -> dict | None:
    """The camera's orientation in the board's frame now, from /rgbd: {"stamp", "R", "corners",
    "commanded"}, or None with too few corners seen (a blurred picture mid-move)."""
    snap = he.http("/rgbd")
    if "error" in snap:
        raise SystemExit(f"/rgbd: {snap['error']}")
    image = cv2.imdecode(np.frombuffer(base64.b64decode(snap["color_jpeg"]), np.uint8), cv2.IMREAD_COLOR)
    pixels, points = he.corners(image)
    if len(pixels) < MIN_CORNERS:
        return None
    _ok, rvec, _tvec = cv2.solvePnP(points, pixels, np.asarray(snap["K"], float), None)
    board_in_camera = cv2.Rodrigues(rvec)[0]
    return {"stamp": float(snap["stamp"]), "R": board_in_camera.T, "corners": len(pixels),
            "commanded": snap["joints"]}


def turned(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The rotation from camera orientation a to b, as a rotation vector (rad) in a's frame."""
    return cv2.Rodrigues(a.T @ b)[0].ravel()


def move(pose: dict, speed: float) -> None:
    reply = he.http("/arm/call", {"method": "move_to", "args": [{str(j): v for j, v in pose.items()}],
                                  "kwargs": {"speed_dps": speed}})
    if "error" in reply:
        raise SystemExit(f"move refused: {reply['error']}")


def still(pose: dict) -> dict:
    """At `pose` (slowly), SETTLE_S still, then a reading -- which must see the board."""
    move(pose, SLOW_DPS)
    time.sleep(SETTLE_S)
    r = reading()
    if r is None:
        raise SystemExit(f"the board is not in view at {pose}: put it where the look pose sees it whole")
    return r


def base_for(joint: int, delta: float) -> dict:
    """The look pose, the joint moved off its limit when +-delta would pass it."""
    pose = {j: float(LOOK[str(j)]) for j in range(1, 6)}
    lo, hi = m3arm.SAFE_LIMITS[joint]
    pose[joint] = min(max(pose[joint], lo + delta), hi - delta)
    return pose


def engaged() -> None:
    if "error" in he.http("/arm/call", {"method": "read"}):   # pose unknown since power-on
        print("engage:", he.http("/arm/call", {"method": "engage"}).get("result"))


def each(joints: list[int], measure) -> dict:
    """measure(joint) for each joint; one that fails (the board out of view, a move refused) is said
    and left, the rest still measured."""
    results = {}
    for joint in joints:
        try:
            results[joint] = measure(joint)
        except SystemExit as exc:
            print(f"J{joint}: {exc}", flush=True)
            results[joint] = {"error": str(exc)}
    return results


def static(joints: list[int], delta: float, repeats: int) -> dict:
    return each(joints, lambda joint: static_one(joint, delta, repeats))


def static_one(joint: int, delta: float, repeats: int) -> dict:
    """The look pose reached from +delta and from -delta, `repeats` times each, read against -delta
    (reached from above, as the readings 'from above' are: their mean over delta is the gain with
    no backlash in it)."""
    base = base_for(joint, delta)
    above, below = dict(base), dict(base)
    above[joint] += delta
    below[joint] -= delta
    r_above, r_below = still(above), still(below)
    axis = turned(r_below["R"], r_above["R"])
    swept = float(np.degrees(np.linalg.norm(axis)))
    axis = axis / np.linalg.norm(axis)
    # a check on the reading: J1 turns about the table's normal, 2-4 about a line along the table
    upright = float(np.degrees(np.arccos(min(1.0, abs((r_below["R"] @ axis)[2])))))
    sides = {"from_above": [], "from_below": []}
    for _ in range(repeats):
        for side, start in (("from_above", above), ("from_below", below)):
            move(start, SLOW_DPS)
            r = still(base)
            sides[side].append(float(np.degrees(turned(r_below["R"], r["R"]) @ axis)))
    a, b = np.array(sides["from_above"]), np.array(sides["from_below"])
    backlash = float(a.mean() - b.mean())
    spread = float(max(a.std(), b.std()))
    gain = float(a.mean() / delta)
    mm = np.radians(1) * LEVER_M * 1000
    print(f"J{joint} at {base[joint]:.0f}: backlash {backlash:+.2f} deg ({backlash * mm:+.1f} mm at "
          f"{LEVER_M * 100:.0f} cm), repeatability sd {spread:.2f} deg ({spread * mm:.1f} mm), gain {gain:.3f} "
          f"({a.mean():.2f} deg for {delta:.0f}; swept {swept:.2f} for {2 * delta:.0f}); "
          f"its axis {upright:.1f} deg off the board's normal", flush=True)
    move(base, SLOW_DPS)
    return {"base": base, "delta": delta, "gain": gain, "swept_deg": swept, "axis_off_normal_deg": upright, **sides,
            "backlash_deg": backlash, "repeat_sd_deg": spread}


def sweep(base: dict, joint: int, delta: float, speed: float) -> dict:
    """One sweep of `joint` by +delta at `speed` from `base`, read all the way."""
    start = still(base)
    goal = dict(base)
    goal[joint] += delta
    seconds = max(0.05, delta / speed)   # as m3arm._sweep times it
    done = threading.Event()
    sent = {}

    def go():
        sent["t"] = time.time()
        try:
            move(goal, speed)
        except SystemExit as exc:
            sent["error"] = str(exc)
        finally:
            done.set()

    threading.Thread(target=go, daemon=True).start()
    samples, tail = [], None
    while tail is None or time.time() < tail:
        r = reading()
        if r is not None:
            samples.append((r["stamp"], r["R"]))
        if done.is_set() and tail is None:
            tail = time.time() + SETTLE_S + 0.5
    if "error" in sent:
        raise SystemExit(sent["error"])
    end = still(goal)
    axis = turned(start["R"], end["R"])
    axis = axis / np.linalg.norm(axis)
    t = np.array([s - sent["t"] for s, _ in samples])
    angle = np.array([np.degrees(turned(start["R"], R) @ axis) for _, R in samples])
    final = float(np.degrees(turned(start["R"], end["R"]) @ axis))

    def commanded(tt):
        return delta * np.clip(tt / seconds, 0.0, 1.0)

    # the lag: the command's ramp shifted to fit the readings best (fitted on the move, not the rest)
    moving = (angle > 0.05 * final) & (angle < 0.95 * final)
    lags = np.arange(-0.1, 0.8, 0.005)
    scale = final / delta
    errors = [np.mean((angle[moving] - scale * commanded(t[moving] - lag)) ** 2) if moving.any() else np.inf
              for lag in lags]
    lag = float(lags[int(np.argmin(errors))])
    near = np.abs(angle - final) < 0.3
    there = next((float(t[i]) for i in range(len(t)) if near[i:].all()), None)
    over = float(max(0.0, angle.max() - final)) if len(angle) else 0.0
    print(f"  J{joint} {delta:+.0f} deg at {speed:.0f} deg/s (commanded {seconds:.2f} s): {len(t)} readings, "
          f"lag {lag * 1000:.0f} ms, there (within 0.3 deg) at {there if there is None else round(there, 2)} s, "
          f"overshoot {over:.2f} deg, turned {final:.2f}", flush=True)
    move(base, SLOW_DPS)
    return {"joint": joint, "delta": delta, "speed": speed, "seconds": seconds, "lag_s": lag,
            "there_s": there, "overshoot_deg": over, "final_deg": final,
            "samples": [[round(float(a), 4), round(float(b), 3)] for a, b in zip(t, angle)]}


def dynamic(joints: list[int], delta: float, speeds: list[float]) -> dict:
    def sweeps(joint):
        base = base_for(joint, delta)
        base[joint] -= delta / 2   # the sweep centred on the look pose: the board stays in view
        return [sweep(base, joint, delta, speed) for speed in speeds]
    return each(joints, sweeps)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("static")
    s.add_argument("--joints", type=int, nargs="+", default=[1, 2, 3, 4])
    s.add_argument("--delta", type=float, default=8.0)
    s.add_argument("--repeats", type=int, default=3)
    d = sub.add_parser("dynamic")
    d.add_argument("--joints", type=int, nargs="+", default=[1, 2, 3, 4])
    d.add_argument("--delta", type=float, default=15.0)
    d.add_argument("--speeds", type=float, nargs="+", default=[15, 30, 60, 90])
    for p in (s, d):
        p.add_argument("--look", type=float, nargs=5, metavar=("J1", "J2", "J3", "J4", "J5"),
                       help="the pose the board is seen from (LOOK: J1 and J2 are refused there, too low over the chassis)")
    args = parser.parse_args()
    if args.look:
        LOOK.update({str(j): v for j, v in zip(range(1, 6), args.look)})
    if any(j not in (1, 2, 3, 4) for j in args.joints):
        raise SystemExit("joints 1-4 only: 5 and 6 turn nothing the camera is on")
    engaged()
    lag = time.time() - (reading() or {"stamp": time.time()})["stamp"]
    print(f"a picture is {lag * 1000:.0f} ms old when read (its capture stamp against this clock)", flush=True)
    if args.cmd == "static":
        result = static(args.joints, args.delta, args.repeats)
    else:
        result = dynamic(args.joints, args.delta, args.speeds)
    move({j: float(LOOK[str(j)]) for j in range(1, 6)}, SLOW_DPS)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.cmd}.json"
    path.write_text(json.dumps({"args": vars(args), "picture_lag_s": lag, "result": result}, indent=1))
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
