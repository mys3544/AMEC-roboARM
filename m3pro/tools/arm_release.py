#!/usr/bin/env python3
"""Release the arm so it can be moved by hand, and watch what each joint reports.

    docker compose run --rm core python tools/arm_release.py          # release + watch
    docker compose run --rm core python tools/arm_release.py --engage # re-stiffen

SAFETY: releasing torque makes the arm limp and it WILL sag under gravity.
Support it, or lower it to a resting position, before the countdown ends.

The diagnostic: torque-off is a command the servo has to *receive*. Reading a
servo needs comms in the other direction. Watching which of those two still
works for J3 tells us where the fault is -- see the summary this prints.
"""

import argparse
import sys
import time

from Rosmaster_Lib import Rosmaster

from roboarm import arm
from roboarm import config as cfg


def connect() -> Rosmaster:
    bot = Rosmaster(com=cfg.SERIAL_PORT)
    bot.create_receive_threading()
    time.sleep(1.0)
    if bot.get_version() == -1:
        sys.exit("board not responding -- is something else holding the port?")
    return bot


def read_all(bot: Rosmaster) -> dict[int, int | None]:
    """Every joint, unclamped so a joint sitting outside its range reports its real
    angle instead of being masked as -1. None = answered neither reader.

    roboarm.arm.read_pose: the batch call, plus a SPACED direct query for what it
    drops and for anything at or below zero -- where J2 now goes, where -1 is a
    real angle and the batch rounds a degree high. Fired back to back the two
    readers corrupt each other; read_pose spaces them.
    """
    return arm.read_pose(bot, attempts=1)


def read_pose(bot: Rosmaster) -> dict[int, int]:
    """Joints that are readable AND inside their limits. Missing key = one or the other."""
    pose = {}
    for _ in range(3):
        for sid, value in read_all(bot).items():
            lo, hi = cfg.HARD_LIMITS[sid]
            if value is not None and lo <= value <= hi:
                pose[sid] = value
        if len(pose) == len(cfg.JOINT_IDS):
            break
        time.sleep(0.05)
    return pose


def engage(bot: Rosmaster) -> None:
    """Re-stiffen, pulling any joint that has drifted out of range back inside.

    Left unpowered, the arm sags and joints park past the limits the SDK can
    command -- we have seen it on J3, J4 and J6. Recovery is simply to command a
    legal angle: the SDK validates the angle you *send*, not where the joint
    currently sits, so a joint at 4 degrees can be told to go to 35 and will.
    """
    pose = read_all(bot)
    missing = [s for s, v in pose.items() if v is None]
    if missing:
        sys.exit(
            f"refusing to engage: J{', J'.join(str(s) for s in missing)} is not replying "
            f"at all. That is a bus problem, not a range problem -- check the connector."
        )

    target, rescued = {}, {}
    for sid, value in pose.items():
        lo, hi = cfg.SAFE_LIMITS[sid]
        target[sid] = min(max(value, lo), hi)
        if target[sid] != value:
            rescued[sid] = (value, target[sid])

    print("holding at:", " ".join(f"J{s}={target[s]}" for s in cfg.JOINT_IDS))
    if rescued:
        print("recovering :", " ".join(f"J{s} {a}->{b}" for s, (a, b) in rescued.items()),
              "(out of range, will move)")
    print("\nKeep a hand on the arm. Torque on in:")
    for i in (3, 2, 1):
        print(f"  {i}...", flush=True)
        time.sleep(1)

    bot.set_uart_servo_torque(True)
    # Target = present position, so there is nothing to snap towards. run_time is
    # slowed from the SDK's 500 ms default so any correction is gentle. send_pose,
    # not the angle array: J2 may be held below 0, which the array silently drops --
    # and then the servo snaps to whatever stale setpoint it still had.
    arm.send_pose(bot, target, run_time=1500)
    time.sleep(2.0)

    after = {s: v for s, v in read_all(bot).items() if v is not None}
    drift = {s: after[s] - target[s] for s in cfg.JOINT_IDS if s in after}
    print("\ntorque ON")
    print("now at:    ", " ".join(f"J{s}={after.get(s, '?')}" for s in cfg.JOINT_IDS))
    moved = {s: d for s, d in drift.items() if abs(d) > cfg.READBACK_TOLERANCE_DEG}
    if moved:
        print("moved:     ", " ".join(f"J{s}{d:+d} deg" for s, d in moved.items()))
    else:
        print("moved:      nothing beyond readback tolerance -- the arm held position")


def release_and_watch(bot: Rosmaster, seconds: int) -> None:
    before = read_all(bot)
    print("pose before release:", " ".join(f"J{s}={v if v is not None else 'no reply'}"
                                           for s, v in before.items()))

    print("\n*** SUPPORT THE ARM NOW -- it goes limp in: ***")
    for i in (5, 4, 3, 2, 1):
        print(f"  {i}...", flush=True)
        time.sleep(1)

    bot.set_uart_servo_torque(False)
    print("\ntorque OFF -- move each joint by hand. Ctrl-C when done.\n")

    seen: dict[int, list[int]] = {s: [] for s in cfg.JOINT_IDS}
    end = time.time() + seconds
    try:
        while time.time() < end:
            for sid, value in read_all(bot).items():
                if value is not None:
                    seen[sid].append(value)
            line = []
            for sid in cfg.JOINT_IDS:
                vals = seen[sid]
                if not vals:
                    line.append(f"J{sid}:--")
                else:
                    span = max(vals) - min(vals)
                    line.append(f"J{sid}:{vals[-1]:>4}{'*' if span >= 3 else ' '}")
            print("  " + "  ".join(line) + "   (* = moved)", end="\r", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass

    print("\n\n--- what each joint reported ---")
    silent, surveyed = [], {}
    for sid in cfg.JOINT_IDS:
        vals = seen[sid]
        if not vals:
            silent.append(sid)
            print(f"  J{sid}: never replied")
        else:
            span = max(vals) - min(vals)
            print(f"  J{sid}: {min(vals)}..{max(vals)}  (moved {span} deg)")
            if span >= 10:  # only a real sweep says anything about where the stops are
                surveyed[sid] = (min(vals), max(vals))

    if surveyed:
        # Pull in from the observed extremes: the hard stop is where damage starts,
        # and readback is only trustworthy to a couple of degrees.
        margin = int(cfg.READBACK_TOLERANCE_DEG) + 1
        print("\n--- reach survey -> paste into SAFE_LIMITS in roboarm/config.py ---")
        print("    (only joints swept 10 deg or more are shown)")
        for sid, (lo, hi) in sorted(surveyed.items()):
            hard_lo, hard_hi = cfg.HARD_LIMITS[sid]
            print(f"    {sid}: ({max(hard_lo, lo + margin)}, {min(hard_hi, hi - margin)}),")

    print("\n--- reading it ---")
    if not silent:
        print("  All six replied. The bus is healthy.")
        return
    print(f"  Servo {', '.join(str(s) for s in silent)} never replied, so it cannot talk to us.")
    print("  Now the part only you can answer -- with torque off, that joint felt:")
    print()
    print("    LIMP, like the others  -> it DID receive the torque-off command.")
    print("                              Comms work one way only: check the data")
    print("                              line / connector pins, not power.")
    print()
    print("    STILL STIFF            -> it received nothing. Dead in both")
    print("                              directions: suspect the connector or")
    print("                              power to that servo.")
    print()
    print("    ALREADY LOOSE before   -> it has no power at all, or has failed.")
    print()
    print("  Torque is left OFF. Re-stiffen with:  ... arm_release.py --engage")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engage", action="store_true", help="re-enable torque instead")
    parser.add_argument("--seconds", type=int, default=180, help="how long to watch")
    args = parser.parse_args()

    bot = connect()
    try:
        if args.engage:
            engage(bot)
        else:
            release_and_watch(bot, args.seconds)
    finally:
        del bot
    return 0


if __name__ == "__main__":
    sys.exit(main())
