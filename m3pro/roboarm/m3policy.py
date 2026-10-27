"""The policy job on the M3's bridge (POST /pick/policy): a policy trained on the M3's own recordings
(ACT: tools/m3_policy.py) drives the arm ten times a second from beside the arm and the camera -- no
WiFi in the loop. Run from the laptop (2026-10-23) it made 4.5-5.5 steps a second, not 10: each move
waited for the arm and two calls crossed the WiFi (some steps 1.9 s), and its 40 deg/s cap was under
half the recordings' speed. Here every tick is the recordings' tick:

  * the newest colour picture (no older than STALE_S) and the pose believed now -- as the recorder
    (roboarm.m3record) kept them, so as the policy learnt them;
  * once a chunk (every n_action_steps ticks) the policy service (tools/policy_service.py, TensorRT on
    127.0.0.1:8763) turns them into the next chunk_size poses;
  * the next of them, held to max_dps / HZ a joint, inside the limits and J6 <= 150 (closing past a
    piece stalls and can burn its servo), goes to the arm as one command over 1 / HZ s, not waited for
    (M3Arm.step_to), checked by the guard against the chassis, the lidars and the floor -- a refusal
    ends the job;
  * the clock is fixed: a tick runs late only when its work takes longer than a tick.

A STOP (the console's, POST /pick/stop) ends it at the next tick; however it ends, the arm holds where
it is. dry: the same, nothing sent, the pose given as START. Each run -> out/policy/run_STAMP.jsonl, a
line a tick; and the recorder keeps it as an episode, as every job.
"""

from __future__ import annotations

import base64
import contextlib
import json
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

from roboarm import m3arm
from roboarm.m3arm import ArmError

SERVICE = "http://127.0.0.1:8763"
HZ = 10
START = {1: 90, 2: 105, 3: 10, 4: 0, 5: 90, 6: 76}   # where every recorded episode began and ended
LIMITS = {**m3arm.SAFE_LIMITS, 6: (0, 150)}
FLOOR_CLEAR = 0.003   # m the fingertips keep over the table, by the arm model. 8 (the picker's first try) until
                      # 2026-10-27: with SAG J2 1.1 the recorded grips' fingertips sit 3.4 mm up (median; 96 % under 8),
                      # where they touched the table -- the policy's learned grip was refused 7 mm up (the user's: 3 mm)
MAX_DPS = 100.0       # the recordings' fastest: the picker's moves reach 100 deg/s on J1-J5 (m3_pick_20261023b)
STALE_S = 0.5         # an older picture is the camera stopped, for a policy that looks ten times a second
JPEG_QUALITY = 90     # as the recorder keeps them
READY_S = 60.0        # the service may still be loading its engine


def ask(path: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    request = urllib.request.Request(SERVICE + path, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            answer = json.loads(reply.read())
    except urllib.error.HTTPError as exc:   # it answered, with what went wrong
        raise RuntimeError(f"the policy: {exc.read().decode(errors='replace')[:300]}") from exc
    except OSError as exc:
        raise RuntimeError(f"the policy does not answer ({exc}): is roboarm-m3policy running?") from exc
    if "error" in answer:
        raise RuntimeError(f"the policy: {answer['error']}")
    return answer


def ready(clock=time.monotonic, sleep=time.sleep) -> dict:
    """The service's /health, waiting up to READY_S for it to come up."""
    deadline = clock() + READY_S
    while True:
        try:
            return ask("/health", timeout=2.0)
        except RuntimeError:
            if clock() > deadline:
                raise
            sleep(1.0)


def safe_step(here: dict, wanted, max_dps: float) -> dict:
    """The policy's pose made a step the arm may take in a tick: every joint at most max_dps / HZ from
    where it is, inside LIMITS, whole degrees."""
    most = max_dps / HZ
    return {j: round(min(max(here[j] + float(np.clip(wanted[j - 1] - here[j], -most, most)), LIMITS[j][0]), LIMITS[j][1]))
            for j in m3arm.JOINTS}


def run(task, colour, seconds: float = 30.0, max_dps: float = MAX_DPS, dry: bool = False,
        clock=time.monotonic, sleep=time.sleep) -> None:
    """`task`: the picker (its call, here, check, step, say, out); `colour()` -> (BGR picture, its age s)."""
    seconds, max_dps = min(max(float(seconds), 1.0), 120.0), min(max(float(max_dps), 5.0), MAX_DPS)
    health = ready(clock, sleep)
    model, n = health["model"], int(health["model"]["n_action_steps"])
    task.say(f"policy {model['checkpoint']} ({health['precision']}), {seconds:.0f} s, at most {max_dps:.0f} deg/s"
             + (" -- dry: nothing sent" if dry else ""))
    if not dry:
        with task.step("to the start"):
            try:
                task.call("read")
            except ArmError:   # not engaged since the bridge started: its pose is unknown
                task.call("engage")
            task.call("move_to", START, speed_dps=20, settle=True)
    out = (task.out.parent if task.out is not None else None)
    log_path = None if out is None else out / "policy" / f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    queue: list = []
    ticks = late = 0
    infer: list[float] = []
    began = tick_at = clock()
    try:
        with task.step("policy"), (open(log_path, "w", encoding="utf-8") if log_path else contextlib.nullcontext()) as log:
            while clock() - began < seconds:
                task.check()   # a STOP: Stopped, and the hold below
                start = clock()
                frame, age = colour()
                if frame is None or age is None or age > STALE_S:
                    raise ArmError("no fresh picture from the camera" + ("" if age is None else f" ({age:.1f} s old)"))
                here = dict(START) if dry else task.here()
                if not queue:
                    jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1]
                    asked = clock()
                    queue = ask("/act", {"state": [here[j] for j in m3arm.JOINTS],
                                         "jpeg": base64.b64encode(jpeg).decode()})["chunk"][:n]
                    infer.append(clock() - asked)
                wanted = queue.pop(0)
                step = safe_step(here, wanted, max_dps)
                if not dry:
                    task.call("step_to", step, 1.0 / HZ, floor_clear=FLOOR_CLEAR)
                ticks += 1
                if log is not None:
                    log.write(json.dumps({"t": round(start - began, 3), "here": [here[j] for j in m3arm.JOINTS],
                                          "wanted": np.round(wanted, 1).tolist(),
                                          "sent": None if dry else [step[j] for j in m3arm.JOINTS],
                                          "picture_age": round(float(age), 3),
                                          "ms": round(1000 * (clock() - start), 1)}) + "\n")
                if ticks % HZ == 0:
                    task.say(f"policy {clock() - began:4.1f} s: wanted {np.round(wanted).astype(int).tolist()} "
                             f"sent {[step[j] for j in m3arm.JOINTS]}")
                tick_at += 1.0 / HZ
                wait = tick_at - clock()
                if wait > 0:
                    sleep(wait)
                else:
                    late += 1
                    if wait < -1.0 / HZ:   # far behind: count the ticks afresh from now
                        tick_at = clock()
    finally:
        if not dry:
            with contextlib.suppress(Exception):
                task.call("hold")
        took = max(clock() - began, 1e-9)
        task.say(f"policy: {ticks} ticks in {took:.1f} s ({ticks / took:.1f} a second), {late} late; the policy "
                 f"answered in {1000 * float(np.median(infer)) if infer else 0:.0f} ms (median of {len(infer)})"
                 + (f" -> {log_path}" if log_path else ""))
