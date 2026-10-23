"""A policy trained on the M3's recordings (tools/m3_episodes.py -> LeRobot; ACT on 39 real episodes,
2026-10-23) judged on the episodes held out of its training, and run on the real M3. In LeRobot's own
python (~/.roboarm/lerobot, CUDA torch: the laptop's GPU does the inference):

    python tools/m3_policy.py eval CHECKPOINT [--dataset DIR]
        each frame of the episodes the training held out (LeRobot's eval_split: the last tenth of each
        task's) -> the policy's chunk against what the picker did next: the mean error per joint at the
        next step and a second ahead, beside "hold still"
    python tools/m3_policy.py run CHECKPOINT [--seconds 30] [--max-dps 40] [--dry]
        on the robot, from the pose every episode began at (the wide look, the fingers open for a 30 mm
        cube): ten times a second the camera's newest picture (the bridge's MJPEG stream: /rgbd took
        57-532 ms over the WiFi) and the pose sent -> the policy's next pose -> the bridge

Safety, run: the bridge's guard checks every pose sent against the chassis, the lidars and the table,
as for the picker, and a refusal ends the run; each step moves no joint more than --max-dps / 10, the
gripper never past J6 150 (closing past a piece stalls and can burn its servo); nothing moves while no
command comes (the board holds the last); STOP: Space or Esc here, the console's STOP (the bridge's
interrupt ends the move under way and so the run), Ctrl-C, or the power switch. --dry predicts and
prints, sending nothing. No pick job may run meanwhile. Each run -> out/policy/run_STAMP.jsonl.
CHECKPOINT: a run's checkpoints/NNNNNN (its pretrained_model inside).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = "http://192.168.73.66:8761"
DATASET = ROOT / "out" / "lerobot" / "m3_pick_20261023"
OUT = ROOT / "out" / "policy"
START = {1: 90, 2: 105, 3: 10, 4: 0, 5: 90, 6: 76}   # where every recorded episode began and ended
TASK = "pick up the 30 mm cube and put it in the box"
LIMITS = {1: (0, 180), 2: (0, 180), 3: (0, 180), 4: (0, 180), 5: (0, 270), 6: (0, 150)}   # J6: see above
FLOOR_CLEAR = 0.008   # m the fingertips keep over the table, as the picker's first try (its setting)
HZ = 10


def load(checkpoint: Path):
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    path = checkpoint / "pretrained_model" if (checkpoint / "pretrained_model").exists() else checkpoint
    policy = ACTPolicy.from_pretrained(str(path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(path))
    return policy, pre, post, device


# ----------------------------------------------------------------------------------- eval --
def held_out(meta) -> list[int]:
    """LeRobot's eval_split=0.1: the last ceil(tenth) of each task's episodes, in order."""
    by_task: dict[str, list[int]] = {}
    for ep in range(meta.total_episodes):
        tasks = meta.episodes["tasks"][ep]
        by_task.setdefault(tasks[0] if tasks else "", []).append(ep)
    return [ep for eps in by_task.values() for ep in eps[len(eps) - math.ceil(len(eps) * 0.1):]]


def evaluate(checkpoint: Path, dataset: Path) -> None:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    policy, pre, post, _device = load(checkpoint)
    chunk = policy.config.chunk_size
    ds = LeRobotDataset(f"local/{dataset.name}", root=dataset,
                        delta_timestamps={"action": [i / HZ for i in range(chunk)]})
    episodes = held_out(ds.meta)
    errors, still = [], []
    for ep in episodes:
        rows = [i for i in range(len(ds)) if int(ds.hf_dataset[i]["episode_index"]) == ep]
        for i in rows:
            item = ds[i]
            batch = {"observation.images.wrist": item["observation.images.wrist"][None],
                     "observation.state": item["observation.state"][None], "task": [item["task"]]}
            with torch.inference_mode():
                predicted = post(policy.predict_action_chunk(pre(batch)))[0].float().cpu().numpy()
            truth, pad = item["action"].numpy(), item["action_is_pad"].numpy()
            keep = ~pad
            errors.append(np.where(keep[:, None], np.abs(predicted - truth), np.nan))
            still.append(np.where(keep[:, None], np.abs(item["observation.state"].numpy()[None] - truth), np.nan))
        print(f"episode {ep}: {len(rows)} frames, task {ds[rows[0]]['task']!r}", flush=True)
    e, s = np.array(errors), np.array(still)
    for name, step in (("next step", 0), ("1 s ahead", min(9, chunk - 1))):
        print(f"{name:10s} mean |error| per joint (deg) J1-J6: policy {np.round(np.nanmean(e[:, step], 0), 1).tolist()}"
              f"  hold still {np.round(np.nanmean(s[:, step], 0), 1).tolist()}", flush=True)
    print(f"over the chunk ({chunk} steps): policy {np.nanmean(e):.2f} deg, hold still {np.nanmean(s):.2f} deg "
          f"({len(episodes)} held-out episodes, {len(e)} frames)", flush=True)


# ------------------------------------------------------------------------------------ run --
def call(method: str, *args, **kwargs):
    body = json.dumps({"method": method, "args": list(args), "kwargs": kwargs}).encode()
    req = urllib.request.Request(BRIDGE + "/arm/call", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            reply = json.load(r)
    except urllib.error.HTTPError as exc:   # the bridge's refusals come as 400 with the reason in the body
        reply = json.loads(exc.read() or b"{}") or {"error": f"HTTP {exc.code}"}
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply.get("result")


def get(path: str):
    with urllib.request.urlopen(BRIDGE + path, timeout=5) as r:
        return json.load(r)


class Stream:
    """The bridge's MJPEG stream read on its own thread: the newest picture (RGB) and when it came."""

    def __init__(self):
        self.frame, self.stamp, self.closed = None, 0.0, False
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        with urllib.request.urlopen(BRIDGE + "/stream.mjpg", timeout=10) as r:
            buf = b""
            while not self.closed:
                buf += r.read(16384)
                end = buf.rfind(b"\xff\xd9")
                start = buf.rfind(b"\xff\xd8", 0, end) if end >= 0 else -1
                if start >= 0:
                    jpeg, buf = buf[start:end + 2], buf[end + 2:]
                    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if image is not None:
                        self.frame, self.stamp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB), time.time()

    def newest(self, max_age: float = 0.5):
        if self.frame is None or time.time() - self.stamp > max_age:
            return None
        return self.frame


def stop_pressed() -> bool:
    try:
        import msvcrt
    except ImportError:
        return False
    while msvcrt.kbhit():
        if msvcrt.getch() in (b" ", b"\x1b"):
            return True
    return False


def safe_step(here: dict, wanted: np.ndarray, max_dps: float) -> dict:
    """The policy's pose made a step the arm may take: every joint at most max_dps / HZ from where it is,
    inside its limits, whole degrees."""
    most = max_dps / HZ
    out = {}
    for j in range(1, 7):
        lo, hi = LIMITS[j]
        out[j] = round(min(max(here[j] + float(np.clip(wanted[j - 1] - here[j], -most, most)), lo), hi))
    return out


def run(checkpoint: Path, seconds: float, max_dps: float, dry: bool) -> None:
    from lerobot.common.control_utils import predict_action

    state = get("/pick/state")
    if state.get("busy"):
        raise SystemExit(f"a pick job is running ({state.get('job')}): stop it first")
    health = get("/health")
    if not health["arm"]["connected"]:
        raise SystemExit(f"the arm is not connected: {health['arm']['error']}")
    policy, pre, post, device = load(checkpoint)
    stream = Stream()
    for _ in range(50):   # its first picture: a second or two after connecting
        if stream.newest() is not None:
            break
        time.sleep(0.1)
    else:
        raise SystemExit("no picture from the bridge's stream in 5 s")
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"battery {state.get('battery_v')} V; to the start {START}" + (" (dry: not moving)" if dry else ""), flush=True)
    if not dry:
        try:
            call("read")
        except RuntimeError:
            call("engage")
        call("move_to", {str(j): v for j, v in START.items()}, speed_dps=20, settle=True)
    policy.reset()
    began, steps, why = time.time(), 0, "time up"
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            while time.time() - began < seconds:
                tick = time.time()
                if stop_pressed():
                    why = "STOP pressed"
                    break
                frame = stream.newest()
                if frame is None:
                    why = "no picture from the camera for half a second"
                    break
                here = {int(j): float(v) for j, v in call("read").items()} if not dry else dict(START)
                obs = {"observation.images.wrist": frame,
                       "observation.state": np.array([here[j] for j in range(1, 7)], np.float32)}
                wanted = predict_action(obs, policy, device, pre, post, use_amp=device.type == "cuda",
                                        task=TASK, robot_type="m3pro").float().cpu().numpy().reshape(-1)
                step = safe_step(here, wanted, max_dps)
                sent = None
                if not dry:
                    try:
                        call("move_to", {str(j): v for j, v in step.items()}, speed_dps=max_dps, settle=False,
                             floor_clear=FLOOR_CLEAR)
                        sent = step
                    except RuntimeError as exc:
                        why = f"refused or stopped: {exc}"
                        break
                log.write(json.dumps({"t": round(tick - began, 3), "here": here, "wanted": np.round(wanted, 1).tolist(),
                                      "sent": sent, "picture_age": round(tick - stream.stamp, 3),
                                      "ms": round(1000 * (time.time() - tick))}) + "\n")
                steps += 1
                if steps % HZ == 0:
                    print(f"  {time.time() - began:4.1f} s: wanted {np.round(wanted).astype(int).tolist()} "
                          f"sent {list(step.values())}", flush=True)
                time.sleep(max(0.0, 1.0 / HZ - (time.time() - tick)))
    except KeyboardInterrupt:
        why = "Ctrl-C"
    finally:
        stream.closed = True
        if not dry:
            try:
                call("hold")
            except (RuntimeError, OSError):
                pass
    print(f"ended after {steps} steps, {time.time() - began:.1f} s: {why} -> {log_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("checkpoint", type=Path)
    e.add_argument("--dataset", type=Path, default=DATASET)
    r = sub.add_parser("run")
    r.add_argument("checkpoint", type=Path)
    r.add_argument("--seconds", type=float, default=30.0)
    r.add_argument("--max-dps", type=float, default=40.0)
    r.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.cmd == "eval":
        evaluate(a.checkpoint, a.dataset)
    else:
        run(a.checkpoint, a.seconds, a.max_dps, a.dry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
