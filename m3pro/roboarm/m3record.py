"""Every job the M3's arm does, kept as an episode a policy can learn from (a VLA taught first in
the simulator, then on the real M3 from a few dozen of these: NEXT.md, "After the event"). Ten
times a second while a job runs: the colour picture, the pose the arm was last sent (the M3 reads
none back -- roboarm.m3arm), the base's odometry and the job's step; at the job's end its narrated
log and the record of each pick attempt. Watched from outside, through the picker's status and log:
nothing in the pick itself changes. Under out/episodes/<YYYYmmdd_HHMMSS>_<job>/:

    000000.jpg ...  the colour pictures, one a tick (JPEG 90, as the camera gives them)
    ticks.jsonl     a line a tick: {"i", "t" (s since the job began), "pose" [J1..J6], "step",
                    "odom" [x, y, yaw] | null, "frame_age" (s; null: no picture this tick)}
    episode.json    {"job", "started", "started_unix", "seconds", "ticks", "log" [{"n", "t" (unix),
                    "text"}], "attempts" [the pick records, "picked" among them]}

The action to learn at a tick is the next tick's pose: the command as the board sweeps through it.
The oldest episodes go once all of them pass KEEP_GB.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

HZ = 10.0
KEEP_GB = 20.0
JPEG_QUALITY = 90


class Recorder:
    """`status()` and `log(since)` are the picker's; `colour()` gives (BGR picture, its age s) or
    (None, None); `pose()` the arm's believed pose {1..6} (raises before the first command);
    `odom()` (x, y, yaw) or None."""

    def __init__(self, status: Callable[[], dict], log: Callable[[int], list[dict]],
                 colour: Callable[[], tuple], pose: Callable[[], dict], odom: Callable[[], tuple | None],
                 out: Path, keep_gb: float = KEEP_GB, clock: Callable[[], float] = time.time):
        self.status, self.log, self.colour, self.pose, self.odom = status, log, colour, pose, odom
        self.out, self.keep_gb, self.clock = Path(out), keep_gb, clock
        self.episode: dict | None = None   # the one being written: {"dir", "job", "started", "ticks", "since", "last", ...}

    # ------------------------------------------------------------------------- the loop --
    def run(self, stop: threading.Event, hz: float = HZ) -> None:
        """Tick until `stop` is set; a failing tick is said once and the loop goes on."""
        said = None
        while not stop.wait(1.0 / hz):
            try:
                self.tick()
                said = None
            except Exception as exc:  # noqa: BLE001 -- the recorder must never take the bridge down
                if str(exc) != said:
                    print(f"recorder: {exc}", flush=True)
                said = str(exc)

    def tick(self) -> None:
        """One look at the picker: an episode opened, added to or closed."""
        state = self.status()
        if state.get("busy"):
            if self.episode is None:
                self._open(state)
            self._add(state)
        elif self.episode is not None:
            self._close(state)

    # ------------------------------------------------------------------------ episodes --
    def _open(self, state: dict) -> None:
        self._prune()
        now = self.clock()
        job = state.get("job") or "job"
        folder = self.out / f"{time.strftime('%Y%m%d_%H%M%S', time.localtime(now))}_{job}"
        folder.mkdir(parents=True, exist_ok=True)
        seen = self.log(0)
        self.episode = {"dir": folder, "job": job, "started": now, "ticks": 0,
                        "since": max((e["n"] for e in seen), default=0), "log": [], "attempts": [],
                        "last": state.get("last")}

    def _add(self, state: dict) -> None:
        ep = self.episode
        try:
            pose = self.pose()
        except Exception:  # noqa: BLE001 -- before the arm's first command its pose is unknown: no tick
            pose = None
        self._news(state)
        if pose is None:
            return
        picture, age = self.colour()
        if picture is not None:
            cv2.imwrite(str(ep["dir"] / f"{ep['ticks']:06d}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        odom = self.odom()
        line = {"i": ep["ticks"], "t": round(self.clock() - ep["started"], 3),
                "pose": [int(pose[j]) for j in range(1, 7)], "step": state.get("step"),
                "odom": None if odom is None else [round(float(v), 4) for v in odom],
                "frame_age": None if picture is None else round(float(age), 3)}
        with open(ep["dir"] / "ticks.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        ep["ticks"] += 1

    def _news(self, state: dict) -> None:
        """The log lines and the pick records come since the last tick."""
        ep = self.episode
        for e in self.log(ep["since"]):
            ep["log"].append(e)
            ep["since"] = max(ep["since"], e["n"])
        last = state.get("last")
        if last is not None and last != ep["last"]:
            ep["attempts"].append(last)
            ep["last"] = last

    def _close(self, state: dict) -> None:
        self._news(state)
        ep, self.episode = self.episode, None
        summary = {"job": ep["job"], "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ep["started"])),
                   "started_unix": round(ep["started"], 3),   # the ticks' t from it; the log's t is unix too
                   "seconds": round(self.clock() - ep["started"], 1), "ticks": ep["ticks"],
                   "error": state.get("error"), "log": ep["log"], "attempts": ep["attempts"]}
        if ep["ticks"] == 0:   # nothing moved that was known: nothing to learn from
            shutil.rmtree(ep["dir"], ignore_errors=True)
            return
        (ep["dir"] / "episode.json").write_text(json.dumps(summary, indent=1, default=_plain), encoding="utf-8")

    def _prune(self) -> None:
        """The oldest episodes out until all of them fit in keep_gb."""
        if not self.out.exists():
            return
        folders = sorted(p for p in self.out.iterdir() if p.is_dir())
        sizes = [sum(f.stat().st_size for f in p.iterdir()) for p in folders]
        while folders and sum(sizes) > self.keep_gb * 1e9:
            with contextlib.suppress(OSError):
                shutil.rmtree(folders.pop(0))
            sizes.pop(0)


def _plain(value):
    """numpy in a pick record, as JSON."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON: {type(value).__name__}")
