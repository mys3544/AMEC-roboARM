"""Where the time goes: named durations, per stage and per job run.

    with metrics.span("pick.descend"):
        ...                                         # timed, recorded under that name
    metrics.record("vision.raised.server", 0.031)   # a duration measured elsewhere

Every duration lands in the session-wide statistics for its name -- count,
total, mean, p50, p90, max, last (the panel's Timing card, GET /api/metrics).
While a RUN is open on a thread -- Session._run_job opens one round every job
-- each span on that thread also goes into the run's timeline, so a finished
run says where its time went from start to finish, and how much of it no span
accounts for. The live view's detection runs on a thread of its own: it counts
in the statistics, never in a run.

Spans nest. A span inside another is recorded with its depth, and a run's
breakdown adds up only its TOP-level spans, so nothing is counted twice; the
nested ones are the detail of where inside those the time went.

Names are dotted by where they are measured: "scan", "refine", "pick", "drop",
"return" at the top of a job; "pick.descend", "detect.depth", "vision.raised",
"arm.move_to" inside them. "*.server" and "*.robot" are the part of a round
trip spent on the far side, as that side measured it; the rest is the network
and the bridge.
"""

from __future__ import annotations

import collections
import contextlib
import json
import math
import threading
import time
from pathlib import Path

# Timed with perf_counter, not monotonic: on Windows (where the panel runs)
# monotonic ticks in 15.6 ms steps, which reads a JPEG encode or a tag search
# as 0 or 15.6 ms.
RECENT = 500        # samples kept per stage for its percentiles
RUNS_KEPT = 20      # finished runs kept for /api/metrics
EVENTS_KEPT = 5000  # timeline entries kept per run (the ride polls the arm ~10 times a second)


class _Stage:
    __slots__ = ("count", "last", "max", "recent", "total")

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.max = 0.0
        self.last = 0.0
        self.recent: collections.deque[float] = collections.deque(maxlen=RECENT)

    def add(self, seconds: float) -> None:
        self.count += 1
        self.total += seconds
        self.max = max(self.max, seconds)
        self.last = seconds
        self.recent.append(seconds)

    def summary(self) -> dict:
        ordered = sorted(self.recent)
        return {
            "count": self.count,
            "total_s": round(self.total, 3),
            "mean_ms": round(1000 * self.total / self.count, 1),
            "p50_ms": round(1000 * _percentile(ordered, 50), 1),
            "p90_ms": round(1000 * _percentile(ordered, 90), 1),
            "max_ms": round(1000 * self.max, 1),
            "last_ms": round(1000 * self.last, 1),
        }


def _percentile(ordered: list[float], p: float) -> float:
    if not ordered:
        return 0.0
    k = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


class Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stages: dict[str, _Stage] = {}
        self._runs: collections.deque[dict] = collections.deque(maxlen=RUNS_KEPT)
        self._local = threading.local()
        # Where finished runs are appended, one JSON object a line, for looking
        # at more than RUNS_KEPT of them afterwards. None: nowhere.
        self.log_path: Path | None = None

    # ----------------------------------------------------------- recording --
    @contextlib.contextmanager
    def span(self, name: str):
        """Time the block and record it under `name`."""
        local = self._local
        depth = getattr(local, "depth", 0)
        local.depth = depth + 1
        start = time.perf_counter()
        try:
            yield
        finally:
            local.depth = depth
            self._add(name, time.perf_counter() - start, start, depth)

    def record(self, name: str, seconds: float) -> None:
        """A duration measured by someone else (the vision service, the bridge),
        recorded as if a span had just ended inside the current one."""
        depth = getattr(self._local, "depth", 0)
        self._add(name, float(seconds), time.perf_counter() - float(seconds), depth)

    @contextlib.contextmanager
    def scope(self, prefix: str):
        """Every name recorded on this thread inside the block gets `prefix.` in
        front: the live view's detection runs the same code as a job's, and its
        numbers should not be mixed with them."""
        local = self._local
        outer = getattr(local, "prefix", "")
        local.prefix = f"{outer}{prefix}."
        try:
            yield
        finally:
            local.prefix = outer

    def _add(self, name: str, seconds: float, start: float, depth: int) -> None:
        name = getattr(self._local, "prefix", "") + name
        with self._lock:
            stage = self._stages.get(name)
            if stage is None:
                stage = self._stages[name] = _Stage()
            stage.add(seconds)
        run = getattr(self._local, "run", None)
        if run is not None and len(run["events"]) < EVENTS_KEPT:
            run["events"].append({"name": name, "at_s": round(start - run["t0"], 3),
                                  "s": round(seconds, 4), "depth": depth})

    @contextlib.contextmanager
    def run(self, name: str):
        """Collect every span on this thread into one run, start to finish.
        Yields the run's dict; the caller may set its "status"."""
        run = {"name": name, "started": time.time(), "t0": time.perf_counter(),
               "status": "running", "events": []}
        local = self._local
        outer = getattr(local, "run", None), getattr(local, "depth", 0)
        local.run, local.depth = run, 0
        try:
            yield run
        finally:
            local.run, local.depth = outer
            finished = _summarise(run, time.perf_counter() - run.pop("t0"))
            with self._lock:
                self._runs.append(finished)
            if self.log_path is not None:
                with contextlib.suppress(OSError), open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(finished) + "\n")

    # ------------------------------------------------------------- reading --
    def snapshot(self, events: bool = True) -> dict:
        """Stage statistics (biggest total first) and the finished runs (newest
        first). The newest run keeps its timeline when `events`."""
        with self._lock:
            stages = {name: stage.summary() for name, stage in self._stages.items()}
            runs = [dict(r) for r in reversed(self._runs)]
        for i, run in enumerate(runs):
            if i or not events:
                run.pop("events", None)
        ordered = dict(sorted(stages.items(), key=lambda kv: -kv[1]["total_s"]))
        return {"stages": ordered, "runs": runs}

    def reset(self) -> None:
        with self._lock:
            self._stages.clear()
            self._runs.clear()


def _summarise(run: dict, total: float) -> dict:
    """The run's own totals: per top-level stage (what adds up to the run) and
    per name at every depth, and what no top-level span covers."""
    events = run["events"]
    top: dict[str, dict] = {}
    every: dict[str, dict] = {}
    for event in events:
        for table in ((top, every) if event["depth"] == 0 else (every,)):
            entry = table.setdefault(event["name"], {"count": 0, "total_s": 0.0})
            entry["count"] += 1
            entry["total_s"] += event["s"]
    covered = sum(entry["total_s"] for entry in top.values())
    for table in (top, every):
        for entry in table.values():
            entry["total_s"] = round(entry["total_s"], 3)
            entry["share"] = round(entry["total_s"] / total, 4) if total > 0 else 0.0
    return {
        **run,
        "total_s": round(total, 3),
        "unaccounted_s": round(max(0.0, total - covered), 3),
        "top": dict(sorted(top.items(), key=lambda kv: -kv[1]["total_s"])),
        "stages": dict(sorted(every.items(), key=lambda kv: -kv[1]["total_s"])),
    }


REGISTRY = Registry()
span = REGISTRY.span
record = REGISTRY.record
run = REGISTRY.run
scope = REGISTRY.scope
snapshot = REGISTRY.snapshot
reset = REGISTRY.reset
