"""The jobs both robots do, the same way (the user's, 2026-10-07: "I want X3 to behave like M3, so
they can perform the same tasks; only difference is that X3 has to use different object detection"):

    pick_once   the drop-off box looked for, the nearest kit thing in reach picked (the base may
                drive up to it), dropped into the box -- or on the line out on the right -- home
    clean_up    every kit thing in reach, one by one; a spot refused or missed twice is left
    home        the arm home
    drop        what is held, to the drop-off

One job at a time, on a thread of its own; what it is doing, its log, its settings and the last
look's marked picture for whoever watches -- the robot side's /pick/* (roboshared.serve), which
the console's arm card reads: the same for both robots.

`Task` is the flow; a robot's subclass is its body (m3pro/roboarm/m3pick.py, x3plus's x3pick.py):
how it looks and what it finds (wide_look, the found dicts below), how it plans and makes a grip
(plan, grip), how it carries and lets go (way_into, drop_on_line, let_go, put_back), where it looks
for the box (go_box_look, to_box_view, view), how its base nudges up to things (approach, why_not)
-- and its constants (SETTINGS, AXIS, FLOOR_Z, FOOTPRINT, DROP_OUT, DROP_SPREAD, BOX_LOOKS).

A found thing is a dict: "centre" (base x, y, z m, z at half its height), "kind" ("cube", "block"
or None: not the kit's), "size" (m across, its narrow way), "length", "height" (its top over the
floor, m), "top" (its top, base z), "footprint" (w, l), "yaw" (deg, a line the fingers may close
along), "period" (90 a cube, 180 a block), "whole" (seen whole); the flow adds "reach".

ROS-free and robot-free: the arm is whatever `call(method, *args, **kwargs)` reaches (the robot
side's roboarm.web.bridge.Bridge), the base whatever `drive(dx, dy, speed)` drives. Every step is
timed (roboshared.metrics: one run per job).
"""

from __future__ import annotations

import collections
import contextlib
import json
import math
import threading
import time
import traceback
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

from roboshared import box as boxes
from roboshared import metrics

MATCH_M = 0.03   # the close look's thing is the wide look's when within this
PART_M = 0.06    # a thing one look saw cut by the picture's edge is another look's whole one within this
                 # (a kit thing's length): from the side looks a lying block cut short read as a cube
                 # 3-4 cm off its middle (2026-10-05, the tests)
MAX_MISSES = 2   # clean_up gives a spot up after this many misses in a row
APPROACHES = 3   # drives towards things before a pick gives up
DRIVE_GAP = 0.03   # m the base's footprint keeps from the drop-off box and the line it has dropped onto


class Stopped(Exception):
    """The job was stopped (pick_stop): the arm holds where it was."""


class NothingToPick(Exception):
    """No kit thing in view, or none in reach."""


class Refused(Exception):
    """The arm refused a move to it (before moving): the thing at `spot` is left for now."""

    def __init__(self, spot: np.ndarray, why: str):
        super().__init__(why)
        self.spot, self.why = np.asarray(spot[:2], float), why


@contextlib.contextmanager
def refusable(spot):
    """The arm's guard refusing a move -- always before it moves -- as Refused(spot)."""
    try:
        yield
    except Stopped:
        raise
    except Exception as exc:   # the bridge's ArmError, or the HTTP wrapper's error
        if "refused" not in str(exc):
            raise
        raise Refused(spot, str(exc)) from exc


def name(o: dict) -> str:
    if o["kind"] == "cube":
        return f"{o['size'] * 1000:.0f} mm cube"
    return "block " + ("lying" if o["length"] > o["size"] else "standing")


def describe(found: list[dict]) -> str:
    return "; ".join(f"{name(o) if o['kind'] else 'something %.0f mm tall' % (o['height'] * 1000)}"
                     f" at x {o['centre'][0]:.3f} y {o['centre'][1]:+.3f}, turned {o['yaw']:.0f} deg"
                     for o in found) or "nothing"


def footprint_gap(offset, slide: float, obstacles: list[np.ndarray], footprint) -> float:
    """m from the base's footprint ((x0, x1), half its width: base frame), driven `offset`
    (forward, left) and then `slide` right, to the nearest of `obstacles` (points, the job's start
    frame); inf when there are none."""
    if not obstacles:
        return math.inf
    (fx0, fx1), half = footprint
    points = np.vstack(obstacles)
    x0, x1 = fx0 + offset[0], fx1 + offset[0]
    y0, y1 = -half + offset[1] - slide, half + offset[1] - slide
    dx = np.maximum.reduce([x0 - points[:, 0], np.zeros(len(points)), points[:, 0] - x1])
    dy = np.maximum.reduce([y0 - points[:, 1], np.zeros(len(points)), points[:, 1] - y1])
    return float(np.hypot(dx, dy).min())


class Task:
    """The flow; a robot's subclass supplies the body (the module's docstring)."""

    # ---- what a robot's subclass sets
    SETTINGS: ClassVar[dict] = {}   # key -> (what it is, unit, lowest, highest, step, default): the console's knobs.
                                 # The flow reads "speed", "free_dps", "near_dps", "gripper_dps", "settle_close_s",
                                 # "release_s", "drive_cm_s", "driven_s", "nudge_cm", "right_cm".
    AXIS = (0.0, 0.0)            # J1's axis, base x, y m: what the arm reaches round
    FLOOR_Z = 0.0                # the floor's height in the base frame, m
    FOOTPRINT = ((-0.16, 0.13), 0.12)   # the base's (x0, x1), half its width: base frame, m
    DROP_OUT = DROP_SPREAD = 0.0  # the line out on the right: from this far out, this much further (m)
    BOX_LOOKS: ClassVar[tuple] = ()   # J1s the box is looked for at, the first where it is looked first

    def __init__(self, call, snapshot, out: Path | None = None, settle_s: float | None = None, rng=None,
                 drive=None, battery=None):
        self.call, self.snapshot, self.out = call, snapshot, out
        self.battery = battery      # () -> volts, or None: the board's reading, without waiting for the arm
        self.settle_s = settle_s    # None: the settings' (the tests set 0)
        self.drive = drive          # drive(dx, dy, speed m/s) the base, m in its own frame -> what it went; None: it cannot
        self.may_drive = False      # per job: what the caller allowed
        self.box: dict | None = None   # the drop-off box this job (box_look), and where it has let go into it
        self.box_drops: list[np.ndarray] = []   # in the box's own terms (in_box): they move with it
        self.box_dir = self.BOX_LOOKS[0] if self.BOX_LOOKS else 0   # J1 it was last seen at: looked for there first
        self.line_drops = 0                     # let go on the line out on the right this job (no box)
        self.offset = np.zeros(2)   # m the base has driven since the job began (forward, left): the drop-off
                                    # box stays where it was, so the base goes back before a drop
        self.cfg = {key: spec[5] for key, spec in self.SETTINGS.items()}
        saved = None if out is None else out / "settings.json"
        if saved is not None and saved.exists():
            with contextlib.suppress(ValueError, OSError):
                self.set_settings(json.loads(saved.read_text()), save=False)
        self.rng = rng or np.random.default_rng()
        self.frame: bytes | None = None   # the last look, marked, as JPEG
        self.found: list[dict] = []
        self.state = {"job": None, "step": "idle", "since": time.time(), "busy": False, "error": None,
                      "last": None, "picked": 0, "missed": 0, "looks": 0}
        self._log: collections.deque = collections.deque(maxlen=300)
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ for the watchers --
    def say(self, text: str) -> None:
        with self._lock:
            self._seq += 1
            self._log.append({"n": self._seq, "t": round(time.time(), 2), "text": text})
        print(text, flush=True)
        if self.out is not None:   # a restart loses the one above
            with contextlib.suppress(OSError), open(self.out / "log.txt", "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text}\n")

    def log(self, since: int = 0) -> list[dict]:
        with self._lock:
            return [e for e in self._log if e["n"] > since]

    def status(self) -> dict:
        with self._lock:
            state = dict(self.state)
        state["now"] = time.time()   # the robot's clock, for how long the step has run
        volts = self.battery() if self.battery is not None else None
        state["battery_v"] = None if volts is None else round(float(volts), 2)
        state["moved_cm"] = [round(100 * float(v), 1) for v in self.offset]
        state["box"] = None if self.box is None else {
            "x": round(float(self.box["centre"][0]), 3), "y": round(float(self.box["centre"][1]), 3),
            "turn": round(self.box["turn"], 1), "drops": len(self.box_drops), "in_it": len(self.box.get("items", ()))}
        state["found"] = [{"kind": o["kind"], "name": name(o) if o["kind"] else None,
                           "x": round(float(o["centre"][0]), 3), "y": round(float(o["centre"][1]), 3),
                           "yaw": round(o["yaw"], 1), "reach": o.get("reach")} for o in self.found]
        return state

    def settings(self) -> list[dict]:
        return [{"key": key, "label": spec[0], "unit": spec[1], "min": spec[2], "max": spec[3], "step": spec[4],
                 "default": spec[5], "value": self.cfg[key]} for key, spec in self.SETTINGS.items()]

    def set_settings(self, values: dict, save: bool = True) -> list[dict]:
        """Change settings (unknown keys refused, values held to their range); they count from the
        next move on, and are kept for the next start."""
        for key, value in values.items():
            if key not in self.SETTINGS:
                raise ValueError(f"no setting {key!r}")
            lo, hi = self.SETTINGS[key][2], self.SETTINGS[key][3]
            self.cfg[key] = min(max(float(value), lo), hi)
        if save and self.out is not None:
            self.out.mkdir(parents=True, exist_ok=True)
            (self.out / "settings.json").write_text(json.dumps(self.cfg, indent=1))
        return self.settings()

    def dps(self, key: str) -> float:
        return self.cfg[key] * self.cfg["speed"] / 100

    def stopping(self) -> bool:
        """A stop is under way: only while a job runs -- a STOP pressed with nothing running must
        not abandon the next move someone else makes through the bridge."""
        return self._stop.is_set() and self.state["busy"]

    def show(self, marked: np.ndarray, found: list[dict] | None = None, tag: str = "") -> None:
        """A look's marked picture for the watchers (GET /pick/frame.jpg), and what it found (`found`,
        when it is a look for things); kept in `out` too."""
        with self._lock:
            self.frame = cv2.imencode(".jpg", marked, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            if found is not None:
                self.found = found
            self.state["looks"] += 1
        if self.out is not None and tag:
            self.out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.out / f"{time.strftime('%H%M%S')}_{tag}.jpg"), marked)

    # --------------------------------------------------------------------------- jobs --
    def start(self, job: str, **kwargs) -> dict:
        """Start `job` ("pick_once", "clean_up", "home", "drop") on its own thread; an error if one runs."""
        work = {"pick_once": self.pick_once, "clean_up": self.clean_up, "home": self.home, "drop": self.drop}[job]
        with self._lock:
            if self.state["busy"]:
                return {"error": f"busy: {self.state['job']} ({self.state['step']})"}
            self._stop.clear()
            self.state.update(job=job, step="starting", since=time.time(), busy=True, error=None)
        self._thread = threading.Thread(target=self._run, args=(job, work, kwargs), daemon=True)
        self._thread.start()
        return {"ok": True, "job": job}

    def stop(self) -> None:
        self._stop.set()

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self, job: str, work, kwargs: dict) -> None:
        error = None
        with metrics.run(job) as run:
            try:
                work(**kwargs)
                run["status"] = "done"
            except Stopped:
                error, run["status"] = "stopped: the arm holds where it was", "stopped"
            except NothingToPick as exc:
                error, run["status"] = str(exc), "nothing"
            except Exception as exc:  # noqa: BLE001 -- shown to the watcher, whatever it is
                error, run["status"] = f"{type(exc).__name__}: {exc}", "failed"
                traceback.print_exc()   # into the bridge's log: where it came from
                if self._stop.is_set():   # a move abandoned by the stop
                    error, run["status"] = "stopped: the arm holds where it was", "stopped"
        if error:
            self.say(error)
        with self._lock:
            self.state.update(step="idle", since=time.time(), busy=False, error=error)
        self._stop.clear()   # it stopped this job only: the arm takes other calls again

    @contextlib.contextmanager
    def step(self, step: str):
        self.check()
        with self._lock:
            self.state.update(step=step, since=time.time())
        with metrics.span(step):
            yield

    def check(self) -> None:
        if self._stop.is_set():
            raise Stopped

    def here(self) -> dict:
        """The arm's pose, as the arm reports it: {joint: angle}."""
        return {int(j): v for j, v in self.call("read").items()}

    def pick_once(self, kind: str | None = None, at=None, drive: bool = False) -> dict:
        """One pick and the drop-off, then home; `drive`: the base may drive up to it."""
        self.may_drive = drive and self.drive is not None
        self.offset = np.zeros(2)
        self.engaged()
        self.box_look()
        try:
            return self.pick(kind, at)[0]
        finally:
            if not self._stop.is_set():
                self.drive_back()
                self.home()

    def clean_up(self, kind: str | None = None, limit: int = 20, drive: bool = False) -> None:
        """Pick every kit thing in reach, one by one, until none is left in reach (or `limit`);
        each pick's last look is the next one's wide look. A spot missed MAX_MISSES times in a row
        is given up. `drive`: the base may drive up to what is out of reach."""
        self.may_drive = drive and self.drive is not None
        self.offset = np.zeros(2)
        self.engaged()
        self.box_look()
        found, misses, skip = None, [], []
        for _ in range(limit):
            try:
                record, found = self.pick(kind, found=found, skip=skip)
            except NothingToPick as exc:
                self.say(f"clean-up done: {exc}" + (f" ({len(skip)} left: refused, or missed twice)" if skip else ""))
                break
            except Refused as exc:   # nothing moved: on with the others, that one left
                self.say(f"left the one at x {exc.spot[0]:.3f} y {exc.spot[1]:+.3f}: {exc.why}")
                skip.append(exc.spot)
                found = None
                continue
            if record["picked"]:
                misses = []
                continue
            misses.append(np.asarray(record["target"][:2]))
            if len(misses) >= MAX_MISSES and np.linalg.norm(misses[-1] - misses[-2]) < MATCH_M:
                # left, and on with the others (2026-10-05: the job stopped there, two more left untried)
                self.say(f"left the one at x {misses[-1][0]:.3f} y {misses[-1][1]:+.3f}: missed twice")
                skip.append(misses[-1])
                misses = []
        self.drive_back()
        self.home()

    def pick(self, kind: str | None = None, at=None, found: list[dict] | None = None,
             skip: list[np.ndarray] = ()) -> tuple[dict, list[dict]]:
        """Pick the nearest kit thing in reach (of `kind`, or the one nearest `at`, base x, y m),
        drop it off and look again from the wide look: (the record of the attempt, what that look
        found -- the next pick's wide look). `found`: the wide look already taken, the arm there
        and the gripper open. Things at the `skip` spots are left; a move the arm refuses on the
        way to the grip raises Refused."""
        started = time.monotonic()
        looked_round = found is None
        if found is None:
            with self.step("look.wide"):
                found = self.wide_look(sides=True)
        at = None if at is None else np.asarray(at, float)
        approach = 0
        while True:
            kit = [o for o in found if o["kind"] and (kind is None or o["kind"] == kind)
                   and all(np.linalg.norm(o["centre"][:2] - spot) > MATCH_M for spot in skip)]
            target = planned = None
            refused = []   # in reach, no plan: the arm would come too near something, or the fingers too low
            with metrics.span("plan"):
                for o in self.order([o for o in kit if o["reach"]], at):
                    planned = self.plan(o, found)
                    if planned:
                        target = o
                        break
                    o["reach"] = False
                    refused.append(o)
            if target is not None:
                break
            if not looked_round:   # the middle look alone: what may be in reach either side of it
                with self.step("look.wide"):
                    found = self.wide_look(sides=True)
                looked_round = True
                continue
            if not kit:
                raise NothingToPick(f"no kit {kind or 'cube or block'} in view")
            move, why = None, "to bring it in reach"
            if self.may_drive and approach < APPROACHES:
                with metrics.span("plan"):
                    move, why = self.approach(refused, found, kit, at)
            if not move:
                raise NothingToPick(self.why_not(kit, found))
            with self.step("drive"):
                self.say(f"driving {100 * move[0]:.0f} cm forward, {100 * move[1]:+.0f} cm left, {why}")
                went = self.base(*move)
                approach += 1
                if at is not None:
                    at = at - went
                found = self.wide_look()
        grip = self.grip(target, planned, found)   # down to it, shut on it, back up: the robot's own way
        held = grip.get("held")   # True / False where the robot can tell (the X3's gripper); None: the look after says
        if held is False:
            self.say("the fingers closed on nothing: nothing to drop off")
        else:
            self.drop(back=(self.offset.copy(), grip.get("line")))
        with self.step("look.after"):   # the next pick's wide look, too
            after = self.wide_look()
        centre = grip["centre"]
        left = [o for o in after if np.linalg.norm(o["centre"][:2] - centre[:2]) < MATCH_M]
        picked = (not left) if held is None else bool(held)
        reading = grip["reading"]
        record = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.monotonic() - started, 1),
                  "kind": name(reading), "target": np.round(centre, 4).tolist(), "size_mm": round(reading["size"] * 1000),
                  "yaw_deg": round(reading["yaw"], 1), **grip.get("record", {}), "picked": picked}
        self._remember(record)
        with self._lock:
            self.state["picked" if picked else "missed"] += 1
        if not picked:
            self.say(f"NOT picked: still there -- {describe(left)}" if left else "NOT picked: closed on nothing")
        else:
            self.say(f"picked the {name(reading)}: {record['seconds']:.0f} s from the wide look to the next")
        return record, after

    # ------------------------------------------------------------------------ the drop --
    def drop(self, back=None) -> None:
        """What is held to the drop-off and let go. Into the box when the job found one: looked at again
        from where it was found (to_box_view: risen and turned, no up and down) -- moved, it is aimed at
        where it is now; not seen there, looked for all round, holding; nowhere, what is held is put
        back where it came from (`back`: (the base's offset then, the robot's own record of the pick))
        and the job stops (the user's, 2026-10-02: the box may be moved between picks) -- then carried
        out over a spot in it (way_into) and straight down. No box at the job's start: on the line out
        on the right, like the X3's (0-DROP_SPREAD further out, at random)."""
        self.drive_back()   # the box did not drive with it
        if self.box is None:
            with self.step("drop"):
                extra = float(self.rng.uniform(0.0, self.DROP_SPREAD))
                self.drop_on_line(extra)
                self.let_go()
                self.line_drops += 1
            self.say(f"dropped {1000 * (self.DROP_OUT + extra):.0f} mm out on the right")
            return
        with self.step("drop"):
            # the box looked at from where it was found (a check pose on the way, and over the spot, read
            # the marker close and askew, 2026-10-05: the box cm and up to 30 deg off, a cube onto its wall)
            self.to_box_view()
            again = self.see_box("box view")
            if again is None:
                self.say(f"the box not seen at J1 {self.box_dir}, where it was: looking for it all round, holding")
                again = self.box_search(holding=True, tried=self.box_dir)
            into = None
            if again is not None:
                if boxes.same_box(again, self.box):
                    self.box = {**again, "items": self.box.get("items", [])}
                else:
                    self.box = again
                    self.say("the box has moved: aimed at where it is now")
                into = self.box_way(self.here())
            if into is None:
                self.put_back(back)
                raise NothingToPick("the drop-off box is not to be found, or nowhere in it in reach: put back"
                                    " what it held where it was -- place the box, its marker facing the arm")
            spot, way = into
            self.carry(way)   # out and round over the spot, straight down onto it
            self.let_go()
        self.box_drops.append(boxes.in_box(self.box, spot))
        self.say(f"dropped into the box at x {spot[0]:.3f} y {spot[1]:+.3f}")

    def box_way(self, here: dict) -> tuple[np.ndarray, list[dict]] | None:
        """The spot in the box to let go over -- the one furthest from what is in it already and
        from this job's drops, of those the arm reaches -- and the way there; None if none."""
        taken = [boxes.out_of_box(self.box, r) for r in [*self.box.get("items", ()), *self.box_drops]]
        spots = boxes.box_spots(self.box)
        self.rng.shuffle(spots)   # ties broken anyhow: not always the same corner
        spots.sort(key=lambda s: -min((float(np.linalg.norm(s - t)) for t in taken), default=1.0))
        for spot in spots:
            way = self.way_into(here, spot)
            if way is not None:
                return spot, way
        return None

    def see_box(self, tag: str) -> dict | None:
        """The box from where the arm is: its marker, the inside from it, and what stands in it (in its
        own terms) -- or None; the marked picture kept."""
        time.sleep(self.settle_s if self.settle_s is not None else self.cfg["settle_close_s"])
        self.check()
        image, K, camera, extra = self.view()
        try:   # the box is a help, never a reason for a look to fail
            seen = boxes.find_marker(image, K, camera, *boxes.BOX_MARKER)
        except Exception as exc:  # noqa: BLE001 -- 2026-10-02: the robot's OpenCV 4.5.4 had no ArucoDetector
            self.say(f"the look for the box failed: {type(exc).__name__}: {exc}")
            seen = None
        if seen is None:
            return None
        box = boxes.box_from_marker(*seen, floor_z=self.FLOOR_Z)
        box["items"] = [boxes.in_box(box, o["centre"]) for o in self.things_in_view(image, K, camera, extra)
                        if boxes.inside(box, o["centre"])]   # what stands in it already
        inv = np.linalg.inv(camera)
        marked = image.copy()
        corners = [K @ ((inv @ np.r_[c, 1.0])[:3] / (inv @ np.r_[c, 1.0])[2])
                   for c in boxes.box_corners(box, floor_z=self.FLOOR_Z)]
        cv2.polylines(marked, [np.array([[int(c[0]), int(c[1])] for c in corners])], True, (0, 255, 0), 2)
        cv2.putText(marked, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        self.show(marked, tag="box")
        return box

    def box_search(self, holding: bool = False, tried: float | None = None) -> dict | None:
        """The box looked for at BOX_LOOKS, where it was last seen first (not at `tried`, looked at
        already); None if nowhere."""
        with self.step("look.box"):
            for j1 in [j for j in [self.box_dir, *(j for j in self.BOX_LOOKS if j != self.box_dir)] if j != tried]:
                self.go_box_look(j1, holding)
                box = self.see_box("box")
                if box is not None:
                    self.box_dir = j1
                    self.say(f"drop-off box at x {box['centre'][0]:.3f} y {box['centre'][1]:+.3f}, turned"
                             f" {box['turn']:.0f} deg, {len(box['items'])} thing(s) in it (looked at J1 {j1})")
                    return box
        return None

    def box_look(self) -> dict | None:
        """At a job's start: the drop-off box, wherever it is in the arm's reach -- or None, and the
        drops go on the line out on the right."""
        self.box, self.box_drops, self.line_drops = self.box_search(), [], 0
        if self.box is None:
            self.say(f"no drop-off box: marker {boxes.BOX_MARKER[1]} not seen all round -- dropping on the line out on the right")
        return self.box

    def dropped_here(self, o: dict) -> bool:
        """`o` lies where this job lets go -- in the drop-off box (its walls and 2 cm round) or on the
        line out on the right once dropped onto -- and is not to be picked (the side looks see there)."""
        p = np.asarray(o["centre"][:2], float) + self.offset   # in the job's start frame, the box's
        if self.box is not None and boxes.inside(self.box, p, boxes.BOX_WALL + 0.02):
            return True
        x = float(self.AXIS[0])
        return bool(self.line_drops) and abs(p[0] - x) < 0.05 \
            and -(self.DROP_OUT + self.DROP_SPREAD + 0.04) < p[1] < -(self.DROP_OUT - 0.04)

    # ------------------------------------------------------------------------- the base --
    def obstacles(self) -> list[np.ndarray]:
        """What the base keeps DRIVE_GAP from, as points round it (the job's start frame): the drop-off
        box, BOX_WALL outside its inside; or the line out on the right, once it has dropped there."""
        if self.box is not None:
            return [boxes.outline_points(boxes.outline(self.box))]
        if self.line_drops:
            x = float(self.AXIS[0])
            near, far = -(self.DROP_OUT - 0.03), -(self.DROP_OUT + self.DROP_SPREAD + 0.03)
            return [boxes.outline_points([np.array(c) for c in ((x - 0.03, near), (x + 0.03, near), (x + 0.03, far), (x - 0.03, far))])]
        return []

    def right_room(self) -> float:
        """m the base may slide right from where it is: right_cm, and no nearer than DRIVE_GAP to
        the drop-off box or the line it has dropped onto."""
        most, obstacles = self.cfg["right_cm"] / 100, self.obstacles()
        slide = 0.0
        while slide + 0.005 <= most + 1e-9 \
                and footprint_gap(self.offset, slide + 0.005, obstacles, self.FOOTPRINT) >= DRIVE_GAP:
            slide += 0.005
        return slide

    def base(self, dx: float, dy: float) -> np.ndarray:
        """Drive the base (m forward, left), at the settings' speed; what it went, added to the
        offset from where the job began; then still for a moment before anyone looks."""
        went = self.drive(dx, dy, self.cfg["drive_cm_s"] / 100)
        went = np.asarray(went if went is not None else (dx, dy), float)
        self.offset = self.offset + went
        time.sleep(self.cfg["driven_s"])
        return went

    def drive_back(self) -> None:
        """Back to where the job began (the drop-off box is there), a move at most at a time."""
        if self.drive is None or float(np.linalg.norm(self.offset)) < 0.005:
            return
        with self.step("drive back"):
            most = self.cfg["nudge_cm"] / 100
            for _ in range(4):
                if float(np.linalg.norm(self.offset)) < 0.005:
                    break
                self.say(f"driving back {100 * self.offset[0]:.0f} cm, {100 * -self.offset[1]:+.0f} cm left, to the start")
                self.base(*np.clip(-self.offset, -most, most))

    def _remember(self, record: dict) -> None:
        with self._lock:
            self.state["last"] = record
        if self.out is not None:
            log = self.out / "picks.json"
            records = json.loads(log.read_text()) if log.exists() else []
            records.append(record)
            log.write_text(json.dumps(records, indent=1))

    # ------------------------------------------------------ what a robot's subclass supplies --
    def engaged(self) -> None:
        """The arm ready to move (the M3's pose is unknown until its first command)."""
        raise NotImplementedError

    def home(self) -> None:
        raise NotImplementedError

    def wide_look(self, sides: bool = False) -> list[dict]:
        """To the wide look, the gripper opening on the way, and what it shows (found dicts, each with
        "reach"); `sides`: the looks either side too. Leaves out what dropped_here, and calls show()."""
        raise NotImplementedError

    def plan(self, thing: dict, around: list[dict]):
        """A grip of `thing`, the fingers clear of `around`, or None (falsy): no way to it from here."""
        raise NotImplementedError

    def grip(self, target: dict, planned, found: list[dict]) -> dict:
        """Down to `target` by `planned`, shut on it, back up holding it: {"reading" (the found dict it
        was gripped by), "centre" (where it was), "line" (what put_back needs to put it back), "record"
        (the robot's own part of the pick's record), "held" (True / False where the robot can tell it
        holds something; left out, the look after the drop says)}; raises Refused where the arm will not."""
        raise NotImplementedError

    def order(self, kit: list[dict], at) -> list[dict]:
        """The kit things in reach in the order to try them: the nearest the robot (or `at`) first."""
        return sorted(kit, key=lambda o: float(np.linalg.norm(o["centre"][:2] - (at if at is not None else [0, 0]))))

    def approach(self, refused: list[dict], found: list[dict], kit: list[dict], at) -> tuple[tuple[float, float] | None, str]:
        """(the base's move (forward, left) m that brings a kit thing in reach -- or None -- and why)."""
        return None, ""

    def why_not(self, kit: list[dict], found: list[dict]) -> str:
        """Why none of `kit` is picked from here."""
        return f"none in reach: {describe(kit)}"

    def view(self):
        """The picture from where the arm is: (image BGR, K, camera base <- optical 4 x 4, anything
        things_in_view wants -- the M3's depth)."""
        raise NotImplementedError

    def things_in_view(self, image, K, camera, extra) -> list[dict]:
        """What stands on the floor in that picture (found dicts); [] when it cannot tell."""
        return []

    def go_box_look(self, j1: float, holding: bool) -> None:
        """The arm to where it looks for the box at `j1`."""
        raise NotImplementedError

    def to_box_view(self) -> None:
        """Holding something, the arm to where it saw the box (box_dir)."""
        raise NotImplementedError

    def way_into(self, here: dict, spot: np.ndarray) -> list[dict] | None:
        """The way from `here`, holding something, to over `spot` in the box, low enough to let go; None."""
        raise NotImplementedError

    def carry(self, way: list[dict]) -> None:
        """Along `way` (way_into's), holding something."""
        raise NotImplementedError

    def drop_on_line(self, extra: float) -> None:
        """Holding something, out over the line on the right, DROP_OUT + `extra` out."""
        raise NotImplementedError

    def let_go(self) -> None:
        raise NotImplementedError

    def put_back(self, back) -> None:
        """What is held put back down where it was picked (`back`: (the base's offset then, grip's "line"))."""
        raise NotImplementedError
