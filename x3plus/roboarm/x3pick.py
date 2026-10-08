"""The X3 Plus's body for the shared jobs (roboshared.task.Task): the same tasks as the M3 -- pick
once, clean up, the drop-off box (marker 10), home -- with the X3's own eyes and hands (the user's,
2026-10-07: "X3 has to use different object detection techniques because of missing depth camera").

  * Its eyes, as the X3's own panel tuned them (2026-09-18..22, restored 2026-10-08 after a first
    run that merged all three looks and missed): the SEARCH walks the stations of the scan looks
    (cfg.SCAN_LOOKS: the primary alone with the C930e -- the outer and near ones are less exact,
    the outer reads ~28 mm far), nearest the arm first, the base still at each, following hints
    (a thing cut by the picture's edge says where to look), and STOPS at the first station that
    shows something it can pick; the SUREST of those is taken (2026-09-18: a shadow ranged as a
    cube, 4 mm nearer than the real one, was picked by "nearest"); after the drop it looks again
    from that station. The detector ladder on every frame (tags, colour, YOLOE masks, Depth
    Anything's raised regions); a thing found again from a standstill before the grip (refine).
  * Its hands: grasp.pick -- open first, hover, the reach passes down, shut, the gripper's own
    read-back saying whether it holds something (so a miss is known at once, "held"), lift.
  * The box: looked for with the angled look of the 2026-10-06/07 depth tests (BOX_LOOK), the
    camera's pose from the arm's fine read-back and the fitted lens (data/cameras/<cam>/
    lens_on_arm.json); dropped into from above (kin.solve over the spot).
  * No driving yet: the base's ROS 2 topics are the next step (x3plus/PLAN.md), so a thing out of
    reach is said, not driven to.

The arm is whatever `call(method, *args, **kwargs)` reaches -- the robot's roboarm.web.bridge.Bridge
(tools/x3bridge.py) -- the camera whatever `frames()` gives (a camera.Stream).
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from roboshared import kit, metrics
from roboshared.box import BOX_DROP_OVER, BOX_INSIDE
from roboshared.task import NothingToPick, Refused, Stopped, Task, describe

from roboarm import camera, detect, grasp, lens, sweep
from roboarm import config as cfg
from roboarm import kinematics as kin

TABLE_Z = -cfg.TABLE_BELOW_PLATE   # the table in the base frame (the base plate is 0)
BLOCK_RATIO = 1.5    # a top face this much longer than wide is a block (the kit's is 30 x 60)
REFIND_M = 0.030     # the standstill look's thing is the search's when within this
REFINE_IF_OFF_DEG = 3   # a look already this close to head-on needs no second one
DROP_ZONE_M = 0.080  # what lies this near the drop line was put there: not picked
SETTLE_S = {"sonix": 1.2, "c930e": 0.4}[cfg.WRIST_CAMERA]
SCAN_POLL_S = 0.02
YAW_SETTLE_S = 0.15   # a look that only turned the base keeps its exposure: this much still
VISION_CHECK_S = 30.0   # how long the vision service's answer to /health is believed
BOX_LOOK = {2: 100, 3: 20, 4: 0, 5: 90}   # the angled look of the depth tests (2026-10-06): the table from
                                          # 15 cm out, a standing marker face on; J1 turns it round
# J4 of the box look, a round of BOX_LOOKS each, in turn: 10 first -- the table 23-87 cm out, an open
# lid's marker standing behind the box whole in the picture (2026-10-08: from J4 0 its top was cut by
# the picture's edge, no box found anywhere, the cubes dropped on the line) -- then BOX_LOOK's own 0
# for a box nearer than that, its marker on a wall.
BOX_TILTS = (10, 0)
# The calibrated looks the survey rides, nearest-seeing first: "near" too (2026-10-08: a 40 mm cube
# 124 mm out was cut by the primary look's near edge in every picture, and left on the table).
SURVEY_LOOKS = ("near", "primary")
# The robot's own front, in the arm's frame: its wheels show in the looks that face down beside it
# (2026-10-08: the front-right one at (0, -105) mm read as raised, once as a lying block, and was
# reached for). Nothing there is picked, and nothing there is a hint.
BODY_FRONT_M = 0.080
BODY_HALF_WIDTH_M = 0.160
LIFT_OVER = 0.06     # m over the release height the hand comes in above a spot in the box

# What the console's arm card tunes (POST /pick/settings): key -> (what it is, unit, lowest, highest, step,
# default). The grasp's own speeds (grasp.py) stay as measured. The driving ones are for the base's ROS 2
# step (next): kept now so the card and the shared flow have them.
SETTINGS = {
    "speed": ("arm speed, the moves between looks", "%", 30, 100, 5, 100),
    "free_dps": ("to a look", "deg/s", 10, 150, 5, 90),
    "return_dps": ("back from the drop-off, empty-handed", "deg/s", 20, 120, 5, 90),
    "near_dps": ("over the thing, before the descent", "deg/s", 10, 120, 5, 60),
    "descend_dps": ("down to the grip (the reach passes)", "deg/s", 5, 60, 1, 25),
    "carry_dps": ("holding it: up, and to the drop-off", "deg/s", 20, 120, 5, 90),
    "scan_dps": ("the base turning while it surveys", "deg/s", 4, 30, 1, 20),
    "settle_s": ("still before a look", "s", 0.0, 2.0, 0.05, SETTLE_S),
    "settle_close_s": ("still before the box look", "s", 0.0, 2.0, 0.05, SETTLE_S),
    "reach_passes": ("corrections after the descent (0: it grips where it lands)", "", 0, 3, 1, 0),
    "grip_deg": ("the fingers close this far past where the piece first touches them (1.8 a mm)", "deg", 0, 12, 0.5, 6),
    "reach_offset_mm": ("aim this much further out than seen", "mm", -30, 40, 1, round(cfg.REACH_OFFSET_M * 1000)),
    "release_s": ("open at the drop-off before going back", "s", 0.0, 1.0, 0.05, 0.15),
    "drive_cm_s": ("driving speed", "cm/s", 1.0, 10.0, 0.5, 5.0),
    "driven_s": ("still after driving, before the look", "s", 0.0, 2.0, 0.05, 0.4),
    "nudge_cm": ("driving per move, at most", "cm", 1, 15, 1, 10),
    "right_cm": ("driving right, at most", "cm", 0, 15, 1, 10),
}


def kind_of(target: detect.Target) -> str | None:
    """"cube" or "block", the M3's kinds (and the console's buttons); None for what the gripper
    cannot take, or what is not the kit's size (the M3 matches the kit too: m3vision.kit_item) --
    2026-10-08 the black tape on the paper's corner, a raised 20 x 24 mm to Depth Anything, cost two
    tries."""
    if not target.graspable or not kit.footprint_fits(target.width_m, target.length_m):
        return None
    return "block" if target.length_m >= BLOCK_RATIO * target.width_m else "cube"


def on_body(target: detect.Target) -> bool:
    return target.x < BODY_FRONT_M and abs(target.y) < BODY_HALF_WIDTH_M


def kit_sized(target: detect.Target) -> detect.Target:
    """A reading that is a kit piece (roboshared.kit.nearest) with the piece's own size -- the user's
    30 and 40 mm cubes and 30 x 60 blocks, nothing else (2026-10-08): the camera reads them ~3 mm
    narrow. Anything else as it was read."""
    item = kit.nearest(target.width_m, target.length_m)
    if item is None:
        return target
    _kind, width, length, height = item
    return replace(target, width_m=width, length_m=length, height_m=height)


def found_of(target: detect.Target, reach: bool | None = None) -> dict:
    """A detect.Target as the shared flow's found dict (roboshared.task)."""
    kind = kind_of(target)
    height = target.height_m or sweep.OBJECT_HEIGHT_M
    period = 180 if kind == "block" else 90
    return {"centre": np.array([target.x, target.y, TABLE_Z + height / 2]), "kind": kind,
            "size": target.width_m, "length": target.length_m, "height": height, "top": height,
            "footprint": (target.width_m, target.length_m), "yaw": float(target.angle_deg) % period,
            "period": period, "whole": not target.clipped, "reach": reach, "target": target,
            "label": target.label, "confidence": round(float(target.confidence), 3)}


def merged(seen: list[tuple[sweep.Look, detect.Target]]) -> list[tuple[sweep.Look, detect.Target]]:
    """sweep.merge, each object with the station that saw it best."""
    station_of = {id(t): look for look, t in seen}
    return [(station_of[id(t)], t) for t in sweep.merge(seen)]


def at_drop_spot(target: detect.Target) -> bool:
    # Drops land anywhere along a line out from under cfg.DROP_POSE (cfg.DROP_SPREAD_M):
    # the zone is that line, widened.
    return grasp.from_drop_line(target.x, target.y) < DROP_ZONE_M


def plannable(target: detect.Target) -> bool:
    if at_drop_spot(target):
        return False
    try:
        grasp.plan(target)
    except grasp.GraspError:
        return False
    return True


class _Arm:
    """The arm as grasp.py and the rides use it: each method through the job's `call` (the
    bridge's lock, its whitelist, its stop)."""

    def __init__(self, call):
        self._call = call

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: self._call(name, *args, **kwargs)


class _Say(io.TextIOBase):
    """grasp.py narrates with print(): each line into the job's log -- whose own print() goes to
    `real`, the stdout this replaced."""

    def __init__(self, say, real):
        self._say, self._real, self._part, self._busy = say, real, "", False

    def write(self, text: str) -> int:
        if self._busy:
            return self._real.write(text)
        lines = (self._part + text).split("\n")
        self._part = lines.pop()
        self._busy = True
        try:
            for line in lines:
                if line.strip():
                    self._say(line.strip())
        finally:
            self._busy = False
        return len(text)

    def flush(self) -> None:
        self._real.flush()


class Picker(Task):
    """The X3's body for the shared jobs (the module's docstring)."""

    SETTINGS = SETTINGS
    AXIS = (0.0, 0.0)          # base x, y are measured from J1's axis
    FLOOR_Z = TABLE_Z
    FOOTPRINT = ((-0.20, 0.09), 0.13)   # m: the chassis round J1's axis, for when it drives
    DROP_OUT = float(math.hypot(*kin.forward({**cfg.DROP_POSE, cfg.GRIPPER_ID: cfg.GRIPPER_CLOSED})[:2]))
    DROP_SPREAD = cfg.DROP_SPREAD_M
    BOX_LOOKS = (180, 135, 90, 45, 10)   # J1 turns 10-180 (cfg.SAFE_LIMITS): the arm's front half circle
    BOX_RELOOK = False   # straight from the pick to the box as the job found it (the user, 2026-10-08)

    def __init__(self, call, frames, calibration, out: Path | None = None, settle_s: float | None = None,
                 rng=None, drive=None, battery=None, detector_url: str | None = cfg.DETECTOR_URL,
                 lens_fit: dict | None = None):
        super().__init__(call, snapshot=None, out=out, settle_s=settle_s, rng=rng, drive=drive, battery=battery)
        self.arm = _Arm(call)
        self.frames = frames             # () -> camera.Stream
        self.calibration = calibration   # () -> [(matrix, survey pose, name)], the calibrated looks
        self.detector_url = detector_url
        self._vision = (0.0, False)
        self.station_seen: sweep.Look | None = None   # the station that saw what is being picked
        self.map: list[tuple[sweep.Look, detect.Target]] | None = None   # this job's survey, kept up to date
        self.box_tilt = BOX_TILTS[0]
        fit = lens_fit or {}
        self.lens = (fit.get("mount_m", lens.NOMINAL_MOUNT), fit.get("turn_rad", (0.0, 0.0, 0.0)),
                     fit.get("sag_deg_per_m", (0.0, 0.0, 0.0)))
        got = camera.intrinsics()   # the pictures' K and what distortion they keep; None: no lens known
        self.K = None if got is None else np.asarray(got[0], float)
        self.dist = np.zeros(5) if got is None else np.asarray(got[1], float)

    # -------------------------------------------------------------------------- helpers --
    def settle(self, key: str = "settle_s") -> float:
        return self.settle_s if self.settle_s is not None else self.cfg[key]

    def calibrated(self) -> list:
        looks = self.calibration()
        if not looks:
            raise NothingToPick("no table calibration: run tools/calibrate_table.py")
        return looks

    def vision_url(self) -> str | None:
        """The vision service's address while it answers (asked again after VISION_CHECK_S); None
        skips the neural rungs rather than wait on a dead socket at every station."""
        if self.detector_url is None:
            return None
        when, ok = self._vision
        if time.monotonic() - when > VISION_CHECK_S:
            ok = detect.vision_available(self.detector_url, timeout=1.5)
            self._vision = (time.monotonic(), ok)
        return self.detector_url if ok else None

    @contextlib.contextmanager
    def narrated(self):
        with contextlib.redirect_stdout(_Say(self.say, sys.stdout)):
            yield

    def as_found(self, target: detect.Target) -> dict:
        return found_of(target, reach=kind_of(target) is not None and plannable(target))

    # --------------------------------------------------------------------------- seeing --
    def detect_at(self, look: sweep.Look, frame, hints: bool = False) -> tuple[list[detect.Target], list[detect.Target]]:
        """What `frame`, taken from `look`, shows: (wholly, cut by the picture's edge -- what the
        hints read); the marked picture for the watchers, the raw one kept (out/looks)."""
        with metrics.span("detect"):
            found = detect.ladder(frame, "auto", look.matrix, nadir=look.nadir, url=self.vision_url(),
                                  lens_m=kin.camera_height(look.pose), note=lambda text: print(text, flush=True),
                                  hints=hints)
        found = [t for t in found if not on_body(t)]
        kept = [t for t in (kit_sized(t) for t in found if not t.clipped)
                if sweep.sees(look, t.x, t.y, sweep.TAG_SPAN_M if t.label.startswith("tag") else t.width_m,
                              t.height_m or sweep.OBJECT_HEIGHT_M)]
        marked = detect.annotate(frame, look.matrix, found)
        cv2.putText(marked, f"{look.name} J1 {look.pose[1]}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        self.show(marked)
        self.save_look(look, frame, found)
        return kept, [t for t in found if t.clipped]

    def save_look(self, look: sweep.Look, frame, found: list[detect.Target]) -> None:
        """Every look kept as the M3 keeps its own: the picture as the camera gave it (undistorted,
        unmarked), the pose it was taken from and what was found in it -- the raw material for depth
        from one camera (x3plus/PLAN.md X2). out/looks/, never overwritten."""
        if self.out is None:
            return
        folder = self.out / "looks"
        now = time.time()
        stem = time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + f"_{int(now * 1000) % 1000:03d}_{look.name}_{look.dyaw:+05.1f}"
        record = {"stamp": round(now, 3), "job": self.state["job"], "look": look.name, "dyaw": float(look.dyaw),
                  "pose": {str(j): float(a) for j, a in look.pose.items()}, "camera": cfg.WRIST_CAMERA,
                  "size": list(frame.shape[1::-1]), "K": None if self.K is None else self.K.tolist(),
                  "dist": self.dist.tolist(),
                  "found": [{"label": t.label, "x": round(float(t.x), 4), "y": round(float(t.y), 4),
                             "width": round(float(t.width_m), 4), "length": round(float(t.length_m), 4),
                             "angle": round(float(t.angle_deg), 1),
                             "height": None if t.height_m is None else round(float(t.height_m), 4),
                             "confidence": round(float(t.confidence), 3), "clipped": bool(t.clipped)} for t in found]}
        with contextlib.suppress(OSError, camera.CameraError):
            folder.mkdir(parents=True, exist_ok=True)
            camera.write_image(str(folder / f"{stem}.jpg"), frame)
            (folder / f"{stem}.json").write_text(json.dumps(record))

    def look_from(self, look: sweep.Look, hints: bool = False, speed: str = "free_dps"):
        """To `look` (repeatably: the homography holds only there), still, and what it shows:
        (wholly, cut by the edge)."""
        self.check()
        here = self.here()
        yaw_only = all(abs(here[j] - look.pose[j]) <= 1 for j in (2, 3, 4, 5))
        with metrics.span("look.move"):
            self.arm.move_to(look.pose, speed_dps=self.dps(speed), repeatable=True)
        with metrics.span("camera.settle"):   # a yaw keeps the same view: the exposure needs no new settle
            frame = self.frames().settled(min(self.settle(), YAW_SETTLE_S) if yaw_only else self.settle())
        self.check()
        return self.detect_at(look, frame, hints)

    def scan_looks(self, calibrated: list) -> list:
        """The calibrated looks the search walks: cfg.SCAN_LOOKS of them (all when None, or when none
        of the named ones is calibrated)."""
        if cfg.SCAN_LOOKS is None:
            return calibrated
        return [e for e in calibrated if e[2] in cfg.SCAN_LOOKS] or calibrated

    @staticmethod
    def station(entry: tuple, j1: float) -> sweep.Look:
        """The calibrated look `entry` (matrix, survey pose, name) with the base at `j1`."""
        matrix, survey, name = entry
        dyaw = float(survey[1] - j1)
        pose = dict(survey)
        pose[1] = round(j1)
        return sweep.Look(dyaw, pose, sweep.rotate(matrix, dyaw), name)

    def passes(self, calibrated: list, looks: list[sweep.Look]) -> list[tuple[tuple, list[float]]]:
        """The stations as one continuous pass per calibrated look, the nearest-seeing look first,
        every other pass backwards (the next starts where this one ended)."""
        out = []
        for k, entry in enumerate(sorted(calibrated, key=lambda e: sweep.nearest_of(e[0]))):
            dyaws = [look.dyaw for look in looks if look.name == entry[2]]
            if k % 2:
                dyaws.reverse()
            if dyaws:
                out.append((entry, dyaws))
        return out

    def ride(self, entry: tuple, dyaws: list[float], on_frame) -> None:
        """One look's pass over `dyaws` in ONE base motion: to the first station, then turning
        steadily to the last, a frame handed to `on_frame(look, frame)` as each station's yaw goes by
        (the look carrying the yaw read round the frame). 2026-09-22, the user's "stable movement"."""
        _matrix, survey, _name = entry
        first = sweep.pose_at(survey, dyaws[0])
        with metrics.span("scan.move"):
            self.arm.move_to(first, speed_dps=self.dps("free_dps"), repeatable=True)
        with metrics.span("camera.settle"):
            frame = self.frames().settled(self.settle())
        on_frame(self.station(entry, first[1]), frame)
        if len(dyaws) == 1:
            return
        with metrics.span("scan.glide"):
            goal = self.arm.glide(sweep.pose_at(survey, dyaws[-1]), speed_dps=self.dps("scan_dps"))
            ahead = 1 if goal[1] > first[1] else -1
            pending = [sweep.pose_at(survey, d)[1] for d in dyaws[1:]]
            last_j1, still = None, 0
            while pending:
                if self._stop.is_set():
                    self.arm.hold()
                    raise Stopped
                before = self.arm.read()
                reached = [s for s in pending if ahead * (before[1] - s) >= 0]
                ended = False
                if not reached:
                    still = still + 1 if before[1] == last_j1 else 0
                    last_j1 = before[1]
                    ended = still >= 3 and not self.arm.moving()   # the glide ended short of the next station
                    if not ended:
                        time.sleep(SCAN_POLL_S)
                        continue
                    reached = [pending[0]]
                frame = self.frames().settled(0.0)
                after = before if ended else self.arm.read()
                pending = [s for s in pending if s not in reached]
                if ended:
                    pending = []
                on_frame(self.station(entry, (before[1] + after[1]) / 2), frame)

    def wide_look(self, sides: bool = False) -> list[dict]:
        """`sides`: a SURVEY (survey()) -- this job's map of the table; else the look again from the
        station that saw what was just picked, and the map brought up to date there: what that look
        sees is as it says now, the rest as the survey left it (the user, 2026-10-08: the whole area
        once, then a final round -- the shared flow asks for a survey again only when the map holds
        nothing left to pick). Before any survey, the primary look straight ahead. What lies where
        this job lets go left out."""
        calibrated = self.calibrated()
        if sides:
            self.map = self.survey(calibrated)
        else:
            look = self.station_seen or self.station(self.scan_looks(calibrated)[0], self.scan_looks(calibrated)[0][1][1])
            fresh = [(look, t) for t in self.look_from(look, speed="return_dps")[0]]
            if self.map is None:
                self.map = merged(fresh)
            else:
                old = [(seen, t) for seen, t in self.map
                       if not sweep.sees(look, t.x, t.y, t.width_m, t.height_m or sweep.OBJECT_HEIGHT_M)]
                self.map = merged(old + fresh)
        found = []
        for station, t in self.map:
            o = self.as_found(t)
            o["station"] = station
            if not self.dropped_here(o):
                found.append(o)
        with self._lock:
            self.found = found
        self.say(("surveyed: " if sides else "looked again: ") + describe(found))
        return found

    def survey(self, calibrated: list) -> list[tuple[sweep.Look, detect.Target]]:
        """Each of SURVEY_LOOKS' rings of stations in ONE smooth turn of the base (ride: a picture as
        each station's yaw goes by), the nearest-seeing look first, from the end nearest the arm,
        the next look back the other way; one object seen from two stations is the one that saw it
        nearer the middle of its picture (sweep.merge). No hints, no stopping at the first find
        (the user, 2026-10-08: the whole area once, smooth -- standstill stations went go-and-stop,
        nearest-first went left, right, left, and searches that stopped at a find covered the table
        in parts, again after every pick)."""
        entries = sorted([e for e in calibrated if e[2] in SURVEY_LOOKS] or self.scan_looks(calibrated),
                         key=lambda e: sweep.nearest_of(e[0]))
        here = self.here()
        seen: list[tuple[sweep.Look, detect.Target]] = []

        def on_frame(look: sweep.Look, frame) -> None:
            seen.extend((look, t) for t in self.detect_at(look, frame)[0])

        forward = None
        for entry in entries:
            dyaws = sorted({look.dyaw for look in sweep.stations([entry], sweep.SURVEY_STEP_DEG)})
            j1s = [sweep.pose_at(entry[1], d)[1] for d in dyaws]
            if forward is None:
                forward = abs(j1s[0] - here[1]) <= abs(j1s[-1] - here[1])
            if dyaws:
                with metrics.span("scan"):
                    self.ride(entry, dyaws if forward else dyaws[::-1], on_frame)
            forward = not forward   # the next look starts where this one ended
        return merged(seen)

    def order(self, kit: list[dict], at) -> list[dict]:
        """The one nearest `at` when asked for; else the SUREST first (the detector's confidence: the
        one number that told a shadow from a cube, 2026-09-18)."""
        if at is not None:
            return super().order(kit, at)
        return sorted(kit, key=lambda o: -o["confidence"])

    def refine(self, target: detect.Target) -> detect.Target:
        """The thing seen again from the calibrated look that faces it, the base still -- the search
        saw it from a turning camera; the search's own reading where that look finds nothing near."""
        span = sweep.TAG_SPAN_M if target.label.startswith("tag") else target.width_m
        try:
            look = sweep.best_refine(self.calibrated(), target.x, target.y, span, target.height_m or sweep.OBJECT_HEIGHT_M)
        except sweep.NoLook as exc:
            self.say(f"cannot centre it ({exc}); the search's reading kept")
            return target
        here = self.here()
        if abs(look.pose[1] - here[1]) <= REFINE_IF_OFF_DEG and all(abs(look.pose[j] - here[j]) <= 3 for j in (2, 3, 4, 5)):
            return target
        near = [t for t in self.look_from(look)[0] if math.hypot(t.x - target.x, t.y - target.y) <= REFIND_M]
        if not near:
            self.say(f"looked again from J1 {look.pose[1]}: nothing within {REFIND_M * 1000:.0f} mm, the search's reading kept")
            return target
        best = min(near, key=lambda t: math.hypot(t.x - target.x, t.y - target.y))
        self.say(f"looked again from J1 {look.pose[1]}: {1000 * math.hypot(best.x - target.x, best.y - target.y):.1f} mm off the search's")
        return best

    def view(self):
        """The picture from where the arm is (the camera's residual distortion taken out), K, and the
        camera's pose from the arm's fine read-back and the fitted lens."""
        frame = self.frames().settled(0.0)
        try:
            pose = self.call("read_fine")
        except Exception:  # noqa: BLE001 -- a bridge without it: the plain read-back
            pose = self.here()
        image = cv2.undistort(frame, self.K, self.dist) if np.any(self.dist) else frame
        R, c = lens.camera_pose({int(j): float(a) for j, a in pose.items()}, *self.lens)
        camera_m = np.eye(4)
        camera_m[:3, :3], camera_m[:3, 3] = R, c
        return image, self.K, camera_m, None

    # --------------------------------------------------------------------------- doing --
    def engaged(self) -> None:
        self.call("read")   # the bridge connects the arm on its first call; its torque is always on
        self.map, self.station_seen = None, None   # a job surveys afresh

    def home(self) -> None:
        with self.step("home"):
            self.arm.home(speed_dps=self.dps("free_dps"))

    def plan(self, thing: dict, around: list[dict]):
        target = thing["target"]
        if at_drop_spot(target):
            return None
        try:
            return grasp.plan(target)
        except grasp.GraspError:
            return None

    def grip(self, target: dict, planned, found: list[dict]) -> dict:
        """Seen again from a standstill, then grasp.pick: whether it holds something, the gripper says."""
        t = target["target"]
        self.station_seen = target.get("station") or self.station_seen   # where to look again after
        with self.step("look.close"):
            t = self.refine(t)
        offset = self.cfg["reach_offset_mm"] / 1000
        with self.step("grip"), self.narrated():
            try:
                held = grasp.pick(self.arm, t, reach_offset_m=offset, speeds=self.grip_speeds(),
                                  passes=int(self.cfg["reach_passes"]), grip_deg=self.cfg["grip_deg"])
            except grasp.GraspError as exc:
                raise Refused(np.array([t.x, t.y]), f"refused: {exc}") from exc
        try:
            step = grasp.plan(grasp.aim_for(t, offset), reach_offset_m=0.0)
        except grasp.GraspError:
            step = None
        reading = found_of(t, reach=True)
        return {"reading": reading, "centre": reading["centre"], "line": (t.x, t.y),
                "held": None if held is None else bool(held),   # None: the look after says, as the M3's
                "record": {"yaw_from": "outline", "pitch": None if step is None else round(float(step.pitch)),
                           "j5": None if step is None else int(step.roll), "label": t.label}}

    def grip_speeds(self) -> dict:
        return {"approach": self.dps("near_dps"), "descend": self.dps("descend_dps"), "carry": self.dps("carry_dps")}

    def why_not(self, kit: list[dict], found: list[dict]) -> str:
        parts = []
        for o in kit:
            t = o["target"]
            if at_drop_spot(t):
                why = "at the drop-off spot -- put there"
            else:
                try:
                    grasp.plan(t)
                    why = "refused on the way"
                except grasp.GraspError as exc:
                    why = str(exc)
            parts.append(f"{describe([o])}: {why}")
        return "; ".join(parts) or "nothing in reach"

    # ------------------------------------------------------------------- the drop-off --
    def go_box_look(self, j1: float, holding: bool) -> None:
        # no settle here: see_box waits settle_close_s before its picture
        self.arm.move_to({1: round(j1), **BOX_LOOK, 4: self.box_tilt}, speed_dps=self.dps("free_dps"), settle=False)

    def box_search(self, holding: bool = False, tried: float | None = None) -> dict | None:
        """Task.box_search at each of BOX_TILTS in turn: where it is found, there it is looked at again."""
        for tilt in BOX_TILTS:
            self.box_tilt = tilt
            box = super().box_search(holding, tried)
            if box is not None:
                return box
        self.box_tilt = BOX_TILTS[0]
        return None

    def to_box_view(self) -> None:
        self.go_box_look(self.box_dir, holding=True)

    def way_into(self, here: dict, spot: np.ndarray) -> list[dict] | None:
        """Over `spot` LIFT_OVER higher, then down to BOX_DROP_OVER over the walls -- the tool as
        upright as it reaches (grasp.DROP_PITCHES), the fingers as they hold it; None out of reach."""
        held = here.get(cfg.GRIPPER_ID, cfg.GRIPPER_CLOSED)
        release = TABLE_Z + BOX_INSIDE[2] + BOX_DROP_OVER
        way = []
        for z in (release + LIFT_OVER, release):
            try:
                pose, _pitch = kin.solve(float(spot[0]), float(spot[1]), z, pitches=grasp.DROP_PITCHES, gripper=held)
            except kin.Unreachable:
                return None
            if not cfg.SAFE_LIMITS[1][0] <= pose[1] <= cfg.SAFE_LIMITS[1][1]:
                return None
            way.append({j: a for j, a in pose.items() if j != cfg.GRIPPER_ID})
        return way

    def carry(self, way: list[dict]) -> None:
        self.check()
        if way:   # one motion through the way, as the M3 carries (Arm.follow)
            self.arm.follow(way, speed_dps=self.dps("carry_dps"))

    def drop_on_line(self, extra: float) -> None:
        try:
            pose = grasp.drop_pose(extra)
        except kin.Unreachable:
            pose = dict(cfg.DROP_POSE)
        self.arm.move_to(pose, speed_dps=self.dps("carry_dps"), settle=False)

    def let_go(self) -> None:
        time.sleep(0.1)
        self.arm.set_gripper(cfg.GRIPPER_OPEN, speed_dps=grasp.DROP_OPEN_DPS, settle=False)
        time.sleep(self.cfg["release_s"])

    def put_back(self, back) -> None:
        if back is None or back[1] is None:
            raise Refused(np.zeros(2), "refused: holding something with nowhere to put it -- take it off by hand")
        x, y = back[1]
        with self.step("put back"), self.narrated():
            grasp.place(self.arm, float(x), float(y), reach_offset_m=self.cfg["reach_offset_mm"] / 1000,
                        speeds=self.grip_speeds())
        self.say("put back where it was picked")
