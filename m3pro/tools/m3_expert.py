"""The privileged expert for the simulator (NEXT.md, "This week"): the M3's own picker (roboarm.m3pick.Picker)
run on the laptop against a simulated table whose layout it is TOLD -- its looks given the pieces where they
are (no depth, no detector), its box looks the box (no marker) -- and its arm an M3Arm whose board is a
recorder on a clock of its own. A clean-up so comes out as the very commands the real board would get and
when (M3Arm's sends: pose, ms), the job's steps and its pick records, in milliseconds of the laptop's time;
m3_sim.py plays them in Isaac Sim, the robot card applied, and records the wrist camera ten times a second in
the real recorder's layout (roboarm.m3record) for tools/m3_episodes.py.

What it is told: each piece's true pose, and that a pick succeeded (the piece gone from the table, into the
box) -- the plan is open-loop over a whole clean-up; whether a pick DID succeed the simulator says, and an
attempt it did not is a miss in the records. Its looks see what the camera at that pose would: a piece whose
top face is whole in the picture (m3pick.in_picture) whole, one whose middle is in it cut, the rest not.

    python tools/m3_expert.py [--seed 1] [--pieces 4]     # one layout planned: the steps and the timings
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roboshared import box as boxes
from roboshared import task as shared_task

from roboarm import m3arm, m3kinematics, m3pick

K = np.array([[480.8, 0, 323.8], [0, 480.9, 240.0], [0, 0, 1]])   # the DCW2's colour camera, 640 x 480
KINDS = (("cube30", 0.6), ("cube40", 0.2), ("block", 0.2))         # what a layout is drawn from, how often
SIZES = {"cube30": (0.03, 0.03, 0.03), "cube40": (0.04, 0.04, 0.04), "block": (0.03, 0.06, 0.03)}   # w, l, h
APART_M = 0.06            # pieces at least this far apart (centre to centre)
REACH_M = (0.11, 0.25)    # from J1's axis: the band a layout's pieces lie in -- a few out of reach, as on the table
BOX_AT = (0.10, -0.215)   # the box's middle beside the robot, as the M3 found it on 2026-10-26 evening (J1 ~165),
                          # facing the arm; (0.212, -0.195) until 2026-10-27, the morning's place: its keep-clear
                          # emptied the right front (m3_sim_report.py placement: the sim leaned left as the real)
# A box of its own each episode (box_place, since 2026-10-27, the user's: "náhodnú polohu, ale dosiahnuteľnú"):
BOX_RING = (0.20, 0.27)                     # m from J1's axis to its middle
BOX_SIDES = ((0.0, 50.0), (130.0, 180.0))   # J1 to it: left of the robot or right of it, not before it
BOX_TURN = 15.0                             # deg it may stand turned off facing the arm
ROBOT_CLEAR = 0.02                          # m its walls keep off the chassis' outline (m3pick.ROBOT_X, ROBOT_Y)


def box_usable(box: dict) -> bool:
    """Whether a job can use the box there: its walls clear of the chassis, its middle in the picture from one
    of the picker's box looks (SIDE_LOOK at BOX_LOOKS, as SimPicker.see_box judges), and three spots or more in it
    the arm carries a 30 mm cube over (m3pick.carry_to, as Picker.way_into, from that look)."""
    axis = np.asarray(m3pick.Picker.AXIS)
    edge = boxes.outline_points(boxes.outline(box))
    (x0, x1), y = m3pick.ROBOT_X, m3pick.ROBOT_Y
    if np.any((edge[:, 0] > x0 - ROBOT_CLEAR) & (edge[:, 0] < x1 + ROBOT_CLEAR) & (np.abs(edge[:, 1]) < y + ROBOT_CLEAR)):
        return False
    look = None
    for j1 in m3pick.BOX_LOOKS:
        camera = m3kinematics.frames({**m3pick.SIDE_LOOK, 1: j1})["arm4"] @ m3kinematics.camera_optical()
        q = np.linalg.inv(camera) @ np.r_[box["centre"], boxes.BOX_INSIDE[2], 1.0]
        if q[2] > 0.05:
            u, v = (K @ (q[:3] / q[2]))[:2]
            if 40 <= u < 600 and 40 <= v < 440:
                look = j1
                break
    if look is None:
        return False
    here = {**m3pick.SIDE_LOOK, 1: look, 6: m3arm.gripper_angle(30)}
    high = m3pick.SETTINGS["carry_cm"][5] / 100 + 0.04   # Picker.carry_high, held_below unknown
    reached = 0
    for spot in boxes.box_spots(box):
        j1 = 90 - math.degrees(math.atan2(spot[1] - axis[1], spot[0] - axis[0]))
        if 0 <= j1 <= 180 and m3pick.carry_to(here, float(np.hypot(*(spot - axis))), j1,
                                               boxes.BOX_INSIDE[2] + boxes.BOX_DROP_OVER, high) is not None:
            reached += 1
            if reached >= 3:
                return True
    return False


def box_place(rng: np.random.Generator, tries: int = 300) -> dict:
    """A drop-off box where a job can use it (box_usable), its middle BOX_RING from J1's axis on a side of
    BOX_SIDES -- either, alike -- facing the arm within BOX_TURN."""
    axis = np.asarray(m3pick.Picker.AXIS)
    for _ in range(tries):
        lo, hi = BOX_SIDES[int(rng.integers(len(BOX_SIDES)))]
        yaw = math.radians(90 - rng.uniform(lo, hi))   # J1 is mirrored: bigger turns right
        box = facing_box(axis + rng.uniform(*BOX_RING) * np.array([math.cos(yaw), math.sin(yaw)]),
                         rng.uniform(-BOX_TURN, BOX_TURN))
        if box_usable(box):
            return box
    raise RuntimeError(f"no usable place for the box in {tries} tries")


class Clock:
    """The planner's time: the arm's moves and the picker's waits advance it, nothing else."""

    def __init__(self):
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += max(0.0, float(seconds))


class FakeTime:
    """The time module for the picker while it plans: its sleeps advance the clock."""

    def __init__(self, clock: Clock):
        self.clock = clock

    def sleep(self, seconds: float) -> None:
        self.clock.advance(seconds)

    def monotonic(self) -> float:
        return self.clock.t

    def __getattr__(self, name):
        return getattr(time, name)


def facing_box(centre, turn_off: float = 0.0) -> dict:
    """The box with its middle at `centre`, its inside facing J1's axis (turned `turn_off` deg off that)."""
    axis = np.asarray(m3pick.Picker.AXIS)
    to_arm = axis - np.asarray(centre, float)
    a = math.atan2(to_arm[1], to_arm[0]) + math.radians(turn_off)
    into = np.array([math.cos(a), math.sin(a)])
    along = np.array([-into[1], into[0]])
    return {"centre": np.asarray(centre, float), "along": along, "into": into,
            "turn": math.degrees(math.atan2(along[1], along[0])), "items": []}


def layout(rng: np.random.Generator, n: int, box: dict) -> list[dict]:
    """`n` pieces lying on the table uniformly over REACH_M before J1's axis (fewer when they do not fit),
    APART_M apart and clear of the box: {"kind", "size", "length", "height", "centre", "yaw" (the line the
    fingers close along), "period"}."""
    axis = np.asarray(m3pick.Picker.AXIS)
    kinds, shares = zip(*KINDS)
    pieces = []
    for _ in range(400):
        if len(pieces) == n:
            break
        r, bearing = rng.uniform(*REACH_M), rng.uniform(-80, 80)
        at = axis + r * np.array([math.cos(math.radians(bearing)), math.sin(math.radians(bearing))])
        if at[0] < m3pick.BUMPER_X + 0.03:   # not under the chassis' front: the arm swept one off at its start
            continue
        if any(np.hypot(*(at - p["centre"][:2])) < APART_M for p in pieces) \
                or boxes.inside(box, at, boxes.BOX_WALL + shared_task.BOX_NEAR + 0.03):
            continue
        kind = str(rng.choice(kinds, p=shares))
        w, length, h = SIZES[kind]
        pieces.append({"kind": "block" if kind == "block" else "cube", "name": kind, "size": w, "length": length,
                       "height": h, "centre": np.r_[at, h / 2], "yaw": float(rng.uniform(0, 180)),
                       "period": 180 if kind == "block" else 90})
    return pieces


class SimPicker(m3pick.Picker):
    """The picker, told the table: `pieces` (layout()'s) and `box` (facing_box()'s)."""

    def __init__(self, pieces: list[dict], box: dict | None, clock: Clock, rng=None):
        self.clock, self.table, self.true_box = clock, [dict(p) for p in pieces], box
        self.sends: list[tuple[float, dict, int]] = []   # (when, pose, ms): what the board was sent
        self.steps: list[tuple[float, str]] = []         # (when, step)
        self.records: list[dict] = []                    # the pick records, in order
        self.said: list[tuple[float, str]] = []          # (when, text): the job's narrated log on this clock
        self.held: dict | None = None
        arm = m3arm.M3Arm(self._send, lambda: 12.0, clock.now, clock.advance)
        super().__init__(lambda method, *a, **kw: getattr(arm, method)(*a, **kw), None, settle_s=None, rng=rng)

    def _send(self, pose: dict, ms: int) -> None:
        self.sends.append((round(self.clock.t, 4), {int(j): int(v) for j, v in pose.items()}, int(ms)))

    def say(self, text: str) -> None:
        self.said.append((round(self.clock.t, 4), text))
        super().say(text)

    @contextlib.contextmanager
    def step(self, step: str):
        self.steps.append((round(self.clock.t, 4), step))
        with super().step(step):
            yield

    def _remember(self, record: dict) -> None:
        super()._remember(record)
        self.records.append(record)

    # ------------------------------------------------------------------------ what it is told --
    def look(self, tag: str, target_near=None):
        """What the camera at the arm's pose sees of the table: whole, cut, or nothing (in_picture)."""
        time.sleep(self.settle_s if self.settle_s is not None else
                   self.cfg["settle_close_s" if tag == "close" else "settle_wide_s"])
        self.check()
        pose = self.here()
        self.K = K
        camera = m3kinematics.frames(pose)["arm4"] @ m3kinematics.camera_optical()
        inv = np.linalg.inv(camera)
        found = []
        for p in self.table:
            q = inv @ np.r_[p["centre"][:2], p["height"], 1.0]
            u, v = (K @ (q[:3] / q[2]))[:2] if q[2] > 0.05 else (-1, -1)
            if not (0 <= u < 640 and 0 <= v < 480):
                continue
            o = {"kind": p["kind"], "size": p["size"], "length": p["length"], "height": p["height"],
                 "top": p["height"], "footprint": (p["size"], p["length"]), "yaw": p["yaw"] % p["period"],
                 "period": p["period"], "centre": np.array(p["centre"], float), "seen": np.array(p["centre"], float),
                 "outline_px": [], "true": p}
            o["whole"] = m3pick.in_picture(pose, o, K)
            radius = float(np.hypot(*(o["centre"][:2] - np.asarray(self.AXIS))))
            o["reach"] = radius <= m3pick.furthest(o["height"]) + 0.005
            found.append(o)
        return camera, found

    def see_box(self, tag: str) -> dict | None:
        """The box when its middle is in the picture from where the arm is (its marker read whole)."""
        time.sleep(self.settle_s if self.settle_s is not None else self.cfg["settle_close_s"])
        if self.true_box is None:
            return None
        camera = m3kinematics.frames(self.here())["arm4"] @ m3kinematics.camera_optical()
        q = np.linalg.inv(camera) @ np.r_[self.true_box["centre"], boxes.BOX_INSIDE[2], 1.0]
        if q[2] <= 0.05:
            return None
        u, v = (K @ (q[:3] / q[2]))[:2]
        return {**self.true_box, "items": []} if 40 <= u < 600 and 40 <= v < 440 else None

    # -------------------------------------------------------------------- what it is told happened --
    def grip(self, target: dict, planned, found: list[dict]) -> dict:
        out = super().grip(target, planned, found)
        self.held = target.get("true") or min(self.table, key=lambda p: np.hypot(*(p["centre"][:2] - target["centre"][:2])))
        self.table = [p for p in self.table if p is not self.held]   # up with the fingers: off the table
        return out

    def let_go(self) -> None:
        super().let_go()
        self.held = None   # into the box (or onto the line): out of every look


def plan_episode(pieces: list[dict], box: dict | None, seed: int = 0) -> dict:
    """A clean-up of the table as the picker would do it: {"sends", "steps", "records", "log", "said" (when,
    text), "seconds", "error"} -- the board's commands from the arm's first (engage) on."""
    clock = Clock()
    picker = SimPicker(pieces, box, clock, rng=np.random.default_rng(seed))
    fake = FakeTime(clock)
    saved = m3pick.time, shared_task.time
    m3pick.time = shared_task.time = fake
    error = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            picker.state.update(busy=True, job="clean_up")
            try:
                picker.clean_up()
            except (shared_task.NothingToPick, shared_task.Stopped, m3arm.ArmError, shared_task.Refused) as exc:
                error = f"{type(exc).__name__}: {exc}"
    finally:
        m3pick.time, shared_task.time = saved
    return {"sends": picker.sends, "steps": picker.steps, "records": picker.records,
            "log": [text for _t, text in picker.said], "said": picker.said, "seconds": round(clock.t, 2),
            "error": error}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pieces", type=int, default=4)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    box = box_place(rng)
    pieces = layout(rng, a.pieces, box)
    began = time.perf_counter()
    plan = plan_episode(pieces, box, a.seed)
    print(f"planned in {time.perf_counter() - began:.2f} s: {plan['seconds']} s of the robot's time, "
          f"{len(plan['sends'])} commands, {len(plan['records'])} picks; error: {plan['error']}")
    for p in pieces:
        print(f"  {p['name']:7s} at x {p['centre'][0]:.3f} y {p['centre'][1]:+.3f}, yaw {p['yaw']:.0f}")
    for t, s in plan["steps"]:
        print(f"  {t:6.1f} s  {s}")
    for line in plan["log"]:
        print("   |", line[:150])
    return 0


if __name__ == "__main__":
    sys.exit(main())
