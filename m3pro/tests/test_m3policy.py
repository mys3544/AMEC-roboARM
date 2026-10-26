"""roboarm/m3policy.py with a pretend picker, camera and policy service: to the start, then a step a tick
held to max_dps / HZ and the limits, sent and not waited for, the service asked once a chunk; a STOP, a
stale picture or a refusal ends it, and however it ends the arm holds; dry sends nothing."""

import contextlib

import numpy as np
import pytest
from roboshared.task import Stopped

from roboarm import m3policy
from roboarm.m3arm import ArmError

CHUNK, N = 20, 10


class Picker:
    def __init__(self, stop_after=None, refuse_after=None, out=None):
        self.calls, self.lines, self.out = [], [], out
        self.pose = {1: 90, 2: 180, 3: 5, 4: 0, 5: 90, 6: 0}
        self.stop_after, self.refuse_after, self.t = stop_after, refuse_after, 0.0

    def call(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        if method == "step_to":
            if self.refuse_after is not None and len(self.steps()) > self.refuse_after:
                raise ArmError("refused: the arm would reach 0.5 cm above the floor")
            self.pose = dict(args[0])
        elif method == "move_to":
            self.pose = {**self.pose, **args[0]}
        elif method == "read":
            return dict(self.pose)
        return None

    def steps(self):
        return [args for method, args, _ in self.calls if method == "step_to"]

    def here(self):
        return dict(self.pose)

    def check(self):
        if self.stop_after is not None and len(self.steps()) >= self.stop_after:
            raise Stopped

    def step(self, _name):
        return contextlib.nullcontext()

    def say(self, text):
        self.lines.append(text)

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += s


@pytest.fixture
def service(monkeypatch):
    asked = []

    def ask(path, payload=None, timeout=10.0):
        asked.append((path, payload))
        if path == "/health":
            return {"model": {"checkpoint": "m3_act/100000", "n_action_steps": N, "chunk_size": CHUNK},
                    "precision": "fp16"}
        return {"chunk": [[150, 105, 10, 0, 90, 170]] * CHUNK}   # J1 far right, J6 shut past its limit

    monkeypatch.setattr(m3policy, "ask", ask)
    return asked


def frame(age=0.05):
    return lambda: (np.zeros((480, 640, 3), np.uint8), age)


def test_from_the_start_a_step_a_tick_held_to_its_speed_and_limits_then_the_arm_holds(service, tmp_path):
    picker = Picker(out=tmp_path / "pick")
    m3policy.run(picker, frame(), seconds=2.0, max_dps=100, clock=picker.clock, sleep=picker.sleep)
    assert picker.calls[0][0] == "read" and picker.calls[1][:2] == ("move_to", (m3policy.START,))
    steps = picker.steps()
    assert 20 <= len(steps) <= 21                                   # ten a second
    assert [s[0][1] for s in steps[:7]] == [100, 110, 120, 130, 140, 150, 150]   # 10 deg a tick at 100 deg/s
    assert max(s[0][6] for s in steps) == 150                       # never shut past 150
    assert all(s[1] == 0.1 for s in steps)
    assert all(kw == {"floor_clear": m3policy.FLOOR_CLEAR} for m, _, kw in picker.calls if m == "step_to")
    assert picker.calls[-1][0] == "hold"
    acts = [p for path, p in service if path == "/act"]
    assert len(acts) == len(steps) // N + (len(steps) % N > 0)      # once a chunk
    assert acts[0]["state"] == [90, 105, 10, 0, 90, 76] and acts[1]["state"][0] == 150
    assert len(next((tmp_path / "policy").glob("run_*.jsonl")).read_text().splitlines()) == len(steps)


def test_a_stop_ends_it_at_the_next_tick_and_the_arm_holds(service):
    picker = Picker(stop_after=3)
    with pytest.raises(Stopped):
        m3policy.run(picker, frame(), seconds=10, clock=picker.clock, sleep=picker.sleep)
    assert len(picker.steps()) == 3 and picker.calls[-1][0] == "hold"


def test_a_refusal_or_a_stale_picture_ends_it_and_the_arm_holds(service):
    picker = Picker(refuse_after=4)
    with pytest.raises(ArmError, match="floor"):
        m3policy.run(picker, frame(), seconds=10, clock=picker.clock, sleep=picker.sleep)
    assert picker.calls[-1][0] == "hold"
    picker = Picker()
    with pytest.raises(ArmError, match="fresh picture"):
        m3policy.run(picker, frame(age=1.0), seconds=10, clock=picker.clock, sleep=picker.sleep)
    assert picker.steps() == [] and picker.calls[-1][0] == "hold"


def test_dry_moves_nothing_and_its_speed_is_held_to_the_recordings(service):
    picker = Picker()
    m3policy.run(picker, frame(), seconds=1.0, max_dps=500, dry=True, clock=picker.clock, sleep=picker.sleep)
    assert picker.calls == []
    assert "at most 100 deg/s" in picker.lines[0] and "ticks in" in picker.lines[-1]
