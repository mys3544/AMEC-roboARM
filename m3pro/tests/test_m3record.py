"""roboarm/m3record.py against a pretend picker: a job becomes one episode -- a picture and a line a
tick, the job's log and its pick records at the end -- nothing is kept while idle or before the arm's
first command, and the oldest episodes go when they pass the limit."""

import json

import numpy as np

from roboarm import m3record


class Picker:
    """A picker's status and log, changed by the test between ticks."""

    def __init__(self):
        self.state = {"job": None, "step": "idle", "busy": False, "error": None, "last": None}
        self.lines = [{"n": 1, "t": 0.0, "text": "before the job"}]
        self.pose = None
        self.t = 1792760000.0   # 2026-10-23, the afternoon: episodes are named by it

    def log(self, since):
        return [e for e in self.lines if e["n"] > since]

    def say(self, text):
        self.lines.append({"n": self.lines[-1]["n"] + 1, "t": self.t, "text": text})

    def read(self):
        if self.pose is None:
            raise RuntimeError("the arm's pose is unknown until the first command")
        return dict(self.pose)

    def recorder(self, out, **kw):
        picture = np.full((48, 64, 3), 128, np.uint8)
        return m3record.Recorder(lambda: dict(self.state), self.log, lambda: (picture, 0.05), self.read,
                                 lambda: (0.0, 0.0, 0.0), out, clock=lambda: self.t, **kw)


def test_a_job_is_one_episode_with_a_picture_and_a_line_a_tick(tmp_path):
    p = Picker()
    rec = p.recorder(tmp_path)
    rec.tick()   # idle: nothing
    assert not list(tmp_path.iterdir())

    p.state.update(job="clean_up", step="look.wide", busy=True)
    rec.tick()   # the arm not commanded yet: the episode opens, no tick
    p.pose = {1: 90, 2: 180, 3: 5, 4: 0, 5: 90, 6: 0}
    for k in range(3):
        p.t += 0.1
        p.pose[1] += 5
        rec.tick()
    p.say("picked the 30 mm cube")
    p.state["last"] = {"kind": "30 mm cube", "target": np.array([0.2, 0.0, 0.015]), "picked": np.bool_(True)}
    p.t += 0.1
    rec.tick()
    p.state.update(job=None, step="idle", busy=False)
    rec.tick()

    (episode,) = list(tmp_path.iterdir())
    assert episode.name.endswith("_clean_up")
    ticks = [json.loads(line) for line in (episode / "ticks.jsonl").read_text().splitlines()]
    assert [t["pose"][0] for t in ticks] == [95, 100, 105, 105]
    assert [t["i"] for t in ticks] == [0, 1, 2, 3] and ticks[0]["step"] == "look.wide"
    assert sorted(f.name for f in episode.glob("*.jpg")) == [f"{i:06d}.jpg" for i in range(4)]
    summary = json.loads((episode / "episode.json").read_text())
    assert summary["job"] == "clean_up" and summary["ticks"] == 4
    assert [e["text"] for e in summary["log"]] == ["picked the 30 mm cube"]   # not what came before the job
    assert summary["attempts"] == [{"kind": "30 mm cube", "target": [0.2, 0.0, 0.015], "picked": True}]

    rec.tick()   # idle again: still one episode
    assert len(list(tmp_path.iterdir())) == 1


def test_a_job_the_arm_never_moved_in_leaves_nothing(tmp_path):
    p = Picker()
    rec = p.recorder(tmp_path)
    p.state.update(job="look", busy=True)
    rec.tick()
    p.state.update(job=None, busy=False, error="no picture from the camera yet")
    rec.tick()
    assert not list(tmp_path.iterdir())


def test_the_oldest_episodes_go_past_the_limit(tmp_path):
    for name in ("20261023_090000_clean_up", "20261023_100000_clean_up"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "000000.jpg").write_bytes(b"x" * 600)
    p = Picker()
    rec = p.recorder(tmp_path, keep_gb=1e-6)   # 1000 bytes
    p.state.update(job="sort", busy=True)
    rec.tick()
    assert min(f.name for f in tmp_path.iterdir()) == "20261023_100000_clean_up"
    assert len(list(tmp_path.iterdir())) == 2   # the newer old one, and the one just opened
