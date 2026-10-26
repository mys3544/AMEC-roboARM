"""tools/m3_episodes.py's cutting of a recorded job, without LeRobot: each attempt from a few ticks
standing before its pregrasp to its arrival back at the look pose, matched to its pick record and drop;
one refused on the way left out; the ticks taken at an even rate; the action the next frame's state;
the task said."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import m3_episodes

STEPS = (["look.box"] * 2 + ["look.wide"] * 4
         + ["pregrasp"] * 2 + ["look.close", "descend", "descend", "close", "lift", "drop", "drop"] + ["look.after"] * 4
         + ["pregrasp"] * 2 + ["home"]                       # refused on the way: no look after, no record
         + ["pregrasp", "look.close", "descend", "close", "lift", "drop"] + ["look.after"] * 3 + ["home"])
LOOK = [90, 105, 10, 0, 90, 76]   # where the wide look and the look after stand


def job():
    """The arm standing at LOOK through the wide look and the look after (but its first tick, on the
    way back), moving everywhere else."""
    ticks = []
    for i, step in enumerate(STEPS):
        standing = step in ("look.box", "look.wide") or step == "look.after" and STEPS[i - 1] == "look.after"
        ticks.append({"i": i, "t": round(0.1 * i, 3), "pose": LOOK if standing else [i, 100, 10, 0, 90, 0], "step": step})
    summary = {"job": "clean_up", "started": "2026-10-23 15:55:03", "started_unix": 1000.0,
               "attempts": [{"kind": "block lying", "picked": True}, {"kind": "30 mm cube", "picked": False}],
               "log": [{"n": 1, "t": 1001.35, "text": "dropped into the box at x 0.138 y -0.197"},
                       {"n": 2, "t": 1002.45, "text": "dropped 261 mm out on the right"}]}
    return ticks, summary


def test_each_attempt_runs_from_just_before_it_leaves_the_look_pose_to_its_arrival_back_there():
    ticks, summary = job()
    found = m3_episodes.attempts(ticks, summary)
    # the first: 3 ticks standing before its pregrasp (6), to its look after's first tick at LOOK (16 of 15-18);
    # the second: none before its pregrasp (22) -- the arm was moving (the refused one's home) --, to 29 of 28-30
    assert [(a["first"], a["last"]) for a in found] == [(3, 16), (22, 29)]
    assert [a["record"]["kind"] for a in found] == ["block lying", "30 mm cube"]
    assert [a["drop"] for a in found] == ["box", "line"]
    assert m3_episodes.task_for(found[0]) == "pick up the block and put it in the box"
    assert m3_episodes.task_for(found[1]) == "pick up the 30 mm cube and put it on the line on the right"


def test_a_piece_shoved_out_of_the_way_is_not_counted_as_picked():
    """2026-10-23: a 30 mm cube shoved 4.3 cm out, past the next look's middle view, was "picked"."""
    ticks, summary = job()
    summary["attempts"][0] = {"kind": "block lying", "picked": True, "target": [0.20, 0.10, 0.015]}
    summary["attempts"][1] = {"kind": "30 mm cube", "picked": True, "target": [0.18, 0.17, 0.015]}
    summary["log"] += [
        {"n": 3, "t": 1000.2, "text": "looked round: block lying at x 0.200 y +0.100, turned 5 deg; "
                                      "30 mm cube at x 0.180 y +0.170, turned 9 deg; 30 mm cube at x 0.150 y +0.120, turned 0 deg"},
        {"n": 4, "t": 1001.5, "text": "wide look: 30 mm cube at x 0.180 y +0.170, turned 9 deg; 30 mm cube at x 0.150 y +0.120, turned 0 deg"},
        {"n": 5, "t": 1002.7, "text": "wide look: nothing"},
        {"n": 6, "t": 1003.0, "text": "looked round: 30 mm cube at x 0.150 y +0.120, turned 0 deg; 30 mm cube at x 0.221 y +0.188, turned 0 deg"}]
    found = m3_episodes.attempts(ticks, summary)
    assert found[0]["record"]["picked"] is True            # its neighbour stayed where it was
    assert found[1]["record"]["picked"] is False and found[1]["record"]["pushed_to"] == [0.221, 0.188]


def test_records_and_attempts_that_do_not_match_are_not_paired_by_guess():
    ticks, summary = job()
    summary["attempts"] = summary["attempts"][:1]
    with pytest.raises(ValueError, match="not matched"):
        m3_episodes.attempts(ticks, summary)


def test_the_ticks_are_taken_evenly_and_the_action_is_the_next_state():
    ticks = [{"t": t, "pose": [k, 0, 0, 0, 0, 0], "step": "descend"} for k, t in enumerate([0.0, 0.1, 0.32, 0.4, 0.5])]
    chosen = m3_episodes.resample(ticks, 0, 4, 10)
    assert chosen == [0, 1, 1, 2, 3, 4]   # 0.2 s has no tick: the nearest, 0.1, again
    out = m3_episodes.frames(ticks, chosen)
    assert [a[0] for _i, _s, a in out] == [1, 1, 2, 3, 4, 4]   # the last holds


def test_a_recording_from_before_started_unix_is_dated_by_the_robots_clock():
    # 15:55:03 in Shanghai is 07:55:03 UTC
    assert m3_episodes.started_unix({"started": "2026-10-23 15:55:03"}) == pytest.approx(1792742103.5)
