"""roboarm.metrics: where the time goes. Pure bookkeeping, no hardware."""

import json
import threading
import time

import pytest

from roboarm import metrics


@pytest.fixture
def reg():
    return metrics.Registry()


def test_a_span_is_counted_and_timed(reg):
    for _ in range(3):
        with reg.span("detect"):
            time.sleep(0.01)
    stage = reg.snapshot()["stages"]["detect"]
    assert stage["count"] == 3
    assert stage["total_s"] == pytest.approx(0.03, abs=0.02)
    assert stage["mean_ms"] >= 9


def test_a_run_adds_up_its_top_level_stages_only(reg):
    with reg.run("pickall") as run:
        with reg.span("scan"), reg.span("detect"):    # nested: detail, not added again
            time.sleep(0.02)
        with reg.span("pick"):
            time.sleep(0.01)
        time.sleep(0.01)                 # between stages
        run["status"] = "done"
    done = reg.snapshot()["runs"][0]
    assert done["name"] == "pickall" and done["status"] == "done"
    assert list(done["top"]) == ["scan", "pick"]
    assert "detect" in done["stages"] and "detect" not in done["top"]
    covered = sum(s["total_s"] for s in done["top"].values())
    assert covered + done["unaccounted_s"] == pytest.approx(done["total_s"], abs=0.002)
    assert done["unaccounted_s"] >= 0.009
    assert [e["depth"] for e in done["events"]] == [1, 0, 0]   # detect ends first


def test_only_the_runs_own_thread_is_in_it(reg):
    def elsewhere():
        with reg.span("live.detect"):
            pass
    with reg.run("sweep"):
        other = threading.Thread(target=elsewhere)
        other.start()
        other.join()
        with reg.span("scan"):
            pass
    done = reg.snapshot()["runs"][0]
    assert list(done["stages"]) == ["scan"]
    assert "live.detect" in reg.snapshot()["stages"], "still in the statistics"


def test_a_scope_prefixes_everything_inside_it(reg):
    with reg.scope("live"), reg.span("detect"), reg.span("detect.depth"):
        reg.record("vision.raised.server", 0.03)
    assert set(reg.snapshot()["stages"]) == {"live.detect", "live.detect.depth",
                                             "live.vision.raised.server"}


def test_a_recorded_duration_nests_inside_the_open_span(reg):
    with reg.run("pick"), reg.span("vision.raised"):
        reg.record("vision.raised.server", 0.025)
    done = reg.snapshot()["runs"][0]
    assert done["stages"]["vision.raised.server"]["total_s"] == pytest.approx(0.025)
    assert "vision.raised.server" not in done["top"]


def test_only_the_newest_run_keeps_its_timeline(reg):
    for name in ("a", "b"):
        with reg.run(name), reg.span("x"):
            pass
    runs = reg.snapshot()["runs"]
    assert [r["name"] for r in runs] == ["b", "a"]
    assert "events" in runs[0] and "events" not in runs[1]
    assert "events" not in reg.snapshot(events=False)["runs"][0]


def test_finished_runs_are_appended_to_the_log(reg, tmp_path):
    reg.log_path = tmp_path / "metrics.jsonl"
    for name in ("sweep", "pick"):
        with reg.run(name), reg.span("scan"):
            pass
    lines = [json.loads(line) for line in reg.log_path.read_text().splitlines()]
    assert [r["name"] for r in lines] == ["sweep", "pick"]


def test_reset_forgets_everything(reg):
    with reg.run("x"), reg.span("y"):
        pass
    reg.reset()
    assert reg.snapshot() == {"stages": {}, "runs": []}
