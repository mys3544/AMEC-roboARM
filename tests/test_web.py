"""The web control panel, end to end against the simulator. No hardware, no browser.

The server is started on a free port and driven with urllib, the way the page
drives it. The simulator's arm moves instantly (time_scale=0) so a whole
sweep-pick-place runs in well under a second.
"""

import json
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from roboarm import camera, detect
from roboarm import config as cfg
from roboarm import kinematics as kin
from roboarm.web import server, session, sim


# ------------------------------------------------------------ fixtures ---
@pytest.fixture
def world():
    return sim.World([
        sim.Block(0.160, 0.020, 0.040, (0, 120, 255)),    # graspable, straight ahead
        # Too SMALL for the gripper to verify a hold on. (A 60 mm cube, the old
        # fixture, is never wholly in shot from any station -- its outline, base
        # to magnified top, spans the picture -- a true fact about this camera.)
        # Half in the survey station's view, wholly in the -25 degree one's, and
        # clear of the other block: two outlines that touch merge into one blob.
        sim.Block(0.145, -0.020, 0.016, (0, 0, 220)),
    ])


@pytest.fixture
def sess(world, monkeypatch):
    monkeypatch.setattr(detect, "BACKGROUND_PATH",
                        Path(tempfile.mkdtemp()) / "table_background.png")
    arm = sim.SimArm(world, time_scale=0)
    s = session.Session(
        arm_factory=lambda: arm,
        stream=lambda: sim.SimStream(arm, fps=40),
        calibration=sim.calibration,
    )
    s.settle_s = 0.05  # the simulated camera has no exposure to settle
    s.start()
    _wait(lambda: s.stream.latest()[0] is not None)
    yield s
    s.close()


@pytest.fixture
def url(sess):
    srv = server.make_server(sess, "127.0.0.1", 0)
    srv.max_stream_frames = 3
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _wait(condition, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    raise AssertionError("timed out waiting")


def get(url: str, path: str) -> dict:
    with urllib.request.urlopen(url + path, timeout=5) as reply:
        return json.loads(reply.read())


def post(url: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(
        url + path, data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def finished(url: str, timeout: float = 20.0) -> dict:
    """Wait for the running job to end and return its record."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = get(url, "/api/state")["job"]
        if job and job["status"] != "running":
            return job
        time.sleep(0.03)
    raise AssertionError("job did not finish")


# ------------------------------------------------------- camera.Stream ---
def test_stream_hands_out_only_newer_frames():
    stream = camera.Stream(device=-1)
    frame = np.zeros((4, 4, 3), np.uint8)
    assert stream.latest() == (None, 0)
    stream.publish(frame)
    got, seq = stream.wait(0, timeout=0.1)
    assert got is frame and seq == 1
    got, seq = stream.wait(1, timeout=0.05)
    assert got is None and seq == 1, "nothing newer, so nothing handed out"


def test_settled_returns_frames_taken_after_the_settle_time():
    stream = camera.Stream(device=-1)
    stamps = []

    def feed():
        for i in range(12):
            frame = np.full((2, 2, 3), i, np.uint8)
            stream.publish(frame)
            stamps.append(time.monotonic())
            time.sleep(0.03)
    threading.Thread(target=feed, daemon=True).start()
    started = time.monotonic()
    frames = stream.settled(settle_s=0.15, count=2)
    assert len(frames) == 2
    # Both frames were published after the settle window, not before.
    assert all(int(f[0, 0, 0]) >= 4 for f in frames)
    assert time.monotonic() - started >= 0.15


def test_settled_gives_up_when_the_camera_is_dead():
    stream = camera.Stream(device=-1)
    stream.error = "camera 0 is not delivering frames"
    with pytest.raises(camera.CameraError):
        stream.settled(settle_s=0.05)


# ---------------------------------------------------------- state & mode ---
def test_state_describes_the_connected_arm_and_camera(url):
    s = get(url, "/api/state")
    assert s["arm"]["connected"] and s["arm"]["pose"] == {str(j): a for j, a in cfg.HOME_POSE.items()}
    assert s["mode"] == "manual"
    assert s["calibration"]["ok"] and s["calibration"]["looks"] == ["primary"]
    assert s["camera_fps"] >= 0 and s["camera_error"] is None
    assert s["arm"]["tip_mm"]["z"] > 0


def test_manual_moves_are_refused_in_auto_mode_and_vice_versa(url):
    assert post(url, "/api/mode", {"mode": "auto"})[0] == 200
    code, body = post(url, "/api/arm/jog", {"joint": 1, "delta": 5})
    assert code == 409 and "manual" in body["error"]
    code, body = post(url, "/api/auto/start", {"job": "survey"})
    assert code == 200
    finished(url)
    assert post(url, "/api/mode", {"mode": "manual"})[0] == 200
    code, body = post(url, "/api/auto/start", {"job": "survey"})
    assert code == 409 and "automatic" in body["error"]


def test_bad_requests_are_400_not_500(url):
    code, body = post(url, "/api/arm/move", {"joints": {"2": 170}})
    assert code == 400 and "safe range" in body["error"]
    code, body = post(url, "/api/arm/jog", {})
    assert code == 400 and "joint" in body["error"]
    code, body = post(url, "/api/mode", {"mode": "sideways"})
    assert code == 400
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(url, "/api/nothing")
    assert exc.value.code == 404


# ------------------------------------------------------------- manual ----
def test_joint_move_jog_and_gripper(url):
    code, body = post(url, "/api/arm/move", {"joints": {"1": 100, "2": 80}})
    assert code == 200 and body["pose"]["1"] == 100 and body["pose"]["2"] == 80
    code, body = post(url, "/api/arm/jog", {"joint": 1, "delta": -5})
    assert body["pose"]["1"] == 95
    code, body = post(url, "/api/arm/gripper", {"action": "close"})
    assert body["pose"]["6"] == cfg.GRIPPER_CLOSED
    code, body = post(url, "/api/arm/gripper", {"angle": 90})
    assert body["pose"]["6"] == 90


def test_jog_is_clamped_to_the_safe_range(url):
    post(url, "/api/arm/move", {"joints": {"1": cfg.SAFE_LIMITS[1][1] - 2}})
    code, body = post(url, "/api/arm/jog", {"joint": 1, "delta": 30})
    assert code == 200 and body["pose"]["1"] == cfg.SAFE_LIMITS[1][1]


def test_cartesian_jog_moves_the_fingertip_and_keeps_the_wrist_roll(url):
    post(url, "/api/arm/preset", {"name": "survey"})
    post(url, "/api/arm/move", {"joints": {"5": 120}})
    before = get(url, "/api/state")["arm"]["tip_mm"]
    code, body = post(url, "/api/arm/cartesian", {"dx": 10})
    assert code == 200
    after = get(url, "/api/state")["arm"]["tip_mm"]
    assert after["x"] - before["x"] == pytest.approx(10, abs=3)
    assert abs(after["y"] - before["y"]) < 3 and abs(after["z"] - before["z"]) < 4
    assert body["pose"]["5"] == 120, "a translation must not un-roll the wrist"


def test_cartesian_absolute_and_unreachable(url):
    code, body = post(url, "/api/arm/cartesian", {"x": 160, "y": 0, "z": 40})
    assert code == 200
    tip = get(url, "/api/state")["arm"]["tip_mm"]
    assert tip["x"] == pytest.approx(160, abs=4) and tip["z"] == pytest.approx(40, abs=4)
    code, body = post(url, "/api/arm/cartesian", {"x": 600, "y": 0, "z": 40})
    assert code == 400 and "reach" in body["error"]


def test_hold_interrupts_a_move_in_progress(sess, url):
    sess.arm.time_scale = 0.02  # slow enough to be caught mid-glide
    result = {}

    def long_move():
        result["code"], result["body"] = post(url, "/api/arm/move", {"joints": {"1": 20}})
    thread = threading.Thread(target=long_move)
    thread.start()
    _wait(lambda: sess.arm.pose[1] < 90)
    code, body = post(url, "/api/arm/hold")
    assert code == 200
    thread.join(timeout=5)
    assert result["code"] == 400 and "interrupted" in result["body"]["error"]
    assert 20 < body["pose"]["1"] < 90, "stopped somewhere along the way, not at the goal"
    assert sess.arm.torque


def test_release_and_engage_track_torque(url):
    assert post(url, "/api/arm/release")[0] == 200
    assert get(url, "/api/state")["arm"]["torque"] is False
    assert post(url, "/api/arm/engage")[0] == 200
    assert get(url, "/api/state")["arm"]["torque"] is True


# ------------------------------------------------------------- frames ----
def test_a_second_pane_can_ask_for_its_own_view(url):
    """?view=mask renders the detector's picture for that client only; the
    page's own view stays what it was."""
    before = get(url, "/api/state")["view"]
    for view in ("mask", "raw", "detect", "nonsense"):
        with urllib.request.urlopen(url + f"/snapshot.jpg?view={view}", timeout=5) as reply:
            assert reply.headers["Content-Type"] == "image/jpeg"
            assert reply.read()[:2] == b"\xff\xd8"
    assert get(url, "/api/state")["view"] == before


def test_snapshot_and_stream_are_jpeg(url):
    with urllib.request.urlopen(url + "/snapshot.jpg", timeout=5) as reply:
        assert reply.headers["Content-Type"] == "image/jpeg"
        assert reply.read()[:2] == b"\xff\xd8"
    with urllib.request.urlopen(url + "/stream.mjpg", timeout=5) as reply:
        assert reply.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        body = reply.read()
    assert body.count(b"--" + server.BOUNDARY.encode()) == 3
    assert body.count(b"Content-Type: image/jpeg") == 3


def test_live_detection_finds_the_block_at_the_survey_station(sess, url):
    post(url, "/api/view", {"view": "detect", "detector": "colour"})
    post(url, "/api/arm/preset", {"name": "survey"})
    _wait(lambda: get(url, "/api/state")["detection"]["at_station"]
          and get(url, "/api/state")["detection"]["targets"])
    d = get(url, "/api/state")["detection"]
    good = [t for t in d["targets"] if t["graspable"]]
    assert len(good) == 1
    assert good[0]["x_mm"] == pytest.approx(160, abs=3)
    assert good[0]["y_mm"] == pytest.approx(20, abs=3)
    assert good[0]["width_mm"] == pytest.approx(40, abs=3)
    # The wide block is clipped by the frame edge from here, so its size is not
    # trusted -- the sweep sees all of it from the next station and says "wider".
    bad = [t for t in d["targets"] if not t["graspable"]]
    assert bad and all("out of frame" in t["why_not"] or "thin" in t["why_not"] for t in bad)


def test_off_station_detections_are_flagged(sess, url):
    post(url, "/api/view", {"view": "detect", "detector": "colour"})
    post(url, "/api/arm/preset", {"name": "home"})
    _wait(lambda: get(url, "/api/state")["detection"]["age_s"] is not None)
    time.sleep(0.5)
    assert get(url, "/api/state")["detection"]["at_station"] is False


def test_each_view_produces_a_frame(sess, url):
    post(url, "/api/arm/preset", {"name": "survey"})
    for view in session.VIEWS:
        post(url, "/api/view", {"view": view})
        time.sleep(0.15)
        with urllib.request.urlopen(url + "/snapshot.jpg", timeout=5) as reply:
            assert reply.read()[:2] == b"\xff\xd8", view


def test_the_mask_view_shows_the_colour_mask(sess):
    sess.set_view(view="mask", detector="colour")
    sess.preset("survey")
    _wait(lambda: sess._mask is not None)
    mask = sess._mask
    assert mask.shape[:2] == tuple(reversed(cfg.WRIST_CAM_SIZE))
    # the detector's own mask at half brightness, with every outline it found
    # drawn on top: green fill for a graspable object
    assert mask.max() > 0, "the colour mask itself, at half brightness"
    assert (mask[:, :, 1] > 150).any(), "the block's outline, in green"


# --------------------------------------------------------------- auto ----
def test_sweep_lists_what_is_on_the_table(url):
    post(url, "/api/mode", {"mode": "auto"})
    code, body = post(url, "/api/auto/start", {"job": "sweep"})
    assert code == 200 and body["status"] == "running"
    job = finished(url)
    assert job["status"] == "done", job
    s = get(url, "/api/state")
    labels = [(t["graspable"], t["x_mm"], t["y_mm"]) for t in s["sweep"]["targets"]]
    assert any(ok and abs(x - 160) < 3 and abs(y - 20) < 3 for ok, x, y in labels)
    assert any(not ok for ok, _x, _y in labels)
    assert s["arm"]["pose"]["2"] == sim.REAL_SURVEY[2], "returns to the survey pose"
    lines = " ".join(x["text"] for x in get(url, "/api/log?since=0")["lines"])
    assert "merged into 2 object(s)" in lines


def test_a_search_stops_at_the_first_station_that_sees_the_block(url):
    """Arm at the survey pose, block straight ahead: the search looks from where
    it is, finds it, and never visits the other six stations."""
    post(url, "/api/mode", {"mode": "auto"})
    post(url, "/api/auto/start", {"job": "survey"})
    finished(url)
    post(url, "/api/auto/start", {"job": "sweep", "first": True})
    assert finished(url)["status"] == "done"
    lines = [x["text"] for x in get(url, "/api/log?since=0")["lines"]]
    stations = [t for t in lines if t.strip().startswith("J1=")]
    assert len(stations) == 1 and "J1= 90" in stations[0]
    assert any("stopping the scan" in t for t in lines)
    assert get(url, "/api/state")["arm"]["pose"]["1"] == 90, "stays where it found it"


def test_a_pick_does_not_look_twice_when_already_head_on(world, url):
    post(url, "/api/mode", {"mode": "auto"})
    post(url, "/api/auto/start", {"job": "survey"})
    finished(url)
    post(url, "/api/auto/start", {"job": "pick"})
    assert finished(url, timeout=20)["status"] == "done"
    lines = " ".join(x["text"] for x in get(url, "/api/log?since=0")["lines"])
    assert "no second look needed" in lines or "looking again" in lines
    assert len([t for t in lines.split("  J1=") if t]) <= 3, "did not scan the whole ring"


def test_second_job_is_refused_while_one_runs(sess, url):
    sess.arm.time_scale = 0.05
    post(url, "/api/mode", {"mode": "auto"})
    assert post(url, "/api/auto/start", {"job": "sweep"})[0] == 200
    code, body = post(url, "/api/auto/start", {"job": "sweep"})
    assert code == 409 and "running" in body["error"]
    code, body = post(url, "/api/mode", {"mode": "manual"})
    assert code == 409
    assert post(url, "/api/auto/stop")[0] == 200
    assert finished(url)["status"] == "stopped"


def test_the_pose_is_tracked_while_a_job_holds_the_arm(sess, url):
    """A sweep yaws the base through every station; the page must see it happen,
    even though nothing may read the servos while the job holds the lock."""
    sess.arm.time_scale = 0.01
    post(url, "/api/mode", {"mode": "auto"})
    post(url, "/api/auto/start", {"job": "sweep"})
    seen = set()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        s = get(url, "/api/state")
        seen.add(s["arm"]["pose"]["1"])
        if s["job"]["status"] != "running":
            break
        time.sleep(0.02)
    assert len(seen) >= 3, f"the base yaw only ever read {seen}"


def at_drop_pose(block) -> bool:
    """Under cfg.DROP_POSE's fingertips, give or take the simulator's reach shortfall."""
    x, y, _z = kin.forward({**cfg.DROP_POSE, cfg.GRIPPER_ID: cfg.GRIPPER_OPEN})
    return abs(block.x - x) < 0.012 and abs(block.y - y) < 0.012


def test_pick_and_drop_moves_the_block(world, url):
    post(url, "/api/mode", {"mode": "auto"})
    post(url, "/api/auto/start", {"job": "pick"})
    job = finished(url, timeout=20)
    assert job["status"] == "done", job
    block = world.blocks[0]
    assert not block.held
    assert at_drop_pose(block)
    assert get(url, "/api/state")["sweep"]["targets"] == [], "the table changed"


def test_what_lies_at_the_drop_spot_is_never_a_target():
    """The pile we dropped is in view from the ring's end stations; it is not
    listed as graspable and never planned (2026-09-21: three grabs at the pile)."""
    x, y, _z = kin.forward({**cfg.DROP_POSE, cfg.GRIPPER_ID: cfg.GRIPPER_OPEN})
    dropped = detect.Target(x=x + 0.030, y=y + 0.040, width_m=0.040, length_m=0.040,
                            angle_deg=0.0, label="cube")
    info = session.Session._target_dict(dropped)
    assert not info["graspable"] and "drop-off" in info["why_not"]
    assert not session.Session._plannable(dropped)
    on_table = detect.Target(x=0.160, y=0.020, width_m=0.040, length_m=0.040,
                             angle_deg=0.0, label="cube")
    assert session.Session._plannable(on_table)


def test_clear_the_table_sweeps_everything_and_sweeps_again(world, url):
    """Full sweep, pick what can be picked, full sweep again, stop when it finds
    nothing left: the 40 mm block is dropped, the 16 mm one is too thin and stays,
    and the last thing in the log is the empty sweep, not a pick."""
    post(url, "/api/mode", {"mode": "auto"})
    post(url, "/api/auto/start", {"job": "pickall"})
    job = finished(url, timeout=60)
    assert job["status"] == "done", job
    lines = [x["text"] for x in get(url, "/api/log?since=0")["lines"]]
    assert sum("round " in x for x in lines) == 2
    assert any("found nothing left" in x for x in lines)
    moved = [b for b in world.blocks if at_drop_pose(b) and not b.held]
    assert [b.size for b in moved] == [0.040]
    assert all(not b.held for b in world.blocks)


def test_one_click_does_the_whole_thing_from_manual_mode(world, url):
    assert get(url, "/api/state")["mode"] == "manual"
    # What the button sends: the auto detector, no declared size.
    code, body = post(url, "/api/auto/oneclick", {})
    assert code == 200 and body["status"] == "running"
    s = get(url, "/api/state")
    assert s["mode"] == "auto" and s["detector"] == "auto" and s["object_mm"] is None
    assert finished(url, timeout=20)["status"] == "done"
    # Each block is measured for itself: the 16 mm one is too thin and is left
    # alone, the 40 mm one is picked and is now under the drop pose.
    moved = [b for b in world.blocks if at_drop_pose(b) and not b.held]
    assert len(moved) == 1 and moved[0].size == 0.040
    code, body = post(url, "/api/auto/oneclick", {})
    assert code == 200, "a second press after it finished simply runs again"
    post(url, "/api/auto/stop")


def test_dry_run_plans_but_does_not_move(world, url):
    post(url, "/api/mode", {"mode": "auto"})
    post(url, "/api/auto/start", {"job": "pick", "dry_run": True})
    assert finished(url)["status"] == "done"
    assert world.blocks[0].x == pytest.approx(0.160, abs=1e-6)
    lines = " ".join(x["text"] for x in get(url, "/api/log?since=0")["lines"])
    assert "dry run" in lines and "not moving" in lines


def test_picking_a_chosen_target_by_index(world, url):
    post(url, "/api/mode", {"mode": "auto"})
    post(url, "/api/auto/start", {"job": "sweep"})
    finished(url)
    targets = get(url, "/api/state")["sweep"]["targets"]
    too_wide = next(i for i, t in enumerate(targets) if not t["graspable"])
    post(url, "/api/auto/start", {"job": "pick", "index": too_wide})
    job = finished(url)
    assert job["status"] == "failed" and "thin" in job["message"]
    good = next(i for i, t in enumerate(targets) if t["graspable"])
    post(url, "/api/auto/start", {"job": "pick", "index": good, "refine": False})
    assert finished(url, timeout=20)["status"] == "done"
    assert at_drop_pose(world.blocks[0])


def test_background_is_captured_per_station(url):
    post(url, "/api/mode", {"mode": "auto"})
    post(url, "/api/auto/start", {"job": "background"})
    assert finished(url)["status"] == "done"
    assert detect.background_path(0.0).exists()
    assert detect.background_path(25.0).exists()
    # ...and the changes detector can now use them.
    post(url, "/api/view", {"view": "detect", "detector": "changes"})
    _wait(lambda: get(url, "/api/state")["detection"]["error"] is None
          and get(url, "/api/state")["detection"]["at_station"])


def test_jobs_need_a_calibration(monkeypatch, world):
    def missing():
        raise FileNotFoundError("no calibration at /app/data/table_homography.json")
    arm = sim.SimArm(world, time_scale=0)
    s = session.Session(arm_factory=lambda: arm,
                        stream=lambda: sim.SimStream(arm), calibration=missing)
    s.start()
    try:
        assert s.snapshot()["calibration"]["ok"] is False
        s.set_mode("auto")
        job = s.start_job("sweep")
        _wait(lambda: job.status != "running")
        assert job.status == "failed" and "calibration" in job.message
    finally:
        s.close()


def test_log_tee_copies_prints_into_the_log(sess, capsys):
    import sys
    original = sys.stdout
    session.LogTee.install(sess)
    try:
        print("hello from a tool")
    finally:
        sys.stdout = original
    assert any(x["text"] == "hello from a tool" for x in sess.log_since(0))


# ---------------------------------------------------------- the ladder ---
def test_ladder_falls_back_to_tags_when_vision_is_down(monkeypatch):
    def offline(*a, **k):
        raise detect.DetectorOffline("nothing listening")
    monkeypatch.setattr(detect, "objects", offline)
    monkeypatch.setattr(detect, "markers", lambda *a, **k: ["tag"])
    notes = []
    frame = np.zeros((480, 640, 3), np.uint8)
    found = detect.ladder(frame, "yolo", np.eye(3), nadir=(0.1, 0.0), note=notes.append)
    assert found == ["tag"]
    assert any("falling back" in n for n in notes)


def test_ladder_rejects_an_unknown_rung():
    with pytest.raises(ValueError):
        detect.ladder(np.zeros((4, 4, 3), np.uint8), "psychic", np.eye(3))


def test_the_clear_the_table_button_is_a_one_click_too(world, url):
    code, body = post(url, "/api/auto/oneclick", {"job": "pickall"})
    assert code == 200 and body["name"] == "pickall" and body["status"] == "running"
    assert get(url, "/api/state")["mode"] == "auto"
    assert finished(url, timeout=60)["status"] == "done"


def test_the_depth_and_mixed_views_are_served_even_without_a_vision_service(url):
    for view in ("depth", "mixed"):
        code, _body = post(url, "/api/view", {"view": view})
        assert code == 200
        assert get(url, "/api/state")["view"] == view
        req = urllib.request.Request(url + f"/snapshot.jpg?view={view}")
        with urllib.request.urlopen(req, timeout=10) as r:
            assert r.status == 200 and r.read(2) == b"\xff\xd8"
