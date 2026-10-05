"""vision_service, the parts worth testing without a GPU: the detector's
extract_detections, health and model resolution, and the depth rung's raised regions.

No network, no torch, no ultralytics: the Ultralytics results object is faked.
"""

import cv2
import numpy as np
import pytest

from vision_service import depth as vdepth
from vision_service.detector import Detector, _resolve_model, extract_detections


# ------------------------------------------------------ service: extract --
class FakeBox:
    """The shapes Ultralytics yields per box: cls/conf (1,), xyxy (1, 4)."""

    def __init__(self, cls, conf, xyxy):
        self.cls = [cls]
        self.conf = [conf]
        self.xyxy = [xyxy]


def test_extract_reads_label_score_and_xywh_from_a_results_object():
    boxes = [FakeBox(2, 0.87, [10.0, 20.0, 40.0, 80.0])]
    out = extract_detections(boxes, {0: "a", 1: "b", 2: "cup"})
    assert out == [{"label": "cup", "confidence": 0.87, "box": [10.0, 20.0, 30.0, 60.0]}]


def test_extract_accepts_a_list_of_names_too():
    boxes = [FakeBox(1, 0.5, [0.0, 0.0, 5.0, 5.0])]
    assert extract_detections(boxes, ["cat", "dog"])[0]["label"] == "dog"


def test_extract_of_nothing_is_an_empty_list():
    assert extract_detections([], {0: "x"}) == []


class FakeMasks:
    def __init__(self, polygons):
        self.xy = polygons


def test_extract_carries_a_segmentation_models_outline():
    boxes = [FakeBox(0, 0.9, [10.0, 20.0, 40.0, 80.0]), FakeBox(0, 0.8, [0.0, 0.0, 5.0, 5.0])]
    masks = FakeMasks([np.array([[10.0, 20.0], [40.0, 20.0], [40.0, 80.0], [10.0, 80.0]]),
                       np.zeros((0, 2))])
    out = extract_detections(boxes, ["cube"], masks)
    assert out[0]["polygon"] == [[10.0, 20.0], [40.0, 20.0], [40.0, 80.0], [10.0, 80.0]]
    assert "polygon" not in out[1], "an empty mask sends no outline"


# ------------------------------------------------------ service: health --
class FakeModel:
    """Just enough of an Ultralytics model for Detector's non-inference paths."""

    def __init__(self):
        self.names = {0: "person", 1: "cup"}

    def predict(self, *a, **k):
        raise AssertionError("predict should not run in these tests")


def test_health_reports_the_model_and_its_classes():
    det = Detector(FakeModel(), "yolo26n.pt", imgsz=640, conf=0.3)
    health = det.health()
    assert health["status"] == "ok"
    assert health["model"] == "yolo26n.pt"
    assert set(health["classes"]) == {"person", "cup"}
    assert health["conf"] == 0.3


# ------------------------------------------------------- model resolution --
def test_resolve_prefers_a_sibling_engine_over_the_configured_pt(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOARM_VISION_NO_AUTO_ENGINE", raising=False)
    (tmp_path / "yolo26n.pt").write_bytes(b"pt")
    (tmp_path / "yolo26n.engine").write_bytes(b"engine")
    assert _resolve_model(str(tmp_path / "yolo26n.pt")).endswith("yolo26n.engine")


def test_resolve_keeps_the_pt_when_no_engine_is_built(tmp_path):
    (tmp_path / "yolo26n.pt").write_bytes(b"pt")
    assert _resolve_model(str(tmp_path / "yolo26n.pt")).endswith("yolo26n.pt")


def test_resolve_honours_the_opt_out(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOARM_VISION_NO_AUTO_ENGINE", "1")
    (tmp_path / "m.pt").write_bytes(b"pt")
    (tmp_path / "m.engine").write_bytes(b"engine")
    assert _resolve_model(str(tmp_path / "m.pt")).endswith("m.pt")


def test_resolve_passes_through_an_explicit_engine_and_an_unfetched_name(tmp_path):
    (tmp_path / "x.engine").write_bytes(b"e")
    assert _resolve_model(str(tmp_path / "x.engine")).endswith("x.engine")
    # a bare name uv/ultralytics would fetch, no local sibling -> unchanged
    assert _resolve_model("yolo26n.pt") == "yolo26n.pt"


# ------------------------------------------------------------ depth rung ----

def _ramp_with_plateau(plateau=None, step=0.15):
    """A relative depth map the way the model draws the table: a slow ramp (the
    near edge looks nearer) plus, optionally, a flat plateau standing above it."""
    h, w = 518, 686
    ramp = np.linspace(0.2, 0.5, h)[:, None] * np.ones((1, w))
    ramp = ramp + 0.0005 * np.random.default_rng(1).standard_normal((h, w))
    if plateau:
        x, y, side = plateau
        ramp[y:y + side, x:x + side] += step
    return ramp.astype(np.float32)


def test_a_plateau_on_the_ramp_is_a_raised_region_with_its_outline():
    found = vdepth.raised_regions(_ramp_with_plateau((300, 200, 120)))
    assert len(found) == 1
    outline, step = found[0]
    assert step > vdepth.MIN_STEP
    x, y, w, h = [int(v) for v in np.concatenate([outline.min(0), outline.max(0) - outline.min(0)])]
    assert abs(x - 300) <= 6 and abs(y - 200) <= 6 and abs(w - 120) <= 12 and abs(h - 120) <= 12


def test_the_bare_ramp_has_nothing_raised():
    assert vdepth.raised_regions(_ramp_with_plateau(None)) == []


def test_a_plateau_cut_off_by_the_border_is_not_a_region():
    assert vdepth.raised_regions(_ramp_with_plateau((600, 200, 120))) == []


def test_a_dip_is_not_raised():
    assert vdepth.raised_regions(_ramp_with_plateau((300, 200, 120), step=-0.15)) == []


def _sloped_plateau(tilt):
    """A plateau whose top the model draws SLOPING across it, as it drew a 30 x 60
    block lying flat on 2026-09-24, on the table's ramp (the C930e's 16:9 map)."""
    h, w = 518, 910
    ramp = np.linspace(0.2, 0.5, h)[:, None] * np.ones((1, w))
    ramp = ramp + 0.0005 * np.random.default_rng(1).standard_normal((h, w))
    ramp[150:350, 300:440] += 0.4 + tilt * np.linspace(0, 1, 140)[None, :]
    return ramp.astype(np.float32)


def test_a_plateau_whose_top_slopes_is_found_whole():
    # With only the noise threshold its whole top was "edge" (found nothing).
    found = vdepth.raised_regions(_sloped_plateau(0.3))
    assert len(found) == 1
    area = cv2.contourArea(found[0][0].astype(np.float32))
    assert area == pytest.approx(140 * 200, rel=0.15)


def _stub_passes(monkeypatch, fine, coarse):
    """raised_regions() with its two passes replaced: `fine` below the floor,
    `coarse` at it. Each is a list of (mask, step)."""
    def regions(_disp, _grad, threshold):
        chosen = coarse if threshold >= vdepth.EDGE_FLOOR else fine
        return [(mask, np.argwhere(mask)[:, ::-1].astype(float), step) for mask, step in chosen]
    monkeypatch.setattr(vdepth, "_regions", regions)


def _box(x0, x1):
    mask = np.zeros((518, 910), bool)
    mask[150:350, x0:x1] = True
    return mask


def test_the_coarse_pass_mends_a_sliver(monkeypatch):
    _stub_passes(monkeypatch, fine=[(_box(360, 380), 9.0)], coarse=[(_box(300, 440), 80.0)])
    assert [step for _outline, step in vdepth.raised_regions(_ramp_with_plateau(None))] == [80.0]


def test_the_coarse_pass_never_merges_what_the_fine_pass_kept_apart(monkeypatch):
    _stub_passes(monkeypatch, fine=[(_box(300, 370), 40.0), (_box(372, 440), 30.0)],
                 coarse=[(_box(300, 440), 80.0)])
    assert [step for _outline, step in vdepth.raised_regions(_ramp_with_plateau(None))] == [40.0, 30.0]
