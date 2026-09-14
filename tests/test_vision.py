"""The neural-detector rung: the client half (roboarm.detect.objects) and the one
piece of the service that is worth testing without a GPU (extract_detections).

No network, no torch, no ultralytics. The HTTP call is stubbed; the Ultralytics
results object is faked.
"""

import numpy as np
import pytest

from roboarm import detect
from vision_service.detector import (
    Detector,
    PromptNotSupported,
    _resolve_model,
    extract_detections,
)

# Same trivial 1 px == 1 mm mapping as test_grasp: lets expected table coords be
# read straight off the pixel boxes.
MM_PER_PIXEL = np.diag([0.001, 0.001, 1.0])


# ------------------------------------------------------ service: extract --
class FakeBox:
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


# --------------------------------------------------- service: prompting --
class FakeModel:
    """Just enough of an Ultralytics model for Detector's non-inference paths."""

    def __init__(self):
        self.names = {0: "person", 1: "cup"}
        self.set_calls = []

    def set_classes(self, names, _pe):
        self.set_calls.append(list(names))
        self.names = {i: n for i, n in enumerate(names)}

    def get_text_pe(self, names):
        return f"pe:{list(names)}"

    def predict(self, *a, **k):
        raise AssertionError("predict should not run in these tests")


def test_a_prompt_against_a_fixed_class_model_is_refused_not_ignored():
    det = Detector(FakeModel(), "yolo26n.pt", open_vocab=False, imgsz=640, conf=0.25)
    with pytest.raises(PromptNotSupported, match="fixed class list"):
        det.infer(b"not-a-real-jpeg", prompt="mug")


def test_setting_the_same_classes_twice_skips_the_text_encoder():
    model = FakeModel()
    det = Detector(model, "yoloe.pt", open_vocab=True, imgsz=640, conf=0.25)
    det._set_classes(["red cube", "mug"])
    det._set_classes(["red cube", "mug"])
    assert model.set_calls == [["red cube", "mug"]], "second identical prompt re-encoded"
    det._set_classes(["spoon"])
    assert model.set_calls == [["red cube", "mug"], ["spoon"]]


def test_health_reports_the_model_and_its_classes():
    det = Detector(FakeModel(), "yolo26n.pt", open_vocab=False, imgsz=640, conf=0.3)
    health = det.health()
    assert health["status"] == "ok"
    assert health["model"] == "yolo26n.pt"
    assert health["open_vocab"] is False
    assert set(health["classes"]) == {"person", "cup"}


# --------------------------------------------------------- client: objects --
def _reply(detections):
    return {"model": "yolo26n.pt", "open_vocab": False, "prompt": None,
            "width": 640, "height": 480, "detections": detections}


def test_objects_maps_pixel_boxes_to_table_targets(monkeypatch):
    posted = {}

    def fake_post(path, body, headers, url, timeout):
        posted["path"], posted["headers"] = path, headers
        return _reply([
            {"label": "cup", "confidence": 0.9, "box": [200, 150, 30, 40]},
        ])

    monkeypatch.setattr(detect, "_vision_post", fake_post)
    frame = np.zeros((480, 640, 3), np.uint8)

    found = detect.objects(frame, MM_PER_PIXEL)
    assert posted["path"] == "/detect"
    assert len(found) == 1
    target = found[0]
    assert target.label == "cup"
    assert target.confidence == pytest.approx(0.9)
    assert target.width_m == pytest.approx(0.030, abs=1e-6)
    assert target.length_m == pytest.approx(0.040, abs=1e-6)
    assert target.x == pytest.approx(0.215, abs=1e-6)  # box centre, 1 px == 1 mm
    assert target.y == pytest.approx(0.170, abs=1e-6)


def test_objects_passes_the_prompt_as_a_header(monkeypatch):
    seen = {}

    def fake_post(path, body, headers, url, timeout):
        seen.update(headers)
        return _reply([])

    monkeypatch.setattr(detect, "_vision_post", fake_post)
    detect.objects(np.zeros((10, 10, 3), np.uint8), MM_PER_PIXEL, prompt="red cube")
    assert seen["X-Vision-Prompt"] == "red cube"


def test_objects_drops_specks_below_the_area_floor(monkeypatch):
    monkeypatch.setattr(detect, "_vision_post", lambda *a, **k: _reply([
        {"label": "dust", "confidence": 0.4, "box": [10, 10, 5, 5]},
        {"label": "cup", "confidence": 0.8, "box": [100, 100, 30, 40]},
    ]))
    found = detect.objects(np.zeros((240, 320, 3), np.uint8), MM_PER_PIXEL)
    assert [t.label for t in found] == ["cup"]


def _square(x, y, side):
    return [[x, y], [x + side, y], [x + side, y + side], [x, y + side]]


def test_objects_drops_an_outline_nested_inside_another(monkeypatch):
    # The tagged cube: a 60 px "traffic sign" and, inside it, the tag's inner
    # square as a "direct". Only the whole cube is an object.
    monkeypatch.setattr(detect, "_vision_post", lambda *a, **k: _reply([
        {"label": "direct", "confidence": 0.9, "box": [120, 120, 30, 30],
         "polygon": _square(120, 120, 30)},
        {"label": "traffic sign", "confidence": 0.6, "box": [100, 100, 60, 60],
         "polygon": _square(100, 100, 60)},
    ]))
    found = detect.objects(np.zeros((240, 320, 3), np.uint8), MM_PER_PIXEL)
    assert [t.label for t in found] == ["traffic sign"]


def test_objects_keeps_two_objects_that_merely_overlap(monkeypatch):
    monkeypatch.setattr(detect, "_vision_post", lambda *a, **k: _reply([
        {"label": "a", "confidence": 0.9, "box": [100, 100, 40, 40],
         "polygon": _square(100, 100, 40)},
        {"label": "b", "confidence": 0.8, "box": [130, 100, 40, 40],
         "polygon": _square(130, 100, 40)},
    ]))
    found = detect.objects(np.zeros((240, 320, 3), np.uint8), MM_PER_PIXEL)
    assert sorted(t.label for t in found) == ["a", "b"]


def test_objects_drops_the_whole_picture_without_it_swallowing_the_table(monkeypatch):
    # Prompt-free YOLOE labels the frame itself ("studio shot") on most shots.
    monkeypatch.setattr(detect, "_vision_post", lambda *a, **k: _reply([
        {"label": "studio shot", "confidence": 0.9, "box": [0, 0, 320, 240],
         "polygon": _square(0, 0, 320)},
        {"label": "stop sign", "confidence": 0.7, "box": [100, 100, 40, 40],
         "polygon": _square(100, 100, 40)},
    ]))
    found = detect.objects(np.zeros((240, 320, 3), np.uint8), MM_PER_PIXEL)
    assert [t.label for t in found] == ["stop sign"]


def test_edges_touched_names_the_sides():
    shape = (240, 320, 3)
    assert detect.edges_touched(_square(100, 100, 40), shape) == frozenset()
    assert detect.edges_touched(_square(0, 100, 40), shape) == {"left"}
    assert detect.edges_touched(_square(100, 0, 40), shape) == {"top"}
    assert detect.edges_touched(_square(280, 200, 40), shape) == {"right", "bottom"}
    assert detect.touches_edge(_square(280, 200, 40), shape) is True


def test_a_clipped_object_records_which_edge_it_left_by(monkeypatch):
    monkeypatch.setattr(detect, "_vision_post", lambda *a, **k: _reply([
        {"label": "cube", "confidence": 0.8, "box": [100, 0, 40, 40],
         "polygon": _square(100, 0, 40)},
    ]))
    found = detect.objects(np.zeros((240, 320, 3), np.uint8), MM_PER_PIXEL)
    assert len(found) == 1
    assert found[0].clipped is True
    assert found[0].edges == {"top"}
    assert found[0].graspable is False


def test_objects_sorts_nearest_first(monkeypatch):
    monkeypatch.setattr(detect, "_vision_post", lambda *a, **k: _reply([
        {"label": "far", "confidence": 0.8, "box": [400, 100, 30, 30]},
        {"label": "near", "confidence": 0.8, "box": [100, 100, 30, 30]},
    ]))
    found = detect.objects(np.zeros((240, 640, 3), np.uint8), MM_PER_PIXEL)
    assert [t.label for t in found] == ["near", "far"]


def test_objects_raises_DetectorOffline_when_nothing_answers(monkeypatch):
    def boom(*a, **k):
        raise detect.DetectorOffline("no vision service at http://vision:8760")

    monkeypatch.setattr(detect, "_vision_post", boom)
    with pytest.raises(detect.DetectorOffline):
        detect.objects(np.zeros((10, 10, 3), np.uint8), MM_PER_PIXEL)


def test_objects_propagates_a_rejected_request_as_valueerror(monkeypatch):
    def rejected(*a, **k):
        raise ValueError("vision service rejected the request (400): fixed class list")

    monkeypatch.setattr(detect, "_vision_post", rejected)
    with pytest.raises(ValueError, match="rejected"):
        detect.objects(np.zeros((10, 10, 3), np.uint8), MM_PER_PIXEL, prompt="mug")


def test_vision_available_is_true_only_on_a_healthy_reply(monkeypatch):
    monkeypatch.setattr(detect, "_vision_post", lambda *a, **k: {"status": "ok"})
    assert detect.vision_available() is True

    monkeypatch.setattr(detect, "_vision_post", lambda *a, **k: {"status": "loading"})
    assert detect.vision_available() is False

    def offline(*a, **k):
        raise detect.DetectorOffline("down")

    monkeypatch.setattr(detect, "_vision_post", offline)
    assert detect.vision_available() is False


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


# --------------------------------------------------------------- Target --
def test_geometric_targets_default_to_full_confidence():
    t = detect.Target(x=0.1, y=0.0, width_m=0.03, length_m=0.03, angle_deg=0.0, label="x")
    assert t.confidence == 1.0
