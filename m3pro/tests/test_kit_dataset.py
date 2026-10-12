"""tools/kit_dataset.py: the depth's named outlines become the kit detector's classes and polygons;
what it could not name stays background; a pick's looks stay on one side of the split."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import kit_dataset


def thing(kind, size, length, outline=((0, 0), (640, 0), (320, 480))):
    return {"kind": kind, "size": size, "length": length, "outline_px": [list(p) for p in outline]}


def test_the_kit_s_pieces_are_named_and_the_rest_left_out():
    assert kit_dataset.kit_class(thing("cube", 0.03, 0.03)) == 0
    assert kit_dataset.kit_class(thing("cube", 0.04, 0.04)) == 1
    assert kit_dataset.kit_class(thing("block", 0.03, 0.06)) == 2
    assert kit_dataset.kit_class(thing("block", 0.03, 0.03)) == 2   # standing
    assert kit_dataset.kit_class(thing(None, None, None)) is None
    lines = kit_dataset.label_lines({"found": [thing("cube", 0.03, 0.03, ((-5, 10), (700, 10), (320, 240))),
                                               thing(None, None, None)]})
    assert lines == ["0 0.00000 0.02083 1.00000 0.02083 0.50000 0.50000"]   # clipped to the picture


def test_a_scene_is_a_pick_s_wide_look_and_its_close_looks():
    stems = ["20261009_174744_wide", "20261009_174825_close", "20261009_175153_wide", "20261009_175201_close",
             "20261009_175214_wide", "20261009_175230_close", "20261009_175235_close"]   # picks in a row
    assert kit_dataset.scenes(stems) == [stems[:2], stems[2:4], stems[4:]]
    assert kit_dataset.scenes(["20261009_174825_close", "20261009_175201_close"]) == [
        ["20261009_174825_close"], ["20261009_175201_close"]]   # a minute apart: apart


def test_the_box_views_are_left_out(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    frames.mkdir()
    for stem in ("20261009_174744_wide", "20261009_172518_wide J1 160"):
        (frames / f"{stem}.json").write_text(json.dumps({"found": [thing("cube", 0.03, 0.03)]}))
        (frames / f"{stem}_rgb.jpg").write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["kit_dataset", "--frames", str(frames), "--out", str(tmp_path / "kit")])
    kit_dataset.main()
    assert [p.name for p in (tmp_path / "kit" / "labels").rglob("*.txt")] == ["20261009_174744_wide.txt"]
    assert "names:" in (tmp_path / "kit" / "kit.yaml").read_text()
