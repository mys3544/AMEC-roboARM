"""roboshared.serve's scan zip: the newest scan folder (or the one asked for) as STAMP/<file>, and
a robot with no scan says so."""

import io
import json
import zipfile

from roboshared import serve


def test_the_newest_scan_or_the_one_asked_for_comes_as_one_folder_in_a_zip(tmp_path):
    for stamp in ("20261012_161735", "20261012_165541"):
        (tmp_path / stamp).mkdir()
        (tmp_path / stamp / "00_0_rgb.jpg").write_bytes(b"jpg " + stamp.encode())
        (tmp_path / stamp / "scan.json").write_text("{}")
    (tmp_path / "notes").mkdir()   # not a scan's stamp: never sent

    status, kind, body = serve.scan_zip(tmp_path, None)
    assert (status, kind) == (200, "application/zip")
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        assert sorted(z.namelist()) == ["20261012_165541/00_0_rgb.jpg", "20261012_165541/scan.json"]
        assert z.read("20261012_165541/00_0_rgb.jpg") == b"jpg 20261012_165541"
    with zipfile.ZipFile(io.BytesIO(serve.scan_zip(tmp_path, "20261012_161735")[2])) as z:
        assert z.read("20261012_161735/00_0_rgb.jpg") == b"jpg 20261012_161735"

    status, _kind, body = serve.scan_zip(tmp_path, "notes")
    assert status == 404 and "no scan notes" in json.loads(body)["error"]
    status, _kind, body = serve.scan_zip(tmp_path / "none yet", None)
    assert status == 404 and json.loads(body)["error"] == "no scan yet"
