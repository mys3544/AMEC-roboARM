"""locate.py without a network: which robot a /health is, and where each robot is found."""

import locate

M3 = {"ok": True, "arm": {}, "camera": {"open": True, "fps": 30, "colour_age_s": 0.0, "depth_age_s": 0.0}}
X3 = {"ok": True, "arm": {}, "camera": {"open": True, "fps": 30}, "calibration": True}
ROBOTS = [{"id": "x3arm", "kind": "x3arm", "url": "http://192.168.73.210:8761"},
          {"id": "m3", "kind": "m3", "url": "http://192.168.73.66:8095"},
          {"id": "m3arm", "kind": "m3arm", "url": "http://192.168.73.66:8761"}]


def where(monkeypatch, names, answers, scanned):
    monkeypatch.setattr(locate, "resolve", lambda name, timeout=2.0: names.get(name))
    monkeypatch.setattr(locate, "health", lambda host, timeout=0.6: answers.get(host))
    monkeypatch.setattr(locate, "scan", lambda: scanned)
    said = []
    return {r["id"]: r["url"] for r in locate.locate(ROBOTS, scan_too=bool(scanned), say=said.append)}, said


def test_a_health_tells_the_robot():
    assert locate.who(M3) == "m3" and locate.who(X3) == "x3" and locate.who({"ok": True}) is None
    assert locate.who(None) is None and locate.robot_of("m3arm") == "m3" and locate.robot_of("x3") == "x3"


def test_by_name_first_the_drive_page_with_its_arm(monkeypatch):
    urls, said = where(monkeypatch, {"m3pro.local": "192.168.43.17", "x3plus.local": "192.168.43.18"},
                       {"192.168.43.17": M3, "192.168.43.18": X3}, {})
    assert urls == {"x3arm": "http://192.168.43.18:8761", "m3": "http://192.168.43.17:8095",
                    "m3arm": "http://192.168.43.17:8761"}
    assert any("by its name m3pro.local" in s for s in said)


def test_a_name_that_answers_as_the_other_robot_is_not_taken(monkeypatch):
    urls, _ = where(monkeypatch, {"m3pro.local": "10.0.0.5"}, {"10.0.0.5": X3, "192.168.73.66": M3, "192.168.73.210": X3}, {})
    assert urls["m3arm"] == "http://192.168.73.66:8761" and urls["x3arm"] == "http://192.168.73.210:8761"


def test_no_scan_unless_asked(monkeypatch):
    monkeypatch.setattr(locate, "resolve", lambda name, timeout=2.0: None)
    monkeypatch.setattr(locate, "health", lambda host, timeout=0.6: None)
    monkeypatch.setattr(locate, "scan", lambda: (_ for _ in ()).throw(AssertionError("scanned")))
    said = []
    assert locate.locate(ROBOTS, say=said.append) == ROBOTS and any("--scan" in s for s in said)


def test_a_scan_when_neither_name_nor_lab_address_answers(monkeypatch):
    urls, said = where(monkeypatch, {}, {}, {"m3": "192.168.43.99"})
    assert urls["m3arm"] == "http://192.168.43.99:8761" and urls["m3"] == "http://192.168.43.99:8095"
    assert urls["x3arm"] == "http://192.168.73.210:8761"   # found nowhere: robots.json's
    assert any("x3: not found" in s for s in said)
