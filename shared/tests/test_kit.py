from roboshared import kit


def test_the_kit_seen_from_above():
    assert kit.footprint_fits(0.027, 0.028)    # the 30 mm cube, read 3 mm narrow (2026-10-08)
    assert kit.footprint_fits(0.037, 0.038)    # the 40 mm cube
    assert kit.footprint_fits(0.028, 0.058)    # the block lying
    assert not kit.footprint_fits(0.020, 0.024)   # the black tape on the paper's corner
    assert not kit.footprint_fits(0.030, 0.080)   # a strip, not a block


def test_a_reading_is_the_nearest_kit_piece():
    assert kit.nearest(0.027, 0.035)[:3] == ("cube", 0.030, 0.030)
    assert kit.nearest(0.037, 0.045)[:3] == ("cube", 0.040, 0.040)
    assert kit.nearest(0.033, 0.038)[:3] == ("cube", 0.040, 0.040)
    assert kit.nearest(0.027, 0.057)[:3] == ("block", 0.030, 0.060)
    assert kit.nearest(0.020, 0.024) is None
