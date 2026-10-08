"""The kit both robots clear (plan v2.2): the 30 and 40 mm cubes and the 30 x 30 x 60 block (two
cubes glued), lying and standing."""

# (kind, width across, length, height) in m. The fingers close across `width`.
KIT = (("cube", 0.030, 0.030, 0.030), ("cube", 0.040, 0.040, 0.040),
       ("block", 0.030, 0.060, 0.030), ("block", 0.030, 0.030, 0.060))

# How far a footprint seen from above may read from a kit piece's and still be one: the X3's wrist
# camera reads the cubes ~3 mm narrow (27 and 37 mm, 2026-10-08), the tape it took for a cube 20.
WIDTH_SLACK_M = 0.007
LENGTH_SLACK_M = 0.012


def footprint_fits(width: float, length: float) -> bool:
    """Whether a top face `width` x `length` m (narrow way first) is a kit piece's, as seen from above
    -- a standing block shows a cube's."""
    return any(abs(width - w) <= WIDTH_SLACK_M and length <= ln + LENGTH_SLACK_M for _k, w, ln, _h in KIT)
