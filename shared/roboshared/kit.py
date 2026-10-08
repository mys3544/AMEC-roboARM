"""The kit both robots clear (plan v2.2): the 30 and 40 mm cubes and the 30 x 30 x 60 block (two
cubes glued), lying and standing."""

# (kind, width across, length, height) in m. The fingers close across `width`.
KIT = (("cube", 0.030, 0.030, 0.030), ("cube", 0.040, 0.040, 0.040),
       ("block", 0.030, 0.060, 0.030), ("block", 0.030, 0.030, 0.060))

# How far a footprint seen from above may read from a kit piece's and still be one: the X3's wrist
# camera reads the cubes ~3 mm narrow (27 and 37 mm, 2026-10-08), the tape it took for a cube 20.
WIDTH_SLACK_M = 0.007
LENGTH_SLACK_M = 0.012


def nearest(width: float, length: float):
    """The kit piece (KIT's row) a top face `width` x `length` m (narrow way first) is, as seen from
    above, the nearest in size -- a standing block shows a cube's, and is taken for the cube -- or
    None."""
    fits = [item for item in KIT if abs(width - item[1]) <= WIDTH_SLACK_M and length <= item[2] + LENGTH_SLACK_M]
    return min(fits, key=lambda item: abs(width - item[1]) + abs(length - item[2])) if fits else None


def footprint_fits(width: float, length: float) -> bool:
    """Whether a top face `width` x `length` m (narrow way first) is a kit piece's."""
    return nearest(width, length) is not None
