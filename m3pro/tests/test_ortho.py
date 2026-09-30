"""arena/ortho.py's crop on a turned 3.2 x 2.8 m fence: the arena and its walls stay, the room
seen over the fence goes -- beyond the walls, and standing against them above the fence top."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "arena"))
import fence_fit
import ortho

FLOOR = 0.013
TURN = np.radians(23)
ORIGIN, EX = np.array([-1.7, 0.4]), np.array([np.cos(TURN), np.sin(TURN)])
FENCE = {"arena_frame": {"origin_map": ORIGIN.tolist(), "x_axis_map": EX.tolist()},
         "fence": {"size_m": [3.2, 2.8]}}


def at(a, b, height):   # a point in the arena frame (a along the long side), height above the floor
    xy = ORIGIN + a * EX + b * fence_fit.rot90(EX)
    return [xy[0], xy[1], FLOOR + height]


def test_the_crop_keeps_the_arena_and_its_walls_and_drops_the_room_over_the_fence():
    points = {   # name: (point, kept?)
        "floor, middle": (at(1.6, 1.4, 0.0), True),
        "a robot in the middle, 60 cm": (at(1.6, 1.4, 0.6), True),
        "landmark near a wall, 28 cm": (at(0.3, 0.12, 0.28), True),
        "wall, low, outer side": (at(-0.03, 1.0, 0.15), True),
        "wall top, far wall": (at(3.23, 1.0, 0.33), True),
        "against a wall outside, 80 cm": (at(0.8, -0.04, 0.8), False),
        "against the far wall, just inside its line, 50 cm": (at(1.0, 2.77, 0.5), False),
        "the room beyond a wall, low": (at(-0.2, 1.0, 0.1), False),
        "the room beyond a corner": (at(3.3, 2.9, 0.2), False),
    }
    xyz = np.array([p for p, _ in points.values()])
    keep, over = ortho.crop(xyz, FENCE, FLOOR)
    assert dict(zip(points, keep.tolist())) == {name: k for name, (_, k) in points.items()}
    assert over.tolist() == [False] * 5 + [True, True, False, False]   # the room beyond is not "over"
