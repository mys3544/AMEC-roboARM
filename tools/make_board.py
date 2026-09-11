#!/usr/bin/env python3
"""Generate a ChArUco board sized for what the wrist camera can actually see.

    docker compose run --rm core python tools/make_board.py --out /out/board_a4.png

The A3 board has 52.5 mm squares. The wrist camera cannot get higher than about
175 mm above the table before the pose becomes unreachable, which gives a field of
view roughly 157 mm wide -- only three squares. Every marker but the middle one is
clipped by the frame edge, and ArUco needs the whole black border, so we never get
the four points a homography needs.

Finer squares fix it: at 20 mm, the same view holds about 7 squares and several
complete markers, with room to spare at the edges.

PRINT AT EXACTLY 100% (no "fit to page"), then MEASURE a square with a ruler and
tell me the real number -- printer scaling is the classic silent error here, and a
few percent would go straight into every position the arm computes.
"""

import argparse
import sys

import cv2

from roboarm import camera
from roboarm import config as cfg

DPI = 300
MM_PER_INCH = 25.4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--squares-x", type=int, default=9)
    parser.add_argument("--squares-y", type=int, default=7)
    parser.add_argument("--square-mm", type=float, default=20.0)
    parser.add_argument("--marker-mm", type=float, default=15.0)
    parser.add_argument("--out", default="/out/board_a4.png")
    args = parser.parse_args()

    width_mm = args.squares_x * args.square_mm
    height_mm = args.squares_y * args.square_mm
    if width_mm > 200 or height_mm > 287:
        print(f"{width_mm:.0f} x {height_mm:.0f} mm will not fit A4 with margins",
              file=sys.stderr)
        return 1

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.ARUCO_DICT))
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_mm / 1000,
        args.marker_mm / 1000,
        dictionary,
    )

    def px(mm: float) -> int:
        return round(mm / MM_PER_INCH * DPI)

    image = board.generateImage((px(width_mm), px(height_mm)), marginSize=0)

    # White margin so the outer squares keep a quiet zone -- ArUco needs contrast
    # around a marker, and a marker flush to the paper edge often fails to detect.
    pad = px(10)
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    camera.write_image(args.out, image)

    print(f"{args.squares_x}x{args.squares_y} squares of {args.square_mm:.1f} mm "
          f"({args.marker_mm:.1f} mm markers)")
    print(f"pattern {width_mm:.0f} x {height_mm:.0f} mm, plus a 10 mm white margin")
    print(f"image {image.shape[1]}x{image.shape[0]} px at {DPI} dpi -> {args.out}")
    print(f"\nthe camera sees about 157 mm across, so roughly "
          f"{157 / args.square_mm:.1f} squares will be in view")
    print("PRINT AT 100%, then measure a square and tell me the real size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
