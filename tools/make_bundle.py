#!/usr/bin/env python3
"""Pack a student copy of the project into one zip.

    python tools/make_bundle.py                  # -> roboarm-2026-<date>.zip, here
    python tools/make_bundle.py --out C:/handout # somewhere else
    python tools/make_bundle.py --list           # print what would go in, write nothing

What goes in is a WHITELIST (PATTERNS), the same idea as tools/sync_robot.py, so
nothing large or machine-specific can wander in by accident. Deliberately absent:

    .venv/      312 MB, and built for whatever machine made it -- `uv sync --frozen`
                rebuilds it in a minute from uv.lock, which IS in the bundle
    data/       the robot's calibration and recorded runs. A laptop in --robot mode
                fetches the calibration from the bridge over HTTP and writes its own
                background photo, so a student never needs this
    out/        annotated frames from past runs; megabytes of nobody's business
    models/     neural weights; they live on the robot, and only the robot runs them
    vendor/     the Rosmaster SDK, needed only by a process wired to the servos.
                roboarm/arm.py tolerates it missing (connecting raises ArmError)

Everything is stored under one top-level folder, so unzipping in a downloads
directory does not spray files everywhere. README.md is what the student reads
first; the script refuses to build a bundle without it.
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEM = "roboarm-2026"

# Everything a student needs to run the simulator, drive the robot over the bridge,
# and read how the robot side is put together. docker/ and compose.yaml are the
# robot's own setup: they will not run them, but the panel makes more sense with
# them to hand.
PATTERNS = [
    "README.md",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    "roboarm/**/*.py",
    "roboarm/web/static/*.html",
    "tools/*.py",
    "tests/*.py",
    "vision_service/*.py",   # tests/test_vision.py imports the detector's pure logic
    "docs/*.md",
    "docker/*",
    "compose.yaml",
]

# Matched by PATTERNS, kept out on purpose.
EXCLUDE = {
    # Uploads the whole tree to the lab robot over SSH. An instructor's tool, and
    # a fine way for a student to overwrite the working copy everyone shares.
    "tools/sync_robot.py",
}

# Anything bigger than this in a source bundle is a mistake worth seeing.
LARGE_FILE_MB = 2.0


def bundle_files() -> list[str]:
    """Project-relative paths to pack, sorted, caches and EXCLUDE dropped."""
    found = set()
    for pattern in PATTERNS:
        for path in ROOT.glob(pattern):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel not in EXCLUDE:
                found.add(rel)
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=".", metavar="DIR",
                        help="where to write the zip (default: the project folder)")
    parser.add_argument("--name", default=None,
                        help=f"zip filename (default: {STEM}-<today>.zip)")
    parser.add_argument("--list", action="store_true", dest="list_only",
                        help="print the file list and the total size, write nothing")
    args = parser.parse_args()

    files = bundle_files()
    if "README.md" not in files:
        print("README.md is missing -- it is the first thing a student reads.",
              file=sys.stderr)
        return 1

    total = sum((ROOT / rel).stat().st_size for rel in files)
    for rel in files:
        size = (ROOT / rel).stat().st_size
        if size > LARGE_FILE_MB * 1e6:
            print(f"  note: {rel} is {size / 1e6:.1f} MB", file=sys.stderr)

    if args.list_only:
        for rel in files:
            print(rel)
        print(f"\n{len(files)} file(s), {total / 1e6:.1f} MB uncompressed")
        return 0

    name = args.name or f"{STEM}-{time.strftime('%Y%m%d')}.zip"
    target = Path(args.out).expanduser().resolve() / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for rel in files:
            archive.write(ROOT / rel, f"{STEM}/{rel}")

    packed = target.stat().st_size
    print(f"{target}\n{len(files)} file(s), {total / 1e6:.1f} MB -> {packed / 1e6:.1f} MB")
    print(f"Unpacks to {STEM}/ ; the student then runs `uv sync --frozen`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
