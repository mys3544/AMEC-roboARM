"""The kit detector's pictures from the X3 (PLAN.md M5): the looks the X3 keeps (x3pick.save_look,
since 2026-10-07: the picture as the camera gave it, the arm's pose, what the ladder found on the
table) labelled by putting each kit piece's box back into the picture -- the wrist camera's model
(roboarm.lens with data/cameras/<camera>/lens_on_arm.json) projects the box's corners, their hull
is the outline. The detector trained on the M3's camera alone took the X3's tag board for cubes
(2026-10-15): these pictures, the board on its own among them, are what it has not seen.

    python tools/kit_dataset_x3.py fetch                    # the X3's out/pick/looks -> out/looks/
    python tools/kit_dataset_x3.py build [--into DIR] [--val-every 5] [--draw 12]

`build` adds the looks to the M3's set (m3pro/tools/kit_dataset.py writes it first, to
m3pro/out/kit) as x3_*, in its layout and classes, so tools/kit_cycle.py --x3 trains on both. A
find is named by its size as the X3's pick names it (roboshared.kit.nearest), its box the kit's own
size (the camera reads the cubes ~3 mm narrow); a 30 mm footprint standing 45 mm or more is the
block on end. What is not kit-sized stays unlabelled, i.e. background, as on the M3.

Labelled by survey, not by look: nothing on the table moves within one survey (its looks come a
second apart), so every piece the survey saw whole in ANY look is put into EVERY look of it --
cut by the picture's edge, or missed by that look's own ladder (2026-10-15: one look at a time
left most pieces in the box, the tagged ones and those at the edge unlabelled, i.e. taught as
background). A survey that saw no piece -- the tag board on its own -- gives pictures with
nothing to find. `--draw N` keeps N labelled pictures in out/kit_x3_check/ to look at first.

A test set no training sees: move a session's looks from out/looks/ to out/looks_test/ (fetch
does not bring them back), then `build --looks out/looks_test --into DIR --val-every 1`, DIR a
set's layout with a kit.yaml, and train/eval_kit.py --test DIR on hekate.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roboshared import kit

from roboarm import config as cfg
from roboarm import lens

ROOT = Path(__file__).resolve().parents[1]
LOOKS = ROOT / "out" / "looks"
HELD_OUT = ROOT / "out" / "looks_test"   # looks moved here are never trained on: the test set
INTO = ROOT.parent / "m3pro" / "out" / "kit"
NAMES = ["cube30", "cube40", "block"]   # the M3's set's classes (m3pro/tools/kit_dataset.py)
HOST, REMOTE = "jetson@192.168.73.210", "roboarm/out/pick/looks"
SCENE_GAP_S = 60   # looks this far apart are separate scenes (the val split goes by scene)
SURVEY_GAP_S = 4.0  # looks closer than this are one survey: nothing on the table moved between them
STANDING_M = 0.045
SAME_M = 0.02      # finds this close are one piece
MIN_INSIDE = 0.25  # of a piece's outline in the picture for it to be labelled there
MIN_AHEAD_M = 0.05  # each corner at least this far in front of the lens to be projected


def kit_class(found: dict) -> tuple[int, tuple] | None:
    """(class, the KIT row) of a find, by its size; None for what is no kit piece."""
    item = kit.nearest(found["width"], found["length"])
    if item is None:
        return None
    kind, width, _length, _height = item
    if kind == "cube" and width == 0.030 and (found.get("height") or 0) >= STANDING_M:
        item = next(row for row in kit.KIT if row[0] == "block" and row[3] == 0.060)   # the block on end
        kind = "block"
    return (2 if kind == "block" else NAMES.index(f"cube{round(width * 1000)}")), item


def box_corners(found: dict, item: tuple) -> np.ndarray:
    """The 8 corners (base frame, m) of the kit piece `item` where `found` puts it: centred there on
    the table, its width along the find's short axis."""
    _kind, width, length, height = item
    a = math.radians(found["angle"])
    across, along = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    centre = np.array([found["x"], found["y"]])
    floor = -cfg.TABLE_BELOW_PLATE
    return np.array([[*(centre + i * width / 2 * across + j * length / 2 * along), floor + k * height]
                     for i in (-1, 1) for j in (-1, 1) for k in (0, 1)])


def outline(found: dict, item: tuple, record: dict, fit: dict) -> np.ndarray | None:
    """The piece's outline in the picture (pixels, its corners' hull); None when a corner is not
    well in front of the lens. A piece reaching past the picture is projected without the lens's
    distortion: its polynomial, fitted inside the picture, throws corners outside it anywhere
    (2026-10-15: long thin triangles across two pictures); the C930e's is small at the edge."""
    pose = {int(j): float(v) for j, v in record["pose"].items()}
    lens_fit = (fit["mount_m"], fit["turn_rad"], fit.get("sag_deg_per_m", (0.0, 0.0, 0.0)))
    corners = box_corners(found, item)
    R, c = lens.camera_pose(pose, *lens_fit)
    ahead = (corners - c) @ R
    if np.min(ahead[:, 2]) < MIN_AHEAD_M:
        return None
    K, (w, h) = np.asarray(record["K"], float), record["size"]
    plain = ahead[:, :2] / ahead[:, 2:] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    within = np.all((plain >= 0) & (plain < [w, h]))
    pixels = lens.project(corners, pose, K, record["dist"] if within else np.zeros(5), *lens_fit)
    return cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2)


def survey_pieces(records: list[dict]) -> list[tuple[dict, int, tuple]]:
    """The kit pieces one survey saw whole in any of its looks: (find, class, KIT row) each. Finds
    within SAME_M of each other are one piece -- several rungs (colour, raised, the open-set
    detector) and several looks find the same one -- the surest standing for it."""
    pieces: list[list] = []
    for record in records:
        for found in record["found"]:
            named = None if found["clipped"] else kit_class(found)
            if named is None:
                continue
            same = next((p for p in pieces if math.hypot(found["x"] - p[0]["x"], found["y"] - p[0]["y"]) < SAME_M), None)
            if same is None:
                pieces.append([found, *named])
            elif found["confidence"] > same[0]["confidence"]:
                same[:] = [found, *named]
    blocks = [p[0] for p in pieces if p[1] == NAMES.index("block")]
    return [tuple(p) for p in pieces if p[1] == NAMES.index("block")   # a lying block's end read as a cube
            or not any(on_block(p[0], b) for b in blocks)]


def on_block(found: dict, block: dict) -> bool:
    """Whether `found`'s centre lies on the lying block `block`'s footprint (30 x 60 mm)."""
    a = math.radians(block["angle"])
    d = np.array([found["x"] - block["x"], found["y"] - block["y"]])
    across, along = abs(d @ [math.cos(a), math.sin(a)]), abs(d @ [-math.sin(a), math.cos(a)])
    return across <= 0.015 + kit.WIDTH_SLACK_M and along <= 0.030 + kit.WIDTH_SLACK_M


def labelled(record: dict, pieces: list[tuple[dict, int, tuple]], fit: dict) -> list[tuple[int, np.ndarray]]:
    """[(class, outline)] for one look: every piece its survey saw, wherever it falls in this
    picture -- whole, or cut by the edge (the outline then ends at the edge), or missed by this
    look's own ladder. A sliver at the edge is left out."""
    record = {**record, "dist": record.get("dist") or [0.0] * 5}   # the first looks (2026-10-06) kept none
    w, h = record["size"]
    frame = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    out = []
    for found, c, item in pieces:
        poly = outline(found, item, record, fit)
        if poly is None:
            continue
        inside, part = cv2.intersectConvexConvex(poly.astype(np.float32), frame)
        if part is not None and inside >= MIN_INSIDE * cv2.contourArea(poly):
            out.append((c, part.reshape(-1, 2)))
    return out


def stamp(stem: str) -> float:
    return time.mktime(time.strptime(stem[:15], "%Y%m%d_%H%M%S")) + int(stem[16:19]) / 1000


def groups(stems: list[str], gap_s: float) -> list[list[str]]:
    """The looks in runs, a new one wherever `gap_s` or more passes between two."""
    out, last = [], None
    for stem in sorted(stems):
        t = stamp(stem)
        if last is None or t - last >= gap_s:
            out.append([])
        out[-1].append(stem)
        last = t
    return out


def build(into: Path, val_every: int, draw: int, looks: Path = LOOKS) -> None:
    if not (into / "kit.yaml").exists():
        raise SystemExit(f"no set at {into}: m3pro/tools/kit_dataset.py writes it first")
    fit = json.loads((ROOT / "data" / "cameras" / cfg.WRIST_CAMERA / "lens_on_arm.json").read_text())
    check = ROOT / "out" / "kit_x3_check"
    if draw:
        shutil.rmtree(check, ignore_errors=True)
        check.mkdir(parents=True)
    stems = sorted(p.stem for p in looks.glob("*.json") if (looks / f"{p.stem}.jpg").exists())
    counts = {"train": [0, 0, [0] * len(NAMES)], "val": [0, 0, [0] * len(NAMES)]}
    left_out, kept = 0, []
    for i, scene in enumerate(groups(stems, SCENE_GAP_S)):
        split = "val" if i % val_every == val_every - 1 else "train"
        for survey in groups(scene, SURVEY_GAP_S):
            records = {stem: json.loads((looks / f"{stem}.json").read_text()) for stem in survey}
            left_out += sum(r.get("K") is None for r in records.values())
            records = {stem: r for stem, r in records.items() if r.get("K") is not None}
            seen = survey_pieces(list(records.values()))
            for stem, record in records.items():
                write(into, split, looks / f"{stem}.jpg", record, labelled(record, seen, fit), counts, kept)
    for split, (n, empty, per) in counts.items():
        print(f"X3 {split}: {n} pictures ({empty} with no kit piece), "
              + ", ".join(f"{name} {k}" for name, k in zip(NAMES, per)))
    print(f"{left_out} looks left out (no camera model) -> {into}")
    if draw:   # spread over all the days: two thirds with pieces, a third without
        full, empty = [k for k in kept if k[1]], [k for k in kept if not k[1]]
        chosen = [g[int(i)] for g, n in ((full, draw - draw // 3), (empty, draw // 3)) if g
                  for i in np.linspace(0, len(g) - 1, min(n, len(g)))]
        for stem, pieces in chosen:
            image = cv2.imread(str(looks / f"{stem}.jpg"))
            for c, poly in pieces:
                cv2.polylines(image, [poly.astype(np.int32)], True, (0, 255, 255), 2)
                cv2.putText(image, NAMES[c], tuple(int(v) for v in poly.min(axis=0)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 2)
            cv2.imwrite(str(check / f"{stem}.jpg"), image)
        print(f"{len(chosen)} drawn in {check}")


def write(into: Path, split: str, picture: Path, record: dict, pieces: list, counts: dict, kept: list) -> None:
    """One look and its labels into the set, counted."""
    stem = picture.stem
    w, h = record["size"]
    lines = [f"{c} " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in poly) for c, poly in pieces]
    shutil.copy(picture, into / "images" / split / f"x3_{stem}.jpg")
    (into / "labels" / split / f"x3_{stem}.txt").write_text("".join(f"{line}\n" for line in lines))
    counts[split][0] += 1
    counts[split][1] += 0 if lines else 1
    for c, _poly in pieces:
        counts[split][2][c] += 1
    kept.append((stem, pieces))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    b = sub.add_parser("build")
    b.add_argument("--into", type=Path, default=INTO)
    b.add_argument("--val-every", type=int, default=5)
    b.add_argument("--draw", type=int, default=0)
    b.add_argument("--looks", type=Path, default=LOOKS, help=f"the looks to label (the test set: {HELD_OUT})")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch()
    else:
        build(a.into, a.val_every, a.draw, a.looks)


def fetch() -> None:
    """The X3's looks newer than the newest here, as one tar stream over SSH (scp -r of its first
    1,800 small files over WiFi took ~8 min, 2026-10-15)."""
    LOOKS.mkdir(parents=True, exist_ok=True)
    newest = max((p.name for folder in (LOOKS, HELD_OUT) if folder.exists() for p in folder.iterdir()), default="")
    with subprocess.Popen(["ssh", "-o", "BatchMode=yes", HOST,
                           f"cd {REMOTE} && ls | awk -v s='{newest}' '$0 > s' | tar cf - -T -"],
                          stdout=subprocess.PIPE) as proc, tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        tar.extractall(LOOKS, filter="data")
    print(f"{len(list(LOOKS.glob('*.jpg')))} looks in {LOOKS}")


if __name__ == "__main__":
    main()
