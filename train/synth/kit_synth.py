"""Synthetic pictures of the kit for its detector (PLAN.md M5): rendered in Isaac Sim 6.0 on hekate,
inside the running streaming instance -- sent there from the laptop, its globals kept:

    python tools/isaac_exec.py --via hekate train/synth/kit_synth.py           # defines generate()
    python tools/isaac_exec.py --via hekate -c "await generate(1500, seed=1)"  # renders, ~0.15 s each

(A call of several minutes loses its reply over the SSH tunnel -- exit 3, "no answer" -- while the
render goes on inside Isaac Sim: count OUT's instance_segmentation_semantics_mapping_*.json instead.)

Each frame: the arena mat (grey or green, yellow and white lines), 1-6 kit pieces (the 30 and 40 mm
cubes, the 30x30x60 block lying or standing; colours as the real kit's, jittered) placed apart, 0-3
things that are not the kit (tape rolls, cans, boxes, balls: no label), the light, and the M3's camera
(fx 480.8 at 640 x 480: m3pro's K) where the pick's wide and close looks put it (LOOKS), turned any way
round. Written by Replicator's BasicWriter (rgb + instance segmentation)
to OUT; train/synth/synth_to_yolo.py turns that into the real set's labels. Why: the real set has 19
pictures of the 40 mm cube in all (2026-10-12), and synthetic pieces cost nothing.
"""

import math
import os

import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, UsdGeom

OUT = os.path.join(os.path.expanduser("~"), "Desktop", "Sim Arm 2026", "train", "datasets", "synth_raw")
W, H, FX = 640, 480, 480.8           # the M3's colour camera (m3pro: K, 2026-10-12)
APERTURE = 20.955                    # mm, the horizontal aperture; the focal length gives FX
KIT = {"cube30": (0.03, 0.03, 0.03), "cube40": (0.04, 0.04, 0.04), "block": (0.06, 0.03, 0.03)}
POOL = {"cube30": 6, "cube40": 6, "block": 5}
CHOICE = (["cube30"] * 3) + (["cube40"] * 3) + (["block"] * 2)   # the 40 mm cube as often as the 30
COLOURS = [(0.72, 0.08, 0.07), (0.10, 0.33, 0.80), (0.04, 0.09, 0.35), (0.92, 0.76, 0.10), (0.10, 0.55, 0.22),
           (0.92, 0.42, 0.08), (0.88, 0.88, 0.86), (0.70, 0.55, 0.36), (0.45, 0.12, 0.55)]
FLOORS = [(0.33, 0.33, 0.38), (0.40, 0.39, 0.44), (0.27, 0.27, 0.30), (0.16, 0.40, 0.33), (0.55, 0.55, 0.57)]
LINES = [(0.92, 0.78, 0.10), (0.92, 0.92, 0.92)]
FAR = (50.0, 50.0, -5.0)             # where an unused prim waits, out of every picture
DOME, SUN = (60.0, 250.0), (0.0, 600.0)   # light: 300-1500 and 0-3000 washed the mat out to white
# The camera where the M3's own looks put it -- (name, height over the floor m, down deg): the pick's
# wide look (m3scan.LOOK_AT) and its close look over a piece. A camera anywhere from 18 to 55 cm away
# made a far 40 mm cube look like a near 30 mm one: trained on that, the 40 mm cubes went unfound
# (0 of 3, against 1 of 3 on the real pictures alone, 2026-10-12) -- seen from the robot's few poses
# a piece's size in the picture is what tells the two apart.
LOOKS = (("wide", 0.279, 69.5), ("close", 0.21, 85.0))
WIDE_SHARE = 0.6


def _colour(prim, rgb):
    UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set([Gf.Vec3f(*[float(c) for c in rgb])])


def _setup():
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    rep.orchestrator.set_capture_on_play(False)
    rep.functional.create.xform(name="World")
    scene = {"dome": rep.functional.create.dome_light(intensity=800, parent="/World", name="Dome"),
             "sun": rep.functional.create.distant_light(intensity=2000, parent="/World", name="Sun"),
             "floor": rep.functional.create.plane(scale=(4, 4, 1), parent="/World", name="Floor"),
             "lines": [rep.functional.create.cube(position=FAR, parent="/World", name=f"Line{i}") for i in range(4)],
             "kit": {k: [rep.functional.create.cube(position=FAR, parent="/World", name=f"{k}_{i}",
                                                    semantics={"class": k}) for i in range(n)]
                     for k, n in POOL.items()},
             "others": [rep.functional.create.cylinder(position=FAR, parent="/World", name=f"Roll{i}") for i in range(2)]
             + [rep.functional.create.cylinder(position=FAR, parent="/World", name="Can")]
             + [rep.functional.create.cube(position=FAR, parent="/World", name=f"Box{i}") for i in range(2)]
             + [rep.functional.create.sphere(position=FAR, parent="/World", name="Ball")]}
    cam = rep.functional.create.camera(position=(0.4, 0, 0.3), look_at=(0, 0, 0), parent="/World", name="Cam",
                                       focal_length=FX * APERTURE / W)
    camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/Cam"))
    camera.GetHorizontalApertureAttr().Set(APERTURE)
    camera.GetVerticalApertureAttr().Set(APERTURE * H / W)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    scene["cam"] = cam
    return scene


def _place(rng, n, near):
    """`n` floor points within `near` m of the middle, at least 6 cm apart (fewer when they do not fit)."""
    out = []
    for _ in range(200):
        if len(out) == n:
            break
        p = rng.uniform(-near, near, 2)
        if all(np.hypot(*(p - q)) >= 0.06 for q in out):
            out.append(p)
    return out


def _frame(scene, rng):
    _colour(scene["floor"], np.clip(np.array(FLOORS[rng.integers(len(FLOORS))]) + rng.normal(0, 0.03, 3), 0, 1))
    for line in scene["lines"]:   # the mat's painted lines: some in view, some away
        if rng.random() < 0.6:
            yaw = rng.uniform(0, 180)
            rep.functional.modify.pose(line, position_value=(*rng.uniform(-0.3, 0.3, 2), 0.0005),
                                       rotation_value=(0, 0, yaw), scale_value=(1.5, rng.uniform(0.015, 0.04), 0.001))
            _colour(line, LINES[rng.integers(len(LINES))])
        else:
            rep.functional.modify.pose(line, position_value=FAR)
    used = {k: 0 for k in KIT}
    kinds = [CHOICE[rng.integers(len(CHOICE))] for _ in range(int(rng.integers(1, 7)))]
    spots = _place(rng, len(kinds) + 3, 0.16)
    for kind, spot in zip(kinds, spots):
        prim = scene["kit"][kind][used[kind]] if used[kind] < POOL[kind] else None
        if prim is None:
            continue
        used[kind] += 1
        size = np.array(KIT[kind])
        if kind == "block" and rng.random() < 0.25:   # standing on its end
            size = size[[1, 2, 0]]
        rep.functional.modify.pose(prim, position_value=(*spot, size[2] / 2), rotation_value=(0, 0, rng.uniform(0, 360)),
                                   scale_value=tuple(size))
        _colour(prim, np.clip(np.array(COLOURS[rng.integers(len(COLOURS))]) * rng.uniform(0.8, 1.15), 0, 1))
    for kind, prims in scene["kit"].items():
        for prim in prims[used[kind]:]:
            rep.functional.modify.pose(prim, position_value=FAR)
    extra = spots[len(kinds):][: int(rng.integers(0, 4))]
    for i, prim in enumerate(scene["others"]):
        if i < len(extra) and rng.random() < 0.7:
            name = prim.GetName() if hasattr(prim, "GetName") else str(prim)
            if "Roll" in name:
                size = (0.065, 0.065, rng.uniform(0.015, 0.02))
            elif "Can" in name:
                size = (0.066, 0.066, rng.uniform(0.10, 0.17))
            elif "Box" in name:
                size = (rng.uniform(0.08, 0.18), rng.uniform(0.05, 0.12), rng.uniform(0.03, 0.08))
            else:
                size = (0.05, 0.05, 0.05)
            rep.functional.modify.pose(prim, position_value=(*extra[i] * 1.6, size[2] / 2),
                                       rotation_value=(0, 0, rng.uniform(0, 360)), scale_value=size)
            _colour(prim, rng.uniform(0.05, 0.95, 3))
        else:
            rep.functional.modify.pose(prim, position_value=FAR)
    _, height, down = LOOKS[int(rng.random() >= WIDE_SHARE)]
    height, down = height + rng.uniform(-0.02, 0.02), math.radians(down + rng.uniform(-4, 4))
    azimuth = rng.uniform(0, 2 * math.pi)
    aim = np.r_[rng.uniform(-0.05, 0.05, 2), 0.0]   # where the optical axis meets the floor
    ahead = height / math.tan(down)
    eye = aim + np.array([-ahead * math.cos(azimuth), -ahead * math.sin(azimuth), height])
    rep.functional.modify.pose(scene["cam"], position_value=tuple(eye), look_at_value=tuple(aim))
    rep.functional.modify.attribute(scene["dome"], "inputs:intensity", float(rng.uniform(DOME[0], DOME[1])))
    rep.functional.modify.attribute(scene["sun"], "inputs:intensity", float(rng.uniform(SUN[0], SUN[1])))
    rep.functional.modify.pose(scene["sun"], rotation_value=(rng.uniform(-50, 50), rng.uniform(-50, 50), 0))


async def generate(n, seed=0, out=OUT):
    """Render `n` pictures with their instance masks into `out` (BasicWriter's files)."""
    rng = np.random.default_rng(seed)
    scene = _setup()
    rp = rep.create.render_product(scene["cam"], (W, H), name="M3")
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=out)
    writer = rep.writers.get("BasicWriter")
    writer.initialize(backend=backend, rgb=True, instance_segmentation=True)
    writer.attach(rp)
    for _ in range(n):
        _frame(scene, rng)
        await rep.orchestrator.step_async()
    await rep.orchestrator.wait_until_complete_async()
    writer.detach()
    rp.destroy()
    return f"{n} frames -> {out}"
