"""Synthetic pictures of the kit for its detector (PLAN.md M5): rendered in Isaac Sim 6.0 on hekate,
inside the running streaming instance -- sent there from the laptop, its globals kept:

    python tools/isaac_exec.py --via hekate train/synth/kit_synth.py           # defines generate()
    python tools/isaac_exec.py --via hekate -c "await generate(1500, seed=1)"  # renders, ~0.15 s each

(A call of several minutes loses its reply over the SSH tunnel -- exit 3, "no answer" -- while the
render goes on inside Isaac Sim: count OUT's instance_segmentation_semantics_mapping_*.json instead.)
Or headless without a running sim, as on Dionysos's Isaac Sim 5.0: train/synth/run_standalone.py.

Each frame: a floor, 1-6 kit pieces (the 30 and 40 mm cubes, the 30x30x60 block lying or standing)
placed apart, 0-3 things that are not the kit (tape rolls, cans, boxes, balls, the yellow cylinder: no
label), the light, and the M3's camera (fx 480.8 at 640 x 480: m3pro's K) where the pick's wide and
close looks put it (LOOKS), turned any way round. Written by Replicator's BasicWriter (rgb + instance
segmentation) to OUT; train/synth/synth_to_yolo.py turns that into the real set's labels. Why: the
real set has few pictures of the 40 mm cube (19 on 2026-10-12), and synthetic pieces cost nothing.

The floor is the arena mat (grey or green, yellow and white lines) or, since 2026-10-19, the lab's
table as the M3 sees it today (TABLE_SHARE): white paper printed faintly with a tag board
(textures/paper.png, made to the photos) on a grey desk, black tape at its edges, a power strip and
cables about, brighter light. The 2026-10-19 set of the mat alone did not help on the table at all.
And the pieces are the lab's own (LAB_SHARE, the rest one colour as before), their faces cut from the
user's photos into textures/: the four 30 mm picture cubes (yellow, green, red, blue paint; pictures
pic_COLOUR_*.png; some faces an AprilTag, tag_*.png), the red block with a tag on its end and the blue
one, and the one 40 mm cube -- bare end grain on two opposite faces (wood_*.png), its sides painted
yellow, green, red and blue: the M3's detector missed it wood up and called it a block side up, having
only seen one-colour cubes here. The yellow cylinder, which both robots' depth names a 30 mm cube,
stands about unlabelled. Each picture comes after FLUSH_UPDATES renders dropped and RT_SUBFRAMES
passes: in one, the previous picture's pieces showed through as ghosts (Isaac Sim 5.0, Dionysos).
"""

import math
import os

import carb
import numpy as np
import omni.kit.app
import omni.kit.commands
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

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
# its folder: this file's, or hekate's copy when isaac_exec.py runs the file's text in the sim (no __file__)
HERE = (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
        else os.path.join(os.path.expanduser("~"), "Desktop", "Sim Arm 2026", "train", "synth"))
TEXTURES = os.path.join(HERE, "textures")
WOOD = [os.path.join(TEXTURES, f"wood_{x}.png") for x in "ab"]
SIDES = [(0.80, 0.60, 0.26), (0.15, 0.45, 0.22), (0.85, 0.26, 0.30), (0.12, 0.42, 0.62)]   # round it: +x +y -x -y
PAINTED_SHARE = 0.7   # of the 40 mm cubes the lab's painted one; the rest one colour (the X3's tagged cubes)
WOOD_UP = 0.6         # it lies wood up mostly; else a painted side up
UPS = [(0, 0), (180, 0)], [(0, -90), (0, 90), (90, 0), (-90, 0)]   # X, Y turns: +-z up / a side up
# The lab's 30 mm picture cubes and blocks (the user's photos, 2026-10-19): paint as measured on them
CUBE_PAINT = {"yellow": (0.85, 0.72, 0.15), "green": (0.23, 0.46, 0.23), "red": (0.72, 0.16, 0.19),
              "blue": (0.22, 0.41, 0.70)}
BLOCK_PAINT = {"red": (0.69, 0.13, 0.15), "blue": (0.18, 0.38, 0.66)}
LAB_SHARE = 0.8       # of the 30 mm cubes and blocks the lab's own; the rest one colour as before
TAG_SHARE = 0.4       # of the picture cubes one with a tag on a painted face
LAB_POOL = {"cube30": 6, "block": 4}
TABLE_SHARE = 0.6     # of the frames the lab table; the rest the arena mat
TABLE = (0.74, 0.74, 0.72)
PAPER_M = 0.6         # m, the side textures/paper.png covers
DOME_TABLE, SUN_TABLE = (150.0, 450.0), (300.0, 1500.0)
RT_SUBFRAMES = 8
FLUSH_UPDATES = 30   # frames rendered and dropped after the pieces move: the history goes with them
# A box of side 1, a face each: (normal, u, v) -- +z and -z, then +x +y -x -y round it
FACES = [((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
         ((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
         ((-1, 0, 0), (0, -1, 0), (0, 0, 1)), ((0, -1, 0), (1, 0, 0), (0, 0, 1))]


def _colour(prim, rgb):
    UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set([Gf.Vec3f(*[float(c) for c in rgb])])


def _jitter(colour, rng, lo=0.8, hi=1.15):
    return Gf.Vec3f(*np.clip(np.array(colour) * rng.uniform(lo, hi), 0, 1))


def _material(stage, name, texture=None, colour=None):
    """An OmniPBR material at /World/Looks/name: (the material, its shader)."""
    path = f"/World/Looks/{name}"
    omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url="OmniPBR.mdl", mtl_name="OmniPBR", mtl_path=path)
    shader = UsdShade.Shader(stage.GetPrimAtPath(f"{path}/Shader"))
    if texture is not None:
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(colour or (1, 1, 1))))
    shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.8)
    return UsdShade.Material(stage.GetPrimAtPath(path)), shader


def _mesh(stage, name, faces, materials, semantic=None):
    """A mesh of side 1 (scaled when placed) from `faces` ((normal, u, v) each), every face its own
    material subset: (the prim, the subsets -- bound again per frame by _bind)."""
    mesh = UsdGeom.Mesh.Define(stage, f"/World/{name}")
    points, normals, st = [], [], []
    for n, u, v in faces:
        n, u, v = np.array(n, float), np.array(u, float), np.array(v, float)
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):   # counter-clockwise seen from outside
            points.append(Gf.Vec3f(*(0.5 * (n + su * u + sv * v))))
            normals.append(Gf.Vec3f(*n))
            st.append(Gf.Vec2f((su + 1) / 2, (sv + 1) / 2))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(list(range(4 * len(faces))))
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                            UsdGeom.Tokens.faceVarying).Set(st)
    subsets = [UsdGeom.Subset.CreateGeomSubset(mesh, f"face{i}", UsdGeom.Tokens.face, [i],
                                               familyName=UsdShade.Tokens.materialBind).GetPrim()
               for i in range(len(faces))]
    _bind(subsets, materials)
    prim = mesh.GetPrim()
    if semantic is not None:
        rep.functional.modify.semantics(prim, {"class": semantic})
    rep.functional.modify.pose(prim, position_value=FAR)
    return prim, subsets


def _bind(subsets, materials):
    for subset, material in zip(subsets, materials):
        UsdShade.MaterialBindingAPI.Apply(subset).Bind(material)


def _distant_light(stage, intensity):
    """The sun: Replicator's own on Isaac Sim 6.0 (hekate); 5.0's functional API has none (Dionysos,
    2026-10-19), so a plain USD one there."""
    create = getattr(rep.functional.create, "distant_light", None)
    if create is not None:
        return create(intensity=intensity, parent="/World", name="Sun")
    light = UsdLux.DistantLight.Define(stage, "/World/Sun")
    light.CreateIntensityAttr(intensity)
    return light.GetPrim()


def _textures(prefix):
    return sorted(os.path.join(TEXTURES, f) for f in os.listdir(TEXTURES) if f.startswith(prefix))


def _setup():
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    rep.orchestrator.set_capture_on_play(False)
    # FXAA keeps no history: DLSS's left the last picture's pieces in as ghosts (Isaac Sim 5.0, 2026-10-19)
    carb.settings.get_settings().set_int("/rtx/post/aa/op", 2)
    rep.functional.create.xform(name="World")
    scene = {"dome": rep.functional.create.dome_light(intensity=800, parent="/World", name="Dome"),
             "sun": _distant_light(stage, 2000),
             "floor": rep.functional.create.plane(scale=(4, 4, 1), parent="/World", name="Floor"),
             "lines": [rep.functional.create.cube(position=FAR, parent="/World", name=f"Line{i}") for i in range(4)],
             "kit": {k: [rep.functional.create.cube(position=FAR, parent="/World", name=f"{k}_{i}",
                                                    semantics={"class": k}) for i in range(n)]
                     for k, n in POOL.items()},
             "others": [rep.functional.create.cylinder(position=FAR, parent="/World", name="YellowCyl")]
             + [rep.functional.create.cylinder(position=FAR, parent="/World", name=f"Roll{i}") for i in range(2)]
             + [rep.functional.create.cylinder(position=FAR, parent="/World", name="Can")]
             + [rep.functional.create.cube(position=FAR, parent="/World", name=f"Box{i}") for i in range(2)]
             + [rep.functional.create.sphere(position=FAR, parent="/World", name="Ball")],
             # the lab table's: tape at the paper's edges, a power strip, cables
             "tape": [rep.functional.create.cube(position=FAR, parent="/World", name=f"Tape{i}") for i in range(4)],
             "strip": rep.functional.create.cube(position=FAR, parent="/World", name="Strip"),
             "cables": [rep.functional.create.cube(position=FAR, parent="/World", name=f"Cable{i}") for i in range(2)]}
    cam = rep.functional.create.camera(position=(0.4, 0, 0.3), look_at=(0, 0, 0), parent="/World", name="Cam",
                                       focal_length=FX * APERTURE / W)
    camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/Cam"))
    camera.GetHorizontalApertureAttr().Set(APERTURE)
    camera.GetVerticalApertureAttr().Set(APERTURE * H / W)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    scene["cam"] = cam
    woods = [_material(stage, f"Wood{i}", texture=t) for i, t in enumerate(WOOD)]
    scene["sides"] = [_material(stage, f"Side{i}", colour=c) for i, c in enumerate(SIDES)]
    scene["woods"] = woods
    scene["painted"] = [_mesh(stage, f"cube40p_{i}", FACES, [woods[i % 2][0], woods[(i + 1) % 2][0]]
                              + [m for m, _ in scene["sides"]], "cube40")[0] for i in range(3)]
    # the lab's picture cubes and blocks: a material per picture, tag and paint, bound per frame
    scene["pictures"] = {c: [_material(stage, f"Pic_{c}_{i}", texture=t)[0]
                             for i, t in enumerate(_textures(f"pic_{c}_"))] for c in CUBE_PAINT}
    scene["tags"] = [_material(stage, f"Tag{i}", texture=t)[0] for i, t in enumerate(_textures("tag_"))]
    scene["cube_paint"] = {c: _material(stage, f"CubePaint_{c}", colour=v) for c, v in CUBE_PAINT.items()}
    scene["block_paint"] = {c: _material(stage, f"BlockPaint_{c}", colour=v) for c, v in BLOCK_PAINT.items()}
    grey = scene["cube_paint"]["blue"][0]
    scene["lab"] = {k: [_mesh(stage, f"lab_{k}_{i}", FACES, [grey] * 6, k) for i in range(n)]
                    for k, n in LAB_POOL.items()}
    scene["paper_shader"] = _material(stage, "Paper", texture=os.path.join(TEXTURES, "paper.png"))
    scene["paper"] = _mesh(stage, "Paper", FACES[:1], [scene["paper_shader"][0]])[0]
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


def _lab_cube(scene, entry, spot, rng):
    """A lab picture cube: two opposite faces its paint (one a tag sometimes), four its pictures."""
    prim, subsets = entry
    colour = list(CUBE_PAINT)[int(rng.integers(len(CUBE_PAINT)))]
    paint = scene["cube_paint"][colour][0]
    caps = [paint, paint]
    if rng.random() < TAG_SHARE:
        caps[int(rng.integers(2))] = scene["tags"][int(rng.integers(len(scene["tags"])))]
    pictures = scene["pictures"][colour]
    sides = [pictures[int(i)] for i in rng.permutation(max(4, len(pictures)))[:4] % len(pictures)]
    _bind(subsets, caps + sides)
    rx, ry = (UPS[0] + UPS[1])[int(rng.integers(6))]
    rep.functional.modify.pose(prim, position_value=(*spot, 0.015), rotation_value=(rx, ry, rng.uniform(0, 360)),
                               rotation_order="XYZ", scale_value=(0.03, 0.03, 0.03))


def _lab_block(scene, entry, spot, rng):
    """A lab block, 60 mm along its x: the red one with a tag on its +x end, or the blue one."""
    prim, subsets = entry
    if rng.random() < 0.5:
        red = scene["block_paint"]["red"][0]
        tag = scene["tags"][int(rng.integers(len(scene["tags"])))]
        _bind(subsets, [red, red, tag, red, red, red])
    else:
        _bind(subsets, [scene["block_paint"]["blue"][0]] * 6)
    standing = rng.random() < 0.25
    rep.functional.modify.pose(prim, position_value=(*spot, 0.03 if standing else 0.015),
                               rotation_value=(float(rng.choice([0, 90, 180, 270])), -90.0 if standing else 0.0,
                                               rng.uniform(0, 360)),
                               rotation_order="XYZ", scale_value=(0.06, 0.03, 0.03))


def _table(scene, rng, on):
    """The lab table: paper with its tape on a grey desk, a power strip and cables -- or all away."""
    if not on:
        for prim in [scene["paper"], scene["strip"], *scene["tape"], *scene["cables"]]:
            rep.functional.modify.pose(prim, position_value=FAR)
        return
    _colour(scene["floor"], np.clip(np.array(TABLE) * rng.uniform(0.85, 1.1) + rng.normal(0, 0.01, 3), 0, 1))
    centre, yaw = rng.uniform(-0.12, 0.12, 2), rng.uniform(0, 360)
    rep.functional.modify.pose(scene["paper"], position_value=(*centre, 0.0), rotation_value=(0, 0, yaw),
                               scale_value=(PAPER_M, PAPER_M, 0.001))   # its face 0.5 mm up
    scene["paper_shader"][1].GetInput("diffuse_tint").Set(_jitter((1, 1, 1), rng, 0.9, 1.08))
    a = math.radians(yaw)
    along, across = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    for tape in scene["tape"]:   # a strip of black tape over an edge of the paper
        if rng.random() < 0.6:
            side, t = float(rng.choice([-1, 1])), rng.uniform(-0.25, 0.25)
            normal, edge = (along, across) if rng.random() < 0.5 else (across, along)
            p = centre + normal * side * PAPER_M / 2 + edge * t
            rep.functional.modify.pose(tape, position_value=(*p, 0.0006), rotation_value=(0, 0, math.degrees(
                math.atan2(normal[1], normal[0]))), scale_value=(0.05, 0.025, 0.0006))
            _colour(tape, (0.03, 0.03, 0.03))
        else:
            rep.functional.modify.pose(tape, position_value=FAR)
    for prim, size, colour in [(scene["strip"], (0.2, 0.05, 0.04), (0.18, 0.18, 0.2))] + \
            [(c, (rng.uniform(0.3, 0.7), 0.005, 0.005), (0.02, 0.02, 0.02)) for c in scene["cables"]]:
        if rng.random() < 0.4:   # beyond the paper, on the desk
            far = rng.uniform(0.2, 0.35) * np.array([math.cos(b := rng.uniform(0, 2 * math.pi)), math.sin(b)])
            rep.functional.modify.pose(prim, position_value=(*(centre + far), size[2] / 2),
                                       rotation_value=(0, 0, rng.uniform(0, 360)), scale_value=size)
            _colour(prim, colour)
        else:
            rep.functional.modify.pose(prim, position_value=FAR)


def _frame(scene, rng):
    table = rng.random() < TABLE_SHARE
    _colour(scene["floor"], np.clip(np.array(FLOORS[rng.integers(len(FLOORS))]) + rng.normal(0, 0.03, 3), 0, 1))
    for line in scene["lines"]:   # the mat's painted lines: some in view, some away -- none on the table
        if not table and rng.random() < 0.6:
            yaw = rng.uniform(0, 180)
            rep.functional.modify.pose(line, position_value=(*rng.uniform(-0.3, 0.3, 2), 0.0005),
                                       rotation_value=(0, 0, yaw), scale_value=(1.5, rng.uniform(0.015, 0.04), 0.001))
            _colour(line, LINES[rng.integers(len(LINES))])
        else:
            rep.functional.modify.pose(line, position_value=FAR)
    _table(scene, rng, table)
    for (_, shader), colour in zip(scene["sides"], SIDES):   # paint and wood under this frame's light
        shader.GetInput("diffuse_color_constant").Set(_jitter(colour, rng))
    for _, shader in scene["woods"]:
        shader.GetInput("diffuse_tint").Set(Gf.Vec3f(*np.clip(rng.uniform(0.85, 1.12) + rng.normal(0, 0.03, 3), 0, 1.2)))
    for paints, values in ((scene["cube_paint"], CUBE_PAINT), (scene["block_paint"], BLOCK_PAINT)):
        for c, (_, shader) in paints.items():
            shader.GetInput("diffuse_color_constant").Set(_jitter(values[c], rng, 0.85, 1.12))
    used = {k: 0 for k in KIT}
    lab = {k: 0 for k in LAB_POOL}
    painted = 0
    kinds = [CHOICE[rng.integers(len(CHOICE))] for _ in range(int(rng.integers(1, 7)))]
    spots = _place(rng, len(kinds) + 3, 0.16)
    for kind, spot in zip(kinds, spots):
        if kind == "cube40" and painted < len(scene["painted"]) and rng.random() < PAINTED_SHARE:
            prim = scene["painted"][painted]
            painted += 1
            ups = UPS[0] if rng.random() < WOOD_UP else UPS[1]
            rx, ry = ups[int(rng.integers(len(ups)))]
            rep.functional.modify.pose(prim, position_value=(*spot, 0.02), rotation_value=(rx, ry, rng.uniform(0, 360)),
                                       rotation_order="XYZ", scale_value=(0.04, 0.04, 0.04))
            continue
        if kind in LAB_POOL and lab[kind] < LAB_POOL[kind] and rng.random() < LAB_SHARE:
            (_lab_cube if kind == "cube30" else _lab_block)(scene, scene["lab"][kind][lab[kind]], spot, rng)
            lab[kind] += 1
            continue
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
    for kind, entries in scene["lab"].items():
        for prim, _ in entries[lab[kind]:]:
            rep.functional.modify.pose(prim, position_value=FAR)
    for prim in scene["painted"][painted:]:
        rep.functional.modify.pose(prim, position_value=FAR)
    extra = spots[len(kinds):][: int(rng.integers(0, 4))]
    for i, prim in enumerate(scene["others"]):
        if i < len(extra) and rng.random() < 0.7:
            name = prim.GetName() if hasattr(prim, "GetName") else str(prim)
            if "YellowCyl" in name:   # the lab's yellow cylinder: no kit piece, read as a 30 mm cube
                size = (0.032, 0.032, 0.033)
            elif "Roll" in name:
                size = (0.065, 0.065, rng.uniform(0.015, 0.02))
            elif "Can" in name:
                size = (0.066, 0.066, rng.uniform(0.10, 0.17))
            elif "Box" in name:
                size = (rng.uniform(0.08, 0.18), rng.uniform(0.05, 0.12), rng.uniform(0.03, 0.08))
            else:
                size = (0.05, 0.05, 0.05)
            rep.functional.modify.pose(prim, position_value=(*extra[i] * 1.6, size[2] / 2),
                                       rotation_value=(0, 0, rng.uniform(0, 360)), scale_value=size)
            _colour(prim, np.clip(np.array((0.82, 0.70, 0.10)) * rng.uniform(0.85, 1.1), 0, 1) if "YellowCyl" in name
                    else rng.uniform(0.05, 0.95, 3))
        else:
            rep.functional.modify.pose(prim, position_value=FAR)
    _, height, down = LOOKS[int(rng.random() >= WIDE_SHARE)]
    height, down = height + rng.uniform(-0.02, 0.02), math.radians(down + rng.uniform(-4, 4))
    azimuth = rng.uniform(0, 2 * math.pi)
    aim = np.r_[rng.uniform(-0.05, 0.05, 2), 0.0]   # where the optical axis meets the floor
    ahead = height / math.tan(down)
    eye = aim + np.array([-ahead * math.cos(azimuth), -ahead * math.sin(azimuth), height])
    rep.functional.modify.pose(scene["cam"], position_value=tuple(eye), look_at_value=tuple(aim))
    dome, sun = (DOME_TABLE, SUN_TABLE) if table else (DOME, SUN)
    rep.functional.modify.attribute(scene["dome"], "inputs:intensity", float(rng.uniform(*dome)))
    rep.functional.modify.attribute(scene["sun"], "inputs:intensity", float(rng.uniform(*sun)))
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
        for _ in range(FLUSH_UPDATES):   # 2026-10-19: the last picture's pieces showed through as ghosts
            await omni.kit.app.get_app().next_update_async()
        await rep.orchestrator.step_async(rt_subframes=RT_SUBFRAMES)
    await rep.orchestrator.wait_until_complete_async()
    writer.detach()
    rp.destroy()
    return f"{n} frames -> {out}"
