"""The M3 Pro's arm in Isaac Sim (NEXT.md, "After the event": a VLA taught in the simulator), run by
Isaac Sim's own Python, headless (Dionysos: 5.0; hekate: 6.0):

    python.bat isaac_m3.py import DESC OUT.usd
        Yahboom's description (DESC/urdf/M3Pro.urdf + DESC/meshes, from the robot's
        yahboomcar_ws/src/yahboom_M3Pro_description) -> a USD: the base fixed, the arm's five joints
        and the gripper's driven finger joint as position drives, the gripper's linkage as mimic joints
    python.bat isaac_m3.py play OUT.usd PLAN.json RESULT.json
        a table at the base's floor (base_link z 0), the plan's pieces as rigid bodies, the plan's
        joint targets (URDF radians at times, held in between as the board sweeps: straight lines)
        played at PHYSICS_HZ; every SAMPLE_S each piece's pose, the joints' positions and the
        fingertips' -> RESULT.json

The laptop's side -- the plan from the picker's own geometry, servo degrees to URDF radians, the
gripper's measured opening -- is m3pro/tools/m3_sim.py, which also copies this over and runs it.
"""

import argparse
import contextlib
import itertools
import json
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
sub = ap.add_subparsers(dest="cmd", required=True)
imp = sub.add_parser("import")
imp.add_argument("desc")
imp.add_argument("usd")
play = sub.add_parser("play")
play.add_argument("usd")
play.add_argument("plan")
play.add_argument("result")
a = ap.parse_args()

ARM = ["arm1_Joint", "arm2_Joint", "arm3_Joint", "arm4_Joiint", "arm5_Joint"]   # Yahboom's spelling
FINGER = "rlink1_Joint"                     # the servo's: the rest of the linkage follows it
# the linkage's joints and their share of the finger's angle (the URDF's mimic multipliers): each step driven,
# stiffly, to its share of where the finger IS -- rigid as the real gears and four-bar are. Tried and dropped
# (2026-10-23): PhysX's mimic joints (the importer left the left side's without a reference joint, and set
# by hand they did not hold: the links hung free), and shares of the finger's TARGET (the finger links turned
# on past where the cube stopped the finger: the pads wedged the cube out, down).
LINKAGE = {"rlink2_Joint": 1.0, "llink1_Joint": 1.0, "llink2_Joint": -1.0, "rlink3_Joint": -1.0, "llink3_Joint": 1.0}
LINK_DRIVE = (200.0, 2.0, 50.0)
FINGER_RANGE = (-1.0, 0.6)                  # rad: the URDF's -0.95..0 opens 49.7 mm; the real one 61.5
PHYSICS_HZ = 120
SAMPLE_S = 0.1
# the drives (USD angular drives: N m per degree, N m s per degree, N m): stiff as a servo holding, with
# the real ones' strength -- the arm's ~1.5 N m, the gripper's less, so a shut finger cannot crush
ARM_DRIVE = (40.0, 2.0, 3.0)
FINGER_DRIVE = (4.0, 0.2, 1.5)
ARM_MAX_DPS = 150.0   # a servo's top speed: without it a target far ahead flung what was held (2026-10-23)
TABLE_GREY = (0.74, 0.74, 0.72)   # the lab's desk round the paper (kit_synth.py's TABLE)
PAPER_M = 0.6                     # m, the side textures/paper.png covers
WARMUP_RENDERS = 60   # renders before the first picture: the first ones came out black while the materials'
                      # shaders compiled (2026-10-26; kit_synth.py drops 30 after every move)
RECORD_SUBFRAMES = 2   # path-tracing passes a recorded picture (a moving arm; 8 for the still looks)
JPEG_QUALITY = 90      # as roboarm.m3record keeps the real ones

from isaacsim import SimulationApp  # before any omni import: the app loads them

began = time.perf_counter()
app = SimulationApp({"headless": True})
print(f"Isaac Sim up in {time.perf_counter() - began:.0f} s", flush=True)

import omni.kit.commands  # the app up first
import omni.usd
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade


def sim_urdf(desc: str) -> str:
    """Yahboom's URDF made importable: its meshes by absolute path (it names them by two packages, one
    misspelt), the wheels' and the camera mount's zero-range joints fixed, the finger's range the real one's."""
    tree = ET.parse(os.path.join(desc, "urdf", "M3Pro.urdf"))
    meshes = os.path.join(desc, "meshes").replace("\\", "/")
    for mesh in tree.getroot().iter("mesh"):
        mesh.set("filename", meshes + "/" + re.sub(r".*/meshes/", "", mesh.get("filename")))
    for joint in tree.getroot().iter("joint"):
        limit = joint.find("limit")
        if joint.get("type") == "revolute" and limit is not None and float(limit.get("lower")) == float(limit.get("upper")):
            joint.set("type", "fixed")
        if joint.get("name") == FINGER:
            limit.set("lower", str(FINGER_RANGE[0]))
            limit.set("upper", str(FINGER_RANGE[1]))
        if joint.find("mimic") is not None and joint.get("type") == "continuous":
            # PhysX's mimic joints need finite limits: the linkage's joints follow the finger, +-1 x its range
            joint.set("type", "revolute")
            ET.SubElement(joint, "limit", lower="-1.2", upper="1.2", effort="100", velocity="10")
    out = os.path.join(desc, "urdf", "M3Pro_sim.urdf")
    tree.write(out)
    return out


def set_drive(joint: Usd.Prim, stiffness: float, damping: float, force: float) -> None:
    drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(damping)
    drive.CreateMaxForceAttr(force)


def do_import(desc: str, usd: str) -> None:
    from isaacsim.asset.importer.urdf._urdf import UrdfJointTargetType

    _ok, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.merge_fixed_joints = False
    cfg.convex_decomp = False             # a convex hull a link: the finger pads stay flat
    cfg.fix_base = True                   # table-top only: the base never moves
    cfg.import_inertia_tensor = True
    cfg.distance_scale = 1.0
    cfg.parse_mimic = True
    cfg.make_default_prim = True
    cfg.self_collision = False
    cfg.create_physics_scene = True
    cfg.default_drive_type = UrdfJointTargetType.JOINT_DRIVE_POSITION
    usd = os.path.abspath(usd).replace("\\", "/")
    _ok, path = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=sim_urdf(desc),
                                          import_config=cfg, dest_path=usd)
    print("imported at", path, flush=True)
    stage = Usd.Stage.Open(usd)
    driven, mimics = [], []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        name = prim.GetName()
        if name in ARM:
            set_drive(prim, *ARM_DRIVE)
            PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(ARM_MAX_DPS)
            driven.append(name)
        elif name == FINGER:
            set_drive(prim, *FINGER_DRIVE)
            driven.append(name)
        else:   # the linkage's: driven stiffly after the finger (LINKAGE), no mimic
            for schema in prim.GetAppliedSchemas():
                if schema.startswith("PhysxMimicJointAPI"):
                    prim.RemoveAppliedSchema(schema)
            set_drive(prim, *LINK_DRIVE)
            joint = UsdPhysics.RevoluteJoint(prim)
            lower, upper = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
            if lower is None or upper is None or not (abs(lower) < 1e6 and abs(upper) < 1e6):
                joint.CreateLowerLimitAttr(-70.0)   # degrees, as USD keeps them
                joint.CreateUpperLimitAttr(70.0)
            mimics.append((name, LINKAGE[name]))
    stage.Save()
    print("driven:", driven, flush=True)
    print("linkage (name, share of the finger):", mimics, flush=True)


# A box of side 1, a face each: (normal, u, v) -- +z and -z, then +x +y -x -y round it (kit_synth.py's)
FACES = [((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
         ((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
         ((-1, 0, 0), (0, -1, 0), (0, 0, 1)), ((0, -1, 0), (1, 0, 0), (0, 0, 1))]


def look_material(stage, path: str, texture: str | None = None, colour=None):
    """An OmniPBR material (a picture, or a colour), matte as paper and paint."""
    from pxr import Sdf

    omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url="OmniPBR.mdl", mtl_name="OmniPBR", mtl_path=path)
    shader = UsdShade.Shader(stage.GetPrimAtPath(f"{path}/Shader"))
    if texture is not None:
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(colour or (1, 1, 1))))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.8)
    return UsdShade.Material(stage.GetPrimAtPath(path))


def box_mesh(stage, path: str, faces: list, looks: list):
    """A mesh of side 1 from `faces`, each face its own `looks` material (a subset each), its st from 0 to 1."""
    from pxr import Sdf

    mesh = UsdGeom.Mesh.Define(stage, path)
    points, normals, st = [], [], []
    for n, u, v in faces:
        n, u, v = (Gf.Vec3f(*x) for x in (n, u, v))
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):   # counter-clockwise seen from outside
            points.append(0.5 * (n + su * u + sv * v))
            normals.append(n)
            st.append(Gf.Vec2f((su + 1) / 2, (sv + 1) / 2))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(list(range(4 * len(faces))))
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                            UsdGeom.Tokens.faceVarying).Set(st)
    for i, look in enumerate(looks):
        subset = UsdGeom.Subset.CreateGeomSubset(mesh, f"face{i}", UsdGeom.Tokens.face, [i],
                                                 familyName=UsdShade.Tokens.materialBind)
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(look)
    return mesh


def lab_table(stage, table: dict) -> None:
    """The lab's table as the M3 sees it (as train/synth/kit_synth.py renders it for the detector): white paper
    printed faintly with a tag board (textures/paper.png, PAPER_M a side) on a grey desk, black tape at its far
    corners, the lab's light -- the ground plane's own look hidden (it still carries the pieces). `table`:
    {"textures": the folder, "paper": [x, y, yaw deg], "paper_texture": its file there (paper.png),
    "dome": intensity, "sun": intensity, "sun_from": the direction the sun shines from (x, y, z)}."""
    from pxr import UsdLux

    UsdGeom.Imageable(stage.GetPrimAtPath("/World/defaultGroundPlane")).MakeInvisible()
    desk = UsdGeom.Cube.Define(stage, "/World/desk")   # looks only: no collider
    desk.CreateSizeAttr(1.0)
    UsdGeom.Xformable(desk).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.0006))
    UsdGeom.Xformable(desk).AddScaleOp().Set(Gf.Vec3f(4.0, 4.0, 0.001))
    desk.CreateDisplayColorAttr([Gf.Vec3f(*TABLE_GREY)])
    x, y, yaw = table["paper"]
    paper = box_mesh(stage, "/World/paper", FACES[:1],
                     [look_material(stage, "/World/Looks/paper",
                                    os.path.join(table["textures"], table.get("paper_texture", "paper.png")))])
    xf = UsdGeom.Xformable(paper)
    xf.AddTranslateOp().Set(Gf.Vec3d(x, y, -0.0005))   # its face at 0: under the pieces' feet
    xf.AddRotateZOp().Set(float(yaw))
    xf.AddScaleOp().Set(Gf.Vec3f(PAPER_M, PAPER_M, 0.001))
    a = math.radians(yaw)
    for i, side in enumerate((-1, 1)):   # tape over the paper's far corners, as on the lab's table
        cx = x + math.cos(a) * PAPER_M / 2 - math.sin(a) * side * (PAPER_M / 2 - 0.02)
        cy = y + math.sin(a) * PAPER_M / 2 + math.cos(a) * side * (PAPER_M / 2 - 0.02)
        tape = UsdGeom.Cube.Define(stage, f"/World/tape{i}")
        tape.CreateSizeAttr(1.0)
        UsdGeom.Xformable(tape).AddTranslateOp().Set(Gf.Vec3d(cx, cy, 0.0003))
        UsdGeom.Xformable(tape).AddRotateZOp().Set(float(yaw))
        UsdGeom.Xformable(tape).AddScaleOp().Set(Gf.Vec3f(0.05, 0.025, 0.0006))
        tape.CreateDisplayColorAttr([Gf.Vec3f(0.03, 0.03, 0.03)])
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(float(table.get("dome", 300.0)))
    sun = UsdLux.DistantLight.Define(stage, "/World/sun")
    sun.CreateIntensityAttr(float(table.get("sun", 900.0)))
    shine = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*table.get("sun_from", (0.5, 0.4, 0.8))), Gf.Vec3d(0, 0, 0),
                                    Gf.Vec3d(0, 0, 1))   # a distant light shines down its -z
    UsdGeom.Xformable(sun).AddTransformOp().Set(shine.GetInverse())


def add_box(stage, path: str, size, centre, yaw_deg: float, mass: float, colour, material, faces=None) -> None:
    """A piece: one colour, or `faces` -- six looks ({"texture": file} or {"colour": rgb}; +z -z +x +y -x -y)."""
    if faces:
        cube = box_mesh(stage, path, FACES, [look_material(stage, f"{path}_look{i}", f.get("texture"), f.get("colour"))
                                             for i, f in enumerate(faces)])
    else:
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*centre))
    xf.AddRotateZOp().Set(float(yaw_deg))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    if faces:
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    PhysxSchema.PhysxCollisionAPI.Apply(prim).CreateContactOffsetAttr(0.002)
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)   # held between two fingers: more solver passes
    body.CreateSolverPositionIterationCountAttr(16)
    body.CreateSolverVelocityIterationCountAttr(4)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")


def material(stage, path: str, static: float, dynamic: float):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    return mat


def targets_at(keys: list[dict], t: float) -> dict:
    """The joint targets at time t: straight lines between the plan's keys, the last held."""
    if t <= keys[0]["t"]:
        return keys[0]["q"]
    for k0, k1 in itertools.pairwise(keys):
        if t <= k1["t"]:
            f = (t - k0["t"]) / max(k1["t"] - k0["t"], 1e-9)
            return {j: k0["q"][j] + f * (k1["q"][j] - k0["q"][j]) for j in k1["q"]}
    return keys[-1]["q"]


def do_play(usd: str, plan_path: str, result_path: str) -> None:
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
    from isaacsim.core.utils.types import ArticulationAction

    with open(plan_path) as f:
        plan = json.load(f)
    world = World(physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / 30, stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    world.scene.add_default_ground_plane(z_position=0.0, static_friction=0.6, dynamic_friction=0.5)
    if plan.get("table"):
        lab_table(stage, plan["table"])
    robot_prim = stage.DefinePrim("/World/m3", "Xform")
    robot_prim.GetReferences().AddReference(os.path.abspath(usd).replace("\\", "/"))
    if plan.get("finger_drive"):   # the finger's drive for this run, over the imported one (m3_sim.FINGER_SQUEEZE)
        for prim in stage.Traverse():
            if prim.GetName() == FINGER and prim.IsA(UsdPhysics.RevoluteJoint):
                set_drive(prim, *plan["finger_drive"])
                print("finger drive:", plan["finger_drive"], flush=True)
    rubber = material(stage, "/World/materials/rubber", 1.2, 1.0)
    wood = material(stage, "/World/materials/wood", plan.get("friction", 0.6), plan.get("friction", 0.6) * 0.8)
    for prim in stage.Traverse():   # the fingertips' pads: rubber
        if prim.GetName() in ("rlink2", "llink2"):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(rubber, UsdShade.Tokens.strongerThanDescendants, "physics")
    for i, piece in enumerate(plan["pieces"]):
        add_box(stage, f"/World/piece{i}", piece["size"], piece["centre"], piece.get("yaw", 0.0),
                piece.get("mass", 0.015), piece.get("colour", (0.9, 0.3, 0.2)), wood, piece.get("faces"))
    for i, part in enumerate(plan.get("statics", [])):   # the drop-off box's floor, walls and lid
        add_static(stage, f"/World/static{i}", part)
    scene = UsdPhysics.Scene.Get(stage, "/physicsScene") if stage.GetPrimAtPath("/physicsScene") else None
    if scene is not None:
        physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physx.CreateSolverTypeAttr("TGS")
    robot = world.scene.add(SingleArticulation("/World/m3", name="m3"))
    shots = sorted(plan.get("marks", {}).items(), key=lambda kv: kv[1]) if plan.get("pictures") else []
    record = plan.get("record")   # {"dir", "hz"}: the wrist camera's pictures, a tick each, as the real recorder
    if record or plan.get("policy"):
        shots = shots or [("recording", math.inf)]
    # one run in three renders black all through, the camera made before the reset or after it (2026-10-26, Dionysos,
    # an ACT training on the same GPU): _picture says so, and the laptop runs it again
    annotator = _camera(plan["pictures"]) if shots else None
    pieces = [world.scene.add(SingleXFormPrim(f"/World/piece{i}", name=f"piece{i}")) for i in range(len(plan["pieces"]))]
    world.reset()
    names = list(robot.dof_names)
    print("dofs:", names, flush=True)
    index = {n: i for i, n in enumerate(names)}
    robot.set_solver_position_iteration_count(32)
    robot.set_solver_velocity_iteration_count(4)
    keys = plan["keys"]
    first = targets_at(keys, 0.0)
    start = robot.get_joint_positions()
    for j, v in {**first, **{j: share * first[FINGER] for j, share in LINKAGE.items()}}.items():
        start[index[j]] = v
    robot.set_joint_positions(start)
    tips = [SingleXFormPrim(p) for p in (_find(stage, "rlink2"), _find(stage, "llink2"))]
    grip = SingleXFormPrim(_find(stage, "Gripping"))   # Yahboom's tool point: between the fingertips
    for _ in range(WARMUP_RENDERS if shots else 0):   # the materials' shaders compiled, the picture settled
        world.render()
    controller = robot.get_articulation_controller()
    if plan.get("policy"):   # a trained policy drives the arm, trial after trial (m3_sim.py evaluate)
        trials = run_policy(world, robot, controller, index, annotator, pieces, grip, plan)
        with open(result_path, "w") as f:
            json.dump({"plan": {k: v for k, v in plan.items() if k != "trials"}, "trials": trials}, f)
        print(f"{len(trials)} trials -> {result_path}", flush=True)
        return
    samples, t, next_sample = [], 0.0, 0.0
    ticks, next_tick = 0, 0.0
    if record:
        import omni.replicator.core as rep

        os.makedirs(record["dir"], exist_ok=True)
        _picture(world, annotator, rep)   # not black: else the laptop runs it again
    end = keys[-1]["t"] + plan.get("settle_s", 1.0)
    while t <= end + 1e-9:
        q = targets_at(keys, t)
        finger_now = float(robot.get_joint_positions()[index[FINGER]])
        q = {**q, **{j: share * finger_now for j, share in LINKAGE.items()}}
        controller.apply_action(ArticulationAction(joint_positions=np.array([q[j] for j in q]),
                                                   joint_indices=np.array([index[j] for j in q])))
        shoot = bool(shots) and t >= shots[0][1]
        tick = bool(record) and t >= next_tick - 1e-9
        world.step(render=shoot or tick)
        if tick:
            rep.orchestrator.step(rt_subframes=RECORD_SUBFRAMES, delta_time=0.0, pause_timeline=False)
            _save_jpeg(os.path.join(record["dir"], f"{ticks:06d}.jpg"), np.asarray(annotator.get_data())[..., :3])
            ticks += 1
            next_tick += 1.0 / record.get("hz", 10)
        t += 1.0 / PHYSICS_HZ
        if shoot:
            import omni.replicator.core as rep

            name = shots.pop(0)[0]
            picture = _picture(world, annotator, rep)
            wrist = stage.GetPrimAtPath(_find(stage, plan["pictures"]["mount"]) + "/wrist") \
                if "mount" in plan["pictures"] else None
            if wrist is not None:   # where it is, to check against the robot's model (row vectors: the last row)
                world_m = UsdGeom.Xformable(wrist).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                print(f"wrist camera at {name}: {[round(v, 4) for v in world_m.ExtractTranslation()]}, looking "
                      f"{[round(-world_m[2][i], 3) for i in range(3)]}", flush=True)
            np.save(os.path.splitext(os.path.abspath(result_path))[0] + f"_{name}.npy", picture)
        if t >= next_sample - 1e-9:
            pos = robot.get_joint_positions()
            samples.append({"t": round(t, 3),
                            "q": {j: round(float(pos[index[j]]), 4) for j in ARM + [FINGER, *LINKAGE]},
                            "target": {j: round(float(v), 4) for j, v in q.items()},
                            "pieces": [_pose(p) for p in pieces],
                            "tips": [_pose(p)[:3] for p in tips],
                            "grip": _pose(grip)[:3]})
            next_sample += SAMPLE_S
    with open(result_path, "w") as f:
        json.dump({"plan": plan, "dofs": names, "samples": samples, "ticks": ticks}, f)
    print(f"{len(samples)} samples, {ticks} pictures over {t:.1f} s -> {result_path}", flush=True)


def servo_to_q(pose: dict, servo_map: dict) -> dict:
    """Servo degrees {1..6} -> the joints' targets, rad: where the arm sits (the command + the model's sag),
    J1 mirrored, 90 the URDF's zero; J6 -> the finger through the gripper as measured (m3_sim.py's mapping)."""
    import numpy as np

    q = {}
    for j in range(1, 6):
        v = pose[j] + servo_map["sag"].get(str(j), 0.0)
        q[ARM[j - 1]] = math.radians(90 - v if j in servo_map["mirrored"] else v - 90)
    q[FINGER] = float(np.interp(pose[6], *zip(*servo_map["j6"])))
    return q


def _ask(url: str, payload: dict | None = None, wait_s: float = 120.0) -> dict:
    """The policy server's answer, waited for while it is still loading."""
    import urllib.request

    deadline = time.time() + wait_s
    while True:
        try:
            request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as reply:
                return json.loads(reply.read())
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(2.0)


def run_policy(world, robot, controller, index, annotator, pieces, grip, plan: dict) -> list[dict]:
    """Each trial: the arm at the start pose, the pieces where the trial has them, then a policy (its server at
    plan["policy"]["url"]) drives the arm ten times a second for the trial's seconds, as the bridge's policy job
    does on the robot -- the newest picture and the pose sent, a chunk every n_action_steps ticks, each step held
    to max_dps / hz a joint (J6 <= 150) and swept over the tick as the board does. What each tick saw and sent,
    and where the pieces and the tool point were -> the trial's record."""
    import base64
    import io

    import numpy as np
    import omni.replicator.core as rep
    from isaacsim.core.utils.types import ArticulationAction
    from PIL import Image

    pol, servo_map = plan["policy"], plan["servo_map"]
    hz, n = pol.get("hz", 10), pol["n_action_steps"]
    most = pol.get("max_dps", 100.0) / hz
    limits = {1: (0, 180), 2: (0, 180), 3: (0, 180), 4: (0, 180), 5: (0, 270), 6: (0, 150)}
    print("policy:", _ask(pol["url"] + "/health"), flush=True)
    _picture(world, annotator, rep)   # not black: else the laptop runs it again

    def drive(pose):
        q = servo_to_q(pose, servo_map)
        finger_now = float(robot.get_joint_positions()[index[FINGER]])
        q = {**q, **{j: share * finger_now for j, share in LINKAGE.items()}}
        controller.apply_action(ArticulationAction(joint_positions=np.array([q[j] for j in q]),
                                                   joint_indices=np.array([index[j] for j in q])))

    out = []
    for number, trial in enumerate(plan["trials"]):
        cmd = {int(j): float(v) for j, v in plan["start"].items()}
        start = robot.get_joint_positions()
        q0 = servo_to_q(cmd, servo_map)
        for j, v in {**q0, **{j: share * q0[FINGER] for j, share in LINKAGE.items()}}.items():
            start[index[j]] = v
        robot.set_joint_positions(start)
        for prim, piece in zip(pieces, trial["pieces"], strict=False):
            yaw = math.radians(piece.get("yaw", 0.0))
            prim.set_world_pose(position=np.array(piece["centre"]), orientation=np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]))
        for _ in range(int(0.5 * PHYSICS_HZ)):   # settled: the arm held at the start, the pieces down
            drive(cmd)
            world.step(render=False)
        queue, ticks, sheet = [], [], []
        for tick in range(int(trial["seconds"] * hz)):
            world.step(render=True)
            rep.orchestrator.step(rt_subframes=RECORD_SUBFRAMES, delta_time=0.0, pause_timeline=False)
            rgb = np.asarray(annotator.get_data())[..., :3].astype("uint8")
            if tick % (2 * hz) == 0:
                sheet.append(tick)
                Image.fromarray(rgb).save(f"trial{number:02d}_{tick:04d}.jpg", quality=80)
            if not queue:
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format="JPEG", quality=JPEG_QUALITY)
                answer = _ask(pol["url"] + "/act", {"state": [cmd[j] for j in range(1, 7)],
                                                    "jpeg": base64.b64encode(buf.getvalue()).decode()})
                queue = answer["chunk"][:n]
            wanted = queue.pop(0)
            step = {j: min(max(cmd[j] + float(np.clip(wanted[j - 1] - cmd[j], -most, most)), limits[j][0]), limits[j][1])
                    for j in range(1, 7)}
            sweeps = int(PHYSICS_HZ / hz)
            for k in range(sweeps):   # the board sweeps from the last command to this one over the tick
                drive({j: cmd[j] + (k + 1) / sweeps * (step[j] - cmd[j]) for j in range(1, 7)})
                world.step(render=False)
            cmd = step
            ticks.append({"t": round(tick / hz, 2), "sent": [round(cmd[j], 1) for j in range(1, 7)],
                          "pieces": [_pose(p) for p in pieces], "grip": _pose(grip)[:3]})
        out.append({"trial": number, "pieces": trial["pieces"], "ticks": ticks, "sheet": sheet})
        print(f"trial {number}: {len(ticks)} ticks", flush=True)
    with contextlib.suppress(OSError):
        _ask(pol["url"] + "/quit", wait_s=1.0)
    return out


def _save_jpeg(path: str, rgb) -> None:
    from PIL import Image

    Image.fromarray(rgb.astype("uint8")).save(path, quality=JPEG_QUALITY)


def add_static(stage, path: str, part: dict) -> None:
    """A fixed box (a collider, no body): {"T": 4 x 4 its middle and turn (column vectors), "size": [x, y, z],
    "colour": rgb | "faces": six looks as add_box's}."""
    if part.get("faces"):
        geom = box_mesh(stage, path, FACES, [look_material(stage, f"{path}_look{i}", f.get("texture"), f.get("colour"))
                                             for i, f in enumerate(part["faces"])])
    else:
        geom = UsdGeom.Cube.Define(stage, path)
        geom.CreateSizeAttr(1.0)
        geom.CreateDisplayColorAttr([Gf.Vec3f(*part.get("colour", (0.03, 0.03, 0.03)))])
    xf = UsdGeom.Xformable(geom)
    xf.AddTransformOp().Set(Gf.Matrix4d([[float(part["T"][r][c]) for r in range(4)] for c in range(4)]))
    xf.AddScaleOp().Set(Gf.Vec3f(*part["size"]))
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())


def _camera(view: dict):
    """A camera and a dome light: its rgb annotator. `view`: {"from": xyz, "at": xyz} -- looking on, 14 mm,
    640 x 480 -- or the wrist camera: {"mount": a link's name, "T": 4 x 4 link -> the USD camera (column
    vectors; a USD camera looks down its -z, +y up), "K": [fx, fy, cx, cy], "size": [w, h]}, fixed to the
    link so it moves with the arm (the principal point taken as the picture's middle: the DCW2's is 3.8 px off)."""
    import omni.replicator.core as rep
    from pxr import UsdLux

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/World/dome"):   # the lab table brings its own light
        UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(1500)
    size = tuple(view.get("size", (640, 480)))
    if "mount" in view:
        path = _find(stage, view["mount"]) + "/wrist"
        cam = UsdGeom.Camera.Define(stage, path)
        fx, fy, _cx, _cy = view["K"]
        focal = 10.0   # any: the field of view is aperture over focal length, set from fx, fy
        cam.CreateFocalLengthAttr(focal)
        cam.CreateHorizontalApertureAttr(focal * size[0] / fx)
        cam.CreateVerticalApertureAttr(focal * size[1] / fy)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.03, 20.0))   # past the DCW2's own housing round its centre
        rows = [[float(view["T"][r][c]) for r in range(4)] for c in range(4)]   # USD: row vectors, transposed
        UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d(rows))
        print("wrist camera at", path, flush=True)
    else:
        path = "/World/sidecam"
        cam = UsdGeom.Camera.Define(stage, path)
        cam.CreateFocalLengthAttr(14.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        look = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*view["from"]), Gf.Vec3d(*view["at"]), Gf.Vec3d(0, 0, 1))
        UsdGeom.Xformable(cam).AddTransformOp().Set(look.GetInverse())
    product = rep.create.render_product(path, size)
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([product])
    return annotator


def _picture(world, annotator, rep, tries: int = 20):
    """The camera's picture (H x W x 3), rendered until it is one: a fresh material's shaders compile for
    longer than any fixed warm-up, and until then the picture is black (2026-10-26: black, then right, then
    black again with a new texture, the logs alike)."""
    import numpy as np

    for _ in range(tries):
        rep.orchestrator.step(rt_subframes=8, delta_time=0.0, pause_timeline=False)
        picture = np.asarray(annotator.get_data())[..., :3]
        if picture.size and float(picture.mean()) > 1.0:
            return picture
        for _ in range(30):
            world.render()
    print("the picture stayed black", flush=True)
    raise SystemExit(3)


def _find(stage, name: str) -> str:
    return next(str(p.GetPath()) for p in stage.Traverse() if p.GetName() == name)


def _pose(prim) -> list:
    pos, quat = prim.get_world_pose()
    return [round(float(v), 4) for v in list(pos) + list(quat)]


try:
    if a.cmd == "import":
        do_import(a.desc, a.usd)
    else:
        do_play(a.usd, a.plan, a.result)
finally:
    app.close()
    sys.stdout.flush()
