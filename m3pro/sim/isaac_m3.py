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
import itertools
import json
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


def add_box(stage, path: str, size, centre, yaw_deg: float, mass: float, colour, material) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*centre))
    xf.AddRotateZOp().Set(float(yaw_deg))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
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
    robot_prim = stage.DefinePrim("/World/m3", "Xform")
    robot_prim.GetReferences().AddReference(os.path.abspath(usd).replace("\\", "/"))
    rubber = material(stage, "/World/materials/rubber", 1.2, 1.0)
    wood = material(stage, "/World/materials/wood", plan.get("friction", 0.6), plan.get("friction", 0.6) * 0.8)
    for prim in stage.Traverse():   # the fingertips' pads: rubber
        if prim.GetName() in ("rlink2", "llink2"):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(rubber, UsdShade.Tokens.strongerThanDescendants, "physics")
    for i, piece in enumerate(plan["pieces"]):
        add_box(stage, f"/World/piece{i}", piece["size"], piece["centre"], piece.get("yaw", 0.0),
                piece.get("mass", 0.015), piece.get("colour", (0.9, 0.3, 0.2)), wood)
    scene = UsdPhysics.Scene.Get(stage, "/physicsScene") if stage.GetPrimAtPath("/physicsScene") else None
    if scene is not None:
        physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physx.CreateSolverTypeAttr("TGS")
    robot = world.scene.add(SingleArticulation("/World/m3", name="m3"))
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
    shots = sorted(plan.get("marks", {}).items(), key=lambda kv: kv[1]) if plan.get("pictures") else []
    annotator = _camera(plan["pictures"]) if shots else None
    controller = robot.get_articulation_controller()
    samples, t, next_sample = [], 0.0, 0.0
    end = keys[-1]["t"] + plan.get("settle_s", 1.0)
    while t <= end + 1e-9:
        q = targets_at(keys, t)
        finger_now = float(robot.get_joint_positions()[index[FINGER]])
        q = {**q, **{j: share * finger_now for j, share in LINKAGE.items()}}
        controller.apply_action(ArticulationAction(joint_positions=np.array([q[j] for j in q]),
                                                   joint_indices=np.array([index[j] for j in q])))
        shoot = bool(shots) and t >= shots[0][1]
        world.step(render=shoot)
        t += 1.0 / PHYSICS_HZ
        if shoot:
            import omni.replicator.core as rep

            name = shots.pop(0)[0]
            rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
            np.save(os.path.splitext(os.path.abspath(result_path))[0] + f"_{name}.npy",
                    np.asarray(annotator.get_data())[..., :3])
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
        json.dump({"plan": plan, "dofs": names, "samples": samples}, f)
    print(f"{len(samples)} samples over {t:.1f} s -> {result_path}", flush=True)


def _camera(view: dict):
    """A camera looking on (view: {"from": xyz, "at": xyz}, 640 x 480) and a dome light: its rgb annotator."""
    import omni.replicator.core as rep
    from pxr import UsdLux

    stage = omni.usd.get_context().get_stage()
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(1500)
    cam = UsdGeom.Camera.Define(stage, "/World/sidecam")   # a USD camera looks down its -z, +y up
    cam.CreateFocalLengthAttr(14.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    look = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*view["from"]), Gf.Vec3d(*view["at"]), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(cam).AddTransformOp().Set(look.GetInverse())
    product = rep.create.render_product("/World/sidecam", (640, 480))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([product])
    return annotator


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
