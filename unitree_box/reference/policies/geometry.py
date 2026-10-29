"""Assemble, check and export the task geometry without policies or plan banks.

Run with mjlab 1.6 / MuJoCo 3.11: python policies/geometry.py
The exported init_state keyframe places the G1 facing the loaded cart.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil

# Set before importing MuJoCo; callers can choose another OpenGL backend.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainEntityCfg

import scene as sc
import g1_23dof as g1

ROOT = Path(__file__).resolve().parents[1]


def geometry_cfg(layout="original") -> SceneCfg:
    """The task's original entities with a deterministic inspection layout.

    Coordinates are metres, z up. The original cart front is at (0, 0).
    The five-stage layout approaches the left table along +y, carries along
    +x to a parallel table offset toward -y, and returns diagonally.
    Table poses and working-edge poses are separate: the destination is
    approached from its west edge while both tables keep the same orientation.
    This config can also be supplied to an mjlab simulation.
    """
    robot = sc.robot_cfg()
    spec_fn = None
    robot.init_state = replace(robot.init_state, pos=(0.95, 0.0, sc.PELVIS_Z),
                               rot=(0.0, 0.0, 0.0, 1.0))
    props = sc.prop_cfgs()
    props["box"].init_state = EntityCfg.InitialStateCfg(
        pos=(sc.BOX_ON_PROP_X, 0.0, sc.CART_TOP + sc.BOX_HALF[2]))
    props["shelf"].init_state = EntityCfg.InitialStateCfg(
        pos=(0.3, -2.6, 0.0), rot=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))
    if layout == "five-stage":
        config = json.loads((ROOT / "policies/five_stage.json").read_text())
        box_half = tuple(v / 2 for v in config["box_dimensions_m"])
        props["box"].spec_fn = partial(sc.box_spec, half_size=box_half, mass=config["box_mass_kg"])
        sx, sy, syaw = config["source_front"]
        offset = -(box_half[0] - sc.OVERHANG)
        props["box"].init_state = EntityCfg.InitialStateCfg(
            pos=(sx + offset*math.cos(syaw), sy + offset*math.sin(syaw), config["source_top_m"] + box_half[2]),
            rot=(math.cos(syaw/2), 0, 0, math.sin(syaw/2)))
        props["cart"].spec_fn = partial(sc.table_spec, name="cart", top_geom="deck",
                                       top_height=config["source_top_m"], size=config["table_size_m"])
        tx, ty, tyaw = config["source_table_pose"]
        props["cart"].init_state = EntityCfg.InitialStateCfg(
            pos=(tx, ty, 0), rot=(math.cos(tyaw/2), 0, 0, math.sin(tyaw/2)))
        x, y, yaw = config["start_stance"]
        robot.init_state = replace(robot.init_state, pos=(x, y, sc.PELVIS_Z),
                                   rot=(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))
        x, y, yaw = config["destination_table_pose"]
        props["shelf"].spec_fn = partial(sc.table_spec, name="shelf", top_geom="top_board",
                                        top_height=config["destination_top_m"], size=config["table_size_m"])
        props["shelf"].init_state = EntityCfg.InitialStateCfg(
            pos=(x, y, 0), rot=(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))
        def add_waypoints(spec):
            for name, color in (("start_stance", (0.95, 0.25, 0.15, 0.8)),
                                ("pickup_reach_stance", (0.2, 0.8, 0.3, 0.8)),
                                ("place_reach_stance", (0.2, 0.5, 1.0, 0.8))):
                x, y, _ = config[name]
                spec.worldbody.add_site(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                        pos=(x, y, 0.004), size=(0.09, 0.003, 0),
                                        rgba=color, group=0)
        spec_fn = add_waypoints
    elif layout != "original":
        raise ValueError(f"Unknown layout: {layout}")
    return SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"),
                    entities={"robot": robot, **props}, num_envs=1, env_spacing=8.0,
                    spec_fn=spec_fn)


def initial_data(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    return data


def verify(model, data, layout="original"):
    """Check geometry and initial contacts using the compiled scene itself."""
    checks = []

    def check(name, condition):
        if not condition:
            raise ValueError(name)
        checks.append(name)

    original = g1.get_spec().compile()
    check("23 robot actuators and two free bodies", model.nu == 23 and model.nq == 37)
    check("original robot mesh vertices", np.array_equal(model.mesh_vert, original.mesh_vert))
    for j in range(1, original.njnt):
        joint = original.joint(j)
        compiled = model.joint("robot/" + joint.name)
        check(joint.name + " limits preserved", np.array_equal(compiled.range, joint.range))
        angle = data.qpos[compiled.qposadr[0]]
        check(joint.name + " initially within limits", joint.range[0] <= angle <= joint.range[1])
    config = json.loads((ROOT / "policies/five_stage.json").read_text()) if layout == "five-stage" else None
    box_half = np.array(config["box_dimensions_m"]) / 2 if config else np.array(sc.BOX_HALF)
    box_mass = config["box_mass_kg"] if config else sc.BOX_MASS
    check("box dimensions", np.allclose(model.geom("box/box_geom").size, box_half))
    check("box mass", np.isclose(model.body("box/box").mass, box_mass))
    box_bottom = data.geom("box/box_geom").xpos[2] - box_half[2]
    deck_top = data.geom("cart/deck").xpos[2] + model.geom("cart/deck").size[2]
    shelf_top = data.geom("shelf/top_board").xpos[2] + model.geom("shelf/top_board").size[2]
    check("box rests on cart deck", np.isclose(box_bottom, deck_top, atol=1e-9))
    if layout == "five-stage":
        config = json.loads((ROOT / "policies/five_stage.json").read_text())
        check("both table surfaces are 90 cm high", np.allclose(
            [deck_top, shelf_top], [config["source_top_m"], config["destination_top_m"]]))
        for a, b in [("deck", "top_board"), *[(f"leg_{i}_{j}", f"leg_{i}_{j}") for i in range(2) for j in range(2)]]:
            first, second = model.geom("cart/" + a), model.geom("shelf/" + b)
            check(f"identical table geometry: {a}", all(np.allclose(getattr(first,k),getattr(second,k))
                  for k in ("size", "pos", "quat", "rgba", "friction", "contype", "conaffinity")))
        centers = []
        for name, geom, key in (("source", "cart/deck", "source_table_pose"),
                                 ("destination", "shelf/top_board", "destination_table_pose")):
            x, y, yaw = config[key]
            center = np.array([x,y])-config["table_size_m"][0]/2*np.array([math.cos(yaw),math.sin(yaw)])
            centers.append(data.geom(geom).xpos[:2].copy())
            check(name+" table placement", np.allclose(centers[-1],center))
            check(name+" top dimensions", np.allclose(model.geom(geom).size[:2]*2, config["table_size_m"]))
            extent = 2*np.abs(data.geom(geom).xmat.reshape(3,3)) @ model.geom(geom).size
            check(name+" world footprint is 60 cm x by 112.5 cm y",
                  np.allclose(extent[:2],[.60,1.125]))
        check("table width is 2.5 times box width", np.isclose(config["table_size_m"][1],2.5*config["box_dimensions_m"][1]))
        check("table centers remain seven feet apart", np.isclose(np.linalg.norm(centers[1]-centers[0]),7*.3048))
        check("right table is offset toward negative y", centers[1][0]>centers[0][0] and centers[1][1]<centers[0][1])
        check("table y offset matches configuration", np.isclose(centers[1][1]-centers[0][1], config["table_y_offset_m"]))
        check("tables stay parallel", np.allclose(data.geom("cart/deck").xmat, data.geom("shelf/top_board").xmat))
        span_x,span_y=config["table_footprint_xy_m"]
        source_front=np.array(config["source_front"][:2])
        destination_front=np.array(config["destination_front"][:2])
        check("pickup uses near edge of left table", np.allclose(source_front,centers[0]-[0,span_y/2]))
        check("placement uses left edge of right table", np.isclose(destination_front[0],centers[1][0]-span_x/2))
        check("placement fits within destination edge span",
              abs(destination_front[1]-centers[1][1])+box_half[1] <= span_y/2)
        start=np.array(config["start_stance"][:2])
        pickup=np.array(config["safe_pickup_stance"][:2])
        reach=np.array(config["pickup_reach_stance"][:2])
        turn=np.array(config["turn_end_stance"][:2])
        carry=np.array(config["carry_end_stance"][:2])
        place=np.array(config["place_reach_stance"][:2])
        check("robot starts at configured position", np.allclose(data.body("robot/pelvis").xpos[:2],start))
        check("five-foot approach is measured to pickup stop", np.allclose(pickup-start,[0,5*.3048]))
        check("initial heading points straight toward pickup", np.isclose(config["start_stance"][2],math.pi/2) and np.isclose(config["safe_pickup_stance"][2],math.pi/2))
        check("pickup stop is 35 cm from table edge", np.allclose(source_front-pickup,[0,.35]))
        check("pickup reach is 20 cm from table edge", np.allclose(source_front-reach,[0,.2]))
        check("pickup turn has no prescribed backstep", np.allclose(turn,reach))
        angle=lambda value: math.atan2(math.sin(value),math.cos(value))
        check("pickup turns right 90 degrees", np.isclose(angle(config["turn_end_stance"][2]-config["pickup_reach_stance"][2]),-math.pi/2))
        check("carry is straight along positive x", np.allclose(carry-turn,[config["carry_distance_m"],0]))
        check("placement is at carry endpoint", np.allclose(place,carry))
        check("placement stop is 27 cm from working edge", np.allclose(destination_front-place,[.27,0]))
        return_start=np.array(config["navigation_start_return"][:2])
        check("putdown clearance step matches configuration", np.allclose(place-return_start,[config["putdown_clearance_m"],0]))
        check("return is one straight diagonal", np.allclose(config["return_waypoints"],[return_start,start]))
        check("return goal equals initial position", np.allclose(config["return_goal"][:2],start))
        heading=math.atan2(start[1]-return_start[1],start[0]-return_start[0])
        check("return heading faces starting point", np.isclose(config["return_goal"][2],heading))
        check("return is a right turn between 90 and 180 degrees", 90<config["return_turn_degrees"]<180 and np.isclose(config["return_turn_degrees"],-math.degrees(heading)))
        check("return distance matches diagonal", np.isclose(np.linalg.norm(start-return_start),config["return_distance_m"]))
        check("proposed carry box clears table surface by 4 cm", np.isclose(
            config["carry_box_center_height_m"] - box_half[2] - shelf_top, 0.04))
    else:
        check("cart and shelf heights", np.allclose([deck_top, shelf_top], [sc.CART_TOP, sc.SHELF_TOP]))
        check("shelf positioned separately from cart", np.allclose(data.geom("shelf/top_board").xpos[:2], [0.3, -2.8]))
    feet = [data.site("robot/" + side + "_foot").xpos.tolist() for side in ("left", "right")]
    check("feet at ground level", all(0 <= foot[2] < 0.005 for foot in feet))
    min_distance = min((float(c.dist) for c in data.contact), default=0.0)
    check("no initial penetration beyond 1 mm", min_distance >= -0.001)
    check("physics clock unchanged", data.time == 0)
    return dict(passed=True, checks=checks, nq=model.nq, nv=model.nv,
                actuators=model.nu, geoms=model.ngeom, meshes=model.nmesh,
                box_dimensions_m=(2 * box_half).tolist(),
                box_mass_kg=box_mass, cart_top_m=float(deck_top),
                shelf_top_m=float(shelf_top), foot_sites_m=feet,
                minimum_contact_distance_m=min_distance,
                validation="CPU forward kinematics and contacts; no policy or physics rollout")


def camera(layout="original"):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.1, -1.15, 0.6)
    cam.distance, cam.azimuth, cam.elevation = 5.0, -45.0, -24.0
    if layout == "five-stage":
        cam.lookat[:] = (0.9, -0.55, 0.55)
        cam.distance, cam.azimuth, cam.elevation = 5.8, 90.0, -42.0
    return cam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("original", "five-stage"), default="original")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-render", action="store_true", help="Compile and check without OpenGL")
    parser.add_argument("--viewer", action="store_true", help="Inspect the static scene in a desktop window")
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / "results/geometry"
        if args.layout != "original":
            args.output /= args.layout
    scene = Scene(geometry_cfg(args.layout), device="cpu")
    scene.spec.option.timestep = 0.005
    model = scene.compile()
    data = initial_data(model)
    report = verify(model, data, args.layout)
    args.output.mkdir(parents=True, exist_ok=True)
    scene.write(args.output)
    # mjlab exports in-memory assets; this robot loads its meshes from disk.
    asset_root = Path(g1.XML).resolve().parent
    shutil.copytree(asset_root / "assets", args.output / "assets", dirs_exist_ok=True)
    shutil.copy2(asset_root / "LICENCE", args.output / "LICENCE")
    mujoco.mj_saveModel(model, str(args.output / "scene.mjb"))
    # Verify the portable XML plus copied assets, including its initial layout.
    restored = mujoco.MjModel.from_xml_path(str(args.output / "scene.xml"))
    verify(restored, initial_data(restored), args.layout)
    report["layout"] = args.layout
    if args.layout == "five-stage":
        shutil.copy2(ROOT / "policies/five_stage.json", args.output / "task_setup.json")
    report["export_reload_passed"] = True
    report["versions"] = {p: importlib.metadata.version(p) for p in ("mjlab", "mujoco", "torch")}
    report["preview"] = None
    if not args.no_render:
        from PIL import Image
        with mujoco.Renderer(model, height=720, width=1080) as renderer:
            renderer.update_scene(data, camera=camera(args.layout))
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            Image.fromarray(renderer.render()).save(args.output / "preview.png")
        report["preview"] = "preview.png"
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(report['checks'])} geometry checks and exported XML reload")
    print(f"Scene and report: {args.output.resolve()}")
    if args.viewer:
        import time
        from mujoco import viewer as native_viewer
        with native_viewer.launch_passive(model, data) as viewer:
            cam = camera(args.layout)
            viewer.cam.lookat[:] = cam.lookat
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = cam.distance, cam.azimuth, cam.elevation
            while viewer.is_running():
                viewer.sync()
                time.sleep(1 / 30)


if __name__ == "__main__":
    main()
