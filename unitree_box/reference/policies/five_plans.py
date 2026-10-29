"""Generate independent references for the configured five-stage route."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("REKIN_GPU", "0")
import mujoco
import numpy as np
import torch

import plan as pl
import scene as sc
from geometry import geometry_cfg, initial_data
from mjlab.scene import Scene
from five_requirements import posture_ok, upright_putdown_ok

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "five_stage.json"
SKILLS = ("go-to", "take", "carry", "put-down", "return")
POLICIES = ("navigation", "take", "carry", "put-down")
OUT = HERE / "five_runs"
ARRAYS = ("qpos", "qcmd", "qvel", "body", "body_lin", "body_ang", "box", "box_lin", "box_wz", "grip", "contact")


def configure_dimensions():
    """Set the legacy planner helpers' dimensions inside a five-stage worker."""
    c = json.loads(CONFIG.read_text())
    sc.BOX_HALF = tuple(v / 2 for v in c["box_dimensions_m"])
    sc.BOX_MASS = pl.BOX_MASS = c["box_mass_kg"]
    sc.CART_TOP, sc.SHELF_TOP = c["source_top_m"], c["destination_top_m"]
    sc.BOX_ON_PROP_X = -(sc.BOX_HALF[0] - sc.OVERHANG)
    pl.BOX_Z0 = sc.CART_TOP + sc.BOX_HALF[2]
    pl.BOX_REST_Z = sc.SHELF_TOP + sc.BOX_HALF[2]
    pl.CARRY_BOX = (c["carry_box_forward_offset_m"], 0., c["carry_box_center_height_m"] - sc.PELVIS_Z)
    pl.READY = (0.20, sc.BOX_HALF[1] + sc.PALM_R + 0.07, pl.BOX_Z0 + .04 - sc.PELVIS_Z)
    return c


def build(skill):
    if skill == "navigation":
        return build_navigation()
    c = configure_dimensions()
    planner = pl.Planner(check_every=10)
    planner.r.max_ik_step = .1
    sx, sy, syaw = c["source_front"]
    source = (sx + sc.BOX_ON_PROP_X * math.cos(syaw),
              sy + sc.BOX_ON_PROP_X * math.sin(syaw), sc.CART_TOP + sc.BOX_HALF[2], syaw)
    fx, fy, fyaw = c["destination_front"]
    placed = (fx + sc.BOX_ON_PROP_X * math.cos(fyaw),
              fy + sc.BOX_ON_PROP_X * math.sin(fyaw), c["destination_top_m"] + sc.BOX_HALF[2], fyaw)
    chunks, phases = [], []
    pose = c[{"go-to": "start_stance", "take": "safe_pickup_stance", "carry": "turn_end_stance",
              "put-down": "carry_end_stance", "return": "carry_end_stance"}[skill]]
    if skill == "return":
        pose = c["navigation_start_return"]

    def segment(name, vias=(), keys=None, seconds=1.5, carry=False, box=None):
        nonlocal pose
        q0 = chunks[-1]["qpos"][-1].astype(np.float64) if chunks else None
        b0 = box if box is not None else (chunks[-1]["box"][-1] if chunks else source)
        if keys is None:
            keys = [pl.Key(0., "carry" if carry else None, "contact" if carry else "rest", grip=carry)]
        phases.append((name, sum((len(p["qpos"]) - 1) / pl.FPS for p in chunks), name))
        p = planner.plan(pose, list(vias), keys, q0=q0, box0=b0, carry=carry,
                         stand=not vias, seconds=seconds, t_post=1.0)
        chunks.append(p)
        if vias:
            pose = vias[-1]

    if skill == "go-to":
        segment("walk", [c["safe_pickup_stance"]])
        segment("stand", seconds=2.)
    elif skill == "take":
        ready = pl.READY
        clear = (.10, ready[1], ready[2])
        segment("raise", keys=[pl.Key(0., None, "rest"), pl.Key(.3, None, "rest"),
                               pl.Key(1.4, None, clear)], seconds=1.7)
        segment("ready", keys=[pl.Key(0., None, clear), pl.Key(1.2, None, "ready")], seconds=1.5)
        segment("approach", [c["pickup_reach_stance"]], [pl.Key(0., None, "ready")])
        lifted = (*source[:2], c["carry_box_center_height_m"], source[3])
        segment("grasp-lift", seconds=3.6, keys=[pl.Key(0., source, "ready"),
                pl.Key(.7, source, "open"), pl.Key(1.2, source, "contact", grip=True),
                pl.Key(2.2, lifted, "contact", grip=True), pl.Key(3.0, "carry", "contact", grip=True)])
        segment("turn-right", [c["turn_end_stance"]], carry=True)
        segment("hold", carry=True, seconds=2.)
    elif skill == "carry":
        segment("carry", [c["carry_end_stance"]], carry=True)
        segment("hold", carry=True, seconds=2.)
    elif skill == "put-down":
        segment("hold", seconds=.6, carry=True)
        above = (*placed[:2], c["carry_box_center_height_m"], placed[3])
        segment("place-release", seconds=3.8, keys=[pl.Key(0., "carry", "contact", grip=True),
                pl.Key(1., above, "contact", grip=True), pl.Key(1.8, placed, "contact", grip=True),
                pl.Key(2.2, placed, "contact"), pl.Key(3.6, placed, "ready")])
        planner.r.upright_manipulation = True
        segment("released-stand", seconds=1., keys=[
            pl.Key(0., placed, "ready", pelvis_z=float(chunks[-1]["qpos"][-1,2])),
            pl.Key(.8, placed, "ready")], box=placed)
        # Move the feet to gain clearance, instead of making the whole-body IK
        # squat or lean backward to satisfy low, retracted palm targets.
        segment("clearance-step", [c["putdown_clearance_stance"]],
                keys=[pl.Key(0.,placed,"ready")], box=placed)
        segment("lower-arms", seconds=3.,
                keys=[pl.Key(0.,placed,"ready"),pl.Key(2.6,placed,"rest")])
        segment("arms-down", seconds=1., keys=[pl.Key(0.,placed,"rest")], box=placed)
        segment("turn-home", [c["navigation_start_return"]],
                keys=[pl.Key(0.,placed,"rest")], box=placed)
        segment("navigation-ready", seconds=2., box=placed)
    else:
        segment("return-start", [c["return_goal"]], box=placed)
        segment("stand", seconds=2., box=placed)

    merged = {k: np.concatenate([p[k] if i == 0 else p[k][1:] for i, p in enumerate(chunks)]) for k in ARRAYS}
    # Recompute derivatives across segment boundaries rather than inheriting local endpoints.
    q = merged["qpos"].astype(np.float64)
    merged["qvel"] = merged["qvel"].astype(np.float64)
    for i in range(len(q)):
        a, b = max(i-1, 0), min(i+1, len(q)-1)
        mujoco.mj_differentiatePos(planner.r.m, merged["qvel"][i], (b-a)*pl.DT, q[a], q[b])
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, q[i, 3:7].astype(float))
        merged["qvel"][i, 3:6] = rotation.reshape(3, 3) @ merged["qvel"][i, 3:6]
    merged["body_lin"] = np.gradient(merged["body"][..., :3], pl.DT, axis=0)
    merged["box_lin"] = np.gradient(merged["box"][:, :3], pl.DT, axis=0)
    merged["box_wz"] = np.gradient(np.unwrap(merged["box"][:, 3]), pl.DT)
    report = validate(merged, planner.r, skill, phases)
    report.update(seconds=(len(q)-1)/pl.FPS, qp_count=planner.stats.n,
                  qp_statuses=planner.stats.statuses, qp_cross_checks=planner.stats.check_n,
                  qp_max_relative_cost_error=planner.stats.check_max_rel,
                  max_ik_error_m=float(max(p["ik_err"].max() for p in chunks)))
    report["segment_ik_errors"] = {phase[0]: {"max_m": float(p["ik_err"].max()),
        "frame": int(p["ik_err"].argmax())} for phase, p in zip(phases, chunks)}
    report["passed"] &= report["max_ik_error_m"] < .08
    folder = OUT / skill / "banks"
    folder.mkdir(parents=True, exist_ok=True)
    bank = dict(joint_names=planner.r.joints, body_names=list(pl.TRACKED), fps=pl.FPS,
                title=skill, phases=phases, hold=(0., (len(q)-1)/pl.FPS),
                walk={"from": q[0,:2].tolist(), "to": q[-1,:2].tolist()}, stats=report,
                config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                plan={"len": torch.tensor([len(q)]), **{k: torch.from_numpy(v.astype(np.float32))[None] for k,v in merged.items()}})
    (folder / "validation.json").write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]:
        torch.save(bank, folder / "rejected-plan.pt")
        raise ValueError(f"Unsafe reference for {skill}: {report}")
    torch.save(bank, folder / "plan.pt")
    print(json.dumps(report, indent=2), flush=True)


def build_navigation():
    """Two variable-length references, one shared actor and observation layout."""
    banks = []
    digest = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    for skill in ("go-to", "return"):
        bank = torch.load(OUT/skill/"banks"/"plan.pt", weights_only=False)
        if bank["config_sha256"] != digest or not bank["stats"]["passed"]:
            raise ValueError(f"Regenerate and validate {skill} before navigation")
        banks.append(bank)
    lengths = torch.cat([b["plan"]["len"] for b in banks])
    maximum = int(lengths.max())
    plan = {"len": lengths}
    for key in ARRAYS:
        padded = []
        for b in banks:
            v = b["plan"][key]
            tail = v[:, -1:].expand(1, maximum-v.shape[1], *v.shape[2:])
            padded.append(torch.cat([v, tail], dim=1))
        plan[key] = torch.cat(padded)
    bank = dict(banks[0], title="Shared arms-down navigation", plan=plan,
                route_names=["go-to", "return"],
                stats={"passed": True, "routes": {s: b["stats"] for s,b in zip(("go-to","return"),banks)},
                       "lengths": lengths.tolist(), "scope": "Kinematic references only"})
    folder = OUT/"navigation"/"banks"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(bank, folder/"plan.pt")
    (folder/"validation.json").write_text(json.dumps(bank["stats"], indent=2)+"\n")
    print(json.dumps(bank["stats"], indent=2), flush=True)


def validate(p, robot, skill, phases=()):
    """Reject out-of-range joints and deeply penetrating robot/table references."""
    scene = Scene(geometry_cfg("five-stage"), device="cpu")
    model = scene.compile()
    data = initial_data(model)
    radr = [model.joint("robot/"+n).qposadr[0] for n in robot.joints]
    root = model.joint("robot/floating_base_joint").qposadr[0]
    box = model.joint("box/box_joint").qposadr[0]
    margins = np.minimum(p["qpos"][:,7:] - robot.m.jnt_range[1:,0], robot.m.jnt_range[1:,1] - p["qpos"][:,7:])
    minimum = 0.
    box_minimum = 0.
    palm_error = 0.
    worst = None
    released_box_minimum = 0.
    released_box_worst = None
    release_clear = next((time for name,time,_ in phases if name == "clearance-step"), float("inf"))
    for i, q in enumerate(p["qpos"]):
        data.qpos[root:root+7] = q[:7]
        data.qpos[radr] = q[7:]
        b = p["box"][i]
        data.qpos[box:box+7] = [*b[:3], math.cos(b[3]/2), 0, 0, math.sin(b[3]/2)]
        mujoco.mj_forward(model, data)
        if p["grip"][i]:
            yaw = math.atan2(2*(q[3]*q[6]+q[4]*q[5]), 1-2*(q[5]**2+q[6]**2))
            left = np.array([-math.sin(yaw), math.cos(yaw), 0.])
            for side, sign in (("left", 1), ("right", -1)):
                target = p["box"][i,:3] + sign*(sc.BOX_HALF[1]+sc.PALM_R)*left
                palm_error = max(palm_error, float(np.linalg.norm(data.site("robot/"+side+"_palm").xpos-target)))
        for contact in data.contact:
            a, b = (model.geom(g).name for g in (contact.geom1, contact.geom2))
            forbidden = (a.startswith("robot/") and b.startswith(("cart/", "shelf/"))) or (b.startswith("robot/") and a.startswith(("cart/", "shelf/")))
            if forbidden and contact.dist < minimum:
                minimum, worst = float(contact.dist), [i, a, b]
            box_table = (a.startswith("box/") and b.startswith(("cart/", "shelf/"))) or (b.startswith("box/") and a.startswith(("cart/", "shelf/")))
            if box_table:
                box_minimum = min(box_minimum, float(contact.dist))
            robot_box = (a.startswith("robot/") and b.startswith("box/")) or (b.startswith("robot/") and a.startswith("box/"))
            if robot_box and i/pl.FPS >= release_clear and contact.dist < released_box_minimum:
                released_box_minimum, released_box_worst = float(contact.dist), [i,a,b]
    joint_step = float(np.abs(np.diff(p["qpos"][:,7:],axis=0)).max())
    root_step = float(np.linalg.norm(np.diff(p["qpos"][:,:3],axis=0),axis=1).max())
    torso = list(pl.TRACKED).index("torso_link")
    quat = torch.tensor(p["body"][-1,torso,3:]).reshape(1,4)
    terminal_posture_passed = bool(posture_ok(quat,quat,skill in ("take","carry"))[0])
    upright_report = {}
    upright_passed = True
    if skill == "put-down":
        first = round(next(t for name,t,_ in phases if name == "released-stand")*pl.FPS)
        height = torch.tensor(p["qpos"][first:,2])
        quats = torch.tensor(p["body"][first:,torso,3:])
        tilt = torch.acos((1.-2.*(quats[:,1:3]**2).sum(-1)).clamp(-1.,1.))
        upright_passed = bool(upright_putdown_ok(height,quats).all())
        upright_report = dict(postrelease_upright_passed=upright_passed,
            postrelease_min_pelvis_m=float(height.min()), postrelease_max_pelvis_m=float(height.max()),
            postrelease_max_torso_tilt_deg=float(tilt.max()*180/math.pi))

    return dict(**upright_report, skill=skill, frames=len(p["qpos"]), min_joint_margin_rad=float(margins.min()),
                min_robot_table_contact_m=minimum, worst_contact=worst,
                min_box_table_contact_m=box_minimum, max_held_palm_error_m=palm_error,
                min_released_robot_box_contact_m=released_box_minimum,
                worst_released_robot_box_contact=released_box_worst,
                terminal_posture_passed=terminal_posture_passed,
                max_joint_step_rad=joint_step, max_root_step_m=root_step,
                passed=bool(margins.min() >= 0 and minimum >= -.002 and box_minimum >= -.002
                            and released_box_minimum >= -1.e-5 and terminal_posture_passed and upright_passed
                            and palm_error < .04 and joint_step < .25 and root_step < .04),
                scope="Kinematic reference checks only; learned dynamics and handoffs require evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", choices=(*SKILLS, "navigation"), required=True)
    build(parser.parse_args().skill)
