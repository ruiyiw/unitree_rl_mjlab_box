"""scene.py — the props, the robot and the carry postures of the box-carrying task.

Runs in the `mjlab` conda env (mjlab 1.6, MuJoCo 3.11).

Every prop has its own frame, with +x the prop's front normal (pointing at
where the robot stands) and the origin on the floor under the middle of its
front edge:

  cart   a flatbed cart: deck 0.60 × 0.56 m with its top at CART_TOP, four
         legs on casters, a push handle at the back.
  shelf  a waist-high shelving unit 0.40 × 0.90 m: top board at SHELF_TOP,
         a lower board, side panels and a back panel.
  box    a cardboard moving box, 0.28 (deep) × 0.30 (wide) × 0.24 (high) m,
         0.5 kg nominal (randomized to 0.33-0.67 kg at startup).

The robot is Unitree's 23-DoF G1 (g1_23dof.py: the model of unitree_rl_mjlab's
Unitree-G1-23Dof-Flat task).  Its legs, waist and wrist rolls keep Unitree's
actuators; the shoulders and elbows get a stiffer PD (kp 40 N·m/rad, the arms' kp
in Unitree's own stand), so that the arms can squeeze the box with a few
centimetres of target offset.  The hand capsules get friction (condim 3, μ = 1) so
the palms can hold the box; no robot geom collides with another (no self-collision).

The robot holds the box between its palms; plan.py solves its postures by
whole-body inverse kinematics, with the arms' joint TARGETS putting the palms
SQUEEZE inside the box's faces while it holds the box.
"""
from __future__ import annotations

import os

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

import g1_23dof as g1

HERE = os.path.dirname(os.path.abspath(__file__))

# the skills' recipe: the base one (default), or fine-tuned against perturbations at
# touchdown and on the box (G1_ROBUST=1, task.py "robust training").  Its actors, hand-off
# banks, reports, runs and media go in a "robust" folder of their own; the plans are shared.
ROBUST = os.environ.get("G1_ROBUST", "0") == "1"


def out_dir(kind, variant=True):
    """banks/, actors/, eval/, runs/ or media/ (and the robust recipe's own folder inside
    it, unless variant=False)."""
    d = os.path.join(HERE, kind)
    return os.path.join(d, "robust") if ROBUST and variant else d

# ------------------------------------------------------------------ geometry

BOX_HALF = (0.14, 0.15, 0.12)       # m: depth (x), width (y), height (z)
BOX_MASS = 0.5                      # kg nominal (an empty-ish moving box)
CART_TOP = 0.68                     # m, top of the cart deck
CART_DECK = (0.60, 0.56)            # deck depth (x), width (y)
SHELF_TOP = 0.60                    # m, top of the shelf's top board
SHELF_BOARD = (0.40, 0.90)          # board depth (x), width (y)
OVERHANG = 0.04                     # box placed this far over a front edge
BOX_ON_PROP_X = -(BOX_HALF[0] - OVERHANG)   # box centre in a prop's frame (x)

PELVIS_Z = 0.76                     # m, standing knees bent (g1.KNEES_BENT_KEYFRAME's joints: the
#                                     foot sites then 1.5 mm off the floor)
HOLD_X = 0.30                       # m, the pelvis in front of a box it picks up or sets down
PALM_R = 0.035                      # radius of the hand capsule
SQUEEZE = 0.03                      # m, palms' targets inside the box faces while holding

BOX_RGBA = (0.72, 0.55, 0.36, 1.0)
TAPE_RGBA = (0.55, 0.40, 0.24, 1.0)
CART_RGBA = (0.20, 0.36, 0.55, 1.0)
METAL_RGBA = (0.62, 0.64, 0.67, 1.0)
WHEEL_RGBA = (0.12, 0.12, 0.13, 1.0)
WOOD_RGBA = (0.80, 0.70, 0.55, 1.0)
WOOD_DARK = (0.66, 0.55, 0.40, 1.0)


PROP_GROUP = 2                      # prop geoms: drawn, never hit by the foot-height rays


def _visual(body, **kw):
    """A geom that is drawn but never collides and carries no mass."""
    body.add_geom(contype=0, conaffinity=0, group=1, density=0, **kw)


def box_spec(half_size=BOX_HALF, mass=BOX_MASS) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    b = spec.worldbody.add_body(name="box", pos=(0, 0, 1.0))
    b.add_freejoint(name="box_joint")
    hx, hy, hz = half_size
    b.add_geom(name="box_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=half_size, group=PROP_GROUP,
               mass=mass, rgba=BOX_RGBA, friction=(0.8, 0.005, 0.0001), condim=3)
    # packing tape over the top seam and down the two short sides
    _visual(b, type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.025, hy + 0.001, 0.001),
            pos=(0, 0, hz + 0.0005), rgba=TAPE_RGBA)
    for s in (1, -1):
        _visual(b, type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.025, 0.001, 0.05),
                pos=(0, s * (hy + 0.0005), hz - 0.05), rgba=TAPE_RGBA)
    return spec


def table_spec(name, top_geom, top_height=0.90, size=(0.60, 0.56)) -> mujoco.MjSpec:
    """Identical four-leg tables, with origin under the front edge midpoint."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    dx, dy = size[0] / 2, size[1] / 2
    thickness, leg_half = 0.03, 0.02
    body.add_geom(name=top_geom, group=PROP_GROUP, type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=(dx, dy, thickness / 2), pos=(-dx, 0, top_height - thickness / 2),
                  rgba=WOOD_RGBA, friction=(0.8, 0.005, 0.0001))
    height = top_height - thickness
    for i, x in enumerate((-0.04, -2 * dx + 0.04)):
        for j, y in enumerate((-dy + 0.04, dy - 0.04)):
            body.add_geom(name=f"leg_{i}_{j}", group=PROP_GROUP, type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=(leg_half, leg_half, height / 2), pos=(x, y, height / 2),
                          rgba=WOOD_DARK)
    return spec


def cart_spec() -> mujoco.MjSpec:
    """Origin on the floor under the middle of the deck's front edge; deck
    spans x ∈ [−0.60, 0]."""
    spec = mujoco.MjSpec()
    c = spec.worldbody.add_body(name="cart")
    dx, dy = CART_DECK[0] / 2, CART_DECK[1] / 2
    t = 0.03
    c.add_geom(name="deck", group=PROP_GROUP, type=mujoco.mjtGeom.mjGEOM_BOX, size=(dx, dy, t / 2),
               pos=(-dx, 0, CART_TOP - t / 2), rgba=CART_RGBA, friction=(0.8, 0.005, 0.0001))
    leg_h = CART_TOP - t - 0.10
    for sx in (-0.04, -2 * dx + 0.04):
        for sy in (dy - 0.04, -dy + 0.04):
            c.add_geom(group=PROP_GROUP, type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.02, 0.02, leg_h / 2),
                       pos=(sx, sy, 0.10 + leg_h / 2), rgba=METAL_RGBA)
            _visual(c, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(0.045, 0.015),
                    pos=(sx, sy, 0.05), quat=(0.7071, 0.7071, 0, 0), rgba=WHEEL_RGBA)
            _visual(c, type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.02, 0.025, 0.01),
                    pos=(sx, sy, 0.095), rgba=METAL_RGBA)
    # lower tray and the push handle at the back
    _visual(c, type=mujoco.mjtGeom.mjGEOM_BOX, size=(dx - 0.02, dy - 0.02, 0.01),
            pos=(-dx, 0, 0.22), rgba=CART_RGBA)
    hx = -2 * dx - 0.02
    for sy in (dy - 0.04, -dy + 0.04):
        _visual(c, type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=(0.012, 0.18),
                pos=(hx, sy, CART_TOP + 0.18), rgba=METAL_RGBA)
    _visual(c, type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=(0.014, dy - 0.04),
            pos=(hx, 0, CART_TOP + 0.36), quat=(0.7071, 0.7071, 0, 0), rgba=METAL_RGBA)
    return spec


def shelf_spec(top_height: float = SHELF_TOP) -> mujoco.MjSpec:
    """Origin on the floor under the middle of the top board's front edge;
    boards span x ∈ [−0.40, 0], y ∈ [−0.45, 0.45]."""
    spec = mujoco.MjSpec()
    s = spec.worldbody.add_body(name="shelf")
    dx, dy = SHELF_BOARD[0] / 2, SHELF_BOARD[1] / 2
    t = 0.025
    s.add_geom(name="top_board", group=PROP_GROUP, type=mujoco.mjtGeom.mjGEOM_BOX, size=(dx, dy, t / 2),
               pos=(-dx, 0, top_height - t / 2), rgba=WOOD_RGBA, friction=(0.8, 0.005, 0.0001))
    s.add_geom(group=PROP_GROUP, type=mujoco.mjtGeom.mjGEOM_BOX, size=(dx, dy, t / 2),
               pos=(-dx, 0, 0.33), rgba=WOOD_RGBA)
    for sy in (dy - t / 2, -dy + t / 2):
        s.add_geom(group=PROP_GROUP, type=mujoco.mjtGeom.mjGEOM_BOX, size=(dx, t / 2, (top_height - t) / 2),
                   pos=(-dx, sy, (top_height - t) / 2), rgba=WOOD_DARK)
    s.add_geom(group=PROP_GROUP, type=mujoco.mjtGeom.mjGEOM_BOX, size=(t / 2, dy, (top_height - t) / 2),
               pos=(-2 * dx + t / 2, 0, (top_height - t) / 2), rgba=WOOD_DARK)
    _visual(s, type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.005, dy - t, 0.03),
            pos=(-0.005, 0, 0.05), rgba=WOOD_DARK)
    return spec


# ------------------------------------------------------------------ robot

ARM_KP, ARM_KD = 40.0, 2.0          # shoulders and elbows (5020 motors, |τ| ≤ 25 N·m)

ROBOT_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=0,
    conaffinity=1,
    condim={r"^(left|right)_foot[1-7]_collision$": 3,
            r"^(left|right)_hand_collision$": 3,
            ".*_collision": 1},
    priority={r"^(left|right)_foot[1-7]_collision$": 1,
              r"^(left|right)_hand_collision$": 1, ".*": 0},
    friction={r"^(left|right)_foot[1-7]_collision$": (0.6,),
              r"^(left|right)_hand_collision$": (1.0,)},
)


def articulation() -> EntityArticulationInfoCfg:
    a5020 = g1.ACTUATOR_5020
    arms = BuiltinPositionActuatorCfg(
        target_names_expr=(".*_elbow_joint", ".*_shoulder_pitch_joint",
                           ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint"),
        stiffness=ARM_KP, damping=ARM_KD, effort_limit=a5020.effort_limit,
        armature=a5020.reflected_inertia)
    wrist_roll = BuiltinPositionActuatorCfg(
        target_names_expr=(".*_wrist_roll_joint",), stiffness=g1.STIFFNESS_5020,
        damping=g1.DAMPING_5020, effort_limit=a5020.effort_limit,
        armature=a5020.reflected_inertia)
    return EntityArticulationInfoCfg(
        actuators=(arms, wrist_roll, g1.G1_ACTUATOR_7520_14, g1.G1_ACTUATOR_7520_22,
                   g1.G1_ACTUATOR_ANKLE),
        soft_joint_pos_limit_factor=0.9)


def action_scale() -> dict[str, float]:
    """Joint target offset per unit action: Unitree's rule (a quarter of the effort
    limit over the stiffness) for the legs, waist and wrist rolls; 0.3 rad for the
    stiffer arm joints (the rule would give 0.16)."""
    s = dict(g1.ACTION_SCALE)
    for k in s:
        if "shoulder" in k or "elbow" in k:
            s[k] = 0.3
    return s


# ------------------------------------------------------------------ arm joints

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")


def robot_spec() -> mujoco.MjSpec:
    """Plain-PD G1 without body-level gravity assistance.

    Body gravcomp also creates non-actuated floating-base lift. Historical
    assisted physics is available explicitly through five_actuation; new
    training must learn arm loading using the same joint/IMU observations.
    """
    spec = g1.get_spec()
    for b in spec.bodies:
        b.gravcomp = 0.0
    for j in spec.joints:
        if any(f"_{k}_joint" in j.name for k in ARM_JOINTS):
            j.actgravcomp = False
    return spec


def robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=g1.KNEES_BENT_KEYFRAME,
        collisions=(ROBOT_COLLISION,),
        spec_fn=robot_spec,
        articulation=articulation(),
    )


def prop_cfgs() -> dict[str, EntityCfg]:
    return {"box": EntityCfg(spec_fn=box_spec),
            "cart": EntityCfg(spec_fn=cart_spec),
            "shelf": EntityCfg(spec_fn=shelf_spec)}
