"""g1_23dof.py — Unitree's 23-DoF G1, for mjlab 1.6.

The model is Unitree's (unitree_rl_mjlab, commit 1425b15): the MJCF and meshes in
../models/unitree_g1_23dof/, unmodified, and the actuator constants of its
g1_23dof_constants.py, the robot of its Unitree-G1-23Dof-Flat task.  That file
imports Unitree's package (`from src import SRC_PATH`) and targets mjlab 1.2; this
is the same robot for mjlab 1.6: the same motors, armatures, PD gains (a 10 Hz
natural frequency, damping ratio 2), effort limits, keyframes and action scale.

23 joints: per leg hip pitch, roll, yaw, knee, ankle pitch, roll (6 + 6); waist yaw
(1); per arm shoulder pitch, roll, yaw, elbow, wrist roll (5 + 5).
"""
from __future__ import annotations

import os

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia_from_two_stage_planetary

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "..", "models", "unitree_g1_23dof", "g1_23dof.xml")


def get_spec() -> mujoco.MjSpec:
    """The MJCF; its meshes load from assets/ beside it."""
    return mujoco.MjSpec.from_file(os.path.normpath(XML))


# motors (Unitree's figures): rotor inertias and gear stages -> reflected inertia
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary((0.139e-4, 0.017e-4, 0.169e-4),
                                                           (1, 1 + 46 / 18, 1 + 56 / 16))
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary((0.489e-4, 0.098e-4, 0.533e-4),
                                                              (1, 4.5, 1 + 48 / 22))
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))

ACTUATOR_5020 = ElectricActuator(reflected_inertia=ARMATURE_5020, velocity_limit=37.0, effort_limit=25.0)
ACTUATOR_7520_14 = ElectricActuator(reflected_inertia=ARMATURE_7520_14, velocity_limit=32.0, effort_limit=88.0)
ACTUATOR_7520_22 = ElectricActuator(reflected_inertia=ARMATURE_7520_22, velocity_limit=20.0, effort_limit=139.0)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535          # 10 Hz
DAMPING_RATIO = 2.0
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ ** 2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ ** 2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ ** 2
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ

G1_ACTUATOR_5020 = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_elbow_joint", ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                       ".*_shoulder_yaw_joint", ".*_wrist_roll_joint"),
    stiffness=STIFFNESS_5020, damping=DAMPING_5020, effort_limit=ACTUATOR_5020.effort_limit,
    armature=ACTUATOR_5020.reflected_inertia)
G1_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
    stiffness=STIFFNESS_7520_14, damping=DAMPING_7520_14, effort_limit=ACTUATOR_7520_14.effort_limit,
    armature=ACTUATOR_7520_14.reflected_inertia)
G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
    stiffness=STIFFNESS_7520_22, damping=DAMPING_7520_22, effort_limit=ACTUATOR_7520_22.effort_limit,
    armature=ACTUATOR_7520_22.reflected_inertia)
# the ankles are 4-bar linkages driven by two 5020s: Unitree takes twice one motor's figures
G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    stiffness=STIFFNESS_5020 * 2, damping=DAMPING_5020 * 2, effort_limit=ACTUATOR_5020.effort_limit * 2,
    armature=ACTUATOR_5020.reflected_inertia * 2)

ARTICULATION = EntityArticulationInfoCfg(
    actuators=(G1_ACTUATOR_5020, G1_ACTUATOR_7520_14, G1_ACTUATOR_7520_22, G1_ACTUATOR_ANKLE),
    soft_joint_pos_limit_factor=0.9)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.78),
    joint_pos={".*_hip_pitch_joint": -0.312, ".*_knee_joint": 0.669, ".*_ankle_pitch_joint": -0.363,
               ".*_elbow_joint": 0.6, "left_shoulder_roll_joint": 0.2, "left_shoulder_pitch_joint": 0.2,
               "right_shoulder_roll_joint": -0.2, "right_shoulder_pitch_joint": 0.2},
    joint_vel={".*": 0.0})

# joint target offset per unit action: a quarter of the effort limit over the stiffness
ACTION_SCALE: dict[str, float] = {n: 0.25 * a.effort_limit / a.stiffness
                                  for a in ARTICULATION.actuators for n in a.target_names_expr}
