"""Episode-level randomization from the user's G1 table-skill specification.

This module is opt-in. It never changes the historical five-stage environments.
Contact structure is compiled once; continuous physical parameters are sampled
at every reset. All sampling starts from the compiled defaults, never from a
previous randomized episode.
"""
from __future__ import annotations

import copy
import math
import re

import torch
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

import task as tk
from five_actuation import configure_actuation

TIERS = ('train', 'dr', 'sensors', 'nominal')


class RandomizedGeo(tk.Geo):
    """Use each world's sampled box width in the existing contact metrics."""
    def __init__(self, env):
        super().__init__(env)
        if not hasattr(self.t.cfg, 'tier'):
            return
        import scene
        index=env.sim.mj_model.geom('box/box_geom').id
        half=env.sim.model.geom_size[:,index].expand(env.num_envs,-1)
        side=torch.zeros(env.num_envs,2,3,device=env.device)
        side[:,0,1]=half[:,1]+scene.PALM_R;side[:,1,1]=-side[:,0,1]
        old_side=torch.tensor([[0.,scene.BOX_HALF[1]+scene.PALM_R,0.],
                              [0.,-scene.BOX_HALF[1]-scene.PALM_R,0.]],device=env.device)
        local=self.palm_err+old_side
        self.palm_err=local-side;self.d_side=self.palm_err.norm(dim=-1)
        self.gap=(local.abs()-half[:,None]).clamp_min(0.).norm(dim=-1)-scene.PALM_R
        self.bottom=self.box[:,2]-half[:,2]
SPEC = {
    'version': 'g1-table-user-dr-v1',
    'sampling': 'episode reset; independent environments; uniform unless specified',
    'foot_friction': [.5, 1.6], 'palm_friction': [.4, 1.],
    'palm_torsion_m': [.01, .025], 'table_friction': [.25, .6],
    'box_friction': .25, 'link_mass_inertia_scale': [.9, 1.1],
    'torso_extra_mass_kg': [-.5, 2.5], 'torso_com_m': [.03, .015, .02],
    'kp_scale': [.85, 1.15], 'kd_scale': [.85, 1.15],
    'torque_limit_scale': [.9, 1.], 'arm_gravity_feedforward_scale': [.85, 1.05],
    'arm_friction_nm': [0., .3], 'leg_waist_friction_nm': [0., .6],
    'joint_damping_nms': [0., .1],
    'motor_delay': {'steps': [0, 3], 'step_ms': 5, 'update_steps': 4, 'hold_probability': .9},
    'box_mass_kg': [.35, .65], 'box_com_m': [.03, .03, .03],
    'box_full_width_delta_m': [-.005, .005],
    'box_placement_take': {'xy_m': .015, 'yaw_deg': 2.},
    'table_placement': {'xy_m': .01, 'yaw_deg': 1.},
    'sensor_noise': {'joint_pos_rad': .01, 'joint_vel_rad_s': 1.5,
                     'gyro_rad_s': .2, 'gravity': .05, 'heading_std_deg': .2,
                     'arm_pd_error_rad': .01},
    'sensor_bias': {'joint_pos_rad': .01, 'ankle_pos_rad': .02,
                    'gyro_rad_s': .02, 'gravity': .025,
                    'heading_drift_deg_s': .2, 'handoff_heading_deg': 1.},
    'observation_delay': {'steps': [0, 1], 'step_ms': 20, 'hold_probability': .95},
    'push': {'interval_s': [1., 3.], 'xy_velocity_m_s': 'max(0.4 * level, 0.15)',
             'level_initial': 0., 'level_step': .05, 'success_up': .7, 'success_down': .3},
    'start_mixture': {'begin': .2, 'handoff': .3, 'carry_putdown_handoff': .6},
    'physics': {'timestep': .005, 'decimation': 4, 'cone': 'elliptic',
                'impratio': 10., 'solver': 'newton', 'iterations': 30, 'ls_iterations': 30},
}


def uniform(shape, lo, hi, device):
    return lo + (hi-lo) * torch.rand(shape, device=device)


def _indices(env):
    if hasattr(env, '_table_dr_indices'):
        return env._table_dr_indices
    m = env.sim.mj_model
    names = [m.body(i).name for i in range(m.nbody)]
    robot = [i for i, name in enumerate(names) if name.startswith('robot/') and m.body_mass[i] > 0]
    arm_bodies = [i for i in robot if any(s in names[i] for s in ('shoulder', 'elbow', 'wrist'))]
    geoms = [m.geom(i).name for i in range(m.ngeom)]
    feet = [i for i, name in enumerate(geoms) if re.fullmatch(r'robot/(left|right)_foot[1-7]_collision', name)]
    palms = [m.geom('robot/'+s+'_hand_collision').id for s in ('left', 'right')]
    joints = [m.joint(i).name for i in range(m.njnt)]
    arms = [int(m.jnt_dofadr[i]) for i, name in enumerate(joints)
            if name.startswith('robot/') and any(s in name for s in ('shoulder', 'elbow', 'wrist'))]
    other = [int(m.jnt_dofadr[i]) for i, name in enumerate(joints)
             if name.startswith('robot/') and m.jnt_type[i] != 0 and int(m.jnt_dofadr[i]) not in arms]
    out = dict(robot=robot, arm_bodies=arm_bodies, feet=feet, palms=palms,
               torso=m.body('robot/torso_link').id, box=m.body('box/box').id,
               box_geom=m.geom('box/box_geom').id,
               tables=[m.geom('cart/deck').id, m.geom('shelf/top_board').id],
               arm_dofs=arms, other_dofs=other)
    env._table_dr_indices = out
    return out


@requires_model_fields('body_mass', 'body_inertia', 'body_ipos', 'body_gravcomp',
                       'geom_friction', 'geom_size', 'geom_rbound', 'geom_aabb',
                       'dof_frictionloss', 'dof_damping', recompute=RecomputeLevel.set_const)
def physical_reset(env, env_ids):
    """Resample mass, inertia, COM, contacts, geometry and passive joint loads."""
    ids = env_ids if env_ids is not None else torch.arange(env.num_envs, device=env.device)
    if not len(ids):
        return
    ix = _indices(env); model = env.sim.model; n = len(ids); dev = env.device
    default = env.sim.get_default_field
    for name in ('body_mass', 'body_inertia', 'body_ipos', 'body_gravcomp',
                 'geom_friction', 'geom_size', 'geom_rbound', 'geom_aabb',
                 'dof_frictionloss', 'dof_damping'):
        field = getattr(model, name)
        field[ids] = default(name).reshape(field.shape[1:])
    bodies = ix['robot']; scale = uniform((n, len(bodies)), .9, 1.1, dev)
    model.body_mass[ids[:, None], bodies] = default('body_mass')[bodies] * scale
    model.body_inertia[ids[:, None], bodies] = default('body_inertia')[bodies] * scale[..., None]
    torso = ix['torso']; old = model.body_mass[ids, torso].clone()
    model.body_mass[ids, torso] += uniform((n,), -.5, 2.5, dev)
    model.body_inertia[ids, torso] *= (model.body_mass[ids, torso]/old)[:, None]
    model.body_ipos[ids, torso] += uniform((n, 3), -1., 1., dev) * torch.tensor([.03, .015, .02], device=dev)
    box = ix['box']; bg = ix['box_geom']
    mass = uniform((n,), .35, .65, dev)
    model.body_mass[ids, box] = mass
    model.body_ipos[ids, box] += uniform((n, 3), -.03, .03, dev)
    size = default('geom_size')[bg].expand(n, -1).clone()
    size[:, 1] += uniform((n,), -.0025, .0025, dev)  # full width ±5 mm
    model.geom_size[ids, bg] = size
    model.geom_rbound[ids, bg] = size.norm(dim=-1)
    # aabb is [center, half-size], in geom coordinates.
    model.geom_aabb[ids, bg, 1] = size
    inertia = mass[:, None]/3. * torch.stack((size[:, 1]**2+size[:, 2]**2,
                size[:, 0]**2+size[:, 2]**2, size[:, 0]**2+size[:, 1]**2), -1)
    model.body_inertia[ids, box] = inertia
    model.geom_friction[ids[:, None], ix['feet'], 0] = uniform((n, 1), .5, 1.6, dev)
    model.geom_friction[ids[:, None], ix['palms'], 0] = uniform((n, 1), .4, 1., dev)
    model.geom_friction[ids[:, None], ix['palms'], 1] = uniform((n, 1), .01, .025, dev)
    model.geom_friction[ids[:, None], ix['tables'], 0] = uniform((n, 2), .25, .6, dev)
    model.geom_friction[ids, bg, 0] = .25
    model.body_gravcomp[ids[:, None], ix['arm_bodies']] = uniform((n, 1), .85, 1.05, dev)
    for dofs, high in ((ix['arm_dofs'], .3), (ix['other_dofs'], .6)):
        model.dof_frictionloss[ids[:, None], dofs] = uniform((n, len(dofs)), 0., high, dev)
        model.dof_damping[ids[:, None], dofs] = uniform((n, len(dofs)), 0., .1, dev)


def push(env, env_ids):
    t = tk.task(env)
    tk.ev_push(env, env_ids, vmax=max(.4*t.level, .15))


def configure(cfg, tier):
    """Compile the contact/actuator model and install episode reset events."""
    if tier not in TIERS:
        raise ValueError(tier)
    cfg = copy.deepcopy(cfg)
    tk.Geo=RandomizedGeo
    configure_actuation(cfg, 'actuator_only')
    robot = cfg.scene.entities['robot']
    for collision in robot.collisions:
        collision.condim = {r'^(left|right)_foot[1-7]_collision$': 3,
                            r'^(left|right)_hand_collision$': 4, '.*_collision': 1}
        collision.priority = {r'^(left|right)_foot[1-7]_collision$': 1, '.*': 0}
        collision.friction = {r'^(left|right)_foot[1-7]_collision$': (.6,),
                              r'^(left|right)_hand_collision$': (.7, .0175, .0001)}
    for name in ('box', 'cart', 'shelf'):
        old = cfg.scene.entities[name].spec_fn
        def spec(old=old, name=name):
            s = old()
            for geom in s.geoms:
                if geom.name in ('box_geom', 'deck', 'top_board'):
                    geom.friction = (.25 if name == 'box' else .425, .005, .0001)
            return s
        cfg.scene.entities[name].spec_fn = spec
    for actuator in robot.articulation.actuators:
        actuator.delay_min_lag = 0
        actuator.delay_max_lag = 3 if tier in ('train', 'dr') else 0
        actuator.delay_hold_prob = .9
        actuator.delay_update_period = 4
        actuator.delay_per_env_phase = False
    cfg.events = {}
    if tier in ('train', 'dr'):
        cfg.events['table_physics'] = EventTermCfg(mode='reset', func=physical_reset)
        cfg.events['table_pd'] = EventTermCfg(mode='reset', func=dr.pd_gains,
            params=dict(kp_range=(.85, 1.15), kd_range=(.85, 1.15), asset_cfg=SceneEntityCfg('robot')))
        cfg.events['table_motor_limits'] = EventTermCfg(mode='reset', func=dr.effort_limits,
            params=dict(effort_limit_range=(.9, 1.), asset_cfg=SceneEntityCfg('robot')))
    if tier == 'train':
        cfg.events['push'] = EventTermCfg(mode='interval', func=push, interval_range_s=(1., 3.))
    for key, value in SPEC['physics'].items():
        if key == 'decimation':
            cfg.decimation = value
        else:
            setattr(cfg.sim.mujoco, key, value)
    return cfg
