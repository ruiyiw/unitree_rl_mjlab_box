"""Training-only initialization from continuous predecessor rollouts.

No additional measurements are exposed to the actor. Historical environments,
references and checkpoints are unchanged. State restoration happens at episode
reset only, never inside a connected sequence evaluation.
"""
from dataclasses import dataclass
from pathlib import Path
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
import task as tk
import five_task as ft
from five_proprio import CONTRACT


SIM_FIELDS = ('qpos', 'qvel', 'act', 'qacc_warmstart', 'ctrl', 'mocap_pos', 'mocap_quat')
ACTION_FIELDS = ('_action', '_prev_action', '_prev_prev_action')
TARGET_FIELDS = ('joint_pos_target', 'joint_vel_target', 'joint_effort_target')


def model_signature(env):
    m = env.sim.mj_model
    return dict(nq=m.nq, nv=m.nv, nu=m.nu, na=m.na,
                joints=[m.joint(i).name for i in range(m.njnt)],
                free_positions=[int(m.jnt_qposadr[i]) for i in range(m.njnt) if m.jnt_type[i] == 0])


def snapshot(env, ids, obs):
    """Remove only the batched environment origin; keep the task frame fixed."""
    origin = env.scene.env_origins[ids]
    out = {k: getattr(env.sim.data, k)[ids].clone() for k in SIM_FIELDS}
    for adr in model_signature(env)['free_positions']:
        out['qpos'][:, adr:adr+3] -= origin
    out['mocap_pos'] -= origin[:, None]
    out.update({k: getattr(env.action_manager, k)[ids].clone() for k in ACTION_FIELDS})
    out.update({k: getattr(env.scene['robot'].data, k)[ids].clone() for k in TARGET_FIELDS})
    out['actor_observation'] = obs[ids].clone()
    out['source_origin'] = origin.clone()
    return {k: v.cpu() for k, v in out.items()}


class HandoffCommand(ft.FiveCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        payload = torch.load(cfg.states, map_location=self.device, weights_only=False)
        if payload['skill'] != (cfg.states_skill or self.skill) or payload['observation_contract'] != CONTRACT:
            raise ValueError('Handoff state contract or destination mismatch')
        if payload['model_signature'] != model_signature(env):
            raise ValueError('Cannot restore handoffs into a different physical model')
        self.samples = payload['states']
        self.sample_count = len(self.samples['qpos'])
        if not self.sample_count: raise ValueError('Empty handoff distribution')
        self.curriculum = False
        self.force_start = tk.BEGIN
        self.just_restored = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.sample_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.metrics['handoff_start'] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        # Ramp 50% -> 90% real handoffs over the first 500 PPO updates.
        fraction = self.cfg.handoff_fraction
        if fraction < 0:
            fraction = .5 + .4 * min(1., self._env.common_step_counter / self.cfg.ramp_steps)
        use = torch.rand(len(env_ids), device=self.device) < fraction
        if self.cfg.handoff_route is not None:
            use &= self.plan_ids[env_ids] == self.cfg.handoff_route
        ids = env_ids[use]
        self.sample_ids[env_ids] = -1
        self.metrics['handoff_start'][env_ids] = use.float()
        if not len(ids): return
        ordered_ids=ids//2 if self.skill=='navigation' and self.cfg.handoff_route is not None else ids
        sample = (ordered_ids % self.sample_count if self.cfg.ordered_samples else
                  torch.randint(self.sample_count, (len(ids),), device=self.device))
        self.sample_ids[ids] = sample
        # Each MuJoCo Warp environment is an independent world. Keep a saved
        # sample at its source origin as well as keeping its local geometry:
        # translating contact configurations across the large visualization
        # grid otherwise changes float32 rounding and the next contact solve.
        self._env.scene.env_origins[ids] = self.samples['source_origin'][sample]
        origin = self._env.scene.env_origins[ids]
        self.frame[ids, :2] = origin[:, :2]
        for key in SIM_FIELDS:
            value = self.samples[key][sample].clone()
            if key == 'qpos':
                for adr in model_signature(self._env)['free_positions']:
                    value[:, adr:adr+3] += origin
            elif key == 'mocap_pos': value += origin[:, None]
            getattr(self._env.sim.data, key)[ids] = value
        for key in ACTION_FIELDS:
            getattr(self._env.action_manager, key)[ids] = self.samples[key][sample]
        for key in TARGET_FIELDS:
            getattr(self.robot.data, key)[ids] = self.samples[key][sample]
        self.clock[ids] = 0
        self.fresh[ids] = True
        self.just_restored[ids] = True
        tk._invalidate(self._env)

    def _update_command(self, env_ids):
        super()._update_command(env_ids)
        if env_ids is None:
            # reset() invokes compute once; the connected controller has just
            # switched and keeps fresh=True for its first physical step.
            self.fresh[self.just_restored] = True
            self.just_restored[:] = False


@dataclass(kw_only=True)
class HandoffCommandCfg(ft.FiveCommandCfg):
    states: str
    handoff_fraction: float = -1.
    ramp_steps: int = 12000
    ordered_samples: bool = False
    states_skill: str | None = None
    handoff_route: int | None = None

    def build(self, env): return HandoffCommand(self, env)


def training_box_lost(env):
    """Allow initial reference mismatch recovery; do not forgive physical drops."""
    t, g = tk.task(env), tk.geo(env)
    seconds = env.episode_length_buf * env.step_dt
    # 45 cm initially, returning to the original 25 cm by three seconds.
    limit = .25 + .20 * (1. - seconds / 3.).clamp(0., 1.)
    far = (t.R('box', 1)[:, :3] - g.box).norm(dim=-1) > limit
    return (g.box[:, 2] < .45) | far


def r_transport_grasp(env):
    g = tk.geo(env)
    return (g.grip > 0).float() * torch.exp(-g.tilt.square() / .15**2)


def navigation_box_lost(env):
    return (tk.task(env).plan_ids==1)&tk.t_box_lost(env)


def env_cfg(skill, n, states, play=False, fraction=-1., ramp_steps=12000):
    cfg = ft.env_cfg(skill, n, play=play)
    cfg.commands['task'] = HandoffCommandCfg(
        plan=cfg.commands['task'].plan, skill=skill, states=str(Path(states).resolve()),
        rsi_noise=False, random_hold=False, handoff_fraction=fraction,
        ramp_steps=ramp_steps, ordered_samples=play,
        states_skill='return' if skill=='navigation' else None,
        handoff_route=1 if skill=='navigation' else None)
    if skill=='go-to':
        # Only load the measured go-to endpoint distribution for shared tooling;
        # validation always starts at the original go-to pose, never a handoff.
        cfg.commands['task'].states_skill='take'
        cfg.commands['task'].handoff_fraction=0.
    if skill in ('return','navigation'):
        cfg.terminations['box_lost']=TerminationTermCfg(
            func=navigation_box_lost if skill=='navigation' else tk.t_box_lost)
    if not play:
        # Handoff mixing replaces the old success curriculum. Do not thereby
        # jump straight to its maximum 0.4 m/s perturbations on the first update.
        cfg.events['push'].params['vmax'] = .1
    if not play and skill == 'carry':
        cfg.terminations['box_lost'] = TerminationTermCfg(func=training_box_lost)
        cfg.rewards['handoff_grasp'] = RewardTermCfg(func=r_transport_grasp, weight=2.)
        cfg.rewards['carry_relative_pose'].weight = 2.
        cfg.rewards['carry_relative_velocity'].weight = .5
        # Broad tracking supplies gradients during recovery; keep the existing
        # tight endpoint/box/palm rewards and all physical success gates.
        cfg.rewards['box_recovery'] = RewardTermCfg(func=tk.r_box, weight=1., params={'std': .20})
    return cfg
