"""hier.py — the pointer network that switches among the frozen skills.

Runs in the `mjlab` env.  The skills (fetch, carry, place) are the actors of
their PPO runs, loaded frozen with their observation normalisers; each reads
its own observation group (its plan ahead, the robot, the box) and outputs a
correction to its plan's joint targets.  Every 0.1 s the pointer reads its
observation (the goals, the robot, the box and the three plans' clocks),
scores each skill and the highest score's skill drives the joints; the
chosen skill's plan clock runs:

  q = MLP(obs) ∈ R^d,   score_k = q · key_k / √d   (key_k: a learned embedding of skill k)

The query MLP and the keys are first fitted to the hand-written switch
(rule_choice) with DAgger (train.imitate_rule), then trained by PPO on the
whole job (task.env_cfg("full")) with Gaussian exploration noise on the
scores; the env takes the argmax.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from tensordict import TensorDict
from torch import nn

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP

import task as tk

SKILL_HIDDEN = (512, 256, 128)
SKILL_DIST = {"class_name": "GaussianDistribution", "init_std": 0.3, "std_type": "scalar"}


def rule_choice(t: "tk.TaskCommand") -> torch.Tensor:
    """The baseline switch: fetch until its plan has ended and the box has been held
    up HOLD_FOR s, then carry until its plan has ended and the robot has stood at
    the shelf stance HOLD_FOR s, then place."""
    t._rule = getattr(t, "_rule", torch.zeros_like(t.phase))
    fresh = t.started.sum(-1) <= 1                  # a new episode: only fetch's plan has started
    t._rule = torch.where(fresh, torch.zeros_like(t._rule), t._rule)
    t._rule = torch.where((t._rule == 0) & (t.held_t >= tk.HOLD_FOR), torch.ones_like(t._rule), t._rule)
    t._rule = torch.where((t._rule == 1) & (t.at_t >= tk.HOLD_FOR), torch.full_like(t._rule, 2), t._rule)
    return t._rule.clone()


class FrozenSkill(nn.Module):
    """A skill's actor from its PPO checkpoint; returns the mean action."""

    def __init__(self, path: str, obs_dim: int, act_dim: int, device, hidden=SKILL_HIDDEN):
        super().__init__()
        sd = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
        obs = TensorDict({"actor": torch.zeros(1, obs_dim, device=device)}, batch_size=[1])
        self.model = MLPModel(obs, {"actor": ["actor"]}, "actor", act_dim, hidden_dims=tuple(hidden),
                              activation="elu", obs_normalization=True, distribution_cfg=dict(SKILL_DIST))
        self.model.load_state_dict(sd)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(TensorDict({"actor": obs}, batch_size=[obs.shape[0]]))


class SkillAction(ActionTerm):
    """Action = one score per skill; the argmax skill's action drives the joints."""

    def __init__(self, cfg: "SkillActionCfg", env):
        super().__init__(cfg, env)
        self.inner = cfg.joint.build(env)
        self.paths = list(cfg.skill_paths)
        self.k = len(self.paths)
        self.skills = None              # loaded on the first step (observation sizes known then)
        n, dev = self.num_envs, self.device
        self._raw = torch.zeros(n, self.k, device=dev)
        self.choice = torch.zeros(n, dtype=torch.long, device=dev)
        self.switched = torch.zeros(n, dtype=torch.bool, device=dev)
        self.joint_action = torch.zeros(n, self.inner.action_dim, device=dev)
        self.forced = None              # play scripts: a fixed skill index per env, or None

    @property
    def action_dim(self) -> int:
        return self.k

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        env = self._env
        obs = [env.obs_buf[f"skill_{s}"] for s in tk.SKILLS]
        if self.skills is None:
            self.skills = [FrozenSkill(p, o.shape[-1], self.inner.action_dim, self.device)
                           for p, o in zip(self.paths, obs)]
        self._raw[:] = actions
        c = actions.argmax(-1) if self.forced is None else self.forced
        if self.cfg.hold_steps > 1 and self.forced is None:     # decide every hold_steps steps
            keep = (env.episode_length_buf % self.cfg.hold_steps) != 0
            c = torch.where(keep, self.choice, c)
        self.switched = c != self.choice
        self.choice = c.clone()
        t = tk.task(env)                     # the chosen skill's plan is the one that runs
        t.active[:] = c
        t.started[torch.arange(self.num_envs, device=self.device), c] = True
        tk._invalidate(env)
        acts = torch.stack([s(o) for s, o in zip(self.skills, obs)], 0)      # (K, N, A)
        ja = acts.gather(0, c[None, :, None].expand(1, -1, acts.shape[-1]))[0]
        self.joint_action[:] = ja
        self.inner.process_actions(ja)

    def apply_actions(self) -> None:
        self.inner.apply_actions()

    def reset(self, env_ids=None) -> None:
        self.inner.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.0
        self.joint_action[ids] = 0.0
        self.choice[ids] = 0
        self.switched[ids] = False


@dataclass(kw_only=True)
class SkillActionCfg(ActionTermCfg):
    entity_name: str = "robot"
    skill_paths: tuple = ()
    joint: ActionTermCfg | None = None
    hold_steps: int = 1                 # the pointer decides every hold_steps policy steps

    def build(self, env) -> SkillAction:
        return SkillAction(self, env)


class PointerNet(nn.Module):
    """Scores of K skills: a query from the observation against K learned keys."""

    def __init__(self, din: int, k: int, hidden=(256, 128), d: int = 32):
        super().__init__()
        self.query = MLP(din, d, hidden, "elu")
        self.keys = nn.Parameter(torch.randn(k, d) / math.sqrt(d))
        self.scale = 1.0 / math.sqrt(d)

    def forward(self, z):
        return self.query(z) @ self.keys.T * self.scale


class PointerModel(MLPModel):
    """rsl_rl's MLPModel with the pointer network as its head."""

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(256, 128), activation="elu",
                 obs_normalization=False, distribution_cfg=None):
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
                         obs_normalization, distribution_cfg)
        out = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = PointerNet(self._get_latent_dim(), out, tuple(hidden_dims))


class PlanModel(MLPModel):
    """rsl_rl's MLPModel with its last layer at zero: the first policy's mean
    correction is exactly zero, i.e. it plays the MPC's plan back (the warm start)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        last = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
