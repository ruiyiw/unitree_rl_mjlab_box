"""models.py — the actors.

ResidualActor   rsl_rl's MLP actor with its last layer at zero: before training the mean
                correction is exactly zero, i.e. the plan is played back.  The policy is
                109 -> 256 -> 256 -> 23 (ELU, observation normalization).
frozen_actor    a trained ResidualActor loaded for inference only (approach 3's base, and
                evaluation outside a runner).
"""
from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn

from rsl_rl.models import MLPModel

HIDDEN = (256, 256)                # the policy network
ADAPTER_HIDDEN = (64, 64)          # approach 3's learned correction
CRITIC_HIDDEN = (512, 256, 128)    # training only; never deployed


def dist_cfg(init_std: float, learn: bool = True, std_range=None) -> dict:
    """Gaussian exploration with one std per joint, kept below 0.4 or within std_range (or fixed at init_std)."""
    return {"class_name": "GaussianDistribution", "init_std": init_std, "std_type": "scalar", "learn_std": learn,
            "std_range": tuple(std_range or (0.01, 0.4)) if learn else (init_std, init_std)}


class ResidualActor(MLPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        last = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)


def frozen_actor(path: str, obs_dim: int, act_dim: int, device, hidden=HIDDEN):
    """obs (N, obs_dim) -> mean action (N, act_dim), from a checkpoint's actor."""
    sd = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
    obs = TensorDict({"actor": torch.zeros(1, obs_dim, device=device)}, batch_size=[1])
    model = MLPModel(obs, {"actor": ["actor"]}, "actor", act_dim, hidden_dims=tuple(hidden), activation="elu",
                     obs_normalization=True, distribution_cfg=dist_cfg(0.3))
    model.load_state_dict(sd)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def act(o: torch.Tensor) -> torch.Tensor:
        return model(TensorDict({"actor": o}, batch_size=[o.shape[0]]))

    act.model = model
    return act


def n_params(hidden, obs_dim: int, act_dim: int) -> int:
    dims = (obs_dim, *hidden, act_dim)
    return sum(a * b + b for a, b in zip(dims[:-1], dims[1:]))
