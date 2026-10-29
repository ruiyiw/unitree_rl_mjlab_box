"""Load the fixed-order controller's frozen actors from one policy file.

Uses the existing sequence environment. The configuration, reference banks and
controller source must match the exported bundle; this is not a distilled MLP.
Version 2 shares navigation between go-to and return. Version 3 gives return
its own 75-input actor while retaining the original go-to actor.
"""
from pathlib import Path
import argparse
import hashlib

import torch
import task as tk
from torch import nn
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from hier import SKILL_DIST
from five_sequence import CHECKPOINTS, ORDER, CompositePolicy
from five_plans import CONFIG,OUT
from five_provenance import fingerprints
from five_proprio import CONTRACT, validate_checkpoint


def export(path):
    actors = {name:validate_checkpoint(p)['actor_state_dict'] for name,p in CHECKPOINTS.items()}
    payload=dict(format='five-stage-composite-proprio-v2',order=ORDER,
        observation_contract=CONTRACT,
        config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        controller_sha256=hashlib.sha256(Path(__file__).with_name('five_sequence.py').read_bytes()).hexdigest(),
        bank_sha256={s:fingerprints(s)['bank_sha256'] for s in ORDER},
        actors=actors,
        checkpoint_paths={name:str(p) for name,p in CHECKPOINTS.items()},
        references={s:torch.load(OUT/s/'banks/plan.pt',map_location='cpu',weights_only=False) for s in ORDER})
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,path)
    return payload


class StoredActor(nn.Module):
    def __init__(self,state,width,device):
        super().__init__()
        obs=TensorDict({'actor':torch.zeros(1,width,device=device)},batch_size=[1])
        self.model=MLPModel(obs,{'actor':['actor']},'actor',23,hidden_dims=(512,256,128),activation='elu',obs_normalization=True,distribution_cfg=dict(SKILL_DIST))
        self.model.load_state_dict(state);self.model.to(device).eval()
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self,obs):return self.model(TensorDict({'actor':obs},batch_size=[obs.shape[0]]))


class BundledPolicy(CompositePolicy):
    def __init__(self,env,obs,path):
        b=torch.load(path,map_location='cpu',weights_only=False)
        assert b['format'] in ('five-stage-composite-proprio-v2','five-stage-composite-proprio-v3'), 'Retrain and export the 75-input actors'
        self.separate_return=b['format']=='five-stage-composite-proprio-v3'
        expected={'navigation','take','carry','put-down'}|({'return'} if self.separate_return else set())
        assert set(b['actors'])==expected,'Unexpected actor selection'
        assert b['observation_contract']==CONTRACT
        assert b['config_sha256']==hashlib.sha256(CONFIG.read_bytes()).hexdigest(),'Task configuration changed'
        assert b['controller_sha256']==hashlib.sha256(Path(__file__).with_name('five_sequence.py').read_bytes()).hexdigest(),'Controller changed'
        assert all(b['bank_sha256'][s]==fingerprints(s)['bank_sha256'] for s in ORDER),'References changed'
        self.env=env
        self.actors={name:StoredActor(state,obs['navigation' if name in ('navigation','return') else 'manipulation'].shape[-1],env.device) for name,state in b['actors'].items()}

    @torch.no_grad()
    def __call__(self,obs):
        if not self.separate_return:return super().__call__(obs)
        stages=tk.task(self.env).plan_ids
        action=torch.empty(self.env.num_envs,23,device=self.env.device)
        for stage,name in enumerate(('navigation','take','carry','put-down','return')):
            mask=stages==stage
            if mask.any():
                group='navigation' if stage in (0,4) else 'manipulation'
                action[mask]=self.actors[name](obs[group][mask])
        return action


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT/'sequence/actors/policy.pt');args=p.parse_args();export(args.out);print(args.out.resolve())
