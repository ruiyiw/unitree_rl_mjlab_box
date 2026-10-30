"""Training rewards for placement and release from actual carry handoffs.

All privileged measurements below are training-only. Evaluation uses the
original environment, physical completion gates, and sensor-only controller.
"""
import torch
from mjlab.managers.reward_manager import RewardTermCfg as Reward
from mjlab.managers.termination_manager import TerminationTermCfg as Termination
import task as tk
import five_handoff as hf
from five_research import (ResearchCommandCfg, VARIANTS, invalid_switch,
    endpoint_precision, endpoint_terms, endpoint_valid, terminal_weight)


def anticipatory_posture(env):
    """Prepare upright posture before the unchanged 6-degree gate activates."""
    import five_task as ft
    t=tk.task(env);height,quat=ft.putdown_posture(env)
    tilt=torch.acos((1.-2.*quat[:,1:3].square().sum(-1)).clamp(-1.,1.))
    ramp=((t.clock[:,0]-(t.upright_frame-50))/50.).clamp(0.,1.)
    supported=env.scene['box_shelf'].data.found.reshape(env.num_envs,-1).sum(-1)>0
    return ramp*supported.float()*torch.exp(-(tilt/.05).square()-((height-.76)/.025).square())


def release_quality(env):
    t=tk.task(env);g=tk.geo(env)
    supported=env.scene['box_shelf'].data.found.reshape(env.num_envs,-1).sum(-1)>0
    clear=(1.-g.touch.amax(-1))*(g.gap.amin(-1)/.08).clamp(0.,1.)
    stable=torch.exp(-(g.box_v.norm(dim=-1)/.12).square()-(g.tilt/.15).square())
    # A dropped box earns no release reward: shelf support is mandatory.
    return (t.clock[:,0]>=t.release_frame).float()*supported.float()*clear*stable


def placement_precision(env):
    t=tk.task(env);g=tk.geo(env)
    supported=env.scene['box_shelf'].data.found.reshape(env.num_envs,-1).sum(-1)>0
    distance=(g.box-t.world(0,'box',99999)[:,:3]).norm(dim=-1)
    return (t.clock[:,0]>=t.release_frame).float()*supported.float()*torch.exp(-(distance/.05).square())


def endpoint_margin(env):
    v=endpoint_terms(env)
    cost=sum(((v[k]-margin).clamp_min(0.)/scale).square().clamp_max(10.)
        for k,margin,scale in [('box',.04,.04),('xy',.05,.05),('heading',.08,.12)])
    return terminal_weight(env)*cost


def early_recovery(env):
    # Dense feedback inside the unchanged 25 cm reference-failure boundary.
    return (env.episode_length_buf*env.step_dt<3.).float()*tk.r_box(env,std=.15)


def env_cfg(skill,n,states,variant):
    if skill!='put-down':raise ValueError('Put-down configuration only')
    spec=VARIANTS[variant]
    cfg=hf.env_cfg(skill,n,states)
    cfg.commands['task']=ResearchCommandCfg(plan=cfg.commands['task'].plan,skill=skill,
        states=str(states.resolve()),rsi_noise=False,random_hold=False,
        handoff_fraction=.95,sensor_success=spec['sensor'])
    if spec['sensor']:cfg.terminations['invalid_switch']=Termination(func=invalid_switch)
    cfg.rewards['success'].weight=150.
    cfg.rewards['endpoint_valid']=Reward(func=endpoint_valid,weight=8.)
    for term,weight in {'box':8.,'xy':8.,'heading':2.}.items():
        cfg.rewards['endpoint_'+term]=Reward(func=endpoint_precision,weight=weight,params={'term':term})
    # Holding palms near the box at the end would conflict with release.
    # Existing grasp rewards remain gated by the reference's holding phase.
    cfg.rewards['release_quality']=Reward(func=release_quality,weight=5.)
    cfg.rewards['placement_precision']=Reward(func=placement_precision,weight=6.)
    cfg.rewards['early_recovery']=Reward(func=early_recovery,weight=2.)
    cfg.rewards['supported_release'].weight=6.
    cfg.rewards['putdown_posture'].weight=4.
    # Plain-PD audit: supported placements often cross the 6-degree limit at
    # released-stand onset. The old posture reward starts at that same instant,
    # leaving no dense anticipation signal before a terminating violation.
    cfg.rewards['anticipatory_posture']=Reward(func=anticipatory_posture,weight=6.)
    if spec['style']=='margin':cfg.rewards['endpoint_margin']=Reward(func=endpoint_margin,weight=-4.)
    if spec['fixed']:
        for name in ('foot_friction','box_mass','push'):cfg.events.pop(name,None)
    else:cfg.events['push'].params['vmax']=.05
    cfg.episode_length_s=max(cfg.episode_length_s,32.)
    return cfg
