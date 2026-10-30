"""Bounded handoff research. Actor contract and original evaluation stay fixed."""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
os.environ.setdefault('MUJOCO_GL','egl')
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.managers.reward_manager import RewardTermCfg as Reward
from mjlab.managers.termination_manager import TerminationTermCfg as Termination
import task as tk
import five_handoff as hf
from five_handoff_train import stage_evaluate
from five_train import ProprioRunner, ppo_cfg, write_json
from five_proprio import CONTRACT, validate_checkpoint
from five_provenance import SOURCES, fingerprints
from five_plans import OUT, CONFIG

HERE=Path(__file__).resolve().parent
RECIPE='sensor-handoff-research-v1'
VARIANTS={
    'endpoint': dict(style='endpoint',sensor=False,std=.025,lr=2e-5,fixed=False),
    'endpoint_explore': dict(style='endpoint',sensor=False,std=.05,lr=2e-5,fixed=False),
    'sensor': dict(style='endpoint',sensor=True,std=.025,lr=2e-5,fixed=False),
    'sensor_low_noise': dict(style='endpoint',sensor=True,std=.015,lr=2e-5,fixed=False),
    'margin': dict(style='margin',sensor=False,std=.025,lr=2e-5,fixed=False),
    'margin_sensor': dict(style='margin',sensor=True,std=.025,lr=2e-5,fixed=False),
    'endpoint_fixed': dict(style='endpoint',sensor=False,std=.025,lr=2e-5,fixed=True),
    'sensor_fixed': dict(style='margin',sensor=True,std=.02,lr=2e-5,fixed=True),
    'sensor_height': dict(style='margin',sensor=True,std=.01,lr=1e-5,fixed=True),
    'sensor_upright': dict(style='margin',sensor=True,std=.008,lr=1e-5,fixed=True),
}


class ResearchCommand(hf.HandoffCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.switch_timer=torch.zeros(self.num_envs,device=self.device)
        self.switch_due=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.metrics['physical_valid_at_switch']=torch.zeros(self.num_envs,device=self.device)

    def _resample_command(self,ids):
        super()._resample_command(ids)
        self.switch_timer[ids]=0.;self.switch_due[ids]=False

    def _update_command(self,ids):
        super()._update_command(ids)
        if ids is not None:return
        ready=self.ended() & self.controller_ready()
        self.switch_timer=torch.where(ready,self.switch_timer+self._env.step_dt,0.)
        self.switch_due=self.switch_timer>=1.
        valid=self.standing()[0]
        if self.cfg.sensor_success:
            self.success=self.switch_due & valid
            self.metrics['success']=self.success.float()
        self.metrics['physical_valid_at_switch']=torch.maximum(
            self.metrics['physical_valid_at_switch'],(self.switch_due & valid).float())


@dataclass(kw_only=True)
class ResearchCommandCfg(hf.HandoffCommandCfg):
    sensor_success: bool=False
    def build(self,env):return ResearchCommand(self,env)


def terminal_weight(env):
    t=tk.task(env)
    return ((t.clock[:,0]-(t.lengths()-1-75))/75.).clamp(0.,1.)


def endpoint_terms(env):
    t=tk.task(env);g=tk.geo(env);r=t.robot.data
    return dict(box=(g.box-t.world(0,'box',99999)[:,:3]).norm(dim=-1),
                xy=(r.root_link_pos_w[:,:2]-t.pose(-1)[:,:2]).norm(dim=-1),
                palm=g.d_side.amax(-1),
                heading=tk.wrap(tk.yaw_of(r.root_link_quat_w)-tk.yaw_of(t.pose(-1)[:,3:7])).abs())


def endpoint_precision(env,term):
    value=endpoint_terms(env)[term]
    width={'box':.06,'xy':.07,'palm':.045,'heading':.12}[term]
    return terminal_weight(env)*torch.exp(-(value/width).square())


def endpoint_margin(env):
    v=endpoint_terms(env)
    # Targets sit inside the unchanged 8cm/10cm/6cm/0.2rad acceptance gates.
    cost=sum(((v[k]-margin).clamp_min(0.)/scale).square().clamp_max(10.)
             for k,margin,scale in [('box',.045,.035),('xy',.05,.05),('palm',.04,.02),('heading',.1,.1)])
    return terminal_weight(env)*cost


def endpoint_valid(env):return tk.task(env).ended().float()*tk.task(env).standing()[0].float()


def endpoint_box_height_cost(env):
    t=tk.task(env)
    error=tk.geo(env).box[:,2]-t.world(0,'box',99999)[:,2]
    return terminal_weight(env)*(error/.03).square().clamp_max(10.)


def carry_box_upright(env,terminal=False):
    value=torch.exp(-(tk.geo(env).tilt/.10).square())
    return value*terminal_weight(env) if terminal else value


def invalid_switch(env):
    t=tk.task(env)
    return t.switch_due & ~t.standing()[0]


def research_cfg(skill,n,states,variant):
    if variant in ('sensor_height','sensor_upright') and skill!='carry':
        raise ValueError('Height refinement is defined only for carry')
    if skill=='put-down':
        from five_putdown_research import env_cfg
        return env_cfg(skill,n,states,variant)
    if skill in ('navigation','return'):
        from five_navigation_research import env_cfg
        return env_cfg(skill,n,states,variant)
    spec=VARIANTS[variant]
    cfg=hf.env_cfg(skill,n,states)
    cfg.commands['task']=ResearchCommandCfg(plan=cfg.commands['task'].plan,skill=skill,
        states=str(states.resolve()),rsi_noise=False,random_hold=False,handoff_fraction=.95,
        sensor_success=spec['sensor'])
    # No relaxed reference termination in this precision-refinement round.
    cfg.terminations['box_lost']=Termination(func=tk.t_box_lost)
    if spec['sensor']:cfg.terminations['invalid_switch']=Termination(func=invalid_switch)
    cfg.rewards['success'].weight=150.
    cfg.rewards['endpoint_valid']=Reward(func=endpoint_valid,weight=8.)
    for term,weight in {'box':12.,'xy':6.,'palm':6.,'heading':2.}.items():
        cfg.rewards['endpoint_'+term]=Reward(func=endpoint_precision,weight=weight,params={'term':term})
    if spec['style']=='margin':cfg.rewards['endpoint_margin']=Reward(func=endpoint_margin,weight=-4.)
    if variant in ('sensor_height','sensor_upright'):
        # The selected actor holds the box about 4 cm below its unchanged goal.
        # Recover vertical margin without weakening the 3-D position gate.
        cfg.rewards['endpoint_box_height']=Reward(func=endpoint_box_height_cost,weight=-3.)
    if variant=='sensor_upright':
        cfg.rewards['carry_box_upright']=Reward(func=carry_box_upright,weight=2.)
        cfg.rewards['endpoint_box_upright']=Reward(func=carry_box_upright,weight=6.,params={'terminal':True})
    if spec['fixed']:
        for name in ('foot_friction','box_mass','push'):cfg.events.pop(name,None)
    else:cfg.events['push'].params['vmax']=.05
    cfg.episode_length_s=max(cfg.episode_length_s,18. if skill=='carry' else 27.)
    return cfg


class ResearchRunner(ProprioRunner):
    def save(self,path,infos=None):
        MjlabOnPolicyRunner.save(self,path,{**(infos or {}),'observation_contract':CONTRACT,
            'reward_recipe':RECIPE,'research':self.research_metadata,
            'simulation_actuation_mode':getattr(self,'actuation_mode','legacy')})
        write_json(Path(path).parent/'training-progress.json',dict(iteration=self.current_learning_iteration,
            checkpoint=str(path),updated_at=datetime.now(timezone.utc).isoformat()))
        if self.current_learning_iteration>0 and Path(path).name not in self.evaluated:
            self.evaluated.add(Path(path).name)
            out=Path(path).with_suffix('.evaluation.json')
            log=open(Path(path).with_suffix('.evaluation.log'),'w')
            cmd=[sys.executable,str(HERE/'five_research.py'),'evaluate','--skill',self.skill,
                 '--checkpoint',str(path),'--states',str(self.validation),'--out',str(out),
                 '--envs',str(getattr(self,'eval_envs',128)),
                 '--actuation-mode',getattr(self,'actuation_mode','legacy')]
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
            log.close();self.evaluators.append((child,out))
        if time.time()>=self.deadline:raise TimeoutError('Nine-hour research deadline reached')


def train(a):
    folder=a.out; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'status.json').exists():raise ValueError('Run already exists')
    spec=VARIANTS[a.variant].copy()
    if a.lr is not None:spec['lr']=a.lr
    if a.std is not None:spec['std']=a.std
    spec.update(head_only=a.head_only,freeze_normalizer=a.freeze_normalizer,noise_scale=a.noise_scale,
                success_weight=a.success_weight,failure_weight=a.failure_weight)
    status=dict(state='initializing',skill=a.skill,variant=a.variant,parameters=spec,
        seed=a.seed,envs=a.envs,iterations=a.iterations,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),pid=os.getpid(),
        checkpoint=str(a.checkpoint.resolve()),states=str(a.states.resolve()),validation=str(a.validation.resolve()),
        actor_inputs=75,critic_inputs=200 if a.skill in ('navigation','return') else 224,deadline_epoch=a.deadline,**fingerprints(a.skill),
        simulation_actuation_mode=a.actuation_mode,
        source_sha256_research=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def update(state,**kwargs):
        status.update(state=state,updated_at=datetime.now(timezone.utc).isoformat(),**kwargs)
        write_json(folder/'status.json',status)
    inputs=folder/'inputs';inputs.mkdir(exist_ok=True)
    for name in (*SOURCES,'five_handoff.py','five_handoff_train.py','five_research.py','five_navigation_research.py','five_putdown_research.py','five_actuation.py'):
        shutil.copy2(HERE/name,inputs/name)
    shutil.copy2(CONFIG,inputs/CONFIG.name)
    env=None;runner=None
    try:
        if time.time()>=a.deadline:raise TimeoutError('Research deadline reached')
        torch.manual_seed(a.seed)
        cfg=research_cfg(a.skill,a.envs,a.states,a.variant);cfg.seed=a.seed
        if a.success_weight is not None:cfg.rewards['success'].weight=a.success_weight
        if a.failure_weight is not None:cfg.rewards['failure'].weight=a.failure_weight
        from five_actuation import configure_actuation
        configure_actuation(cfg,a.actuation_mode)
        if a.noise_scale!=1.:
            for term in cfg.observations['actor'].terms.values():
                if term.noise is not None:
                    term.noise.n_min*=a.noise_scale;term.noise.n_max*=a.noise_scale
            if 'encoder_bias' in cfg.events:
                cfg.events['encoder_bias'].params['bias_range']=tuple(
                    x*a.noise_scale for x in cfg.events['encoder_bias'].params['bias_range'])
        env=ManagerBasedRlEnv(cfg,device='cuda:0');vec=RslRlVecEnvWrapper(env,clip_actions=None)
        agent=ppo_cfg(a.skill,a.iterations);agent.seed=a.seed;agent.save_interval=200
        agent.algorithm.learning_rate=spec['lr'];agent.algorithm.schedule='fixed'
        agent.actor.distribution_cfg.update(init_std=spec['std'],std_range=(spec['std'],spec['std']))
        runner=ResearchRunner(vec,asdict(agent),str(folder),'cuda:0')
        runner.research_metadata=status.copy();runner.skill=a.skill;runner.validation=a.validation.resolve()
        runner.evaluated=set();runner.evaluators=[];runner.deadline=a.deadline
        runner.eval_envs=a.eval_envs
        runner.actuation_mode=a.actuation_mode
        validate_checkpoint(a.checkpoint)
        runner.load(str(a.checkpoint),load_cfg={'actor':True,'critic':not a.fresh_critic},map_location='cuda:0')
        runner.current_learning_iteration=0
        actor=runner.alg._raw_actor
        if a.head_only:
            actor.mlp.requires_grad_(False);actor.mlp[-1].requires_grad_(True)
        if a.freeze_normalizer:actor.update_normalization=lambda observations:None
        frozen={name:value.detach().clone() for name,value in actor.state_dict().items()
            if (a.head_only and name.startswith('mlp.') and not name.startswith('mlp.6.'))
            or (a.freeze_normalizer and name.startswith('obs_normalizer.'))}
        with torch.no_grad():runner.alg._raw_actor.distribution.std_param.fill_(spec['std'])
        obs=vec.get_observations()
        assert obs['actor'].shape[-1]==75 and obs['critic'].shape[-1]==(200 if a.skill in ('navigation','return') else 224)
        runner.save(str(folder/'initial.pt'));update('training')
        runner.learn(a.iterations,init_at_random_ep_len=False)
        if any(not torch.equal(value,actor.state_dict()[name]) for name,value in frozen.items()):
            raise RuntimeError('Frozen actor parameters or normalization changed')
        del actor,frozen
        env.close();env=None
        evaluators=runner.evaluators
        del runner,vec;runner=None;torch.cuda.empty_cache()
        update('evaluating')
        for child,out in evaluators:
            child.wait(timeout=max(1.,a.deadline-time.time()))
            if child.returncode:raise RuntimeError(f'Evaluation failed: {out}')
        reports=[json.loads(out.read_text()) for _,out in evaluators]
        def score(r):
            routes=r.get('route_completion',{})
            if routes:return (routes['go-to']>=.99,routes['return'],routes['go-to'])
            return (r['valid_handoff'],r['completion_ever'],r['sensor_handoff'])
        best=max(reports,key=score)
        write_json(folder/'selection.json',reports)
        update('complete',selected_checkpoint=best['checkpoint'],evaluation=best)
    except BaseException:
        update('failed',error=traceback.format_exc());raise
    finally:
        if env is not None:env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['train','evaluate'])
    p.add_argument('--skill',choices=['go-to','navigation','return','take','carry','put-down'],default='carry')
    p.add_argument('--variant',choices=VARIANTS,default='endpoint')
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--states',type=Path,required=True)
    p.add_argument('--validation',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--envs',type=int,default=4096);p.add_argument('--iterations',type=int,default=800)
    p.add_argument('--eval-envs',type=int,default=128)
    p.add_argument('--seed',type=int,default=101);p.add_argument('--fresh-critic',action='store_true')
    p.add_argument('--lr',type=float);p.add_argument('--std',type=float)
    p.add_argument('--head-only',action='store_true');p.add_argument('--freeze-normalizer',action='store_true')
    p.add_argument('--noise-scale',type=float,default=1.)
    p.add_argument('--success-weight',type=float);p.add_argument('--failure-weight',type=float)
    p.add_argument('--actuation-mode',choices=['legacy','none','actuator_only'],default='none')
    p.add_argument('--deadline',type=float,default=1793293265)
    a=p.parse_args();torch.set_num_threads(1)
    if a.mode=='evaluate':
        r=stage_evaluate(a.skill,a.checkpoint,a.states,a.envs,actuation_mode=a.actuation_mode);write_json(a.out,r);print(json.dumps(r),flush=True)
    else:train(a)


if __name__=='__main__':main()
