"""Matched 151/376-input PPO refinement with the previous per-skill recipes."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault('MUJOCO_GL','egl')
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from hier import SKILL_DIST
import task as tk
import motion
from five_train import ppo_cfg,write_json
from five_proprio import validate_checkpoint
from table_observations import CRITICS,contract,env_cfg
from table_domain_randomization import SPEC
from five_actuation import audit_actuation

ROOT=Path(__file__).resolve().parents[1]
RECIPES={
    'go-to':dict(lr=1e-4,std=.05,iterations=4000,freeze_normalizer=False,
        source='proprio-teacher-scratch-4000-20261028; completed guidance, PPO continuation'),
    'take':dict(lr=5e-5,std=.05,iterations=1600,freeze_normalizer=True,
        source='autoresearch-physical-none-r1-preload_fast-20261029'),
    'carry':dict(lr=2e-5,std=.025,iterations=1600,freeze_normalizer=True,
        source='autoresearch-physical-none-r1-preload-20261029'),
    'put-down':dict(lr=1e-6,std=.015,iterations=600,freeze_normalizer=True,
        source='physical-none-putdown-r3-selected-handoffs'),
}
# Matt's skill PPO (task.ppo_cfg 'fetch' with the noise capped): learned std from
# 0.3 within [0.05, 0.15], entropy 0.005, adaptive lr from 1e-3 at KL 0.01, gamma 0.99.
MATT_PPO=dict(ppo='matt',lr=1e-3,schedule='adaptive',desired_kl=.01,init_std=.3,std_range=(.05,.15),
    entropy_coef=.005,gamma=.99,lam=.95)
CRITIC_EXTRA=dict(legacy=0,actor=None,state=73)
SCRATCH_RECIPE=dict(lr=1e-4,std=.05,iterations=4000,freeze_normalizer=False,
    source='Fresh residual actor and privileged critic; original scratch PPO settings without imitation')


def audit_fresh_runner(runner, observations, history, skill, critic_kind='actor'):
    """Reject accidental checkpoint loading or fitted normalization in scratch runs."""
    actor,critic=runner.alg._raw_actor,runner.alg._raw_critic
    for model in (actor,critic):
        norm=model.obs_normalizer
        if int(norm.count)!=0 or torch.count_nonzero(norm._mean) or not torch.equal(norm._var,torch.ones_like(norm._var)):
            raise AssertionError('Scratch normalization is not fresh')
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise AssertionError('Nonfinite initial model')
    last=[m for m in actor.mlp.modules() if isinstance(m,torch.nn.Linear)][-1]
    if torch.count_nonzero(last.weight) or torch.count_nonzero(last.bias):
        raise AssertionError('Scratch residual output must start at zero')
    if runner.alg.optimizer.state or runner.current_learning_iteration!=0:
        raise AssertionError('Scratch optimizer/iteration is not fresh')
    if observations['actor'].shape[-1]!=76+75*history:
        raise AssertionError('Incorrect actor dimensions')
    actor_width=76+75*history
    critic_width=critic_widths(history,skill)[critic_kind]
    if observations['critic'].shape[-1]!=critic_width:
        raise AssertionError('Incorrect privileged critic dimensions')
    if critic_kind=='actor':
        torch.testing.assert_close(observations['critic'][:,:actor_width],observations['actor'],rtol=0,atol=0)
    return dict(actor='random hidden weights; zero residual output head',critic='random weights',
        actor_inputs=76+75*history,critic_inputs=critic_width,normalization='fresh, trainable',
        optimizer='empty',checkpoint_loaded=False,teacher_loaded=False,exact_actor_prefix=critic_kind=='actor',
        reference='Existing reference trajectories and predecessor-state banks retained')


def critic_widths(history,skill):
    legacy=200 if skill=='go-to' else 224
    return dict(legacy=legacy,actor=legacy+76+75*history,state=legacy+CRITIC_EXTRA['state'])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expand_actor(old,new,history):
    """Exact functional transfer: only the current proprioceptive frame is used."""
    offset=76+75*(history-1)
    for key,value in new.items():
        if key=='mlp.0.weight':
            value.zero_();value[:,offset:offset+75]=old[key]
        elif key.startswith('obs_normalizer.') and value.ndim and value.shape[-1]==76+75*history:
            value.fill_(0. if key.endswith('_mean') else 1.)
            value[:,offset:offset+75]=old[key]
        elif key in old and value.shape==old[key].shape:value.copy_(old[key])
    return new


def expand_critic(old,new,legacy,extra):
    """Exact functional transfer of the legacy critic: its columns are copied, the
    appended critic_state columns get zero weights and normalization from `extra`,
    the first observed batch, so they enter on the scale of the old ones."""
    var=extra.var(0,keepdim=True).clamp_min(1e-4);mean=extra.mean(0,keepdim=True)
    for key,value in new.items():
        if key=='mlp.0.weight':
            value.zero_();value[:,:legacy]=old[key]
        elif key.startswith('obs_normalizer.') and value.ndim and value.shape[-1]>legacy:
            value[:,:legacy]=old[key]
            value[:,legacy:]={'_mean':mean,'_var':var,'_std':var.sqrt()}[key.split('.')[-1]]
        elif key in old and value.shape==old[key].shape:value.copy_(old[key])
    return new


def critic_model(state,width,device):
    obs=TensorDict({'critic':torch.zeros(1,width,device=device)},batch_size=[1])
    m=MLPModel(obs,{'critic':['critic']},'critic',1,hidden_dims=(512,256,128),
        activation='elu',obs_normalization=True)
    m.to(device);m.load_state_dict(state);m.eval();m.requires_grad_(False)
    return m


def actor_model(state,width,device):
    obs=TensorDict({'actor':torch.zeros(1,width,device=device)},batch_size=[1])
    m=MLPModel(obs,{'actor':['actor']},'actor',23,hidden_dims=(512,256,128),
        activation='elu',obs_normalization=True,distribution_cfg=dict(SKILL_DIST))
    m.to(device);m.load_state_dict(state);m.eval();m.requires_grad_(False)
    return m


class Runner(MjlabOnPolicyRunner):
    def save(self,path,infos=None):
        super().save(path,{**(infos or {}),'observation_contract':self.table_contract,
            'critic_contract':getattr(self,'critic_contract',None),
            'domain_randomization':SPEC,'recipe':self.recipe,'initial_checkpoint':self.initial_checkpoint})
        write_json(Path(path).parent/'training-progress.json',dict(checkpoint=str(path),
            iteration=self.current_learning_iteration,updated_at=datetime.now(timezone.utc).isoformat()))


def evaluate(a):
    torch.manual_seed(a.seed)
    cfg=env_cfg(a.skill,a.envs,a.states,a.history,a.tier);cfg.seed=a.seed
    cfg.auto_reset=False
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        d=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        if d['infos']['observation_contract']!=contract(a.history):raise ValueError('Wrong observation contract')
        model=actor_model(d['actor_state_dict'],76+75*a.history,env.device)
        obs,_=env.reset();t=tk.task(env);physics=audit_actuation(env,'actuator_only')
        live=torch.ones(a.envs,dtype=torch.bool,device=env.device);valid=torch.zeros_like(live)
        sensor=torch.zeros_like(live);ever=torch.zeros_like(live)
        failures={k:0 for k in env.termination_manager.active_terms}
        # auto_reset remains enabled below: live masks ensure one trial per row.
        env.cfg.auto_reset=True
        with torch.no_grad():
            for _ in range(round(40/env.step_dt)):
                ready=t.switch_due & live
                physical=t.standing()[0]
                valid |= ready & physical;sensor |= ready;ever |= live&t.ended()&physical
                live &= ~ready
                if not live.any():break
                action=model(TensorDict({'actor':obs['actor']},batch_size=[a.envs]))
                obs,_,term,trunc,_=env.step(action)
                for key in failures:failures[key]+=int((live&env.termination_manager.get_term(key)).sum())
                live &= ~(term|trunc)
        result=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=sha(a.checkpoint),
            skill=a.skill,history=a.history,actor_inputs=76+75*a.history,tier=a.tier,seed=a.seed,
            episodes=a.envs,valid_handoff=float(valid.float().mean()),
            sensor_handoff=float(sensor.float().mean()),completion_ever=float(ever.float().mean()),
            unfinished=int(live.sum()),failures=failures,actuation_audit=physics,
            evaluation='Independent stage, held-out predecessor bank; original physical gates; first sensor handoff')
        write_json(a.out,result);print(json.dumps(result),flush=True)
    finally:env.close()


def train(a):
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'status.json').exists():raise RuntimeError('Run already exists')
    recipe=(SCRATCH_RECIPE if a.from_scratch else RECIPES[a.skill]).copy()
    iterations=a.iterations or recipe['iterations']
    if a.ppo=='matt':recipe.update(MATT_PPO,std=None)
    else:recipe.update(ppo='yours')
    recipe['critic']=a.critic
    metadata=dict(state='initializing',skill=a.skill,history=a.history,actor_inputs=76+75*a.history,
        envs=a.envs,iterations=iterations,seed=a.seed,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        pid=os.getpid(),recipe=recipe,checkpoint=str(a.checkpoint.resolve()) if a.checkpoint else None,
        states=str(a.states.resolve()),validation=str(a.validation.resolve()),
        observation_contract=contract(a.history),domain_randomization=SPEC,
        unchanged=['reward functions','PPO losses','hidden layers','action/controller rules'],
        initialization=('from_scratch' if a.from_scratch else
            'Selected actor transferred exactly; new input columns zero; matching critic loaded'
            +(' and widened (new columns zero)' if a.critic=='state' else '')+'; fresh optimizer'),
        teacher='None; PPO only',
        input_sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
            ROOT/'policies/table_observations.py',ROOT/'policies/table_domain_randomization.py']})
    def update(state,**kwargs):
        metadata.update(state=state,updated_at=datetime.now(timezone.utc).isoformat(),**kwargs)
        write_json(a.out/'status.json',metadata)
    inputs=a.out/'inputs';inputs.mkdir()
    for name in ('table_dr_train.py','table_observations.py','table_domain_randomization.py',
                 'five_task.py','five_research.py','five_putdown_research.py','five_actuation.py',
                 'five_train.py','motion.py','task.py','scene.py','five_stage.json'):
        shutil.copy2(ROOT/'policies'/name,inputs/name)
    env=None
    try:
        update('initializing');torch.manual_seed(a.seed)
        cfg=env_cfg(a.skill,a.envs,a.states,a.history,'train',critic=a.critic);cfg.seed=a.seed
        env=ManagerBasedRlEnv(cfg,device='cuda:0');vec=RslRlVecEnvWrapper(env,clip_actions=None)
        if a.ppo=='matt':
            ppo=motion.ppo_cfg(str(a.out),iterations)
            ppo.actor.distribution_cfg.update(init_std=MATT_PPO['init_std'],std_range=MATT_PPO['std_range'])
            for key in ('learning_rate','schedule','desired_kl','entropy_coef','gamma','lam'):
                if getattr(ppo.algorithm,key)!=MATT_PPO['lr' if key=='learning_rate' else key]:
                    raise AssertionError(f'task.ppo_cfg no longer matches Matt\'s {key}')
        else:
            ppo=ppo_cfg(a.skill,iterations)
            ppo.algorithm.learning_rate=recipe['lr'];ppo.algorithm.schedule='fixed'
            ppo.actor.distribution_cfg.update(init_std=recipe['std'],std_range=(recipe['std'],recipe['std']))
        ppo.seed=a.seed;ppo.save_interval=200
        resolved=dict(ppo=asdict(ppo),
            rewards={k:dict(function=v.func.__module__+'.'+v.func.__name__,weight=v.weight,params=v.params)
                     for k,v in cfg.rewards.items()},
            terminations={k:dict(function=v.func.__module__+'.'+v.func.__name__,params=v.params)
                          for k,v in cfg.terminations.items()})
        (a.out/'resolved-training.json').write_text(json.dumps(resolved,indent=2,default=str)+'\n')
        runner=Runner(vec,asdict(ppo),str(a.out),'cuda:0')
        runner.table_contract=contract(a.history);runner.recipe=recipe
        runner.initial_checkpoint=str(a.checkpoint.resolve()) if a.checkpoint else None
        actor=runner.alg._raw_actor
        if not a.from_scratch:
            old=validate_checkpoint(a.checkpoint)
            actor.load_state_dict(expand_actor(old['actor_state_dict'],actor.state_dict(),a.history))
            legacy=200 if a.skill=='go-to' else 224;critic=runner.alg._raw_critic
            if a.critic=='state':
                extra=vec.get_observations()['critic'][:,legacy:]
                critic.load_state_dict(expand_critic(old['critic_state_dict'],critic.state_dict(),legacy,extra))
            elif 'critic_state_dict' in old:critic.load_state_dict(old['critic_state_dict'])
        # A warm start copies the checkpoint's std; each PPO recipe sets its own.
        with torch.no_grad():
            actor.distribution.std_param.fill_(recipe['std'] if a.ppo=='yours' else MATT_PPO['init_std'])
        if recipe['freeze_normalizer']:actor.update_normalization=lambda observations:None
        obs=vec.get_observations();assert obs['actor'].shape[-1]==76+75*a.history
        runner.critic_contract=dict(id={'legacy':'legacy-privileged','actor':'actor-plus-legacy-privileged-v1',
                'state':'legacy-plus-table-state-v1'}[a.critic],
            width=obs['critic'].shape[-1],actor_prefix_width=76+75*a.history if a.critic=='actor' else 0,
            legacy_width=200 if a.skill=='go-to' else 224)
        if runner.critic_contract['width']!=critic_widths(a.history,a.skill)[a.critic]:
            raise AssertionError(f"Critic width {runner.critic_contract['width']}")
        metadata['critic_contract']=runner.critic_contract
        error=None
        if a.from_scratch:
            metadata['initialization_audit']=audit_fresh_runner(runner,obs,a.history,a.skill,a.critic)
        else:
            # Prove expansion has not changed the initial mapping.
            reference=actor_model(old['actor_state_dict'],75,'cuda:0')
            x=torch.randn(32,76+75*a.history,device='cuda:0');offset=76+75*(a.history-1)
            with torch.no_grad():
                y=actor(TensorDict({'actor':x},batch_size=[32]));z=reference(TensorDict({'actor':x[:,offset:offset+75]},batch_size=[32]))
                error=float((y-z).abs().max())
            if error>2e-5:raise AssertionError(f'Input expansion changed actor: {error}')
            del reference
            if a.critic=='state':
                width=obs['critic'].shape[-1];reference=critic_model(old['critic_state_dict'],legacy,'cuda:0')
                x=torch.randn(32,width,device='cuda:0')
                with torch.no_grad():
                    y=critic(TensorDict({'critic':x},batch_size=[32]));z=reference(TensorDict({'critic':x[:,:legacy]},batch_size=[32]))
                # Values reach thousands: compare relative to their scale (float32 rounding).
                critic_error=float((y-z).abs().max()/(z.abs().max()+1.));metadata['initial_critic_parity_max_error']=critic_error
                if critic_error>1e-5:raise AssertionError(f'Critic expansion changed critic: {critic_error}')
                del reference
        torch.manual_seed(a.seed);runner.current_learning_iteration=0
        runner.save(str(a.out/'initial.pt'));update('training',initial_actor_parity_max_error=error,
            actuation_audit=audit_actuation(env,'actuator_only'))
        runner.learn(iterations,init_at_random_ep_len=False)
        env.close();env=None;del runner,vec,actor;torch.cuda.empty_cache()
        if a.smoke_only:
            update('complete',scope='PPO smoke test only; checkpoint selection not performed')
            return
        update('evaluating')
        checkpoints=[a.out/'initial.pt']+sorted(a.out.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
        reports=[]
        for cp in checkpoints:
            output=cp.with_suffix('.dr.json')
            cmd=[sys.executable,__file__,'evaluate','--skill',a.skill,'--history',str(a.history),
                 '--checkpoint',str(cp),'--states',str(a.validation),'--out',str(output),
                 '--envs',str(a.eval_envs),'--seed','19011','--tier','dr']
            with cp.with_suffix('.dr.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            reports.append(json.loads(output.read_text()))
            write_json(a.out/'selection.json',reports)
        best=max(reports,key=lambda r:(r['valid_handoff'],r['completion_ever'],r['sensor_handoff']))
        for tier in ('nominal','sensors'):
            output=a.out/f'selected-{tier}.json'
            cmd=[sys.executable,__file__,'evaluate','--skill',a.skill,'--history',str(a.history),
                 '--checkpoint',best['checkpoint'],'--states',str(a.validation),'--out',str(output),
                 '--envs',str(a.eval_envs),'--seed','19012','--tier',tier]
            with (a.out/f'selected-{tier}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        selected=torch.load(best['checkpoint'],map_location='cpu',weights_only=False)
        actors=a.out/'actors';actors.mkdir(exist_ok=True)
        torch.save(dict(actor_state_dict=selected['actor_state_dict'],
            observation_contract=contract(a.history),source=best['checkpoint'],
            source_sha256=best['checkpoint_sha256'],domain_randomization=SPEC),actors/'policy.pt')
        update('complete',selected_checkpoint=best['checkpoint'],evaluation=best)
    except BaseException:
        update('failed',error=traceback.format_exc());raise
    finally:
        if env is not None:env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('train','evaluate'))
    p.add_argument('--skill',choices=RECIPES,required=True);p.add_argument('--history',type=int,choices=(1,4),required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--states',type=Path,required=True)
    p.add_argument('--from-scratch',action='store_true')
    p.add_argument('--critic',choices=CRITICS,help='default: actor from scratch, legacy when warm-starting')
    p.add_argument('--ppo',choices=('yours','matt'),default='yours')
    p.add_argument('--validation',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--envs',type=int,default=4096);p.add_argument('--iterations',type=int)
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--eval-envs',type=int,default=256);p.add_argument('--seed',type=int,default=120101)
    p.add_argument('--tier',choices=('dr','sensors','nominal'),default='dr')
    a=p.parse_args();torch.set_num_threads(1)
    if a.from_scratch and (a.mode!='train' or a.checkpoint):
        p.error('--from-scratch requires train mode and prohibits --checkpoint')
    if not a.from_scratch and a.checkpoint is None:
        p.error('--checkpoint is required unless training --from-scratch')
    if a.mode=='train' and a.validation is None:
        p.error('Training requires --validation')
    a.critic=a.critic or ('actor' if a.from_scratch else 'legacy')
    if not a.from_scratch and a.critic=='actor':
        p.error('Warm-starting loads the legacy critic: use --critic legacy or state')
    if a.mode=='train':train(a)
    else:evaluate(a)

if __name__=='__main__':main()
