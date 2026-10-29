"""Fixed-order composite controller: four frozen actors, five continuous phases.

There are no simulation resets, teleports, or state writes at phase handoffs.
Only the initial episode reset places the robot and box at their initial poses.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

os.environ.setdefault('MUJOCO_GL','egl')
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.termination_manager import TerminationTermCfg

import motion
import task as tk
import five_task as ft
from five_plans import OUT, CONFIG
from five_provenance import fingerprints
from five_train import write_json
from hier import FrozenSkill
from five_proprio import validate_checkpoint

ORDER = ('go-to','take','carry','put-down','return')
CHECKPOINTS = {name: OUT/name/'actors/proprio-policy.pt'
               for name in ('navigation', 'take', 'carry', 'put-down')}


class SequenceCommand(ft.FiveCommand):
    def __init__(self,cfg,env):
        # Initialize put-down's reference landmarks from its native reference.
        super().__init__(cfg,env)
        plans=[motion.Plan(str(OUT/s/'banks/plan.pt'),self.device,list(self.robot.joint_names)) for s in ORDER]
        self.phase_plans=plans
        maximum=max(int(p.sk['len'].max()) for p in plans)
        self.plan.sk={'len':torch.cat([p.sk['len'] for p in plans]) , **{
            k:torch.cat([torch.cat([p.sk[k],p.sk[k][:,-1:].expand(1,maximum-p.sk[k].shape[1],*p.sk[k].shape[2:])],dim=1) for p in plans])
            for k in plans[0].sk if k!='len'}}
        self.skill='sequence'
        self.force_start=tk.BEGIN
        self.curriculum=False
        self.completed=torch.zeros(self.num_envs,5,dtype=torch.bool,device=self.device)
        self.controller_done=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.phase_elapsed=torch.zeros(self.num_envs,device=self.device)
        self.transition_counts=torch.zeros(4,dtype=torch.long,device=self.device)
        self.max_state_jump=0.

    def _resample_command(self,env_ids):
        self.plan_ids[env_ids]=0
        self.skill='go-to'
        super()._resample_command(env_ids)
        self.skill='sequence'
        self.completed[env_ids]=False
        self.phase_elapsed[env_ids]=0.
        self.controller_done[env_ids]=False

    def standing(self):
        ok=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        for i,name in enumerate(ORDER):
            self.skill=name
            candidate,errors=super().standing()
            ok |= (self.plan_ids==i)&candidate
        self.skill='sequence'
        return ok,errors

    def _update_command(self,env_ids):
        if env_ids is not None:return
        tk._invalidate(self._env)
        next_clock=self.clock[:,0]+(~self.fresh).long()
        self.clock[:,0]=torch.minimum(next_clock,self.lengths()-1)
        self.phase_elapsed += self._env.step_dt*(~self.fresh)
        self.fresh[:]=False
        tk._invalidate(self._env)
        self.phase[:]=self.R('grip').long()
        ready=self.controller_ready() & self.ended()
        self.stood=torch.where(ready,self.stood+self._env.step_dt,0.)
        handoff=self.stood>=1.
        # Ground truth scores the result but never chooses a phase or clock.
        ok,_=self.standing()
        ids=handoff.nonzero().flatten()
        self.completed[ids,self.plan_ids[ids]] |= ok[ids]
        self.controller_done |= handoff&(self.plan_ids==4)
        self.success=self.controller_done & self.completed.all(-1)
        advance=(handoff&(self.plan_ids<4)).nonzero().flatten()
        if len(advance):self.advance(advance)
        self.metrics['success'][:]=self.success.float()

    def advance(self,ids):
        # Audit the actual simulator state, not merely the desired references.
        before_q=self._env.sim.data.qpos[ids].clone()
        before_v=self._env.sim.data.qvel[ids].clone()
        for i in range(4):self.transition_counts[i]+=(self.plan_ids[ids]==i).sum()
        self.plan_ids[ids]+=1
        self.clock[ids]=0
        self.fresh[ids]=True
        self.stood[ids]=0.
        self.success[ids]=False
        self.phase_elapsed[ids]=0.
        tk._invalidate(self._env)
        self.phase[:]=self.R('grip').long()
        assert torch.equal(before_q,self._env.sim.data.qpos[ids])
        assert torch.equal(before_v,self._env.sim.data.qvel[ids])


@dataclass(kw_only=True)
class SequenceCommandCfg(ft.FiveCommandCfg):
    def build(self,env):return SequenceCommand(self,env)


def phase_timeout(env):
    t=tk.task(env)
    # A completed return may keep standing during the final stability check.
    done=t.controller_done & (t.plan_ids==4)
    return (~done)&(t.phase_elapsed > (t.lengths()-1)*env.step_dt+12.)


def scoped_termination(env,name):
    t=tk.task(env)
    if name=='box_lost':return ((t.plan_ids>=1)&tk.t_box_lost(env))
    fn={'heading':ft.t_putdown_heading,'postrelease_collision':ft.t_postrelease_collision,
        'postrelease_posture':ft.t_putdown_posture}[name]
    return (t.plan_ids==3)&fn(env)


def sequence_cfg(n):
    cfg=ft.env_cfg('put-down',n,play=True)
    nav=ft.env_cfg('navigation',n,play=True)
    cfg.commands['task']=SequenceCommandCfg(plan=str(OUT/'put-down/banks/plan.pt'),skill='put-down',rsi_noise=False,random_hold=False)
    cfg.observations={'navigation':nav.observations['actor'],'manipulation':cfg.observations['actor']}
    cfg.rewards={}
    cfg.terminations.pop('success')
    cfg.terminations['phase_timeout']=TerminationTermCfg(func=phase_timeout)
    for name in ('box_lost','heading','postrelease_collision','postrelease_posture'):
        cfg.terminations[name]=TerminationTermCfg(func=scoped_termination,params={'name':name})
    cfg.episode_length_s=130.
    cfg.seed=10001
    return cfg


class CompositePolicy:
    """One callable controller, with a deterministic state machine in the command."""
    def __init__(self,env,obs):
        self.env=env
        for path in CHECKPOINTS.values():
            validate_checkpoint(path)
        self.actors={name:FrozenSkill(str(path),obs['navigation' if name=='navigation' else 'manipulation'].shape[-1],23,env.device,hidden=(512,256,128)) for name,path in CHECKPOINTS.items()}

    @torch.no_grad()
    def __call__(self,obs):
        stages=tk.task(self.env).plan_ids
        action=torch.empty(self.env.num_envs,23,device=self.env.device)
        for name,actor in self.actors.items():
            mask=((stages==0)|(stages==4)) if name=='navigation' else stages==ORDER.index(name)
            if mask.any():action[mask]=actor(obs['navigation' if name=='navigation' else 'manipulation'][mask])
        return action


def run(n,output,record=False,seed=10001):
    torch.manual_seed(seed);torch.set_num_threads(1)
    cfg=sequence_cfg(n);cfg.seed=seed
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        obs,_=env.reset();t=tk.task(env);policy=CompositePolicy(env,obs)
        live=torch.ones(n,dtype=torch.bool,device=env.device)
        completed=torch.zeros(n,5,dtype=torch.bool,device=env.device)
        final_time=torch.full((n,),-1.,device=env.device)
        finished=torch.zeros_like(live)
        stable=torch.zeros_like(live)
        failures={name:torch.zeros_like(live) for name in env.termination_manager.active_terms}
        failure_phase=torch.full((n,),-1,dtype=torch.long,device=env.device)
        records=[];transitions=[];last_phase=t.plan_ids.clone()
        for step in range(round(125/env.step_dt)):
            completed |= live[:,None]&t.completed
            now=step*env.step_dt
            final_time=torch.where(live&t.success&(final_time<0),now,final_time)
            held=live&(final_time>=0)&(now-final_time>=5.)
            # Match standalone evaluation: completion occurred, and the
            # physical endpoint conditions still hold after the extra dwell.
            stable |= held&t.standing()[0]
            finished |= held
            live &= ~held
            if not live.any():break
            if record and step%2==0:
                records.append(dict(qpos=env.sim.data.qpos[0].cpu().numpy().copy(),
                    mocap_pos=env.sim.data.mocap_pos[0].cpu().numpy().copy(),mocap_quat=env.sim.data.mocap_quat[0].cpu().numpy().copy(),
                    time=now,stage=int(t.plan_ids[0]),clock=int(t.clock[0,0])))
            before=t.plan_ids.clone()
            with torch.no_grad():obs,_,terminated,truncated,_=env.step(policy(obs))
            done=terminated|truncated
            for name in failures:failures[name] |= live&env.termination_manager.get_term(name).bool()
            failure_phase=torch.where(live&done,before,failure_phase)
            live &= ~done
            changed=live&(t.plan_ids!=before)
            if changed.any():
                event=dict(time=round(now+env.step_dt,3),counts={ORDER[i]:int((changed&(t.plan_ids==i)).sum()) for i in range(1,5)})
                transitions.append(event)
                print('TRANSITION',json.dumps(event),flush=True)
        report=dict(episodes=n,seconds_limit=125,final_hold_seconds=5,final_stability_definition="Endpoint conditions at five seconds after first full completion",
            full_sequence_completion=float(completed.all(-1).float().mean()),completed_and_stable=float(stable.float().mean()),
            phase_completion={name:float(completed[:,i].float().mean()) for i,name in enumerate(ORDER)},
            failures={name:float(mask.float().mean()) for name,mask in failures.items()},
            failure_phase={name:int((failure_phase==i).sum()) for i,name in enumerate(ORDER)},
            unfinished=int((~finished&(failure_phase<0)).sum()),
            transition_state_audit='qpos and qvel exactly unchanged by every phase switch',
            scope='Continuous end-to-end simulation, original start, fixed order, mean actions, no pushes; no phase resets',
            checkpoints={k:str(v) for k,v in CHECKPOINTS.items()},seed=seed,transitions=transitions)
        write_json(output/'evaluation.json',report)
        if record:
            render(env,records,report,output)
        print(json.dumps({k:v for k,v in report.items() if k!='transitions'},indent=2),flush=True)
        return report
    finally:env.close()


def render(env,records,report,output):
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image,ImageDraw
    model=env.sim.mj_model;motion.restyle(model)
    model.vis.global_.offwidth=960;model.vis.global_.offheight=720
    data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=720,width=960)
    camera=mujoco.MjvCamera();camera.azimuth=115;camera.elevation=-32;camera.distance=5.2;camera.lookat[:]=[.8,-.7,.65]
    path=output/'sequence.mp4'
    with imageio.get_writer(path,fps=25,codec='libx264',quality=7,pixelformat='yuv420p',macro_block_size=16,ffmpeg_params=['-movflags','+faststart']) as writer:
        for i,state in enumerate(records):
            data.qpos[:]=state['qpos'];data.mocap_pos[:]=state['mocap_pos'];data.mocap_quat[:]=state['mocap_quat'];mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=camera);im=Image.fromarray(renderer.render());d=ImageDraw.Draw(im)
            d.rectangle((0,0,960,72),fill='white');d.text((18,10),'Five-stage composite policy | continuous physics rollout',font=motion.font(22,True),fill='black')
            d.text((18,42),f"{state['stage']+1}/5  {ORDER[state['stage']]} | {state['time']:.2f}s | no resets between phases",font=motion.font(18),fill='black')
            if i==len(records)-1:
                d.rectangle((0,665,960,720),fill='white')
                d.text((18,680),'Full task completed' if report['full_sequence_completion']==1 else 'Task incomplete — see evaluation report',font=motion.font(20,True),fill='black')
            writer.append_data(np.asarray(im))
            if i%250==0:im.save(output/f'frame-{i:04d}.png')
        for _ in range(25):writer.append_data(np.asarray(im))
        im.save(output/'sequence-end.png')
    renderer.close()
    print('VIDEO',str(path.resolve()),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--envs',type=int,default=64);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--record',action='store_true');parser.add_argument('--seed',type=int,default=10001);args=parser.parse_args()
    if args.record and args.envs!=1:parser.error('Recording requires --envs 1')
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=dict(order=ORDER,created_at=datetime.now(timezone.utc).isoformat(),config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),sequence_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs={s:fingerprints(s) for s in ORDER},checkpoints={name:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for name,p in CHECKPOINTS.items()})
    write_json(args.out/'manifest.json',manifest)
    run(args.envs,args.out,args.record,args.seed)


if __name__=='__main__':main()
