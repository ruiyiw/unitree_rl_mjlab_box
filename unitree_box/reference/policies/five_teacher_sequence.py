"""Assemble and evaluate selected teacher-trained students as one continuous policy.

The bundle contains frozen 75-input actors and all five local references.
The existing sensor-only state machine selects actors; no teacher or estimator
is loaded. This module leaves historical controllers and artifacts unchanged.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
os.environ.setdefault('MUJOCO_GL','egl')
import torch
from mjlab.envs import ManagerBasedRlEnv
import task as tk
from five_sequence import ORDER,sequence_cfg
from five_sequence_bundle import BundledPolicy
from five_proprio import CONTRACT,validate_checkpoint
from five_provenance import SOURCES,fingerprints
from five_plans import OUT,CONFIG,POLICIES
from five_train import write_json

DEFAULT_RUN='proprio-teacher-scratch-4000-20261028'
HERE=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assemble(run_id,path):
    checkpoints={};actors={};evidence={}
    for skill in POLICIES:
        folder=OUT/skill/'runs'/run_id
        status=json.loads((folder/'status.json').read_text())
        if status['state']!='complete':raise ValueError(f'Incomplete source run: {skill}')
        for name in SOURCES:
            if (folder/'inputs'/name).read_bytes()!=(HERE/name).read_bytes():
                raise ValueError(f'Simulation/controller source changed: {skill}/{name}')
        if digest(CONFIG)!=status['config_sha256'] or digest(OUT/skill/'banks/plan.pt')!=status['bank_sha256']:
            raise ValueError(f'Training configuration/reference changed: {skill}')
        checkpoint=Path(status['selected_checkpoint'])
        state=validate_checkpoint(checkpoint)
        if state['actor_state_dict']['mlp.0.weight'].shape!=(512,75):
            raise ValueError('Sequence actor must have exactly 75 inputs')
        checkpoints[skill]=str(checkpoint.resolve());actors[skill]=state['actor_state_dict']
        evidence[skill]=dict(checkpoint_sha256=digest(checkpoint),
            initialization=status.get('initialization'),evaluation=status['evaluation'])
    payload=dict(format='five-stage-composite-proprio-v2',order=ORDER,observation_contract=CONTRACT,
        config_sha256=digest(CONFIG),controller_sha256=digest(HERE/'five_sequence.py'),
        bank_sha256={s:fingerprints(s)['bank_sha256'] for s in ORDER},actors=actors,
        checkpoint_paths=checkpoints,
        references={s:torch.load(OUT/s/'banks/plan.pt',map_location='cpu',weights_only=False) for s in ORDER},
        source_run=run_id,source_evidence=evidence,hardware_validated=False,
        transition='Reference ended plus one continuous second of joint/IMU settling; no world/contact gates',
        created_at=datetime.now(timezone.utc).isoformat())
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,path)
    write_json(path.with_suffix('.json'),{k:v for k,v in payload.items() if k not in ('actors','references')})
    inputs=path.parent/'controller';inputs.mkdir(exist_ok=True)
    for name in ('five_teacher_sequence.py','five_sequence.py','five_sequence_bundle.py','five_proprio.py'):
        shutil.copy2(HERE/name,inputs/name)
    print('BUNDLE',str(path.resolve()),flush=True)
    return payload


def evaluate(bundle,output,n=64,seed=10001,record=False,physics_mode=None,gravity_scale=1.):
    torch.set_num_threads(1);torch.manual_seed(seed)
    cfg=sequence_cfg(n);cfg.seed=seed
    payload=torch.load(bundle,map_location='cpu',weights_only=False)
    actuation_mode=physics_mode or payload.get('simulation_actuation_mode','legacy')
    from five_actuation import configure_actuation
    configure_actuation(cfg,actuation_mode,gravity_scale)
    # Capture native failure data before the environment's automatic reset.
    # This wrapper returns the unmodified termination result.
    box_failures=[];seen_box_failures=set()
    native_box_failure=cfg.terminations['box_lost'].func
    def audited_box_failure(env,**kwargs):
        result=native_box_failure(env,**kwargs)
        ids=[int(i) for i in result.nonzero().flatten().cpu().tolist() if int(i) not in seen_box_failures]
        if ids:
            t=tk.task(env);g=tk.geo(env)
            error=(t.R('box',1)[:,:3]-g.box).norm(dim=-1)
            for i in ids:
                box_failures.append(dict(environment=i,stage=ORDER[int(t.plan_ids[i])],
                    phase_seconds=float(t.phase_elapsed[i]),box_height_m=float(g.box[i,2]),
                    reference_error_m=float(error[i]),height_gate=bool(g.box[i,2]<.45),
                    reference_gate=bool(error[i]>tk.TERM_BOX),reference_limit_m=tk.TERM_BOX,
                    bilateral_contact=bool(g.grip[i]>0)))
            seen_box_failures.update(ids)
        return result
    cfg.terminations['box_lost'].func=audited_box_failure
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    output.mkdir(parents=True,exist_ok=True)
    try:
        obs,_=env.reset();term=tk.task(env);policy=BundledPolicy(env,obs,bundle)
        from five_actuation import audit_actuation
        actuation_audit=audit_actuation(env,actuation_mode)
        payload=torch.load(bundle,map_location='cpu',weights_only=False)
        live=torch.ones(n,dtype=torch.bool,device=env.device)
        completed=torch.zeros(n,5,dtype=torch.bool,device=env.device)
        entered=completed.clone();entered[:,0]=True
        finish_time=torch.full((n,),-1.,device=env.device)
        stable=torch.zeros_like(live);finished=torch.zeros_like(live)
        failure_phase=torch.full((n,),-1,device=env.device,dtype=torch.long)
        failures={name:torch.zeros_like(live) for name in env.termination_manager.active_terms}
        records=[];transitions=[];last_time=0.
        with torch.no_grad():
            for step in range(round(125/env.step_dt)):
                now=step*env.step_dt;last_time=now
                completed |= live[:,None]&term.completed
                finish_time=torch.where(live&term.success&(finish_time<0),now,finish_time)
                held=live&(finish_time>=0)&(now-finish_time>=5.)
                stable |= held&term.standing()[0];finished |= held;live &= ~held
                if not live.any():break
                if record:
                    records.append(dict(qpos=env.sim.data.qpos[0].cpu().numpy().copy(),
                        mocap_pos=env.sim.data.mocap_pos[0].cpu().numpy().copy(),
                        mocap_quat=env.sim.data.mocap_quat[0].cpu().numpy().copy(),
                        time=now,stage=int(term.plan_ids[0]),clock=int(term.clock[0,0])))
                before=term.plan_ids.clone()
                obs,_,terminated,truncated,_=env.step(policy(obs))
                done=terminated|truncated
                for name in failures:failures[name] |= live&env.termination_manager.get_term(name).bool()
                failure_phase=torch.where(live&done,before,failure_phase)
                live &= ~done
                changed=live&(term.plan_ids!=before)
                for stage in range(5):entered[:,stage] |= live&(term.plan_ids==stage)
                if changed.any():
                    delta=(term.robot.data.joint_pos-term.raw(0,'qpos')[:,7:]).abs().amax(-1)[changed]
                    xy=(term.robot.data.root_link_pos_w[:,:2]-term.pose(0)[:,:2]).norm(dim=-1)[changed]
                    event=dict(time_s=round(now+env.step_dt,3),
                        entered={ORDER[i]:int((changed&(term.plan_ids==i)).sum()) for i in range(1,5)},
                        next_reference_max_joint_error_rad=dict(mean=float(delta.mean()),max=float(delta.max())),
                        next_reference_xy_error_m=dict(mean=float(xy.mean()),max=float(xy.max())))
                    transitions.append(event);print('TRANSITION',json.dumps(event),flush=True)
        report=dict(bundle=str(bundle.resolve()),bundle_sha256=digest(bundle),source_run=payload['source_run'],
            episodes=n,seed=seed,seconds_limit=125,elapsed_simulation_s=last_time,
            final_hold_seconds=5,full_sequence_completion=float(completed.all(-1).float().mean()),
            completed_and_stable=float(stable.float().mean()),
            phase_entered={name:int(entered[:,i].sum()) for i,name in enumerate(ORDER)},
            phase_completion={name:float(completed[:,i].float().mean()) for i,name in enumerate(ORDER)},
            prefix_completion={name:float(completed[:,:i+1].all(-1).float().mean())
                               for i,name in enumerate(ORDER)},
            conditional_completion={name:(float(completed[:,:i+1].all(-1).sum()) /
                int(completed[:,:i].all(-1).sum()) if completed[:,:i].all(-1).any() else None)
                for i,name in enumerate(ORDER)},
            failures={name:int(v.sum()) for name,v in failures.items()},
            failure_phase={name:int((failure_phase==i).sum()) for i,name in enumerate(ORDER)},
            unfinished=int((~finished&(failure_phase<0)).sum()),transitions=transitions,
            box_failure_details=box_failures,evaluation_driver_sha256=digest(__file__),
            checkpoints=payload['checkpoint_paths'],observation_contract=CONTRACT,
            transition_state_audit='Existing controller asserts qpos/qvel unchanged at every handoff; previous residual preserved',
            scope='Continuous physics from original start; sensor-only switches, no stage resets, mean actions, no pushes',
            evaluation_randomization='None: fixed initial pose, layout, physics parameters and uncorrupted actor observations. Batch contact numerics can still vary.',
            hardware_validated=False)
        report['simulation_actuation_mode']=actuation_mode
        report['actuation_audit']=actuation_audit
        write_json(output/'evaluation.json',report)
        if record:render(env,records,report,output)
        print(json.dumps({k:v for k,v in report.items() if k!='transitions'},indent=2),flush=True)
        return report
    finally:env.close()


def render(env,records,report,output):
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image,ImageDraw
    import motion
    model=env.sim.mj_model;motion.restyle(model)
    model.vis.global_.offwidth=960;model.vis.global_.offheight=720
    data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=720,width=960)
    camera=mujoco.MjvCamera();camera.azimuth=115;camera.elevation=-32
    camera.distance=5.2;camera.lookat[:]=[.8,-.7,.65]
    target=output/'sequence.mp4';temporary=output/'sequence.pending.mp4'
    try:
        with imageio.get_writer(temporary,fps=50,codec='libx264',quality=7,pixelformat='yuv420p',
                macro_block_size=16,ffmpeg_params=['-movflags','+faststart']) as writer:
            for i,state in enumerate(records):
                data.qpos[:]=state['qpos'];data.mocap_pos[:]=state['mocap_pos'];data.mocap_quat[:]=state['mocap_quat']
                mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera)
                im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,960,75),fill='white')
                draw.text((18,10),'75-input actors | continuous five-stage controller',font=motion.font(22,True),fill='black')
                draw.text((18,43),f"{state['stage']+1}/5 {ORDER[state['stage']]} | {state['time']:.2f} s | no stage resets",
                    font=motion.font(18),fill='black')
                if i==len(records)-1:
                    draw.rectangle((0,665,960,720),fill='white')
                    reasons=', '.join(k for k,v in report['failures'].items() if v)
                    failed_stages=', '.join(s for s in ORDER if report['phase_completion'][s]<1.)
                    label='Full sequence completed' if report['full_sequence_completion']==1 else 'Incomplete: '+(reasons or failed_stages+' physical checks')
                    draw.text((18,680),label,font=motion.font(19,True),fill='black')
                writer.append_data(np.asarray(im))
                if i==len(records)//2:im.save(output/'sequence-preview.png')
            im.save(output/'sequence-end.png')
        temporary.replace(target)
    finally:renderer.close()
    print('VIDEO',str(target.resolve()),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['build','evaluate'])
    p.add_argument('--run-id',default=DEFAULT_RUN);p.add_argument('--bundle',type=Path)
    p.add_argument('--out',type=Path);p.add_argument('--envs',type=int,default=64)
    p.add_argument('--seed',type=int,default=10001);p.add_argument('--record',action='store_true')
    a=p.parse_args();bundle=a.bundle or OUT/'sequence/runs'/a.run_id/'policy.pt'
    if a.record and a.envs!=1:p.error('Recording requires --envs 1')
    if a.mode=='build':assemble(a.run_id,bundle)
    else:
        if a.out is None:p.error('Evaluation requires --out')
        evaluate(bundle,a.out,a.envs,a.seed,a.record)


if __name__=='__main__':main()
