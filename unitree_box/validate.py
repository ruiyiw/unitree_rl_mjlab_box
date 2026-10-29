"""CPU parity, sensor-switching and reference-timing regression checks."""
import json
from pathlib import Path
import numpy as np
import torch
from . import ACTORS, MOTOR_IDS, ORDER, POLICY_DIR
from .runtime import Core, build_core, reference_path, verify


def validate(output,samples=256):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    manifest=verify();reference_path()
    from five_sequence_bundle import StoredActor
    from five_proprio import observation as source_observation
    payload=torch.load(POLICY_DIR/'policy.pt',weights_only=False,map_location='cpu')
    torch.set_num_threads(1)
    library=build_core(output)
    default=np.asarray(manifest['interface']['default_joint_positions'],np.float32)
    scale=np.asarray(manifest['interface']['action_scale'],np.float32)
    ids=np.asarray(MOTOR_IDS)
    rng=np.random.default_rng(20261029);results={}
    actors={name:StoredActor(payload['actors'][name],75,'cpu') for name in ACTORS}
    for stage,name in enumerate(ORDER):
        bank=payload['references'][name];plan=bank['plan']
        previous=np.zeros(23,np.float32); max_obs=max_action=max_target=0.
        with Core(library,stage) as core:
            for i in range(samples):
                telemetry=rng.normal(size=65).astype(np.float32)
                telemetry[:4]/=np.linalg.norm(telemetry[:4])
                telemetry[7+ids]=default+rng.uniform(-.5,.5,23)
                obs,action,target,state=core.step(telemetry)
                w,x,y,z=telemetry[:4]
                gravity=np.asarray([2*(w*y-x*z),-2*(y*z+w*x),-1+2*(x*x+y*y)],np.float32)
                expected=source_observation(*[torch.from_numpy(v[None]) for v in
                    (telemetry[4:7],gravity,telemetry[7+ids],telemetry[36+ids],previous,default)]).numpy()[0]
                ref=actors[ACTORS[stage]](torch.from_numpy(obs[None])).numpy()[0]
                qcmd=plan['qcmd'][0,min(int(state[1])+1,int(plan['len'][0])-1)].numpy()
                np.testing.assert_allclose(obs,expected,atol=2e-6,rtol=1e-6)
                np.testing.assert_allclose(action,ref,atol=1e-4,rtol=1e-4)
                np.testing.assert_allclose(target,qcmd+scale*ref,atol=1e-4,rtol=1e-4)
                max_obs=max(max_obs,float(abs(obs-expected).max()))
                max_action=max(max_action,float(abs(action-ref).max()))
                max_target=max(max_target,float(abs(target-(qcmd+scale*ref)).max()))
                previous=action
            core.reset()
            obs,_,_,state=core.step(telemetry)
            assert not obs[52:].any() and tuple(state)==(stage,0,0)
            for invalid in (0.,float('nan')):
                bad=telemetry.copy();bad[:4]=invalid
                try:core.step(bad)
                except RuntimeError:pass
                else:raise AssertionError('Invalid IMU accepted')
        results[name]={'samples':samples,'max_observation_error':max_obs,
            'max_action_error':max_action,'max_target_error':max_target}
    # Run the actual archived SequenceCommand._update_command against a tiny
    # sensor-only adapter, and compare every C++ clock and handoff. No physics or
    # copied update equations are used as the reference for this regression.
    from five_sequence import SequenceCommand
    from five_proprio import settled
    from types import SimpleNamespace
    class Reference:
        def __init__(self):
            self._env=SimpleNamespace(step_dt=.02)
            self.clock=torch.zeros(1,1,dtype=torch.long);self.fresh=torch.zeros(1,dtype=torch.bool)
            self.phase_elapsed=torch.zeros(1);self.phase=torch.zeros(1,dtype=torch.long)
            self.stood=torch.zeros(1);self.plan_ids=torch.zeros(1,dtype=torch.long)
            self.completed=torch.zeros(1,5,dtype=torch.bool);self.controller_done=torch.zeros(1,dtype=torch.bool)
            self.metrics={'success':torch.zeros(1)};self.success=torch.zeros(1,dtype=torch.bool)
        def lengths(self):return torch.tensor([int(payload['references'][ORDER[int(self.plan_ids[0])]]['plan']['len'][0])])
        def R(self,key):return torch.zeros(1)
        def ended(self):return self.clock[:,0]>=self.lengths()-1
        def controller_ready(self):
            end=payload['references'][ORDER[int(self.plan_ids[0])]]['plan']['qpos'][0,-1,7:]
            return settled(self.obs,end-torch.from_numpy(default))
        def standing(self):return torch.ones(1,dtype=torch.bool),None
        def advance(self,ids):
            self.plan_ids[ids]+=1;self.clock[ids]=0;self.fresh[ids]=True
            self.stood[ids]=0.;self.phase_elapsed[ids]=0.;self.success[ids]=False
    ref=Reference();transitions=[];last_stage=0;previous=np.zeros(23,np.float32)
    telemetry=np.zeros(65,np.float32);telemetry[0]=1
    with Core(library) as core:
        for tick in range(6500):
            stage=int(ref.plan_ids[0]); plan=payload['references'][ORDER[stage]]['plan']
            telemetry[7+ids]=plan['qpos'][0,-1,7:].numpy()
            # Break a near-complete dwell to exercise consecutive settling.
            telemetry[4]=.3 if tick%113==0 else 0
            ref.obs=torch.from_numpy(np.concatenate((telemetry[4:7],[0,0,-1],telemetry[7+ids]-default,telemetry[36+ids],previous)).astype(np.float32)[None])
            if tick:SequenceCommand._update_command(ref,None)
            obs,action,target,state=core.step(telemetry)
            assert tuple(state)==(int(ref.plan_ids[0]),int(ref.clock[0,0]),int(ref.controller_done[0])), (tick,state,ref.clock)
            np.testing.assert_array_equal(obs[52:],previous)
            current=int(state[0]);current_plan=payload['references'][ORDER[current]]['plan']
            qcmd=current_plan['qcmd'][0,min(int(state[1])+1,int(current_plan['len'][0])-1)].numpy()
            np.testing.assert_allclose(target,qcmd+scale*action,atol=1e-6)
            if current!=last_stage:transitions.append({'tick':tick,'stage':ORDER[current]});last_stage=current
            previous=action
            if state[2]:break
        else:raise AssertionError('Sequence never completed')
    assert len(transitions)==4
    report={'actors':results,'sequence_ticks':tick+1,'transitions':transitions,
            'archived_source_controller_parity':True,'reset_and_invalid_imu':True,
            'hardware_tested':False,'physics_tested':False}
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return report
