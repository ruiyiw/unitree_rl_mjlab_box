"""Deployable 151/376-value observations and documented reset randomization.

Actor order: 76 current reference/controller values, then oldest-first 75-value
proprioceptive frames. No measured base position, velocity, box or contacts are
used by actor_observation. Plan references are local controller memory.
"""
from dataclasses import dataclass
import math
from pathlib import Path

import torch
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_mul

import motion
import task as tk
import five_task as ft
import five_handoff as hf
from five_research import ResearchCommand, ResearchCommandCfg, research_cfg
from table_domain_randomization import configure, uniform
import table_domain_randomization as tdr


def contract(history):
    if history not in (1,4):raise ValueError(history)
    return dict(id='g1-table-history-v1',actor_dim=76+75*history,history=history,
        current_terms=dict(ref_joints=46,ref_clock=2,plan_pelvis=12,plan_gravity=3,
                           plan_grip=3,arm_pd_error=10),
        history_terms=dict(gyro=3,gravity=3,joint_pos=23,joint_vel=23,actions=23),
        order='current terms, then frame-major history oldest first',
        position_feedback=False,critic_only=['base position','base velocity','box state','contacts'],
        history_initialization='repeat first reading, zero action history',
        action='unchanged original PlanJointAction; last residual preserved by controller',
        plan_grip_definition='reference holding flag; reference lift clearance / 0.04 clipped to [0,1]; planned 18 N per hand when holding, divided by 18')


class TableCommand(ResearchCommand):
    """Keep existing reward/success logic; replace only reset and actor inputs."""
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        n=self.num_envs;dev=self.device
        self.curriculum=False  # ev_push already applies its explicit level.
        self.force_start=tk.BEGIN
        self.start_kind=torch.zeros(n,dtype=torch.long,device=dev)
        self.heading_origin=torch.zeros(n,device=dev)
        self.heading_bias=torch.zeros(n,device=dev)
        self.heading_drift=torch.zeros(n,device=dev)
        self.gyro_bias=torch.zeros(n,3,device=dev)
        self.gravity_bias=torch.zeros(n,3,device=dev)
        self.sensor_previous=torch.zeros(n,86,device=dev) # 75 + heading + PD error 10
        self.sensor_current=torch.zeros_like(self.sensor_previous)
        self.delayed=torch.zeros_like(self.sensor_previous)
        self.history=torch.zeros(n,cfg.history,75,device=dev)
        self.delay_lag=torch.zeros(n,dtype=torch.long,device=dev)
        self.sensor_fresh=torch.ones(n,dtype=torch.bool,device=dev)
        self.sensor_step=torch.full((n,),-1,dtype=torch.long,device=dev)
        self.reset_count=torch.zeros(n,dtype=torch.long,device=dev)
        self.lower_ids=[i for i,name in enumerate(self.robot.joint_names)
                        if any(s in name for s in ('hip','knee','ankle','waist'))]
        self.ankle_ids=[i for i,name in enumerate(self.robot.joint_names) if 'ankle' in name]
        self.success_sum=0.;self.success_count=0

    def _write_reference(self,ids):
        q=self.world(0,'qpos',ids=ids);v=self.world(0,'qvel',ids=ids)
        self.robot.write_root_state_to_sim(torch.cat((q[:,:7],v[:,:6]),-1),env_ids=ids)
        self.robot.write_joint_state_to_sim(q[:,7:],v[:,6:],env_ids=ids)
        b=self.world(0,'box',ids=ids);bv=self.world(0,'box_lin',ids=ids)
        w=torch.zeros(len(ids),3,device=self.device);w[:,2]=self.raw(0,'box_wz',ids=ids)
        self.boxe.write_root_state_to_sim(torch.cat((b[:,:3],tk.yaw_quat(b[:,3]),bv,w),-1),env_ids=ids)

    def _restore(self,ids):
        if not len(ids):return
        sample=torch.randint(self.sample_count,(len(ids),),device=self.device)
        self.sample_ids[ids]=sample
        origin=self._env.scene.env_origins[ids]
        for key in hf.SIM_FIELDS:
            value=self.samples[key][sample].clone()
            if key=='qpos':
                for adr in hf.model_signature(self._env)['free_positions']:value[:,adr:adr+3]+=origin
            elif key=='mocap_pos':value+=origin[:,None]
            getattr(self._env.sim.data,key)[ids]=value
        for key in hf.ACTION_FIELDS:getattr(self._env.action_manager,key)[ids]=self.samples[key][sample]
        for key in hf.TARGET_FIELDS:getattr(self.robot.data,key)[ids]=self.samples[key][sample]

    def _resample_command(self,ids):
        # Preserve the existing curriculum thresholds, but aggregate one 4096-
        # episode window rather than changing level on every tiny reset batch.
        # Random-frame starts are excluded: a late reference frame succeeds easily.
        done=ids[self.start_kind[ids]!=2]
        if self.cfg.tier=='train' and len(done):
            self.success_sum+=float(self.success[done].sum());self.success_count+=len(done)
            if self.success_count>=self.num_envs:
                rate=self.success_sum/self.success_count
                self.level=max(0.,min(1.,self.level+(.05 if rate>.7 else -.05 if rate<.3 else 0.)))
                self.success_sum=0.;self.success_count=0
        ft.FiveCommand._resample_command(self,ids)
        self.switch_timer[ids]=0.;self.switch_due[ids]=False
        n=len(ids);dev=self.device;train=self.cfg.tier=='train'
        u=torch.rand(n,device=dev)
        hand=.6 if self.skill in ('carry','put-down') else .3
        kind=torch.where(u<.2,0,torch.where(u<.2+hand,1,2)) if train else torch.full((n,),0 if self.skill=='go-to' else 1,device=dev)
        self.start_kind[ids]=kind;self.sample_ids[ids]=-1
        rsi=ids[kind==2]
        if len(rsi):self.clock[rsi,0]=(torch.rand(len(rsi),device=dev)*(self.lengths(ids=rsi)-1)).long()
        self._write_reference(ids)
        self.robot.data.joint_pos_target[ids]=self.raw(0,'qcmd',ids=ids)
        self._restore(ids[kind==1])
        # Returning to go-to starts a new table task: its box is replaced on A.
        # The predecessor's robot/controller state remains intact.
        if self.skill=='go-to':
            b=self.world(0,'box',ids=ids);z=torch.zeros(n,6,device=dev)
            self.boxe.write_root_state_to_sim(torch.cat((b[:,:3],tk.yaw_quat(b[:,3]),z),-1),env_ids=ids)
        m=self._env.sim.mj_model;d=self._env.sim.data
        rq=int(m.joint('robot/floating_base_joint').qposadr[0]);rv=int(m.joint('robot/floating_base_joint').dofadr[0])
        bq=int(m.joint('box/box_joint').qposadr[0]);bv=int(m.joint('box/box_joint').dofadr[0])
        q=d.qpos[ids].clone();v=d.qvel[ids].clone()
        if train:
            xy_limit=torch.where(kind==0,.05 if self.skill=='go-to' else .02,.02)
            yaw_limit=torch.where(kind==0,math.radians(5 if self.skill=='go-to' else 2),.05)
            extra=(kind==1)&(self.skill in ('carry','put-down'))
            xy_limit=torch.where(extra,.05,torch.where(kind==1,0.,xy_limit))
            yaw_limit=torch.where(extra,math.radians(8),torch.where(kind==1,0.,yaw_limit))
            delta=uniform((n,2),-1.,1.,dev)*xy_limit[:,None]
            yaw=uniform((n,),-1.,1.,dev)*yaw_limit
            oldxy=q[:,rq:rq+2].clone();q[:,rq:rq+2]+=delta
            q[:,rq+3:rq+7]=quat_mul(tk.yaw_quat(yaw),q[:,rq+3:rq+7])
            held=extra & (self.raw(0,'grip',ids=ids)>0)
            q[held,bq:bq+2]=oldxy[held]+delta[held]+tk.rot2(q[held,bq:bq+2]-oldxy[held],yaw[held])
            q[held,bq+3:bq+7]=quat_mul(tk.yaw_quat(yaw[held]),q[held,bq+3:bq+7])
            v[:,rv:rv+2]=tk.rot2(v[:,rv:rv+2],yaw)
            v[held,bv:bv+2]=tk.rot2(v[held,bv:bv+2],yaw[held])
            leg_amp=torch.where(extra,.05,torch.where(kind==2,.03,0.))
            joint_q=[int(m.joint('robot/'+self.robot.joint_names[i]).qposadr[0]) for i in self.lower_ids]
            joint_v=[int(m.joint('robot/'+self.robot.joint_names[i]).dofadr[0]) for i in self.lower_ids]
            leg_columns=torch.tensor(['waist' not in self.robot.joint_names[i]
                                      for i in self.lower_ids],device=dev)
            amp=leg_amp[:,None]*torch.where(extra[:,None],True,leg_columns[None])
            q[:,joint_q]+=uniform((n,len(joint_q)),-1.,1.,dev)*amp
            v[:,joint_v]+=uniform((n,len(joint_v)),-.3,.3,dev)*extra[:,None]
            v[:,rv:rv+3]+=uniform((n,3),-.1,.1,dev)*extra[:,None]
            self.robot.data.joint_pos_target[ids]+=uniform((n,23),-.03,.03,dev)*extra[:,None]
        d.qpos[ids]=q;d.qvel[ids]=v
        # Plan frame is fixed for the episode, initialized from its start pose.
        raw=self.raw(0,'qpos',ids=ids)
        fy=tk.wrap(tk.yaw_of(q[:,rq+3:rq+7])-tk.yaw_of(raw[:,3:7]))
        self.frame[ids,2]=fy
        self.frame[ids,:2]=q[:,rq:rq+2]-tk.rot2(raw[:,:2],fy)
        self.heading_origin[ids]=tk.yaw_of(q[:,rq+3:rq+7])-tk.yaw_of(raw[:,3:7])
        # Tables remain in the fixed layout, not moved with the anchored plan.
        for name,pose in [('cart',self.setup['source_table_pose']),('shelf',self.setup['destination_table_pose'])]:
            pos=self._env.scene.env_origins[ids].clone();pos[:,:2]+=torch.tensor(pose[:2],device=dev)
            yaw=torch.full((n,),pose[2],device=dev)
            if self.cfg.tier in ('train','dr'):
                pos[:,:2]+=uniform((n,2),-.01,.01,dev);yaw+=uniform((n,),-math.pi/180,math.pi/180,dev)
            self._env.scene[name].write_mocap_pose_to_sim(torch.cat((pos,tk.yaw_quat(yaw)),-1),ids)
        if self.skill=='take' and self.cfg.tier in ('train','dr'):
            q=d.qpos[ids].clone();q[:,bq:bq+2]+=uniform((n,2),-.015,.015,dev)
            yaw=uniform((n,),-math.radians(2),math.radians(2),dev)
            q[:,bq+3:bq+7]=quat_mul(tk.yaw_quat(yaw),q[:,bq+3:bq+7]);d.qpos[ids]=q
        corrupt=self.cfg.tier!='nominal'
        self.robot.data.encoder_bias[ids]=uniform((n,23),-.01,.01,dev) if corrupt else 0.
        if corrupt:
            self.robot.data.encoder_bias[ids[:,None],self.ankle_ids]=uniform((n,len(self.ankle_ids)),-.02,.02,dev)
        self.gyro_bias[ids]=uniform((n,3),-.02,.02,dev) if corrupt else 0.
        self.gravity_bias[ids]=uniform((n,3),-.025,.025,dev) if corrupt else 0.
        self.heading_drift[ids]=uniform((n,),-math.radians(.2),math.radians(.2),dev) if corrupt else 0.
        self.heading_bias[ids]=(uniform((n,),-math.pi/180,math.pi/180,dev)*(kind==1)) if corrupt else 0.
        self.sensor_fresh[ids]=True;self.sensor_step[ids]=-1;self.reset_count[ids]+=1
        self.delay_lag[ids]=torch.randint(2,(n,),device=dev) if self.cfg.tier in ('train','dr') else 0
        # Initialize motor delay from the preceding command, rather than
        # silently making the first command of every episode delay-free.
        for group in self.robot._builtin_group._delayed_groups:
            buffer=group.delay_buffer
            targets=getattr(self.robot.data,group.target_attr)[:,group.target_ids]
            if not buffer.is_initialized:buffer.append(targets)
            buffer.backfill(targets,ids)
            buffer._current_lags[ids]=torch.randint(4,(n,),device=dev)
        self.just_restored[ids]=kind==1
        tk._invalidate(self._env)

    def sensors(self):
        env=self._env;step=env.common_step_counter
        ids=(self.sensor_step!=step).nonzero().flatten()
        if not len(ids):return self.delayed
        r=self.robot.data;dev=self.device;n=len(ids)
        gyro=env.scene['robot/imu_ang_vel'].data[ids].clone()
        gravity=r.projected_gravity_b[ids].clone()
        jp=r.joint_pos_biased[ids]-r.default_joint_pos[ids]
        jv=r.joint_vel[ids].clone();actions=tk.obs_last_joint_action(env)[ids].clone()
        yaw=tk.wrap(tk.yaw_of(r.root_link_quat_w[ids])-self.heading_origin[ids])
        pd=(r.joint_pos_target[ids]-r.joint_pos_biased[ids])[:,self.arm_ids]
        fresh=self.sensor_fresh[ids]
        actions[fresh]=0.
        if self.cfg.tier!='nominal':
            gyro+=self.gyro_bias[ids]+uniform((n,3),-.2,.2,dev)
            gravity+=self.gravity_bias[ids]+uniform((n,3),-.05,.05,dev)
            jp+=uniform((n,23),-.01,.01,dev);jv+=uniform((n,23),-1.5,1.5,dev)
            pd+=uniform((n,10),-.01,.01,dev)
            yaw+=self.heading_bias[ids]+self.heading_drift[ids]*env.episode_length_buf[ids]*env.step_dt+torch.randn(n,device=dev)*math.radians(.2)
        current=torch.cat((gyro,gravity,jp,jv,actions,yaw[:,None],pd),-1)
        prev=self.sensor_current[ids].clone();prev[fresh]=current[fresh]
        self.sensor_previous[ids]=prev;self.sensor_current[ids]=current
        if self.cfg.tier in ('train','dr'):
            refresh=torch.rand(n,device=dev)>.95
            self.delay_lag[ids]=torch.where(refresh,torch.randint(2,(n,),device=dev),self.delay_lag[ids])
        self.delayed[ids]=torch.where(self.delay_lag[ids,None].bool(),prev,current)
        h=self.history[ids].clone()
        h[:,:-1]=h[:,1:].clone();h[:,-1]=self.delayed[ids,:75]
        h[fresh]=self.delayed[ids[fresh],None,:75]
        self.history[ids]=h;self.sensor_fresh[ids]=False;self.sensor_step[ids]=step
        return self.delayed


@dataclass(kw_only=True)
class TableCommandCfg(ResearchCommandCfg):
    history:int=4
    tier:str='train'
    def build(self,env):return TableCommand(self,env)


def actor_observation(env):
    t=tk.task(env);r=t.robot.data;s=t.sensors()
    q=t.raw(0,'qpos');qd=t.raw(0,'qvel')
    refs=torch.cat((q[:,7:]-r.default_joint_pos,qd[:,6:]),-1)
    clock=torch.stack((t.clock[:,0]/(t.lengths()-1).clamp_min(1),t.ended().float()),-1)
    out=torch.cat((refs,clock,plan_features(env,s[:,75]),s[:,76:86],t.history.flatten(1)),-1)
    if out.shape[-1]!=76+75*t.cfg.history:raise AssertionError(out.shape)
    return out


def plan_features(env,yaw):
    """Planned pelvis/heading at 0/10/25 frames in the estimated heading, reference gravity, planned grip (18)."""
    t=tk.task(env);q=t.raw(0,'qpos')
    # Position comes ONLY from the plan, not r.root_link_pos_w.
    parts=[]
    for ahead in (0,10,25):
        p=t.raw(0,'qpos',ahead)
        xy=tk.rot2(p[:,:2]-q[:,:2],-yaw)
        error=tk.wrap(tk.yaw_of(p[:,3:7])-yaw)
        parts.append(torch.cat((xy,torch.sin(error)[:,None],torch.cos(error)[:,None]),-1))
    gravity=torch.zeros(env.num_envs,3,device=env.device);gravity[:,2]=-1.
    gravity=quat_apply_inverse(q[:,3:7],gravity)
    holding=t.raw(0,'grip').float()
    box=t.raw(0,'box');bottom=box[:,2]-float(t.setup['box_dimensions_m'][2])/2
    source=float(t.setup['source_top_m']);dest=float(t.setup['destination_top_m'])
    support=source if t.skill in ('go-to','take') else dest
    share=((bottom-support)/.04).clamp(0.,1.)*holding
    grip=torch.stack((holding,share,holding),-1) # nominal 18 N / 18 N
    return torch.cat((*parts,gravity,grip),-1)


def _per_env(x,n):
    """A model field indexed per world; unrandomized fields have a single world."""
    return x.expand(n,*x.shape[1:]) if x.shape[0]==1 else x


def critic_state(env):
    """Privileged additions to the legacy critic (73): what the actor acts on but
    the legacy critic lacks. Plan features in the actor's heading estimate and
    clean arm PD errors (28); the actor's persistent sensor errors and delays
    (32); episode physics randomization (13)."""
    t=tk.task(env);r=t.robot.data;s=t.sensors();n=env.num_envs
    plan=plan_features(env,s[:,75])
    pd=(r.joint_pos_target-r.joint_pos)[:,t.arm_ids]
    true_yaw=tk.wrap(tk.yaw_of(r.root_link_quat_w)-t.heading_origin)
    heading_error=tk.wrap(s[:,75]-true_yaw)
    groups=t.robot._builtin_group._delayed_groups
    motor_lag=(torch.stack([g.delay_buffer.current_lags.float() for g in groups],-1).mean(-1)
               if groups else torch.zeros(n,device=env.device))
    sensor=torch.cat((r.encoder_bias,t.gyro_bias,t.gravity_bias,heading_error[:,None],
                      t.delay_lag.float()[:,None],motor_lag[:,None]),-1)
    ix=tdr._indices(env);m=env.sim.model;default=env.sim.get_default_field
    box,torso=ix['box'],ix['torso']
    gain=_per_env(m.actuator_gainprm[...,0],n);bias=_per_env(m.actuator_biasprm[...,2],n)
    g0=default('actuator_gainprm')[:,0];b0=default('actuator_biasprm')[:,2]
    kp=(gain[:,g0!=0]/g0[g0!=0]).mean(-1);kd=(bias[:,b0!=0]/b0[b0!=0]).mean(-1)
    friction=_per_env(m.geom_friction,n)
    physics=torch.stack((_per_env(m.body_mass[:,box],n),
        *(_per_env(m.body_ipos[:,box],n)-default('body_ipos')[box]).unbind(-1),
        _per_env(m.body_mass[:,torso],n)-default('body_mass')[torso],
        friction[:,ix['palms'][0],0],friction[:,ix['palms'][0],1],friction[:,ix['feet'][0],0],
        friction[:,ix['tables'][0],0],friction[:,ix['tables'][1],0],
        _per_env(m.body_gravcomp[:,ix['arm_bodies'][0]],n),kp,kd),-1)
    return torch.cat((plan,pd,sensor,physics),-1)


CRITICS=('legacy','actor','state')


def env_cfg(skill,n,states,history,tier,actor_in_critic=False,critic=None):
    # critic: 'legacy' privileged (200/224); 'actor' = actor inputs + legacy
    # (576/600); 'state' = legacy + critic_state (273/297).
    critic=critic or ('actor' if actor_in_critic else 'legacy')
    if critic not in CRITICS:raise ValueError(critic)
    # Preserve the existing per-stage reward/termination recipe verbatim.
    cfg=(ft.env_cfg(skill,n,play=False) if skill=='go-to'
         else research_cfg(skill,n,Path(states),'sensor_fixed'))
    cfg=configure(cfg,tier)
    cfg.commands['task']=TableCommandCfg(plan=cfg.commands['task'].plan,skill=skill,
        states=str(Path(states).resolve()),states_skill=skill,history=history,tier=tier,
        rsi_noise=False,random_hold=(skill=='go-to' and tier=='train'),
        # Training success is the physical hold (FiveCommand): plan ended and all
        # task gates held 1 s. The joint-speed settle check cannot hold under
        # exploration noise (0% at std .05); evaluation keeps the sensor handoff.
        sensor_success=(skill!='go-to' and tier!='train'))
    cfg.observations['actor']=ObservationGroupCfg({'table':ObservationTermCfg(func=actor_observation)},enable_corruption=False)
    if critic=='actor':
        # Keep the complete previous privileged observation, including clean
        # sensors, and prepend the exact actor inputs (same cached noise/history).
        # Critic redundancy is intentional: no privileged signal is discarded.
        cfg.observations['critic'].terms={
            'actor_observation':ObservationTermCfg(func=actor_observation),
            **cfg.observations['critic'].terms}
    elif critic=='state':
        cfg.observations['critic'].terms['table_state']=ObservationTermCfg(func=critic_state)
    if tier!='train':
        cfg.terminations.pop('success',None);cfg.terminations.pop('invalid_switch',None)
        cfg.episode_length_s=max(cfg.episode_length_s,40.)
    return cfg
