"""Offline regression coverage for the frozen box deployment."""
import json
import shutil
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from unitree_box import ACTORS,MOTOR_IDS,ORDER,POLICY_DIR,ROOT
from unitree_box.runtime import Core,build_core,reference_path,verify


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    return build_core(tmp_path_factory.mktemp('box-core'))


@pytest.fixture(scope='module')
def payload():
    return torch.load(POLICY_DIR/'policy.pt',map_location='cpu',weights_only=False)


def test_packaged_selection_and_profiles(payload):
    manifest=verify()
    assert tuple(payload['order'])==ORDER
    assert set(payload['actors'])==set(ACTORS)
    assert payload['simulation_actuation_mode']=='none'
    assert payload['format']=='five-stage-composite-proprio-v3'
    ids=np.asarray(MOTOR_IDS); gaps=sorted(set(range(29))-set(MOTOR_IDS))
    for stage in (*ORDER,'sequence'):
        cfg=yaml.safe_load((ROOT/f'deploy/robots/g1_23dof/config/box_{stage}.yaml').read_text())['FSM']
        assert list(cfg['_'])==['Passive','FixStand','BoxPolicy']
        assert cfg['BoxPolicy']['stage']==stage
        assert cfg['FixStand']['transitions']['BoxPolicy']=='RT + A.on_pressed'
        for key in ('kp','kd'):
            np.testing.assert_allclose(np.asarray(cfg['FixStand'][key])[ids],manifest['interface'][key])
            assert not np.asarray(cfg['FixStand'][key])[gaps].any()
        assert not np.asarray(cfg['Passive']['mode'])[gaps].any()
        first=ORDER[0] if stage=='sequence' else stage
        np.testing.assert_allclose(np.asarray(cfg['FixStand']['qs'][-1])[ids],payload['references'][first]['plan']['qpos'][0,0,7:].numpy())
    # Return must retain its selected weights instead of sharing navigation.
    assert manifest['actors']['return']['checkpoint_sha256']!=manifest['actors']['go-to']['checkpoint_sha256']


def test_source_model_contract_and_plain_pd():
    reference_path()
    from five_task import env_cfg
    from unitree_balance.audit import contract
    saved=json.loads((POLICY_DIR/'params/contract.json').read_text())
    for stage in ORDER:
        model,actual=contract(env_cfg(stage,1,play=True))
        assert actual==saved
        assert model.nu==23 and not model.body_gravcomp.any()
    human=json.loads((ROOT/'deploy/robots/g1_23dof/config/policy/human_push/v1/params/contract.json').read_text())
    assert saved['joint_order']==human['joint_order']
    assert saved['default_joint_positions']==human['default_joint_positions']
    # Stronger box shoulder/elbow gains must not leak into human-push.
    assert saved['kp'][13]==40 and human['kp'][13]!=40
    assert saved['action_scale'][13]==.3


def test_wrong_actor_reference_and_invalid_inputs_rejected(library,tmp_path):
    stage=tmp_path/'go-to';stage.mkdir()
    shutil.copy2(POLICY_DIR/'return/policy.onnx',stage/'policy.onnx')
    shutil.copy2(POLICY_DIR/'go-to/reference.bin',stage/'reference.bin')
    with pytest.raises(RuntimeError,match='identity mismatch'):Core(library,0,tmp_path)
    shutil.copy2(POLICY_DIR/'go-to/policy.onnx',stage/'policy.onnx')
    with (stage/'reference.bin').open('ab') as stream:stream.write(b'bad')
    with pytest.raises(RuntimeError,match='identity mismatch'):Core(library,0,tmp_path)
    with pytest.raises(RuntimeError,match='Invalid box stage'):Core(library,5)
    with Core(library,0) as core:
        telemetry=np.zeros(65,np.float32)
        with pytest.raises(RuntimeError,match='IMU'):core.step(telemetry)
        telemetry[0]=1;telemetry[7+MOTOR_IDS[0]]=np.nan
        with pytest.raises(RuntimeError,match='Non-finite'):core.step(telemetry)


def test_stage_timeout_and_no_automatic_next_stage(library,payload):
    ids=np.asarray(MOTOR_IDS)
    telemetry=np.zeros(65,np.float32);telemetry[0]=1
    plan=payload['references']['go-to']['plan'];length=int(plan['len'][0])
    telemetry[7+ids]=plan['qpos'][0,-1,7:].numpy()
    with Core(library,0) as core:
        for _ in range(length+80):obs,action,target,state=core.step(telemetry)
        assert tuple(state)==(0,length-1,1)
        core.reset();obs,_,_,state=core.step(telemetry)
        assert tuple(state)==(0,0,0) and not obs[52:].any()
        telemetry[4]=.3 # Never satisfies the gyro settling gate.
        with pytest.raises(RuntimeError,match='timeout'):
            for _ in range(length+650):core.step(telemetry)


def test_actor_and_source_sequence_parity(tmp_path):
    from unitree_box.validate import validate
    report=validate(tmp_path/'validation',32)
    assert report['archived_source_controller_parity']
    assert len(report['transitions'])==4


@pytest.fixture(scope='module')
def controller_binary():
    binary=ROOT/'deploy/robots/g1_23dof/build/g1_ctrl'
    if not binary.exists():
        pytest.skip('Build g1_ctrl to run executable preflight checks')
    return binary


@pytest.mark.parametrize('stage',(*ORDER,'sequence'))
def test_real_executable_preflight(controller_binary,stage):
    import subprocess
    cfg=ROOT/f'deploy/robots/g1_23dof/config/box_{stage}.yaml'
    result=subprocess.run([str(controller_binary),'--check-box-config',str(cfg)],
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert 'no DDS connection or motor commands' in result.stdout


@pytest.mark.parametrize('corruption',('absent_motor','gain','pose','state_order'))
def test_executable_rejects_bad_profile(controller_binary,tmp_path,corruption):
    import subprocess
    cfg=yaml.safe_load((ROOT/'deploy/robots/g1_23dof/config/box_sequence.yaml').read_text())
    fsm=cfg['FSM']
    if corruption=='absent_motor':fsm['Passive']['mode'][13]=1
    elif corruption=='gain':fsm['FixStand']['kp'][15]=14.25
    elif corruption=='pose':fsm['FixStand']['qs'][1][0]=0.7
    else:
        states=fsm['_']
        fsm['_']={'BoxPolicy':states['BoxPolicy'],'Passive':states['Passive'],'FixStand':states['FixStand']}
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    result=subprocess.run([str(controller_binary),'--check-box-config',str(path)],
                          capture_output=True,text=True,timeout=30)
    assert result.returncode!=0
    assert 'Box preflight:' in result.stderr


def test_human_push_hardware_setup_matches():
    config=ROOT/'deploy/robots/g1_23dof/config'
    human=yaml.safe_load((config/'human_push.yaml').read_text())['FSM']
    hp=json.loads((config/'policy/human_push/v1/params/contract.json').read_text())
    box=json.loads((POLICY_DIR/'params/contract.json').read_text())
    for key in ('joint_order','action_order','actor_columns','default_joint_positions',
                'physics_dt','decimation','policy_dt','dimensions'):
        assert hp[key]==box[key],key
    changed_gains={name for i,name in enumerate(box['joint_order'])
                   if hp['kp'][i]!=box['kp'][i] or hp['kd'][i]!=box['kd'][i]}
    arms={name for name in box['joint_order'] if 'shoulder' in name or 'elbow' in name}
    assert changed_gains==arms
    hip_roll={'left_hip_roll_joint','right_hip_roll_joint'}
    changed_scales={name for i,name in enumerate(box['joint_order'])
                    if hp['action_scale'][i]!=box['action_scale'][i]}
    assert changed_scales==arms|hip_roll
    for stage in (*ORDER,'sequence'):
        fsm=yaml.safe_load((config/f'box_{stage}.yaml').read_text())['FSM']
        assert fsm['Passive']==human['Passive']
        assert fsm['FixStand']['ts']==human['FixStand']['ts']
        assert fsm['FixStand']['transitions']=={
            ('BoxPolicy' if key=='HumanPushRecovery' else key):value
            for key,value in human['FixStand']['transitions'].items()}
        assert fsm['BoxPolicy']['transitions']==human['HumanPushRecovery']['transitions']


def test_cpp_observations_match_human_push_on_identical_telemetry(library,tmp_path):
    from unitree_balance.human_push_validate import DeploymentCore
    human=DeploymentCore(tmp_path/'human_core')
    rng=np.random.default_rng(20261029)
    previous=np.zeros((1,23),np.float32)
    try:
        with Core(library,0) as box:
            for _ in range(128):
                telemetry=rng.normal(size=(1,65)).astype(np.float32)
                telemetry[0,:4]/=np.linalg.norm(telemetry[0,:4])
                hp_obs,_,_=human.run(telemetry,previous.copy())
                box_obs,action,_,_=box.step(telemetry[0])
                np.testing.assert_allclose(box_obs,hp_obs[0],atol=1e-6,rtol=1e-6)
                previous[0]=action
    finally:
        human.close()
