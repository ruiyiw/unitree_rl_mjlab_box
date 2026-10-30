"""Deployment contract and failure-path tests for the three history actors."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from unitree_box import ROOT, MOTOR_IDS
from unitree_box.table_export import PACKAGE, SELECTED
from unitree_box.table_runtime import Core, build_core, verify
from unitree_box.table_validate import validate


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    return build_core(tmp_path_factory.mktemp('table_core'))


def test_packaged_checkpoint_identity_and_profile():
    manifest = verify()
    assert manifest['observation_contract']['actor_dim'] == 376
    assert manifest['actuation'] == 'actuator_only'
    for stage, (_, expected) in SELECTED.items():
        assert manifest['actors'][stage]['checkpoint_sha256'] == expected
    cfg = yaml.safe_load((ROOT / 'deploy/robots/g1_23dof/config/box_three.yaml').read_text())['FSM']
    assert list(cfg['_']) == ['Passive', 'FixStand', 'Velocity', 'GoTo', 'Take', 'PutDown']
    assert cfg['FixStand']['transitions'] == {'Passive': 'LT + B.on_pressed', 'Velocity': 'RB + X.on_pressed'}
    assert cfg['Velocity']['policy_dir'] == 'config/policy/velocity/v0'
    assert cfg['Take']['transitions'] == {'Passive': 'LT + B.on_pressed', 'PutDown': 'RB + A.on_pressed'}
    assert 'Sequence' not in cfg and 'Carry' not in cfg


def test_source_observation_actor_gravity_parity(tmp_path):
    report = validate(tmp_path / 'parity')
    assert set(report['stages']) == set(SELECTED)
    assert sum(s['samples'] for s in report['stages'].values()) == 2383


@pytest.mark.parametrize('stage', range(3))
def test_timeout_and_reset(library, stage):
    name = tuple(SELECTED)[stage]
    plan = dict(np.load(PACKAGE / name / 'reference.npz'))
    telemetry = np.zeros(65, np.float32); telemetry[0] = 1
    telemetry[7 + np.array(MOTOR_IDS)] = plan['qpos'][-1, 7:]
    telemetry[4] = .4  # Cannot satisfy the gyro settling gate.
    with Core(library, stage) as core:
        with pytest.raises(RuntimeError, match='timeout'):
            for _ in range(len(plan['qpos']) + 650):
                core.step(telemetry, plan['qcmd'][0])
        core.reset(np.full(23, .8, np.float32))
        obs, _, _, _, state = core.step(telemetry, plan['qcmd'][0])
        assert tuple(state) == (0, 0)
        np.testing.assert_array_equal(obs[128:151], 0)
        telemetry[0] = np.nan
        with pytest.raises(RuntimeError, match='IMU'):
            core.step(telemetry, plan['qcmd'][0])


@pytest.mark.parametrize('artifact', ['policy.onnx', 'reference.bin'])
def test_rejects_changed_artifacts(library, tmp_path, artifact):
    folder = tmp_path / 'go-to'; folder.mkdir()
    for name in ('policy.onnx', 'reference.bin'):
        shutil.copyfile(PACKAGE / 'go-to' / name, folder / name)
    path = folder / artifact
    data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    with pytest.raises(RuntimeError, match='identity mismatch'):
        Core(library, 0, tmp_path)


@pytest.mark.parametrize('corruption', [None, 'absent_joint', 'pose', 'gain', 'walk_while_holding', 'old_policy', 'blend', 'velocity'])
def test_actual_executable_profile(tmp_path, corruption):
    binary = ROOT / 'deploy/robots/g1_23dof/build/g1_ctrl'
    if not binary.exists(): pytest.skip('Build g1_ctrl for executable checks')
    cfg = yaml.safe_load((ROOT / 'deploy/robots/g1_23dof/config/box_three.yaml').read_text())
    fsm = cfg['FSM']
    if corruption == 'absent_joint': fsm['Passive']['mode'][13] = 1
    elif corruption == 'pose': fsm['FixStand']['qs'][1][0] += .1
    elif corruption == 'gain': fsm['FixStand']['kp'][15] = 14.25
    elif corruption == 'walk_while_holding': fsm['Take']['transitions']['Velocity'] = 'RB + X.on_pressed'
    elif corruption == 'old_policy': fsm['GoTo']['policy_dir'] = 'config/policy/box/v1'
    elif corruption == 'blend': fsm['Take']['blend_s'] = -1
    elif corruption == 'velocity': fsm['Velocity']['policy_dir'] = 'config/policy/box_three/v1'
    path = tmp_path / 'profile.yaml'; path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    result = subprocess.run([str(binary), '--check-box-config', str(path)], capture_output=True, text=True, timeout=30)
    if corruption is None:
        assert result.returncode == 0, result.stderr
        assert result.stdout.count('376 inputs, 23 outputs') == 3
        assert 'no DDS connection or motor commands' in result.stdout
    else:
        assert result.returncode != 0


@pytest.mark.parametrize('stage', range(3))
def test_reference_start_clock_matches_training_reset(library, stage):
    from unitree_box.table_clock import source_clock
    name = tuple(SELECTED)[stage]
    plan = dict(np.load(PACKAGE / name / 'reference.npz'))
    reference = source_clock(name, len(plan['qpos']))
    telemetry = np.zeros(65, np.float32); telemetry[0] = 1
    telemetry[7 + np.array(MOTOR_IDS)] = plan['qpos'][0, 7:]
    with Core(library, stage) as core:
        for tick in range(8):
            if tick: reference._update_command(None)
            *_, state = core.step(telemetry, plan['qcmd'][0])
            assert state[0] == int(reference.clock[0, 0])


@pytest.mark.parametrize('stage', range(3))
def test_completed_stage_keeps_holding_like_previous_box(library, stage):
    name = tuple(SELECTED)[stage]
    plan = dict(np.load(PACKAGE / name / 'reference.npz'))
    telemetry = np.zeros(65, np.float32); telemetry[0] = 1
    telemetry[7 + np.array(MOTOR_IDS)] = plan['qpos'][-1, 7:]
    with Core(library, stage) as core:
        for _ in range(len(plan['qpos']) + 65):
            *_, state = core.step(telemetry, plan['qcmd'][-1])
        assert state[1] == 1
        telemetry[4] = .4
        # Once completed, a later loss of settling must not restart the stage timeout.
        for _ in range(650):
            *_, state = core.step(telemetry, plan['qcmd'][-1])
            assert state[1] == 1
