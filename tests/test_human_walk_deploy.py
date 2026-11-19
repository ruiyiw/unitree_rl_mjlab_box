"""Deployment contract and failure-path tests for the human-walk policy."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from unitree_human_walk import CHECKPOINT, MOTOR_IDS, OBSERVATION, PACKAGE, PROFILE, ROOT, TERMS, TRAINING, term
from unitree_human_walk.runtime import Core, build_core, reference, telemetry, verify
from unitree_human_walk.validate import FREE, TOLERANCE, replay

FIXTURES = Path(__file__).parent / 'fixtures/human_walk'
BINARY = ROOT / 'deploy/robots/g1_23dof/build/g1_ctrl'
ROLLOUTS = sorted(p.stem for p in FIXTURES.glob('*.npz'))


@pytest.fixture(scope='module')
def core_library(tmp_path_factory):
    return build_core(tmp_path_factory.mktemp('human_walk_core'))


@pytest.fixture(scope='module')
def frames():
    return reference()


def standing(frames, yaw=0.):
    """Clean sensors of a robot standing in the walk's starting pose, facing `yaw`."""
    return telemetry([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], [0, 0, 0], [0, 0, 9.81], frames[0, 23:46], np.zeros(23))


def test_package_identity_and_profile():
    manifest = verify()
    assert manifest['format'] == 'g1-human-walk-v1' and not manifest['hardware_validated']
    actor = manifest['actor']
    assert actor['iteration'] == 500 and actor['checkpoint_source'].endswith(CHECKPOINT)
    assert actor['trained_with_estimator'] is None            # trained on the simulator's pose: the estimate is new to it
    assert manifest['observation'] == [list(t) for t in TERMS] and sum(w for _, w in TERMS) == OBSERVATION
    assert manifest['interface']['dimensions'] == {'nq': 30, 'nv': 29, 'nu': 23}
    assert len(manifest['interface']['arm_gravity_compensation']['bodies']) == 10
    box = json.loads((ROOT / 'deploy/robots/g1_23dof/config/policy/box/v1/params/contract.json').read_text())
    for key in ('joint_order', 'action_scale', 'default_joint_positions', 'kp', 'kd'):
        assert manifest['interface'][key] == box[key]                           # the robot the box controller drives
    walk = json.loads((ROOT / 'deploy/robots/g1_23dof/config/policy/parametrized_walk/v2/params/estimator.json').read_text())
    mine = json.loads((PACKAGE / 'params/estimator.json').read_text())
    assert mine['feet'] == walk['feet'] and mine['filter'] == walk['constants']  # the controller reuses the walk's filter
    cfg = yaml.safe_load(PROFILE.read_text())
    fsm = cfg['FSM']
    assert list(fsm['_']) == ['Passive', 'FixStand', 'Velocity', 'HumanWalk']
    assert fsm['FixStand']['transitions'] == {'Passive': 'LT + B.on_pressed', 'Velocity': 'RB + X.on_pressed'}
    assert fsm['Velocity']['transitions'] == {'Passive': 'LT + B.on_pressed', 'HumanWalk': 'RB + Y.on_pressed'}
    assert fsm['HumanWalk']['transitions'] == {'Passive': 'LT + B.on_pressed', 'Velocity': 'RB + X.on_pressed'}
    assert fsm['Velocity']['policy_dir'] == 'config/policy/velocity/v0'
    assert (fsm['HumanWalk']['blend_s'], fsm['HumanWalk']['hold_s'], fsm['HumanWalk']['odometry_scale']) == (.5, 1., 1.)
    assert fsm['HumanWalk']['return_s'] == .5                                  # Velocity takes over after the walk


def test_reference_is_one_walk_from_a_stand(frames):
    manifest = json.loads((PACKAGE / 'manifest.json').read_text())
    assert frames.shape == (323, 78) and np.isfinite(frames).all() and manifest['reference']['frames'] == 323
    pose = yaml.safe_load(PROFILE.read_text())['FSM']['FixStand']['qs'][1]
    np.testing.assert_allclose(np.array(pose)[list(MOTOR_IDS)], frames[0, 23:46], atol=1e-6)   # FixStand ends where the walk begins
    np.testing.assert_allclose(frames[0, :23], frames[0, 23:46], atol=1e-6)                     # the first target is the first pose
    assert np.abs(frames[0, 76:78]).max() < 1e-6 and np.abs(frames[0, 46:69]).max() < 1e-3       # under the pelvis, at rest
    assert abs(frames[-1, 76] - 2.2015) < 1e-3 and abs(frames[-1, 77] + .1406) < 1e-3            # 2.2 m ahead, 14 cm to the right
    assert np.abs(frames[-1, 46:69]).max() < 1e-3                                               # and at rest again
    np.testing.assert_allclose(np.linalg.norm(frames[:, 72:76], axis=1), 1, atol=1e-5)          # torso quaternions


@pytest.mark.parametrize('rollout', ROLLOUTS)
def test_controller_matches_training_rollout(core_library, rollout):
    """Sensors recorded in the mjlab training env, replayed through the C++ controller."""
    f = dict(np.load(FIXTURES / f'{rollout}.npz'))
    with Core(core_library) as core:
        worst, by_term, state = replay(core, f)
        alone, _, _ = replay(core, f, free=True)
    for key, limit in TOLERANCE.items():
        assert worst[key] <= limit, (key, worst[key])
    for key, limit in FREE.items():
        assert alone[key] <= limit, (key, alone[key])
    if rollout == 'nominal':
        assert tuple(state) == (int(f['plan_frames']) - 1, 1, 1)      # ended, and settled while standing at the end


def test_actor_matches_pytorch_and_report(tmp_path):
    pytest.importorskip('rsl_rl')
    from unitree_human_walk.validate import validate
    report = validate(FIXTURES, tmp_path / 'check')
    assert set(report['rollouts']) == set(ROLLOUTS) == {'nominal', 'pushes'}
    assert all(r['max_difference']['onnx_vs_pytorch_actor'] < 1e-4 for r in report['rollouts'].values())
    assert sum(r['steps'] for r in report['rollouts'].values()) == 1192


def test_entry_holds_the_first_frame_then_plays(core_library, frames):
    sensors = standing(frames)
    scale = np.array(json.loads((PACKAGE / 'manifest.json').read_text())['interface']['action_scale'], np.float32)
    with Core(core_library) as core:
        core.start(hold_s=1.)
        clocks = [int(core.step(sensors)[4][0]) for _ in range(57)]
        assert clocks == [0] * 52 + [1, 2, 3, 4, 5]               # the entry step, 50 held, training's repeat, then one frame per step
        core.start()
        assert [int(core.step(sensors)[4][0]) for _ in range(4)] == [0, 0, 1, 2]   # training's timeline
        obs, action, target, torque, state, estimate = core.step(sensors)
        assert obs.shape == (OBSERVATION,) and np.isfinite(target).all() and not state[1]
        np.testing.assert_allclose(target, frames[4, :23] + scale * action, atol=1e-6)
        assert not torque[:13].any() and np.abs(torque[13:]).max() > .3       # the arm motors only


def test_standing_at_the_start_the_robot_is_where_the_reference_starts(core_library, frames):
    with Core(core_library) as core:
        core.start()
        obs, _, _, _, _, estimate = core.step(standing(frames))
    np.testing.assert_allclose(estimate[:2], 0, atol=1e-6)
    assert .75 < estimate[2] < .82 and np.abs(estimate[3:]).max() < 1e-6
    assert np.abs(obs[term('ref_anchor')][:3]).max() < 5e-3                  # the torso is at the reference's torso
    np.testing.assert_allclose(obs[term('ref_anchor')][3:], [1, 0, 0, 1, 0, 0], atol=5e-3)   # the reference starts within 0.2 degrees of upright
    np.testing.assert_allclose(obs[term('ref_clock')], [0, 0])
    np.testing.assert_allclose(obs[term('projected_gravity')], [0, 0, -1], atol=1e-6)
    np.testing.assert_allclose(obs[term('joint_pos')] + 0, obs[term('ref_joints')][:23], atol=1e-6)


def test_walk_frame_is_the_pose_at_entry(core_library):
    """The route is straight ahead of wherever the robot faces when the walk starts."""
    f = dict(np.load(FIXTURES / 'nominal.npz'))
    turn = np.array([np.cos(1.35), 0, 0, np.sin(1.35)])           # 2.7 rad about the vertical

    def turned(q):
        w, x, y, z = q
        return [turn[0] * w - turn[3] * z, turn[0] * x - turn[3] * y, turn[0] * y + turn[3] * x, turn[0] * z + turn[3] * w]
    with Core(core_library) as a, Core(core_library) as b:
        a.start()
        b.start()
        for t in range(300):
            args = (f['gyro'][t], f['accelerometer'][t], f['joint_pos'][t], f['joint_vel'][t])
            if t:
                a.previous(f['action'][t - 1])
                b.previous(f['action'][t - 1])
            one = a.step(telemetry(f['quaternion'][t], *args))
            two = b.step(telemetry(turned(f['quaternion'][t]), *args))
            np.testing.assert_allclose(one[0], two[0], atol=2e-4)
            np.testing.assert_allclose(one[5], two[5], atol=2e-5)


def test_estimate_follows_the_robot(core_library):
    """The position the actor is given stays within centimetres of the simulator's over the walk."""
    f = dict(np.load(FIXTURES / 'nominal.npz'))
    with Core(core_library) as core:
        core.start()
        worst = 0.
        for t in range(len(f['clock'])):
            if t:
                core.previous(f['action'][t - 1])
            *_, estimate = core.step(telemetry(f['quaternion'][t], f['gyro'][t], f['accelerometer'][t], f['joint_pos'][t], f['joint_vel'][t]))
            worst = max(worst, float(np.abs(estimate[:3] - (f['pelvis'][t] - [f['pelvis'][0, 0], f['pelvis'][0, 1], 0])).max()))
    assert worst < .08
    assert abs(estimate[2] - f['pelvis'][-1, 2]) < .005                      # the height from the soles


def test_odometry_scale_stretches_the_estimate(core_library):
    f = dict(np.load(FIXTURES / 'nominal.npz'))
    ends = []
    for scale in (1., 1.02):
        with Core(core_library) as core:
            core.start(scale=scale)
            for t in range(int(f['plan_frames'])):
                if t:
                    core.previous(f['action'][t - 1])
                *_, estimate = core.step(telemetry(f['quaternion'][t], f['gyro'][t], f['accelerometer'][t],
                                                   f['joint_pos'][t], f['joint_vel'][t]))
            ends.append(estimate[0])
    assert abs(ends[1] / ends[0] - 1.02) < 2e-3


def test_bad_sensors(core_library, frames):
    sensors = standing(frames)
    with Core(core_library) as core:
        with pytest.raises(RuntimeError, match='hold time'):
            core.start(hold_s=-1.)
        for index, match in ((0, 'IMU'), (5, 'observation'), (8, 'estimate|observation|target'), (10, 'leg joint|observation')):
            core.start()
            core.step(sensors)
            bad = sensors.copy()
            bad[index] = np.nan
            if index == 0:
                bad[:4] = np.nan
            with pytest.raises(RuntimeError, match=match):
                core.step(bad)


@pytest.mark.parametrize('artifact', ['policy.onnx', 'reference.bin'])
def test_rejects_changed_artifacts(core_library, tmp_path, artifact):
    for name in ('policy.onnx', 'reference.bin'):
        shutil.copyfile(PACKAGE / name, tmp_path / name)
    Core(core_library, tmp_path).close()
    path = tmp_path / artifact
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(RuntimeError, match='identity mismatch'):
        Core(core_library, tmp_path)


CORRUPTIONS = [None, 'hold_at_the_end', 'absent_joint', 'pose', 'gain', 'walk_from_fixstand', 'blend', 'no_hold', 'scale', 'velocity',
               'extra_state', 'late_return', 'no_return_setting']


@pytest.mark.parametrize('corruption', CORRUPTIONS)
def test_actual_executable_profile(tmp_path, corruption):
    if not BINARY.is_file():
        pytest.skip('g1_ctrl is not built')
    cfg = yaml.safe_load(PROFILE.read_text())
    fsm = cfg['FSM']
    if corruption == 'absent_joint': fsm['Passive']['mode'][13] = 1
    elif corruption == 'pose': fsm['FixStand']['qs'][1][3] += .05
    elif corruption == 'gain': fsm['FixStand']['kp'][15] = 14.25
    elif corruption == 'walk_from_fixstand': fsm['FixStand']['transitions']['HumanWalk'] = 'RB + Y.on_pressed'
    elif corruption == 'blend': fsm['HumanWalk']['blend_s'] = 0
    elif corruption == 'no_hold': fsm['HumanWalk']['hold_s'] = 0
    elif corruption == 'scale': fsm['HumanWalk']['odometry_scale'] = 2
    elif corruption == 'velocity': fsm['Velocity']['policy_dir'] = 'config/policy/human_walk/v1'
    elif corruption == 'extra_state': fsm['_']['Again'] = {'id': 5, 'type': 'HumanWalk'}
    elif corruption == 'late_return': fsm['HumanWalk']['return_s'] = 30
    elif corruption == 'no_return_setting': del fsm['HumanWalk']['return_s']
    elif corruption == 'hold_at_the_end': fsm['HumanWalk']['return_s'] = 0      # allowed: the operator returns with RB + X
    path = tmp_path / 'profile.yaml'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    result = subprocess.run([str(BINARY), '--check-box-config', str(path)], capture_output=True, text=True, timeout=60)
    if corruption in (None, 'hold_at_the_end'):
        assert result.returncode == 0, result.stderr
        assert '145 inputs, 23 outputs, arm gravity compensation' in result.stdout
        assert ('Velocity takes over 0.5 s after' if corruption is None else 'holds at the end until [RB + X]') in result.stdout
        assert '323 reference frames' in result.stdout and 'no DDS connection or motor commands' in result.stdout
    else:
        assert result.returncode != 0


def test_native_simulation_from_velocity(tmp_path):
    """Velocity stands, hands over with the profile's blend and hold, and the robot walks the reference."""
    if not (TRAINING / 'policy-walk-human/banks/plan.pt').is_file():
        pytest.skip('the training checkout is not beside this repository')
    pytest.importorskip('mjlab')
    from unitree_human_walk.simulate import play
    report = play(tmp_path / 'sim', from_velocity=True)
    walk = report['walk']
    assert report['stop_reason'] == 'completed' and walk['returned_to_velocity_s_after_the_reference'] == .5
    assert walk['largest_pelvis_tilt_deg'] < 15 and walk['error_m'] < .20     # where it stands 10 s after Velocity took over
    assert walk['on_velocity_afterwards']['seconds'] == 10.
