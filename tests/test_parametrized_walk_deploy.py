"""Deployment contract and failure-path tests for the two distance-parametrized walk policies."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from unitree_parametrized_skills import MOTOR_IDS, OBSERVATION, PACKAGE, POLICIES, PROFILE, ROOT
from unitree_parametrized_skills.runtime import Core, build_core, library, telemetry, verify
from unitree_parametrized_skills.validate import TOLERANCE, replay

FIXTURES = Path(__file__).parent / 'fixtures/parametrized_walk'
BINARY = ROOT / 'deploy/robots/g1_23dof/build/g1_ctrl'
ROLLOUTS = sorted(p.stem for p in FIXTURES.glob('*.npz'))


@pytest.fixture(scope='module')
def core_library(tmp_path_factory):
    return build_core(tmp_path_factory.mktemp('walk_core'))


@pytest.fixture(scope='module')
def plans():
    return library()[0]


def standing(plans, yaw=0.):
    """Clean sensors of a robot standing in the walks' starting pose, facing `yaw`."""
    return telemetry([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], [0, 0, 0], [0, 0, 9.81], plans[100][0, :23], np.zeros(23))


def test_package_identity_and_profile():
    manifest = verify()
    assert manifest['estimator'] == 'kalman' and not manifest['hardware_validated']
    assert list(manifest['actors']) == list(POLICIES) == ['goal-conditioned', 'adapter']
    assert manifest['format'] == 'g1-parametrized-walk-v2' and manifest['history'] == 4
    goal, adapter = manifest['actors']['goal-conditioned'], manifest['actors']['adapter']
    assert (goal['run'], goal['profile'], goal['history']) == ('goal20_hk2', 'v1', 4)          # not retrained (PARAMETRIZED_WALK.md)
    assert (adapter['run'], adapter['base_run'], adapter['profile']) == ('adapter20_r', 'base_r_seed2', 'robust')
    assert goal['trained_with_estimator'] == adapter['trained_with_estimator'] == 'kalman'
    assert adapter['adapter_bound'] == 1.0
    assert manifest['interface']['dimensions'] == {'nq': 30, 'nv': 29, 'nu': 23}
    box = json.loads((ROOT / 'deploy/robots/g1_23dof/config/policy/box/v1/params/contract.json').read_text())
    for key in ('joint_order', 'action_scale', 'default_joint_positions', 'kp', 'kd'):
        assert manifest['interface'][key] == box[key]                           # the robot the box controller drives
    cfg = yaml.safe_load(PROFILE.read_text())
    fsm = cfg['FSM']
    assert list(fsm['_']) == ['Passive', 'FixStand', 'Velocity', 'WalkGoal', 'WalkAdapter']
    assert fsm['FixStand']['transitions'] == {'Passive': 'LT + B.on_pressed', 'Velocity': 'RB + X.on_pressed'}
    assert fsm['Velocity']['policy_dir'] == 'config/policy/velocity/v0'
    assert 'WalkGoal' not in fsm['FixStand']['transitions']                     # a walk starts from Velocity
    assert (fsm['WalkGoal']['shortest_m'], fsm['WalkAdapter']['shortest_m']) == (.2, .3)
    assert cfg['WalkDistance']['default_m'] == 2.


def test_library_is_a_grid_of_walks_from_one_stand(plans):
    manifest = json.loads((PACKAGE / 'manifest.json').read_text())
    assert sorted(plans) == [100 * i for i in range(1, 201)] == [round(d * 1000) for d in manifest['library']['distances_m']]
    start = plans[100][0]
    pose = yaml.safe_load(PROFILE.read_text())['FSM']['FixStand']['qs'][1]
    np.testing.assert_allclose(np.array(pose)[list(MOTOR_IDS)], start[:23], atol=1e-6)   # FixStand ends where walks begin
    for mm, frames in plans.items():
        assert frames.shape[1] == 28 and np.isfinite(frames).all()
        np.testing.assert_allclose(frames[0, :23], start[:23], atol=1e-6)
        assert abs(frames[-1, 23] - frames[0, 23] - mm / 1000) < 1e-3                 # the plan walks the distance
        assert abs(frames[-1, 24]) < 5e-3 and frames[0, 27] == 0 and frames[-1, 27] == 0
        assert (len(frames) - 1) * .02 < 51


@pytest.mark.parametrize('rollout', ROLLOUTS)
def test_controller_matches_training_rollout(core_library, plans, rollout):
    """Sensors recorded in the mjlab training env, replayed through the C++ controller."""
    f = dict(np.load(FIXTURES / f'{rollout}.npz'))
    with Core(core_library, rollout.split('_')[0]) as core:
        worst, state = replay(core, f, json.loads((PACKAGE / 'manifest.json').read_text()).get('history', 1))
    for key, limit in TOLERANCE.items():
        assert worst[key] <= limit, (key, worst[key])
    assert tuple(state) == (int(f['plan_frames']) - 1, 1, 1)      # ended, and settled while standing at the goal


def test_actors_match_pytorch_and_report(tmp_path):
    pytest.importorskip('rsl_rl')
    from unitree_parametrized_skills.validate import validate
    report = validate(FIXTURES, tmp_path / 'check')
    assert set(report['rollouts']) == set(ROLLOUTS) and len(ROLLOUTS) == 6
    assert all(r['max_difference']['onnx_vs_pytorch_actor'] < 1e-4 for r in report['rollouts'].values())
    assert sum(r['steps'] for r in report['rollouts'].values()) == 7726


@pytest.mark.parametrize('policy', tuple(POLICIES))
def test_entry_holds_the_first_frame_then_plays(core_library, plans, policy):
    sensors = standing(plans)
    with Core(core_library, policy) as core:
        assert core.start(2., hold_s=1.) == 2.
        clocks = [int(core.step(sensors)[3][0]) for _ in range(56)]
        assert clocks == [0] * 51 + [1, 2, 3, 4, 5]               # the entry step, 50 held, then one frame per step
        assert core.start(2.) == 2.
        assert [int(core.step(sensors)[3][0]) for _ in range(4)] == [0, 1, 2, 3]   # training's timeline
        obs, action, target, state, estimate = core.step(sensors)
        assert obs.shape == (OBSERVATION,) and np.isfinite(target).all() and not state[1]
        np.testing.assert_allclose(target, plans[2000][5, :23] + np.array(
            json.loads((PACKAGE / 'manifest.json').read_text())['interface']['action_scale'], np.float32) * action, atol=1e-6)


def test_walk_frame_is_the_pose_at_entry(core_library, plans):
    """The route is straight ahead of wherever the robot faces when the walk starts."""
    f = dict(np.load(FIXTURES / 'goal-conditioned_2.5m.npz'))
    turn = np.array([np.cos(1.35), 0, 0, np.sin(1.35)])           # 2.7 rad about the vertical

    def turned(q):
        w, x, y, z = q
        return [turn[0] * w - turn[3] * z, turn[0] * x - turn[3] * y, turn[0] * y + turn[3] * x, turn[0] * z + turn[3] * w]
    with Core(core_library, 'goal-conditioned') as a, Core(core_library, 'goal-conditioned') as b:
        a.start(2.5)
        b.start(2.5)
        for t in range(300):
            args = (f['gyro'][t], f['accelerometer'][t], f['joint_pos'][t], f['joint_vel'][t])
            one = a.step(telemetry(f['quaternion'][t], *args))
            two = b.step(telemetry(turned(f['quaternion'][t]), *args))
            np.testing.assert_allclose(one[0], two[0], atol=2e-4)
            np.testing.assert_allclose(one[4], two[4], atol=2e-5)


def test_walking_again_keeps_the_correction_and_restarts_the_estimate(core_library, plans):
    sensors = standing(plans)
    with Core(core_library, 'adapter') as core:
        core.start(1.)
        for _ in range(400):
            *_, state, estimate = core.step(sensors)
        assert state[1] and state[2]                              # a still robot: ended and settled
        assert core.start(.5, previous=np.full(23, .3, np.float32)) == .5
        obs, _, _, state, estimate = core.step(sensors)
        np.testing.assert_array_equal(obs[52:75], np.float32(.3))
        assert tuple(state) == (0, 0, 0)
        np.testing.assert_allclose(estimate[:2], plans[500][0, 23:25], atol=1e-7)
        assert abs(obs[101] - .5) < 2e-3 and abs(obs[105] - .5) < 2e-3   # half a metre to go, by plan and by estimate


def test_odometry_scale_stretches_the_estimate(core_library):
    f = dict(np.load(FIXTURES / 'adapter_2.5m.npz'))
    ends = []
    for scale in (1., 1.02):
        with Core(core_library, 'adapter') as core:
            core.start(2.5, scale=scale)
            for t in range(int(f['plan_frames'])):
                *_, estimate = core.step(telemetry(f['quaternion'][t], f['gyro'][t], f['accelerometer'][t],
                                                   f['joint_pos'][t], f['joint_vel'][t]))
            ends.append(estimate[0] - f['estimate'][0, 0])
    assert abs(ends[1] / ends[0] - 1.02) < 2e-3


def test_distance_selection(core_library):
    with Core(core_library, 'goal-conditioned') as core:
        assert core.distances()[0] == .1 and core.distances()[-1] == 20. and len(core.distances()) == 200
        assert core.select(3.5) == 3.5 and core.select(3.14) == 3.1 and core.select(.1) == .1 and core.select(20.04) == 20.
        assert core.select(3.1) == 3.1 and core.select_step(1.) == 4.1
        assert core.select_step(-.1) == 4. and core.select_step(100.) == 20. and core.select_step(-100.) == .1
        for bad in (0., -1., 20.2, 1e9, float('nan'), float('inf')):
            with pytest.raises(ValueError, match='outside the planned range'):
                core.select(bad)
        assert core.select_step(.3) == .4                         # a refused request leaves the selection alone


def test_bad_sensors_and_missing_start(core_library, plans):
    sensors = standing(plans)
    with Core(core_library, 'goal-conditioned') as core:
        with pytest.raises(RuntimeError, match='No walk distance'):
            core.step(sensors)
        core.start(1.)
        for index, match in ((0, 'IMU'), (5, 'observation'), (8, 'estimate|observation|target'), (10, 'leg joint|observation')):
            core.start(1.)
            core.step(sensors)
            bad = sensors.copy()
            bad[index] = np.nan
            if index == 0:
                bad[:4] = np.nan
            with pytest.raises(RuntimeError, match=match):
                core.step(bad)


@pytest.mark.parametrize('artifact', ['goal-conditioned/policy.onnx', 'adapter/policy.onnx', 'plans.bin'])
def test_rejects_changed_artifacts(core_library, tmp_path, artifact):
    for name in ('goal-conditioned/policy.onnx', 'adapter/policy.onnx', 'plans.bin'):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        shutil.copyfile(PACKAGE / name, tmp_path / name)
    path = tmp_path / artifact
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    changed = artifact.split('/')[0] if '/' in artifact else None
    for policy in POLICIES:
        if changed in (None, policy):
            with pytest.raises(RuntimeError, match='identity mismatch'):
                Core(core_library, policy, tmp_path)
        else:
            Core(core_library, policy, tmp_path).close()


CORRUPTIONS = [None, 'absent_joint', 'pose', 'gain', 'walk_from_fixstand', 'blend', 'no_hold', 'shortest', 'scale',
               'default_distance', 'button', 'again', 'swapped_policy', 'velocity']


@pytest.mark.parametrize('corruption', CORRUPTIONS)
def test_actual_executable_profile(tmp_path, corruption):
    if not BINARY.is_file():
        pytest.skip('g1_ctrl is not built')
    cfg = yaml.safe_load(PROFILE.read_text())
    fsm = cfg['FSM']
    if corruption == 'absent_joint': fsm['Passive']['mode'][13] = 1
    elif corruption == 'pose': fsm['FixStand']['qs'][1][3] += .05
    elif corruption == 'gain': fsm['FixStand']['kp'][15] = 14.25
    elif corruption == 'walk_from_fixstand': fsm['FixStand']['transitions']['WalkGoal'] = 'RB + Y.on_pressed'
    elif corruption == 'blend': fsm['WalkGoal']['blend_s'] = 0
    elif corruption == 'no_hold': fsm['WalkAdapter']['hold_s'] = 0
    elif corruption == 'shortest': fsm['WalkAdapter']['shortest_m'] = 0
    elif corruption == 'scale': fsm['WalkGoal']['odometry_scale'] = 2
    elif corruption == 'default_distance': cfg['WalkDistance']['default_m'] = 50
    elif corruption == 'button': cfg['WalkDistance']['buttons'] = {'RT + sideways.on_pressed': 1.}
    elif corruption == 'again': fsm['WalkGoal']['again'] = 'RB + Q.on_pressed'
    elif corruption == 'swapped_policy': fsm['WalkGoal']['policy'] = 'adapter'
    elif corruption == 'velocity': fsm['Velocity']['policy_dir'] = 'config/policy/parametrized_walk/v1'
    path = tmp_path / 'profile.yaml'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    result = subprocess.run([str(BINARY), '--check-box-config', str(path)], capture_output=True, text=True, timeout=60)
    if corruption is None:
        assert result.returncode == 0, result.stderr
        assert result.stdout.count('inputs (4 sensor frames), 23 outputs') == 2
        assert '200 distances' in result.stdout and 'no DDS connection or motor commands' in result.stdout
    else:
        assert result.returncode != 0


@pytest.mark.parametrize('distance, ok', [('3.5', True), ('30', False), ('0', False), ('nan', False)])
def test_launch_distance_is_checked_before_dds(distance, ok):
    """A bad --distance stops the launch at the preflight, before any DDS object exists."""
    if not BINARY.is_file():
        pytest.skip('g1_ctrl is not built')
    try:
        result = subprocess.run([str(BINARY), '--config', str(PROFILE), '--network', 'lo', '--distance', distance],
                                capture_output=True, text=True, timeout=6, stdin=subprocess.DEVNULL)
        assert not ok and result.returncode == 1 and 'Waiting for connection' not in result.stdout
        assert 'outside the planned range' in result.stderr
    except subprocess.TimeoutExpired as waiting:        # accepted: it went on to wait for a robot
        out = waiting.stdout or ''
        out = out.decode() if isinstance(out, bytes) else out
        assert ok and f'Walk distance: {float(distance):.1f} m (--distance)' in out


@pytest.mark.parametrize('policy', tuple(POLICIES))
def test_native_simulation_from_velocity(tmp_path, policy):
    """Velocity stands, hands over with the profile's blend and hold, and the robot walks the distance."""
    if not (ROOT.parent / 'factory/parametrization/walk/runs/banks/eval.pt').is_file():
        pytest.skip('the training checkout is not beside this repository')
    pytest.importorskip('mjlab')
    from unitree_parametrized_skills.simulate import play
    report = play(policy, 1.5, tmp_path / 'sim', from_velocity=True, again=.5)
    assert report['stop_reason'] == 'completed' and len(report['walks']) == 2
    for walk in report['walks']:
        assert walk['settled_s_after_entry'] is not None and walk['error_m'] < .10
