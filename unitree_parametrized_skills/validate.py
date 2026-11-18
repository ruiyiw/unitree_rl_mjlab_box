"""Check the C++ walk controller against the training environment's recorded rollouts.

record.py saved, for every step of both policies on a few distances, the robot's sensors and
what training made of them.  Here the same sensors go through the C++ controller (the header
g1_ctrl compiles) and its observation, position estimate, correction and joint targets are
compared with training's, step by step.  Also checked: the library holds the same plans the
training planner makes, and the ONNX actors equal the PyTorch ones.

  python -m unitree_parametrized_skills.validate --fixtures tests/fixtures/parametrized_walk --output logs/walk_check
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import PACKAGE, POLICIES, actor_inputs, newest_frame
from .runtime import Core, build_core, library, telemetry, verify

# float32 arithmetic in two implementations: agreement, not identity.  The observation is
# compared without its previous-action entries, which are each side's own correction and are
# covered by `action`.  The actors amplify the estimate's last digits (the goal inputs are
# normalized by a few centimetres), hence the looser limit on the correction.
# v2's adapter is steeper still: one position input moved by the 8e-5 m the two estimates differ by
# changes its correction by 2e-3, and four inputs carry the estimate; the largest difference seen is 7.7e-3.
TOLERANCE = {'observation': 2e-4, 'estimate_m': 2e-4, 'action': 1e-2, 'target_rad': 5e-3}
PREVIOUS = slice(52, 75)


def replay(core, fixture, history=1):
    """The C++ controller on a recorded rollout's sensors; the largest differences from training."""
    f = fixture
    planned = core.start(float(f['distance_m']))
    assert abs(planned - float(f['distance_m'])) < 1e-6, 'the distance is not in the library'
    worst = dict.fromkeys(TOLERANCE, 0.0)
    for t in range(len(f['clock'])):
        sensors = telemetry(f['quaternion'][t], f['gyro'][t], f['accelerometer'][t], f['joint_pos'][t], f['joint_vel'][t])
        obs, action, target, state, estimate = core.step(sensors)
        assert state[0] == int(f['clock'][t, 0]), f'clock {state[0]} != {int(f["clock"][t, 0])} at step {t}'
        difference = np.abs(obs - newest_frame(f['observation'][t], history))
        difference[PREVIOUS] = 0
        worst['observation'] = max(worst['observation'], float(difference.max()))
        worst['estimate_m'] = max(worst['estimate_m'], float(np.abs(estimate[:2] - f['estimate'][t]).max()))
        worst['action'] = max(worst['action'], float(np.abs(action - f['action'][t]).max()))
        worst['target_rad'] = max(worst['target_rad'], float(np.abs(target - f['target'][t]).max()))
    return worst, state


def onnx_actor(lib, path, width):
    """The packaged ONNX actor through the same ONNX Runtime the controller links."""
    import ctypes
    lib = ctypes.CDLL(str(lib))
    ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    lib.onnx_open.argtypes, lib.onnx_open.restype = [ctypes.c_char_p], ctypes.c_void_p
    lib.onnx_run.argtypes = [ctypes.c_void_p, ptr, ctypes.c_int, ptr, ctypes.c_int]
    handle = lib.onnx_open(str(path).encode())
    assert handle

    def run(observations):
        out = np.empty((len(observations), 23), np.float32)
        for row, o in zip(out, np.ascontiguousarray(observations, dtype=np.float32)):
            assert not lib.onnx_run(handle, o, width, row, 23)
        return out
    return run


def check_actors(report):
    """Each ONNX actor against the PyTorch checkpoint(s) it was exported from, on recorded observations."""
    import torch
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    def actor(path, hidden):
        state = torch.load(path, map_location='cpu', weights_only=False)['actor_state_dict']
        dummy = TensorDict({'actor': torch.zeros(1, actor_inputs(manifest.get('history', 1)))}, batch_size=[1])
        model = MLPModel(dummy, {'actor': ['actor']}, 'actor', 23, hidden_dims=hidden, activation='elu',
                         obs_normalization=True, distribution_cfg={'class_name': 'GaussianDistribution',
                                                                   'init_std': .3, 'std_type': 'scalar'}).eval()
        model.load_state_dict(state, strict=True)
        return lambda o: model(TensorDict({'actor': o}, batch_size=[len(o)]))
    manifest = json.loads((PACKAGE / 'manifest.json').read_text())
    out = {}
    for name in POLICIES:
        folder = PACKAGE / name
        if (folder / 'base_checkpoint.pt').exists():
            base, adapter = actor(folder / 'base_checkpoint.pt', (256, 256)), actor(folder / 'checkpoint.pt', (64, 64))
            bound = manifest['actors'][name]['adapter_bound']
            out[name] = lambda o, base=base, adapter=adapter, bound=bound: base(o) + adapter(o).clamp(-bound, bound)
        else:
            out[name] = actor(folder / 'checkpoint.pt', (256, 256))
    return out


def validate(fixtures, output):
    import torch
    fixtures, output = Path(fixtures), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = verify()
    lib = build_core(output)
    plans, _ = library()
    report = {'hardware_tested': False, 'tolerance': TOLERANCE, 'rollouts': {},
              'method': 'Sensors recorded in the mjlab training env (nominal robot, Kalman estimator) replayed through '
                        'the C++ controller; its outputs compared with training\'s at every step'}
    torch.set_num_threads(1)
    reference = check_actors(report)
    history = manifest.get('history', 1)
    for name in POLICIES:
        with Core(lib, name) as core:
            assert core.distances() == [round(d, 3) for d in manifest['library']['distances_m']]
            for path in sorted(fixtures.glob(f'{name}_*m.npz')):
                f = dict(np.load(path))
                # the library's plan is the one training planned for this distance
                frames = plans[round(float(f['distance_m']) * 1000)]
                assert len(frames) == int(f['plan_frames'])
                clock = f['clock'][:, 0].astype(int)
                ref_joints = frames[np.minimum(clock + 1, len(frames) - 1), :23] - np.array(manifest['interface']['default_joint_positions'], np.float32)
                assert np.abs(ref_joints - newest_frame(f['observation'], history)[:, 75:98]).max() < 1e-6, 'library plan differs from the training plan'
                worst, state = replay(core, f, history)
                with torch.inference_mode():
                    expected = reference[name](torch.tensor(f['observation'])).numpy()
                worst['pytorch_actor_vs_recorded'] = float(np.abs(expected - f['action']).max())
                worst['onnx_vs_pytorch_actor'] = float(np.abs(onnx_actor(lib, PACKAGE / name / 'policy.onnx', actor_inputs(history))(f['observation']) - expected).max())
                for key, limit in TOLERANCE.items():
                    assert worst[key] <= limit, f'{path.name}: {key} differs by {worst[key]:.2e} (limit {limit:.0e})'
                assert worst['pytorch_actor_vs_recorded'] < 1e-4 and worst['onnx_vs_pytorch_actor'] < 1e-4
                report['rollouts'][path.stem] = dict(
                    steps=len(clock), plan_frames=int(f['plan_frames']), ended_early_in_training=bool(f['ended_early']),
                    walked_m=float(f['pelvis'][-1, 0] - f['pelvis'][0, 0]),
                    estimate_error_at_end_m=float(np.linalg.norm(f['estimate'][-1] - f['pelvis'][-1])),
                    settled_flag=bool(state[2]), max_difference=worst)
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    validate(args.fixtures, args.output)
