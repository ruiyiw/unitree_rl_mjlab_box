"""Check the C++ human-walk controller against the training environment's recorded rollouts.

record.py saved, for every step, the robot's sensors and what the training environment (with
estimate.py's inputs) made of them.  Here the same sensors go through the C++ controller (the
header g1_ctrl compiles) and its clock, observation, estimate, correction, joint targets and
arm torque are compared with the recording, step by step.  Also checked: the reference file
holds the training plan, and the ONNX actor equals the PyTorch one.

  python -m unitree_human_walk.validate --fixtures tests/fixtures/human_walk --output logs/human_walk_check
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import OBSERVATION, PACKAGE, TERMS, term
from .runtime import Core, build_core, reference, telemetry, verify

# float32 arithmetic in two implementations: agreement, not identity.  At each step the
# controller is given training's previous correction, so every step is compared on the same
# inputs.  Left to read back its own correction the difference grows where the actor is steep
# (FREE: up to 2e-2 in the pushed rollout, where corrections reach 1.75).
TOLERANCE = {'observation': 1e-4, 'estimate_m': 1e-4, 'velocity_m_s': 1e-3, 'action': 1e-3, 'target_rad': 5e-4, 'torque_nm': 1e-4}
FREE = {'action': 5e-2, 'target_rad': 2.5e-2}


def replay(core, fixture, free=False):
    """The C++ controller on a recorded rollout's sensors; the largest differences from training.
    free: the controller reads back its own previous correction, as it does on the robot."""
    f = fixture
    core.start(0.)
    worst = dict.fromkeys(TOLERANCE, 0.0)
    worst_term = {}
    previous = term('actions')
    for t in range(len(f['clock'])):
        sensors = telemetry(f['quaternion'][t], f['gyro'][t], f['accelerometer'][t], f['joint_pos'][t], f['joint_vel'][t])
        if t and not free:
            core.previous(f['action'][t - 1])
        obs, action, target, torque, state, estimate = core.step(sensors)
        assert state[0] == int(f['clock'][t]), f'clock {state[0]} != {int(f["clock"][t])} at step {t}'
        difference = np.abs(obs - f['observation'][t])
        if free:
            difference[previous] = 0
        worst['observation'] = max(worst['observation'], float(difference.max()))
        for name, _ in TERMS:
            worst_term[name] = max(worst_term.get(name, 0.), float(difference[term(name)].max()))
        worst['estimate_m'] = max(worst['estimate_m'], float(np.abs(estimate[:3] - f['estimate'][t]).max()))
        worst['velocity_m_s'] = max(worst['velocity_m_s'], float(np.abs(estimate[3:6] - f['velocity'][t]).max()))
        worst['action'] = max(worst['action'], float(np.abs(action - f['action'][t]).max()))
        worst['target_rad'] = max(worst['target_rad'], float(np.abs(target - f['target'][t]).max()))
        # the ten arm motors; MuJoCo's compensation also turns the waist and lifts the pelvis, which no motor does
        assert not torque[:13].any()
        worst['torque_nm'] = max(worst['torque_nm'], float(np.abs(torque[13:] - f['torque'][t, 13:]).max()))
    return worst, worst_term, state


def onnx_actor(lib, path):
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
            assert not lib.onnx_run(handle, o, OBSERVATION, row, 23)
        return out
    return run


def validate(fixtures, output):
    import torch
    from .export import actor
    fixtures, output = Path(fixtures), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = verify()
    lib = build_core(output)
    frames = reference()
    default = np.array(manifest['interface']['default_joint_positions'], np.float32)
    torch.set_num_threads(1)
    module, _ = actor(PACKAGE / 'checkpoint.pt', tuple(manifest['actor']['hidden']))
    onnx = onnx_actor(lib, PACKAGE / 'policy.onnx')
    report = {'hardware_tested': False, 'tolerance': TOLERANCE, 'rollouts': {},
              'method': 'Sensors recorded in the mjlab training env (nominal robot, estimate.py inputs) replayed through '
                        'the C++ controller; its outputs compared with the recording at every step'}
    for path in sorted(fixtures.glob('*.npz')):
        f = dict(np.load(path))
        assert len(frames) == int(f['plan_frames'])
        clock = f['clock'].astype(int)
        # the reference file is the plan training read
        assert np.abs(frames[clock, 23:46] - default - f['observation'][:, :23]).max() < 1e-6, 'reference differs from the training plan'
        assert np.abs(frames[clock, 46:69] - f['observation'][:, 23:46]).max() < 1e-6
        with Core(lib) as core:
            worst, by_term, state = replay(core, f)
            alone, _, _ = replay(core, f, free=True)
        for key, limit in FREE.items():
            assert alone[key] <= limit, f'{path.name}: reading back its own correction, {key} differs by {alone[key]:.2e}'
            worst[f'{key}_reading_back_its_own_correction'] = alone[key]
        with torch.inference_mode():
            expected = module(torch.tensor(f['observation'])).numpy()
        worst['pytorch_actor_vs_recorded'] = float(np.abs(expected - f['action']).max())
        worst['onnx_vs_pytorch_actor'] = float(np.abs(onnx(f['observation']) - expected).max())
        for key, limit in TOLERANCE.items():
            assert worst[key] <= limit, f'{path.name}: {key} differs by {worst[key]:.2e} (limit {limit:.0e})'
        assert worst['pytorch_actor_vs_recorded'] < 1e-4 and worst['onnx_vs_pytorch_actor'] < 1e-4
        report['rollouts'][path.stem] = dict(
            steps=len(clock), plan_frames=int(f['plan_frames']), ended_early_in_training=bool(f['ended_early']),
            pushes=bool(f['pushes']), walked_m=float(f['pelvis'][-1, 0] - f['pelvis'][0, 0]),
            estimate_error_at_end_m=float(np.linalg.norm(f['estimate'][-1, :2] - f['pelvis'][-1, :2])),
            largest_arm_torque_nm=float(np.abs(f['torque'][:, 13:]).max()),
            settled_flag=bool(state[2]), max_difference=worst, max_observation_difference_by_term=by_term)
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    validate(args.fixtures, args.output)
