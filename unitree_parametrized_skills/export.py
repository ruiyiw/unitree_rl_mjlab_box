"""Export walk(N) for C++: two frozen actors, the library of planned distances, the robot's
constants, the launch profile.

Needs the training checkout (factory: parametrization/walk and its runs) and the planner it
uses (humanoid).  Nothing is retrained and no weight or normalizer is changed.

  python -m unitree_parametrized_skills.export --factory-root ../factory --output <new directory>

The distances in the library are a choice made here (--step, --max, --extra); the C++ build
fingerprints the library, so a library with other distances needs a rebuild.
"""
import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

from . import FRAME, MOTOR_IDS, OBSERVATION, PACKAGE, POLICIES, PROFILE, REFERENCE, ROOT

MAGIC = b'G1WALKN1'
SOURCES = ('env.py', 'odometry.py', 'plans.py', 'models.py', 'evaluate.py', 'README.md')


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fnv(path):
    value = 14695981039346656037
    for byte in Path(path).read_bytes():
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def training(factory):
    """The training package, importable: parametrization.walk and the humanoid planner."""
    factory = Path(factory).resolve()
    if str(factory) not in sys.path:
        sys.path.insert(0, str(factory))
    from parametrization.walk import env, models, odometry, paths, plans
    return env, models, odometry, paths, plans


def contract(cfg):
    """Joint order, gains, action scales and nominal pose of the env's robot (as unitree_balance.audit)."""
    import re
    from mjlab.entity import Entity
    entity = Entity(cfg.scene.entities['robot'])
    model = entity.spec.compile()
    joints = list(entity.joint_names)
    _, actions = entity.find_joints_by_actuator_names(cfg.actions['joint_pos'].actuator_names)
    scales = []
    for name in actions:
        matches = [v for pattern, v in cfg.actions['joint_pos'].scale.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous action scale: {name}')
        scales.append(matches[0])
    actuated = [model.joint(i).name for i in model.actuator_trnid[:, 0]]
    actuator_ids = [actuated.index(name) for name in actions]
    joint_ids = [model.joint(name).id for name in joints]
    return model, {
        'joint_order': joints, 'action_order': list(actions), 'action_scale': scales,
        'default_joint_positions': model.key_qpos[0, model.jnt_qposadr[joint_ids]].tolist(),
        'kp': model.actuator_gainprm[actuator_ids, 0].tolist(),
        'kd': (-model.actuator_biasprm[actuator_ids, 2]).tolist(),
        'force_limits': model.actuator_forcerange[actuator_ids].tolist(),
        'physics_dt': cfg.sim.mujoco.timestep, 'decimation': cfg.decimation,
        'policy_dt': cfg.sim.mujoco.timestep * cfg.decimation,
        'dimensions': {'nq': model.nq, 'nv': model.nv, 'nu': model.nu},
    }


def inference_module(models, policy, base, bound):
    """observations (1, 109) -> the correction on the joints (1, 23), normalization included.
    With a base walker: base(obs) + clamp(adapter(obs), -bound, bound), as AdapterJointAction."""
    import torch

    def parts(path, hidden):
        actor = models.frozen_actor(str(path), OBSERVATION, 23, 'cpu', hidden).model
        return actor, actor.obs_normalizer, actor.mlp

    class Goal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor, self.norm, self.mlp = parts(policy, models.HIDDEN)

        def forward(self, observations):
            return self.mlp(self.norm(observations))

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_actor, self.base_norm, self.base_mlp = parts(base, models.HIDDEN)
            self.actor, self.norm, self.mlp = parts(policy, models.ADAPTER_HIDDEN)

        def forward(self, observations):
            return self.base_mlp(self.base_norm(observations)) + torch.clamp(
                self.mlp(self.norm(observations)), -bound, bound)

    return (Adapter() if base else Goal()).eval()


def plan_library(plans, paths, distances, workers):
    """Plan every distance with the training planner; joints in the robot's order come later."""
    import multiprocessing as mp
    order = sorted(range(len(distances)), key=lambda i: -distances[i])
    workers = max(1, min(workers, len(distances)))
    chunks = [[distances[i] for i in order[w::workers]] for w in range(workers)]
    with mp.get_context('spawn').Pool(workers) as pool:
        results = pool.map(plans._worker, chunks)
    out = sorted((p for group, _ in results for p in group), key=lambda p: p['n_m'])
    statuses = {}
    for _, (_, _, s) in results:
        for k, v in s.items():
            statuses[k] = statuses.get(k, 0) + v
    paths.use_humanoid()
    import plan as planner
    return out, planner.Robot().joints, statuses


def write_library(path, planned, perm):
    """One file: a table of the distances, then every plan's frames (little-endian float32).

    frame = 23 joint targets, the plan's pelvis x and y, sin and cos of the stride phase, walking."""
    import numpy as np
    table, frames, first = b'', [], 0
    for p in planned:
        length = len(p['qpos'])
        f = np.concatenate((p['qcmd'][:, perm], p['qpos'][:, :2], p['phase']), -1).astype('<f4')
        assert f.shape == (length, FRAME) and np.isfinite(f).all()
        end = p['qpos'][-1, 7:][perm].astype('<f4')
        table += struct.pack('<III', round(p['n_m'] * 1000), length, first) + end.tobytes()
        frames.append(f)
        first += length
    Path(path).write_bytes(MAGIC + struct.pack('<II', len(planned), FRAME) + table
                           + np.concatenate(frames).tobytes())
    return first


def kinematics(odometry, model, joints):
    """The two legs' chains from the pelvis to the foot sites, as LegKinematics has them."""
    kin = odometry.LegKinematics(model, joints, 'cpu')
    feet = []
    for links, site in kin.chains:
        feet.append({'site': site.tolist(),
                     'links': [{'position': ln['pos'].tolist(), 'rotation': ln['R'].reshape(-1).tolist(),
                                'axis': ln['axis'].tolist(), 'anchor': ln['anchor'].tolist(),
                                'joint': int(ln['col'])} for ln in links]})
    assert all(len(f['links']) == 6 for f in feet)
    return feet


def header(interface, start, feet, estimator, hashes, library_hash, count, frames, bound):
    def array(name, values, kind='float'):
        def value(x):
            return f'{float(x):.10e}f' if kind == 'float' else str(x) + ('ULL' if kind == 'uint64_t' else '')
        return f'inline constexpr std::array<{kind},{len(values)}> {name}{{{{' + ','.join(map(value, values)) + '}};\n'

    def vec(values):
        return '{{' + ','.join(f'{float(v):.10e}f' for v in values) + '}}'
    h = ('// Generated by unitree_parametrized_skills.export. Do not edit.\n'
         '#pragma once\n#include <array>\n#include <cstdint>\nnamespace parametrized_walk {\n'
         'inline constexpr float kStepDt = .02f;\n'
         f'inline constexpr std::array<const char*,{len(POLICIES)}> kNames{{{{'
         + ','.join(f'"{n}"' for n in POLICIES) + '}};\n')
    h += array('kActorHashes', hashes, 'uint64_t')
    h += f'inline constexpr uint64_t kLibraryHash = {library_hash}ULL;\n'
    h += f'inline constexpr int kPlans = {count};\ninline constexpr int kLibraryFrames = {frames};\n'
    h += f'inline constexpr float kAdapterBound = {bound:.10e}f; // already inside the adapter actor\n'
    h += array('kMotorIds', MOTOR_IDS, 'int')
    for name, key in (('kDefault', 'default_joint_positions'), ('kScale', 'action_scale'), ('kKp', 'kp'), ('kKd', 'kd')):
        h += array(name, interface[key])
    h += array('kStart', start)
    for name, value in estimator.items():
        h += f'inline constexpr float {name} = {value:.10e}f;\n'
    h += ('struct Link { std::array<float,3> position; std::array<float,9> rotation; '
          'std::array<float,3> axis,anchor; int joint; };\n'
          'struct Leg { std::array<Link,6> links; std::array<float,3> site; };\n'
          'inline constexpr std::array<Leg,2> kLegs{{\n')
    h += ',\n'.join('{{{' + ',\n'.join('{' + ','.join(vec(ln[k]) for k in ('position', 'rotation', 'axis', 'anchor'))
                                       + f',{ln["joint"]}' + '}' for ln in f['links']) + '}},' + vec(f['site']) + '}'
                    for f in feet)
    return h + '\n}};\n}\n'


def write_profile(interface, start, default_m):
    """Passive, FixStand, Velocity and the two walk policies; the box profiles' Passive/FixStand/Velocity."""
    import yaml
    config = ROOT / 'deploy/robots/g1_23dof/config'
    cfg = yaml.safe_load((config / 'box_three.yaml').read_text())
    old = cfg['FSM']
    passive = {'Passive': 'LT + B.on_pressed'}
    fsm = {'_': {'Passive': {'id': 1}, 'FixStand': {'id': 2}, 'Velocity': {'id': 3, 'type': 'RLBase'},
                 'WalkGoal': {'id': 4, 'type': 'ParametrizedWalk'},
                 'WalkAdapter': {'id': 5, 'type': 'ParametrizedWalk'}}}
    for name in ('Passive', 'FixStand', 'Velocity'):
        fsm[name] = old[name]
    fsm['FixStand']['transitions'] = {**passive, 'Velocity': 'RB + X.on_pressed'}
    pose = [0.] * 29
    for field in ('kp', 'kd'):
        fsm['FixStand'][field] = [0.] * 29
        for i, motor in enumerate(MOTOR_IDS):
            fsm['FixStand'][field][motor] = interface[field][i]
    for i, motor in enumerate(MOTOR_IDS):
        pose[motor] = start[i]
    fsm['FixStand']['qs'] = [[], pose]
    fsm['Velocity']['transitions'] = {**passive, 'WalkGoal': 'RB + Y.on_pressed', 'WalkAdapter': 'RB + A.on_pressed'}
    directory = str(PACKAGE.relative_to(ROOT / 'deploy/robots/g1_23dof'))
    # shortest_m: below it the policy lost its posture in more than 1 in 20 randomized training-env trials
    for name, policy, other, again, shortest in (('WalkGoal', 'goal-conditioned', 'WalkAdapter', 'RB + Y.on_pressed', .2),
                                                 ('WalkAdapter', 'adapter', 'WalkGoal', 'RB + A.on_pressed', .3)):
        to_other = 'RB + A.on_pressed' if other == 'WalkAdapter' else 'RB + Y.on_pressed'
        fsm[name] = {'transitions': {**passive, 'Velocity': 'RB + X.on_pressed', other: to_other},
                     'policy_dir': directory, 'policy': policy, 'blend_s': .5, 'hold_s': 1.,
                     'shortest_m': shortest, 'again': again, 'odometry_scale': 1.}
    cfg['FSM'] = fsm
    cfg['WalkDistance'] = {'default_m': default_m,
                           'buttons': {'RT + up.on_pressed': 1., 'RT + down.on_pressed': -1.,
                                       'RT + right.on_pressed': .1, 'RT + left.on_pressed': -.1}}
    PROFILE.write_text(yaml.safe_dump(cfg, sort_keys=False))


def export(factory, output, step, longest, extra, workers, default_m):
    import numpy as np
    import torch
    env, models, odometry, paths, plans = training(factory)
    factory, output = Path(factory).resolve(), Path(output).resolve()
    runs = Path(paths.RUNS)
    distances = sorted({round(i * step, 3) for i in range(1, round(longest / step) + 1)} | set(extra))
    assert distances[0] > 0 and len({round(d * 1000) for d in distances}) == len(distances)

    # the robot the policies were trained on must be the one the box controller already drives
    cfg = env.env_cfg(plans.bank_path('eval'), num_envs=1, mode='nominal')
    model, interface = contract(cfg)
    assert model.nu == 23 and not model.body_gravcomp.any() and interface['policy_dt'] == .02
    box = json.loads((ROOT / 'deploy/robots/g1_23dof/config/policy/box/v1/params/contract.json').read_text())
    for key in ('joint_order', 'action_order', 'action_scale', 'default_joint_positions', 'kp', 'kd'):
        if interface[key] != box[key]:
            raise ValueError(f'The walk robot differs from the box robot in {key}')
    assert tuple(env.ACTOR_TERMS) == ('gyro', 'gravity', 'joint_pos', 'joint_vel', 'actions', 'ref_joints',
                                      'ref_phase', 'goal') and env.ACTOR_DIM == OBSERVATION

    output.mkdir(parents=True, exist_ok=False)
    params = output / 'params'
    params.mkdir()
    manifest = {'format': 'g1-parametrized-walk-v1', 'hardware_validated': False, 'estimator': 'kalman',
                'source_root': str(factory), 'actors': {}, 'interface': interface}
    hashes, bound = [], 0.
    for name, (run, base_run) in POLICIES.items():
        folder = output / name
        folder.mkdir()
        meta = json.loads((runs / run / 'run.json').read_text())
        variant = paths.run_variant(str(runs / run / 'selected.pt'))
        assert variant['history'] == 1 and not variant['sole_contacts'] and meta['actor_dim'] == OBSERVATION
        shutil.copy2(runs / run / 'selected.pt', folder / 'checkpoint.pt')
        info = {'run': run, 'checkpoint_source': str(runs / run / 'selected.pt'),
                'checkpoint_sha256': digest(folder / 'checkpoint.pt'), 'trained_with_estimator': variant['estimator'],
                'selection': json.loads((runs / run / 'selection.json').read_text())['selected'], 'training': meta}
        base = None
        if base_run:
            assert meta['kind'] == 'adapter' and paths.run_variant(str(runs / base_run / 'selected.pt'))['history'] == 1
            base, bound = folder / 'base_checkpoint.pt', float(meta['adapter_bound'])
            shutil.copy2(runs / base_run / 'selected.pt', base)
            info.update(base_run=base_run, base_checkpoint_sha256=digest(base), adapter_bound=bound,
                        base_training=json.loads((runs / base_run / 'run.json').read_text()))
        module = inference_module(models, folder / 'checkpoint.pt', base, bound)
        with torch.inference_mode():
            torch.onnx.export(module, torch.zeros(1, OBSERVATION), str(folder / 'policy.onnx'),
                              input_names=['observations'], output_names=['actions'], opset_version=18, dynamo=False)
        (folder / 'source_metadata.json').write_text(json.dumps(info, indent=2) + '\n')
        hashes.append(fnv(folder / 'policy.onnx'))
        manifest['actors'][name] = {k: info[k] for k in info if k not in ('training', 'base_training')}
        manifest['actors'][name]['onnx_sha256'] = digest(folder / 'policy.onnx')

    planned, plan_joints, statuses = plan_library(plans, paths, distances, workers)
    perm = [plan_joints.index(n) for n in interface['joint_order']]
    start = planned[0]['qpos'][0, 7:][perm]
    for p in planned:                       # every walk starts from the same stand, facing along x
        assert np.abs(p['qpos'][0, 7:][perm] - start).max() < 1e-6 and np.abs(p['qcmd'][0, perm] - start).max() < 1e-6
        assert abs(p['qpos'][0, 1]) < 1e-3 and np.abs(p['qpos'][0, 4:7]).max() < 1e-3
    frames = write_library(output / 'plans.bin', planned, perm)
    manifest['library'] = {
        'distances_m': distances, 'plans': len(planned), 'frames': frames, 'fps': plans.FPS,
        'longest_s': (max(len(p['qpos']) for p in planned) - 1) / plans.FPS,
        'ik_error_max_m': float(max(p['ik_err'].max() for p in planned)),
        'pelvis_end_minus_distance_max_m': float(max(abs(p['qpos'][-1, 0] - p['qpos'][0, 0] - p['n_m']) for p in planned)),
        'qp_statuses': statuses,
        'frame': ['qcmd/' + n for n in interface['joint_order']] + ['pelvis_x', 'pelvis_y', 'phase_sin', 'phase_cos', 'walking']}

    K = odometry.KalmanOdometry
    estimator = {'kSoft': K.SOFT, 'kSigA': K.SIG_A, 'kSigB': K.SIG_B, 'kSigV': K.SIG_V, 'kGate': K.GATE,
                 'kP0Position': K.P0[0], 'kP0Velocity': K.P0[1], 'kP0Bias': K.P0[2]}
    feet = kinematics(odometry, model, interface['joint_order'])
    (params / 'contract.json').write_text(json.dumps(interface, indent=2) + '\n')
    (params / 'estimator.json').write_text(json.dumps({'constants': estimator, 'feet': feet}, indent=2) + '\n')
    (params / 'PolicyConfig.h').write_text(header(interface, start.tolist(), feet, estimator, hashes,
                                                  fnv(output / 'plans.bin'), len(planned), frames, bound))
    if output == PACKAGE.resolve():
        write_profile(interface, start.tolist(), default_m)
        REFERENCE.mkdir(exist_ok=True)      # the training source the C++ port follows, for the record
        for name in SOURCES:
            shutil.copy2(factory / 'parametrization/walk' / name, REFERENCE / name)
        manifest['source_files'] = {name: digest(REFERENCE / name) for name in SOURCES}
    manifest['files'] = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(output, f'{len(planned)} distances, {frames} frames')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factory-root', type=Path, default=ROOT.parent / 'factory')
    parser.add_argument('--output', type=Path, default=PACKAGE)
    parser.add_argument('--step', type=float, default=.1, help='spacing of the library distances (m)')
    parser.add_argument('--max', type=float, default=20., help='longest library distance (m)')
    parser.add_argument('--extra', type=float, nargs='*', default=[], help='more distances to plan (m)')
    parser.add_argument('--default', type=float, default=2., help='the profile\'s distance at launch (m)')
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    export(args.factory_root, args.output, args.step, args.max, args.extra, args.workers, args.default)
