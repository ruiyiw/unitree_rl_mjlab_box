"""Export the three selected 376-input checkpoints without changing the five-policy package."""
import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

from . import ROOT, MOTOR_IDS
from .export import digest, fnv

PACKAGE = ROOT / 'deploy/robots/g1_23dof/config/policy/box_three/v1'
SELECTED = {
    'go-to': ('go-to-scratch-matt/model_3999.pt', '363da7e10cafe132e8e2a118738c57ba28d770402bfcf644aa1a5dc52a768987'),
    'take': ('take-warm-yours/model_3999.pt', 'b9289259632bfc631c01a4c55238eeef7a4b007b6d5993067468adb844f8d989'),
    'put-down': ('put-down-scratch-matt/model_1600.pt', '51acfeb52e5fe730f44e7e55185d62e4ddb0cfaa64521959d34ce50c194388ad'),
}


def gravity_xml(model):
    """Robot inertials/joints only: keep the independent torque oracle small."""
    import xml.etree.ElementTree as ET
    root = ET.Element('mujoco', model='box_three_gravity')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true')
    ET.SubElement(root, 'option', gravity='0 0 -9.81')
    world = ET.SubElement(root, 'worldbody'); nodes = {0: world}
    vec = lambda values: ' '.join(format(float(v), '.17g') for v in values)
    for b in range(1, model.nbody):
        node = ET.SubElement(nodes[int(model.body_parentid[b])], 'body', name=model.body(b).name,
                             pos=vec(model.body_pos[b]), quat=vec(model.body_quat[b]),
                             gravcomp=str(float(model.body_gravcomp[b])))
        nodes[b] = node
        ET.SubElement(node, 'inertial', pos=vec(model.body_ipos[b]), quat=vec(model.body_iquat[b]),
                      mass=str(float(model.body_mass[b])), diaginertia=vec(model.body_inertia[b]))
        for j in range(model.body_jntadr[b], model.body_jntadr[b]+model.body_jntnum[b]):
            attrs = dict(name=model.joint(j).name, type='free' if model.jnt_type[j] == 0 else 'hinge',
                         pos=vec(model.jnt_pos[j]), actuatorgravcomp=str(bool(model.jnt_actgravcomp[j])).lower())
            if model.jnt_type[j] != 0:
                attrs.update(axis=vec(model.jnt_axis[j]), ref=str(float(model.qpos0[model.jnt_qposadr[j]])))
            ET.SubElement(node, 'joint', **attrs)
    ET.indent(root)
    return ET.tostring(root, encoding='unicode') + '\n'


def write_profile(interface, start):
    import yaml
    config = ROOT / 'deploy/robots/g1_23dof/config'
    cfg = yaml.safe_load((config / 'config.yaml').read_text())
    old = cfg['FSM']
    fsm = {'_': {'Passive': {'id': 1}, 'FixStand': {'id': 2},
                  'Velocity': {'id': 3, 'type': 'RLBase'}}}
    for name in ('Passive', 'FixStand', 'Velocity'):
        fsm[name] = old[name]
    fsm['FixStand']['transitions'] = {'Passive': 'LT + B.on_pressed', 'Velocity': 'RB + X.on_pressed'}
    for field in ('kp', 'kd'):
        fsm['FixStand'][field] = [0.] * 29
        for i, motor in enumerate(MOTOR_IDS):
            fsm['FixStand'][field][motor] = interface[field][i]
    pose = [0.] * 29
    for i, motor in enumerate(MOTOR_IDS):
        pose[motor] = start[i]
    fsm['FixStand']['qs'] = [[], pose]
    fsm['Velocity']['policy_dir'] = 'config/policy/velocity/v0'
    fsm['Velocity']['transitions'] = {'Passive': 'LT + B.on_pressed',
                                     'GoTo': 'RB + Y.on_pressed', 'Take': 'RB + A.on_pressed'}
    for name, stage, after in (
        ('GoTo', 'go-to', {'Take': 'RB + A.on_pressed', 'Velocity': 'RB + X.on_pressed'}),
        ('Take', 'take', {'PutDown': 'RB + A.on_pressed'}),
        ('PutDown', 'put-down', {'Velocity': 'RB + X.on_pressed'}),
    ):
        fsm['_'][name] = {'id': len(fsm['_']) + 1, 'type': 'TableBoxPolicy'}
        fsm[name] = {'transitions': {'Passive': 'LT + B.on_pressed', **after},
                     'policy_dir': 'config/policy/box_three/v1', 'stage': stage, 'blend_s': .5}
    cfg['FSM'] = fsm
    (config / 'box_three.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))


def export(source, runs, output):
    import numpy as np
    import torch
    import mujoco
    source, runs, output = source.resolve(), runs.resolve(), output.resolve()
    snapshots = runs / 'go-to-scratch-matt/inputs'
    # Scene modules resolve assets relative to their own file; verify these
    # against the saved inputs before importing from the complete worktree.
    for p in snapshots.iterdir():
        if p.is_file() and p.name != 'table_dr_train.py':
            if digest(p) != digest(source / 'policies' / p.name):
                raise ValueError(f'Source differs from training snapshot: {p.name}')
    sys.path.insert(0, str(source / 'policies'))
    import five_task
    from table_observations import contract as obs_contract
    import importlib.util
    spec = importlib.util.spec_from_file_location('saved_table_training', snapshots / 'table_dr_train.py')
    training = importlib.util.module_from_spec(spec); spec.loader.exec_module(training)
    actor_model = training.actor_model
    from table_domain_randomization import configure
    from unitree_balance.audit import contract
    from mjlab.entity import Entity

    for rel, sha in SELECTED.values():
        if digest(runs / rel) != sha:
            raise ValueError(f'Selected checkpoint changed: {rel}')
        for p in snapshots.iterdir():
            if p.is_file() and digest(p) != digest((runs / rel).parent / 'inputs' / p.name):
                raise ValueError(f'Training snapshots differ: {rel}: {p.name}')
    output.mkdir(parents=True, exist_ok=False)
    params = output / 'params'; params.mkdir()
    archived = ROOT / 'unitree_box/three_reference'
    archived.mkdir(exist_ok=True)
    for p in snapshots.iterdir():
        if p.is_file(): shutil.copy2(p, archived / p.name)
    shutil.copy2(source / 'policies/five_handoff.py', archived / 'five_handoff.py')
    import inspect
    from mjlab.envs import ManagerBasedRlEnv
    shutil.copy2(inspect.getfile(ManagerBasedRlEnv), archived / 'manager_based_rl_env.py')
    manifest = {'format': 'g1-three-table-history-v1', 'hardware_validated': False,
                'observation_contract': obs_contract(4), 'actuation': 'actuator_only',
                'source_root': str(source), 'actors': {}, 'files': {}}
    old = json.loads((ROOT / 'deploy/robots/g1_23dof/config/policy/box/v1/params/contract.json').read_text())
    starts, ends, lengths, actor_hashes, reference_hashes = [], [], [], [], []
    reference_width = 30 + 29 + 23 + 3  # root+joint qpos, qvel, qcmd, planned grip features
    for stage, (rel, _) in SELECTED.items():
        cfg = five_task.env_cfg(stage, 1, play=True)
        model, interface = contract(cfg)
        if interface != old:
            raise ValueError(f'{stage}: actuator/observation joint ordering changed')
        nominal = configure(cfg, 'nominal')
        gravity_model = Entity(nominal.scene.entities['robot']).spec.compile()
        folder = output / stage; folder.mkdir()
        cp = runs / rel
        data = torch.load(cp, map_location='cpu', weights_only=False)
        if data['infos']['observation_contract'] != obs_contract(4):
            raise ValueError(f'{stage}: observation contract mismatch')
        actor = actor_model(data['actor_state_dict'], 376, 'cpu')
        class Inference(torch.nn.Module):
            def __init__(self, actor):
                super().__init__(); self.norm = actor.obs_normalizer; self.mlp = actor.mlp
            def forward(self, observations):
                return self.mlp(self.norm(observations))
        with torch.inference_mode():
            torch.onnx.export(Inference(actor).eval(), torch.zeros(1, 376), str(folder / 'policy.onnx'),
                              input_names=['observations'], output_names=['actions'],
                              opset_version=18, dynamo=False)
        shutil.copy2(cp, folder / 'checkpoint.pt')
        (folder / 'source_metadata.json').write_text(json.dumps(data['infos'], indent=2) + '\n')
        bank_path = source / f'policies/five_runs/{stage}/banks/plan.pt'
        bank = torch.load(bank_path, map_location='cpu', weights_only=False)
        layout = json.loads((snapshots / 'five_stage.json').read_text())
        if bank['config_sha256'] != digest(snapshots / 'five_stage.json'):
            raise ValueError(f'{stage}: reference layout differs from training snapshot')
        if bank['joint_names'] != interface['joint_order'] or bank['fps'] != 50:
            raise ValueError(f'{stage}: reference joint order or frequency mismatch')
        plan = bank['plan']; length = int(plan['len'][0]); lengths.append(length)
        arrays = {key: plan[key][0, :length].numpy() for key in ('qpos', 'qvel', 'qcmd', 'grip', 'box')}
        np.savez(folder / 'reference.npz', **arrays)
        support = layout['source_top_m' if stage in ('go-to', 'take') else 'destination_top_m']
        held = arrays['grip'].astype(np.float32)
        share = np.clip((arrays['box'][:, 2] - layout['box_dimensions_m'][2] / 2 - support) / .04, 0, 1) * held
        features = np.stack((held, share, held), -1)
        frames = np.concatenate((arrays['qpos'], arrays['qvel'], arrays['qcmd'], features), -1).astype('<f4')
        assert frames.shape == (length, reference_width) and np.isfinite(frames).all()
        (folder / 'reference.bin').write_bytes(struct.pack('<II', length, reference_width) + frames.tobytes())
        starts.append(arrays['qpos'][0, 7:].tolist()); ends.append(arrays['qpos'][-1, 7:].tolist())
        actor_hashes.append(fnv(folder / 'policy.onnx')); reference_hashes.append(fnv(folder / 'reference.bin'))
        manifest['actors'][stage] = {'checkpoint_source': str(cp), 'checkpoint_sha256': digest(cp),
                                     'reference_source': str(bank_path), 'reference_sha256': digest(bank_path),
                                     'frames': length}
    manifest['interface'] = interface
    (params / 'contract.json').write_text(json.dumps(interface, indent=2) + '\n')
    shutil.copy2(snapshots / 'five_stage.json', params / 'layout.json')
    # Export the exact nominal arm gravity model. C++ evaluates only arm motor torques;
    # no floating-base or leg gravity forces are commanded.
    m = gravity_model
    assert m.body_jntnum[1] == 1 and m.jnt_type[m.body_jntadr[1]] == mujoco.mjtJoint.mjJNT_FREE
    assert not m.body_gravcomp[1]
    assert all(m.qpos0[m.jnt_qposadr[m.joint(n).id]] == 0 for n in interface['joint_order'])
    bodies = []
    for b in range(m.nbody):
        js = list(range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]))
        assert len(js) <= 1
        j = js[0] if js else -1
        motor = interface['joint_order'].index(m.joint(j).name) if j >= 0 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE else -1
        bodies.append(dict(parent=int(m.body_parentid[b]), motor=motor,
                           position=m.body_pos[b].tolist(), quaternion=m.body_quat[b].tolist(),
                           axis=m.jnt_axis[j].tolist() if motor >= 0 else [0., 0., 0.],
                           anchor=m.jnt_pos[j].tolist() if motor >= 0 else [0., 0., 0.],
                           com=m.body_ipos[b].tolist(), mass=float(m.body_mass[b] * m.body_gravcomp[b])))
    assert sum(b['mass'] > 0 for b in bodies) == 10
    assert all(m.jnt_actgravcomp[m.joint(n).id] == (i >= 13) for i, n in enumerate(interface['joint_order']))
    (params / 'gravity.json').write_text(json.dumps(bodies, indent=2) + '\n')
    # A portable robot-only model for independent gravity tests; never needed by C++.
    (params / 'gravity.xml').write_text(gravity_xml(m))
    def array(name, vals, kind='float'):
        def value(x):
            return f'{float(x):.10e}f' if kind == 'float' else str(x) + ('ULL' if kind == 'uint64_t' else '')
        return f'inline constexpr std::array<{kind},{len(vals)}> {name}{{{{' + ','.join(map(value, vals)) + '}};\n'
    h = '#pragma once\n#include <array>\n#include <cstdint>\nnamespace table_box {\n'
    h += 'inline constexpr std::array<const char*,3> kNames{{"go-to","take","put-down"}};\n'
    for name, vals, kind in [('kLengths',lengths,'int'), ('kActorHashes',actor_hashes,'uint64_t'), ('kReferenceHashes',reference_hashes,'uint64_t')]:
        h += array(name, vals, kind)
    for name, rows in [('kStarts',starts), ('kEnds',ends)]:
        h += f'inline constexpr std::array<std::array<float,23>,3> {name}{{{{\n'
        h += ',\n'.join('{{' + ','.join(f'{v:.10e}f' for v in row) + '}}' for row in rows) + '\n}};\n'
    h += 'struct Body { int parent,motor; std::array<float,3> position; std::array<float,4> quaternion; std::array<float,3> axis,anchor,com; float mass; };\n'
    h += f'inline constexpr std::array<Body,{len(bodies)}> kBodies{{{{\n'
    def vec(values): return '{{' + ','.join(f'{v:.10e}f' for v in values) + '}}'
    h += ',\n'.join('{' + f'{b["parent"]},{b["motor"]},' + ','.join(vec(b[k]) for k in ('position','quaternion','axis','anchor','com')) + f',{b["mass"]:.10e}f' + '}' for b in bodies)
    h += '\n}};\n}\n'
    (params / 'PolicyConfig.h').write_text(h)
    write_profile(interface, starts[0])
    manifest['source_files'] = {p.name: digest(p) for p in archived.iterdir() if p.is_file()}
    manifest['files'] = {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=PACKAGE)
    args = parser.parse_args()
    export(args.source_root, args.runs, args.output)
