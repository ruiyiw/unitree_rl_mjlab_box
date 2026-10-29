"""Import the selected immutable actors, references and source runtime."""
import argparse
import copy
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

from . import ACTORS, MOTOR_IDS, ORDER, POLICY_DIR, ROOT

SELECTED = 'results/autoresearch-20261029/final-selection'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fnv(path):
    value = 14695981039346656037
    for byte in Path(path).read_bytes():
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def export(source, output):
    import numpy as np
    import torch
    import yaml
    source, output = source.resolve(), output.resolve()
    selected = source / SELECTED
    if digest(selected / 'policy.pt') != '324a76c836b5a8e1f24b53ca93c0d636fae5fcb481bc008cce3d5a8ddbbbeed7':
        raise ValueError('Expected the final selected 2026-10-29 sequence bundle')
    payload = torch.load(selected / 'policy.pt', map_location='cpu', weights_only=False)
    if payload['format'] != 'five-stage-composite-proprio-v3' or tuple(payload['order']) != ORDER:
        raise ValueError('Expected five separate selected actors')
    if payload['simulation_actuation_mode'] != 'none':
        raise ValueError('Only plain-PD selected policies are supported')
    sys.path.insert(0, str(source / 'policies'))
    import five_proprio
    import five_task
    from unitree_balance.audit import contract
    if payload['observation_contract'] != five_proprio.CONTRACT:
        raise ValueError('Observation contract mismatch')
    if digest(source / 'policies/five_sequence.py') != payload['controller_sha256']:
        raise ValueError('Reference controller changed')
    if digest(source / 'policies/five_stage.json') != payload['config_sha256']:
        raise ValueError('Task layout changed')
    model, interface = contract(five_task.env_cfg('go-to', 1, play=True))
    assert model.nu == 23 and not model.body_gravcomp.any()
    assert interface['policy_dt'] == .02
    for stage in ORDER:
        _, other = contract(five_task.env_cfg(stage, 1, play=True))
        assert other == interface, stage
    output.mkdir(parents=True, exist_ok=False)
    params = output / 'params'; params.mkdir()
    manifest = {'source': str(selected), 'bundle_sha256': digest(selected / 'policy.pt'),
                'hardware_validated': False, 'actors': {}, 'files': {}, 'interface': interface,
                'observation_contract': payload['observation_contract']}
    shutil.copy2(selected / 'policy.pt', output / 'policy.pt')
    for name in ('README.txt', 'report.json', 'evaluation.json'):
        shutil.copy2(selected / name, output / name)
    (params / 'contract.json').write_text(json.dumps(interface, indent=2)+'\n')
    shutil.copy2(source / 'policies/five_stage.json', params / 'layout.json')
    starts=[]; ends=[]; lengths=[]; actor_hashes=[]; reference_hashes=[]
    for stage, actor in zip(ORDER, ACTORS):
        metadata = json.loads((selected / 'actors' / actor / 'metadata.json').read_text())
        checkpoint = Path(payload['checkpoint_paths'][actor])
        if digest(checkpoint) != metadata['checkpoint_sha256']:
            raise ValueError(f'Checkpoint mismatch: {actor}')
        state = five_proprio.validate_checkpoint(checkpoint)['actor_state_dict']
        assert set(state) == set(payload['actors'][actor])
        assert all(torch.equal(state[k], payload['actors'][actor][k]) for k in state)
        folder = output / stage; folder.mkdir()
        shutil.copy2(checkpoint, folder / 'checkpoint.pt')
        shutil.copy2(selected / 'actors' / actor / 'policy.onnx', folder / 'policy.onnx')
        shutil.copy2(selected / 'actors' / actor / 'metadata.json', folder / 'source_metadata.json')
        bank = payload['references'][stage]
        reference = source / 'policies/five_runs' / stage / 'banks/plan.pt'
        assert digest(reference) == payload['bank_sha256'][stage]
        original = torch.load(reference, map_location='cpu', weights_only=False)
        assert all(torch.equal(original['plan'][k], bank['plan'][k]) for k in bank['plan'])
        assert bank['fps'] == 50
        perm = [bank['joint_names'].index(n) for n in interface['joint_order']]
        length = int(bank['plan']['len'][0]); lengths.append(length)
        qcmd = bank['plan']['qcmd'][0, :length, perm].numpy().astype('<f4')
        assert qcmd.shape == (length, 23) and np.isfinite(qcmd).all()
        (folder / 'reference.bin').write_bytes(struct.pack('<I', length) + qcmd.tobytes())
        starts.append(bank['plan']['qpos'][0, 0, 7:][perm].tolist())
        ends.append(bank['plan']['qpos'][0, length-1, 7:][perm].tolist())
        actor_hashes.append(fnv(folder / 'policy.onnx')); reference_hashes.append(fnv(folder / 'reference.bin'))
        manifest['actors'][stage] = {'actor': actor, 'checkpoint_source': str(checkpoint),
            'checkpoint_sha256': digest(checkpoint), 'onnx_sha256': digest(folder / 'policy.onnx'),
            'reference_source_sha256': payload['bank_sha256'][stage], 'frames': length}
    # The source runtime is kept intact, in its own import path, for reproducibility.
    runtime = ROOT / 'unitree_box/reference'
    runtime.mkdir(exist_ok=True)
    policies = runtime / 'policies'; policies.mkdir(exist_ok=True)
    from five_provenance import SOURCES
    for name in (*SOURCES, 'five_teacher_sequence.py', 'five_actuation.py', 'five_stage.json'):
        shutil.copy2(source / 'policies' / name, policies / name)
    shutil.copytree(source / 'models/unitree_g1_23dof', runtime / 'models/unitree_g1_23dof', dirs_exist_ok=True)
    for stage in (*ORDER, 'navigation'):
        dest = policies / 'five_runs' / stage / 'banks'; dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / 'policies/five_runs' / stage / 'banks/plan.pt', dest / 'plan.pt')
    provenance = {str(p.relative_to(runtime)): digest(p) for p in runtime.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    (ROOT / 'unitree_box/source_manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    def array(name, values, kind='float'):
        vals = ', '.join(f'{float(x):.10e}f' if kind == 'float' else str(x) + ('ULL' if kind == 'uint64_t' else '') for x in values)
        return f'inline constexpr std::array<{kind}, {len(values)}> {name}{{{{{vals}}}}};\n'
    header = '#pragma once\n#include <array>\n#include <cstdint>\nnamespace box_policy {\ninline constexpr float kStepDt = .02f;\n'
    header += array('kMotorIds', MOTOR_IDS, 'int')
    for name,key in [('kDefault','default_joint_positions'),('kScale','action_scale'),('kKp','kp'),('kKd','kd')]:
        header += array(name,interface[key])
    header += array('kLengths',lengths,'int') + array('kActorHashes',actor_hashes,'uint64_t') + array('kReferenceHashes',reference_hashes,'uint64_t')
    header += 'inline constexpr std::array<const char*,5> kNames{{"go-to","take","carry","put-down","return"}};\n'
    for name,values in [('kStarts',starts),('kEnds',ends)]:
        header += f'inline constexpr std::array<std::array<float,23>,5> {name}{{{{\n'
        header += ',\n'.join('{{'+', '.join(f'{v:.10e}f' for v in row)+'}}' for row in values)+'\n}};\n'
    (params / 'PolicyConfig.h').write_text(header+'}\n')
    base = yaml.safe_load((ROOT / 'deploy/robots/g1_23dof/config/human_push.yaml').read_text())
    for choice in (*ORDER,'sequence'):
        cfg=copy.deepcopy(base); fsm=cfg['FSM']
        fsm['_'].pop('HumanPushRecovery'); fsm.pop('HumanPushRecovery')
        fsm['_']['BoxPolicy']={'id':7,'type':'BoxPolicy'}
        fsm['BoxPolicy']={'transitions':{'Passive':'LT + B.on_pressed'},'policy_dir':'config/policy/box/v1','stage':choice}
        fsm['FixStand']['transitions']={'Passive':'LT + B.on_pressed','BoxPolicy':'RT + A.on_pressed'}
        for key in ('kp','kd'):
            values=[0.]*29
            for i,motor in enumerate(MOTOR_IDS): values[motor]=interface[key][i]
            fsm['FixStand'][key]=values
        pose=[0.]*29
        for i,motor in enumerate(MOTOR_IDS): pose[motor]=starts[0 if choice=='sequence' else ORDER.index(choice)][i]
        fsm['FixStand']['qs']=[[],pose]
        (ROOT / f'deploy/robots/g1_23dof/config/box_{choice}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    manifest['files']={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
    (output / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=ROOT.parent/'humanoid-full')
    p.add_argument('--output',type=Path,default=POLICY_DIR)
    a=p.parse_args();export(a.source_root,a.output)

if __name__=='__main__':main()
