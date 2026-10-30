"""Exercise the actual archived reset/update methods without physics or CUDA."""
import ast
from types import SimpleNamespace as NS

import torch
from . import ROOT


def source_clock(stage, length):
    root = ROOT / 'unitree_box/three_reference'
    scope = {'torch': torch, 'tk': NS(_invalidate=lambda env: None)}
    def method(file, cls, name):
        tree = ast.parse((root / file).read_text())
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
        return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    # Keep the inheritance chain: HandoffCommand and ResearchCommand call super().
    for file, name, base in [('five_task.py', 'FiveCommand', 'object'),
                             ('five_handoff.py', 'HandoffCommand', 'FiveCommand'),
                             ('five_research.py', 'ResearchCommand', 'HandoffCommand')]:
        node = ast.ClassDef(name=name, bases=[ast.Name(id=base, ctx=ast.Load())], keywords=[],
                            body=[method(file, name, '_update_command')], decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(module, str(root / file), 'exec'), scope)
    t = scope['ResearchCommand']()
    t.clock = torch.zeros(1, 1, dtype=torch.long)
    t.fresh = torch.ones(1, dtype=torch.bool)
    # TableCommand nominal reset: go-to begins; take and put-down restore a predecessor.
    t.just_restored = torch.tensor([stage != 'go-to'])
    t.phase = torch.zeros(1, dtype=torch.long)
    t.stood = torch.zeros(1); t.hold_required = torch.ones(1)
    t.switch_timer = torch.zeros(1); t.switch_due = torch.zeros(1, dtype=torch.bool)
    t.metrics = {'success': torch.zeros(1), 'grip': torch.zeros(1), 'box_error': torch.zeros(1), 'physical_valid_at_switch': torch.zeros(1)}
    t.cfg = NS(sensor_success=False)
    t.lengths = lambda: torch.tensor([length])
    t.R = lambda name: torch.zeros(1, 4) if name == 'box' else torch.zeros(1)
    t.standing = lambda: (torch.zeros(1, dtype=torch.bool), None)
    t.ended = lambda: t.clock[:, 0] >= length - 1
    t.controller_ready = lambda: torch.ones(1, dtype=torch.bool)
    t._env = NS(step_dt=.02)
    scope['tk'].geo = lambda env: NS(grip=torch.zeros(1), box=torch.zeros(1, 3))
    # Execute the recorded mjlab reset method too: its env_ids argument matters.
    reset = method('manager_based_rl_env.py', 'ManagerBasedRlEnv', 'reset')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), reset], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), 'saved_mjlab_reset', 'exec'), scope)
    env = NS(num_envs=1, device='cpu', extras={}, _reset_idx=lambda ids: None,
             scene=NS(write_data_to_sim=lambda: None), sim=NS(forward=lambda: None, sense=lambda: None),
             command_manager=NS(compute=lambda dt, env_ids: t._update_command(env_ids)),
             observation_manager=NS(compute=lambda **kwargs: {}),
             recorder_manager=NS(record_post_reset=lambda ids: None))
    scope['reset'](env)
    return t
