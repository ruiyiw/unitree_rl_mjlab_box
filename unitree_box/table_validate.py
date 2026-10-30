"""Compare the C++ controller with the saved training observation functions and PyTorch actor."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from mjlab.utils.lab_api.math import quat_apply_inverse

from . import ROOT, MOTOR_IDS
from .table_export import PACKAGE, SELECTED
from .table_runtime import Core, build_core, verify
from .table_clock import source_clock


def saved_functions():
    """Execute the archived function bodies without importing the CUDA training environment."""
    source = ROOT / 'unitree_box/three_reference'
    scope = {'torch': torch, 'quat_apply_inverse': quat_apply_inverse}
    def extract(file, names, cls=None):
        tree = ast.parse((source / file).read_text())
        nodes = tree.body if cls is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
        selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in selected} == set(names)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source / file), 'exec'), scope)
    extract('task.py', ('wrap', 'yaw_of', 'rot2'))
    tk = NS(**{k: scope[k] for k in ('wrap', 'yaw_of', 'rot2')}, task=lambda env: env.task,
            obs_last_joint_action=lambda env: env.previous)
    scope['tk'] = tk
    extract('table_observations.py', ('sensors',), 'TableCommand')
    extract('table_observations.py', ('actor_observation', 'plan_features'))
    return scope


class Oracle:
    def __init__(self, stage, plan, interface):
        self.func = saved_functions()
        self.skill = stage; self.device = 'cpu'; self.num_envs = 1; self.cfg = NS(tier='nominal', history=4)
        self.setup = json.loads((PACKAGE / 'params/layout.json').read_text())
        self.plan = {k: torch.tensor(v) for k, v in plan.items()}
        self.clock = torch.zeros((1, 1), dtype=torch.long)
        self.heading_origin = torch.zeros(1); self.arm_ids = list(range(13, 23))
        self.sensor_step = torch.full((1,), -1, dtype=torch.long)
        self.sensor_fresh = torch.ones(1, dtype=torch.bool)
        self.delay_lag = torch.zeros(1, dtype=torch.long)
        self.sensor_current = torch.zeros(1, 86); self.sensor_previous = torch.zeros(1, 86)
        self.delayed = torch.zeros(1, 86); self.history = torch.zeros(1, 4, 75)
        self.robot = NS(data=NS(default_joint_pos=torch.tensor(interface['default_joint_positions'])[None]))
        self._env = NS(task=self, num_envs=1, device='cpu', common_step_counter=0,
                       scene={'robot/imu_ang_vel': NS(data=torch.zeros(1, 3))}, previous=torch.zeros(1, 23))
        self.sensors = lambda: self.func['sensors'](self)

    def raw(self, k, key, off=0):
        return self.plan[key][min(int(self.clock[0, 0]) + off, len(self.plan['qpos']) - 1)][None]

    def lengths(self): return torch.tensor([len(self.plan['qpos'])])
    def ended(self): return self.clock[:, 0] >= self.lengths() - 1

    def observation(self, telemetry, command, previous, tick, clock):
        q = torch.tensor(telemetry[:4])[None]; q = q / q.norm(dim=-1, keepdim=True)
        r = self.robot.data
        r.root_link_quat_w = q
        r.projected_gravity_b = quat_apply_inverse(q, torch.tensor([[0., 0., -1.]]))
        r.joint_pos_biased = torch.tensor(telemetry[7 + np.array(MOTOR_IDS)])[None]
        r.joint_vel = torch.tensor(telemetry[36 + np.array(MOTOR_IDS)])[None]
        r.joint_pos_target = torch.tensor(command)[None]
        self._env.scene['robot/imu_ang_vel'].data = torch.tensor(telemetry[4:7])[None]
        self._env.previous = torch.tensor(previous)[None]
        if tick == 0:
            self.heading_origin = self.func['yaw_of'](q) - self.func['yaw_of'](self.raw(0, 'qpos')[:, 3:7])
        self.clock[0, 0] = clock
        self._env.common_step_counter = tick
        return self.func['actor_observation'](self._env)


def load_actor(stage):
    state = torch.load(PACKAGE / stage / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor_state_dict']
    dummy = TensorDict({'actor': torch.zeros(1, 376)}, batch_size=[1])
    model = MLPModel(dummy, {'actor': ['actor']}, 'actor', 23, hidden_dims=(512, 256, 128),
                     activation='elu', obs_normalization=True,
                     distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': .3, 'std_type': 'scalar'}).eval()
    model.load_state_dict(state, strict=True)
    return model


def validate(output):
    import mujoco
    torch.set_num_threads(1)
    manifest = verify(); interface = manifest['interface']
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    lib = build_core(output)
    gravity = mujoco.MjModel.from_xml_path(str(PACKAGE / 'params/gravity.xml'))
    data = mujoco.MjData(gravity)
    joints = [gravity.joint(n).id for n in interface['joint_order']]
    qadr = gravity.jnt_qposadr[joints]; vadr = gravity.jnt_dofadr[joints]
    rng = np.random.default_rng(20261030); ids = np.array(MOTOR_IDS)
    report = {'hardware_tested': False, 'stages': {}, 'method': 'Saved training functions, frozen PyTorch actors, native MuJoCo gravity'}
    for stage_id, stage in enumerate(SELECTED):
        plan = dict(np.load(PACKAGE / stage / 'reference.npz'))
        actor = load_actor(stage); oracle = Oracle(stage, plan, interface)
        previous = np.zeros(23, np.float32); command = plan['qcmd'][0].copy()
        maxima = np.zeros(4); length = len(plan['qpos'])
        clock_source = source_clock(stage, length)
        clock_trace = []
        with Core(lib, stage_id) as core, torch.inference_mode():
            for tick in range(length + 65):
                if tick: clock_source._update_command(None)
                frame = int(clock_source.clock[0, 0])
                if tick < 6: clock_trace.append(frame)
                telemetry = np.zeros(65, np.float32)
                # Both heading turns and nonzero roll/pitch; arbitrary yaw offset
                # checks episode-local heading anchoring, including wrap at +/-pi.
                angle = 2.7
                q = plan['qpos'][frame, 3:7].copy()
                w,x,y,z=q; c,s=np.cos(angle/2),np.sin(angle/2)
                telemetry[:4] = [c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w]
                telemetry[4:7] = rng.uniform(-.1,.1,3) if tick < length else 0
                telemetry[7 + ids] = plan['qpos'][frame, 7:] + (rng.uniform(-.02,.02,23) if tick < length else 0)
                telemetry[36 + ids] = rng.uniform(-.2,.2,23) if tick < length else 0
                expected_obs = oracle.observation(telemetry, command, previous, tick, frame)
                expected_action = actor({'actor': expected_obs})[0].numpy()
                obs, action, target, torque, state = core.step(telemetry, command)
                expected_target = plan['qcmd'][min(frame+1,length-1)] + np.array(interface['action_scale'])*expected_action
                # Independently compute MuJoCo's gravity generalized forces for
                # this pose; actuator_only sends these on arm motors exclusively.
                data.qpos[:7] = [0,0,.76,*telemetry[:4]]; data.qpos[qadr] = telemetry[7+ids]
                data.qvel[:] = 0; mujoco.mj_forward(gravity, data)
                expected_torque = data.qfrc_gravcomp[vadr].copy(); expected_torque[:13] = 0
                errors = [np.max(np.abs(obs-expected_obs[0].numpy())), np.max(np.abs(action-expected_action)),
                          np.max(np.abs(target-expected_target)), np.max(np.abs(torque-expected_torque))]
                maxima = np.maximum(maxima, errors)
                np.testing.assert_allclose(obs, expected_obs[0], atol=4e-6, rtol=2e-6)
                np.testing.assert_allclose(action, expected_action, atol=2e-5, rtol=2e-5)
                np.testing.assert_allclose(target, expected_target, atol=1e-5, rtol=2e-5)
                np.testing.assert_allclose(torque, expected_torque, atol=5e-6, rtol=3e-6)
                assert state[0] == frame
                # Use the same residual/actual command in both implementations;
                # avoids compounding numeric error when validating observation math.
                previous = action.copy(); command = target.copy()
            assert state[1] == 1, 'Endpoint did not settle'
            core.reset(np.ones(23, np.float32))
            reset_obs, *_ = core.step(telemetry, command)
            for h in range(4):
                np.testing.assert_array_equal(reset_obs[76+h*75+52:76+(h+1)*75], 0)
                np.testing.assert_array_equal(reset_obs[76+h*75:76+(h+1)*75], reset_obs[76:151])
        report['stages'][stage] = dict(samples=length+65, observation_max_error=maxima[0], actor_max_error=maxima[1],
                                       target_max_error_rad=maxima[2], gravity_max_error_nm=maxima[3],
                                       clock_and_endpoint_passed=True, initial_reference_clocks=clock_trace, history_reset_passed=True)
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2)); return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    validate(parser.parse_args().output)
