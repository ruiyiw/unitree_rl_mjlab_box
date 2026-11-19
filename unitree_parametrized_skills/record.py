"""Record what the training environment does, for the C++ controller to be checked against.

Runs both policies in the mjlab training env (GPU; nominal robot, clean sensors, the Kalman
estimator) on a few distances and saves, for every 20 ms step, the sensors a real robot has
and what training computed from them: the actor's observation, its correction and the joint
targets.  validate.py replays the sensors through the C++ controller.

  python -m unitree_parametrized_skills.record --output tests/fixtures/parametrized_walk
"""
import argparse
import json
from pathlib import Path

from . import PACKAGE, POLICIES, ROOT
from .export import training

DISTANCES = (0.3, 2.5, 5.0)        # the last is the library's longest (v3: 5 m)
AFTER_S = 5.0                    # recorded after the plan's end: standing at the goal


def record(factory, output, distances=DISTANCES, after_s=AFTER_S, seed=7):
    import numpy as np
    import torch
    env_module, models, _, paths, plans = training(factory)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.lab_api.math import quat_apply_inverse
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bank = output / 'bank.pt'
    plans.make_bank(distances, str(bank), workers=len(distances))
    manifest = json.loads((PACKAGE / 'manifest.json').read_text())
    runs = Path(paths.RUNS)
    history = manifest.get('history', 1)

    # the accelerometer's reading in the pelvis frame, as WalkCommand._accel forms it
    original = env_module.WalkCommand._accel

    def accel(self, R_imu, fresh):
        r = self.robot.data
        v = r.root_link_lin_vel_w
        a = torch.where(fresh[:, None], torch.zeros_like(v), (v - self._vel_w) / self._env.step_dt)
        a = a + torch.tensor([0.0, 0.0, env_module.GRAVITY], device=v.device)
        self.specific_force = quat_apply_inverse(r.root_link_quat_w, a)
        return original(self, R_imu, fresh)
    env_module.WalkCommand._accel = accel

    for name, (run, base_run) in POLICIES.items():
        info = manifest['actors'][name]
        run, base_run = info['run'], info.get('base_run')      # the runs the package was exported from
        base = str(runs / base_run / 'selected.pt') if base_run else None
        b = torch.load(bank, map_location='cpu', weights_only=False)
        seconds = (int(b['len'].max()) - 1) / b['fps'] + after_s + 2.0
        cfg = env_module.env_cfg(str(bank), num_envs=len(distances), mode='nominal', adapter_base=base,
                                 adapter_bound=info.get('adapter_bound', 0.5), episode_s=seconds, estimator='kalman',
                                 history=history, base_history=history if base else 1)
        cfg.seed = seed
        torch.manual_seed(seed)
        env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
        c = env_module.cmd(env)
        obs, _ = env.reset()
        act = models.frozen_actor(str(runs / run / 'selected.pt'), obs['actor'].shape[-1], 23, env.device,
                                  models.ADAPTER_HIDDEN if base else models.HIDDEN)
        term = env.action_manager.get_term('joint_pos')
        robot = env.scene['robot'].data
        c.specific_force = torch.zeros(env.num_envs, 3, device=env.device)
        keys = ('quaternion', 'gyro', 'accelerometer', 'joint_pos', 'joint_vel', 'observation', 'action', 'target',
                'clock', 'estimate', 'heading', 'pelvis', 'yaw')
        log = {k: [] for k in keys}
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        steps = torch.zeros(env.num_envs, dtype=torch.long)
        for _ in range(int(b['len'].max()) + round(after_s / env.step_dt)):
            o = obs['actor']
            now = {'quaternion': robot.root_link_quat_w, 'gyro': o[:, 3 * (history - 1):3 * history], 'accelerometer': c.specific_force,
                   'joint_pos': robot.joint_pos_biased, 'joint_vel': robot.joint_vel, 'observation': o,
                   'clock': c.clock[:, None].float(), 'estimate': c.odo.pos, 'heading': c.yaw_est[:, None],
                   'pelvis': c.pos_true, 'yaw': c.yaw_true[:, None]}
            with torch.no_grad():
                action = act(o)
            obs, _, terminated, truncated, _ = env.step(action)
            now['action'], now['target'] = term.applied, term._processed_actions
            for k in keys:
                log[k].append(now[k].detach().cpu().numpy().astype(np.float32).copy())
            steps += alive.cpu().long()
            alive &= ~(terminated | truncated)          # an episode that ended early is compared up to there
        env.close()
        for i, n in enumerate(distances):
            length = min(int(steps[i]), int(b['len'][i]) + round(after_s / env.step_dt))
            np.savez_compressed(output / f'{name}_{n:g}m.npz', distance_m=n, plan_frames=int(b['len'][i]),
                                ended_early=int(steps[i]) < len(log['clock']),
                                **{k: np.stack(log[k])[:length, i] for k in keys})
            print(name, n, 'steps', length, 'plan', int(b['len'][i]), 'walked', float(log['pelvis'][length - 1][i, 0] - log['pelvis'][0][i, 0]))
    bank.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factory-root', type=Path, default=ROOT.parent / 'factory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--distances', type=float, nargs='*', default=list(DISTANCES))
    args = parser.parse_args()
    record(args.factory_root, args.output, tuple(args.distances))
