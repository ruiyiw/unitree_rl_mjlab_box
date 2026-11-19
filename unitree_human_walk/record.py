"""Record the policy in the training environment for the C++ comparison (validate.py).

One robot from the reference's first frame, mean actions, the torso's pose and the IMU's
velocity from estimate.py.  Saved for every step: the sensors, and what the environment made
of them (observation, estimate, correction, joint targets, the arm motors' gravity torque).

  python -m unitree_human_walk.record --output tests/fixtures/human_walk
"""
import argparse
from pathlib import Path

import numpy as np

from . import CHECKPOINT, MOTION, TRAINING
from .training import install


def record(output, root=TRAINING, checkpoint=CHECKPOINT, pushes=False, after_s=3., seed=7):
    import mujoco
    import torch
    motion, tk = install(root)
    folder = Path(root) / MOTION
    torch.manual_seed(seed)
    seconds = motion.episode_s(motion.plan_path(str(folder))) + after_s
    player = motion.Player(str(folder), 1, seconds, seed=seed, pushes=pushes)
    env, t = player.env, player.task
    policy = player.policy(str(folder / checkpoint))
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    joints = [model.joint('robot/' + n) for n in env.scene['robot'].joint_names]
    dofs = [int(j.dofadr[0]) for j in joints]
    assert all(model.jnt_actgravcomp[j.id] == (i >= 13) for i, j in enumerate(joints))   # the ten arm motors compensate
    obs, _ = player.vec.reset()
    r = env.scene['robot'].data
    rows = {k: [] for k in ('quaternion', 'gyro', 'accelerometer', 'joint_pos', 'joint_vel', 'observation', 'action',
                            'target', 'clock', 'estimate', 'velocity', 'pelvis', 'torque')}
    ended = False
    with torch.no_grad():
        for _ in range(int(round(seconds / env.step_dt)) - 1):
            rb = env._human_walk
            s = rb.sensors
            for k in ('quaternion', 'gyro', 'accelerometer', 'joint_pos', 'joint_vel'):
                rows[k].append(s[k][0].cpu().numpy().copy())
            rows['observation'].append(obs['actor'][0].cpu().numpy().copy())
            rows['clock'].append(int(t.clock[0, 0]))
            rows['estimate'].append(rb.estimate.pelvis()[0].cpu().numpy().copy())
            rows['velocity'].append(rb.estimate.velocity()[0].cpu().numpy().copy())
            rows['pelvis'].append((r.root_link_pos_w[0] - env.scene.env_origins[0]).cpu().numpy().copy())
            data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            rows['torque'].append(data.qfrc_gravcomp[dofs].copy())      # what MuJoCo's arm actuators add
            action = policy(obs)
            rows['action'].append(action[0].cpu().numpy().copy())
            obs, _, dones, _ = player.vec.step(action)
            rows['target'].append(env.action_manager.get_term('joint_pos')._processed_actions[0].cpu().numpy().copy())
            if bool(dones[0]):
                ended = True
                break
    out = {k: np.asarray(v) for k, v in rows.items()}
    out.update(plan_frames=int(t.plan.sk['len'][0]), ended_early=ended, pushes=pushes, dt=env.step_dt)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / ('pushes.npz' if pushes else 'nominal.npz')
    np.savez_compressed(path, **out)
    print(path, len(out['clock']), 'steps; clock', out['clock'][:4], '...', out['clock'][-1], 'ended early:', ended,
          '; pelvis', out['pelvis'][-1].round(3), 'estimate', out['estimate'][-1].round(3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, default=TRAINING)
    parser.add_argument('--pushes', action='store_true')
    a = parser.parse_args()
    record(a.output, a.training_root, pushes=a.pushes)
