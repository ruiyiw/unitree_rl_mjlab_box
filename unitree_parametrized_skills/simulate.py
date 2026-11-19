"""Walk N metres in native MuJoCo with the C++ controller g1_ctrl runs.

A CPU integration test without DDS: the training robot model, plain PD joints, and the
compiled walk controller in the loop.  It is not the batched mjlab evaluation (one robot,
nominal physics, no randomization) and says nothing about hardware.

  python -m unitree_parametrized_skills.simulate --policy adapter --distance 3.5 --output logs/walk_sim
  ... --from-velocity     start in the Velocity policy (sticks centred), then hand over as the profile does
  ... --again 1.0         when settled, walk a second distance straight away (the profile's "again" button)
"""
import argparse
import ctypes
import json
import math
from pathlib import Path

import numpy as np

from . import MOTOR_IDS, PACKAGE, POLICIES, PROFILE, ROOT
from .export import training
from .runtime import Core, build_core, library, telemetry, verify

GRAVITY = 9.81


class Velocity:
    """The profile's Velocity state, as State_RLBase runs it: its actor, gains, action scaling and rate limit."""

    def __init__(self, lib, directory):
        import yaml
        cfg = yaml.safe_load((directory / 'params/deploy.yaml').read_text())
        assert cfg['joint_ids_map'] == list(MOTOR_IDS) and cfg['step_dt'] == .02
        assert list(cfg['observations']) == ['base_ang_vel', 'projected_gravity', 'velocity_commands', 'gait_phase',
                                             'joint_pos_rel', 'joint_vel_rel', 'last_action']
        action = cfg['actions']['JointPositionAction']
        f = lambda v: np.array(v, np.float32)  # noqa: E731
        self.kp, self.kd, self.default = f(cfg['stiffness']), f(cfg['damping']), f(cfg['default_joint_pos'])
        self.scale, self.offset, self.clip, self.rate = f(action['scale']), f(action['offset']), f(action['clip']), f(action['rate_limit'])
        self.period = cfg['observations']['gait_phase']['params']['period']
        self.lib = ctypes.CDLL(str(lib))
        self.lib.onnx_open.argtypes, self.lib.onnx_open.restype = [ctypes.c_char_p], ctypes.c_void_p
        ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        self.lib.onnx_run.argtypes = [ctypes.c_void_p, ptr, ctypes.c_int, ptr, ctypes.c_int]
        self.handle = self.lib.onnx_open(str(directory / 'exported/policy.onnx').encode())
        assert self.handle
        self.raw, self.phase = np.zeros(23, np.float32), 0.

    def step(self, quaternion, gyro, joint_pos, joint_vel, command):
        w, x, y, z = quaternion
        gravity = [2 * (w * y - x * z), -2 * (y * z + w * x), -1 + 2 * (x * x + y * y)]
        self.phase = (self.phase + .02 / self.period) % 1.
        moving = np.linalg.norm(command) >= .1
        gait = [math.sin(2 * math.pi * self.phase), math.cos(2 * math.pi * self.phase)] if moving else [0., 0.]
        obs = np.concatenate((gyro, gravity, command, gait, joint_pos - self.default, joint_vel, self.raw)).astype(np.float32)
        out = np.empty(23, np.float32)
        assert not self.lib.onnx_run(self.handle, obs, len(obs), out, 23)
        self.raw = self.raw + np.clip(out - self.raw, -self.rate, self.rate)
        return np.clip(self.raw * self.scale + self.offset, self.clip[:, 0], self.clip[:, 1])


def robot_model(factory):
    """The training env's robot on its floor, compiled for CPU MuJoCo with the training solver settings."""
    import mujoco
    env_module, _, _, _, plans = training(factory)
    from mjlab.scene import Scene
    cfg = env_module.env_cfg(plans.bank_path('eval'), num_envs=1, mode='nominal')
    cfg.scene.num_envs = 1
    model = Scene(cfg.scene, device='cpu').compile()
    cfg.sim.mujoco.apply(model)
    assert model.nu == 23 and not model.body_gravcomp.any()
    return model, mujoco.MjData(model)


def play(policy, distance, output, factory=ROOT.parent / 'factory', from_velocity=False, stick=(0., 0., 0.),
         again=None, after_s=6., viewer=False, hold_s=None, blend_s=None, scale=1.):
    import mujoco
    import yaml
    manifest = verify()
    interface = manifest['interface']
    profile = yaml.safe_load(PROFILE.read_text())['FSM']
    state = profile[{'goal-conditioned': 'WalkGoal', 'adapter': 'WalkAdapter'}[policy]]
    hold_s = state['hold_s'] if hold_s is None else hold_s
    blend_s = state['blend_s'] if blend_s is None else blend_s
    model, data = robot_model(factory)
    plans, _ = library()
    joint_ids = [model.joint('robot/' + n).id for n in interface['joint_order']]
    qadr, vadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    actuated = model.actuator_trnid[:, 0].tolist()
    act = np.array([actuated.index(j) for j in joint_ids])
    kp, kd = np.array(interface['kp']), np.array(interface['kd'])
    np.testing.assert_allclose(model.actuator_gainprm[act, 0], kp)
    np.testing.assert_allclose(-model.actuator_biasprm[act, 2], kd)
    root = model.joint('robot/floating_base_joint')
    qa, va = int(root.qposadr[0]), int(root.dofadr[0])

    def gains(p, d):
        model.actuator_gainprm[act, 0], model.actuator_biasprm[act, 1], model.actuator_biasprm[act, 2] = p, -p, -d

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    lib = build_core(output)
    start = plans[100][0]                       # every plan starts in the same stand
    data.qpos[qa:qa + 7] = [0, 0, .76, 1, 0, 0, 0]
    data.qpos[qadr] = start[:23]
    mujoco.mj_forward(model, data)
    command = start[:23].astype(np.float64).copy()
    data.ctrl[act] = command
    display = None
    if viewer:
        import mujoco.viewer
        display = mujoco.viewer.launch_passive(model, data)
    frames, reason, last_v = [], 'completed', data.qvel[va:va + 3].copy()

    def sensors():
        nonlocal last_v
        q = data.qpos[qa + 3:qa + 7].copy()
        v = data.qvel[va:va + 3].copy()            # the pelvis's velocity in the world
        a = (v - last_v) / .02 + [0, 0, GRAVITY]
        last_v = v
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        return q, data.qvel[va + 3:va + 6].copy(), R.reshape(3, 3).T @ a, data.qpos[qadr].copy(), data.qvel[vadr].copy()

    def advance(target):
        frames.append(data.qpos.copy())
        data.ctrl[act] = target
        for _ in range(4):
            mujoco.mj_step(model, data)
        if display:
            display.sync()

    def fallen():
        up = np.zeros(9)
        mujoco.mju_quat2Mat(up, data.qpos[qa + 3:qa + 7])
        return up[8] < math.cos(math.pi / 4) or data.qpos[qa + 2] < .45

    if from_velocity:                              # stand (and optionally walk) on the Velocity policy first
        velocity = Velocity(lib, ROOT / 'deploy/robots/g1_23dof' / profile['Velocity']['policy_dir'])
        gains(velocity.kp, velocity.kd)
        for tick in range(400):
            q, gyro, _, jp, jv = sensors()
            moving = 50 <= tick < 250              # 4 s of stick, then 3 s with the sticks centred
            command = velocity.step(q, gyro, jp, jv, np.array(stick if moving else (0., 0., 0.), np.float32))
            advance(command)
            if fallen():
                reason = 'fell on the Velocity policy'
                break
        kp0, kd0 = velocity.kp.astype(float), velocity.kd.astype(float)
    else:
        kp0, kd0 = kp, kd
    walks = []
    with Core(lib, policy) as core:
        core.rate_limit(state.get('rate_limit'))             # the profile's action-rate limit, as State_ParametrizedWalk
        previous = None
        for n, entry in [(distance, True)] + ([(again, False)] if again else []):
            if reason != 'completed':
                break
            planned = core.start(n, hold_s if entry else 0., previous, scale)
            q0 = np.array(command, float)
            origin, yaw0 = data.qpos[qa:qa + 2].copy(), math.atan2(2 * (data.qpos[qa + 3] * data.qpos[qa + 6]), 1 - 2 * data.qpos[qa + 6] ** 2)
            t_enter, done_at, steps = len(frames), None, 0
            total = round(hold_s / .02) * entry + len(plans[round(planned * 1000)]) + round(after_s / .02)
            for tick in range(total):
                try:
                    obs, action, target, flags, estimate = core.step(telemetry(*sensors()))
                except RuntimeError as error:
                    reason = str(error)
                    break
                s = .5 - .5 * math.cos(math.pi * min(tick * .02 / blend_s, 1.)) if entry and blend_s > 0 else 1.
                gains(kp0 + (kp - kp0) * s, kd0 + (kd - kd0) * s)
                command = q0 + (target - q0) * s
                advance(command)
                steps += 1
                if flags[2] and done_at is None:
                    done_at = tick * .02
                    if again and entry:
                        break
                if fallen():
                    reason = 'fell'
                    break
            c, s_ = math.cos(yaw0), math.sin(yaw0)
            d = data.qpos[qa:qa + 2] - origin
            along, across = c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]
            yaw = math.atan2(2 * (data.qpos[qa + 3] * data.qpos[qa + 6] + data.qpos[qa + 4] * data.qpos[qa + 5]),
                             1 - 2 * (data.qpos[qa + 5] ** 2 + data.qpos[qa + 6] ** 2))
            walks.append({'requested_m': n, 'planned_m': planned, 'walked_m': float(along), 'sideways_m': float(across),
                          'error_m': float(math.hypot(along - planned, across)),
                          'estimate_minus_truth_m': float(estimate[0] - plans[round(planned * 1000)][0, 23] - along),
                          'heading_change_deg': math.degrees(math.atan2(math.sin(yaw - yaw0), math.cos(yaw - yaw0))),
                          'settled_s_after_entry': done_at, 'plan_s': (len(plans[round(planned * 1000)]) - 1) * .02,
                          'steps': steps, 'first_frame': t_enter})
            previous = action
    if display:
        display.close()
    report = {'policy': policy, 'from_velocity': from_velocity, 'stick_before': list(stick), 'blend_s': blend_s,
              'hold_s': hold_s, 'stop_reason': reason, 'walks': walks,
              'physics': 'native CPU MuJoCo, nominal robot, plain PD; not the batched GPU evaluation',
              'hardware_tested': False}
    np.savez_compressed(output / 'trajectory.npz', qpos=np.asarray(frames), dt=.02)
    (output / 'simulation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=tuple(POLICIES), required=True)
    parser.add_argument('--distance', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--factory-root', type=Path, default=ROOT.parent / 'factory')
    parser.add_argument('--from-velocity', action='store_true')
    parser.add_argument('--stick', type=float, nargs=3, default=(0., 0., 0.), help='forward, left, turn before the hand-over')
    parser.add_argument('--again', type=float, default=None)
    parser.add_argument('--viewer', action='store_true')
    a = parser.parse_args()
    play(a.policy, a.distance, a.output, a.factory_root, a.from_velocity, tuple(a.stick), a.again, viewer=a.viewer)
