"""The training environment (humanoid-full/policies/motion.py) with the robot's own inputs.

install() replaces the three observation terms that read the simulator's pose and velocity
(the torso relative to the plan, the plan ahead of the torso, the IMU's linear velocity) with
the same terms computed from estimate.py.  Everything else is training's.
"""
import sys

import torch

from . import TRAINING
from .estimate import Estimate, yaw_of

GRAVITY = 9.81


def modules(root=TRAINING):
    """motion.py and the task it builds on, importable."""
    root = str(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    import motion
    import task
    return motion, task


class Robot:
    """One env's worth of state per robot: the sensors the G1 has, read from the simulator, and
    the estimate made from them, updated once per control step."""

    def __init__(self, env, motion):
        self.env, self.motion = env, motion
        r = env.scene['robot']
        self.estimate = Estimate(env.sim.mj_model, r.joint_names, env.num_envs, env.device, prefix='robot/')
        self.last_velocity = torch.zeros(env.num_envs, 3, device=env.device)
        self.at = -1
        self.scale = 1.
        self.sensors = {}

    def sync(self):
        env = self.env
        if env.common_step_counter == self.at:
            return self
        from mjlab.tasks.velocity import mdp
        from mjlab.utils.lab_api.math import quat_apply_inverse
        r = env.scene['robot'].data
        plan = self.motion.task(env).plan.sk['qpos'][0, 0]
        new = (env.episode_length_buf == 0) | (self.at < 0)
        v = r.root_link_lin_vel_w
        self.last_velocity[new] = v[new]
        change = (v - self.last_velocity) / env.step_dt
        change[:, 2] += GRAVITY
        self.last_velocity = v.clone()
        # the gyro is read through the term's own function (install() has kept the original)
        self.sensors = {'quaternion': r.root_link_quat_w.clone(), 'gyro': self.gyro(env).clone(),
                        'accelerometer': quat_apply_inverse(r.root_link_quat_w, change),
                        'joint_pos': r.joint_pos.clone(), 'joint_vel': r.joint_vel.clone(), 'new': new.clone()}
        s = self.sensors
        self.estimate.step(s['quaternion'], s['accelerometer'], s['joint_pos'], new, plan[:2],
                           float(yaw_of(plan[3:7])), env.step_dt, self.scale)
        self.at = env.common_step_counter
        return self


def install(root=TRAINING, errors=None):
    """Patch motion.py's actor terms; returns (motion, task).  errors: a dict that collects, per
    term, the estimate's largest difference from the simulator's value at each step."""
    motion, tk = modules(root)
    if getattr(motion, '_human_walk_installed', False):
        return motion, tk
    from mjlab.utils.lab_api.math import matrix_from_quat
    true_anchor, true_preview, true_sensor = tk.obs_ref_anchor, tk.obs_ref_preview, motion.vel_mdp.builtin_sensor

    def robot(env):
        if not hasattr(env, '_human_walk'):
            env._human_walk = Robot(env, motion)
            env._human_walk.gyro = lambda e: true_sensor(e, 'robot/imu_ang_vel')
        return env._human_walk.sync()

    def note(key, mine, truth):
        if errors is not None:
            errors.setdefault(key, []).append((mine - truth).abs().max(-1).values.detach())

    def ref_anchor(env, skill):
        t = motion.task(env)
        rb = robot(env)
        p, R = rb.estimate.torso(rb.sensors['joint_pos'])
        ref = t.raw(0, 'body')[:, t.anchor_j]
        position = (R.transpose(1, 2) @ (ref[:, :3] - p)[..., None]).squeeze(-1)
        relative = R.transpose(1, 2) @ matrix_from_quat(ref[:, 3:])
        out = torch.cat([position, relative[..., :2].reshape(-1, 6)], -1)
        truth = true_anchor(env, skill)
        note('ref_anchor_position_m', out[:, :3], truth[:, :3])
        note('ref_anchor_rotation', out[:, 3:], truth[:, 3:])
        return out

    def ref_preview(env, skill):
        t = motion.task(env)
        rb = robot(env)
        p, R = rb.estimate.torso(rb.sensors['joint_pos'])
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        parts = []
        for ahead in (10, 25):
            ref = t.raw(0, 'body', ahead)[:, t.anchor_j]
            e = tk.wrap(tk.yaw_of(ref[:, 3:]) - yaw)
            parts += [tk._yaw_frame(ref[:, :3] - p, yaw), torch.sin(e)[:, None], torch.cos(e)[:, None]]
        out = torch.cat(parts, -1)
        note('ref_preview', out, true_preview(env, skill))
        return out

    def sensor(env, sensor_name):
        if sensor_name != 'robot/imu_lin_vel':
            return true_sensor(env, sensor_name)
        rb = robot(env)
        out = rb.estimate.imu_velocity(rb.sensors['gyro'])
        note('imu_lin_vel_m_s', out, true_sensor(env, sensor_name))
        return out

    tk.obs_ref_anchor, tk.obs_ref_preview, motion.vel_mdp.builtin_sensor = ref_anchor, ref_preview, sensor
    motion._human_walk_installed = True
    return motion, tk


def arm_weight(motion):
    """Training's arms are weightless to the pelvis: MuJoCo's body gravity compensation also
    pushes the floating base up.  After this only the arm motors compensate, as on the robot
    (the box policies' actuator_only mode)."""
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'unitree_box/three_reference'))
    from five_actuation import configure_actuation
    if not getattr(motion, '_human_walk_arm_weight', False):
        native = motion.env_cfg
        motion.env_cfg = lambda *a, **k: configure_actuation(native(*a, **k), 'actuator_only')
        motion._human_walk_arm_weight = True
