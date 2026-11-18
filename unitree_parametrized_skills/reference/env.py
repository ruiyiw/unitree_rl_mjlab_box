"""env.py — the mjlab env of walk(N): Unitree's 23-DoF G1 walks N metres and stops.

Each episode draws a plan from a bank (plans.py: the humanoid repository's MPC and IK for
one distance N).  The policy outputs a CORRECTION to the plan's joint targets,

    joint target = plan target one frame ahead + scale * action,

so a zero action plays the plan back on the PD joints (the warm start "from an MPC").

What the actor reads is what a real G1 has (ACTOR_TERMS):

  gyro, gravity         the pelvis IMU: angular velocity, gravity direction
  joint_pos, joint_vel  the 23 encoders (with their bias), from the default pose
  actions               the previous correction
  ref_joints, ref_phase the plan's next joint targets, its stride phase, how far the plan
                        still has to go, whether it has ended: controller memory
  goal                  where the plan is and where the goal is along and across the route,
                        relative to where the robot believes it is: LEG ODOMETRY
                        (odometry.py) from the encoders and the IMU's heading.  No
                        simulator position or velocity.

The critic also reads the truth (base velocity, true errors to the plan and the goal, the
tracked bodies' errors, foot contacts); rewards, terminations and the success test use the
truth, as they only exist in training and evaluation.

Rewards: tracking the plan (BeyondMimic's terms, as in the humanoid repository), staying on
the route, standing still at the goal, a success bonus, action-rate and joint-limit costs,
and a cost on the odometry's drift rate, so that the gait keeps its own estimate right.

Domain randomization for hardware (dr_events): floor friction, link masses and inertias, a
payload and COM shift on the torso, PD gains, torque limits, joint friction and damping,
command delay, encoder bias, sensor noise and bias, observation delay, IMU heading drift
and tilt bias in the odometry, and pushes.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import CameraSensorCfg, ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import (quat_apply, quat_apply_inverse, quat_error_magnitude, quat_inv,
                                      quat_mul)
from mjlab.utils.lab_api.math import yaw_quat as yaw_of_quat
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.spec_config import CameraCfg
from mjlab.viewer import ViewerConfig

from . import paths, plans
from .odometry import KalmanOdometry, LegKinematics, LegOdometry, _quat_to_mat

paths.use_humanoid()
import scene as sc  # noqa: E402   humanoid: the robot's collisions, arm gains, action scale

ANCHOR = "torso_link"

# the head camera: the G1's RealSense D435i, at the pose of `d435_joint` in Unitree's URDF
# (unitree_ros, g1_description/g1_23dof_rev_1_0.urdf; torso_link is the same frame there and in
# the MJCF).  The policies do not read it.
CAMERA = "d435"
CAMERA_BODY = "torso_link"
CAMERA_POS = (0.0576235, 0.01753, 0.42987)       # m, in torso_link
CAMERA_PITCH = 0.8307767239493009                # rad about +y: the lens 47.6 degrees below the horizon
CAMERA_FOVY = 42.0                               # degrees: the D435's colour stream, 69 x 42 at 16:9
CAMERA_SIZE = (640, 360)                         # px, a 16:9 image (the colour stream's shape)


def camera_quat() -> tuple[float, float, float, float]:
    """The camera's orientation in torso_link (w, x, y, z).  The URDF's d435_link has +x along
    the lens and +z up; a MuJoCo camera looks along its -z with +y up."""
    h = 0.5 * CAMERA_PITCH
    c, s = 0.5 * math.cos(h), 0.5 * math.sin(h)  # (cos h, 0, sin h, 0) * (0.5, 0.5, -0.5, -0.5)
    return (c + s, c - s, s - c, -c - s)


def camera_cfg() -> CameraCfg:
    return CameraCfg(name=CAMERA, body=CAMERA_BODY, fovy=CAMERA_FOVY, pos=CAMERA_POS, quat=camera_quat())
EE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_roll_rubber_hand",
             "right_wrist_roll_rubber_hand")
ACTOR_TERMS = ("gyro", "gravity", "joint_pos", "joint_vel", "actions", "ref_joints", "ref_phase", "goal")
SENSOR_TERMS = {"gyro": 3, "gravity": 3, "joint_pos": 23, "joint_vel": 23, "actions": 23}   # may carry history
CONTROL_DIM = 23 + 5 + 6           # ref_joints, ref_phase, goal: the current values only
ACTOR_DIM = sum(SENSOR_TERMS.values()) + CONTROL_DIM       # 109, one frame


def actor_dim(history: int = 1) -> int:
    """The actor's input size with `history` frames of the sensor terms (oldest first per term)."""
    return history * sum(SENSOR_TERMS.values()) + CONTROL_DIM


def newest_frames(obs: torch.Tensor, history: int, keep: int = 1) -> torch.Tensor:
    """From an actor observation with `history` sensor frames, the one with the newest `keep`."""
    out, i = [], 0
    for d in SENSOR_TERMS.values():
        out.append(obs[:, i + d * (history - keep): i + d * history])
        i += d * history
    return torch.cat(out + [obs[:, i:]], -1)


# starts
BEGIN, RSI = 0, 1
P_BEGIN = (0.3, 0.5)               # share of first-frame starts at curriculum level 0 and 1
RSI_NOISE = (0.02, 0.05, 0.03)     # random-frame starts: root xy (m), yaw (rad), leg joints (rad)

# the success tests (the truth)
GOAL_TOL = 0.10                    # m, pelvis from the goal, horizontally
HEADING_TOL = math.radians(10.0)
SPEED_TOL = 0.10                   # m/s
UPRIGHT_TOL = 0.20                 # rad, pelvis tilt
ARRIVE_FOR = 0.5                   # s: the strict test (README) held this long = stable arrival
TRAIN_AT = (0.10, 0.2, 0.10)       # training's test: m, rad heading, rad torso tilt ...
TRAIN_FOR = 1.0                    # ... held this long (no speed test: it must survive exploration noise)
HOLD_S = 4.0                       # s a training episode goes on after the plan's end: standing at the goal
                                   # is part of the skill (an episode does NOT end at success)

# tracking terminations (BeyondMimic's, and drift), as in the humanoid repository
TERM_ANCHOR_Z, TERM_ANCHOR_ORI, TERM_EE_Z, TERM_DRIFT = 0.25, 0.8, 0.25, 0.5

# curriculum: pushes and first-frame starts grow with the level
PUSH_V = (0.1, 0.4)                # m/s per horizontal axis at level 0 and 1
LEVEL_UP, LEVEL_DOWN, LEVEL_STEP, LEVEL_EVERY = 0.7, 0.3, 0.05, 200

# sensors the odometry reads
YAW_DRIFT = math.radians(0.1)      # rad/s, the IMU heading's drift (per episode, uniform ±)
YAW_BIAS = math.radians(1.0)       # rad, its offset at the start
YAW_NOISE = math.radians(0.05)     # rad, per step (a fused IMU heading is smooth)
TILT_BIAS = 0.02                   # rad, roll and pitch of the IMU's attitude estimate
ODO_START_NOISE = 0.02             # m, the estimate at a random-frame start
ACC_NOISE = 0.3                    # m/s^2, the accelerometer per 20 ms sample (Kalman estimator only)
ACC_BIAS = 0.1                     # m/s^2 per axis, per episode
# Randomization profiles (env_cfg(profile=...)).  "v1" is what the first policies were trained
# with.  "robust" widens what the first hardware trials showed to be too narrow: that robot's
# ankle encoders and IMU are 1-2 deg off, its accelerometer 0.14 m/s^2, and v1's pushes stayed
# near 0.1 m/s because they grew with a success rate that stayed low (README, "Robust profile").
# push: m/s per horizontal axis at curriculum level 0 and 1.
PROFILES = {
    "v1": {"encoder_bias": 0.01, "ankle_encoder_bias": 0.01, "gravity_bias": 0.025, "tilt_bias": TILT_BIAS,
           "acc_noise": ACC_NOISE, "acc_bias": ACC_BIAS, "push": PUSH_V},
    "robust": {"encoder_bias": 0.015, "ankle_encoder_bias": 0.04, "gravity_bias": 0.045, "tilt_bias": 0.045,
               "acc_noise": 1.0, "acc_bias": 0.25, "push": (0.3, 0.5)},
}
PROFILES["robust_p"] = {**PROFILES["robust"], "push": (0.15, 0.5)}     # robust with a gentler push floor
GRAVITY = 9.81
ESTIMATORS = {"legs": LegOdometry, "kalman": KalmanOdometry}

# (weight, std)
TRACK = dict(anchor_pos=(0.5, 0.15), anchor_ori=(0.5, 0.3), body_pos=(1.0, 0.1), body_ori=(1.0, 0.3),
             body_lin_vel=(1.0, 1.0), body_ang_vel=(1.0, 3.14))
ROUTE = dict(along=(1.0, 0.06), heading=(0.5, 0.15), goal=(3.0, 0.05), still=(2.0, 0.10))
CROSS = (-1.0, 0.02, 0.08)         # weight, free sway (m), scale (m)
SUCCESS_BONUS = 50.0
REG = dict(action_rate=-0.2, joint_limit=-10.0, odometry_drift=-30.0, adapter=-0.5)


# ------------------------------------------------------------------ small maths

def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_quat(yaw):
    z = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)], -1)


def rot2(v, yaw):
    """Rotate (..., 2) vectors by yaw (broadcast over the leading dims)."""
    while yaw.dim() < v.dim() - 1:
        yaw = yaw[..., None]
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def rot_z(yaw):
    c, s, z, o = torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(-1, 3, 3)


def rot_xy(roll, pitch):
    """Small attitude error: R_y(pitch) R_x(roll)."""
    cr, sr, cp, sp = torch.cos(roll), torch.sin(roll), torch.cos(pitch), torch.sin(pitch)
    z, o = torch.zeros_like(roll), torch.ones_like(roll)
    Rx = torch.stack([o, z, z, z, cr, -sr, z, sr, cr], -1).reshape(-1, 3, 3)
    Ry = torch.stack([cp, z, sp, z, o, z, -sp, z, cp], -1).reshape(-1, 3, 3)
    return Ry @ Rx


def to_world(v, F, key):
    """A plan quantity `key` (plan frame) in the world, for plan frames F (x, y, yaw)."""
    yaw = F[:, 2]
    if key == "qpos":
        p = torch.cat([F[:, :2] + rot2(v[:, :2], yaw), v[:, 2:3]], -1)
        return torch.cat([p, quat_mul(yaw_quat(yaw), v[:, 3:7]), v[:, 7:]], -1)
    if key == "qvel":
        return torch.cat([rot2(v[:, :2], yaw), v[:, 2:3], rot2(v[:, 3:5], yaw), v[:, 5:]], -1)
    if key == "body":
        nb = v.shape[1]
        p = torch.cat([F[:, None, :2] + rot2(v[..., :2], yaw), v[..., 2:3]], -1)
        q = quat_mul(yaw_quat(yaw)[:, None].expand(-1, nb, -1).reshape(-1, 4), v[..., 3:].reshape(-1, 4))
        return torch.cat([p, q.reshape(-1, nb, 4)], -1)
    if key in ("body_lin", "body_ang"):
        return torch.cat([rot2(v[..., :2], yaw), v[..., 2:3]], -1)
    return v


def _u(n, lo, hi, dev):
    return lo + (hi - lo) * torch.rand(n, device=dev)


# ------------------------------------------------------------------ the bank

class Bank:
    """A plan bank (plans.py) on the device, joints in the simulated robot's order."""

    def __init__(self, path, device, joint_names):
        b = torch.load(path, map_location="cpu", weights_only=False)
        perm = torch.tensor([b["joint_names"].index(n) for n in joint_names])
        d = {k: b[k] for k in plans.KEYS}
        d["qpos"] = torch.cat([d["qpos"][..., :7], d["qpos"][..., 7:][..., perm]], -1)
        d["qvel"] = torch.cat([d["qvel"][..., :6], d["qvel"][..., 6:][..., perm]], -1)
        d["qcmd"] = d["qcmd"][..., perm]
        self.d = {k: v.to(device).float() for k, v in d.items()}
        self.len = b["len"].to(device)
        self.n_m = b["n_m"].to(device)
        self.n = len(self.n_m)
        self.body_names = b["body_names"]
        rows = torch.arange(self.n, device=device)
        self.start_xy = self.d["qpos"][:, 0, :2].clone()
        self.end_xy = self.d["qpos"][rows, self.len - 1, :2].clone()
        self.fps = b["fps"]
        self.extra(b, device)

    def extra(self, b, device):
        """For a task built on walk: whatever else its bank holds (carry_box: the boxes)."""


# ------------------------------------------------------------------ the task command

class WalkCommand(CommandTerm):
    """Lays out each episode from a plan; keeps its clock, the odometry, the success timers
    and the curriculum level; serves the reference."""

    cfg: "WalkCommandCfg"
    BANK = Bank                    # the bank's class (a task built on walk has its own)

    def __init__(self, cfg: "WalkCommandCfg", env):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.robot: Entity = env.scene["robot"]
        jn = list(self.robot.joint_names)
        self.bank = self.BANK(cfg.bank, dev, jn)
        bn = list(self.robot.body_names)
        self.body_ids = [bn.index(b) for b in self.bank.body_names]
        self.anchor_j = self.bank.body_names.index(ANCHOR)
        self.ee_j = [self.bank.body_names.index(b) for b in EE_BODIES]
        self.legs = torch.tensor([("hip" in j or "knee" in j or "ankle" in j) for j in jn], device=dev)
        z = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731
        self.plan_ids = torch.zeros(n, dtype=torch.long, device=dev)
        self.frame = z(n, 3)                          # the plan's frame in the world (x, y, yaw)
        self.clock = torch.zeros(n, dtype=torch.long, device=dev)
        self.fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self.start = torch.zeros(n, dtype=torch.long, device=dev)
        self.elapsed, self.after_end = z(n), z(n)
        self.stood, self.held = z(n), z(n)
        self.success = torch.zeros(n, dtype=torch.bool, device=dev)       # training's test, latched
        self.success_now = torch.zeros(n, dtype=torch.bool, device=dev)   # ... the step it was first met
        self.arrived = torch.zeros(n, dtype=torch.bool, device=dev)       # the strict test, latched
        self.t_arrived = torch.full((n,), float("nan"), device=dev)
        self.overshoot = z(n)                         # max forward distance past the goal
        self.odo_err = z(n)                           # |estimate − truth|, now
        self.odo_drift, self._odo_err_xy = z(n), z(n, 2)   # its rate of change (m/s)
        # the odometry: kinematics from the compiled model, the IMU's heading error
        self.kin = LegKinematics(env.sim.mj_model, jn, dev, prefix="robot/")
        self.odo = ESTIMATORS[cfg.estimator](self.kin, n, dev, sole=cfg.sole_contacts)
        self.odo.scale = cfg.odometry_scale
        # how often a training episode draws each plan: uniform, plans shorter than 0.5 m short_weight times
        self._plan_p = torch.where(self.bank.end_xy[:, 0] < 0.5, cfg.short_weight, 1.0).to(dev)
        self.yaw_drift, self.yaw_bias, self.tilt_bias = z(n), z(n), z(n, 2)
        self.acc_bias, self._vel_w = z(n, 3), z(n, 3)
        self.yaw_est = z(n)
        self.pos_true, self.yaw_true = z(n, 2), z(n)  # the pelvis in the plan frame
        for k in ("success", "arrived", "level", "odometry_error", "goal_error"):
            self.metrics[k] = z(n)
        self.level = 0.0 if cfg.curriculum else 1.0
        self._ema, self._last_update = 0.0, 0

    @property
    def command(self):
        return self.clock[:, None].float()

    # -------------------------------------------------- the reference
    def lengths(self, ids=None):
        return self.bank.len[self.plan_ids if ids is None else self.plan_ids[ids]]

    def ref(self, key, off=0, ids=None):
        """The plan's quantity at its clock + off (held at the last frame), plan frame."""
        clk = self.clock if ids is None else self.clock[ids]
        idx = self.plan_ids if ids is None else self.plan_ids[ids]
        frame = torch.minimum(torch.clamp(clk + off, min=0), self.bank.len[idx] - 1)
        return self.bank.d[key][idx, frame]

    def ref_w(self, key, off=0, ids=None):
        return to_world(self.ref(key, off, ids), self.frame if ids is None else self.frame[ids], key)

    def ended(self):
        return self.clock >= self.lengths() - 1

    def goal_xy(self):
        """The goal (the plan's last pelvis position), plan frame."""
        return self.bank.end_xy[self.plan_ids]

    # -------------------------------------------------- curriculum
    def _curriculum(self, env_ids):
        if not self.cfg.curriculum or len(env_ids) == 0:
            return
        rate = self.kept()[env_ids].float().mean().item()
        self._ema = 0.98 * self._ema + 0.02 * rate
        step = self._env.common_step_counter
        if step - self._last_update >= LEVEL_EVERY:
            self._last_update = step
            if self._ema > LEVEL_UP:
                self.level = min(1.0, self.level + LEVEL_STEP)
            elif self._ema < LEVEL_DOWN:
                self.level = max(0.0, self.level - LEVEL_STEP)

    # -------------------------------------------------- reset
    def _resample_command(self, env_ids):
        self._curriculum(env_ids)
        cfg = self.cfg
        n, dev = len(env_ids), self.device
        if cfg.fixed_plans:                             # evaluation: env i always plays plan i mod C
            self.plan_ids[env_ids] = env_ids % self.bank.n
        else:
            self.plan_ids[env_ids] = torch.multinomial(self._plan_p, n, replacement=True)
        p_b = cfg.begin_fraction if cfg.begin_fraction is not None else (
            P_BEGIN[0] + (P_BEGIN[1] - P_BEGIN[0]) * self.level)
        mode = torch.where(torch.rand(n, device=dev) < p_b, BEGIN, RSI)
        self.start[env_ids] = mode
        yaw = _u(n, -math.pi, math.pi, dev) if cfg.random_heading else torch.zeros(n, device=dev)
        self.frame[env_ids] = torch.cat([self._env.scene.env_origins[env_ids, :2], yaw[:, None]], -1)
        L = self.lengths(env_ids)
        self.clock[env_ids] = torch.where(mode == RSI, (torch.rand(n, device=dev) * (L - 1)).long(),
                                          torch.zeros(n, dtype=torch.long, device=dev))
        self.fresh[env_ids] = True
        q = self.ref_w("qpos", 0, env_ids)
        qd = self.ref_w("qvel", 0, env_ids)
        root = torch.cat([q[:, :7], qd[:, :6]], -1)
        jp, jv = q[:, 7:].clone(), qd[:, 6:].clone()
        if cfg.rsi_noise:
            rsi = (mode == RSI).float()
            root[:, :2] += rsi[:, None] * (2 * torch.rand(n, 2, device=dev) - 1) * RSI_NOISE[0]
            dy = rsi * _u(n, -RSI_NOISE[1], RSI_NOISE[1], dev)
            root[:, 3:7] = quat_mul(yaw_quat(dy), root[:, 3:7])
            jp += rsi[:, None] * self.legs[None].float() * (2 * torch.rand_like(jp) - 1) * RSI_NOISE[2]
        self.robot.write_root_state_to_sim(root, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
        am = self._env.action_manager
        for buf in (am._action, am._prev_action, am._prev_prev_action):
            buf[env_ids] = 0.0
        s = 1.0 if cfg.sensor_errors else 0.0
        h = s if cfg.heading_errors else 0.0
        self.yaw_drift[env_ids] = h * _u(n, -YAW_DRIFT, YAW_DRIFT, dev)
        self.yaw_bias[env_ids] = h * _u(n, -YAW_BIAS, YAW_BIAS, dev)
        self.tilt_bias[env_ids] = s * (2 * torch.rand(n, 2, device=dev) - 1) * cfg.tilt_bias
        self.acc_bias[env_ids] = s * (2 * torch.rand(n, 3, device=dev) - 1) * cfg.acc_bias
        for b in (self.elapsed, self.after_end, self.stood, self.held, self.overshoot):
            b[env_ids] = 0.0
        self.success[env_ids] = False
        self.success_now[env_ids] = False
        self.arrived[env_ids] = False
        self.t_arrived[env_ids] = float("nan")

    # -------------------------------------------------- per step
    def _sense(self):
        """The truth (pelvis in the plan frame) and the IMU's orientation as the robot knows it."""
        r = self.robot.data
        q = r.root_link_quat_w
        yaw_w = yaw_of(q)
        self.pos_true = rot2(r.root_link_pos_w[:, :2] - self.frame[:, :2], -self.frame[:, 2])
        self.yaw_true = wrap(yaw_w - self.frame[:, 2])
        noise = YAW_NOISE * torch.randn_like(yaw_w) if self.cfg.sensor_errors and self.cfg.heading_errors else 0.0
        self.yaw_est = self.yaw_true + self.yaw_bias + self.yaw_drift * self.elapsed + noise
        tilt = _quat_to_mat(quat_mul(yaw_quat(-yaw_w), q))             # roll and pitch only
        return rot_z(self.yaw_est) @ rot_xy(self.tilt_bias[:, 0], self.tilt_bias[:, 1]) @ tilt

    def _accel(self, R_imu, fresh):
        """The pelvis's horizontal acceleration as the IMU gives it, in the start-heading frame:
        the specific force over the last control step (the true change of velocity, plus
        gravity, in the body frame, with the accelerometer's bias and noise), turned by the
        IMU's attitude estimate.  The accelerometer is taken to be at the pelvis origin."""
        r = self.robot.data
        v = r.root_link_lin_vel_w
        a = torch.where(fresh[:, None], torch.zeros_like(v), (v - self._vel_w) / self._env.step_dt)
        self._vel_w = v.clone()
        a = a + torch.tensor([0.0, 0.0, GRAVITY], device=v.device)
        f = quat_apply_inverse(r.root_link_quat_w, a)
        if self.cfg.sensor_errors:
            f = f + self.acc_bias + self.cfg.acc_noise * (2 * torch.rand_like(f) - 1)
        return (R_imu @ f[..., None]).squeeze(-1)[:, :2]

    def _update_command(self, env_ids):
        env = self._env
        dt = env.step_dt
        ids = slice(None) if env_ids is None else env_ids
        fresh = self.fresh.clone()
        if env_ids is None:
            # the clock stops at the plan's last frame; it does not tick on the step of a reset
            self.clock = torch.minimum(self.clock + (~fresh).long(), self.lengths() - 1)
            self.elapsed = self.elapsed + (~fresh).float() * dt
        R_imu = self._sense()
        jp = self.robot.data.joint_pos_biased
        new = fresh.nonzero().squeeze(-1) if env_ids is None else env_ids[fresh[env_ids]]
        if len(new):
            pos0 = self.pos_true[new].clone()
            if self.cfg.sensor_errors:
                pos0 += (self.start[new] == RSI).float()[:, None] * torch.randn_like(pos0) * ODO_START_NOISE
            self.odo.reset(new, jp[new], R_imu[new], pos0, self.ref("qvel", 0, new)[:, :2])
        if env_ids is None:
            self.odo.update(jp, R_imu, self._accel(R_imu, fresh), dt)
        self.fresh[ids] = False
        err = self.odo.pos - self.pos_true
        if env_ids is None:                              # how fast the estimate left the truth this step
            self.odo_drift = torch.where(fresh, torch.zeros_like(self.odo_drift),
                                         (err - self._odo_err_xy).norm(dim=-1) / dt)
        self._odo_err_xy = err
        self.odo_err = err.norm(dim=-1)
        if env_ids is not None:
            return
        # the success tests, on the truth
        ok, ok_train, d, _ = self.at_goal()
        ended = self.ended()
        self.stood = torch.where(ok_train & ended, self.stood + dt, torch.zeros_like(self.stood))
        self.success_now = (self.stood >= TRAIN_FOR) & ~self.success
        self.success |= self.success_now
        self.held = torch.where(ok & ended, self.held + dt, torch.zeros_like(self.held))
        hit = (self.held >= ARRIVE_FOR) & ~self.arrived
        self.t_arrived = torch.where(hit, self.elapsed, self.t_arrived)
        self.arrived |= hit
        self.after_end = torch.where(ended, self.after_end + dt, torch.zeros_like(self.after_end))
        self.overshoot = torch.maximum(self.overshoot, self.pos_true[:, 0] - self.goal_xy()[:, 0])
        m = self.metrics
        m["success"] = self.kept().float()
        m["arrived"] = torch.maximum(m["arrived"], self.arrived.float())
        m["odometry_error"] = torch.maximum(m["odometry_error"], self.odo_err)
        m["goal_error"] = d

    def kept(self):
        """Training's success, and still at the goal now (what the curriculum and the logs count)."""
        return self.success & (self.stood > 0)

    def at_goal(self):
        """The strict test (README: within 10 cm and 10 degrees, slower than 0.1 m/s, upright,
        both feet on the floor), training's looser one, the distance and the heading error."""
        r = self.robot.data
        d = (self.goal_xy() - self.pos_true).norm(dim=-1)
        herr = self.yaw_true.abs()
        speed = r.root_link_lin_vel_w[:, :2].norm(dim=-1)
        tilt = torch.acos((-r.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
        feet = (self._env.scene["feet_ground_contact"].data.found > 0).all(dim=-1)
        strict = (d < GOAL_TOL) & (herr < HEADING_TOL) & (speed < SPEED_TOL) & (tilt < UPRIGHT_TOL) & feet
        train = (d < TRAIN_AT[0]) & (herr < TRAIN_AT[1]) & (tilt < TRAIN_AT[2])
        return strict, train, d, herr

    def _update_metrics(self):
        self.metrics["level"][:] = self.level


@dataclass(kw_only=True)
class WalkCommandCfg(CommandTermCfg):
    bank: str
    curriculum: bool = True          # the level rises with training success (else fixed at 1)
    rsi_noise: bool = True           # offsets on random-frame starts
    begin_fraction: float | None = None   # share of first-frame starts (None: from the level)
    fixed_plans: bool = False        # env i plays plan i mod C (evaluation)
    random_heading: bool = True      # the plan frame's heading in the world
    sensor_errors: bool = True       # IMU heading drift and bias, tilt bias, odometry start noise
    heading_errors: bool = True      # ... of which the heading's (off: an exact heading, the rest kept)
    estimator: str = "legs"          # "legs" (LegOdometry) or "kalman" (KalmanOdometry)
    sole_contacts: bool = False      # the estimator's contact points: the sole's ends (else the foot site)
    short_weight: float = 1.0        # training draws plans shorter than 0.5 m this many times as often
    odometry_scale: float = 1.0      # the estimator's calibrated scale (calibrate.py)
    tilt_bias: float = TILT_BIAS     # rad, the IMU attitude's roll and pitch bias in the estimator (PROFILES)
    acc_bias: float = ACC_BIAS       # m/s^2, the accelerometer's bias per axis
    acc_noise: float = ACC_NOISE     # m/s^2, the accelerometer's noise per sample
    resampling_time_range: tuple[float, float] = (1e9, 1e9)

    def build(self, env) -> WalkCommand:
        return WalkCommand(self, env)


def cmd(env) -> WalkCommand:
    return env.command_manager.get_term("task")


# ------------------------------------------------------------------ the action

class PlanJointAction(JointPositionAction):
    """Joint targets = the plan's targets one frame ahead + scale * action."""

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        ref = cmd(self._env).ref("qcmd", 1)
        self._processed_actions = self._raw_actions * self._scale + ref[:, self._target_ids]

    @property
    def applied(self):
        """The correction on the joints (what a policy sees as its previous action)."""
        return self._raw_actions


@dataclass(kw_only=True)
class PlanJointActionCfg(JointPositionActionCfg):
    def build(self, env) -> PlanJointAction:
        return PlanJointAction(self, env)


class AdapterJointAction(PlanJointAction):
    """Approach 3: a frozen walking actor plus a bounded learned correction.

    The action is the adapter's output; the joints get plan + scale * (base(obs) + clamp(action)).
    The frozen actor reads the same actor observation the adapter does, and sees the summed
    correction as its previous action, as it did when it was trained alone."""

    cfg: "AdapterJointActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.base = None
        self._total = torch.zeros_like(self._raw_actions)

    def base_obs(self):
        """What the frozen actor reads: the input it was trained on."""
        obs = self._env.obs_buf["actor"]
        if self.cfg.base_history != self.cfg.history:
            obs = newest_frames(obs, self.cfg.history, self.cfg.base_history)
        return obs

    def base_action(self):
        """The frozen actor's correction on the current observation."""
        from .models import frozen_actor
        obs = self.base_obs()
        if self.base is None:
            self.base = frozen_actor(self.cfg.base, obs.shape[-1], self.action_dim, self.device)
        return self.base(obs)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._total = self.base_action() + actions.clamp(-self.cfg.bound, self.cfg.bound)
        ref = cmd(self._env).ref("qcmd", 1)
        self._processed_actions = self._total * self._scale + ref[:, self._target_ids]

    @property
    def applied(self):
        return self._total

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._total[slice(None) if env_ids is None else env_ids] = 0.0


@dataclass(kw_only=True)
class AdapterJointActionCfg(JointPositionActionCfg):
    base: str = ""                   # the frozen actor's checkpoint
    bound: float = 0.5               # |correction| per joint, in action units
    history: int = 1                 # sensor frames in the actor observation
    base_history: int = 1            # ... of which the frozen actor reads the newest

    def build(self, env) -> AdapterJointAction:
        return AdapterJointAction(self, env)


# ------------------------------------------------------------------ observations

def obs_last_action(env):
    return env.action_manager.get_term("joint_pos").applied


def obs_ref_joints(env):
    """The plan's next joint targets, from the default pose (controller memory)."""
    return cmd(env).ref("qcmd", 1) - env.scene["robot"].data.default_joint_pos


def obs_ref_phase(env):
    """The plan's stride phase (sin, cos, walking), how far it still has to go (m, capped at
    1) and whether it has ended (controller memory)."""
    c = cmd(env)
    togo = (c.goal_xy()[:, 0] - c.ref("qpos")[:, 0]).clamp(0.0, 1.0)
    return torch.cat([c.ref("phase"), togo[:, None], c.ended().float()[:, None]], -1)


def _goal_terms(c: WalkCommand, pos, yaw):
    """For a pelvis at (pos, yaw) in the plan frame: where the plan is from it (along the route
    and across), how far the goal still is along the route (capped at 1 m: every far goal
    looks the same) and across it, sin and cos of the heading off the route's.

    All in the ROUTE's frame, not the robot's heading frame: seen from the robot, a goal 20 m
    ahead moves 35 cm sideways per degree of heading, and policies that read that turned
    toward it and left the plan (the first version; README "What went wrong")."""
    e = (c.ref("qpos")[:, :2] - pos).clamp(-0.5, 0.5)
    g = c.goal_xy() - pos
    g = torch.stack([g[:, 0].clamp(-0.5, 1.0), g[:, 1].clamp(-0.5, 0.5)], -1)
    return torch.cat([e, g, torch.sin(yaw)[:, None], torch.cos(yaw)[:, None]], -1)


def obs_goal(env):
    """From the leg odometry and the IMU's heading: what the robot can know."""
    c = cmd(env)
    return _goal_terms(c, c.odo.pos, c.yaw_est)


def obs_goal_true(env):
    """Critic only: the same from the simulator's pose."""
    c = cmd(env)
    return _goal_terms(c, c.pos_true, c.yaw_true)


def obs_ref_joint_vel(env):
    return cmd(env).ref("qvel")[:, 6:]


def obs_body_err(env):
    """Critic only: each tracked body's offset from its plan pose (relative to the torso)."""
    return (_rel_ref_bodies(env, 0)[..., :3] - _robot_bodies(env)[0]).reshape(env.num_envs, -1)


def obs_odometry_err(env):
    """Critic only: the odometry's error (plan frame)."""
    c = cmd(env)
    return torch.cat([c.odo.pos - c.pos_true, (c.yaw_est - c.yaw_true)[:, None]], -1)


# ------------------------------------------------------------------ rewards: tracking (BeyondMimic's)

def _anchor(env):
    r = env.scene["robot"]
    c = cmd(env)
    i = c.body_ids[c.anchor_j]
    return r.data.body_link_pos_w[:, i], r.data.body_link_quat_w[:, i]


def _robot_bodies(env):
    d = env.scene["robot"].data
    i = cmd(env).body_ids
    return d.body_link_pos_w[:, i], d.body_link_quat_w[:, i], d.body_link_lin_vel_w[:, i], d.body_link_ang_vel_w[:, i]


def _rel_ref_bodies(env, off=1):
    """The plan's bodies moved so that its torso sits on the robot's torso (its height kept)
    and turned by the heading difference."""
    c = cmd(env)
    ref = c.ref_w("body", off)
    pa, qa = _anchor(env)
    ra = ref[:, c.anchor_j]
    dpos = pa.clone()
    dpos[:, 2] = ra[:, 2]
    dq = yaw_of_quat(quat_mul(qa, quat_inv(ra[:, 3:])))
    nb = ref.shape[1]
    dqe = dq[:, None].expand(-1, nb, -1).reshape(-1, 4)
    pos = dpos[:, None] + quat_apply(dqe, (ref[..., :3] - ra[:, None, :3]).reshape(-1, 3)).reshape(-1, nb, 3)
    quat = quat_mul(dqe, ref[..., 3:].reshape(-1, 4)).reshape(-1, nb, 4)
    return torch.cat([pos, quat], -1)


def r_anchor_pos(env, std: float):
    c = cmd(env)
    p, _ = _anchor(env)
    e = (c.ref_w("body", 1)[:, c.anchor_j, :3] - p).square().sum(-1)
    return torch.exp(-e / std ** 2)


def r_anchor_ori(env, std: float):
    c = cmd(env)
    _, q = _anchor(env)
    e = quat_error_magnitude(c.ref_w("body", 1)[:, c.anchor_j, 3:], q).square()
    return torch.exp(-e / std ** 2)


def r_body_pos(env, std: float):
    e = (_rel_ref_bodies(env)[..., :3] - _robot_bodies(env)[0]).square().sum(-1).mean(-1)
    return torch.exp(-e / std ** 2)


def r_body_ori(env, std: float):
    ref = _rel_ref_bodies(env)[..., 3:]
    q = _robot_bodies(env)[1]
    e = quat_error_magnitude(ref.reshape(-1, 4), q.reshape(-1, 4)).reshape(q.shape[:2]).square().mean(-1)
    return torch.exp(-e / std ** 2)


def r_body_lin_vel(env, std: float):
    e = (cmd(env).ref_w("body_lin", 1) - _robot_bodies(env)[2]).square().sum(-1).mean(-1)
    return torch.exp(-e / std ** 2)


def r_body_ang_vel(env, std: float):
    e = (cmd(env).ref_w("body_ang", 1) - _robot_bodies(env)[3]).square().sum(-1).mean(-1)
    return torch.exp(-e / std ** 2)


# ------------------------------------------------------------------ rewards: the route and the goal

def r_along(env, std: float):
    """The pelvis where the plan has it along the route."""
    c = cmd(env)
    e = c.ref("qpos", 1)[:, 0] - c.pos_true[:, 0]
    return torch.exp(-e.square() / std ** 2)


def r_cross(env, free: float, scale: float):
    """Sideways off the route, beyond the gait's sway (a cost, capped)."""
    c = cmd(env)
    e = ((c.pos_true[:, 1] - c.ref("qpos", 1)[:, 1]).abs() - free).clamp(min=0.0) / scale
    return e.square().clamp(max=4.0)


def r_heading(env, std: float):
    return torch.exp(-cmd(env).yaw_true.square() / std ** 2)


def r_goal(env, std: float):
    """After the plan's end: the pelvis on the goal."""
    c = cmd(env)
    d = (c.goal_xy() - c.pos_true).norm(dim=-1)
    return c.ended().float() * torch.exp(-d.square() / std ** 2)


def r_still(env, std: float):
    """After the plan's end: near the goal, not moving, both feet on the floor."""
    c = cmd(env)
    r = env.scene["robot"].data
    d = (c.goal_xy() - c.pos_true).norm(dim=-1)
    speed = r.root_link_lin_vel_w[:, :2].norm(dim=-1)
    feet = (env.scene["feet_ground_contact"].data.found > 0).all(dim=-1).float()
    return c.ended().float() * feet * torch.exp(-d.square() / std ** 2 - speed.square() / SPEED_TOL ** 2)


def r_odometry_drift(env):
    """How fast the leg odometry is leaving the truth (m/s): the weighted feet moving on the
    floor.  A cost: the policy learns to plant its feet, which is also what hardware wants."""
    return cmd(env).odo_drift.clamp(max=1.0)


def r_success(env):
    """Training's success, once an episode: BONUS / dt on the step the test is first met.  The
    episode goes on: the robot has to keep standing there (r_goal, r_still)."""
    return cmd(env).success_now.float() / env.step_dt


def r_adapter(env):
    """Approach 3: the size of the learned correction."""
    return env.action_manager.get_term("joint_pos").raw_action.square().mean(-1)


# ------------------------------------------------------------------ terminations and events

def t_tracking(env, drift: bool = True):
    """The torso's height or tilt, or a foot or hand's height, far from the plan's; in training
    (drift) also the torso TERM_DRIFT from the plan's spot.  Evaluation leaves that last test
    out: a robot that walks well but ends off the goal is measured, not cut short."""
    c = cmd(env)
    p, q = _anchor(env)
    ra = c.ref_w("body", 1)[:, c.anchor_j]
    bad = (ra[:, 2] - p[:, 2]).abs() > TERM_ANCHOR_Z
    gw = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(len(p), 3)
    bad |= (quat_apply_inverse(ra[:, 3:], gw)[:, 2] - quat_apply_inverse(q, gw)[:, 2]).abs() > TERM_ANCHOR_ORI
    rel = _rel_ref_bodies(env)[:, c.ee_j, 2]
    bad |= ((rel - _robot_bodies(env)[0][:, c.ee_j, 2]).abs() > TERM_EE_Z).any(-1)
    if drift:
        bad |= (ra[:, :2] - p[:, :2]).norm(dim=-1) > TERM_DRIFT
    return bad


def t_plan_time_out(env, hold: float):
    """`hold` s after the plan's end (a time-out: the value is bootstrapped)."""
    return cmd(env).after_end >= hold


def ev_push(env, env_ids, lo: float = PUSH_V[0], hi: float = PUSH_V[1]):
    """A velocity kick to the pelvis, growing with the curriculum level."""
    if env_ids is None or len(env_ids) == 0:
        return
    c = cmd(env)
    v = lo + (hi - lo) * c.level
    if v <= 0:
        return
    r: Entity = env.scene["robot"]
    vel = r.data.root_link_vel_w[env_ids].clone()
    vel[:, :2] += (2 * torch.rand(len(env_ids), 2, device=env.device) - 1) * v
    r.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


FOOT_GEOMS = tuple(f"{s}_foot{i}_collision" for s in ("left", "right") for i in range(1, 8))
ARM_JOINTS = (".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*")
LEG_WAIST_JOINTS = (".*_hip_.*", ".*_knee_.*", ".*_ankle_.*", "waist_.*")

# what is randomized, for the record (README and checkpoints): profile v1; dr_spec() for the others
DR_SPEC = {
    "foot_friction": [0.4, 1.4], "link_mass_inertia_scale": [0.9, 1.1],
    "torso_extra_mass_kg": [-0.5, 2.5], "torso_com_m": [0.03, 0.015, 0.02],
    "kp_scale": [0.85, 1.15], "kd_scale": [0.85, 1.15], "torque_limit_scale": [0.9, 1.0],
    "joint_friction_nm": {"legs_waist": [0.0, 0.6], "arms": [0.0, 0.3]}, "joint_damping_nms": [0.0, 0.1],
    "command_delay_ms": [0, 15], "encoder_bias_rad": 0.01,
    "sensor_noise": {"gyro_rad_s": 0.2, "gravity": 0.05, "joint_pos_rad": 0.01, "joint_vel_rad_s": 1.5},
    "sensor_bias": {"gyro_rad_s": 0.02, "gravity": 0.025},
    "observation_delay_ms": [0, 20],
    "odometry": {"heading_drift_deg_s": 0.1, "heading_bias_deg": 1.0, "heading_noise_deg": 0.05,
                 "tilt_bias_rad": TILT_BIAS, "start_noise_m": ODO_START_NOISE},
    "push_m_s": {"every_s": [1.0, 3.0], "level0": PUSH_V[0], "level1": PUSH_V[1]},
    "accelerometer_m_s2": {"noise": ACC_NOISE, "bias": ACC_BIAS, "used_by": "the Kalman estimator only"},
}


def dr_spec(profile: str = "v1") -> dict:
    """DR_SPEC with a profile's ranges."""
    p, d = PROFILES[profile], copy.deepcopy(DR_SPEC)
    d.update({"profile": profile, "encoder_bias_rad": p["encoder_bias"], "ankle_encoder_bias_rad": p["ankle_encoder_bias"]})
    d["sensor_bias"]["gravity"] = p["gravity_bias"]
    d["odometry"]["tilt_bias_rad"] = p["tilt_bias"]
    d["push_m_s"].update({"level0": p["push"][0], "level1": p["push"][1]})
    d["accelerometer_m_s2"].update({"noise": p["acc_noise"], "bias": p["acc_bias"]})
    return d


def light_events(push: bool) -> dict:
    """The humanoid repository's original recipe: floor friction, encoder bias, pushes from zero."""
    E, S = EventTermCfg, SceneEntityCfg
    ev = {
        "foot_friction": E(mode="startup", func=dr.geom_friction,
                           params={"asset_cfg": S("robot", geom_names=FOOT_GEOMS), "operation": "abs",
                                   "ranges": (0.4, 1.2), "shared_random": True}),
        "encoder_bias": E(mode="startup", func=dr.encoder_bias,
                          params={"asset_cfg": S("robot"), "bias_range": (-0.01, 0.01)}),
    }
    if push:
        ev["push"] = E(mode="interval", func=ev_push, interval_range_s=(1.0, 3.0), params={"lo": 0.0, "hi": 0.4})
    return ev


def dr_events(push: bool, profile: str = "v1") -> dict:
    """Physical randomization: masses once per env at startup (they need the model's
    constants recomputed), the rest every episode."""
    E, S = EventTermCfg, SceneEntityCfg
    p = PROFILES[profile]
    ev = {
        "link_inertia": E(mode="startup", func=dr.pseudo_inertia,
                          params={"asset_cfg": S("robot"), "alpha_range": (-0.05, 0.05)}),
        "torso_mass": E(mode="startup", func=dr.body_mass,
                        params={"asset_cfg": S("robot", body_names=("torso_link",)), "ranges": (-0.5, 2.5),
                                "operation": "add"}),
        "torso_com": E(mode="startup", func=dr.body_com_offset,
                       params={"asset_cfg": S("robot", body_names=("torso_link",)),
                               "ranges": {0: (-0.03, 0.03), 1: (-0.015, 0.015), 2: (-0.02, 0.02)}}),
        "foot_friction": E(mode="reset", func=dr.geom_friction,
                           params={"asset_cfg": S("robot", geom_names=FOOT_GEOMS), "operation": "abs",
                                   "ranges": (0.4, 1.4), "shared_random": True}),
        "pd_gains": E(mode="reset", func=dr.pd_gains,
                      params={"asset_cfg": S("robot"), "kp_range": (0.85, 1.15), "kd_range": (0.85, 1.15)}),
        "torque_limits": E(mode="reset", func=dr.effort_limits,
                           params={"asset_cfg": S("robot"), "effort_limit_range": (0.9, 1.0)}),
        "leg_friction": E(mode="reset", func=dr.dof_frictionloss,
                          params={"asset_cfg": S("robot", joint_names=LEG_WAIST_JOINTS), "ranges": (0.0, 0.6)}),
        "arm_friction": E(mode="reset", func=dr.dof_frictionloss,
                          params={"asset_cfg": S("robot", joint_names=ARM_JOINTS), "ranges": (0.0, 0.3)}),
        "joint_damping": E(mode="reset", func=dr.dof_damping,
                           params={"asset_cfg": S("robot"), "ranges": (0.0, 0.1)}),
        "encoder_bias": E(mode="reset", func=dr.encoder_bias,
                          params={"asset_cfg": S("robot"), "bias_range": (-p["encoder_bias"], p["encoder_bias"])}),
    }
    if p["ankle_encoder_bias"] != p["encoder_bias"]:      # after the event above: the ankles' wider range
        ev["ankle_encoder_bias"] = E(mode="reset", func=dr.encoder_bias,
                                     params={"asset_cfg": S("robot", joint_names=(".*_ankle_.*",)),
                                             "bias_range": (-p["ankle_encoder_bias"], p["ankle_encoder_bias"])})
    if push:
        ev["push"] = E(mode="interval", func=ev_push, interval_range_s=(1.0, 3.0),
                       params={"lo": p["push"][0], "hi": p["push"][1]})
    return ev


# ------------------------------------------------------------------ config

def _sensor_noise(noisy: bool, light: bool = False):
    """Per-step noise and a per-episode bias on a sensor, and up to one control step of delay
    (light: the noise alone)."""
    def make(noise, bias=0.0):
        if not noisy:
            return {}
        n = Unoise(n_min=-noise, n_max=noise)
        if light:
            return {"noise": n}
        if bias > 0:
            n = NoiseModelWithAdditiveBiasCfg(noise_cfg=n, bias_noise_cfg=Unoise(n_min=-bias, n_max=bias))
        return {"noise": n, "delay_min_lag": 0, "delay_max_lag": 1, "delay_hold_prob": 0.95}
    return make


def actor_obs(noisy: bool, light: bool = False, history: int = 1, gravity_bias: float = 0.025) -> dict:
    """What the real robot has (module docstring).  The order is ACTOR_TERMS.  With history > 1
    each sensor term (SENSOR_TERMS) is its last `history` readings, oldest first; after a reset
    the first reading fills them."""
    O, n = ObservationTermCfg, _sensor_noise(noisy, light)
    h = {"history_length": history} if history > 1 else {}
    return {
        "gyro": O(func=vel_mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}, **n(0.2, 0.02), **h),
        "gravity": O(func=vel_mdp.projected_gravity, **n(0.05, gravity_bias), **h),
        "joint_pos": O(func=vel_mdp.joint_pos_rel, params={"biased": True}, **n(0.01), **h),
        "joint_vel": O(func=vel_mdp.joint_vel_rel, **n(1.5), **h),
        "actions": O(func=obs_last_action, **h),
        "ref_joints": O(func=obs_ref_joints),
        "ref_phase": O(func=obs_ref_phase),
        "goal": O(func=obs_goal),
    }


def critic_obs() -> dict:
    O = ObservationTermCfg
    return {
        **actor_obs(False),
        "base_lin_vel": O(func=vel_mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}),
        "goal_true": O(func=obs_goal_true),
        "odometry_err": O(func=obs_odometry_err),
        "ref_joint_vel": O(func=obs_ref_joint_vel),
        "body_err": O(func=obs_body_err),
        "foot_contact": O(func=vel_mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}),
    }


def rewards(adapter: bool = False) -> dict:
    R = RewardTermCfg
    fn = {"anchor_pos": r_anchor_pos, "anchor_ori": r_anchor_ori, "body_pos": r_body_pos,
          "body_ori": r_body_ori, "body_lin_vel": r_body_lin_vel, "body_ang_vel": r_body_ang_vel}
    out = {k: R(func=fn[k], weight=w, params={"std": s}) for k, (w, s) in TRACK.items()}
    rt = {"along": r_along, "heading": r_heading, "goal": r_goal, "still": r_still}
    out.update({k: R(func=rt[k], weight=w, params={"std": s}) for k, (w, s) in ROUTE.items()})
    out.update({
        "cross": R(func=r_cross, weight=CROSS[0], params={"free": CROSS[1], "scale": CROSS[2]}),
        "success": R(func=r_success, weight=SUCCESS_BONUS),
        "action_rate": R(func=envs_mdp.action_rate_l2, weight=REG["action_rate"]),
        "joint_limit": R(func=envs_mdp.joint_pos_limits, weight=REG["joint_limit"]),
        "odometry_drift": R(func=r_odometry_drift, weight=REG["odometry_drift"]),
    })
    if adapter:
        out["adapter"] = R(func=r_adapter, weight=REG["adapter"])
    return out


def env_cfg(bank: str, num_envs: int = 4096, mode: str = "train", adapter_base: str | None = None,
            adapter_bound: float = 0.5, episode_s: float | None = None, randomize: str = "full",
            privileged_actor: bool = False, teacher_group: bool = False, history: int = 1,
            estimator: str = "legs", sole_contacts: bool = False, base_history: int = 1,
            short_weight: float = 1.0, odometry_scale: float = 1.0,
            camera: tuple[int, int] | None = None, profile: str = "v1") -> ManagerBasedRlEnvCfg:
    """The env on a plan bank.

    mode  "train"    randomized physics and sensors, pushes, random-frame starts, the curriculum;
                     an episode ends HOLD_S after the plan's end (a time-out), not at success
          "nominal"  evaluation: the nominal robot, clean sensors, every episode from the start,
                     env i on plan i mod C; no drift termination
          "dr"       evaluation as "nominal" but with training's randomized physics and sensors
                     (no pushes)
          "dr_h"     as "dr" with an exact IMU heading (no heading offset, drift or noise): what
                     is left when the heading is not the limit
    history: sensor frames the actor reads (1 or more; the critic and a teacher always read one).
    estimator, sole_contacts: the position estimator behind the `goal` terms (odometry.py).
    base_history: approach 3, the frames the frozen actor reads.
    short_weight: training draws plans shorter than 0.5 m this many times as often as the others
          (with the 0-20 m bank the long plans otherwise take most of the steps).
    odometry_scale: the estimator's distance scale, from a calibration walk (calibrate.py).
    adapter_base: approach 3, the frozen actor's checkpoint; the action is then the correction.
    randomize: "full" (dr_events, sensor bias and delay, command delay, odometry errors) or
          "light" (light_events and sensor noise only), in "train" and "dr".
    privileged_actor: the actor reads the critic's observation (a teacher; never deployed).
    teacher_group: also serve that observation as the group "teacher" (guided.py).
    profile: the randomization ranges (PROFILES), in "train" and "dr" with randomize "full".
    camera: (width, height) to render the head camera in every env each step, as the sensor
          "head_camera" (env.scene["head_camera"].data.rgb and .depth).  The camera itself
          (CAMERA) is always in the robot's model; without this nothing is rendered."""
    assert mode in ("train", "nominal", "dr", "dr_h") and randomize in ("full", "light")
    P = PROFILES[profile]
    train, noisy = mode == "train", mode != "nominal"
    light = randomize == "light"
    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="netforce", num_slots=1)
    robot = copy.deepcopy(sc.robot_cfg())       # (its actuator configs are shared module objects)
    robot.cameras = (*robot.cameras, camera_cfg())
    sensors = [feet_ground]
    if camera:
        sensors.append(CameraSensorCfg(name="head_camera", camera_name=f"robot/{CAMERA}", width=camera[0],
                                       height=camera[1], data_types=("rgb", "depth")))
    for a in robot.articulation.actuators:      # command latency: 0-3 physics steps of 5 ms
        a.delay_min_lag, a.delay_max_lag = 0, 3 if noisy and not light else 0
        a.delay_hold_prob, a.delay_update_period = 0.9, 4
    if adapter_base:
        action = AdapterJointActionCfg(entity_name="robot", actuator_names=(".*",), scale=sc.action_scale(),
                                       base=adapter_base, bound=adapter_bound, history=history,
                                       base_history=base_history)
    else:
        action = PlanJointActionCfg(entity_name="robot", actuator_names=(".*",), scale=sc.action_scale())
    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "tracking": TerminationTermCfg(func=t_tracking, params={"drift": train}),
        "fell": TerminationTermCfg(func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}),
    }
    if train:
        terminations["plan_time_out"] = TerminationTermCfg(func=t_plan_time_out, params={"hold": HOLD_S},
                                                           time_out=True)
    if episode_s is None:
        b = torch.load(bank, map_location="cpu", weights_only=False)
        episode_s = float(math.ceil((int(b["len"].max()) - 1) / b["fps"] + HOLD_S + 1.0))
    observations = {"actor": ObservationGroupCfg(critic_obs() if privileged_actor else actor_obs(noisy, light, history, P["gravity_bias"]),
                                                 enable_corruption=noisy and not privileged_actor),
                    "critic": ObservationGroupCfg(critic_obs(), enable_corruption=False)}
    if teacher_group:
        observations["teacher"] = ObservationGroupCfg(critic_obs(), enable_corruption=False)
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), entities={"robot": robot},
                       sensors=tuple(sensors), num_envs=num_envs, env_spacing=8.0),
        observations=observations,
        actions={"joint_pos": action},
        commands={"task": WalkCommandCfg(bank=bank, curriculum=train, rsi_noise=train,
                                         begin_fraction=None if train else 1.0, fixed_plans=not train,
                                         sensor_errors=noisy and not light, heading_errors=mode != "dr_h",
                                         estimator=estimator, sole_contacts=sole_contacts,
                                         short_weight=short_weight, odometry_scale=odometry_scale,
                                         tilt_bias=P["tilt_bias"], acc_bias=P["acc_bias"], acc_noise=P["acc_noise"])},
        events=(light_events(push=train) if light else dr_events(push=train, profile=profile)) if noisy else {},
        rewards=rewards(adapter=bool(adapter_base)),
        terminations=terminations,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot",
                            body_name="torso_link", distance=4.0, elevation=-20.0, azimuth=120.0),
        sim=SimulationCfg(nconmax=None, njmax=400, contact_sensor_maxmatch=64,
                          mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=50)),
        decimation=4,
        episode_length_s=episode_s,
    )
