#!/usr/bin/env python3
"""motion.py — a box-free policy for one planned motion: the env, training, measurement, GIF.

Runs in the `mjlab` env, from a motion's folder (policy-go-to/, policy-carry-without-box/),
whose plans.py has written banks/plan.pt: one plan from the MPC (plan.py), the times and
captions of its phases, when the arms hold the holding pose, and the start and goal of
its walk (the pelvis's planned poses).

The recipe is the box-carrying skills' (task.py), with no box: not in the scene and
not in the observation.  The policy follows the plan and outputs a CORRECTION to its
joint targets: target = plan target + scale · action, so a zero action plays the plan
back on the PD joints.  Episodes start at the plan's first frame or at a random frame
of it (reference state initialisation, with small offsets in training).  The reward is
the skills' tracking reward (BeyondMimic's terms, tighter, and the hands), the success
bonus and the regularisers.  Training randomises the feet's friction and the encoders'
bias, adds observation noise, and pushes the robot, harder as the curriculum level ℓ
rises.  An episode succeeds once the plan has ended and the robot has stood at the
plan's end pose (within AT) and upright for STAND_FOR s.

  python ../motion.py train [--iterations 1000]   # runs/policy/model_<it>.pt
  python ../motion.py select --from 500           # every saved checkpoint; the best -> actors/policy.pt
  python ../motion.py play [--zero]               # 1,024 runs from the start, 60 s each -> eval/policy.json
  python ../motion.py render                      # g1_<motion>.gif
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mjlab.entity import Entity  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg  # noqa: E402
from mjlab.envs import mdp as envs_mdp  # noqa: E402
from mjlab.envs.mdp import dr  # noqa: E402
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg  # noqa: E402
from mjlab.managers.event_manager import EventTermCfg  # noqa: E402
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg  # noqa: E402
from mjlab.managers.reward_manager import RewardTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.managers.termination_manager import TerminationTermCfg  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from mjlab.scene import SceneCfg  # noqa: E402
from mjlab.sensor import ContactMatch, ContactSensorCfg  # noqa: E402
from mjlab.sim import MujocoCfg, SimulationCfg  # noqa: E402
from mjlab.tasks.velocity import mdp as vel_mdp  # noqa: E402
from mjlab.terrains import TerrainEntityCfg  # noqa: E402
from mjlab.utils.lab_api.math import quat_apply, quat_mul  # noqa: E402
from mjlab.utils.noise import UniformNoiseCfg as Unoise  # noqa: E402
from mjlab.viewer import ViewerConfig  # noqa: E402

import scene as sc  # noqa: E402
import task as tk  # noqa: E402   the box-carrying task: its reference terms, rewards and pushes

STAND_FOR = 1.0                    # s stood at the end: success
AT = (0.10, 0.2)                   # m, rad: the pelvis from the plan's end pose
UPRIGHT = 0.10                     # rad: the torso's tilt from vertical
PLAY_S = 60.0                      # s per measured run
FOOT = 0.3048                      # m


def slug(folder):
    """policy-carry-without-box -> carry_without_box."""
    return os.path.basename(os.path.abspath(folder)).removeprefix("policy-").replace("-", "_")


# ------------------------------------------------------------------ the plan

class Plan:
    """A motion's plan (its plans.py) on the device, as (1, T, ...) arrays in the plan's
    frame, joints in the simulated robot's order; and its phases."""

    def __init__(self, path, device, joint_names):
        b = torch.load(path, map_location="cpu", weights_only=False)
        perm = torch.tensor([b["joint_names"].index(n) for n in joint_names], device=device)
        d = {k: v.to(device) for k, v in b["plan"].items()}
        d["qpos"] = torch.cat([d["qpos"][..., :7], d["qpos"][..., 7:][..., perm]], -1)
        d["qvel"] = torch.cat([d["qvel"][..., :6], d["qvel"][..., 6:][..., perm]], -1)
        d["qcmd"] = d["qcmd"][..., perm]
        self.sk = d
        self.body_names = b["body_names"]
        self.seconds = (int(d["len"].max()) - 1) / b["fps"]
        for k in ("phases", "hold", "walk", "title", "stats"):
            setattr(self, k, b[k])


def episode_s(path):
    """The plan's length and 2.5 s to stand at its end, in whole seconds."""
    b = torch.load(path, map_location="cpu", weights_only=False)
    return float(math.ceil((int(b["plan"]["len"].max()) - 1) / b["fps"] + 2.5))


# ------------------------------------------------------------------ the task command

class MotionCommand(CommandTerm):
    """Lays out each episode from the plan, keeps its clock, the standing timer and the
    curriculum level, and serves the reference.  It has the box-carrying task's
    interface (TaskCommand) that that task's observations, rewards and terminations
    use; their skill index `k` is ignored, as there is one plan."""

    cfg: "MotionCommandCfg"

    def __init__(self, cfg: "MotionCommandCfg", env):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.robot: Entity = env.scene["robot"]
        self.plan = Plan(cfg.plan, dev, list(self.robot.joint_names))
        self.plan_ids = torch.zeros(n, dtype=torch.long, device=dev)
        bn = list(self.robot.body_names)
        self.body_ids = [bn.index(b) for b in self.plan.body_names]
        self.anchor_j = self.plan.body_names.index(tk.ANCHOR)
        self.ee_j = [self.plan.body_names.index(b) for b in tk.EE_BODIES]
        self.hand_j = [self.plan.body_names.index(b) for b in tk.EE_BODIES[2:]]
        self.legs = torch.tensor([("hip" in j or "knee" in j or "ankle" in j) for j in self.robot.joint_names],
                                 device=dev)
        z = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731
        self.frame = z(n, 3)                     # the plan's frame in the world
        self.clock = torch.zeros(n, 1, dtype=torch.long, device=dev)
        self.fresh = torch.zeros(n, dtype=torch.bool, device=dev)
        self.start = torch.zeros(n, dtype=torch.long, device=dev)
        self.stood = z(n)
        self.success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.metrics["success"] = z(n)
        self.metrics["level"] = z(n)
        self.level = 0.0                # curriculum level ℓ
        self.curriculum = True          # play freezes it (at ℓ = 1)
        self._ema, self._last_update = 0.0, 0
        self.force_start = None         # play: start every episode in this mode

    @property
    def command(self):
        return self.clock.float()

    # -------------------------------------------------- the reference
    def lengths(self, k=0, ids=None):
        idx = self.plan_ids if ids is None else self.plan_ids[ids]
        return self.plan.sk["len"][idx]

    def raw(self, k, key, off=0, ids=None):
        """The plan's quantity at its clock + off (held at the last frame), plan frame."""
        clk = self.clock[:, 0] if ids is None else self.clock[ids, 0]
        idx = self.plan_ids if ids is None else self.plan_ids[ids]
        frame = torch.minimum(torch.clamp(clk + off, min=0), self.lengths(ids=ids) - 1)
        return self.plan.sk[key][idx, frame]

    def world(self, k, key, off=0, ids=None):
        """The plan's quantity in the world."""
        return tk.to_world(self.raw(k, key, off, ids), self.frame if ids is None else self.frame[ids], key)

    def R(self, key, off=0):
        """The plan's quantity in the world, every env (cached per step)."""
        return tk._cached(self._env, f"ref:{key}:{off}", lambda: self.world(0, key, off))

    def ended(self, k=0):
        return self.clock[:, 0] >= self.lengths() - 1

    def pose(self, frame):
        """The plan's root pose at a frame, in the world (N, 7)."""
        frame = self.lengths() + frame if frame < 0 else torch.minimum(
            torch.full_like(self.plan_ids, frame), self.lengths() - 1)
        q = self.plan.sk["qpos"][self.plan_ids, frame]
        return tk.to_world(q, self.frame, "qpos")[:, :7]

    # the box-carrying task's curriculum, on this command's success
    _curriculum = tk.TaskCommand._curriculum

    # -------------------------------------------------- reset
    def _resample_command(self, env_ids):
        self._curriculum(env_ids)
        tk._invalidate(self._env)
        n, dev = len(env_ids), self.device
        if self.force_start is not None:
            mode = torch.full((n,), self.force_start, dtype=torch.long, device=dev)
        else:
            p_b = tk._lerp(*tk.P_BEGIN, self.level if self.curriculum else 1.0)
            if self.cfg.begin_fraction is not None:
                p_b = self.cfg.begin_fraction
            mode = torch.where(torch.rand(n, device=dev) < p_b, tk.BEGIN, tk.RSI)
        self.start[env_ids] = mode
        # the plan's frame: the env origin, x along the world's
        self.frame[env_ids] = torch.cat([self._env.scene.env_origins[env_ids, :2],
                                         torch.zeros(n, 1, device=dev)], -1)
        # the clock: the plan's first frame (BEGIN) or a random one (RSI); the robot there
        L = self.lengths(ids=env_ids)
        self.clock[env_ids, 0] = torch.where(mode == tk.RSI, (torch.rand(n, device=dev) * (L - 1)).long(),
                                             torch.zeros(n, dtype=torch.long, device=dev))
        self.fresh[env_ids] = True
        q = self.world(0, "qpos", 0, env_ids)
        qd = self.world(0, "qvel", 0, env_ids)
        root = torch.cat([q[:, :7], qd[:, :6]], -1)
        jp, jv = q[:, 7:].clone(), qd[:, 6:].clone()
        if self.cfg.rsi_noise:
            rsi = (mode == tk.RSI).float()
            root[:, :2] += rsi[:, None] * (2 * torch.rand(n, 2, device=dev) - 1) * tk.RSI_NOISE[0]
            dy = rsi * tk._u(n, -tk.RSI_NOISE[1], tk.RSI_NOISE[1], dev)
            root[:, 3:7] = quat_mul(tk.yaw_quat(dy), root[:, 3:7])
            jp += rsi[:, None] * self.legs[None].float() * (2 * torch.rand_like(jp) - 1) * tk.RSI_NOISE[2]
        self.robot.write_root_state_to_sim(root, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
        am = self._env.action_manager
        for buf in (am._action, am._prev_action, am._prev_prev_action):
            buf[env_ids] = 0.0
        self.stood[env_ids] = 0.0
        self.success[env_ids] = False

    # -------------------------------------------------- per step
    def standing(self):
        """At the plan's end pose (within AT) with the torso upright; and the numbers: the
        pelvis's distance and heading error, its horizontal speed, the torso's tilt."""
        r = self.robot.data
        end = self.pose(-1)
        d = (end[:, :2] - r.root_link_pos_w[:, :2]).norm(dim=-1)
        herr = tk.wrap(tk.yaw_of(end[:, 3:7]) - tk.yaw_of(r.root_link_quat_w))
        speed = r.root_link_lin_vel_w[:, :2].norm(dim=-1)
        _, qa = tk._anchor(self._env)
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(len(qa), 3)
        tilt = torch.acos(quat_apply(qa, up)[:, 2].clamp(-1.0, 1.0))
        ok = (d < AT[0]) & (herr.abs() < AT[1]) & (tilt < UPRIGHT)
        return ok, (d, herr, speed, tilt)

    def _update_command(self, env_ids):
        env = self._env
        if env_ids is not None:
            return
        # the clock stops at the plan's last frame: the progress observed then stays at 1, as
        # in training, however long the robot stands (unbounded, it drew robots off the pose)
        self.clock[:, 0] = torch.minimum(self.clock[:, 0] + (~self.fresh).long(), self.lengths() - 1)
        self.fresh[:] = False
        tk._invalidate(env)
        ok, _ = self.standing()
        ok &= self.ended()
        self.stood = torch.where(ok, self.stood + env.step_dt, torch.zeros_like(self.stood))
        self.success = self.stood >= STAND_FOR
        self.metrics["success"] = torch.maximum(self.metrics["success"], self.success.float())

    def _update_metrics(self):
        self.metrics["level"][:] = self.level


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
    plan: str
    rsi_noise: bool = True           # offsets to the random-frame starts (training)
    begin_fraction: float | None = None  # optional fixed share of full-motion starts
    resampling_time_range: tuple[float, float] = (1e9, 1e9)

    def build(self, env) -> MotionCommand:
        return MotionCommand(self, env)


def task(env) -> MotionCommand:
    return env.command_manager.get_term("task")


# ------------------------------------------------------------------ observations, rewards, terminations

def observations(noise) -> dict:
    """The policy's observation: the plan ahead and its own body (145 numbers)."""
    O = ObservationTermCfg
    return {
        "ref_joints": O(func=tk.obs_ref_joints, params={"skill": 0}),
        "ref_anchor": O(func=tk.obs_ref_anchor, params={"skill": 0}, noise=noise(0.05)),
        "ref_preview": O(func=tk.obs_ref_preview, params={"skill": 0}, noise=noise(0.05)),
        "ref_clock": O(func=tk.obs_ref_clock, params={"skill": 0}),
        "base_lin_vel": O(func=vel_mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}, noise=noise(0.3)),
        "base_ang_vel": O(func=vel_mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}, noise=noise(0.2)),
        "projected_gravity": O(func=vel_mdp.projected_gravity, noise=noise(0.05)),
        "joint_pos": O(func=vel_mdp.joint_pos_rel, noise=noise(0.01)),
        "joint_vel": O(func=vel_mdp.joint_vel_rel, noise=noise(1.0)),
        "actions": O(func=tk.obs_last_joint_action),
    }


def t_success(env):
    return task(env).success


def rewards() -> dict:
    R = RewardTermCfg
    fn = {"anchor_pos": tk.r_anchor_pos, "anchor_ori": tk.r_anchor_ori, "body_pos": tk.r_body_pos,
          "body_ori": tk.r_body_ori, "body_lin_vel": tk.r_body_lin_vel, "body_ang_vel": tk.r_body_ang_vel,
          "hands": tk.r_hands}
    out = {k: R(func=fn[k], weight=w, params={"std": s}) for k, (w, s) in tk.TRACK.items()}
    out.update({
        "success": R(func=tk.r_success, weight=tk.SUCCESS_BONUS),
        "action_rate": R(func=envs_mdp.action_rate_l2, weight=tk.TASK["action_rate"]),
        "joint_limit": R(func=envs_mdp.joint_pos_limits, weight=tk.TASK["joint_limit"]),
    })
    return out


# ------------------------------------------------------------------ config

def plan_path(folder):
    return os.path.join(folder, "banks", "plan.pt")


def env_cfg(folder: str, num_envs: int = 4096, play: bool = False) -> ManagerBasedRlEnvCfg:
    """A motion's env: training's randomisation and pushes, or none (play)."""
    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="netforce", num_slots=1)

    noise = (lambda lo: Unoise(n_min=-lo, n_max=lo)) if not play else (lambda lo: None)
    clean = lambda lo: None  # noqa: E731
    critic = {**observations(clean),
              "body_err": ObservationTermCfg(func=tk.obs_body_err),
              "foot_contact": ObservationTermCfg(func=vel_mdp.foot_contact,
                                                 params={"sensor_name": "feet_ground_contact"})}
    events = {}
    if not play:
        foot_geoms = tuple(f"{s}_foot{i}_collision" for s in ("left", "right") for i in range(1, 8))
        events = {
            "foot_friction": EventTermCfg(mode="startup", func=dr.geom_friction,
                                          params={"asset_cfg": SceneEntityCfg("robot", geom_names=foot_geoms),
                                                  "operation": "abs", "ranges": (0.4, 1.2), "shared_random": True}),
            "encoder_bias": EventTermCfg(mode="startup", func=dr.encoder_bias,
                                         params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)}),
            "push": EventTermCfg(mode="interval", func=tk.ev_push, interval_range_s=(1.0, 3.0),
                                 params={"vmax": tk.PUSH_V}),
        }
    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "success": TerminationTermCfg(func=t_success),
        "tracking": TerminationTermCfg(func=tk.t_tracking),
        "fell": TerminationTermCfg(func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}),
    }
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), entities={"robot": sc.robot_cfg()},
                       sensors=(feet_ground,), num_envs=num_envs, env_spacing=8.0),
        observations={"actor": ObservationGroupCfg(observations(noise), enable_corruption=not play),
                      "critic": ObservationGroupCfg(critic, enable_corruption=False)},
        actions={"joint_pos": tk.PlanJointActionCfg(entity_name="robot", actuator_names=(".*",),
                                                    scale=sc.action_scale())},
        commands={"task": MotionCommandCfg(plan=plan_path(folder), rsi_noise=not play)},
        events=events,
        rewards=rewards(),
        terminations=terminations,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot",
                            body_name="torso_link", distance=4.0, elevation=-20.0, azimuth=120.0),
        sim=SimulationCfg(nconmax=None, njmax=400, contact_sensor_maxmatch=64,
                          mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=50)),
        decimation=4,
        episode_length_s=episode_s(plan_path(folder)),
    )


def ppo_cfg(folder: str, iterations: int = 1000) -> RslRlOnPolicyRunnerCfg:
    """The skills' PPO settings: the correction's last layer starts at zero (the plan played back)."""
    cfg = tk.ppo_cfg("fetch", iterations)
    cfg.experiment_name = f"g1_{slug(folder)}"
    return cfg


# ------------------------------------------------------------------ training

def train(folder, iterations=1000, envs=4096, seed=1):
    """PPO; checkpoints every 100 iterations and at the end, to runs/policy/."""
    import time
    torch.manual_seed(seed)
    cfg = env_cfg(folder, num_envs=envs)
    cfg.seed = seed
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    vec = RslRlVecEnvWrapper(env, clip_actions=None)
    agent = ppo_cfg(folder, iterations)
    agent.seed = seed
    log_dir = os.path.join(folder, "runs", "policy")
    runner = MjlabOnPolicyRunner(vec, asdict(agent), log_dir, "cuda:0")
    t0 = time.time()
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
    runner.save(os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt"))
    print(f"trained {iterations} iterations in {time.time() - t0:.0f} s; "
          f"curriculum level {task(env).level:.2f}; log {log_dir}")


# ------------------------------------------------------------------ measurement

def actor_path(folder):
    return os.path.join(folder, "actors", "policy.pt")


class Player:
    """The env in play mode, without the success ending, and a runner to load the policy into."""

    def __init__(self, folder, num_envs=64, seconds=PLAY_S, seed=7):
        cfg = env_cfg(folder, num_envs=num_envs, play=True)
        del cfg.terminations["success"]
        cfg.episode_length_s = seconds
        cfg.seed = seed
        self.folder = folder
        self.env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
        self.task = t = task(self.env)
        t.curriculum, t.level = False, 1.0
        t.force_start = tk.BEGIN
        self.vec = RslRlVecEnvWrapper(self.env, clip_actions=None)
        self.runner = MjlabOnPolicyRunner(self.vec, asdict(ppo_cfg(folder)), None, "cuda:0")
        self.seconds = seconds

    def policy(self, path=None, zero=False):
        if zero:
            env = self.env
            return lambda obs: torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        self.runner.load(path, load_cfg={"actor": True}, map_location="cuda:0")
        return self.runner.get_inference_policy(device="cuda:0")

    def evaluate(self, path=None, zero=False, seed=7):
        """Each env plays the motion once from its first frame, with mean actions, for
        `seconds`.  Success: the plan ended and the robot stood 1 s at its end pose; still
        standing: at the end pose and upright when the seconds are up."""
        torch.manual_seed(seed)
        env, t = self.env, self.task
        n, dt, dev = env.num_envs, env.step_dt, env.device
        policy = self.policy(path, zero)
        obs, _ = self.vec.reset()
        r = env.scene["robot"].data
        yaw0 = tk.yaw_of(r.root_link_quat_w).clone()
        start = r.root_link_pos_w[:, :2].clone()
        hold = [round(x / dt) for x in t.plan.hold]
        live = torch.ones(n, dtype=torch.bool, device=dev)
        t_ok = torch.full((n,), float("nan"), device=dev)
        end = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in ("fell", "tracking")}
        last = start.clone()
        hands = torch.zeros(n, device=dev)
        tm = env.termination_manager
        with torch.no_grad():
            for k in range(int(round(self.seconds / dt)) - 1):      # the last step would reset
                obs, _, dones, _ = self.vec.step(policy(obs))
                hit = live & t.success & torch.isnan(t_ok)
                t_ok[hit] = (k + 1) * dt
                going = live & ~dones.bool()
                clk = t.clock[:, 0]
                last = torch.where(going[:, None], r.root_link_pos_w[:, :2], last)
                if hold[0] <= int(clk.max()) and int(clk.min()) <= hold[1]:
                    e = (tk._rel_ref_bodies(env, 0)[:, t.hand_j, :3] - tk._robot_bodies(env)[0][:, t.hand_j]).norm(dim=-1)
                    during = going & (clk >= hold[0]) & (clk <= hold[1])
                    hands = torch.where(during, torch.maximum(hands, e.max(-1).values), hands)
                d = dones.bool() & live
                if d.any():
                    for key in end:
                        end[key] |= d & tm.get_term(key).bool()
                    live &= ~d
                if not live.any():
                    break
        c = lambda x: x.cpu().numpy()  # noqa: E731
        ok = ~np.isnan(c(t_ok))
        standing = c(live & t.standing()[0])
        turned = c(tk.wrap(tk.yaw_of(r.root_link_quat_w) - yaw0).abs())
        pct = lambda x: [round(float(np.percentile(x, q)), 3) for q in (0, 50, 100)]  # noqa: E731
        w = t.plan.walk
        return {"policy": "plan played back (zero correction)" if zero else os.path.relpath(path, self.folder),
                "episodes": int(n), "success": round(float(ok.mean()), 4),
                **{k: round(float(c(v).mean()), 4) for k, v in end.items()},
                "time_to_success_s": pct(c(t_ok)[ok]) if ok.any() else None,
                f"standing_at_{self.seconds:g}_s": round(float(standing.mean()), 4),
                "plan_walk_m": round(float(np.hypot(w["to"][0] - w["from"][0], w["to"][1] - w["from"][1])), 4),
                "walked_m_min_median_max": pct(c((last - start).norm(dim=-1))),
                "turned_rad_min_median_max": pct(turned),
                "hands_off_holding_pose_m_min_median_max": pct(c(hands))}


def export(folder, ckpt):
    """actors/policy.pt: the checkpoint's actor (weights and observation normaliser)."""
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    os.makedirs(os.path.dirname(actor_path(folder)), exist_ok=True)
    torch.save({"actor_state_dict": d["actor_state_dict"], "iter": d["iter"], "infos": None,
                "source": os.path.relpath(ckpt, folder)}, actor_path(folder))
    return actor_path(folder)


def select(folder, first=500, envs=64):
    """Every saved checkpoint from `first` on; the best (success and still standing, then
    the hands closest to the holding pose) is exported."""
    pl = Player(folder, envs)
    run = os.path.join(folder, "runs", "policy")
    res = []
    for it, f in sorted((int(f[6:-3]), f) for f in os.listdir(run) if f.startswith("model_")):
        if it >= first:
            rep = pl.evaluate(os.path.join(run, f))
            stand = rep[f"standing_at_{PLAY_S:g}_s"]
            res.append((min(rep["success"], stand), -rep["hands_off_holding_pose_m_min_median_max"][1], it, f))
            print(f"{f:16s} success {rep['success']:.4f}  standing {stand:.4f}  walked {rep['walked_m_min_median_max']}  "
                  f"turned {rep['turned_rad_min_median_max']}  hands {rep['hands_off_holding_pose_m_min_median_max']}",
                  flush=True)
    best = max(res)
    print("best:", best[3], best[0])
    print(os.path.relpath(export(folder, os.path.join(run, best[3])), folder), "<-", best[3])


# ------------------------------------------------------------------ the GIF

W, H = 640, 360
INK, MUTED, PAPER = (24, 32, 44), (110, 120, 135), (244, 244, 241)


def font(size, bold=False):
    from PIL import ImageFont
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")
    return ImageFont.truetype(p, size) if os.path.exists(p) else ImageFont.load_default()


def restyle(m):
    """Concrete floor, soft light, light haze (the box-carrying GIF's look)."""
    for i in range(m.ngeom):
        if m.geom_matid[i] == 0:                      # mjlab's checker ground plane
            m.geom_matid[i] = -1
            m.geom_rgba[i] = (0.58, 0.57, 0.54, 1.0)
    m.vis.rgba.haze[:] = (0.86, 0.87, 0.88, 1.0)
    m.vis.headlight.ambient[:] = (0.30, 0.30, 0.30)
    m.vis.headlight.diffuse[:] = (0.35, 0.35, 0.35)
    m.vis.headlight.specular[:] = (0.05, 0.05, 0.05)
    for i in range(m.nlight):
        m.light_diffuse[i] = (0.55, 0.55, 0.55)
        m.light_specular[i] = (0.1, 0.1, 0.1)
    m.vis.quality.shadowsize = 4096


def render(folder, zero=False, fps=20, azimuth=90.0, distance=3.4, elevation=-18.0, out=None):
    """One run from the plan's first frame, recorded with mjlab and drawn with MuJoCo's
    renderer: a side view of the walk, a tick on the floor every foot of it, the caption of
    the plan's phase."""
    import mujoco
    from PIL import Image, ImageDraw
    torch.manual_seed(0)
    pl = Player(folder, 1, episode_s(plan_path(folder)))
    env, t = pl.env, pl.task
    policy = pl.policy(actor_path(folder), zero)
    obs, _ = pl.vec.reset()
    S = []
    with torch.no_grad():
        for k in range(int(round(pl.seconds / env.step_dt)) - 1):      # the last step would reset
            S.append((env.sim.data.qpos[0].cpu().numpy().copy(), int(t.clock[0, 0])))
            obs, _, dones, _ = pl.vec.step(policy(obs))
            if bool(dones[0]):
                print(f"ended at {(k + 1) * env.step_dt:.2f} s")
                break
    m = env.sim.mj_model
    restyle(m)
    d = mujoco.MjData(m)
    ren = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    dt = env.step_dt
    org = env.scene.env_origins[0].cpu().numpy()
    w = t.plan.walk
    a, b = np.array(w["from"]), np.array(w["to"])
    u = (b - a) / np.linalg.norm(b - a)
    tick_rot = np.array([[u[0], -u[1], 0.0], [u[1], u[0], 0.0], [0.0, 0.0, 1.0]]).flatten()
    look = np.r_[org[:2] + 0.5 * (a + b), 0.65]
    phases = sorted(t.plan.phases, key=lambda p: p[1])
    every = max(1, int(round(1.0 / fps / dt)))
    f_big, f_sm = font(15, True), font(12)
    sub = "mjlab · MPC (Rekin) plan, PPO correction · " + ("plan played back, no correction" if zero else
                                                          "the policy's mean actions")
    frames = []
    for k in range(0, len(S), every):
        qpos, clk = S[k]
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        cam.lookat[:] = look
        cam.distance, cam.elevation, cam.azimuth = distance, elevation, azimuth
        ren.update_scene(d, camera=cam)
        for i in range(6):                            # a tick every foot of the walk
            if ren.scene.ngeom >= ren.scene.maxgeom:
                break
            p = org[:2] + a + i * FOOT * u
            mujoco.mjv_initGeom(ren.scene.geoms[ren.scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                                np.array([0.006, 0.12, 0.001]), np.array([p[0], p[1], 0.001]), tick_rot,
                                np.array([0.95, 0.95, 0.92, 1.0], np.float32))
            ren.scene.ngeom += 1
        img = Image.fromarray(ren.render())
        dr = ImageDraw.Draw(img, "RGBA")
        dr.rectangle([0, 0, W, 54], fill=(*PAPER, 225))
        dr.text((14, 8), t.plan.title, fill=INK, font=f_big)
        says = [c for _, s, c in phases if s <= clk * dt and c][-1]
        dr.text((14, 31), says, fill=INK, font=f_sm)
        clock = f"{k * dt:5.1f} s"
        dr.text((W - 14 - dr.textlength(clock, font=f_sm), 31), clock, fill=MUTED, font=f_sm)
        dr.rectangle([0, H - 22, W, H], fill=(*PAPER, 200))
        dr.text((14, H - 18), sub, fill=MUTED, font=f_sm)
        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=160))
    out = out or os.path.join(folder, f"g1_{slug(folder)}.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:] + [frames[-1]] * fps,
                   duration=int(1000 / fps), loop=0, optimize=True)
    print(out, os.path.getsize(out) // 1024, "KB,", len(frames), "frames")


# ------------------------------------------------------------------ command line

def main():
    ap = argparse.ArgumentParser(description="a box-free policy for one planned motion (run from its folder)")
    ap.add_argument("command", choices=("train", "select", "play", "render"))
    ap.add_argument("--dir", default=".", help="the motion's folder (default: here)")
    ap.add_argument("--iterations", type=int, default=1000, help="train")
    ap.add_argument("--envs", type=int, default=None, help="train: 4096; select: 64; play: 1,024")
    ap.add_argument("--from", dest="first", type=int, default=500, help="select: from this iteration on")
    ap.add_argument("--zero", action="store_true", help="play, render: the plan played back, no correction")
    ap.add_argument("--policy", default=None, help="play: a checkpoint (default: actors/policy.pt)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    folder = os.path.abspath(a.dir)
    if a.command == "train":
        train(folder, a.iterations, a.envs or 4096)
    elif a.command == "select":
        select(folder, a.first, a.envs or 64)
    elif a.command == "play":
        pl = Player(folder, a.envs or 1024)
        rep = pl.evaluate(a.policy or actor_path(folder), zero=a.zero)
        print(json.dumps(rep, indent=1))
        out = a.out or os.path.join(folder, "eval", "zero.json" if a.zero else "policy.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(rep, open(out, "w"), indent=1)
    else:
        render(folder, zero=a.zero, out=a.out)


if __name__ == "__main__":
    main()
