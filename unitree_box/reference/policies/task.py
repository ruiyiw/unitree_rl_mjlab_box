"""task.py — the mjlab task: Unitree's 23-DoF G1 takes a moving box off a cart,
walks to a shelf and puts the box on it, warm-started by the MPC's plans.

Runs in the `mjlab` env.  The job is split into three skills, each a PPO
policy trained in its own segment with its own curriculum (train.py --skill):

  fetch  walk to the cart, reach around the box, close the hands, lift it
         to the carry pose; done when the plan has ended and the box has been
         held 3 cm off the deck, upright and within BOX_TOL of where the plan
         holds it, neither robot nor box touching the cart, for HOLD_FOR s
  carry  step back, turn and walk the box to the stance in front of the
         shelf; done when the plan has ended and the robot has stood there
         holding the box (upright, within BOX_TOL of the plan's) for HOLD_FOR s
  place  lower the box onto the shelf's top board and let go; done when the
         plan has ended and the hands have been off the resting box for
         SUCCESS_HOLD s

Each skill follows its plan from the MPC (plan.py, banks/plans.pt).  The
policy observes the plan's next frames and outputs a CORRECTION to the plan's
joint targets: target = plan target + scale · action.  A zero action plays the
plan back on the PD joints, so every skill starts from the MPC's motion (the
warm start); episodes start at random frames of a random layout's plan
(reference state initialisation), at its first frame, or from the previous
skill's recorded end states (bank.py).  The reward is the tracking reward of
BeyondMimic (mjlab's tracking task: the torso's pose, the bodies' poses and
velocities relative to it), tighter, plus the hands' positions, the box's pose,
the hands on the box when the plan holds it, the skill's success and the
usual regularisers.  The arm motors carry the arms' weight (gravity
compensation, scene.py), so a plan's arm targets hold the plan's arm pose.

A pointer network on top (hier.py, train.py --skill full) reads the goals,
the robot, the box and the three plans' clocks, and picks every 0.1 s which
skill drives the joints; a skill's plan clock runs only while it is chosen.

Every plan lives in the frame of the box it handles (plan.py); an episode
places that frame anywhere in the env, the cart and the shelf around it.

Bookkeeping phases (the same in every env): FETCH → CARRY when both hands
touch the box and it is 3 cm off the deck; CARRY → DONE when the box rests on
the board with both palms still at its sides (set down, not dropped).
"""
from __future__ import annotations

import math
import os
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
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import (matrix_from_quat, quat_apply, quat_apply_inverse,
                                      quat_error_magnitude, quat_inv, quat_mul)
from mjlab.utils.lab_api.math import yaw_quat as yaw_of_quat
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import scene as sc

SKILLS = ("fetch", "carry", "place")
FETCH, CARRY, DONE = 0, 1, 2
NPHASE = 3
PLANS = os.path.join(sc.out_dir("banks", variant=False), "plans.pt")
PLANS_EVAL = os.path.join(sc.out_dir("banks", variant=False), "plans_eval.pt")
ANCHOR = "torso_link"
EE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_roll_rubber_hand",
             "right_wrist_roll_rubber_hand")

# ------------------------------------------------------------------ layout noise

ORIGIN_NOISE = 0.5                 # m, the cart's box frame around the env origin
CART_NOISE = (0.02, 0.02, 0.05)    # the cart against the box: x, y (m), yaw (rad)
SHELF_NOISE = (0.02, 0.02, 0.05)   # the shelf against the box's place point
RSI_NOISE = (0.02, 0.05, 0.03)     # reference-state starts: root xy (m), yaw (rad), leg joints (rad)

# start modes
BEGIN = 0       # the plan's first frame
RSI = 1         # a random frame of the plan
HANDOFF = 2     # one of the previous skill's recorded end states (plan's first frame)

# curricula: each skill's starts and pushes grow with its level ℓ ∈ [0, 1]
P_BEGIN = (0.2, 0.4)               # share of BEGIN starts at ℓ = 0 and 1
P_HANDOFF = (0.0, 0.4)             # share of HANDOFF starts (carry, place)
PUSH_V = 0.4                       # m/s, the largest push at ℓ = 1
LEVEL_UP, LEVEL_DOWN, LEVEL_STEP = 0.7, 0.3, 0.05
LEVEL_EVERY = 200                  # policy steps between curriculum updates

# ------------------------------------------------------------------ phase and success tests

LIFT = 0.03                        # m off the deck to count as lifted
REST_TOL = 0.015                   # m, box bottom from the board when resting
REST_SPEED = 0.10                  # m/s
TILT_MAX = 0.15                    # rad
PLACE_X = (-0.26, -0.02)           # box centre inside the board, shelf frame
PLACE_Y = 0.32
PLACE_GRIP = 0.06                  # m, each palm from its side target when set down
RELEASE_GAP = 0.05                 # m between palm surface and box: released
HOLD_FOR = 0.5                     # s: fetch and carry successes
SUCCESS_HOLD = 1.0                 # s released with the box resting: place and the whole job
AT_STANCE = (0.12, 0.2)            # m, rad: carry has arrived
BOX_TOL = 0.10                     # m: fetch and carry end with the box where the plan holds it,
HOLD_TILT = 0.25                   # rad: and upright (without it, fetch tilted it 30 degrees)
EPISODE_S = {"fetch": 10.0, "carry": 14.0, "place": 6.0, "full": 32.0}

# tracking terminations (BeyondMimic's, plus drift and the box)
TERM_ANCHOR_Z, TERM_ANCHOR_ORI, TERM_EE_Z = 0.25, 0.8, 0.25
TERM_DRIFT = 0.5                   # m, torso off its plan position, horizontally
TERM_BOX = 0.25                    # m, box off its plan position


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


def compose(F, G):
    """Frame G (x, y, yaw), given in frame F, in the world."""
    return torch.cat([F[:, :2] + rot2(G[:, :2], F[:, 2]), (F[:, 2] + G[:, 2])[:, None]], -1)


def _u(n, lo, hi, dev):
    return lo + (hi - lo) * torch.rand(n, device=dev)


def _lerp(a, b, t):
    return a + (b - a) * t


# ------------------------------------------------------------------ the plans

class Plans:
    """The MPC's plans (plan.py --bank) on the device: per skill, padded (C, T, ...)
    arrays over C layouts, in the plan frames (cart's box frame for fetch and
    carry, shelf's for place), joints in the bank's order."""

    def __init__(self, path, device, joint_names):
        b = torch.load(path, map_location="cpu", weights_only=False)
        self.n = len(b["start"])
        self.shelf = b["shelf"].to(device)                     # (C, 3) shelf box frame in the cart's
        perm = torch.tensor([b["joint_names"].index(n) for n in joint_names], device=device)
        self.sk = []
        for k in SKILLS:
            d = {key: v.to(device) for key, v in b[k].items() if torch.is_tensor(v)}
            d["qpos"] = torch.cat([d["qpos"][..., :7], d["qpos"][..., 7:][..., perm]], -1)
            d["qvel"] = torch.cat([d["qvel"][..., :6], d["qvel"][..., 6:][..., perm]], -1)
            d["qcmd"] = d["qcmd"][..., perm]
            d["grip"] = d["grip"].float()
            self.sk.append(d)
        self.body_names = b["body_names"]
        self.stats = b["stats"]


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
    if key in ("body_lin", "body_ang", "box_lin"):
        return torch.cat([rot2(v[..., :2], yaw), v[..., 2:3]], -1)
    if key == "box":
        return torch.cat([F[:, :2] + rot2(v[:, :2], yaw), v[:, 2:3], (v[:, 3] + yaw)[:, None]], -1)
    return v


# ------------------------------------------------------------------ geometry helpers

class Geo:
    """Per-step geometric quantities, computed from the simulator on demand."""

    def __init__(self, env):
        self.env = env
        r: Entity = env.scene["robot"]
        b: Entity = env.scene["box"]
        t: TaskCommand = env.command_manager.get_term("task")
        self.t = t
        self.base = r.data.root_link_pos_w
        self.yaw = yaw_of(r.data.root_link_quat_w)
        self.box = b.data.root_link_pos_w
        self.box_q = b.data.root_link_quat_w
        self.box_v = b.data.root_link_lin_vel_w
        palms = r.data.site_pos_w[:, t.palm_ids]               # (N, 2, 3) left, right
        # palms in the box frame; box y axis chosen to point at the robot's left
        rel = palms - self.box[:, None]
        q = self.box_q[:, None].expand(-1, 2, -1)
        pb = quat_apply_inverse(q.reshape(-1, 4), rel.reshape(-1, 3)).reshape(-1, 2, 3)
        n = len(self.yaw)
        left_w = torch.stack([-torch.sin(self.yaw), torch.cos(self.yaw), torch.zeros_like(self.yaw)], -1)
        yb = quat_apply(self.box_q, torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(n, 3))
        flip = torch.where((yb * left_w).sum(-1) < 0, -1.0, 1.0)
        pb = pb * torch.stack([flip, flip, torch.ones_like(flip)], -1)[:, None]
        half = torch.tensor(sc.BOX_HALF, device=env.device)
        side = torch.tensor([[0.0, sc.BOX_HALF[1] + sc.PALM_R, 0.0],
                             [0.0, -sc.BOX_HALF[1] - sc.PALM_R, 0.0]], device=env.device)
        self.palm_err = pb - side[None]                       # (N, 2, 3) to side targets
        self.d_side = self.palm_err.norm(dim=-1)              # (N, 2)
        out = (pb.abs() - half).clamp_min(0.0)
        self.gap = out.norm(dim=-1) - sc.PALM_R               # palm surface to box surface
        self.bottom = self.box[:, 2] - sc.BOX_HALF[2]
        zb = quat_apply(self.box_q, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(n, 3))
        self.tilt = torch.acos(zb[:, 2].abs().clamp(max=1.0))
        hb = env.scene["hand_box"].data.found
        self.touch = (hb > 0).float()[:, t.hand_cols]         # (N, 2)
        self.on_cart = ((env.scene["robot_cart"].data.found.sum(-1) > 0)
                        | (env.scene["box_cart"].data.found.sum(-1) > 0))   # robot or box touching the cart
        self.on_shelf = env.scene["robot_shelf"].data.found.sum(-1) > 0     # robot touching the shelf
        self.grip = self.touch[:, 0] * self.touch[:, 1]
        self.phase = t.phase

    def stance(self, which: str):
        """A stance (xy, heading) and the robot's distance and heading error to it:
        "fetch" in front of the box on the cart, "shelf" in front of its place point."""
        F = self.t.frame_c if which == "fetch" else self.t.frame_s
        xy = F[:, :2] + rot2(torch.tensor([[sc.HOLD_X, 0.0]], device=F.device).expand(len(F), 2), F[:, 2])
        hd = F[:, 2] + math.pi
        return xy, hd, (xy - self.base[:, :2]).norm(dim=-1), wrap(hd - self.yaw)

    def in_shelf(self):
        t = self.t
        loc = rot2(self.box[:, :2] - t.shelf_xy, -t.shelf_yaw)
        return ((loc[:, 0] > PLACE_X[0]) & (loc[:, 0] < PLACE_X[1]) & (loc[:, 1].abs() < PLACE_Y))

    def resting_on_shelf(self):
        return (self.in_shelf() & ((self.bottom - sc.SHELF_TOP).abs() < REST_TOL)
                & (self.box_v.norm(dim=-1) < REST_SPEED) & (self.tilt < TILT_MAX))

    def holding(self):
        """Both hands on the box, the palms at its sides."""
        return (self.grip > 0) & (self.d_side.max(dim=1).values < PLACE_GRIP)


def _cached(env, name, make):
    """A value of the current simulator state, shared by every term that runs
    between two state changes (a physics step, a reset, a clock tick)."""
    key = (env._sim_step_counter, getattr(env, "_geo_epoch", 0))
    c = env.__dict__.setdefault("_task_cache", {})
    if c.get("key") != key:
        c.clear()
        c["key"] = key
    if name not in c:
        c[name] = make()
    return c[name]


def geo(env) -> Geo:
    return _cached(env, "geo", lambda: Geo(env))


def _invalidate(env):
    env._geo_epoch = getattr(env, "_geo_epoch", 0) + 1


# ------------------------------------------------------------------ the task command

class TaskCommand(CommandTerm):
    """Lays out each episode from a plan, keeps the plans' clocks, the phase,
    the success timers and the curriculum level, and serves the reference."""

    cfg: "TaskCommandCfg"

    def __init__(self, cfg: "TaskCommandCfg", env):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.skill = cfg.skill
        self.robot: Entity = env.scene["robot"]
        self.boxe: Entity = env.scene["box"]
        self.cart: Entity = env.scene["cart"]
        self.shelf: Entity = env.scene["shelf"]
        self.palm_ids = [self.robot.site_names.index(s) for s in ("left_palm", "right_palm")]
        names = env.scene["hand_box"].primary_names
        self.hand_cols = [i for side in ("left", "right") for i, nm in enumerate(names)
                          if nm.startswith(side)]
        assert len(self.hand_cols) == 2, names
        self.plans = Plans(cfg.plans, dev, list(self.robot.joint_names))
        bn = list(self.robot.body_names)
        self.body_ids = [bn.index(b) for b in self.plans.body_names]
        self.anchor_j = self.plans.body_names.index(ANCHOR)
        self.ee_j = [self.plans.body_names.index(b) for b in EE_BODIES]
        self.hand_j = [self.plans.body_names.index(b) for b in EE_BODIES[2:]]
        z = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731
        self.chain = torch.zeros(n, dtype=torch.long, device=dev)
        self.frame_c, self.frame_s = z(n, 3), z(n, 3)          # plan frames in the world
        self.shelf_xy, self.shelf_yaw = z(n, 2), z(n)          # the shelf prop
        self.clock = torch.zeros(n, 3, dtype=torch.long, device=dev)
        self.started = torch.zeros(n, 3, dtype=torch.bool, device=dev)
        self.fresh = torch.zeros(n, dtype=torch.bool, device=dev)
        k0 = SKILLS.index(cfg.skill) if cfg.skill in SKILLS else 0
        self.active = torch.full((n,), k0, dtype=torch.long, device=dev)
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.start = torch.zeros(n, dtype=torch.long, device=dev)
        self.held_t, self.at_t, self.released = z(n), z(n), z(n)
        self.success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.dist_job = z(n)
        for k in ("success", "lifted", "placed"):
            self.metrics[k] = z(n)
        self.metrics["level"] = z(n)
        self.level = 0.0                # curriculum level ℓ
        self.curriculum = True          # play scripts freeze it (at ℓ = 1)
        self._ema, self._last_update = 0.0, 0
        self.fixed_chain = None         # play scripts: a layout index for every env
        self.fixed_frame = None         # play scripts: the cart's box frame (x, y, yaw) in the env
        self.force_start = None         # play scripts: start every episode in this mode
        self.start_frame = None         # play scripts: RSI starts at this plan frame
        self.handoff = load_bank(cfg.handoff, dev)

    @property
    def command(self):
        return self.clock.float()

    # -------------------------------------------------- the reference
    def lengths(self, k, ids=None):
        c = self.chain if ids is None else self.chain[ids]
        return self.plans.sk[k]["len"][c]

    def raw(self, k, key, off=0, ids=None):
        """Skill k's plan quantity at its clock + off (held at the last frame), plan frame."""
        sk = self.plans.sk[k]
        c = self.chain if ids is None else self.chain[ids]
        clk = self.clock[:, k] if ids is None else self.clock[ids, k]
        t = torch.minimum(clk + off, sk["len"][c] - 1)
        return sk[key][c, t]

    def world(self, k, key, off=0, ids=None):
        """Skill k's plan quantity in the world."""
        F = self.frame_s if k == 2 else self.frame_c
        return to_world(self.raw(k, key, off, ids), F if ids is None else F[ids], key)

    def R(self, key, off=0):
        """The ACTIVE skill's plan quantity in the world, every env (cached per step)."""
        def make():
            if self.skill in SKILLS:
                return self.world(SKILLS.index(self.skill), key, off)
            out = None
            for k in range(3):
                v = self.world(k, key, off)
                sel = (self.active == k).view(-1, *([1] * (v.dim() - 1)))
                out = v if out is None else torch.where(sel, v, out)
            return out
        return _cached(self._env, f"ref:{key}:{off}", make)

    def ended(self, k):
        return self.clock[:, k] >= self.lengths(k) - 1

    # -------------------------------------------------- curriculum
    def _curriculum(self, env_ids):
        if not self.curriculum or len(env_ids) == 0:
            return
        rate = self.success[env_ids].float().mean().item()
        self._ema = 0.98 * self._ema + 0.02 * rate
        step = self._env.common_step_counter
        if step - self._last_update >= LEVEL_EVERY:
            self._last_update = step
            if self._ema > LEVEL_UP:
                self.level = min(1.0, self.level + LEVEL_STEP)
            elif self._ema < LEVEL_DOWN:
                self.level = max(0.0, self.level - LEVEL_STEP)

    def _modes(self, n, dev):
        if self.force_start is not None:
            return torch.full((n,), self.force_start, dtype=torch.long, device=dev)
        if self.skill == "full":
            return torch.full((n,), BEGIN, dtype=torch.long, device=dev)
        L = self.level if self.curriculum else 1.0
        u = torch.rand(n, device=dev)
        p_h = _lerp(*P_HANDOFF, L) if self.handoff is not None else 0.0
        p_b = _lerp(*P_BEGIN, L)
        m = torch.where(u < p_h, HANDOFF, torch.where(u < p_h + p_b, BEGIN, RSI))
        return m

    # -------------------------------------------------- reset
    def _resample_command(self, env_ids):
        self._curriculum(env_ids)
        _invalidate(self._env)
        n, dev = len(env_ids), self.device
        k0 = SKILLS.index(self.skill) if self.skill in SKILLS else 0
        mode = self._modes(n, dev)
        self.start[env_ids] = mode
        if self.fixed_chain is not None:
            chain = torch.full((n,), self.fixed_chain, dtype=torch.long, device=dev)
        else:
            chain = torch.randint(self.plans.n, (n,), device=dev)
        bank_j = None
        hand = mode == HANDOFF
        if self.handoff is not None and bool(hand.any()):
            bank_j = torch.randint(len(self.handoff["chain"]), (n,), device=dev)
            chain = torch.where(hand, self.handoff["chain"][bank_j], chain)
        self.chain[env_ids] = chain
        # the plan frames: the cart's box frame around the env origin, the shelf's from the layout
        org = self._env.scene.env_origins[env_ids]
        if self.fixed_frame is not None:
            Fc = torch.tensor(self.fixed_frame, device=dev, dtype=torch.float32)[None].expand(n, 3).clone()
            Fc[:, :2] += org[:, :2]
        else:
            Fc = torch.cat([org[:, :2] + (2 * torch.rand(n, 2, device=dev) - 1) * ORIGIN_NOISE,
                            _u(n, -math.pi, math.pi, dev)[:, None]], -1)
        Fs = compose(Fc, self.plans.shelf[chain])
        self.frame_c[env_ids], self.frame_s[env_ids] = Fc, Fs
        noise = 1.0 if self.fixed_frame is None else 0.0

        def prop(F, nz):
            off = torch.stack([-sc.BOX_ON_PROP_X + noise * _u(n, -nz[0], nz[0], dev),
                               noise * _u(n, -nz[1], nz[1], dev), noise * _u(n, -nz[2], nz[2], dev)], -1)
            return compose(F, off)

        cart, shelf = prop(Fc, CART_NOISE), prop(Fs, SHELF_NOISE)
        self.shelf_xy[env_ids], self.shelf_yaw[env_ids] = shelf[:, :2], shelf[:, 2]
        zc = torch.zeros(n, 1, device=dev)
        self.cart.write_mocap_pose_to_sim(torch.cat([cart[:, :2], zc, yaw_quat(cart[:, 2])], -1), env_ids)
        self.shelf.write_mocap_pose_to_sim(torch.cat([shelf[:, :2], zc, yaw_quat(shelf[:, 2])], -1), env_ids)
        # the clocks: the starting skill at frame 0 (BEGIN, HANDOFF) or a random frame (RSI)
        self.clock[env_ids] = 0
        lens = self.plans.sk[k0]["len"][chain]
        t0 = torch.where(mode == RSI, (torch.rand(n, device=dev) * (lens - 1).float()).long(),
                         torch.zeros_like(lens))
        if self.start_frame is not None:
            t0 = torch.where(mode == RSI, torch.clamp(torch.full_like(t0, self.start_frame), max=lens - 1), t0)
        self.clock[env_ids, k0] = t0
        self.started[env_ids] = False
        self.started[env_ids, k0] = True
        self.active[env_ids] = k0
        self.fresh[env_ids] = True
        # the robot and the box from the plan
        q = self.world(k0, "qpos", 0, env_ids)
        qd = self.world(k0, "qvel", 0, env_ids)
        bx = self.world(k0, "box", 0, env_ids)
        bv = self.world(k0, "box_lin", 0, env_ids)
        grip = self.raw(k0, "grip", 0, env_ids) > 0
        root = torch.cat([q[:, :7], qd[:, :6]], -1)
        jp, jv = q[:, 7:].clone(), qd[:, 6:].clone()
        if noise and self.fixed_chain is None:
            rsi = (mode == RSI).float()
            root[:, :2] += rsi[:, None] * (2 * torch.rand(n, 2, device=dev) - 1) * RSI_NOISE[0]
            dy = rsi * _u(n, -RSI_NOISE[1], RSI_NOISE[1], dev)
            root[:, 3:7] = quat_mul(yaw_quat(dy), root[:, 3:7])
            legs = self._leg_mask[None].float()
            jp += rsi[:, None] * legs * (2 * torch.rand_like(jp) - 1) * RSI_NOISE[2]
        self.robot.write_root_state_to_sim(root, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
        bq = yaw_quat(bx[:, 3])
        bvel = torch.cat([bv * grip[:, None].float(), torch.zeros(n, 3, device=dev)], -1)
        bpos = bx[:, :3] + torch.tensor([0, 0, 0.001], device=dev) * (~grip)[:, None].float()
        self.boxe.write_root_state_to_sim(torch.cat([bpos, bq, bvel], -1), env_ids=env_ids)
        am = self._env.action_manager
        for buf in (am._action, am._prev_action, am._prev_prev_action):
            buf[env_ids] = 0.0
        if bank_j is not None and bool(hand.any()):
            i = hand.nonzero().squeeze(-1)
            self._write_bank(env_ids[i], bank_j[i], (Fs if k0 == 2 else Fc)[i])
        # bookkeeping phase from where the plan starts
        lifted = grip & (bx[:, 2] - sc.BOX_HALF[2] > sc.CART_TOP + LIFT / 2)
        set_down = (k0 == 2) & ~grip & (t0 > 0)
        ph = torch.full((n,), FETCH if k0 == 0 else CARRY, dtype=torch.long, device=dev)
        ph = torch.where(lifted, torch.full_like(ph, CARRY), ph)
        ph = torch.where(set_down, torch.full_like(ph, DONE), ph)
        self.phase[env_ids] = ph
        for b in (self.held_t, self.at_t, self.released):
            b[env_ids] = 0.0
        self.success[env_ids] = False
        self.dist_job[env_ids] = 0.0
        fs = self._env.__dict__.get("_feet")
        if fs is not None:
            fs.reset(env_ids)

    @property
    def _leg_mask(self):
        if not hasattr(self, "_legs"):
            jn = self.robot.joint_names
            self._legs = torch.tensor([("hip" in j or "knee" in j or "ankle" in j) for j in jn], device=self.device)
        return self._legs

    def _write_bank(self, eids, j, F):
        """Start eids from hand-off bank entries j, in plan frames F."""
        b = self.handoff
        pos = lambda p: torch.cat([F[:, :2] + rot2(p[j, :2], F[:, 2]), p[j, 2:3]], -1)  # noqa: E731
        vec = lambda v: torch.cat([rot2(v[j, :2], F[:, 2]), v[j, 2:3]], -1)             # noqa: E731
        rot = lambda q: quat_mul(yaw_quat(F[:, 2]), q[j])                               # noqa: E731
        for ent, k in ((self.robot, "root"), (self.boxe, "box")):
            ent.write_root_state_to_sim(torch.cat([pos(b[f"{k}_pos"]), rot(b[f"{k}_quat"]),
                                                   vec(b[f"{k}_lin"]), vec(b[f"{k}_ang"])], -1), env_ids=eids)
        self.robot.write_joint_state_to_sim(b["joint_pos"][j], b["joint_vel"][j], env_ids=eids)
        # a position-controlled robot's joint targets are part of its state: the
        # previous skill's last correction (to a plan target that matches this plan's first)
        am = self._env.action_manager
        if am.action.shape[-1] == b["action"].shape[-1]:
            for buf in (am._action, am._prev_action, am._prev_prev_action):
                buf[eids] = b["action"][j]

    def snapshot(self, ids, frame: str) -> dict:
        """The robot's and the box's states of envs ids in a plan frame ("cart" or
        "shelf"), with the layout and the last correction: a hand-off bank entry."""
        F = (self.frame_c if frame == "cart" else self.frame_s)[ids]
        pos = lambda p: torch.cat([rot2(p[:, :2] - F[:, :2], -F[:, 2]), p[:, 2:3]], -1)  # noqa: E731
        vec = lambda v: torch.cat([rot2(v[:, :2], -F[:, 2]), v[:, 2:3]], -1)              # noqa: E731
        rot = lambda q: quat_mul(yaw_quat(-F[:, 2]), q)                                   # noqa: E731
        out = {}
        for ent, k in ((self.robot, "root"), (self.boxe, "box")):
            d = ent.data
            out[f"{k}_pos"] = pos(d.root_link_pos_w[ids])
            out[f"{k}_quat"] = rot(d.root_link_quat_w[ids])
            out[f"{k}_lin"] = vec(d.root_link_lin_vel_w[ids])
            out[f"{k}_ang"] = vec(d.root_link_ang_vel_w[ids])
        out["joint_pos"] = self.robot.data.joint_pos[ids].clone()
        out["joint_vel"] = self.robot.data.joint_vel[ids].clone()
        out["action"] = obs_last_joint_action(self._env)[ids].clone()
        out["chain"] = self.chain[ids].clone()
        return out

    # -------------------------------------------------- per step
    def _update_command(self, env_ids):
        env = self._env
        if env_ids is not None:
            return
        # the active plan's clock ticks (not on the step an env was reset)
        tick = (~self.fresh).long()
        self.clock.scatter_add_(1, self.active[:, None], tick[:, None])
        self.fresh[:] = False
        _invalidate(env)
        g = geo(env)
        dt = env.step_dt
        lifted = (self.phase == FETCH) & (g.grip > 0) & (g.bottom > sc.CART_TOP + LIFT)
        self.phase[lifted] = CARRY
        placed = (self.phase == CARRY) & g.holding() & g.resting_on_shelf()
        self.phase[placed] = DONE
        on_plan = ((self.R("box", 0)[:, :3] - g.box).norm(dim=-1) < BOX_TOL) & (g.tilt < HOLD_TILT)
        up = ((self.phase == CARRY) & g.holding() & (g.bottom > sc.CART_TOP + LIFT) & ~g.on_cart
              & self.ended(0) & on_plan)
        self.held_t = torch.where(up, self.held_t + dt, torch.zeros_like(self.held_t))
        _, _, ds, hs = g.stance("shelf")
        there = ((self.phase == CARRY) & g.holding() & (g.bottom > sc.SHELF_TOP + LIFT)
                 & (ds < AT_STANCE[0]) & (hs.abs() < AT_STANCE[1]) & ~g.on_shelf & self.ended(1) & on_plan)
        self.at_t = torch.where(there, self.at_t + dt, torch.zeros_like(self.at_t))
        free = ((self.phase == DONE) & (g.gap.min(dim=1).values > RELEASE_GAP) & g.resting_on_shelf()
                & ~g.on_shelf & self.ended(2))
        self.released = torch.where(free, self.released + dt, torch.zeros_like(self.released))
        x = self.cfg.extra_hold
        if self.skill == "fetch":
            self.success = self.held_t >= HOLD_FOR + x
        elif self.skill == "carry":
            self.success = self.at_t >= HOLD_FOR + x
        else:
            self.success = self.released >= SUCCESS_HOLD + x
        m = self.metrics
        m["success"] = torch.maximum(m["success"], self.success.float())
        m["lifted"] = torch.maximum(m["lifted"], (self.phase >= CARRY).float())
        m["placed"] = torch.maximum(m["placed"], (self.phase == DONE).float())

    def _update_metrics(self):
        self.metrics["level"][:] = self.level   # the others are updated with the phases


def load_bank(path, device):
    if not path or not os.path.exists(path):
        return None
    return {k: v.to(device) for k, v in torch.load(path, map_location=device).items()}


@dataclass(kw_only=True)
class TaskCommandCfg(CommandTermCfg):
    skill: str = "fetch"
    plans: str = PLANS
    handoff: str | None = None
    extra_hold: float = 0.0          # s a skill must hold its end before success counts, beyond HOLD_FOR /
    #                                  SUCCESS_HOLD (robust training: ROBUST_HOLD)
    resampling_time_range: tuple[float, float] = (1e9, 1e9)

    def build(self, env) -> TaskCommand:
        return TaskCommand(self, env)


def task(env) -> TaskCommand:
    return env.command_manager.get_term("task")


# ------------------------------------------------------------------ the action: corrections to the plan

class PlanJointAction(JointPositionAction):
    """Joint targets = the active plan's targets one frame ahead + scale · action."""

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        ref = task(self._env).R("qcmd", off=1)
        self._processed_actions = self._raw_actions * self._scale + ref[:, self._target_ids]


@dataclass(kw_only=True)
class PlanJointActionCfg(JointPositionActionCfg):
    def build(self, env) -> PlanJointAction:
        return PlanJointAction(self, env)


# ------------------------------------------------------------------ observations

def _yaw_frame(v, yaw):
    """World vectors (N, 3) into a yaw frame."""
    return torch.cat([rot2(v[:, :2], -yaw), v[:, 2:3]], -1)


def _anchor(env):
    r = env.scene["robot"]
    t = task(env)
    i = t.body_ids[t.anchor_j]
    return r.data.body_link_pos_w[:, i], r.data.body_link_quat_w[:, i]


def obs_ref_joints(env, skill: int):
    """The plan's joint positions (from the default) and velocities."""
    t = task(env)
    q = t.world(skill, "qpos")[:, 7:] - env.scene["robot"].data.default_joint_pos
    return torch.cat([q, t.world(skill, "qvel")[:, 6:]], -1)


def obs_ref_anchor(env, skill: int):
    """The plan's torso position and orientation (two matrix columns) in the robot's torso frame."""
    t = task(env)
    p, q = _anchor(env)
    ref = t.world(skill, "body")[:, t.anchor_j]
    pos = quat_apply_inverse(q, ref[:, :3] - p)
    rel = matrix_from_quat(quat_mul(quat_inv(q), ref[:, 3:]))
    return torch.cat([pos, rel[..., :2].reshape(-1, 6)], -1)


def obs_ref_preview(env, skill: int):
    """The plan's torso 0.2 s and 0.5 s ahead: position in the robot's yaw frame,
    sin and cos of its heading error."""
    t = task(env)
    p, q = _anchor(env)
    yaw = yaw_of(q)
    parts = []
    for off in (10, 25):
        ref = t.world(skill, "body", off)[:, t.anchor_j]
        e = wrap(yaw_of(ref[:, 3:]) - yaw)
        parts += [_yaw_frame(ref[:, :3] - p, yaw), torch.sin(e)[:, None], torch.cos(e)[:, None]]
    return torch.cat(parts, -1)


def obs_ref_box(env, skill: int):
    """The plan's box position in the robot's yaw frame, and whether the plan holds it."""
    t = task(env)
    g = geo(env)
    b = t.world(skill, "box")
    return torch.cat([_yaw_frame(b[:, :3] - g.base, g.yaw), t.raw(skill, "grip")[:, None]], -1)


def obs_ref_clock(env, skill: int):
    """The plan's progress (0 → 1) and whether it has ended."""
    t = task(env)
    L = t.lengths(skill).float()
    return torch.stack([t.clock[:, skill].float() / L, t.ended(skill).float()], -1)


def obs_clocks(env):
    """Every skill's plan: started, progress, ended (the pointer's view)."""
    t = task(env)
    parts = []
    for k in range(3):
        parts.append(torch.stack([t.started[:, k].float(), t.clock[:, k].float() / t.lengths(k).float(),
                                  t.ended(k).float()], -1))
    return torch.cat(parts, -1)


def obs_box(env):
    """Box position, orientation (two axes) and velocity in the robot's yaw frame."""
    g = geo(env)
    n = len(g.yaw)
    qy = yaw_quat(-g.yaw)
    ax = quat_apply(qy, quat_apply(g.box_q, torch.tensor([1.0, 0, 0], device=env.device).expand(n, 3)))
    az = quat_apply(qy, quat_apply(g.box_q, torch.tensor([0, 0, 1.0], device=env.device).expand(n, 3)))
    return torch.cat([_yaw_frame(g.box - g.base, g.yaw), ax, az, _yaw_frame(g.box_v, g.yaw)], -1)


def obs_palms(env):
    """Each palm's offset from its side target, in the box frame (6), and the
    two hand-box contact flags (2)."""
    g = geo(env)
    return torch.cat([g.palm_err.reshape(-1, 6), g.touch], -1)


def obs_goals(env):
    """Both stances (dx, dy in the yaw frame; sin, cos of the heading error), and
    the box's place point on the shelf and its home on the cart (yaw frame)."""
    g = geo(env)
    t = task(env)
    parts = []
    for which, F, z in (("fetch", t.frame_c, sc.CART_TOP), ("shelf", t.frame_s, sc.SHELF_TOP)):
        xy, hd, dist, herr = g.stance(which)
        parts += [rot2(xy - g.base[:, :2], -g.yaw), torch.sin(herr)[:, None], torch.cos(herr)[:, None]]
        spot = torch.cat([F[:, :2], torch.full((len(F), 1), z + sc.BOX_HALF[2], device=env.device)], -1)
        parts.append(_yaw_frame(spot - g.base, g.yaw))
    return torch.cat(parts, -1)


def obs_last_joint_action(env):
    """The correction of the last step (whichever skill produced it)."""
    am = env.action_manager
    if "skill" in am.active_terms:
        return am.get_term("skill").joint_action
    return am.action


def obs_phase(env):
    return torch.nn.functional.one_hot(task(env).phase, NPHASE).float()


def obs_body_err(env):
    """Critic only: each tracked body's offset from its plan pose (relative to the torso)."""
    return (_rel_ref_bodies(env)[..., :3] - _robot_bodies(env)[0]).reshape(env.num_envs, -1)


def obs_box_mass(env):
    bm = env.sim.model.body_mass
    i = task(env).boxe.indexing.body_ids[0]
    m = bm[:, i] if bm.dim() == 2 else bm[i].expand(env.num_envs)
    return (m - sc.BOX_MASS)[:, None]


# ------------------------------------------------------------------ rewards: tracking (BeyondMimic's)

def _robot_bodies(env):
    t = task(env)
    d = env.scene["robot"].data
    i = t.body_ids
    return d.body_link_pos_w[:, i], d.body_link_quat_w[:, i], d.body_link_lin_vel_w[:, i], d.body_link_ang_vel_w[:, i]


def _rel_ref_bodies(env, off=1):
    """The plan's bodies moved so that its torso sits on the robot's torso (its
    height kept) and turned by the heading difference."""
    def make():
        t = task(env)
        ref = t.R("body", off)
        pa, qa = _anchor(env)
        ra = ref[:, t.anchor_j]
        dpos = pa.clone()
        dpos[:, 2] = ra[:, 2]
        dq = yaw_of_quat(quat_mul(qa, quat_inv(ra[:, 3:])))
        nb = ref.shape[1]
        dqe = dq[:, None].expand(-1, nb, -1).reshape(-1, 4)
        pos = dpos[:, None] + quat_apply(dqe, (ref[..., :3] - ra[:, None, :3]).reshape(-1, 3)).reshape(-1, nb, 3)
        quat = quat_mul(dqe, ref[..., 3:].reshape(-1, 4)).reshape(-1, nb, 4)
        return torch.cat([pos, quat], -1)
    return _cached(env, f"relref:{off}", make)


def r_anchor_pos(env, std: float):
    t = task(env)
    p, _ = _anchor(env)
    e = (t.R("body", 1)[:, t.anchor_j, :3] - p).square().sum(-1)
    return torch.exp(-e / std ** 2)


def r_anchor_ori(env, std: float):
    t = task(env)
    _, q = _anchor(env)
    e = quat_error_magnitude(t.R("body", 1)[:, t.anchor_j, 3:], q).square()
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
    e = (task(env).R("body_lin", 1) - _robot_bodies(env)[2]).square().sum(-1).mean(-1)
    return torch.exp(-e / std ** 2)


def r_body_ang_vel(env, std: float):
    e = (task(env).R("body_ang", 1) - _robot_bodies(env)[3]).square().sum(-1).mean(-1)
    return torch.exp(-e / std ** 2)


def r_box(env, std: float = 0.05):
    """The box at its plan position, upright."""
    g = geo(env)
    e = (task(env).R("box", 1)[:, :3] - g.box).square().sum(-1)
    return torch.exp(-e / std ** 2) * torch.exp(-(g.tilt / 0.2) ** 2)


def r_hands(env, std: float):
    """The hands (wrists) at their plan positions, relative to the torso."""
    t = task(env)
    e = (_rel_ref_bodies(env)[:, t.hand_j, :3] - _robot_bodies(env)[0][:, t.hand_j]).square().sum(-1).mean(-1)
    return torch.exp(-e / std ** 2)


def r_grip(env):
    """Both hands on the box while the plan holds it; neither touching it otherwise."""
    g = geo(env)
    hold = task(env).R("grip", 1) > 0
    return torch.where(hold, g.grip, 1.0 - g.touch.max(-1).values)


def r_success(env):
    """The skill's (or the whole job's) success, once: BONUS/dt on that step."""
    return task(env).success.float() / env.step_dt


def r_prop_contact(env):
    """Any robot geom touching the cart or the shelf."""
    c = env.scene["robot_cart"].data.found.sum(-1) + env.scene["robot_shelf"].data.found.sum(-1)
    return (c > 0).float()


def r_stage(env):
    """The whole job: 1 per second in CARRY, 2 in DONE."""
    return task(env).phase.float()


def r_progress_job(env):
    """The whole job: speed toward the cart's stance before the lift, the shelf's after."""
    g = geo(env)
    t = task(env)
    _, _, df, _ = g.stance("fetch")
    _, _, ds, _ = g.stance("shelf")
    d = torch.where(g.phase == FETCH, df, ds)
    v = torch.where(t.dist_job > 0, (t.dist_job - d) / env.step_dt, torch.zeros_like(d))
    t.dist_job = d
    return v.clamp(-1.0, 1.0) * (g.phase != DONE).float()


def r_switch(env):
    """The pointer changed skill this step."""
    return env.action_manager.get_term("skill").switched.float()


# ------------------------------------------------------------------ terminations

def t_tracking(env):
    """BeyondMimic's: the torso's height or tilt, or a foot or hand's height, far
    from the plan's; and the torso drifted TERM_DRIFT from the plan's spot."""
    t = task(env)
    p, q = _anchor(env)
    ref = t.R("body", 1)
    ra = ref[:, t.anchor_j]
    bad = (ra[:, 2] - p[:, 2]).abs() > TERM_ANCHOR_Z
    gw = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(len(p), 3)
    bad |= (quat_apply_inverse(ra[:, 3:], gw)[:, 2] - quat_apply_inverse(q, gw)[:, 2]).abs() > TERM_ANCHOR_ORI
    rel = _rel_ref_bodies(env)[:, t.ee_j, 2]
    bad |= ((rel - _robot_bodies(env)[0][:, t.ee_j, 2]).abs() > TERM_EE_Z).any(-1)
    bad |= (ra[:, :2] - p[:, :2]).norm(dim=-1) > TERM_DRIFT
    return bad


def t_box_lost(env):
    g = geo(env)
    far = (task(env).R("box", 1)[:, :3] - g.box).norm(dim=-1) > TERM_BOX
    return (g.box[:, 2] < 0.45) | far


def t_success(env):
    return task(env).success


def ev_push(env, env_ids, vmax: float):
    """A velocity kick to the robot's base, up to vmax·ℓ in each horizontal axis."""
    if env_ids is None or len(env_ids) == 0:
        return
    t = task(env)
    v = vmax * (t.level if t.curriculum else 1.0)
    if v <= 0:
        return
    r: Entity = env.scene["robot"]
    vel = r.data.root_link_vel_w[env_ids].clone()
    vel[:, :2] += (2 * torch.rand(len(env_ids), 2, device=env.device) - 1) * v
    r.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
    _count(env, "push", env_ids)


def _count(env, name, ids):
    """How many perturbations of each kind each env has had (render_robust.py draws them)."""
    c = env.__dict__.setdefault("_perturbed", {})
    if name not in c:
        c[name] = torch.zeros(env.num_envs, device=env.device)
    c[name][ids] += 1


KICK = (1.0, 0.15)                 # box kicks: rad/s about its horizontal axes, m/s per axis (at ℓ = 1)


def _holding(env):
    """The plan holds the box and both hands touch it."""
    return (task(env).R("grip", 0) > 0) & (geo(env).touch.min(-1).values > 0)


def ev_kick(env, env_ids, wmax: float, vmax: float):
    """A kick to the box while the plan holds it and both hands touch it: its
    angular velocity about its horizontal axes up to wmax·ℓ, its velocity up to vmax·ℓ."""
    if env_ids is None or len(env_ids) == 0:
        return
    t = task(env)
    lvl = t.level if t.curriculum else 1.0
    held = _holding(env)[env_ids]
    ids = env_ids[held]
    if len(ids) == 0 or lvl <= 0:
        return
    b: Entity = env.scene["box"]
    vel = b.data.root_link_vel_w[ids].clone()
    w_b = torch.zeros(len(ids), 3, device=env.device)
    w_b[:, :2] = (2 * torch.rand(len(ids), 2, device=env.device) - 1) * wmax * lvl
    vel[:, 3:] += quat_apply(b.data.root_link_quat_w[ids], w_b)
    vel[:, :3] += (2 * torch.rand(len(ids), 3, device=env.device) - 1) * vmax * lvl
    b.write_root_link_velocity_to_sim(vel, env_ids=ids)
    _count(env, "kick", ids)


# ------------------------------------------------------------------ robust training (G1_ROBUST=1)
#
# Fine-tuning the skills against what a search for counterexamples to stability
# certificates found around the 29-DoF G1's run (README "Robust training"): a foot's
# contact switching on a 5 ms substep earlier or later moves the next tick's ankle by up
# to 0.66 rad/s and 1.5°, from states 0.1 mm apart, and the loop is slow to take that out
# (11 of 111 directions still grow after five ticks).  So, while training:
#   at each touchdown, a jolt of that size to the landing leg (its ankle's pitch and roll,
#   less to the knee and the hip pitch), with chance JOLT_P;
#   kicks to the held box (the box's height and offset in the grip are the directions
#   the linearization keeps longest);
#   softer touchdowns: the jump grows with the speed the foot meets the floor at (the
#   human walk's skills land at a median 0.34 m/s, 10% faster than 0.71 m/s), so its
#   square is penalised at each touchdown.
JOLT_P = 0.5                       # chance of a jolt at a touchdown
JOLT_V = (1.0, 0.5)                # rad/s at ℓ = 1: the ankle's pitch and roll; the knee and hip pitch
JOLT_Q = 0.03                      # rad at ℓ = 1: the ankle's pitch and roll
IMPACT = -20.0                     # weight of the touchdown speed² (m²/s²): about a quarter of the tracking terms
# In the whole job, a skill holds its end until the pointer switches, and every perturbation
# goes on meanwhile; its own episodes end at success, so it never practised that.  With
# pushes and box kicks, the first robust round lost the box in 15% of whole jobs in fetch's
# end, before the switch to carry.  So a skill's success, in robust training only, needs
# ROBUST_HOLD s more of holding its end (the episodes are as much longer).
ROBUST_HOLD = float(os.environ.get("G1_ROBUST_HOLD", "1.5"))      # (round 1 of the robust skills: 0)
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


class Feet:
    """Each foot's floor contact and vertical speed over the last policy steps."""

    def __init__(self, env):
        r: Entity = env.scene["robot"]
        n, dev = env.num_envs, env.device
        self.ids = [r.body_names.index(b) for b in FEET]
        jn = list(r.joint_names)
        self.ankle = [[jn.index(f"{s}_ankle_pitch_joint"), jn.index(f"{s}_ankle_roll_joint")] for s in ("left", "right")]
        self.upper = [[jn.index(f"{s}_knee_joint"), jn.index(f"{s}_hip_pitch_joint")] for s in ("left", "right")]
        self.found = torch.ones(n, 2, dtype=torch.bool, device=dev)
        self.vz = torch.zeros(3, n, 2, device=dev)              # the last three policy steps
        self.impact = torch.zeros(n, device=dev)                # Σ touchdown speed², this step
        self.touchdowns = torch.zeros(n, device=dev)
        self.jolts = torch.zeros(n, device=dev)
        self.log = None                                         # a list: each touchdown's speed (stress.py)

    def reset(self, env_ids):
        self.found[env_ids] = True
        self.vz[:, env_ids] = 0.0
        self.impact[env_ids] = 0.0


def feet(env) -> Feet:
    if "_feet" not in env.__dict__:
        env._feet = Feet(env)
    return env._feet


def ev_touchdown(env, env_ids, p: float, wv: tuple, q: float):
    """Every step: each foot's touchdowns (its floor contact switching on), their speed,
    and with chance p a jolt to the landing leg, up to wv·ℓ (rad/s) and q·ℓ (rad)."""
    f = feet(env)
    r: Entity = env.scene["robot"]
    found = env.scene["feet_ground_contact"].data.found > 0
    f.vz = torch.roll(f.vz, 1, 0)
    f.vz[0] = r.data.body_link_lin_vel_w[:, f.ids, 2]
    td = found & ~f.found
    f.found = found
    f.impact = (td.float() * f.vz.min(0).values.clamp(max=0.0) ** 2).sum(-1)
    f.touchdowns += td.float().sum(-1)
    if f.log is not None and bool(td.any()):
        f.log.append((-f.vz.min(0).values.clamp(max=0.0))[td])
    t = task(env)
    lvl = t.level if t.curriculum else 1.0
    if p <= 0 or lvl <= 0:
        return
    hit = td & (torch.rand(td.shape, device=env.device) < p)
    for side in (0, 1):
        ids = hit[:, side].nonzero().squeeze(-1)
        if len(ids) == 0:
            continue
        pos, vel = r.data.joint_pos[ids].clone(), r.data.joint_vel[ids].clone()
        u = lambda k: 2 * torch.rand(len(ids), k, device=env.device) - 1  # noqa: E731
        vel[:, f.ankle[side]] += u(2) * wv[0] * lvl
        vel[:, f.upper[side]] += u(2) * wv[1] * lvl
        pos[:, f.ankle[side]] += u(2) * q * lvl
        r.write_joint_state_to_sim(pos, vel, env_ids=ids)
        f.jolts[ids] += 1
        _count(env, "jolt", ids)


def r_impact(env):
    """Σ over this step's touchdowns of the foot's vertical speed² before it met the floor."""
    return feet(env).impact


# ------------------------------------------------------------------ config

SUCCESS_BONUS = 50.0
POINTER_EVERY = 5                  # the pointer decides every 5 policy steps (10 Hz)

# (weight, std): BeyondMimic's terms, tighter (its 0.3 m / 0.4 rad let the policy trade
# the plan's posture for an easier lift: box up by the head, leaning back), and the hands
TRACK = dict(anchor_pos=(0.5, 0.15), anchor_ori=(0.5, 0.3), body_pos=(1.0, 0.1), body_ori=(1.0, 0.3),
             body_lin_vel=(1.0, 1.0), body_ang_vel=(1.0, 3.14), hands=(1.0, 0.05))
TASK = dict(box=2.0, grip=0.5, success=SUCCESS_BONUS, prop_contact=-1.0, action_rate=-0.1,
            joint_limit=-10.0)
POINTER = dict(stage=1.0, progress_job=2.0, success=SUCCESS_BONUS, switch=-1.0)


def _rewards(skill: str) -> dict:
    R = RewardTermCfg
    fn = {"anchor_pos": r_anchor_pos, "anchor_ori": r_anchor_ori, "body_pos": r_body_pos,
          "body_ori": r_body_ori, "body_lin_vel": r_body_lin_vel, "body_ang_vel": r_body_ang_vel,
          "hands": r_hands}
    if skill == "full":
        return {"stage": R(func=r_stage, weight=POINTER["stage"]),
                "progress_job": R(func=r_progress_job, weight=POINTER["progress_job"]),
                "success": R(func=r_success, weight=POINTER["success"]),
                "switch": R(func=r_switch, weight=POINTER["switch"])}
    out = {k: R(func=fn[k], weight=w, params={"std": s}) for k, (w, s) in TRACK.items()}
    out.update({
        "box": R(func=r_box, weight=TASK["box"]),
        "grip": R(func=r_grip, weight=TASK["grip"]),
        "success": R(func=r_success, weight=TASK["success"]),
        "prop_contact": R(func=r_prop_contact, weight=TASK["prop_contact"]),
        "action_rate": R(func=envs_mdp.action_rate_l2, weight=TASK["action_rate"]),
        "joint_limit": R(func=envs_mdp.joint_pos_limits, weight=TASK["joint_limit"]),
    })
    return out


def skill_obs(k: int, noise) -> dict:
    """A skill's observation: the plan ahead, the robot, the box."""
    O = ObservationTermCfg
    return {
        "ref_joints": O(func=obs_ref_joints, params={"skill": k}),
        "ref_anchor": O(func=obs_ref_anchor, params={"skill": k}, noise=noise(0.05)),
        "ref_preview": O(func=obs_ref_preview, params={"skill": k}, noise=noise(0.05)),
        "ref_box": O(func=obs_ref_box, params={"skill": k}, noise=noise(0.01)),
        "ref_clock": O(func=obs_ref_clock, params={"skill": k}),
        **robot_obs(noise),
    }


def robot_obs(noise) -> dict:
    O = ObservationTermCfg
    return {
        "base_lin_vel": O(func=vel_mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}, noise=noise(0.3)),
        "base_ang_vel": O(func=vel_mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}, noise=noise(0.2)),
        "projected_gravity": O(func=vel_mdp.projected_gravity, noise=noise(0.05)),
        "joint_pos": O(func=vel_mdp.joint_pos_rel, noise=noise(0.01)),
        "joint_vel": O(func=vel_mdp.joint_vel_rel, noise=noise(1.0)),
        "actions": O(func=obs_last_joint_action),
        "box": O(func=obs_box, noise=noise(0.01)),
        "palms": O(func=obs_palms, noise=noise(0.005)),
    }


def env_cfg(skill: str = "fetch", num_envs: int = 4096, play: bool = False, skill_paths=None,
            plans: str | None = None, robust: bool | None = None,
            stress: dict | None = None) -> ManagerBasedRlEnvCfg:
    """The env of one skill ("fetch", "carry", "place"), or of the whole job
    driven by the pointer over the frozen skills ("full", skill_paths).
    robust: the robust recipe's training events and touchdown penalty (default: G1_ROBUST).
    stress: in play, perturbations at these multiples of their ℓ = 1 sizes,
    {"jolt": ..., "push": ..., "kick": ...} (stress.py)."""
    robust = sc.ROBUST if robust is None else robust
    hand_box = ContactSensorCfg(
        name="hand_box",
        primary=ContactMatch(mode="subtree", pattern=r"^(left_wrist_roll_rubber_hand|right_wrist_roll_rubber_hand)$",
                             entity="robot"),
        secondary=ContactMatch(mode="body", pattern="box", entity="box"),
        fields=("found", "force"), reduce="netforce", num_slots=1)
    box_cart = ContactSensorCfg(
        name="box_cart", primary=ContactMatch(mode="body", pattern="box", entity="box"),
        secondary=ContactMatch(mode="body", pattern="cart", entity="cart"),
        fields=("found",), reduce="netforce", num_slots=1)
    robot_props = [ContactSensorCfg(
        name=f"robot_{p}", primary=ContactMatch(mode="body", pattern=p, entity=p),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found",), reduce="netforce", num_slots=1) for p in ("cart", "shelf")]
    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="netforce", num_slots=1)

    noise = (lambda lo: Unoise(n_min=-lo, n_max=lo)) if not play else (lambda lo: None)
    clean = lambda lo: None  # noqa: E731
    observations = {}
    if skill == "full":
        import hier
        actor = {**robot_obs(noise),
                 "goals": ObservationTermCfg(func=obs_goals, noise=noise(0.01)),
                 "clocks": ObservationTermCfg(func=obs_clocks)}
        critic = {**robot_obs(clean), "goals": ObservationTermCfg(func=obs_goals),
                  "clocks": ObservationTermCfg(func=obs_clocks),
                  "phase": ObservationTermCfg(func=obs_phase),
                  "box_mass": ObservationTermCfg(func=obs_box_mass)}
        for k, s in enumerate(SKILLS):     # what each frozen skill sees (as in its training, noise included)
            observations[f"skill_{s}"] = ObservationGroupCfg(skill_obs(k, noise), enable_corruption=not play)
        joint = PlanJointActionCfg(entity_name="robot", actuator_names=(".*",), scale=sc.action_scale())
        actions = {"skill": hier.SkillActionCfg(skill_paths=tuple(skill_paths), joint=joint,
                                                hold_steps=POINTER_EVERY)}
    else:
        k = SKILLS.index(skill)
        actor = skill_obs(k, noise)
        critic = {**skill_obs(k, clean),
                  "body_err": ObservationTermCfg(func=obs_body_err),
                  "foot_contact": ObservationTermCfg(func=vel_mdp.foot_contact,
                                                     params={"sensor_name": "feet_ground_contact"}),
                  "box_mass": ObservationTermCfg(func=obs_box_mass),
                  "phase": ObservationTermCfg(func=obs_phase)}
        actions = {"joint_pos": PlanJointActionCfg(entity_name="robot", actuator_names=(".*",),
                                                   scale=sc.action_scale())}
    observations["actor"] = ObservationGroupCfg(actor, enable_corruption=not play)
    observations["critic"] = ObservationGroupCfg(critic, enable_corruption=False)

    events = {}
    if not play:
        foot_geoms = tuple(f"{s}_foot{i}_collision" for s in ("left", "right") for i in range(1, 8))
        events = {
            "foot_friction": EventTermCfg(mode="startup", func=dr.geom_friction,
                                          params={"asset_cfg": SceneEntityCfg("robot", geom_names=foot_geoms),
                                                  "operation": "abs", "ranges": (0.4, 1.2), "shared_random": True}),
            "box_mass": EventTermCfg(mode="startup", func=dr.pseudo_inertia,
                                     params={"asset_cfg": SceneEntityCfg("box", body_names=("box",)),
                                             "alpha_range": (-0.2027, 0.1438)}),
            "encoder_bias": EventTermCfg(mode="startup", func=dr.encoder_bias,
                                         params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)}),
            "push": EventTermCfg(mode="interval", func=ev_push, interval_range_s=(1.0, 3.0),
                                 params={"vmax": PUSH_V}),
        }
        if robust:
            events["kick"] = EventTermCfg(mode="interval", func=ev_kick, interval_range_s=(1.0, 3.0),
                                          params={"wmax": KICK[0], "vmax": KICK[1]})
            events["touchdown"] = EventTermCfg(mode="step", func=ev_touchdown,
                                               params={"p": JOLT_P, "wv": JOLT_V, "q": JOLT_Q})
    elif stress:
        m = lambda k: float(stress.get(k, 0.0))  # noqa: E731
        events["touchdown"] = EventTermCfg(mode="step", func=ev_touchdown,
                                           params={"p": JOLT_P if m("jolt") > 0 else 0.0,
                                                   "wv": tuple(m("jolt") * w for w in JOLT_V), "q": m("jolt") * JOLT_Q})
        if m("push") > 0:
            events["push"] = EventTermCfg(mode="interval", func=ev_push, interval_range_s=(1.0, 3.0),
                                          params={"vmax": m("push") * PUSH_V})
        if m("kick") > 0:
            events["kick"] = EventTermCfg(mode="interval", func=ev_kick, interval_range_s=(1.0, 3.0),
                                          params={"wmax": m("kick") * KICK[0], "vmax": m("kick") * KICK[1]})
    # hand-off banks hold layout indices: the evaluation layouts have their own
    handoff = {"carry": "fetch_end", "place": "carry_end"}.get(skill)
    if handoff:
        handoff += ("_eval" if plans == PLANS_EVAL else "") + ".pt"
    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "success": TerminationTermCfg(func=t_success),
        "tracking": TerminationTermCfg(func=t_tracking),
        "fell": TerminationTermCfg(func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}),
        "box_lost": TerminationTermCfg(func=t_box_lost),
    }
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"),
                       entities={"robot": sc.robot_cfg(), **sc.prop_cfgs()},
                       sensors=(hand_box, box_cart, *robot_props, feet_ground),
                       num_envs=num_envs, env_spacing=8.0),
        observations=observations,
        actions=actions,
        commands={"task": TaskCommandCfg(
            skill=skill, plans=plans or PLANS,
            extra_hold=ROBUST_HOLD if robust and not play and skill != "full" else 0.0,
            handoff=os.path.join(sc.out_dir("banks"), handoff) if handoff else None)},
        events=events,
        rewards=_rewards(skill)
        | ({"impact": RewardTermCfg(func=r_impact, weight=IMPACT)} if robust and skill != "full" else {}),
        terminations=terminations,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot",
                            body_name="torso_link", distance=4.0, elevation=-20.0, azimuth=120.0),
        sim=SimulationCfg(nconmax=None, njmax=400, contact_sensor_maxmatch=64,
                          mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=50)),
        decimation=4,
        episode_length_s=EPISODE_S[skill] + (ROBUST_HOLD if robust and not play and skill != "full" else 0.0),
    )


def ppo_cfg(skill: str = "fetch", iterations: int = 3000) -> RslRlOnPolicyRunnerCfg:
    if skill == "full":
        # the pointer starts from the hand-written switch (train.imitate_rule): little
        # exploration noise on its scores and no entropy bonus (with 0.01 the noise grew
        # from 0.5 to 5 and every choice became random)
        actor = RslRlModelCfg(hidden_dims=(256, 128), activation="elu", obs_normalization=True,
                              class_name="hier:PointerModel",
                              distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3,
                                                "std_type": "scalar"})
        algo = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
                                    entropy_coef=0.0, num_learning_epochs=5, num_mini_batches=4,
                                    learning_rate=3e-4, schedule="adaptive", gamma=0.99, lam=0.95,
                                    desired_kl=0.01, max_grad_norm=1.0)
    else:
        # the correction starts small: the first policy plays the MPC's plan back
        actor = RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                              class_name="hier:PlanModel",
                              distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3,
                                                "std_type": "scalar"})
        algo = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
                                    entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
                                    learning_rate=1e-3, schedule="adaptive", gamma=0.99, lam=0.95,
                                    desired_kl=0.01, max_grad_norm=1.0)
    return RslRlOnPolicyRunnerCfg(
        actor=actor,
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=algo, num_steps_per_env=24, max_iterations=iterations, save_interval=100,
        experiment_name=f"g1_box_{skill}", logger="tensorboard", upload_model=False,
    )
