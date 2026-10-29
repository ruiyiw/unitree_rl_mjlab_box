"""gait.py — footsteps along a path of pelvis poses, and their support schedule.

A walk goes through via poses (x, y, heading): every step moves the pelvis an
equal share of the way to the next via pose, within a step's limits
(forward, backward, sideways, turning), and alternates feet, turning feet
leading.  The last step brings the trailing foot beside the other.  Each
step is the walk's swing and double support; the walk is padded with
standing double support before and after.  The footsteps are only nominal:
the MPC (mpc.py) moves them.

The walk is the human one: human_gait.json, from people carrying a box (OmniRetarget's
G1 box-carrying clips; README "The human walk"): 0.4 s swing + 0.1 s double support,
0.21 m forward steps 0.12 m apart (the MPC may bring them to 0.11 m; people: 0.09 m,
and the G1's feet are 0.07 m wide), the pelvis risen and bobbing as theirs, heel
strike and the swing foot's pitch, the ZMP rolling heel to toe.  Only forward steps
walk like this; backing up and turning on the spot stay flat-footed at standing width.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np

import scene as sc
from mpc import TICK, Footstep, Gait, G, H, rot

STANCE_W = 0.237                 # m, between the foot sites (the G1 standing)
FOOT_X = 0.039                   # m, foot sites ahead of the pelvis when standing
HEEL_X, TOE_X = -0.094, 0.092    # m, the sole's back and front edges from the foot site
ZMP_HEEL, ZMP_TOE = -0.025, 0.035   # m, the ZMP target's roll along a forward step's sole
RAMP = 0.4                       # s, the pelvis rises into the walk and settles out of it


@dataclass
class Style:
    """How the robot walks (module docstring)."""
    name: str
    t_ss: float
    t_ds: float
    step: tuple                  # forward, backward, sideways (m), turning (rad) per step
    walk_w: float                # m between the feet while walking forward
    reach_y: tuple
    human: dict                  # human_gait.json

    def __post_init__(self):
        h = self.human
        ph = np.array(h["stride_phase"])
        ev = h["events"]
        st = h["stride"]
        s = np.linspace(0.0, 1.0, 101)
        sw = ev[3] + s * (1.0 - ev[3])             # the swing: lift-off to touchdown
        self.sw_s = s
        self.sw_pitch = np.interp(sw, ph, st["pitch"])
        floor = float(np.median(np.interp(np.linspace(ev[1], ev[2], 20), ph, st["foot_z"])))
        self.sw_z = np.interp(sw, ph, st["foot_z"]) - floor
        self.td_pitch = 0.5 * (st["pitch"][0] + st["pitch"][-1])     # deg, toe up at heel strike
        # the pelvis per step (touchdown at 0), as a periodic Fourier series
        z = np.array(h["step_pelvis_z_m"])
        z = z - np.linspace(z[0], z[-1], len(z))
        z = z - z.mean()
        x = np.linspace(0.0, 1.0, len(z))[:-1]
        self.bob_c = [(2 * np.mean(z[:-1] * np.cos(2 * np.pi * k * x)), 2 * np.mean(z[:-1] * np.sin(2 * np.pi * k * x)))
                      for k in (1, 2, 3)]
        self.rise = float(h.get("planner", {}).get("pelvis_rise_m", 0.0))

    def bob(self, phi):
        """The pelvis's height around its walking mean at step phase phi."""
        return sum(a * np.cos(2 * np.pi * k * phi) + b * np.sin(2 * np.pi * k * phi)
                   for k, (a, b) in zip((1, 2, 3), self.bob_c))


def load_style():
    h = json.load(open(os.path.join(sc.HERE, "human_gait.json")))
    return Style("human", 0.4, 0.1, (round(h["step_length_m"], 2), 0.12, 0.08, 0.30), 0.12, (0.11, 0.32), h)


STYLE = load_style()
STEP_FWD, STEP_BACK, STEP_SIDE, STEP_TURN = STYLE.step


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def feet_at(pose, width=STANCE_W):
    """Foot sites (left, right) at a pelvis pose (x, y, yaw), `width` apart."""
    x, y, th = pose
    R = rot(th)
    p = np.array([x, y])
    return (p + R @ np.array([FOOT_X, width / 2]), p + R @ np.array([FOOT_X, -width / 2]))


def n_steps(a, b):
    """Steps to move the pelvis from pose a to pose b."""
    d = rot(-a[2]) @ (np.array(b[:2]) - np.array(a[:2]))
    fwd = d[0] / STEP_FWD if d[0] >= 0 else -d[0] / STEP_BACK
    return max(1, math.ceil(max(fwd, abs(d[1]) / STEP_SIDE, abs(wrap(b[2] - a[2])) / STEP_TURN) - 1e-9))


def interp_pose(a, b, s):
    return (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + wrap(b[2] - a[2]) * s)


def step_poses(vias):
    """Pelvis poses after each step through the via poses (the first is the start)."""
    out = []
    for a, b in zip(vias[:-1], vias[1:]):
        n = n_steps(a, b)
        out += [interp_pose(a, b, (k + 1) / n) for k in range(n)]
    return out


def forward(a, b):
    """The pelvis walks forward from pose a to pose b (not backing up, sidestepping
    or turning on the spot)."""
    d = rot(-a[2]) @ (np.array(b[:2]) - np.array(a[:2]))
    return d[0] > 0.5 * STEP_FWD and abs(d[1]) < d[0] and abs(wrap(b[2] - a[2])) < STEP_TURN


def make_gait(start, vias, t_pre=0.4, t_post=0.6, style=None):
    """The gait from standing at pelvis pose `start` through `vias` (the last
    is where it stops, feet together).  Returns the Gait, the start tick of
    the walk, and per tick the reference CoM position and velocity (T, 2)."""
    style = style or STYLE
    poses = step_poses([start] + list(vias))
    fl, fr = feet_at(start)
    # the first swing foot: the side the walk turns or moves toward
    d0 = rot(-start[2]) @ (np.array(poses[0][:2]) - np.array(start[:2]))
    turn = wrap(poses[0][2] - start[2])
    side = 1 if (turn > 1e-3 or (abs(turn) <= 1e-3 and d0[1] >= 0)) else -1
    y0 = start[2]
    first = Footstep(side, (fl if side > 0 else fr).copy(), y0, 0, True)
    stance = Footstep(-side, (fr if side > 0 else fl).copy(), y0, 0, True)
    steps = [first, stance]
    k_pre = int(round(t_pre / TICK))
    k_ss, k_ds = int(round(style.t_ss / TICK)), int(round(style.t_ds / TICK))
    support = [(0, 1, 0.5 + 0.5 * k / max(k_pre - 1, 1)) for k in range(k_pre)]
    lift = {}
    s = side
    targets = []
    # a forward step between two forward steps lands at the walking width; the
    # walk's first and last, and every other step, at standing width
    fwd = [forward(a, b) for a, b in zip([start] + poses[:-1], poses)]
    for k, q in enumerate(poses):
        narrow = fwd[k] and k + 1 < len(poses) and fwd[k + 1]
        L, R = feet_at(q, style.walk_w if narrow else STANCE_W)
        targets.append((s, L if s > 0 else R, q[2], fwd[k]))
        s = -s
    # the trailing foot beside the other at the last pose (a closing step, not a stride)
    L, R = feet_at(poses[-1])
    targets.append((s, L if s > 0 else R, poses[-1][2], False))
    walk = {}
    for sd, xy, yaw, fw in targets:
        i = len(steps)
        lift[i] = len(support)
        support += [(i - 1,)] * k_ss
        steps.append(Footstep(sd, xy.copy(), yaw, len(support)))
        support += [(i - 1, i, (k + 1) / k_ds) for k in range(k_ds)]
        walk[i] = fw
    n = len(steps)
    k_post = int(round(t_post / TICK))
    support += [(n - 2, n - 1, 0.5)] * k_post
    gait = Gait(steps, support, lift, reach_y=style.reach_y, fwd=walk)
    gait.style = style
    gait.k_ds = k_ds
    gait.eta = np.array([eta_at(gait, k * TICK) for k in range(len(support))])
    gait.zmp_off = zmp_offsets(gait, k_ss)
    return gait, k_pre, ref_com(gait)


def _runs(gait):
    """The runs of consecutive forward steps, each a list of step windows (s):
    from the previous foot's touchdown to this one's."""
    runs, cur = [], []
    for i in range(2, len(gait.steps)):
        if gait.fwd.get(i):
            cur.append((gait.steps[i - 1].land * TICK, gait.steps[i].land * TICK))
        elif cur:
            runs.append(cur)
            cur = []
    return runs + ([cur] if cur else [])


def _ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (10 - 15 * u + 6 * u * u)


def pelvis_dz(gait, t, d=0):
    """The human walk's pelvis height above the standing one at time t (d=2: its
    second derivative): the rise and the bob within each forward step, eased in
    over RAMP s at a run of forward steps' start and out at its end."""
    st = getattr(gait, "style", None)
    if st is None:
        return 0.0
    if d == 2:
        e = 0.01
        return (pelvis_dz(gait, t + e) - 2 * pelvis_dz(gait, t) + pelvis_dz(gait, t - e)) / (e * e)
    out = 0.0
    for run in _runs(gait):
        t0, t1 = run[0][0], run[-1][1]
        if not t0 <= t <= t1:
            continue
        env = _ease((t - t0) / RAMP) * _ease((t1 - t) / RAMP)
        bob = next((st.bob((t - a) / (b - a)) for a, b in run if a <= t < b), 0.0)
        out += env * (st.rise + bob)
    return out


def eta_at(gait, t):
    """The ZMP's factor h/(g + ḧ) for the human walk's centre-of-mass height."""
    return (H + pelvis_dz(gait, t)) / (G + pelvis_dz(gait, t, 2))


def zmp_offsets(gait, k_ss):
    """Per tick, the ZMP target along the support feet: on a forward step it rolls
    from heel to toe over the stance sole in single support, and from the trailing
    toe to the leading heel in double support."""
    out = np.zeros((len(gait.support), 2))
    for t, sup in enumerate(gait.support):
        if len(sup) == 1:
            i = sup[0] + 1                       # the step swinging over this stance
            if gait.fwd.get(i):
                k = t - gait.lift[i]
                u = (k + 0.5) / k_ss
                out[t] = ZMP_HEEL + (ZMP_TOE - ZMP_HEEL) * u, 0.0
        elif gait.fwd.get(sup[1]) and sup[1] >= 2 and t < gait.steps[sup[1]].land + gait.k_ds:
            out[t] = ZMP_TOE, ZMP_HEEL
    return out


def ref_com(gait: Gait):
    """Reference CoM per tick: the feet's midpoint at each landing, linear in between."""
    T = len(gait.support)
    keys = [(0, 0.5 * (gait.steps[0].xy + gait.steps[1].xy))]
    for i in range(2, len(gait.steps)):
        keys.append((gait.steps[i].land, 0.5 * (gait.steps[i].xy + gait.steps[i - 1].xy)))
    keys.append((T - 1, keys[-1][1]))
    ts = np.array([k for k, _ in keys], float)
    ps = np.array([p for _, p in keys])
    t = np.arange(T, dtype=float)
    p = np.stack([np.interp(t, ts, ps[:, 0]), np.interp(t, ts, ps[:, 1])], -1)
    v = np.gradient(p, TICK, axis=0)
    return p, v


def stand_gait(pose, seconds):
    """Standing still for `seconds`: double support, no steps."""
    fl, fr = feet_at(pose)
    steps = [Footstep(1, fl.copy(), pose[2], 0, True), Footstep(-1, fr.copy(), pose[2], 0, True)]
    T = int(round(seconds / TICK)) + 1
    gait = Gait(steps, [(0, 1, 0.5)] * T, {})
    return gait, 0, ref_com(gait)
