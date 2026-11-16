#!/usr/bin/env python3
"""plans.py — the reference motion of walk(N), from the humanoid repository's planner.

For a distance N the planner (humanoid/policies: gait.py footsteps on the human-walk
template, mpc.py linear-inverted-pendulum MPC with free footsteps, plan.py whole-body IK
on the 23-DoF G1) produces, at 50 Hz: stand 0.4 s, walk N m straight ahead in
ceil(N / 0.21) equal steps, close the feet, stand.  N = 0 is a standing plan.  The plan's
frame is the start: x straight ahead, origin on the floor under the pelvis.

A bank is a set of such plans, padded to one length, that the env samples from:

  python -m parametrization.walk.plans train                     # parametrization/walk/runs/banks/train.pt  (N in [0, 5] m)
  python -m parametrization.walk.plans fixed --n 2.0             # parametrization/walk/runs/banks/fixed_2.00.pt (one plan)
  python -m parametrization.walk.plans eval                      # parametrization/walk/runs/banks/eval.pt (held-out, N = 0, 8, 20)
  python -m parametrization.walk.plans select                    # parametrization/walk/runs/banks/select.pt (to choose checkpoints on)
  python -m parametrization.walk.plans train20 | eval20 | select20   # the same for training to 20 m
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("REKIN_GPU", "0")

import numpy as np

from . import paths

FPS = 50
T_PRE = 0.4                        # s standing before the first step (the planner's default)
T_POST = 1.6                       # s standing after the closing step, in the plan
STRIDE_S = 1.0                     # s: two steps (0.4 s swing + 0.1 s double support each)
KEYS = ("qpos", "qcmd", "qvel", "body", "body_lin", "body_ang", "contact", "phase")

# evaluation distances (m): held out of the training bank; below one step; standing; beyond training
EVAL_IN_RANGE = (0.75, 1.4, 2.5, 3.5, 4.2)       # 4, 7, 12, 17, 20 steps: plans ending on either foot
EVAL_SHORT = (0.1, 0.2, 0.3)
EVAL_EXTRAPOLATE = (8.0, 20.0)
SELECT = (0.0, 0.15, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)     # checkpoint selection: not the evaluation distances
TRAIN_RANGE = (0.0, 5.0)
# the long-range variation: training to 20 m; evaluation inside that range and beyond it
FAR_RANGE = (5.0, 20.0)
EVAL_FAR = (8.0, 12.0, 16.5, 20.0)                     # 40, 59, 80 and 97 steps
EVAL_BEYOND = (25.0, 40.0)                             # 121 and 192 steps
SELECT_FAR = (7.0, 10.0, 14.0, 18.0)


def _planner():
    paths.use_humanoid()
    import plan as pl
    return pl, pl.Planner(check_every=0)


def walk_plan(P, pl, n_m: float) -> dict:
    """One plan: stand, walk n_m metres straight ahead, stop with the feet side by side."""
    keys = [pl.Key(0.0, None, "rest")]
    if n_m < 1e-6:
        p = P.plan((0.0, 0.0, 0.0), [], keys, stand=True, seconds=T_PRE + T_POST)
        t0 = t1 = 0.0
    else:
        p = P.plan((0.0, 0.0, 0.0), [(float(n_m), 0.0, 0.0)], keys, t_pre=T_PRE, t_post=T_POST)
        t0 = T_PRE
        t1 = (len(p["qpos"]) - 1) / FPS - T_POST          # the closing step has landed
    nf = len(p["qpos"])
    t = np.arange(nf) / FPS
    on = ((t >= t0) & (t < t1)).astype(np.float32)
    ang = 2 * math.pi * (t - t0) / STRIDE_S
    p["phase"] = np.stack([np.sin(ang) * on, np.cos(ang) * on, on], -1).astype(np.float32)
    p["contact"] = p["contact"].astype(np.float32)
    p["n_m"] = float(n_m)
    p["walk_frames"] = (int(round(t0 * FPS)), int(round(t1 * FPS)))
    return p


def _worker(ns):
    pl, P = _planner()
    out = []
    for n in ns:
        p = walk_plan(P, pl, n)
        out.append({k: p[k] for k in KEYS + ("n_m", "walk_frames", "ik_err")})
    st = P.stats
    return out, (st.n, st.secs, dict(st.statuses))


def make_bank(ns, out, workers=32) -> dict:
    """Plan every distance in ns (parallel), pad to one length, save."""
    import multiprocessing as mp

    import torch
    ns = [float(n) for n in ns]
    order = sorted(range(len(ns)), key=lambda i: -ns[i])            # longest first, dealt round-robin
    workers = max(1, min(workers, len(ns)))
    chunks = [[ns[i] for i in order[w::workers]] for w in range(workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(_worker, chunks)
    plans = sorted((p for ps, _ in res for p in ps), key=lambda p: p["n_m"])
    paths.use_humanoid()
    import plan as pl
    lens = [len(p["qpos"]) for p in plans]
    T = max(lens)
    bank = {"joint_names": pl.Robot().joints, "body_names": list(pl.TRACKED), "fps": FPS,
            "n_m": torch.tensor([p["n_m"] for p in plans], dtype=torch.float32),
            "len": torch.tensor(lens),
            "walk_frames": torch.tensor([p["walk_frames"] for p in plans]),
            "ik_err": torch.tensor([float(p["ik_err"].max()) for p in plans])}
    for k in KEYS:
        bank[k] = torch.from_numpy(np.stack(
            [np.concatenate([p[k], np.repeat(p[k][-1:], T - len(p[k]), axis=0)], 0) for p in plans]))
    statuses = {}
    for _, (_, _, s) in res:
        for k, v in s.items():
            statuses[k] = statuses.get(k, 0) + v
    end_x = bank["qpos"][torch.arange(len(plans)), bank["len"] - 1, 0]
    bank["stats"] = {"plans": len(plans), "frames_max": T, "wall_s": round(time.time() - t0, 1),
                     "qps": sum(r[1][0] for r in res), "qp_statuses": statuses,
                     "ik_err_max_m": round(float(bank["ik_err"].max()), 4),
                     "pelvis_end_minus_n_max_m": round(float((end_x - bank["n_m"]).abs().max()), 4)}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(bank, out)
    return bank["stats"]


def train_distances(n_short=24, n_long=135, seed=0, n_far=0):
    """Standing, n_short distances below 0.5 m, n_long in 0.5-5 m, n_far in 5-20 m: stratified,
    jittered, and never within 2 cm of an evaluation distance.  The first 1 + n_short + n_long
    are the same distances whatever n_far is."""
    rng = np.random.default_rng(seed)
    held = np.array(EVAL_IN_RANGE + EVAL_SHORT + EVAL_FAR)

    def strat(lo, hi, n):
        edges = np.linspace(lo, hi, n + 1)
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            for _ in range(100):
                x = rng.uniform(a, b)
                if np.abs(held - x).min() > 0.02:
                    break
            out.append(round(float(x), 3))
        return out

    return ([0.0] + strat(0.02, 0.5, n_short) + strat(0.5, TRAIN_RANGE[1], n_long)
            + (strat(*FAR_RANGE, n_far) if n_far else []))


def bank_path(name):
    return os.path.join(paths.BANKS, f"{name}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=("train", "fixed", "eval", "select", "one", "train20", "eval20", "select20", "select20s"))
    ap.add_argument("--n", type=float, default=2.0, help="fixed / one: the distance (m)")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    if a.which == "train":
        st = make_bank(train_distances(), bank_path("train"), a.workers)
    elif a.which == "train20":                # the 160 distances of "train" and 100 more in 5-20 m
        st = make_bank(train_distances(n_far=100), bank_path("train20"), a.workers)
    elif a.which == "eval20":
        st = make_bank((0.0,) + EVAL_SHORT + EVAL_IN_RANGE + EVAL_FAR + EVAL_BEYOND, bank_path("eval20"), a.workers)
    elif a.which == "select20":
        st = make_bank(SELECT + SELECT_FAR, bank_path("select20"), a.workers)
    elif a.which == "select20s":              # ... and a half-length single step, which 0.15 m does not cover
        st = make_bank(SELECT + (0.08,) + SELECT_FAR, bank_path("select20s"), a.workers)
    elif a.which == "fixed":
        st = make_bank([a.n], bank_path(f"fixed_{a.n:.2f}"), 1)
    elif a.which == "select":
        st = make_bank(SELECT, bank_path("select"), a.workers)
    elif a.which == "eval":
        st = make_bank((0.0,) + EVAL_SHORT + EVAL_IN_RANGE + EVAL_EXTRAPOLATE, bank_path("eval"), a.workers)
    else:
        t0 = time.time()
        pl, P = _planner()
        p = walk_plan(P, pl, a.n)
        st = {"n_m": a.n, "frames": len(p["qpos"]), "seconds": round(time.time() - t0, 2),
              "pelvis_end_x": round(float(p["qpos"][-1, 0]), 4), "ik_err_max": round(float(p["ik_err"].max()), 4),
              "qps": P.stats.n, "qp_statuses": dict(P.stats.statuses), "steps": int(len(p["steps"]) - 2)}
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
