#!/usr/bin/env python3
"""evaluate.py — walk(N) measured on the truth, per distance.

Every distance in a plan bank is run `repeats` times from standing with the policy's mean
actions.  One constructed controller is used for all N.  Two tiers:

  nominal  the nominal robot, clean sensors (the odometry still comes from encoders and IMU)
  dr       training's randomized physics, sensor noise, bias and delay, heading drift; no pushes

Per distance: stable-arrival rate (within 10 cm and 10 degrees of the goal, slower than
0.1 m/s, upright, both feet down, for 0.5 s, AND still so SETTLE s after the plan's end),
falls and lost posture (the torso or a limb far from the plan's height or tilt), the final
position error SETTLE s after the plan's end, the largest forward overshoot, the time from
invocation to stable arrival, and the odometry's error at the end.

  python -m parametrization.walk.evaluate --policy parametrization/walk/runs/goal/model_1999.pt --tier nominal
  python -m parametrization.walk.evaluate --policy parametrization/walk/runs/adapter/model_999.pt --adapter-base parametrization/walk/runs/base/model_999.pt
  python -m parametrization.walk.evaluate --zero                       # the plans played back, no policy
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from . import env as we  # noqa: E402
from . import models, paths, plans  # noqa: E402

SETTLE = 5.0                       # s after the plan's end before the final measurement


def evaluate(policy=None, bank="eval", tier="nominal", repeats=64, adapter_base=None, zero=False, seed=7,
             record=None, privileged=False, settle=SETTLE, estimator=None, sole_contacts=None, odometry_scale=1.0,
             raw=None, profile=None):
    """Returns {"per_distance": [...], "overall": {...}}.  record: env indices whose qpos to keep.
    The observation variant (history, estimator) is the one the policy was trained with (its
    run.json); estimator / sole_contacts override it, to try a policy on another estimator.
    odometry_scale: the estimator's calibrated distance scale (calibrate.py).
    raw: a path; every episode's numbers are saved there (.npz), for pooling across distances.
    profile: the randomization ranges of the "dr" tiers (env.PROFILES); default: the policy's own."""
    torch.manual_seed(seed)
    bank_path = plans.bank_path(bank)
    b = torch.load(bank_path, map_location="cpu", weights_only=False)
    C = len(b["n_m"])
    seconds = (int(b["len"].max()) - 1) / b["fps"] + settle + 2.0
    bound = 0.5
    if adapter_base:                                     # the correction's bound the adapter was trained with
        meta = os.path.join(os.path.dirname(policy), "run.json")
        if os.path.exists(meta):
            bound = json.load(open(meta)).get("adapter_bound") or bound
    v = paths.run_variant(policy)
    cfg = we.env_cfg(bank_path, num_envs=C * repeats, mode=tier, adapter_base=adapter_base, adapter_bound=bound,
                     episode_s=seconds, privileged_actor=privileged, history=v["history"],
                     estimator=estimator or v["estimator"],
                     sole_contacts=v["sole_contacts"] if sole_contacts is None else sole_contacts,
                     base_history=paths.run_variant(adapter_base)["history"], odometry_scale=odometry_scale,
                     profile=profile or v["profile"])
    cfg.seed = seed
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    c = we.cmd(env)
    n, dev = env.num_envs, env.device
    obs, _ = env.reset()
    adim = env.action_manager.total_action_dim
    if zero:
        act = lambda o: torch.zeros(n, adim, device=dev)  # noqa: E731
    else:
        hidden = models.ADAPTER_HIDDEN if adapter_base else models.HIDDEN
        act = models.frozen_actor(policy, obs["actor"].shape[-1], adim, dev, hidden)
    live = torch.ones(n, dtype=torch.bool, device=dev)
    z = lambda: torch.full((n,), float("nan"), device=dev)  # noqa: E731
    out = {k: z() for k in ("final_err", "final_heading", "overshoot", "t_arrive", "odo_err", "walked", "odo_along")}
    arrived = torch.zeros(n, dtype=torch.bool, device=dev)
    failed = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in ("fell", "tracking")}
    tm = env.termination_manager
    frames = {i: [] for i in (record or [])}
    snap = None
    for _ in range(int(round(seconds / env.step_dt)) - 1):
        for i in frames:
            if bool(live[i]):
                frames[i].append(env.sim.data.qpos[i].cpu().numpy().copy())
        _, _, d, herr = c.at_goal()
        snap = {"final_err": d.clone(), "final_heading": herr.clone(), "overshoot": c.overshoot.clone(),
                "odo_err": c.odo_err.clone(), "walked": c.pos_true[:, 0] - c.bank.start_xy[c.plan_ids, 0],
                "odo_along": c.odo.pos[:, 0] - c.pos_true[:, 0],
                "arrived": c.arrived.clone(), "t_arrive": c.t_arrived.clone()}
        obs, _, term, trunc, _ = env.step(act(obs["actor"]))
        dead = (term | trunc) & live                      # ended early: keep the last step's numbers
        if dead.any():
            for k in failed:
                failed[k] |= dead & tm.get_term(k).bool()
            for k in out:
                out[k] = torch.where(dead, snap[k], out[k])
            live &= ~dead
        done = live & (c.after_end >= settle)             # stood SETTLE s after the plan: measure
        if done.any():
            _, _, d, herr = c.at_goal()
            now = {"final_err": d, "final_heading": herr, "overshoot": c.overshoot, "odo_err": c.odo_err,
                   "walked": c.pos_true[:, 0] - c.bank.start_xy[c.plan_ids, 0], "t_arrive": c.t_arrived,
                   "odo_along": c.odo.pos[:, 0] - c.pos_true[:, 0]}
            for k in out:
                out[k] = torch.where(done, now[k], out[k])
            arrived |= done & c.arrived & (c.held > 0)   # arrived, and still passing the test now
            live &= ~done
        if not live.any():
            break
    plan = (torch.arange(n, device=dev) % C).cpu().numpy()
    g = lambda t: t.float().cpu().numpy()  # noqa: E731
    arr, fail = g(arrived), {k: g(v) for k, v in failed.items()}
    o = {k: g(v) for k, v in out.items()}
    if raw:
        np.savez(raw, n_m=np.asarray(b["n_m"], dtype=float)[plan], arrived=arr, fell=fail["fell"],
                 tracking=fail["tracking"], **o)
    rows = []
    for j in range(C):
        m = plan == j
        ok = m & (arr > 0)
        med = lambda x: round(float(np.nanmedian(x)), 4) if np.isfinite(x).any() else None  # noqa: E731
        rows.append({
            "n_m": round(float(b["n_m"][j]), 3), "episodes": int(m.sum()),
            "stable_arrival": round(float(arr[m].mean()), 4),
            "fell": round(float(fail["fell"][m].mean()), 4), "tracking_lost": round(float(fail["tracking"][m].mean()), 4),
            "ended_early": round(float(np.maximum(fail["fell"], fail["tracking"])[m].mean()), 4),   # either
            "final_err_m_median": med(o["final_err"][m]),
            "final_err_m_p95": round(float(np.nanpercentile(o["final_err"][m], 95)), 4),
            "walked_m_median": med(o["walked"][m]),
            "overshoot_m_max": round(float(np.nanmax(o["overshoot"][m])), 4),
            "heading_err_deg_median": round(float(np.degrees(np.nanmedian(o["final_heading"][m]))), 2),
            "time_to_arrival_s_median": med(o["t_arrive"][ok]) if ok.any() else None,
            "odometry_err_m_median": med(o["odo_err"][m]),
            "odometry_along_m_median": med(o["odo_along"][m]),      # signed: the estimate ahead of the truth
        })
    rep = {"policy": "plan played back (zero action)" if zero else policy, "adapter_base": adapter_base,
           "bank": bank, "tier": tier, "repeats": repeats, "settle_s": settle,
           "overall": {"stable_arrival": round(float(arr.mean()), 4),
                       "failed": round(float(np.maximum(fail["fell"], fail["tracking"]).mean()), 4)},
           "per_distance": rows}
    extra = {"frames": frames, "model": env.sim.mj_model, "plan_frame": c.frame.cpu().numpy(),
             "origins": env.scene.env_origins.cpu().numpy(), "dt": env.step_dt, "plans": C,
             "n_m": [float(x) for x in b["n_m"]]}
    env.close()
    return (rep, extra) if record else rep


def table(rep) -> str:
    h = f"{'N (m)':>6} {'arrive':>7} {'fail':>6} {'err med':>8} {'err p95':>8} {'walked':>7} {'overshoot':>9} {'t (s)':>6} {'odo err':>8}"
    lines = [h]
    for r in rep["per_distance"]:
        t = "-" if r["time_to_arrival_s_median"] is None else f"{r['time_to_arrival_s_median']:.1f}"
        lines.append(f"{r['n_m']:6.2f} {r['stable_arrival']:7.3f} {r['fell'] + r['tracking_lost']:6.3f} "
                     f"{r['final_err_m_median']:8.3f} {r['final_err_m_p95']:8.3f} {r['walked_m_median']:7.3f} "
                     f"{r['overshoot_m_max']:9.3f} {t:>6} {r['odometry_err_m_median']:8.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    ap.add_argument("--adapter-base", default=None)
    ap.add_argument("--zero", action="store_true")
    ap.add_argument("--bank", default="eval")
    ap.add_argument("--tier", default="nominal", choices=("nominal", "dr", "dr_h"))
    ap.add_argument("--estimator", default=None, choices=tuple(we.ESTIMATORS), help="override the policy's estimator")
    ap.add_argument("--sole-contacts", action="store_true", default=None)
    ap.add_argument("--odometry-scale", type=float, default=1.0, help="the estimator's calibrated scale")
    ap.add_argument("--repeats", type=int, default=64)
    ap.add_argument("--out", default=None)
    ap.add_argument("--privileged", action="store_true", help="the policy is a teacher (reads the critic's observation)")
    ap.add_argument("--settle", type=float, default=SETTLE, help="s standing after the plan's end before measuring")
    a = ap.parse_args()
    if not a.zero and not a.policy:
        ap.error("--policy or --zero")
    rep = evaluate(a.policy, a.bank, a.tier, a.repeats, a.adapter_base, a.zero, privileged=a.privileged,
                   settle=a.settle, estimator=a.estimator, sole_contacts=a.sole_contacts,
                   odometry_scale=a.odometry_scale)
    print(table(rep))
    print(json.dumps(rep["overall"]))
    out = a.out or os.path.join(os.path.dirname(a.policy) if a.policy else plans.paths.RUNS,
                                f"eval_{'zero_' if a.zero else ''}{a.bank}_{a.tier}.json")
    json.dump(rep, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
