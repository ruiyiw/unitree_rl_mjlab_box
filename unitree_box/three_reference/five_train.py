"""Train/evaluate a single five-stage policy on its visible GPU.

CUDA_VISIBLE_DEVICES=0 python policies/five_train.py train --skill go-to
Each process owns its actor, critic, normalization, checkpoints and evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

import motion
import task as tk
from five_plans import CONFIG, OUT, SKILLS, POLICIES
from five_task import env_cfg, navigation_scores
from five_provenance import SOURCES, fingerprints
from five_proprio import CONTRACT, OBSERVATION_DIM, validate_checkpoint
from five_rewards import RECIPE

EXPLORATION_STD = .05


def ppo_cfg(skill, iterations=1000):
    cfg = motion.ppo_cfg(str(OUT/skill),iterations)
    cfg.algorithm.entropy_coef = 0.
    cfg.algorithm.learning_rate = 3e-4
    cfg.algorithm.gamma = .995
    cfg.actor.distribution_cfg.update(init_std=EXPLORATION_STD,learn_std=False,
                                     std_range=(EXPLORATION_STD,EXPLORATION_STD))
    return cfg


def warm_start(runner, checkpoint):
    runner.load(str(checkpoint),load_cfg={"actor":True},map_location="cuda:0")
    with torch.no_grad():
        runner.alg._raw_actor.distribution.std_param.fill_(EXPLORATION_STD)
    runner.current_learning_iteration = 0
    runner.env.unwrapped.common_step_counter = 0


def selection_score(r):
    return (r.get("worst_route_completed_and_stable",r["completed_and_stable"]),
            r["completed_and_stable"],r["success"],r["stable_at_end"],
            -sum(r["failures"].values()),
            r.get("mean_survival_seconds",0.),
            -r.get("route_tracking",{}).get("mean_final_endpoint_error_m",0.),
            -r.get("route_tracking",{}).get("p95_max_cross_track_m",0.))


class ProprioRunner(MjlabOnPolicyRunner):
    def save(self, path, infos=None):
        return super().save(path, {**(infos or {}), "observation_contract": CONTRACT,
                                   "reward_recipe":RECIPE})

    def load(self, path, *args, **kwargs):
        validate_checkpoint(path)
        return super().load(path, *args, **kwargs)


def export_actor(folder, checkpoint):
    data = validate_checkpoint(checkpoint)
    path = Path(folder)/"actors"/"proprio-policy.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor_state_dict": data["actor_state_dict"], "iter": data["iter"],
                "observation_contract": CONTRACT,
                "reward_recipe":data.get("infos",{}).get("reward_recipe"),
                "evaluation_recipe":RECIPE,"source": str(checkpoint)}, path)
    return str(path)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2)+"\n")
    temp.replace(path)


class Evaluator:
    def __init__(self, skill, envs, seconds=60.):
        if skill == "navigation" and envs < 2:
            raise ValueError("Navigation evaluation needs at least two environments to cover both routes")
        cfg = env_cfg(skill, envs, play=True)
        cfg.terminations.pop("success")
        cfg.episode_length_s = max(seconds, cfg.episode_length_s) + .1
        self.seconds = cfg.episode_length_s - .1
        cfg.seed = 10001
        self.env = ManagerBasedRlEnv(cfg, device="cuda:0")
        self.term = tk.task(self.env)
        self.term.curriculum = False
        self.term.force_start = tk.BEGIN
        self.skill = skill
        self.vec = RslRlVecEnvWrapper(self.env, clip_actions=None)
        self.runner = ProprioRunner(self.vec, asdict(ppo_cfg(skill)), None, "cuda:0")

    def run(self, checkpoint):
        torch.manual_seed(10001)
        self.runner.load(str(checkpoint), load_cfg={"actor": True}, map_location="cuda:0")
        policy = self.runner.get_inference_policy(device="cuda:0")
        obs, _ = self.vec.reset()
        routes = self.term.plan_ids.clone()
        n = self.env.num_envs
        live = torch.ones(n, dtype=torch.bool, device="cuda:0")
        success = torch.zeros_like(live)
        max_drift = torch.zeros(n, device="cuda:0")
        drift_sum = torch.zeros_like(max_drift)
        drift_samples = torch.zeros_like(max_drift)
        endpoint_error = torch.zeros_like(max_drift)
        lifetime = torch.zeros_like(max_drift)
        failures = {k: torch.zeros_like(live) for k in self.env.termination_manager.active_terms if k != "time_out"}
        # Environment metrics survive across checkpoint evaluations. Keep them
        # ordinary tensors so the next reset can update them outside this scope.
        with torch.no_grad():
            for _ in range(round(self.seconds/self.env.step_dt)):
                lifetime += live.float()*self.env.step_dt
                if self.skill in ("navigation", "go-to", "return", "carry"):
                    scores = navigation_scores(self.env)
                    drift = scores["cross_track_m"]
                    max_drift = torch.maximum(max_drift, torch.where(live, drift, 0.))
                    drift_sum += torch.where(live, drift, 0.)
                    drift_samples += live.float()
                    endpoint_error = torch.where(live, scores["endpoint_error_m"], endpoint_error)
                obs, _, dones, _ = self.vec.step(policy(obs))
                success |= live & ~dones.bool() & self.term.success
                for k in failures:
                    failures[k] |= live & self.env.termination_manager.get_term(k).bool()
                live &= ~dones.bool()
                if not live.any():
                    break
        stable = live & self.term.standing()[0]
        report = dict(checkpoint=str(checkpoint), episodes=n, seconds=self.seconds,
                    success=float(success.float().mean()), stable_at_end=float(stable.float().mean()),
                    completed_and_stable=float((success & stable).float().mean()),
                    handoff_ready_at_end=float((live & self.term.ended() & self.term.controller_ready()).float().mean()),
                    reward_recipe=RECIPE,
                    mean_survival_seconds=float(lifetime.mean()),
                    failures={k:float(v.float().mean()) for k,v in failures.items()},
                    scope="Independent skill from reference initial state; not sequential handoff validation")
        if self.skill in ("navigation", "go-to", "return", "carry"):
            def route_metrics(mask):
                return dict(mean_cross_track_m=float((drift_sum[mask]/drift_samples[mask].clamp_min(1)).mean()),
                            mean_max_cross_track_m=float(max_drift[mask].mean()),
                            p95_max_cross_track_m=float(torch.quantile(max_drift[mask], .95)),
                            mean_final_endpoint_error_m=float(endpoint_error[mask].mean()))
            report["route_tracking"] = route_metrics(torch.ones_like(live))
        if self.skill == "navigation":
            report["routes"] = {name: dict(episodes=int((routes==i).sum()),
                success=float(success[routes==i].float().mean()),
                completed_and_stable=float((success & stable)[routes==i].float().mean()),
                **route_metrics(routes==i))
                for i,name in enumerate(("go-to", "return"))}
            report["worst_route_completed_and_stable"] = min(
                r["completed_and_stable"] for r in report["routes"].values())
        return report

    def close(self):
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "smoke", "pilot", "evaluate"))
    parser.add_argument("--skill", choices=(*SKILLS, "navigation"), required=True)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warm-start", type=Path, help="Load actor and normalization only; reset critic, optimizer and iteration")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval-envs", type=int, default=1024)
    args = parser.parse_args()
    if args.resume and args.warm_start:
        parser.error("Choose either --resume or --warm-start")
    if args.mode in ("train","pilot") and args.skill not in POLICIES:
        parser.error("go-to and return are trained together with --skill navigation")
    for checkpoint in (args.resume, args.warm_start, args.checkpoint):
        if checkpoint is not None:
            validate_checkpoint(checkpoint)
    if args.resume and (validate_checkpoint(args.resume).get("infos") or {}).get("reward_recipe") != RECIPE:
        parser.error("Reward recipe changed: use --warm-start to reset critic, optimizer and exploration")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This program must run with host GPU access.")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    folder = OUT/args.skill/"runs"/args.run_id
    folder.mkdir(parents=True, exist_ok=True)
    snapshot = folder/"inputs"
    snapshot.mkdir(exist_ok=True)
    for path in (CONFIG, OUT/args.skill/"banks"/"plan.pt", OUT/args.skill/"banks"/"validation.json",
                 *[Path(__file__).parent/name for name in SOURCES]):
        shutil.copy2(path, snapshot/path.name)
    status = dict(skill=args.skill, run_id=args.run_id, pid=os.getpid(),
                  gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                  started_at=datetime.now(timezone.utc).isoformat(),
                  **fingerprints(args.skill), warm_start=str(args.warm_start) if args.warm_start else None,
                  observation_contract=CONTRACT,
                  reward_recipe=RECIPE,exploration_std=EXPLORATION_STD,
                  versions={k:importlib.metadata.version(k) for k in ("mjlab","mujoco","torch","rekin","casadi")},
                  state="initializing")
    def update(state, **more):
        status.update(state=state, updated_at=datetime.now(timezone.utc).isoformat(), **more)
        write_json(folder/"status.json", status)
    update("initializing")
    env = None
    evaluator = None
    try:
        if args.mode != "evaluate":
            count = min(args.envs,32) if args.mode == "smoke" else args.envs
            cfg = env_cfg(args.skill,count)
            cfg.seed = args.seed
            env = ManagerBasedRlEnv(cfg,device="cuda:0")
            vec = RslRlVecEnvWrapper(env,clip_actions=None)
            obs,_ = vec.reset()
            if obs["actor"].shape[-1] != OBSERVATION_DIM:
                raise RuntimeError("Actor must have exactly 75 joint/IMU/controller-memory observations")
            for _ in range(8):
                obs, reward, _, _ = vec.step(torch.zeros((count,23),device="cuda:0"))
                if not torch.isfinite(reward).all() or not all(torch.isfinite(v).all() for v in obs.values()):
                    raise RuntimeError("Non-finite simulation state during preflight")
            agent = ppo_cfg(args.skill,args.iterations)
            agent.seed = args.seed
            agent.save_interval = 100
            runner = ProprioRunner(vec,asdict(agent),str(folder),"cuda:0")
            if args.resume:
                runner.load(str(args.resume),map_location="cuda:0")
            elif args.warm_start:
                warm_start(runner,args.warm_start)
            count_iterations = 2 if args.mode == "smoke" else args.iterations
            update("training",envs=count,iterations=count_iterations,
                   actor_obs=int(obs["actor"].shape[-1]),critic_obs=int(obs["critic"].shape[-1]))
            runner.learn(num_learning_iterations=count_iterations,init_at_random_ep_len=True)
            checkpoint = folder/f"model_{runner.current_learning_iteration}.pt"
            runner.save(str(checkpoint))
            env.close()
            env = None
            del runner,vec
            torch.cuda.empty_cache()
            if args.mode == "smoke":
                update("smoke_passed",checkpoint=str(checkpoint))
                print(f"SMOKE PASSED: {args.skill}",flush=True)
                return
            if args.mode == "pilot":
                update("evaluating_pilot",checkpoint=str(checkpoint))
                evaluator = Evaluator(args.skill,args.eval_envs,seconds=30.)
                report = evaluator.run(checkpoint)
                write_json(folder/"evaluation.json",report)
                evaluator.close();evaluator=None
                update("pilot_complete",checkpoint=str(checkpoint),evaluation=report)
                print(json.dumps(report,indent=2),flush=True)
                return
            update("selecting_checkpoint",last_checkpoint=str(checkpoint))
            candidates = sorted(folder.glob("model_*.pt"), key=lambda p:int(p.stem.split("_")[-1]))
            candidates = [p for p in candidates if int(p.stem.split("_")[-1]) >= args.iterations//2]
            # Evaluate the starting actor under exactly the same success gates.
            # Fine-tuning must earn selection over the baseline, including when
            # its older evaluation used a different training-completion gate.
            if args.warm_start:
                candidates.insert(0,args.warm_start.resolve())
            evaluator = Evaluator(args.skill,64,seconds=30.)
            selection = []
            for candidate in candidates:
                result = evaluator.run(candidate)
                selection.append(result)
                write_json(folder/"selection.json",selection)
                update("selecting_checkpoint", evaluated_checkpoints=len(selection), candidate_count=len(candidates))
                print(json.dumps(result),flush=True)
            selected = max(selection,key=selection_score)
            checkpoint = Path(selected["checkpoint"])
            evaluator.close()
            evaluator = None
            torch.cuda.empty_cache()
            update("evaluating",selected_checkpoint=str(checkpoint))
        else:
            if args.checkpoint is None:
                raise ValueError("--checkpoint is required for evaluation")
            checkpoint = args.checkpoint
            update("evaluating",selected_checkpoint=str(checkpoint))
        evaluator = Evaluator(args.skill,args.eval_envs,seconds=60.)
        report = evaluator.run(checkpoint)
        write_json(folder/"evaluation.json",report)
        evaluator.close()
        evaluator = None
        # Preserve experiment checkpoints, but do not replace the active actor
        # with an all-failing candidate. Positive success is not hardware approval.
        exportable = report.get("worst_route_completed_and_stable",report["completed_and_stable"]) > 0
        actor = export_actor(str(OUT/args.skill),str(checkpoint)) if args.mode=="train" and exportable else None
        update("complete",evaluation=report,actor=actor,exported=actor is not None,
               export_note="Successful simulation trials; hardware validation still required" if actor else
                           "No active actor replaced; evaluation-only run or no stable completion on every route")
        print(json.dumps(report,indent=2),flush=True)
    except BaseException:
        update("failed",error=traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()
        if evaluator is not None:
            evaluator.close()


if __name__ == "__main__":
    main()
