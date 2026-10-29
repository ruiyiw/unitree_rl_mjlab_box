"""Start four policies on GPUs 0–3; navigation serves both go-to and return."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from five_provenance import fingerprints
from five_proprio import CONTRACT, validate_checkpoint
from five_rewards import RECIPE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SKILLS = ("navigation", "take", "carry", "put-down")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="/home/rwang/humanoid/.venv/bin/python")
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--warm-starts", type=Path, help="JSON mapping policy names to evaluated actor checkpoints")
    parser.add_argument("--skills", nargs="+", choices=SKILLS, default=list(SKILLS))
    parser.add_argument("--iterations", type=int, default=1500)
    args = parser.parse_args()
    out = HERE/"five_runs"
    manifest_path = out/f"launch-{args.run_id}.json"
    if manifest_path.exists():
        raise RuntimeError(f"Run already exists: {manifest_path}")
    setup_hash = hashlib.sha256((HERE/"five_stage.json").read_bytes()).hexdigest()
    warm_starts = json.loads(args.warm_starts.read_text()) if args.warm_starts else {}
    for checkpoint in warm_starts.values():
        validate_checkpoint(checkpoint)
    if len(set(args.skills)) != len(args.skills):
        parser.error("Duplicate policy")
    for skill in args.skills:
        report = json.loads((out/skill/"banks"/"validation.json").read_text())
        if not report["passed"]:
            raise RuntimeError(f"Reference validation failed: {skill}")
        smoke = json.loads((out/skill/"runs"/"smoke-launch"/"status.json").read_text())
        if smoke["state"] != "smoke_passed":
            raise RuntimeError(f"GPU smoke test has not passed: {skill}")
        if any(smoke.get(k) != v for k,v in fingerprints(skill).items()):
            raise RuntimeError(f"GPU smoke test is stale after code, geometry or reference changes: {skill}")
    gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    memory = {int(a):int(b) for a,b in (row.split(",") for row in gpu_rows.strip().splitlines())}
    if any(i not in memory or memory[i] > 1024 for i in range(len(args.skills))):
        raise RuntimeError("Assigned GPUs must be available and idle before launching.")
    manifest = dict(run_id=args.run_id, started_at=datetime.now(timezone.utc).isoformat(),
                    observation_contract=CONTRACT,
                    reward_recipe=RECIPE,
                    config_sha256=hashlib.sha256((HERE/"five_stage.json").read_bytes()).hexdigest(),
                    envs_per_gpu=args.envs, jobs=[])
    for gpu, skill in enumerate(args.skills):
        folder = out/skill/"runs"/args.run_id
        folder.mkdir(parents=True, exist_ok=False)
        log = folder/"training.log"
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MPLCONFIGDIR="/tmp/humanoid-mpl", MUJOCO_GL="egl", PYTHONUNBUFFERED="1",
                   PYTHONPATH=str(ROOT/".deps")+os.pathsep+str(HERE))
        iterations = args.iterations
        command = [args.python, str(HERE/"five_train.py"), "train", "--skill", skill,
                   "--run-id", args.run_id, "--envs", str(args.envs), "--iterations", str(iterations)]
        if skill in warm_starts:
            command += ["--warm-start", warm_starts[skill]]
        with log.open("w") as stream:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        manifest["jobs"].append(dict(skill=skill, gpu=gpu, pid=proc.pid, iterations=iterations,
                                     log=str(log), status=str(folder/"status.json"), command=command))
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    (out/"latest-launch.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
