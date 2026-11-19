"""The policy in the batched training environment, on the robot's own inputs.

motion.py's measurement (mean actions, no observation noise, 60 s per run) with the torso's
pose and the IMU's velocity taken from estimate.py, not from the simulator.

  python -m unitree_human_walk.evaluate --output logs/human_walk_eval.json
  ... --pushes        training's pushes (up to 0.4 m/s to the pelvis every 1-3 s)
  ... --arm-weight    the arms' weight carried by the robot, as on hardware (training's is not)
  ... --truth         the simulator's pose and velocity, as in training (for comparison)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import CHECKPOINT, MOTION, TRAINING
from .training import arm_weight, install, modules


def evaluate(output, root=TRAINING, checkpoint=CHECKPOINT, envs=256, pushes=False, weight=False, truth=False):
    import torch
    errors = {}
    motion, _ = modules(root) if truth else install(root, errors)
    if weight:
        arm_weight(motion)
    folder = Path(root) / MOTION
    player = motion.Player(str(folder), envs, pushes=pushes)
    result = player.evaluate(str(folder / checkpoint))
    result.update(checkpoint=checkpoint, arm_weight_on_the_pelvis=weight,
                  position_and_velocity_inputs='simulator' if truth else 'estimate.py (leg odometry, sole height, IMU, waist encoder)')
    for key, values in errors.items():
        e = torch.stack(values).flatten().cpu().numpy()
        result[f'estimate_error_{key}_median_p95_max'] = [round(float(np.percentile(e, q)), 4) for q in (50, 95, 100)]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result, indent=1))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, default=TRAINING)
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--envs', type=int, default=256)
    parser.add_argument('--pushes', action='store_true')
    parser.add_argument('--arm-weight', action='store_true')
    parser.add_argument('--truth', action='store_true')
    a = parser.parse_args()
    evaluate(a.output, a.training_root, a.checkpoint, a.envs, a.pushes, a.arm_weight, a.truth)
