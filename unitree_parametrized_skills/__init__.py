"""Distance-parametrized walking, walk(N), for the 23-DoF G1: two frozen policies and a plan library."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'deploy/robots/g1_23dof/config/policy/parametrized_walk/v3'
PROFILE = ROOT / 'deploy/robots/g1_23dof/config/parametrized_walk.yaml'
REFERENCE = Path(__file__).parent / 'reference'
# name: (training run of the selected checkpoint, frozen base walker's run or None)
# v3: the goal-conditioned policy is abl_256_ppo_goto_s3 (PPO alone with the box go-to policy's settings, fall
# penalty and randomization ranges, 8,000 updates, trained on 0-5 m plans; four sensor frames, Kalman estimate),
# the first to match go-to in the controller tests (PARAMETRIZED_WALK.md).  The library therefore ends at 5 m
# (export --max 5).  The adapter and its base walker are v2's, unchanged.
# v2 was goal20_hk2 with the same adapter (0-20 m); v1 was goal20 and adapter20_k on base (one frame).  The
# goal-conditioned and adapter policies of v1 and v2 fell on the robot.
POLICIES = {'goal-conditioned': ('abl_256_ppo_goto_s3', None), 'adapter': ('adapter20_r', 'base_r_seed2')}
MOTOR_IDS = tuple(range(13)) + tuple(range(15, 20)) + tuple(range(22, 27))
OBSERVATION = 109          # one frame: gyro 3, gravity 3, joints 23, joint speeds 23, previous action 23, plan 28, goal 6
SENSORS = 75               # of which the sensor terms, which the actor reads `history` frames of


def actor_inputs(history):
    """The actor's input width with `history` sensor frames."""
    return OBSERVATION + SENSORS * (history - 1)
FRAME = 28                 # a plan frame: joint targets 23, pelvis x y, stride phase sin cos, walking
TELEMETRY = 68             # IMU quaternion 4, gyro 3, accelerometer 3, 29 motor positions, 29 motor speeds


def newest_frame(observation, history):
    """The one-frame observation (109) inside an actor input with `history` sensor frames
    (each sensor term's readings together, oldest first, then the plan and goal terms)."""
    import numpy as np
    o, parts, at = np.asarray(observation), [], 0
    for width in (3, 3, 23, 23, 23):
        parts.append(o[..., at + width * (history - 1):at + width * history])
        at += width * history
    return np.concatenate(parts + [o[..., at:]], -1)
