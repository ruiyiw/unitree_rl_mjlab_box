"""Distance-parametrized walking, walk(N), for the 23-DoF G1: two frozen policies and a plan library."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'deploy/robots/g1_23dof/config/policy/parametrized_walk/v2'
PROFILE = ROOT / 'deploy/robots/g1_23dof/config/parametrized_walk.yaml'
REFERENCE = Path(__file__).parent / 'reference'
# name: (training run of the selected checkpoint, frozen base walker's run or None)
# v2: four sensor frames, trained with the Kalman estimate.  The adapter and its base walker are
# retrained with the robust randomization profile.  The goal-conditioned policy is goal20_hk2
# (the first profile): every retraining of it with the robust profile did worse in the controller
# tests (PARAMETRIZED_WALK.md).  v1 was goal20 and adapter20_k on base (one frame); it fell on the robot.
POLICIES = {'goal-conditioned': ('goal20_hk2', None), 'adapter': ('adapter20_r', 'base_r_seed2')}
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
