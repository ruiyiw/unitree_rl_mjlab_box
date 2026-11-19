"""A person's walk on the 23-DoF G1: one policy that tracks a retargeted motion-capture walk (2.2 m)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'deploy/robots/g1_23dof/config/policy/human_walk/v1'
PROFILE = ROOT / 'deploy/robots/g1_23dof/config/human_walk.yaml'
REFERENCE = Path(__file__).parent / 'reference'
TRAINING = ROOT.parent / 'humanoid-full/policies'      # the training checkout: motion.py and the motion's folder
MOTION = 'policy-walk-human'
# Iteration 500 of 1,000.  Later checkpoints withstand pushes better in training, but they shuffle
# their feet while standing, and the leg odometry the robot must use then drifts (HUMAN_WALK.md).
CHECKPOINT = 'runs/policy/model_500.pt'
MOTOR_IDS = tuple(range(13)) + tuple(range(15, 20)) + tuple(range(22, 27))
OBSERVATION = 145
# the actor's input, in order: name, width
TERMS = (('ref_joints', 46), ('ref_anchor', 9), ('ref_preview', 10), ('ref_clock', 2), ('base_lin_vel', 3),
         ('base_ang_vel', 3), ('projected_gravity', 3), ('joint_pos', 23), ('joint_vel', 23), ('actions', 23))
FRAME = 78                 # a plan frame: joint targets 23, joint positions 23, joint speeds 23, torso position 3 and quaternion 4, pelvis x y
TELEMETRY = 68             # IMU quaternion 4, gyro 3, accelerometer 3, 29 motor positions, 29 motor speeds
MAGIC = b'G1HWALK1'


def term(name):
    """The slice of a term in the actor's input."""
    at = 0
    for n, width in TERMS:
        if n == name:
            return slice(at, at + width)
        at += width
    raise KeyError(name)


assert sum(w for _, w in TERMS) == OBSERVATION
