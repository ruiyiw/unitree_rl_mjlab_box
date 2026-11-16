"""Distance-parametrized walking, walk(N), for the 23-DoF G1: two frozen policies and a plan library."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'deploy/robots/g1_23dof/config/policy/parametrized_walk/v1'
PROFILE = ROOT / 'deploy/robots/g1_23dof/config/parametrized_walk.yaml'
REFERENCE = Path(__file__).parent / 'reference'
# name: (training run of the selected checkpoint, frozen base walker's run or None)
POLICIES = {'goal-conditioned': ('goal20', None), 'adapter': ('adapter20_k', 'base')}
MOTOR_IDS = tuple(range(13)) + tuple(range(15, 20)) + tuple(range(22, 27))
OBSERVATION = 109          # gyro 3, gravity 3, joints 23, joint speeds 23, previous action 23, plan 28, goal 6
FRAME = 28                 # a plan frame: joint targets 23, pelvis x y, stride phase sin cos, walking
TELEMETRY = 68             # IMU quaternion 4, gyro 3, accelerometer 3, 29 motor positions, 29 motor speeds
