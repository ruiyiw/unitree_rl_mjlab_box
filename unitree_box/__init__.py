"""Frozen five-stage box policies for the 23-DoF G1."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / 'deploy/robots/g1_23dof/config/policy/box/v1'
REFERENCE = Path(__file__).parent / 'reference/policies'
ORDER = ('go-to', 'take', 'carry', 'put-down', 'return')
ACTORS = ('navigation', 'take', 'carry', 'put-down', 'return')
MOTOR_IDS = tuple(range(13)) + tuple(range(15, 20)) + tuple(range(22, 27))
