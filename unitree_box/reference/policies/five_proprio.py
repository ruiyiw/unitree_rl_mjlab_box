"""Hardware observation contract and controller checks, independent of simulation.

Matches humanoid_balance's 23-DoF actor / HumanPushPolicy.h observation order.
The residual action still uses a locally stored joint reference and clock; those
are controller memory, never measurements of the robot or box in the world.
"""
import math

import torch

OBSERVATION_DIM = 75
ACTOR_TERMS = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions")
CONTRACT = {
    "id": "g1-five-proprio-v1",
    "actor_dim": OBSERVATION_DIM,
    "terms": list(ACTOR_TERMS),
    "widths": [3, 3, 23, 23, 23],
    "frame": "pelvis",
    "joint_order": "robot.joint_names",
    "joint_position": "measured minus default_joint_pos",
    "joint_velocity": "rad/s (zero default velocity)",
    "previous_action": "unclipped residual, preserved across stage switches",
    "action": "local qcmd[clock+1] + action_scale * residual",
    "reference_clock": "50 Hz local counter; no contact or world-state gates",
    "handoff": "reference ended and proprioceptive endpoint settled for 1 second",
}


def observation(gyro, gravity, joint_position, joint_velocity, previous_action, default_position):
    """All arguments are batched tensors in policy joint order, in SI units."""
    result = torch.cat((gyro, gravity, joint_position-default_position,
                        joint_velocity, previous_action), dim=-1)
    if result.shape[-1] != OBSERVATION_DIM:
        raise ValueError("Expected 23 joints and a 75-value human-push observation")
    return result


def settled(obs, endpoint_relative_joints):
    """Sensor-only handoff readiness; does NOT establish task success or release."""
    return ((obs[..., :3].norm(dim=-1) < .25)
            & (obs[..., 5] < -math.cos(math.radians(15)))
            & ((obs[..., 6:29]-endpoint_relative_joints).abs().amax(dim=-1) < .3)
            & (obs[..., 29:52].abs().amax(dim=-1) < .5))


def validate_checkpoint(path):
    """Reject privileged-input weights before resume, evaluation or export."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    contract = checkpoint.get("observation_contract", (checkpoint.get("infos") or {}).get("observation_contract"))
    if contract != CONTRACT:
        raise ValueError(f"{path}: incompatible observation/controller contract. "
                         "Retrain with five_train.py; legacy 156/180-input actors cannot be reused.")
    return checkpoint
