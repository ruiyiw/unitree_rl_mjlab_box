"""Posture requirements shared by reference validation and task completion."""
import math

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

LOADED_MAX_TILT = math.radians(15.)
LOADED_POSTURE_ERROR = .10
UNLOADED_MAX_TILT = .10


def posture_ok(actual_quat, reference_quat, loaded):
    """Ignore heading here: compare gravity directions to assess roll/pitch."""
    up = torch.zeros_like(actual_quat[..., :3])
    up[..., 2] = 1.
    actual_up = quat_apply_inverse(actual_quat, up)
    reference_up = quat_apply_inverse(reference_quat, up)
    tilt = torch.acos(actual_up[..., 2].clamp(-1.,1.))
    error = torch.acos((actual_up*reference_up).sum(-1).clamp(-1.,1.))
    if loaded:
        return (tilt < LOADED_MAX_TILT) & (error < LOADED_POSTURE_ERROR)
    return tilt < UNLOADED_MAX_TILT


PUTDOWN_MIN_HEIGHT = .73
PUTDOWN_MAX_HEIGHT = .81
PUTDOWN_MAX_TILT = math.radians(6.)


def upright_putdown_ok(height, torso_quat):
    """Whole post-release motion, not just its final standing pose."""
    w, x, y, z = torso_quat.unbind(-1)
    cos_tilt = 1. - 2.*(x.square()+y.square())
    return ((height >= PUTDOWN_MIN_HEIGHT) & (height <= PUTDOWN_MAX_HEIGHT)
            & (cos_tilt >= math.cos(PUTDOWN_MAX_TILT)))
