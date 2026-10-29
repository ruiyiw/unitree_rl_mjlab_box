"""Training-only route objectives; no world-state input reaches the actor.

Use each route's start/end frame, so go-to (+y), carry (+x), and the diagonal
return all penalize sideways drift in the direction appropriate to that route.
"""
import torch

RECIPE = "proprio-precision-v2"


def palm_alignment(side_errors):
    """Both palms must have margin inside the 6 cm holding tolerance."""
    return torch.exp(-side_errors.amax(-1).square()/.04**2)


def capsule_box_clearance(endpoints, half_size, radius):
    """Conservative capsule/box clearance, all coordinates in the box frame.

    endpoints: (..., 2, 3), half_size: (3,), radius: broadcast over (...).
    Subtract half the sampling interval as a Lipschitz error bound. Uses the
    entire hand capsule, not the palm site (which missed the tabletop collision).
    """
    samples = torch.linspace(0., 1., 17, device=endpoints.device, dtype=endpoints.dtype)
    delta = endpoints[..., 1, :] - endpoints[..., 0, :]
    points = endpoints[..., :1, :] + samples[:, None]*delta[..., None, :]
    outside = points.abs()-half_size
    signed = outside.clamp_min(0.).norm(dim=-1) + outside.amax(-1).clamp_max(0.)
    return signed.amin(-1)-radius-delta.norm(dim=-1)/32.


def clearance_cost(clearance):
    """A 3 cm noncontact margin, bounded to avoid extreme reward outliers."""
    return ((.03-clearance).clamp(0., .12)/.03).square().amax(-1)


def route_scores(position, velocity, start, goal):
    direction = goal-start
    length = direction.norm(dim=-1).clamp_min(1e-6)
    forward = direction/length[..., None]
    lateral = torch.stack((-forward[..., 1], forward[..., 0]), dim=-1)
    offset = position-start
    along = (offset*forward).sum(-1)
    cross = (offset*lateral).sum(-1)
    forward_speed = (velocity*forward).sum(-1)
    lateral_speed = (velocity*lateral).sum(-1)
    remaining = length-along
    distance_sq = (position-goal).square().sum(-1)
    return {
        # Signed progress: moving away or overshooting is penalized. Saturation
        # prevents high speed from overpowering endpoint and drift objectives.
        "progress": (forward_speed/.3).clamp(-1., 1.) * (remaining/.1).clamp(-1., 1.),
        "endpoint": torch.exp(-distance_sq/.25**2),
        "precision": torch.exp(-distance_sq/.08**2),
        "stop": torch.exp(-distance_sq/.08**2 - velocity.square().sum(-1)/.1**2),
        "terminal_error": (distance_sq/.1**2).clamp_max(25.),
        # Allow 2 cm of normal gait sway; larger route deviation costs quadratically.
        "cross_track": ((cross.abs()-.02).clamp_min(0.)/.08).square(),
        "lateral_velocity": (lateral_speed/.2).square(),
        "cross_track_m": cross.abs(),
        "endpoint_error_m": distance_sq.sqrt(),
    }
