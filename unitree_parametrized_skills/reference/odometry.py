"""odometry.py — where the pelvis is, from joint encoders and the IMU only.

The real G1 has no position sensor.  What it does have is its leg encoders and the IMU's
orientation, and that is enough for kinematic (leg) odometry: the foot that carries the
weight does not move on the floor, so the pelvis moves by minus the change of that foot's
position in the pelvis frame.  This module is that estimator, batched over envs:

  LegKinematics   forward kinematics of both foot sites in the pelvis frame, from the 12
                  measured leg joint angles.  Built from the compiled MuJoCo model (body
                  offsets, joint axes), so the same numbers can be exported to the robot.
  LegOdometry     the pelvis's horizontal position in the frame of the skill's start: each
                  step it moves by minus the displacement of the feet relative to the pelvis,
                  weighted toward the lowest foot (in the IMU's gravity-aligned frame).
  KalmanOdometry  the same kinematic velocity fused with the IMU's accelerometer in a Kalman
                  filter (position, velocity, accelerometer bias); with an untrusted
                  accelerometer it reduces to LegOdometry.

Nothing here reads the simulator's base position or velocity, or its contacts.  Errors come
from where they come from on hardware: foot slip and rolling, encoder bias, IMU tilt and
heading error.
"""
from __future__ import annotations

import mujoco
import torch

SIDES = ("left", "right")


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) wxyz -> (..., 3, 3)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def _axis_angle_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues: unit axis (3,), angles (N,) -> (N, 3, 3)."""
    K = torch.zeros(3, 3, device=angle.device, dtype=angle.dtype)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -axis[2], axis[1], axis[2], -axis[0], -axis[1], axis[0]
    eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
    s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
    return eye + s * K + (1 - c) * (K @ K)


def _sole_points(m: mujoco.MjModel, body: int):
    """The ends of a foot body's collision capsules, lowered by their radius (body frame)."""
    import numpy as np
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != body or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        axis, (rad, half) = R.reshape(3, 3)[:, 2], m.geom_size[g][:2]
        for sgn in (-1.0, 1.0):
            pts.append(m.geom_pos[g] + sgn * half * axis - np.array([0.0, 0.0, rad]))
    assert pts, "no collision capsules on the foot body"
    return np.stack(pts)


class LegKinematics:
    """Both foot sites in the pelvis frame from joint angles (in a given joint order)."""

    def __init__(self, mj_model: mujoco.MjModel, joint_names: list[str], device, prefix: str = "",
                 root: str = "pelvis", sites=("left_foot", "right_foot")):
        m = mj_model
        root_id = m.body(prefix + root).id
        self.chains, self.soles = [], []
        f = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        for site in sites:
            sid = m.site(prefix + site).id
            b = int(m.site_bodyid[sid])
            links = []
            while b != root_id:
                assert b != 0, f"{site} is not under {root}"
                assert m.body_jntnum[b] == 1, "one hinge per leg body expected"
                j = int(m.body_jntadr[b])
                assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                name = m.joint(j).name.removeprefix(prefix)
                links.append({"pos": f(m.body_pos[b]), "R": _quat_to_mat(f(m.body_quat[b])),
                              "axis": f(m.jnt_axis[j]), "anchor": f(m.jnt_pos[j]),
                              "col": joint_names.index(name)})
                b = int(m.body_parentid[b])
            self.chains.append((links[::-1], f(m.site_pos[sid])))
            self.soles.append(f(_sole_points(m, int(m.site_bodyid[sid]))))
        self.joint_cols = sorted({ln["col"] for links, _ in self.chains for ln in links})

    def _frames(self, joint_pos: torch.Tensor):
        """Each foot body's position (N, 3) and orientation (N, 3, 3) in the pelvis frame."""
        n = joint_pos.shape[0]
        out = []
        for links, _ in self.chains:
            p = torch.zeros(n, 3, device=joint_pos.device)
            R = torch.eye(3, device=joint_pos.device).expand(n, 3, 3)
            for ln in links:
                p = p + (R @ ln["pos"])
                R = R @ ln["R"]
                Rj = _axis_angle_mat(ln["axis"], joint_pos[:, ln["col"]])
                p = p + (R @ (ln["anchor"] - Rj @ ln["anchor"])[..., None]).squeeze(-1)
                R = R @ Rj
            out.append((p, R))
        return out

    def feet(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """joint_pos (N, nj) absolute angles -> foot sites (N, 2, 3) in the pelvis frame."""
        return torch.stack([p + R @ site for (p, R), (_, site) in zip(self._frames(joint_pos), self.chains)], 1)

    def points(self, joint_pos: torch.Tensor, sole: bool = False) -> torch.Tensor:
        """Candidate contact points of each foot, (N, 2, K, 3) in the pelvis frame: the foot
        site alone (K = 1), or the ends of the sole's collision capsules at the sole's
        underside (sole: the foot pivots on its heel and toe, which the site does not show)."""
        if not sole:
            return self.feet(joint_pos)[:, :, None]
        return torch.stack([p[:, None] + (R[:, None] @ pts[None, :, :, None]).squeeze(-1)
                            for (p, R), pts in zip(self._frames(joint_pos), self.soles)], 1)


class LegOdometry:
    """The pelvis's (x, y) in the start frame, integrated over the feet on the floor.

    Each step, every foot's displacement relative to the pelvis says how the pelvis moved IF
    that foot stood still.  The feet are weighted toward the lowest one, softmax(-height /
    SOFT): in single support the stance foot has all the weight, in double support both, and
    a foot that lifts loses its weight within a few millimetres.

    The estimate is exactly as good as the weighted feet are still, so a foot dragged along
    the floor while it is unloaded is an error.  Training therefore penalizes the drift rate
    (env.r_odometry_drift): the policy learns to walk so that its odometry is right.  Two
    alternatives were measured on a policy trained WITHOUT that penalty (2 m walk, clean
    sensors): anchoring on the lower foot with a 1 cm hysteresis lost 25% of the distance,
    this weighting (then with SOFT = 3 mm) 10%, and weighting by foot load estimated from the
    leg motors' torques (J^-T tau) 21%, because the feet slid while still loaded.

    SOFT is 1 mm.  At 3 mm a foot skimming 2-5 mm above the floor kept a quarter of the
    weight; whether that mattered depended on the gait a training run happened to find
    (0.7% of the distance lost in one run, 3-5% in the next).  Replaying four trained
    policies' recorded 2 m walks under randomized sensors, 1 mm gave a median error of
    3-5 cm where 3 mm gave 3-10 cm; simulated foot contacts were no better a stance signal
    than height (feet slide while touching)."""

    SOFT = 0.001                     # m: the height scale of the stance weights
    SCALE = 1.0                      # true distance per kinematic distance; `scale` is the instance's value,
                                     # set from a calibration walk (calibrate.py) when a gait slips steadily

    def __init__(self, kin: LegKinematics, num_envs: int, device, sole: bool = False):
        self.kin, self.sole, self.scale = kin, sole, self.SCALE
        k = kin.soles[0].shape[0] if sole else 1
        self.pos = torch.zeros(num_envs, 2, device=device)             # estimate
        self.vel = torch.zeros(num_envs, 2, device=device)             # its rate (m/s)
        self.feet = torch.zeros(num_envs, 2, k, 3, device=device)      # last step's contact points
        self.weights = torch.full((num_envs, 2), 0.5, device=device)   # per foot

    def _feet(self, joint_pos, R_imu):
        """Contact points relative to the pelvis in the start-heading, gravity-aligned frame."""
        return (R_imu[:, None, None] @ self.kin.points(joint_pos, self.sole)[..., None]).squeeze(-1)

    def reset(self, ids: torch.Tensor, joint_pos: torch.Tensor, R_imu: torch.Tensor, pos0: torch.Tensor | None = None,
              vel0: torch.Tensor | None = None):
        """Start envs `ids` at estimate pos0 (default: the origin).  joint_pos, R_imu: of those envs."""
        self.pos[ids] = 0.0 if pos0 is None else pos0
        self.vel[ids] = 0.0 if vel0 is None else vel0
        self.feet[ids] = self._feet(joint_pos, R_imu)

    def _kinematic(self, joint_pos, R_imu):
        """This step's pelvis displacement as each foot alone would have it (N, 2, 2), and the
        feet's weights (N, 2).  Within a foot its lowest points count; between feet the lowest."""
        feet = self._feet(joint_pos, R_imu)
        n, _, k, _ = feet.shape
        z = 0.5 * (feet[..., 2] + self.feet[..., 2])
        w = torch.softmax(-z.reshape(n, -1) / self.SOFT, dim=1).reshape(n, 2, k)
        per_foot = w.sum(-1)
        d = -self.scale * (w[..., None] * (feet - self.feet)[..., :2]).sum(2) / per_foot.clamp_min(1e-9)[..., None]
        self.feet = feet
        return d, per_foot

    def update(self, joint_pos: torch.Tensor, R_imu: torch.Tensor, acc: torch.Tensor | None = None,
               dt: float = 0.02) -> torch.Tensor:
        """One control step, every env.  joint_pos (N, nj) measured; R_imu (N, 3, 3) the IMU's
        orientation with its heading measured from the start heading.  (acc is not used.)"""
        d, self.weights = self._kinematic(joint_pos, R_imu)
        step = (self.weights[..., None] * d).sum(1)
        self.vel = step / dt
        self.pos = self.pos + step
        return self.pos


class KalmanOdometry(LegOdometry):
    """Leg kinematics fused with the IMU's accelerometer.

    Per horizontal axis the state is (position, velocity, accelerometer bias).  Each control
    step it is predicted with the measured horizontal acceleration (the specific force
    turned by the IMU's attitude, gravity removed) and corrected with the pelvis velocity
    each foot's kinematics imply, trusted in proportion to the foot's stance weight.  A foot
    whose velocity disagrees with the prediction by more than GATE standard deviations is
    trusted less (a slipping or landing foot).  Position is not measured: it is the filtered
    velocity integrated, so heading error and a steady slip still accumulate.

    With SIG_A large the filter ignores the accelerometer and approaches LegOdometry.

    The gate is what matters.  Replaying recorded walks of two trained policies (nominal
    robot and randomized with an exact heading), the filter without a useful gate (3 sigma)
    ended within 1 cm of LegOdometry everywhere; LegOdometry reads 0.4-0.7% long, steadily,
    and at 0.7 sigma the feet's landing and push-off transients that cause it are trusted
    less: at 20 m, 13.7 -> 4.6 cm and 10.9 -> 6.1 cm nominal, 11.2 -> 8.4 cm and
    12.1 -> 6.6 cm randomized.  Using the sole's ends as contact points instead of the foot
    site was worse for both estimators (sole=True; 2-4 times the error)."""

    SIG_A = 1.0                      # m/s^2: accelerometer noise per step (attitude error, vibration)
    SIG_B = 0.02                     # m/s^2 per sqrt(s): how fast its bias may wander
    SIG_V = 0.03                     # m/s: a planted foot's kinematic velocity
    GATE = 0.7                       # standard deviations before a foot is trusted less
    P0 = (1e-4, 0.05 ** 2, 0.3 ** 2)  # initial variances: position, velocity, bias

    def __init__(self, kin: LegKinematics, num_envs: int, device, sole: bool = False):
        super().__init__(kin, num_envs, device, sole)
        self.bias = torch.zeros(num_envs, 2, device=device)
        self.P = torch.diag(torch.tensor(self.P0, device=device)).repeat(num_envs, 1, 1)

    def reset(self, ids, joint_pos, R_imu, pos0=None, vel0=None):
        super().reset(ids, joint_pos, R_imu, pos0, vel0)
        self.bias[ids] = 0.0
        self.P[ids] = torch.diag(torch.tensor(self.P0, device=self.P.device))

    def update(self, joint_pos, R_imu, acc=None, dt: float = 0.02):
        d, self.weights = self._kinematic(joint_pos, R_imu)
        z = d / dt                                                    # (N, 2 feet, 2 axes)
        a = torch.zeros_like(self.pos) if acc is None else acc - self.bias
        # predict
        pos = self.pos + self.vel * dt + 0.5 * a * dt * dt
        vel = self.vel + a * dt
        F = torch.eye(3, device=pos.device).repeat(len(pos), 1, 1)
        F[:, 0, 1], F[:, 0, 2], F[:, 1, 2] = dt, -0.5 * dt * dt, -dt
        P = F @ self.P @ F.transpose(1, 2)
        P[:, 1, 1] += (self.SIG_A * dt) ** 2
        P[:, 2, 2] += self.SIG_B ** 2 * dt
        # the two feet's velocities as one measurement (information-weighted), gated
        r = self.SIG_V ** 2 / self.weights.clamp_min(1e-6)            # (N, 2)
        e = (z - vel[:, None]).norm(dim=-1)                           # innovation of each foot
        sd = torch.sqrt(P[:, 1, 1, None] + r)
        r = r * (e / (self.GATE * sd)).clamp_min(1.0) ** 2
        info = 1.0 / r
        R = 1.0 / info.sum(1)
        zbar = (info[..., None] * z).sum(1) * R[:, None]
        # correct
        K = P[:, :, 1] / (P[:, 1, 1] + R)[:, None]                    # (N, 3)
        innov = zbar - vel                                            # (N, 2)
        self.pos = pos + K[:, 0:1] * innov
        self.vel = vel + K[:, 1:2] * innov
        self.bias = self.bias + K[:, 2:3] * innov
        P = P - K[:, :, None] * P[:, 1:2, :]
        self.P = 0.5 * (P + P.transpose(1, 2))
        return self.pos
