"""Where the robot is, from what it measures: the two inputs of the human-walk actor that the G1
has no sensor for.

Training gave the actor the torso's true pose relative to the plan and the IMU's true linear
velocity.  Here they are estimated from the pelvis IMU and the joint encoders:

  pelvis x, y and their rates   the walk controller's Kalman leg odometry (a planted foot does
                                not move), unitree_parametrized_skills/reference/odometry.py
  pelvis height and its rate    minus the height of the lowest sole point under the pelvis
  orientation                   the IMU's, with its heading measured from the start of the walk
  torso                         the pelvis and the waist encoder

Batched over robots (torch).  HumanWalkPolicy.h is the same arithmetic for one robot; validate.py
compares the two step by step.
"""
import torch

from unitree_parametrized_skills.reference import odometry


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Estimate:
    def __init__(self, mj_model, joint_names, num, device, prefix=''):
        m = mj_model
        self.kin = odometry.LegKinematics(m, list(joint_names), device, prefix=prefix)
        self.kalman = odometry.KalmanOdometry(self.kin, num, device)
        f = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        torso = m.body(prefix + 'torso_link')
        assert m.body(torso.parentid[0]).name == prefix + 'pelvis' and torso.jntnum[0] == 1
        j = int(torso.jntadr[0])
        self.waist = list(joint_names).index(m.joint(j).name.removeprefix(prefix))
        self.torso_position, self.torso_rotation = f(torso.pos), odometry._quat_to_mat(f(torso.quat))
        self.waist_axis, self.waist_anchor = f(m.jnt_axis[j]), f(m.jnt_pos[j])
        imu = m.site(prefix + 'imu_in_pelvis')
        assert m.site_bodyid[imu.id] == m.body(prefix + 'pelvis').id
        self.imu = f(imu.pos)
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        self.heading_origin, self.height, self.rise = z(num), z(num), z(num)
        self.attitude = torch.eye(3, device=device).repeat(num, 1, 1)

    def constants(self):
        """What the C++ controller needs beyond the walk controller's leg kinematics."""
        t = lambda x: x.reshape(-1).tolist()  # noqa: E731
        return {'soles': [t(s) for s in self.kin.soles], 'torso_position': t(self.torso_position),
                'torso_rotation': t(self.torso_rotation), 'waist_axis': t(self.waist_axis),
                'waist_anchor': t(self.waist_anchor), 'waist_joint': self.waist, 'imu': t(self.imu)}

    def _attitude(self, quaternion):
        """The IMU's orientation with its heading measured from the heading at the start."""
        c, s = torch.cos(-self.heading_origin), torch.sin(-self.heading_origin)
        z, o = torch.zeros_like(c), torch.ones_like(c)
        return torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(-1, 3, 3) @ odometry._quat_to_mat(quaternion)

    def _height(self, joint_pos, attitude):
        points = (attitude[:, None, None] @ self.kin.points(joint_pos, sole=True)[..., None]).squeeze(-1)
        return -points[..., 2].reshape(len(joint_pos), -1).min(1).values

    def step(self, quaternion, accelerometer, joint_pos, new, start, start_yaw=0., dt=.02, scale=1.):
        """One control step for every robot.  new (N,) bool: the robots that start a walk now, at
        the plan's first pelvis position `start` (2,) and heading `start_yaw`.
        accelerometer: specific force in the pelvis frame (about +9.81 on z at rest)."""
        ids, old = new.nonzero().squeeze(-1), (~new).nonzero().squeeze(-1)
        if len(ids):
            self.heading_origin[ids] = yaw_of(quaternion[ids]) - start_yaw
        R = self._attitude(quaternion)
        k = self.kalman
        k.scale = scale
        if len(ids):
            k.reset(ids, joint_pos[ids], R[ids], start.expand(len(ids), 2))
            self.height[ids] = self._height(joint_pos[ids], R[ids])
            self.rise[ids] = 0.
        if len(old):
            kept = [x.clone() for x in (k.pos, k.vel, k.bias, k.P, k.feet)]
            k.update(joint_pos, R, (R @ accelerometer[..., None]).squeeze(-1)[:, :2], dt)
            for now, before in zip((k.pos, k.vel, k.bias, k.P, k.feet), kept):
                now[ids] = before[ids]
            h = self._height(joint_pos, R)
            self.rise[old] = ((h - self.height) / dt)[old]
            self.height[old] = h[old]
        self.attitude = R

    def pelvis(self):
        """Position (N, 3) in the plan's frame."""
        return torch.cat([self.kalman.pos, self.height[:, None]], -1)

    def velocity(self):
        return torch.cat([self.kalman.vel, self.rise[:, None]], -1)

    def torso(self, joint_pos):
        """The torso link's position (N, 3) and orientation (N, 3, 3) in the plan's frame."""
        R = self.attitude
        turn = odometry._axis_angle_mat(self.waist_axis, joint_pos[:, self.waist])
        base = R @ self.torso_rotation
        position = (self.pelvis() + (R @ self.torso_position)
                    + (base @ (self.waist_anchor - turn @ self.waist_anchor)[..., None]).squeeze(-1))
        return position, base @ turn

    def imu_velocity(self, gyro):
        """What a velocimeter at the IMU would read: its velocity in its own frame."""
        v = (self.attitude.transpose(1, 2) @ self.velocity()[..., None]).squeeze(-1)
        return v + torch.cross(gyro, self.imu.expand_as(gyro), dim=-1)
