"""The human walk in native MuJoCo with the C++ controller g1_ctrl runs.

A CPU integration test without DDS: the training robot model, PD joints, and the compiled
controller in the loop.  Unlike training, the arms' weight is carried by the robot: MuJoCo's
body gravity compensation is switched off and the controller's arm torque is applied at the
ten arm motors, as on hardware.  One robot, no randomization; it says nothing about hardware.

  python -m unitree_human_walk.simulate --output logs/human_walk_sim
  ... --from-velocity         start in the Velocity policy (sticks centred), then hand over as the profile does
  ... --training-physics      training's weightless arms (for comparison with the batched evaluation)
  ... --ankle-offsets 1.0     multiples of one robot's measured ankle encoder offsets (PARAMETRIZED_WALK.md)
  ... --push 0.3 90 3.0       a push to the pelvis: m/s, direction in degrees from ahead (90 = to the left), seconds into the walk
  ... --hold                  keep the walk policy standing at the end (the profile hands back to Velocity instead)
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from unitree_parametrized_skills.simulate import Velocity

from . import MOTION, MOTOR_IDS, PROFILE, ROOT, TRAINING
from .runtime import Core, build_core, reference, telemetry, verify
from .training import modules

GRAVITY = 9.81
# The ankle encoder offsets implied by one G1's standing posture (PARAMETRIZED_WALK.md):
# pitch 1.15 and 1.75 degrees, roll -0.5 and -1.9 degrees (left, right).
ANKLE_OFFSETS = {'left_ankle_pitch_joint': 1.15, 'right_ankle_pitch_joint': 1.75,
                 'left_ankle_roll_joint': -.5, 'right_ankle_roll_joint': -1.9}


def robot_model(root=TRAINING):
    """The training env's robot on its floor, compiled for CPU MuJoCo with the training solver settings."""
    import mujoco
    motion, _ = modules(root)
    from mjlab.scene import Scene
    cfg = motion.env_cfg(str(Path(root) / MOTION), num_envs=1, play=True)
    cfg.scene.num_envs = 1
    model = Scene(cfg.scene, device='cpu').compile()
    cfg.sim.mujoco.apply(model)
    assert model.nu == 23
    return model, mujoco.MjData(model)


def play(output, root=TRAINING, from_velocity=False, training_physics=False, ankle_offsets=0., push=None,
         after_s=10., viewer=False, hold_s=None, blend_s=None, scale=1., lib=None, friction=None, save=True,
         return_s='profile'):
    """return_s: seconds after the reference ends at which Velocity takes over again, as State_HumanWalk
    does ('profile': the profile's value; None or 0: the walk policy keeps standing).  after_s: how long
    the robot is then watched."""
    import mujoco
    import yaml
    manifest = verify()
    interface = manifest['interface']
    profile = yaml.safe_load(PROFILE.read_text())['FSM']
    state = profile['HumanWalk']
    hold_s = state['hold_s'] if hold_s is None else hold_s
    blend_s = state['blend_s'] if blend_s is None else blend_s
    return_s = state.get('return_s', 0.) if return_s == 'profile' else (return_s or 0.)
    model, data = robot_model(root)
    frames = reference()
    names = interface['joint_order']
    joint_ids = [model.joint('robot/' + n).id for n in names]
    qadr, vadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    actuated = model.actuator_trnid[:, 0].tolist()
    act = np.array([actuated.index(j) for j in joint_ids])
    kp, kd = np.array(interface['kp']), np.array(interface['kd'])
    np.testing.assert_allclose(model.actuator_gainprm[act, 0], kp)
    np.testing.assert_allclose(-model.actuator_biasprm[act, 2], kd)
    root_joint = model.joint('robot/floating_base_joint')
    qa, va = int(root_joint.qposadr[0]), int(root_joint.dofadr[0])
    pelvis = model.body('robot/pelvis').id
    if not training_physics:
        model.body_gravcomp[:] = 0          # the arms weigh on the robot; the arm motors compensate (below)
    if friction is not None:
        model.geom_friction[:, 0] = friction
    offset = np.array([math.radians(ANKLE_OFFSETS.get(n, 0.)) * ankle_offsets for n in names])

    def gains(p, d):
        model.actuator_gainprm[act, 0], model.actuator_biasprm[act, 1], model.actuator_biasprm[act, 2] = p, -p, -d

    output = Path(output)
    if save:
        output.mkdir(parents=True, exist_ok=False)
    lib = lib or build_core(output)
    start = frames[0, 23:46]
    data.qpos[qa:qa + 7] = [0, 0, .80, 1, 0, 0, 0]
    data.qpos[qadr] = start
    mujoco.mj_forward(model, data)
    # set the robot on the floor: its lowest collision geom at z = 0
    low = min(data.geom_xpos[g, 2] - model.geom_rbound[g] for g in range(model.ngeom)
              if model.geom_bodyid[g] and (model.geom_contype[g] or model.geom_conaffinity[g])
              and model.body(model.geom_bodyid[g]).name.startswith('robot/') and 'ankle' in model.body(model.geom_bodyid[g]).name)
    data.qpos[qa + 2] -= low - .002
    mujoco.mj_forward(model, data)
    command = start.astype(np.float64).copy()
    display = None
    if viewer:
        import mujoco.viewer
        display = mujoco.viewer.launch_passive(model, data)
    trajectory, reason, last_v = [], 'completed', data.qvel[va:va + 3].copy()

    def sensors():
        """What the robot's IMU and encoders report (the encoders with their offsets)."""
        nonlocal last_v
        q = data.qpos[qa + 3:qa + 7].copy()
        v = data.qvel[va:va + 3].copy()            # the pelvis's velocity in the world
        a = (v - last_v) / .02 + [0, 0, GRAVITY]
        last_v = v
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        return q, data.qvel[va + 3:va + 6].copy(), R.reshape(3, 3).T @ a, data.qpos[qadr] + offset, data.qvel[vadr].copy()

    def advance(target, torque=None):
        trajectory.append(data.qpos.copy())
        data.ctrl[act] = target - offset           # the motor's PD closes on its own (offset) encoder
        data.qfrc_applied[:] = 0
        if torque is not None and not training_physics:
            data.qfrc_applied[vadr] = torque
        for _ in range(4):
            mujoco.mj_step(model, data)
        if display:
            display.sync()

    def fallen():
        up = np.zeros(9)
        mujoco.mju_quat2Mat(up, data.qpos[qa + 3:qa + 7])
        return up[8] < math.cos(math.pi / 4) or data.qpos[qa + 2] < .45

    velocity = None
    if from_velocity or return_s:
        velocity = Velocity(lib, ROOT / 'deploy/robots/g1_23dof' / profile['Velocity']['policy_dir'])
    if from_velocity:                              # stand on the Velocity policy first, sticks centred
        gains(velocity.kp, velocity.kd)
        for _ in range(250):
            q, gyro, _, jp, jv = sensors()
            command = velocity.step(q, gyro, jp, jv, np.zeros(3, np.float32))
            advance(command)
            if fallen():
                reason = 'fell on the Velocity policy'
                break
        kp0, kd0 = velocity.kp.astype(float), velocity.kd.astype(float)
    else:
        kp0, kd0 = kp, kd
    walk = {}
    if reason == 'completed':
        with Core(lib) as core:
            core.start(hold_s, scale)
            q0 = np.array(command, float)
            origin = data.qpos[qa:qa + 2].copy()
            w, x, y, z = data.qpos[qa + 3:qa + 7]
            yaw0 = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            first, done_at, ended_at, estimate = len(trajectory), None, None, np.zeros(7)
            total = round(hold_s / .02) + 1 + len(frames) + round((return_s if return_s else after_s) / .02)
            pushed = None
            lowest_height, largest_tilt = 1e9, 0.
            for tick in range(total):
                if push and pushed is None and tick * .02 >= hold_s + push[2]:
                    c, s = math.cos(yaw0 + math.radians(push[1])), math.sin(yaw0 + math.radians(push[1]))
                    data.qvel[va:va + 2] += push[0] * np.array([c, s])     # an impulse: the pelvis's velocity changes at once
                    pushed = tick * .02
                try:
                    obs, action, target, torque, flags, estimate = core.step(telemetry(*sensors()))
                except RuntimeError as error:
                    reason = str(error)
                    break
                s = .5 - .5 * math.cos(math.pi * min(tick * .02 / blend_s, 1.)) if blend_s > 0 else 1.
                gains(kp0 + (kp - kp0) * s, kd0 + (kd - kd0) * s)
                command = q0 + (target - q0) * s
                advance(command, torque * s)
                up = np.zeros(9)
                mujoco.mju_quat2Mat(up, data.qpos[qa + 3:qa + 7])
                largest_tilt = max(largest_tilt, math.degrees(math.acos(min(1., up[8]))))
                lowest_height = min(lowest_height, float(data.qpos[qa + 2]))
                if flags[1] and ended_at is None:
                    ended_at = tick * .02
                if return_s and ended_at is not None and tick * .02 >= ended_at + return_s:
                    break                          # the state hands back to Velocity
                if flags[2] and done_at is None:
                    done_at = tick * .02
                if fallen():
                    reason = 'fell'
                    break
            stood = None
            if return_s and reason == 'completed':
                # Velocity again (State_RLBase): its targets and gains blended in over 0.5 s from the last
                # command, the arm torque blended out; then the robot is watched for after_s
                velocity.raw[:] = 0
                velocity.phase = 0.
                q1, tau1 = np.array(command, float), np.array(torque, float)
                at_return = data.qpos[qa:qa + 2].copy()
                for tick in range(round(after_s / .02)):
                    q, gyro, _, jp, jv = sensors()
                    target = velocity.step(q, gyro, jp, jv, np.zeros(3, np.float32))
                    s = .5 - .5 * math.cos(math.pi * min(tick * .02 / .5, 1.))
                    gains(kp + (velocity.kp - kp) * s, kd + (velocity.kd - kd) * s)
                    command = q1 + (target - q1) * s
                    advance(command, tau1 * (1 - s))
                    if fallen():
                        reason = 'fell on the Velocity policy after the walk'
                        break
                d0 = at_return - origin
                end = frames[-1, 76:78] - frames[0, 76:78]
                stood = {'seconds': after_s, 'drift_m': float(np.linalg.norm(data.qpos[qa:qa + 2] - at_return)),
                         'error_at_the_hand_back_m': float(math.hypot(math.cos(yaw0) * d0[0] + math.sin(yaw0) * d0[1] - end[0],
                                                                      -math.sin(yaw0) * d0[0] + math.cos(yaw0) * d0[1] - end[1]))}
            c, s_ = math.cos(yaw0), math.sin(yaw0)
            d = data.qpos[qa:qa + 2] - origin
            along, across = c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]
            w, x, y, z = data.qpos[qa + 3:qa + 7]
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            goal = frames[-1, 76:78] - frames[0, 76:78]
            walk = {'reference_m': [float(goal[0]), float(goal[1])], 'walked_m': float(along), 'sideways_m': float(across),
                    'error_m': float(math.hypot(along - goal[0], across - goal[1])),
                    'estimate_m': [float(estimate[0]), float(estimate[1])],
                    'estimate_minus_truth_m': [float(estimate[0] - along), float(estimate[1] - across)],
                    'heading_change_deg': math.degrees(math.atan2(math.sin(yaw - yaw0), math.cos(yaw - yaw0))),
                    'reference_ended_s_after_entry': ended_at, 'settled_s_after_entry': done_at,
                    'largest_pelvis_tilt_deg': largest_tilt, 'lowest_pelvis_m': lowest_height,
                    'pushed_s_after_entry': pushed, 'first_frame': first,
                    'returned_to_velocity_s_after_the_reference': return_s or None, 'on_velocity_afterwards': stood}
    if display:
        display.close()
    report = {'from_velocity': from_velocity, 'blend_s': blend_s, 'hold_s': hold_s, 'stop_reason': reason,
              'arm_weight': 'weightless to the pelvis, as in training' if training_physics
                            else 'carried by the robot; the controller\'s torque at the ten arm motors',
              'ankle_offset_multiple': ankle_offsets, 'push_m_s_deg_s': list(push) if push else None,
              'foot_friction': friction, 'walk': walk,
              'physics': 'native CPU MuJoCo, one nominal robot, PD joints; not the batched GPU evaluation',
              'hardware_tested': False}
    if save:
        np.savez_compressed(output / 'trajectory.npz', qpos=np.asarray(trajectory), dt=.02)
        (output / 'simulation.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, default=TRAINING)
    parser.add_argument('--from-velocity', action='store_true')
    parser.add_argument('--training-physics', action='store_true')
    parser.add_argument('--ankle-offsets', type=float, default=0.)
    parser.add_argument('--push', type=float, nargs=3, default=None)
    parser.add_argument('--hold', action='store_true')
    parser.add_argument('--viewer', action='store_true')
    a = parser.parse_args()
    play(a.output, a.training_root, a.from_velocity, a.training_physics, a.ankle_offsets, a.push, viewer=a.viewer,
         return_s=None if a.hold else 'profile')
