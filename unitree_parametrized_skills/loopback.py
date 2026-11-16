"""The real g1_ctrl binary against a simulated robot over DDS, driven as an operator would.

g1_ctrl runs unchanged (--network lo).  A small DDS bridge (tests/fixtures/walk_dds_bridge.cpp)
publishes the robot state of a native MuJoCo simulation and returns the motor commands; this
script presses gamepad buttons and types distances into the controller's terminal:

  Passive -> FixStand (hanging) -> lowered -> Velocity -> released -> type 1.5, gamepad +1.0 and -0.1
  -> goal-conditioned walk (2.4 m) -> type 1, walk again (1.0 m) -> adapter walk (1.0 m)
  -> Velocity -> type 0.1 -> adapter refused -> Passive

It needs the full controller build and the Unitree SDK it was built with (a development
machine), and wall-clock time: the controller's 20 ms deadlines are real.

  python -m unitree_parametrized_skills.loopback --output logs/walk_loopback
"""
import argparse
import json
import math
import re
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from . import MOTOR_IDS, PROFILE, ROOT
from .runtime import library, verify
from .simulate import GRAVITY, robot_model

BUILD = ROOT / 'deploy/robots/g1_23dof/build'
BUTTONS = {'RB': 0, 'LB': 1, 'start': 2, 'back': 3, 'RT': 4, 'LT': 5, 'F1': 6, 'F2': 7,
           'A': 8, 'B': 9, 'X': 10, 'Y': 11, 'up': 12, 'right': 13, 'down': 14, 'left': 15}


def build_bridge(directory, sdk):
    arch = 'x86_64'
    out = Path(directory) / 'walk_dds_bridge'
    libs = [sdk / 'lib' / arch, sdk / 'thirdparty/lib' / arch]
    subprocess.run(['g++', '-std=c++17', '-O2', str(ROOT / 'tests/fixtures/walk_dds_bridge.cpp'),
                    '-I' + str(sdk / 'include'), '-I' + str(sdk / 'thirdparty/include'),
                    '-I' + str(sdk / 'thirdparty/include/ddscxx'),
                    *[f'-L{p}' for p in libs], *[f'-Wl,-rpath,{p}' for p in libs],
                    '-lunitree_sdk2', '-lddscxx', '-lddsc', '-lpthread', '-lrt', '-o', str(out)], check=True)
    return out


def remote(*pressed):
    """The 40 wireless-remote bytes with these buttons held and the sticks centred."""
    bits = sum(1 << BUTTONS[b] for b in pressed)
    return struct.pack('<2BH5f', 0x55, 0x51, bits, 0., 0., 0., 0., 0.).ljust(40, b'\0')


def run(output, factory=ROOT.parent / 'factory', sdk=BUILD / 'local-deps/unitree_sdk2', binary=BUILD / 'g1_ctrl'):
    import mujoco
    manifest = verify()
    interface = manifest['interface']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bridge_binary = build_bridge(output, Path(sdk))
    model, data = robot_model(factory)
    plans, _ = library()
    joint_ids = [model.joint('robot/' + n).id for n in interface['joint_order']]
    qadr, vadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    actuated = model.actuator_trnid[:, 0].tolist()
    act = np.array([actuated.index(j) for j in joint_ids])
    root = model.joint('robot/floating_base_joint')
    qa, va = int(root.qposadr[0]), int(root.dofadr[0])
    ids = np.array(MOTOR_IDS)
    hang = np.array([0, 0, .90, 1, 0, 0, 0.])           # suspended: the feet clear of the floor
    data.qpos[qa:qa + 7] = hang
    data.qpos[qadr] = np.array(interface['default_joint_positions'])
    mujoco.mj_forward(model, data)

    log = open(output / 'g1_ctrl.log', 'w')
    lines = []
    bridge = subprocess.Popen([str(bridge_binary), 'lo'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    ctrl = subprocess.Popen([str(binary), '--config', str(PROFILE), '--network', 'lo'], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=binary.parent)

    def reader():
        for line in ctrl.stdout:
            lines.append((time.monotonic(), line.rstrip()))
            log.write(line)
            log.flush()
    threading.Thread(target=reader, daemon=True).start()

    def seen(pattern, since=0):
        return next((m for _, text in lines[since:] if (m := re.search(pattern, text))), None)

    held, pinned = [], True
    velocities = [np.zeros(3)] * 2
    events, trace = [], []

    def exchange():
        """One 5 ms physics step in step with the wall clock: state out, command in."""
        q = data.qpos[qa + 3:qa + 7]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        v = data.qvel[va:va + 3].copy()
        accel = R.reshape(3, 3).T @ ((v - velocities[-1]) / model.opt.timestep + [0, 0, GRAVITY])
        velocities.append(v)
        state = np.zeros(68, np.float32)
        state[:4], state[4:7], state[7:10] = q, data.qvel[va + 3:va + 6], accel
        state[10 + ids], state[39 + ids] = data.qpos[qadr], data.qvel[vadr]
        bridge.stdin.write(state.tobytes() + remote(*held))
        bridge.stdin.flush()
        reply = bridge.stdout.read(4 + 29 * 5 * 4 + 29)
        count = struct.unpack_from('<I', reply)[0]
        cmd = np.frombuffer(reply, '<f4', 29 * 5, 4).reshape(29, 5)[ids]
        if count:
            model.actuator_gainprm[act, 0], model.actuator_biasprm[act, 1] = cmd[:, 2], -cmd[:, 2]
            model.actuator_biasprm[act, 2] = -cmd[:, 3]
            data.ctrl[act] = cmd[:, 0]
            data.qfrc_applied[vadr] = cmd[:, 4]
        if pinned:
            data.qpos[qa:qa + 7], data.qvel[va:va + 6] = hang, 0
        mujoco.mj_step(model, data)
        return count

    start = time.monotonic()
    step = 0

    def advance(seconds=None, until=None, limit=90.):
        """Run the simulation in real time for `seconds`, or until the controller prints `until`."""
        nonlocal step
        since, t0 = len(lines), step
        while True:
            exchange()
            step += 1
            if step % 4 == 0:
                trace.append(data.qpos[qa:qa + 7].copy())
            delay = start + step * model.opt.timestep - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elapsed = (step - t0) * model.opt.timestep
            if until is None and elapsed >= seconds:
                return True
            if until is not None and seen(until, since):
                return True
            if until is not None and elapsed > limit:
                return False
            if ctrl.poll() is not None:
                raise RuntimeError('g1_ctrl exited: ' + (lines[-1][1] if lines else ''))

    def press(modifier, button):
        # As a hand does it: the shoulder button first (the SDK low-passes L2/R2), then the key.
        held[:] = [modifier]
        advance(.2)
        held[:] = [modifier, button]
        advance(.1)
        held.clear()
        advance(.1)

    def type_distance(text):
        ctrl.stdin.write(text + '\n')
        ctrl.stdin.flush()

    def pose():
        q = data.qpos
        return np.array([q[qa], q[qa + 1], math.atan2(2 * (q[qa + 3] * q[qa + 6] + q[qa + 4] * q[qa + 5]),
                                                       1 - 2 * (q[qa + 5] ** 2 + q[qa + 6] ** 2))])

    def walk(label, button, expect_m):
        """Press the walk button; measure the pelvis from entry to standing still."""
        before, since = pose(), len(lines)
        press(*button)
        ok = advance(until=r'standing still', limit=expect_m / .4 + 25)
        after = pose()
        d = after[:2] - before[:2]
        along = math.cos(before[2]) * d[0] + math.sin(before[2]) * d[1]
        across = -math.sin(before[2]) * d[0] + math.cos(before[2]) * d[1]
        announced = seen(r'walking ([0-9.]+) m', since)
        events.append({'walk': label, 'expected_m': expect_m, 'announced_m': float(announced.group(1)) if announced else None,
                       'settled': bool(ok), 'walked_m': along, 'sideways_m': across,
                       'error_m': math.hypot(along - expect_m, across), 'pelvis_height_m': float(data.qpos[qa + 2])})

    report = {'hardware_tested': False, 'dds': 'CycloneDDS over lo, domain 0', 'failure': None}
    try:
        assert advance(until=r'FSM: Start Passive', limit=30), 'the controller did not connect'
        advance(1.)
        press('LT', 'up')                                   # Passive -> FixStand, hanging
        advance(3.5)
        # Lowered until the feet carry the weight, still held by the hoist: FixStand is an
        # open-loop pose and does not balance.  Released once Velocity has blended in.
        hang[2] = .76
        advance(.5)
        press('RB', 'X')                                    # FixStand -> Velocity
        advance(.7)
        pinned = False
        advance(3.)
        events.append({'velocity_standing_height_m': float(data.qpos[qa + 2])})
        type_distance('1.5')
        advance(.3)
        press('RT', 'up')                                   # +1.0 -> 2.5 m
        press('RT', 'left')                                 # -0.1 -> 2.4 m
        type_distance('35')                                 # refused: outside the library
        type_distance('hello')                              # refused: not a distance
        advance(.3)
        walk('goal-conditioned from Velocity', ('RB', 'Y'), 2.4)
        type_distance('1')
        advance(.3)
        walk('goal-conditioned again', ('RB', 'Y'), 1.0)
        walk('adapter from the goal-conditioned policy', ('RB', 'A'), 1.0)
        press('RB', 'X')                                    # back to Velocity
        advance(3.)
        type_distance('0.1')
        advance(.3)
        since = len(lines)
        press('RB', 'A')                                    # the adapter's shortest walk is 0.3 m: refused
        advance(2.)
        events.append({'short_walk_refused': bool(seen(r'refused', since)),
                       'back_in_velocity': bool(seen(r'to Velocity', since)),
                       'standing_height_m': float(data.qpos[qa + 2])})
        press('LT', 'B')                                    # Passive
        advance(.5)
    except Exception as error:                              # report what happened, then stop the processes
        report['failure'] = repr(error)
    finally:
        ctrl.send_signal(2)
        try:
            ctrl.wait(10)
        except subprocess.TimeoutExpired:
            ctrl.kill()
        bridge.stdin.close()
        bridge.wait(5)
        log.close()
    text = [t for _, t in lines]
    report.update(events=events, transitions=[t.split('FSM: ')[1] for t in text if 'FSM: Change state' in t],
                  distance_messages=[t.split('] ', 2)[-1] for t in text if 'Walk distance' in t or 'outside the planned' in t or 'Not a distance' in t],
                  walk_messages=[t.split('] ', 2)[-1] for t in text if 'ParametrizedWalk' in t],
                  errors=[t for t in text if '[error]' in t or '[critical]' in t])
    np.savez_compressed(output / 'trajectory.npz', pelvis=np.asarray(trace), dt=.02)
    (output / 'loopback.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--factory-root', type=Path, default=ROOT.parent / 'factory')
    a = parser.parse_args()
    run(a.output, a.factory_root)
