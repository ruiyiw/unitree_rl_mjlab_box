"""Offline access to the walk controller g1_ctrl runs (the same C++ header, no DDS)."""
import ctypes
import json
import platform
import struct
import subprocess
from pathlib import Path

import numpy as np

from . import FRAME, MOTOR_IDS, OBSERVATION, PACKAGE, POLICIES, REFERENCE, ROOT, TELEMETRY
from .export import MAGIC, digest


def verify(package=PACKAGE):
    """The package is the one that was exported (and, in place, so is the archived training source)."""
    manifest = json.loads((package / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(package / name) != expected:
            raise ValueError(f'Parametrized-walk artifact changed: {name}')
    for name, expected in manifest.get('source_files', {}).items():
        if digest(REFERENCE / name) != expected:
            raise ValueError(f'Training source changed: {name}')
    return manifest


def library(package=PACKAGE):
    """plans.bin as {millimetres: frames (length, 28)} and each plan's end joint pose."""
    data = (package / 'plans.bin').read_bytes()
    assert data[:8] == MAGIC
    count, width = struct.unpack_from('<II', data, 8)
    assert width == FRAME
    table = np.frombuffer(data, np.dtype([('mm', '<u4'), ('length', '<u4'), ('first', '<u4'), ('end', '<f4', 23)]), count, 16)
    frames = np.frombuffer(data, '<f4', offset=16 + table.nbytes).reshape(-1, FRAME)
    return {int(t['mm']): frames[t['first']:t['first'] + t['length']] for t in table}, {int(t['mm']): t['end'] for t in table}


def build_core(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    arch = {'x86_64': 'x64', 'aarch64': 'aarch64'}[platform.machine()]
    ort = ROOT / f'deploy/thirdparty/onnxruntime-linux-{arch}-1.22.0'
    out = directory / 'libparametrized_walk_core.so'
    subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-shared', '-fPIC',
                    str(ROOT / 'deploy/tests/parametrized_walk_core.cpp'),
                    '-I' + str(ROOT / 'deploy/robots/g1_23dof/include'), '-I' + str(ort / 'include'),
                    str(ort / 'lib/libonnxruntime.so.1.22.0'), '-Wl,-rpath,' + str(ort / 'lib'),
                    '-o', str(out)], check=True)
    return out


def telemetry(quaternion, gyro, accelerometer, joint_pos, joint_vel):
    """The controller's sensor vector from joint-ordered values (absent motor slots stay zero)."""
    t = np.zeros(TELEMETRY, np.float32)
    t[:4], t[4:7], t[7:10] = quaternion, gyro, accelerometer
    t[10 + np.array(MOTOR_IDS)] = joint_pos
    t[39 + np.array(MOTOR_IDS)] = joint_vel
    return t


class Core:
    def __init__(self, lib, policy, package=PACKAGE):
        self.lib = ctypes.CDLL(str(lib))
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        L = self.lib
        L.walk_error.restype = ctypes.c_char_p
        L.walk_create.argtypes, L.walk_create.restype = [ctypes.c_char_p, ctypes.c_int], ctypes.c_void_p
        L.walk_destroy.argtypes = [ctypes.c_void_p]
        L.walk_plans.argtypes, L.walk_plans.restype = [ctypes.c_void_p], ctypes.c_int
        L.walk_plan_metres.argtypes, L.walk_plan_metres.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_float
        L.walk_start.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_void_p, ctypes.c_float]
        L.walk_start.restype = ctypes.c_float
        L.walk_step.argtypes = [ctypes.c_void_p, f32, f32, f32, f32, i32, f32]
        L.walk_select.argtypes, L.walk_select.restype = [ctypes.c_char_p, ctypes.c_float], ctypes.c_float
        L.walk_select_step.argtypes, L.walk_select_step.restype = [ctypes.c_float], ctypes.c_float
        self.package = str(package).encode()
        index = policy if isinstance(policy, int) else tuple(POLICIES).index(policy)
        self.handle = L.walk_create(self.package, index)
        if not self.handle:
            raise RuntimeError(L.walk_error().decode())

    def distances(self):
        return [round(float(self.lib.walk_plan_metres(self.handle, i)), 3) for i in range(self.lib.walk_plans(self.handle))]

    def start(self, metres, hold_s=0., previous=None, scale=1.):
        """Begin a walk; returns the planned distance that will be walked."""
        if previous is not None:
            previous = np.ascontiguousarray(previous, dtype=np.float32)
            assert previous.shape == (23,)
            previous = previous.ctypes.data_as(ctypes.c_void_p)
        planned = self.lib.walk_start(self.handle, metres, hold_s, previous, scale)
        if planned < 0:
            raise RuntimeError(self.lib.walk_error().decode())
        return round(float(planned), 3)

    def step(self, sensors):
        """-> observation (109), correction (23), joint targets (23), [clock, ended, done], [x, y, heading]."""
        sensors = np.ascontiguousarray(sensors, dtype=np.float32)
        if sensors.shape != (TELEMETRY,):
            raise ValueError(f'Expected {TELEMETRY} telemetry values')
        obs, action, target = (np.empty(n, np.float32) for n in (OBSERVATION, 23, 23))
        state, estimate = np.empty(3, np.int32), np.empty(3, np.float32)
        if self.lib.walk_step(self.handle, sensors, obs, action, target, state, estimate):
            raise RuntimeError(self.lib.walk_error().decode())
        return obs, action, target, state, estimate

    def select(self, metres):
        """What the operator's selector answers to a request: the planned distance, or an error."""
        planned = self.lib.walk_select(self.package, metres)
        if planned < 0:
            raise ValueError(self.lib.walk_error().decode())
        return round(float(planned), 3)

    def select_step(self, delta):
        return round(float(self.lib.walk_select_step(delta)), 3)

    def close(self):
        if self.handle:
            self.lib.walk_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
