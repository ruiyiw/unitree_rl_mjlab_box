"""Offline access to the human-walk controller g1_ctrl runs (the same C++ header, no DDS)."""
import ctypes
import json
import platform
import struct
import subprocess
from pathlib import Path

import numpy as np

from unitree_parametrized_skills.export import digest
from unitree_parametrized_skills.runtime import telemetry  # noqa: F401  (the same sensor vector)

from . import FRAME, MAGIC, OBSERVATION, PACKAGE, REFERENCE, ROOT, TELEMETRY


def verify(package=PACKAGE):
    """The package is the one that was exported (and, in place, so is the archived training source)."""
    manifest = json.loads((package / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(package / name) != expected:
            raise ValueError(f'Human-walk artifact changed: {name}')
    for name, expected in manifest.get('source_files', {}).items():
        if digest(REFERENCE / name) != expected:
            raise ValueError(f'Training source changed: {name}')
    return manifest


def reference(package=PACKAGE):
    """reference.bin as frames (length, 78)."""
    data = (package / 'reference.bin').read_bytes()
    assert data[:8] == MAGIC
    length, width = struct.unpack_from('<II', data, 8)
    assert width == FRAME
    return np.frombuffer(data, '<f4', offset=16).reshape(length, FRAME)


def build_core(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    arch = {'x86_64': 'x64', 'aarch64': 'aarch64'}[platform.machine()]
    ort = ROOT / f'deploy/thirdparty/onnxruntime-linux-{arch}-1.22.0'
    out = directory / 'libhuman_walk_core.so'
    subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-shared', '-fPIC',
                    str(ROOT / 'deploy/tests/human_walk_core.cpp'),
                    '-I' + str(ROOT / 'deploy/robots/g1_23dof/include'), '-I' + str(ort / 'include'),
                    str(ort / 'lib/libonnxruntime.so.1.22.0'), '-Wl,-rpath,' + str(ort / 'lib'),
                    '-o', str(out)], check=True)
    return out


class Core:
    def __init__(self, lib, package=PACKAGE):
        self.lib = ctypes.CDLL(str(lib))
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        L = self.lib
        L.hw_error.restype = ctypes.c_char_p
        L.hw_create.argtypes, L.hw_create.restype = [ctypes.c_char_p], ctypes.c_void_p
        L.hw_destroy.argtypes = [ctypes.c_void_p]
        L.hw_start.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
        L.hw_previous.argtypes = [ctypes.c_void_p, f32]
        L.hw_step.argtypes = [ctypes.c_void_p, f32, f32, f32, f32, f32, i32, f32]
        self.handle = L.hw_create(str(package).encode())
        if not self.handle:
            raise RuntimeError(L.hw_error().decode())

    def start(self, hold_s=0., scale=1.):
        if self.lib.hw_start(self.handle, hold_s, scale):
            raise RuntimeError(self.lib.hw_error().decode())

    def previous(self, action):
        """Set the correction the actor reads back at the next step (the offline comparison with training)."""
        self.lib.hw_previous(self.handle, np.ascontiguousarray(action, dtype=np.float32))

    def step(self, sensors):
        """-> observation (145), correction (23), joint targets (23), arm torque (23),
        [clock, ended, done], [pelvis x y z, their rates, heading]."""
        sensors = np.ascontiguousarray(sensors, dtype=np.float32)
        if sensors.shape != (TELEMETRY,):
            raise ValueError(f'Expected {TELEMETRY} telemetry values')
        obs, action, target, torque = (np.empty(n, np.float32) for n in (OBSERVATION, 23, 23, 23))
        state, estimate = np.empty(3, np.int32), np.empty(7, np.float32)
        if self.lib.hw_step(self.handle, sensors, obs, action, target, torque, state, estimate):
            raise RuntimeError(self.lib.hw_error().decode())
        return obs, action, target, torque, state, estimate

    def close(self):
        if self.handle:
            self.lib.hw_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
