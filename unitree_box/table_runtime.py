"""Offline access to the same 376-input controller used by g1_ctrl."""
import ctypes
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
from . import ROOT
from .table_export import PACKAGE


def verify():
    from .export import digest
    manifest = json.loads((PACKAGE / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(PACKAGE / name) != expected:
            raise ValueError(f'Three-policy artifact changed: {name}')
    for name, expected in manifest['source_files'].items():
        if digest(ROOT / 'unitree_box/three_reference' / name) != expected:
            raise ValueError(f'Training source changed: {name}')
    return manifest


def build_core(directory):
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    arch = {'x86_64': 'x64', 'aarch64': 'aarch64'}[platform.machine()]
    ort = ROOT / f'deploy/thirdparty/onnxruntime-linux-{arch}-1.22.0'
    library = directory / 'libtable_box_core.so'
    subprocess.run(['g++', '-std=c++17', '-O2', '-shared', '-fPIC',
                    str(ROOT / 'deploy/tests/table_box_core.cpp'),
                    '-I' + str(ROOT / 'deploy/robots/g1_23dof/include'), '-I' + str(ort / 'include'),
                    str(ort / 'lib/libonnxruntime.so.1.22.0'), '-Wl,-rpath,' + str(ort / 'lib'),
                    '-o', str(library)], check=True)
    return library


class Core:
    def __init__(self, library, stage, package=PACKAGE):
        self.lib = ctypes.CDLL(str(library))
        self.lib.table_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.table_create.restype = ctypes.c_void_p
        self.lib.table_error.restype = ctypes.c_char_p
        self.lib.table_destroy.argtypes = [ctypes.c_void_p]
        ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        ints = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        self.lib.table_reset.argtypes = [ctypes.c_void_p, ptr]
        self.lib.table_step.argtypes = [ctypes.c_void_p, ptr, ptr, ptr, ptr, ptr, ptr, ints]
        self.handle = self.lib.table_create(str(package).encode(), stage)
        if not self.handle:
            raise RuntimeError(self.lib.table_error().decode())

    def step(self, telemetry, command):
        telemetry = np.ascontiguousarray(telemetry, dtype=np.float32)
        command = np.ascontiguousarray(command, dtype=np.float32)
        if telemetry.shape != (65,) or command.shape != (23,):
            raise ValueError('Expected 65 telemetry and 23 position-command values')
        obs, action, target, torque = (np.empty(n, np.float32) for n in (376, 23, 23, 23))
        state = np.empty(2, np.int32)
        if self.lib.table_step(self.handle, telemetry, command, obs, action, target, torque, state):
            raise RuntimeError(self.lib.table_error().decode())
        return obs, action, target, torque, state

    def reset(self, previous=None):
        previous = np.zeros(23, np.float32) if previous is None else np.ascontiguousarray(previous, dtype=np.float32)
        if previous.shape != (23,): raise ValueError('Expected 23 previous actions')
        self.lib.table_reset(self.handle, previous)

    def close(self):
        if self.handle: self.lib.table_destroy(self.handle); self.handle = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
