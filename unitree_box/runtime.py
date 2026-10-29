"""Build and call the deployment core without DDS or a robot connection."""
import ctypes
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from . import POLICY_DIR, REFERENCE, ROOT


def reference_path():
    # Source modules use sibling imports. Only the dedicated entry point installs
    # this path; the stock mjlab 1.2 task registry is never imported.
    sys.path.insert(0, str(REFERENCE))


def verify():
    from .export import digest
    manifest=json.loads((POLICY_DIR/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if digest(POLICY_DIR/name)!=expected: raise ValueError(f'Box artifact changed: {name}')
    for name,expected in json.loads((ROOT/'unitree_box/source_manifest.json').read_text()).items():
        if digest(REFERENCE.parent/name)!=expected: raise ValueError(f'Box reference runtime changed: {name}')
    return manifest


def build_core(directory):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    arch={'x86_64':'x64','aarch64':'aarch64'}[platform.machine()]
    ort=ROOT/f'deploy/thirdparty/onnxruntime-linux-{arch}-1.22.0'
    library=directory/'libbox_core.so'
    subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC',
        str(ROOT/'deploy/tests/box_core.cpp'),'-I'+str(ROOT/'deploy/robots/g1_23dof/include'),
        '-I'+str(ort/'include'),str(ort/'lib/libonnxruntime.so.1.22.0'),
        '-Wl,-rpath,'+str(ort/'lib'),'-o',str(library)],check=True)
    return library


class Core:
    def __init__(self,library,stage=-1,bundle=POLICY_DIR):
        self.lib=ctypes.CDLL(str(library))
        self.lib.box_create.argtypes=[ctypes.c_char_p,ctypes.c_int]
        self.lib.box_create.restype=ctypes.c_void_p
        self.lib.box_error.restype=ctypes.c_char_p
        self.lib.box_destroy.argtypes=[ctypes.c_void_p]
        self.lib.box_reset.argtypes=[ctypes.c_void_p]
        ptr=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
        ints=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        self.lib.box_step.argtypes=[ctypes.c_void_p,ptr,ptr,ptr,ptr,ints]
        self.handle=self.lib.box_create(str(bundle).encode(),stage)
        if not self.handle: raise RuntimeError(self.lib.box_error().decode())

    def step(self,telemetry):
        telemetry=np.ascontiguousarray(telemetry,dtype=np.float32)
        if telemetry.shape!=(65,):raise ValueError('Expected 65 SDK telemetry values')
        obs,action,target=(np.empty(n,np.float32) for n in (75,23,23))
        state=np.empty(3,np.int32)
        if self.lib.box_step(self.handle,telemetry,obs,action,target,state):
            raise RuntimeError(self.lib.box_error().decode())
        return obs,action,target,state

    def reset(self):self.lib.box_reset(self.handle)

    def close(self):
        if self.handle:self.lib.box_destroy(self.handle);self.handle=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
