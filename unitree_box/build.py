"""Build with installed prerequisites or this machine's isolated dependency cache."""
import os
import platform
import shutil
import subprocess
from pathlib import Path
from . import ROOT


def build_controller(build_dir):
    build_dir = Path(build_dir).resolve()
    local = build_dir / 'local-deps'
    prefix = local / 'prefix/usr'
    sdk = local / 'unitree_sdk2'
    cmake = prefix / 'bin/cmake'
    env = dict(os.environ)
    args = []
    if cmake.is_file():
        # This cache is a local build artifact, not part of the policy package.
        arch = platform.machine()
        multiarch = {'x86_64': 'x86_64-linux-gnu', 'aarch64': 'aarch64-linux-gnu'}[arch]
        lib = prefix / 'lib' / multiarch
        env['LD_LIBRARY_PATH'] = str(lib) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        includes = [prefix/'include', prefix/'include/eigen3', sdk/'include',
                    sdk/'thirdparty/include', sdk/'thirdparty/include/ddscxx']
        libs = [sdk/'lib'/arch, sdk/'thirdparty/lib'/arch, lib]
        # The staged CMake is 4.2.3. Clear the old cache so no /tmp paths survive.
        args = ['--fresh', '-DCMAKE_PREFIX_PATH=' + str(prefix),
                '-DZLIB_INCLUDE_DIR=' + str(prefix/'include'),
                '-DZLIB_LIBRARY=' + str(lib/'libz.a'),
                '-DCMAKE_CXX_FLAGS=' + ' '.join('-I'+str(p) for p in includes),
                '-DCMAKE_EXE_LINKER_FLAGS=' + ' '.join('-L'+str(p)+' -Wl,-rpath,'+str(p) for p in libs)]
    else:
        found = shutil.which('cmake')
        if not found:
            raise RuntimeError('CMake is missing. Install the C++ prerequisites in BOX.md; --core-only only needs g++.')
        cmake = Path(found)
    subprocess.run([str(cmake), '-S', str(ROOT/'deploy/robots/g1_23dof'),
                    '-B', str(build_dir), *args], env=env, check=True)
    subprocess.run([str(cmake), '--build', str(build_dir), '-j4'], env=env, check=True)
