"""Build, validate and launch the frozen 23-DoF G1 box policies."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unitree_box import ORDER,POLICY_DIR,ROOT


def main():
    build_dir=ROOT/'deploy/robots/g1_23dof/build'
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    build=commands.add_parser('build',help='Build the G1 controller, or just its offline core')
    build.add_argument('--core-only',action='store_true')
    validate=commands.add_parser('validate',help='CPU parity with all five PyTorch actors and source switching')
    validate.add_argument('--output',type=Path,required=True)
    validate.add_argument('--samples',type=int,default=256)
    play=commands.add_parser('play',help='CPU MuJoCo playback using the C++ controller; no DDS')
    play.add_argument('--stage',choices=(*ORDER,'sequence'),default='sequence')
    play.add_argument('--output',type=Path,required=True)
    play.add_argument('--seconds',type=float,default=125.)
    play.add_argument('--viewer',action='store_true')
    evaluate=commands.add_parser('evaluate',help='Preserved full-sequence GPU evaluation with physical criteria')
    evaluate.add_argument('--output',type=Path,required=True)
    evaluate.add_argument('--num-envs',type=int,default=64)
    evaluate.add_argument('--seed',type=int,default=120001)
    evaluate.add_argument('--record',action='store_true')
    launch=commands.add_parser('launch',help='Launch the DDS G1 controller with one of six dedicated profiles')
    launch.add_argument('--stage',choices=(*ORDER,'sequence'),default='sequence')
    launch.add_argument('--network',required=True)
    launch.add_argument('--dry-run',action='store_true',help='Print the command without connecting to DDS')
    preflight=commands.add_parser('preflight',help='Load and validate C++ profiles and actors without opening DDS')
    preflight.add_argument('--stage',choices=(*ORDER,'sequence','all'),default='all')
    commands.add_parser('verify',help='Verify all packaged artifact and source hashes')
    a=p.parse_args()
    from unitree_box.runtime import verify
    verify()
    if a.command=='verify':print('All box artifacts and reference sources match their manifest.');return
    if a.command=='build':
        if a.core_only:
            from unitree_box.runtime import build_core
            print(build_core(build_dir));return
        from unitree_box.build import build_controller
        build_controller(build_dir)
    elif a.command=='validate':
        if not 1<=a.samples<=256:p.error('--samples must be in 1..256 (within the shortest stage timeout)')
        from unitree_box.validate import validate
        validate(a.output,a.samples)
    elif a.command=='play':
        if not 0<a.seconds<=130:p.error('--seconds must be in (0,130]')
        from unitree_box.simulate import play
        play(a.stage,a.output,a.seconds,a.viewer)
    elif a.command=='evaluate':
        import torch
        if a.num_envs<1:p.error('--num-envs must be positive')
        if a.record and a.num_envs!=1:p.error('--record requires --num-envs 1')
        if not torch.cuda.is_available():p.error('The preserved evaluation requires a working CUDA GPU; use play for a CPU smoke test.')
        if a.output.exists():p.error('--output must be a new directory')
        from unitree_box.runtime import reference_path
        reference_path()
        from five_teacher_sequence import evaluate
        evaluate(POLICY_DIR/'policy.pt',a.output,a.num_envs,a.seed,a.record)
    elif a.command=='preflight':
        binary=build_dir/'g1_ctrl'
        if not binary.is_file():p.error('Build g1_ctrl first with scripts/box.py build')
        for stage in (*ORDER,'sequence') if a.stage=='all' else (a.stage,):
            cfg=ROOT/f'deploy/robots/g1_23dof/config/box_{stage}.yaml'
            subprocess.run([str(binary),'--check-box-config',str(cfg)],check=True)
    elif a.command=='launch':
        binary=build_dir.resolve()/'g1_ctrl'
        cfg=ROOT/f'deploy/robots/g1_23dof/config/box_{a.stage}.yaml'
        argv=[str(binary),'--config',str(cfg),'--network',a.network]
        if a.dry_run:
            import shlex
            print(shlex.join(argv));return
        if not binary.is_file():p.error('Build g1_ctrl first with scripts/box.py build')
        os.chdir(ROOT/'deploy/robots/g1_23dof')
        os.execv(str(binary),argv)


if __name__=='__main__':main()
