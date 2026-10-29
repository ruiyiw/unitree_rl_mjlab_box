# Box integration validation — 2026-10-29

The five selected actors and coordinated sequence are integrated with the same
23-DoF G1 deployment executable as human-push. The full x86-64 executable builds,
CPU controller checks pass, and a native MuJoCo sequence reaches its final sensor
hold. Physical hardware and batched CUDA performance remain untested here.

## Frozen identities and interface

All selected checkpoints and ONNX exports are byte-identical copies. The bundle
is the original `five-stage-composite-proprio-v3`, including a distinct return
actor and `simulation_actuation_mode=none`. Import also verifies each checkpoint's
actor tensors against the bundle, all reference tensors against their source
banks, and controller/configuration hashes.

All five source model contracts match the generated C++ contract: 23 joints,
75 actor inputs, 23 outputs, source joint/action order, nominal observation pose,
PD gains, action scales and 20 ms policy period. All compiled body gravity
compensation coefficients are zero. Tests compare the motor mapping and nominal
pose with human-push and explicitly check the box-specific shoulder/elbow gains.

The manifests cover the packaged artifacts, original runtime and robot assets.
`python scripts/box.py verify` checks their SHA256 identities. C++ additionally
rejects mismatched ONNX/reference bytes using compiled fingerprints.

## Actual C++ inference and switching

The same `BoxPolicy.h` used by the hardware FSM was compiled into an offline
shared library using g++ and bundled ONNX Runtime 1.22.0. Each actor was checked
on 128 sensor samples against its frozen PyTorch actor:

| Stage | Maximum actor error | Maximum motor-target error |
|---|---:|---:|
| Go-to | 2.3842e-6 | 1.0729e-6 rad |
| Take | 1.0729e-6 | 4.7684e-7 rad |
| Carry | 1.9073e-6 | 8.3447e-7 rad |
| Put-down | 8.3447e-7 | 3.5763e-7 rad |
| Return | 1.4305e-6 | 7.1526e-7 rad |

Maximum observation error was 5.9605e-7. The test checks unclipped action memory,
next-frame reference targets and zeroed history after reset.

A 3,329-step synthetic sensor replay compared every C++ stage, reference clock
and completion flag directly with the archived
`SequenceCommand._update_command`, using its original settling function. All
four handoffs matched. Interrupted dwell windows, the fresh-frame behavior and
residual history across switches were exercised. Synthetic settling is a logic
regression test, not a physical rollout.

Separate tests reject swapped actors, corrupt references, invalid stage IDs,
invalid IMU and non-finite observations. They verify a non-settling stage times
out and an isolated stage holds its final actor instead of advancing.

## Native CPU simulation

The CPU runner uses the source geometry, plain-PD model, full source MuJoCo
solver settings and 5 ms × 4 simulation steps per C++ controller action. Each
standalone stage ran for two simulated seconds without an exception or tilt/
height stop, starting from its own reference robot/box state.

One continuous sequence ran for 125 simulated seconds, with these handoffs:

| Entered stage | Time |
|---|---:|
| Go-to | 0.00 s |
| Take | 8.80 s |
| Carry | 25.72 s |
| Put-down | 34.54 s |
| Return | 55.56 s |

It reached final sensor completion and continued holding through the end of the
run. There were no stage resets. This runner does **not** score all original
physical criteria such as supported release and box/robot endpoint position;
this result must not be reported as physical box-task success or a success rate.

## Build and regression checks

The full `deploy/robots/g1_23dof/build/g1_ctrl` compiled and linked successfully
with GNU 15.2.0, CMake 4.2.3, bundled ONNX Runtime 1.22.0 and Unitree SDK2 commit
`d1abf001220d3e5f59e00a0c4aaa595e00089a4a`. `g1_ctrl --help` passed. All six profiles were loaded and checked by the executable without opening DDS.

A follow-up deployment audit found the initial test executable depended on
`/tmp/box-build-deps`. That dependency has been removed: prerequisites now reside
in `deploy/robots/g1_23dof/build/local-deps`, and the documented Python build
wrapper rebuilds against them. No system packages were installed. `ldd` resolves
all libraries and the rebuilt executable's runtime search paths contain no
`/tmp` directories. The dependency cache is local and ignored by Git; a fresh
checkout requires normal installed prerequisites.

The actual executable now supports `--check-box-config`. All six profiles passed
its offline checks and inference, including the five-actor sequence. Tests also
confirm it rejects an enabled absent motor, incorrect gains, incorrect initial
pose, and a non-Passive first state. The same check runs before DDS initialization
on a real box launch. It checks software configuration only; it never opens DDS
or issues motor commands in offline mode.

Only x86-64 was compiled and executed here. The included aarch64 ONNX library
does not constitute an aarch64 controller build. Build natively and run preflight
and `ldd` on the target computer; Linux/glibc/libstdc++ compatibility also matters.

```bash
../humanoid/.venv/bin/python -m pytest -q \
  tests/test_box_deploy.py tests/test_human_push_deploy.py \
  tests/test_balance_port.py tests/test_human_push.py
# 33 passed
```

The stock controller profile, human-push profile and balance environment were
not changed. The G1 executable registers the additional BoxPolicy FSM type;
only the six new profiles activate it.

## Evidence and remaining validation

Machine-readable results are stored in:

- `unitree_box/validation/controller.json`
- `unitree_box/validation/simulation.json`
- `unitree_box/validation/build.json`
- `unitree_box/validation/human_push_comparison.json`

The archived reference evaluation reports 45.90% nominal and 41.02% repeat
full-sequence completion. Those measurements are copied source evidence. CUDA
was unavailable here, so the preserved batched evaluator was not rerun. The CPU
rollout uses different contact numerics and is not a replacement for that audit.

No DDS transport, motor firmware, physical box contact or real-robot execution
was tested. Hardware switching measures proprioceptive settling and cannot
confirm box placement or global route accuracy. Follow the staged startup and
physical-layout instructions in [BOX.md](BOX.md) and [BOX_SEQUENCE.md](BOX_SEQUENCE.md).

## Direct comparison with the working human-push setup

The user reports that following HUMAN_PUSH.md worked on the humanoid. A direct
comparison confirms the same CMake/executable, DDS domain/interface selection,
low-state acquisition and low-command publishing path, 23 motor mapping,
`mode_machine=4` / `mode_pr=0`, Passive configuration, FixStand duration and
controller buttons. The human-push state/profile have not been modified.

A new regression feeds 128 identical telemetry/previous-action samples to both
actual C++ observation builders and compares all 75 values. Another compares all
six box profiles with human-push and identifies the exact gain/scale differences.

The differences are box reference-start poses, moving references and unclipped
residual memory, eight shoulder/elbow gains/scales, two hip-roll scales, sequence
switching, stricter deadline faults, stage timeout and preflight. The source
box model's hip-roll effort limit is ±139 Nm versus ±88 Nm for human-push; these
are simulator limits, not new firmware settings. This difference is preserved
from the selected training setup and does not establish the real motor's capacity.

The direct C++ launch procedure, reusing the user's working interface and
installed dependencies, is now documented in BOX.md. The Python/mjlab environment
is not required for that hardware launch. This comparison confirms setup
consistency; the user's successful human-push run is not a measured box trial.
