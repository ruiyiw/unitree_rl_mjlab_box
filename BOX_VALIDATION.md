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

The test workstation used an ignored dependency cache at
`deploy/robots/g1_23dof/build/local-deps`. All runtime libraries resolve, with no
`/tmp` runtime search paths. This cache is not shipped in Git; on the controller
computer, use the normal installed human-push prerequisites and the direct
CMake commands in [BOX.md](BOX.md#build-once).

The actual executable now supports `--check-box-config`. All six profiles passed
its offline checks and inference, including the five-actor sequence. Tests also
confirm it rejects an enabled absent motor, incorrect gains, incorrect initial
pose, and a non-Passive first state. The same check runs before DDS initialization
on a real box launch. It checks software configuration only; it never opens DDS
or issues motor commands in offline mode.

Only x86-64 was compiled and executed here. The included aarch64 ONNX library
does not constitute an aarch64 controller build. Build natively on the target computer. Optional dependency and profile checks
are listed below; Linux/glibc/libstdc++ compatibility also matters.

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

## Optional development tools

These commands are for offline checks and simulation. Hardware deployment uses
only the CMake and `g1_ctrl` commands in [BOX.md](BOX.md).

From the repository root, these direct C++ diagnostics need no Python:

```bash
deploy/robots/g1_23dof/build/g1_ctrl --check-box-config deploy/robots/g1_23dof/config/box_sequence.yaml
ldd deploy/robots/g1_23dof/build/g1_ctrl
```

The first command checks configuration and inference without opening DDS or
issuing motor commands. The second lists shared-library dependencies; none
should say `not found`.

The optional `scripts/box.py` helpers provide these conveniences:

| Helper | Purpose |
|---|---|
| `build` | Calls CMake; can use the test workstation's local dependency cache. |
| `preflight` | Runs the C++ offline check for all six profiles. |
| `launch --stage <stage> --network <interface>` | Verifies package hashes, selects a profile and executes `g1_ctrl`. |

`build --core-only` builds just the offline inference library. `launch --dry-run`
prints the command without connecting to a robot. These helpers do not change
the hardware controller.

### Python environment and checks

Use the separate mjlab 1.6 environment from the balance integration. On this
machine, `../humanoid/.venv/bin/python` supplies it. A separate installation can
use `requirements-box.txt`. Do not install the stock mjlab 1.2 `setup.py` into
that environment. Run these commands from this repository root:

```bash
../humanoid/.venv/bin/python scripts/box.py verify
../humanoid/.venv/bin/python scripts/box.py validate --output logs/box_validation_new
../humanoid/.venv/bin/python -m pytest -q tests/test_box_deploy.py
```

Validation compiles the actual C++ deployment core with g++, compares all five
actors with their selected PyTorch weights, and compares every clock/handoff
against the archived Python sequence controller. No robot connection is made.
Measured results and limitations are recorded above.


### Standalone CPU playback

CPU MuJoCo playback uses the C++ deployment controller and the packaged source
scene. It sets the robot and box to the selected stage's reference start once,
then runs continuous physics. Add `--viewer` on a machine with a display:

```bash
../humanoid/.venv/bin/python scripts/box.py play --stage go-to --output logs/box_goto
../humanoid/.venv/bin/python scripts/box.py play --stage take --output logs/box_take
../humanoid/.venv/bin/python scripts/box.py play --stage carry --output logs/box_carry
../humanoid/.venv/bin/python scripts/box.py play --stage put-down --output logs/box_putdown
../humanoid/.venv/bin/python scripts/box.py play --stage return --output logs/box_return
```

Output directories must be new. `--seconds 2` selects a short smoke test. Results
include `simulation.json` and `trajectory.npz`. This runner reports controller
completion, not the source evaluator's physical box-success criteria.

### Full-sequence CPU playback and GPU evaluation

Run the following Python commands from the repository root, using the separate
mjlab environment. They are simulation tools, not hardware-launch requirements.

CPU playback runs the actual C++ deployment controller against the source scene:

```bash
../humanoid/.venv/bin/python scripts/box.py play \
  --stage sequence --seconds 125 --output logs/box_sequence_cpu
# Add --viewer for interactive playback on a machine with a display.
```

The original batched mjlab evaluator is also preserved. It uses the original
PyTorch bundle, continuous physics, sensor-only switches and physical success
criteria, with plain PD and no floating-base gravity assistance:

```bash
CUDA_VISIBLE_DEVICES=0 ../humanoid/.venv/bin/python scripts/box.py evaluate \
  --output logs/box_sequence_gpu --num-envs 1024 --seed 120001
# For a rendered single rollout: --num-envs 1 --record
```

This command requires working CUDA. The bundled source results report **45.90%**
full-sequence completion in the nominal 1,024-trial evaluation and **41.02%** in
the repeat. Those are historical source results, not new measurements from this
integration. The CPU runner does not implement all those physical scoring
criteria. The results above distinguish current controller checks from source evaluation
and hardware validation.

## Reproduce the import

The package is self-contained; execution does not need `humanoid-full`. The
archived reference runtime and robot assets live under `unitree_box/reference`,
with original source hashes in `unitree_box/source_manifest.json` and the robot
asset license preserved. To regenerate from the selected source into a new
policy directory:

```bash
../humanoid/.venv/bin/python -m unitree_box.export \
  --source-root ../humanoid-full --output /tmp/box_reexport
```

This also refreshes the local source snapshot and six launch profiles. The
importer pins the selected sequence hash, checks checkpoint/actor-state identity,
source controller/configuration and reference hashes, audits all five model
contracts, and generates C++ constants. New policies require a deliberate new
import and rebuild; swapping a shape-compatible ONNX file is rejected.
