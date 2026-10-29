# G1 23-DoF box policies

The final five selected actors and their coordinated sequence are bundled in
`deploy/robots/g1_23dof/config/policy/box/v1`. Each actor has **75 inputs and 23
outputs**. The sequence is go-to → take → carry → put-down → return, with a
separate return actor. No retraining or weight/normalizer updates were performed.

| Stage | Selected checkpoint | Packaged actor | Launch profile |
|---|---|---|---|
| Go-to | `navigation/runs/proprio-teacher-scratch-4000-20261028/model_3600.pt` | `go-to/policy.onnx` | `box_go-to.yaml` |
| Take | `physical-none-prefix-precision-take/model_2.pt` | `take/policy.onnx` | `box_take.yaml` |
| Carry | `physical-none-alternate-carry_refine/model_12.pt` | `carry/policy.onnx` | `box_carry.yaml` |
| Put-down | `physical-none-putdown-r3-selected-handoffs/model_200.pt` | `put-down/policy.onnx` | `box_put-down.yaml` |
| Return | `navigation/runs/autoresearch-navigation-r1-shared_full-20261029/model_200.pt` | `return/policy.onnx` | `box_return.yaml` |
| Full sequence | Original selected `policy.pt` bundle | All five actors and references | `box_sequence.yaml` |

Actor paths are relative to the bundled policy directory; profiles live in
`deploy/robots/g1_23dof/config`. Each stage includes its original checkpoint,
byte-identical ONNX actor, source metadata and exported joint reference. The
manifest records full original paths and SHA256 hashes. The original sequence
bundle SHA256 is
`324a76c836b5a8e1f24b53ca93c0d636fae5fcb481bc008cce3d5a8ddbbbeed7`.

## Comparison with human-push and balance

The physical robot and observation ordering match [HUMAN_PUSH.md](HUMAN_PUSH.md)
and [BALANCE.md](BALANCE.md). Their controller parameters are not interchangeable:

| Property | Human-push/balance | Box policies |
|---|---|---|
| Robot | 23-DoF G1 | Same 23 motor IDs and joint order |
| Observations | Gyro 3, projected gravity 3, relative joints 23, joint velocities 23, previous actions 23 | Same layout and nominal observation pose |
| Actor | 256/256 ELU | 512/256/128 ELU; saved normalization included in ONNX |
| Previous action | Clipped to [-1, 1] | Unclipped residual; retained across handoffs |
| Position target | Nominal + scale × action | Stored `qcmd[min(clock+1,last)]` + scale × residual |
| Shoulder/elbow PD | Original standing gains | kp = 40, kd = 2 |
| Shoulder/elbow action scale | 0.438577 rad | 0.3 rad |
| Hip-roll action scale | 0.222002 rad | 0.350661 rad |
| Hip-roll effort limit in the training model | ±88 Nm | ±139 Nm |
| Missed 20 ms period | Reschedule; no explicit inference-deadline fault | Fault to damping/Passive on missed deadline |
| Policy period | 20 ms | 20 ms; source physics is 5 ms × 4 substeps |
| Switching | One recovery actor | Reference end plus continuous proprioceptive settling |

SDK motor IDs are `0..12, 15..19, 22..26`; the six absent joints remain disabled.
The controller requires `mode_machine=4` and uses ankle pitch/roll `mode_pr=0`.
Wrists, legs and waist retain the source box-task gains and scales. All five
compiled model contracts were compared, including joint/action ordering, gains,
scales, nominal pose and effort limits. The box scene uses plain PD, with no
floating-base gravity assistance.

The ONNX signature is float32 `observations[1,75]` → `actions[1,23]`. It requires
no external normalizer, `.onnx.data`, teacher, estimator or privileged actor
inputs. The references and switching controller remain required. C++ checks
compiled artifact fingerprints before loading; `verify` also checks SHA256
hashes of the full package and archived runtime.

## Switching from your working human-push setup

Your reported successful human-push deployment is the hardware-setup baseline.
Box uses the same robot project, CMake target, Unitree SDK2/DDS path, Ethernet
interface selection, `mode_machine=4`, `mode_pr=0`, IMU/motor-state snapshot,
23 motor IDs, and `LowCmd` publisher. Both run synchronous 50 Hz inference inside
the same 1 kHz FSM. Both send position/PD commands with target velocity and
feedforward torque zero, and use immediate damping followed by Passive on a
fault. All six box profiles have exactly the same Passive configuration as
human-push, and use the same three-second FixStand interpolation and buttons.

On the computer that already ran human-push, keep its installed SDK/DDS and
network setup. Build this updated robot project with the same commands in
[HUMAN_PUSH.md](HUMAN_PUSH.md). Then, from `deploy/robots/g1_23dof/build`, change
the profile in the known working launch command:

```bash
# Working human-push command (use your actual interface in both commands):
./g1_ctrl --config ../config/human_push.yaml --network enp5s0

# Offline box loading check; no robot connection:
./g1_ctrl --check-box-config ../config/box_sequence.yaml

# Box sequence, run separately after stopping the other controller:
./g1_ctrl --config ../config/box_sequence.yaml --network enp5s0
```

Use `box_go-to.yaml`, `box_take.yaml`, `box_carry.yaml`, `box_put-down.yaml`, or
`box_return.yaml` for a single stage. The Python wrapper and mjlab environment
are optional for hardware launching; the C++ executable, config directory and
ONNX/reference files perform deployment.

The policy settings must remain those from box training: FixStand goes to that
stage's reference start, the actor adds an **unclipped residual to a moving
reference**, and the arm gains and hip-roll scales differ as shown above.
Copying human-push's gains, pose or action clipping into the box profile would
change the trained controller. The hip-roll effort figures are simulation-model
limits; neither C++ controller programs those limits into motor firmware or
establishes that the real hip-roll motor can produce the larger torque. This is
a training-to-hardware difference that human-push success alone does not verify.

Box also has stricter deadline faults and a stage-settling timeout, plus an
offline profile/artifact check before DDS startup. These are explicit wrapper
differences from human-push, not changes to its network or motor interface.
The human-push FSM and profile themselves remain unchanged.

## Runtime and checks

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
See [BOX_VALIDATION.md](BOX_VALIDATION.md) for measured results and limitations.

## Five separate policies: playback and launch

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

An isolated hardware stage requires its corresponding physical starting setup:

| Stage | Required setup before activating the policy |
|---|---|
| Go-to | Robot at route start facing the source table; box on source table |
| Take | Robot at the source pickup stance; box in its reference location |
| Carry | Box already held between palms; robot at the post-take stance facing the destination |
| Put-down | Box already held; robot at the destination placement stance |
| Return | Box supported on the destination table; arms lowered and robot facing home |

Each profile's FixStand pose is generated from that stage's first reference
joint pose. FixStand does not place or grasp the box, move the floating base to
its starting location, or recreate the incoming residual history. Standalone
entry initializes residual history to zero. The sequence retains history and
physical state across handoffs, so isolated starts are not equivalent to
continuous handoffs.

## Build and launch on the same 23-DoF G1

**Deployment status:** built and checked on this x86-64 workstation; not installed
or tested on a physical G1. The software is available for a supervised hardware
trial after the target computer and physical setup are checked. A controller
computer running aarch64 needs a native aarch64 build; do not copy this x86-64
executable to it. Matching CPU architecture alone is insufficient: the target
must also provide compatible system libraries (including glibc/libstdc++).


Use the same C++ prerequisites as human-push: CMake, C++17 compiler, Unitree SDK2,
CycloneDDS/CycloneDDS-CXX, Boost program_options, yaml-cpp, Eigen, fmt, spdlog and
zlib. The repository includes x86-64 and aarch64 ONNX Runtime libraries.

```bash
cmake -S deploy/robots/g1_23dof -B deploy/robots/g1_23dof/build
cmake --build deploy/robots/g1_23dof/build -j4
```

On this workstation, use the wrapper below. It uses the isolated prerequisites
under `deploy/robots/g1_23dof/build/local-deps`, so it does not require CMake on
PATH or dependencies under `/tmp`. This cache is an ignored local build artifact;
it is not included in a fresh checkout. On a new controller computer, install
the prerequisites above and build there.

```bash
../humanoid/.venv/bin/python scripts/box.py build
../humanoid/.venv/bin/python scripts/box.py preflight
```

`preflight` loads all six profiles into the actual `g1_ctrl` executable, verifies
FSM startup/exit transitions, 23 motor slots, disabled absent joints, gains,
initial poses and actor/reference fingerprints, and runs inference. It does not
initialize DDS or issue motor commands. Direct C++ checking needs no Python:

```bash
cd deploy/robots/g1_23dof/build
./g1_ctrl --check-box-config ../config/box_sequence.yaml
ldd ./g1_ctrl  # Every dependency must resolve; none may say "not found".
```

The real box launch repeats the C++ profile check before initializing DDS.
Passing it confirms software loading, not robot connectivity, correct firmware,
physical contact behavior or successful box placement. The policy files can be
copied between supported architectures; build the executable for the computer
that actually runs it. Recheck dependencies and profiles on that computer.

Keep the robot project's `config` directory alongside its `build` directory.
The wrapper verifies the package and selects one dedicated profile:

```bash
../humanoid/.venv/bin/python scripts/box.py launch --stage go-to --network enp5s0
../humanoid/.venv/bin/python scripts/box.py launch --stage take --network enp5s0
../humanoid/.venv/bin/python scripts/box.py launch --stage carry --network enp5s0
../humanoid/.venv/bin/python scripts/box.py launch --stage put-down --network enp5s0
../humanoid/.venv/bin/python scripts/box.py launch --stage return --network enp5s0
```

Run one controller at a time. Replace `enp5s0` with the robot Ethernet interface;
`--dry-run` prints the command without opening DDS. Direct launch also works:

```bash
cd deploy/robots/g1_23dof/build
./g1_ctrl --config ../config/box_sequence.yaml --network enp5s0
```

Follow the repository's suspended startup/debug-mode procedure. The controller
starts in **Passive**. **L2 + Up** enters the three-second FixStand transition;
after reaching the appropriate setup, **R2 + A** activates the selected policy.
**L2 + B** returns to Passive. Each profile loads only Passive, FixStand and
BoxPolicy; it needs no walking, dance or human-push actor files. At an isolated
stage's sensor completion it holds that stage's final reference with the same
actor until the operator exits.

The FSM applies damping and requests Passive on invalid telemetry/actions,
sensor timeout, control deadline misses, stage-settling timeout or pelvis tilt
beyond 45°. It does not measure physical box success. No physical robot has been
connected or tested by this integration.

For machines without DDS dependencies, this builds only the offline core:

```bash
../humanoid/.venv/bin/python scripts/box.py build --core-only
```

## Full sequence

See [BOX_SEQUENCE.md](BOX_SEQUENCE.md) for the continuous launch, exact switching
rules, layout, and preserved GPU evaluation. This is a coordinated controller
containing five actors, not a separately trained single neural network.

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
