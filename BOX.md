# G1 23-DoF box policies

Five selected actors, each with **75 inputs and 23 outputs**, run on the same
23-DoF G1 and C++ executable as [human-push](HUMAN_PUSH.md). Choose one stage or
run **go-to → take → carry → put-down → return** as a coordinated sequence.

## Build once

Use the same installed C++ prerequisites and robot Ethernet interface as your
working human-push setup. For a new controller computer, install CMake, a C++17
compiler, Unitree SDK2, CycloneDDS/CycloneDDS-CXX, Boost program_options, yaml-cpp,
Eigen, fmt, spdlog and zlib. Build on the computer that will run the controller.

From the repository root:

```bash
cmake -S deploy/robots/g1_23dof -B deploy/robots/g1_23dof/build
cmake --build deploy/robots/g1_23dof/build -j4
cd deploy/robots/g1_23dof/build
```

Keep the robot project's `config` directory, including `config/policy/box/v1`,
alongside `build`. **No Python wrapper or virtual environment is needed for
hardware deployment.**

Any launch below can add `--record ../recordings/box_001` after
`mkdir -p ../recordings`. Use a new session name each time. See
[deployment recording](DEPLOYMENT_RECORDING.md) for offline replay.

## Launch one policy

Run **one** of the following commands from `deploy/robots/g1_23dof/build`.
Replace `enp5s0` with the interface used by your working human-push command.
Stop the current controller before launching another.

### Go-to

```bash
./g1_ctrl --config ../config/box_go-to.yaml --network enp5s0
```

### Take

```bash
./g1_ctrl --config ../config/box_take.yaml --network enp5s0
```

### Carry

```bash
./g1_ctrl --config ../config/box_carry.yaml --network enp5s0
```

### Put-down

```bash
./g1_ctrl --config ../config/box_put-down.yaml --network enp5s0
```

### Return

```bash
./g1_ctrl --config ../config/box_return.yaml --network enp5s0
```

### Full sequence

```bash
./g1_ctrl --config ../config/box_sequence.yaml --network enp5s0
```

The sequence switches between all five actors automatically. See
[BOX_SEQUENCE.md](BOX_SEQUENCE.md) for its physical layout and switching rules.

## Activate with the same controls as human-push

Follow the same suspended startup and debug-mode procedure as human-push.
Each profile starts in Passive:

1. **L2 + Up**: move to the selected stage's starting joint pose over three seconds.
2. **R2 + A**: activate the selected policy after FixStand and physical setup are ready.
3. **L2 + B**: return to Passive.

A standalone stage holds its final reference after sensor completion until you
exit. The full sequence holds the final return reference after completion.
The executable checks the profile and policy files automatically before DDS
startup; no separate preflight command is required.

## Starting setup

The full sequence starts with the go-to setup. For a standalone stage, prepare
that stage's setup before activation:

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

## What matches human-push

The CMake target, SDK2/DDS connection, 23 motor IDs, Passive state, three-second
FixStand transition and gamepad controls are the same. Box uses its own trained
poses, gains and action handling, so keep the supplied box profiles:

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
compiled artifact fingerprints before loading.

The hip-roll effort figures are simulation-model limits; the controller does
not program them into motor firmware or establish real motor torque capacity.
The human-push profile and FSM are unchanged.

The C++ build and offline checks passed. Real-robot box execution has not been
tested here. See [BOX_VALIDATION.md](BOX_VALIDATION.md) for evidence and optional
simulation/development commands.

## Selected artifacts

The package is self-contained in `deploy/robots/g1_23dof/config/policy/box/v1`.
C++ deployment uses the ONNX actors and reference files. The `.pt` files preserve
training provenance and support the Python evaluator; C++ does not load them.
No retraining or weight/normalizer updates were performed.

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
