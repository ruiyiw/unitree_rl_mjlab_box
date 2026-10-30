# G1 23-DoF human-push recovery

Task: `Mjlab-Human-Push-Recovery-G1-23DoF`.

The C++ controller supports `--record <new-directory>` to capture measured
states, commands and FSM transitions. See [deployment recording](DEPLOYMENT_RECORDING.md)
for the launch flag and offline replay commands.

The selected frozen checkpoint is bundled in
`deploy/robots/g1_23dof/config/policy/human_push/v1/checkpoints/best_policy.pt`.
It is a byte-for-byte copy of
`/home/rwang/humanoid/logs/human_push_autoresearch_20261027/search/best_policy.pt`:
`ca298fd1e906b56be26b83cd6092260dcbb6a924aa432901c5ca2d9baafd0242`.
No fine-tuning or normalization updates were performed.

## Simulation

Use the same mjlab 1.6 runtime as [the balance port](BALANCE.md), with
`requirements-human-push.txt` for the export/test dependencies. The separate
entry point avoids importing the stock mjlab 1.2 task package. Run from this
repository root:

```bash
../humanoid/.venv/bin/python scripts/human_push.py play --num-envs 1
../humanoid/.venv/bin/python scripts/human_push.py validate \
  --output logs/human_push_cpp_validation --num-envs 128 --seed 2026102803
```

`play` selects the bundled checkpoint by default. `validate` builds the actual
C++ inference/observation core using g++, checks 4,096 inputs against the frozen
PyTorch actor, then controls the mjlab robot with C++ actions for 25-second
trials. Use `--cpu-only` to omit GPU simulation. Each output directory must be
new. Results include exact initial states/push plans and per-trial outcomes.
Neither command sends robot commands.

The task preserves the source rewards, resets, force pulses and termination
conditions. Trials contain 20% quiet, 60% single-push and 20% double-push cases,
with 10–100 N smooth pulses of 0.1–0.6 seconds, applied to pelvis/torso near the
center of mass. Play mode keeps the pushes and disables training randomization.
`unitree_balance.tasks` registers both preserved tasks. `scripts/human_push.py
train` is available for subsequent experiments, but does not reproduce the
historical autoresearch optimizer schedule automatically.

## Hardware interface

The actor uses 75 measurements in this order:

| Columns | Input |
|---|---|
| 0–2 | Pelvis IMU angular velocity, local axes |
| 3–5 | Gravity projected into pelvis frame, from normalized wxyz quaternion |
| 6–28 | Joint positions minus trained nominal pose |
| 29–51 | Joint velocities |
| 52–74 | Previous clipped normalized actions; zero on entry |

No world position, base linear velocity, contact force, push information or
critic input reaches the actor. The ONNX file includes the saved normalization
(epsilon 0.01), the 256/256 ELU actor and clipping to [-1, 1]. It is a single
file with float32 `[1,75]` input `obs` and `[1,23]` output `actions`; it needs no
`.onnx.data` file or external normalizer.

The original joint/action order, bent-knee pose, action scales, gains and 20 ms
policy period are retained. Target positions are `nominal + scale * action`.
The 23 policy joints map to SDK motor IDs `0..12, 15..19, 22..26`.
The controller requires `mode_machine=4` and uses ankle pitch/roll `mode_pr=0`,
consistent with the [SDK G1 definitions](https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/dds_wrapper/robots/g1/defines.h).

`params/contract.json` records the compiled model interface; `params/deploy.yaml`
records hardware parameters; generated `params/PolicyConfig.h` supplies those
same constants to C++. Policy identity metadata must match the compiled config.
The checkpoint's original metadata and archived training source hashes are
included in the bundle. The training XML and every compiled model array were
compared against the archived training setup during export.

## Build and launch on the same 23-DoF G1

Install the repository's normal C++ deployment prerequisites (Unitree SDK2,
CycloneDDS, Boost program_options, yaml-cpp, Eigen, fmt, spdlog and zlib), then
build for the controller computer's architecture:

```bash
cmake -S deploy/robots/g1_23dof -B deploy/robots/g1_23dof/build
cmake --build deploy/robots/g1_23dof/build -j4
```

Keep the `config/policy/human_push/v1` directory alongside this robot project.
Use the dedicated profile, after the repository's normal suspended startup and
debug-mode procedure. Replace `enp5s0` with the actual robot Ethernet interface:

```bash
cd deploy/robots/g1_23dof/build
./g1_ctrl --config ../config/human_push.yaml --network enp5s0
```

Start in Passive; press **L2 + Up** for the three-second transition to the trained
standing pose, then **R2 + A** for HumanPushRecovery. **L2 + B** returns to
Passive. This profile loads only these three states and the bundled policy.
It does not require walking or dance policy files.

Inference runs synchronously in the FSM with a 50 Hz deadline and fresh sensor
snapshot. Bad IMU/observations, non-finite actions, sensor timeout or tilt beyond
45 degrees immediately set damping and request Passive. The simulator's height
termination cannot run on hardware without a height estimate; height is not
added to the actor. The mjlab rollout tests policy calculations and model PD
control, not DDS transport, motor firmware limits or physical contact dynamics.

The complete x86-64 controller compiled and its CLI was checked. The C++ actor
also passed the simulation checks in [HUMAN_PUSH_VALIDATION.md](HUMAN_PUSH_VALIDATION.md).
No physical robot was connected or tested. The first supported hardware trial
remains necessary before claiming real-robot push-recovery performance.

## Re-export with provenance checks

The bundle is already exported. To reproduce it into a new directory:

```bash
../humanoid/.venv/bin/python scripts/human_push.py export \
  --checkpoint ../humanoid/logs/human_push_autoresearch_20261027/search/best_policy.pt \
  --source-runtime ../humanoid/logs/human_push_autoresearch_20261027/search/runtime/src/humanoid_balance \
  --output /tmp/human_push_export
```

Export checks task identity, 23 DoF, selected checkpoint hash, archived training
sources, model arrays and interface before generating artifacts. A new checkpoint
needs matching generated constants and a controller rebuild; swapping the ONNX
file alone is rejected when identity metadata differs.
