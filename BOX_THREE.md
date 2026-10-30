# G1 23-DoF: three new box policies with velocity control

This profile contains **three individual policies** and the existing velocity
policy from `velocity_integration`. Switching is manual, using the gamepad.

| Policy | Selected checkpoint | Inputs → outputs |
|---|---|---|
| Go-to | `go-to-scratch-matt/model_3999.pt` | 376 → 23 |
| Take | `take-warm-yours/model_3999.pt` | 376 → 23 |
| Put-down | `put-down-scratch-matt/model_1600.pt` | 376 → 23 |

Checkpoints came from `humanoid-full/results/critic297-20261030/runs`.
The self-contained deployment package is
`deploy/robots/g1_23dof/config/policy/box_three/v1`.

## Build and launch on the same 23-DoF G1

Use the same installed C++ prerequisites, controller computer and Ethernet
interface as your working human-push/velocity deployment. From the repository
root:

```bash
cmake -S deploy/robots/g1_23dof -B deploy/robots/g1_23dof/build
cmake --build deploy/robots/g1_23dof/build -j4
cd deploy/robots/g1_23dof/build
./g1_ctrl --config ../config/box_three.yaml --network enp5s0
```

Replace `enp5s0` with your robot interface. Run one controller at a time. Keep
`config` alongside `build`. Python and the source training checkout are not
needed for this launch.

To record the deployment, append `--record ../recordings/box_three_001` after
creating the parent directory with `mkdir -p ../recordings`. Use a new session
name each time. See [deployment recording](DEPLOYMENT_RECORDING.md) for captured
states, commands, shutdown and offline replay.

## Start with velocity control

Follow the same suspended startup and debug-mode procedure as human-push.

1. **L2 + Up**: Passive → FixStand; wait three seconds.
2. **R1 + X**: FixStand → Velocity.
3. Use the sticks for velocity control. Centre both sticks before selecting a
   box policy.

**L2 + B** returns to Passive from every active state.
In YAML, `LT/RT/LB/RB` mean the remote's `L2/R2/L1/R1` buttons.

## Select the individual policies

| Current state | Button | Next policy | Required physical setup |
|---|---|---|---|
| Velocity | **R1 + Y** | Go-to | At the route start, facing the source table; box on the table |
| Velocity or Go-to | **R1 + A** | Take | At the source pickup stance, box aligned with the reference |
| Take | **R1 + A** | Put-down | Box held, with the robot and destination table aligned to the put-down start |
| Go-to or Put-down | **R1 + X** | Velocity | Reference has ended; hands empty |

Box-to-box buttons become available after the reference ends and the robot
settles for one second. Press the button after that condition is met; an early
press is not queued. Returning to Velocity waits for the reference to end.
Take has no Velocity exit because it ends holding the box. Each actor holds
its endpoint until you select the next state or return to Passive.

**There is no carry policy or automatic sequence in this profile.** The saved
take endpoint and put-down starting location are about **1.5 m apart** in the
training layout. A Take → Put-down button press does not transport the box or
align a destination table. Prepare a compatible physical setup before that
transition; these three actors alone do not reproduce the original two-table
transport task. Joint settling does not verify box contact or table alignment.

The reference box is **0.28 × 0.45 × 0.325 m**, nominal mass **0.5 kg**, and the
table tops are **0.90 m** high. Exact layout and reference poses are packaged in
`params/layout.json` and each stage's `reference.npz`.

## Controller behavior

The profile follows the merged velocity integration: the same 23 motor slots,
SDK2/DDS connection, stick-centering gate, three-second FixStand, and 0.5-second
cosine blend when entering from Velocity. Direct settled box-to-box transitions
retain the preceding residual and skip the entry blend.

These actors use **76 current reference/controller values plus four 75-value
sensor frames**, oldest first. Each policy entry initializes all four frames
from the first reading with zero action history, matching the saved training
code. Heading is anchored from the pelvis IMU on entry. No measured world
position, box state or contact sensor is an actor input.

Targets remain next-frame reference joints plus an unclipped actor residual.
The new training setup also requires **arm motor gravity-compensation torque**;
C++ computes it from the saved robot model, measured joints and pelvis IMU.
The old five-policy package continues to use its original controller.

The stage-entry clock follows the new training runtime's explicit reset:
go-to starts at reference frames `0, 0, 1, 2, …`; take and put-down start at
`0, 0, 0, 1, …`. Each observation uses the current frame and each action uses
`qcmd[min(clock + 1, last)]`. Completing the one-second settling check is latched:
the controller continues holding without restarting its stage timeout, matching
the previous box controller.

| Deployment detail | Previous box + velocity | Three new policies |
|---|---|---|
| Motor mapping, joint order, PD gains and action scales | 23-DoF G1 | Identical |
| Passive / FixStand / emergency exit | Existing velocity integration | Identical |
| Enter from Velocity | Sticks centred; 0.5 s blend | Same |
| Actor input | 75 values | 376 values, including four sensor frames |
| Position target | Next-frame reference + unclipped residual | Same formula |
| Arm feedforward torque | Zero after the entry blend | Nominal trained arm gravity compensation |
| Stage entry observations and clock | Previous reference controller | New training reset/history semantics |
| After completion | Hold until operator exit | Same; completion stays latched |

## Checks

The executable checks the profile and actors automatically before opening DDS.
To check loading separately without connecting to the robot, run from `build`:

```bash
./g1_ctrl --check-box-config ../config/box_three.yaml
```

The x86-64 C++ build and all **58 regression tests** passed, including the
previous box, human-push and balance checks. Validation executes the archived
training reset/update methods to check the reference clock, and checks that a
completed stage keeps holding after a later loss of settling. Across 2,383 reference/sensor
steps, the C++ observations matched the archived training functions within
`6e-8`; actor outputs matched PyTorch within `2e-6`; arm gravity torques matched
MuJoCo within `4.4e-6 Nm`. Results are in
[the validation report](unitree_box/validation/three_policies.json).
These are software checks; real-robot execution of these three policies has not
been tested here.

For optional reproduction from the repository root, use the existing box Python
environment (`requirements-box.txt`):

```bash
../humanoid/.venv/bin/python -m unitree_box.table_validate --output logs/box_three_check
../humanoid/.venv/bin/python -m pytest -q tests/test_table_box_deploy.py
```

Use a new output directory. Original checkpoints, ONNX actors, frozen
normalization, references and SHA256 provenance are included in the package.
C++ loads ONNX/reference files; `.pt` files support provenance and offline tests.
