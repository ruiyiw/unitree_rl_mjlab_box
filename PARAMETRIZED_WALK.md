# G1 23-DoF: walk a commanded distance, with velocity control

This profile adds **walk(N)**: the robot walks N metres straight ahead and stops.
It starts in the existing velocity policy; the operator sets N and switches to one
of two walk policies with the gamepad.

| Policy | Trained as | Selected checkpoint | Inputs → outputs |
|---|---|---|---|
| Goal-conditioned | One policy across 0–20 m (approach 2), trained on leg odometry; deployed with the Kalman position filter swapped in | `goal20/selected.pt` (update 1800) | 109 → 23 |
| Adapter | Frozen 2 m base walker plus a bounded correction (approach 3), retrained across 0–20 m with the Kalman filter | `adapter20_k/selected.pt` (update 1200) on `base/selected.pt` | 109 → 23 |

Checkpoints came from `factory/parametrization/walk/runs`. The self-contained
deployment package is `deploy/robots/g1_23dof/config/policy/parametrized_walk/v1`.
Nothing was retrained for deployment.

**Nothing here has run on a robot.** The C++ controller was checked against the
training environment and in simulation only; see [Checks](#checks) and
[What is not verified](#what-is-not-verified).

## Build and launch on the same 23-DoF G1

Use the same C++ prerequisites, controller computer and Ethernet interface as the
working human-push/velocity deployment. From the repository root:

```bash
cmake -S deploy/robots/g1_23dof -B deploy/robots/g1_23dof/build
cmake --build deploy/robots/g1_23dof/build -j4
cd deploy/robots/g1_23dof/build
./g1_ctrl --config ../config/parametrized_walk.yaml --network enp5s0 --distance 2.0
```

Replace `enp5s0` with your robot interface. Run one controller at a time. Keep
`config` alongside `build`. Python is not needed for this launch. `--record` works
as for the other profiles ([deployment recording](DEPLOYMENT_RECORDING.md)).

## Tell the robot how far to walk

The distance is **not compiled in**. It is read each time a walk starts, and can
be changed at any moment before that without rebuilding or relaunching:

| How | Example | Effect |
|---|---|---|
| At launch | `--distance 3.5` | First distance. Without it the profile's `WalkDistance.default_m` (2.0 m) is used. |
| Typed into the controller's terminal | `3.5` then Enter | Sets the distance for the next walk. |
| Gamepad | **R2 + Up / Down** | ± 1.0 m |
| Gamepad | **R2 + Right / Left** | ± 0.1 m |

Every change is answered in the terminal with the distance that will be walked,
for example `Walk distance: 3.5 m (terminal). It applies to the next walk.`
Read that line before pressing a walk button: the gamepad has no display.

- **Range and step:** 0.1 to 20.0 m in steps of 0.1 m (200 planned walks). A
  request between two steps walks the nearest one (`3.14` → 3.1 m) and says so.
- **Out of range:** a typed request outside 0.1–20 m is refused and the previous
  distance kept. Gamepad steps stop at the ends. A bad `--distance` stops the
  launch before any connection to the robot.
- **Shortest walk per policy:** 0.2 m for the goal-conditioned policy and 0.3 m
  for the adapter (`shortest_m` in the profile). A shorter selection is refused
  when the walk button is pressed, and the robot stays in velocity control. Below
  these distances the policies lost their posture in more than 1 in 20 randomized
  simulation trials.
- **While a walk is running** a change is accepted and applies to the next walk.

The planned walks are a library file, `plans.bin`, made by the same planner that
made the training plans. The build fingerprints it, so walking a distance that is
not on the 0.1 m grid means exporting a library that contains it and rebuilding;
see [Other distances](#other-distances).

## Start with velocity control

Follow the same suspended startup and debug-mode procedure as human-push.

1. **L2 + Up**: Passive → FixStand; wait three seconds.
2. **R1 + X**: FixStand → Velocity.
3. Use the sticks to place the robot. It will walk straight ahead from where it
   stands, the way it faces. Centre both sticks before starting a walk.

**L2 + B** returns to Passive from every active state.
In YAML, `LT/RT/LB/RB` mean the remote's `L2/R2/L1/R1` buttons. Press the
shoulder button first, then the key.

## Start a walk

| Current state | Button | Next | Available when |
|---|---|---|---|
| Velocity | **R1 + Y** | Goal-conditioned walk | Sticks centred |
| Velocity | **R1 + A** | Adapter walk | Sticks centred |
| Goal-conditioned walk | **R1 + Y** | Walk again, same policy | Robot standing still after the walk |
| Adapter walk | **R1 + A** | Walk again, same policy | Robot standing still after the walk |
| Goal-conditioned walk | **R1 + A** | Adapter walk | Robot standing still after the walk |
| Adapter walk | **R1 + Y** | Goal-conditioned walk | Robot standing still after the walk |
| Either walk | **R1 + X** | Velocity | The planned walk has played out |

A walk takes about 2.4 s plus 2.4 s per metre (2 m: 7.4 s; 20 m: 50.4 s), after a
one-second hold on entry from Velocity. The terminal reports the start, the end
of the plan, and when the robot stands still. An early press is not queued.

**The robot does not sense obstacles and does not steer around anything.** Clear
the whole route, plus margin, before pressing a walk button. There is no way to
shorten a walk in progress other than **L2 + B** (Passive).

## What the controller does

- **Frame of the walk:** the origin and the forward direction are the robot's
  position and IMU heading when the walk starts. Each walk, including "walk
  again", starts a new frame: errors of one walk are not corrected by the next.
- **Targets:** `target = plan[min(clock + 1, last)] + scale × actor(observation)`,
  plain PD, no feedforward torque. The actor's output is not clipped.
- **Observation (109):** gyro 3, projected gravity 3, joint positions 23, joint
  speeds 23, previous correction 23, the plan's next joint targets 23, stride
  phase and remaining plan 5, and 6 numbers that place the plan and the goal
  relative to where the robot believes it is.
- **Position estimate:** no position sensor is read. The pelvis position comes
  from leg kinematics (a planted foot does not move) fused with the IMU
  accelerometer in a Kalman filter; heading comes from the IMU. Both policies
  use the filter.
- **Adapter:** its ONNX actor contains both networks: the frozen base walker's
  correction plus the adapter's, the adapter's part limited to ±1.0.
- **Entry from Velocity:** 0.5 s cosine blend of targets and gains, as the box
  states do. The plan's first frame is held for 1.0 s (`hold_s`) so the first
  step starts after the blend.
- **After the plan:** the last frame is held and the actor keeps correcting
  toward the goal. "Standing still" means gyro below 0.25 rad/s, tilt below 15°
  and every joint slower than 0.5 rad/s for one second.
- **Faults → damping/Passive:** pelvis tilt beyond 45°, a missed 20 ms period,
  inference over 20 ms, a robot-state timeout, or a non-finite sensor or target.

| Deployment detail | Three box policies | Parametrized walk |
|---|---|---|
| Motor mapping, joint order, PD gains, action scales | 23-DoF G1 | Identical (checked at export) |
| Passive / FixStand / Velocity / emergency exit | Velocity integration | Identical; FixStand ends on the walks' start pose |
| Reference | One fixed plan per stage | One of 200 plans, chosen by the operator's distance at entry |
| Actor input | 376 values, four sensor frames | 109 values, one frame |
| Extra sensor | None | IMU accelerometer (position filter) |
| Arm feedforward torque | Arm gravity compensation | None: these policies were trained on plain PD |
| Entry clock | Stage-specific repeats | First frame held 1.0 s after entry from Velocity |
| Settled test | Stillness and joints near the plan's end pose | Stillness only: these policies stand in their own posture |
| Stage that never settles | Faults to Passive after 12 s | Keeps holding; Velocity stays available |

## How accurate it is

Measured in the training simulation (randomized physics and sensors, exact IMU
heading, 48 trials per distance), final distance from the goal, median:

| Distance | Goal-conditioned | Adapter |
|---|---:|---:|
| up to 4.2 m | 3 cm | 4 cm |
| 8 m | 3 cm | 4 cm |
| 12 m | 5 cm | 5 cm |
| 20 m | 8 cm | 7 cm |

These hold only if the IMU heading is exact. With a heading offset of ±1° and a
drift of ±0.1°/s, as assumed for hardware in training, the median error is about
9 cm at 8 m and 50 cm at 20 m for both policies: sideways error grows with
distance and neither policy can see it.

Two calibrations may help on hardware, both without retraining:

- **`odometry_scale`** (per walk state in the profile, default 1.0): true
  distance per estimated distance. In simulation the estimate reads 0.5–0.7%
  long, so the robot stops about 10 cm short at 20 m. Measure a long walk with a
  tape and set the scale to walked ÷ commanded.
- **`shortest_m`**: raise it if short walks are unreliable on the real floor.

## Checks

The executable checks the profile, the library and both actors before opening
DDS. To run that check alone, from `build`:

```bash
./g1_ctrl --check-box-config ../config/parametrized_walk.yaml
```

What was verified, all in software ([reports](unitree_parametrized_skills/validation)):

- **C++ against training, step by step.** Sensors recorded in the mjlab training
  environment for both policies at 0.3, 2.5 and 20 m (7,726 steps) were replayed
  through the C++ controller. The reference clock matched at every step, the
  observation within `6e-5` (leaving out the previous-action entries), the
  position estimate within `0.06 mm`, the actor's correction within `2.6e-3` and
  the joint targets within `1.1e-3 rad`. On identical inputs the ONNX actors
  match the PyTorch checkpoints within `2e-6`.
- **Native MuJoCo with the C++ controller in the loop.** 50 runs: both policies,
  0.2–20 m, started directly, from the Velocity policy standing, and from the
  Velocity policy after walking and turning. No falls. Error 2–6 cm up to 10 m
  and 10–11 cm at 20 m.
- **The real `g1_ctrl` over DDS against a simulated robot**, operated by script:
  FixStand, Velocity, a typed distance, gamepad steps, a 2.4 m goal-conditioned
  walk (3.5 cm error), walking 1.0 m again (5.1 cm), a hand-over to the adapter
  for 1.0 m (5.0 cm), back to Velocity, and a refused 0.1 m adapter walk.
- **Regression tests:** the 39 tests of `tests/test_parametrized_walk_deploy.py`
  pass, including 13 rejected profiles and rejected changed artifacts. The
  repository's other 69 tests still pass, and the box profiles still pass their
  preflight with this build.

For optional reproduction from the repository root, with the training checkout
beside this repository (`../factory`, and `../humanoid` for its planner):

```bash
PY=../factory/.venv/bin/python
$PY -m unitree_parametrized_skills.validate --fixtures tests/fixtures/parametrized_walk --output logs/walk_check
$PY -m unitree_parametrized_skills.simulate --policy adapter --distance 3.5 --from-velocity --output logs/walk_sim
$PY -m unitree_parametrized_skills.loopback --output logs/walk_loopback
$PY -m pytest -q tests/test_parametrized_walk_deploy.py
```

Use new output directories. `loopback` needs the built `g1_ctrl` and the Unitree
SDK it was built with.

## What is not verified

- **No hardware run.** Balance, foot slip, joint friction and delays on the real
  robot are covered only by the training randomization.
- **The IMU accelerometer.** The filter assumes `imu_state.accelerometer` is the
  specific force in the pelvis frame in m/s² (about +9.81 on z at rest), from the
  same IMU as the quaternion. Check this on the robot before a first walk. The
  controller averages it over each 20 ms period, as training did.
- **IMU heading.** Drift during a walk becomes sideways error at the goal.
- **The floor.** Training assumed a flat, level floor with friction 0.4–1.4.
- **The hoist.** FixStand is an open-loop pose and does not balance: keep the
  robot supported until Velocity is active, as for the other profiles.

## Other distances

The library's distances are chosen at export. To add a distance off the grid, or
change the grid, re-export into a new package directory and rebuild, since the
build fingerprints `plans.bin`:

```bash
mv deploy/robots/g1_23dof/config/policy/parametrized_walk/v1 /tmp/parametrized_walk_v1_old
../factory/.venv/bin/python -m unitree_parametrized_skills.export --extra 3.14 7.25
cmake --build deploy/robots/g1_23dof/build -j4
```

`--step` and `--max` change the grid (default 0.1 m to 20 m, a 30 MB library).
The export plans each distance with the training planner, takes about two
minutes, and rewrites `parametrized_walk.yaml` and the generated
`params/PolicyConfig.h`. Beyond 20 m the policies were not trained; in simulation,
with an exact heading, arrival within 10 cm fell to 38% (goal-conditioned) and 52%
(adapter) at 25 m.

## Files

| Path | Contents |
|---|---|
| `deploy/robots/g1_23dof/config/parametrized_walk.yaml` | The launch profile |
| `…/config/policy/parametrized_walk/v1` | ONNX actors, original checkpoints, `plans.bin`, generated constants, manifest with SHA256 |
| `deploy/robots/g1_23dof/include/ParametrizedWalkPolicy.h` | Plan library, position filter, controller, distance selection (no DDS) |
| `…/include/State_ParametrizedWalk.h` | The FSM state |
| `…/include/ParametrizedWalkPreflight.h`, `ParametrizedWalkOperator.h` | Offline profile check; terminal and gamepad distance input |
| `unitree_parametrized_skills/` | Export, recording, validation, simulation and loopback tools; archived training source |
