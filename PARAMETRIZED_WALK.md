# G1 23-DoF: walk a commanded distance, with velocity control

This profile adds **walk(N)**: the robot walks N metres straight ahead and stops.
It starts in the existing velocity policy; the operator sets N and switches to one
of two walk policies with the gamepad.

| Policy | Trained as | Selected checkpoint | Inputs → outputs |
|---|---|---|---|
| Goal-conditioned | One policy across 0–5 m (approach 2), PPO alone with the box go-to policy's settings, fall penalty and randomization ranges; the Kalman position filter and four sensor frames | `abl_256_ppo_goto_s3/selected.pt` (update 8000) | 334 → 23 |
| Adapter | Frozen 2 m base walker plus a bounded correction (approach 3); both retrained with the robust randomization profile, the Kalman filter and four sensor frames | `adapter20_r/selected.pt` (update 1400) on `base_r_seed2/selected.pt` (update 1900) | 334 → 23 |

Checkpoints came from `factory/parametrization/walk/runs`. The self-contained
deployment package is `deploy/robots/g1_23dof/config/policy/parametrized_walk/v3`.

**The v3 package has not run on a robot.** The C++ controller was checked against
the training environment and in simulation only; see [Checks](#checks) and
[What is not verified](#what-is-not-verified). Both earlier packages did run on a
G1 and lost their balance on a 1 m walk: `v1` (`goal20` and `adapter20_k` on
`base`, one sensor frame) on 2026-11-17, and `v2` (`goal20_hk2`, with the adapter
v3 still ships) on 2026-11-18. [Why v3](#why-v3) says what was changed in the
goal-conditioned policy and what was measured; [Why v2](#why-v2) is the earlier
round. The adapter is v2's, unchanged.

## Why v3

v3 replaces the goal-conditioned policy and shortens the library to the 0–5 m it
was trained on. The adapter and its base walker are v2's.

The new policy (`abl_256_ppo_goto_s3`) was trained in `factory/parametrization/walk`
as the box go-to policy was: PPO alone (no teacher guidance) with go-to's learning
rate schedule, exploration noise and entropy bonus, go-to's fall penalty and go-to's
randomization ranges, for 8,000 updates, with the same 256-256 actor, four sensor
frames and Kalman estimate as v2. Of eight combinations tried (actor size, with or
without teacher guidance, go-to's or wider randomization ranges), only this recipe
reached go-to's level, and only after 6,000-8,000 updates; three seeds of it did.

Failed runs in native MuJoCo with the compiled controller, sensor noise and
accelerometer bias as measured on the robots, and the measured ankle or IMU
offsets of two robots scaled by 1 to 3; walks of 0.3, 1, 2 and 5 m (go-to: its own
1.5 m) from the Velocity policy or from five handover states recorded on the
robot (`factory/parametrization/walk/docs/robustness_tests/evalset`):

| Test (runs) | Go-to | v1 goal-conditioned | v2 goal-conditioned | v3 goal-conditioned |
|---|---:|---:|---:|---:|
| Offsets 1× (216) | 0 | 7 | 8 | 0 |
| Offsets 1.5× (216) | 0 | 48 | 43 | 0 |
| Offsets 2× (216) | 0 | 90 | 75 | 0 |
| Offsets 3× (216) | 91 | 165 | 163 | 12 |
| Pushes of 0.2-0.4 m/s at 1× (288) | 18 | 106 | 84 | 1 |
| Physics variation at 1× (216) | 0 | 28 | 24 | 0 |

The package in this repository was run through the whole set after export: all
1,368 runs give the same result as the checkpoint tested before packaging. The
table is without the [action-rate limit](#the-action-rate-limit); with the
shipped limit the v3 column is 0, 0, 0, 12, 0, 0.

- **Where it stops:** a median 8 cm from the goal at 1× offsets (18 cm at the
  95th percentile), against 5 cm (12 cm) for v2; the error grows with distance.
- **After arriving** its joints keep moving more than go-to's (mean joint speed
  0.19 rad/s against 0.07 in the last 3 s of these runs).
- **Not on the robot yet.** The earlier packages also completed simulated walks
  at the measured offsets; this set is harder than the tests they passed, but it
  is still simulation.

## Why v2

On the robot both v1 policies got through the hold and two steps, then rocked
further sideways each step and fell to the left at about 3.5–4 s. The same robot
walks with the box go-to policy, so the comparison below uses that policy as the
reference. Every number is from native MuJoCo with the compiled controller,
starting in the Velocity policy, on a 1 m walk (go-to: its own 1.5 m walk).
"This robot's offsets" are the ankle encoder offsets implied by its standing
posture: 1.15° and 1.75° pitch, −0.5° and −1.9° roll (left, right).

| Test | Go-to | v1 goal-conditioned | v1 adapter | v2 goal-conditioned | v2 adapter |
|---|---|---|---|---|---|
| Multiple of this robot's ankle offsets at which it first falls | above 3× | 2× | 1.5× | 2× | 3× |
| Falls with the offsets plus physics variation inside the training ranges (40 draws) | 0 | 0 | 10 | 1 | 1 |
| Falls after a 0.3 m/s sideways push with the offsets (6 pushes) | 0 | 2 | 4 | 2 | 0 |
| Falls after a 0.4 m/s sideways push with the offsets (6 pushes) | 0 | 4 | 6 | 2 | 2 |
| Falls with the offsets plus this robot's accelerometer bias (10 runs) | not used | 0 | 10 | 0 | 0 |

- **The adapter is the one that improved.** Its base walker and correction were
  retrained with wider ankle and IMU offsets, accelerometer bias and noise, and
  pushes of 0.3–0.5 m/s from the start of training (`--profile robust` in
  `factory/parametrization/walk`). v1's pushes grew with a success rate that
  stayed low, so those policies trained on pushes of about 0.1–0.16 m/s.
- **The goal-conditioned policy was not improved by retraining.** Eight
  retrained checkpoints with the robust profile (stronger or gentler pushes, stronger
  teacher guidance, twice the training, starting from the robust base walker)
  all did worse in these tests than a checkpoint that already existed,
  `goal20_hk2`. v2 ships that one. It has four sensor frames and was trained on
  the filter it is deployed with, but with the first randomization profile.
- **What these tests do not show.** Simulated falls from offsets alone are
  forward and early; the robot's falls were sideways and later, like the
  simulated response to a 0.2–0.3 m/s push to the left. What supplies that
  disturbance on the robot was not identified.

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

- **Range and step:** 0.1 to 5.0 m in steps of 0.1 m (50 planned walks). A
  request between two steps walks the nearest one (`3.14` → 3.1 m) and says so.
- **Out of range:** a typed request outside 0.1–5 m is refused and the previous
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

A walk takes about 2.4 s plus 2.4 s per metre (2 m: 7.4 s; 5 m: 14.4 s), after a
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
- **Observation (109 per frame):** gyro 3, projected gravity 3, joint positions 23, joint
  speeds 23, previous correction 23, the plan's next joint targets 23, stride
  phase and remaining plan 5, and 6 numbers that place the plan and the goal
  relative to where the robot believes it is.
- **Actor input (334):** the last four readings of the first five terms (each
  term's readings together, oldest first), then the plan and goal numbers once.
  At a walk's start the first reading fills all four. The number of frames is a
  constant of the package (`kHistory` in `PolicyConfig.h`).
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
| Reference | One fixed plan per stage | One of 50 plans, chosen by the operator's distance at entry |
| Actor input | 376 values, four sensor frames | 334 values, four sensor frames |
| Extra sensor | None | IMU accelerometer (position filter) |
| Arm feedforward torque | Arm gravity compensation | None: these policies were trained on plain PD |
| Entry clock | Stage-specific repeats | First frame held 1.0 s after entry from Velocity |
| Settled test | Stillness and joints near the plan's end pose | Stillness only: these policies stand in their own posture |
| Stage that never settles | Faults to Passive after 12 s | Keeps holding; Velocity stays available |

## The action-rate limit

Neither training nor the controller smooths the policies' output: each 20 ms
step the joint target is the plan's plus the actor's correction. With sensor
noise the goal-conditioned policy's correction changes by 0.05 rad per step on
average while walking and by up to 2.0 rad in one step on an ankle (the adapter:
up to 1.3 rad). As in the humanoid repository's deployment, each step's change
of each raw action is therefore held to a per-joint `rate_limit` (raw action
units per 20 ms step; `rate_limit` of each walk state in the profile, `null`
for none). The limited action is the one applied, and the one the actor reads as
its previous action. An action inside the limit is passed on unchanged.

The limits come from `unitree_parametrized_skills/action_rates.json`: per joint,
the largest change over healthy simulated runs (the two robots' measured offsets,
sensor noise, pushes of 0.2–0.4 m/s; about 500 completed runs per policy), times
a factor, at least 0.2.

| Policy | Factor | Largest command step left | Why |
|---|---:|---:|---|
| Goal-conditioned | 0.5 | 1.0 rad (ankle pitch) | The tightest limit tested that costs no robustness (below) |
| Adapter | 1.25 | 1.65 rad | A guard only, as in the humanoid repository: tighter limits were not tested for it |

Failed runs of the goal-conditioned policy in the controller tests of
[Why v3](#why-v3), by factor:

| Factor | Offsets 1× | 1.5× | 2× | 3× | Pushes | Physics variation |
|---|---:|---:|---:|---:|---:|---:|
| no limit | 0 | 0 | 0 | 12 | 1 | 0 |
| 1.25 | 0 | 0 | 0 | 12 | 1 | 0 |
| **0.5 (shipped)** | 0 | 0 | 0 | 12 | 0 | 0 |
| 0.25 | 0 | 1 | 5 | 36 | 12 | 0 |
| 0.125 | 31 | 39 | 74 | 126 | 96 | 53 |

- **It cuts peaks, not the ordinary jitter.** At 0.5 the limit acts on about 0.3%
  of joint-steps; the mean change per step stays 0.05 rad. The policy needs its
  fast corrections: at 0.125 it falls at the measured offsets.
- **Changing it:** edit `factor` in `action_rates.json`, then
  `python -m unitree_parametrized_skills.export --rate-limits` (rewrites only the
  profile's limits; no rebuild). The preflight accepts 23 values of 0.05 to 20,
  or `null`.
- **Not in training.** The policies were trained without a limit; the tests above
  are the evidence that they tolerate it.

## How accurate it is

The numbers in this section were measured on the v1 policies, with a library
that reached 20 m, and were not repeated for v2 or v3. v3's library ends at 5 m;
its goal-conditioned policy's error in the controller tests is in
[Why v3](#why-v3).

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
  environment for both policies at 0.3, 2.5 and 5 m (4,126 steps) were replayed
  through the C++ controller. The reference clock matched at every step, the
  observation within `8.2e-5` (leaving out the previous-action entries), the
  position estimate within `0.082 mm`, the actor's correction within `7.3e-3` and
  the joint targets within `3.2e-3 rad` (v3; the largest are the adapter's, which
  turns the estimate's last digits into larger corrections than v1's did, so the
  test's limits are `1e-2` and `5e-3 rad`; the new goal-conditioned policy stays
  within `1.7e-3` and `7.1e-4 rad`). On identical inputs the ONNX actors match the
  PyTorch checkpoints within `2e-6`.
- **Offsets, physics variation and pushes** in native MuJoCo with the C++
  controller: the table in [Why v3](#why-v3) for the goal-conditioned policy, and
  in [Why v2](#why-v2) for the adapter.
- **Native MuJoCo with the C++ controller in the loop (v1, not repeated since).** 50 runs: both policies,
  0.2–20 m, started directly, from the Velocity policy standing, and from the
  Velocity policy after walking and turning. No falls. Error 2–6 cm up to 10 m
  and 10–11 cm at 20 m.
- **The real `g1_ctrl` over DDS against a simulated robot (v3, with the action-rate limit)**, operated by script:
  FixStand, Velocity, a typed distance, gamepad steps, a refused 35 m request, a 2.4 m
  goal-conditioned walk (2.7 cm error), walking 1.0 m again (4.7 cm), a hand-over to the adapter
  for 1.0 m (1.9 cm), back to Velocity, and a refused 0.1 m adapter walk. No missed
  deadline or fault. This run has the nominal robot and clean sensors.
- **Regression tests:** the 43 tests of `tests/test_parametrized_walk_deploy.py`
  pass, including 15 rejected profiles and rejected changed artifacts. The
  repository's other 69 tests still pass, and the walk and box profiles pass their
  preflight with the v3 build.

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

- **No hardware run of v3.** Balance, foot slip, joint friction and delays on the real
  robot are covered only by the training randomization and the tests in
  [Why v3](#why-v3). v1 and v2 lost their balance on the robot.
- **The adapter.** It is v2's, which lost its balance on the robot. Try the
  goal-conditioned policy first.
- **Beyond 5 m.** The goal-conditioned policy was trained on 0–5 m, so the
  library ends there; the adapter, trained to 20 m, is limited with it.
- **`shortest_m`.** The profile keeps v1's 0.2 m and 0.3 m. In the training
  environment the v2 adapter lost its posture in under 2% of randomized trials
  at 0.1–0.3 m, so its limit is now conservative.
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
mv deploy/robots/g1_23dof/config/policy/parametrized_walk/v3 /tmp/parametrized_walk_v3_old
../factory/.venv/bin/python -m unitree_parametrized_skills.export --extra 3.14 4.25
cmake --build deploy/robots/g1_23dof/build -j4
```

`--step` and `--max` change the grid (default 0.1 m to 5 m, a 2.4 MB library).
The export plans each distance with the training planner, takes under a minute,
and rewrites `parametrized_walk.yaml` and the generated `params/PolicyConfig.h`.
Beyond 5 m the goal-conditioned policy was not trained. The validation
recordings (`record.py`) and several tests assume the 0.1–5 m grid.

## Files

| Path | Contents |
|---|---|
| `deploy/robots/g1_23dof/config/parametrized_walk.yaml` | The launch profile |
| `…/config/policy/parametrized_walk/v3` | ONNX actors, original checkpoints, `plans.bin`, generated constants, manifest with SHA256 |
| `deploy/robots/g1_23dof/include/ParametrizedWalkPolicy.h` | Plan library, position filter, controller, distance selection (no DDS) |
| `…/include/State_ParametrizedWalk.h` | The FSM state |
| `…/include/ParametrizedWalkPreflight.h`, `ParametrizedWalkOperator.h` | Offline profile check; terminal and gamepad distance input |
| `unitree_parametrized_skills/` | Export, recording, validation, simulation and loopback tools; archived training source; `action_rates.json` (the measured action rates behind the profile's `rate_limit`) |
