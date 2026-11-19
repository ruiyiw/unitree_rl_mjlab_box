# G1 23-DoF: a person's walk, with velocity control

This profile adds one policy: the robot walks **2.2 m straight ahead as a person
does** (a motion-capture walk retargeted to the 23-DoF G1), stops, and hands back
to the existing velocity policy. The operator starts it from velocity control with
the gamepad.

| Policy | Trained as | Selected checkpoint | Inputs → outputs |
|---|---|---|---|
| Human walk | PPO correction to a fixed reference motion (`humanoid-full/policies/policy-walk-human`, trained by `motion.py`) | `runs/policy/model_500.pt` (iteration 500 of 1,000) | 145 → 23 |

The self-contained deployment package is
`deploy/robots/g1_23dof/config/policy/human_walk/v1`.

**This package has not run on a robot, and the policy was not trained for one.**
Read [What is different about this policy](#what-is-different-about-this-policy)
and [What is not verified](#what-is-not-verified) before a first run.

## What is different about this policy

The box-three and parametrized-walk actors read only what the robot measures and
were trained with randomized physics. This one was not. Three things follow, and
the deployment handles each as below.

| | In training | On the robot | What it costs |
|---|---|---|---|
| Torso pose relative to the reference, IMU linear velocity | The simulator's true values (with noise) | Estimated: the walk controller's Kalman leg odometry for x and y, the lowest sole point for height, the IMU for orientation, the waist encoder for the torso | The estimate reads about 4–6 cm short at the end of the walk, because the feet slide a little. The robot stops 2–6 cm past the reference's end. |
| Arms' weight | Compensated by MuJoCo for the arm joints **and for the pelvis** (the arms weighed nothing on the legs) | The ten arm motors get the box-three gravity torque; the legs carry the arms | The undisturbed walk still completes in every simulated run. Under training's repeated pushes 77% complete the walk, against 92% in training's own conditions. |
| Standing after the walk | Fine on the simulator's true position | Unreliable on the estimate | Left standing, the policy fell in simulation on floors with friction 1.2 and more, and with 2× ankle offsets. The profile therefore hands back to Velocity 0.5 s after the reference ends. |

Training randomized only foot friction (0.4–1.2), encoder bias (0.01 rad) and
pushes (up to 0.4 m/s). Mass, motor gains, delays and sensor errors were not
randomized.

**Why iteration 500 and not the last one.** Later checkpoints withstand pushes
better in training, but they shuffle their feet while standing. Leg odometry
assumes a planted foot does not move, so the estimate drifts and the policy
walks away: on estimated inputs, 98.8% of iteration 999's runs left the
tracking bounds within 60 s, against none for iteration 500
([evaluation](unitree_human_walk/validation/evaluation)).

## Build and launch on the same 23-DoF G1

Use the same C++ prerequisites, controller computer and Ethernet interface as the
working human-push/velocity deployment. From the repository root:

```bash
cmake -S deploy/robots/g1_23dof -B deploy/robots/g1_23dof/build
cmake --build deploy/robots/g1_23dof/build -j4
cd deploy/robots/g1_23dof/build
./g1_ctrl --config ../config/human_walk.yaml --network enp5s0
```

Replace `enp5s0` with your robot interface. Run one controller at a time. Keep
`config` alongside `build`. Python is not needed for this launch. `--record` works
as for the other profiles ([deployment recording](DEPLOYMENT_RECORDING.md)).

## Start with velocity control

Follow the same suspended startup and debug-mode procedure as human-push.

1. **L2 + Up**: Passive → FixStand; wait three seconds.
2. **R1 + X**: FixStand → Velocity.
3. Use the sticks to place the robot. It will walk straight ahead from where it
   stands, the way it faces. **Centre both sticks and leave them centred** until
   the walk is over.

**L2 + B** returns to Passive from every active state.
In YAML, `LT/RT/LB/RB` mean the remote's `L2/R2/L1/R1` buttons. Press the
shoulder button first, then the key.

## Start the walk

| Current state | Button | Next | Available when |
|---|---|---|---|
| Velocity | **R1 + Y** | Human walk | Sticks centred |
| Human walk | *(none)* | Velocity | By itself, 0.5 s after the reference ends |
| Human walk | **R1 + X** | Velocity | The reference has played out (only needed if `return_s` is 0) |

- **Timeline:** 1.0 s holding the first pose (the 0.5 s blend is inside it),
  6.4 s of reference (0.4 s standing, about 4.7 s walking, the rest standing),
  then 0.5 s until Velocity takes over: about 8 s from the button to velocity
  control.
- **Where it goes:** 2.20 m ahead and 0.14 m to the right of where it stood, as
  the person in the clip did.
- **Speed:** the pelvis reaches about 1.05 m/s, twice the go-to policy's 0.57 m/s,
  with steps up to 0.51 m.
- **To walk again:** place the robot with the sticks, centre them, press
  **R1 + Y**. There is no "again" button: every walk starts from Velocity.

**The robot does not sense obstacles and does not steer.** Clear 3 m ahead and
1 m to each side before pressing **R1 + Y**. A walk in progress can only be
stopped with **L2 + B** (Passive).

**Velocity takes over with whatever the sticks command.** Keep them centred
through the walk; a deflected stick moves the robot as soon as the walk ends.

## What the controller does

- **Frame of the walk:** the origin and the forward direction are the robot's
  position and IMU heading when the walk starts.
- **Targets:** `target = reference[min(clock + 1, last)] + scale × actor(observation)`,
  PD, with gravity-compensation torque on the ten arm motors (the box-three
  controller's `table_box::gravity_torque`). The actor's output is not clipped.
- **Observation (145):** the reference's joint positions and speeds 46, the
  reference's torso in the robot's torso frame 9, the reference's torso 0.2 s and
  0.5 s ahead 10, progress and ended 2, IMU linear velocity 3, gyro 3, projected
  gravity 3, joint positions 23, joint speeds 23, previous correction 23. One
  frame, no history.
- **Position estimate:** no position sensor is read. Pelvis x and y and their
  rates come from the parametrized-walk controller's Kalman filter (leg
  kinematics fused with the IMU accelerometer), unchanged. Height is minus the
  lowest of the soles' 28 corner points under the pelvis; its rate is the
  difference over one step. The torso follows from the pelvis and the waist
  encoder.
- **Entry from Velocity:** 0.5 s cosine blend of targets, gains and arm torque.
  The reference's first frame is held for 1.0 s (`hold_s`) and for one step more,
  which reproduces training's clock (`0, 0, 1, 2, …`).
- **After the reference:** the last frame is held. After `return_s` (0.5 s) the
  state hands back to Velocity, which blends in over its own 0.5 s. With
  `return_s: 0` the walk policy keeps standing until **R1 + X**; see the table
  above for why that is not the default.
- **Faults → damping/Passive:** pelvis tilt beyond 45°, a missed 20 ms period,
  inference over 20 ms, a robot-state timeout, or a non-finite sensor or target.

| Deployment detail | Three box policies | Parametrized walk | Human walk |
|---|---|---|---|
| Motor mapping, joint order, PD gains, action scales | 23-DoF G1 | Identical | Identical (checked at export and at compile time) |
| Passive / FixStand / Velocity / emergency exit | Velocity integration | Identical | Identical; FixStand ends on the walk's start pose |
| Reference | One fixed plan per stage | One of 200 plans | One fixed motion (323 frames) |
| Actor input | 376 values, four sensor frames | 334 values, four sensor frames | 145 values, one frame |
| Position and velocity inputs | None | Kalman leg odometry, trained with it | Kalman leg odometry and sole height, **not trained with them** |
| Arm feedforward torque | Arm gravity compensation | None | Arm gravity compensation (the box-three torque) |
| Entry clock | Stage-specific repeats | First frame held 1.0 s | First frame held 1.0 s, and one step |
| After the reference | Holds until the operator's button | Holds; Velocity stays available | Velocity takes over by itself after 0.5 s |

## Checks

The executable checks the profile, the reference and the actor before opening
DDS. To run that check alone, from `build`:

```bash
./g1_ctrl --check-box-config ../config/human_walk.yaml
```

What was verified, all in software ([reports](unitree_human_walk/validation)):

- **C++ against training, step by step.** Sensors recorded in the mjlab training
  environment (one undisturbed rollout and one under pushes, 1,192 steps) were
  replayed through the C++ controller. The clock matched at every step, the
  observation within `2.0e-5`, the position estimate within `7.6e-6 m`, the
  correction within `4.7e-5`, the joint targets within `2.1e-5 rad` and the arm
  torques within `1.5e-6 Nm` of MuJoCo's. On identical inputs the ONNX actor
  matches the PyTorch checkpoint within `8.3e-7`.
- **The policy on estimated inputs**, in the batched training environment (256
  runs of 60 s, [evaluation](unitree_human_walk/validation/evaluation)):

  | Inputs | Arms' weight | Completes the walk | At the end pose at 60 s | Walks (reference 2.20 m) |
  |---|---|--:|--:|--:|
  | Simulator's | On the robot | 100% | 100% | 2.19 m |
  | Estimated | Weightless (training's) | 100% | 100% | 2.24 m |
  | Estimated | On the robot | 100% | 100% | 2.25 m |
  | Estimated, under training's pushes | On the robot | 77% | 0% | – |

  No run tilted 60°; the runs that fail leave the tracking bounds.
- **Native MuJoCo with the C++ controller in the loop**
  ([simulation](unitree_human_walk/validation/simulation.json)): started from the
  Velocity policy with the profile's blend and hold, arms' weight on the robot,
  the robot is 2.4 cm from the reference's end when Velocity takes over, and
  10 cm from there after 10 s on Velocity.
- **Disturbances, one run each, same setup**
  ([robustness](unitree_human_walk/validation/robustness.json)). The ankle
  offsets are the ones PARAMETRIZED_WALK.md uses.

  | Test | Falls with the profile | Falls if the walk policy is left standing |
  |---|--:|--:|
  | Ankle encoder offsets 0.5×, 1×, 1.5×, 2×, 3× | 0 of 5 | 2 of 5 (2×, 3×) |
  | Floor friction 0.4, 0.6, 0.8, 1.0, 1.2, 1.4 | 0 of 6 | 2 of 6 (1.2, 1.4) |
  | One sideways push of 0.2, 0.3 or 0.4 m/s during the walk | 0 of 18 | not run |
  | The same pushes with 1× ankle offsets | 0 of 18 | not run |

  With the offsets the robot ends further from the reference's end: 12 cm at
  1×, 31 cm at 3×. How the offsets and pushes were applied for
  PARAMETRIZED_WALK.md's table is not recorded here, so the two tables are not
  directly comparable.
- **Regression tests:** the 27 tests of `tests/test_human_walk_deploy.py` pass,
  including 11 rejected profiles and rejected changed artifacts. The
  repository's other 107 tests still pass (2 skipped: they need `../factory`),
  and the parametrized-walk and box-three profiles pass their preflight with
  this build.

For optional reproduction from the repository root, with the training checkout
beside this repository (`../humanoid-full/policies`, with
`policy-walk-human/banks/plan.pt` and `runs/policy/model_500.pt`):

```bash
PY=../humanoid/.venv/bin/python
$PY -m unitree_human_walk.validate --fixtures tests/fixtures/human_walk --output logs/human_walk_check
$PY -m unitree_human_walk.simulate --from-velocity --output logs/human_walk_sim
$PY -m unitree_human_walk.robustness --output logs/human_walk_robustness
$PY -m unitree_human_walk.evaluate --arm-weight --output logs/human_walk_eval.json
$PY -m pytest -q tests/test_human_walk_deploy.py
```

Use new output directories. `simulate --viewer` opens MuJoCo's viewer.

## What is not verified

- **No hardware run.** Everything above is simulation.
- **Robot-to-robot variation.** The policy was not trained with randomized mass,
  motor gains, delays or sensor errors. The box and parametrized-walk policies
  were, and the first parametrized-walk package still fell on the robot.
- **Landing speed.** The feet touch down at a median 0.71 m/s (largest
  1.08 m/s), against 0.61 m/s for the go-to policy.
- **Foot sliding.** The stance feet slide a little in simulation, which is what
  the estimate's 4–6 cm error comes from. On a floor with different friction
  the slide, the estimate and the stopping point change.
- **Standing on this policy.** The profile avoids it. With `return_s: 0`, expect
  it to drift.
- **The IMU accelerometer**, as for the parametrized walk: the filter assumes
  `imu_state.accelerometer` is the specific force in the pelvis frame in m/s²
  (about +9.81 on z at rest).
- **The start pose.** FixStand ends on the clip's standing pose, which is
  straighter-legged than the other profiles' stand. The walk itself is entered
  from Velocity's stance, with the blend.
- **The DDS path.** `g1_ctrl` was built and its preflight run; it was not run
  against a simulated robot over DDS (`loopback`), as the parametrized walk was.
- **The hoist.** FixStand is an open-loop pose and does not balance: keep the
  robot supported until Velocity is active, as for the other profiles.

A first run should be on the hoist with slack, with the route clear, and after
the Velocity policy has been checked on the same floor.

## Another checkpoint or motion

Re-export into a new package directory and rebuild; the build fingerprints the
actor and the reference:

```bash
mv deploy/robots/g1_23dof/config/policy/human_walk/v1 /tmp/human_walk_v1_old
../humanoid/.venv/bin/python -m unitree_human_walk.export --checkpoint runs/policy/model_500.pt
cmake --build deploy/robots/g1_23dof/build -j4
```

The export rewrites `human_walk.yaml` and the generated `params/PolicyConfig.h`,
and refuses a robot whose joints, gains, leg kinematics or filter constants
differ from the packages the controller reuses.

## Files

| Path | Contents |
|---|---|
| `deploy/robots/g1_23dof/config/human_walk.yaml` | The launch profile |
| `…/config/policy/human_walk/v1` | ONNX actor, original checkpoint, `reference.bin`, generated constants, manifest with SHA256 |
| `deploy/robots/g1_23dof/include/HumanWalkPolicy.h` | Reference, estimate, controller (no DDS) |
| `…/include/State_HumanWalk.h` | The FSM state |
| `…/include/HumanWalkPreflight.h` | Offline profile check |
| `unitree_human_walk/` | Export, the Python estimate, recording, validation, simulation and robustness tools; validation reports; archived training source |
