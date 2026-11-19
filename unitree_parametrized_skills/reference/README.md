# walk(N): a simulated 23-DoF G1 walks N metres and stops

One fixed set of weights walks Unitree's 23-DoF G1 a requested distance N over flat ground
and stops, in mjlab (MuJoCo-Warp). The actor reads only what the real robot has. This
directory holds the environment, the two approaches from [parametrization/walk/docs/README.md](docs/README.md)
that were asked for (approach 2, goal-conditioned PPO; approach 3, a frozen base walker plus
a small adapter), and the evaluation.

Everything here is simulation. Nothing has run on hardware.

## Results

Full tables: [parametrization/walk/docs/RESULTS.md](docs/RESULTS.md) (128 trials per distance and
tier). Videos: `parametrization/walk/runs/videos/{goal,adapter,base}.mp4`, and `goal_camera.mp4` with the
head camera's view.

Stable-arrival rate (%), randomized physics and sensors:

| Controller | 0 m | 0.1 m | 0.2 m | 0.3 m | 0.75 m | 1.4 m | 2.5 m | 3.5 m | 4.2 m | 8 m | 20 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Plan played back, no policy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Base walker alone (2 m plan only) | 99 | 0 | 0 | 99 | 100 | 0 | 98 | 0 | 90 | 0 | 4 |
| Approach 2, seed 1 | 100 | 100 | 100 | 100 | 100 | 99 | 98 | 98 | 88 | 49 | 3 |
| Approach 2, seed 2 | 100 | 98 | 99 | 98 | 99 | 97 | 98 | 92 | 90 | 49 | 5 |
| Approach 3, bound 0.5 | 99 | 96 | 94 | 91 | 91 | 96 | 92 | 88 | 88 | 48 | 3 |
| Approach 3, bound 1.0 | 100 | 98 | 100 | 99 | 98 | 95 | 98 | 85 | 83 | 45 | 4 |
| Privileged teacher (not deployable) | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 81 |

- **Inside the training range (0–5 m)** both approaches work. Under randomization
  approach 2 arrives in 88–100% of trials and approach 3 in 83–100%; averaged over the nine
  distances from 0 to 4.2 m that is 98% and 97% for approach 2's seeds, 93% and 95% for
  approach 3's bounds. On the nominal robot
  approach 3 and approach 2's first seed are at 99–100% from 0 to 8 m; the second seed is at
  96–100% to 4.2 m and 76% at 8 m.
- **Beyond the range** every deployable controller is limited by its position estimate, not
  by walking: about half the randomized trials arrive at 8 m and 3–5% at 20 m, where the
  leg odometry is off by a median of about 10 cm and 40–50 cm. The teacher, which reads the
  true position, arrives in 100% and 81%. On the nominal robot all four miss 20 m by
  14–20 cm.
- **The base walker alone does not generalize.** It arrives on plans whose step count has
  the parity of its training plan and walks about 40 cm past the goal on the others. The
  adapter fixes this.
- **Falls.** Under randomization 2–4% of trials end in a fall or lost posture for all four.
  On the nominal robot approach 3 and approach 2's first seed never fall; the second seed
  falls or loses posture in 2–7% of trials at 1.4–4.2 m and about 20% at 8 and 20 m.
- **Standing.** Measured 30 s after arrival instead of 5 s, arrival on the selection
  distances is 91–94% for all four.

Construction cost (4,096 envs on one A100; training only):

| Controller | Deployed weights trained in | Everything it needs | Trainable parameters |
|---|---|---|---:|
| Approach 2 | 197 M steps, 56 min (student) | 344 M steps, 97 min (with its teacher) | 99,863 |
| Approach 3 | 197 M steps, 52 min (adapter) | 590 M steps, 161 min (with the base walker and its teacher) | 12,695 |
| Approach 3, adapter stopped at 1,000 updates | 98 M steps, 26 min | 492 M steps, 135 min | 12,695 |

Given an existing base walker, the adapter reaches approach 2's nominal result with an
eighth of the trainable parameters and no teacher of its own, in about the same number of
steps as approach 2's student, and is 2–5 points lower overall under randomization. At half
the steps it is 5–10 points lower. Building the base walker first costs more than building
approach 2 outright. These are single runs per configuration (two seeds for approach 2);
differences of a few points are within what I saw between checkpoints of one run.

## Variations: history, a Kalman estimate, training to 20 m

Three changes were tried on both approaches. Every table and the per-distance numbers are in
[parametrization/walk/docs/VARIATIONS.md](docs/VARIATIONS.md); these are single runs, the long
distances have 48 trials each, and differences of a few points are noise. Cells are
stable-arrival rates (%) as **nominal robot / randomized / randomized with an exact IMU
heading**. The third tier was added because the assumed heading error (±1° offset, ±0.1°/s
drift) alone moves a 20 m goal by tens of centimetres: the estimate's median error at 20 m is
45 cm with it and 12 cm without.

**1. Four frames of sensor history (trained on 0–5 m).**

"Ended early" is an episode cut short by a fall or by a posture far from the plan's,
randomized tier.

| Controller | 0–4.2 m | 8 m | Ended early, 0–4.2 m | Ended early, 0–20 m |
|---|---:|---:|---:|---:|
| Approach 2 | 100 / 98 / 99 | 99 / 48 / 83 | 0.6–1.0% | 2.6% |
| Approach 2 + history | 100 / 99 / 99 | 27 / 52 / 88 | 0.3–0.5% | 0.4% |
| Approach 3 | 100 / 96 / 99 | 100 / 46 / 80 | 0.5% | 0.8% |
| Approach 3 + history in the adapter | 100 / 92 / 94 | 0 / 45 / 83 | 5.0% | 6.7% |
| Approach 3 + history in base and adapter | 100 / 87 / 91 | 100 / 40 / 71 | 6.0% | 5.7% |

- **Approach 2: history removes early endings and does not change arrival.** Inside the
  trained range the gain is small (within the trial noise); most of it is at 8–20 m, where
  the baseline ends early in 8–12% of trials and the history policy in 2–3%. It holds at
  every saved checkpoint from update 1000 on: 0.0–0.8% with history, 0.6–9.1% without. The
  same shows in the 0–20 m runs (Kalman estimate, 8–20 m: 15–21% to 0.3%). The ranges are
  because the stored per-distance results count a fall and a lost posture separately, and
  one trial can be both.
- **Why, as far as these runs show.** The history student copies the teacher faster (training
  success at update 200: 0.74 against 0.52), so the push curriculum climbs higher (level 0.60
  against 0.40) and it trains under stronger pushes. History and the harder curriculum are
  not separated here.
- **The nominal 8 m miss is a threshold effect.** The history policy stops 10.3 cm from the
  goal (median) against 6.9 cm, and the tolerance is 10 cm; the nominal tier is one robot, so
  every trial lands on the same side. Randomized, the two arrive equally often at 8 m.
- **Approach 3: history makes it worse in both forms.** The adapter is trained by plain PPO
  with no teacher, and its early training is no faster with history. The failures are not
  falls: with history in the adapter, 86% are a hand more than 25 cm off the plan's height.
  With history in the base as well, the 0.1 m walk fails in 46% of randomized trials on the
  foot-height test, although the history base walker alone behaves like the baseline one.
- The actor grows from 99,863 to 157,463 parameters (the adapter from 12,695 to 27,095) for
  the same number of environment steps.

**2. Kalman-filter position estimate (trained on 0–5 m).**

| Controller | 0–4.2 m | 8–20 m | 25–40 m |
|---|---:|---:|---:|
| Approach 2 | 100 / 98 / 99 | 38 / 17 / 58 | 0 / 0 / 14 |
| Approach 2, filter swapped in, not retrained | 100 / 97 / 98 | 99 / 22 / 78 | 41 / 2 / 33 |
| Approach 2, retrained with the filter | 100 / 96 / 97 | 100 / 20 / 78 | 50 / 2 / 25 |
| Approach 3 | 100 / 96 / 99 | 49 / 17 / 53 | 0 / 3 / 18 |
| Approach 3, filter swapped in, not retrained | 100 / 96 / 99 | 50 / 19 / 70 | 0 / 3 / 32 |
| Approach 3, adapter retrained with the filter | 100 / 92 / 92 | 50 / 20 / 66 | 0 / 2 / 28 |

Inside the training range the filter changes nothing; the estimate was not the limit
there. Beyond it, the filter is what lets approach 2 reach 20 m on the nominal robot (its
estimate is off by 6 cm there instead of 14) and raises arrival with an exact heading from
58% to 78% over 8–20 m. With the heading error included nothing changes, because no filter
on these sensors observes heading. Retraining with the filter is not needed: swapping it
under the existing policy gives the same gain, and the retrained policies lose their
posture on the 0.1 m walk in 8% (approach 2) and 13% (approach 3) of randomized trials.

A second evaluation of these six controllers kept every episode, which separates falls
(torso past 60°) from posture losses and gives the error of the walks that finished. It is
a fresh sample: arrival differs from the table above by up to 5 points. Randomized, both
heading tiers pooled:

| Controller | Fell, 0–4.2 m | Posture lost, 0–4.2 m | Fell, 8–20 m | Posture lost, 8–20 m |
|---|---:|---:|---:|---:|
| Approach 2 | 0.9% | 0.2% | 5.5% | 3.6% |
| Approach 2, filter swapped in | 0.9% | 0.9% | 8.0% | 6.8% |
| Approach 2, retrained with the filter | 0.6% | 1.4% | 3.1% | 4.8% |
| Approach 3 | 0.0% | 1.0% | 0.0% | 2.5% |
| Approach 3, filter swapped in | 0.0% | 1.6% | 0.0% | 3.3% |
| Approach 3, adapter retrained with the filter | 0.0% | 7.1% | 0.0% | 9.5% |

Swapping the filter under approach 2 raises early endings beyond the trained range (9% to
15% at 8–20 m); retraining with it does not. Approach 3 never fell; its retrained adapter
loses posture seven times as often in range. The final error of finished walks is 2–4 cm
(median) in range for all six. At 20 m with an exact heading the filter cuts it from 10.9 to
8.1 cm (approach 2) and from 12.9 to 9.2 cm (approach 3); with the heading error it stays
near 50 cm.

**3. Training on 0–20 m, tested inside (to 20 m) and outside (25, 40 m).** Arrival by
distance, randomized with an exact heading:

| Controller | 0–4.2 m | 8 m | 12 m | 16.5 m | 20 m | 25 m | 40 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| Approach 2, trained 0–5 m | 99 | 88 | 60 | 52 | 33 | 17 | 10 |
| Approach 2, 0–20 m, original setting | 98 | 100 | 83 | 54 | 21 | 12 | 2 |
| Approach 2, 0–20 m, history + Kalman (best) | 100 | 100 | 100 | 79 | 62 | 25 | 2 |
| … with the estimate's scale calibrated | 100 | 100 | 94 | 83 | 73 | 71 | 31 |
| Approach 3, trained 0–5 m | 99 | 90 | 65 | 29 | 29 | 31 | 4 |
| Approach 3, 0–20 m, original setting | 89 | 92 | 79 | 52 | 31 | 44 | 12 |
| Approach 3, 0–20 m, Kalman, short plans ×3 (best) | 96 | 96 | 90 | 83 | 56 | 52 | 19 |
| … with the estimate's scale calibrated | 97 | 98 | 90 | 81 | 85 | 71 | 33 |
| Privileged teacher, 0–20 m (not deployable) | 100 | 100 | 100 | 100 | 100 | 100 | 98 |

The same controllers on the nominal robot and with the heading error:

| Controller | 0–4.2 m | 8–20 m | 25–40 m |
|---|---:|---:|---:|
| Approach 2, trained 0–5 m | 100 / 98 / 99 | 38 / 17 / 58 | 0 / 0 / 14 |
| Approach 2, 0–20 m, original setting | 100 / 96 / 98 | 50 / 18 / 65 | 0 / 0 / 7 |
| Approach 2, 0–20 m, history + Kalman (best) | 100 / 99 / 100 | 75 / 21 / 85 | 0 / 1 / 14 |
| … with the estimate's scale calibrated | 100 / 97 / 100 | 100 / 22 / 88 | 100 / 2 / 51 |
| Approach 3, trained 0–5 m | 100 / 96 / 99 | 49 / 17 / 53 | 0 / 3 / 18 |
| Approach 3, 0–20 m, original setting | 100 / 85 / 89 | 50 / 18 / 64 | 0 / 3 / 28 |
| Approach 3, 0–20 m, Kalman, short plans ×3 (best) | 100 / 94 / 96 | 100 / 23 / 81 | 21 / 2 / 35 |
| … with the estimate's scale calibrated | 100 / 94 / 97 | 100 / 23 / 89 | 100 / 3 / 52 |
| Privileged teacher, 0–20 m (not deployable) | 100 / 100 / 100 | 100 / 98 / 100 | 100 / 57 / 99 |

- **The original setting** (one frame, leg odometry) did not work well. Approach 2 walks
  the longer plans more reliably (8 and 12 m improve, falls stay near 2%) but still misses
  16.5 m and beyond, because the estimate reads 0.7–1.0% long. Approach 3 lost the 0.1 m
  walk (38% randomized arrival): long plans took the training steps, and plans under 0.5 m
  fell from 9% of them to 3%.
- **Best approach 2: history and the Kalman estimate together.** With an exact heading it
  arrives in 98–100% of trials from 0 to 12 m and 62% at 20 m, with 0–2% falls. With the
  Kalman estimate alone the policy lost its posture in 6–38% of trials at 8 m and beyond
  (two runs); history alone did not improve the estimate.
- **Best approach 3: the Kalman estimate, with plans under 0.5 m drawn three times as
  often** (`--short-weight 3`), which restores the 0.1 m walk to 100% nominal and 80%
  randomized (the 0–5 m adapter has 99%). History made the 0–20 m adapter worse.
- **The filter swapped under the 0–20 m policies, with no retraining** (a second
  evaluation that kept every episode; falls and posture losses are separate there). For
  approach 2 it beats retraining with the filter: 8–20 m arrival with an exact heading is 64%
  with leg odometry, 85% swapped in and 72% retrained, and early endings at 8–20 m are 2%,
  5% and 16%. For approach 3 retraining is slightly ahead (67%, 77%, 84%) and all three end
  early in 1–2% of trials; the adapter never fell. The median error of finished walks at
  20 m drops from 12.4 to 7.6 cm (approach 2, swapped in) and from 10.2 to 6.8 cm (approach
  3, retrained). With the heading error it is about 50 cm for all six.
- **Out of range** (25 and 40 m) neither arrives reliably: 2–52% with an exact heading, 0–4%
  with the heading error. Training on a wider range did not make the policies extrapolate.
- **Calibrating the estimate's scale** (`calibrate.py`: one number per controller, from
  7–18 m walks on the nominal robot, no retraining) is outside the three variations but
  shows what is left. The over-reading is steady on the nominal robot, so there every
  distance to 40 m reaches 100%. Under randomization it recovers part: each randomized robot
  slips differently, and a real calibration would be done on the robot itself.
- **With the heading error included**, arrival is 40–55% at 8 m, 15–25% at 12 and 16.5 m
  and under 10% at 20 m and beyond, for every controller and variation. Reaching a point 20 m away within 10 cm
  needs a heading reference or a position fix; none of the three variations supplies one.

## The environment (`env.py`)

| | |
|---|---|
| Robot | Unitree G1, 23 actuated joints (the MJCF, gains and collisions of `../humanoid`) |
| Simulator | mjlab 1.6 on MuJoCo-Warp, physics 200 Hz, control 50 Hz, flat floor |
| Reference | for each N, a plan from `../humanoid`'s planner: footsteps, linear-inverted-pendulum MPC, whole-body IK (`plans.py`). Stand 0.4 s, walk N in ceil(N / 0.21) steps plus a closing step, stand. About 0.42 m/s. |
| Action | 23 numbers: joint target = plan's joint target one frame ahead + scale × action. A zero action plays the plan back, and the policy's last layer starts at zero, so training starts from the MPC plan. |
| Policy | MLP 109 → 256 → 256 → 23, ELU, observation normalization (`models.py`) |

### What the actor reads (109 numbers)

| Term | Size | Source on the robot |
|---|---:|---|
| `gyro` | 3 | pelvis IMU angular velocity |
| `gravity` | 3 | gravity direction from the IMU's attitude |
| `joint_pos` | 23 | encoders, relative to the default pose (with encoder bias) |
| `joint_vel` | 23 | encoder velocities |
| `actions` | 23 | the previous correction |
| `ref_joints` | 23 | the plan's next joint targets (controller memory) |
| `ref_phase` | 5 | stride phase (sin, cos, walking), plan distance still to go (capped at 1 m), plan ended |
| `goal` | 6 | where the plan and the goal are from where the robot believes it is, along and across the route, and its heading off the route: from leg odometry |

No simulator position, velocity or contact reaches the actor. `python -m parametrization.walk.check`
asserts that the actor group is exactly these terms.

**History (`--history 4`, a variation).** The baseline actor reads one frame. `../humanoid`'s
current policies read the last four frames of the five sensor terms (its 376-input setup).
With `--history 4` the actor here does the same: 4 × 75 sensor numbers, oldest first, plus
the 34 plan and goal numbers once, 334 in all. The critic and the teacher stay single-frame,
so the same teachers are used.

**Leg odometry (`odometry.py`).** The real G1 has no position sensor. The pelvis position is
integrated from the leg encoders and the IMU's orientation: each control step the pelvis
moves by minus the displacement of the feet relative to it, weighted toward the lowest foot
(softmax over foot height, 1 mm scale). Its errors in simulation are the ones it has on
hardware: foot slip, encoder bias, IMU tilt bias and heading drift. Training adds a cost on
the odometry's drift rate, so the gait plants its feet.

**Kalman estimate (`--estimator kalman`, a variation).** Per horizontal axis a three-state
filter (position, velocity, accelerometer bias). It predicts with the pelvis accelerometer
rotated by the IMU attitude (simulated with ±0.1 m/s² bias and ±0.3 m/s² noise) and corrects
with each foot's kinematic velocity, weighted by the same height softmax. A foot whose
velocity disagrees with the prediction by more than 0.7 standard deviations is
down-weighted, which is what removes most of the leg odometry's over-reading when a foot
slides. It uses the same sensors plus the accelerometer, which the G1's IMU has. It does not
observe heading or absolute position, so it cannot remove heading drift.

**Training-only signals.** The critic (188 numbers) also reads true base velocity, true
errors to the plan and goal, the odometry's error, tracked-body errors and foot contacts.
Rewards, terminations and the success test use the truth. A privileged teacher actor reads
the critic's observation; it is used to train the deployed actor and is never deployed.

### Rewards and episode ends

- Tracking the plan: BeyondMimic's terms as in `../humanoid` (torso position and orientation;
  body positions, orientations and velocities relative to the torso).
- The route: pelvis where the plan has it along the route, a cost for leaving it sideways
  beyond the gait's sway, heading.
- The goal, after the plan ends: pelvis on the goal, standing still with both feet down, and
  a one-time bonus the first time the robot has stayed within 10 cm for 1 s.
- Costs: action rate, joint limits, odometry drift rate, and for approach 3 the adapter's size.
- A training episode runs until 4 s after the plan ends (the robot must keep standing), or
  until a fall or the posture leaving the plan's (torso height or tilt, a hand or foot
  height, or 0.5 m from the plan's position).
- Starts: 30–50% from standing at the plan's first frame, the rest from a random frame of
  the plan with small offsets. A curriculum raises pushes with the success rate.

### Domain randomization

Applied in training and in the randomized evaluation tier (pushes in training only).

| What | Range |
|---|---|
| Foot–floor friction | 0.4 – 1.4, per episode |
| Link masses and inertias | ±10%, per env |
| Torso payload | −0.5 to +2.5 kg, per env |
| Torso centre of mass | ±3 cm forward, ±1.5 cm sideways, ±2 cm up, per env |
| PD gains (kp, kd) | × 0.85 – 1.15 per joint, per episode |
| Torque limits | × 0.9 – 1.0 |
| Joint friction | 0 – 0.6 N·m legs and waist, 0 – 0.3 N·m arms |
| Joint damping | 0 – 0.1 N·m·s/rad |
| Command delay | 0 – 15 ms |
| Encoder bias | ±0.01 rad per joint, per episode |
| Sensor noise per step | gyro ±0.2 rad/s, gravity ±0.05, joint position ±0.01 rad, joint velocity ±1.5 rad/s |
| Sensor bias per episode | gyro ±0.02 rad/s, gravity ±0.025 |
| Observation delay | 0 – 20 ms |
| IMU heading (odometry) | drift ±0.1°/s, initial offset ±1°, noise 0.05° |
| IMU tilt (odometry) | ±0.02 rad roll and pitch bias |
| Odometry start error | 2 cm (random-frame starts) |
| Pushes | pelvis velocity kicks every 1–3 s, 0.1 m/s growing to 0.4 m/s with the curriculum |

The ranges are judgement, not measurements of a particular robot. They are recorded in each
run's `run.json`.

### Robust profile (after the first hardware trials)

The first hardware trials (2026-11-17) ran `goal20` and `adapter20_k` on a G1. Both fell on a
1 m walk. That robot's ankle encoders and IMU read 1–2° off and its accelerometer 0.14 m/s²,
and in a controller-in-the-loop test both policies fell after sideways pushes of 0.2–0.3 m/s.
Their training pushes had stayed near 0.1–0.16 m/s: pushes grew with the curriculum level,
and the level ended at 0.2 (`goal20`) and 0.0 (`adapter20_k`).

`--profile robust` (`env.PROFILES`) changes only these ranges; `robust_p` is the same with a
push floor of 0.15 m/s.

| What | First profile (`v1`) | `robust` |
|---|---|---|
| Encoder bias, ankles | ±0.01 rad | ±0.04 rad |
| Encoder bias, other joints | ±0.01 rad | ±0.015 rad |
| Gravity observation bias | ±0.025 | ±0.045 |
| IMU tilt bias in the estimator | ±0.02 rad | ±0.045 rad |
| Accelerometer bias, noise | 0.1, 0.3 m/s² | 0.25, 1.0 m/s² |
| Pushes at level 0 and 1 | 0.1, 0.4 m/s | 0.3, 0.5 m/s |

What it gave, measured with the deployed C++ controller in native MuJoCo on a 1 m walk
(`../unitree_rl_mjlab_box/PARAMETRIZED_WALK.md`, "Why v2"):

- **Approach 3 improved.** `base_r_seed2` and `adapter20_r` (four sensor frames, Kalman
  estimate) first fall at three times that robot's ankle offsets (`adapter20_k`: at 1.5 times),
  fall in 1 of 40 runs with the offsets plus physics variation (10 of 40), and stand 0.3 m/s
  sideways pushes with the offsets (fell from 0.2 m/s). In the training env under the robust
  tier it loses its posture in 0–9% of trials up to 4.2 m (`adapter20_k`: 17–29% from 0.75 m),
  and at 0.1 m it arrives in 98% (37%).
- **Approach 2 did not.** Eight retrained checkpoints (`goal20_r`, `_ra` to `_rg`: stronger
  or gentler pushes, guidance floor 0.1, 4,000 updates, starting from `base_r_seed2`) did
  worse in the same tests than `goal20_hk2`, trained with the first profile. Their teacher
  (`teacher_goal20_rw`) has no falls under the robust tier, so the loss is in the student.
  Alone among the widened ranges, the ankle encoder bias is what the students cannot handle.
- Warm-starting a teacher from the first profile's teacher reached a higher push level than
  training it from scratch (base 1.0 against 0.6, goal 0.45 against 0.0).

The per-run numbers are in each run's `selection.json`; the deployed pair is `goal20_hk2` and
`adapter20_r` on `base_r_seed2`. The scripts behind the controller-in-the-loop tests are kept,
as they were run, in `docs/robustness_tests/` (they expect the scratch paths named in them).

### Head camera

The robot's model carries the G1's head camera, a RealSense D435i, as the MuJoCo camera
`robot/d435` (`env.py`: `CAMERA`): on `torso_link` at (0.0576, 0.0175, 0.4299) m, the lens
47.6° below the horizon. That is the pose of `d435_joint` in Unitree's
[URDF](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_23dof_rev_1_0.urdf#L522-L526).
The vertical field of view is 42°, which at 16:9 gives the 69° × 42° of the D435's colour
stream. Standing, the lens is 1.23 m above the floor and sees the floor from about 0.5 m to
2.5 m ahead.

The policies do not read the camera, and by default nothing is rendered, so training and
evaluation are unchanged. `env_cfg(..., camera=(640, 360))` renders it in every env each step
as the sensor `head_camera`: `env.scene["head_camera"].data.rgb` (uint8) and `.depth` (m).
`render.py --camera` puts its view beside the side view of a rollout.

## Training

Direct PPO on the 109-number observation did not learn the task (about 15% training
success). All deployed actors are therefore trained in two stages, the recipe `../humanoid`
uses for its deployed controllers (`guided.py`):

1. **Teacher.** PPO with a privileged actor that reads the critic's observation.
2. **Student.** PPO for the 109-input actor on its own rollouts, with one extra loss term:
   the squared distance of its mean action from the frozen teacher's on the same states.
   The term's weight is 0.1 for 200 updates, then falls linearly until update 1,000: to zero
   for the base walker, to a floor of 0.02 that it keeps for approach 2. That choice was made
   per controller on the selection distances (see "What went wrong").

PPO settings: 4,096 envs, 24 steps per update, γ 0.99, λ 0.95, 5 epochs × 4 minibatches.
Teachers use an adaptive learning rate from 1e-3 and a learned exploration std; students a
fixed 3e-4 and a fixed std of 0.05.

**Approach 2 — goal-conditioned policy (`train.py goal`).** One policy trained across a bank
of 160 plans with N from 0 to 5 m (stratified, never within 2 cm of an evaluation
distance). N reaches the policy through the plan's joint targets and the `goal` terms.

**Approach 3 — frozen base plus adapter (`train.py adapter`).** The base walker is a policy
trained on the 2 m plan only, then frozen. A 109 → 64 → 64 → 23 adapter (12,695 parameters)
reads the same observation and adds a correction clamped to ±0.5 (or ±1.0) action units per
joint; it is trained by PPO for 2,000 updates across the same 160-plan bank with a cost on
its size. No teacher is used for the adapter.

**Training to 20 m (a variation).** The `train20` bank is the same 160 plans plus 100 with N
from 5 to 20 m. Approach 2 gets its own teacher on that bank (2,000 updates); approach 3
keeps the 2 m base walker. `--short-weight 3` draws plans under 0.5 m three times as often.
Checkpoints are selected on the selection distances plus 7, 10, 14 and 18 m.

Checkpoints are chosen by measurement (`select.py`) on eight distances that are neither
training nor evaluation distances, averaged over the nominal robot and randomization.

## Evaluation (`evaluate.py`, `report.py`)

- Every trial starts from standing and uses the policy's mean action.
- **Stable arrival**: within 10 cm and 10° of the goal, slower than 0.1 m/s, upright, both
  feet down, held 0.5 s, and still so 5 s after the plan ends.
- Tiers: the nominal robot with clean sensors, and randomized physics and sensors
  (training's ranges, no pushes). The variations add a third, randomized with an exact IMU
  heading (`--tier dr_h`), to separate heading drift from everything else. The position
  estimate is the actor's only position source in all of them.
- Distances: 0 m; 0.1, 0.2, 0.3 m (one or two steps); 0.75, 1.4, 2.5, 3.5, 4.2 m (inside
  the training range, held out, both step parities); 8 and 20 m (beyond the range).
- Evaluation episodes end early only for a fall or lost posture, not for leaving the route.

## Running it

```bash
parametrization/walk/pipeline.sh plans                 # plan banks (CPU, ~2 min)
GPU=0 parametrization/walk/pipeline.sh base            # teacher on the 2 m plan → base walker → selection
GPU=1 parametrization/walk/pipeline.sh goal            # teacher across distances → approach 2 → selection
GPU=1 parametrization/walk/pipeline.sh goal 2          # a second seed
GPU=0 parametrization/walk/pipeline.sh adapter         # approach 3 on the selected base (bound 0.5)
GPU=2 parametrization/walk/pipeline.sh adapter 1.0     # the same with bound 1.0
GPU=0 parametrization/walk/pipeline.sh report          # parametrization/walk/runs/report.json and parametrization/walk/docs/RESULTS.md
GPU=3 parametrization/walk/pipeline.sh vary h          # a variation of both approaches: h, k, far, far_h, far_k, far_hk
SHORT=3 GPU=4 parametrization/walk/pipeline.sh vary far_k   # ... with plans under 0.5 m drawn three times as often
GPU=0 INIT=parametrization/walk/runs/teacher_base/model_1999.pt parametrization/walk/pipeline.sh robust base_teacher   # the robust profile: teacher (warm-started), ...
GPU=0 TEACHER_BASE=teacher_base_rw parametrization/walk/pipeline.sh robust base 2      # ... base walker (seed 2), ...
GPU=0 BASE=base_r_seed2 parametrization/walk/pipeline.sh robust adapter               # ... and its adapter; goal_teacher and goal likewise
python -m parametrization.walk.calibrate goal20_hk     # the estimate's distance scale for one controller
GPU=0 parametrization/walk/pipeline.sh variations goal_h,goal_k   # measure those → parametrization/walk/docs/VARIATIONS.md
python -m parametrization.walk.check                   # observation contract, kinematics, odometry, plan playback
python -m parametrization.walk.render --policy parametrization/walk/runs/goal/selected.pt --n 0.2 2.5 --out goal.mp4
python -m parametrization.walk.render --policy parametrization/walk/runs/goal/selected.pt --n 2.5 --camera --out goal_camera.mp4   # with the head camera's view
```

Requirements: the `.venv` here (mjlab 1.6, rsl-rl 5.4, torch), and the sibling repository
`../humanoid` (or `HUMANOID_ROOT`) for the robot model and the planner. Runs, banks and
checkpoints go to `parametrization/walk/runs/` (git-ignored).

## Differences from the design document

- **Observations.** The design document allows base velocity and goal errors from simulator
  pose. Here the actor gets neither: goal errors come from leg odometry.
- **Reference motion.** The design document trains from rewards alone. Here every policy
  outputs a correction to an MPC plan for the requested distance, as in `../humanoid`, and
  tracking that plan is part of the reward. The requested N therefore reaches the robot as
  a plan as well as a goal.
- **Teacher.** Deployed actors are trained with a privileged teacher's guidance.
- **Approach 3** is the residual-adapter variant: the frozen base walker is a fixed gait
  trained on one plan, and the adapter corrects its joint targets.
- **Success test.** Stricter than the document's: both feet down, and still passing 5 s
  after the plan ends.

## What went wrong

Five training rounds were needed. Each of these changed the code or the protocol.

1. **Direct PPO on the real-robot observation did not learn.** It plateaued near 15%
   training success. The teacher-then-student recipe replaced it.
2. **Goal seen from the robot's heading frame (round 1).** A goal 20 m ahead moves 35 cm
   sideways per degree of heading error, and policies that read it turned toward it and left
   the plan. The `goal` terms are now in the route's frame.
3. **Standing was never trained (round 2).** Training episodes ended 1 s after arrival, so
   what the policy did afterwards was untrained: the approach-2 teacher walked away from a
   0 m goal in 30% of trials, and student checkpoints swung between 29% and 92%. Episodes
   now continue 4 s after the plan ends, the success bonus is paid once, and evaluation
   requires the robot to still pass 5 s after the plan ends.
4. **The evaluation distances hid a failure (round 2).** All in-range evaluation plans had
   the same step parity as the base walker's 2 m plan. The distances now cover both
   parities; on the other parity the base walker fails every trial.
5. **Leg odometry was fooled by the gait (round 3).** With a 3 mm height scale, a swing foot
   skimming 2–5 mm above the floor kept a quarter of the weight, and the estimate fell short
   by 3–5% of the distance (19 cm at 4.2 m for the teacher). The scale is now 1 mm and the
   drift cost three times larger; the teacher's odometry error at 4.2 m is 5 cm. Weighting
   by planned contact or by simulated contact was worse than by height.
6. **The base walker's teacher was under-trained (round 4).** After 1,000 updates it stopped
   7 cm to the side of the goal, and the base walker reached 90% on its own plan. With 2,000
   updates the teacher is at 100% with a 1.5 cm median error.
7. **Approach-2 students degraded once guidance reached zero (round 4).** In both seeds,
   training success fell from about 0.6 to about 0.4 over the 1,000 updates after the fade,
   and the randomized score on the selection distances at the last checkpoint was 0.85 and
   0.90. With a floor of 0.02 every checkpoint from update 1,000 to 2,000 scores 0.91–0.95. At the selected
   checkpoints the difference is small: 83.5% and 80.3% overall under randomization without
   the floor, 83.2% and 85.2% with it (64 trials per distance). For the one-plan base walker
   the floor did not help: 96.9% on its plan with it, 99.0% without, so the base walker
   fades to zero. The no-floor students are in `parametrization/walk/runs/v4/`, the floor base walker in
   `parametrization/walk/runs/alt/`.
8. **Selecting checkpoints under randomization only (round 5).** The checkpoint it chose
   for approach 2's second seed fell forward while standing in 26–38% of nominal trials at
   1.4, 3.5 and 8 m. Approach-2 checkpoints vary a lot on the nominal robot (0.65–1.00 on the
   selection distances within one run); every adapter checkpoint scored 1.00. Selection now
   averages the nominal and randomized tiers. The second seed still falls in a few percent
   of nominal trials.
9. **Adapters were stopped too early (round 5).** At 1,000 updates they were still
   improving; they now train for 2,000, and the results list both.
10. **The 0–20 m plan bank starved the short walks (variation 3).** Plans are drawn
    uniformly per episode, and a 20 m episode is ten times as long as a 0.3 m one, so plans
    under 0.5 m fell from 9% of training steps to 3%. The 0–20 m adapters ended 30–40 cm
    from a 0.1 m goal in about 60% of randomized trials. Drawing those plans three times as often
    (`--short-weight 3`) restores their share. The selection distances also missed it: the
    shortest was 0.15 m, and the failure is at 0.1 m. The second-round selection adds 0.08 m.
11. **Policies retrained with the Kalman estimate are weaker on the 0.1 m walk
    (variation 2).** In the failing trials the estimate is within 2 cm of the truth. The
    0–5 m policy trained with the filter fails a third as often when run on leg odometry
    (2.5% against 7.2%); the 0–20 m one fails as often on either (about 15%); the baseline
    policy with the filter swapped in does not fail. So it is something these policies
    learn, not the filter's output at run time. I did not find what.

## Limits

- Simulation only. The randomization ranges are not fitted to a robot, and no part of this
  has run on hardware. Exporting the policy and the odometry to the robot is not done.
- The position source is dead reckoning. Under randomization its median error is about 5 cm
  at 4.2 m, 10 cm at 8 m and 45 cm at 20 m. With an exact heading it is 5 cm at 8 m and
  12 cm at 20 m, so about three quarters of the 20 m error is the assumed heading error
  (±1° offset, ±0.1°/s drift) and the rest the gait's slip. Arrival within 10 cm beyond
  about 5 m needs a heading reference or an external position fix.
- The variations are single runs. Several shared a GPU, so their wall-clock times in
  VARIATIONS.md are inflated; environment steps are the comparable cost.
- One gait: straight ahead at about 0.42 m/s on flat ground, from the planner in
  `../humanoid`. Plans are computed per distance before the walk (about a second each on a
  CPU); nothing replans during the walk.
- Approach 1 from the design document (a hand-written distance controller around a
  velocity-conditioned walking policy) is not built; it was not part of this request.
- Results are from two seeds for approach 2 and one base walker; adapters have one seed per
  bound.

## Files

| File | What it is |
|---|---|
| `env.py` | the mjlab env: plan bank command, observations, rewards, randomization, terminations |
| `odometry.py` | leg kinematics; leg odometry from encoders and IMU; the Kalman variant |
| `plans.py` | plans for a distance from `../humanoid`'s planner; plan banks |
| `models.py` | the 256 × 256 actor with a zero last layer; frozen actors |
| `guided.py` | teacher-guided PPO for the deployed actor |
| `train.py` | PPO for the base walker, approach 2 and approach 3 |
| `select.py` | checkpoint selection on the selection distances |
| `evaluate.py`, `report.py` | per-distance evaluation; the results document |
| `variations.py` | the history, Kalman and 0–20 m variations, measured against the baseline |
| `calibrate.py` | the position estimate's distance scale from long walks on the nominal robot |
| `render.py`, `curves.py`, `check.py` | videos, training curves, contract checks |
| `pipeline.sh` | the construction in order |
