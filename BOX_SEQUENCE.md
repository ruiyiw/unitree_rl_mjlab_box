# G1 23-DoF box sequence

The selected `policy.pt` bundles five frozen actors and references for
**go-to → take → carry → put-down → return**. Return uses its separately selected
actor. This coordinated sequence retains the source reference controller and
sensor-only switching; it is not a separately trained single network.

## Build and launch

Use the same prerequisites as human-push, listed in [BOX.md](BOX.md#build-once).
From the repository root:

```bash
cmake -S deploy/robots/g1_23dof -B deploy/robots/g1_23dof/build
cmake --build deploy/robots/g1_23dof/build -j4
cd deploy/robots/g1_23dof/build
./g1_ctrl --config ../config/box_sequence.yaml --network enp5s0
```

Replace `enp5s0` with your working human-push Ethernet interface. Run one
controller at a time. No Python wrapper or virtual environment is needed.
The executable checks the profile automatically at startup.
For separate stage commands, see [BOX.md](BOX.md#launch-one-policy).

## Activate

Follow the same suspended startup and debug-mode procedure as human-push.
Start at the go-to pose, facing the source table with the box on it.

1. **L2 + Up**: transition from Passive to FixStand over three seconds.
2. **R2 + A**: start the sequence once FixStand and the physical setup are ready.
3. **L2 + B**: return to Passive.

There are no commanded base resets or joint-history resets between stages.

## Reference clock and handoffs

Each actor receives the same 75 proprioceptive inputs. Each 20 ms step computes
`target = qcmd[min(clock + 1, last)] + scale * residual`. Actor normalization
stays frozen; residuals are unclipped and retained across stage switches.

| Stage | Reference frames | Reference duration |
|---|---:|---:|
| Go-to | 391 | 7.80 s |
| Take | 796 | 15.90 s |
| Carry | 391 | 7.80 s |
| Put-down | 1001 | 20.00 s |
| Return | 466 | 9.30 s |

The reference clock holds at its last frame. A handoff requires the reference
to have ended and all of these conditions continuously for one second:

- Pelvis gyro norm < 0.25 rad/s.
- Pelvis tilt < 15°, from projected gravity.
- Every measured joint within 0.3 rad of the stored endpoint **qpos**.
- Every joint speed < 0.5 rad/s.

A failed condition resets the dwell timer. The C++ float32 timer and the source
controller's fresh-frame behavior are preserved, including the first-frame
repeat following handoff. A stage that fails to settle within its reference
duration plus 12 seconds faults to Passive. Completion holds the final return
reference and actor until the operator exits.

These signals establish sensor settling only. Box contact, position, support,
release, and global robot position do not gate hardware switching. In the source
GPU evaluator they score physical success separately; therefore a completed
controller sequence is not itself evidence that the box was delivered.

## Physical layout

`config/policy/box/v1/params/layout.json` preserves the training layout. Its
historical four-policy/shared-navigation labels are retained for source-hash
compatibility; the v3 bundle and C++ controller select five distinct actors.
The physical dimensions are:

- Box: 0.28 × 0.45 × 0.325 m, mass 0.5 kg.
- Both table tops: 0.90 m high, footprints 0.60 × 1.125 m.
- Source table pose: `(0.30, 0.30, 0)`; destination: `(2.374186, -0.20, 0)`.
- Planned start stance: `(0, -2.1365, π/2)`; go-to travels 1.524 m toward pickup.
- Take picks up and turns right 90°; carry travels along +x.
- Put-down places, releases, steps back 0.18 m, lowers arms, then turns toward home.
- Return walks approximately 2.1344 m home, with its own selected actor.

Pose triples above are metres/metres/yaw radians in the shared scene frame.
The reference root poses include IK offsets from these planned stance anchors;
use the bundled references for exact initialization. The controller has no
world-position correction input and does not adapt this route to new furniture
positions or box dimensions.

## Validation and simulation

The C++ build and offline checks passed; real-robot box execution remains
untested here. [BOX_VALIDATION.md](BOX_VALIDATION.md) records the results and
optional CPU playback and GPU evaluation commands.
