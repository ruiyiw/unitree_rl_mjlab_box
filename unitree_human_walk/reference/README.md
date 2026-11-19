# Walk 2.2 m as a person does: a policy for Unitree's 23-DoF G1, tracked from motion capture

![Unitree's 23-DoF G1 walking 2.2 m as the person in the clip does](g1_walk_human.gif)

*The policy's mean actions in mjlab, in real time. The white ticks on the floor are a foot apart.*

One fixed motion: stand, walk 2.2 m straight ahead in about 4.7 s, stop and stand.

The reference is a person's walk, not an MPC plan. Everything after the reference is the same as [policy-go-to](../policy-go-to/README.md): [../motion.py](../motion.py) trains a PPO correction to the reference's joint targets, with the same observation, rewards, terminations, curriculum and PPO settings.

**Result** (1,024 runs from the start, the policy's mean actions, 60 s each, no pushes):
- every run does the whole motion and is still standing, upright, at 60 s;
- it walks 2.20 m of the reference's 2.21 m, at 0.55 m/s (peak 1.05 m/s);
- the torso stays a median 2.2 cm from the reference's.

**Under training's pushes** (up to 0.4 m/s to the pelvis every 1–3 s, for the whole 60 s) the kept checkpoint is weak: 92.1% do the motion, and 0.3% are still at the end spot at 60 s (the other 99.7% were stopped for leaving the tracking bounds; none tilted 60°, the fall check). See [Checkpoints](#checkpoints).

## The reference

[plans.py](plans.py) makes it from one segment of [Unitree's retargeting of LAFAN1](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) to the 29-DoF G1: clip `walk3_subject1`, frames 6679–6821 at 30 fps. The clip is not in the repository (LAFAN1's licence is CC BY-NC-ND 4.0); `python plans.py --download` fetches it.

1. **The segment** runs from the stillest frame before the walk to the stillest after it: six steps, the longest 0.49 m, the last closing the feet.
2. **The 23-DoF pose.** The root and the 23 shared joints are copied. The two models' legs are the same, so the feet land where the clip's do (within 3 mm). Inverse kinematics then makes up for the 6 missing joints (waist roll and pitch, wrist pitch and yaw): the pelvis leans to bring the torso near the clip's. The torso ends within 0.07 rad of the clip's, against 0.22 rad with the joints only copied.
3. **The plan's frame:** the origin under the pelvis at the first frame, x along its heading, the soles on the floor.
4. **50 Hz** by cubic splines. The clip's clock is eased in and out over 0.3 s, so the plan starts and ends at rest. It stands 0.4 s before the walk and 1.0 s after.

**Checked kinematically** (the numbers `plans.py` prints):
- **joint limits:** the nearest joint is 0.07 rad inside its limit; the fastest joint moves at 8.1 rad/s;
- **foot sliding:** a sole on the floor moves at a median 0.8 cm/s (95th percentile 6.7 cm/s);
- **the floor:** a sole's edge dips at most 9 mm below it, at toe-off.

The reference is kinematic only. Played back with no correction, every run leaves the tracking bounds after 0.48 m.

## Training

`motion.py train` ran 1,000 PPO iterations of 4,096 envs × 24 steps (19 min on an A100). The curriculum level reached 1 before iteration 500.

## Checkpoints

Every checkpoint from 500 on does the motion in all 64 undisturbed runs. They differ in how they stand afterwards:

| Iteration | Torso from the reference | Feet apart 4 s after the end (reference: 0.20 m) | Under pushes: does the motion | Under pushes: at the end spot, upright, at 60 s |
|--:|--:|--:|--:|--:|
| 500 (kept) | 2.1 cm | 0.13 m | 92.1% | 0.3% |
| 600 | 5.5 cm | 0.26 m | 99.4% | 75.6% |
| 700 | 3.7 cm | 0.26 m | 99.4% | 86.9% |
| 800 | 6.6 cm | 0.46 m | 99.0% | 88.3% |
| 900 | 7.7 cm | 0.57 m | 99.6% | 93.8% |
| 999 | 5.1 cm | 0.56 m | 100% | 97.1% |

Iterations 500 and 999 were measured under pushes on 1,024 runs, the others on 512; the first two columns are from 64 undisturbed runs.

The later checkpoints withstand the pushes by leaving the reference's stance: after the plan ends they spread their feet about 0.5 m apart and hold an arm out. The retrained MPC go-to policy does the same (feet 0.51–0.62 m apart). Iteration 500 keeps the person's stance and is the one in `actors/policy.pt` and the GIF. `motion.py select --pushes` picks 999.

The table task's go-to policy avoids this with rewards on the feet (`feet_parallel`, `feet_track`); `motion.py` has none.

## Against the MPC go-to policies

[../../results/human-walk-20261119/](../../results/human-walk-20261119/README.md) has the comparison and its scripts.

## Reproduce

In the env of the [box-carrying policy](../README.md#reproduce), with scipy:

```sh
cd policies/policy-walk-human
python plans.py --download       # data/lafan1/g1/walk3_subject1.csv
python plans.py                  # banks/plan.pt (1 s)
python ../motion.py train        # runs/policy/
python ../motion.py select       # checkpoints 500 on; the best -> actors/policy.pt
python ../motion.py play         # eval/policy.json
python ../motion.py play --pushes
python ../motion.py render       # g1_walk_human.gif
```
