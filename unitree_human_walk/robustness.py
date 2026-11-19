"""How much the human walk tolerates, with the C++ controller in native MuJoCo.

Every run starts in the Velocity policy and hands over as the profile does; the arms' weight
is on the robot; Velocity takes over again after the walk.  One nominal robot per run: these
are single trials, not rates.

  python -m unitree_human_walk.robustness --output logs/human_walk_robustness
"""
import argparse
import json
from pathlib import Path

from . import TRAINING
from .runtime import build_core
from .simulate import play


def sweep(output, root=TRAINING):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    lib = build_core(output)

    def run(**kw):
        r = play(output, root, from_velocity=True, lib=lib, save=False, **kw)
        w = r['walk']
        after = w.get('on_velocity_afterwards')
        return {'result': r['stop_reason'], 'error_m': round(w['error_m'], 3) if w else None,
                'largest_pelvis_tilt_deg': round(w['largest_pelvis_tilt_deg'], 1) if w else None,
                'error_at_the_hand_back_m': round(after['error_at_the_hand_back_m'], 3) if after else None,
                'drift_on_velocity_m': round(after['drift_m'], 3) if after else None}

    report = {'method': 'unitree_human_walk.simulate, from Velocity, arm weight on the robot, Velocity again after the '
                        'walk as the profile sets it, then 10 s watched; one run per row',
              'hardware_tested': False, 'nominal': run(), 'ankle_offsets': {}, 'pushes': {}, 'pushes_with_offsets': {},
              'foot_friction': {}, 'holding_at_the_end_instead': {}}
    for multiple in (.5, 1., 1.5, 2., 3.):
        report['ankle_offsets'][f'{multiple:g}x'] = run(ankle_offsets=multiple)
        report['holding_at_the_end_instead'][f'ankle offsets {multiple:g}x'] = run(ankle_offsets=multiple, return_s=None)
    for key, offsets in (('pushes', 0.), ('pushes_with_offsets', 1.)):
        for speed in (.2, .3, .4):
            for direction, name in ((90., 'left'), (-90., 'right')):
                for at in (1.5, 3., 4.5):
                    report[key][f'{speed:g} m/s to the {name} at {at:g} s'] = run(push=(speed, direction, at), ankle_offsets=offsets)
    for friction in (.4, .6, .8, 1., 1.2, 1.4):
        report['foot_friction'][f'{friction:g}'] = run(friction=friction)
        report['holding_at_the_end_instead'][f'foot friction {friction:g}'] = run(friction=friction, return_s=None)

    def falls(rows):
        return sum(r['result'] != 'completed' for r in rows.values())
    report['summary'] = {
        'first_ankle_offset_multiple_that_falls': next((k for k, r in report['ankle_offsets'].items() if r['result'] != 'completed'), 'none up to 3x'),
        **{f'falls_{key}_{speed:g}_m_s_of_6': falls({k: r for k, r in report[key].items() if k.startswith(f'{speed:g} ')})
           for key in ('pushes', 'pushes_with_offsets') for speed in (.2, .3, .4)},
        'falls_foot_friction_of_6': falls(report['foot_friction']),
        'falls_holding_at_the_end_instead_of_11': falls(report['holding_at_the_end_instead']),
        'which_fell_holding': [k for k, r in report['holding_at_the_end_instead'].items() if r['result'] != 'completed']}
    (output / 'robustness.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, default=TRAINING)
    a = parser.parse_args()
    sweep(a.output, a.training_root)
