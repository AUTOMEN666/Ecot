"""Prepare exactly30 distinct formal cases AFTER a source-verified pilot acceptance."""
from __future__ import annotations
import json
from pathlib import Path
from copy import copy
from prepare_native_sim import parser, prepare
from acceptance import verify_engineering
from layout_manifest import validate_manifest
from task_contract import LAYOUT_IDS, OUTPUT_NAMESPACE


def prepare_cohort(a):
    validate_manifest(a.layout_manifest, expected_sha256=a.manifest_sha256)
    if a.engineering_acceptance is None:
        raise ValueError('Separate engineering acceptance is mandatory before formal30')
    verify_engineering(a.engineering_acceptance, a.manifest_sha256)
    if OUTPUT_NAMESPACE not in a.output.resolve().parts or a.output.exists():
        raise ValueError('Fresh isolated formal cohort directory required')
    a.output.mkdir(parents=True)
    plans = []
    for index in LAYOUT_IDS:
        case = copy(a); case.phase = 'formal'; case.layout = index
        case.output = a.output / f'layout_{index:03d}'
        prepare(case)
        plans.append({'layout_id': index, 'plan': str(case.output / 'sim_launch.json')})
    result = {'schema': 'ecot.stack_bowls_random.formal30_plan.v1', 'scene_seed': 1,
              'episodes': 30, 'layout_ids': list(LAYOUT_IDS), 'noise_seed': 42,
              'manifest_sha256': a.manifest_sha256, 'plans': plans,
              'engineering_in_denominator': False, 'new_layouts_generated': 0,
              'native_API_GPU_started': False, 'execution_mode': 'sequential cases; fresh policy process and immutable policy audit per case; reset selector/history/RNG'}
    (a.output / 'cohort.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = parser(include_layout=False); p.description = __doc__
    a = p.parse_args()
    if a.phase != 'formal':
        p.error('prepare_cohort requires --phase formal; engineering is separate')
    print(json.dumps(prepare_cohort(a)))
