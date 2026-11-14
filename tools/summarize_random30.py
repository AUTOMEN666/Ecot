"""Publish a complete30-case SDK success rate; do not count engineering or stops."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'online_subtask/runtime'))
from export_binding import read, sha
from acceptance import SCHEMA
from task_contract import TASK, CHECKPOINT_STEP, PROTOCOL_CONFIG_ID


def summarize(paths):
    records = [read(path) for path in paths]
    if len(records) != 30:
        raise ValueError('Require all30 accepted formal cases; interruptions are not completed failures')
    if sorted(row.get('layout_id', -1) for row in records) != list(range(30)):
        raise ValueError('Exactly one formal case per seed1/layout0..29 required')
    if len({row.get('episode_id') for row in records}) != 30 or len({row.get('manifest_sha256') for row in records}) != 1:
        raise ValueError('Unique episodes and one frozen scene cohort required')
    for row in records:
        if (row.get('schema') != SCHEMA or row.get('passed') is not True or row.get('phase') != 'formal'
                or row.get('task') != TASK or row.get('scene_seed') != 1 or row.get('step') != CHECKPOINT_STEP
                or row.get('protocol_config_id') != PROTOCOL_CONFIG_ID or type(row.get('SDK_success')) is not bool):
            raise ValueError('Only actual formal Stack Bowls EMA10000 cases are eligible')
        for path, digest in row['source_sha256'].items():
            if sha(path) != digest:
                raise ValueError('Actual case evidence changed')
    successes = sum(row['SDK_success'] for row in records)
    return {'schema': 'ecot.stack_bowls_random.formal30_results.v1', 'task': TASK,
            'scene_seed': 1, 'noise_seed': 42, 'step': CHECKPOINT_STEP,
            'completed_episodes': 30, 'unique_layouts': 30, 'successes': successes,
            'success_rate': successes / 30, 'SDK_score_mean': sum(row['SDK_score'] for row in records) / 30,
            'engineering_in_denominator': False, 'interruption_as_failure': False,
            'manifest_sha256': records[0]['manifest_sha256'],
            'records': sorted(records, key=lambda row: row['layout_id'])}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipts', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); d = summarize(a.receipts)
    with a.output.open('x') as out:
        json.dump(d, out, indent=2); out.write('\n')
    print(json.dumps({k: v for k, v in d.items() if k != 'records'}))
