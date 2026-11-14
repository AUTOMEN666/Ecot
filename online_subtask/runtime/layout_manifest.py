"""Freeze exactly the existing30 Random layouts; never generate or screen scenes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from export_binding import read, sha, canonical_sha
from task_contract import TASK, SCENE_SEED, LAYOUT_IDS, NOISE_SEED

SCHEMA = 'ecot.stack_bowls_random.seed1_existing30.v1'
SOURCE_FILES = (
    'env/global_configs.py', 'env/seed_manager/seed_manager.py',
    'src/eval_client/main.py', 'src/eval_client/eval_env.py',
    'task/RoboDojo/task_registry.py', 'task/RoboDojo/tasks/stack_bowls_random.py',
    'task/RoboDojo/config/stack_bowls_random.yml',
    'XPolicyLab/client_server/ws/model_server.py', 'XPolicyLab/client_server/ws/model_client.py',
    'third_party/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py',
)


def create_manifest(sdk_source, output):
    sdk = Path(sdk_source).resolve(strict=True)
    entries = []
    for index in LAYOUT_IDS:
        p = sdk / 'Assets/Eval_Layout/RoboDojo/arx_x5' / str(SCENE_SEED) / f'{TASK}_{index}.json'
        p = p.resolve(strict=True)
        if not isinstance(read(p), dict):
            raise ValueError('Native layout JSON object required')
        entries.append({'seed': SCENE_SEED, 'layout_id': index, 'path': str(p), 'sha256': sha(p)})
    d = {
        'schema': SCHEMA, 'task': TASK, 'sdk_source': str(sdk),
        'scene_seeds': [SCENE_SEED], 'layout_ids': list(LAYOUT_IDS), 'episodes': 30,
        'noise_seed': NOISE_SEED, 'entries': entries,
        'source_kind': 'existing_SDK_layouts_no_generation_no_outcome_screening',
        'source_sha256': {rel: sha(sdk / rel) for rel in SOURCE_FILES},
        'new_layouts_generated': 0, 'layout_repetitions': 0,
        'common_layout_digest': canonical_sha([{k: row[k] for k in ('seed', 'layout_id', 'sha256')}
                                               for row in entries]),
    }
    if len({row['sha256'] for row in entries}) != 30:
        raise ValueError('Thirty distinct scene files required; no duplicate padding')
    with Path(output).open('x') as out:
        json.dump(d, out, indent=2)
        out.write('\n')
    return d


def validate_manifest(path, *, check_files=True, expected_sha256=None):
    path = Path(path).resolve(strict=True)
    if expected_sha256 is not None and sha(path) != expected_sha256:
        raise ValueError('Externally pinned layout manifest SHA differs')
    d = read(path)
    expected = {'schema': SCHEMA, 'task': TASK, 'scene_seeds': [1],
                'layout_ids': list(LAYOUT_IDS), 'episodes': 30, 'noise_seed': 42,
                'new_layouts_generated': 0, 'layout_repetitions': 0,
                'source_kind': 'existing_SDK_layouts_no_generation_no_outcome_screening'}
    if any(d.get(k) != v for k, v in expected.items()):
        raise ValueError('Exact existing seed1 Random30 protocol required')
    rows = d.get('entries')
    if not isinstance(rows, list) or len(rows) != 30:
        raise ValueError('All30 frozen entries required')
    sdk = Path(d['sdk_source']).resolve(strict=True)
    if set(d.get('source_sha256', {})) != set(SOURCE_FILES):
        raise ValueError('Native task/source pin inventory differs')
    for relative, digest in d['source_sha256'].items():
        if sha(sdk / relative) != digest:
            raise ValueError('Native task, cap, rewards or preparation source changed')
    for index, row in enumerate(rows):
        if (set(row) != {'seed', 'layout_id', 'path', 'sha256'} or type(row['seed']) is not int
                or row['seed'] != 1 or type(row['layout_id']) is not int or row['layout_id'] != index):
            raise ValueError('Exact seed1/layout0..29 order required')
        wanted = sdk / 'Assets/Eval_Layout/RoboDojo/arx_x5/1' / f'{TASK}_{index}.json'
        if Path(row['path']).resolve(strict=True) != wanted.resolve(strict=True):
            raise ValueError('Random layout cannot be replaced by a clean or foreign task layout')
        if check_files and sha(row['path']) != row['sha256']:
            raise ValueError('Frozen scene bytes changed')
    if len({r['sha256'] for r in rows}) != 30 or d['common_layout_digest'] != canonical_sha([
            {k: r[k] for k in ('seed', 'layout_id', 'sha256')} for r in rows]):
        raise ValueError('Unique30 layout digest differs')
    return d


def rows_for(manifest, seed, ids):
    if (type(seed) is not int or seed != SCENE_SEED or not ids or len(set(ids)) != len(ids)
            or any(type(index) is not int or index not in LAYOUT_IDS for index in ids)):
        raise ValueError('Exact seed1/existing Random30 subset required')
    table = {row['layout_id']: row for row in manifest['entries']}
    return [dict(table[index]) for index in ids]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    d = create_manifest(a.sdk_root, a.output)
    print(json.dumps({'manifest': str(a.output), 'sha256': sha(a.output),
                      'episodes': d['episodes'], 'native_started': False}))


if __name__ == '__main__':
    main()
