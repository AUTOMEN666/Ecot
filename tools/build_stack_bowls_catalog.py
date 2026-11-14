"""Extract original training strings only. No API, future test frames or rewrites."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as src:
        for block in iter(lambda: src.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('Duplicate JSON key')
        out[key] = value
    return out


def read(raw):
    return json.loads(raw, object_pairs_hook=unique)


def build(dataset_manifest, context_manifest):
    data = read(Path(dataset_manifest).read_text())
    context = read(Path(context_manifest).read_text())
    episodes = data['episodes']
    if (data['task_name'] != 'stack_bowls' or data['horizon'] != 50
            or data['fps'] != 25 or data['action_dim'] != 14
            or sorted(e['episode_index'] for e in episodes) != list(range(3000, 3100))
            or any(e['instruction'] != 'Stack the three bowls together.' for e in episodes)
            or context['source_field'] != 'Planning.high_level_subtask.instruction'
            or context['language_values_modified'] is not False
            or context['dropout_probability'] != 0
            or context['baseline_manifest_sha256'] != sha(dataset_manifest)):
        raise ValueError('Require the actual approved Stack Bowls100 original-text projection')
    cache = Path(context['cache'])
    before = cache.stat()
    cache_sha = sha(cache)
    if cache_sha != context['cache_sha256']:
        raise ValueError('Original training cache hash differs')
    expected = {(e['episode_index'], frame) for e in episodes
                for frame in range(e['valid_action_count'])}
    seen, counts = set(), Counter()
    with cache.open() as src:
        for line in src:
            row = read(line)
            key = (row['episode_index'], row['frame_index'])
            if key in seen or key not in expected:
                raise ValueError('Duplicate, foreign or terminal observation in training projection')
            seen.add(key)
            value = read(row['context']) if isinstance(row['context'], str) else row['context']
            if (set(value) != {'subtask'} or not isinstance(value['subtask'], str)
                    or not value['subtask'].strip() or value['subtask'] != value['subtask'].strip()
                    or len(value['subtask']) > 1200):
                raise ValueError('Original nonempty Subtask only; do not normalize or rewrite it')
            counts[value['subtask']] += 1
    after = cache.stat()
    if seen != expected or (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Training coverage incomplete or source changed during extraction')
    return {
        'schema': 'ecot.stack_bowls.original_training_subtask_catalog.v1',
        'training_task': 'stack_bowls', 'instruction': 'Stack the three bowls together.',
        'source_field': 'Planning.high_level_subtask.instruction',
        'dataset_manifest_sha256': sha(dataset_manifest),
        'context_manifest_sha256': sha(context_manifest), 'source_cache_sha256': cache_sha,
        'source_episode_indices': list(range(3000, 3100)), 'valid_training_starts': len(seen),
        'language_values_modified': False, 'test_observations_used': False,
        'candidate_count': len(counts),
        'candidates': [{'id': f'U{index:04d}', 'text': text, 'training_start_count': counts[text]}
                       for index, text in enumerate(sorted(counts), 1)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest', type=Path, required=True)
    parser.add_argument('--context-manifest', type=Path, required=True)
    parser.add_argument('--output', required=True, help='Fresh path, or - for stdout')
    args = parser.parse_args()
    raw = json.dumps(build(args.dataset_manifest, args.context_manifest), ensure_ascii=False,
                     indent=2, allow_nan=False) + '\n'
    if args.output == '-':
        print(raw, end='')
    else:
        with Path(args.output).open('x') as out:
            out.write(raw)


if __name__ == '__main__':
    main()
