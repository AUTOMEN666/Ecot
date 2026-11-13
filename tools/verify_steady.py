"""Verify explicit steady adaptations without rewriting historical provenance."""
import hashlib
import json
from pathlib import Path


def verify(root):
    manifest = json.loads((root / 'online_subtask/provenance/STEADY_SOURCE_MAP.json').read_text())
    if manifest['base_commit'] != '8e4f340940c56fbbe5c30e0ee578d6e5b8678321':
        raise ValueError('Unexpected source-branch base commit')
    files = {}
    for row in manifest['files']:
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts or '.git' in rel.parts or str(rel) in files:
            raise ValueError('Invalid/duplicate steady manifest path')
        path = root / rel
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or nonregular steady source: ' + str(rel))
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Steady source differs from explicit manifest: ' + str(rel))
        files[str(rel)] = row
    for name in ('mainhist93_tok65536_v1', 'mainhist93_tok65536_steady_v1'):
        config = json.loads((root / 'online_subtask/configs' / (name + '.json')).read_text())
        sections = [config['selector']] + ([config['observer']] if 'observer' in config else [])
        for section in sections:
            path = root / 'online_subtask' / section['prompt_relative_to_online_subtask']
            if hashlib.sha256(path.read_bytes()).hexdigest() != section['prompt_sha256']:
                raise ValueError('Prompt differs from its declared configuration')
    return files


def main():
    files = verify(Path(__file__).resolve().parents[1])
    print(json.dumps(dict(verified_files=len(files),
                          modified=sum(r['base_sha256'] is not None and r['base_sha256'] != r['sha256'] for r in files.values()),
                          added=sum(r['base_sha256'] is None for r in files.values()),
                          actual_API_GPU_simulator_called=False)))


if __name__ == '__main__':
    main()
