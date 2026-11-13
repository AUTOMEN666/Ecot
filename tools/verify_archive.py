"""Verify preserved-source mappings without imports, credentials or network calls."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    amendments = {}
    if (root / 'online_subtask/provenance/STEADY_SOURCE_MAP.json').exists():
        from verify_steady import verify
        amendments = {name: row['sha256'] for name, row in verify(root).items()
                      if row['base_sha256'] != row['sha256']}
    mapping = json.loads((root / 'online_subtask/provenance/SOURCE_MAP.json').read_text())
    seen = set()
    changes = []
    for row in mapping['mapped_files']:
        relative = Path(row['destination'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen:
            raise ValueError('Invalid or duplicate destination: ' + str(relative))
        seen.add(str(relative))
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing/nonregular archived source: ' + str(relative))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != amendments.get(str(relative), row['destination_sha256']):
            raise ValueError('Archived source changed: ' + str(relative))
        if row['byte_identical'] and str(relative) not in amendments:
            if digest != row['source_sha256']:
                raise ValueError('Original source identity differs: ' + str(relative))
        elif not row['byte_identical']:
            changes.append(str(relative))
    expected_change = ['online_subtask/runtime/test_mainhistory93_65536.py']
    if changes != expected_change:
        raise ValueError('Unexpected deployed-source adaptations: ' + repr(changes))
    generated = set()
    for row in mapping['generated_files']:
        relative = Path(row['destination'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen | generated:
            raise ValueError('Invalid or duplicate generated path: ' + str(relative))
        generated.add(str(relative))
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing/nonregular archive utility: ' + str(relative))
        if hashlib.sha256(path.read_bytes()).hexdigest() != amendments.get(str(relative), row['destination_sha256']):
            raise ValueError('Archive utility/configuration changed: ' + str(relative))
    config = json.loads((root / 'online_subtask/configs/mainhist93_tok65536_v1.json').read_text())
    for path_key, digest_key in [('catalog_relative_to_online_subtask', 'catalog_sha256'),
                                 ('prompt_relative_to_online_subtask', 'prompt_sha256')]:
        path = root / 'online_subtask' / config['selector'][path_key]
        if hashlib.sha256(path.read_bytes()).hexdigest() != config['selector'][digest_key]:
            raise ValueError('Fixed catalogue/prompt differs from configuration')
    print(json.dumps({'verified_historical_mappings': len(seen), 'verified_archive_utilities': len(generated),
                      'adapted_sources': changes,
                      'explicit_steady_adaptations': sorted(amendments),
                      'configuration_id': config['config_id'], 'network_or_models_called': False}))


if __name__ == '__main__':
    main()
