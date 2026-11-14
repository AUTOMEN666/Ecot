"""Verify current source hashes and Stack Bowls protocol without API/models."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    index = json.loads((root / 'online_subtask/provenance/SOURCE_INDEX.json').read_text())
    if index['source_commit'] != '95188146ce7191aa15d8a23a545707712e267d81':
        raise ValueError('Source branch identity changed')
    seen = set()
    for row in index['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen:
            raise ValueError('Invalid/duplicate source path')
        seen.add(str(relative))
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing/nonregular source: ' + str(relative))
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Source changed: ' + str(relative))
    config = json.loads((root / 'online_subtask/configs/stack_bowls_random_mainhist93_ema10000_v1.json').read_text())
    for path_key, digest_key in [('catalog_relative_to_online_subtask', 'catalog_sha256'),
                                 ('prompt_relative_to_online_subtask', 'prompt_sha256')]:
        path = root / 'online_subtask' / config['selector'][path_key]
        if hashlib.sha256(path.read_bytes()).hexdigest() != config['selector'][digest_key]:
            raise ValueError('Frozen catalogue/prompt differs')
    if (config['selector']['catalog_count'] != 297 or config['policy']['EMA_step'] != 10000
            or config['policy']['SDK_action_limit'] != 800 or config['evaluation']['episodes'] != 30
            or config['evaluation']['layout_ids'] != list(range(30))
            or config['policy']['source_training_eval_ecot'] is not False
            or config['policy']['effective_eval_ecot'] is not True):
        raise ValueError('Stack Bowls evaluation contract differs')
    print(json.dumps(dict(verified_source_files=len(seen), source_commit=index['source_commit'],
        configuration_id=config['config_id'], actual_API_GPU_simulator_called=False)))


if __name__ == '__main__':
    main()
