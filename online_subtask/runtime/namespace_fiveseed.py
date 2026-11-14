"""Private SDK namespace: preserve official robot, cameras, layouts and rewards."""
import hashlib
import ast
import importlib.util
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def links(source, destination, exclude=()):
    destination.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        if item.name not in exclude:
            (destination / item.name).symlink_to(item, target_is_directory=item.is_dir())


def patch_seed_source(text, rows):
    ids = [row['layout_id'] for row in rows]
    if not ids or len(set(ids)) != len(ids) or not all(type(i) is int and i >= 0 for i in ids):
        raise ValueError('Distinct nonnegative official seed1 stack_bowls_random layout IDs required')
    needle = '        all_layout_ids = list(range(len(matching_files)))'
    if text.count(needle) != 1:
        raise ValueError('Native seed declaration changed')
    text = text.replace(needle, needle + '\n        _pi05_ids = ' + repr(ids) + '\n'
        '        if not set(_pi05_ids) <= set(all_layout_ids):\n'
        "            raise ValueError('Missing requested official layout')\n"
        '        all_layout_ids = _pi05_ids')
    needle = '        data = load_json(file_path)'
    if text.count(needle) != 1:
        raise ValueError('Native layout load site changed')
    hashes = {row['layout_id']: {'path': row['path'], 'sha256': row['sha256']} for row in rows}
    guard = ('        import hashlib\n        _pi05_layouts = ' + repr(hashes) + '\n'
             '        _expected = _pi05_layouts.get(seed)\n'
             '        if (_expected is None or str(Path(file_path).resolve()) != _expected["path"]\n'
             '                or hashlib.sha256(Path(file_path).read_bytes()).hexdigest() != _expected["sha256"]):\n'
             '            raise ValueError("Official layout bytes changed")\n')
    return text.replace(needle, guard + needle)


def verify_private_assets(source):
    """Read actual native stdlib-only config through its private import path."""
    source=Path(source);path=source/'env/global_configs.py';tree=ast.parse(path.read_text())
    allowed={'os.path.abspath','os.path.dirname','os.path.join','os.environ.get'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Import)and any(alias.name!='os'for alias in node.names):
            raise ValueError('Native global config imports changed; inspect before overlay use')
        if isinstance(node,ast.ImportFrom):raise ValueError('Native global config imports changed')
        if isinstance(node,ast.Call)and ast.unparse(node.func)not in allowed:
            raise ValueError('Native global config has an unreviewed executable call')
    spec=importlib.util.spec_from_file_location('_paired_native_private_global_config',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    actual=Path(module.ASSETS_PATH).resolve(strict=True)
    if actual!=(source/'Assets').resolve(strict=True)or module.BENCHMARK!='RoboDojo':
        raise ValueError('Native loader does not address the private frozen Assets overlay')
    return {'global_configs_sha256':sha(path),'actual_private_ASSETS_PATH':str(actual)}


def patch_readback_source(text):
    needle = '            self.reward_manager.init_state()'
    if text.count(needle) != 1:
        raise ValueError('Native reward initialize site changed')
    legacy = ('            from support.probe_readback import install_private_reward_readback\n'
              '            if not hasattr(self, "_ervla_readback_audit"):\n'
              '                self._ervla_readback_audit = install_private_reward_readback(self)\n')
    own = ('            from pi05_policy.probe_readback import install_private_reward_readback\n'
           '            if not hasattr(self, "_pi05_reward_readback_audit"):\n'
           '                self._pi05_reward_readback_audit = install_private_reward_readback(self)\n')
    if legacy in text:
        if text.count(legacy) != 1 or text.count('support.') != 1:
            raise ValueError('Unreviewed legacy project dependencies in native SDK')
        return text.replace(legacy, own)
    if 'support.' in text or 'install_private_reward_readback' in text:
        raise ValueError('Unreviewed reward readback adapter; do not duplicate it')
    return text.replace(needle, own + needle)


def build(folder, ids, adapter, alias, *, seed, manifest):
    from layout_manifest import rows_for
    source_root=Path(manifest['sdk_source'])
    folder = Path(folder)
    source = folder / 'src/RoboDojo'
    source.parent.mkdir(parents=True, exist_ok=True)
    rows=rows_for(manifest,seed,ids)
    pool=rows_for(manifest,seed,list(range(30)))
    for row in rows:
        if sha(row['path'])!=row['sha256']:raise ValueError('Frozen scene/layout changed before namespace construction')
    links(source_root, source, {'Assets','env', 'src', 'XPolicyLab', 'eval_result', '.git'})
    # Only this private SDK sees the frozen overlay. Original shared Assets
    # remain untouched; all30 filenames retain native loader ID semantics.
    relative=Path('Assets')
    for next_name in ('Eval_Layout','RoboDojo','arx_x5',str(seed)):
        links(source_root/relative,source/relative,{next_name})
        relative=relative/next_name
    (source/relative).mkdir()
    for row in pool:
        (source/relative/f"stack_bowls_random_{row['layout_id']}.json").symlink_to(Path(row['path']).resolve(strict=True))
    links(source_root / 'env', source / 'env', {'seed_manager', '__pycache__'})
    links(source_root / 'env/seed_manager', source / 'env/seed_manager', {'seed_manager.py', '__pycache__'})
    (source / 'env/seed_manager/seed_manager.py').write_text(
        patch_seed_source((source_root / 'env/seed_manager/seed_manager.py').read_text(),
            [{**row,'path':str(Path(row['path']).resolve(strict=True))}for row in rows]))
    links(source_root / 'src', source / 'src', {'eval_client', '__pycache__'})
    links(source_root / 'src/eval_client', source / 'src/eval_client', {'eval_env.py', '__pycache__'})
    # Existing SDK's reward parser mixes CUDA tensors with NumPy. This bridge
    # only converts pose/point readbacks to NumPy; no rule/threshold is changed.
    text = (source_root / 'src/eval_client/eval_env.py').read_text()
    text = patch_readback_source(text)
    (source / 'src/eval_client/eval_env.py').write_text(text)
    links(source_root / 'XPolicyLab', source / 'XPolicyLab', {'policy', '__pycache__'})
    (source / 'XPolicyLab/policy').mkdir()
    init = source_root / 'XPolicyLab/policy/__init__.py'
    if init.exists():
        (source / 'XPolicyLab/policy/__init__.py').symlink_to(init)
    (source / 'XPolicyLab/policy' / alias).symlink_to(adapter, target_is_directory=True)
    results = folder / 'results/eval'
    results.mkdir(parents=True)
    (source / 'eval_result').symlink_to(results, target_is_directory=True)
    asset_proof=verify_private_assets(source)
    return {**asset_proof,'scene_seed':seed,'sdk_source':str(source_root),'layout_pool':pool,'private_base': str(folder), 'private_source': str(source),
            'result_root': str(results), 'policy_alias': alias, 'layout_ids': ids, 'layouts': rows,
            'shared_SDK_modified': False, 'reward_rules_modified': False,
            'seed_source_sha256': sha(source / 'env/seed_manager/seed_manager.py'),
            'eval_source_sha256': sha(source / 'src/eval_client/eval_env.py')}
