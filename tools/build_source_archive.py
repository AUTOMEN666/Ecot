"""Build a deterministic ZIP from a clean committed checkout, outside the repo."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def git(*options):
        return subprocess.check_output(['git', '-C', str(root), *options])
    if git('status', '--porcelain'):
        raise ValueError('Commit reviewed files before building the archive')
    output = args.output_dir.resolve()
    if output == root or root in output.parents:
        raise ValueError('Build artifacts must be outside the source checkout')
    output.mkdir(parents=True, exist_ok=True)
    head = git('rev-parse', 'HEAD').decode().strip()
    names = sorted(name.decode() for name in git('ls-tree', '-r', '--name-only', '-z', head).split(b'\0') if name)
    if not names:
        raise ValueError('No tracked source files')
    target = output / ('ecot_online_generation_subtask_' + head[:12] + '.zip')
    files = []
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in names:
            path = root / name
            if path.is_symlink() or not path.is_file():
                raise ValueError('Nonregular source file: ' + name)
            raw = path.read_bytes()
            # Read the exact committed bytes, and reject modifications racing the build.
            committed = git('show', head + ':' + name)
            if raw != committed:
                raise ValueError('Source changed while building: ' + name)
            info = zipfile.ZipInfo('Ecot/' + name, date_time=(2026, 11, 13, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            files.append({'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    if git('rev-parse', 'HEAD').decode().strip() != head or git('status', '--porcelain'):
        target.unlink()
        raise ValueError('Checkout changed while building the archive')
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.zip.sha256').write_text(digest + '  ' + target.name + '\n')
    receipt = {'schema': 'ecot.clean_source_archive.v1', 'git_commit': head,
               'git_branch': git('branch', '--show-current').decode().strip(),
               'archive_name': target.name, 'archive_sha256': digest,
               'archive_bytes': target.stat().st_size, 'files': files,
               'contains_git_metadata': False, 'source_checkout_clean': True}
    target.with_suffix('.manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'files'}))


if __name__ == '__main__':
    main()
