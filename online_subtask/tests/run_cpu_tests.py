"""Run the preserved CPU behavior suite; optional loopback-only HTTP fixtures.

This runner never loads policy weights, calls a real API or launches a simulator.
The complete suite needs NumPy and permission to bind temporary127.0.0.1 ports.
Run with python -I -B when NumPy is installed in that interpreter's site-packages.
python -E -B also ignores inherited PYTHONPATH while allowing installed user-site
NumPy; use a separate copy/cwd outside the original workspace for either command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exclude-http', action='store_true',
                        help='Partial diagnostic: omit the loopback HTTP transport fixtures.')
    parser.add_argument('--report', type=Path, help='Optional JSON receipt path; no env/credentials captured.')
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parents[1] / 'runtime'
    def source_pins():
        return {str(path.relative_to(runtime)):hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(runtime.rglob('*')) if path.is_file()
                and '__pycache__' not in path.parts and path.suffix in ('.py','.md','.json','.txt')}
    before_source=source_pins()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(runtime))
    modules = [path.stem for path in sorted(runtime.glob('test_*.py'))
               if not (args.exclude_http and path.stem == 'test_http_transport')]
    if not {'test_subtask_guard','test_steady_provider','test_steady_runtime','test_steady_concurrency',
            'test_steady_invariants','test_steady_delivery'} <= set(modules):
        raise ValueError('Steady branch must include state, provider, runtime and lifecycle checks')
    suite = unittest.TestLoader().loadTestsFromNames(modules)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after_source=source_pins()
    source_stable=after_source==before_source
    receipt = dict(schema='ecot.online_subtask.steady.cpu_test_run.v1',
                   interpreter=sys.executable, version=sys.version, cwd=str(Path.cwd()),
                   runner=str(Path(__file__).resolve()), runtime=str(runtime),
                   isolated_python=bool(sys.flags.isolated),
                   ignore_environment=bool(sys.flags.ignore_environment), sys_path=list(sys.path),
                   modules=modules, tests_run=result.testsRun,
                   failures=len(result.failures), errors=len(result.errors),
                   skipped=len(result.skipped), passed=result.wasSuccessful() and source_stable,
                   runtime_source_unchanged_during_tests=source_stable,
                   runtime_source_sha256=before_source,
                   loopback_HTTP_tests_included=not args.exclude_http,
                   actual_API_GPU_simulator_called=False)
    if args.report:
        args.report.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    return 0 if result.wasSuccessful() and source_stable else 1


if __name__ == '__main__':
    raise SystemExit(main())
