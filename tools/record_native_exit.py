"""Run one already-prepared native plan under the operator's accepted GPU owner.

Does not acquire or replace the external whole-node admission/environment guard.
Its receipt records the ACTUAL child exit; no acceptance is inferred from exit0.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--execute-under-accepted-gpu-owner', action='store_true', required=True)
    p.add_argument('--timeout', type=int, default=28800)
    a = p.parse_args()
    if hashlib.sha256(a.plan.read_bytes()).hexdigest() != a.plan_sha256:
        raise ValueError('Externally pinned native plan required')
    plan = json.loads(a.plan.read_text())
    if plan.get('schema') != 'ecot.stack_bowls_random.native_case_plan.v1' or plan.get('launched') is not False:
        raise ValueError('Actual Stack Bowls native plan required')
    if not 1 <= a.timeout <= 28800:
        raise ValueError('Finite native process timeout required')
    folder = a.plan.parent
    if (folder / 'NATIVE_EXIT.json').exists():
        raise ValueError('Do not overwrite or silently retry an actual case')
    runtime = Path(__file__).resolve().parents[1] / 'online_subtask/runtime'
    sys.path.insert(0, str(runtime))
    from preflight import validate_native_plan
    preflight = validate_native_plan(plan, a.plan)
    env = dict(os.environ)
    for key in plan['environment_must_be_unset']:
        env.pop(key, None)
    for key in ('GH_TOKEN', 'GITHUB_TOKEN'):
        env.pop(key, None)
    env.update(plan['required_environment_overrides'])
    env['PYTHONPATH'] = os.pathsep.join(plan['runtime_PYTHONPATH_prefix'] +
        ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    started = time.time()
    with (folder / 'native.stdout.log').open('xb') as stdout, (folder / 'native.stderr.log').open('xb') as stderr:
        child = subprocess.Popen(plan['argv'], cwd=plan['cwd'], env=env,
                                 stdout=stdout, stderr=stderr, start_new_session=True)
        timed_out = False
        try:
            code = child.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                code = child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                code = child.wait()
    receipt = {'schema': 'ecot.stack_bowls.actual_native_exit.v1', 'pid': child.pid,
               'started': started, 'ended': time.time(), 'exit_code': code,
               'timed_out': timed_out, 'phase': plan['phase'], 'layout_id': plan['layout_id'],
               'plan_path': str(a.plan.resolve()), 'plan_sha256': a.plan_sha256,
               'SDK_success_claimed': False, 'acceptance_claimed': False}
    receipt['launch_preflight'] = preflight
    with (folder / 'NATIVE_EXIT.json').open('x') as out:
        json.dump(receipt, out, indent=2); out.write('\n')
    print(json.dumps(receipt))
    return 124 if timed_out else code


if __name__ == '__main__':
    raise SystemExit(main())
