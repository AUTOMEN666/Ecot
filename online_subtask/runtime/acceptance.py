"""Audit actual case artifacts; an SDK failure is distinct from an interruption.

No simulator/model is imported. Engineering acceptance means the integration
worked, not that the robot succeeded. Formal preparation rechecks source hashes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from export_binding import read, sha, canonical_sha
from multiview_protocol import CATALOG_SHA, load_catalog
from pi05_policy.physics_clock import verify_clock_delta
from pi05_policy.freeze import verify_static
from task_contract import TASK, CHECKPOINT_STEP, PROTOCOL_CONFIG_ID, NATIVE_ACTION_LIMIT, LAYOUT_IDS, INSTRUCTION, compose_prompt

SCHEMA = 'ecot.stack_bowls_random.actual_case_acceptance.v1'


def lines(path):
    raw = Path(path).read_text()
    if raw and not raw.endswith('\n'):
        raise ValueError('Partial audit log line')
    return [json.loads(line) for line in raw.splitlines() if line]


def audit_case(output, policy_audit, layout_manifest, manifest_sha256, native_exit,
               *, phase, layout_id, result_path, decode_receipt):
    from layout_manifest import validate_manifest
    manifest = validate_manifest(layout_manifest, expected_sha256=manifest_sha256)
    if phase not in ('engineering', 'formal') or type(layout_id) is not int or layout_id not in LAYOUT_IDS:
        raise ValueError('Exact phase and existing Random30 layout required')
    output = Path(output).resolve(strict=True)
    exit_data = read(native_exit)
    if (exit_data.get('schema') != 'ecot.stack_bowls.actual_native_exit.v1'
            or exit_data.get('exit_code') != 0 or exit_data.get('timed_out') is not False
            or exit_data.get('phase') != phase or exit_data.get('layout_id') != layout_id):
        raise ValueError('Actual native child must exit0; interruption is not SDK failure')
    plan_path = exit_data['plan_path']
    if sha(plan_path) != exit_data['plan_sha256']:
        raise ValueError('Actual child plan changed')
    actual_plan = read(plan_path)
    if (actual_plan.get('schema') != 'ecot.stack_bowls_random.native_case_plan.v1'
            or actual_plan.get('task') != TASK or actual_plan.get('phase') != phase
            or actual_plan.get('layout_id') != layout_id
            or actual_plan.get('manifest_sha256') != manifest_sha256):
        raise ValueError('Actual native plan differs from audited case')
    terminals = lines(output / 'terminal.jsonl')
    if len(terminals) != 1:
        raise ValueError('One real native terminal required')
    terminal = terminals[0]
    commands = terminal['executed_commands']
    if (type(commands) is not int or not 1 <= commands <= NATIVE_ACTION_LIMIT
            or terminal.get('actual_environment_terminal') is not True
            or terminal.get('engineering_stop') is not False
            or terminal.get('engineering_command_limit') != 0
            or terminal.get('native_step_limit') != NATIVE_ACTION_LIMIT
            or terminal.get('scene_seed') != 1 or terminal.get('layout_id') != layout_id
            or terminal.get('engineering_probe') is not (phase == 'engineering')):
        raise ValueError('Complete original native800 episode/phase/layout required')
    episode = terminal['episode_id']
    clocks = lines(output / 'execution_clock.jsonl')
    if [row.get('execution_frame') for row in clocks] != list(range(1, commands + 1)):
        raise ValueError('Missing actual action clock rows')
    previous = None
    for row in clocks:
        if row.get('episode_id') != episode or row.get('native_take_action_count') != row['execution_frame']:
            raise ValueError('Clock episode/native count differs')
        if previous is not None:
            verify_clock_delta(previous, row['before'], 0, 'audit_between_actions')
        verify_clock_delta(row['before'], row['after'], 10, 'audit_actual25Hz')
        previous = row['after']
    injected = lines(output / 'injected_text.jsonl')
    plans = [row for row in injected if row.get('event') == 'actual_injection']
    texts = [row for row in injected if row.get('event') == 'actual_executed_action_text']
    if [row.get('execution_frame') for row in plans] != list(range(0, commands, 8)):
        raise ValueError('Every8 actual injection required')
    if [row.get('execution_action') for row in texts] != list(range(commands)):
        raise ValueError('Every executed action needs actual injected text')
    catalogue = {row['id']: row['text'] for row in load_catalog()['candidates']}
    by_frame = {row['execution_frame']: row for row in plans}
    for plan in plans:
        lineage = plan['lineage']; selected = lineage['selected_id']
        if (plan.get('episode_id') != episode or plan.get('actual_injected') is not True
                or plan.get('checkpoint_step') != CHECKPOINT_STEP
                or plan.get('training_composition_byte_equal') is not True
                or plan.get('context') != {'subtask': catalogue.get(selected)}
                or selected not in catalogue
                or plan.get('prompt') != compose_prompt(INSTRUCTION, catalogue[selected])
                or plan.get('prompt_sha256') != hashlib.sha256(plan['prompt'].encode()).hexdigest()
                or lineage.get('catalog_sha256') != CATALOG_SHA
                or lineage.get('protocol_config_id') != PROTOCOL_CONFIG_ID
                or lineage.get('current_frame') != plan['execution_frame']
                or any(frame >= plan['execution_frame'] for frame in lineage['history_frames'])):
            raise ValueError('Genuine current-only Subtask/EMA10000 injection required')
    for row in texts:
        p = by_frame[8 * (row['execution_action'] // 8)]
        if (row.get('episode_id') != episode or row.get('injection_execution_frame') != p['execution_frame']
            or row.get('context') != p['context'] or row.get('prompt') != p['prompt']
            or row.get('observed_outcome_frame') != row['execution_action'] + 1
            or row.get('prompt_sha256') != p['prompt_sha256']):
            raise ValueError('Displayed text must match actual executed injection')
    for name in ('freeze.jsonl', 'history_capture_freeze.jsonl', 'executed_subtask_ack.jsonl'):
        records = lines(output / name)
        if not records:
            raise ValueError('Missing real freeze/capture/ACK evidence')
        for row in records:
            verify_static(row['before'], row['after'])
            verify_clock_delta(row['before']['physics_clock'], row['after']['physics_clock'], 0, name)
    acks = lines(output / 'executed_subtask_ack.jsonl')
    if len(acks) != len(plans) or any(row.get('ACK_RPC_returned') is not True for row in acks):
        raise ValueError('One successful actual-execution ACK per plan required')
    model = [row for row in lines(policy_audit) if row.get('episode_id') == episode and row.get('event') == 'inference']
    if [row['executed_action_step'] for row in model] != list(range(0, commands, 8)):
        raise ValueError('One matching actual policy call per native plan required')
    for row in model:
        p = by_frame[row['executed_action_step']]
        if (row.get('context') != p['context'] or row.get('prompt') != p['prompt']
                or row['binding'].get('step') != CHECKPOINT_STEP
                or row['binding'].get('effective_eval_ecot') is not True
                or row['binding'].get('source_training_eval_ecot') is not False):
            raise ValueError('Actual model consumption differs from native injection')
    # Reconstruct history from actual model commands and native action clocks,
    # rather than trusting a saved "ACK returned" flag.
    import numpy as np
    from executed_history import ExecutedSubtaskHistory, canonical
    from pi05_policy.native_model import to_commands
    from pi05_policy.freeze import plain
    history = ExecutedSubtaskHistory(load_catalog()); history.reset(episode)
    for plan, row, ack in zip(plans, model, acks):
        f = plan['execution_frame']
        expected_injection = {k: v for k, v in plan.items() if k != 'event'}
        commands8, executed8 = to_commands(np.asarray(row['raw_predicted_absolute_joint14'], dtype=np.float32))
        if canonical(executed8.tolist()) != canonical(row['executed_first8_absolute_joint14']):
            raise ValueError('Model first8 commands differ from the recorded full50 prediction')
        actual_clocks = clocks[f:min(f + 8, commands)]
        if canonical(ack['ack_source']['execution_clocks']) != canonical(actual_clocks):
            raise ValueError('ACK does not describe actual native actions')
        receipt = history.commit(ack['ack_source'], expected_injection=expected_injection,
                                 expected_actions=plain(commands8), expected_clock=row['physics_clock'])
        if canonical(receipt) != canonical(ack['receipt']):
            raise ValueError('Executed history receipt differs from reconstructed ACK')
    recording = terminal['main_camera_action25hz']
    decoded = read(decode_receipt)
    if (recording.get('status') != 'complete' or recording.get('frame_count') != commands + 1
            or decoded.get('fully_decoded') is not True or decoded.get('encoded_video_fully_decoded') is not True
            or decoded.get('frame_count') != commands + 1
            or decoded.get('fps') != 25 or decoded.get('actual_text_timeline_verified') is not True
            or decoded.get('source_video_sha256') != sha(recording['video_path'])
            or decoded.get('injection_log_sha256') != sha(output / 'injected_text.jsonl')
            or recording.get('frames_sha256') != sha(recording['frames_path'])
            or sha(decoded['output_video']) != decoded.get('output_video_sha256')):
        raise ValueError('Fully decoded25Hz video with actual synchronous Subtask text required')
    result = read(result_path)
    details = result.get('details')
    if result.get('eval_time') != 1 or not isinstance(details, dict) or len(details) != 1:
        raise ValueError('Original single-case SDK result required')
    outcome = next(iter(details.values()))
    if (outcome.get('layout_id') != layout_id or type(outcome.get('success')) is not bool
            or type(outcome.get('score')) not in (int, float) or not math.isfinite(outcome['score'])):
        raise ValueError('Original SDK outcome/scene binding differs')
    if Path(result_path).resolve() != (Path(recording['original_SDK_save_dir']) / '_result.json').resolve():
        raise ValueError('Must use this recorded episode original SDK result')
    evidence = [Path(output / name) for name in ('terminal.jsonl', 'execution_clock.jsonl', 'injected_text.jsonl',
                'freeze.jsonl', 'history_capture_freeze.jsonl', 'executed_subtask_ack.jsonl')]
    evidence += [Path(p) for p in (policy_audit, native_exit, result_path, decode_receipt,
                 layout_manifest, recording['video_path'], recording['frames_path'], decoded['output_video'], plan_path)]
    return {'schema': SCHEMA, 'passed': True, 'phase': phase, 'task': TASK, 'scene_seed': 1,
            'layout_id': layout_id, 'step': CHECKPOINT_STEP, 'episode_id': episode,
            'protocol_config_id': PROTOCOL_CONFIG_ID, 'manifest_sha256': manifest_sha256,
            'executed_commands': commands, 'SDK_success': outcome['success'], 'SDK_score': outcome['score'],
            'scope': 'Actual integration/clock/text/video acceptance; SDK success remains the original outcome.',
            'source_sha256': {str(p.resolve(strict=True)): sha(p) for p in evidence}}


def verify_engineering(path, manifest_sha256):
    d = read(path)
    if (d.get('schema') != SCHEMA or d.get('passed') is not True or d.get('phase') != 'engineering'
            or d.get('task') != TASK or d.get('step') != CHECKPOINT_STEP
            or d.get('manifest_sha256') != manifest_sha256 or not d.get('source_sha256')):
        raise ValueError('Separate accepted engineering case required before formal preparation')
    for p, digest in d['source_sha256'].items():
        if sha(p) != digest:
            raise ValueError('Engineering acceptance source changed')
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'policy-audit', 'layout-manifest', 'native-exit', 'result-path', 'decode-receipt', 'receipt'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--phase', choices=['engineering', 'formal'], required=True)
    p.add_argument('--layout-id', type=int, required=True)
    a = p.parse_args()
    d = audit_case(a.output, a.policy_audit, a.layout_manifest, a.manifest_sha256, a.native_exit,
                   phase=a.phase, layout_id=a.layout_id, result_path=a.result_path, decode_receipt=a.decode_receipt)
    with a.receipt.open('x') as out:
        json.dump(d, out, indent=2); out.write('\n')
    print(json.dumps({'receipt': str(a.receipt), 'phase': a.phase, 'SDK_success': d['SDK_success']}))


if __name__ == '__main__':
    main()
