"""Visual scene admission + singleton goal lock; policy still receives Subtask only."""
from copy import deepcopy
from pathlib import Path
import time
import http.client
import urllib.error
from uuid import uuid4

from multiview_provider import (MultiviewHistoryProvider, DEEPSEEK_MODEL, write,
                               transient_transport_error)
from multiview_protocol import canonical, sha, strict_json, RETRY_BUDGET_SECONDS
from subtask_guard import SubtaskGuard, SubtaskConstraintError, STEADY_CONFIG_ID
from rpc_lifecycle import exclusive_rpc, lifecycle_lock

HERE = Path(__file__).resolve().parent


class SteadySubtaskProvider(MultiviewHistoryProvider):
    protocol_config_id = STEADY_CONFIG_ID
    def __init__(self, *args, **kwargs):
        self._rpc_lock = lifecycle_lock()
        super().__init__(*args, **kwargs)
        # The existing Codex relay has no observer-schema contract. Fail before use.
        if self.selector_kind != 'deepseek':
            raise ValueError('Steady mode requires the explicit DeepSeek scene-observer contract')
        self.guard = SubtaskGuard(self.catalog)
        self.observer_receipt = None
        self.stopped = None

    @exclusive_rpc
    def reset(self, episode_id):
        super().reset(episode_id)
        self.guard.reset(episode_id)
        self.observer_receipt = None
        self.stopped = None

    @exclusive_rpc
    def capture_history(self, *args, **kwargs):
        return super().capture_history(*args, **kwargs)

    @exclusive_rpc
    def acknowledge_executed_subtask(self, *args, **kwargs):
        return super().acknowledge_executed_subtask(*args, **kwargs)

    def observer_query(self, request, proof, *, started, unchanged):
        """Same transport, model/settings and shared1850s budget, max3 same-body attempts.

        Parse/schema/semantic errors are NEVER retried. A request-only worker is
        killed/reaped by the inherited transport before an admitted retry.
        """
        body = canonical(request).encode()
        ids = []
        for attempt in range(1, 4):
            unchanged()
            if time.monotonic() - started + self.timeout > RETRY_BUDGET_SECONDS:
                raise TimeoutError('Shared steady observer/selector budget exhausted; HTTP not started')
            ident = 'observer_' + uuid4().hex
            ids.append(ident)
            meta = dict(proof, call_role='scene_observer', request_id=ident,
                        messages_sha256=sha(canonical(request['messages']).encode()),
                        request_sha256=sha(body), attempt_index=attempt,
                        maximum_identical_request_attempts=3, shared_retry_budget_seconds=1850)
            write(self.output_dir / (ident + '_request.json'), body)
            write(self.output_dir / (ident + '_metadata.json'), meta)
            raw = None
            empty_length = False
            try:
                attempt_request = strict_json(body)
                raw, headers = self._post(attempt_request, meta)
                write(self.output_dir / (ident + '_response.raw.json'), raw)
                write(self.output_dir / (ident + '_response_headers.json'), headers)
                if canonical(attempt_request).encode() != body:
                    raise ValueError('Observer attempt mutated canonical request bytes')
                unchanged()
                response = strict_json(raw)
                if (response.get('model') != DEEPSEEK_MODEL
                        or type(response.get('choices')) is not list or len(response['choices']) != 1):
                    raise ValueError('Observer actual model/choices mismatch')
                choice = response['choices'][0]
                message = choice.get('message', {})
                text = message.get('content')
                if (choice.get('finish_reason') == 'length' and text in (None, '')
                        and not message.get('tool_calls') and not message.get('refusal')):
                    empty_length = True
                    raise ValueError('Observer length limit with empty public content')
                if (choice.get('finish_reason') != 'stop' or message.get('tool_calls')
                        or message.get('refusal')):
                    raise ValueError('Incomplete/tool/refused observer response')
                evidence = strict_json(text)
                receipt = dict(meta, response_sha256=sha(raw), completion_id=response.get('id'),
                               actual_model=response['model'], attempt_count=attempt,
                               attempt_request_ids=ids, evidence_sha256=sha(canonical(evidence).encode()),
                               HTTP_transport_receipt=meta.get('HTTP_transport_receipt'))
                write(self.output_dir / (ident + '_result.json'),
                      {'evidence': evidence, 'lineage': receipt})
                return evidence, receipt
            except BaseException as error:
                error_body = getattr(error, 'response_body', None)
                if raw is None and error_body is None and isinstance(error, urllib.error.HTTPError):
                    error_body = error.read()
                reason = error.reason if isinstance(error, urllib.error.URLError) else error
                if raw is None and error_body is None and isinstance(reason, http.client.IncompleteRead):
                    error_body = reason.partial
                if raw is None and error_body is not None:
                    write(self.output_dir / (ident + '_http_error.raw'), error_body)
                retry = empty_length or raw is None and transient_transport_error(error)
                scheduled = retry and attempt < 3 and time.monotonic() - started + self.timeout <= RETRY_BUDGET_SECONDS
                write(self.output_dir / (ident + '_failure.json'), {
                    'call_role': 'scene_observer', 'error_type': type(error).__name__, 'error': str(error),
                    'attempt_index': attempt, 'attempt_request_ids': ids, 'request_sha256': sha(body),
                    'available_error_body_sha256': sha(error_body) if error_body is not None else None,
                    'HTTP_transport_receipt': meta.get('HTTP_transport_receipt'),
                    'retry_scheduled': scheduled, 'actual_injected': False, 'policy_called': False})
                if not scheduled:
                    raise
        raise AssertionError('Unreachable observer retry state')

    def prepare_selection(self, request, proof, *, started, unchanged):
        observer = deepcopy(request)
        observer['messages'][0]['content'] = (HERE / 'scene_observer_prompt.md').read_text()
        descriptors = {c['descriptor_id']: c['object_phrase'] for c in self.catalog['candidates']}
        observer['messages'][1]['content'][0]['text'] = canonical({'descriptors': descriptors})
        scene = self.guard.snapshot()
        content = observer['messages'][-1]['content']
        metadata = strict_json(content[0]['text'])
        metadata['scene_registry'] = scene
        metadata['registry_is_fallible_visual_memory_not_GT'] = True
        content[0]['text'] = canonical(metadata)
        state_sha = sha(canonical(scene).encode())
        def unchanged_observer():
            unchanged()
            if sha(canonical(self.guard.snapshot()).encode()) != state_sha:
                raise SubtaskConstraintError('Scene registry changed during observer request')
        evidence, receipt = self.observer_query(observer, proof, started=started, unchanged=unchanged_observer)
        unchanged_observer()
        decision = self.guard.observe(evidence, episode_id=proof['episode_id'], frame=proof['current_frame'],
                                      available_frames=[o['frame'] for o in proof['observations']],
                                      executed_records=proof['executed_subtask_history'])
        self.observer_receipt = receipt
        request = deepcopy(request)
        request['messages'][0]['content'] = (HERE / 'steady_selector_prompt.md').read_text()
        allowed = set(decision['allowed_ids'])
        request['messages'][1]['content'][0]['text'] = canonical({
            'allowed_candidates': [{'id': c['id'], 'text': c['text']} for c in self.catalog['candidates'] if c['id'] in allowed],
            'original_catalogue_count': 288, 'mode': decision['mode'], 'locked_id': decision['locked_id'],
            'category_to_basket': decision['category_to_basket'],
            'basket_to_category': decision['basket_to_category'],
            'selection_rule': 'Exactly one allowed ID. No semantic retry, replacement or repair.'})
        proof.update(schema='ecot.mainhist93_tok65536.steady.request.v1',
                     protocol_config_id=STEADY_CONFIG_ID, steady_protocol=STEADY_CONFIG_ID, call_role='subtask_selector',
                     observer_request_sha256=receipt['request_sha256'], observer_response_sha256=receipt['response_sha256'],
                     observer_prompt_sha256=sha((HERE / 'scene_observer_prompt.md').read_bytes()),
                     prompt_sha256=sha((HERE / 'steady_selector_prompt.md').read_bytes()),
                     messages_sha256=sha(canonical(request['messages']).encode()),
                     steady_decision_sha256=sha(canonical(decision).encode()),
                     shared_observer_selector_budget_seconds=1850)
        return request, proof

    def validate_selection(self, context, resolution, proof):
        admission = self.guard.select(resolution['selected_id'])
        proof['steady_guard'] = admission
        self.guard.validate_injection(context, dict(proof, selected_id=resolution['selected_id'], resolution=resolution),
                                      episode_id=self.episode_id, frame=proof['current_frame'])

    @exclusive_rpc
    def validate_injection(self, context, lineage, *, episode_id, frame):
        self.guard.validate_injection(context, lineage, episode_id=episode_id, frame=frame)

    @exclusive_rpc
    def generate(self, *args, **kwargs):
        # Reject a delayed caller from a prior episode BEFORE the decision-stop
        # handler can latch the current episode. No API request is admitted.
        if not self.episode_id or kwargs.get('episode_id') != self.episode_id:
            raise ValueError('Provider requires actual same-episode reset')
        if self.stopped is not None:
            raise SubtaskConstraintError('Steady decision already stopped; an explicit new episode reset is required')
        try:
            return super().generate(*args, **kwargs)
        except Exception as error:
            self.stopped = {'frame': kwargs.get('frame'), 'error_type': type(error).__name__}
            # Separate decision stops from SDK outcomes. No history is manufactured.
            with (self.output_dir / 'steady_decision_stops.jsonl').open('a') as stream:
                stream.write(canonical({'event': 'steady_decision_stop', 'episode_id': kwargs.get('episode_id'),
                    'frame': kwargs.get('frame'), 'error_type': type(error).__name__, 'error': str(error),
                    'actual_injected': False, 'policy_called': False, 'SDK_outcome': None,
                    'scene_state': self.guard.snapshot(), 'decision': deepcopy(self.guard.decision)}) + '\n')
            raise
