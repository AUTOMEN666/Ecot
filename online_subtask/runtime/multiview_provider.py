"""Every8-frame single-ID provider:93 pastMAIN pluscurrent3,96-image cap and65536 tokens."""
from __future__ import annotations
import json,os,time,urllib.error,urllib.request
from pathlib import Path
from uuid import uuid4
from executed_history import ExecutedSubtaskHistory,digest
from multiview_protocol import (CAMERAS,CATALOG_SHA,PROTOCOLS,build_request,canonical,load_catalog,resolve,sha,strict_json,make_history_point,history_source_frame,HTTP_TIMEOUT_SECONDS,RETRY_BUDGET_SECONDS)

DEEPSEEK_MODEL='deepseek-v4-1-flash-261017'
DEEPSEEK_URL='https://ark.cn-beijing.volces.com/api/v3/chat/completions'

def write(path,value):
    data=value if isinstance(value,bytes) else (canonical(value)+'\n').encode()
    with Path(path).open('xb') as f:f.write(data)

def transient_transport_error(error):
    import errno,http.client,socket
    code=getattr(error,'code',None)
    if (type(error)is RuntimeError and getattr(error,'response_body',None)is not None
            and str(error).startswith('deepseek HTTP ')):
        suffix=str(error)[len('deepseek HTTP '):]
        if suffix.isdecimal():code=int(suffix)
    if type(code)is int:return code==429 or 500<=code<=599
    reason=error.reason if isinstance(error,urllib.error.URLError)else error
    return(isinstance(reason,(TimeoutError,ConnectionError,http.client.IncompleteRead,http.client.RemoteDisconnected))
        or isinstance(reason,socket.gaierror)and reason.errno==socket.EAI_AGAIN
        or isinstance(reason,OSError)and reason.errno in(errno.ETIMEDOUT,errno.ECONNRESET,errno.ECONNABORTED,errno.ECONNREFUSED,errno.EPIPE))

class MultiviewHistoryProvider:
    def __init__(self,output_dir,*,selector_kind,relay_url='http://127.0.0.1:28782/select',timeout_seconds=HTTP_TIMEOUT_SECONDS,transport=None):
        if selector_kind not in ('deepseek','codex'):raise ValueError('Explicit DeepSeek/Codex provider required')
        if type(timeout_seconds)not in (int,float)or timeout_seconds!=HTTP_TIMEOUT_SECONDS:raise ValueError('This protocol requires exact600-second HTTP timeout')
        if selector_kind=='codex' and not relay_url.startswith('http://127.0.0.1:'):raise ValueError('Codex relay must be host loopback')
        self.output_dir=Path(output_dir);self.output_dir.mkdir(parents=True,exist_ok=True)
        self.selector_kind=selector_kind;self.relay_url=relay_url;self.timeout=timeout_seconds;self.transport=transport
        self.catalog=load_catalog();self.executed_history=ExecutedSubtaskHistory(self.catalog);self.episode_id=None;self.frames=[];self.last_frame=-8
        write(self.output_dir/'fixed_catalog_snapshot.json',self.catalog)

    def reset(self,episode_id):
        if type(episode_id)is not str or not episode_id:raise ValueError('Actual explicit episode reset required')
        self.episode_id=episode_id;self.frames=[];self.last_frame=-8;self.executed_history.reset(episode_id)

    def acknowledge_executed_subtask(self,ack,*,expected_injection,expected_actions,expected_clock):
        receipt=self.executed_history.commit(ack,expected_injection=expected_injection,
            expected_actions=expected_actions,expected_clock=expected_clock)
        row={**receipt,'ack_source':ack}
        with (self.output_dir/'executed_subtask_history.jsonl').open('a')as stream:
            stream.write(canonical(row)+'\n')
        return receipt

    def capture_history(self,pngs,instruction,*,state14,episode_id,frame):
        if episode_id!=self.episode_id:raise ValueError('History RPC needs same actual episode reset')
        if frame!=history_source_frame(len(self.frames)):raise ValueError('History must capture every actual floor(2.5*k) boundary once')
        if instruction!='Sort the objects by category into the three baskets.':raise ValueError('Original Instruction required')
        point=make_history_point(episode_id,frame,state14,pngs)
        folder=self.output_dir/'history_captures'/episode_id;folder.mkdir(parents=True,exist_ok=True)
        views={}
        for camera,view in zip(CAMERAS,point.views):
            write(folder/f'frame_{frame}_{camera}.png',pngs[camera])
            views[camera]={'frame':frame,'image_sha256':view.image_sha256,'original_camera_chw_sha256':view.camera_chw_sha256,
                'state14':list(view.state14),'state_sha256':view.state_sha256}
        record={'episode_id':episode_id,'frame':frame,'history_source_index':len(self.frames),'frame_rule':'floor(2.5*k)',
            'instruction':instruction,'measured_state14':list(point.state14),'state_source':'official_RoboDojo_observation_joint_and_gripper',
            'current_views':views,'policy_called':False,'RNG_advanced':False,'simulation_steps_by_this_provider':0}
        write(folder/f'frame_{frame}.json',record)
        self.frames.append(point)
        return record

    def _post(self,request,proof):
        if self.transport is not None:return self.transport(request,proof)
        if self.selector_kind=='deepseek':
            request=dict(request,model=DEEPSEEK_MODEL)
            key=os.environ.get('ECOT_AUTHORIZED_API_KEY')
            if not key:raise RuntimeError('Authorized API key absent from process memory')
            body=canonical(request).encode();url=DEEPSEEK_URL
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
        else:
            body=canonical({'request':request,'proof':proof}).encode();url=self.relay_url
            headers={'Content-Type':'application/json'}
        from http_transport import bounded_http_post
        receipt={};proof['HTTP_transport_receipt']=receipt
        return bounded_http_post(url,body,headers,timeout_seconds=self.timeout,
            selector_kind=self.selector_kind,receipt=receipt)

    def prepare_selection(self,request,proof,*,started,unchanged):
        """Legacy pass-through; steady mode adds its visual admission here."""
        return request,proof

    def validate_selection(self,context,resolution,proof):
        """Legacy ID lookup already validated; steady mode also checks admission."""
        return None

    def generate(self,pngs,instruction,*,state14,episode_id,frame):
        import errno,http.client,socket
        if self.episode_id!=episode_id:raise ValueError('Provider requires actual same-episode reset')
        if type(frame)is not int or frame!=self.last_frame+8:raise ValueError('One update exactly every8 executed frames, starting0; no skipped/reused frame')
        started=time.monotonic()
        original_last_frame=self.last_frame;original_frames=tuple(self.frames)
        executed_proof=self.executed_history.for_request(episode_id,frame)
        original_executed_sha=digest(executed_proof)
        request,proof,current=build_request(episode_id=episode_id,frame=frame,state14=state14,pngs=pngs,instruction=instruction,past_frames=self.frames,executed_subtask_history=executed_proof['executed_subtask_history'],catalog=self.catalog)
        proof.update(executed_proof)
        if self.selector_kind=='deepseek':request['model']=DEEPSEEK_MODEL
        def unchanged():
            if (self.episode_id!=episode_id or self.last_frame!=original_last_frame or tuple(self.frames)!=original_frames
                    or digest(self.executed_history.for_request(episode_id,frame))!=original_executed_sha):
                raise ValueError('Retry altered actual episode/frame/history')
        request,proof=self.prepare_selection(request,proof,started=started,unchanged=unchanged)
        unchanged()
        request_body=canonical(request).encode()
        capture_dir=self.output_dir/'captured_observations'/episode_id;capture_dir.mkdir(parents=True,exist_ok=True)
        for camera in CAMERAS:write(capture_dir/f'frame_{frame}_{camera}.png',pngs[camera])
        write(capture_dir/f'frame_{frame}.json',{'episode_id':episode_id,'frame':frame,'instruction':instruction,'measured_state14':list(current.state14),
            'state_sha256':current.state_sha256,'current_views':{camera:next(x for x in proof['observations'] if x['frame']==frame and x['camera']==camera)for camera in CAMERAS}})
        protocol_id=getattr(self,'protocol_config_id',PROTOCOLS['expanded_catalog_id'])
        kind_suffix=protocol_id if hasattr(self,'protocol_config_id')else 'mainhist93_tok65536'
        proof.update(provider_kind=self.selector_kind+'_'+kind_suffix,generator_protocol=protocol_id,
            request_sha256=sha(request_body),live_provider_source_sha256=sha(Path(__file__).read_bytes()),
            executed_history_source_sha256=sha((Path(__file__).parent/'executed_history.py').read_bytes()),
            current_execution_frame=frame,current_episode_id=episode_id,image_sha256=current.image_sha256,
            current_capture_json=str(capture_dir/f'frame_{frame}.json'),context_fields=['subtask'])
        base_proof=dict(proof);attempt_ids=[];last_retry_error=None
        rpc_retry_budget_seconds=RETRY_BUDGET_SECONDS
        maximum_attempts=3 if self.selector_kind=='deepseek'else 1
        for attempt in range(1,maximum_attempts+1):
            unchanged()
            elapsed=time.monotonic()-started
            if elapsed+self.timeout>rpc_retry_budget_seconds:
                write(self.output_dir/((attempt_ids[-1]if attempt_ids else uuid4().hex)+'_retry_budget_exhausted.json'),{
                    'event':'retry_not_started_RPC_safety_budget_exhausted','frame':frame,'episode_id':episode_id,
                    'attempt_request_ids':list(attempt_ids),'next_attempt_not_started':attempt,
                    'elapsed_seconds':elapsed,'unchanged_transport_timeout_seconds':self.timeout,
                    'RPC_retry_budget_seconds':rpc_retry_budget_seconds,'request_sha256':sha(request_body),
                    'actual_injected':False,'policy_called':False,'fallback_used':False})
                if last_retry_error is None:raise TimeoutError('Request preparation exhausted bounded retry admission budget; HTTP not started')
                raise last_retry_error
            request_id=uuid4().hex;attempt_ids.append(request_id)
            attempt_request=strict_json(request_body)
            proof=dict(base_proof,request_id=request_id,attempt_index=attempt,
                maximum_identical_request_attempts=maximum_attempts,RPC_retry_budget_seconds=rpc_retry_budget_seconds,
                attempt_admission_elapsed_seconds=elapsed,unchanged_transport_timeout_seconds=self.timeout)
            write(self.output_dir/(request_id+'_request.json'),request_body)
            write(self.output_dir/(request_id+'_metadata.json'),proof)
            raw=None;headers={};retry_empty_length=False;response=None
            try:
                raw,headers=self._post(attempt_request,proof)
                write(self.output_dir/(request_id+'_response.raw.json'),raw)
                write(self.output_dir/(request_id+'_response_headers.json'),headers)
                if canonical(attempt_request).encode()!=request_body:raise ValueError('Attempt mutated canonical request bytes')
                unchanged()
                response=strict_json(raw)
                if self.selector_kind=='deepseek':
                    choices=response.get('choices')
                    if response.get('model')!=DEEPSEEK_MODEL or type(choices)is not list or len(choices)!=1:raise ValueError('DeepSeek actual model/choices mismatch')
                    choice=choices[0];message=choice.get('message',{})
                    text=message.get('content')
                    if (choice.get('finish_reason')=='length'and text in(None,'')
                            and not message.get('tool_calls')and not message.get('refusal')):
                        retry_empty_length=True
                        raise ValueError('DeepSeek length limit with empty public content; no ID')
                    if choice.get('finish_reason')!='stop' or message.get('tool_calls') or message.get('refusal'):raise ValueError('Incomplete/tool/refused generation')
                    completion_id=response.get('id');actual_model=response.get('model');usage=response.get('usage')
                else:
                    text=raw.decode();completion_id=headers.get('X-Completion-ID',headers.get('x-completion-id'))
                    actual_model=headers.get('X-Model',headers.get('x-model'));usage=None
                    messages_sha=headers.get('X-Messages-SHA256',headers.get('x-messages-sha256'))
                    if messages_sha!=proof['messages_sha256']:raise ValueError('Codex relay actual messages SHA differs')
                    if not isinstance(completion_id,str)or not completion_id or not isinstance(actual_model,str)or not actual_model:
                        raise ValueError('Codex relay must return actual completion/model identity')
                context,resolution=resolve(text,self.catalog)
                self.validate_selection(context,resolution,proof)
                proof.update(response_sha256=sha(raw),response_headers=headers,completion_id=completion_id,actual_model=actual_model,
                    raw_text=text,usage=usage,resolution=resolution,selected_id=resolution['selected_id'],context_sha256=sha(canonical(context).encode()),
                    generation_total_seconds=time.monotonic()-started,generation_seconds=time.monotonic()-started,attempt_count=attempt,
                    attempt_request_ids=list(attempt_ids),all_attempt_request_sha256=sha(request_body),
                    identical_request_retries=attempt-1,fallback_used=False,SSR_generated_then_discarded=False)
                write(self.output_dir/(request_id+'_id_resolution.json'),resolution)
                write(self.output_dir/(request_id+'_result.json'),{'context':context,'lineage':proof})
                unchanged()
                self.last_frame=frame
                return context,proof
            except BaseException as error:
                error_body=getattr(error,'response_body',None)
                if raw is None and error_body is None and isinstance(error,urllib.error.HTTPError):error_body=error.read()
                transport_error=error.reason if isinstance(error,urllib.error.URLError)else error
                if raw is None and error_body is None and isinstance(transport_error,http.client.IncompleteRead):error_body=transport_error.partial
                if raw is None and error_body is not None:write(self.output_dir/(request_id+'_http_error.raw'),error_body)
                stable=(canonical(attempt_request).encode()==request_body and self.episode_id==episode_id
                    and self.last_frame==original_last_frame and tuple(self.frames)==original_frames
                    and digest(self.executed_history.for_request(episode_id,frame))==original_executed_sha)
                retry_allowed=(self.selector_kind=='deepseek'and stable and(retry_empty_length or raw is None and transient_transport_error(error)))
                elapsed=time.monotonic()-started
                within_rpc_budget=elapsed+self.timeout<=rpc_retry_budget_seconds
                write(self.output_dir/(request_id+'_failure.json'),{'request_id':request_id,'frame':frame,'episode_id':episode_id,
                    'error_type':type(error).__name__,'error':str(error),'actual_injected':False,'policy_called':False,'fallback_used':False,
                    'generation_seconds':time.monotonic()-started,'attempt_index':attempt,'attempt_request_ids':list(attempt_ids),
                    'request_sha256':sha(request_body),'response_sha256':sha(raw)if raw is not None else None,
                    'available_error_body_sha256':sha(error_body)if error_body is not None else None,
                    'response_headers':headers,'HTTP_transport_receipt':proof.get('HTTP_transport_receipt'),'retry_eligible':retry_allowed,
                    'retry_scheduled':retry_allowed and attempt<maximum_attempts and within_rpc_budget,
                    'retry_reason':'empty_public_length'if retry_empty_length else'known_transient_transport'if retry_allowed else None,
                    'RPC_retry_budget_seconds':rpc_retry_budget_seconds,'retry_budget_exhausted':retry_allowed and attempt<maximum_attempts and not within_rpc_budget,
                    'unchanged_transport_timeout_seconds':self.timeout,'elapsed_seconds':elapsed,
                    'frame_state_history_unchanged':stable})
                if not retry_allowed or attempt>=maximum_attempts or not within_rpc_budget:raise
                last_retry_error=error
        raise AssertionError('Unreachable bounded retry state')
