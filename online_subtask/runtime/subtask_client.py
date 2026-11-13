"""Strict single-field, current-image-only DeepSeek annotation with raw provenance.

This module never imports CUDA, a policy, scene labels, state or history. Semantic
keyword observations are advisory; valid nonempty Subtask is never rejected by
noun/color regular expressions. HTTP/schema retries use the identical body.
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import time
from uuid import uuid4
import urllib.error
import urllib.request
import zlib

MODEL='deepseek-v4-1-flash-261017'
BASE_URL='https://ark.cn-beijing.volces.com/api/v3'
PROMPT=Path(__file__).with_name('subtask_prompt.md')


def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def strict_json(raw):
    def pairs(items):
        value={}
        for k,v in items:
            if k in value:raise ValueError('Duplicate JSON key')
            value[k]=v
        return value
    def reject(x):raise ValueError('Nonfinite JSON')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=reject)
def validate_context(value):
    if (not isinstance(value,dict)or set(value)!={'subtask'}or not isinstance(value['subtask'],str)
            or not value['subtask'].strip()or value['subtask']!=value['subtask'].strip()
            or len(value['subtask'])>1200):
        raise ValueError('Expected only one nonempty, trimmed subtask string <=1200 characters')
    return value

def png_current_rgb(chw):
    import numpy as np
    a=np.asarray(chw)
    if a.shape!=(3,480,640)or a.dtype!=np.uint8:raise ValueError('Actual uint8 CHW current main image required')
    rgb=np.ascontiguousarray(a.transpose(1,2,0))
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    # PNG Up/filter2 is an exact modulo256 difference against the preceding
    # original RGB row. Only wire compression changes; every pixel is retained.
    filtered=rgb.copy()
    np.subtract(rgb[1:],rgb[:-1],out=filtered[1:])
    raw=b''.join(b'\2'+row.tobytes()for row in filtered)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',640,480,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,9))+chunk(b'IEND',b'')

class SubtaskClient:
    def __init__(self,output_dir,*,timeout=60,attempts=3):
        if timeout!=60 or attempts!=3:raise ValueError('Bounded60s/3 identical-current-frame attempts required')
        self.output_dir=Path(output_dir);self.timeout=timeout;self.attempts=attempts
        self.output_dir.mkdir(parents=True,exist_ok=True)
    def record(self,name,raw):
        if not isinstance(raw,bytes):raw=canonical(raw)+b'\n'
        with(self.output_dir/name).open('xb')as f:f.write(raw)
    def post(self,body):
        key=os.environ.get('ECOT_AUTHORIZED_API_KEY')
        if not key:raise ValueError('Authorized credential absent from process memory')
        request=urllib.request.Request(BASE_URL+'/chat/completions',data=body,method='POST',headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request,timeout=self.timeout)as response:return response.read()
        except urllib.error.HTTPError as e:
            # Error bytes are retained without headers/credential. No body is
            # converted into a context or passed to the policy.
            error=RuntimeError('DeepSeek HTTP '+str(e.code));error.response_body=e.read();raise error from None
    def generate(self,png,instruction):
        if not isinstance(png,bytes)or not png.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('Lossless current image PNG required')
        if not isinstance(instruction,str)or not instruction.strip():raise ValueError('Full original Instruction required')
        prompt=PROMPT.read_bytes();body=canonical({'model':MODEL,'temperature':0,'max_tokens':450,
            'thinking':{'type':'disabled'},'response_format':{'type':'json_object'},
            'messages':[{'role':'system','content':prompt.decode()},{'role':'user','content':[
                {'type':'text','text':canonical({'instruction':instruction}).decode()},
                {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode()}}]}]})
        started=time.monotonic();ids=[]
        for attempt in range(self.attempts):
            rid=uuid4().hex;ids.append(rid);begin=time.monotonic();raw=None
            proof={'request_id':rid,'model':MODEL,'endpoint':BASE_URL,'thinking_type':'disabled',
                   'prompt_sha256':sha(prompt),'request_sha256':sha(body),'image_sha256':sha(png),
                   'instruction_sha256':sha(instruction.encode()),'input_fields':['current_image','instruction'],
                   'history_frames':0,'context_fields':['subtask']}
            self.record(rid+'_request.json',body);self.record(rid+'_metadata.json',proof)
            try:
                raw=self.post(body);self.record(rid+'_response.json',raw);response=strict_json(raw)
                choices=response.get('choices')
                if response.get('model')!=MODEL or not isinstance(choices,list)or len(choices)!=1:raise ValueError('Actual response model/choices differ')
                choice=choices[0];message=choice.get('message',{})
                if(choice.get('finish_reason')!='stop'or message.get('tool_calls')or message.get('refusal')or not isinstance(message.get('content'),str)):
                    raise ValueError('Incomplete or refused Subtask output')
                context=validate_context(strict_json(message['content']))
                text=context['subtask']
                proof.update(response_sha256=sha(raw),context_sha256=sha(canonical(context)),
                             generation_seconds=time.monotonic()-begin,generation_total_seconds=time.monotonic()-started,
                             attempt_count=len(ids),attempt_request_ids=list(ids),usage=response.get('usage'),
                             lexical_diagnostics={'advisory_only':True,'semantic_accuracy_guaranteed':False,
                             'generic_reference':bool(re.search(r'\b(?:target object|the object|an object)\b',text,re.I)),
                             'has_color_word':bool(re.search(r'\b(?:red|green|blue|white|black|yellow|orange|purple|brown|silver|gold)\b',text,re.I))})
                self.record(rid+'_result.json',{'context':context,'lineage':proof});return context,proof
            except Exception as error:
                if raw is None and getattr(error,'response_body',None)is not None:self.record(rid+'_http_error_body.json',error.response_body)
                self.record(rid+'_failure.json',{'request_id':rid,'error_type':type(error).__name__,
                            'validated_context_available':False,'generation_seconds':time.monotonic()-begin})
                if attempt+1==self.attempts:raise RuntimeError('Three same-frame Subtask API/schema attempts failed')from error
        raise AssertionError('Unreachable retry state')
