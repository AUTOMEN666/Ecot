"""HTTP wall-clock deadline with an exactly reaped, request-only subprocess.

The policy server invokes synchronous methods on worker threads, so process-wide
signals cannot safely enforce a deadline. The child imports no policy/GPU code,
receives credentials only through stdin, and starts no further processes.
"""
from __future__ import annotations
import base64
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def _bytes(value):
    return base64.b64decode(value,validate=True)


def bounded_http_post(url,body,headers,*,timeout_seconds,selector_kind,receipt=None):
    if type(timeout_seconds)not in(int,float)or timeout_seconds<=0:
        raise ValueError('Positive HTTP wall-clock timeout required')
    receipt={}if receipt is None else receipt
    started=time.monotonic()
    payload=json.dumps({'url':url,'body_b64':base64.b64encode(body).decode(),
        'headers':headers,'timeout_seconds':timeout_seconds},ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()
    # The key is transmitted in the private stdin pipe, never argv or a file.
    environment=dict(os.environ)
    environment.pop('ECOT_AUTHORIZED_API_KEY',None)
    process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=environment)
    receipt.update(kind='reaped_subprocess_wall_clock',pid=process.pid,
        wall_clock_timeout_seconds=timeout_seconds,credential_route='private_stdin_pipe',
        request_child_reaped=False,request_child_killed=False)
    timed_out=False
    try:
        remaining=timeout_seconds-(time.monotonic()-started)
        if remaining<=0:raise subprocess.TimeoutExpired(process.args,timeout_seconds)
        output,_stderr=process.communicate(payload,timeout=remaining)
    except subprocess.TimeoutExpired:
        timed_out=True
        process.kill()
        process.communicate()
        receipt['request_child_killed']=True
    finally:
        # No retry is possible until this exact request process has exited.
        if process.poll()is None:
            process.kill();process.communicate();receipt['request_child_killed']=True
        receipt.update(request_child_reaped=process.returncode is not None,
            returncode=process.returncode,elapsed_seconds=time.monotonic()-started)
    if timed_out:raise TimeoutError('HTTP request exceeded wall-clock deadline; request child killed and reaped')
    if process.returncode!=0:raise RuntimeError('HTTP request worker exited without a transport envelope')
    envelope=json.loads(output)
    if envelope.get('ok')is True:return _bytes(envelope['raw_b64']),envelope['headers']
    kind=envelope.get('kind');message=envelope.get('error','HTTP request failed')
    if kind=='http':
        error=RuntimeError(selector_kind+' HTTP '+str(envelope['status']))
        error.response_body=_bytes(envelope['raw_b64']);raise error
    if kind=='incomplete_read':raise http.client.IncompleteRead(_bytes(envelope['partial_b64']))
    if kind=='remote_disconnected':raise http.client.RemoteDisconnected(message)
    if kind=='timeout':raise TimeoutError(message)
    if kind=='connection':raise ConnectionError(message)
    if kind=='dns':raise socket.gaierror(envelope.get('errno'),message)
    if kind=='os':raise OSError(envelope.get('errno'),message)
    raise RuntimeError('HTTP worker '+envelope.get('error_type','error')+': '+message)


def _worker():
    payload=json.loads(sys.stdin.buffer.read())
    req=urllib.request.Request(payload['url'],data=_bytes(payload['body_b64']),
        headers=payload['headers'],method='POST')
    try:
        with urllib.request.urlopen(req,timeout=payload['timeout_seconds'])as response:
            meta={k:v for k,v in response.headers.items()if k.lower()in(
                'x-completion-id','x-model','x-request-sha256','x-messages-sha256',
                'x-request-id','x-tt-logid','date')}
            envelope={'ok':True,'raw_b64':base64.b64encode(response.read()).decode(),'headers':meta}
    except urllib.error.HTTPError as error:
        envelope={'ok':False,'kind':'http','status':error.code,
            'raw_b64':base64.b64encode(error.read()).decode()}
    except Exception as error:
        reason=error.reason if isinstance(error,urllib.error.URLError)else error
        kind=('incomplete_read'if isinstance(reason,http.client.IncompleteRead)else
            'remote_disconnected'if isinstance(reason,http.client.RemoteDisconnected)else
            'timeout'if isinstance(reason,TimeoutError)else
            'dns'if isinstance(reason,socket.gaierror)else
            'connection'if isinstance(reason,ConnectionError)else
            'os'if isinstance(reason,OSError)else'other')
        envelope={'ok':False,'kind':kind,'error_type':type(reason).__name__,
            'error':str(reason),'errno':getattr(reason,'errno',None)}
        if kind=='incomplete_read':envelope['partial_b64']=base64.b64encode(reason.partial).decode()
    sys.stdout.write(json.dumps(envelope,ensure_ascii=False,separators=(',',':'),allow_nan=False))


if __name__=='__main__':
    if sys.argv[1:]!=['--worker']:raise SystemExit('Request-only HTTP worker')
    _worker()
