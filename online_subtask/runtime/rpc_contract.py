"""Narrow private SDK client timeout adapter; no shared SDK edits."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path

TIMEOUT_SECONDS=2100.0


def _call(tree):
    calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Name)and n.func.id=='WsModelClient']
    if len(calls)!=1:raise ValueError('Exactly one original WsModelClient constructor required')
    return calls[0]


def _other_signature(call):
    value={'func':ast.dump(call.func),'args':[ast.dump(n)for n in call.args],
           'keywords':[(n.arg,ast.dump(n.value))for n in call.keywords
                       if n.arg not in ('ws_ping_timeout_s','request_timeout_s')]}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def validate_rpc_text(text):
    call=_call(ast.parse(text))
    for name in ['ws_ping_timeout_s','request_timeout_s']:
        found=[n for n in call.keywords if n.arg==name]
        if (len(found)!=1 or not isinstance(found[0].value,ast.Constant)
                or found[0].value.value!=TIMEOUT_SECONDS):
            raise ValueError('Private client requires explicit2100-second '+name)
    return {'request_timeout_seconds':TIMEOUT_SECONDS,'ws_ping_timeout_seconds':TIMEOUT_SECONDS,
            'constructor_count':1,'other_constructor_signature_sha256':_other_signature(call)}


def patch_rpc_text(text):
    """Replace only original get(...,20.0), add request timeout, prove rest same."""
    tree=ast.parse(text);call=_call(tree)
    if any(n.arg=='request_timeout_s'for n in call.keywords):
        raise ValueError('Original client already declares request timeout; inspect rather than overwrite')
    found=[n for n in call.keywords if n.arg=='ws_ping_timeout_s']
    if len(found)!=1:raise ValueError('Original client ping-timeout keyword missing/duplicated')
    value=found[0].value
    if (not isinstance(value,ast.Call)or not isinstance(value.func,ast.Attribute)
            or value.func.attr!='get' or not isinstance(value.func.value,ast.Attribute)
            or value.func.value.attr!='deploy_cfg' or not isinstance(value.func.value.value,ast.Name)
            or value.func.value.value.id!='self' or len(value.args)!=2 or value.keywords
            or not isinstance(value.args[0],ast.Constant)or value.args[0].value!='ws_ping_timeout_s'
            or not isinstance(value.args[1],ast.Constant)or value.args[1].value!=20.0):
        raise ValueError('Original ping timeout expression is not the inspected deploy_cfg.get(...,20.0)')
    # AST columns are byte offsets. Preserve all unrelated source bytes exactly.
    lines=text.encode('utf-8').splitlines(keepends=True)
    start=sum(map(len,lines[:value.lineno-1]))+value.col_offset
    end=sum(map(len,lines[:value.end_lineno-1]))+value.end_col_offset
    raw=text.encode('utf-8')
    replacement=b'2100.0, request_timeout_s=2100.0'
    patched=(raw[:start]+replacement+raw[end:]).decode('utf-8')
    post=validate_rpc_text(patched)
    if _other_signature(call)!=post['other_constructor_signature_sha256']:
        raise ValueError('Unrelated client constructor arguments changed')
    expected_tree=ast.parse(text)
    expected_call=_call(expected_tree)
    for keyword in expected_call.keywords:
        if keyword.arg=='ws_ping_timeout_s':keyword.value=ast.Constant(value=TIMEOUT_SECONDS)
    expected_call.keywords.append(ast.keyword(arg='request_timeout_s',value=ast.Constant(value=TIMEOUT_SECONDS)))
    # Keyword insertion occurs directly after ping, so compare independent of
    # ordering while proving every other AST node in the whole file is unchanged.
    expected_call.keywords.sort(key=lambda n:n.arg or '')
    actual_tree=ast.parse(patched);_call(actual_tree).keywords.sort(key=lambda n:n.arg or '')
    if ast.dump(expected_tree)!=ast.dump(actual_tree):raise ValueError('RPC patch changed unrelated syntax')
    return patched,{**post,'adapter':'private_sdk_rpc2100_v1',
        'source_sha256':hashlib.sha256(raw).hexdigest(),
        'patched_sha256':hashlib.sha256(patched.encode()).hexdigest(),
        'shared_SDK_modified':False,'only_timeout_keywords_changed':True}


def patch_private_namespace(namespace):
    base=Path(namespace['private_base']).resolve(strict=True)
    source=Path(namespace['private_source'])/'src/eval_client/eval_env.py'
    if source.is_symlink()or not source.resolve(strict=True).is_relative_to(base):
        raise ValueError('RPC adapter may write only the newly isolated regular SDK file')
    original=source.read_text()
    if hashlib.sha256(original.encode()).hexdigest()!=namespace['eval_source_sha256']:
        raise ValueError('Private SDK source changed before RPC patch')
    patched,proof=patch_rpc_text(original);source.write_text(patched)
    namespace['eval_source_sha256']=proof['patched_sha256']
    namespace['rpc_timeout_contract']=proof
    return namespace


def validate_rpc_namespace(namespace):
    source=Path(namespace['private_source'])/'src/eval_client/eval_env.py'
    proof=namespace.get('rpc_timeout_contract',{})
    actual=validate_rpc_text(source.read_text())
    if (proof.get('adapter')!='private_sdk_rpc2100_v1' or proof.get('shared_SDK_modified')is not False
            or proof.get('only_timeout_keywords_changed')is not True
            or any(proof.get(k)!=v for k,v in actual.items())
            or hashlib.sha256(source.read_bytes()).hexdigest()!=proof.get('patched_sha256')
            or namespace['eval_source_sha256']!=proof.get('patched_sha256')):
        raise ValueError('Private client timeout proof/source changed')
    return actual
