# Current isolated A800 main-history93 /65536 invocation

The root execution agent is the sole remote deployment, API, credentials,
process-signal and GPU owner. These sources target only authorized A800 hosts
42132 and43585, four rolling policy/simulator GPU pairs per host. They do not
resume ERVLA or launch/restore synthetic GPU parking. Sampled idle GPUs are not
an authorized reservation.

## Frozen scientific/runtime contract

- `config_id` and `protocol_config_id`: `mainhist93_tok65536_v1`.
- Current observation: MAIN/left-wrist/right-wrist RGB plus synchronized State14
  and original instruction. Strictly past imagery: MAIN only, at most93 points
  uniformly sampled over all actual approximately10Hz floor(2.5*k) captures;
  maximum96 images including the current three. Captured wrist history is audit
  material and never enters the request.
- Full actual-executed Subtask ID, exact288-catalogue text, observation frame and
  actual action range. Failed requests and unexecuted proposals are excluded.
- DeepSeek `deepseek-v4-1-flash-261017`, max_tokens65536, thinking enabled,
  reasoning_effort high, temperature0, JSON with only subtask_id. No fallback,
  semantic retry, ID substitution or human replacement.
- HTTP absolute wall-clock600 seconds per request, request-only subprocess
  killed and reaped before retry. Total identical-request admission budget1850
  seconds including request construction, at most3 attempts. Private client
  request/ping and policy-server ping timeout2100 seconds.
- Original A_subtask_online EMA5000 binding, predict50/execute8/flow10/noise42,
  25Hz, native1100-action limit and original SDK success/termination unchanged.
  All waits freeze simulation. Actual post-execution ACK remains unchanged.

## Deployment and acceptance

Reuse the existing environment and the corresponding host's frozen
`export_binding.py`. The authoritative new root invocation is documented in the
new report's root operations and receipts. Use the sibling `prepare50.py` then
`controller50.py`, with the matching `CORE_<host>.json` manifest and prepared
scope. The fixed fresh remote base is:

```
/mnt/xzt/ecot/diagnostics/online96_10hz_20261112/seed1_native50_mainhistory93_65536_20261113_v1
```

The new portbase is30110. Each host prepares25 cases: seed1/layout0..24 on42132,
seed1/layout25..49 on43585. Prepare is CPU/filesystem only. Root must verify real
GPU availability/owned lifecycle, real96-image65536 API acceptance (93pastMAIN pluscurrent3), source and
prepared-model guards, then actual model/native8-action ACK and next-request
history before claiming runtime acceptance. CPU tests do not establish SDK
success or semantic quality.

Keep all prior65536/34-image and old16384/96-image records, namespaces, source, API evidence and videos
separate. Never hot-edit a running frozen core. Retire only exact owned old
process identities through the documented root handoff, verify cleanup, then
launch the new namespace. No third host or foreign process may be signalled.

The stable viewer remains:
http://127.0.0.1:8766/experiments_viewer/deepseek_seed1_native50_executed_history96_20261113/index.html

Publish actual per-case video and synchronized injected Subtask at case closure;
never reuse another configuration or attempt's footage. Missing SDK output is
an engineering interruption, not SDK false. Preserve unrecoverable recordings
with truthful missing/unfinalized status.

## Local CPU verification

From the workspace root:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s reports/deepseek_seed1_mainhistory93_65536_20261113/code/core42132 -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s reports/deepseek_seed1_mainhistory93_65536_20261113/code/core43585 -p 'test_*.py' -v
```

`ONLINE8_REMOTE_INVOCATION_REFERENCE.md` is retained solely as a historical
source reference; its historical placement/timeouts are superseded here.
