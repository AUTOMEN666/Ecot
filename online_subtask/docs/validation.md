# Verification scope

## Packaged checkout

The release owner ran the complete suite from a copied checkout outside the
original experiment workspace, using `python3 -E -B` with Python3.9.6 and
NumPy2.0.2. All31 checks passed, including3 loopback HTTP tests, with0 failures,
0 errors and0 skips. The original workspace was absent from the test import
path. This is separate from the observed deployed policy environment
(Python3.11.17/NumPy1.26.4) and makes no GPU-environment equivalence claim.

| Check | Packaged-checkout result |
|---|---|
| Preserved-source map verification |48 mapped sources passed;47 retain original bytes, one test-only fixture path adaptation |
| CPU behavior suite, including3 loopback HTTP checks |31 passed;0 failures/errors/skips in the separate copied checkout |
| No implicit original workspace/history dependency | Original workspace absent from the successful test import path; old prompt is an explicit package fixture |
| Final public file list and secret/data exclusions | Pending final review |
| Complete file inventory / clean committed clone / extracted ZIP checks | Pending final release-owner receipts |
| External software/version provenance | Observed versions and selected source hashes recorded; unknown upstream commits remain unpinned |
| New API/GPU/native execution from this archive | Not performed by the documentation/packaging checks |

The [source map](../provenance/SOURCE_MAP.json) records each original/destination
hash and the sole test adaptation. Recheck it with:

```sh
python -B tools/verify_archive.py
```

The final release owner must attach actual clone/ZIP receipts before claiming
those pending checks passed. The copied-checkout CPU result does not fill them.

## CPU commands and meaning

From the repository root, with NumPy installed:

```sh
python -B -m unittest discover -s online_subtask/runtime -p 'test_*.py' -v
python -E -B online_subtask/tests/run_cpu_tests.py
```

`-E` ignores inherited Python environment settings, including `PYTHONPATH`.
`-I` may replace `-E` when NumPy is installed in the interpreter's regular
site-packages; it hides user-site packages. A fresh virtual environment avoids
that ambiguity. Isolation verification must also use a copied checkout outside
the original experiment workspace.

The full existing suite has31 checks. Three local HTTP transport tests require
permission to bind a temporary loopback port. The runner's `--exclude-http`
option is a deliberately partial28-check diagnostic, not a substitute for the
full suite. A loopback permission error is an environment restriction, not a
passing HTTP test. No test requires a real API key or external model call.

The tests cover strict ID/JSON behavior, causal image sampling, prompt
compatibility, bounded local HTTP behavior and actual-execution-history
contracts, including runtime integration with mocked policy/JAX/transport/parent
dependencies. These mocks do not load the real EMA checkpoint or simulate a
real native episode. `native_cpu_smoke.py` is a separate asset-dependent tool
that imports actual model dependencies; it is not part of the minimal NumPy
test promise.

## Historical source and deployment evidence

Before packaging, the original frozen source manifest's77 entries were
independently checked. Existing workspace CPU coverage comprised28 direct
checks plus3 HTTP checks rerun where local loopback binding was permitted.
Those are source/workspace facts, not packaged-checkout test results.

Historical deployment receipts recorded generation before inference on selected
frame0/8 inputs, real first8-action ACKs with next-request executed history, and
actual acceptance of the maximum96-image request. Sanitized provenance may
summarize that evidence. Reading a receipt is not a fresh API/GPU execution, a
complete50-case result, a semantic-quality pass or validation of every external
asset on a new machine.

A complete reproduction claim additionally requires the genuine checkpoint,
normalizer, training/data-audit bindings, exact software/SDK sources, same layout
data, actual execution/ACK evidence and original SDK outcomes. Real rollout
quality and synchronized videos are distinct from CPU/source verification.
