# Steady online Subtask: Follow ECoT

This branch adds executable goal continuity and category/basket admission to the
original online Subtask implementation. An unfinished physical-instance goal is
locked to one exact ID/text; a later target must respect observed category/basket
assignments. Invalid/uncertain decisions stop before policy inference and are not
reported as SDK outcomes. The original source branch is unchanged.

**Start with [the steady protocol and integration instructions](online_subtask/docs/steady.md).**
The new model-server CLI requires a steady configuration by default; create it
from a genuine original model spec with `runtime/prepare_steady_spec.py`.
The current ID is `mainhist93_tok65536_steady_v1`. A separate visual-observer call
provides fallible causal scene evidence; deterministic code filters the original
288 candidates, then the selector still returns only `subtask_id`. Each call uses
at most96 images, with a shared1850s budget. No Summary/Reasoning enters π0.5.

The implementation and CPU fixtures are supplied here. A new real API/GPU/native
rollout or semantic-quality improvement is not claimed. The existing checkpoint,
normalization, policy inputs and action/SDK settings are retained.
The complete offline suite passed **78 checks**, including the3 loopback HTTP
checks, with0 failures/errors/skips; see the [steady CPU receipt](online_subtask/provenance/steady_cpu_validation.json).
The [follow-up reviews](online_subtask/docs/steady_review.md) document reproduced
defects and corrections, including worker-thread overlap and basket visibility.

## Historical base archive

This repository preserves the deployed `mainhist93_tok65536_v1` online Subtask
generation and π0.5 integration. DeepSeek selects one exact ID from a fixed
288-entry catalogue using current observations, sampled visual history and
acknowledged executed Subtasks. The selected catalogue text conditions the next
policy prediction.

This is an experiment-specific source archive. It preserves strict asset,
observation, timing and execution checks; it is not a general training repository
or a self-contained simulator distribution.

## Preserved configuration

| Component | Contract |
|---|---|
| Current observation | Synchronized main, left-wrist and right-wrist RGB; measured State14; complete task instruction |
| Visual history | Up to93 strictly past main-camera observations, uniformly sampled over the whole episode's approximately10Hz capture grid; up to96 images including the current3 |
| Executed history | Complete chronological exact ID/text, observation frame and executed action range; added only after a verified execution ACK |
| DeepSeek | `deepseek-v4-1-flash-261017`, `max_tokens=65536`, temperature0, thinking enabled, reasoning effort high, JSON output |
| Output | Only `{"subtask_id":"one listed ID"}`; exact text lookup in the frozen288 catalogue |
| Time bounds | HTTP600s; identical-request retry admission budget1850s; policy RPC/ping2100s; at most3 total identical attempts |
| Policy | `A_subtask_online` EMA5000; predict50, execute up to8, flow10, noise42 |
| Environment | 25Hz control; original SDK1100-action limit and success/termination rules; simulation frozen during waits |

The historical generation entry point is
[`MultiviewHistoryProvider`](online_subtask/runtime/multiview_provider.py), using
[`multiview_protocol.py`](online_subtask/runtime/multiview_protocol.py) and the
complete [`selector_prompt.md`](online_subtask/runtime/selector_prompt.md).
`SubtaskClient` and `HistoryClient` retain older implementations and shared
helpers; their legacy generation methods are not this configuration's entry
points.

The original runtime order was:

```text
current observation + causal history
  → DeepSeek single ID → unchanged catalogue text
  → Instruction + " Current subtask: " + Subtask
  → π0.5 predicts50 → SDK executes up to8
  → verify actual actions/clocks and commit execution ACK
  → next observation and request
```

π0.5 receives only the current three images, State14 and composed prompt. The
visual and executed histories are selector inputs, not policy image history.
See the [base protocol](online_subtask/docs/protocol.md) for preserved sampling
and ACK semantics, and [steady.md](online_subtask/docs/steady.md) for the current
visual admission, hard lock and legal-switching semantics.

## CPU checks

From the repository root, use a separate Python3.11 environment with NumPy.
The observed policy environment used Python3.11.17 and NumPy1.26.4:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install 'numpy==1.26.4'
python -B -m unittest discover -s online_subtask/runtime -p 'test_*.py' -v
```

These checks use local fixtures and mocked policy/transport dependencies. Three
HTTP transport checks bind a temporary loopback port; allow local loopback
binding for the full suite. They do not need an API key, external API access,
checkpoint, GPU or native simulator. NumPy installation alone is not an exact
environment lock for full policy execution.

The packaged runner can also ignore inherited `PYTHONPATH`:

```sh
python -E -B online_subtask/tests/run_cpu_tests.py
```

Use `-I` in place of `-E` for stronger interpreter isolation when NumPy is
installed in that interpreter's regular site-packages, as in the virtual
environment above. `-I` hides user-site packages.

The release owner's copied checkout passed all31 CPU checks, including the3
loopback tests, with0 failures/errors/skips and without the original workspace
on its import path. That check used Python3.9.6 and NumPy2.0.2; it is distinct
from the observed policy deployment environment above. Final clone/ZIP
verification remains pending in
[validation.md](online_subtask/docs/validation.md). CPU checks do not establish
native success, semantic correctness or complete GPU reproducibility.

Verify the historical source mapping, explicit steady adaptations and declared
prompt/catalogue hashes locally:

```sh
python -B tools/verify_archive.py
```

From a clean committed checkout, build a deterministic source ZIP into a
directory outside the repository:

```sh
python -B tools/build_source_archive.py --output-dir ../source-archives
```

The builder also writes SHA256 and per-file manifest sidecars. It excludes Git
metadata and refuses uncommitted source changes. This command documents the
build procedure; final archive/clone acceptance remains a separate release
check.

## Integration and repository layout

| Path | Purpose |
|---|---|
| `online_subtask/runtime/` | Runtime, steady visual guard/provider, catalogue and behavior tests |
| `online_subtask/integration/experiment_contract/` | Parent experiment's source/support validation dependencies; reference material, not a training quickstart |
| `online_subtask/configs/mainhist93_tok65536_v1.json` | Declarative snapshot of the fixed scientific configuration |
| `online_subtask/configs/mainhist93_tok65536_steady_v1.json` | New steady protocol configuration and enforcement contract |
| `online_subtask/configs/external_paths.example.json` | External-asset path placeholders; not a new runtime loader |
| `online_subtask/tests/` | Isolated CPU runner and the explicitly identified prior-prompt fixture |
| `online_subtask/docs/` | Protocol, integration requirements, verification scope and provenance |
| `online_subtask/provenance/` | Source mappings, hashes and release evidence supplied with this archive |

Full GPU execution additionally needs the original compatible OpenPI/SDK
sources, trained checkpoint and EMA export, normalizer, parent configuration,
tokenizer assets, layout data and genuine training/data-audit/binding evidence.
Their identity checks remain enforced. Paths alone cannot recreate missing
assets or authorize a new machine binding. Follow the
[integration requirements](online_subtask/docs/integration.md) before adapting
the runtime.

`prepare_policy.py` is a retained historical CPU preparation tool.
`prepare_native_sim.py` retains a seed0 pilot interface that is not a current
50-case launcher. Neither is a complete quickstart for this configuration.
Machine-specific queues, recovery supervisors and web/video publishers are not
part of this archive's public runtime startup flow. The separate GT-trajectory
offline SSR annotation experiment is outside this archive.

Weights, datasets, trajectory images/videos, raw API responses, private reasoning,
credentials and operational host/process logs are not distributed here.
See [provenance](online_subtask/docs/provenance.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source and licensing scope.
No new license is assigned to the experiment-specific code by this archive.
