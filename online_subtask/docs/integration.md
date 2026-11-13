# Integration requirements and retained boundaries

The archive preserves a deployed experiment's contracts. The configuration
snapshots describe those contracts; they do not make the preserved files a
portable launcher.

## Active components

| Component | Role |
|---|---|
| `multiview_protocol.py` | Validates current/history observations, constructs causal requests and resolves the exact catalogue ID |
| `multiview_provider.py` | Per-episode capture, DeepSeek transport/retry lifecycle and acknowledged executed history |
| `executed_history.py` | Exact action/injection/clock ACK validation and committed text memory |
| `model_server.py` | Generates the Subtask before π0.5 inference and enforces the experiment configuration |
| `pi05_policy/` | Observation/action conversion, original SDK execution, frozen waits and clock checks |
| `integration/experiment_contract/` | Preserved parent source/support validation dependencies |

Older `SubtaskClient.generate` and `HistoryClient` generation paths retain
earlier prompts/budgets and are not supported startup choices for this archive.
Their modules are still imported for shared encoding and validation helpers.
Do not add a missing legacy reference or prompt to make those old paths appear
equivalent to the current protocol.

## Full policy and native execution prerequisites

Supply and independently verify all of the following outside the public source
archive:

- The actual `A_subtask_online` EMA5000 checkpoint/export, normalizer, training
  identity, context-cache/data-audit records, parameter manifests and export
  binding evidence.
- The compatible parent experiment configuration and directory structure.
  `model_server.initialize_paths` imports parent `src/` and
  `evaluation_native/support/`; the retained contract source is a reference
  closure, not a replacement for missing audit assets or a new training run.
- The exact OpenPI source and local strict input-transform behavior used by
  that export; matching JAX/Flax/Optax/Orbax and tokenizer assets; the compatible
  WebSocket client/server transport.
- The original RoboDojo/XPolicyLab/IsaacSim integration, native task and camera
  interfaces, and genuine layout source data/manifests. Video support uses the
  original SDK recording components.
- A genuine external API authorization supplied by the operator when network
  generation is deliberately enabled. The preserved provider reads
  `ECOT_AUTHORIZED_API_KEY`; CPU tests need no key. Never commit an actual value
  in configuration, commands, logs or provenance.

Software source/version evidence is tracked in [provenance.md](provenance.md).
Where an exact deployed version is not recorded, it remains unpinned; installing
the latest packages cannot establish equivalence.

`configs/external_paths.example.json` contains placeholders for operator-owned
assets. The preserved guard also requires the experiment namespace component
`seed1_native50_mainhistory93_65536_20261113_v1` in the audit path. Checkpoint
bindings contain source identities, original paths/stat fingerprints and real
data-audit hashes. Moving assets or changing machines can correctly fail those
checks. Do not edit binding receipts, reuse a different machine's attestation,
remove validation or invent an audit to bypass rejection. A portable integration
would require a separately reviewed adaptation and fresh genuine evidence.

## Episode lifecycle

Use a fresh provider output directory. The provider exclusively creates some
files and keeps episode frames, last query frame and committed history in
memory. It is not a transparent process-restart/resume library.

1. Initialize the verified model and explicitly reset the actual episode.
2. Capture every required history source-grid observation, including frame0,
   without advancing policy RNG or simulation for capture itself.
3. At frame0 and each subsequent multiple of8, provide the current synchronized
   observation; generate the ID and exact text before inference.
4. Execute returned commands through the original SDK with clock evidence.
5. ACK only the actions actually executed. A full8-action ACK permits the next
   normal query; a terminal partial chunk ends that episode's query sequence.

The ACK entry point is `acknowledge_executed_subtask`; it must receive the real
injection, expected commands and clock evidence. Passing a fabricated record or
simply appending a predicted ID is not an equivalent integration.

## Historical tools and operational scope

`prepare_policy.py` retains a dated CPU preparation path. `prepare_native_sim.py`
retains a seed0 pilot interface, while the current layout/namespace contracts
require the seed1 pool. They are preserved provenance, not a tested end-to-end
launcher for this archive's50-case evaluation.

The deployed cohort used a separate experiment-specific scheduler and asset/
resource admission workflow. Machine-specific queues, continuation/quarantine/
drain supervisors, GPU ownership operations, live dashboards and video
publishers are outside the public runtime startup flow. Their presence in
historical evidence does not create a general command for a new allocation.
The separate offline GT-trajectory SSR comparison, which predicts additional
annotation fields on expert observations, is not included in this single-ID
online policy contract.
