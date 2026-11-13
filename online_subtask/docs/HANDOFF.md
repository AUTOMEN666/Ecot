# Steady Subtask implementation handoff

Work is confined to `ecot_online_generation_subtask_steady_classify_objects`, based on original
`ecot_online_generation_subtask_classify_objects` at `8e4f340940c56fbbe5c30e0ee578d6e5b8678321`.
Do not edit or force-update the original branch. No training, GPU process, real
API request or simulator episode was started for this implementation/review.

The user's objective is hard goal continuity and legal category/basket choices
for testing a trained policy's Follow ECoT behavior. The completed implementation
locks one physical instance and exact288-catalogue ID/text until supported
completion, enforces observed bidirectional basket mappings, gives held objects
priority for new goals, rejects descriptor/category and instance ambiguities,
and admits history only from validated actual execution ACKs with durable logs.
The visual observer is internal; the policy receives only current3RGB, State14,
the original instruction and the current Subtask. No Summary/Reasoning is added.

Preserve the original source branch's actual settings: EMA5000, predict50,
execute8, flow10, noise42,25Hz, current3 images plus at most93 strictly past MAIN
images for DeepSeek, max_tokens65536, and the existing transport/time budgets.
Historical earlier proposals using31 three-view history points or16384 tokens
are not the configuration of this source branch. The14 pinned native/action
source files and checkpoint binding remain unchanged.

Validation is102 offline test methods, including loopback HTTP, controlled
concurrency, persistence failure injection, task-rule enumeration and one complete
mocked three-goal model/provider/execution-ACK flow. Receipt:
`../provenance/steady_cpu_validation.json`. Details: [steady_review.md](steady_review.md).

Run from the repository root:

```sh
python3 -E -B online_subtask/tests/run_cpu_tests.py --report /tmp/steady_cpu.json
python3 -E -B tools/verify_archive.py
python3 -E -B tools/verify_steady.py
```

The test suite uses no external API/weights/GPU/simulator; localhost tests need
permission to bind temporary loopback ports. `--exclude-http` is a partial check.

Remaining empirical work is real API/native acceptance and any explicitly
authorized50-case evaluation with genuine external assets and an adapted runner.
Do not fabricate those receipts or report offline checks as policy success. A
decision stop, final visual placement and SDK whole-task outcome are separate.
Do not omit stopped cases. The observer's visual identity/category/completion
facts remain fallible; deterministic admission guarantees consistency with the
accepted evidence rather than ground-truth recognition.

Workspace operational constraints still apply: ERVLA H20-6/H20-5 are paused;
synthetic GPU parking is retired. Do not restart them or touch other projects.
