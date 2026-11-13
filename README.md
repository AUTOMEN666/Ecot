# Online Subtask, Reasoning and Action Summary for classify_objects

This branch implements causal generation of all three public ECoT fields and
actual three-field pi0.5 conditioning. Start with the [SSR protocol and integration
guide](online_subtask/docs/ssr_branch.md). Active ID: `mainhist93_tok65536_steady_ssr_v1`.

The inherited visual observer/steady guard is followed by one joint selector
returning `subtask_id`, `reasoning` and `action.summary`. The host looks up exact
Subtask text from the original288 candidates. Seven source-bound real training
examples constrain action-stage language. Reasoning/Summary describe a grounded
next action over a short16-step window, separating observed facts from intended
outcomes. Current-test future images, expert actions and GT labels are excluded.

The API input remains current3RGB + State14 + instruction, up to93 strictly past
MAIN images uniformly sampled over the whole episode prefix (96total), and
verified actual-execution Subtask history. Both calls use temperature0, thinking
enabled/high and65536 output tokens, with one shared1850s budget,600s HTTP/max3
identical attempts, and2100s policy RPC. No semantic repair/redraw or fallback.

The runtime generates and validates all three fields BEFORE policy inference,
compares the prompt byte-for-byte with the actual original noDropout training
serializer, then gives pi0.5 only current3RGB, State14 and Instruction/SSR. It
retains predict50/execute8/flow10/noise42/25Hz/native1100 and the original SDK
freeze/action/termination rules. Only a real action/injection/clock ACK commits
history; complete actual SSR text stays in the execution audit.

A genuine existing **noDropout three-field SSR EMA checkpoint** and original
weights/normalization/tokenizer binding are required. The checkpoint step is
explicit; current inventory found noDropout EMA17392 and EMA19566. The preparer
validates a selected genuine binding without starting API/model/GPU. The server
requires SSR by default; --allow-legacy-replay explicitly opts into inherited
Subtask-only behavior.

## Task-scoped branches

- [Original Subtask](https://github.com/AUTOMEN666/Ecot/tree/ecot_online_generation_subtask_classify_objects)
- [Steady Subtask](https://github.com/AUTOMEN666/Ecot/tree/ecot_online_generation_subtask_steady_classify_objects)
- [Subtask + Reasoning + Action Summary](https://github.com/AUTOMEN666/Ecot/tree/ecot_online_generation_subtask_reasoning_and_action_summary_classify_objects)

Instruction,288 candidates and category/basket constraints target classify_objects.
Other tasks need their own catalogue and integration. Historical protocol IDs,
archive names and source receipts retain their identities. Inherited steady.md
records the old Subtask-only path; use ssr_branch.md for this branch's active path.

## Verification boundaries

See provenance/ssr_cpu_validation.json for the actual CPU receipt and
SSR_TRAINING_FORMAT_REFERENCE.json for seven original-training prompt comparisons.
Tests cover generation-before-inference, all three policy fields, actual ACK
methods/next history,96-image capacity, error/mutation/retry/concurrency gates,
private-thinking separation and original weight identity checks.

```sh
python3 -E -B online_subtask/tests/run_cpu_tests.py
python3 -E -B tools/verify_archive.py
python3 -E -B tools/verify_ssr.py
```

Code/CPU checks do not establish semantic accuracy, real API/GPU compatibility
or task-success improvement. Those empirical checks remain pending an explicit
checkpoint and an isolated online run. No new training or50-case scheduler is
claimed.
