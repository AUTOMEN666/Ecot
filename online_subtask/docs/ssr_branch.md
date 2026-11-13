# Online Subtask, Reasoning and Action Summary for classify_objects

Active protocol: `mainhist93_tok65536_steady_ssr_v1`.
Branch: `ecot_online_generation_subtask_reasoning_and_action_summary_classify_objects`.

## Implemented generation and execution path

Each replan makes two DeepSeek calls: the inherited visual observer, then a joint
selector returning exactly `subtask_id`, `reasoning` and `action.summary`. Subtask
text is looked up from the original288 catalogue. Reasoning/Summary strings are
copied unchanged after strict schema validation. An unfinished physical-instance
and basket goal stays locked; its upcoming action phase can change the two
annotations without changing the goal ID.

Both calls use synchronized current main/left-wrist/right-wrist RGB, State14,
instruction, at most93 strictly past MAIN images uniformly sampled over the whole
approximately10Hz prefix, and acknowledged actual Subtask history. Total images
remain96. No future test images, expert actions, test labels, reward or SDK success
truth are supplied. The visual observer's evidence is fallible, not GT.

The selector also receives seven exact original training label examples, with
source episode/frame and projection SHA in `runtime/ssr_training_reference.json`.
They are actual training starts of training_episode200, explicitly style/phase
references rather than current scene evidence. Their object order, basket colors
and progress must not become current-test truth. Seven byte-exact comparisons
against the original noDropout training formatter are preserved in
`provenance/SSR_TRAINING_FORMAT_REFERENCE.json`.

The prompt requests a grounded short-term action intent over up to16 control
steps (~0.64s): arm/target, approach/alignment, gripper state or transition,
contact prerequisite, lift/carry/release as appropriate. Action Summary concisely
names the next action and purpose. Observed facts and planned outcomes are
separate; the offline teacher's access to future expert actions is unavailable
online. Public `reasoning` comes from JSON content, never private API thinking.
Prompt/schema checks cannot prove the generated plan is semantically correct.

```text
current observations + causal history
 -> visual observer + steady goal/category/basket admission
 -> joint allowed-ID / public Reasoning / Action Summary
 -> strict schema + exact Subtask lookup + full-context authorization
 -> actual original noDropout training serializer, byte-equality check
 -> current3RGB + State14 + Instruction/SSR into pi0.5
 -> predict50 / execute up to8 through original SDK
 -> actual action/injection/clock ACK and durable history commit
 -> next observation and acknowledged Subtask prefix
```

Missing/unsupported fields, unknown IDs, unauthorized goals or mutated text stop
before inference. There is no semantic redraw, ID repair, fallback or automatic
text replacement. Proposals and failed requests do not create execution history.
ACKs preserve complete actual SSR text/prompt plus actions/clocks; the generator's
Subtask-history rows retain their original five fields and137-record maximum.

The camera/action decoder, flow10, noise42,25Hz,1100-action cap and original SDK
success, termination and frozen-wait mechanisms remain unchanged. The two API
calls share one1850s generation budget: HTTP600s, at most3 identical attempts for
the preserved transient/empty-length cases, and policy RPC2100s. Lifecycle
exclusion spans generation AND policy inference.

## Genuine noDropout SSR checkpoint and startup

Use an existing summary_subtask_reasoning, dropout0.0, full-parameter EMA model
with its genuine original weights binding, normalizer, tokenizer and training
identity. Subtask-only and dropout0.5 weights are rejected. Source training
`eval_ecot=false` stays unmodified; runtime three-field evaluation is separately
recorded as effective_eval_ecot=true.

The checkpoint step is explicit. Current operator inventory found committed
noDropout attempt2 EMA17392 and EMA19566; presence is not a quality comparison.
Obtain a selected checkpoint's fresh binding through the original training
project's `evaluation/support/weights_contract.py:prepare_weights`. The archive
does not manufacture or bypass that original source/weight/normalization evidence.

```sh
python3 -B online_subtask/runtime/prepare_ssr_spec.py \
  --weights-binding /path/to/genuine_noDropout_SSR_weights_binding.json \
  --parent-root /path/to/original_noDropout_SSR_training_root \
  --audit-path /new/run/seed1_native50_steady_ssr_mainhist93_65536_20261113_v1/case/actions.jsonl \
  --output /new/run/ssr_model.json

python3 -B online_subtask/runtime/model_server.py \
  --config /new/run/ssr_model.json --port 30111 \
  --ready-output /new/run/ssr_ready.json
```

The preparer validates existing external weights but starts no API/model/GPU.
This branch's server defaults to SSR. Inherited Subtask-only replay requires
explicit --allow-legacy-replay; it is not an SSR test.

The native SDK must use `runtime/pi05_ssr_policy`, with
`PI05_SSR_CHECKPOINT_STEP` set to the genuine selected step (19566 for the current
operator choice). The inherited `pi05_policy` adapter accepts only Subtask and
legacy checkpoint milestones; it cannot run this three-field model. The SSR
adapter retains the original action loop/camera/clock implementations, validates
all three fields and the full training prompt before executing any command, and
uses the original RPC `obs` keyword for execution ACKs and terminal reports.

```sh
python3 -B online_subtask/runtime/prepare_ssr_native.py \
  --model-config /new/run/ssr_model.json \
  --layout-manifest /path/to/verified_original_C5000_seed1_native50_manifest.json \
  --sim-python /path/to/existing/native/python --sim-gpu-uuid GPU-REPLACE \
  --port 30111 --layout 0 --command-limit 128 \
  --output /new/run/seed1_native50_steady_ssr_mainhist93_65536_20261113_v1/engineering_case
```

This produces a private namespace and `sim_launch.json`, without starting a job.
The128-command stop is an engineering diagnostic, never an SDK failure or formal
episode. Real external assets, exclusive resources, platform renderer/device
acceptance and an authorized API key are still required. In particular an A800
plan does not establish H20 renderer compatibility. No new50-case scheduler or
native rollout result is supplied by the archive.

## Verification scope

```sh
python3 -E -B online_subtask/tests/run_cpu_tests.py
python3 -E -B tools/verify_archive.py
python3 -E -B tools/verify_ssr.py
```

The fixtures verify generation-before-inference, three actual policy prompt
fields, original-training golden bytes, full8-action ACK and next history,
96-image/29-ACK prefixes, absence of phantom history, partial termination,
mutation rejection, same-request retries, concurrency, public/private reasoning
separation, and rejection of incompatible policy conditioning. A direct SDK
adapter-to-model RPC fixture also checks history replies, actual `obs` keyword
ACKs, full8/partial terminal execution and source equality of the action loop.
See provenance/ssr_cpu_validation.json for the actual CPU result. Historical
steady receipts remain historical. CPU fixtures are not real API/GPU/native
acceptance. Semantic quality and task-success improvement remain unmeasured.
