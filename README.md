# Online Subtask, Reasoning and Action Summary for classify_objects

This branch implements causal online generation of three public ECoT fields and
conditions the existing noDropout SSR π0.5 checkpoint **EMA19566** on those fields.
The active experimental protocol is `mainhist93_tok65536_grounded_strict_ssr_v7`.
Start with the [grounded SSR guide](online_subtask/docs/grounded_ssr.md).
Whole-trajectory annotation quality and genuine H20 GPU/native acceptance are
still being evaluated. Neither code tests nor a rendered source video establish
that the annotations are reliable or that task success improves.

## Generation and execution order

1. Observe the current MAIN, left-wrist and right-wrist RGB images, synchronized
   State14, instruction and causal history. The first DeepSeek call reports
   current objects, basket contents and arm/grasp facts with local uncertainty.
2. Preserve physical-instance/goal continuity and admit actions supported by
   those fallible visual facts. The second call returns exactly `subtask_id`,
   public `reasoning` and `action.summary`. Copy Subtask text byte-for-byte from
   the original288 candidates; preserve generated Reasoning/Summary verbatim.
3. Validate all three fields **before** π0.5 inference. Compose the prompt using
   the exact original noDropout training serializer. π0.5 receives current3RGB,
   State14, instruction and the three public fields.
4. Execute up to8 actions from the50-step prediction. Only verified actual
   injection/action/clock ACKs commit a real online Subtask-history record.

V7 retains v4 strict JSON Schema with a finite semantic category vocabulary plus
null. Categories are visually chosen, never inferred by the host from descriptor
IDs or SDK labels. The selector schema remains byte-for-byte equivalent to the
v3 selector factory. Earlier current3 probes showed limited format compatibility,
but later v3 responses violated a category string pattern; strict mode is not a
guarantee. Both output format and visual/action constraints remain checked, with
errors preserved and no JSON-object fallback. Real v6 API regression runs have
already exposed both an observer grasp contradiction and a false rejection of a
completed-release clause. No v6 whole trajectory has passed quality acceptance.
These failures are preserved while the implementation is revised; a strict
schema does not certify the facts.

V7 keeps those failed versions and adds a current hand/object consistency
reminder. It also distinguishes one narrowly recognized completed-release
clause from a future release instruction: “Retract the empty left arm from the
white basket after releasing the pen.” This requires the exact same current
instance visibly released in that basket, an open/empty acting hand and a
sealed actual selector-proof/goal/envelope binding. Primary actions, actors,
state claims and destinations still undergo the original checks. The public
text is never rewritten. Other grammar can still be rejected; this is not a
general tense parser or a visual accuracy guarantee.

V5 also admits opening as the physical implementation of an already admitted
release. A Summary may omit an actor only for Lift/Carry/Lower/Release when the
Reasoning names one hand and current facts establish exactly one clear, closed,
secure holder of that exact target. The source text is never rewritten. Unknown
support, another holder or an ambiguous hand still stops. Explicit Summary chains
such as “then” are rejected; this does not establish general language accuracy.

V6 adds two scoped corrections. A direct constraint such as “without releasing
it” is not an instruction to release. During opening, continued release requires
the previous real adjacent observation to show the same closed, securely held
target ready over the same basket, with current supported opening and the other
hand clear/empty. At most two additional8-step windows permit Open/Release/Hold;
no facts are rewritten and no lift/carry is admitted. Earlier readiness must
already be confirmed: a current-ready report after a prior-not-ready report still
stops. Release-budget recovery after a dropped/regrasped same goal is not yet
implemented. These limitations remain visible in evaluation.

Reasoning explains the current constraint and a useful next adjustment. Summary
specifies a feasible immediate action over about8–16 control steps. Training
examples guide the language, never supply current test truth. Close requires an
aligned object between the fingers; lift requires a supported grasp; release
requires the correct basket and placement readiness. These conservative checks
are not a perception or English-semantic accuracy oracle. No output rewriting,
semantic redraw, replacement ID or fallback is used.

## Fixed settings

| Item | Setting |
| --- | --- |
| Current input | Three RGB views + synchronized State14 + full instruction |
| Visual history | Up to93 strictly past MAIN images, uniform across the entire approximately10Hz episode prefix, keeping earliest/latest past timestamps |
| Maximum image count |93 history +3 current =96 per call |
| Real online text history | Up to137 actual executed Subtasks: exact ID/text, observation frame and executed action range; reset per episode |
| Generator | Existing DeepSeek model, temperature0, thinking enabled/high, max_tokens65536, JSON output |
| Calls per replan | Current visual observer, then joint three-field selector |
| Timeouts/retries | HTTP600s, shared generation1850s, policy RPC2100s; max3 identical transport/empty-length attempts |
| Policy | Genuine noDropout SSR EMA19566; no retraining or weight substitution |
| Actions | Predict50, execute8, flow10, noise42,25Hz, native limit1100 |
| Simulation | Frozen during API/policy waits; original SDK termination/success rules |

Historical images and the visual observer's internal JSON are not π0.5 inputs.
Failed requests, unexecuted proposals and retries do not extend real history.
The same physical object and destination retain an ID through approach, grasp,
transport and placement; clear visual contradictions permit an explicit
correction. Another similar object in a basket does not complete this instance.

A real H20 numerical probe loaded the genuine EMA19566 with actual v4 generated
three-field/current-frame input. Two fresh seed42 predictions were byte-identical.
That establishes a narrow loader/numerical/input result; v7 live generation,
executed8-action ACKs and SDK task outcomes remain separate pending checks.

A separate real v6 observer component accepted93 past MAIN images plus current3
at source frame600 in about115 seconds, without retry. Independent standard
JSON Schema validation passed. Its history was explicitly empty because this
standalone probe had never generated earlier labels or executed actions. It
does not establish selector96, a complete label-history pipeline or annotation
accuracy, and does not serve as v7 prompt acceptance.

## Native adapter and evaluation

Use `pi05_ssr_policy`, `PI05_SSR_CHECKPOINT_STEP=19566` and
`PI05_SSR_PROTOCOL_ID=mainhist93_tok65536_grounded_strict_ssr_v7`. The preparer requires
a genuine original weights/normalization/tokenizer binding. The original seven
`pi05_policy` camera/action/clock files remain unchanged; the SSR adapter's
execution loop remains equivalent to that original loop. H20 renderer/device and
resource admission require separate verification; the A800 launcher alone does
not establish H20 compatibility.

Causal expert replay is a separate evaluation mode. Its recorded expert actions
were not conditioned on generated annotations, so exposure of those transitions
is never described as actual policy injection or ACK. Test GT, future images,
expert actions and segment labels are excluded from generator requests. A
separate renderer/evaluator can inspect same-frame GT and the following16-frame
window. Legal alternative object order is not automatically an error, and GT
text agreement is not measured semantic accuracy. Preserve complete source
videos, missing annotations, stop reasons and all failed versions.

The previous [steady SSR guide](online_subtask/docs/ssr_branch.md), protocol
`mainhist93_tok65536_steady_ssr_v1`, and historical receipts remain available.

## Task-scoped branches

- [Original Subtask](https://github.com/AUTOMEN666/Ecot/tree/ecot_online_generation_subtask_classify_objects)
- [Steady Subtask](https://github.com/AUTOMEN666/Ecot/tree/ecot_online_generation_subtask_steady_classify_objects)
- [Subtask + Reasoning + Action Summary](https://github.com/AUTOMEN666/Ecot/tree/ecot_online_generation_subtask_reasoning_and_action_summary_classify_objects)

Instruction, catalogue and category/basket constraints are specific to
`classify_objects`. Other tasks require their own catalogue and integration.

```sh
python3 -E -B online_subtask/tests/run_cpu_tests.py
python3 -E -B tools/verify_archive.py
python3 -E -B tools/verify_ssr.py
```

The CPU suite covers generation-before-inference, exact three-field prompting,
ACK/history methods,96-image capacity, retries/timeouts, mutation/concurrency
failures, current-fact/phase counterexamples and original weight contracts.
Actual API compatibility, whole-trajectory language quality, GPU/native
execution and SDK outcomes must each be reported from their own evidence.
