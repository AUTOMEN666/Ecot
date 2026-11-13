# Steady Subtask: runtime constraints for Follow ECoT tests

Branch: `ecot_online_generation_subtask_steady`, based on
`ecot_online_generation_subtask` commit `8e4f340940c56fbbe5c30e0ee578d6e5b8678321`.
The source branch and its deployment/results are unchanged. This version is a
new protocol, `mainhist93_tok65536_steady_v1`, with its own output namespace.

## What is enforced

| Constraint | Executable behavior |
| --- | --- |
| Unfinished goal continuity | Once selected, the same physical instance, candidate ID, exact text and destination stay locked. The next selector request contains a singleton allowed ID. A different ID is rejected before policy inference. |
| Goal completion | The locked instance must be visibly released inside its own destination in at least two distinct, transmitted observations after selection AND after its most recent observed drop/regrasp/unreleased state, including the current frame. Another similar instance, execution ACK, elapsed actions or an ID change cannot complete it. |
| Same category, same basket | Category-to-basket assignments are established by visually observed released containment, including initial basket contents. A prior textual plan does not establish an assignment. |
| Different categories, different baskets | Both directions of the mapping are checked. If pens occupy red, camera-to-red is not an allowed candidate, even though that ID exists in the288 catalogue. Mixed or contradictory observed contents stop the decision. |
| New destination visibility | Every newly selected destination must have currently inspected contents, including a previously assigned basket. An unfinished locked goal remains locked through basket occlusion. |
| Category names | Color-qualified and generic categories are rejected. Broad word families only catch obvious descriptor/category contradictions; they NEVER replace or merge the observed category keys. Distinct reported categories such as toy_car and toy_bus remain distinct. Ambiguous words such as box/device/tool/ring are not reliable category labels. |
| Physical instance identity | Matching descriptions do not merge different instances. Ambiguous new target identity stops admission. Omission/occlusion cannot complete a target; a positively observed later drop clears that instance's completion. |
| Actual injection | The runtime independently checks the authorized ID/text, frame, episode, active goal and category mappings immediately before `policy.infer`. |
| Uncertainty or conflict | Raise a recorded decision stop with `SDK_outcome=null`; do not inject, repair an ID, retry for semantics or substitute an answer. |
| No next legal target | Preserve any supported current-goal completion and accepted scene facts, then stop without selecting a substitute. Completion of a last pick/place goal is not the SDK whole-task result. |
| Worker-thread lifecycle | Reject overlapping reset, observation, history, ACK, injection-read, terminal and action RPCs before mutation. The model gate spans both API calls AND policy inference; the provider also protects direct use. |

This tests continuous following of the CURRENT goal, with legal switching after
confirmed completion. It does not lock one target for the entire sorting episode.
Decision stops are latched until an explicit new episode reset: another RPC call
cannot silently redraw a rejected semantic answer in the same episode.
An overlapping RPC raises `RPCBusyError` without being queued, starting another
API request or latching a decision stop. Reset can be issued after the active RPC
returns; it cannot mutate a running generation or inference. A delayed direct
provider request naming an old episode is rejected before the stop handler, so it
cannot poison the new episode. These gates do not change legacy replay behavior.
No simulator labels, object poses, reward, task-success truth, expert annotations
or future images are introduced as generator/policy inputs.

## Why there is a separate visual observer

The original288 catalogue has wording and basket colors, not reliable scene
category, occupancy, identity or completion facts. Those facts cannot honestly be
inferred from an executed Subtask ledger. Each replan therefore performs:

1. One visual-observer request using the SAME causal image payload as the selector.
   It returns typed object/category/location/release evidence and image-frame
   references, including placement_frames and unplaced_frames. Positive and
   negative witnesses cannot overlap. These are fallible model observations, not GT.
2. Deterministic admission: update observed category/basket assignments, check
   completion of the active instance, and compute the legal subset of the288 IDs.
3. One Subtask-selector request returning ONLY `{"subtask_id":"..."}`. Its ID must
   belong to that subset; when the goal is unfinished, the subset has one ID.
4. Unchanged exact text lookup and policy input composition, then original SDK
   execution and execution ACK.

The observer is an additional DeepSeek call; both requests use the original model,
temperature0, thinking enabled/high and max_tokens65536. Each request contains at
most93 strictly past MAIN images plus current3 views. They share ONE1850-second
admission budget; each request allows at most3 identical attempts for the preserved
transport/empty-length cases. Each HTTP attempt remains bounded by600 seconds,
with the request child killed/reaped on timeout. Policy RPC/ping remain2100 seconds.
The existing synchronous model call and SDK freeze envelope cover BOTH requests.
No real API/GPU/native validation of the new two-call path is claimed by CPU tests.

π0.5 still receives ONLY current3RGB, measured State14 and
`Instruction + ' Current subtask: ' + exact_catalog_text`. Internal scene evidence
and maps are not added to its observations. No ECoT Summary/Reasoning is generated
or injected. There is no new training, weight selection, normalization, action
decoder, prediction50/execution8/flow10/noise42/25Hz or SDK success-rule change.

The guarantee is consistency WITH admitted observations. A visual model can still
misrecognize a physical object, placement or category. Program checks cannot prove
pixel truth. Stable wrong perception remains an error, not successful following.

## Using the new protocol

Keep the genuine checkpoint, normalizer, external source and binding checks from
[integration.md](integration.md). Do not rewrite old receipts or reuse an old run
directory. Convert a genuine original model specification into a NEW file:

```sh
python3 -B online_subtask/runtime/prepare_steady_spec.py \
  --base-spec /path/to/verified_original_model.json \
  --audit-path /new/run/seed1_native50_subtask_steady_20261113_v1/case/policy/actions.jsonl \
  --output /new/run/steady_model.json
```

The converter does not load weights, call an API, start a simulator or claim a
fresh resource/asset acceptance. It preserves the weight/export/normalizer and
action configuration and checks the original and new protocol specifications.
It refuses to overwrite its output. Start the existing isolated model-server/SDK
integration with this specification and this branch's runtime source. The server
CLI requires steady enforcement by default. Historical unguarded replay requires
the explicit `--allow-legacy-replay` option; it is not a steady test.
The old experiment-specific scheduler's pinned source/protocol hashes must not be
bypassed to launch this new version. This archive does not supply a complete new
50-case scheduler, original checkpoints or simulator environment.

The old Codex relay has no scene-observer schema and is explicitly rejected in
steady mode. DeepSeek is the supported backend for this implementation.

## Evidence and testing

Observer and selector request/response files are recorded separately. Selector
lineage includes `steady_guard`: proposed/actual ID (identical), active instance,
legal IDs, observed maps, mode and any supported completion. An invalid proposal
is kept in its raw response and failure file; it is never silently overwritten.
`steady_decision_stops.jsonl` is distinct from actual SDK outcomes. The original
action/ACK ledger still contains only instructions followed by verified execution.
Every reset clears instance tracking, maps, active goal and visual/exec histories.
If no legal next target exists after the last supported placement, the stop record
retains that completion, but leaves SDK_outcome null. This policy does not invent
a return-to-origin Subtask or silently continue another target to obtain an SDK
terminal result. A full-task benchmark must report such decision stops separately.

```sh
python3 -E -B online_subtask/tests/run_cpu_tests.py
python3 -B tools/verify_archive.py
```

Fixtures cover illegal first-frame choices, no-jump sequences, category/basket
collisions, non-merging of category names, false completion by a similar instance, drops,
legal completion/switching, missing/uncertain/future evidence, exact retries,
shared deadlines, mode/namespace enforcement, actual runtime-before-infer admission
and real ACK method wiring using mocked policy/transport. They do not measure
real image understanding, new API compatibility or physical Follow ECoT success.
Historical provenance files describe the parent archive. The separate steady
source manifest records all intentional branch adaptations without relabelling
them as bytes from the original deployment.

See [the follow-up review](steady_review.md) for reproduced defects in the initial
steady commit, their regression checks and the remaining validation limits.
