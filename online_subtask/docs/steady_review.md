# Follow-up review of the steady implementation

Initially reviewed candidate: `8b82a0a2859534a00003146bd5e430a46b48ab26`.
The first review reproduced three behavioral defects in that initial steady commit
and a missing error-evidence artifact.
The original `ecot_online_generation_subtask` branch is unchanged.

| Finding | Reproduced behavior before correction | Correction |
| --- | --- | --- |
| Host category merging | Reports labelled toy_car and toy_bus were both rewritten to toy_vehicle, permitting bus-to-red after car-to-red. | Preserve exact observed category keys. Lexical families only check obvious descriptor contradictions and never determine basket mapping keys. |
| Pre-drop placement reuse | A placement frame from before an observed drop and a new current placement could satisfy the two-frame completion threshold. | Track the active instance's latest unplaced witness. Require two placement witnesses AFTER that barrier. The observer schema also reports intervening unplaced_frames, so a drop within one observation interval invalidates older placement evidence. |
| Lost final-goal completion | With the only goal visibly completed and no next legal target, an exception occurred before completion was committed, leaving the old active goal in state. | Commit accepted visual state and completion before recording a no_legal_target stop. Do not call the selector or manufacture an SDK result. |
| Missing observer error evidence | Observer HTTP rejection/partial-response bodies were omitted from the error artifacts. | Retain available raw error/partial bodies and SHA256, using the existing retry classification. Semantic errors remain nonretryable. |

Regression checks cover each finding, including contradictory placed/unplaced
witnesses, distinct writing-related categories, two different toy categories,
drops across calls and within one report, and the real provider/ACK path for the
last-goal stop. That review's complete offline suite passed **71 tests**, including the3
loopback HTTP tests, with0 failures/errors/skips. The retained test receipt records
the command and scope. There were no real API, checkpoint, GPU or native episodes.

## Second follow-up review

Reviewed candidate: `51b81245a5c3bb4946c973017643e6c614907ba0`.
Controlled worker threads reproduced a reset during an observer wait: reset
changed the model/provider episode, the old request failed its causal-state check,
and its exception handler then latched `stopped` on the NEW episode. No policy
call was required to reproduce this state corruption. A guard fixture also
admitted a new target into an assigned basket whose current contents were unknown.

| Finding | Correction and regression evidence |
| --- | --- |
| Model lifecycle RPC overlap | A nonblocking per-model gate covers reset, observation updates, captures, ACK, injection reads, terminal records and the entire action call. Overlap is rejected before mutation. Tests hold the observer and, separately, policy inference; reset/update/duplicate actions cannot change the active call's episode, frame, inputs or RNG. |
| Direct provider overlap and delayed old calls | A separate provider gate covers its public lifecycle methods. Busy rejection occurs outside the stop handler and starts no additional API attempt. Old-episode generation is rejected before it can latch the new episode. Tests verify the original request still completes, a later reset works, and an inference exception releases the model gate. |
| Unknown contents of an assigned destination | Require current clear basket-content observation for ALL new goals. A test completes one pen goal while red contents are unknown and rejects the next pen-to-red goal, preserving the first completion. A separate test confirms that occlusion alone does not change an unfinished locked goal. |

That review's complete suite passed **78 tests**, including all3 localhost HTTP tests,
with0 failures/errors/skips. Seven added checks cover the two visibility cases
and five lifecycle cases. The test receipt lists the modules and offline scope.
No actual API, weights, GPU or native simulator was used. A busy RPC is a rejected
caller operation, not a generated decision, execution record or SDK outcome.

## Final requirement review

Reviewed candidate: `46c59ecb80c046f1434d7b3606a52189824b6970`.
The review checked the task requirements across the guard, provider, actual model
adapter, ACK persistence, prompts and delivered source package.

| Finding | Correction and evidence |
| --- | --- |
| A new goal could ignore a held object | Restrict new targets to held objects when any are observed. An unnameable held object stops admission; two identified held objects can supply targets. Existing unfinished goals remain locked. |
| The same descriptor could be renamed into a new category on another instance | Keep an episode-local descriptor/category ledger. Contradictions stop atomically, including when the original object is occluded. Exact category keys remain distinct; there is no host semantic merging. |
| A new instance ID could bypass an occluded known object's identity | Persist the ambiguity of newly named targets sharing a descriptor with an absent known instance. Repetition and intervening goals cannot clear it; explicit co-visibility can. Previously distinguished IDs remain usable, and explicitly separate visible instances can still use the same catalogue ID. |
| A regrasp inside a basket could leave stale completion | Any current unreleased/regrasped observation clears that instance's completion, including inside a basket. It does not switch another active goal or erase the established category mapping. |
| NumPy input buffers and policy transforms could change stored observations | Copy the admitted observation/clock at update and isolate policy input transforms. Tests mutate caller arrays and policy inputs and check the stored/current evidence remains unchanged. Full-scene digests also bind injection authorization. |
| ACK state could advance before execution evidence was safely logged | Validate on a private ledger. Write a complete temporary log, sync it, replace the old log atomically and sync the directory before publishing the ledger. Persistence faults stop the episode; tests preserve a previous committed prefix and admit no new history. Invalid ACK validation can still be corrected without a semantic redraw. |
| Observer and stop artifacts could misidentify the evidence | Observer records now identify the actual steady protocol, prompt and registry hashes. A failed new observation links its observer receipt, sets current decision to null and preserves the last admitted decision separately. Stopped providers reject old injection authorizations. |

All **102 test methods** pass, including3 localhost transport tests, with0
failures/errors/skips. The24 new methods include24 category/basket/visibility
combinations and all287 alternative IDs while locked. A complete mocked episode
executes and ACKs six policy chunks across three objects/categories: each ID is
kept through grasp, two supported placements permit each switch, and final
completion is preserved separately from the SDK result. These are software
behavior checks; they are not50 real cases or a visual recognition accuracy score.

## Remaining limits

- Visual object identity, category and placement are still model observations.
  Internally consistent but visually wrong evidence can pass deterministic checks.
  These tests do not establish recognition quality or Follow ECoT task success.
- Ambiguous instances, uncertain evidence and conflicting contents deliberately
  stop admission. A stopped case is not an SDK success/failure and must remain
  visible in evaluation totals instead of being silently omitted or retried.
- When no legal next target exists, this diagnostic stops; it does not synthesize
  a return-to-origin command. A visual last-goal completion and an SDK whole-task
  terminal outcome remain separate measurements.
- The two-call observer/selector path has not been accepted against a real API or
  native simulator. It also still requires the genuine external checkpoint,
  normalizer, source/binding evidence and a separately adapted evaluation runner.

Passing this review means the reproduced defects were corrected and regression
fixtures pass. It does not mean the online system has been proven problem-free.
