# Follow-up review of the steady implementation

Reviewed candidate: `8b82a0a2859534a00003146bd5e430a46b48ab26`.
This review reproduced three behavioral defects in that initial steady commit.
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
last-goal stop. The complete offline suite passes **71 tests**, including the3
loopback HTTP tests, with0 failures/errors/skips. The retained test receipt records
the command and scope. There were no real API, checkpoint, GPU or native episodes.

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
