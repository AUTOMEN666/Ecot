# Online protocol: mainhist93_tok65536_v1

The source of truth is the preserved runtime and complete selector prompt.
This document explains that contract; it does not substitute a shortened prompt.

## Inputs and causal visual sampling

The complete instruction is:

```text
Sort the objects by category into the three baskets.
```

At replanning frames0,8,16,…, the selector receives current `cam_high`,
`cam_left_wrist` and `cam_right_wrist` RGB images from the same actual SDK
observation, synchronized measured State14, and the instruction. Image
validation preserves the collector's640×480 lossless RGB encoding and binds
decoded pixels to their hashes.

State14 order is left-arm joints0–5, left gripper measurement, right-arm
joints0–5, right gripper measurement. These are native SDK measurements; the
protocol does not infer Cartesian coordinates, units or gripper-sign semantics.
Each historical image is paired with its own synchronized State14 and frame.

The collector retains every source frame `floor(2.5*k)`, k=0,1,…, on the25Hz
execution frame axis, giving approximately10Hz history. Only frames strictly
earlier than the current query may enter visual history. If there are N past
points:

- N≤93: use all points.
- N>93: use point indices `i*(N-1)//92` for i=0,…,92, preserving the earliest
  and most recent past observations without duplicates.

Only the main image from each selected past point enters the API request.
Current three-view images follow the past images, so total image count is
`min(N,93)+3≤96`. Past wrist images remain local capture/audit material and are
not sent. Missing source captures, future frames and cross-episode observations
are rejected. This is whole-episode sampling, not a moving window of93 recent
images or sampling only at replanning frames.

## Executed Subtask history

Every committed record contains exactly:

```json
{
  "observation_frame": 0,
  "subtask_id": "U0001",
  "subtask": "Pick up the beige plush doll in blue overalls and place it into the white basket.",
  "executed_action_start": 1,
  "executed_action_end": 8
}
```

The example is a catalogue entry, not a scene label or evidence that this goal
was completed. Action ranges are one-based and inclusive. At request frame8k,
exactly k preceding full8-action records must cover actions1…8k in the same
episode. The first request has empty history; the latest record is identified
explicitly. There is no truncation of the earlier executed text history.

`ExecutedSubtaskHistory.commit` checks the exact injection, ID/text lookup,
composed prompt, returned commands, episode/frame order and per-action physical
clocks before appending. A successful generation alone does not commit a
record. Failed requests, repeated transport attempts and unexecuted proposals
add no history. Duplicate, missing and inconsistent ACKs are rejected.

Under the SDK1100-action limit, the last possible request is frame1096, with137
past records. Its final four executed actions may produce committed record138,
ending at action1100. There is no next request after that terminal partial
chunk. The137 limit on history supplied to a query is therefore compatible with
a terminal ledger of138 records. Native termination, and explicitly recorded
engineering termination, may ACK a shorter final chunk; they do not authorize
continuing a normal query after a partial chunk.

Text history records past instructions and actual action ranges. It is not
ground-truth object identity, correct placement or goal completion.

## Selection and continuity

The full prompt keeps the same ID for the same physical object instance and
destination while observations support continuing that goal. Approach, grasp,
carry, occlusion or viewpoint change alone does not justify changing wording.
Clear contradictory visual evidence permits correction. Placement completion
requires visible release/placement or consistent before-and-after evidence;
elapsed actions and an earlier instruction are insufficient. Identical-looking
different instances may legitimately share one catalogue ID. Category-to-basket
consistency also depends on non-conflicting observed placements.

The fixed catalogue contains96 known training lexical descriptors paired with
three basket colors, giving288 candidates. Some pairings are derived text, not
verbatim training sentences. This is neither an open-vocabulary output space
nor test ground truth. Catalogue SHA256:

```text
97734cb3db4e92fcf62beec96d635896d09d8f23c99c646170bf7736f8b48600
```

The only accepted public output is a JSON object containing the sole key
`subtask_id`, with one listed string ID. Duplicate/extra fields, unknown IDs,
null, `NO_MATCH`, additional text or a new Subtask sentence are rejected. The
program copies the selected catalogue text unchanged. It performs no ID repair,
human substitution, semantic redraw or fallback. A valid ID does not establish
semantic correctness.

## Generation, action and waiting

The active path is `model_server.create_model → RuntimeModel.get_action →
MultiviewHistoryProvider.generate → multiview_protocol.build_request/resolve`.
The selected text is composed as
`Instruction + ' Current subtask: ' + Subtask` before π0.5 inference. The policy
receives current three-view RGB, State14 and that prompt; it predicts50 commands
and the SDK executes at most8 before acknowledgement and replanning.

DeepSeek uses `deepseek-v4-1-flash-261017`, max_tokens65536, temperature0,
thinking enabled/high and JSON object output. The HTTP worker has a600-second
wall-clock limit and is killed/reaped before a retry. At most3 total attempts
use the identical canonical request bytes, episode, frame and history within
the1850-second retry admission budget. Only the explicitly recognized transient
transport/429/5xx conditions, or a length-limited response with empty public
content, are retryable. A semantically invalid public answer is not redrawn.
Policy RPC and ping limits are2100 seconds.

The original EMA5000 policy uses flow10, noise42 and25Hz control. One action
advances10 physics steps at0.004 seconds each. The original SDK1100-action cap
is44 seconds of simulated control, with its original success/termination
checks. Model/API, capture and ACK waits freeze simulation; robot, object and
clock readbacks check that no simulation step advanced during those waits.
