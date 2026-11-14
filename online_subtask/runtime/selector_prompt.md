Select exactly one object-centered Subtask ID from the supplied Stack Bowls
training-text catalogue for the complete Instruction:
Stack the three bowls together.

INPUT EVIDENCE
You receive current main, left-wrist and right-wrist RGB views with measured
State14, plus up to93 uniformly sampled strictly past MAIN-camera RGB views
and their State14 (at most96 images total). Only the current observation has
wrist views. All images are current or strictly past within this episode.
There are no future images, expert actions, test labels or completion truth.

The chronological executed Subtask history contains exact IDs, original texts,
observation frames and actual executed action ranges, committed only after
execution acknowledgement. Failed requests and unexecuted proposals are absent.
An earlier instruction is NOT evidence that its goal was achieved. State14
contains measured robot joints and grippers, not object identity, contact truth
or Cartesian motion directions. Do not infer gripper sign conventions.

STACKING TASK
Identify the three task bowls from the observations. Random-setting bowl models
may differ in appearance from training, and there are distracting objects.
Do not manipulate a distractor or substitute a robot part for a bowl. Use main
view for bowl/stack relationships and wrists for grasp, support and release
evidence. Use an unchanged catalogue sentence appropriate to this actual scene;
do not invent a color or geometric relation just to fit a sentence.

Keep one physical bowl instance, intended support bowl or existing stack, and
the selected ID/text while that Subtask remains unfinished. Temporary occlusion,
viewpoint changes, arm motion or eight elapsed actions do not justify switching
goals or rephrasing. A left/right/center description anchors the instance when
selected; if that bowl is subsequently moved, do not silently transfer the goal
to another bowl now occupying that position. In particular, do not keep selecting
the other currently rightmost bowl after the selected rightmost bowl is grasped.

COMPLETION IS SPECIFIC TO THE ORIGINAL SUBTASK TEXT
- A pickup-only instruction such as 'Pick up the rightmost bowl.' finishes when
  that SAME bowl is visibly secured and lifted clear of its support. Approaching,
  closing the gripper, contact or an ambiguous hidden grasp alone is insufficient.
  After a supported pickup, select a compatible placement/stacking instruction
  for the held bowl. Do not wait for the entire three-bowl task to finish before
  leaving a pickup-only Subtask.
- A placement/stacking instruction finishes when that SAME bowl is aligned with
  the intended support bowl/stack, visibly supported or nested there, and the
  gripper has released/disengaged without the bowl falling away. Hovering above
  a bowl, carrying it or opening away from the intended support is insufficient.
- A combined pickup-and-stack sentence requires both phases; a successful pickup
  alone does not complete it. A retreat/reset sentence concerns withdrawal from
  the completed manipulation, not a new grasp.

During an unfinished Subtask, preserve its physical goal through approach,
grasping, lifting, carrying, alignment and release as applicable to that text.
Do not generate a new micro-motion instruction every replan. Maintain the
existing support stack when placing the next bowl; do not break a visibly valid
stack merely to change ordering. A two-bowl stack does not complete a three-bowl
task. Use actual visible support/release and consistent before-and-after evidence,
not elapsed time or text history, to decide whether to advance.

Once all three bowls are visibly stacked, preserve that stack and select the
supported release-and-retract instruction while withdrawing the arms. Do not
restart a pickup or disturb the completed stack merely because the simulator
has not terminated yet. Native success also requires the robot to return to
its original posture; keep any already released gripper clear of the stack.
Use only visible evidence and measured past/current robot state, not hidden
scene labels, to guide withdrawal. The simulator alone decides final success.

If a bowl slips, is dropped or a placement fails, do not count it as completed.
Continue or select a supported recovery instruction for that same goal. Clear
visual evidence may correct a previously mistaken instance/support identification;
consistency never requires preserving a known error. A correction or ID switch
does not retroactively mark the preceding goal complete. Identical-looking bowls
remain distinct physical instances and may legitimately share catalogue wording.

The catalogue is a vocabulary of original training Subtasks, not a test plan or
test ground truth. Do not output Summary, Reasoning, coordinates, future actions,
hidden scene labels or a success/termination decision. Episode success and
termination are exclusively the native simulator's responsibility.

OUTPUT CONTRACT
Return ONLY JSON {"subtask_id":"one listed ID"}.
Select one supported listed ID and copy no new text into the response. No extra
keys, null, NO_MATCH, Markdown or unlisted IDs. The host resolves the ID to its
exact original training sentence without rewriting, semantic retries or fallback.
