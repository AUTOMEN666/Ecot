# Prospective SSR branch for classify_objects

Branch: `ecot_online_generation_subtask_reasoning_and_action_summary_classify_objects`.
Parent: `ecot_online_generation_subtask_steady_classify_objects` at `28fc9730adf5222382f4aa177916217062a1e7d5`.

## Current status

This naming/documentation change prepares the requested branch. Its executable
runtime is still the inherited steady Subtask-only implementation. SSR generation,
three-field policy conditioning and a new real API/GPU/native evaluation are
pending implementation and acceptance.

## Intended generation contract

- Retain the classify_objects instruction, exact288 Subtask ID/text lookup and
  steady instance-goal/category/basket admission.
- Generate public Subtask, Reasoning and Action Summary annotations using actual
  training-label semantics, source-bound examples and the training serializer.
- Reasoning describes grounded short-term action intent: which arm and object,
  approach/alignment, gripper transition, lift/transport/release as appropriate.
  Action Summary concisely names the next action and its purpose.
- Use causal current observations and same-episode history to anticipate the next
  action. Separate observed facts from planned outcomes; future test images,
  expert actions, GT labels and SDK success truth are not generator inputs.
- Preserve current3RGB/State14/instruction plus up to93 strictly past MAIN images
  (96images total), acknowledged actual execution history and frozen waits.
- Generate and validate the three fields before policy inference; commit actual
  injection history only after verified execution ACK.

## Policy/checkpoint boundary

The inherited server currently validates `A_subtask_online` EMA5000 and injects
only Subtask. Three-field testing needs a separately verified existing
summary_subtask_reasoning (SSR) checkpoint/export, normalizer, training identity
and exact prompt serializer. Selecting that checkpoint and implementing its
binding are pending; the branch name is not evidence of SSR support.

The task-specific suffix does not rename historical protocol IDs, scientific
configuration, source hashes, API budgets or old experiment/result namespaces.
