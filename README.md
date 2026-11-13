# ECoT — classify_objects

Preserved implementations and task-specific development branches.

## Task scope and branches

This implementation targets **`classify_objects`**: the sorting instruction,
fixed288 Subtask catalogue, object descriptors, basket rules and policy bindings
are specific to this task. Other tasks require their own catalogue, constraints
and integration evidence.

| Branch | Purpose |
|---|---|
| [ecot_online_generation_subtask_classify_objects](https://github.com/AUTOMEN666/Ecot/tree/ecot_online_generation_subtask_classify_objects) | Original online Subtask implementation |
| [ecot_online_generation_subtask_steady_classify_objects](https://github.com/AUTOMEN666/Ecot/tree/ecot_online_generation_subtask_steady_classify_objects) | Subtask implementation with steady goal and category/basket checks |
| [ecot_online_generation_subtask_reasoning_and_action_summary_classify_objects](https://github.com/AUTOMEN666/Ecot/tree/ecot_online_generation_subtask_reasoning_and_action_summary_classify_objects) | Branch prepared for prospective Subtask, Reasoning and Action Summary work |

The `_classify_objects` suffix identifies the task. The third branch currently
inherits the steady Subtask-only runtime; its name does not claim that SSR
generation, SSR policy injection or a new online evaluation has been completed.
Historical protocol IDs, archived ZIP names, source commits and validation
receipts keep their original identities.

