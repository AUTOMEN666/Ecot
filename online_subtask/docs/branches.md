# classify_objects branch names

## Task scope and branches

This implementation targets **`classify_objects`**: the sorting instruction,
fixed288 Subtask catalogue, object descriptors, basket rules and policy bindings
are specific to this task. Other tasks require their own catalogue, constraints
and integration evidence.

| Branch | Purpose |
|---|---|
| [ecot_online_generation_subtask_classify_objects](https://github.com/AUTOMEN666/Ecot/tree/ecot_online_generation_subtask_classify_objects) | Original online Subtask implementation |
| [ecot_online_generation_subtask_steady_classify_objects](https://github.com/AUTOMEN666/Ecot/tree/ecot_online_generation_subtask_steady_classify_objects) | Subtask implementation with steady goal and category/basket checks |
| [ecot_online_generation_subtask_reasoning_and_action_summary_classify_objects](https://github.com/AUTOMEN666/Ecot/tree/ecot_online_generation_subtask_reasoning_and_action_summary_classify_objects) | Prospective SSR generation and noDropout three-field policy conditioning |

The `_classify_objects` suffix identifies the task. The third branch implements
public SSR generation and noDropout three-field policy conditioning; empirical
API/GPU/native semantic-quality acceptance is a separate step.
Historical protocol IDs, archived ZIP names, source commits and validation
receipts keep their original identities.

Current Git references replace the former unsuffixed names. The original Subtask source commit remains `8e4f340940c56fbbe5c30e0ee578d6e5b8678321`; the steady runtime source before this documentation update remains `19046d3b9c0dea754c1f3937cc6d4ca698e9d0ea`.
