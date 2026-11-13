Generate an actionable short-term ECoT annotation for classify_objects, before
the action policy is called. Return ONLY this exact public JSON shape:
{"subtask_id":"one allowed ID","reasoning":"English action guidance","action":{"summary":"English next-action summary"}}

Subtask ID must be from allowed_candidates. In locked mode keep that exact ID,
physical object instance and basket until admitted visual completion. Approach,
grasp, lift, transport and alignment are phases of that same goal. Follow the
admitted category/basket mappings. Another similar object is a different instance.

Reasoning is a public training-style action annotation, not private chain of
thought. Use one or two concise action-oriented sentences (normally20–65 words).
Explain the next useful micro-action: which arm, target, relative approach or
alignment, gripper state/transition, contact prerequisite, lift, carry or release
as warranted. Action Summary should name the upcoming action and its purpose,
normally4–25 words, in the style of the supplied real training labels.

Anticipate a short action window of up to16 control steps (~0.64s at25Hz); the
policy still predicts50 and executes8 before reobservation. Choose a feasible
immediate action supporting the admitted Subtask, rather than narrating only the
current image or past motion. A grasp plan can keep the gripper open while
approaching, close after alignment/contact, then lift only when supported. A
placement plan can align over the admitted basket and release once positioned.
Keep current observed facts separate from intended next actions. Do not assert
that planned grasping, placement, release, SDK success or completion already
happened. Reassess after each real observation, including failed grasps and drops.

The reference examples are exact labels from OTHER training observations. They
define language/phase semantics, not current scene facts. Do not copy their object
sequence, category/basket mapping, frame progress or motion into this episode.
The offline teacher could use expert future-action windows; you cannot see the
current test episode's future. Predict an intended action from causal current
three-view RGB, State14, instruction, strictly past MAIN images and acknowledged
executed Subtask history. Prior selected/executed text is not proof of completion.
Prefer wrist evidence for the actually held object and MAIN for scene/baskets.

Use only supported relative motion; do not invent metric distances, coordinates,
forces or hidden object states. If a grounded legal short-term plan cannot be
formed, return null for the unsupported annotation field(s); this stops before
policy inference. Never substitute a new ID, fabricate facts or request a semantic
redraw. Do not add prose, analysis, completion lists or extra output fields.
