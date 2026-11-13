Select exactly ONE Subtask ID for "Sort the objects by category into the three
baskets." The supplied allowed_candidates are a PROGRAM-ENFORCED subset of the
original 288 catalogue entries. Their text is unchanged. Return ONLY JSON
{"subtask_id":"one allowed ID"}. No Summary, Reasoning, extra text or fields.

When mode=locked, the only allowed ID is the active, unfinished goal. Return that
exact ID. Approach, grasp, lift, transport, alignment, occlusion and viewpoint
changes do not authorize another object or basket. Do not rephrase the goal.

When mode=new_goal, select an observed unfinished physical instance from the
allowed candidates. Category/basket assignments come from observed placements:
same category uses one basket and different categories do not share a basket.
Object color is not a category. An old plan or executed instruction is not proof
of placement. The actual-execution ledger is instruction history, not ground truth.
The visual observer is fallible; the program's admission does not prove correct
recognition. Do not use an unlisted candidate even if another seems plausible.
If you cannot obey this contract, do not invent a new ID or replacement answer.
