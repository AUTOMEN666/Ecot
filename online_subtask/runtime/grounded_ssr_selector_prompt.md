Generate the next short actionable ECoT annotation for classify_objects BEFORE
the action policy is called. Return ONLY this exact public JSON shape:
{"subtask_id":"one allowed ID","reasoning":"English action guidance","action":{"summary":"English next-action summary"}}

Select one exact allowed_candidates ID. Keep the SAME physical object and basket
through approach, alignment, grasp, lift and transport until visual completion.
Use the admitted category/basket rules, not previous plans. An explicit admitted
visual correction is different from changing IDs for a phase. When the last
object has visibly been released, post-placement withdrawal/settling can retain
that object's Subtask; this does not assert SDK success or ask for another pickup.

ACCURACY FIRST. The scene/arm observations and motion_options are fallible visual
evidence, not ground truth. Check them against the actual current wrist/MAIN
images. If the selected instance, current arm, grasp or basket claim conflicts
with visible evidence, do not invent support. Return null for an unsupported
public field and stop. No fallback, semantic redraw or host text repair is allowed.

Reasoning is a public training-style micro-action annotation, not private chain
of thought. In normally15–45 words, state the relevant CURRENT constraint and
the next feasible arm/gripper adjustment. Action Summary is normally6–22 words
describing ONE immediate action and purpose. The exact real training examples
define vocabulary/style, not the object sequence or current facts of this case.
Keep all current facts distinct from intended future actions.

SHORT-HORIZON FORECAST: infer useful intent over the NEXT8–16 control steps
(0.32–0.64s at25Hz). The policy predicts50 but executes8 then observes again.
Do not just narrate a still image. Also do not summarize the entire pickup/carry/
place sequence as if it fits this short window. A correct goal does not establish
readiness for the next manipulation phase. Do not copy a training example's
gripper/contact state into the current observation.

Start Action Summary with ONE permitted imperative verb from motion_options:
Approach, Align, Open, Close, Lift, Carry, Lower, Release, Retract, or Hold.
Identify the acting left/right arm or gripper (both only when both are admitted).
Use that arm's permitted verbs for the exact candidate instance/basket. Avoid a
generic 'Move' or purely descriptive 'The gripper is...' Summary. This is normal
short action-label language, not a request to output an action code or checklist.

Respect unmet prerequisites in BOTH Reasoning and Summary:
* Separated/approaching target + open empty gripper: continue approach/alignment
  with open jaws. A target visible in a wrist image is not ready to grasp.
* Close only when that target is CURRENTLY enclosed between aligned opposing
  fingers at grasp height. An earlier pickup proposal is not such evidence.
* A close attempt is not a secure grasp. Lift/Carry only after current supported
  holding of that same instance is established. Do not promise a lift before it.
  A narrow known-contact transition may admit Close/Hold for the SAME locked
  object while two adjacent actual observations show the same enclosing jaws
  already closing. This does NOT establish holding or secure support. Preserve
  that uncertainty in Reasoning and do not forecast a lift/carry/release yet.
* Carry/align/lower toward the admitted basket while retaining a secure grasp.
  Release only when the actual target basket and placement readiness are visible.
* After visible release: retract/settle the empty arm. Do not describe it as still
  carrying or begin grasping the already placed object without new evidence.

Do not mention a disallowed later phase even as 'then lift', 'after that release'
or a conditional multi-stage plan while still approaching. The public text
conditions a motor policy, so a premature future verb can bias the next8 actions.
For example, predict continued open-jaw approach until readiness changes, rather
than forecasting a grasp/lift that cannot yet be supported. Predictive does NOT
mean moving the action phase ahead of its visual prerequisites.

Use observed relative directions only, without invented metric distances, forces,
hidden object poses or future coordinates. Do not claim the gripper is/keeps
closed when currently open; use an admitted Open/Close transition when needed.
Do not assert planned contact, secured grasp, release or task completion already
happened. Reasoning and Summary must agree on object, arm, basket and phase.
Past Subtask text is not proof of object identity, grasp or completion. In expert
replay mode the recorded motion was NOT controlled by those generated labels.

No current-test future images, actions, GT labels, segment boundaries, remaining
length or SDK outcomes are supplied. Public reasoning comes only from the JSON
content, never private model thinking. No extra output keys, prose or completion
lists. Unsupported null fields stop before policy inference; never guess an ID.
