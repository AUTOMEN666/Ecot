Observe the CURRENT manipulation state for classify_objects. Report visual facts,
not a future plan and not task success. Use the supplied current MAIN, LEFT wrist
and RIGHT wrist RGB, synchronized State14 and strictly past MAIN images only.
Past selected/executed instructions do not prove contact, a grasp or completion.

Report uncertainty locally. scene_usable is false ONLY if the input views are
unusable/misaligned or the scene cannot be observed at all. An uncertain watch,
black device or basket does not make a clearly visible pen uncertain. Describe
each uncertain object with nullable category and empty/possible descriptor IDs;
do not guess its identity to keep moving. Unknown basket contents block that
basket, not unrelated clear baskets. Unknown material in a gripper MUST be
reported as holding=unknown; never hide it to propose a table pickup.

Distinguish the actual object instance from its description, category and planned
destination. Track stable visual instance IDs O1,O2,... across the supplied
history and scene_registry. Different similar objects are separate instances;
the same object must not receive a new ID just because it is occluded or moved.
Include CURRENTLY visible objects only. Omission is occlusion, not completion.
Robot fingers, wrist housings and attachments are not table objects. Use wrists
to establish what is between the fingers and MAIN to inspect all baskets.

For each object use category=an explicit lowercase singular semantic kind
(pen,watch,camera,toy_car,plush_doll,block,...) or null when uncertain. Color is
not a category. descriptor_ids contains ONLY exact ID KEYS from the supplied
descriptor table, such as "D004" or "D070". It NEVER contains object phrases
like "black and gold pen", candidate IDs like "U0012", or combined ID strings.
Use the text associated with an ID to check compatibility with the
actual visible object, including color; generic words like device do not by
themselves establish a category. Clear identity/category/descriptor confidence
means supported by current visual evidence, not confidence inherited from text.
location is table,held,white,blue,red or unknown. Basket locations mean visibly
INSIDE the basket, never merely hovering above it. released=true requires the
object to be disengaged from the fingers. Do not call an empty closed gripper a
grasp. A dropped/regrasped object is unfinished. Report current facts ONLY: the
program records their actual frame and accumulates later visual confirmation.
DO NOT return evidence_frames, guessed timestamps or retrospective frame arrays.

Each arm reports its actual relation to the observed interaction_instance_id:
* separated: object is outside the closing corridor, with a visible gap;
* approaching: reaching toward it, not yet aligned around it;
* aligned: nearly aligned, but object is not yet clearly enclosed by both jaws;
* enclosing: target is visibly BETWEEN the two opposing finger pads, at their
  grasp height, so closing can establish contact without another approach;
* contact: supported physical contact, not merely overlapping in one projection;
* held: an identified object actually supported in the gripper;
* over_basket: actual held object is over the identified observed_basket;
* released/retreat: current empty-gripper post-placement/withdrawal state;
* unknown: current view cannot establish this relation.
Being in the wrist image, near its center, or assigned as the Subtask is NOT
enclosing/contact. At early approach a pen can be large/near in the wrist view
while still well outside the jaws. Do not jump to enclosing on that basis.

object_secured=yes requires clear identified grasp support: the object is between
closed fingers and moves/supports with the gripper, using temporal evidence when
available. Closed jaws alone, an attempted close, or a selected pickup is not
secure grasp evidence. Use no/unknown when not established. gripper_state is the
CURRENT visual state, not a future command. Do not infer Cartesian positions,
distances or contact solely from State14 joint values.
Keep material identity and grasp security separate: a clearly visible pen
between enclosing, closing fingers can remain identifiable while supported
holding is not yet established. Report that same interaction_instance_id and
the actual closing trend; do not rename it or claim object_secured=yes. If
material inside a hand truly cannot be identified, keep holding=unknown.

release_ready=yes requires secure holding over the correct observed basket
opening at a supported placement/release position. Being carried toward a basket
is not release readiness. observed_basket is what is visibly under/at the current
gripper, not the basket named by a plan. motion_trend describes observed causal
motion, never unseen future movement. Empty hands at rest remain empty.

If new clear visual evidence corrects a previously wrong object category or
description, use corrections with the exact old values, a short VISUAL reason,
and at least two CURRENT cameras corroborating the same physical instance.
Do not rename or merge instances to avoid a contradiction. goal_correction is
normally null. It may explicitly identify a current actual held object different
from the locked target, or a reliable released-containment/category mapping that
contradicts the locked basket. A different phase, another similar object placed,
motion toward another empty basket or a proposed plan is NOT such evidence.
An old goal interrupted by correction is not thereby completed.

Return JSON ONLY with exactly these top-level fields:
{
 "scene_usable":true,
 "baskets":{"white":{"visibility":"clear","unknown_contents":false},"blue":{"visibility":"clear","unknown_contents":false},"red":{"visibility":"clear","unknown_contents":false}},
 "objects":[{"instance_id":"O1","identity_confidence":"uncertain","category":null,"category_confidence":"uncertain","descriptor_ids":[],"descriptor_confidence":"uncertain","location":"unknown","released":null}],
 "corrections":[],
 "goal_correction":null,
 "arms":{
  "left":{"holding":"empty","held_instance_id":null,"gripper_state":"open","interaction_instance_id":null,"relation":"unknown","object_secured":"no","observed_basket":null,"release_ready":"no","visibility":"clear","motion_trend":"unknown"},
  "right":{"holding":"empty","held_instance_id":null,"gripper_state":"open","interaction_instance_id":null,"relation":"unknown","object_secured":"no","observed_basket":null,"release_ready":"no","visibility":"clear","motion_trend":"unknown"}
 }
}
The displayed values are SHAPE examples, not current scene facts. Report all
visible objects and the actual state of both arms. Enums: confidence clear|uncertain;
For a clear matching description, an example ID array is ["D004"], NEVER
["black-and-gold pen"]. If no supplied ID matches, use [] and mark descriptor
confidence uncertain; do not invent a new ID or replace the ID with prose.
visibility clear|unknown for baskets, clear|occluded for arms; gripper_state
open|closed|transitioning|unknown; holding empty|identified|unknown;
object_secured/release_ready yes|no|unknown; motion_trend
approach|align|close|lift|transport|lower|release|retract|stationary|unknown.
corrections items have exactly instance_id,previous_category,previous_descriptor_ids,
reason,evidence_cameras. goal_correction when present has previous_subtask_id,
actual_instance_id,reason,evidence_cameras and optionally corrected_basket.
The response ROOT has exactly SIX keys: scene_usable, baskets, objects,
corrections, goal_correction, arms. No seventh key is permitted. In particular,
do not copy API/request envelope fields into the answer: type, response_format,
json_object, messages, scene_registry and current_report_frame are NOT response
fields. Output the scene object itself, without an API envelope or schema wrapper.
Before returning, check these six root keys, every descriptor ID and both arm
records against the contract above. Never return Subtask, Reasoning, Summary,
SDK labels or extra fields here.


CATEGORY TOKEN CONTRACT — revision category_tokens_v2.
This contract REPLACES the earlier open-ended category examples above. The ONLY
non-null category values, copied exactly, are:
pen, watch, camera, plush_toy, toy_vehicle, block, box, ring, bracelet, pouch,
strap, tool, bottle_cap, can, tracker.
All fifteen tokens belong to a fixed ANNOTATOR VOCABULARY, not a complete SDK
or real-world taxonomy and not truth about this scene. The first five broad
semantic kinds do not exhaust the vocabulary. Do not drop the other ten kinds.

The category field is ONE semantic token or null, never a descriptor, a color,
a material phrase or an object name sentence. Thus "black-and-brown plush doll"
is not a category token. The older toy_car/plush_doll examples above are not
permitted output tokens in this revision. Use a permitted semantic kind only
when the current visible object supports it; otherwise use category=null and
category_confidence=uncertain. Do not force a nearest token for an unfamiliar,
occluded, ambiguous or vocabulary-external object. Never pair category=null
with category_confidence=clear.

Choose category independently from the actual current MAIN/wrist evidence and
causal visual history. A selected descriptor_id, its text, a prior Subtask or a
category word in old text is NOT proof of category. Descriptor IDs still convey
the exact candidate wording; do not rewrite or merge O instances, D descriptors
or U candidate IDs to fit a token. The same category can contain multiple
physical instances. This does not mean one instance completed another's goal.

Use one stable token for the SAME physical instance while visual category stays
the same; motion phase, grasp, transport and placement do not change category.
A clear visual category correction still requires the exact explicit correction
and corroborating current cameras already specified above. previous_category
in each correction must also be one exact token from this vocabulary or null.
Do not carry invalid prior output into a corrected field by guessing its token.

Do not infer category-to-basket mappings from this token list, descriptor text,
SDK labels or old plans. The existing released-containment/visual mapping rules
remain. Null/uncertainty is local; it must not erase unrelated visible facts.
No host normalization, removed fields, replacement answer or semantic retry is
part of this contract. Generate the exact category token yourself only when
visually supported. Return the same six-field current-observation object.


CURRENT HOLDING/LOCATION SELF-CONSISTENCY CONTRACT.
Clear object identity is NOT proof of holding. Report holding=identified only
when an identifiable object is actually supported in that current hand; its
held_instance_id and interaction_instance_id must be the SAME visible instance,
and that object's current location must be held. Do not call a recognizable
object held merely because it is enclosed between fingers or a close is planned.

A known target still supported on the table during enclosure/contact remains
location=table. If no object is yet supported in the hand, report holding=empty,
held_instance_id=null and retain the actual interaction_instance_id/relation.
If actual material in the hand truly cannot be identified/observed, retain
holding=unknown and held_instance_id=null; do not hide uncertainty to advance.
Never change an object's observed table location to held just to make a hand
report or old instruction consistent. The CURRENT physical evidence decides.

Identified holding and secure support are distinct. identified+object_secured=no
is permitted when the same identifiable object is actually supported in the
hand but secure retention is not yet established. Do not forbid or falsely
upgrade that legitimate contact phase. Keep the observed open/closed/transitioning
state and no/unknown security as they actually appear. Without secure support,
do not forecast Lift/Carry as the next action or a promised subsequent phase.
Continue only the currently supported immediate contact/alignment action;
future policy commands or selected text never prove a completed grasp.


BOUNDED RELEASE-IN-PROGRESS OBSERVATION CONTRACT — revision release_progress_v1.
This narrowly distinguishes an object STILL visibly supported while the jaws
are opening for release from an object ALREADY disengaged from an empty hand.
Preserve every existing current-observation field and report what is visible;
do not output a phase flag, inferred timestamps or host authorization.

An identifiable target can remain holding=identified, location=held and
released=false while the gripper is visibly transitioning OPEN during a
release over the currently observed basket. Use motion_trend=release only for
that visible opening/release trend. Keep the SAME held_instance_id and
interaction_instance_id. If present visual support still establishes that the
object remains supported, object_secured=yes describes that CURRENT support
only; it does NOT assert closed fingers, authorize lifting/carrying or promise
that the grasp will remain secure as opening continues. This is the sole
narrow exception to the earlier closed-fingers wording. release_ready=yes
still requires the currently visible supported object over the basket opening
at a suitable release position, not a plan to reach it.

Never change gripper_state=transitioning to closed just to satisfy old text.
Never call an open gripper, an unknown/occluded grasp, a falling object or an
unsupported object secure. An object already disengaged from the fingers is
released; report its actual current location and the hand's empty state. A
drop to the table remains unfinished and is not basket containment. If the
object's current identity or support is unclear, retain the appropriate
unknown/no facts rather than hiding uncertainty.

Only the program may bind adjacent actual observations, the still-active goal
and a finite opening-window budget. Previous selected/executed release text is
NOT evidence that the fingers opened or the object was released. Do not infer
completion from this exception, invent future motion, return extra fields or
change instance/category/basket to obtain an admission.
