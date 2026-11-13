"""Admit immediate actions from fallible CURRENT visual facts, never expert future.

These checks cannot certify perception or arbitrary English. They bind a public
imperative Summary to the observed arm/phase and reject explicit contradictions;
independent trajectory/video review remains necessary. No output is rewritten.
"""
from copy import deepcopy
import re

from subtask_guard import SubtaskConstraintError,lexical_family

ARM_FIELDS={'holding','held_instance_id','gripper_state','interaction_instance_id',
    'relation','object_secured','observed_basket','release_ready','visibility','motion_trend'}
RELATIONS={'separated','approaching','aligned','enclosing','contact','held','over_basket','released','retreat','unknown'}
ACTIONS={'approach','align','open','close','lift','carry','lower','release','retract','hold'}
BASKETS={'white','blue','red'}

def require(condition,message):
    if not condition:raise SubtaskConstraintError(message)

def validate_arms(arms,objects):
    require(type(arms)is dict and set(arms)=={'left','right'},'Both current arm observations required')
    ids={o['instance_id']:o for o in objects}
    for name,arm in arms.items():
        require(type(arm)is dict and set(arm)==ARM_FIELDS,'Exact current arm/phase fields required')
        require(arm['holding']in{'empty','identified','unknown'}and arm['gripper_state']in{'open','closed','transitioning','unknown'},'Unknown holding/gripper enum')
        require(arm['relation']in RELATIONS and arm['object_secured']in{'yes','no','unknown'}
            and arm['release_ready']in{'yes','no','unknown'}and arm['visibility']in{'clear','occluded'},'Unknown current motion evidence enum')
        require(arm['motion_trend']in{'approach','align','close','lift','transport','lower','release','retract','stationary','unknown'},'Current observed trend required')
        require(arm['observed_basket']is None or arm['observed_basket']in BASKETS,'Only actually observed basket color')
        for key in['held_instance_id','interaction_instance_id']:
            require(arm[key]is None or type(arm[key])is str and arm[key]in ids,'Arm must reference a current observed instance')
        if arm['holding']=='identified':
            require(arm['held_instance_id']is not None and arm['interaction_instance_id']==arm['held_instance_id'],
                'Identified grasp must bind the actually held current instance')
            require(ids[arm['held_instance_id']]['location']=='held','Held arm/object observations contradict')
        else:require(arm['held_instance_id']is None,'No fabricated held identity for empty/unknown gripper')
        if arm['object_secured']=='yes':
            require(arm['holding']=='identified'and arm['gripper_state']=='closed'and arm['visibility']=='clear',
                'Secure grasp needs identified object, closed gripper and clear visual support')
        if arm['release_ready']=='yes':
            require(arm['holding']=='identified'and arm['object_secured']=='yes'
                and arm['relation']=='over_basket'and arm['observed_basket']in BASKETS,
                'Release readiness requires current secure grasp over a visibly identified basket')
    return deepcopy(arms)

def action_envelopes(evidence,target,*,contact_transition=None):
    """Each arm's permitted immediate action verbs for this exact physical goal."""
    objects={o['instance_id']:o for o in evidence['objects']}
    arms=validate_arms(evidence['arms'],evidence['objects'])
    ident,basket=target['instance_id'],target['basket_color'];obj=objects.get(ident)
    result={}
    for name,arm in arms.items():
        actions=set();why=[]
        matches=arm['interaction_instance_id']==ident
        if (contact_transition is not None and contact_transition.get('arm')==name
                and contact_transition.get('instance_id')==ident):
            require(obj is not None and obj['location']=='table' and obj['released']is not True
                and arm['holding']=='unknown' and arm['held_instance_id']is None
                and arm['visibility']=='clear' and matches and arm['relation']=='enclosing'
                and arm['gripper_state']=='transitioning' and arm['motion_trend']=='close'
                and arm['object_secured']=='no' and arm['release_ready']=='no'
                and contact_transition.get('allowed_verbs')==['close','hold']
                and contact_transition.get('holding_identity_or_grasp_support_promoted')is False
                and contact_transition.get('new_goal_or_completion_admitted')is False
                and contact_transition.get('uncertainty_windows')in(1,2),
                'Known contact-transition evidence differs from locked visual admission')
            actions={'close','hold'};why=['same known target visibly enclosed across adjacent actual reports; continue existing closure or hold alignment, never infer secure grasp']
        elif contact_transition is not None:
            actions={'hold'};why=['the other arm must not initiate a competing pickup during the known closing contact transition']
        elif arm['holding']=='unknown' or arm['visibility']!='clear':
            # A grounded stationary instruction may be generated, never a host
            # fallback. Repeated holds are separately a quality/progress failure.
            actions={'hold'};why=['current interaction occluded/unknown; no new contact/release claim']
        elif arm['holding']=='identified':
            if arm['held_instance_id']!=ident:
                actions={'hold'};why=['this arm holds a DIFFERENT observed instance']
            elif arm['object_secured']!='yes':
                actions={'hold'};why=['grasp identity known but secure support not yet established']
                if arm['gripper_state']in{'open','transitioning'}and arm['relation']in{'enclosing','contact','held'}:
                    actions.add('close');why.append('continue closing around the already enclosed target; no lifting yet')
            elif arm['relation']=='over_basket'and arm['observed_basket']==basket:
                actions={'align','lower','hold'};why=['securely held target above the ADMITTED basket']
                if arm['release_ready']=='yes':actions.add('release');why.append('placement/release prerequisites currently visible')
            else:
                actions={'lift','carry','hold'};why=['securely grasped actual target; retain grip while lifting/transporting']
        elif obj is not None and obj['location']in BASKETS and obj.get('released')is True:
            actions={'retract','hold'};why=['target visibly released; empty-arm withdrawal/settling, not new pickup']
        elif obj is not None and obj['location']=='table':
            if arm['relation']in{'released','retreat'}and not matches:
                actions={'retract','hold'};why=['empty arm is still clearing a prior interaction']
            elif arm['gripper_state']=='open':
                actions={'approach','align'};why=['table target, open/empty gripper; maintain open jaws while reaching/alignment']
                if matches and arm['relation']in{'enclosing','contact'}:
                    actions={'close','align','hold'};why=['target CURRENTLY enclosed between aligned jaws; closing can establish contact']
            elif arm['gripper_state']=='closed':
                actions={'open','approach'};why=['empty closed gripper; open before contact while approaching']
            else:
                actions={'hold'};why=['gripper transition/state not established; do not invent closed/open state']
        else:
            actions={'hold'};why=['locked instance not currently observable; no new object interaction asserted']
        result[name]={'allowed_verbs':sorted(actions),'observed':arm,'grounds':why}
    return {'target_instance_id':ident,'target_basket':basket,'category':target['category'],
        'intent_horizon_steps':16,'execute_steps_before_reobserve':8,'by_arm':result,
        'perception_is_fallible':True,'output_repair_allowed':False,
        'known_contact_transition':deepcopy(contact_transition)}

def mentioned_arms(text):
    refs=set(re.findall(r'\b(left|right|both)(?:-hand)?\s+(?:(?:open|closed|empty|free|robot)\s+){0,3}(?:arm|arms|gripper|grippers|hand|hands|fingers)\b',text,re.I))
    refs={x.lower()for x in refs}
    if'both'in refs:refs.discard('both');refs.update({'left','right'})
    return refs

def explicit_action_actors(text):
    """Conservative extraction of explicit actor/action phrases, not a grader."""
    forms={'approach':'approach','approaches':'approach','approaching':'approach',
        'align':'align','aligns':'align','aligning':'align','open':'open','opens':'open','opening':'open',
        'close':'close','closes':'close','closing':'close','lift':'lift','lifts':'lift','lifting':'lift',
        'raise':'lift','raises':'lift','raising':'lift','carry':'carry','carries':'carry','carrying':'carry',
        'transport':'carry','transports':'carry','transporting':'carry','lower':'lower','lowers':'lower','lowering':'lower',
        'release':'release','releases':'release','releasing':'release','retract':'retract','retracts':'retract',
        'retracting':'retract'}
    verbs='(?:'+'|'.join(sorted(forms,key=len,reverse=True))+')'
    actor=r'(left|right|both)(?:-hand)?\s+(?:(?:open|closed|empty|free|robot)\s+){0,3}(?:arm|arms|gripper|grippers|hand|hands|fingers)'
    patterns=[(re.compile(r'\b('+verbs+r')\b[^.;]{0,130}?\bwith\s+(?:the\s+)?'+actor,re.I),False),
        (re.compile(r'\b'+actor+r'\s+(?:(?:should|will|can|must|to|gently|slightly|continues|continue|starts|start)\s+){0,4}('+verbs+r')\b',re.I),True),
        (re.compile(r'\b('+verbs+r')\s+(?:(?:the|your|empty|open|closed)\s+){0,2}'+actor,re.I),False)]
    result=[]
    for clause in re.split(r'[.;]|\b(?:while|and|then|but)\b',text,flags=re.I):
        for pattern,actor_first in patterns:
            for match in pattern.finditer(clause):
                a,b=match.groups();who,word=(a,b)if actor_first else(b,a)
                verb_start=match.start(2 if actor_first else 1)
                before=clause[:verb_start]
                if word.lower()in{'open','opening'}and re.search(r'\b(?:the|an|a|keep|keeps|remain|remains|stay|stays|is|are)\s+(?:fully\s+)?$',before,re.I):
                    continue  # 'the open left gripper' is a state adjective.
                result.append((forms[word.lower()],{'left','right'}if who.lower()=='both'else{who.lower()}))
    return result

def validate_plan(context,envelope):
    """Reject, never edit, an unsupported explicit next-action annotation."""
    summary=context['action']['summary'];reasoning=context['reasoning']
    match=re.match(r'^([A-Za-z]+)\b',summary)
    verb=match[1].lower()if match else None
    require(verb in ACTIONS,'Summary must begin with one admitted immediate-action verb')
    arms=mentioned_arms(summary)
    require(bool(arms),'Action Summary must identify its acting left/right/both arm or gripper')
    for arm in arms:
        require(verb in envelope['by_arm'][arm]['allowed_verbs'],
            f'Immediate {verb} is not visually admitted for the {arm} arm')
    for action,actors in explicit_action_actors(summary+'; '+reasoning):
        for actor in actors:
            state_adjective=action=='open'and envelope['by_arm'][actor]['observed']['gripper_state']=='open'
            require(state_adjective or action in envelope['by_arm'][actor]['allowed_verbs'],
                f'Reasoning/Summary explicitly assigns unsupported {action} to the {actor} arm')
    allowed=set().union(*(set(envelope['by_arm'][a]['allowed_verbs'])for a in arms))
    # Explicit imperative/intended close/lift/release claims are especially
    # harmful in the16-step window. Do not validate by exact GT wording or add
    # future labels. Conditional multi-stage plans are also inappropriate when
    # their prerequisite cannot yet be seen within this short action window.
    critical={
        'close':r'\b(?:close|closing)\s+(?:the|your|its|left|right|both)\b',
        'lift':r'\b(?:lift|raise|lifting|raising)\s+(?:the|it|them|this|that|slightly|gently|left|right|both)\b',
        'carry':r'\b(?:carry|carrying|transport|transporting)\s+(?:the|it|them|this|that)\b',
        'release':r'\b(?:release|releasing|drop|dropping)\s+(?:the|it|them|this|that)\b'}
    for action,pattern in critical.items():
        if action not in allowed:
            require(re.search(pattern,summary+' '+reasoning,re.I)is None,
                'Public short-term plan skips a currently unobserved prerequisite: '+action)
    family=lexical_family(envelope['category'])
    if family is not None:
        for text in [summary,reasoning]:
            require(lexical_family(text)in(None,family),'Reasoning/Summary names a different explicit object category')
    # A current-state assertion must not contradict the observer. Intended
    # transitions use Open/Close rather than pretending the state already holds.
    for name,item in envelope['by_arm'].items():
        observed=item['observed']['gripper_state']
        for claimed in ['open','closed']:
            state_claim=(r'\b'+name+r' gripper (?:is|stays?|remains?) (?:fully )?'+claimed+r'\b|'
                r'\bkeep (?:the )?'+name+r' gripper (?:fully )?'+claimed+r'\b')
            if re.search(state_claim,reasoning,re.I):
                require(observed==claimed,'Reasoning contradicts currently observed '+name+' gripper state')
    if verb in{'carry','lower','release'}:
        destinations=re.findall(r'\b(?:into|toward|towards|over|above|in) (?:the )?(white|blue|red) basket\b',summary,re.I)
        require(all(x.lower()==envelope['target_basket']for x in destinations),'Immediate action uses a different target basket')
    intended_destinations=re.findall(r'\b(?:carry|carrying|transport|transporting|lower|lowering|release|releasing|place|move|bring)\b[^.;]{0,130}?\b(?:to|toward|towards|into|over|above|in) (?:the )?(white|blue|red) basket\b',reasoning,re.I)
    require(all(x.lower()==envelope['target_basket']for x in intended_destinations),
        'Reasoning directs the next action toward a different target basket')
    return {'summary_primary_action':verb,'acting_arms':sorted(arms),'permitted_by_current_visual_evidence':True,
        'all_public_text_preserved':True,'semantic_accuracy_not_certified_by_lexical_checks':True}
