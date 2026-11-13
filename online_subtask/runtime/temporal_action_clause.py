"""A finite completed-release clause recognizer, separate from negation.

Only a terminal after-releasing/after-having-released clause in an explicit
Retract-empty-arm Summary can qualify. The caller must already have validated
the actual request/capture and current scene/goal admission. This helper checks
their binding and current released facts; it neither establishes visual truth
nor certifies the rest of the plan. It never edits input or grants new actions,
changes a goal, declares task completion, or weakens grasp validation.
"""
import hashlib
import json
import math
import re


CAMERAS = {'cam_high', 'cam_left_wrist', 'cam_right_wrist'}
CATEGORIES = {'pen', 'watch', 'camera', 'plush_toy', 'toy_vehicle', 'block',
              'box', 'ring', 'bracelet', 'pouch', 'strap', 'tool', 'bottle_cap',
              'can', 'tracker'}
SUMMARY = re.compile(
    r'^Retract the empty (?P<arm>left|right) (?:arm|gripper|hand) '
    r'from the (?P<basket>white|blue|red) basket '
    r'after (?P<form>releasing|having released) the '
    r'(?P<object>[A-Za-z0-9_\-]+(?: [A-Za-z0-9_\-]+){0,5}?)'
    r'(?: into the (?P<destination>white|blue|red) basket)?\.?$', re.I,
)


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _has_unparsed_quote(text):
    for index, char in enumerate(text):
        if char in {'"', '“', '”', '‘', '`'}:
            return True
        if char in {"'", '’'} and not (0 < index < len(text) - 1
                and text[index - 1].isalnum() and text[index + 1].isalnum()):
            return True
    return False


def is_completed_release_clause(text, action_start_index, *, context, evidence,
                                target, proof, envelope):
    """Recognize one exact historical release token using CURRENT observations.

    target has subtask_id/instance_id/category/basket_color from the actual guard.
    proof is that same current selector proof, including its current observation
    SHA, guard decision and three synchronized capture records. No prior-only
    completion, text, execution history, future action or stale frame suffices.

    True exempts only this token from a *future release intention* check. It is
    not negation and must never bypass actor/state/primary-action/destination,
    temporal-chain, scene, history, capture or injection/ACK validation.
    """
    try:
        return _is_completed_release_clause(text, action_start_index,
            context=context, evidence=evidence, target=target, proof=proof, envelope=envelope)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError, OverflowError):
        return False


def _is_completed_release_clause(text, index, *, context, evidence, target, proof, envelope):
    if type(text) is not str or type(index) is not int:
        return False
    if not all(type(item) is dict for item in (context, evidence, target, proof, envelope)):
        return False
    summary, reason = context['action']['summary'], context['reasoning']
    if (type(summary) is not str or type(reason) is not str
            or text not in (summary, summary + ' ' + reason, summary + '; ' + reason)
            or _has_unparsed_quote(text)):
        return False
    match = SUMMARY.fullmatch(summary)
    if match is None:
        return False
    action_start = match.start('form') + (len('having ') if match['form'].lower().startswith('having ') else 0)
    if index != action_start:
        return False
    arm, basket = match['arm'].lower(), match['basket'].lower()
    ident, category, sid = target['instance_id'], target['category'], target['subtask_id']
    if (not re.fullmatch(r'O[0-9]+', ident) or category not in CATEGORIES
            or not re.fullmatch(r'U[0-9]{4}', sid) or target['basket_color'] != basket):
        return False
    noun = category.replace('_', ' ')
    if match['object'].lower() not in {noun, ident.lower(), noun + ' ' + ident.lower()}:
        return False
    if match['destination'] is not None and match['destination'].lower() != basket:
        return False
    episode, frame = proof['episode_id'], proof['current_frame']
    if (proof.get('call_role') != 'subtask_selector'
            or type(episode) is not str or not episode or type(frame) is not int or frame < 0 or frame % 8
            or proof['current_visual_observation'] != evidence
            or proof['current_visual_observation_sha256'] != digest(evidence)):
        return False
    observations = proof['observations']
    if type(observations) is not list or not 3 <= len(observations) <= 96:
        return False
    if not all(type(row) is dict and type(row.get('frame')) is int
               and 0 <= row['frame'] <= frame for row in observations):
        return False
    current = [row for row in observations if row.get('is_current') is True]
    if (len(current) != 3 or {row.get('camera') for row in current} != CAMERAS
            or not all(row['frame'] == frame for row in current)):
        return False
    states = []
    for row in current:
        state = row.get('state14')
        if (type(row.get('image_sha256')) is not str
                or re.fullmatch(r'[0-9a-f]{64}', row['image_sha256']) is None
                or type(state) is not list or len(state) != 14
                or not all(type(x) in (int, float) and math.isfinite(x) for x in state)):
            return False
        states.append(state)
    if any(state != states[0] for state in states[1:]):
        return False
    decision = proof['grounded_decision']
    if (type(decision) is not dict or decision.get('episode_id') != episode
            or decision.get('frame') != frame or decision.get('locked_id') != sid
            or decision.get('allowed_ids') != [sid]):
        return False
    actual_target = decision['targets'][sid]
    if any(actual_target.get(key) != target[key] for key in ('instance_id', 'category', 'basket_color')):
        return False
    if (evidence.get('scene_usable') is not True or evidence.get('goal_correction') is not None
            or evidence.get('corrections') != []):
        return False
    objects = evidence['objects']
    if type(objects) is not list:
        return False
    matches = [obj for obj in objects if type(obj) is dict and obj.get('instance_id') == ident]
    if len(matches) != 1:
        return False
    obj = matches[0]
    if (obj.get('identity_confidence') != 'clear' or obj.get('category_confidence') != 'clear'
            or obj.get('descriptor_confidence') != 'clear' or not obj.get('descriptor_ids')
            or obj.get('category') != category or obj.get('location') != basket
            or obj.get('released') is not True):
        return False
    basket_facts = evidence['baskets'][basket]
    if basket_facts.get('visibility') != 'clear' or basket_facts.get('unknown_contents') is not False:
        return False
    observed = evidence['arms'][arm]
    if not all(observed.get(key) == value for key, value in {
            'holding': 'empty', 'held_instance_id': None, 'gripper_state': 'open',
            'interaction_instance_id': ident, 'object_secured': 'no', 'observed_basket': basket,
            'release_ready': 'no', 'visibility': 'clear'}.items()):
        return False
    if observed.get('relation') not in {'released', 'retreat'} or observed.get('motion_trend') not in {'retract', 'stationary'}:
        return False
    other = evidence['arms']['right' if arm == 'left' else 'left']
    if (other.get('holding') != 'empty' or other.get('held_instance_id') is not None
            or other.get('visibility') != 'clear' or other.get('object_secured') != 'no'
            or other.get('interaction_instance_id') == ident):
        return False
    if any(envelope.get(key) != value for key, value in {
            'target_instance_id': ident, 'target_basket': basket, 'category': category}.items()):
        return False
    item = envelope['by_arm'][arm]
    if item.get('observed') != observed or set(item.get('allowed_verbs', [])) != {'retract', 'hold'}:
        return False
    return True
