"""Deterministic admission of Subtasks from fallible, causal visual evidence.

No simulator labels, rewards, object poses or model calls enter this module.
It guarantees consistency WITH the accepted evidence, not pixel recognition.
"""
from copy import deepcopy
import re

from executed_history import digest


STEADY_CONFIG_ID = 'mainhist93_tok65536_steady_v1'
STEADY_NAMESPACE = 'seed1_native50_subtask_steady_20261113_v1'
BASKETS = ('white', 'blue', 'red')
OBJECT_FIELDS = {'instance_id', 'category', 'descriptor_ids', 'location',
                 'released', 'evidence_frames', 'placement_frames', 'unplaced_frames'}
COLOR_WORDS = {'white', 'black', 'blue', 'red', 'green', 'yellow', 'orange',
               'purple', 'pink', 'gold', 'silver', 'beige', 'cyan', 'brown', 'gray', 'grey'}
def lexical_family(phrase):
    """Broad lexical sanity check, NEVER a category key or an alias merger.

    Box/device/tool/ring and other ambiguous wording is left to visual evidence.
    This catches, for example, an observer calling an explicit camera a pen.
    """
    words = set(re.findall(r'[a-z]+', phrase.lower().replace('_', ' ')))
    if words.intersection({'pen', 'marker', 'stylus'}) or {'writing', 'instrument'} <= words:
        return 'pen'
    if words.intersection({'watch', 'smartwatch', 'wristwatch', 'wristband'}) or 'fitness band' in phrase:
        return 'watch'
    if 'camera' in words:
        return 'camera'
    if words.intersection({'doll', 'plush'}):
        return 'plush_toy'
    if 'toy' in words and words.intersection({'car', 'bus', 'truck', 'engine', 'vehicle'}):
        return 'toy_vehicle'
    return None


class SubtaskConstraintError(ValueError):
    """An unmeasured decision stop: never an SDK failure or substitute ID."""


def require(value, message):
    if not value:
        raise SubtaskConstraintError(message)


class SubtaskGuard:
    def __init__(self, catalog):
        self.by_id = {item['id']: item for item in catalog['candidates']}
        self.descriptors = {item['descriptor_id'] for item in self.by_id.values()}
        self.descriptor_families = {item['descriptor_id']: lexical_family(item['object_phrase'])
                                      for item in self.by_id.values()}
        self.reset(None)

    def reset(self, episode_id):
        self.episode_id = episode_id
        self.objects = {}
        self.category_to_basket = {}
        self.basket_to_category = {}
        self.descriptor_to_category = {}
        self.identity_ambiguities = {}
        self.completed = set()
        self.active = None
        self.decision = None
        self.authorization = None

    def snapshot(self):
        return deepcopy({'episode_id': self.episode_id, 'known_objects': self.objects,
                         'category_to_basket': self.category_to_basket,
                         'basket_to_category': self.basket_to_category,
                         'descriptor_to_category': self.descriptor_to_category,
                         'identity_ambiguities': self.identity_ambiguities,
                         'completed_instance_ids': sorted(self.completed),
                         'active_goal': self.active})

    def observe(self, evidence, *, episode_id, frame, available_frames, executed_records):
        """Validate atomically; an invalid report cannot mutate the scene ledger.

        Omitted objects remain unknown/occluded; omission never means completion.
        Placement evidence is sourced from transmitted images, not action count.
        """
        require(episode_id == self.episode_id and bool(episode_id), 'Wrong guard episode')
        require(type(frame) is int and frame >= 0 and frame % 8 == 0, 'Wrong guard frame')
        require(type(executed_records) is list and len(executed_records) == frame // 8,
                'Actual executed prefix required before the visual decision')
        require(all(row['observation_frame'] == i * 8 and row['executed_action_start'] == i * 8 + 1
                    and row['executed_action_end'] == i * 8 + 8
                    for i, row in enumerate(executed_records)), 'Noncontiguous executed prefix')
        require(type(evidence) is dict and set(evidence) == {'uncertain', 'basket_visibility', 'objects'},
                'Exact scene evidence schema required')
        require(type(evidence['uncertain']) is bool and not evidence['uncertain'],
                'Visual scene uncertain; do not invent a target')
        visibility = evidence['basket_visibility']
        require(type(visibility) is dict and set(visibility) == set(BASKETS)
                and all(v in ('clear', 'unknown') for v in visibility.values()),
                'All three basket visibility states required')
        require(type(evidence['objects']) is list and len(evidence['objects']) <= 64,
                'Bounded visual object list required')
        frames = set(available_frames)
        require(frame in frames and all(type(f) is int and 0 <= f <= frame for f in frames),
                'Only actually transmitted causal image frames are admissible')
        known = deepcopy(self.objects)
        categories = dict(self.category_to_basket)
        baskets = dict(self.basket_to_category)
        descriptor_categories = dict(self.descriptor_to_category)
        ambiguities = deepcopy(self.identity_ambiguities)
        completed = set(self.completed)
        visible = {}
        for obj in evidence['objects']:
            require(type(obj) is dict and set(obj) == OBJECT_FIELDS, 'Exact visual object fields required')
            ident, category = obj['instance_id'], obj['category']
            require(type(ident) is str and re.fullmatch(r'O[1-9][0-9]*', ident)
                    and ident not in visible, 'Unique stable visual instance IDs required')
            require(type(category) is str and re.fullmatch(r'[a-z]+(?:_[a-z]+)*', category)
                    and category not in ('unknown', 'object', 'item', 'thing', 'toy', 'device')
                    and not COLOR_WORDS.intersection(category.split('_')),
                    'Semantic category must be explicit and independent of object color')
            descriptors = obj['descriptor_ids']
            require(type(descriptors) is list
                    and all(type(d) is str and d in self.descriptors for d in descriptors)
                    and len(set(descriptors)) == len(descriptors),
                    'Only original catalogue descriptor IDs are allowed')
            require(all(self.descriptor_families[d] in (None, lexical_family(category)) for d in descriptors),
                    'Visual category contradicts an explicit catalogue descriptor')
            for desc in descriptors:
                require(descriptor_categories.get(desc, category) == category,
                        'Same catalogue descriptor assigned contradictory visual categories')
                descriptor_categories[desc] = category
            require(obj['location'] in ('table', 'held') + BASKETS and type(obj['released']) is bool,
                    'Explicit currently observed location/release required')
            require(not (obj['location'] == 'held' and obj['released']), 'A held object is not released')
            refs, placements, unplaced = obj['evidence_frames'], obj['placement_frames'], obj['unplaced_frames']
            for values in (refs, placements, unplaced):
                require(type(values) is list and all(type(f) is int and f in frames for f in values)
                        and values == sorted(set(values)), 'Evidence refers to an absent/future/duplicate frame')
            require(bool(refs) and refs[-1] == frame, 'Object facts require a current image observation')
            require(set(placements) <= set(refs), 'Placement needs its own transmitted image evidence')
            require(set(unplaced) <= set(refs) and not set(unplaced).intersection(placements),
                    'Placed/unplaced witnesses must be supported and cannot contradict at one frame')
            if obj['location'] in ('table', 'held') or not obj['released']:
                require(frame in unplaced, 'Current unplaced/unreleased state needs a negative witness')
            require(not placements or obj['location'] in BASKETS and obj['released'],
                    'Placement frames require visible basket containment and release')
            if ident in known:
                require(known[ident]['category'] == category,
                        'Visual identity/category contradiction; explicit review required')
            visible[ident] = dict(deepcopy(obj), category=category)
            known[ident] = dict(deepcopy(obj), category=category)
            if obj['location'] in ('table', 'held') or not obj['released']:
                # Positive evidence of a drop undoes completion, not its category assignment.
                completed.discard(ident)
            if obj['location'] in BASKETS and obj['released']:
                require(bool(placements) and placements[-1] == frame,
                        'Basket occupancy needs a current placement witness')
                basket = obj['location']
                require(categories.get(category, basket) == basket,
                        'Same category observed in different baskets')
                require(baskets.get(basket, category) == category,
                        'Different categories observed in the same basket')
                categories[category] = basket
                baskets[basket] = category
        require(len(known) <= 64, 'Visual instance registry exceeded its bounded object limit')
        for ident, obj in visible.items():
            conflicts = set(ambiguities.get(ident, []))
            if ident not in self.objects or not self.objects[ident]['descriptor_ids']:
                conflicts.update(old_id for old_id, old in self.objects.items()
                    if old_id != ident and old_id not in visible
                    and set(old['descriptor_ids']).intersection(obj['descriptor_ids']))
            # Repeatedly reporting a new ID is not independent identity evidence.
            # Resolve a suspected alias only with both instances explicitly visible.
            conflicts.difference_update(visible)
            if conflicts:
                ambiguities[ident] = sorted(conflicts)
            else:
                ambiguities.pop(ident, None)
        active = deepcopy(self.active)
        completion = None
        if active is not None:
            category, basket = active['category'], active['basket_color']
            require(categories.get(category, basket) == basket
                    and baskets.get(basket, category) == category,
                    'Active goal contradicts observed category/basket assignments')
            target = visible.get(active['instance_id'])
            if target is not None:
                require(self.by_id[active['subtask_id']]['descriptor_id'] in target['descriptor_ids'],
                        'Current visual identity contradicts the locked descriptor; stop without ID repair')
                active['last_unplaced_at'] = max([active['last_unplaced_at']] + target['unplaced_frames'])
                # A different similar instance cannot complete this locked goal.
                barrier = max(active['selected_at'], active['last_unplaced_at'])
                after_start = [f for f in target['placement_frames'] if f > barrier]
                if (target['location'] == basket and target['released'] and len(after_start) >= 2
                        and after_start[-1] == frame and frame > active['selected_at']):
                    completion = dict(active, confirmed_at=frame, placement_frames=after_start)
                    completed.add(active['instance_id'])
                    active = None
        targets = {}
        if active is not None:
            # A program-enforced singleton, even when another candidate appears plausible.
            targets[active['subtask_id']] = {k: active[k] for k in
                                           ('instance_id', 'category', 'basket_color')}
        else:
            # A textual ID must identify one visible unfinished physical instance.
            owners = {}
            held = any(obj['location'] == 'held' for obj in visible.values())
            for obj in visible.values():
                if obj['location'] not in ('table', 'held') or obj['instance_id'] in completed:
                    continue
                if held and obj['location'] != 'held':
                    continue  # Finish an actually held object before selecting a table pickup.
                for desc in obj['descriptor_ids']:
                    owners.setdefault(desc, []).append(obj)
            for ident, candidate in self.by_id.items():
                objects = owners.get(candidate['descriptor_id'], [])
                if len(objects) != 1:
                    continue  # ambiguous identity is never resolved by a host guess
                obj = objects[0]
                if obj['instance_id'] in ambiguities:
                    continue
                category, basket = obj['category'], candidate['basket_color']
                if (categories.get(category, basket) != basket
                        or baskets.get(basket, category) != category):
                    continue
                if visibility[basket] != 'clear':
                    continue  # even an assigned basket may now have unseen conflicting contents
                targets[ident] = {'instance_id': obj['instance_id'], 'category': category,
                                  'basket_color': basket}
        # Preserve accepted visual facts and a completed FINAL goal even when
        # there is no next selectable target. This is not SDK task success.
        self.objects, self.category_to_basket, self.basket_to_category = known, categories, baskets
        self.descriptor_to_category = descriptor_categories
        self.identity_ambiguities = ambiguities
        self.completed, self.active = completed, active
        self.decision = {'episode_id': episode_id, 'frame': frame,
                         'mode': 'locked' if active else 'new_goal' if targets else 'no_legal_target',
                         'locked_id': active['subtask_id'] if active else None,
                         'allowed_ids': sorted(targets), 'targets': targets,
                         'category_to_basket': categories, 'basket_to_category': baskets,
                         'confirmed_completion': completion}
        self.authorization = None
        require(bool(targets), 'No unambiguous legal Subtask; stop before policy inference')
        return deepcopy(self.decision)

    def select(self, selected_id):
        require(self.decision is not None and selected_id in self.decision['targets'],
                'Candidate violates the goal lock or category/basket constraints')
        if self.active is None:
            self.active = dict(self.decision['targets'][selected_id], subtask_id=selected_id,
                               selected_at=self.decision['frame'], last_unplaced_at=self.decision['frame'])
        require(self.active['subtask_id'] == selected_id, 'Locked Subtask changed')
        result = dict(deepcopy(self.decision), active_goal=deepcopy(self.active),
                      raw_proposed_id=selected_id, actual_selected_id=selected_id,
                      scene_state_sha256=digest(self.snapshot()),
                      program_substitution=False, evidence_is_visual_model_output_not_GT=True)
        self.authorization = deepcopy(result)
        return result

    def validate_injection(self, context, lineage, *, episode_id, frame):
        require(self.authorization is not None and episode_id == self.episode_id
                and self.authorization['frame'] == frame, 'No current steady authorization')
        require(digest(lineage.get('steady_guard')) == digest(self.authorization),
                'Steady authorization was changed or belongs to an earlier decision')
        require(digest(self.snapshot()) == self.authorization['scene_state_sha256'],
                'Scene evidence changed after candidate admission')
        require(self.category_to_basket == self.authorization['category_to_basket']
                and self.basket_to_category == self.authorization['basket_to_category']
                and self.active == self.authorization['active_goal'],
                'Scene constraints changed after candidate admission')
        ident = self.active['subtask_id']
        require(context == {'subtask': self.by_id[ident]['text']}
                and lineage.get('selected_id') == ident
                and lineage.get('resolution', {}).get('selected_id') == ident,
                'Policy injection does not match the authorized locked ID/text')
