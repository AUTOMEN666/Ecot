"""Current visual facts, local uncertainty, instance goals and causal corrections.

The model reports CURRENT facts, never historical frame numbers. The caller
supplies the real query frame; two consecutive current observations establish
released containment. This proves consistency with fallible visual reports, not
pixel recognition or SDK success. No proposal is an executed-history record.
"""
from copy import deepcopy
import re

from executed_history import digest, validate_records
from subtask_guard import BASKETS, COLOR_WORDS, SubtaskConstraintError, lexical_family, require

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
OBJECT_FIELDS = {'instance_id', 'identity_confidence', 'category', 'category_confidence',
                 'descriptor_ids', 'descriptor_confidence', 'location', 'released'}
BASE_FIELDS = {'scene_usable', 'baskets', 'objects', 'corrections', 'arms'}
CORRECTION_FIELDS = {'instance_id', 'previous_category', 'previous_descriptor_ids',
                     'reason', 'evidence_cameras'}


def instance_id(value):
    return type(value) is str and re.fullmatch(r'O[1-9][0-9]*', value) is not None


def clear_identity(obj):
    return obj['identity_confidence'] == 'clear'


def clear_category(obj):
    return clear_identity(obj) and obj['category_confidence'] == 'clear' and obj['category'] is not None


def clear_candidate(obj):
    return clear_category(obj) and obj['descriptor_confidence'] == 'clear' and bool(obj['descriptor_ids'])


def explanation(row):
    require(type(row['reason']) is str and 0 < len(row['reason'].strip()) <= 800
            and not any(ord(c) < 32 for c in row['reason']), 'A bounded current visual correction reason is required')
    cameras = row['evidence_cameras']
    require(type(cameras) is list and bool(cameras) and len(cameras) == len(set(cameras))
            and all(type(c) is str and c in CAMERAS for c in cameras),
            'Correction must identify actually supplied current cameras')
    return set(cameras)


class GroundedSceneGuard:
    """Preserve IDs across phases; unclear unrelated entities are locally blocked."""
    def __init__(self, catalog):
        self.catalog = deepcopy(catalog)
        self.by_id = {x['id']: x for x in catalog['candidates']}
        self.descriptors = {x['descriptor_id'] for x in self.by_id.values()}
        self.descriptor_families = {x['descriptor_id']: lexical_family(x['object_phrase']) for x in self.by_id.values()}
        self.descriptor_colors = {x['descriptor_id']: COLOR_WORDS.intersection(re.findall(r'[a-z]+', x['object_phrase'].lower()))
                                  for x in self.by_id.values()}
        self.reset(None)

    def reset(self, episode_id):
        self.episode_id = episode_id
        self.frame = None
        self.objects = {}
        self.visible = {}
        self.arms = {}
        self.baskets = {}
        self.category_to_basket = {}
        self.basket_to_category = {}
        self.blocked_baskets = set()
        self.ambiguous_categories = set()
        self.identity_ambiguities = {}
        self.placement_streaks = {}
        self.mapping_witnesses = {}
        self.completed = set()
        self.active = None
        self.last_completed_goal = None
        self.correction_log = []
        self.interrupted_goals = []
        self.observation_ledger = []
        self.decision = None
        self.authorization = None
        self.closing_transition_streak = 0
        self.closing_transition_arm = None

    def snapshot(self):
        return deepcopy({'episode_id': self.episode_id, 'current_frame': self.frame,
            'known_objects': self.objects, 'current_visible_objects': self.visible,
            'arms': self.arms, 'baskets': self.baskets,
            'category_to_basket': self.category_to_basket, 'basket_to_category': self.basket_to_category,
            'blocked_baskets': sorted(self.blocked_baskets), 'ambiguous_categories': sorted(self.ambiguous_categories),
            'identity_ambiguities': self.identity_ambiguities, 'placement_streaks': self.placement_streaks,
            'mapping_witnesses': self.mapping_witnesses, 'completed_instance_ids': sorted(self.completed),
            'active_goal': self.active, 'last_completed_goal': self.last_completed_goal,
            'recent_corrections': self.correction_log[-4:], 'correction_log_sha256': digest(self.correction_log),
            'recent_interrupted_goals': self.interrupted_goals[-4:], 'interrupted_goals_sha256': digest(self.interrupted_goals),
            'visual_observation_count': len(self.observation_ledger),
            'closing_transition_uncertainty_windows': self.closing_transition_streak,
            'closing_transition_arm': self.closing_transition_arm,
            'visual_observation_prefix_sha256': digest(self.observation_ledger)})

    def _known_closing_transition(self, previous_arms, previous_visible, evidence, frame):
        """Continue an existing enclosed-target close; never infer secured holding.

        Both witnesses are actual adjacent visual reports, not selected text or
        planned commands. At most two8-action uncertainty windows are admitted.
        An unidentified material, a single contact, two unknown hands or an
        occluded target cannot use this exception to authorize a new pickup.
        """
        if (self.active is None or self.frame is None or frame != self.frame + 8
                or self.closing_transition_streak >= 2 or not self.observation_ledger):
            return None
        ident = self.active['instance_id']
        unknown = [name for name, arm in self.arms.items() if arm['holding'] == 'unknown']
        if len(unknown) != 1:
            return None
        name = unknown[0]
        if self.closing_transition_streak and name != self.closing_transition_arm:
            return None
        if any(o['location'] == 'held' for rows in (self.visible, previous_visible, self.objects) for o in rows.values()):
            return None  # A separate unresolved held material cannot hide behind the known table target.
        obj, prior_obj = self.visible.get(ident), previous_visible.get(ident)
        if not all(o is not None and clear_candidate(o) and o['location'] == 'table'
                   and o['released'] is not True for o in (obj, prior_obj)):
            return None
        if (obj['category'] != self.active['category']
                or self.by_id[self.active['subtask_id']]['descriptor_id'] not in obj['descriptor_ids']):
            return None
        current, prior = self.arms[name], previous_arms.get(name)
        if prior is None or prior['holding'] not in ('empty', 'unknown'):
            return None
        for arm in (current, prior):
            if not (arm.get('interaction_instance_id') == ident and arm.get('held_instance_id') is None
                    and arm.get('relation') == 'enclosing' and arm.get('gripper_state') == 'transitioning'
                    and arm.get('motion_trend') == 'close' and arm.get('object_secured') == 'no'
                    and arm.get('visibility') == 'clear' and arm.get('release_ready') == 'no'):
                return None
        other = 'right' if name == 'left' else 'left'
        if not all(arms.get(other, {}).get('holding') == 'empty'
                   and arms.get(other, {}).get('visibility') == 'clear'
                   for arms in (self.arms, previous_arms)):
            return None
        return {'arm': name, 'instance_id': ident, 'previous_frame': self.frame, 'current_frame': frame,
            'previous_visual_report_sha256': self.observation_ledger[-1]['visual_report_sha256'],
            'current_visual_report_sha256': digest(evidence),
            'uncertainty_windows': self.closing_transition_streak + 1,
            'allowed_verbs': ['close', 'hold'], 'holding_identity_or_grasp_support_promoted': False,
            'new_goal_or_completion_admitted': False}

    def observe(self, evidence, *, episode_id, frame, available_frames, executed_records):
        self.authorization = None
        trial = deepcopy(self)
        decision = trial._observe(evidence, episode_id=episode_id, frame=frame,
                                  available_frames=available_frames, executed_records=executed_records)
        self.__dict__.update(trial.__dict__)
        require(bool(decision['allowed_ids']), decision.get('stop_reason', 'No grounded legal Subtask'))
        return deepcopy(decision)

    def validate_observation_domain(self, episode_id, frame, available_frames, executed_records):
        """Strict default timing/execution domain; separate audits must override explicitly."""
        require(episode_id == self.episode_id and bool(episode_id), 'Wrong grounded scene episode')
        require(type(frame) is int and frame == (0 if self.frame is None else self.frame + 8),
                'Current visual observations must start0 and advance exactly8 real actions')
        try:
            validate_records(executed_records, episode_id, frame, self.catalog)
        except (ValueError, TypeError, KeyError) as error:
            raise SubtaskConstraintError('Complete exact executed Subtask prefix required') from error
        require(type(available_frames) in (list, tuple) and frame in available_frames
                and all(type(f) is int and 0 <= f <= frame for f in available_frames),
                'Only actual causal input image frames are admissible')

    def _validate(self, evidence, episode_id, frame, available_frames, executed_records):
        self.validate_observation_domain(episode_id, frame, available_frames, executed_records)
        require(type(evidence) is dict and BASE_FIELDS <= set(evidence)
                and set(evidence) <= BASE_FIELDS | {'goal_correction'}, 'Exact current-observation scene schema required')
        require(evidence['scene_usable'] is True, 'Current scene images unusable; no grounded action')
        require(type(evidence['baskets']) is dict and set(evidence['baskets']) == set(BASKETS),
                'All three current basket reports required')
        for row in evidence['baskets'].values():
            require(type(row) is dict and set(row) == {'visibility', 'unknown_contents'}
                    and row['visibility'] in ('clear', 'unknown') and type(row['unknown_contents']) is bool,
                    'Explicit basket visibility and local unknown contents required')
        require(type(evidence['objects']) is list and len(evidence['objects']) <= 64, 'Bounded current object list required')
        visible = {}
        for raw in evidence['objects']:
            require(type(raw) is dict and set(raw) == OBJECT_FIELDS, 'Exact local object observation fields required')
            obj = deepcopy(raw)
            ident = obj['instance_id']
            require(instance_id(ident) and ident not in visible, 'Unique stable physical instance IDs required')
            for key in ('identity_confidence', 'category_confidence', 'descriptor_confidence'):
                require(obj[key] in ('clear', 'uncertain'), 'Explicit local confidence required: ' + key)
            category = obj['category']
            require(category is None or type(category) is str and re.fullmatch(r'[a-z]+(?:_[a-z]+)*', category)
                    and category not in ('unknown', 'object', 'item', 'thing', 'toy', 'device')
                    and not COLOR_WORDS.intersection(category.split('_')), 'Category must be semantic, not color or generic object')
            require(obj['category_confidence'] != 'clear' or category is not None,
                    'Unknown category cannot be declared clear')
            descs = obj['descriptor_ids']
            require(type(descs) is list and all(type(d) is str and d in self.descriptors for d in descs)
                    and len(descs) == len(set(descs)), 'Only exact original descriptor IDs permitted')
            if clear_category(obj) and obj['descriptor_confidence'] == 'clear':
                require(all(self.descriptor_families[d] in (None, lexical_family(category)) for d in descs),
                        'Observed category contradicts an explicit catalogue descriptor')
            require(obj['location'] in ('table', 'held', 'unknown') + BASKETS
                    and (obj['released'] is None or type(obj['released']) is bool), 'Observed location and release state required')
            require(not (obj['location'] == 'held' and obj['released'] is True), 'Held object cannot be released')
            visible[ident] = obj
        require(type(evidence['arms']) is dict and set(evidence['arms']) == {'left', 'right'}, 'Both current arm reports required')
        held = set()
        for name, arm in evidence['arms'].items():
            require(type(arm) is dict and {'holding', 'held_instance_id'} <= set(arm)
                    and arm['holding'] in ('empty', 'identified', 'unknown'), 'Current held/empty/unknown arm state required')
            ident = arm['held_instance_id']
            require(ident is None or instance_id(ident), 'Held instance must use a stable observed ID')
            if arm['holding'] == 'identified':
                require(ident in visible and clear_identity(visible[ident]) and visible[ident]['location'] == 'held',
                        'Identified held object must be the same currently visible physical instance')
                held.add(ident)
            else:
                require(ident is None, 'Empty or unknown hand cannot assert an identified held instance')
        for ident, obj in visible.items():
            if obj['location'] == 'held':
                require(ident in held or any(a['holding'] == 'unknown' for a in evidence['arms'].values()),
                        'A held object cannot be hidden behind two empty-hand reports')
        corrections = evidence['corrections']
        require(type(corrections) is list and len(corrections) <= 64, 'Bounded explicit correction list required')
        corrected = {}
        for row in corrections:
            require(type(row) is dict and set(row) == CORRECTION_FIELDS, 'Exact visual identity correction fields required')
            ident = row['instance_id']
            require(instance_id(ident) and ident not in corrected and ident in self.objects and ident in visible,
                    'Correction must name one known, currently observed physical instance')
            old, new = self.objects[ident], visible[ident]
            require(row['previous_category'] == old['category'] and type(row['previous_descriptor_ids']) is list
                    and all(type(x) is str for x in row['previous_descriptor_ids'])
                    and len(row['previous_descriptor_ids']) == len(set(row['previous_descriptor_ids']))
                    and set(row['previous_descriptor_ids']) == set(old['descriptor_ids']),
                    'Correction must bind the exact previous instance interpretation')
            explanation(row)
            require(clear_category(new) and new['descriptor_confidence'] == 'clear',
                    'Correction requires a clear current identity/category/description')
            corrected[ident] = deepcopy(row)
        return visible, held, corrected

    def _merge_objects(self, visible, corrected, frame):
        for ident, obj in visible.items():
            previous = self.objects.get(ident)
            if previous is not None and clear_category(obj) and previous['category_confidence'] == 'clear':
                require(previous['category'] == obj['category'] or ident in corrected,
                        'Visual category contradiction needs an explicit same-instance correction')
            if previous is not None and ident not in corrected and clear_candidate(obj) and previous['descriptor_confidence'] == 'clear':
                old_colors = set().union(*(self.descriptor_colors[d] for d in previous['descriptor_ids']))
                new_colors = set().union(*(self.descriptor_colors[d] for d in obj['descriptor_ids']))
                old_primary, new_primary = old_colors - {'gold', 'silver'}, new_colors - {'gold', 'silver'}
                color_conflict = (old_primary and new_primary and not old_primary.intersection(new_primary))
                require(not color_conflict and (not old_colors or not new_colors or bool(old_colors.intersection(new_colors))),
                        'Clear color/descriptor contradiction requires explicit visual correction')
            stored = deepcopy(obj)
            stored['last_placement_basket'] = previous.get('last_placement_basket') if previous else None
            stored['category_observed_at'] = (frame if clear_category(obj)
                else previous.get('category_observed_at') if previous else None)
            stored['descriptor_observed_at'] = (frame if clear_identity(obj) and obj['descriptor_confidence'] == 'clear'
                else previous.get('descriptor_observed_at') if previous else None)
            if previous and ident not in corrected:
                if obj['category_confidence'] != 'clear' or not clear_identity(obj):
                    stored['category'] = previous['category']
                    stored['category_confidence'] = previous['category_confidence']
                if obj['descriptor_confidence'] != 'clear' or not clear_identity(obj):
                    stored['descriptor_ids'] = deepcopy(previous['descriptor_ids'])
                    stored['descriptor_confidence'] = previous['descriptor_confidence']
                elif previous['category'] == obj['category']:
                    # An omitted compatible synonym is not a visual contradiction.
                    stored['descriptor_ids'] = sorted(set(previous['descriptor_ids']) | set(obj['descriptor_ids']))
            stored['last_seen_at'] = frame
            self.objects[ident] = stored
            if ident in corrected:
                self.correction_log.append(dict(corrected[ident], frame=frame,
                    corrected_category=obj['category'], corrected_descriptor_ids=deepcopy(obj['descriptor_ids'])))
                if previous['category'] != obj['category']:
                    # A newly learned category cannot retrospectively outrank an
                    # independent category assignment already visible earlier.
                    for witness in self.mapping_witnesses.values():
                        if witness['instance_id'] == ident:
                            witness['category_reinterpreted_at'] = frame
        require(len(self.objects) <= 64, 'Bounded physical instance registry exceeded')

    def _update_placements(self, visible, frame):
        for ident in list(self.placement_streaks):
            if ident not in visible:
                self.placement_streaks[ident] = []
        for ident, obj in visible.items():
            placed = clear_category(obj) and obj['location'] in BASKETS and obj['released'] is True
            old_streak = self.placement_streaks.get(ident, [])
            previous = self.objects[ident].get('last_placement_basket')
            if placed:
                streak = old_streak + [frame] if old_streak and old_streak[-1] == frame - 8 and previous == obj['location'] else [frame]
                self.placement_streaks[ident] = streak[-2:]
                self.objects[ident]['last_placement_basket'] = obj['location']
                key = ident + ':' + obj['location']
                self.mapping_witnesses.setdefault(key, {'instance_id': ident, 'basket_color': obj['location'],
                    'first_observed_at': frame, 'category_when_observed': obj['category']})
            else:
                self.placement_streaks[ident] = []
                self.objects[ident]['last_placement_basket'] = None
                if obj['location'] in ('table', 'held') or obj['released'] is False:
                    self.completed.discard(ident)
                    if self.active and self.active['instance_id'] == ident:
                        self.active['last_unplaced_at'] = frame

    def _derive_mappings(self):
        categories, baskets, blocked, ambiguous = {}, {}, set(), set()
        grouped = {}
        for row in self.mapping_witnesses.values():
            obj = self.objects[row['instance_id']]
            if obj['category'] is not None and obj['category_confidence'] == 'clear':
                effective_frame = max(row['first_observed_at'], row.get('category_reinterpreted_at', -1))
                grouped.setdefault(effective_frame, []).append((obj['category'], row['basket_color']))
        for frame in sorted(grouped):
            batch = set(grouped[frame])
            by_basket = {b: {c for c, d in batch if d == b} for b in BASKETS}
            by_category = {c: {b for d, b in batch if d == c} for c, _ in batch}
            for cat, basket in sorted(batch):
                if cat in categories and categories[cat] != basket:
                    blocked.add(basket)
                elif basket in baskets and baskets[basket] != cat:
                    blocked.add(basket)
                elif cat not in categories and len(by_category[cat]) > 1:
                    ambiguous.add(cat); blocked.update(by_category[cat])
                elif basket not in baskets and len(by_basket[basket]) > 1:
                    blocked.add(basket); ambiguous.update(by_basket[basket])
                elif cat not in ambiguous and basket not in blocked:
                    categories[cat], baskets[basket] = basket, cat
        for color, report in self.baskets.items():
            if report['visibility'] != 'clear' or report['unknown_contents']:
                blocked.add(color)
        current_contents = {}
        for obj in self.visible.values():
            if obj['location'] in BASKETS:
                if not clear_category(obj) or obj['released'] is not True:
                    blocked.add(obj['location'])
                else:
                    current_contents.setdefault(obj['location'], set()).add(obj['category'])
        for color, kinds in current_contents.items():
            if len(kinds) > 1 or color in baskets and kinds != {baskets[color]}:
                blocked.add(color)
        self.category_to_basket, self.basket_to_category = categories, baskets
        self.blocked_baskets, self.ambiguous_categories = blocked, ambiguous

    def _update_alias_ambiguity(self, visible):
        for ident, obj in visible.items():
            conflicts = set(self.identity_ambiguities.get(ident, []))
            if ident not in self._previous_objects or not self._previous_objects[ident]['descriptor_ids']:
                conflicts.update(old_id for old_id, old in self._previous_objects.items()
                    if old_id != ident and old_id not in visible
                    and set(old['descriptor_ids']).intersection(self.objects[ident]['descriptor_ids']))
            conflicts.difference_update(k for k, v in visible.items() if clear_identity(v))
            if conflicts:
                self.identity_ambiguities[ident] = sorted(conflicts)
            else:
                self.identity_ambiguities.pop(ident, None)

    def _apply_goal_correction(self, row, visible, held, corrected, prior_categories, frame):
        if row is None:
            return None
        required = {'previous_subtask_id', 'actual_instance_id', 'reason', 'evidence_cameras'}
        require(type(row) is dict and required <= set(row) and set(row) <= required | {'corrected_basket'},
                'Exact explicit goal correction required')
        require(self.active is not None and row['previous_subtask_id'] == self.active['subtask_id'],
                'Goal correction must bind the exact unfinished prior goal')
        cams = explanation(row)
        ident = row['actual_instance_id']
        require(ident in visible and clear_candidate(visible[ident]), 'Goal correction needs one clear current actual instance')
        obj = visible[ident]
        destination = row.get('corrected_basket')
        require(destination is None or destination in BASKETS, 'Corrected basket must be explicit or null')
        different_held = ident != self.active['instance_id'] and ident in held
        identity_fixed = ident == self.active['instance_id'] and ident in corrected
        destination_fixed = False
        if destination is not None:
            require(ident == self.active['instance_id'] or different_held,
                    'Another released instance cannot correct the unfinished active goal destination')
            require(destination != self.active['basket_color'], 'A basket correction must change a contradicted destination')
            prior = prior_categories.get(obj['category'])
            established_mapping = prior == destination and self.category_to_basket.get(obj['category']) == destination
            new_released_mapping = (prior is None and obj['location'] == destination and obj['released'] is True
                                    and self.category_to_basket.get(obj['category']) == destination)
            require(established_mapping or new_released_mapping,
                    'Destination correction needs actual released containment or an established visual category mapping')
            require(len(cams) >= 2, 'Destination correction needs corroborating current views')
            require(destination not in self.blocked_baskets, 'Unknown or conflicting basket contents cannot authorize correction')
            destination_fixed = True
        require(different_held or identity_fixed or destination_fixed,
                'A phase change, another placed instance or motion toward an empty basket cannot unlock a goal')
        if different_held:
            wrists = {'cam_' + name + '_wrist' for name, arm in self.arms.items() if arm['held_instance_id'] == ident}
            require(len(cams) >= 2 and bool(wrists.intersection(cams)),
                    'Changing to another held physical object needs wrist plus corroborating visual evidence')
        event = dict(deepcopy(self.active), interrupted_at=frame, completed=False,
                     correction=deepcopy(row), event='goal_interrupted_by_explicit_current_visual_correction')
        self.interrupted_goals.append(event)
        self.active = None
        return {'instance_id': ident, 'basket_color': destination,
                'allow_contained': destination_fixed and obj['location'] == destination and obj['released'] is True,
                'same_instance': ident == event['instance_id'],
                'previous_basket': event['basket_color'], 'previous_category': event['category']}

    def _new_targets(self, held, correction):
        unknown_hand = any(x['holding'] == 'unknown' for x in self.arms.values())
        if unknown_hand and not held:
            return {}
        eligible = {}
        for ident, raw in self.visible.items():
            obj = self.objects[ident]
            if not clear_candidate(raw) or ident in self.identity_ambiguities:
                continue
            if held and ident not in held:
                continue
            if correction and ident != correction['instance_id']:
                continue
            misplaced = (raw['location'] in BASKETS and raw['released'] is True
                and (self.category_to_basket.get(raw['category'], raw['location']) != raw['location']
                     or self.basket_to_category.get(raw['location'], raw['category']) != raw['category']))
            contained_correction = correction and correction['allow_contained']
            if raw['location'] not in ('table', 'held') and not misplaced and not contained_correction:
                continue
            if ident in self.completed and not contained_correction and not misplaced:
                continue
            eligible[ident] = obj
        owners = {}
        for obj in eligible.values():
            for desc in obj['descriptor_ids']:
                owners.setdefault(desc, []).append(obj)
        current_focus = set()
        for arm in self.arms.values():
            ident = arm.get('interaction_instance_id')
            if (arm['holding'] == 'empty' and arm.get('visibility') == 'clear'
                    and arm.get('relation') in ('approaching', 'aligned', 'enclosing', 'contact')
                    and ident in eligible):
                current_focus.add(ident)
        targets = {}
        for ident, candidate in self.by_id.items():
            objects = owners.get(candidate['descriptor_id'], [])
            if len(objects) > 1:
                focused = [o for o in objects if o['instance_id'] in current_focus]
                if len(focused) == 1:
                    objects = focused
            if len(objects) != 1:
                continue
            obj = objects[0]
            category, basket = obj['category'], candidate['basket_color']
            if category in self.ambiguous_categories or basket in self.blocked_baskets:
                continue
            if self.category_to_basket.get(category, basket) != basket or self.basket_to_category.get(basket, category) != category:
                continue
            if correction:
                required_basket = correction['basket_color']
                if required_basket is None and correction['same_instance']:
                    # Correcting a physical object's description does not by
                    # itself authorize changing its still-compatible destination.
                    established = self.category_to_basket.get(category)
                    if established is not None:
                        required_basket = established
                    elif self.basket_to_category.get(correction['previous_basket'], category) == category:
                        required_basket = correction['previous_basket']
                if required_basket is not None and basket != required_basket:
                    continue
            targets[ident] = {'instance_id': obj['instance_id'], 'category': category, 'basket_color': basket}
        return targets

    def _cleanup_target(self):
        goal = self.last_completed_goal
        if goal is None or goal['instance_id'] not in self.completed:
            return {}
        obj = self.objects[goal['instance_id']]
        if (obj['category'] != goal['category'] or obj['location'] != goal['basket_color']
                or self.by_id[goal['subtask_id']]['descriptor_id'] not in obj['descriptor_ids']
                or self.category_to_basket.get(goal['category']) != goal['basket_color']):
            return {}
        if not all(x['holding'] == 'empty' for x in self.arms.values()):
            return {}
        observed_withdrawal = any(x.get('relation') in ('released', 'retreat')
                                 or x.get('motion_trend') in ('release', 'retract') for x in self.arms.values())
        all_known_contained = (bool(self.objects) and all(ident in self.completed and obj['location'] in BASKETS
            and obj['released'] is True for ident, obj in self.objects.items())
            and not self.blocked_baskets
            and all(row['visibility'] == 'clear' and not row['unknown_contents'] for row in self.baskets.values()))
        if not observed_withdrawal and not all_known_contained:
            return {}
        # Empty-hand observed withdrawal after verified placement, not a default
        # new pickup, all-done assertion, semantic fallback or task-success flag.
        return {goal['subtask_id']: {k: goal[k] for k in ('instance_id', 'category', 'basket_color')}}

    def _observe(self, evidence, *, episode_id, frame, available_frames, executed_records):
        visible, held, corrected = self._validate(evidence, episode_id, frame, available_frames, executed_records)
        previous_arms, previous_visible = deepcopy(self.arms), deepcopy(self.visible)
        self._previous_objects = deepcopy(self.objects)
        prior_categories = dict(self.category_to_basket)
        self.visible, self.arms, self.baskets = deepcopy(visible), deepcopy(evidence['arms']), deepcopy(evidence['baskets'])
        self._merge_objects(visible, corrected, frame)
        self._update_alias_ambiguity(visible)
        self._update_placements(visible, frame)
        self._derive_mappings()
        correction = self._apply_goal_correction(evidence.get('goal_correction'), visible, held, corrected, prior_categories, frame)
        contact_transition = self._known_closing_transition(previous_arms, previous_visible, evidence, frame)
        completion = None
        stop = None
        if self.active is not None:
            active = self.active
            target = visible.get(active['instance_id'])
            category, basket = active['category'], active['basket_color']
            if (self.category_to_basket.get(category, basket) != basket
                    or self.basket_to_category.get(basket, category) != category):
                stop = 'Current visual category/basket facts contradict the locked goal; explicit correction required'
            elif active['instance_id'] in corrected and (self.objects[active['instance_id']]['category'] != category
                    or self.by_id[active['subtask_id']]['descriptor_id'] not in self.objects[active['instance_id']]['descriptor_ids']):
                stop = 'Current identity correction contradicts the locked ID; explicit goal correction required'
            elif target is not None and clear_category(target) and target['category'] != category:
                stop = 'Locked physical instance category changed without goal correction'
            else:
                witnessed = [f for f in self.placement_streaks.get(active['instance_id'], [])
                             if f > max(active['selected_at'], active['last_unplaced_at'])]
                if (target is not None and clear_category(target) and target['location'] == basket
                        and target['released'] is True and len(witnessed) >= 2 and active['instance_id'] not in held):
                    completion = dict(active, confirmed_at=frame, placement_frames=witnessed)
                    self.completed.add(active['instance_id'])
                    self.last_completed_goal = deepcopy(completion)
                    self.active = None
        for ident, obj in visible.items():
            if self.active and self.active['instance_id'] == ident:
                continue
            if (clear_category(obj) and obj['location'] in BASKETS and obj['released'] is True
                    and len(self.placement_streaks.get(ident, [])) >= 2
                    and self.category_to_basket.get(obj['category']) == obj['location']):
                self.completed.add(ident)
        if stop:
            targets, mode = {}, 'correction_required'
        elif self.active is not None:
            if (any(x['holding'] == 'unknown' for x in self.arms.values())
                    and self.active['instance_id'] not in held and contact_transition is None):
                targets, mode, stop = {}, 'unresolved_held_object', 'Unknown held object prevents a table-target action'
            elif held and self.active['instance_id'] not in held:
                targets, mode, stop = {}, 'correction_required', 'A different actual held instance needs explicit visual goal correction'
            else:
                targets = {self.active['subtask_id']: {k: self.active[k] for k in ('instance_id', 'category', 'basket_color')}}
                mode = 'locked_contact_transition' if contact_transition else 'locked'
        else:
            targets = self._new_targets(held, correction)
            mode = 'new_goal' if targets else 'no_legal_target'
            if not targets and correction is None:
                targets = self._cleanup_target()
                if targets:
                    mode = 'cleanup'
        if mode != 'locked_contact_transition':contact_transition = None
        if contact_transition:
            self.closing_transition_streak = contact_transition['uncertainty_windows']
            self.closing_transition_arm = contact_transition['arm']
        else:
            # Empty/unknown label oscillations during the same unconfirmed
            # closure do not create another two-window budget. Reset only for
            # a real visual phase change or a changed/completed physical goal.
            ident = self.active['instance_id'] if self.active else None
            secured = any(a['held_instance_id'] == ident and a.get('object_secured') == 'yes'
                          for a in self.arms.values()) if ident else False
            reopened = any(name == self.closing_transition_arm and a['holding'] == 'empty' and a.get('visibility') == 'clear'
                           and a.get('gripper_state') == 'open'
                           and a.get('interaction_instance_id') == ident
                           and a.get('relation') in ('separated', 'approaching', 'aligned')
                           for name, a in self.arms.items()) if ident else False
            if ident is None or secured or reopened:
                self.closing_transition_streak = 0
                self.closing_transition_arm = None
        self.frame = frame
        self.observation_ledger.append({'frame': frame, 'visual_report_sha256': digest(evidence),
            'actual_input_frame_domain_sha256': digest(sorted(set(available_frames)))})
        self.decision = {'episode_id': episode_id, 'frame': frame, 'mode': mode,
            'locked_id': self.active['subtask_id'] if self.active else None,
            'allowed_ids': sorted(targets), 'targets': targets,
            'category_to_basket': dict(self.category_to_basket), 'basket_to_category': dict(self.basket_to_category),
            'confirmed_completion': completion, 'goal_correction_admitted': deepcopy(correction),
            'known_contact_transition': deepcopy(contact_transition),
            'cleanup_only': mode == 'cleanup', 'stop_reason': stop or 'No clearly identified legal current target',
            'program_substitution': False, 'SDK_success': None}
        self.authorization = None
        del self._previous_objects
        return self.decision

    def select(self, selected_id):
        require(self.decision is not None and selected_id in self.decision['targets'],
                'Selected ID violates the physical goal or local visual constraints')
        if self.decision['mode'] != 'cleanup':
            if self.active is None:
                self.active = dict(self.decision['targets'][selected_id], subtask_id=selected_id,
                    selected_at=self.decision['frame'], last_unplaced_at=self.decision['frame'])
            require(self.active['subtask_id'] == selected_id, 'Locked goal cannot change for an action phase')
        result = dict(deepcopy(self.decision), active_goal=deepcopy(self.active),
            raw_proposed_id=selected_id, actual_selected_id=selected_id,
            scene_state_sha256=digest(self.snapshot()), evidence_is_visual_model_output_not_GT=True)
        self.authorization = deepcopy(result)
        return result

    def validate_injection(self, context, lineage, *, episode_id, frame):
        require(self.authorization is not None and episode_id == self.episode_id
                and frame == self.frame == self.authorization['frame'], 'No current grounded visual authorization')
        require(digest(lineage.get('steady_guard')) == digest(self.authorization), 'Grounded authorization mutated')
        require(digest(self.snapshot()) == self.authorization['scene_state_sha256'], 'Grounded scene state mutated')
        ident = self.authorization['actual_selected_id']
        require(context == {'subtask': self.by_id[ident]['text']}
                and lineage.get('selected_id') == ident and lineage.get('resolution', {}).get('selected_id') == ident,
                'Injection differs from the model-selected exact candidate')
