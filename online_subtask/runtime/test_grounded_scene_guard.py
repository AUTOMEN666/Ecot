"""Causal guard counterexamples; these are not perception or native success tests."""
from copy import deepcopy
import unittest

from executed_history import digest
from grounded_scene_guard import GroundedSceneGuard
from multiview_protocol import load_catalog
from subtask_guard import SubtaskConstraintError

CATALOG = load_catalog()
BY_ID = {x['id']: x for x in CATALOG['candidates']}


def candidate(desc='D004', basket='red'):
    return next(x['id'] for x in CATALOG['candidates'] if x['descriptor_id'] == desc and x['basket_color'] == basket)


def obj(ident='O1', category='pen', desc='D004', location='table', released=True,
        identity='clear', category_confidence=None, descriptor_confidence='clear'):
    return {'instance_id': ident, 'identity_confidence': identity, 'category': category,
        'category_confidence': category_confidence or ('clear' if category is not None else 'uncertain'),
        'descriptor_ids': [desc] if isinstance(desc, str) else list(desc or []),
        'descriptor_confidence': descriptor_confidence, 'location': location, 'released': released}


def scene(*objects, left=None, right=None, unknown_left=False, unknown_right=False, relation='stationary'):
    def arm(ident, unknown):
        return {'holding': 'unknown' if unknown else 'identified' if ident else 'empty',
                'held_instance_id': ident, 'relation': relation, 'motion_trend': relation}
    return {'scene_usable': True,
        'baskets': {b: {'visibility': 'clear', 'unknown_contents': False} for b in ('white', 'blue', 'red')},
        'objects': list(objects), 'corrections': [],
        'arms': {'left': arm(left, unknown_left), 'right': arm(right, unknown_right)}}


def records(frame):
    """Explicit synthetic valid-history fixture, never a real policy ACK."""
    ident = candidate()
    return [{'observation_frame': n, 'subtask_id': ident, 'subtask': BY_ID[ident]['text'],
             'executed_action_start': n + 1, 'executed_action_end': n + 8} for n in range(0, frame, 8)]


def observe(g, evidence, frame=0, available_frames=None):
    return g.observe(evidence, episode_id='fixture', frame=frame,
        available_frames=available_frames or [frame], executed_records=records(frame))


def correction(old, **changes):
    return {'instance_id': old['instance_id'], 'previous_category': old['category'],
        'previous_descriptor_ids': old['descriptor_ids'],
        'reason': 'The current wrist image clearly reveals the object identity.',
        'evidence_cameras': ['cam_left_wrist'], **changes}


def goal_fix(previous, actual='O1', basket=None):
    return {'previous_subtask_id': previous, 'actual_instance_id': actual,
        'reason': 'Current wrist and main images clearly contradict the earlier object or destination.',
        'evidence_cameras': ['cam_high', 'cam_right_wrist'], 'corrected_basket': basket}


class GroundedSceneTests(unittest.TestCase):
    def setUp(self):
        self.g = GroundedSceneGuard(CATALOG)
        self.g.reset('fixture')

    def start(self, *others):
        d = observe(self.g, scene(obj(), *others))
        self.assertIn(candidate(), d['allowed_ids'])
        self.g.select(candidate())

    def test_unclear_unrelated_watch_does_not_block_clear_pen(self):
        d = observe(self.g, scene(obj(), obj('O2', 'watch', None,
            identity='uncertain', descriptor_confidence='uncertain')))
        self.assertEqual(d['allowed_ids'], [candidate('D004', b) for b in ('white', 'blue', 'red')])
        self.assertEqual(self.g.objects['O2']['identity_confidence'], 'uncertain')

    def test_unknown_contents_only_blocks_that_basket(self):
        evidence = scene(obj())
        evidence['baskets']['white']['unknown_contents'] = True
        d = observe(self.g, evidence)
        self.assertNotIn(candidate('D004', 'white'), d['allowed_ids'])
        self.assertIn(candidate('D004', 'red'), d['allowed_ids'])
        self.assertIn(candidate('D004', 'blue'), d['allowed_ids'])

    def test_unknown_held_object_prevents_table_target(self):
        with self.assertRaisesRegex(SubtaskConstraintError, 'No clearly identified legal'):
            observe(self.g, scene(obj(), unknown_right=True))
        self.assertEqual(self.g.decision['allowed_ids'], [])
        self.assertIsNone(self.g.decision['SDK_success'])

    def test_held_object_cannot_be_hidden_behind_empty_hands(self):
        before = self.g.snapshot()
        with self.assertRaisesRegex(SubtaskConstraintError, 'hidden behind'):
            observe(self.g, scene(obj(location='held', released=False)))
        self.assertEqual(before, self.g.snapshot())

    def test_actual_held_instance_precedes_another_table_pickup(self):
        d = observe(self.g, scene(obj(), obj('O2', 'camera', 'D003', 'held', released=False), right='O2'))
        self.assertTrue(d['allowed_ids'])
        self.assertTrue(all(x['instance_id'] == 'O2' for x in d['targets'].values()))

    def test_phase_change_or_occlusion_does_not_change_the_macro_ID(self):
        self.start()
        for frame, evidence in [(8, scene(obj(location='held', released=False), left='O1')),
                                (16, scene()), (24, scene(obj()))]:
            d = observe(self.g, evidence, frame)
            self.assertEqual(d['allowed_ids'], [candidate()])
            self.g.select(candidate())

    def test_program_stamps_actual_queries_without_LLM_historical_frames(self):
        self.start()
        observe(self.g, scene(obj()), 8, [0, 2, 5, 7, 8]); self.g.select(candidate())
        observe(self.g, scene(obj()), 16, [0, 2, 5, 7, 10, 12, 15, 16]); self.g.select(candidate())
        d = observe(self.g, scene(obj()), 24, [0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 24])
        self.assertEqual(d['allowed_ids'], [candidate()])
        self.assertEqual([x['frame'] for x in self.g.observation_ledger], [0, 8, 16, 24])

    def test_two_post_selection_current_placements_confirm_same_instance(self):
        self.start()
        d = observe(self.g, scene(obj(location='red'), relation='released'), 8)
        self.assertIsNone(d['confirmed_completion']); self.g.select(candidate())
        d = observe(self.g, scene(obj(location='red'), relation='retreat'), 16)
        self.assertEqual(d['confirmed_completion']['placement_frames'], [8, 16])
        self.assertIn('O1', self.g.completed)
        self.assertEqual(d['mode'], 'cleanup')
        self.assertEqual(d['allowed_ids'], [candidate()])

    def test_another_similar_placed_instance_does_not_complete_active_object(self):
        self.start(obj('O2', 'pen', 'D005'))
        for frame in (8, 16):
            d = observe(self.g, scene(obj(), obj('O2', 'pen', 'D005', 'red')), frame)
            self.assertEqual(d['allowed_ids'], [candidate()])
            self.assertIsNone(d['confirmed_completion'])
            self.g.select(candidate())
        self.assertNotIn('O1', self.g.completed)

    def test_drop_breaks_placement_streak(self):
        self.start()
        for frame, location in [(8, 'red'), (16, 'table'), (24, 'red')]:
            d = observe(self.g, scene(obj(location=location), relation='released'), frame)
            self.assertIsNone(d['confirmed_completion']); self.g.select(candidate())
        d = observe(self.g, scene(obj(location='red'), relation='retreat'), 32)
        self.assertEqual(d['confirmed_completion']['placement_frames'], [24, 32])

    def test_occlusion_does_not_fabricate_a_second_placement_witness(self):
        self.start()
        for frame, evidence in [(8, scene(obj(location='red'))), (16, scene()), (24, scene(obj(location='red')))]:
            d = observe(self.g, evidence, frame)
            self.assertIsNone(d['confirmed_completion']); self.g.select(candidate())

    def test_basket_unknown_does_not_redirect_unfinished_held_goal(self):
        self.start()
        evidence = scene(obj(location='held', released=False), left='O1')
        evidence['baskets']['red']['visibility'] = 'unknown'
        d = observe(self.g, evidence, 8)
        self.assertEqual(d['allowed_ids'], [candidate()])
        self.assertEqual(d['mode'], 'locked')

    def test_generic_descriptor_is_not_a_global_category_binding(self):
        generic = next(x['descriptor_id'] for x in CATALOG['candidates']
                       if x['object_phrase'] == 'black device')
        d = observe(self.g, scene(obj('O1', 'camera', [generic, 'D003']), obj('O2', 'block', generic)))
        self.assertIn(candidate('D003', 'red'), d['allowed_ids'])
        self.assertNotIn(candidate(generic, 'red'), d['allowed_ids'])
        self.assertNotIn('descriptor_to_category', self.g.snapshot())

    def test_distinct_table_instances_with_only_same_descriptor_are_not_guessed(self):
        with self.assertRaisesRegex(SubtaskConstraintError, 'No clearly identified legal'):
            observe(self.g, scene(obj(), obj('O2', desc='D004')))

    def test_held_instance_disambiguates_same_description_from_table_instance(self):
        d = observe(self.g, scene(obj(location='held', released=False), obj('O2'), left='O1'))
        self.assertTrue(d['allowed_ids'])
        self.assertTrue(all(x['instance_id'] == 'O1' for x in d['targets'].values()))

    def test_one_clear_current_interaction_resolves_same_descriptor_without_host_guess(self):
        evidence = scene(obj(), obj('O2'))
        evidence['arms']['left'].update(visibility='clear', relation='approaching', interaction_instance_id='O2')
        d = observe(self.g, evidence)
        self.assertTrue(d['allowed_ids'])
        self.assertTrue(all(x['instance_id'] == 'O2' for x in d['targets'].values()))
        self.g.select(candidate())
        for frame in [8, 16]:
            evidence = scene(obj(location='red'), obj('O2'))
            evidence['arms']['left'].update(visibility='clear', relation='approaching', interaction_instance_id='O2')
            d = observe(self.g, evidence, frame)
            self.assertEqual(d['allowed_ids'], [candidate()])
            self.assertEqual(self.g.active['instance_id'], 'O2')
            self.assertIsNone(d['confirmed_completion'])
            self.g.select(candidate())
        self.assertNotIn('O2', self.g.completed)

    def test_two_different_current_focus_owners_remain_ambiguous(self):
        evidence = scene(obj(), obj('O2'))
        evidence['arms']['left'].update(visibility='clear', relation='approaching', interaction_instance_id='O1')
        evidence['arms']['right'].update(visibility='clear', relation='aligned', interaction_instance_id='O2')
        with self.assertRaisesRegex(SubtaskConstraintError, 'No clearly identified legal'):
            observe(self.g, evidence)

    def test_occluded_or_stationary_focus_claim_cannot_resolve_ambiguous_owners(self):
        for visibility, relation in [('occluded', 'approaching'), ('clear', 'stationary'), ('clear', 'released')]:
            g = GroundedSceneGuard(CATALOG); g.reset('fixture')
            evidence = scene(obj(), obj('O2'))
            evidence['arms']['left'].update(visibility=visibility, relation=relation, interaction_instance_id='O2')
            with self.subTest(visibility=visibility, relation=relation), self.assertRaises(SubtaskConstraintError):
                observe(g, evidence)

    def test_unrelated_explicit_visual_identity_correction_is_local(self):
        old = obj('O2', 'block', 'D006')
        self.start(old)
        new = obj('O2', 'camera', 'D003')
        evidence = scene(obj(), new); evidence['corrections'] = [correction(old)]
        d = observe(self.g, evidence, 8)
        self.assertEqual(d['allowed_ids'], [candidate()])
        self.assertEqual(self.g.objects['O2']['category'], 'camera')
        self.assertEqual(self.g.correction_log[-1]['frame'], 8)

    def test_missing_identity_correction_is_not_silently_applied(self):
        self.start(obj('O2', 'block', 'D006'))
        before = self.g.snapshot()
        with self.assertRaisesRegex(SubtaskConstraintError, 'explicit same-instance correction'):
            observe(self.g, scene(obj(), obj('O2', 'camera', 'D003')), 8)
        self.assertEqual(before, self.g.snapshot())

    def test_newly_identified_unrelated_object_can_change_uncertain_label_without_fake_correction(self):
        self.start(obj('O2', None, None))
        d = observe(self.g, scene(obj(), obj('O2', 'camera', 'D003')), 8)
        self.assertEqual(d['allowed_ids'], [candidate()])
        self.assertEqual(self.g.objects['O2']['category'], 'camera')

    def test_different_real_held_instance_needs_explicit_goal_correction(self):
        self.start(obj('O2', 'camera', 'D003'))
        with self.assertRaisesRegex(SubtaskConstraintError, 'different actual held'):
            observe(self.g, scene(obj(), obj('O2', 'camera', 'D003', 'held', released=False), right='O2'), 8)

    def test_explicit_held_object_goal_correction_reopens_exact_model_choice(self):
        self.start(obj('O2', 'camera', 'D003'))
        evidence = scene(obj(), obj('O2', 'camera', 'D003', 'held', released=False), right='O2')
        evidence['goal_correction'] = goal_fix(candidate(), actual='O2')
        d = observe(self.g, evidence, 8)
        self.assertEqual(d['mode'], 'new_goal')
        self.assertTrue(all(x['instance_id'] == 'O2' for x in d['targets'].values()))
        self.assertEqual(len(self.g.interrupted_goals), 1)
        self.assertIs(self.g.interrupted_goals[0]['completed'], False)
        self.assertNotIn('O1', self.g.completed)
        chosen = candidate('D003', 'white'); self.g.select(chosen)
        self.assertEqual(self.g.active['subtask_id'], chosen)

    def test_same_instance_identity_correction_preserves_compatible_destination(self):
        old = obj('O1', 'block', 'D006')
        observe(self.g, scene(old)); previous = candidate('D006', 'red'); self.g.select(previous)
        evidence = scene(obj('O1', 'camera', 'D003', 'held', released=False), left='O1')
        evidence['corrections'] = [correction(old)]
        evidence['goal_correction'] = goal_fix(previous)
        d = observe(self.g, evidence, 8)
        self.assertEqual(d['allowed_ids'], [candidate('D003', 'red')])
        self.assertTrue(d['goal_correction_admitted']['same_instance'])
        self.assertEqual(self.g.interrupted_goals[-1]['completed'], False)

    def test_same_instance_identity_correction_uses_prior_independent_class_mapping(self):
        old = obj('O1', 'block', 'D006')
        observe(self.g, scene(old, obj('O2', 'camera', 'D003', 'blue')))
        previous = candidate('D006', 'red'); self.g.select(previous)
        evidence = scene(obj('O1', 'camera', 'D003', 'held', released=False),
                         obj('O2', 'camera', 'D003', 'blue'), left='O1')
        evidence['corrections'] = [correction(old)]
        evidence['goal_correction'] = goal_fix(previous)
        d = observe(self.g, evidence, 8)
        self.assertEqual(d['allowed_ids'], [candidate('D003', 'blue')])

    def test_another_placed_object_or_action_phase_cannot_unlock_goal(self):
        self.start(obj('O2', 'camera', 'D003'))
        evidence = scene(obj(), obj('O2', 'camera', 'D003', 'blue'))
        evidence['goal_correction'] = goal_fix(candidate(), actual='O2')
        with self.assertRaisesRegex(SubtaskConstraintError, 'phase change'):
            observe(self.g, evidence, 8)

    def test_another_placed_object_cannot_use_destination_correction_to_unlock_goal(self):
        self.start(obj('O2', 'camera', 'D003'))
        evidence = scene(obj(), obj('O2', 'camera', 'D003', 'blue'))
        evidence['goal_correction'] = goal_fix(candidate(), actual='O2', basket='blue')
        with self.assertRaisesRegex(SubtaskConstraintError, 'Another released instance'):
            observe(self.g, evidence, 8)

    def test_new_category_interpretation_cannot_override_prior_independent_visual_mapping(self):
        old = obj('O2', 'block', 'D006', 'white')
        evidence = scene(obj(), old)
        observe(self.g, evidence); self.g.select(candidate())
        observe(self.g, scene(obj(), old, obj('O3', 'camera', 'D003', 'blue')), 8); self.g.select(candidate())
        new = obj('O2', 'camera', 'D003', 'white')
        evidence = scene(obj(), new, obj('O3', 'camera', 'D003', 'blue'))
        evidence['corrections'] = [correction(old)]
        observe(self.g, evidence, 16)
        self.assertEqual(self.g.category_to_basket['camera'], 'blue')
        self.assertIn('white', self.g.blocked_baskets)

    def test_hovering_over_another_empty_basket_is_not_destination_correction(self):
        self.start()
        evidence = scene(obj(location='held', released=False), left='O1', relation='over_basket')
        evidence['goal_correction'] = goal_fix(candidate(), basket='blue')
        with self.assertRaisesRegex(SubtaskConstraintError, 'released containment or an established'):
            observe(self.g, evidence, 8)

    def test_explicit_released_destination_correction_requires_no_prior_opposing_mapping(self):
        self.start()
        evidence = scene(obj(location='blue'), relation='released')
        evidence['goal_correction'] = goal_fix(candidate(), basket='blue')
        d = observe(self.g, evidence, 8)
        self.assertEqual(d['allowed_ids'], [candidate('D004', 'blue')])
        self.assertEqual(self.g.category_to_basket, {'pen': 'blue'})
        self.assertIsNone(self.g.active)
        self.assertEqual(self.g.interrupted_goals[-1]['basket_color'], 'red')
        self.g.select(candidate('D004', 'blue'))
        self.assertEqual(self.g.active['subtask_id'], candidate('D004', 'blue'))

    def test_wrong_placement_cannot_rewrite_established_category_mapping(self):
        self.start(obj('O2', 'pen', 'D005', 'red'))
        evidence = scene(obj(location='blue'), obj('O2', 'pen', 'D005', 'red'))
        evidence['goal_correction'] = goal_fix(candidate(), basket='blue')
        with self.assertRaisesRegex(SubtaskConstraintError, 'released containment or an established'):
            observe(self.g, evidence, 8)
        self.assertEqual(self.g.category_to_basket, {'pen': 'red'})

    def test_mapping_is_never_created_by_selected_text(self):
        self.start()
        self.assertEqual(self.g.category_to_basket, {})
        self.assertEqual(self.g.basket_to_category, {})

    def test_simultaneous_mixed_basket_does_not_choose_mapping_by_ID_order(self):
        evidence = scene(obj('O1', 'pen', 'D004', 'red'), obj('O2', 'camera', 'D003', 'red'),
                         obj('O3', 'watch', 'D067'))
        d = observe(self.g, evidence)
        self.assertNotIn('red', self.g.basket_to_category)
        self.assertIn('red', self.g.blocked_baskets)
        self.assertTrue(all(x['basket_color'] != 'red' for x in d['targets'].values()))

    def test_cleanup_requires_verified_placement_and_known_empty_hands(self):
        self.start()
        observe(self.g, scene(obj(location='red'), relation='released'), 8); self.g.select(candidate())
        d = observe(self.g, scene(obj(location='red'), relation='retreat'), 16)
        self.assertEqual(d['mode'], 'cleanup')
        admission = self.g.select(candidate())
        lineage = {'steady_guard': admission, 'selected_id': candidate(), 'resolution': {'selected_id': candidate()}}
        self.g.validate_injection({'subtask': BY_ID[candidate()]['text']}, lineage, episode_id='fixture', frame=16)
        self.assertIsNone(self.g.active)

    def test_fully_verified_final_rest_can_keep_last_ID_without_new_pickup(self):
        self.start()
        observe(self.g, scene(obj(location='red'), relation='released'), 8); self.g.select(candidate())
        observe(self.g, scene(obj(location='red'), relation='retreat'), 16); self.g.select(candidate())
        d = observe(self.g, scene(obj(location='red'), relation='stationary'), 24)
        self.assertEqual(d['mode'], 'cleanup')
        self.assertIsNone(d['SDK_success'])

    def test_unidentified_remaining_table_object_cannot_be_masked_by_stationary_cleanup(self):
        unknown = obj('O2', None, None)
        self.start(unknown)
        observe(self.g, scene(obj(location='red'), unknown, relation='released'), 8); self.g.select(candidate())
        observe(self.g, scene(obj(location='red'), unknown, relation='retreat'), 16); self.g.select(candidate())
        with self.assertRaisesRegex(SubtaskConstraintError, 'No clearly identified legal'):
            observe(self.g, scene(obj(location='red'), unknown), 24)

    def test_later_drop_cancels_completion_and_cleanup(self):
        self.start()
        observe(self.g, scene(obj(location='red'), relation='released'), 8); self.g.select(candidate())
        observe(self.g, scene(obj(location='red'), relation='retreat'), 16); self.g.select(candidate())
        d = observe(self.g, scene(obj()), 24)
        self.assertNotIn('O1', self.g.completed)
        self.assertEqual(d['mode'], 'new_goal')
        self.assertEqual(d['allowed_ids'], [candidate()])

    def test_exact_history_future_frame_episode_and_query_order_are_checked(self):
        for change in ('future', 'episode', 'history', 'order'):
            g = GroundedSceneGuard(CATALOG); g.reset('fixture')
            kwargs = dict(episode_id='fixture', frame=0, available_frames=[0], executed_records=[])
            if change == 'future': kwargs['available_frames'] = [0, 8]
            if change == 'episode': kwargs['episode_id'] = 'other'
            if change == 'history': kwargs['executed_records'] = records(8)
            if change == 'order': kwargs.update(frame=8, available_frames=[8], executed_records=records(8))
            with self.subTest(change=change), self.assertRaises(SubtaskConstraintError):
                g.observe(scene(obj()), **kwargs)
            self.assertIsNone(g.frame)

    def test_explicit_lexical_contradiction_and_global_uncertainty_schema_are_rejected(self):
        for evidence in (scene(obj(category='pen', desc='D003')), dict(scene(obj()), uncertain=False)):
            with self.assertRaises(SubtaskConstraintError): observe(self.g, evidence)

    def test_source_binding_and_model_selected_exact_ID_are_authorized(self):
        self.start()
        admission = deepcopy(self.g.authorization)
        lineage = {'steady_guard': admission, 'selected_id': candidate(), 'resolution': {'selected_id': candidate()}}
        self.g.validate_injection({'subtask': BY_ID[candidate()]['text']}, lineage, episode_id='fixture', frame=0)
        self.g.arms['left']['holding'] = 'unknown'
        with self.assertRaisesRegex(SubtaskConstraintError, 'scene state mutated'):
            self.g.validate_injection({'subtask': BY_ID[candidate()]['text']}, lineage, episode_id='fixture', frame=0)

    def test_reset_clears_all_causal_state(self):
        self.start(); self.g.reset('next')
        self.assertIsNone(self.g.active)
        self.assertIsNone(self.g.frame)
        self.assertEqual(self.g.objects, {})
        self.assertEqual(self.g.placement_streaks, {})
        self.assertEqual(self.g.observation_ledger, [])


if __name__ == '__main__': unittest.main(verbosity=2)
