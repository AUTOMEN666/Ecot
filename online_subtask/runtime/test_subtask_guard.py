"""Adversarial evidence fixtures, not visual recognition or robot success tests."""
from copy import deepcopy
import unittest

from multiview_protocol import load_catalog
from subtask_guard import SubtaskGuard, SubtaskConstraintError


def obj(ident='O1', category='pen', descriptor='D004', location='table', frame=0,
        placements=None, released=True, unplaced=None):
    placements = [] if placements is None else placements
    unplaced = ([frame] if location in ('table', 'held') or not released else []) if unplaced is None else unplaced
    return dict(instance_id=ident, category=category, descriptor_ids=[descriptor] if descriptor else [],
                location=location, released=released,
                evidence_frames=sorted(set(placements + unplaced + [frame])), placement_frames=placements,
                unplaced_frames=unplaced)


def scene(*objects):
    return dict(uncertain=False, basket_visibility=dict.fromkeys(('white', 'blue', 'red'), 'clear'),
                objects=list(objects))


def prefix(frame):
    return [dict(observation_frame=f, executed_action_start=f + 1, executed_action_end=f + 8)
            for f in range(0, frame, 8)]


def observe(guard, evidence, frame=0, frames=None):
    return guard.observe(evidence, episode_id='fixture', frame=frame,
                         available_frames=frames if frames is not None else range(frame + 1),
                         executed_records=prefix(frame))


class GoalAndSortingConstraints(unittest.TestCase):
    def setUp(self):
        self.guard = SubtaskGuard(load_catalog())
        self.guard.reset('fixture')

    def start(self):
        observe(self.guard, scene(obj()))
        self.guard.select('U0012')  # pen -> red; this is a test fixture choice

    def test_initial_goal_is_free_but_a_plan_does_not_assign_a_basket(self):
        decision = observe(self.guard, scene(obj()))
        self.assertEqual(decision['allowed_ids'], ['U0010', 'U0011', 'U0012'])
        self.guard.select('U0012')
        self.assertEqual(self.guard.category_to_basket, {})
        self.assertEqual(self.guard.basket_to_category, {})

    def test_no_jump_across_approach_grasp_carry_and_occlusion(self):
        self.start()
        for frame, evidence in [(8, scene(obj(location='held', frame=8, released=False))),
                                (16, scene()), (24, scene(obj(frame=24)))]:
            decision = observe(self.guard, evidence, frame)
            self.assertEqual(decision['allowed_ids'], ['U0012'])
            with self.assertRaises(SubtaskConstraintError):
                self.guard.select('U0001')
            self.guard.select('U0012')

    def test_pen_in_red_blocks_camera_red_and_requires_other_pen_red(self):
        evidence = scene(obj(location='red', placements=[0]),
                         obj('O2', 'camera', 'D003'), obj('O3', 'pen', 'D005'))
        decision = observe(self.guard, evidence)
        self.assertEqual(decision['category_to_basket'], {'pen': 'red'})
        self.assertIn('U0007', decision['allowed_ids'])  # camera -> white
        self.assertNotIn('U0009', decision['allowed_ids'])  # camera -> red
        self.assertIn('U0015', decision['allowed_ids'])  # other pen -> red
        self.assertNotIn('U0013', decision['allowed_ids'])  # other pen -> white
        with self.assertRaises(SubtaskConstraintError):
            self.guard.select('U0009')

    def test_mixed_basket_is_a_stop_and_does_not_poison_the_ledger(self):
        before = self.guard.snapshot()
        with self.assertRaisesRegex(SubtaskConstraintError, 'Different categories'):
            observe(self.guard, scene(obj(location='red', placements=[0]),
                                     obj('O2', 'camera', 'D003', 'red', placements=[0])))
        self.assertEqual(before, self.guard.snapshot())

    def test_same_category_in_two_baskets_is_rejected(self):
        with self.assertRaisesRegex(SubtaskConstraintError, 'different baskets'):
            observe(self.guard, scene(obj(location='red', placements=[0]),
                                     obj('O2', 'pen', 'D005', 'blue', placements=[0])))

    def test_one_placement_frame_or_grasp_does_not_unlock(self):
        self.start()
        decision = observe(self.guard, scene(obj(location='red', frame=8, placements=[8])), 8)
        self.assertEqual(decision['allowed_ids'], ['U0012'])
        self.guard.select('U0012')
        self.assertIsNone(decision['confirmed_completion'])

    def test_only_the_locked_physical_instance_can_complete(self):
        self.start()
        decision = observe(self.guard, scene(obj(frame=8),
            obj('O2', 'pen', 'D005', 'red', 8, placements=[7, 8])), 8)
        self.assertEqual(decision['allowed_ids'], ['U0012'])
        self.assertIsNone(decision['confirmed_completion'])

    def test_two_post_selection_placement_frames_unlock_for_a_legal_next_goal(self):
        self.start()
        decision = observe(self.guard, scene(obj(location='red', frame=8, placements=[7, 8]),
                                            obj('O2', 'camera', 'D003', frame=8)), 8)
        self.assertEqual(decision['mode'], 'new_goal')
        self.assertEqual(decision['confirmed_completion']['instance_id'], 'O1')
        self.assertNotIn('U0009', decision['allowed_ids'])
        self.guard.select('U0007')
        self.assertEqual(self.guard.active['instance_id'], 'O2')

    def test_drop_back_to_table_remains_unfinished(self):
        self.start()
        decision = observe(self.guard, scene(obj(frame=8)), 8)
        self.assertEqual(decision['mode'], 'locked')
        self.assertNotIn('O1', self.guard.completed)

    def test_clear_descriptor_contradiction_stops_instead_of_silently_locking_a_wrong_object(self):
        self.start()
        with self.assertRaisesRegex(SubtaskConstraintError, 'locked descriptor'):
            observe(self.guard, scene(obj(descriptor='D089', frame=8)), 8)
        self.assertEqual(self.guard.active['subtask_id'], 'U0012')

    def test_same_ID_can_start_a_different_identical_instance_after_supported_completion(self):
        self.start()
        decision = observe(self.guard, scene(obj(location='red', frame=8, placements=[7, 8]),
                                             obj('O2', 'pen', 'D004', frame=8)), 8)
        self.assertEqual(decision['mode'], 'new_goal')
        self.assertEqual(decision['allowed_ids'], ['U0012'])
        self.guard.select('U0012')
        self.assertEqual(self.guard.active['instance_id'], 'O2')

    def test_wrong_destination_is_not_relabelled_as_success_or_new_mapping(self):
        self.start()
        before = self.guard.snapshot()
        with self.assertRaisesRegex(SubtaskConstraintError, 'Active goal contradicts'):
            observe(self.guard, scene(obj(location='blue', frame=8, placements=[7, 8])), 8)
        self.assertEqual(self.guard.snapshot(), before)

    def test_later_drop_clears_instance_completion_but_preserves_category_assignment(self):
        self.start()
        observe(self.guard, scene(obj(location='red', frame=8, placements=[7, 8]),
                                 obj('O2', 'camera', 'D003', frame=8)), 8)
        self.guard.select('U0007')
        observe(self.guard, scene(obj(frame=16), obj('O2', 'camera', 'D003', 'held',
                                                  frame=16, released=False)), 16)
        self.guard.select('U0007')
        self.assertNotIn('O1', self.guard.completed)
        decision = observe(self.guard, scene(obj(frame=24),
            obj('O2', 'camera', 'D003', 'white', 24, placements=[23, 24])), 24)
        self.assertEqual(decision['allowed_ids'], ['U0012'])
        self.assertEqual(decision['category_to_basket'], {'pen': 'red', 'camera': 'white'})

    def test_missing_or_occluded_basket_cannot_be_assumed_empty_for_a_new_goal(self):
        evidence = scene(obj())
        evidence['basket_visibility'] = dict.fromkeys(('white', 'blue', 'red'), 'unknown')
        with self.assertRaisesRegex(SubtaskConstraintError, 'No unambiguous legal'):
            observe(self.guard, evidence)

    def test_ambiguous_matching_instances_are_not_arbitrarily_merged(self):
        with self.assertRaisesRegex(SubtaskConstraintError, 'No unambiguous legal'):
            observe(self.guard, scene(obj(), obj('O2', 'pen', 'D004')))

    def test_uncertain_or_untransmitted_evidence_is_rejected(self):
        uncertain = scene(obj())
        uncertain['uncertain'] = True
        future = scene(obj())
        future['objects'][0]['evidence_frames'] = [0, 8]
        unseen = scene(obj(frame=8))
        unseen['objects'][0]['evidence_frames'] = [6, 8]
        for evidence, frame, frames in [(uncertain, 0, [0]), (future, 0, [0]),
                                        (unseen, 8, [0, 2, 5, 7, 8])]:
            with self.subTest(evidence=evidence), self.assertRaises(SubtaskConstraintError):
                observe(self.guard, evidence, frame, frames)

    def test_category_renaming_and_color_categories_are_rejected(self):
        self.start()
        with self.assertRaisesRegex(SubtaskConstraintError, 'identity/category contradiction'):
            observe(self.guard, scene(obj(category='camera', descriptor='D003', frame=8)), 8)
        with self.assertRaisesRegex(SubtaskConstraintError, 'independent of object color'):
            observe(self.guard, scene(obj(category='red_pen', frame=8)), 8)

    def test_explicit_camera_descriptor_cannot_be_declared_a_pen_to_bypass_sorting(self):
        with self.assertRaisesRegex(SubtaskConstraintError, 'explicit catalogue descriptor'):
            observe(self.guard, scene(obj(category='pen', descriptor='D003')))

    def test_word_families_do_not_merge_distinct_declared_categories(self):
        decision = observe(self.guard, scene(obj(category='marker', descriptor='D018',
                                                location='red', placements=[0]),
                                             obj('O2', 'pen', 'D005')))
        self.assertEqual(decision['category_to_basket'], {'marker': 'red'})
        self.assertNotIn('U0015', decision['allowed_ids'])
        self.assertIn('U0013', decision['allowed_ids'])

    def test_different_toy_categories_cannot_be_folded_into_a_vehicle_class(self):
        decision = observe(self.guard, scene(obj('O1', 'toy_car', 'D032', 'red', placements=[0]),
                                             obj('O2', 'toy_bus', 'D093')))
        self.assertEqual(decision['category_to_basket'], {'toy_car': 'red'})
        self.assertNotIn('U0279', decision['allowed_ids'])
        self.assertIn('U0277', decision['allowed_ids'])

    def test_pre_drop_placement_is_not_reused_after_a_later_observed_drop(self):
        self.start()
        observe(self.guard, scene(obj(location='red', frame=8, placements=[8])), 8)
        self.guard.select('U0012')
        observe(self.guard, scene(obj(frame=16)), 16)
        self.guard.select('U0012')
        decision = observe(self.guard, scene(obj(location='red', frame=24, placements=[7, 24]),
                                             obj('O2', 'camera', 'D003', frame=24)), 24)
        self.assertEqual(decision['mode'], 'locked')
        self.assertIsNone(decision['confirmed_completion'])
        self.guard.select('U0012')
        decision = observe(self.guard, scene(obj(location='red', frame=32, placements=[27, 32]),
                                             obj('O2', 'camera', 'D003', frame=32)), 32)
        self.assertEqual(decision['mode'], 'new_goal')
        self.assertEqual(decision['confirmed_completion']['placement_frames'], [27, 32])

    def test_drop_inside_one_observer_report_also_invalidates_older_placement(self):
        self.start()
        decision = observe(self.guard, scene(obj(location='red', frame=16, placements=[7, 16], unplaced=[10]),
                                             obj('O2', 'camera', 'D003', frame=16)), 16)
        self.assertEqual(decision['mode'], 'locked')
        self.assertEqual(self.guard.active['last_unplaced_at'], 10)
        self.assertIsNone(decision['confirmed_completion'])

    def test_last_goal_completion_is_kept_when_no_next_target_exists(self):
        self.start()
        with self.assertRaisesRegex(SubtaskConstraintError, 'No unambiguous legal'):
            observe(self.guard, scene(obj(location='red', frame=8, placements=[7, 8])), 8)
        self.assertIsNone(self.guard.active)
        self.assertIn('O1', self.guard.completed)
        self.assertEqual(self.guard.decision['mode'], 'no_legal_target')
        self.assertEqual(self.guard.decision['confirmed_completion']['instance_id'], 'O1')
        self.assertEqual(self.guard.decision['allowed_ids'], [])

    def test_one_frame_cannot_be_both_placed_and_unplaced(self):
        with self.assertRaisesRegex(SubtaskConstraintError, 'cannot contradict'):
            observe(self.guard, scene(obj(location='red', placements=[0], unplaced=[0])))

    def test_reset_clears_goal_mapping_instances_and_authorization(self):
        observe(self.guard, scene(obj(location='red', placements=[0]), obj('O2', 'pen', 'D005')))
        self.guard.select('U0015')
        self.guard.reset('next')
        self.assertIsNone(self.guard.active)
        self.assertIsNone(self.guard.authorization)
        self.assertEqual(self.guard.category_to_basket, {})
        self.assertEqual(self.guard.objects, {})


if __name__ == '__main__':
    unittest.main()
