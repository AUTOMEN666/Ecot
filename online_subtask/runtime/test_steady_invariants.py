"""Task-rule counterexamples and exhaustive finite admission checks, not perception scores."""
from copy import deepcopy
from itertools import permutations
import unittest

from executed_history import digest
from multiview_protocol import load_catalog
from subtask_guard import SubtaskGuard, SubtaskConstraintError
from test_subtask_guard import obj, scene, observe


class SteadyTaskInvariants(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog()
        self.guard = SubtaskGuard(self.catalog)
        self.guard.reset('fixture')

    def test_new_goal_prioritizes_actual_held_object_over_table_pickup(self):
        decision = observe(self.guard, scene(obj(location='held', released=False),
                                             obj('O2', 'camera', 'D003')))
        self.assertEqual(decision['allowed_ids'], ['U0010', 'U0011', 'U0012'])

    def test_unnameable_held_object_cannot_be_hidden_by_a_table_target(self):
        with self.assertRaisesRegex(SubtaskConstraintError, 'No unambiguous legal'):
            observe(self.guard, scene(obj(descriptor=None, location='held', released=False),
                                      obj('O2', 'camera', 'D003')))

    def test_two_identified_held_objects_allow_either_but_no_table_target(self):
        decision = observe(self.guard, scene(obj(location='held', released=False),
            obj('O2', 'camera', 'D003', 'held', released=False), obj('O3', 'pen', 'D005')))
        self.assertEqual(decision['allowed_ids'], ['U0007', 'U0008', 'U0009', 'U0010', 'U0011', 'U0012'])

    def test_same_descriptor_cannot_gain_a_new_category_to_reassign_basket(self):
        before = self.guard.snapshot()
        with self.assertRaisesRegex(SubtaskConstraintError, 'descriptor assigned contradictory'):
            observe(self.guard, scene(obj(location='red', placements=[0]), obj('O2', 'marker', 'D004')))
        self.assertEqual(before, self.guard.snapshot())

    def test_descriptor_category_binding_survives_occlusion_and_new_instance_ID(self):
        observe(self.guard, scene(obj(location='red', placements=[0]), obj('O2', 'camera', 'D003')))
        self.guard.select('U0007')
        before = self.guard.snapshot()
        with self.assertRaisesRegex(SubtaskConstraintError, 'descriptor assigned contradictory'):
            observe(self.guard, scene(obj('O3', 'marker', 'D004', frame=8),
                obj('O2', 'camera', 'D003', 'held', frame=8, released=False)), 8)
        self.assertEqual(before, self.guard.snapshot())

    def test_new_instance_cannot_alias_an_occluded_previously_placed_object(self):
        observe(self.guard, scene(obj(location='red', placements=[0]), obj('O2', 'camera', 'D003')))
        self.guard.select('U0007')
        with self.assertRaisesRegex(SubtaskConstraintError, 'No unambiguous legal'):
            observe(self.guard, scene(obj('O2', 'camera', 'D003', 'white', frame=8, placements=[7, 8]),
                                      obj('O9', 'pen', 'D004', frame=8)), 8)
        self.assertIn('O2', self.guard.completed)
        self.assertIsNone(self.guard.active)

    def test_previously_distinguished_instance_can_use_same_ID_when_another_is_occluded(self):
        observe(self.guard, scene(obj(location='red', placements=[0]), obj('O2', 'pen', 'D004'),
                                 obj('O3', 'camera', 'D003')))
        self.guard.select('U0007')
        decision = observe(self.guard, scene(obj('O3', 'camera', 'D003', 'white', frame=8, placements=[7, 8]),
                                            obj('O2', 'pen', 'D004', frame=8)), 8)
        self.assertEqual(decision['allowed_ids'], ['U0012'])
        self.guard.select('U0012')
        self.assertEqual(self.guard.active['instance_id'], 'O2')

    def test_identity_ambiguity_persists_while_other_legal_goals_are_executed(self):
        watch = next(c['descriptor_id'] for c in self.catalog['candidates'] if 'watch' in c['object_phrase'])
        watch_id = next(c['id'] for c in self.catalog['candidates'] if c['descriptor_id'] == watch and c['basket_color'] == 'blue')
        observe(self.guard, scene(obj(location='red', placements=[0]), obj('O2', 'camera', 'D003')))
        self.guard.select('U0007')
        decision = observe(self.guard, scene(obj('O2', 'camera', 'D003', 'white', frame=8, placements=[7, 8]),
            obj('O9', 'pen', 'D004', frame=8), obj('O3', 'watch', watch, frame=8)), 8)
        self.assertNotIn('U0012', decision['allowed_ids'])
        self.guard.select(watch_id)
        with self.assertRaisesRegex(SubtaskConstraintError, 'No unambiguous legal'):
            observe(self.guard, scene(obj('O3', 'watch', watch, 'blue', frame=16, placements=[15, 16]),
                                      obj('O9', 'pen', 'D004', frame=16)), 16)
        self.assertEqual(self.guard.identity_ambiguities['O9'], ['O1'])

    def test_identity_ambiguity_requires_visible_separation_not_repeated_naming(self):
        observe(self.guard, scene(obj(location='red', placements=[0]), obj('O2', 'camera', 'D003')))
        self.guard.select('U0007')
        observe(self.guard, scene(obj('O2', 'camera', 'D003', 'held', frame=8, released=False),
                                 obj('O9', 'pen', 'D004', frame=8)), 8)
        self.guard.select('U0007')
        decision = observe(self.guard, scene(obj(location='red', frame=16, placements=[16]),
            obj('O2', 'camera', 'D003', 'white', frame=16, placements=[15, 16]),
            obj('O9', 'pen', 'D004', frame=16)), 16)
        self.assertEqual(decision['allowed_ids'], ['U0012'])
        self.assertNotIn('O9', self.guard.identity_ambiguities)

    def test_nested_or_nonstring_descriptor_data_is_a_typed_constraint_stop(self):
        for invalid in ([[]], [{}], [1], [None], ['D004', 'D004']):
            evidence = scene(obj()); evidence['objects'][0]['descriptor_ids'] = invalid
            with self.subTest(value=invalid), self.assertRaises(SubtaskConstraintError):
                observe(self.guard, evidence)
        self.assertEqual(self.guard.objects, {})

    def test_regrasp_inside_basket_also_undoes_instance_completion(self):
        observe(self.guard, scene(obj())); self.guard.select('U0012')
        observe(self.guard, scene(obj(location='red', frame=8, placements=[7, 8]),
                                 obj('O2', 'camera', 'D003', frame=8)), 8)
        self.guard.select('U0007')
        observe(self.guard, scene(obj(location='red', frame=16, released=False),
            obj('O2', 'camera', 'D003', 'held', frame=16, released=False)), 16)
        self.assertNotIn('O1', self.guard.completed)
        self.assertEqual(self.guard.active['subtask_id'], 'U0007')

    def test_authorization_binds_the_entire_scene_evidence(self):
        observe(self.guard, scene(obj())); admission = self.guard.select('U0012')
        context = {'subtask': self.guard.by_id['U0012']['text']}
        proof = dict(steady_guard=admission, selected_id='U0012', resolution={'selected_id': 'U0012'})
        self.guard.objects['O1']['evidence_frames'] = []
        with self.assertRaisesRegex(SubtaskConstraintError, 'Scene evidence changed'):
            self.guard.validate_injection(context, proof, episode_id='fixture', frame=0)

    def test_all_287_alternative_IDs_are_rejected_while_the_goal_is_locked(self):
        observe(self.guard, scene(obj())); self.guard.select('U0012')
        observe(self.guard, scene(obj(location='held', released=False, frame=8)), 8)
        before = digest(self.guard.snapshot())
        for ident in self.guard.by_id:
            if ident == 'U0012':
                continue
            with self.subTest(ident=ident), self.assertRaises(SubtaskConstraintError):
                self.guard.select(ident)
            self.assertEqual(digest(self.guard.snapshot()), before)

    def test_all_basket_permutations_preserve_both_mapping_directions(self):
        watch = next(c['descriptor_id'] for c in self.catalog['candidates'] if 'watch' in c['object_phrase'])
        kinds = [('pen', 'D004'), ('camera', 'D003'), ('watch', watch)]
        for colors in permutations(('white', 'blue', 'red')):
            for unseen in (None, 'white', 'blue', 'red'):
                with self.subTest(colors=colors, unseen=unseen):
                    self.guard.reset('fixture')
                    evidence = scene(*[obj('O' + str(i + 1), category, desc, color, placements=[0])
                        for i, ((category, desc), color) in enumerate(zip(kinds, colors))],
                        *[obj('O' + str(i + 4), category, desc) for i, (category, desc) in enumerate(kinds)])
                    if unseen:
                        evidence['basket_visibility'][unseen] = 'unknown'
                    decision = observe(self.guard, evidence)
                    self.assertEqual(len(decision['allowed_ids']), 3 if unseen is None else 2)
                    for ident in decision['allowed_ids']:
                        target = decision['targets'][ident]
                        self.assertEqual(decision['category_to_basket'][target['category']], target['basket_color'])
                        self.assertEqual(decision['basket_to_category'][target['basket_color']], target['category'])

    def test_registry_cannot_grow_without_bound_by_inventing_new_instance_IDs(self):
        observe(self.guard, scene(obj(), *[obj('O' + str(i), descriptor=None) for i in range(2, 65)]))
        self.guard.select('U0012'); before = deepcopy(self.guard.snapshot())
        with self.assertRaisesRegex(SubtaskConstraintError, 'bounded object limit'):
            observe(self.guard, scene(obj(frame=8), obj('O65', descriptor=None, frame=8)), 8)
        self.assertEqual(self.guard.snapshot(), before)


if __name__ == '__main__':
    unittest.main()
