"""Finite regression cases, not a general English or annotation accuracy proof."""
from copy import deepcopy
import re
import unittest

from negative_action_guard import is_negated
import grounded_motion
from subtask_guard import SubtaskConstraintError


ACTUAL_202_SUMMARY = 'Carry the held beige plush doll toward the white basket.'
ACTUAL_202_REASON = (
    'The left gripper is closed around the beige plush doll and appears to hold it securely; '
    'the white basket is visible ahead, so continue transporting the doll toward the basket '
    'without releasing it.'
)


class NegativeActionGuardTests(unittest.TestCase):
    @staticmethod
    def motion_fixture(*, closing=False):
        def arm(**updates):
            value = dict(holding='empty', held_instance_id=None, gripper_state='open',
                         interaction_instance_id=None, relation='separated', object_secured='no',
                         observed_basket=None, release_ready='no', visibility='clear', motion_trend='stationary')
            value.update(updates)
            return value
        left = arm(holding='identified', held_instance_id='O1', gripper_state='closed',
                   interaction_instance_id='O1', relation='held', object_secured='yes', motion_trend='transport')
        if closing:
            left.update(holding='unknown', held_instance_id=None, gripper_state='transitioning',
                        relation='enclosing', object_secured='no', motion_trend='close')
        evidence = {'objects': [dict(instance_id='O1', category='plush_toy',
                                     location='table' if closing else 'held', released=False)],
                    'arms': {'left': left, 'right': arm()}}
        target = dict(instance_id='O1', category='plush_toy', basket_color='white')
        transition = None
        if closing:
            transition = dict(arm='left', instance_id='O1', allowed_verbs=['close', 'hold'],
                              holding_identity_or_grasp_support_promoted=False,
                              new_goal_or_completion_admitted=False, uncertainty_windows=1)
        return evidence, target, grounded_motion.action_envelopes(evidence, target, contact_transition=transition)

    def check_action(self, text, word, expected, *, occurrence=0):
        matches = list(re.finditer(r'\b' + re.escape(word) + r'\b', text, re.I))
        self.assertGreater(len(matches), occurrence)
        self.assertIs(is_negated(text, matches[occurrence].start()), expected, text)

    def test_actual_202_frame72_negative_release_constraint(self):
        self.check_action(ACTUAL_202_REASON, 'releasing', True)
        self.check_action(ACTUAL_202_REASON, 'transporting', False)
        self.check_action(ACTUAL_202_SUMMARY, 'Carry', False)

    def test_finite_direct_prefixes(self):
        for prefix in (
            'without', 'avoid', 'avoiding', 'never', 'cannot',
            'do not', 'does not', 'did not', 'must not', 'should not',
            'can not', 'could not', 'will not', 'would not', 'shall not',
            "don't", "doesn't", "didn't", "mustn't", "shouldn't", "can't",
            "couldn't", "won't", "wouldn't", "shan't", 'don’t',
        ):
            with self.subTest(prefix=prefix):
                self.check_action(prefix + ' release the pen.', 'release', True)

    def test_do_not_open_and_close_stays_positive(self):
        text = 'Close the left gripper around the pen, and do not open the left gripper.'
        self.check_action(text, 'Close', False)
        self.check_action(text, 'open', True)

    def test_carry_stays_positive_with_negative_release(self):
        text = 'Carry the pen with the left gripper without releasing it.'
        self.check_action(text, 'Carry', False)
        self.check_action(text, 'releasing', True)

    def test_at_most_two_finite_adverbs(self):
        for text in (
            'Do not immediately release the pen.',
            'Avoid accidentally releasing it.',
            'Without ever prematurely releasing it.',
            'NEVER gently fully open the left gripper.',
        ):
            with self.subTest(text=text):
                word = 'open' if 'open' in text else ('releasing' if 'releasing' in text else 'release')
                self.check_action(text, word, True)
        self.check_action('Do not ever gently fully release it.', 'release', False)

    def test_positive_release_and_new_clauses_are_not_suppressed(self):
        for text in (
            'Release the pen with the left gripper.',
            'Do not lift it, then release the pen.',
            'Without lifting it, but release the pen.',
            'Avoid lifting it and release the pen.',
            'Never lift it; release the pen.',
            'Do not lift it. Release the pen.',
        ):
            with self.subTest(text=text):
                self.check_action(text, 'release', False)

    def test_new_clause_can_have_its_own_direct_negative_prefix(self):
        text = 'Do not lift it, but avoid releasing it.'
        self.check_action(text, 'lift', True)
        self.check_action(text, 'releasing', True)

    def test_not_only_is_not_a_direct_negative_constraint(self):
        for text in (
            'Not only release the pen but also retract the arm.',
            'Do not only release the pen.',
            'Never only release the pen.',
        ):
            with self.subTest(text=text):
                self.check_action(text, 'release', False)

    def test_intervening_words_never_extend_scope(self):
        for text in (
            'Without waiting to release it.',
            'Without having to release it.',
            'Avoid trying to release it.',
            'Do not fail to release it.',
            "Don't fail to release it.",
            'Never stop releasing it.',
            'Without necessarily releasing it.',
            'Do not necessarily release it.',
            'Do not, release it.',
            'Without\nreleasing it.',
        ):
            with self.subTest(text=text):
                word = 'releasing' if 'releasing' in text else 'release'
                self.check_action(text, word, False)

    def test_double_negative_and_scope_changing_verbs_are_ambiguous(self):
        for text in (
            'Do not avoid releasing it.',
            "Don't ever avoid releasing it.",
            'Not without releasing it.',
            'Not necessarily without releasing it.',
            'Never avoid releasing it.',
            'Without avoiding release of the pen.',
            'Fail to avoid releasing it.',
            'Stop avoiding release of the pen.',
            "It isn't true that you should not release it.",
        ):
            with self.subTest(text=text):
                word = 'releasing' if 'releasing' in text else 'release'
                self.check_action(text, word, False)

    def test_quotes_are_unparsed_and_do_not_suppress_actions(self):
        for text in (
            '"Do not release it."',
            "'Do not release it.'",
            '“Without releasing it.”',
            '‘Without releasing it.’',
            '`Never release it`',
            'The quoted command is "avoid releasing it".',
            '"Do not release it',
            'Do not release the "pen".',
        ):
            with self.subTest(text=text):
                word = 'releasing' if 'releasing' in text else 'release'
                self.check_action(text, word, False)

    def test_matching_uses_each_actual_action_start(self):
        text = 'Without releasing it, then release it without releasing the grip.'
        self.check_action(text, 'releasing', True, occurrence=0)
        self.check_action(text, 'release', False)
        self.check_action(text, 'releasing', True, occurrence=1)

    def test_invalid_indices_do_not_suppress(self):
        for text, index in (
            (None, 0), ('without release', None), ('without release', True),
            ('without release', -1), ('without release', 15),
            ('without release', 7), ('without release', 9),
            ('without _release', 9), ('without xrelease', 9),
        ):
            with self.subTest(text=text, index=index):
                self.assertIs(is_negated(text, index), False)

    def test_prefix_must_be_a_complete_word(self):
        for text in ('notwithout release it.', 'unavoid releasing it.', 'inever release it.'):
            with self.subTest(text=text):
                self.check_action(text, 'releasing' if 'releasing' in text else 'release', False)

    def test_public_context_and_exact_strings_remain_unchanged(self):
        context = {
            'reasoning': ACTUAL_202_REASON,
            'action': {'summary': ACTUAL_202_SUMMARY},
            'subtask_id': 'U0001',
        }
        before = deepcopy(context)
        reason_bytes = context['reasoning'].encode('utf-8')
        summary_bytes = context['action']['summary'].encode('utf-8')
        self.check_action(context['reasoning'], 'releasing', True)
        self.check_action(context['action']['summary'], 'Carry', False)
        self.assertEqual(context, before)
        self.assertEqual(context['reasoning'].encode('utf-8'), reason_bytes)
        self.assertEqual(context['action']['summary'].encode('utf-8'), summary_bytes)

    def test_real_core_optional_hook_accepts_actual_carry_without_rewrite(self):
        evidence, target, envelope = self.motion_fixture()
        context = {'subtask': 'unchanged exact candidate text', 'reasoning': ACTUAL_202_REASON,
                   'action': {'summary': ACTUAL_202_SUMMARY}}
        before = deepcopy((context, evidence, target, envelope))
        with self.assertRaisesRegex(SubtaskConstraintError, 'unobserved prerequisite: release'):
            grounded_motion.validate_plan(context, envelope, acting_arms={'left'})
        result = grounded_motion.validate_plan(context, envelope, acting_arms={'left'}, negation_detector=is_negated)
        self.assertEqual(result['summary_primary_action'], 'carry')
        self.assertTrue(result['all_public_text_preserved'])
        self.assertEqual((context, evidence, target, envelope), before)

    def test_real_core_optional_hook_keeps_positive_release_rejected(self):
        _, _, envelope = self.motion_fixture()
        for reason in (
            'Continue transporting the doll; release it now.',
            'Continue without releasing it, then release the doll.',
            'Without lifting it, but release the doll.',
            'Do not fail to release it.',
            'The next instruction is "release the doll".',
        ):
            context = {'reasoning': reason, 'action': {'summary': ACTUAL_202_SUMMARY}}
            before = deepcopy(context)
            with self.subTest(reason=reason), self.assertRaises(SubtaskConstraintError):
                grounded_motion.validate_plan(context, envelope, acting_arms={'left'}, negation_detector=is_negated)
            self.assertEqual(context, before)

    def test_real_core_optional_hook_accepts_close_and_rejects_positive_open(self):
        evidence, target, envelope = self.motion_fixture(closing=True)
        context = {'reasoning': 'The left gripper encloses the beige plush doll; do not open the left gripper.',
                   'action': {'summary': 'Close the left gripper around the beige plush doll.'}}
        before = deepcopy((context, evidence, target, envelope))
        with self.assertRaisesRegex(SubtaskConstraintError, 'unsupported open'):
            grounded_motion.validate_plan(context, envelope)
        result = grounded_motion.validate_plan(context, envelope, negation_detector=is_negated)
        self.assertEqual(result['summary_primary_action'], 'close')
        self.assertEqual((context, evidence, target, envelope), before)
        context['reasoning'] = 'Open the left gripper.'
        with self.assertRaisesRegex(SubtaskConstraintError, 'unsupported open'):
            grounded_motion.validate_plan(context, envelope, negation_detector=is_negated)

    def test_real_actor_extractor_uses_each_local_action_start(self):
        text = 'Carry the doll with the left gripper; do not open the left gripper, then release it with the left gripper.'
        old = grounded_motion.explicit_action_actors(text)
        new = grounded_motion.explicit_action_actors(text, negation_detector=is_negated)
        self.assertIn(('open', {'left'}), old)
        self.assertNotIn(('open', {'left'}), new)
        self.assertIn(('carry', {'left'}), new)
        self.assertIn(('release', {'left'}), new)


if __name__ == '__main__':
    unittest.main()
