"""V5 actual-terminal regressions and causal motion counterexamples; CPU only."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import grounded_motion as legacy
from grounded_motion_v5 import action_envelopes,unique_secured_actor,validate_plan
from phase_grounded_ssr_protocol import CONFIG_ID,NAMESPACE,MOTION_INTERPRETATION,validate_spec
from phase_grounded_ssr_provider import PhaseGroundedSSRProvider
from enum_grounded_ssr_provider import EnumGroundedSSRProvider
from multiview_protocol import INSTRUCTION,canonical,load_catalog,strict_json
from subtask_guard import SubtaskConstraintError

CATALOG={row['id']:row['text']for row in load_catalog()['candidates']}


def arm(**updates):
    value=dict(holding='empty',held_instance_id=None,gripper_state='open',interaction_instance_id=None,
        relation='unknown',object_secured='no',observed_basket=None,release_ready='no',visibility='clear',motion_trend='stationary')
    value.update(updates);return value


def obj(ident,category,location,descriptor,released=None):
    return dict(instance_id=ident,identity_confidence='clear',category=category,category_confidence='clear',
        descriptor_ids=[descriptor],descriptor_confidence='clear',location=location,released=released)


def actual(episode):
    """Verbatim PUBLIC text; current motion facts from real v4 terminal rows."""
    if episode==201:
        summary='Carry the grasped black-and-gold pen toward the white basket to prepare placement.'
        reasoning='The left gripper is closed and holding O1 securely, but white-basket alignment is not yet confirmed; keep the grasp and continue transporting the pen toward the white basket.'
        ident='U0115';category='pen'
        objects=[obj('O1','pen','held','D039',False),obj('O5','watch','table','D070')]
        arms={'left':arm(holding='identified',held_instance_id='O1',gripper_state='closed',interaction_instance_id='O1',
            motion_trend='transport',object_secured='yes',relation='held'),
            'right':arm(interaction_instance_id='O5',relation='separated')}
    elif episode==202:
        summary='Release the held beige plush doll into the white basket by opening the left gripper.'
        reasoning='The left gripper remains closed around O1 above the white basket; current wrist and high views show the doll over the basket, so opening the jaws is the next feasible release adjustment.'
        ident='U0001';category='plush_toy'
        objects=[obj('O1','plush_toy','held','D001',False)]
        arms={'left':arm(holding='identified',held_instance_id='O1',gripper_state='closed',interaction_instance_id='O1',
            motion_trend='lower',object_secured='yes',observed_basket='white',relation='over_basket',release_ready='yes'),'right':arm()}
    elif episode==203:
        summary='Hold the left gripper open and refine alignment over the green watch.'
        reasoning='The left gripper is empty and open near O1, but the green watch remains separated and unenclosed. Keep the jaws open and refine the left-arm alignment over the watch.'
        ident='U0208';category='watch'
        objects=[obj('O1','watch','table','D070')]
        arms={'left':arm(gripper_state='unknown',interaction_instance_id='O1',motion_trend='approach',relation='approaching'),
            'right':arm(gripper_state='unknown',motion_trend='unknown')}
    else:raise ValueError('Only actual fixed v4 terminal regressions')
    context={'subtask':CATALOG[ident],'action':{'summary':summary},'reasoning':reasoning}
    evidence={'scene_usable':True,'objects':objects,'arms':arms,
        'baskets':{x:{'visibility':'clear','unknown_contents':False}for x in('white','blue','red')},'corrections':[],'goal_correction':None}
    return context,evidence,dict(instance_id='O1',category=category,basket_color='white')


class PhaseMotionTests(unittest.TestCase):
    def test_actual201_actor_is_derived_without_public_rewrite_v4_still_rejects(self):
        context,evidence,target=actual(201);before=deepcopy((context,evidence,target))
        with self.assertRaisesRegex(SubtaskConstraintError,'identify its acting'):
            legacy.validate_plan(context,legacy.action_envelopes(evidence,target))
        envelope=action_envelopes(evidence,target);result=validate_plan(context,envelope)
        self.assertEqual(result['acting_arms'],['left'])
        self.assertEqual(result['actor_override_from_current_evidence'],['left'])
        self.assertEqual(result['motion_interpretation'],MOTION_INTERPRETATION)
        self.assertEqual((context,evidence,target),before)
        self.assertTrue(result['all_public_text_preserved'])

    def test_actual202_release_physically_entails_open_v4_still_rejects(self):
        context,evidence,target=actual(202);before=deepcopy((context,evidence,target))
        old=legacy.action_envelopes(evidence,target)
        with self.assertRaisesRegex(SubtaskConstraintError,'unsupported open'):
            legacy.validate_plan(context,old)
        envelope=action_envelopes(evidence,target)
        self.assertEqual(set(envelope['by_arm']['left']['allowed_verbs'])-set(old['by_arm']['left']['allowed_verbs']),{'open'})
        self.assertEqual(envelope['by_arm']['right'],old['by_arm']['right'])
        self.assertEqual(envelope['release_open_entailments'],['left'])
        self.assertEqual(validate_plan(context,envelope)['acting_arms'],['left'])
        self.assertEqual((context,evidence,target),before)

    def test_actual203_unknown_gripper_open_claim_still_rejected(self):
        context,evidence,target=actual(203)
        for make,check in[(legacy.action_envelopes,legacy.validate_plan),(action_envelopes,validate_plan)]:
            with self.assertRaisesRegex(SubtaskConstraintError,'unsupported open'):
                check(context,make(evidence,target))

    def test_actor_requires_exact_single_reasoning_hand_and_current_target(self):
        context,evidence,target=actual(201)
        for reasoning in['Keep the grasp and carry the pen toward the white basket.',
                'The right gripper securely holds the pen; carry toward the white basket.',
                'The left gripper is closed; the right gripper remains open while transporting.']:
            with self.subTest(reasoning=reasoning):
                altered=deepcopy(context);altered['reasoning']=reasoning
                self.assertIsNone(unique_secured_actor(altered,action_envelopes(evidence,target)))
                with self.assertRaises(SubtaskConstraintError):validate_plan(altered,action_envelopes(evidence,target))
        evidence['objects'].append(obj('O2','pen','table','D089'))
        wrong=dict(target,instance_id='O2')
        with self.assertRaises(SubtaskConstraintError):validate_plan(context,action_envelopes(evidence,wrong))

    def test_actor_never_defaults_for_non_supported_primary_phases(self):
        context,evidence,target=actual(201)
        for word in['Approach','Align','Hold','Open','Close','Retract']:
            altered=deepcopy(context);altered['action']['summary']=word+' the target carefully.'
            with self.subTest(word=word),self.assertRaises(SubtaskConstraintError):
                validate_plan(altered,action_envelopes(evidence,target))

    def test_explicit_wrong_hand_or_both_cannot_be_replaced_by_reasoning(self):
        context,evidence,target=actual(201)
        for suffix in['with the right gripper.','with both grippers.']:
            altered=deepcopy(context);altered['action']['summary']='Carry the pen toward the white basket '+suffix
            self.assertIsNone(unique_secured_actor(altered,action_envelopes(evidence,target)))
            with self.assertRaises(SubtaskConstraintError):validate_plan(altered,action_envelopes(evidence,target))

    def test_unparsed_summary_side_word_cannot_be_silently_replaced_by_left(self):
        context,evidence,target=actual(201)
        for words in['with the right-arm','with the right-hand','using both-hands','to the right side']:
            altered=deepcopy(context);altered['action']['summary']='Carry the pen toward the white basket '+words+'.'
            with self.subTest(words=words):
                self.assertIsNone(unique_secured_actor(altered,action_envelopes(evidence,target)))
                with self.assertRaises(SubtaskConstraintError):validate_plan(altered,action_envelopes(evidence,target))

    def test_reasoning_hyphenated_other_hand_still_counts_against_single_actor(self):
        context,evidence,target=actual(201)
        context['reasoning']+=' The right-arm stays still.'
        self.assertIsNone(unique_secured_actor(context,action_envelopes(evidence,target)))
        with self.assertRaises(SubtaskConstraintError):validate_plan(context,action_envelopes(evidence,target))

    def test_unknown_not_secure_open_transition_or_occluded_target_cannot_supply_actor(self):
        mutations=[{'object_secured':'no'},
            {'object_secured':'unknown'},
            {'object_secured':'no','gripper_state':'open'},
            {'object_secured':'no','gripper_state':'transitioning'},
            {'object_secured':'unknown','visibility':'occluded'},
            {'object_secured':'no','holding':'unknown','held_instance_id':None},
            {'object_secured':'no','holding':'empty','held_instance_id':None}]
        for mutation in mutations:
            context,evidence,target=actual(201);evidence['arms']['left'].update(mutation)
            with self.subTest(mutation=mutation),self.assertRaises(SubtaskConstraintError):
                validate_plan(context,action_envelopes(evidence,target))

    def test_any_second_secure_holder_blocks_an_omitted_actor(self):
        for same_instance in[True,False]:
            context,evidence,target=actual(201)
            ident='O1'if same_instance else'O2'
            if not same_instance:evidence['objects'].append(obj('O2','pen','held','D089',False))
            evidence['arms']['right']=arm(holding='identified',held_instance_id=ident,gripper_state='closed',
                interaction_instance_id=ident,object_secured='yes',relation='held')
            with self.subTest(same_instance=same_instance),self.assertRaises(SubtaskConstraintError):
                validate_plan(context,action_envelopes(evidence,target))

    def test_inferred_actor_does_not_grant_later_lower_release_phase(self):
        context,evidence,target=actual(201)
        for word in['Lower','Release']:
            altered=deepcopy(context);altered['action']['summary']=word+' the pen into the white basket.'
            with self.subTest(word=word),self.assertRaisesRegex(SubtaskConstraintError,'not visually admitted'):
                validate_plan(altered,action_envelopes(evidence,target))

    def test_all_four_omitted_actor_actions_require_their_current_phase(self):
        for word,episode in[('Lift',201),('Carry',201),('Lower',202),('Release',202)]:
            context,evidence,target=actual(episode)
            noun='black-and-gold pen'if episode==201 else'beige plush doll'
            context['action']['summary']=word+' the securely held '+noun+' toward the white basket.'
            with self.subTest(word=word):
                receipt=validate_plan(context,action_envelopes(evidence,target))
                self.assertEqual(receipt['acting_arms'],['left'])
                self.assertEqual(receipt['actor_override_from_current_evidence'],['left'])

    def test_known_closing_transition_cannot_be_promoted_to_secure_actor_or_release_open(self):
        _,evidence,target=actual(203)
        evidence['arms']['left'].update(holding='unknown',gripper_state='transitioning',relation='enclosing',motion_trend='close')
        evidence['arms']['right']=arm()
        transition={'arm':'left','instance_id':'O1','allowed_verbs':['close','hold'],
            'holding_identity_or_grasp_support_promoted':False,'new_goal_or_completion_admitted':False,'uncertainty_windows':1}
        envelope=action_envelopes(evidence,target,contact_transition=transition)
        self.assertEqual(envelope['by_arm']['left']['allowed_verbs'],['close','hold'])
        self.assertEqual(envelope['by_arm']['right']['allowed_verbs'],['hold'])
        self.assertEqual(envelope['release_open_entailments'],[])
        context={'subtask':CATALOG['U0208'],'reasoning':'The known watch remains enclosed while the left gripper continues closing; support is not yet established.',
            'action':{'summary':'Close the left gripper around the enclosed watch to continue closure.'}}
        self.assertEqual(validate_plan(context,envelope)['summary_primary_action'],'close')
        for summary in['Carry the watch toward the white basket.','Open the left gripper to release the watch.']:
            context['action']['summary']=summary
            with self.subTest(summary=summary),self.assertRaises(SubtaskConstraintError):validate_plan(context,envelope)

    def test_open_is_added_only_where_base_release_already_admitted(self):
        mutations=[{'release_ready':'no'},{'release_ready':'unknown'},
            {'observed_basket':'blue'},
            {'relation':'held','observed_basket':None,'release_ready':'no'},
            {'object_secured':'no','release_ready':'no'},
            {'visibility':'occluded','object_secured':'unknown','release_ready':'no'}]
        for mutation in mutations:
            context,evidence,target=actual(202);evidence['arms']['left'].update(mutation)
            with self.subTest(mutation=mutation):
                old=legacy.action_envelopes(evidence,target);new=action_envelopes(evidence,target)
                self.assertNotIn('release',old['by_arm']['left']['allowed_verbs'])
                self.assertNotIn('open',new['by_arm']['left']['allowed_verbs'])
                with self.assertRaises(SubtaskConstraintError):validate_plan(context,new)

    def test_explicit_temporal_chains_rejected_even_when_both_verbs_individually_allowed(self):
        context,evidence,target=actual(201)
        for marker in['then','After that','followed by']:
            altered=deepcopy(context)
            altered['action']['summary']='Carry the pen with the left gripper, '+marker+' hold it steady.'
            with self.subTest(marker=marker),self.assertRaisesRegex(SubtaskConstraintError,'temporal chaining'):
                validate_plan(altered,action_envelopes(evidence,target))

    def test_review_counterexample_approach_squeeze_then_hoist_rejected(self):
        _,evidence,target=actual(203);evidence['arms']['left']['gripper_state']='open'
        context={'subtask':CATALOG['U0208'],'reasoning':'The left gripper is open near the watch; approach without assuming secure contact.',
            'action':{'summary':'Approach the watch with the left gripper until the fingers squeeze shut, then hoist it clear.'}}
        with self.assertRaisesRegex(SubtaskConstraintError,'temporal chaining'):
            validate_plan(context,action_envelopes(evidence,target))

    def test_same_immediate_release_open_is_not_mistaken_for_a_sequence(self):
        context,evidence,target=actual(202)
        self.assertEqual(validate_plan(context,action_envelopes(evidence,target))['summary_primary_action'],'release')
        context['reasoning']+=' Future later phases remain unspecified.'
        self.assertEqual(validate_plan(context,action_envelopes(evidence,target))['acting_arms'],['left'])
        context['action']['summary']='Open the left gripper to release the beige plush doll into the white basket.'
        self.assertEqual(validate_plan(context,action_envelopes(evidence,target))['summary_primary_action'],'open')

    def test_valid_explicit_actor_and_original_category_basket_checks_still_apply(self):
        context,evidence,target=actual(201)
        context['action']['summary']='Carry the black-and-gold pen with the left gripper toward the white basket.'
        result=validate_plan(context,action_envelopes(evidence,target))
        self.assertIsNone(result['actor_override_from_current_evidence'])
        for bad in['Carry the pen with the left gripper toward the blue basket.',
                'Carry the watch with the left gripper toward the white basket.']:
            altered=deepcopy(context);altered['action']['summary']=bad
            with self.assertRaises(SubtaskConstraintError):validate_plan(altered,action_envelopes(evidence,target))


class PhaseProviderProtocolTests(unittest.TestCase):
    def test_protocol_preserves_science_dual_factory_and_requires_new_interpretation(self):
        from prepare_ssr_spec import build
        import enum_grounded_ssr_protocol as old
        with tempfile.TemporaryDirectory()as folder:
            binding=Path(folder)/'SYNTHETIC_PROTOCOL_ONLY_NO_WEIGHTS.json';binding.write_text('{"step":19566}')
            spec=build(binding,parent_root='/SYNTHETIC_PROTOCOL_ONLY',audit_path=Path(folder)/old.NAMESPACE/'audit.jsonl',config_id=old.CONFIG_ID)
            spec.update(config_id=CONFIG_ID,protocol_config_id=CONFIG_ID,generator_protocol=CONFIG_ID,
                audit_path=str(Path(folder)/NAMESPACE/'audit.jsonl'),motion_interpretation=MOTION_INTERPRETATION)
            self.assertEqual(validate_spec(spec)['motion_interpretation'],MOTION_INTERPRETATION)
            for key,bad in[('motion_interpretation',None),('response_schema_factory_sha256','0'*64),
                    ('response_schema_base_factory_sha256','0'*64),('execute_steps',16),('maximum_past_timepoints',31),('max_tokens',16384)]:
                altered=deepcopy(spec);altered[key]=bad
                with self.subTest(key=key),self.assertRaises(ValueError):validate_spec(altered)

    def test_same_v4_schema_prompt_and_hook_dispatch(self):
        provider=object.__new__(PhaseGroundedSSRProvider);old=object.__new__(EnumGroundedSSRProvider)
        provider.catalog=load_catalog();old.catalog=provider.catalog
        self.assertEqual(provider.observer_prompt_path,old.observer_prompt_path)
        self.assertEqual(provider.observer_response_format(),old.observer_response_format())
        self.assertEqual(provider.selector_response_format(['U0001']),old.selector_response_format(['U0001']))
        context,evidence,target=actual(202)
        self.assertEqual(provider.validate_public_plan(context,provider.make_action_envelopes(evidence,target))['acting_arms'],['left'])
        with self.assertRaises(SubtaskConstraintError):old.validate_public_plan(context,old.make_action_envelopes(evidence,target))

    def test_interpretation_marker_bound_before_both_roles_and_input_not_mutated(self):
        provider=object.__new__(PhaseGroundedSSRProvider)
        request={'messages':[{'role':'user','content':[{'type':'text','text':canonical({'instruction':INSTRUCTION,'current_frame':0,'executed_subtask_history':[]})}]}]}
        proof={'episode_id':'CPU_fixture_only','current_frame':0};before=deepcopy((request,proof))
        sent,lineage=provider.history_metadata(request,proof)
        self.assertEqual(strict_json(sent['messages'][-1]['content'][0]['text'])['motion_interpretation'],MOTION_INTERPRETATION)
        self.assertEqual(lineage['motion_interpretation'],MOTION_INTERPRETATION)
        self.assertEqual((request,proof),before)


if __name__=='__main__':unittest.main()
