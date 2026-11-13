"""Behavioral counterexamples and actual public-SSR/policy/ACK method integration."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from grounded_motion import action_envelopes,validate_plan
from grounded_ssr_protocol import CONFIG_ID,NAMESPACE
from grounded_ssr_provider import GroundedSSRProvider
from multiview_protocol import INSTRUCTION,canonical
from prepare_ssr_spec import build
from subtask_guard import SubtaskConstraintError
from test_executed_history_integration import response
from test_history96 import triplet,state
from test_ssr import runtime,acknowledge,capture_to
from test_steady_runtime import obs

def arm(**kwargs):
    value=dict(holding='empty',held_instance_id=None,gripper_state='open',interaction_instance_id=None,
        relation='separated',object_secured='no',observed_basket=None,release_ready='no',
        visibility='clear',motion_trend='stationary')
    value.update(kwargs);return value

def facts(phase='approach'):
    obj=dict(instance_id='O1',identity_confidence='clear',category='pen',category_confidence='clear',
        descriptor_ids=['D004'],descriptor_confidence='clear',location='table',released=True)
    left=arm(interaction_instance_id='O1',relation='approaching',motion_trend='approach')
    if phase=='close':left=arm(interaction_instance_id='O1',relation='enclosing',motion_trend='align')
    elif phase in('lift','release'):
        obj.update(location='held',released=False)
        left=arm(holding='identified',held_instance_id='O1',gripper_state='closed',interaction_instance_id='O1',
            relation='held',object_secured='yes',motion_trend='lift')
        if phase=='release':left.update(relation='over_basket',observed_basket='red',release_ready='yes',motion_trend='lower')
    elif phase=='retract':
        obj.update(location='red',released=True);left=arm(interaction_instance_id='O1',relation='released',motion_trend='retract')
    return dict(scene_usable=True,baskets={x:{'visibility':'clear','unknown_contents':False}for x in('white','blue','red')},
                objects=[obj],corrections=[],goal_correction=None,arms={'left':left,'right':arm()})

def public(verb='Approach',reason=None):
    summaries={'Approach':'Approach the pen with the open left gripper to prepare a grasp.',
        'Close':'Close the left gripper around the aligned pen to establish a grasp.',
        'Lift':'Lift the securely held pen with the left arm for transfer.',
        'Release':'Release the pen from the left gripper into the red basket.',
        'Retract':'Retract the empty left arm away from the red basket after placement.'}
    reasons={'Approach':'Keep the left gripper open while approaching and aligning the pen between the jaws.',
        'Close':'The pen is enclosed between the aligned left fingers; close the left gripper to establish contact.',
        'Lift':'The left gripper is closed around the supported pen; lift it slightly while retaining the grasp.',
        'Release':'The securely held pen is positioned over the red basket; release the pen from the left gripper.',
        'Retract':'The pen is visibly released inside the red basket; retract the empty left arm clear of the basket.'}
    return dict(subtask_id='U0012',reasoning=reason or reasons[verb],action={'summary':summaries[verb]})

def wire(request,proof):
    phases={0:'approach',8:'close',16:'lift',24:'release',32:'retract',40:'retract'}
    f=proof['current_frame'];phase=phases.get(f,'approach')
    return response(canonical(facts(phase)if proof['call_role']=='scene_observer'else public(phase.capitalize())))

@contextmanager
def grounded_runtime(tmp,transport=wire):
    # Existing fixture binds independent original training formatter, JAX noise,
    # weight identity and actual NativeModel methods. No real model is loaded.
    with runtime(tmp,step=19566)as(_,policy,_,oldspec):
        import model_server
        spec=build(oldspec['weights_binding_file'],parent_root=oldspec['parent_root'],
            audit_path=Path(tmp)/NAMESPACE/'actions.jsonl',config_id=CONFIG_ID)
        p=GroundedSSRProvider(Path(tmp)/'grounded_requests',selector_kind='deepseek',checkpoint_step=19566,transport=transport)
        import types,numpy as np
        class Random:
            key=staticmethod(lambda seed:np.array([seed],np.int32));key_data=staticmethod(lambda key:key)
            split=staticmethod(lambda key:(key,key+1));normal=staticmethod(lambda key,shape:np.zeros(shape,np.float32))
        model=model_server.create_model(spec,policy=policy,jax_module=types.SimpleNamespace(random=Random),provider=p)
        yield model,policy,p,spec

class GroundedSSRTests(unittest.TestCase):
    def test_known_enclosed_closing_target_preserves_unknown_grasp_and_only_close_hold(self):
        seen=[]
        def transport(request,proof):
            frame=proof['current_frame'];e=facts('approach'if frame==0 else 'close')
            if frame>=8:
                e['objects'][0]['released']=None
                e['arms']['left'].update(gripper_state='transitioning',motion_trend='close',
                    holding='unknown'if frame>=16 else 'empty')
            if proof['call_role']=='scene_observer':return response(canonical(e))
            meta=deepcopy(request);seen.append((frame,meta))
            if frame==0:return response(canonical(public()))
            value=public('Close'if frame>=16 else 'Approach')
            if frame==8:value.update(reasoning='The known pen remains enclosed while the left jaws transition closed; maintain alignment without asserting a supported grasp.',
                action={'summary':'Hold the left gripper alignment as the jaws close around the pen.'})
            else:value['reasoning']='The known pen is visibly enclosed and the left jaws continue closing; continue closure while support remains unconfirmed.'
            return response(canonical(value))
        with tempfile.TemporaryDirectory()as tmp,grounded_runtime(tmp,transport)as(model,policy,p,spec):
            model.reset()
            for frame in [0,8,16,24]:
                capture_to(p,frame);model.update_obs(obs(frame));commands=model.get_action();injection=model.current_injection()
                if frame>=16:
                    evidence=injection['lineage']['current_visual_observation']
                    self.assertEqual(evidence['arms']['left']['holding'],'unknown')
                    self.assertIsNone(evidence['arms']['left']['held_instance_id'])
                    self.assertEqual(evidence['arms']['left']['object_secured'],'no')
                    decision=injection['lineage']['grounded_decision']
                    self.assertEqual(decision['mode'],'locked_contact_transition')
                    self.assertEqual(decision['known_contact_transition']['uncertainty_windows'],(frame-8)//8)
                    envelope=p._motion_envelopes['U0012']
                    self.assertEqual(envelope['by_arm']['left']['allowed_verbs'],['close','hold'])
                    self.assertEqual(envelope['by_arm']['right']['allowed_verbs'],['hold'])
                    with self.assertRaises(SubtaskConstraintError):validate_plan({'subtask':'unused',**{k:v for k,v in public('Lift').items()if k!='subtask_id'}},envelope)
                    with self.assertRaises(SubtaskConstraintError):validate_plan({'subtask':'unused','reasoning':'Keep the other arm steady.',
                        'action':{'summary':'Approach the pen with the right gripper.'}},envelope)
                acknowledge(model,commands,frame)
            capture_to(p,32);model.update_obs(obs(32))
            with self.assertRaisesRegex(SubtaskConstraintError,'Unknown held object'):model.get_action()
            self.assertEqual(len(policy.observations),4);self.assertEqual(len(p.executed_history.records),4)
            self.assertEqual([f for f,_ in seen],[0,8,16,24])

    def test_closing_transition_cannot_bypass_unknown_material_contact_opening_or_other_hand(self):
        mutations=[lambda e:e['arms']['left'].update(relation='contact'),
            lambda e:e['arms']['left'].update(motion_trend='release'),
            lambda e:e['arms']['left'].update(visibility='occluded'),
            lambda e:e['arms']['left'].update(interaction_instance_id=None),
            lambda e:e['arms']['right'].update(holding='unknown'),
            lambda e:e['objects'][0].update(identity_confidence='uncertain'),
            lambda e:e['objects'].append(dict(instance_id='O2',identity_confidence='uncertain',category=None,
                category_confidence='uncertain',descriptor_ids=[],descriptor_confidence='uncertain',location='held',released=False))]
        for mutate in mutations:
            with self.subTest(mutation=mutations.index(mutate)):
                selectors=[]
                def transport(request,proof):
                    frame=proof['current_frame'];e=facts('approach'if frame==0 else 'close')
                    if frame>=8:
                        e['objects'][0]['released']=None
                        e['arms']['left'].update(gripper_state='transitioning',motion_trend='close',holding='unknown'if frame==16 else 'empty')
                    if frame==16:mutate(e)
                    if proof['call_role']=='scene_observer':return response(canonical(e))
                    selectors.append(frame)
                    return response(canonical(public()if frame==0 else dict(subtask_id='U0012',
                        reasoning='The known pen is enclosed during closure; maintain alignment without assuming secure support.',
                        action={'summary':'Hold the left gripper alignment around the pen.'})))
                with tempfile.TemporaryDirectory()as tmp,grounded_runtime(tmp,transport)as(model,policy,p,spec):
                    model.reset()
                    for frame in [0,8]:
                        capture_to(p,frame);model.update_obs(obs(frame));commands=model.get_action();acknowledge(model,commands,frame)
                    capture_to(p,16);model.update_obs(obs(16))
                    with self.assertRaises(SubtaskConstraintError):model.get_action()
                    self.assertEqual(selectors,[0,8]);self.assertEqual(len(policy.observations),2)
                    self.assertEqual(len(p.executed_history.records),2)

    def test_empty_unknown_oscillation_cannot_reset_the_same_closure_uncertainty_budget(self):
        selectors=[]
        def transport(request,proof):
            frame=proof['current_frame'];e=facts('approach'if frame==0 else 'close')
            if frame>=8:
                e['objects'][0]['released']=None
                e['arms']['left'].update(gripper_state='transitioning',motion_trend='close',
                    holding='unknown'if frame in(16,24,40)else 'empty')
            if frame==32:
                e['arms']['right'].update(interaction_instance_id='O1',relation='approaching',motion_trend='approach')
            if proof['call_role']=='scene_observer':return response(canonical(e))
            selectors.append(frame)
            return response(canonical(public()if frame==0 else dict(subtask_id='U0012',
                reasoning='The known pen remains enclosed during closing; maintain alignment without claiming support.',
                action={'summary':'Hold the left gripper alignment around the pen.'})))
        with tempfile.TemporaryDirectory()as tmp,grounded_runtime(tmp,transport)as(model,policy,p,spec):
            model.reset()
            for frame in [0,8,16,24,32]:
                capture_to(p,frame);model.update_obs(obs(frame));commands=model.get_action();acknowledge(model,commands,frame)
            self.assertEqual(p.guard.closing_transition_streak,2)
            capture_to(p,40);model.update_obs(obs(40))
            with self.assertRaisesRegex(SubtaskConstraintError,'Unknown held object'):model.get_action()
            self.assertEqual(selectors,[0,8,16,24,32]);self.assertEqual(len(policy.observations),5)
            self.assertEqual(len(p.executed_history.records),5)

    def test_real_frame8_right_open_gripper_phrase_is_an_unambiguous_actor(self):
        evidence=facts();evidence['objects'][0].update(category='watch',descriptor_ids=['D070'])
        evidence['arms']={'left':arm(),'right':arm(interaction_instance_id='O1',relation='approaching',motion_trend='approach')}
        context={'subtask':'unused','reasoning':'The right gripper is open and empty, while the green watch remains separated ahead below the red basket; continue an open-jaw approach and adjust toward the watch without closing before it is enclosed.',
            'action':{'summary':'Approach the green watch with the right open gripper to prepare alignment.'}}
        receipt=validate_plan(context,action_envelopes(evidence,dict(instance_id='O1',basket_color='red',category='watch')))
        self.assertEqual(receipt['acting_arms'],['right']);self.assertEqual(context['action']['summary'],'Approach the green watch with the right open gripper to prepare alignment.')

    def test_current_approach_does_not_admit_frame16_premature_close_lift(self):
        target=dict(instance_id='O1',basket_color='red',category='pen')
        envelope=action_envelopes(facts(),target)
        with self.assertRaises(SubtaskConstraintError):validate_plan({'subtask':'unused',**{k:v for k,v in public('Close').items()if k!='subtask_id'}},envelope)
        good=public();good.pop('subtask_id');good['subtask']='unused'
        good['reasoning']='The pen is not between the fingers yet; close the left gripper and lift it slightly once secured.'
        with self.assertRaises(SubtaskConstraintError):validate_plan(good,envelope)

    def test_open_current_gripper_cannot_be_described_as_already_closed(self):
        target=dict(instance_id='O1',basket_color='red',category='pen');value=public(reason='The left gripper is closed around the pen while approaching.')
        value.pop('subtask_id');value['subtask']='unused'
        with self.assertRaisesRegex(SubtaskConstraintError,'gripper state'):validate_plan(value,action_envelopes(facts(),target))

    def test_reasoning_cannot_assign_lift_to_the_empty_opposite_arm(self):
        target=dict(instance_id='O1',basket_color='red',category='pen')
        value=public('Lift',reason='Lift the pen with the right arm while the left gripper holds it securely.')
        value.pop('subtask_id');value['subtask']='unused'
        with self.assertRaisesRegex(SubtaskConstraintError,'right arm'):validate_plan(value,action_envelopes(facts('lift'),target))

    def test_reasoning_cannot_direct_carry_to_another_basket(self):
        target=dict(instance_id='O1',basket_color='red',category='pen')
        value=public('Lift',reason='Carry the pen toward the blue basket with the left arm while keeping the grasp.')
        value.pop('subtask_id');value['subtask']='unused'
        with self.assertRaisesRegex(SubtaskConstraintError,'different target basket'):validate_plan(value,action_envelopes(facts('lift'),target))

    def test_grounded_future_actions_progress_across_real_methods_with_full_ACK_prefix(self):
        requests=[]
        def transport(r,p):requests.append((deepcopy(r),deepcopy(p)));return wire(r,p)
        with tempfile.TemporaryDirectory()as tmp,grounded_runtime(tmp,transport)as(model,policy,p,spec):
            model.reset()
            for f in [0,8,16,24,32,40]:
                capture_to(p,f);model.update_obs(obs(f));commands=model.get_action();injection=model.current_injection()
                self.assertEqual(injection['context_protocol'],CONFIG_ID)
                self.assertEqual(injection['checkpoint_step'],19566)
                self.assertEqual(injection['context']['action']['summary'],public({0:'Approach',8:'Close',16:'Lift',24:'Release',32:'Retract',40:'Retract'}[f])['action']['summary'])
                self.assertIn(injection['context']['reasoning'],policy.observations[-1]['prompt'])
                self.assertEqual(injection['lineage']['executed_subtask_history_count'],f//8)
                acknowledge(model,commands,f)
            self.assertEqual(len(policy.observations),6);self.assertEqual(len(p.executed_history.records),6)
            self.assertEqual([r['executed_action_end']for r in p.executed_history.records],[8,16,24,32,40,48])
            for request,proof in requests:
                self.assertEqual(proof['generator_protocol'],CONFIG_ID)
                self.assertEqual(request['max_tokens'],65536)

    def test_actual_SDK_adapter_accepts_only_the_explicit_new_protocol(self):
        import os
        from unittest.mock import patch
        import test_ssr_native_loop as original_fixture
        with patch.dict(os.environ,{'PI05_SSR_PROTOCOL_ID':CONFIG_ID}), \
            patch.object(original_fixture,'runtime',lambda tmp,step:grounded_runtime(tmp)):
            case=original_fixture.SSRNativeLoop('test_actual_adapter_RPC_loop_executes8_then3_and_commits_three_fields_after_each')
            case.test_actual_adapter_RPC_loop_executes8_then3_and_commits_three_fields_after_each()

    def test_bad_future_phase_never_calls_policy_or_creates_ACK(self):
        seen=[]
        def bad(request,proof):
            seen.append(proof['call_role']);return response(canonical(facts()if proof['call_role']=='scene_observer'else public('Close')))
        with tempfile.TemporaryDirectory()as tmp,grounded_runtime(tmp,bad)as(model,policy,p,spec):
            model.reset();capture_to(p,0);model.update_obs(obs(0))
            with self.assertRaises(SubtaskConstraintError):model.get_action()
            self.assertEqual(policy.observations,[]);self.assertEqual(p.executed_history.records,[])
            with self.assertRaises(SubtaskConstraintError):model.get_action()
            self.assertEqual(seen,['scene_observer','subtask_selector'])

if __name__=='__main__':unittest.main(verbosity=2)
