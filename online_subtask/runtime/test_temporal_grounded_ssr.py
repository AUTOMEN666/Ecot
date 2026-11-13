"""V7 actual PUBLIC regressions and real host methods with CPU fixture policy.

Historical public scene facts/text/response SHA are v6 regressions, NOT a new
v7 API request. All capture/State14/history proofs are explicitly SYNTHETIC.
Model/provider/ACK chain tests use SYNTHETIC captures/policy/HTTP;
there is no real GPU, native execution, SDK outcome or accuracy certification.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

import grounded_motion as original
import grounded_motion_v6 as v6
from grounded_motion_v7 import action_envelopes,validate_plan
from multiview_protocol import INSTRUCTION,canonical,load_catalog,strict_json
from negative_action_guard import is_negated
from ssr_protocol import digest
from subtask_guard import SubtaskConstraintError
from temporal_action_clause import is_completed_release_clause
from temporal_grounded_ssr_protocol import (CONFIG_ID,NAMESPACE,MOTION_INTERPRETATION,
    COMPLETED_RELEASE_HELPER_SHA256,validate_spec)
from temporal_grounded_ssr_provider import TemporalGroundedSSRProvider
from test_executed_history_integration import response

HERE=Path(__file__).resolve().parent
CATALOG={row['id']:row['text']for row in load_catalog()['candidates']}

PUBLIC_REGRESSIONS={'201_112_scene_observer': {'arms': {'left': {'gripper_state': 'transitioning',
                                              'held_instance_id': 'O1',
                                              'holding': 'identified',
                                              'interaction_instance_id': 'O1',
                                              'motion_trend': 'release',
                                              'object_secured': 'yes',
                                              'observed_basket': 'white',
                                              'relation': 'over_basket',
                                              'release_ready': 'yes',
                                              'visibility': 'clear'},
                                     'right': {'gripper_state': 'open',
                                               'held_instance_id': None,
                                               'holding': 'empty',
                                               'interaction_instance_id': None,
                                               'motion_trend': 'stationary',
                                               'object_secured': 'no',
                                               'observed_basket': None,
                                               'relation': 'separated',
                                               'release_ready': 'no',
                                               'visibility': 'clear'}},
                            'baskets': {'blue': {'unknown_contents': False, 'visibility': 'clear'},
                                        'red': {'unknown_contents': False, 'visibility': 'clear'},
                                        'white': {'unknown_contents': False, 'visibility': 'clear'}},
                            'corrections': [],
                            'goal_correction': None,
                            'objects': [{'category': 'pen',
                                         'category_confidence': 'clear',
                                         'descriptor_confidence': 'clear',
                                         'descriptor_ids': ['D004', 'D039', 'D060'],
                                         'identity_confidence': 'clear',
                                         'instance_id': 'O1',
                                         'location': 'held',
                                         'released': False},
                                        {'category': 'pen',
                                         'category_confidence': 'clear',
                                         'descriptor_confidence': 'clear',
                                         'descriptor_ids': ['D089', 'D091'],
                                         'identity_confidence': 'clear',
                                         'instance_id': 'O2',
                                         'location': 'table',
                                         'released': True},
                                        {'category': 'block',
                                         'category_confidence': 'clear',
                                         'descriptor_confidence': 'clear',
                                         'descriptor_ids': ['D006', 'D012'],
                                         'identity_confidence': 'clear',
                                         'instance_id': 'O3',
                                         'location': 'table',
                                         'released': True},
                                        {'category': 'watch',
                                         'category_confidence': 'clear',
                                         'descriptor_confidence': 'clear',
                                         'descriptor_ids': ['D070', 'D072'],
                                         'identity_confidence': 'clear',
                                         'instance_id': 'O4',
                                         'location': 'table',
                                         'released': True},
                                        {'category': 'watch',
                                         'category_confidence': 'clear',
                                         'descriptor_confidence': 'clear',
                                         'descriptor_ids': ['D070', 'D072'],
                                         'identity_confidence': 'clear',
                                         'instance_id': 'O5',
                                         'location': 'table',
                                         'released': True}],
                            'scene_usable': True},
 '201_112_subtask_selector': {'action': {'summary': 'Release the black and gold pen into the white basket '
                                                    'with the left gripper.'},
                              'reasoning': 'The left gripper remains over the white basket with the black '
                                           'and gold pen still between its jaws; placement readiness is '
                                           'visible, so continue opening the left jaws to deposit it.',
                              'subtask_id': 'U0010'},
 '201_120_scene_observer': {'arms': {'left': {'gripper_state': 'open',
                                              'held_instance_id': None,
                                              'holding': 'empty',
                                              'interaction_instance_id': 'O1',
                                              'motion_trend': 'retract',
                                              'object_secured': 'no',
                                              'observed_basket': 'white',
                                              'relation': 'released',
                                              'release_ready': 'no',
                                              'visibility': 'clear'},
                                     'right': {'gripper_state': 'open',
                                               'held_instance_id': None,
                                               'holding': 'empty',
                                               'interaction_instance_id': None,
                                               'motion_trend': 'stationary',
                                               'object_secured': 'no',
                                               'observed_basket': None,
                                               'relation': 'separated',
                                               'release_ready': 'no',
                                               'visibility': 'clear'}},
                            'baskets': {'blue': {'unknown_contents': False, 'visibility': 'clear'},
                                        'red': {'unknown_contents': False, 'visibility': 'clear'},
                                        'white': {'unknown_contents': False, 'visibility': 'clear'}},
                            'corrections': [],
                            'goal_correction': None,
                            'objects': [{'category': 'pen',
                                         'category_confidence': 'clear',
                                         'descriptor_confidence': 'clear',
                                         'descriptor_ids': ['D004', 'D039', 'D060'],
                                         'identity_confidence': 'clear',
                                         'instance_id': 'O1',
                                         'location': 'white',
                                         'released': True},
                                        {'category': 'pen',
                                         'category_confidence': 'clear',
                                         'descriptor_confidence': 'clear',
                                         'descriptor_ids': ['D089', 'D091'],
                                         'identity_confidence': 'clear',
                                         'instance_id': 'O2',
                                         'location': 'table',
                                         'released': True},
                                        {'category': 'block',
                                         'category_confidence': 'clear',
                                         'descriptor_confidence': 'clear',
                                         'descriptor_ids': ['D006', 'D012'],
                                         'identity_confidence': 'clear',
                                         'instance_id': 'O3',
                                         'location': 'table',
                                         'released': True},
                                        {'category': 'watch',
                                         'category_confidence': 'clear',
                                         'descriptor_confidence': 'clear',
                                         'descriptor_ids': ['D070', 'D072'],
                                         'identity_confidence': 'clear',
                                         'instance_id': 'O4',
                                         'location': 'table',
                                         'released': True},
                                        {'category': 'watch',
                                         'category_confidence': 'clear',
                                         'descriptor_confidence': 'clear',
                                         'descriptor_ids': ['D070', 'D072'],
                                         'identity_confidence': 'clear',
                                         'instance_id': 'O5',
                                         'location': 'table',
                                         'released': True}],
                            'scene_usable': True},
 '201_120_subtask_selector': {'action': {'summary': 'Retract the empty left arm from the white basket after '
                                                    'releasing the pen.'},
                              'reasoning': 'The black and gold pen is already visible inside the white '
                                           'basket, and the left gripper is open and empty. The next '
                                           'feasible adjustment is to withdraw the left arm clear of the '
                                           'basket.',
                              'subtask_id': 'U0010'},
 '203_56_scene_observer': {'arms': {'left': {'gripper_state': 'closed',
                                             'held_instance_id': None,
                                             'holding': 'empty',
                                             'interaction_instance_id': 'O6',
                                             'motion_trend': 'lift',
                                             'object_secured': 'yes',
                                             'observed_basket': None,
                                             'relation': 'held',
                                             'release_ready': 'no',
                                             'visibility': 'clear'},
                                    'right': {'gripper_state': 'open',
                                              'held_instance_id': None,
                                              'holding': 'empty',
                                              'interaction_instance_id': None,
                                              'motion_trend': 'stationary',
                                              'object_secured': 'no',
                                              'observed_basket': None,
                                              'relation': 'unknown',
                                              'release_ready': 'no',
                                              'visibility': 'clear'}},
                           'baskets': {'blue': {'unknown_contents': False, 'visibility': 'clear'},
                                       'red': {'unknown_contents': False, 'visibility': 'clear'},
                                       'white': {'unknown_contents': False, 'visibility': 'clear'}},
                           'corrections': [],
                           'goal_correction': None,
                           'objects': [{'category': 'watch',
                                        'category_confidence': 'clear',
                                        'descriptor_confidence': 'clear',
                                        'descriptor_ids': ['D070'],
                                        'identity_confidence': 'clear',
                                        'instance_id': 'O1',
                                        'location': 'table',
                                        'released': None},
                                       {'category': 'watch',
                                        'category_confidence': 'uncertain',
                                        'descriptor_confidence': 'uncertain',
                                        'descriptor_ids': ['D034'],
                                        'identity_confidence': 'uncertain',
                                        'instance_id': 'O2',
                                        'location': 'table',
                                        'released': None},
                                       {'category': 'pen',
                                        'category_confidence': 'clear',
                                        'descriptor_confidence': 'clear',
                                        'descriptor_ids': ['D004'],
                                        'identity_confidence': 'clear',
                                        'instance_id': 'O3',
                                        'location': 'table',
                                        'released': None},
                                       {'category': 'pen',
                                        'category_confidence': 'clear',
                                        'descriptor_confidence': 'clear',
                                        'descriptor_ids': ['D073'],
                                        'identity_confidence': 'clear',
                                        'instance_id': 'O4',
                                        'location': 'table',
                                        'released': None},
                                       {'category': 'block',
                                        'category_confidence': 'uncertain',
                                        'descriptor_confidence': 'uncertain',
                                        'descriptor_ids': ['D012'],
                                        'identity_confidence': 'uncertain',
                                        'instance_id': 'O5',
                                        'location': 'table',
                                        'released': None},
                                       {'category': None,
                                        'category_confidence': 'uncertain',
                                        'descriptor_confidence': 'uncertain',
                                        'descriptor_ids': [],
                                        'identity_confidence': 'uncertain',
                                        'instance_id': 'O6',
                                        'location': 'table',
                                        'released': None}],
                           'scene_usable': True}}

PUBLIC_RESPONSE_SHA256={'201_112_scene_observer': '0642266bf55d1c1e55db40476d2834bc63ff1f80d833650ae20ef821ee0ab826',
 '201_112_subtask_selector': '5df3c47ddc28630ee3a798ab7b368913c6c8fb9d3cae33e3b5fb44d42fc87c99',
 '201_120_scene_observer': '0d8966b6f1a9a3bf6dd8a5c85374c02aa421caa8eed8934d1748741ebb56a73d',
 '201_120_subtask_selector': '9d9cfb97a1fafc7e0424d93b41e4889defe34305309a0bfd1d32d5ab73e50666',
 '203_56_scene_observer': '9674f21a3189fda879cc24225c86eb27d6d469a8386c1d40bc5fc90512580d30'}


def synthetic_current_proof(evidence):
    """Explicit CPU current captures/state and guard binding, no real dataset.

    Historical full actual capture/history proof remains a PRIVATE report-only
    replay artifact; none of its image SHA, State14 or history values is shipped.
    This minimal pure-helper fixture does not claim a complete executed prefix.
    """
    episode="SYNTHETIC_CPU_CURRENT_RELEASED_201"
    target=dict(instance_id="O1",category="pen",basket_color="white")
    return {"episode_id":episode,"current_frame":120,"call_role":"subtask_selector",
        "fixture":True,"synthetic_current_capture_and_State14":True,
        "actual_policy_injection_or_ACK":False,
        "current_visual_observation":deepcopy(evidence),"current_visual_observation_sha256":digest(evidence),
        "grounded_decision":{"episode_id":episode,"frame":120,"locked_id":"U0010",
            "allowed_ids":["U0010"],"targets":{"U0010":target}},
        "observations":[{"camera":camera,"frame":120,"is_current":True,
            "image_sha256":"0"*64,"state14":[0.]*14}for camera in
            ("cam_high","cam_left_wrist","cam_right_wrist")]}


def actual_context():
    value=deepcopy(PUBLIC_REGRESSIONS['201_120_subtask_selector'])
    return {'subtask':CATALOG[value['subtask_id']],'reasoning':value['reasoning'],'action':value['action']}


@contextmanager
def temporal_chain(folder,transport,provider_class=TemporalGroundedSSRProvider):
    from test_ssr import runtime
    with runtime(folder,step=19566)as(_,policy,_,oldspec):
        import model_server,types,numpy as np
        from prepare_ssr_spec import build
        spec=build(oldspec['weights_binding_file'],parent_root=oldspec['parent_root'],
            audit_path=Path(folder)/NAMESPACE/'actions.jsonl',config_id=CONFIG_ID)
        provider=provider_class(Path(folder)/'temporal_requests',selector_kind='deepseek',
            checkpoint_step=19566,transport=transport)
        class Random:
            key=staticmethod(lambda seed:np.array([seed],np.int32));key_data=staticmethod(lambda key:key)
            split=staticmethod(lambda key:(key,key+1));normal=staticmethod(lambda key,shape:np.zeros(shape,np.float32))
        model=model_server.create_model(spec,policy=policy,jax_module=types.SimpleNamespace(random=Random),provider=provider)
        yield model,policy,provider,spec


def synthetic_scene(phase):
    # Only the phase/setup is a CPU prerequisite. The final120 observation is
    # copied verbatim from the preserved actual public source, not corrected.
    evidence=deepcopy(PUBLIC_REGRESSIONS['201_120_scene_observer'])
    obj=evidence['objects'][0];left=evidence['arms']['left']
    if phase in('approach','close'):
        obj.update(location='table',released=None)
        left.update(holding='empty',held_instance_id=None,gripper_state='open',interaction_instance_id='O1',
            object_secured='no',observed_basket=None,release_ready='no',
            relation='approaching'if phase=='approach'else'enclosing',motion_trend='approach'if phase=='approach'else'align')
    elif phase in('carry','ready'):
        obj.update(location='held',released=False)
        left.update(holding='identified',held_instance_id='O1',gripper_state='closed',interaction_instance_id='O1',
            object_secured='yes',observed_basket='white'if phase=='ready'else None,
            release_ready='yes'if phase=='ready'else'no',relation='over_basket'if phase=='ready'else'held',
            motion_trend='lower'if phase=='ready'else'transport')
    elif phase=='opening':
        evidence=deepcopy(PUBLIC_REGRESSIONS['201_112_scene_observer'])
    return evidence


def synthetic_public(phase):
    summary,reason={
        'approach':('Approach the pen with the open left gripper.', 'The left gripper is open; approach the separated pen.'),
        'close':('Close the left gripper around the enclosed pen.', 'The pen is enclosed between the left fingers; continue closing without lifting it.'),
        'carry':('Carry the pen with the left gripper toward the white basket.', 'The left gripper is closed around the supported pen; continue transporting it toward the white basket without releasing it.'),
        'ready':('Release the pen with the left gripper into the white basket.', 'The left gripper supports the pen over the white basket opening and is ready for release.'),
        'cleanup':('Retract the empty left arm from the white basket.', 'The pen remains visible inside the white basket; withdraw the empty left arm.')
    }[phase]
    return {'subtask_id':'U0010','reasoning':reason,'action':{'summary':summary}}


def wire(seen):
    def transport(request,proof):
        seen.append((deepcopy(request),deepcopy(proof)));frame=proof['current_frame']
        phase='approach'if frame==0 else'close'if frame==8 else'ready'if frame==104 else'opening'if frame==112 else'retract'if frame>=120 else'carry'
        evidence=synthetic_scene(phase)
        if frame==112:value=deepcopy(PUBLIC_REGRESSIONS['201_112_subtask_selector'])
        elif frame==120:value=deepcopy(PUBLIC_REGRESSIONS['201_120_subtask_selector'])
        else:
            value=synthetic_public('cleanup'if frame>120 else phase)
            # At128 the two real-method released reports confirm O1 and the
            # actual guard admits O2 as a new pen goal. This explicit mock-model
            # response selects U0265; it must not reuse the now-retired U0010.
            if frame>120:value['subtask_id']='U0265'
        return response(canonical(evidence if proof['call_role']=='scene_observer'else value))
    return transport


class TemporalPureTests(unittest.TestCase):
    def test_exact_frozen_helper_and_new_prompt_full_v6_prefix_same_full_schemas(self):
        from release_grounded_ssr_provider import ReleaseGroundedSSRProvider
        self.assertEqual(hashlib.sha256((HERE/'temporal_action_clause.py').read_bytes()).hexdigest(),COMPLETED_RELEASE_HELPER_SHA256)
        new=object.__new__(TemporalGroundedSSRProvider);old=object.__new__(ReleaseGroundedSSRProvider)
        new.catalog=load_catalog();old.catalog=new.catalog
        self.assertTrue(new.observer_prompt_path.read_bytes().startswith(old.observer_prompt_path.read_bytes()))
        self.assertEqual(new.observer_response_format(),old.observer_response_format())
        self.assertEqual(new.selector_response_format(['U0010']),old.selector_response_format(['U0010']))

    def test_actual201120_public_facts_and_synthetic_current_proof_not_a_real_API(self):
        context=actual_context();evidence=deepcopy(PUBLIC_REGRESSIONS['201_120_scene_observer'])
        proof=synthetic_current_proof(evidence)
        target=dict(proof['grounded_decision']['targets']['U0010'],subtask_id='U0010')
        old=v6.action_envelopes(evidence,target)
        self.assertEqual(proof['current_frame'],120);self.assertEqual(len(proof['observations']),3)
        self.assertTrue(proof['synthetic_current_capture_and_State14'])
        self.assertFalse(proof['actual_policy_injection_or_ACK'])
        self.assertEqual(PUBLIC_RESPONSE_SHA256['201_120_subtask_selector'],
            '9d9cfb97a1fafc7e0424d93b41e4889defe34305309a0bfd1d32d5ab73e50666')
        before=deepcopy((context,evidence,proof,target));env=action_envelopes(evidence,target)
        with self.assertRaisesRegex(SubtaskConstraintError,'unobserved prerequisite: release'):v6.validate_plan(context,old)
        result=validate_plan(context,env,completed_release_predicate=lambda text,index:
            is_completed_release_clause(text,index,context=context,evidence=evidence,target=target,proof=proof,envelope=env))
        self.assertEqual(result['summary_primary_action'],'retract');self.assertEqual(len(result['completed_release_clause_mentions']),1)
        self.assertEqual((context,evidence,proof,target),before)

    def test_actual20356_is_closed_but_empty_null_held_secure_yes_still_rejected(self):
        e=deepcopy(PUBLIC_REGRESSIONS['203_56_scene_observer']);before=deepcopy(e)
        self.assertEqual(tuple(e['arms']['left'][k]for k in('gripper_state','holding','held_instance_id','object_secured')),
            ('closed','empty',None,'yes'))
        with self.assertRaisesRegex(SubtaskConstraintError,'Secure grasp needs identified object'):original.validate_arms(e['arms'],e['objects'])
        self.assertEqual(e,before)

    def test_v6_release_actions_budget_negation_actor_and_temporal_rules_remain(self):
        from test_release_grounded_ssr import fixture,derive
        e,b,p,s,t=fixture(203);phases,_=derive(e,b,p,s)
        old=v6.action_envelopes(e,t,known_release_transitions=phases);new=action_envelopes(e,t,known_release_transitions=phases)
        expected=deepcopy(old);expected['motion_interpretation']=MOTION_INTERPRETATION
        self.assertEqual(new,expected);self.assertEqual(new['by_arm']['left']['allowed_verbs'],['hold','open','release'])
        self.assertEqual(phases['left']['maximum_steps_from_last_closed_ready_through_final_block'],24)
        context={'subtask':CATALOG['U0208'],'reasoning':'The left gripper is transitioning during release; the watch remains supported over the white basket.',
            'action':{'summary':'Open the left gripper to continue release of the watch into the white basket.'}}
        self.assertEqual(validate_plan(context,new,completed_release_predicate=lambda text,index:False)['summary_primary_action'],'open')
        context['action']['summary']='Open the left gripper, then carry the watch toward the basket.'
        with self.assertRaisesRegex(SubtaskConstraintError,'temporal chaining'):validate_plan(context,new,completed_release_predicate=lambda text,index:False)

    def test_new_protocol_marker_dual_SHA_and_science_reject_old_or_reduced_settings(self):
        from prepare_ssr_spec import build
        with tempfile.TemporaryDirectory()as tmp:
            binding=Path(tmp)/'CPU_ONLY.json';binding.write_text('{"step":19566}')
            spec=build(binding,parent_root='/CPU_ONLY',audit_path=Path(tmp)/NAMESPACE/'audit.jsonl',config_id=CONFIG_ID)
            self.assertEqual(validate_spec(spec)['motion_interpretation'],MOTION_INTERPRETATION)
            for key,value in [('motion_interpretation','release_progress_and_negative_constraints_v1'),
                ('response_schema_factory_sha256','0'*64),('response_schema_base_factory_sha256','0'*64),
                ('execute_steps',16),('maximum_past_timepoints',31),('max_tokens',16384)]:
                changed=deepcopy(spec);changed[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):validate_spec(changed)


class TemporalProviderMethodTests(unittest.TestCase):
    def test_real_provider_model_ACK_methods_actual120_public_text_then_next_history(self):
        from test_ssr import acknowledge,capture_to
        from test_steady_runtime import obs
        seen=[]
        with tempfile.TemporaryDirectory()as tmp,temporal_chain(tmp,wire(seen))as(model,policy,p,spec):
            model.reset()
            for frame in range(0,129,8):
                capture_to(p,frame);model.update_obs(obs(frame));commands=model.get_action();injection=model.current_injection()
                self.assertEqual(len(commands),8);self.assertEqual(injection['context_protocol'],CONFIG_ID)
                self.assertEqual(injection['checkpoint_step'],19566)
                p.validate_injection(injection['context'],injection['lineage'],episode_id=model.episode_id,frame=frame)
                if frame==120:
                    self.assertEqual(injection['context'],actual_context())
                    motion=injection['lineage']['grounded_motion_authorization']
                    self.assertEqual(len(motion['completed_release_clause_mentions']),1)
                    self.assertEqual(motion['temporal_completed_release_binding_sha256'],p._temporal_binding_sha256)
                    self.assertIn(injection['context']['reasoning'],policy.observations[-1]['prompt'])
                    self.assertIn(injection['context']['action']['summary'],policy.observations[-1]['prompt'])
                acknowledge(model,commands,frame)
            self.assertEqual(len(policy.observations),17);self.assertEqual(len(p.executed_history.records),17)
            for request,proof in seen:
                self.assertEqual(proof['motion_interpretation'],MOTION_INTERPRETATION)
                self.assertEqual(strict_json(request['messages'][-1]['content'][0]['text'])['motion_interpretation'],MOTION_INTERPRETATION)
            nextproof=next(proof for _,proof in seen if proof['current_frame']==128 and proof['call_role']=='subtask_selector')
            self.assertEqual(nextproof['executed_subtask_history_count'],16)
            self.assertEqual(nextproof['executed_subtask_history'][-1],{
                'observation_frame':120,'subtask_id':'U0010','subtask':CATALOG['U0010'],
                'executed_action_start':121,'executed_action_end':128})
            self.assertTrue(nextproof['executed_subtask_ack_source_sha256'])

    def test_actual20356_invalid_observer_stops_before_selector_policy_or_ACK(self):
        from test_ssr import capture_to
        from test_steady_runtime import obs
        calls=[]
        def transport(request,proof):
            calls.append(proof['call_role']);return response(canonical(PUBLIC_REGRESSIONS['203_56_scene_observer']))
        with tempfile.TemporaryDirectory()as tmp,temporal_chain(tmp,transport)as(model,policy,p,spec):
            model.reset();capture_to(p,0);model.update_obs(obs(0))
            with self.assertRaisesRegex(SubtaskConstraintError,'Secure grasp needs identified object'):model.get_action()
            self.assertEqual(calls,['scene_observer']);self.assertEqual(policy.observations,[])
            self.assertEqual(p.executed_history.records,[]);self.assertIsNone(p._temporal_binding)

    def test_mutated_binding_fails_before_policy(self):
        from test_ssr import capture_to
        from test_steady_runtime import obs
        class Mutated(TemporalGroundedSSRProvider):
            def validate_selection(self,context,resolution,proof):
                super().validate_selection(context,resolution,proof)
                self._temporal_binding['context_sha256']='0'*64
        seen=[]
        with tempfile.TemporaryDirectory()as tmp,temporal_chain(tmp,wire(seen),Mutated)as(model,policy,p,spec):
            model.reset();capture_to(p,0);model.update_obs(obs(0))
            with self.assertRaisesRegex(SubtaskConstraintError,'Temporal clause binding'):model.get_action()
            self.assertEqual(len(seen),2);self.assertEqual(policy.observations,[]);self.assertEqual(p.executed_history.records,[])

    def test_cache_target_proof_context_envelope_authorized_scene_reset_or_staleness_fail_revalidation(self):
        from test_ssr import capture_to
        from test_steady_runtime import obs
        seen=[]
        with tempfile.TemporaryDirectory()as tmp,temporal_chain(tmp,wire(seen))as(model,policy,p,spec):
            model.reset();capture_to(p,0);model.update_obs(obs(0));model.get_action();injection=model.current_injection()
            keys=('_temporal_binding','_temporal_binding_sha256','_temporal_authorized_state','_temporal_authorized_state_sha256',
                '_temporal_phase','_motion_envelopes','_ssr_authorization','episode_id','stopped')
            saved={k:deepcopy(getattr(p,k))for k in keys};guard=deepcopy(p.guard.__dict__)
            for mutation in('missing','proof','evidence','target','context','envelope','authorization','authorization_removed','phase','guard','frame','episode','stopped'):
                for k,v in saved.items():setattr(p,k,deepcopy(v))
                p.guard.__dict__.clear();p.guard.__dict__.update(deepcopy(guard))
                if mutation=='missing':p._temporal_binding=None
                if mutation=='proof':p._temporal_binding['selector_proof']['request_sha256']='0'*64
                if mutation=='evidence':p._temporal_binding['evidence']['objects'][0]['released']=True
                if mutation=='target':p._temporal_binding['target']['instance_id']='O2'
                if mutation=='context':p._temporal_binding['context_sha256']='0'*64
                if mutation=='envelope':p._motion_envelopes['U0010']['by_arm']['left']['allowed_verbs'].append('release')
                if mutation=='authorization':p._temporal_authorized_state['scene_sha256']='0'*64
                if mutation=='authorization_removed':p._temporal_authorized_state=None;p._temporal_authorized_state_sha256=None
                if mutation=='phase':p._temporal_phase='selecting'
                if mutation=='guard':p.guard.arms['left']['gripper_state']='closed'
                if mutation=='frame':p.guard.frame=8
                if mutation=='episode':p.episode_id='OTHER_EPISODE'
                if mutation=='stopped':p.stopped={'error':'EXPLICIT_CPU_STOP'}
                with self.subTest(mutation=mutation),self.assertRaises(SubtaskConstraintError):
                    p.validate_injection(injection['context'],injection['lineage'],episode_id=model.episode_id,frame=0)

    def test_actual_selector_lineage_fields_cannot_be_mutated_after_selection(self):
        from test_ssr import capture_to
        from test_steady_runtime import obs
        seen=[]
        with tempfile.TemporaryDirectory()as tmp,temporal_chain(tmp,wire(seen))as(model,policy,p,spec):
            model.reset();capture_to(p,0);model.update_obs(obs(0));model.get_action();injection=model.current_injection()
            for key in('request_sha256','current_visual_observation_sha256','motion_interpretation',
                'temporal_completed_release_binding_sha256','temporal_actual_selector_proof_sha256'):
                changed=deepcopy(injection['lineage']);changed[key]='EXPLICIT_CPU_MUTATION'
                with self.subTest(key=key),self.assertRaises(SubtaskConstraintError):
                    p.validate_injection(injection['context'],changed,episode_id=model.episode_id,frame=0)

    def test_episode_reset_clears_all_temporal_and_release_state(self):
        p=object.__new__(TemporalGroundedSSRProvider);p.episode_id='NEW_EPISODE'
        p._temporal_binding={'old':True};p._temporal_binding_sha256='old';p._temporal_authorized_state={'old':True}
        p._temporal_authorized_state_sha256='old';p._temporal_phase='authorized'
        p.reset_context_state()
        for key in('_temporal_binding','_temporal_binding_sha256','_temporal_authorized_state','_temporal_authorized_state_sha256','_temporal_phase'):
            self.assertIsNone(getattr(p,key))
        self.assertEqual(p._release_transition_state,{'episode_id':'NEW_EPISODE','bursts':{}})
        self.assertIsNone(p._ssr_authorization);self.assertIsNone(p._release_pending)


if __name__=='__main__':unittest.main()
