"""V6 causal CPU fixtures; no API/GPU/policy/SDK execution or acceptance.

ACTUAL_OBSERVATIONS preserves PUBLIC current facts from V5 terminal 201/112
and 203/120. Previous-ready observations below are explicitly SYNTHETIC CPU
prerequisites, not claims that the actual prior frames passed.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import grounded_motion as original
import grounded_motion_v5 as v5
from grounded_motion_v6 import (MAX_OPENING_WINDOWS,action_envelopes,
    derive_release_transitions,proof_binding,validate_plan)
from release_grounded_ssr_provider import ReleaseGroundedSSRProvider
from release_grounded_ssr_protocol import CONFIG_ID,NAMESPACE,MOTION_INTERPRETATION,validate_spec
from enum_grounded_ssr_provider import EnumGroundedSSRProvider
from multiview_protocol import INSTRUCTION,canonical,load_catalog,strict_json
from ssr_protocol import digest
from subtask_guard import SubtaskConstraintError

ACTUAL_OBSERVATIONS={201: {'arms': {'left': {'gripper_state': 'transitioning', 'held_instance_id': 'O1', 'holding': 'identified', 'interaction_instance_id': 'O1', 'motion_trend': 'release', 'object_secured': 'yes', 'observed_basket': 'white', 'relation': 'over_basket', 'release_ready': 'yes', 'visibility': 'clear'}, 'right': {'gripper_state': 'open', 'held_instance_id': None, 'holding': 'empty', 'interaction_instance_id': 'O5', 'motion_trend': 'stationary', 'object_secured': 'no', 'observed_basket': None, 'relation': 'approaching', 'release_ready': 'no', 'visibility': 'clear'}}, 'baskets': {'blue': {'unknown_contents': False, 'visibility': 'clear'}, 'red': {'unknown_contents': False, 'visibility': 'clear'}, 'white': {'unknown_contents': False, 'visibility': 'clear'}}, 'corrections': [], 'goal_correction': None, 'objects': [{'category': 'pen', 'category_confidence': 'clear', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D004'], 'identity_confidence': 'clear', 'instance_id': 'O1', 'location': 'held', 'released': False}, {'category': 'pen', 'category_confidence': 'clear', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D089'], 'identity_confidence': 'clear', 'instance_id': 'O2', 'location': 'table', 'released': None}, {'category': None, 'category_confidence': 'uncertain', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D012'], 'identity_confidence': 'uncertain', 'instance_id': 'O3', 'location': 'table', 'released': None}, {'category': 'ring', 'category_confidence': 'uncertain', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D066'], 'identity_confidence': 'uncertain', 'instance_id': 'O4', 'location': 'table', 'released': None}, {'category': 'watch', 'category_confidence': 'clear', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D070'], 'identity_confidence': 'clear', 'instance_id': 'O5', 'location': 'table', 'released': None}], 'scene_usable': True}, 203: {'arms': {'left': {'gripper_state': 'transitioning', 'held_instance_id': 'O2', 'holding': 'identified', 'interaction_instance_id': 'O2', 'motion_trend': 'release', 'object_secured': 'yes', 'observed_basket': 'white', 'relation': 'over_basket', 'release_ready': 'yes', 'visibility': 'clear'}, 'right': {'gripper_state': 'open', 'held_instance_id': None, 'holding': 'empty', 'interaction_instance_id': None, 'motion_trend': 'stationary', 'object_secured': 'no', 'observed_basket': None, 'relation': 'unknown', 'release_ready': 'no', 'visibility': 'clear'}}, 'baskets': {'blue': {'unknown_contents': False, 'visibility': 'clear'}, 'red': {'unknown_contents': False, 'visibility': 'clear'}, 'white': {'unknown_contents': False, 'visibility': 'clear'}}, 'corrections': [], 'goal_correction': None, 'objects': [{'category': 'watch', 'category_confidence': 'clear', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D070', 'D072'], 'identity_confidence': 'clear', 'instance_id': 'O2', 'location': 'held', 'released': False}, {'category': None, 'category_confidence': 'uncertain', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D012'], 'identity_confidence': 'uncertain', 'instance_id': 'O3', 'location': 'table', 'released': None}, {'category': 'pen', 'category_confidence': 'clear', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D004', 'D039'], 'identity_confidence': 'clear', 'instance_id': 'O4', 'location': 'table', 'released': None}, {'category': 'pen', 'category_confidence': 'clear', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D021', 'D073'], 'identity_confidence': 'clear', 'instance_id': 'O5', 'location': 'table', 'released': None}, {'category': None, 'category_confidence': 'uncertain', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D012'], 'identity_confidence': 'uncertain', 'instance_id': 'O6', 'location': 'table', 'released': None}, {'category': None, 'category_confidence': 'uncertain', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D011'], 'identity_confidence': 'uncertain', 'instance_id': 'O7', 'location': 'table', 'released': None}], 'scene_usable': True}}

ACTUAL_PRIOR_CURRENT_PAIRS={201: {'frame': 112, 'before': {'episode_id': 'expert_replay_episode201_04fdae2e26934dfeaef02e48e2090783', 'current_frame': 104, 'active_goal': {'basket_color': 'white', 'category': 'pen', 'instance_id': 'O1', 'last_unplaced_at': 104, 'selected_at': 0, 'subtask_id': 'U0010'}, 'arms': {'left': {'gripper_state': 'closed', 'held_instance_id': 'O1', 'holding': 'identified', 'interaction_instance_id': 'O1', 'motion_trend': 'lower', 'object_secured': 'yes', 'observed_basket': 'white', 'relation': 'over_basket', 'release_ready': 'no', 'visibility': 'clear'}, 'right': {'gripper_state': 'open', 'held_instance_id': None, 'holding': 'empty', 'interaction_instance_id': 'O5', 'motion_trend': 'stationary', 'object_secured': 'no', 'observed_basket': None, 'relation': 'approaching', 'release_ready': 'no', 'visibility': 'clear'}}, 'current_visible_objects': {'O1': {'category': 'pen', 'category_confidence': 'clear', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D004'], 'identity_confidence': 'clear', 'instance_id': 'O1', 'location': 'held', 'released': False}, 'O2': {'category': 'pen', 'category_confidence': 'clear', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D089'], 'identity_confidence': 'clear', 'instance_id': 'O2', 'location': 'table', 'released': None}, 'O3': {'category': None, 'category_confidence': 'uncertain', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D012'], 'identity_confidence': 'uncertain', 'instance_id': 'O3', 'location': 'table', 'released': None}, 'O4': {'category': 'ring', 'category_confidence': 'uncertain', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D066'], 'identity_confidence': 'uncertain', 'instance_id': 'O4', 'location': 'table', 'released': None}, 'O5': {'category': 'watch', 'category_confidence': 'clear', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D070'], 'identity_confidence': 'clear', 'instance_id': 'O5', 'location': 'table', 'released': None}}, 'baskets': {'blue': {'unknown_contents': False, 'visibility': 'clear'}, 'red': {'unknown_contents': False, 'visibility': 'clear'}, 'white': {'unknown_contents': False, 'visibility': 'clear'}}}, 'evidence': {'arms': {'left': {'gripper_state': 'transitioning', 'held_instance_id': 'O1', 'holding': 'identified', 'interaction_instance_id': 'O1', 'motion_trend': 'release', 'object_secured': 'yes', 'observed_basket': 'white', 'relation': 'over_basket', 'release_ready': 'yes', 'visibility': 'clear'}, 'right': {'gripper_state': 'open', 'held_instance_id': None, 'holding': 'empty', 'interaction_instance_id': 'O5', 'motion_trend': 'stationary', 'object_secured': 'no', 'observed_basket': None, 'relation': 'approaching', 'release_ready': 'no', 'visibility': 'clear'}}, 'baskets': {'blue': {'unknown_contents': False, 'visibility': 'clear'}, 'red': {'unknown_contents': False, 'visibility': 'clear'}, 'white': {'unknown_contents': False, 'visibility': 'clear'}}, 'corrections': [], 'goal_correction': None, 'objects': [{'category': 'pen', 'category_confidence': 'clear', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D004'], 'identity_confidence': 'clear', 'instance_id': 'O1', 'location': 'held', 'released': False}, {'category': 'pen', 'category_confidence': 'clear', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D089'], 'identity_confidence': 'clear', 'instance_id': 'O2', 'location': 'table', 'released': None}, {'category': None, 'category_confidence': 'uncertain', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D012'], 'identity_confidence': 'uncertain', 'instance_id': 'O3', 'location': 'table', 'released': None}, {'category': 'ring', 'category_confidence': 'uncertain', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D066'], 'identity_confidence': 'uncertain', 'instance_id': 'O4', 'location': 'table', 'released': None}, {'category': 'watch', 'category_confidence': 'clear', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D070'], 'identity_confidence': 'clear', 'instance_id': 'O5', 'location': 'table', 'released': None}], 'scene_usable': True}, 'request_sha256': '1681f2fce20d674a8ffc3b1dd8e5deeaa5eb5740f370fba1a6c968ab8e18fb17', 'response_sha256': '58e0020bcfdb0e2fce22ba27c1f4342e25d60414a9bed32f517cbd749f0dc617'}, 203: {'frame': 120, 'before': {'episode_id': 'expert_replay_episode203_f87489659bc9478b9e5c09211dae8e0c', 'current_frame': 112, 'active_goal': {'basket_color': 'white', 'category': 'watch', 'instance_id': 'O2', 'last_unplaced_at': 112, 'selected_at': 0, 'subtask_id': 'U0208'}, 'arms': {'left': {'gripper_state': 'closed', 'held_instance_id': 'O2', 'holding': 'identified', 'interaction_instance_id': 'O2', 'motion_trend': 'lower', 'object_secured': 'yes', 'observed_basket': 'white', 'relation': 'over_basket', 'release_ready': 'yes', 'visibility': 'clear'}, 'right': {'gripper_state': 'open', 'held_instance_id': None, 'holding': 'empty', 'interaction_instance_id': None, 'motion_trend': 'stationary', 'object_secured': 'no', 'observed_basket': None, 'relation': 'unknown', 'release_ready': 'no', 'visibility': 'clear'}}, 'current_visible_objects': {'O2': {'category': 'watch', 'category_confidence': 'clear', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D070', 'D072'], 'identity_confidence': 'clear', 'instance_id': 'O2', 'location': 'held', 'released': False}, 'O3': {'category': None, 'category_confidence': 'uncertain', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D012'], 'identity_confidence': 'uncertain', 'instance_id': 'O3', 'location': 'table', 'released': None}, 'O4': {'category': 'pen', 'category_confidence': 'clear', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D004', 'D039'], 'identity_confidence': 'clear', 'instance_id': 'O4', 'location': 'table', 'released': None}, 'O5': {'category': 'pen', 'category_confidence': 'clear', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D021', 'D073'], 'identity_confidence': 'clear', 'instance_id': 'O5', 'location': 'table', 'released': None}, 'O6': {'category': None, 'category_confidence': 'uncertain', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D012'], 'identity_confidence': 'uncertain', 'instance_id': 'O6', 'location': 'table', 'released': None}, 'O7': {'category': None, 'category_confidence': 'uncertain', 'descriptor_confidence': 'uncertain', 'descriptor_ids': ['D011'], 'identity_confidence': 'uncertain', 'instance_id': 'O7', 'location': 'table', 'released': None}}, 'baskets': {'blue': {'unknown_contents': False, 'visibility': 'clear'}, 'red': {'unknown_contents': False, 'visibility': 'clear'}, 'white': {'unknown_contents': False, 'visibility': 'clear'}}}, 'evidence': {'arms': {'left': {'gripper_state': 'transitioning', 'held_instance_id': 'O2', 'holding': 'identified', 'interaction_instance_id': 'O2', 'motion_trend': 'release', 'object_secured': 'yes', 'observed_basket': 'white', 'relation': 'over_basket', 'release_ready': 'yes', 'visibility': 'clear'}, 'right': {'gripper_state': 'open', 'held_instance_id': None, 'holding': 'empty', 'interaction_instance_id': None, 'motion_trend': 'stationary', 'object_secured': 'no', 'observed_basket': None, 'relation': 'unknown', 'release_ready': 'no', 'visibility': 'clear'}}, 'baskets': {'blue': {'unknown_contents': False, 'visibility': 'clear'}, 'red': {'unknown_contents': False, 'visibility': 'clear'}, 'white': {'unknown_contents': False, 'visibility': 'clear'}}, 'corrections': [], 'goal_correction': None, 'objects': [{'category': 'watch', 'category_confidence': 'clear', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D070', 'D072'], 'identity_confidence': 'clear', 'instance_id': 'O2', 'location': 'held', 'released': False}, {'category': None, 'category_confidence': 'uncertain', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D012'], 'identity_confidence': 'uncertain', 'instance_id': 'O3', 'location': 'table', 'released': None}, {'category': 'pen', 'category_confidence': 'clear', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D004', 'D039'], 'identity_confidence': 'clear', 'instance_id': 'O4', 'location': 'table', 'released': None}, {'category': 'pen', 'category_confidence': 'clear', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D021', 'D073'], 'identity_confidence': 'clear', 'instance_id': 'O5', 'location': 'table', 'released': None}, {'category': None, 'category_confidence': 'uncertain', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D012'], 'identity_confidence': 'uncertain', 'instance_id': 'O6', 'location': 'table', 'released': None}, {'category': None, 'category_confidence': 'uncertain', 'descriptor_confidence': 'clear', 'descriptor_ids': ['D011'], 'identity_confidence': 'uncertain', 'instance_id': 'O7', 'location': 'table', 'released': None}], 'scene_usable': True}, 'request_sha256': '6efbb533d6cd7ff9553bb9c00f30b27755cfea75c3b1159c6b4a4e779df5f6d2', 'response_sha256': '97d38bd2d91195b013a6650e2a832d7506f2ff3dc232437cea953bee17fa9214'}}

CATALOG={x['id']:x for x in load_catalog()['candidates']}


def proof(episode,frame):
    return {'episode_id':episode,'current_frame':frame,'executed_subtask_history':[],
        'observations':[{'frame':frame,'camera':c,'is_current':True,'image_sha256':'0'*64}
            for c in('cam_high','cam_left_wrist','cam_right_wrist')]}


def fixture(episode=201):
    evidence=deepcopy(ACTUAL_OBSERVATIONS[episode]);obj=next(x for x in evidence['objects']if x['location']=='held')
    row=next(x for x in CATALOG.values()if x['descriptor_id']in obj['descriptor_ids']and x['basket_color']=='white')
    frame=112 if episode==201 else 120;eid='SYNTHETIC_CPU_PRIOR_'+str(episode)
    goal={'subtask_id':row['id'],'instance_id':obj['instance_id'],'category':obj['category'],
        'basket_color':'white','selected_at':0,'last_unplaced_at':frame-8}
    prior=deepcopy(evidence);prior['arms']['left'].update(gripper_state='closed',motion_trend='lower')
    before={'episode_id':eid,'current_frame':frame-8,'active_goal':goal,
        'arms':prior['arms'],'current_visible_objects':{x['instance_id']:x for x in prior['objects']},'baskets':prior['baskets']}
    state={'episode_id':eid,'bursts':{}}
    target={k:goal[k]for k in('instance_id','category','basket_color')}
    return evidence,before,proof(eid,frame),state,target


def advance_snapshot(before,evidence,frame):
    out=deepcopy(before);out.update(current_frame=frame,arms=deepcopy(evidence['arms']),
        current_visible_objects={x['instance_id']:deepcopy(x)for x in evidence['objects']},baskets=deepcopy(evidence['baskets']))
    out['active_goal']['last_unplaced_at']=frame
    return out


def derive(evidence,before,p,state):
    return derive_release_transitions(evidence,before=before,proof=p,transition_state=state)


class ReleasePhaseTests(unittest.TestCase):
    def test_actual203_prior112_current120_supports_opening_admission_without_edit(self):
        # Both semantic observations and locked goal are copied from preserved
        # PUBLIC records. Capture hashes here remain explicit CPU placeholders;
        # this is not a real request/capture/native acceptance receipt.
        pair=deepcopy(ACTUAL_PRIOR_CURRENT_PAIRS[203]);e,b=pair['evidence'],pair['before']
        self.assertEqual(e,ACTUAL_OBSERVATIONS[203]);self.assertEqual(b['current_frame'],112)
        self.assertEqual(b['arms']['left']['release_ready'],'yes')
        state={'episode_id':b['episode_id'],'bursts':{}};saved=deepcopy(pair)
        phases,_=derive(e,b,proof(b['episode_id'],pair['frame']),state)
        target={k:b['active_goal'][k]for k in('instance_id','category','basket_color')}
        envelope=action_envelopes(e,target,known_release_transitions=phases)
        self.assertEqual(envelope['by_arm']['left']['allowed_verbs'],['hold','open','release'])
        self.assertEqual(phases['left']['subtask_id'],'U0208');self.assertEqual(pair,saved)

    def test_actual201_prior104_not_ready_must_still_reject_current112(self):
        pair=deepcopy(ACTUAL_PRIOR_CURRENT_PAIRS[201]);e,b=pair['evidence'],pair['before']
        self.assertEqual(e,ACTUAL_OBSERVATIONS[201]);self.assertEqual(b['current_frame'],104)
        self.assertEqual(b['arms']['left']['release_ready'],'no')
        self.assertEqual(e['arms']['left']['release_ready'],'yes');saved=deepcopy(pair)
        with self.assertRaisesRegex(SubtaskConstraintError,'previous actual closed secure ready'):
            derive(e,b,proof(b['episode_id'],pair['frame']),{'episode_id':b['episode_id'],'bursts':{}})
        self.assertEqual(pair,saved)

    def test_both_actual_current_terminal_observations_pass_only_with_synthetic_ready_prior(self):
        for episode in(201,203):
            evidence,before,p,state,target=fixture(episode);saved=deepcopy((evidence,before,p,state))
            with self.subTest(episode=episode):
                with self.assertRaisesRegex(SubtaskConstraintError,'Secure grasp'):
                    original.validate_arms(evidence['arms'],evidence['objects'])
                with self.assertRaisesRegex(SubtaskConstraintError,'Secure grasp'):
                    v5.action_envelopes(evidence,target)
                phases,after=derive(evidence,before,p,state)
                original.validate_arms(evidence['arms'],evidence['objects'],known_release_transitions=phases)
                env=action_envelopes(evidence,target,known_release_transitions=phases)
                self.assertEqual(env['by_arm']['left']['allowed_verbs'],['hold','open','release'])
                self.assertEqual(env['by_arm']['right']['allowed_verbs'],['hold'])
                self.assertEqual(env['by_arm']['left']['observed']['gripper_state'],'transitioning')
                self.assertEqual(phases['left']['window_index'],1)
                self.assertEqual((evidence,before,p,state),saved)
                self.assertEqual(next(iter(after['bursts'].values()))['windows'],1)

    def test_exactly_two_adjacent_opening_windows_and_no_third(self):
        e,b,p,s,t=fixture();one,s1=derive(e,b,p,s)
        b2=advance_snapshot(b,e,p['current_frame']);p2=proof(p['episode_id'],p['current_frame']+8)
        two,s2=derive(e,b2,p2,s1)
        self.assertEqual(two['left']['window_index'],MAX_OPENING_WINDOWS)
        self.assertEqual(two['left']['initial_closed_ready_frame'],b['current_frame'])
        self.assertEqual((two['left']['current_frame']-two['left']['initial_closed_ready_frame'])/25,.64)
        self.assertEqual(two['left']['maximum_additional_execution_steps'],16)
        self.assertEqual(two['left']['maximum_additional_execution_seconds'],.64)
        self.assertEqual(two['left']['maximum_steps_from_last_closed_ready_through_final_block'],24)
        with self.assertRaisesRegex(SubtaskConstraintError,'exhausted'):
            derive(e,advance_snapshot(b2,e,p2['current_frame']),proof(p['episode_id'],p2['current_frame']+8),s2)

    def test_closed_transition_label_oscillation_cannot_restart_spent_burst(self):
        e,b,p,s,t=fixture();_,s1=derive(e,b,p,s)
        closed=deepcopy(e);closed['arms']['left'].update(gripper_state='closed',motion_trend='stationary')
        b2=advance_snapshot(b,e,p['current_frame']);p2=proof(p['episode_id'],p['current_frame']+8)
        phases,s2=derive(closed,b2,p2,s1)
        self.assertEqual(phases,{});self.assertEqual(s1,s2)
        with self.assertRaisesRegex(SubtaskConstraintError,'exhausted'):
            derive(e,advance_snapshot(b2,closed,p2['current_frame']),proof(p['episode_id'],p2['current_frame']+8),s2)

    def test_missing_active_goal_previous_frame_wrong_episode_or_gap_rejected(self):
        for change in('goal','frame','episode','gap'):
            e,b,p,s,t=fixture()
            if change=='goal':b['active_goal']=None
            if change=='frame':b['current_frame']=None
            if change=='episode':b['episode_id']='OTHER'
            if change=='gap':b['current_frame']-=8
            with self.subTest(change=change),self.assertRaises(SubtaskConstraintError):derive(e,b,p,s)

    def test_every_prior_closed_ready_prerequisite_is_required(self):
        mutations=[{'gripper_state':'transitioning'},{'gripper_state':'open'},
            {'gripper_state':'unknown'},{'object_secured':'no'},{'object_secured':'unknown'},
            {'release_ready':'no'},{'release_ready':'unknown'},{'visibility':'occluded'},
            {'relation':'held'},{'observed_basket':'blue'},{'holding':'unknown','held_instance_id':None},
            {'interaction_instance_id':'O2'},{'held_instance_id':'O2'}]
        for mutation in mutations:
            e,b,p,s,t=fixture();b['arms']['left'].update(mutation)
            with self.subTest(mutation=mutation),self.assertRaises(SubtaskConstraintError):derive(e,b,p,s)

    def test_every_current_opening_prerequisite_is_required(self):
        mutations=[{'gripper_state':'open'},{'gripper_state':'unknown'},
            {'object_secured':'no'},{'object_secured':'unknown'},
            {'release_ready':'no'},{'release_ready':'unknown'},{'visibility':'occluded'},
            {'relation':'held'},{'observed_basket':'blue'},{'motion_trend':'close'},
            {'motion_trend':'transport'},{'holding':'unknown','held_instance_id':None},
            {'interaction_instance_id':'O2'},{'held_instance_id':'O2'}]
        for mutation in mutations:
            e,b,p,s,t=fixture();e['arms']['left'].update(mutation)
            with self.subTest(mutation=mutation),self.assertRaises(SubtaskConstraintError):
                phases,_=derive(e,b,p,s)
                original.validate_arms(e['arms'],e['objects'],known_release_transitions=phases)

    def test_nonsecure_opening_does_not_get_exception_or_release(self):
        e,b,p,s,t=fixture();e['arms']['left'].update(object_secured='no',release_ready='no')
        phases,after=derive(e,b,p,s);self.assertEqual(phases,{})
        env=action_envelopes(e,t,known_release_transitions=phases)
        self.assertNotIn('release',env['by_arm']['left']['allowed_verbs'])
        self.assertNotIn('open',env['by_arm']['left']['allowed_verbs']);self.assertEqual(s,after)

    def test_current_or_prior_object_drop_release_unknown_identity_or_category_change_rejected(self):
        for where in('current','previous'):
            for mutation in[{'location':'table'},{'released':True},{'identity_confidence':'uncertain'},
                    {'category_confidence':'uncertain'},{'category':'watch'}]:
                e,b,p,s,t=fixture()
                obj=next(x for x in e['objects']if x['instance_id']==t['instance_id'])if where=='current'else b['current_visible_objects'][t['instance_id']]
                obj.update(mutation)
                with self.subTest(where=where,mutation=mutation),self.assertRaises(SubtaskConstraintError):derive(e,b,p,s)

    def test_other_hand_material_occlusion_or_second_opening_rejected(self):
        for where in('current','previous'):
            for mutation in[{'holding':'unknown'},{'visibility':'occluded'},
                    {'holding':'identified','held_instance_id':'O1'},
                    {'gripper_state':'transitioning','object_secured':'yes'}]:
                e,b,p,s,t=fixture();arms=e['arms']if where=='current'else b['arms'];arms['right'].update(mutation)
                with self.subTest(where=where,mutation=mutation),self.assertRaises(SubtaskConstraintError):derive(e,b,p,s)

    def test_goal_basket_correction_and_unknown_basket_do_not_get_exception(self):
        for change in('basket','correction','goal_correction','contents','visibility'):
            e,b,p,s,t=fixture()
            if change=='basket':b['active_goal']['basket_color']='blue'
            if change=='correction':e['corrections']=[{'instance_id':'O1'}]
            if change=='goal_correction':e['goal_correction']={'actual_instance_id':'O1'}
            if change=='contents':e['baskets']['white']['unknown_contents']=True
            if change=='visibility':b['baskets']['white']['visibility']='unknown'
            with self.subTest(change=change),self.assertRaises(SubtaskConstraintError):derive(e,b,p,s)

    def test_only_open_release_hold_not_lower_lift_carry_close_or_other_arm(self):
        e,b,p,s,t=fixture();phases,_=derive(e,b,p,s);env=action_envelopes(e,t,known_release_transitions=phases)
        for verb in('Open','Release','Hold','Lower','Lift','Carry','Close'):
            context={'subtask':CATALOG[b['active_goal']['subtask_id']]['text'],
                'reasoning':'The left gripper is transitioning during release; the same pen remains supported over the white basket.',
                'action':{'summary':verb+' the left gripper to continue the present release.'}}
            with self.subTest(verb=verb):
                if verb in('Open','Release','Hold'):self.assertEqual(validate_plan(context,env)['summary_primary_action'],verb.lower())
                else:
                    with self.assertRaises(SubtaskConstraintError):validate_plan(context,env)
        context['action']['summary']='Open the right gripper to release the pen.'
        with self.assertRaises(SubtaskConstraintError):validate_plan(context,env)

    def test_release_progress_cannot_claim_already_closed_open_or_infer_an_omitted_actor(self):
        e,b,p,s,t=fixture();phases,_=derive(e,b,p,s);env=action_envelopes(e,t,known_release_transitions=phases)
        context={'subtask':CATALOG[b['active_goal']['subtask_id']]['text'],
            'reasoning':'The left gripper is transitioning during the supported release over the white basket.',
            'action':{'summary':'Open the left gripper to continue the current release.'}}
        for claim in('The left gripper is closed around the pen.','The left gripper is open around the pen.'):
            altered=deepcopy(context);altered['reasoning']=claim
            with self.subTest(claim=claim),self.assertRaisesRegex(SubtaskConstraintError,'gripper state'):validate_plan(altered,env)
        context['action']['summary']='Release the pen into the white basket.'
        with self.assertRaisesRegex(SubtaskConstraintError,'identify its acting'):validate_plan(context,env)

    def test_actual202_negative_release_constraint_is_preserved_without_semantic_rewrite(self):
        # Verbatim public terminal text from V5/202/frame72; original current
        # evidence is secure held/closed, not aligned/release-ready.
        from test_phase_grounded_ssr import actual
        _,e,t=actual(202)
        e['arms']['left'].update(relation='held',motion_trend='transport',release_ready='no')
        context={'subtask':CATALOG['U0001']['text'],
            'action':{'summary':'Carry the held beige plush doll toward the white basket.'},
            'reasoning':'The left gripper is closed around the beige plush doll and appears to hold it securely; the white basket is visible ahead, so continue transporting the doll toward the basket without releasing it.'}
        saved=deepcopy((context,e,t))
        with self.assertRaisesRegex(SubtaskConstraintError,'unobserved prerequisite: release'):
            v5.validate_plan(context,v5.action_envelopes(e,t))
        result=validate_plan(context,action_envelopes(e,t))
        self.assertEqual(result['summary_primary_action'],'carry');self.assertEqual(result['acting_arms'],['left'])
        self.assertEqual((context,e,t),saved)
        context['reasoning']+=' Release the doll into the white basket.'
        with self.assertRaisesRegex(SubtaskConstraintError,'unobserved prerequisite: release'):
            validate_plan(context,action_envelopes(e,t))

    def test_prior_v5_actual_unknown_gripper_and_temporal_chains_still_rejected(self):
        from test_phase_grounded_ssr import actual
        context,e,t=actual(203)
        with self.assertRaises(SubtaskConstraintError):validate_plan(context,action_envelopes(e,t))
        context,e,t=actual(202)
        context['action']['summary']='Release the doll with the left gripper, then retract the left arm.'
        with self.assertRaisesRegex(SubtaskConstraintError,'temporal chaining'):validate_plan(context,action_envelopes(e,t))

    def test_transition_cannot_apply_to_another_target_or_fake_current_phase(self):
        e,b,p,s,t=fixture();phases,_=derive(e,b,p,s)
        for change in('target','basket','window','verbs','state'):
            target=deepcopy(t);phase=deepcopy(phases);current=deepcopy(e)
            if change=='target':target['instance_id']='O2'
            if change=='basket':target['basket_color']='blue'
            if change=='window':phase['left']['window_index']=3
            if change=='verbs':phase['left']['allowed_verbs'].append('carry')
            if change=='state':current['arms']['left']['gripper_state']='unknown'
            with self.subTest(change=change),self.assertRaises(SubtaskConstraintError):action_envelopes(current,target,known_release_transitions=phase)

    def test_actual_proof_current_frame_cameras_and_sha_required(self):
        for change in('frame','camera','sha','missing','history_only','episode'):
            e,b,p,s,t=fixture()
            if change=='frame':p['observations'][0]['frame']-=8
            if change=='camera':p['observations'][0]['camera']='cam_right_wrist'
            if change=='sha':p['observations'][0]['image_sha256']='missing'
            if change=='missing':p['observations']=[]
            if change=='history_only':p['observations'][0]['is_current']=False
            if change=='episode':s['episode_id']='OTHER'
            with self.subTest(change=change),self.assertRaises(SubtaskConstraintError):derive(e,b,p,s)

    def test_budget_and_previous_opening_evidence_must_remain_bound(self):
        for change in('evidence','arm','gap','goal'):
            e,b,p,s,t=fixture();_,s1=derive(e,b,p,s)
            next_before=advance_snapshot(b,e,p['current_frame']);next_proof=proof(p['episode_id'],p['current_frame']+8)
            if change=='evidence':next_before['arms']['left']['motion_trend']='stationary'
            if change=='arm':next(iter(s1['bursts'].values()))['arm']='right'
            if change=='gap':next(iter(s1['bursts'].values()))['last_frame']-=8
            if change=='goal':next_before['active_goal']['selected_at']=8
            with self.subTest(change=change),self.assertRaises(SubtaskConstraintError):derive(e,next_before,next_proof,s1)

    def test_drop_regrasp_budget_rearming_is_explicitly_not_implemented(self):
        e,b,p,s,t=fixture();_,s1=derive(e,b,p,s)
        # A later same-goal closed/ready report is not a proven whole regrasp
        # chain. Keep the ID; reject the narrow transition exception.
        b['current_frame']+=32;p=proof(p['episode_id'],b['current_frame']+8)
        with self.assertRaisesRegex(SubtaskConstraintError,'exhausted'):derive(e,b,p,s1)


class ReleaseProviderTests(unittest.TestCase):
    def provider(self,before,state):
        class ReadOnlyFixtureGuard:
            def snapshot(self):return deepcopy(self.value)
        provider=object.__new__(ReleaseGroundedSSRProvider);provider.guard=ReadOnlyFixtureGuard();provider.guard.value=before
        provider._release_transition_state=state;provider._release_pending=None
        return provider

    def test_validation_caches_without_spending_until_scene_accepts_same_evidence(self):
        e,b,p,s,t=fixture();provider=self.provider(b,s);provider.validate_current_arms(e,p)
        self.assertEqual(provider._release_transition_state,s)
        with self.assertRaisesRegex(SubtaskConstraintError,'actual scene admission'):provider.make_action_envelopes(e,t)
        self.assertEqual(provider._release_transition_state,s)
        provider.guard.value=advance_snapshot(b,e,p['current_frame'])
        env=provider.make_action_envelopes(e,t)
        self.assertEqual(env['known_release_transitions']['left']['window_index'],1)
        self.assertEqual(next(iter(provider._release_transition_state['bursts'].values()))['windows'],1)
        provider.make_action_envelopes(e,t)
        self.assertEqual(next(iter(provider._release_transition_state['bursts'].values()))['windows'],1)

    def test_pending_current_evidence_target_and_goal_must_match_after_scene(self):
        for change in('evidence','target','goal','frame','episode'):
            e,b,p,s,t=fixture();provider=self.provider(b,s);provider.validate_current_arms(e,p)
            provider.guard.value=advance_snapshot(b,e,p['current_frame'])
            if change=='evidence':e['arms']['left']['motion_trend']='stationary'
            if change=='target':t['basket_color']='blue'
            if change=='goal':provider.guard.value['active_goal']['selected_at']=8
            if change=='frame':provider.guard.value['current_frame']+=8
            if change=='episode':provider.guard.value['episode_id']='OTHER'
            with self.subTest(change=change),self.assertRaises(SubtaskConstraintError):provider.make_action_envelopes(e,t)
            self.assertEqual(provider._release_transition_state,s)

    def test_episode_reset_clears_budget_and_authorization(self):
        provider=object.__new__(ReleaseGroundedSSRProvider);provider.episode_id='NEW'
        provider._release_transition_state={'episode_id':'OLD','bursts':{'spent':{}}};provider._release_pending={'old':True}
        provider.reset_context_state()
        self.assertEqual(provider._release_transition_state,{'episode_id':'NEW','bursts':{}})
        self.assertIsNone(provider._release_pending);self.assertIsNone(provider._ssr_authorization)

    def test_normal_frame_zero_needs_no_previous_ready_observation(self):
        e,b,p,s,t=fixture();e['arms']['left'].update(gripper_state='closed',motion_trend='lower')
        b.update(current_frame=None,active_goal=None);p=proof(p['episode_id'],0)
        phases,after=derive(e,b,p,s);self.assertEqual(phases,{});self.assertEqual(s,after)

    def test_both_request_roles_use_new_marker_and_no_input_mutation(self):
        provider=object.__new__(ReleaseGroundedSSRProvider)
        request={'messages':[{'role':'user','content':[{'type':'text','text':canonical({'instruction':INSTRUCTION})}]}]}
        p={'episode_id':'CPU_ONLY','current_frame':0};before=deepcopy((request,p))
        request2,p2=provider.history_metadata(request,p)
        self.assertEqual(p2['motion_interpretation'],MOTION_INTERPRETATION)
        self.assertEqual(strict_json(request2['messages'][-1]['content'][0]['text'])['motion_interpretation'],MOTION_INTERPRETATION)
        self.assertNotIn('release_open_and_unique_secured_actor_v1',canonical((request2,p2)))
        self.assertEqual((request,p),before)

    def test_v4_prompt_exact_prefix_new_explicit_release_contract_and_schema_unchanged(self):
        new=object.__new__(ReleaseGroundedSSRProvider);old=object.__new__(EnumGroundedSSRProvider)
        new.catalog=load_catalog();old.catalog=new.catalog
        self.assertTrue(new.observer_prompt_path.read_bytes().startswith(old.observer_prompt_path.read_bytes()))
        self.assertNotEqual(new.observer_prompt_path,old.observer_prompt_path)
        self.assertEqual(new.observer_response_format(),old.observer_response_format())
        self.assertEqual(new.selector_response_format(['U0001']),old.selector_response_format(['U0001']))

    def test_protocol_requires_new_id_marker_with_unchanged_science_and_dual_schema(self):
        from prepare_ssr_spec import build
        import enum_grounded_ssr_protocol as v4
        with tempfile.TemporaryDirectory()as folder:
            binding=Path(folder)/'CPU_ONLY.json';binding.write_text('{"step":19566}')
            spec=build(binding,parent_root='/CPU_ONLY',audit_path=Path(folder)/v4.NAMESPACE/'audit.jsonl',config_id=v4.CONFIG_ID)
            spec.update(config_id=CONFIG_ID,protocol_config_id=CONFIG_ID,generator_protocol=CONFIG_ID,
                audit_path=str(Path(folder)/NAMESPACE/'audit.jsonl'),motion_interpretation=MOTION_INTERPRETATION)
            self.assertEqual(validate_spec(spec)['motion_interpretation'],MOTION_INTERPRETATION)
            for key,bad in[('motion_interpretation',None),('response_schema_factory_sha256','0'*64),
                    ('response_schema_base_factory_sha256','0'*64),('execute_steps',16),('maximum_past_timepoints',31),('max_tokens',16384)]:
                altered=deepcopy(spec);altered[key]=bad
                with self.subTest(key=key),self.assertRaises(ValueError):validate_spec(altered)


class ReleaseActualMethodFixtureTests(unittest.TestCase):
    """Real host methods and ACK validators, SYNTHETIC policy/transport only."""
    def chain(self,folder,transport):
        from contextlib import contextmanager
        from test_ssr import runtime
        @contextmanager
        def context():
            with runtime(folder,step=19566)as(_,policy,_,oldspec):
                import model_server,types,numpy as np
                from prepare_ssr_spec import build
                spec=build(oldspec['weights_binding_file'],parent_root=oldspec['parent_root'],
                    audit_path=Path(folder)/NAMESPACE/'actions.jsonl',config_id=CONFIG_ID)
                provider=ReleaseGroundedSSRProvider(Path(folder)/'release_requests',selector_kind='deepseek',
                    checkpoint_step=19566,transport=transport)
                class Random:
                    key=staticmethod(lambda seed:np.array([seed],np.int32));key_data=staticmethod(lambda key:key)
                    split=staticmethod(lambda key:(key,key+1));normal=staticmethod(lambda key,shape:np.zeros(shape,np.float32))
                model=model_server.create_model(spec,policy=policy,jax_module=types.SimpleNamespace(random=Random),provider=provider)
                yield model,policy,provider,spec
        return context()

    def transport(self,seen,*,third_opening=False,no_prior_ready=False):
        from test_grounded_ssr import facts,public
        from test_executed_history_integration import response
        def wire(request,p):
            seen.append((deepcopy(request),deepcopy(p)));frame=p['current_frame']
            phase={0:'approach',8:'close',16:'lift',24:'release',32:'release',40:'release',48:'retract',56:'retract'}[frame]
            evidence=facts(phase)
            if frame==24 and no_prior_ready:evidence['arms']['left']['release_ready']='no'
            if frame in(32,40)or(frame==48 and third_opening):
                evidence=facts('release');evidence['arms']['left'].update(gripper_state='transitioning',motion_trend='release')
            value=public(phase.capitalize())
            if frame in(32,40,48)and evidence['arms']['left']['gripper_state']=='transitioning':
                value={'subtask_id':'U0012',
                    'reasoning':'The left gripper is transitioning during release; the same pen remains visibly supported over the red basket.',
                    'action':{'summary':'Open the left gripper to continue the current release into the red basket.'}}
            if frame==24 and no_prior_ready:
                value={'subtask_id':'U0012','reasoning':'The held pen remains supported above the red basket; release readiness is not established.',
                    'action':{'summary':'Hold the left arm steady above the red basket.'}}
            return response(canonical(evidence if p['call_role']=='scene_observer'else value))
        return wire

    def test_real_model_provider_ack_methods_two_opening_windows_then_visual_completion(self):
        from test_ssr import acknowledge,capture_to
        from test_steady_runtime import obs
        seen=[]
        with tempfile.TemporaryDirectory()as tmp,self.chain(tmp,self.transport(seen))as(model,policy,provider,spec):
            model.reset()
            for frame in(0,8,16,24,32,40,48,56):
                capture_to(provider,frame);model.update_obs(obs(frame));commands=model.get_action()
                injection=model.current_injection();lineage=injection['lineage']
                self.assertEqual(injection['context_protocol'],CONFIG_ID);self.assertEqual(injection['checkpoint_step'],19566)
                self.assertEqual(lineage['motion_interpretation'],MOTION_INTERPRETATION)
                self.assertEqual(lineage['executed_subtask_history_count'],frame//8)
                self.assertIn(injection['context']['reasoning'],policy.observations[-1]['prompt'])
                provider.validate_injection(injection['context'],lineage,episode_id=model.episode_id,frame=frame)
                phases=lineage['grounded_motion_authorization']['known_release_transitions']
                if frame in(32,40):
                    self.assertEqual(phases['left']['window_index'],(frame-24)//8)
                    self.assertEqual(lineage['current_visual_observation']['arms']['left']['gripper_state'],'transitioning')
                    self.assertFalse(lineage['grounded_decision']['confirmed_completion'])
                else:self.assertEqual(phases,{})
                acknowledge(model,commands,frame)
            self.assertEqual(len(policy.observations),8);self.assertEqual(len(provider.executed_history.records),8)
            self.assertEqual(provider.guard.snapshot()['completed_instance_ids'],['O1'])
            self.assertEqual(next(iter(provider._release_transition_state['bursts'].values()))['windows'],2)
            for request,p in seen:
                self.assertEqual(p['motion_interpretation'],MOTION_INTERPRETATION)
                self.assertEqual(strict_json(request['messages'][-1]['content'][0]['text'])['motion_interpretation'],MOTION_INTERPRETATION)
                if p['call_role']=='scene_observer':self.assertTrue(request['messages'][0]['content'].startswith(provider.observer_prompt_path.read_text()))

    def test_missing_previous_ready_stops_before_selector_policy_and_new_ack(self):
        from test_ssr import acknowledge,capture_to
        from test_steady_runtime import obs
        seen=[]
        with tempfile.TemporaryDirectory()as tmp,self.chain(tmp,self.transport(seen,no_prior_ready=True))as(model,policy,provider,spec):
            model.reset()
            for frame in(0,8,16,24):
                capture_to(provider,frame);model.update_obs(obs(frame));commands=model.get_action();acknowledge(model,commands,frame)
            capture_to(provider,32);model.update_obs(obs(32))
            with self.assertRaisesRegex(SubtaskConstraintError,'previous actual closed secure ready'):model.get_action()
            self.assertEqual(len(policy.observations),4);self.assertEqual(len(provider.executed_history.records),4)
            self.assertEqual([(p['current_frame'],p['call_role'])for _,p in seen if p['current_frame']==32],[(32,'scene_observer')])
            self.assertEqual(provider._release_transition_state['bursts'],{})

    def test_third_opening_stops_before_selector_policy_and_new_ack(self):
        from test_ssr import acknowledge,capture_to
        from test_steady_runtime import obs
        seen=[]
        with tempfile.TemporaryDirectory()as tmp,self.chain(tmp,self.transport(seen,third_opening=True))as(model,policy,provider,spec):
            model.reset()
            for frame in(0,8,16,24,32,40):
                capture_to(provider,frame);model.update_obs(obs(frame));commands=model.get_action();acknowledge(model,commands,frame)
            capture_to(provider,48);model.update_obs(obs(48))
            with self.assertRaisesRegex(SubtaskConstraintError,'exhausted'):model.get_action()
            self.assertEqual(len(policy.observations),6);self.assertEqual(len(provider.executed_history.records),6)
            self.assertEqual([(p['current_frame'],p['call_role'])for _,p in seen if p['current_frame']==48],[(48,'scene_observer')])
            self.assertEqual(next(iter(provider._release_transition_state['bursts'].values()))['windows'],2)


if __name__=='__main__':unittest.main()

