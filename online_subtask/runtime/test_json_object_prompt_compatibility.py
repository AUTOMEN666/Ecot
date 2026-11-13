"""Ark JSON-mode input compatibility regression, entirely offline fixtures."""
import json,re,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import multiview_protocol
from multiview_protocol import INSTRUCTION,canonical
from multiview_provider import MultiviewHistoryProvider
from test_history96 import triplet,state
from test_executed_history_integration import response,capture_to

OLD_LINE='Return ONLY {"subtask_id":"one listed ID"}.'
NEW_LINE='Return ONLY JSON {"subtask_id":"one listed ID"}.'

def message_texts(request):
    texts=[]
    for message in request['messages']:
        content=message['content']
        if isinstance(content,str):texts.append(content)
        else:texts.extend(block['text']for block in content if block.get('type')=='text')
    return texts

def json_mode_compatible(request):
    return request.get('response_format')!={'type':'json_object'}or any(re.search('json',text,re.I)for text in message_texts(request))

class JSONModePromptCompatibility(unittest.TestCase):
    def test_actual_request_contains_JSON_and_retains_generation_contract(self):
        calls=[]
        def transport(request,proof):
            calls.append(request)
            self.assertTrue(json_mode_compatible(request))
            return response()
        with tempfile.TemporaryDirectory()as tmp:
            provider=MultiviewHistoryProvider(tmp,selector_kind='deepseek',transport=transport)
            provider.reset('fixture');capture_to(provider,0)
            _,proof=provider.generate(triplet(0),INSTRUCTION,state14=state(0),episode_id='fixture',frame=0)
            request=calls[0]
            self.assertIn(NEW_LINE,request['messages'][0]['content'])
            self.assertNotIn(OLD_LINE,request['messages'][0]['content'])
            self.assertEqual(request['response_format'],{'type':'json_object'})
            self.assertEqual(request['temperature'],0);self.assertEqual(request['thinking'],{'type':'enabled'})
            self.assertEqual(request['reasoning_effort'],'high');self.assertEqual(request['max_tokens'],65536)
            self.assertEqual(len(calls),1);self.assertEqual(proof['attempt_count'],1)
            self.assertEqual(proof['executed_subtask_history'],[]);self.assertEqual(proof['total_image_count'],3)
    def test_original_missing_word_reproduces_parameter_rejection_without_retry(self):
        # This reproduces only the provider's input validation rule; no API is called.
        calls=[]
        def transport(request,proof):
            calls.append(request)
            self.assertFalse(json_mode_compatible(request))
            error=RuntimeError('deepseek HTTP 400')
            error.response_body=canonical({'error':{'code':'InvalidParameter',
                'message':"Prompt must contain the word 'json' in some form to use 'response_format' of type 'json_object'."}}).encode()
            raise error
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);prompt=(multiview_protocol.HERE/'selector_prompt.md').read_text()
            self.assertEqual(prompt.count(NEW_LINE),1)
            (root/'selector_prompt.md').write_text(prompt.replace(NEW_LINE,OLD_LINE))
            provider=MultiviewHistoryProvider(root/'output',selector_kind='deepseek',transport=transport)
            provider.reset('fixture');capture_to(provider,0)
            with patch.object(multiview_protocol,'HERE',root):
                with self.assertRaisesRegex(RuntimeError,'deepseek HTTP 400'):
                    provider.generate(triplet(0),INSTRUCTION,state14=state(0),episode_id='fixture',frame=0)
            self.assertEqual(len(calls),1);self.assertEqual(provider.executed_history.records,[])
            failures=list((root/'output').glob('*_failure.json'));self.assertEqual(len(failures),1)
            row=json.loads(failures[0].read_text())
            self.assertFalse(row['retry_eligible']);self.assertFalse(row['retry_scheduled'])
            self.assertFalse(row['actual_injected']);self.assertFalse(row['policy_called'])

if __name__=='__main__':unittest.main()
