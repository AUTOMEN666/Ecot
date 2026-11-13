"""Local fake HTTP endpoints only; no external API, policy or GPU calls."""
import http.server
import os
import threading
import time
import unittest
from unittest.mock import patch
from http_transport import bounded_http_post
import http_transport


class WallClockTransport(unittest.TestCase):
    def setUp(self):
        self.received=[]
        received=self.received
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                received.append((self.rfile.read(int(self.headers['Content-Length'])),self.headers.get('Authorization')))
                if self.path=='/slow':
                    self.send_response(200);self.send_header('Content-Length','100');self.end_headers()
                    try:
                        for _ in range(100):self.wfile.write(b'x');self.wfile.flush();time.sleep(.03)
                    except(BrokenPipeError,ConnectionResetError):pass
                else:
                    raw=b'actual-error-body'if self.path=='/503'else b'{"subtask_id":"U0001"}'
                    self.send_response(503 if self.path=='/503'else 200)
                    self.send_header('Content-Length',str(len(raw)));self.send_header('X-Model','fake-local-fixture')
                    self.end_headers();self.wfile.write(raw)
        self.server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.01},daemon=True)
        self.thread.start();self.base='http://127.0.0.1:'+str(self.server.server_port)
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()
    def test_exact_body_and_header_only_enter_child_through_stdin(self):
        body=b'{"request":"exact bytes, no semantic mutation"}'
        token='fixture-only-private-token';receipt={};calls=[]
        original=http_transport.subprocess.Popen
        def spy(*args,**kwargs):calls.append((args,kwargs));return original(*args,**kwargs)
        with patch.dict(os.environ,{'ECOT_AUTHORIZED_API_KEY':token}),patch.object(http_transport.subprocess,'Popen',side_effect=spy):
            raw,headers=bounded_http_post(self.base+'/ok',body,{'Authorization':'Bearer '+token},timeout_seconds=3,
                selector_kind='deepseek',receipt=receipt)
        self.assertEqual(raw,b'{"subtask_id":"U0001"}');self.assertEqual(headers['X-Model'],'fake-local-fixture')
        self.assertEqual(self.received,[(body,'Bearer '+token)])
        self.assertNotIn(token,str(calls[0][0]));self.assertNotIn('ECOT_AUTHORIZED_API_KEY',calls[0][1]['env'])
        self.assertTrue(receipt['request_child_reaped']);self.assertFalse(receipt['request_child_killed'])
    def test_slow_drip_is_bounded_and_exact_child_reaped_before_return(self):
        receipt={};started=time.monotonic()
        with self.assertRaises(TimeoutError):
            bounded_http_post(self.base+'/slow',b'exact',{},timeout_seconds=.25,selector_kind='deepseek',receipt=receipt)
        self.assertLess(time.monotonic()-started,1.5)
        self.assertTrue(receipt['request_child_killed']);self.assertTrue(receipt['request_child_reaped'])
        with self.assertRaises(ProcessLookupError):os.kill(receipt['pid'],0)
    def test_HTTP_error_keeps_existing_retry_classification_and_raw_body(self):
        receipt={}
        with self.assertRaisesRegex(RuntimeError,'^deepseek HTTP 503$')as caught:
            bounded_http_post(self.base+'/503',b'exact',{},timeout_seconds=3,selector_kind='deepseek',receipt=receipt)
        self.assertEqual(caught.exception.response_body,b'actual-error-body');self.assertTrue(receipt['request_child_reaped'])


if __name__=='__main__':unittest.main()
