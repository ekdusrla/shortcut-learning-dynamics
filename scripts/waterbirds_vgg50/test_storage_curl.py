import io
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request
from scripts.waterbirds_vgg50 import download as d, storage as s
from scripts.waterbirds_vgg50.curl_transport import CurlOpener, CurlResponse
from scripts.waterbirds_vgg50.test_download import Endpoint, FixturePolicy, objects
from scripts.waterbirds_vgg50.test_waterbirds import fixture
from scripts.waterbirds_vgg50.protocol import prepare as protocol_prepare, load


class StorageTests(unittest.TestCase):
    def test_local_policy_and_no_transfer_on_failed_plan(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root=base/'ssd'/'birds';cache=base/'network'/'cache'
            root.parent.mkdir();cache.parent.mkdir()
            def fs(path):return 'nfs4' if Path(path).is_relative_to(cache.parent) else 'ext4'
            with patch.object(s,'filesystem',side_effect=fs):
                policy=s.StoragePolicy(root,cache,base=root.parent)
                plan=policy.preflight()
                self.assertLess(plan['ssd_peak_with_reserve_bytes'],50_000_000_000)
                self.assertEqual(plan['final_cap_bytes'],16*1024**3)
                self.assertFalse(root.exists());self.assertFalse(cache.exists())
                with patch.object(s.shutil,'disk_usage',return_value=type('Disk',(),{'free':1})()):
                    with self.assertRaisesRegex(ValueError,'Insufficient'):
                        d.prepare(root,opener=lambda *a,**k:self.fail('Network must not start'),policy=policy)
                self.assertFalse(root.exists());self.assertFalse(cache.exists())
                with patch.object(s,'FINAL_CAP',50_000_000_000):
                    with self.assertRaisesRegex(ValueError,'50GB'):policy.preflight()
                root.mkdir();(root/'old.tgz').write_bytes(b'preserve')
                with self.assertRaisesRegex(ValueError,'Archive'):policy.preflight()
                self.assertEqual((root/'old.tgz').read_bytes(),b'preserve')
            with self.assertRaises(ValueError):s.local_root('/data/example/birds')
            with patch.object(s,'filesystem',return_value='nfs4'):
                with self.assertRaisesRegex(ValueError,'node-local'):s.local_root(root,root.parent)

    def test_hard_limits_stage_cleanup_and_cache_retention(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);budget=s.DataBudget(root,limit=32*1024**2)
            with self.assertRaisesRegex(ValueError,'hard byte'):budget.claim(33*1024**2)
            final=root/'final'
            def fail(stage):
                (stage/'partial.jpg').write_bytes(b'partial');raise ValueError('injected failure')
            with self.assertRaises(ValueError):d.reuse_or_publish(final,fail,lambda p:{})
            self.assertFalse(final.exists());self.assertFalse(list(root.glob('.final.stage-*')))
            cache=root/'network';endpoint=Endpoint({d.CUB_URL:b'abcdef'*500});endpoint.truncate=True
            with self.assertRaises(ValueError):d.fetch_archive(d.CUB_URL,cache/'CUB.tgz',endpoint)
            part=cache/'CUB.tgz.part';before=part.read_bytes()
            endpoint.truncate=False
            with self.assertRaisesRegex(ValueError,'cache hard budget'):
                d.fetch_archive(d.CUB_URL,cache/'CUB.tgz',endpoint,cache_limit=1000)
            self.assertEqual(part.read_bytes(),before)
            d.fetch_archive(d.CUB_URL,cache/'CUB.tgz',endpoint)
            d.cleanup_archive(cache/'CUB.tgz',d.CUB_URL)
            self.assertFalse(list(cache.glob('*.tgz*')))
            unknown=cache/'unknown.tgz';unknown.write_bytes(b'normal existing data')
            with self.assertRaises(ValueError):d.cleanup_archive(unknown,d.CUB_URL)
            self.assertTrue(unknown.exists())

    def test_archive_removal_node_manifest_and_exact_counts(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'source';source.mkdir();fixture(source)
            root=Path(td)/'ssd';endpoint=Endpoint(objects(source));policy=FixturePolicy(root)
            d.prepare(root,endpoint,bird_count=160,scene_count=20,policy=policy)
            self.assertFalse(list(root.rglob('*.tgz')));self.assertFalse(list(policy.cache.glob('*.tgz*')))
            stamp=root/'.waterbirds-node.json';self.assertTrue(stamp.exists())
            manifest=Path(td)/'manifest';protocol_prepare(root,manifest)
            with patch('scripts.waterbirds_vgg50.protocol.socket.gethostname',return_value='other-node'):
                with self.assertRaisesRegex(ValueError,'another node'):load(manifest/'protocol.json',.95)
            mask=next((root/'CUB_200_2011/segmentations').rglob('*.png'))
            (mask.parent/'extra.png').write_bytes(mask.read_bytes())
            with self.assertRaisesRegex(ValueError,'file count'):
                d.validate_cub(root/'CUB_200_2011/segmentations',160,masks=True,rgb_root=root/'CUB_200_2011')


class CurlTests(unittest.TestCase):
    def test_argv_and_error_headers_no_body_read(self):
        class Process:
            def __init__(self):self.stdout=io.BytesIO(b'HTTP/1.1 200 OK\r\nContent-Length: 999999999\r\n\r\nDO NOT READ');self.returncode=None
            def poll(self):return self.returncode
            def terminate(self):self.returncode=-15
            def wait(self,timeout=None):return self.returncode or 0
        process=Process()
        def launch(argv,**kw):
            self.assertIsInstance(argv,list);self.assertFalse(kw['shell'])
            self.assertNotIn('-L',argv);self.assertIn('--suppress-connect-headers',argv)
            self.assertIn('Range: bytes=0-0',argv)
            return process
        with patch('scripts.waterbirds_vgg50.curl_transport.subprocess.Popen',side_effect=launch):
            response=CurlResponse(urllib.request.Request('https://example.invalid',headers={'Range':'bytes=0-0'}),30)
            self.assertEqual(response.status,200)
            self.assertEqual(process.stdout.read(),b'DO NOT READ') # Test inspection only; production did not read body.
            response.close();self.assertEqual(process.returncode,-15)

    def test_real_curl_loopback_redirect_range_resume(self):
        state={'data':b'abcdef'*400,'truncate':False,'ignore':False,'requests':[]}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*a):pass
            def do_HEAD(self):self.handle_request(True)
            def do_GET(self):self.handle_request(False)
            def handle_request(self,head):
                state['requests'].append((self.command,self.path,self.headers.get('Range'),self.headers.get('If-Range'),self.headers.get('User-Agent')))
                if self.path=='/start':
                    self.send_response(302);self.send_header('Location','/object?sig=%2F%2B');self.send_header('Content-Length','0');self.end_headers();return
                if head:self.send_response(403);self.send_header('Content-Length','0');self.end_headers();return
                payload=state['data'];interval=self.headers.get('Range')
                code=200
                if interval and not state['ignore']:
                    a,b=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',interval).groups())
                    payload=payload[a:b+1];code=206
                self.send_response(code)
                self.send_header('Content-Length',str(len(payload)));self.send_header('ETag','"stable"')
                self.send_header('Last-Modified','Mon, 16 Nov 2026 00:00:00 GMT')
                if code==206:self.send_header('Content-Range',f'bytes {a}-{b}/{len(state["data"])}')
                self.end_headers()
                if state['truncate'] and interval!='bytes=0-0':payload=payload[:len(payload)//2]
                try:self.wfile.write(payload)
                except (BrokenPipeError,ConnectionResetError):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with patch.dict(os.environ,{'NO_PROXY':'127.0.0.1','no_proxy':'127.0.0.1'}):
                url=f'http://127.0.0.1:{server.server_port}/start';opener=CurlOpener()
                remote=d.Remote(url,opener)
                self.assertEqual(remote.metadata_method,'Range GET');self.assertEqual(remote.range(0,0),b'a')
                self.assertTrue(any(method=='HEAD' and path.startswith('/object') for method,path,*rest in state['requests']))
                self.assertTrue(all(ua.startswith('curl/') for *rest,ua in state['requests']))
                self.assertTrue(any(path.endswith('?sig=%2F%2B') and interval=='bytes=0-0' for method,path,interval,*rest in state['requests']))
                state['ignore']=True
                with self.assertRaisesRegex(ValueError,'206'):remote.range(0,0)
                state['ignore']=False;state['truncate']=True
                with tempfile.TemporaryDirectory() as td:
                    dest=Path(td)/'CUB.tgz'
                    with self.assertRaises(OSError):d.fetch_archive(url,dest,opener)
                    self.assertFalse(dest.exists());self.assertTrue(Path(str(dest)+'.part').exists())
                    # The first truncated curl read can be buffered; force a verified prefix for the resume test.
                    part=Path(str(dest)+'.part');part.write_bytes(state['data'][:100])
                    state['truncate']=False;d.fetch_archive(url,dest,opener)
                    self.assertEqual(dest.read_bytes(),state['data'])
                    self.assertTrue(any(interval=='bytes=100-2399' and validator=='"stable"' for method,path,interval,validator,ua in state['requests']))
        finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
