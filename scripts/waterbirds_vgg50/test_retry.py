import json
import io
import urllib.request
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.waterbirds_vgg50 import download as d
from scripts.waterbirds_vgg50.curl_transport import RetryPolicy, CurlTransportError, RetryExhausted
from scripts.waterbirds_vgg50.range_cache import RangeCache
from scripts.waterbirds_vgg50.test_download import Endpoint, Reply, FixturePolicy, objects
from scripts.waterbirds_vgg50.test_waterbirds import fixture


class Tests(unittest.TestCase):
    def test_curl_process_dns_error_is_classified(self):
        from scripts.waterbirds_vgg50.curl_transport import CurlResponse
        class Process:
            def __init__(self):self.stdout=io.BytesIO(b'')
            def wait(self,timeout=None):return 6
            def poll(self):return 6
        def launch(argv,**kwargs):
            kwargs['stderr'].write(b'curl: (6) Could not resolve host: data.csail.mit.edu')
            return Process()
        with patch('scripts.waterbirds_vgg50.curl_transport.subprocess.Popen',side_effect=launch):
            with self.assertRaises(CurlTransportError) as raised:
                CurlResponse(urllib.request.Request('https://data.csail.mit.edu/places/archive.tar'),30)
        self.assertEqual(raised.exception.code,6)
        self.assertEqual(raised.exception.host,'data.csail.mit.edu')

    def test_retry_bound_backoff_and_final_host_code(self):
        delays=[];calls=[];policy=RetryPolicy(sleep=delays.append)
        def fail():
            calls.append(1);raise CurlTransportError(6,'data.csail.mit.edu','DNS failure')
        with self.assertRaises(RetryExhausted) as raised:policy.run('Range bytes=10-19',fail)
        self.assertEqual(len(calls),6);self.assertEqual(delays,[1,2,4,8,16])
        error=raised.exception
        self.assertEqual(error.retries,5);self.assertEqual(error.code,6)
        for text in ('data.csail.mit.edu','retries=5','attempts=6','last_error_code=6','bytes=10-19'):
            self.assertIn(text,str(error))
        for code in (5,6,7,28,52,55,56):
            with self.subTest(code=code):
                calls=[]
                def once():
                    calls.append(1)
                    if len(calls)==1:raise CurlTransportError(code,'target.invalid','temporary')
                    return b'ok'
                self.assertEqual(policy.run('probe',once),b'ok');self.assertEqual(len(calls),2)
        with self.assertRaises(ValueError):RetryPolicy(max_retries=6)

    def test_permanent_failures_are_not_retried(self):
        delays=[];policy=RetryPolicy(sleep=delays.append)
        for code in (18,22,60):
            with self.subTest(code=code):
                calls=[]
                def fail():calls.append(1);raise CurlTransportError(code,'target.invalid','permanent')
                with self.assertRaises(CurlTransportError):policy.run('range',fail)
                self.assertEqual(len(calls),1)
        self.assertEqual(delays,[])
        for status,changes in ((403,{}),(404,{}),(206,{'Content-Range':'bytes 1-10/100'}),
                               (206,{'ETag':'"changed"'}),(206,{'Last-Modified':'changed'})):
            endpoint=Endpoint({'https://target.invalid':b'x'*100});endpoint.retry_policy=policy
            def opener(req,timeout=0):
                response=endpoint(req,timeout)
                response.headers['Last-Modified']='fixed'
                if req.get_method()=='GET':response.status=status;response.headers.update(changes)
                return response
            opener.retry_policy=policy
            remote=d.Remote('https://target.invalid',opener)
            with self.assertRaises(ValueError):remote.range(0,9)
            self.assertEqual(len(endpoint.requests),2) # HEAD plus one GET only.
            self.assertFalse(endpoint.last_response.body_read)
        self.assertEqual(delays,[])

    def test_failed_range_only_retried_including_body_error(self):
        url='https://target.invalid';endpoint=Endpoint({url:b'abcdefghijklmnopqrst'})
        policy=RetryPolicy(sleep=lambda n:None);counts={}
        def opener(req,timeout=0):
            interval=req.get_header('Range');counts[interval]=counts.get(interval,0)+1
            if interval=='bytes=10-19' and counts[interval]==1:
                response=endpoint(req,timeout)
                def fail(n):raise CurlTransportError(28,'target.invalid','body timeout')
                response.read=fail
                return response
            return endpoint(req,timeout)
        opener.retry_policy=policy
        remote=d.Remote(url,opener,budget=30)
        self.assertEqual(remote.range(0,9),b'abcdefghij')
        self.assertEqual(remote.range(10,19),b'klmnopqrst')
        self.assertEqual(counts['bytes=0-9'],1);self.assertEqual(counts['bytes=10-19'],2)
        self.assertEqual(remote.used,30) # Retried byte ranges count against the cap.
        with self.assertRaisesRegex(ValueError,'budget'):remote.range(0,0)

    def test_cache_reuse_integrity_and_changed_validator(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cache=RangeCache(root/'category',root)
            endpoint=Endpoint({'https://target.invalid':b'x'*100})
            remote=d.Remote('https://target.invalid',endpoint,range_cache=cache)
            remote.range(0,9);before=len(endpoint.requests)
            self.assertEqual(remote.range(0,9),b'x'*10)
            self.assertEqual(len(endpoint.requests),before)
            endpoint.etag='"new-version"'
            changed=d.Remote('https://target.invalid',endpoint,range_cache=cache)
            with self.assertRaisesRegex(ValueError,'validator changed'):changed.range(0,9)
            self.assertEqual(len(endpoint.requests),before+1) # New HEAD, no GET on source mismatch.
            endpoint.etag='"v1"'
            blob=next((root/'category').glob('*.chunk'));blob.write_bytes(b'y'*10)
            with self.assertRaisesRegex(ValueError,'integrity'):remote.range(0,9)
            with patch('scripts.waterbirds_vgg50.range_cache.CACHE_CAP',1):
                with self.assertRaisesRegex(ValueError,'hard budget'):remote.range(10,19)
            unexpected=root/'category'/'user-note.txt';unexpected.write_text('preserve')
            with self.assertRaisesRegex(ValueError,'preserving cache'):cache.clear()
            self.assertEqual(unexpected.read_text(),'preserve')

    def test_category_restart_uses_ranges_and_preserves_cub(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'ssd';source=Path(td)/'source';source.mkdir();fixture(source)
            # Few MiB total; exercise multiple 1MiB TAR reads, no external network.
            for path in (source/'places365').rglob('*.jpg'):
                with path.open('ab') as f:f.write(b'\0'*(64*1024))
            endpoint=Endpoint(objects(source));active=[''];failures=[True];attempts=[]
            def opener(req,timeout=0):
                interval=req.get_header('Range');attempts.append((active[0],req.full_url,req.get_method(),interval))
                if failures[0] and active[0]=='b/bamboo_forest' and interval=='bytes=1048576-2097151':
                    raise CurlTransportError(6,'data.csail.mit.edu','injected DNS failure')
                return endpoint(req,timeout)
            opener.retry_policy=RetryPolicy(sleep=lambda n:None)
            original=d.fetch_category
            def category(url,name,*args,**kwargs):
                active[0]=name
                try:return original(url,name,*args,**kwargs)
                finally:active[0]=''
            policy=FixturePolicy(root)
            with patch.object(d,'fetch_category',side_effect=category):
                with self.assertRaises(RetryExhausted):d.prepare(root,opener,bird_count=160,scene_count=20,policy=policy)
                self.assertTrue((root/'CUB_200_2011/.complete').exists())
                self.assertTrue((root/'CUB_200_2011/segmentations/.complete').exists())
                before={str(p):d.sha(p) for p in (root/'CUB_200_2011').rglob('*') if p.is_file()}
                category_root=root/'places365/data_256/b/bamboo_forest'
                self.assertFalse(category_root.exists());self.assertFalse(list(root.rglob('*.stage-*')))
                self.assertTrue(list((policy.cache/'.places-ranges').rglob('*.chunk')))
                successful_first=('b/bamboo_forest',d.PLACES_URL,'GET','bytes=0-1048575')
                self.assertIn(successful_first,attempts)
                attempts.clear();failures[0]=False
                d.prepare(root,opener,bird_count=160,scene_count=20,policy=policy)
                self.assertNotIn(successful_first,attempts)
                self.assertFalse(any(url in (d.CUB_URL,d.SEG_URL) for _,url,_,_ in attempts))
                self.assertEqual(before,{str(p):d.sha(p) for p in (root/'CUB_200_2011').rglob('*') if p.is_file()})
                self.assertEqual(len(list(category_root.glob('*.jpg'))),20)
                self.assertTrue((category_root/'.complete').exists())
                self.assertFalse(list((policy.cache/'.places-ranges').rglob('*.chunk')))
                attempts.clear();d.prepare(root,opener,bird_count=160,scene_count=20,policy=policy)
                self.assertEqual(attempts,[])

if __name__=='__main__':unittest.main()
