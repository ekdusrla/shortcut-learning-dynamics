"""No sockets/network: real tiny JPEG/tar archives and injected HTTP responses."""
import io
import json
import re
import tarfile
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import pandas as pd
from scripts.waterbirds_vgg50 import download as d
from scripts.waterbirds_vgg50.test_waterbirds import fixture
from scripts.waterbirds_vgg50.protocol import prepare as make_protocol, load, CORRS


class Reply(io.BytesIO):
    def __init__(self, data=b'', status=200, headers=None):
        super().__init__(data);self.status=status;self.headers=headers or {};self.body_read=False;self.read_sizes=[]
    def read(self,n=-1):self.body_read=True;self.read_sizes.append(n);return super().read(n)


class Endpoint:
    def __init__(self, objects):self.objects=objects;self.requests=[];self.ignore_range=False;self.truncate=False;self.etag='"v1"';self.last_response=None
    def __call__(self,req,timeout=0):
        body=self.objects[req.full_url];headers={'Content-Length':str(len(body)),'ETag':self.etag}
        interval=req.get_header('Range');self.requests.append((req.full_url,req.get_method(),interval))
        status=200
        if req.get_method()=='HEAD':payload=b''
        elif interval and not self.ignore_range:
            match=re.fullmatch(r'bytes=(\d+)-(\d+)',interval)
            first,last=map(int,match.groups());payload=body[first:last+1]
            status=206;headers.update({'Content-Range':f'bytes {first}-{last}/{len(body)}','Content-Length':str(last-first+1)})
        else:payload=body
        if self.truncate and req.get_method()!='HEAD':payload=payload[:max(1,len(payload)//2)]
        self.last_response=Reply(payload,status,headers);return self.last_response


class HeadForbiddenEndpoint(Endpoint):
    def __init__(self, objects, raised=True):
        super().__init__(objects);self.raised=raised;self.errors=[];self.if_ranges=[]
        self.modified='Mon, 16 Nov 2026 00:00:00 GMT'
    def __call__(self, req, timeout=0):
        if req.get_method()=='HEAD':
            self.requests.append((req.full_url,'HEAD',None))
            response=Reply(b'forbidden error page',403,{'ETag':'"error-page"'})
            self.errors.append(response)
            if self.raised:
                raise urllib.error.HTTPError(req.full_url,403,'Forbidden',response.headers,response)
            return response
        if req.get_header('Range') and req.get_header('Range')!='bytes=0-0':
            self.if_ranges.append(req.get_header('If-range'))
        response=super().__call__(req,timeout)
        response.headers['Last-Modified']=self.modified
        return response


def tar_bytes(root, paths, mode='w:gz'):
    out=io.BytesIO()
    with tarfile.open(fileobj=out,mode=mode,format=tarfile.USTAR_FORMAT) as archive:
        for source,name in paths:
            payload=Path(source).read_bytes();info=tarfile.TarInfo(name);info.size=len(payload)
            archive.addfile(info,io.BytesIO(payload))
    return out.getvalue()


def objects(root):
    cub=root/'CUB_200_2011'
    rgb=[(cub/name,'CUB_200_2011/'+name) for name in ('images.txt','train_test_split.txt')]
    rgb += [(p,'CUB_200_2011/'+str(p.relative_to(cub))) for p in sorted((cub/'images').rglob('*.jpg'))]
    extra=cub/'attributes/unneeded.txt';extra.parent.mkdir(exist_ok=True);extra.write_text('unused annotation')
    rgb.append((extra,'CUB_200_2011/attributes/unneeded.txt'))
    masks=[(p,'segmentations/'+str(p.relative_to(cub/'segmentations'))) for p in sorted((cub/'segmentations').rglob('*.png'))]
    scenes=[(p,'data_256/'+str(p.relative_to(root/'places365/data_256'))) for p in sorted((root/'places365/data_256').rglob('*.jpg'))]
    return {d.CUB_URL:tar_bytes(root,rgb),d.SEG_URL:tar_bytes(root,masks),d.PLACES_URL:tar_bytes(root,scenes,'w')}


class FixturePolicy:
    def __init__(self,root):self.cache=Path(root).parent/'network-cache'
    def preflight(self):return {'fixture_storage':True}


class Tests(unittest.TestCase):
    def test_head403_range_metadata(self):
        url=d.CUB_URL
        for raised in (True,False):
            with self.subTest(raised_http_error=raised):
                endpoint=HeadForbiddenEndpoint({url:b'abc'*100},raised=raised)
                remote=d.Remote(url,endpoint,budget=10)
                self.assertEqual(remote.total,300)
                self.assertEqual(remote.used,1)
                self.assertEqual(remote.metadata_method,'Range GET')
                self.assertEqual(remote.etag,'"v1"')
                self.assertEqual(remote.modified,endpoint.modified)
                self.assertEqual(endpoint.last_response.read_sizes,[2])
                self.assertEqual(remote.range(0,2),b'abc')
                self.assertEqual(endpoint.if_ranges,['"v1"'])
                self.assertTrue(all(e.closed and not e.body_read for e in endpoint.errors))
                endpoint.modified='Tue, 17 Nov 2026 00:00:00 GMT'
                with self.assertRaisesRegex(ValueError,'Last-Modified'):remote.range(0,2)
        endpoint=HeadForbiddenEndpoint({url:b'ab'})
        def failed_head(req,timeout=0):
            if req.get_method()=='HEAD':
                raise urllib.error.HTTPError(url,500,'Error',{},None)
            self.fail('Only HEAD 403 may trigger fallback')
        with self.assertRaises(urllib.error.HTTPError):d.Remote(url,failed_head)

    def test_head403_probe_rejects_unsafe_responses(self):
        url=d.CUB_URL
        endpoint=HeadForbiddenEndpoint({url:b'a'*2048});endpoint.ignore_range=True
        with self.assertRaisesRegex(ValueError,'206'):d.Remote(url,endpoint)
        self.assertFalse(endpoint.last_response.body_read)
        for change in ({'Content-Range':'bytes 1-1/2048'}, {'Content-Range':'bytes 0-0/*'},
                       {'Content-Range':'bytes 0-0/0'}, {'Content-Length':'2048'},
                       {'Content-Length':None}, {'Content-Encoding':'gzip'}):
            with self.subTest(headers=change):
                endpoint=HeadForbiddenEndpoint({url:b'a'*2048})
                def altered(req,timeout=0):
                    response=endpoint(req,timeout)
                    response.headers.update(change)
                    return response
                with self.assertRaises(ValueError):d.Remote(url,altered)
                self.assertFalse(endpoint.last_response.body_read)
        for payload in (b'',b'ab'):
            endpoint=HeadForbiddenEndpoint({url:b'a'*2048})
            def body_override(req,timeout=0):
                response=endpoint(req,timeout)
                response.seek(0);response.truncate();response.write(payload);response.seek(0)
                return response
            with self.assertRaisesRegex(ValueError,'Truncated or oversized'):d.Remote(url,body_override)
            self.assertEqual(endpoint.last_response.read_sizes,[2])
        endpoint=HeadForbiddenEndpoint({url:b'a'*2048})
        with self.assertRaisesRegex(ValueError,'budget'):d.Remote(url,endpoint,budget=0)
        self.assertEqual(len(endpoint.requests),1) # HEAD only; no GET exceeds the budget.

    def test_head403_archive_resume_and_places_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            endpoint=HeadForbiddenEndpoint({d.CUB_URL:b'abcde'*300});endpoint.truncate=True
            dest=Path(td)/'archive.tgz'
            with self.assertRaisesRegex(ValueError,'truncated'):d.fetch_archive(d.CUB_URL,dest,endpoint)
            part=Path(str(dest)+'.part');partial_size=part.stat().st_size
            metadata=json.loads(Path(str(part)+'.json').read_text())
            self.assertEqual(metadata['modified'],endpoint.modified)
            endpoint.truncate=False
            d.fetch_archive(d.CUB_URL,dest,endpoint)
            self.assertEqual(dest.read_bytes(),endpoint.objects[d.CUB_URL])
            self.assertIn((d.CUB_URL,'GET',f'bytes={partial_size}-1499'),endpoint.requests)
            self.assertEqual(endpoint.if_ranges[-1],'"v1"')
            self.assertFalse(list(Path(td).glob('*.stale-*')))
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'source';source.mkdir();fixture(source)
            endpoint=HeadForbiddenEndpoint(objects(source));root=Path(td)/'prepared'
            d.prepare(root,endpoint,bird_count=160,scene_count=20,policy=FixturePolicy(root))
            protocol=Path(td)/'protocol';meta=make_protocol(root,protocol)
            self.assertEqual(meta['val_corr'],.5);self.assertEqual(meta['test_corr'],.5)
            for corr in CORRS:self.assertIn('cf_pixel_values',load(protocol/'protocol.json',corr)['test'][0])
            endpoint.requests.clear();d.prepare(root,endpoint,bird_count=160,scene_count=20,policy=FixturePolicy(root))
            self.assertEqual(endpoint.requests,[])

    def test_range_validation_and_budget(self):
        endpoint=Endpoint({'https://example.invalid/archive':b'a'*2048});remote=d.Remote('https://example.invalid/archive',endpoint)
        self.assertEqual(remote.range(512,1023),b'a'*512)
        endpoint.ignore_range=True
        with self.assertRaisesRegex(ValueError,'206'):remote.range(0,511)
        self.assertFalse(endpoint.last_response.body_read)
        endpoint.ignore_range=False;endpoint.truncate=True
        with self.assertRaisesRegex(ValueError,'Truncated'):remote.range(0,511)
        endpoint.truncate=False;endpoint.etag='"v2"'
        with self.assertRaisesRegex(ValueError,'changed'):remote.range(0,511)
        endpoint.etag='"v1"';remote=d.Remote('https://example.invalid/archive',endpoint,budget=511)
        with self.assertRaisesRegex(ValueError,'budget'):remote.range(0,511)
        for change in ({'Content-Range':'bytes 1-512/2048'}, {'Content-Length':'513'},{'Content-Encoding':'gzip'}):
            def altered(req,timeout=0):
                response=endpoint(req,timeout)
                if req.get_method()!='HEAD':response.headers.update(change)
                return response
            r=d.Remote('https://example.invalid/archive',altered)
            with self.assertRaises(ValueError):r.range(0,511)
            self.assertFalse(endpoint.last_response.body_read)

    def test_archive_interruption_resume_and_version_change(self):
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/'archive.tgz';endpoint=Endpoint({'https://example.invalid/archive':b'abcde'*300})
            endpoint.truncate=True
            with self.assertRaisesRegex(ValueError,'truncated'):d.fetch_archive('https://example.invalid/archive',dest,endpoint)
            self.assertFalse(dest.exists());part=Path(str(dest)+'.part');self.assertGreater(part.stat().st_size,0)
            endpoint.truncate=False
            d.fetch_archive('https://example.invalid/archive',dest,endpoint);self.assertEqual(dest.read_bytes(),endpoint.objects['https://example.invalid/archive'])
            self.assertTrue(any(method=='GET' and interval for _,method,interval in endpoint.requests))
            requests=len(endpoint.requests);d.fetch_archive('https://example.invalid/archive',dest,endpoint)
            self.assertEqual(len(endpoint.requests),requests+1) # Only HEAD on verified cache reuse.
            endpoint.etag='"v2"'
            with self.assertRaisesRegex(ValueError,'source changed'):d.fetch_archive('https://example.invalid/archive',dest,endpoint)
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/'archive';endpoint=Endpoint({'https://example.invalid/archive':b'a'*2000});endpoint.truncate=True
            with self.assertRaises(ValueError):d.fetch_archive('https://example.invalid/archive',dest,endpoint)
            endpoint.truncate=False;endpoint.ignore_range=True
            before=Path(str(dest)+'.part').read_bytes()
            with self.assertRaisesRegex(ValueError,'resume'):d.fetch_archive('https://example.invalid/archive',dest,endpoint)
            self.assertEqual(Path(str(dest)+'.part').read_bytes(),before)
            self.assertFalse(endpoint.last_response.body_read)
            endpoint.ignore_range=False;endpoint.etag='"v2"'
            d.fetch_archive('https://example.invalid/archive',dest,endpoint)
            self.assertEqual(len(list(Path(td).glob('*.stale-*'))),2) # Old partial + metadata preserved.

    def test_extraction_failure_never_publishes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);archive=root/'bad.tgz';archive.write_bytes(b'not gzip')
            final=root/'final'
            with self.assertRaises(tarfile.TarError):
                d.reuse_or_publish(final,lambda stage:d.extract_archive(archive,stage),lambda stage:{})
            self.assertFalse(final.exists());self.assertFalse(any(root.glob('.*.stage-*/.complete')))
            out=io.BytesIO()
            with tarfile.open(fileobj=out,mode='w:gz') as tf:
                info=tarfile.TarInfo('../escape');info.size=1;tf.addfile(info,io.BytesIO(b'x'))
            archive.write_bytes(out.getvalue())
            with self.assertRaisesRegex(ValueError,'Unsafe'):d.extract_archive(archive,root/'safe')
            self.assertFalse((root/'escape').exists())
            out=io.BytesIO()
            with tarfile.open(fileobj=out,mode='w:gz') as tf:
                info=tarfile.TarInfo('link');info.type=tarfile.SYMTYPE;info.linkname='/tmp';tf.addfile(info)
            archive.write_bytes(out.getvalue())
            with self.assertRaisesRegex(ValueError,'Unsupported'):d.extract_archive(archive,root/'safe')

    def test_marker_partial_component_and_hash_change(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)/'category';folder.mkdir();(folder/'.complete').write_text('5000')
            with self.assertRaises(ValueError):d.reuse_or_publish(folder,lambda p:self.fail('must not overwrite'),lambda p:d.validate_category(p,1))
            self.assertEqual((folder/'.complete').read_text(),'5000')
            linked=Path(td)/'linked';linked.symlink_to(folder,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'symlink'):
                d.reuse_or_publish(linked/'missing',lambda p:None,lambda p:{})
            self.assertFalse((folder/'missing').exists())
            new=Path(td)/'new'
            def populate(p):Image.new('RGB',(8,8),'red').save(p/'x.jpg')
            d.reuse_or_publish(new,populate,lambda p:d.validate_category(p,1))
            meta=json.loads((new/'.complete').read_text());self.assertEqual(len(meta['files']),1)
            Image.new('RGB',(8,8),'blue').save(new/'x.jpg')
            with self.assertRaisesRegex(ValueError,'changed'):d.reuse_or_publish(new,populate,lambda p:d.validate_category(p,1))

    def test_places_search_and_category_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=io.BytesIO()
            jpeg=io.BytesIO();Image.new('RGB',(8,8),'red').save(jpeg,format='JPEG')
            payload=jpeg.getvalue()+b'\0'*(128*1024)
            with tarfile.open(fileobj=out,mode='w',format=tarfile.USTAR_FORMAT) as tf:
                for category in sorted(d.CATEGORIES):
                    for i in range(3):
                        info=tarfile.TarInfo(f'data_256/{category}/{i}.jpg');info.size=len(payload)
                        tf.addfile(info,io.BytesIO(payload))
            endpoint=Endpoint({d.PLACES_URL:out.getvalue()})
            for category in d.CATEGORIES:
                folder=root/category.split('/')[-1];folder.mkdir()
                d.fetch_category(d.PLACES_URL,category,folder,endpoint,expected=3)
                self.assertEqual(len(d.validate_category(folder,3)),3)
            endpoint.truncate=True
            final=root/'failed'
            with self.assertRaises(ValueError):
                d.reuse_or_publish(final,lambda stage:d.fetch_category(d.PLACES_URL,'b/bamboo_forest',stage,endpoint,3),lambda p:d.validate_category(p,3))
            self.assertFalse(final.exists());self.assertFalse(any(root.glob('.failed.stage-*/.complete')))
            endpoint.truncate=False
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                d.reuse_or_publish(final,lambda stage:d.fetch_category(d.PLACES_URL,'b/bamboo_forest',stage,endpoint,4),lambda p:d.validate_category(p,4))
            d.reuse_or_publish(final,lambda stage:d.fetch_category(d.PLACES_URL,'b/bamboo_forest',stage,endpoint,3),lambda p:d.validate_category(p,3))
            self.assertTrue((final/'.complete').exists())

    def test_crc_mask_shape_and_lock(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source';source.mkdir();fixture(source)
            image=source/'CUB_200_2011/images/001.Albatross/1.jpg'
            archive=root/'crc.tgz'
            body=tar_bytes(source,[(image,'sample.jpg')])
            corrupt=bytearray(body);corrupt[-8]^=255;archive.write_bytes(corrupt)
            with self.assertRaises((OSError,EOFError,tarfile.TarError)):
                d.reuse_or_publish(root/'bad',lambda stage:d.extract_archive(archive,stage),lambda p:{})
            self.assertFalse((root/'bad').exists())
            Image.new('RGB',(10,10),'white').save(source/'CUB_200_2011/segmentations/001.Albatross/1.png')
            with self.assertRaisesRegex(ValueError,'shape mismatch'):
                d.validate_cub(source/'CUB_200_2011/segmentations',160,masks=True,rgb_root=source/'CUB_200_2011')
            with d.lock_root(root/'locked'):
                with self.assertRaisesRegex(RuntimeError,'Another'):
                    with d.lock_root(root/'locked'):pass

    def test_complete_pipeline_loader_and_restart_without_download(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'source';source.mkdir();fixture(source)
            endpoint=Endpoint(objects(source));root=Path(td)/'prepared'
            d.prepare(root,endpoint,bird_count=160,scene_count=20,policy=FixturePolicy(root))
            for category in d.CATEGORIES:self.assertEqual(len(list((root/'places365/data_256'/category).glob('*.jpg'))),20)
            self.assertTrue((root/'CUB_200_2011/.complete').is_file())
            self.assertFalse((root/'CUB_200_2011/attributes').exists())
            self.assertFalse(list(FixturePolicy(root).cache.glob('*.tgz')))
            self.assertFalse(list(root.rglob('*.tgz')))
            before={str(p):d.sha(p) for p in root.rglob('*') if p.is_file() and not p.name.endswith('.lock')}
            endpoint.requests.clear();d.prepare(root,endpoint,bird_count=160,scene_count=20,policy=FixturePolicy(root))
            self.assertEqual(endpoint.requests,[])
            self.assertEqual(before,{str(p):d.sha(p) for p in root.rglob('*') if p.is_file() and not p.name.endswith('.lock')})
            protocol=Path(td)/'protocol';meta=make_protocol(root,protocol)
            self.assertEqual(meta['val_corr'],.5);self.assertEqual(meta['test_corr'],.5)
            for corr in CORRS:
                datasets=load(protocol/'protocol.json',corr)
                sample=datasets['test'][0]
                self.assertEqual(tuple(sample['pixel_values'].shape),(3,224,224))
                self.assertIn('cf_pixel_values',sample)
                self.assertEqual(set(datasets['test'].labels),{0,1})

if __name__=='__main__':unittest.main()
