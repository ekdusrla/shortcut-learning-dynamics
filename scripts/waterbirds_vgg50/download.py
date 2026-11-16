"""Transactional CUB+Places preparation, isolated from shared experiment code.

Network is used only by the normal CLI. Tests inject an in-memory HTTP endpoint.
Existing final data is validated and reused, never overwritten or repaired in place.
"""
import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request
import urllib.error
import uuid
import sys
import socket
from contextlib import contextmanager
from PIL import Image

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.waterbirds_vgg50.storage import StoragePolicy, DataBudget, CACHE_CAP, tree_bytes
from scripts.waterbirds_vgg50.curl_transport import CurlOpener, RetryPolicy
from scripts.waterbirds_vgg50.range_cache import RangeCache

CUB_URL = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
SEG_URL = 'https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1'
PLACES_URL = 'https://data.csail.mit.edu/places/places365/train_256_places365standard.tar'
CATEGORIES = ('b/bamboo_forest', 'f/forest/broadleaf', 'o/ocean', 'l/lake/natural')
CHUNK = 1024 * 1024


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with tmp.open('x') as f:
        json.dump(payload, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def checked_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts or '\\' in name:
        raise ValueError(f'Unsafe archive/metadata path: {name}')
    return path


class Remote:
    def __init__(self, url, opener=None, budget=None, range_cache=None):
        self.url = url
        self.opener = opener or urllib.request.urlopen
        self.budget = budget
        self.used = 0
        self.retry=getattr(self.opener,'retry_policy',RetryPolicy(max_retries=0))
        self.range_cache=range_cache
        # Redirected object stores can forbid HEAD while allowing Range GET.
        # Do not use the 403 error page as object metadata or consume its body.
        try:
            response = self.retry.run('HEAD metadata',lambda:self.opener(urllib.request.Request(
                url, method='HEAD', headers={'Accept-Encoding':'identity'}), timeout=30))
        except urllib.error.HTTPError as error:
            if error.code != 403:
                error.close()
                raise
            error.close()
            response = None
        if response is not None:
            with response as r:
                if r.status == 200:
                    self._set_metadata(r.headers, int(r.headers['Content-Length']))
                    self.metadata_method = 'HEAD'
                elif r.status != 403:
                    raise ValueError(f'Invalid HEAD response: {r.status}')
        if not hasattr(self, 'total'):
            self._range_metadata()
        self.validator = self.etag if self.etag and not self.etag.startswith('W/') else self.modified

    def _set_metadata(self, headers, total):
        if headers.get('Content-Encoding', 'identity') != 'identity':
            raise ValueError('Encoded response is incompatible with byte offsets')
        if total <= 0:
            raise ValueError('Empty remote object')
        self.total = total
        self.etag = headers.get('ETag')
        self.modified = headers.get('Last-Modified')

    def _range_metadata(self):
        return self.retry.run('Range metadata bytes=0-0',self._range_metadata_once)

    def reserve(self,n):
        if self.budget is not None and self.used+n>self.budget:
            raise ValueError('Category transfer budget exceeded; retries included')
        self.used+=n # Conservative requested byte upper bound, including failed attempts.

    def _range_metadata_once(self):
        self.reserve(1)
        request = urllib.request.Request(self.url, headers={
            'Range':'bytes=0-0', 'Accept-Encoding':'identity'})
        with self.opener(request, timeout=30) as r:
            if r.status != 206:
                raise ValueError(f'Expected HTTP 206 for metadata, got {r.status}; body not read')
            match = re.fullmatch(r'bytes 0-0/([1-9][0-9]*)', r.headers.get('Content-Range', ''))
            if not match:
                raise ValueError('Metadata Content-Range mismatch')
            if r.headers.get('Content-Length') != '1':
                raise ValueError('Metadata Content-Length must be 1')
            self._set_metadata(r.headers, int(match[1]))
            payload = r.read(2)  # At most two bytes: also detect an oversized body.
            if len(payload) != 1:
                raise ValueError('Truncated or oversized metadata Range response')
            self.metadata_method = 'Range GET'

    def identity(self):
        return dict(url=self.url, total=self.total, etag=self.etag, modified=self.modified)

    def check_identity(self, response):
        for name, expected in (('ETag', self.etag), ('Last-Modified', self.modified)):
            if expected is not None and response.headers.get(name) != expected:
                raise ValueError(f'Remote object changed or validator missing: {name}')
        if response.headers.get('Content-Encoding', 'identity') != 'identity':
            raise ValueError('Encoded response is incompatible with byte offsets')

    def range(self, start, end):
        if not (0<=start<=end<self.total):raise ValueError('Invalid byte interval')
        if self.range_cache is not None:
            cached=self.range_cache.get(self,start,end)
            if cached is not None:return cached
        payload=self.retry.run(f'Range bytes={start}-{end}',lambda:self._range_once(start,end))
        if self.range_cache is not None:self.range_cache.put(self,start,end,payload)
        return payload

    def _range_once(self, start, end):
        if not (0 <= start <= end < self.total): raise ValueError('Invalid byte interval')
        n = end - start + 1
        self.reserve(n)
        headers = {'Range':f'bytes={start}-{end}', 'Accept-Encoding':'identity'}
        if self.validator: headers['If-Range'] = self.validator
        with self.opener(urllib.request.Request(self.url, headers=headers), timeout=60) as r:
            if r.status != 206:
                raise ValueError(f'Expected HTTP 206, got {r.status}; body not read')
            expected = f'bytes {start}-{end}/{self.total}'
            if r.headers.get('Content-Range') != expected:
                raise ValueError('Content-Range mismatch')
            if r.headers.get('Content-Length') not in (None, str(n)):
                raise ValueError('Range Content-Length mismatch')
            self.check_identity(r)
            payload = r.read(n + 1)
            if len(payload) != n: raise ValueError('Truncated or oversized Range response')
            return payload


def fetch_archive(url, path, opener=None, cache_limit=None):
    """Resume only matching source validators; archive finalization follows exact length.

    Failed reads retain .part + source metadata. Extraction validates gzip/tar too.
    """
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    remote = Remote(url, opener)
    if remote.total > 4*1024*1024*1024:
        raise ValueError('Unexpected CUB archive size exceeds 4 GiB; no transfer started')
    stamp = path.with_name(path.name + '.download.json')
    part = path.with_name(path.name + '.part')
    source = part.with_name(part.name + '.json')
    if path.exists():
        meta = json.loads(stamp.read_text()) if stamp.exists() else {}
        if meta.get('source') != remote.identity() or meta.get('sha256') != sha(path) or path.stat().st_size != remote.total:
            raise ValueError(f'Existing cache is unverified or source changed: {path}; use a new cache/root')
        return path
    offset = 0
    if part.exists():
        info = json.loads(source.read_text()) if source.exists() else {}
        if info != remote.identity() or not remote.validator or part.stat().st_size > remote.total:
            # Preserve the old partial; start a new one without mixing versions.
            stale = part.with_name(part.name + '.stale-' + uuid.uuid4().hex)
            part.rename(stale)
            if source.exists(): source.rename(stale.with_name(stale.name + '.json'))
        else:
            offset = part.stat().st_size
    if cache_limit is not None:
        additional=remote.total-offset
        if tree_bytes(path.parent)+additional+1024*1024>cache_limit:
            raise ValueError('Archive cache hard budget exceeded before transfer')
    atomic_json(source, remote.identity())
    if offset == remote.total:
        # This can be an interrupted last step or truncated EOF error: revalidate tail.
        n = min(512, remote.total)
        with part.open('rb') as f:
            f.seek(-n, 2); tail = f.read()
        if tail != remote.range(remote.total-n, remote.total-1):
            raise ValueError('Partial archive tail differs from source')
    else:
        headers = {'Accept-Encoding':'identity'}
        if offset:
            headers.update(Range=f'bytes={offset}-{remote.total-1}', **{'If-Range':remote.validator})
        with remote.retry.run('archive connection',lambda:remote.opener(urllib.request.Request(url, headers=headers), timeout=60)) as r:
            if offset:
                if r.status != 206 or r.headers.get('Content-Range') != f'bytes {offset}-{remote.total-1}/{remote.total}':
                    raise ValueError('Unsafe archive resume response; partial retained')
            elif r.status != 200:
                raise ValueError('Expected HTTP 200 for new archive')
            remote.check_identity(r)
            remain = remote.total - offset
            if r.headers.get('Content-Length') not in (None, str(remain)):
                raise ValueError('Archive Content-Length mismatch')
            with part.open('ab' if offset else 'wb') as f:
                while remain:
                    block = r.read(min(CHUNK, remain))
                    if not block: raise ValueError('Interrupted/truncated archive; rerun to resume')
                    if len(block) > remain: raise ValueError('Oversized archive response')
                    f.write(block); remain -= len(block)
                f.flush(); os.fsync(f.fileno())
                if r.read(1): raise ValueError('Oversized archive response')
    # Stamp first: a crash before rename leaves a resumable part, never trusted final data.
    atomic_json(stamp, dict(source=remote.identity(), sha256=sha(part)))
    part.rename(path)
    return path


def extract_archive(archive, stage, select=None, storage_budget=None):
    """Reject links/path traversal/duplicate entries; drain gzip to verify trailer CRC."""
    stage = Path(stage)
    with tarfile.open(archive, 'r|gz') as tf:
        for member in tf:
            rel = checked_path(member.name)
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsupported archive entry: {member.name}')
            if select is not None and not select(rel):
                continue
            target = stage / str(rel)
            if member.isdir(): target.mkdir(parents=True, exist_ok=True); continue
            if member.size > 32*1024*1024: raise ValueError('Unexpected oversized image/metadata entry')
            if storage_budget is not None:storage_budget.claim(member.size)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst, CHUNK)
    # Tar EOF may precede gzip CRC; explicitly validate the whole compressed stream.
    import gzip
    with gzip.open(archive, 'rb') as stream:
        for _ in iter(lambda:stream.read(CHUNK), b''): pass


def rows(cub, expected):
    images = [line.split() for line in (cub/'images.txt').read_text().splitlines() if line.strip()]
    split = [line.split() for line in (cub/'train_test_split.txt').read_text().splitlines() if line.strip()]
    if len(images) != expected or any(len(row)!=2 for row in images+split):
        raise ValueError('Invalid CUB metadata count/columns')
    if len({r[0] for r in images})!=expected or len({r[1] for r in images})!=expected:
        raise ValueError('Duplicate CUB identity')
    if len(split)!=expected or len({r[0] for r in split})!=expected or {r[0] for r in split}!={r[0] for r in images} or any(r[1] not in ('0','1') for r in split):
        raise ValueError('Invalid CUB split identities/flags')
    for identity, rel in images:
        if not identity.isdigit() or int(identity)<=0 or len(checked_path(rel).parts)!=2 or '.' not in checked_path(rel).parts[0] or not checked_path(rel).parts[0].split('.',1)[1] or not rel.endswith('.jpg'):
            raise ValueError('CUB species/filename convention differs from loader')
    return images


def validate_cub(cub, expected=11788, masks=False, rgb_root=None):
    cub = Path(cub)
    reference = Path(rgb_root) if rgb_root is not None else cub
    records = rows(reference, expected)
    actual=list(cub.rglob('*.png')) if masks else list((cub/'images').rglob('*.jpg'))
    if len(actual)!=expected:raise ValueError('Unexpected RGB/mask file count')
    files = {}
    if not masks:
        for name in ('images.txt','train_test_split.txt'):
            files[name] = sha(cub/name)
    for _, rel in records:
        path = cub/Path(rel).with_suffix('.png') if masks else cub/'images'/rel
        with Image.open(path) as image:
            image.load()
            if masks:
                with Image.open(reference/'images'/rel) as rgb:
                    if image.size != rgb.size: raise ValueError(f'Mask/image shape mismatch: {rel}')
        files[str(path.relative_to(cub))] = sha(path)
    return files


def validate_category(folder, expected=5000):
    files = sorted(Path(folder).glob('*.jpg'))
    if len(files)!=expected: raise ValueError(f'Expected {expected} JPEGs in {folder}, got {len(files)}')
    hashes = {}
    for path in files:
        with Image.open(path) as image: image.load()
        hashes[path.name] = sha(path)
    return hashes


def reuse_or_publish(final, populate, validate):
    """Incomplete old final directories are never merged/replaced automatically."""
    final = Path(final)
    if final.exists() or final.is_symlink():
        hashes = validate(final)
        marker = final/'.complete'
        if marker.exists():
            try: recorded = json.loads(marker.read_text())
            except (ValueError, OSError): recorded = None
            if isinstance(recorded, dict) and recorded.get('format')=='waterbirds-download-v1' and recorded.get('files')!=hashes:
                raise ValueError(f'Completed component has changed: {final}')
        print(f'[reuse, verified] {final}')
        return
    if any(parent.is_symlink() for parent in final.parents):
        raise ValueError(f'Refusing to publish through a symlink ancestor: {final}; assemble missing components in a private tree')
    final.parent.mkdir(parents=True, exist_ok=True)
    # Stage name is unique. Interrupted stages are left for inspection, not reused blindly.
    stage = final.parent / ('.'+final.name+'.stage-'+uuid.uuid4().hex)
    stage.mkdir()
    try:
        populate(stage)
        hashes = validate(stage)
        atomic_json(stage/'.complete', dict(format='waterbirds-download-v1',files=hashes))
        if final.exists() or final.is_symlink(): raise FileExistsError(final)
        stage.rename(final)
    except BaseException:
        # Only this invocation's freshly created stage; never preexisting/final data.
        if stage.exists():shutil.rmtree(stage)
        raise
    print(f'[published, verified] {final}')


class RangeStream(io.RawIOBase):
    def __init__(self, remote, start): self.remote=remote; self.pos=start; self.buffer=b''
    def readable(self): return True
    def read(self, n=-1):
        if n<0: raise ValueError('Unbounded range stream read refused')
        n=min(n, CHUNK)
        if n==0:return b''
        while len(self.buffer)<n and self.pos<self.remote.total:
            count=min(CHUNK,self.remote.total-self.pos)
            self.buffer+=self.remote.range(self.pos,self.pos+count-1)
            self.pos+=count
        payload,self.buffer=self.buffer[:n],self.buffer[n:]
        return payload


def header_at(remote, offset):
    offset -= offset % 512
    span = 64*1024
    while span <= 8*1024*1024:
        body = remote.range(offset, min(offset+span,remote.total)-1)
        for i in range(0,len(body)-511,512):
            try: info=tarfile.TarInfo.frombuf(body[i:i+512], 'utf-8', 'strict')
            except (tarfile.HeaderError, UnicodeError): continue
            if info.name.startswith('data_256/'):
                checked_path(info.name)
                return offset+i, info.name
        if offset+len(body)>=remote.total: break
        span*=4
    raise ValueError('Cannot find Places tar header within bounded probe')


def category_key(name): return tuple(PurePosixPath(name).parts[1:-1])


def fetch_category(url, category, stage, opener=None, expected=5000, budget=1024*1024*1024, storage_budget=None, range_cache=None):
    remote=Remote(url, opener, budget, range_cache=range_cache)
    target=tuple(category.split('/'));lo=0;hi=remote.total
    while hi-lo > 1024*1024:
        mid=(lo+hi)//2
        pos,name=header_at(remote,mid)
        if category_key(name)<target: lo=pos
        else: hi=mid
    start,_=header_at(remote,lo)
    count=0
    with tarfile.open(fileobj=RangeStream(remote,start), mode='r|') as archive:
        for member in archive:
            checked_path(member.name)
            key=category_key(member.name)
            if not member.isfile():
                if member.issym() or member.islnk(): raise ValueError('Tar link refused')
                continue
            if key<target:continue
            if key>target:break
            if not member.name.endswith('.jpg') or member.size>32*1024*1024:
                raise ValueError('Unexpected category entry')
            if count>=expected:raise ValueError('Too many category images')
            if storage_budget is not None:storage_budget.claim(member.size)
            path=Path(stage)/PurePosixPath(member.name).name
            with archive.extractfile(member) as src,path.open('xb') as dst:
                shutil.copyfileobj(src,dst,CHUNK)
            count+=1
    if count!=expected:raise ValueError(f'Incomplete category: {count}/{expected}')


@contextmanager
def lock_root(root):
    import fcntl
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.waterbirds-download.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another preparation process owns this root')
        yield


def prepare(root, opener=None, bird_count=11788, scene_count=5000, category_budget=1024*1024*1024, policy=None):
    if not 0<category_budget<=1024**3:raise ValueError('Places category budget must be at most 1 GiB')
    root=Path(root).absolute()
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
        raise ValueError('Writable root must not have symlink ancestors; use --check-only for shared roots')
    policy=policy or StoragePolicy(root,Path(__file__).resolve().parents[2]/'.cache/waterbirds-archives'/socket.gethostname().split('.')[0])
    plan=policy.preflight() # No network or filesystem writes before this check.
    print('[storage plan] '+json.dumps(plan))
    with lock_root(root), lock_root(policy.cache):
        policy.preflight()
        if not (root/'.waterbirds-node.json').exists():
            atomic_json(root/'.waterbirds-node.json',dict(hostname=socket.gethostname(),root=str(root)))
        budget=DataBudget(root)
        cache=policy.cache
        cub=root/'CUB_200_2011'
        def populate_cub(stage):
            archive=fetch_archive(CUB_URL,cache/'CUB_200_2011.tgz',opener,cache_limit=CACHE_CAP)
            extract_archive(archive,stage,select=lambda rel: len(rel.parts)>=2 and rel.parts[0]=='CUB_200_2011' and (rel.parts[1]=='images' or str(rel) in ('CUB_200_2011/images.txt','CUB_200_2011/train_test_split.txt')),storage_budget=budget)
            source=stage/'CUB_200_2011'
            if not source.is_dir():raise ValueError('Unexpected CUB archive root')
            for path in source.iterdir():path.rename(stage/path.name)
            source.rmdir()
        reuse_or_publish(cub,populate_cub,lambda p:validate_cub(p,bird_count))
        cleanup_archive(cache/'CUB_200_2011.tgz',CUB_URL)
        def populate_masks(stage):
            archive=fetch_archive(SEG_URL,cache/'segmentations.tgz',opener,cache_limit=CACHE_CAP)
            extract_archive(archive,stage,select=lambda rel: rel.parts[0]=='segmentations',storage_budget=budget)
            source=stage/'segmentations'
            if not source.is_dir():raise ValueError('Unexpected segmentation archive root')
            for path in source.iterdir():path.rename(stage/path.name)
            source.rmdir()
        reuse_or_publish(cub/'segmentations',populate_masks,
                         lambda p:validate_cub(p,bird_count,masks=True,rgb_root=cub))
        cleanup_archive(cache/'segmentations.tgz',SEG_URL)
        for category in CATEGORIES:
            final=root/'places365/data_256'/category
            ranges=RangeCache(cache/'.places-ranges'/category.replace('/','_'),cache)
            reuse_or_publish(final,lambda stage,c=category:fetch_category(PLACES_URL,c,stage,opener,scene_count,category_budget,storage_budget=budget,range_cache=ranges),
                             lambda p:validate_category(p,scene_count))
            ranges.clear()


def cleanup_archive(path, url):
    """Delete only verified, managed temporary archives after component publication.

    Incomplete .part files remain on network storage for a later explicit retry.
    """
    path=Path(path);stamp=path.with_name(path.name+'.download.json')
    if not path.exists():return
    if path.is_symlink() or not stamp.is_file():raise ValueError('Unmanaged archive cache; not deleting')
    meta=json.loads(stamp.read_text())
    if meta.get('source',{}).get('url')!=url or meta.get('sha256')!=sha(path):
        raise ValueError('Unverified archive cache; not deleting')
    path.unlink();stamp.unlink()
    source=path.with_name(path.name+'.part.json')
    if source.exists():source.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/local_datasets')/os.environ.get('USER','user')/'waterbirds_v1')
    parser.add_argument('--expected-node',default='moana-u2')
    parser.add_argument('--transport',choices=('curl','urllib'),default='curl')
    parser.add_argument('--archive-cache',type=Path,default=Path(__file__).resolve().parents[2]/'.cache/waterbirds-archives'/socket.gethostname().split('.')[0])
    parser.add_argument('--plan-only',action='store_true',help='Check local SSD and network-cache budgets; no writes/network')
    parser.add_argument('--check-only',action='store_true',help='Read-only full component validation; no network or writes')
    parser.add_argument('--category-budget-mib',type=int,default=1024)
    args=parser.parse_args()
    if socket.gethostname().split('.')[0]!=args.expected_node:
        parser.error(f'Run on {args.expected_node}; local_datasets is node-specific')
    if not 0<args.category_budget_mib<=1024:parser.error('Category budget must be 1..1024 MiB')
    if args.check_only:
        from scripts.waterbirds_vgg50.storage import local_root
        local_root(args.root)
        validate_cub(args.root/'CUB_200_2011')
        validate_cub(args.root/'CUB_200_2011/segmentations',masks=True,rgb_root=args.root/'CUB_200_2011')
        for category in CATEGORIES:validate_category(args.root/'places365/data_256'/category)
        print('Validated RGB/masks/four scene categories; correlation generation is a separate prepare step.')
        return
    policy=StoragePolicy(args.root,args.archive_cache)
    if args.plan_only:
        print(json.dumps(policy.preflight(),indent=2));return
    opener=CurlOpener() if args.transport=='curl' else None
    prepare(args.root,opener=opener,category_budget=args.category_budget_mib*1024*1024,policy=policy)

if __name__=='__main__':main()
