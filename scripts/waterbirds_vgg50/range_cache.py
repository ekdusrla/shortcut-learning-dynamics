"""Validated successful Range responses in the bounded NFS cache, never on SSD."""
import hashlib
import json
import re
from pathlib import Path
import shutil
import uuid
from scripts.waterbirds_vgg50.storage import CACHE_CAP, tree_bytes

FORMAT='waterbirds-range-cache-v1'


class RangeCache:
    def __init__(self,path,cache_root):
        self.path=Path(path);self.root=Path(cache_root)
        if self.path.resolve()==self.root.resolve() or not self.path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('Category Range cache must be a subtree of archive cache')
    def _identity(self,remote,start,end):
        return dict(source=remote.identity(),start=start,end=end)
    def _key(self,identity):return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    def _owner(self):
        marker=self.path/'.owner.json'
        if self.path.exists():
            if not marker.is_file() or json.loads(marker.read_text()).get('format')!=FORMAT:
                raise ValueError('Unmanaged Range cache; preserved without modification')
        else:
            self.path.mkdir(parents=True)
            marker.write_text(json.dumps(dict(format=FORMAT)))
    def _source(self,remote,write=False):
        stamp=self.path/'.source.json'
        if stamp.exists():
            if json.loads(stamp.read_text())!=remote.identity():
                raise ValueError('Range cache source validator changed; preserved without retry')
        elif write:
            from scripts.waterbirds_vgg50.download import atomic_json
            atomic_json(stamp,remote.identity())

    def get(self,remote,start,end):
        if not self.path.exists():return None
        self._owner();self._source(remote)
        if not remote.validator:return None
        identity=self._identity(remote,start,end);key=self._key(identity)
        stamp=self.path/(key+'.json');blob=self.path/(key+'.chunk')
        if not stamp.exists():return None
        meta=json.loads(stamp.read_text())
        if blob.stat().st_size!=end-start+1:raise ValueError('Range cache length mismatch')
        payload=blob.read_bytes()
        if meta.get('identity')!=identity or len(payload)!=end-start+1 or hashlib.sha256(payload).hexdigest()!=meta.get('sha256'):
            raise ValueError('Range cache integrity mismatch; not redownloading or overwriting it')
        return payload
    def put(self,remote,start,end,payload):
        if not remote.validator:return # Cross-run reuse needs a source validator.
        if tree_bytes(self.root)+len(payload)+1024*1024>CACHE_CAP:raise ValueError('NFS cache hard budget exceeded')
        self._owner();self._source(remote,write=True);identity=self._identity(remote,start,end);key=self._key(identity)
        blob=self.path/(key+'.chunk');stamp=self.path/(key+'.json')
        digest=hashlib.sha256(payload).hexdigest()
        if blob.exists():
            if hashlib.sha256(blob.read_bytes()).hexdigest()!=digest:raise ValueError('Orphan Range cache differs; preserved')
        else:
            temp=self.path/(key+'.tmp-'+uuid.uuid4().hex)
            try:temp.write_bytes(payload);temp.rename(blob)
            finally:
                if temp.exists():temp.unlink()
        from scripts.waterbirds_vgg50.download import atomic_json
        atomic_json(stamp,dict(identity=identity,sha256=digest))
    def clear(self):
        if not self.path.exists():return
        self._owner()
        for path in self.path.iterdir():
            allowed=path.name in ('.owner.json','.source.json') or re.fullmatch(r'[0-9a-f]{64}\.(chunk|json|tmp-[0-9a-f]{32}|json\.tmp-[0-9a-f]{32})',path.name) or re.fullmatch(r'\.source\.json\.tmp-[0-9a-f]{32}',path.name)
            if path.is_symlink() or not path.is_file() or not allowed:
                raise ValueError('Unexpected Range cache files; preserving cache without deletion')
        # Only our explicitly owned temporary category cache, after final validation.
        shutil.rmtree(self.path)
