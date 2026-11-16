"""Node-local SSD policy; archives live in a separate network-filesystem cache."""
import os
from pathlib import Path
import shutil
import socket

GiB=1024**3
FINAL_CAP=16*GiB
CACHE_CAP=4*GiB
SSD_RESERVE=2*GiB
CACHE_RESERVE=GiB
POLICY_LIMIT=50_000_000_000


def tree_bytes(root):
    root=Path(root)
    if not root.exists():return 0
    total=0
    for base,dirs,files in os.walk(root,followlinks=False):
        for name in dirs+files:
            p=Path(base)/name
            if p.is_symlink():raise ValueError(f'Storage tree must not contain symlinks: {p}')
        total+=sum((Path(base)/name).stat().st_size for name in files)
    return total


def existing_parent(path):
    path=Path(path)
    while not path.exists():path=path.parent
    return path


def filesystem(path):
    path=Path(path).resolve();matches=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left,right=line.split(' - ',1);mount=Path(left.split()[4].replace('\\040',' '))
        if path==mount or mount in path.parents:matches.append((len(str(mount)),right.split()[0]))
    return max(matches)[1]


def local_root(root,base=Path('/local_datasets')):
    root=Path(root).absolute();base=Path(base).resolve()
    if root.resolve()==base or not root.resolve().is_relative_to(base):
        raise ValueError('Final dataset must be a private subtree of /local_datasets')
    if filesystem(existing_parent(root)) not in ('ext4','xfs','btrfs','zfs'):
        raise ValueError('Final root must use node-local disk, not NFS/overlay/tmpfs')
    return root


class DataBudget:
    def __init__(self,root,limit=FINAL_CAP):self.root=Path(root);self.limit=limit;self.used=tree_bytes(root)
    def claim(self,n):
        if self.used+n>self.limit-16*1024**2:raise ValueError('Final dataset hard byte budget exceeded before write')
        if shutil.disk_usage(existing_parent(self.root)).free<n+SSD_RESERVE:raise ValueError('SSD free space dropped below reserve')
        self.used+=n


class StoragePolicy:
    def __init__(self,root,cache,base=Path('/local_datasets')):
        self.root=local_root(root,base);self.cache=Path(cache).absolute()
        if self.cache.is_symlink() or any(p.is_symlink() for p in self.cache.parents):
            raise ValueError('Archive cache must not use symlink ancestors')
        if self.cache.resolve().is_relative_to(self.root.resolve()) or self.root.resolve().is_relative_to(self.cache.resolve()):
            raise ValueError('Archive cache and dataset must be disjoint')
        if filesystem(existing_parent(self.cache)) not in ('nfs','nfs4','cifs','ceph','fuse.ceph'):
            raise ValueError('Resumable archive cache must be network storage, not computing-node SSD')
    def preflight(self):
        import json
        if FINAL_CAP+SSD_RESERVE>=POLICY_LIMIT:raise ValueError('Planned SSD peak is not below 50GB')
        for p in self.root.rglob('*') if self.root.exists() else []:
            if p.is_file() and any(ext in p.name.lower().split('.')[1:] for ext in ('tar','tgz','zip')):
                raise ValueError('Archive found in final SSD root; preserve it and use a clean root')
        used=tree_bytes(self.root);cached=tree_bytes(self.cache)
        if used>FINAL_CAP or cached>CACHE_CAP:raise ValueError('Existing data/cache exceeds hard budget; no download started')
        if shutil.disk_usage(existing_parent(self.root)).free<FINAL_CAP-used+SSD_RESERVE:
            raise ValueError('Insufficient SSD space for bounded final data plus reserve')
        if shutil.disk_usage(existing_parent(self.cache)).free<CACHE_CAP-cached+CACHE_RESERVE:
            raise ValueError('Insufficient network cache space')
        stamp=self.root/'.waterbirds-node.json'
        if stamp.exists() and json.loads(stamp.read_text())['hostname']!=socket.gethostname():
            raise ValueError('Dataset belongs to another node')
        return dict(final_cap_bytes=FINAL_CAP,archive_cache_cap_bytes=CACHE_CAP,
                    ssd_peak_with_reserve_bytes=FINAL_CAP+SSD_RESERVE,hostname=socket.gethostname())
