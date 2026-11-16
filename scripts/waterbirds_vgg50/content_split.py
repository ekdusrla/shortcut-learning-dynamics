"""Read-only source inventory and deterministic content-group split repair."""
from collections import defaultdict
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import data

SPLITS=('train','val','test')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def counts(frame):
    return {split:{'n':int((frame.split==split).sum()),
                   'classes':{str(y):int(((frame.split==split)&(frame.y==y)).sum()) for y in (0,1)}} for split in SPLITS}


def legacy_scenes(root,base,seed):
    frame=base.copy()
    for kind,column in enumerate(('land_file','water_file')):
        files=sorted(set(data.place_files(Path(root)/'places365',data.TARGET_PLACES[kind])))
        order=np.random.default_rng(seed+1000+kind).permutation(len(files))
        pools=np.array_split(np.asarray(files)[order],[int(.6*len(files)),int(.8*len(files))])
        for split,pool in zip(SPLITS,pools):
            if not len(pool):raise ValueError('Empty legacy background pool')
            ids=frame.index[frame.split==split]
            frame.loc[ids,column]=[pool[i%len(pool)] for i in range(len(ids))]
    return frame


def inventory(root,seed=42):
    root=Path(root).resolve()
    base=data.waterbirds_metadata(root,dict(train=.95,val=.5,test=.5),seed)
    if base.img_id.duplicated().any() or base.img_filename.duplicated().any():raise ValueError('Duplicate CUB identity')
    base['original_split']=base['split']
    assets={};sizes={};cub_groups=defaultdict(list);scene_groups=defaultdict(list)
    def inspect(path):
        rel=str(path.relative_to(root))
        if rel not in assets:
            with Image.open(path) as image:image.load();sizes[rel]=image.size
            assets[rel]=sha(path)
        return assets[rel]
    for row in base.itertuples():
        image=root/'CUB_200_2011/images'/row.img_filename
        mask=root/'CUB_200_2011/segmentations'/Path(row.img_filename).with_suffix('.png')
        key=inspect(image);inspect(mask)
        if sizes[str(image.relative_to(root))]!=sizes[str(mask.relative_to(root))]:raise ValueError(f'Mask/image shape mismatch: {image}')
        cub_groups[key].append(dict(img_id=int(row.img_id),path=str(image.relative_to(root)),y=int(row.y),split=row.split))
    for kind,categories in enumerate(data.TARGET_PLACES):
        for rel in sorted(set(data.place_files(root/'places365',categories))):
            path=root/'places365'/data.PLACES_PREFIX/rel
            key=inspect(path)
            scene_groups[key].append(dict(path=str(path.relative_to(root)),scene_file=rel,kind=kind))
    for name in ('images.txt','train_test_split.txt'):assets['CUB_200_2011/'+name]=sha(root/'CUB_200_2011'/name)
    return base,assets,dict(cub_groups),dict(scene_groups)


def ownership(frame,assets,mode='all_candidates'):
    owners=defaultdict(set)
    for row in frame.itertuples():
        pairs=(('CUB','CUB_200_2011/images/'+row.img_filename),
               ('Places','places365/'+data.PLACES_PREFIX+'/'+row.land_file),
               ('Places','places365/'+data.PLACES_PREFIX+'/'+row.water_file))
        if mode=='factual' or (mode=='runtime' and row.split=='train'):
            pairs=(pairs[0],pairs[1+int(row.place)])
        elif mode not in ('all_candidates','runtime'):raise ValueError('Unknown asset usage mode')
        for domain,rel in pairs:owners[assets[rel]].add((domain,rel,row.split))
    return owners


def leakage_report(frame,assets,mode='all_candidates'):
    owners=ownership(frame,assets,mode)
    collisions=[]
    for digest,entries in sorted(owners.items()):
        if len({split for _,_,split in entries})>1:
            collisions.append(dict(sha256=digest,occurrences=[dict(domain=d,path=p,split=s) for d,p,s in sorted(entries)]))
    domains={}
    for domain in ('CUB','Places'):
        groups=[]
        for digest,entries in sorted(owners.items()):
            selected=[entry for entry in entries if entry[0]==domain]
            if len({s for _,_,s in selected})>1:groups.append(digest)
        domains[domain]=dict(cross_split_hashes=len(groups),sha256=groups)
    return dict(usage_mode=mode,cross_split_hashes=len(collisions),by_domain=domains,collisions=collisions)


def _validation_subset(groups,target,seed):
    # Exact per-class quota, minimum number of changes from the old val assignment.
    rng=np.random.default_rng(seed);tie=rng.permutation(len(groups))
    order=sorted(range(len(groups)),key=lambda i:(len(groups[i])-2*sum(r['split']=='val' for r in groups[i]),int(tie[i])))
    inf=10**9;cost=[inf]*(target+1);bits=[0]*(target+1);cost[0]=0
    for i in order:
        weight=len(groups[i]);delta=weight-2*sum(r['split']=='val' for r in groups[i])
        for n in range(target,weight-1,-1):
            if cost[n-weight]!=inf and cost[n-weight]+delta<cost[n]:
                cost[n]=cost[n-weight]+delta;bits[n]=bits[n-weight]|(1<<i)
    if cost[target]==inf:raise ValueError('Content groups cannot preserve exact validation class quota; inspect source audit')
    return {i for i in range(len(groups)) if bits[target]&(1<<i)}


def group_cub(root,base,cub_groups,seed):
    frame=base.copy();official=pd.read_csv(Path(root)/'CUB_200_2011/train_test_split.txt',sep=r'\s+',header=None,names=['img_id','is_train'])
    flags=dict(zip(official.img_id,official.is_train));assignment={};train_groups=[]
    for digest,members in sorted(cub_groups.items()):
        if len({r['y'] for r in members})!=1:raise ValueError(f'Identical CUB content has conflicting labels: {digest}; source audit required')
        if any(flags[r['img_id']]==0 for r in members):
            assignment.update({r['img_id']:'test' for r in members})
        else:train_groups.append(members)
    # Do nothing to a clean CUB split. Places-only duplicates must not change birds.
    original=dict(zip(base.img_id,base.split))
    clean=all(len({r['split'] for r in g})==1 for g in cub_groups.values()) and all(
        assignment[i]==original[i] for i in assignment)
    if clean:return frame
    for y in (0,1):
        groups=[g for g in train_groups if g[0]['y']==y]
        target=int(((base.split=='val')&(base.y==y)).sum())
        chosen=_validation_subset(groups,target,seed+200+y)
        for i,members in enumerate(groups):assignment.update({r['img_id']:'val' if i in chosen else 'train' for r in members})
    frame['split']=frame.img_id.map(assignment)
    if frame.split.isna().any():raise ValueError('Lost CUB group assignment')
    return frame


def group_scenes(base,cub_groups,scene_groups,seed):
    frame=base.copy();cub_owner={}
    by_id=dict(zip(frame.img_id,frame.split))
    for digest,members in cub_groups.items():
        splits={by_id[r['img_id']] for r in members}
        if len(splits)!=1:raise ValueError('CUB content group still crosses splits')
        cub_owner[digest]=next(iter(splits))
    free=sorted(set(scene_groups)-set(cub_owner));order=np.random.default_rng(seed+1000).permutation(len(free))
    pools=np.array_split(np.asarray(free)[order],[int(.6*len(free)),int(.8*len(free))])
    owner={h:s for s,pool in zip(SPLITS,pools) for h in pool}
    owner.update({h:cub_owner[h] for h in scene_groups if h in cub_owner})
    used_pools={}
    scene_hash={r['scene_file']:h for h,rows in scene_groups.items() for r in rows}
    for split in SPLITS:
        representatives={kind:[] for kind in (0,1)}
        for digest,rows in sorted(scene_groups.items()):
            if owner[digest]!=split:continue
            for kind in (0,1):
                paths=sorted(r['scene_file'] for r in rows if r['kind']==kind)
                if paths:representatives[kind].append(paths[0])
        for kind in (0,1):
            if not representatives[kind]:raise ValueError(f'Not enough independent background content groups: {split}, kind={kind}')
        ids=frame.index[frame.split==split]
        land=representatives[0];water=representatives[1]
        for i,index in enumerate(ids):
            a=land[i%len(land)];b=water[i%len(water)]
            if scene_hash[a]==scene_hash[b]:
                alternatives=[w for w in water if scene_hash[w]!=scene_hash[a]]
                if not alternatives:raise ValueError('No distinct-background counterfactual available')
                b=alternatives[0]
            frame.loc[index,'land_file']=a;frame.loc[index,'water_file']=b
        used_pools[split]={str(k):len(v) for k,v in representatives.items()}
    return frame,owner,used_pools


def assign_places(base,corr,seed,eval_corr):
    frame=base.copy();frame['place']=0
    for split in SPLITS:
        rng=np.random.default_rng(seed+data.SPLIT_SEEDS[split]);p=corr if split=='train' else eval_corr
        for y in (0,1):
            ids=np.flatnonzero((frame.split==split)&(frame.y==y));fraction=p if y else 1-p
            chosen=rng.choice(ids,size=int(np.round(fraction*len(ids))),replace=False)
            frame.loc[chosen,'place']=1
    return frame
