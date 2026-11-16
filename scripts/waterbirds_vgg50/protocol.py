"""Isolated Waterbirds A protocol. Never changes shared dataset/model code."""
import hashlib
import json
import socket
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import data

CORRS = (.95, .80, .65)
EVAL_CORR = .50
PROTOCOL_VERSION = 3

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


from scripts.waterbirds_vgg50.content_split import (inventory, legacy_scenes, leakage_report,
    group_cub, group_scenes, assign_places, counts)


def validate_frame(frame,assets,corr):
    if frame.img_id.duplicated().any() or frame.img_filename.duplicated().any():
        raise ValueError('Duplicate CUB identity in manifest')
    if not frame.split.isin(('train','val','test')).all() or not frame.y.isin((0,1)).all() or not frame.place.isin((0,1)).all():
        raise ValueError('Invalid split or binary labels')
    leakage=leakage_report(frame,assets)
    if leakage['cross_split_hashes']:
        first=leakage['collisions'][0]
        raise ValueError(f'Cross-split image content leakage: {first}')
    for row in frame.itertuples():
        if assets['places365/'+data.PLACES_PREFIX+'/'+row.land_file]==assets['places365/'+data.PLACES_PREFIX+'/'+row.water_file]:
            raise ValueError('Counterfactual backgrounds have identical content')
    reports={}
    for split in ('train','val','test'):
        f=frame[frame.split==split];target=corr if split=='train' else EVAL_CORR
        groups={f'g{y}{s}':int(((f.y==y)&(f.place==s)).sum()) for y in (0,1) for s in (0,1)}
        for y in (0,1):
            sub=f[f.y==y]
            if not len(sub) or abs(int((sub.place==y).sum())-target*len(sub))>.500001:
                raise ValueError('Conditional correlation failed')
        if min(groups.values())<1:raise ValueError('Empty evaluation/training group')
        reports[split]=dict(n=len(f),groups=groups,target_correlation=target,
            correlation=float((f.y==f.place).mean()),
            by_label={str(y):dict(n=int((f.y==y).sum()),aligned_n=int(((f.y==y)&(f.place==y)).sum()),
                correlation=float((f.loc[f.y==y,'place']==y).mean())) for y in (0,1)})
    return reports


def prepare(root,output,seed=42,expected_birds=None,expected_scenes_per_category=None):
    root,output=Path(root).resolve(),Path(output)
    if output.exists():raise FileExistsError(output)
    if output.resolve().is_relative_to(root):raise ValueError('Manifest output must be outside the source dataset')
    base,assets,cub_groups,scene_groups=inventory(root,seed)
    source_counts=dict(cub_metadata_rows=len(base),cub_rgb_files=sum(1 for _ in (root/'CUB_200_2011/images').rglob('*.jpg')),
        cub_mask_files=sum(1 for _ in (root/'CUB_200_2011/segmentations').rglob('*.png')),
        scene_files=sum(len(rows) for rows in scene_groups.values()),cub_unique_content=len(cub_groups),scene_unique_content=len(scene_groups))
    if expected_birds is not None and any(source_counts[k]!=expected_birds for k in ('cub_metadata_rows','cub_rgb_files','cub_mask_files')):
        raise ValueError('Unexpected CUB image/mask sample count')
    if expected_scenes_per_category is not None:
        for categories in data.TARGET_PLACES:
            for category in categories:
                files=data.place_files(root/'places365',[category])
                if len(files)!=expected_scenes_per_category:raise ValueError(f'Unexpected scene count: {category}')
    legacy=legacy_scenes(root,base,seed)
    original_counts=counts(base)
    grouped=group_cub(root,base,cub_groups,seed)
    grouped,scene_owner,pool_counts=group_scenes(grouped,cub_groups,scene_groups,seed)
    frames=[assign_places(grouped,c,seed,EVAL_CORR) for c in CORRS]
    reports=[validate_frame(frame,assets,c) for frame,c in zip(frames,CORRS)]
    for frame in frames[1:]:
        if not frame.drop(columns='place').equals(frames[0].drop(columns='place')):raise ValueError('Paired comparison lost')
        if not frame[frame.split!='train'].equals(frames[0][frames[0].split!='train']):raise ValueError('Evaluation distribution changed across correlations')
    final_counts=counts(grouped)
    cub_original=dict(zip(base.img_id,base.split));final=dict(zip(grouped.img_id,grouped.split))
    audit=dict(hash_definition='SHA256 of original encoded bytes; masks excluded from leakage identity',
        legacy=leakage_report(legacy,assets),final=leakage_report(frames[0],assets),
        legacy_usage_by_correlation={str(c):{mode:leakage_report(assign_places(legacy,c,seed,EVAL_CORR),assets,mode) for mode in ('factual','runtime')} for c in CORRS},
        source_counts=source_counts,original_counts=original_counts,final_counts=final_counts,total_ids_preserved=len(base)==len(grouped),
        moved_bird_ids=[int(i) for i in base.img_id if cub_original[i]!=final[i]],
        cub_duplicates={h:rows for h,rows in cub_groups.items() if len(rows)>1},
        scene_duplicates={h:rows for h,rows in scene_groups.items() if len(rows)>1},
        scene_group_split=scene_owner,scene_unique_pool_counts=pool_counts,
        scene_groups_pinned_to_cub=sorted(set(cub_groups)&set(scene_groups)),
        cub_policy='Preserve clean split. Duplicate groups with official test members stay wholly in test; exact per-class validation quota, minimal membership changes; conflicting labels/infeasible quotas fail.',
        scene_policy='All Places content hashes split 60/20/20 as groups; CUB-identical groups pinned to CUB split; one representative per kind; within-split cycling allowed.')
    # No output is published until all conditions/ownership/count checks succeed.
    output.mkdir(parents=True)
    files={}
    for c,frame,report in zip(CORRS,frames,reports):
        name=f'A_corr{round(c*100):03d}.csv';frame.to_csv(output/name,index=False)
        files[name]=dict(sha256=digest(output/name),splits=report)
    (output/'split_audit.json').write_text(json.dumps(audit,indent=2))
    payload=dict(version=PROTOCOL_VERSION,condition='A',condition_name='Background',
        val_corr=EVAL_CORR,test_corr=EVAL_CORR,seed=seed,root=str(root),files=files,assets=assets,
        scene_policy=audit['scene_policy'],split_audit_sha256=digest(output/'split_audit.json'),
        original_counts=original_counts,final_counts=final_counts,source_counts=source_counts)
    node_stamp=root/'.waterbirds-node.json'
    if node_stamp.exists():payload['data_node']=json.loads(node_stamp.read_text())['hostname']
    (output/'protocol.json').write_text(json.dumps(payload,indent=2))
    return payload


def load(manifest, corr, verify=True, require_local=False):
    manifest=Path(manifest).resolve()
    meta=json.loads(manifest.read_text())
    if meta.get('version') not in (1, 2, PROTOCOL_VERSION) or any(meta.get(k, EVAL_CORR)!=EVAL_CORR for k in ('val_corr','test_corr')):
        raise ValueError('Expected val/test correlation=0.50; prepare a new manifest without overwriting old results')
    name=f'A_corr{round(corr*100):03d}.csv'
    if corr not in CORRS or digest(manifest.parent/name)!=meta['files'][name]['sha256']:
        raise ValueError('Unsupported correlation or modified manifest')
    root=Path(meta['root'])
    if require_local:
        from scripts.waterbirds_vgg50.storage import local_root
        local_root(root)
    if meta.get('data_node') not in (None,socket.gethostname()):
        raise ValueError('Manifest belongs to another node-local dataset')
    if verify:
        for rel, expected in meta['assets'].items():
            if digest(root/rel)!=expected:
                raise ValueError(f'Source changed: {rel}')
    frame=pd.read_csv(manifest.parent/name)
    validate_frame(frame,meta['assets'],corr)
    return {s:data.Waterbirds(root, frame[frame.split==s],224,train=s=='train',
                              augment=False,counterfactual=s!='train') for s in ('train','val','test')}
