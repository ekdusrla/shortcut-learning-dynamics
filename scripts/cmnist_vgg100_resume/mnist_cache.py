"""Prepare one immutable official MNIST revision, then verify existing data.py offline.
CPU only. Never imports or runs the trainer, and never writes to original runs.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[2]
PIN='77f3279092a1c1579b2250db8eafed0ad422088c'
REPO='ylecun/mnist'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def env(cache,offline):
 os.environ.update(HF_HOME=str(cache),HF_DATASETS_CACHE=str(cache/'datasets'),HF_HUB_CACHE=str(cache/'hub'),
                   HF_MODULES_CACHE=str(cache/'modules'),HF_HUB_OFFLINE=str(int(offline)),
                   HF_DATASETS_OFFLINE=str(int(offline)),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
                   HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_XET='1')
def signatures(raw):
 import numpy as np
 result={}
 for split,n in [('train',60000),('test',10000)]:
  ds=raw[split];assert len(ds)==n and ds.column_names==['image','label']
  labels=np.asarray(ds['label'],dtype='<i8');assert set(labels)==set(range(10))
  pixels=hashlib.sha256()
  for batch in ds.iter(batch_size=512):
   for image in batch['image']:
    a=np.asarray(image.convert('L'),dtype=np.uint8);assert a.shape==(28,28)
    pixels.update(a.tobytes(order='C'))
  result[split]={'samples':n,'fingerprint':ds._fingerprint,'label_sha256':hashlib.sha256(labels.tobytes()).hexdigest(),
                 'ordered_grayscale_pixels_sha256':pixels.hexdigest(),'class_counts':np.bincount(labels,minlength=10).tolist(),
                 'features':ds.features.to_dict()}
  print(f'{split}: N={n}, fingerprint={ds._fingerprint}',flush=True)
 return result

def verify(cache):
 env(cache,True)
 # Deny actual sockets as well as setting HF flags; offline success cannot depend on Hub access.
 network_calls=[]
 def blocked(*args,**kwargs):
  network_calls.append(True)
  raise RuntimeError('Network prohibited during offline validation')
 socket.socket.connect=blocked
 socket.socket.connect_ex=blocked
 from datasets import load_dataset
 import numpy as np
 import torch
 sys.path.insert(0,str(ROOT))
 import data
 manifest=json.loads((cache/'mnist_manifest.json').read_text())
 for f in manifest['cache_files']:
  assert sha(cache/f['path'])==f['sha256'],f['path']
 raw=load_dataset(REPO)  # exact unpinned call used by current, unchanged data.py
 sig=signatures(raw)
 for split in sig:
  for key in ['samples','label_sha256','ordered_grayscale_pixels_sha256','features']:
   assert sig[split][key]==manifest['raw_splits'][split][key],(split,key)
 results=[]
 for tag,mode in [('bg','background'),('fg','foreground')]:
  parent=ROOT/f'runs/vgg11_{tag}_corr095_seed42_e50_img224_all'
  protected=[parent/'config.json',parent/'history.csv',parent/'test_metrics.json',parent/'checkpoints/epoch_050.pt',parent/'features/epoch_050.npz']
  before={str(p.relative_to(ROOT)):sha(p) for p in protected}
  saved=torch.load(parent/'checkpoints/epoch_050.pt',map_location='cpu',weights_only=True)
  cfg=saved['config'];assert cfg==json.loads((parent/'config.json').read_text());assert saved['epoch']==50
  del saved
  required={'dataset':'colored_mnist','seed':42,'mnist_val_size':5000,'image_size':224,'train_corr':.95,
            'val_corr':.1,'test_corr':.1,'color_pairing':'all','color_mode':mode,'augment':False}
  assert all(cfg[k]==v for k,v in required.items())
  # Calls unchanged production construction/loading path on the real prepared data.
  ds=data.build_datasets(cfg)
  perm=np.random.default_rng(42).permutation(60000)
  expected_ids={'train':perm[5000:],'val':perm[:5000],'test':np.arange(10000)}
  split_info={}
  for split,n,seed in [('train',55000,142),('val',5000,242),('test',10000,342)]:
   d=ds[split];assert len(d)==n and np.array_equal(d.indices,expected_ids[split])
   base_labels=np.asarray(raw['test' if split=='test' else 'train']['label'],dtype=np.int64)
   labels=base_labels[expected_ids[split]];assert np.array_equal(d.labels,labels)
   rng=np.random.default_rng(seed);aligned=rng.random(n)<cfg[f'{split}_corr']
   alt=(labels+rng.integers(1,10,n))%10
   assert np.array_equal(d.spurious,np.where(aligned,labels,alt))
   assert np.array_equal(d.cf_spurious,np.where(aligned,alt,labels))
   counts=np.bincount(d.labels*10+d.spurious,minlength=100).reshape(10,10)
   split_info[split]={'samples':n,'color_seed':seed,'aligned_n':int((d.labels==d.spurious).sum()),
       'realized_aligned_fraction':float((d.labels==d.spurious).mean()),
       'indices_sha256':hashlib.sha256(d.indices.astype('<i8').tobytes()).hexdigest(),
       'labels_sha256':hashlib.sha256(d.labels.astype('<i8').tobytes()).hexdigest(),
       'spurious_sha256':hashlib.sha256(d.spurious.astype('<i8').tobytes()).hexdigest(),
       'counts':counts.tolist()}
   if split=='val':
    with (parent/'history.csv').open() as f:first=next(csv.DictReader(f))
    assert all(counts[y,s]==int(first[f'val_g{y}{s}_n']) for y in range(10) for s in range(10))
   if split=='test':
    old=json.loads((parent/'test_metrics.json').read_text())['metrics']
    assert all(counts[y,s]==old[f'g{y}{s}_n'] for y in range(10) for s in range(10))
   # Actual 224px tensors, manual painting formula, paired/all-color ordering.
   from PIL import Image
   for i in [0,n-1]:
    item=d[i];assert item['pixel_values'].shape==(3,224,224)
    base=raw['test' if split=='test' else 'train'][int(d.indices[i])]['image']
    a=np.asarray(base.convert('L'),dtype=np.float32)[...,None]/255
    c=data.MNIST_COLORS[int(d.spurious[i])]
    rgb=a*c+(1-a)*data.BLACK if mode=='foreground' else a*data.WHITE+(1-a)*data.BACKGROUND_SCALE*c
    expected=data.image_transform(224)(Image.fromarray(rgb.round().clip(0,255).astype(np.uint8),'RGB'))
    assert torch.equal(item['pixel_values'],expected)
    assert torch.isfinite(item['pixel_values']).all()
    assert ('cf_pixel_values' in item)==(split!='train')
    if split!='train':
     assert torch.equal(item['cf_pixel_values'],d.paint(base,d.cf_spurious[i],i))
     colors=[c for c in range(10) if c!=int(d.spurious[i])]
     assert item['cf_all_spurious'].tolist()==colors
     assert all(torch.equal(item['cf_all_pixel_values'][k],d.paint(base,c,i)) for k,c in enumerate(colors))
  with np.load(parent/'features/epoch_050.npz',allow_pickle=False) as z:
   lookup={int(idx):j for j,idx in enumerate(ds['val'].indices)}
   positions=np.array([lookup[int(idx)] for idx in z['indices']])
   assert np.array_equal(z['labels'],ds['val'].labels[positions])
   assert np.array_equal(z['spurious'],ds['val'].spurious[positions])
   feature_n=len(positions)
  assert before=={str(p.relative_to(ROOT)):sha(p) for p in protected}
  results.append({'mode':mode,'config':cfg,'splits':split_info,'original_validation_feature_metadata_matches':feature_n,
                  'original_validation_and_test_group_counts_match':True,'protected_sha256':before})
 assert all(results[0]['splits'][k]==results[1]['splits'][k] for k in expected_ids)
 assert not network_calls
 assert not torch.cuda.is_initialized()
 blob=subprocess.check_output(['git','rev-parse','HEAD:data.py'],cwd=ROOT,text=True).strip()
 current_blob=subprocess.check_output(['git','hash-object','data.py'],cwd=ROOT,text=True).strip()
 assert current_blob==blob
 report={'success':True,'validated_utc':datetime.now(timezone.utc).isoformat(),'cache_root':str(cache),
         'repo':REPO,'revision':manifest['revision'],'offline_raw_splits':sig,'network_attempts':len(network_calls),
         'gpu_used':False,'data_py_sha256':sha(ROOT/'data.py'),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'data_py_git_blob':blob,'data_py_matches_head':True,'conditions':results,
         'limitations':['Original raw revision/fingerprint/image hashes were not recorded; historical pixel identity cannot be proven.',
                        'Historical validation/test group counts and 3200 saved validation labels/colors per mode match; no model inference performed.',
                        'GPU-node mount/accessibility has not been tested here.']}
 write(cache/'offline_validation.json',report)
 print('PASS: real offline MNIST + production splits/colors + original metadata; no GPU, no network',flush=True)

def prepare(cache,revision):
 if cache.exists():raise FileExistsError(f'Refusing to overwrite {cache}; use verify for an existing cache')
 cache.parent.mkdir(parents=True,exist_ok=True)
 stage=cache.with_name(cache.name+'.preparing-'+uuid.uuid4().hex[:10]);stage.mkdir()
 env(stage,False)
 from huggingface_hub import HfApi
 from datasets import load_dataset
 info=HfApi(token=False).dataset_info(REPO,revision=revision,timeout=30)
 commit=info.sha;assert len(commit)==40
 print('Preparing',REPO,'revision',commit,'stage',stage,flush=True)
 raw=load_dataset(REPO,revision=commit,token=False)
 sig=signatures(raw)
 files=[]
 for d in raw.values():
  for item in d.cache_files:
   p=Path(item['filename']).resolve();files.append({'path':str(p.relative_to(stage)), 'bytes':p.stat().st_size,'sha256':sha(p)})
 manifest={'repo':REPO,'revision':commit,'requested_revision':revision,'prepared_utc':datetime.now(timezone.utc).isoformat(),
           'historical_revision_known':False,'raw_splits':sig,'cache_files':files,
           'versions':{p:importlib.metadata.version(p) for p in ['datasets','huggingface_hub','numpy','pillow','torch','torchvision']},
           'process':'Official pinned repo -> fresh staging cache -> CPU offline validation -> publish; original runs never modified'}
 write(stage/'mnist_manifest.json',manifest)
 child_env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':''}
 subprocess.run([sys.executable,str(Path(__file__).resolve()),'verify','--cache-root',str(stage)],env=child_env,check=True)
 if cache.exists():raise FileExistsError(f'Destination appeared during preparation: {cache}')
 stage.rename(cache)
 # Published paths differ; revalidate the exact final location as well.
 subprocess.run([sys.executable,str(Path(__file__).resolve()),'verify','--cache-root',str(cache)],env=child_env,check=True)
 print('Published verified cache:',cache,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('action',choices=['prepare','verify'])
 p.add_argument('--cache-root',type=Path,default=ROOT/'.cache/huggingface')
 p.add_argument('--revision',default=PIN,help='Immutable official dataset commit; resolves and records SHA before download')
 args=p.parse_args();cache=args.cache_root.resolve()
 if args.action=='prepare':prepare(cache,args.revision)
 else:verify(cache)
