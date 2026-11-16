"""Resumable CPU linear readouts of saved snapshots; no model/checkpoint inference."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/cmnist-probe-mpl')
from pathlib import Path
import sys,json,hashlib,time,logging,traceback,csv
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
MODES=['background','foreground'];CORRS=[.95,.80,.65];SEEDS=[1701,1702,1703]
LAYER_EPOCHS=[0,1,2,3,5,10,20,27,28,29,46,47,48,49,50]
SOFT_BG=[0,1,28,49,50]
def write(path,obj):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n');os.replace(tmp,path)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def run_path(mode,corr):return ROOT/f'runs/vgg11_{"bg" if mode=="background" else "fg"}_corr{round(corr*100):03d}_seed42_e50_img224_all'
def update(**kw):
 p=OUT/'progress.json';old=json.loads(p.read_text()) if p.exists() else {};old.update(kw);
 if kw.get('status') not in ['failed','interrupted']:
  old.pop('error',None);old.pop('traceback',None)
 old['updated_unix']=time.time();write(p,old)
def plan():
 if (OUT/'protocol.json').exists():return json.loads((OUT/'protocol.json').read_text())
 # These rules are fixed before fitting or observing any probe score.
 protocol=dict(primary_layer='avgpool',primary_method='standardized ridge one-hot argmax',ridge_alpha=1.,
   split_seeds=SEEDS,primary_split_seed=1701,labels={'core':'MNIST digit 0..9','shortcut':'color index 0..9'},
   split='Fixed source-image-disjoint stratification over digit x color; all epochs and conditions share splits',
   duplicate_rule='Remove snapshot images whose grayscale pixels occur in main-model training; retain only one of within-pool duplicates, then downsample every joint cell to common minimum',
   train_fraction=.75,scaling='train-only per-feature mean/std; intercept via centered targets; never fit preprocessing on held-out data',
   primary_curve_layers=['features.3','avgpool'],additional_layer_epochs=LAYER_EPOCHS,
   sensitivity_alphas=[.1,10.],sensitivity_epochs='bg .95: 0,1,28,49,50; other conditions: 0,1,50',
   softmax='Existing evaluate.DecodedCorrelation: 100 epochs, batch256, AdamW lr.001 wd.0001; train-only standardized features; primary split only; same sensitivity epochs',
   decodability_criteria='For each target, earliest observed start of 3 consecutive epochs with held-out accuracy >=90% (secondary95%) on ALL three fixed splits at avgpool. Include epoch0; report left censoring if already decodable, never call this true core-learning onset.',
   statistics='Repeated partitions are not model seeds; no p-values or seed-generalization claims; anomaly paired bootstrap only on primary split, conditional on frozen readouts',
   resource='CPU1, BLAS1; one snapshot/layer at a time, no GPU, no model inference')
 write(OUT/'protocol.json',protocol);return protocol

def prepare_split():
 if (OUT/'split_manifest.json').exists():
  m=json.loads((OUT/'split_manifest.json').read_text());return m
 cache=ROOT/'.cache/huggingface'
 os.environ.update(HF_HOME=str(cache),HF_DATASETS_CACHE=str(cache/'datasets'),HF_HUB_CACHE=str(cache/'hub'),HF_MODULES_CACHE=str(cache/'modules'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
 from datasets import load_dataset
 raw=load_dataset('ylecun/mnist')['train']
 with np.load(run_path('background',.95)/'features/epoch_000.npz',allow_pickle=False) as z:
  ids=z['indices'];labels=z['labels'];colors=z['spurious']
 val_ids=np.random.default_rng(42).permutation(60000)[:5000];assert set(ids)<=set(val_ids)
 val_set=set(val_ids.tolist());pool_set=set(ids.tolist());pool_hash={};train_hash=set()
 logging.info('Audit grayscale identities against the main-model training split (CPU metadata only)')
 for i in range(60000):
  h=hashlib.sha256(np.asarray(raw[i]['image'].convert('L'),dtype=np.uint8).tobytes()).hexdigest()
  if i not in val_set:train_hash.add(h)
  if i in pool_set:pool_hash[i]=h
 keep=[];seen=set();excluded=[]
 for j,i in enumerate(ids):
  h=pool_hash[int(i)]
  reason='same grayscale pixels in model training' if h in train_hash else ('duplicate grayscale in snapshot' if h in seen else None)
  if reason:excluded.append({'index':int(i),'reason':reason})
  else:keep.append(j);seen.add(h)
 groups={g:[j for j in keep if int(labels[j]*10+colors[j])==g] for g in range(100)}
 m=min(map(len,groups.values()));ntrain=int(np.floor(.75*m));assert ntrain>=8 and m-ntrain>=4
 splits=[]
 for seed in SEEDS:
  rng=np.random.default_rng(seed);train=[];test=[]
  for g in range(100):
   choices=rng.permutation(groups[g])[:m];train.extend(choices[:ntrain]);test.extend(choices[ntrain:])
  train=sorted(train);test=sorted(test)
  assert not set(ids[train])&set(ids[test])
  assert not {pool_hash[int(i)] for i in ids[train]}&{pool_hash[int(i)] for i in ids[test]}
  assert np.all(np.bincount(labels[train]*10+colors[train],minlength=100)==ntrain)
  assert np.all(np.bincount(labels[test]*10+colors[test],minlength=100)==m-ntrain)
  splits.append({'seed':seed,'train_rows':train,'test_rows':test})
 manifest=dict(original_n=len(ids),ids=ids.tolist(),labels=labels.tolist(),colors=colors.tolist(),excluded=excluded,
   common_cell_n=m,train_per_cell=ntrain,test_per_cell=m-ntrain,splits=splits,
   source='Original validation snapshots; neither probe partition uses main-model training IDs',
   raw_cache_manifest_sha256=sha(cache/'mnist_manifest.json'),pixel_hashes={str(i):h for i,h in pool_hash.items()},
   content_identity_limit='Pixel duplicate check uses newly pinned official raw cache; historical raw pixel identity was not recorded')
 write(OUT/'split_manifest.json',manifest)
 logging.info('Split ready: removed %d images; %d train / %d test per joint cell',len(excluded),ntrain,m-ntrain)
 return manifest

def fit_ridge(X,labels,colors,train,test,alpha):
 tr=np.asarray(train);te=np.asarray(test)
 x=X[tr].astype(np.float64);mean=x.mean(0);scale=x.std(0);scale[scale<1e-12]=1
 x=(x-mean)/scale;z=(X[te].astype(np.float64)-mean)/scale
 Y=np.concatenate([np.eye(10)[labels[tr]],np.eye(10)[colors[tr]]],axis=1);base=Y.mean(0)
 gram=x.T@x;gram.flat[::len(gram)+1]+=alpha
 w=cho_solve(cho_factor(gram,lower=True,check_finite=False),x.T@(Y-base),check_finite=False)
 logits=z@w+base;train_logits=x@w+base
 pred=np.stack([logits[:,:10].argmax(1),logits[:,10:].argmax(1)])
 train_pred=np.stack([train_logits[:,:10].argmax(1),train_logits[:,10:].argmax(1)])
 return pred,train_pred,(mean,scale,x,z)

def scores(pred,y):
 a=float(np.mean(pred==y));return dict(accuracy=a,chance_accuracy=.1,chance_adjusted_accuracy=(a-.1)/.9,legacy_decoded_correlation=2*a-1)

def main():
 OUT.mkdir(exist_ok=True);(OUT/'batches').mkdir(exist_ok=True)
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'execution.log'),logging.StreamHandler(sys.stdout)])
 start=time.monotonic();plan();update(stage='split preparation',status='running')
 split=prepare_split();ids=np.array(split['ids']);labels=np.array(split['labels']);colors=np.array(split['colors'])
 from evaluate import DecodedCorrelation
 import torch
 torch.set_num_threads(1)
 from probe_dynamics import probe_scores
 # Audit input source files once; no original writes. Resume consumes atomic completed batches.
 for mode in MODES:
  for corr in CORRS:
   parent=run_path(mode,corr);tag=f'{mode}_{round(corr*100):03d}'
   cfg=json.loads((parent/'config.json').read_text());assert cfg['seed']==42 and cfg['train_corr']==corr
   for epoch in range(51):
    dest=OUT/'batches'/f'{tag}_e{epoch:03d}.json';pred_file=dest.with_suffix('.npz')
    if dest.exists() and pred_file.exists():
     prior=json.loads(dest.read_text());src=ROOT/prior['source'];st=src.stat()
     assert st.st_size==prior['source_bytes'] and st.st_mtime_ns==prior['source_mtime_ns'],str(src)
     continue
    path=parent/f'features/epoch_{epoch:03d}.npz';stat=path.stat();digest=sha(path)
    rows=[];pred_saved={};alias=True
    with np.load(path,allow_pickle=False) as z:
     assert np.array_equal(z['indices'],ids) and np.array_equal(z['labels'],labels) and np.array_equal(z['spurious'],colors)
     layers=z['layers'].tolist();selected=[0,5]+([1,2,3] if epoch in LAYER_EPOCHS else [])
     alias=np.array_equal(z['f4'],z['f5']);assert alias
     sensitivity=epoch in (SOFT_BG if mode=='background' and corr==.95 else [0,1,50])
     for li in selected:
      X=z[f'f{li}'];assert np.isfinite(X).all()
      for si,part in enumerate(split['splits']):
       train,test=part['train_rows'],part['test_rows'];tr=np.array(train);te=np.array(test)
       pred,train_pred,scaling=fit_ridge(X,labels,colors,train,test,1.)
       for ti,(target,y) in enumerate([('core',labels),('shortcut',colors)]):
        rows.append(dict(mode=mode,train_corr=corr,epoch=epoch,layer=layers[li],dim=X.shape[1],split_seed=part['seed'],method='ridge',alpha=1.,target=target,train_n=len(tr),test_n=len(te),train_accuracy=float(np.mean(train_pred[ti]==y[tr])),**probe_scores(float(np.mean(pred[ti]==y[te])))))
       if li==5:pred_saved[f'split{si}']=pred.astype(np.uint8)
       if li==5 and si==0 and sensitivity:
        for alpha in [.1,10.]:
         pa,pt,_=fit_ridge(X,labels,colors,train,test,alpha)
         for ti,(target,y) in enumerate([('core',labels),('shortcut',colors)]):
          rows.append(dict(mode=mode,train_corr=corr,epoch=epoch,layer=layers[li],dim=X.shape[1],split_seed=part['seed'],method='ridge',alpha=alpha,target=target,train_n=len(tr),test_n=len(te),train_accuracy=float(np.mean(pt[ti]==y[tr])),**probe_scores(float(np.mean(pa[ti]==y[te])))))
        _,_,x,ztest=scaling
        for target,y in [('core',labels),('shortcut',colors)]:
         torch.manual_seed(1701)
         probe=DecodedCorrelation(X.shape[1],10).cpu()
         probe.fit(torch.tensor(x,dtype=torch.float32),torch.tensor(y[tr]),epochs=100,batch_size=256,lr=.001,weight_decay=.0001)
         score=probe.evaluate(torch.tensor(ztest,dtype=torch.float32),torch.tensor(y[te]))['accuracy']
         rows.append(dict(mode=mode,train_corr=corr,epoch=epoch,layer=layers[li],dim=X.shape[1],split_seed=part['seed'],method='softmax_AdamW',alpha=None,target=target,train_n=len(tr),test_n=len(te),train_accuracy=None,**probe_scores(score)))
    assert path.stat().st_size==stat.st_size and path.stat().st_mtime_ns==stat.st_mtime_ns
    temp=pred_file.with_suffix('.tmp.npz');np.savez_compressed(temp,**pred_saved);os.replace(temp,pred_file)
    write(dest,dict(source=str(path.relative_to(ROOT)),source_sha256=digest,source_bytes=stat.st_size,source_mtime_ns=stat.st_mtime_ns,alias_features28_equals_avgpool=alias,rows=rows))
    count=len(list((OUT/'batches').glob('*.json')));update(stage='probe fits',status='running',completed_snapshots=count,total_snapshots=306,last=f'{tag} e{epoch}',elapsed_seconds=time.monotonic()-start)
    if epoch%5==0 or epoch in [27,28,29,46,47,48,49]:logging.info('%s e%d saved (%d/306)',tag,epoch,count)
 assert not torch.cuda.is_initialized()
 import pandas as pd
 batches=[json.loads(p.read_text()) for p in sorted((OUT/'batches').glob('*.json'))]
 results=pd.DataFrame([r for b in batches for r in b['rows']]);results.to_csv(OUT/'probe_results.csv',index=False)
 write(OUT/'feature_source_manifest.json',[{k:v for k,v in b.items() if k!='rows'} for b in batches])
 update(stage='probe fits complete',status='awaiting report',completed_snapshots=306,total_snapshots=306,probe_rows=len(results),elapsed_seconds=time.monotonic()-start)
 logging.info('Saved %d probe rows; no GPU initialized',len(results))
if __name__=='__main__':
 try:
  with threadpool_limits(limits=1):main()
 except BaseException as e:
  update(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e),traceback=traceback.format_exc());raise
