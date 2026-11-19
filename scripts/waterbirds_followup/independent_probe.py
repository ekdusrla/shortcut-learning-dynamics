"""CPU probes for manually extracted independent Val-fit/Test-score features."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from scripts.waterbirds_followup.probes import probe

def main():
 p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 torch.set_num_threads(1);rows=[];identities={}
 for code in ('095','080','065'):
  root=a.features/f'A_corr{code}';done=json.loads((root/'EXTRACTION_SUCCESS').read_text());assert done['epochs']==list(range(51))
  for e in range(51):
   v=np.load(root/f'val_epoch_{e:03d}.npz');t=np.load(root/f'test_epoch_{e:03d}.npz');assert not set(v['indices'])&set(t['indices']);assert np.array_equal(v['layers'],t['layers'])
   for split,f in [('val',v),('test',t)]:
    ids=tuple(f['indices']);labels=tuple(f['labels']);places=tuple(f['spurious']);current=(ids,labels,places)
    if split not in identities:identities[split]=current
    assert identities[split]==current
    assert all(np.sum((f['labels']==y)&(f['spurious']==s))==32 for y in (0,1) for s in (0,1))
   for i,l in enumerate(v['layers']):
    x=np.concatenate([v[f'f{i}'],t[f'f{i}']]);fit=np.arange(128);score=np.arange(128,256)
    for target,key in [('core','labels'),('spurious','spurious')]:
     y=np.concatenate([v[key],t[key]]);rows.append(dict(condition='A_corr'+code,epoch=e,layer=str(l),target=target,evaluation_split='independent_test',**probe(x,y,fit,score)))
 a.output.mkdir(parents=True);pd.DataFrame(rows).to_csv(a.output/'probe_summary.csv',index=False)
 (a.output/'protocol.json').write_text(json.dumps(dict(protocol='original run.py analyze Val-fit/Test-score',fit_n=128,score_n=128,model='evaluate.DecodedCorrelation',epochs=100,batch_size=256,lr=.001,weight_decay=.0001,seed=42,note='Keep separate from validation content holdout results.'),indent=2))
if __name__=='__main__':main()
