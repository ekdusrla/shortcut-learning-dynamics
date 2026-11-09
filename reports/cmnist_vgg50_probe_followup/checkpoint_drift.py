"""CPU-only descriptive late-checkpoint head/BN state comparisons; no model is built or inferred."""
from pathlib import Path
import hashlib,json,os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import pandas as pd
O=Path(__file__).resolve().parent;ROOT=O.parents[1];P=ROOT/'runs/vgg11_bg_corr095_seed42_e50_img224_all/checkpoints'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
if (O/'checkpoint_drift.csv').exists():
 print('Reusing saved checkpoint diagnostics');raise SystemExit()
torch.set_num_threads(1);rows=[];manifest=[];previous=None
for e in range(25,51):
 p=P/f'epoch_{e:03d}.pt';state=torch.load(p,map_location='cpu',weights_only=True);assert state['epoch']==e
 keep={k:v.detach().clone() for k,v in state['model'].items() if k.startswith('classifier.') or k.endswith('running_mean') or k.endswith('running_var')}
 manifest.append({'epoch':e,'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
 del state
 if previous is not None:
  for key,x in keep.items():
   old=previous[key];delta=torch.linalg.vector_norm(x-old).item();base=torch.linalg.vector_norm(old).item()
   rows.append({'from_epoch':e-1,'to_epoch':e,'tensor':key,'delta_l2':delta,'previous_l2':base,'relative_l2':delta/(base+1e-12)})
  for digit in range(10):
   x=keep['classifier.weight'][digit];old=previous['classifier.weight'][digit]
   rows.append({'from_epoch':e-1,'to_epoch':e,'tensor':f'classifier.weight_digit{digit}','delta_l2':torch.linalg.vector_norm(x-old).item(),
                'previous_l2':torch.linalg.vector_norm(old).item(),'relative_l2':(torch.linalg.vector_norm(x-old)/(torch.linalg.vector_norm(old)+1e-12)).item()})
 previous=keep
 assert not torch.cuda.is_initialized()
d=pd.DataFrame(rows);d['rank_relative_l2_desc_within_tensor']=d.groupby('tensor').relative_l2.rank(method='min',ascending=False)
d.to_csv(O/'checkpoint_drift.csv',index=False)
(O/'checkpoint_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(d[(d.to_epoch.isin([28,49]))&d.tensor.isin(['classifier.weight','classifier.weight_digit5'])].to_string(index=False))
