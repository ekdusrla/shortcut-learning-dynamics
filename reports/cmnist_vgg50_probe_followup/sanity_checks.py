"""Small CPU checks of linear readout implementation and label-permutation controls."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_probes as rp
O=rp.OUT;split=json.loads((O/'split_manifest.json').read_text());part=split['splits'][0]
tr=np.array(part['train_rows']);te=np.array(part['test_rows']);y=np.array(split['labels']);c=np.array(split['colors']);rows=[]
with threadpool_limits(limits=1):
 for mode in rp.MODES:
  for corr in rp.CORRS:
   for epoch in [0,50]:
    with np.load(rp.run_path(mode,corr)/f'features/epoch_{epoch:03d}.npz') as z:X=z['f5']
    rng=np.random.default_rng(91701);yp=y.copy();cp=c.copy();yp[tr]=rng.permutation(y[tr]);cp[tr]=rng.permutation(c[tr])
    pred,_,_=rp.fit_ridge(X,yp,cp,tr,te,1.)
    for k,(target,truth) in enumerate([('core',y),('shortcut',c)]):rows.append(dict(mode=mode,train_corr=corr,epoch=epoch,target=target,permuted_train_label_accuracy=float(np.mean(pred[k]==truth[te])),chance=.1,test_n=len(te)))
 with np.load(rp.run_path('background',.95)/'features/epoch_001.npz') as z:X=z['f5']
 pred,_,(_,_,x,z)=rp.fit_ridge(X,y,c,tr,te,1.)
 Y=np.concatenate([np.eye(10)[y[tr]],np.eye(10)[c[tr]]],axis=1)
 model=Ridge(alpha=1.,solver='cholesky').fit(x,Y);score=model.predict(z)
 reference=np.stack([score[:,:10].argmax(1),score[:,10:].argmax(1)])
 assert np.array_equal(pred,reference)
pd.DataFrame(rows).to_csv(O/'permutation_controls.csv',index=False)
assert np.mean([r['permuted_train_label_accuracy'] for r in rows])<.15
(O/'probe_sanity_checks.json').write_text(json.dumps({'success':True,'ridge_predictions_match_sklearn_on_reference_snapshot':True,
 'permutation_control_rows':len(rows),'permutation_accuracy_min':min(r['permuted_train_label_accuracy'] for r in rows),
 'permutation_accuracy_max':max(r['permuted_train_label_accuracy'] for r in rows),
 'permutation_accuracy_mean':float(np.mean([r['permuted_train_label_accuracy'] for r in rows])),
 'meaning':'Label-permuted probe fits are near chance; this does not establish causal feature use or historical raw pixel identity.'},indent=2)+'\n')
print((O/'probe_sanity_checks.json').read_text())
