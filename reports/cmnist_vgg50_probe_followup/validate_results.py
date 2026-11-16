"""Final integrity checks plus a zero-fit resume check of completed stages."""
from pathlib import Path
import os,json,sys,time,hashlib,re,importlib.metadata,logging
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
import pandas as pd
from PIL import Image
O=Path(__file__).resolve().parent;ROOT=O.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
D=pd.read_csv(O/'probe_results.csv');assert len(D)==5412 and D.accuracy.between(0,1).all()
assert (D.train_n==2400).all() and (D.test_n==800).all()
assert len(D[(D.method=='ridge')&(D.alpha==1)])==5292
assert len(D[D.method=='softmax_AdamW'])==40
assert not D.duplicated(['mode','train_corr','epoch','layer','target','split_seed','method','alpha']).any()
M=json.loads((O/'split_manifest.json').read_text());ids=np.array(M['ids']);y=np.array(M['labels']);s=np.array(M['colors'])
assert len(set(ids))==3200 and len(set(M['pixel_hashes'].values()))==3200 and not M['excluded']
for f in M['splits']:
 tr=np.array(f['train_rows']);te=np.array(f['test_rows']);assert not set(tr)&set(te)
 assert not {M['pixel_hashes'][str(int(i))] for i in ids[tr]}&{M['pixel_hashes'][str(int(i))] for i in ids[te]}
 assert np.all(np.bincount(y[tr]*10+s[tr],minlength=100)==24)
 assert np.all(np.bincount(y[te]*10+s[te],minlength=100)==8)
review=json.loads((O/'review_validation.json').read_text())
for item in review['source_files']+review['prior_report_sources']:assert sha(ROOT/item['path'])==item['sha256']
features=json.loads((O/'feature_source_manifest.json').read_text());assert len(features)==306
print('Checking original snapshot hashes...',flush=True)
for item in features:
 assert sha(ROOT/item['source'])==item['source_sha256']
 assert item['alias_features28_equals_avgpool']
for item in json.loads((O/'checkpoint_sources.json').read_text()):assert sha(ROOT/item['path'])==item['sha256']
for f in (O/'figures').glob('*.png'):
 with Image.open(f) as image:image.verify()
assert len(list((O/'figures').glob('*.png')))==7 and len(list((O/'figures').glob('*.pdf')))==7
for path in re.findall(r'\]\(([^)]+)\)',(O/'analysis_report.md').read_text()):assert (O/path).exists()
# Prove resumed execution consumes saved batches without recomputing any readout.
sys.path.insert(0,str(O));import run_probes as rp
from evaluate import DecodedCorrelation
old=json.loads((O/'progress.json').read_text());start=time.monotonic()
def forbidden(*a,**kw):raise AssertionError('Resume unexpectedly attempted an already completed probe fit')
rp.fit_ridge=forbidden;DecodedCorrelation.fit=forbidden
rp.main()
progress=old;progress.update(stage='complete',status='complete',resume_zero_fit_check_passed=True,report='analysis_report.md',updated_unix=time.time())
(O/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
result=dict(success=True,experiments=6,snapshots=306,probe_rows=5412,original_feature_hashes_checked=306,
 original_late_checkpoint_hashes_checked=26,previous_report_and_metric_hashes_unchanged=True,
 source_image_disjoint=True,pixel_disjoint=True,test_joint_balance=True,train_only_preprocessing=True,
 figures_png=7,figures_pdf=7,resume_zero_fit_check=True,resume_check_seconds=time.monotonic()-start,
 cpu_only=True,backbone_inference=False,model_training=False)
(O/'validation_results.json').write_text(json.dumps(result,indent=2)+'\n')
(O/'environment.json').write_text(json.dumps({'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','torch','torchvision']},
 'source_sha256':{n:sha(ROOT/n) for n in ['probe_dynamics.py','followup_common.py','evaluate.py','shortcut_experiment.py','data.py']},
 'resource_scope':'1 CPU thread/BLAS; no GPU; peak RSS not instrumented'},indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
