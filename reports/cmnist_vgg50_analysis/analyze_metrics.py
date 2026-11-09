"""Read-only, CPU-only analysis of the six Job 153567 runs. Outputs stay here.
Reuses analyze.py's saved-history curve definitions; does not call its curves()
because that function writes into original runs. No model inference or unpickling.
"""
import os
os.environ['MPLCONFIGDIR']='/tmp/cmnist-analysis-mpl'
from pathlib import Path
import json, zipfile, pickletools, hashlib, re, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
MODES=['background','foreground']; CORRS=[.95,.80,.65]
COLORS={.95:'#c44e52',.80:'#dd9222',.65:'#2878b5'}
audit=[]; runs={}; rows=[]; sources=[]; feature_errors=[]
metrics=['acc','aligned_acc','conflicting_acc','wga','cf_flip_rate','cf_acc','cf_acc_drop','cf_all_flip_rate','cf_all_acc','cf_all_color_rate']
def source(p):
 sources.append({'path':str(p.relative_to(ROOT)),'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
def crossing(d, target):
 # Fixed common operational rule: conflict and all-CF accuracy >= target,
 # all-CF flip <= 1-target, three consecutive observed epochs, excluding initialization.
 ok=(d.val_conflicting_acc>=target)&(d.val_cf_all_acc>=target)&(d.val_cf_all_flip_rate<=1-target)&(d.epoch>=1)
 for i in range(len(d)-2):
  if ok.iloc[i:i+3].all() and np.array_equal(d.epoch.iloc[i:i+3],np.arange(d.epoch.iloc[i],d.epoch.iloc[i]+3)):
   return int(d.epoch.iloc[i])
 return None
for task,(mode,corr) in enumerate((m,c) for m in MODES for c in CORRS):
 tag='bg' if mode=='background' else 'fg'; p=ROOT/f'runs/vgg11_{tag}_corr{round(corr*100):03d}_seed42_e50_img224_all'
 cfg=json.loads((p/'config.json').read_text()); d=pd.read_csv(p/'history.csv'); test=json.loads((p/'test_metrics.json').read_text()); t=test['metrics']
 for f in ['config.json','history.csv','test_metrics.json','latest.json','TRAIN_SUCCESS']: source(p/f)
 log=(ROOT/f'logs/cmnist_153567_{task}.out').read_text(); err=(ROOT/f'logs/cmnist_153567_{task}.err').read_text()
 for ext in ['out','err']: source(ROOT/f'logs/cmnist_153567_{task}.{ext}')
 logged=json.loads(re.search(r'^\[config\] (.*)$',log,re.M).group(1))
 assert logged==cfg and cfg['color_mode']==mode and cfg['train_corr']==corr
 assert d.epoch.tolist()==list(range(51)) and not d.epoch.duplicated().any()
 assert d.loc[d.epoch>0,'train_acc'].notna().all() and pd.isna(d.loc[0,'train_acc'])
 assert np.isfinite(d[['val_'+m for m in metrics]].to_numpy()).all()
 assert all(d['val_'+m].between(0,1).all() for m in metrics if m!='cf_acc_drop')
 assert np.allclose(d.val_acc-d.val_cf_acc,d.val_cf_acc_drop)
 assert test['epoch']==50 and json.loads((p/'latest.json').read_text())['epoch']==50
 assert 'FINAL TEST:' in log and f'Run directory: {p}' in log
 assert (p/'TRAIN_SUCCESS').exists()
 counts=[c for c in d if re.fullmatch(r'val_g\d\d_n',c)]
 assert (d[counts].sum(axis=1)==5000).all()
 assert (d[counts]==d.loc[0,counts]).all().all()
 # Independently reconstruct WGA from 20 digit x {aligned, other} group metrics.
 wg=[f'val_g{y}{y}_acc' for y in range(10)]+[f'val_g{y}other_acc' for y in range(10)]
 assert np.allclose(d[wg].min(axis=1),d.val_wga)
 assert all(np.isfinite(t[m]) for m in metrics)
 assert sum(t[f'g{y}{s}_n'] for y in range(10) for s in range(10))==10000
 ck=[]; ft=[]
 for e in range(51):
  cp=p/f'checkpoints/epoch_{e:03d}.pt'
  with zipfile.ZipFile(cp) as z:
   info=z.infolist(); assert all(x.file_size>0 and x.header_offset<cp.stat().st_size for x in info)
   pk=z.read(next(x.filename for x in info if x.filename.endswith('/data.pkl')))
   ops=list(pickletools.genops(pk)); saved_epoch=None
   for j,(op,arg,pos) in enumerate(ops[:-1]):
    if arg=='epoch': saved_epoch=next(arg2 for op2,arg2,pos2 in ops[j+1:] if op2.name not in ['BINPUT','LONG_BINPUT','MEMOIZE']); break
   assert saved_epoch==e
  ck.append({'epoch':e,'bytes':cp.stat().st_size,'archive_members':len(info),'metadata_epoch':saved_epoch})
  fp=p/f'features/epoch_{e:03d}.npz'
  with np.load(fp,allow_pickle=False) as z:
   layers=z['layers'].tolist(); labels=z['labels']; spur=z['spurious']; ids=z['indices']
   assert len(labels)==3200 and len(np.unique(ids))==3200
   shapes={}
   for k in range(len(layers)):
    a=z[f'f{k}']; assert a.shape[0]==3200 and np.isfinite(a).all(); shapes[layers[k]]=list(a.shape)
   assert np.all(np.bincount(labels*10+spur,minlength=100)==32)
  ft.append({'epoch':e,'bytes':fp.stat().st_size,'shapes':shapes})
 a={'task':task,'mode':mode,'corr':corr,'path':str(p.relative_to(ROOT)),'config':cfg,'log_config_matches':True,'history_epochs':d.epoch.tolist(),'checkpoints':ck,'features':ft,'stderr':err.strip(),'validation_n':5000,'test_n':10000,'aligned_validation_n':int(sum(d.loc[0,f'val_g{y}{y}_n'] for y in range(10))),'gradcam_exists':(p/'gradcam_comparison').exists(),'extra_eval_exists':(p/'evaluation_e050').exists()}
 audit.append(a); runs[(mode,corr)]=(d,t)
 r={'task':task,'mode':mode,'train_corr':corr,'seed':42,'history_rows':len(d),'checkpoint_count':len(ck),'feature_count':len(ft)}
 for m in metrics:
  r['val_e1_'+m]=d.loc[1,'val_'+m]; r['val_e50_'+m]=d.loc[50,'val_'+m]; r['test_e50_'+m]=t[m]
 r['train_e50_acc']=d.loc[50,'train_acc']
 for threshold in [.90,.95,.98]: r[f'proxy_{round(threshold*100)}_first_epoch']=crossing(d,threshold)
 for lo,hi,label in [(1,5,'early_1_5'),(6,25,'middle_6_25'),(26,50,'late_26_50')]:
  for m in metrics:r[f'{label}_{m}_mean']=d.loc[d.epoch.between(lo,hi),'val_'+m].mean()
 for m in ['cf_flip_rate','cf_all_flip_rate','cf_all_color_rate']:
  s=d.loc[d.epoch>=1,'val_'+m]; r[m+'_increasing_steps']=int((s.diff()>1e-12).sum()); r[m+'_min_epoch']=int(d.loc[s.idxmin(),'epoch']); r[m+'_min']=s.min(); r[m+'_max_rebound_pp']=100*(s-s.cummin()).max()
 r['best_val_epoch']=int(d.loc[d.val_acc.idxmax(),'epoch']);r['best_val_acc']=d.val_acc.max()
 rows.append(r)
 print(f'Validated task {task}: {mode} {corr}, 51 history/checkpoints/features',flush=True)
base={k:v for k,v in audit[0]['config'].items() if k not in ['color_mode','train_corr']}
assert all({k:v for k,v in a['config'].items() if k not in ['color_mode','train_corr']}==base for a in audit)
s=pd.DataFrame(rows); s.to_csv(OUT/'summary_table.csv',index=False)
(OUT/'artifact_audit.json').write_text(json.dumps(audit,indent=2))
(OUT/'source_manifest.json').write_text(json.dumps(sources,indent=2))
pd.concat([d.assign(mode=m,train_corr=c) for (m,c),(d,t) in runs.items()]).to_csv(OUT/'epoch_metrics.csv',index=False)
# Same plot semantics as analyze.py, with explicit test-only final markers and separate panels.
def panel_grid(name, columns, labels, ylim=None):
 fig,axs=plt.subplots(2,len(columns),figsize=(5*len(columns),7.3),squeeze=False,layout='constrained')
 for i,mode in enumerate(MODES):
  for j,(col,label) in enumerate(zip(columns,labels)):
   ax=axs[i,j]
   for corr in CORRS:
    d,t=runs[(mode,corr)]; y=d['val_'+col]*100
    ax.plot(d.epoch,y,color=COLORS[corr],label=f'train corr={corr:.2f}')
   ax.set(title=f'{mode}: {label}',xlabel='Epoch',ylabel=('Accuracy difference (pp)' if col=='cf_acc_drop' else 'Percent (%)'));ax.grid(alpha=.25);ax.legend(fontsize=8)
   if ylim: ax.set_ylim(*ylim)
 fig.suptitle('VGG11_BN / Colored MNIST / seed 42 / validation corr=0.10 (N=5000)')
 fig.savefig(FIG/(name+'.png'),dpi=180);fig.savefig(FIG/(name+'.pdf'));plt.close(fig)
panel_grid('02_groups',['aligned_acc','conflicting_acc','wga'],['Aligned accuracy','Conflicting accuracy','Worst-group accuracy'])
panel_grid('03_flip_rates',['cf_flip_rate','cf_all_flip_rate'],['Paired CF flip rate','All-color CF flip rate'])
panel_grid('04_counterfactual',['cf_acc','cf_acc_drop','cf_all_acc'],['Paired CF accuracy','Paired CF accuracy drop (pp)','All-color conflicting CF accuracy'])
panel_grid('05_correlation_dependence',['cf_all_flip_rate','cf_all_color_rate','cf_all_acc'],['All-color flip rate','Conflicting recolor: prediction=color','All-color conflicting CF accuracy'])
fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
for i,mode in enumerate(MODES):
 for j,corr in enumerate(CORRS):
  d,t=runs[(mode,corr)];ax=axs[i,j]
  ax.plot(d.epoch,d.train_acc*100,label='Train (online)');ax.plot(d.epoch,d.val_acc*100,label='Validation')
  ax.scatter([50],[t['acc']*100],marker='*',s=100,label='Test: epoch 50 only',zorder=4)
  ax.set(title=f'{mode}, train corr={corr:.2f}',xlabel='Epoch',ylabel='Accuracy (%)');ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.suptitle('VGG11_BN / seed 42 / validation & test corr=0.10');fig.savefig(FIG/'01_accuracy.png',dpi=180);fig.savefig(FIG/'01_accuracy.pdf');plt.close(fig)
fig,axs=plt.subplots(3,3,figsize=(15,11),layout='constrained')
for j,corr in enumerate(CORRS):
 for i,(metric,label) in enumerate([('cf_all_flip_rate','All-color flip rate'),('cf_all_acc','All-color conflicting CF accuracy'),('conflicting_acc','Conflicting accuracy')]):
  ax=axs[i,j]
  for mode,style in zip(MODES,['-','--']):
   d,t=runs[(mode,corr)];ax.plot(d.epoch,100*d['val_'+metric],style,label=mode)
  ax.set(title=f'corr={corr:.2f}: {label}',xlabel='Epoch',ylabel='Percent (%)');ax.legend();ax.grid(alpha=.25)
fig.suptitle('VGG11_BN / seed 42 / validation corr=0.10');fig.savefig(FIG/'06_background_vs_foreground.png',dpi=180);fig.savefig(FIG/'06_background_vs_foreground.pdf');plt.close(fig)
print(s[['mode','train_corr','proxy_90_first_epoch','proxy_95_first_epoch','proxy_98_first_epoch','cf_all_flip_rate_increasing_steps','cf_all_flip_rate_max_rebound_pp','best_val_epoch']].to_string(index=False))
