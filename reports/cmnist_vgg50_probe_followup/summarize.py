"""Join saved probe batches and existing behavior; no feature refits or inference."""
from pathlib import Path
import sys
FIGURES_ONLY = "--figures-only" in sys.argv
import json,os
os.environ['MPLCONFIGDIR']='/tmp/cmnist-probe-mpl'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;ROOT=O.parents[1];FIG=O/'figures';FIG.mkdir(exist_ok=True)
D=pd.read_csv(O/'probe_results.csv');S=json.loads((O/'split_manifest.json').read_text())
primary=D[(D.method=='ridge')&(D.alpha==1)].copy()
curves=primary.groupby(['mode','train_corr','epoch','layer','target'],as_index=False).accuracy.agg(['mean','min','max']).reset_index()
curves.to_csv(O/'probe_curves.csv',index=False)
A=curves[curves.layer=='avgpool'];OLD=ROOT/'reports/cmnist_vgg50_analysis_review';behavior=pd.read_csv(OLD/'epoch_metrics.csv')
old_summary=pd.read_csv(OLD/'summary_table.csv');timings=[]
for (mode,corr,target),part in A.groupby(['mode','train_corr','target']):
 part=part.sort_values('epoch');assert part.epoch.tolist()==list(range(51))
 for threshold in [.9,.95]:
  good=part['min'].to_numpy()>=threshold;first=None
  for i in range(49):
   if good[i:i+3].all():first=i;break
  timings.append(dict(mode=mode,train_corr=corr,target=target,threshold=threshold,consecutive_epochs=3,first_observed_criterion_epoch=first,
    criterion_met_untrained_epoch0=bool(good[0]),interpretation='criterion crossing only; true learning onset not identified'))
T=pd.DataFrame(timings);T.to_csv(O/'decodability_times.csv',index=False)
summary=[]
for mode in ['background','foreground']:
 for corr in [.95,.8,.65]:
  b=old_summary[(old_summary['mode']==mode)&(old_summary.train_corr==corr)].iloc[0]
  r={'mode':mode,'train_corr':corr,'behavior95_first_epoch':b.proxy_95_first_epoch,'final_behavior_all_flip':b.val_e50_cf_all_flip_rate}
  for target in ['core','shortcut']:
   for e in [0,1,2,28,49,50]:
    q=A[(A['mode']==mode)&(A.train_corr==corr)&(A.target==target)&(A.epoch==e)].iloc[0]
    for k in ['mean','min','max']:r[f'{target}_e{e}_{k}']=q[k]
   for level in [.9,.95]:r[f'{target}_{round(level*100)}_first_epoch']=T[(T['mode']==mode)&(T.train_corr==corr)&(T.target==target)&(T.threshold==level)].iloc[0].first_observed_criterion_epoch
  summary.append(r)
pd.DataFrame(summary).to_csv(O/'probe_summary.csv',index=False)
D[(D.method!='ridge')|(D.alpha!=1)].to_csv(O/'sensitivity_results.csv',index=False)
if not FIGURES_ONLY:
    # Same held-out images across epoch comparisons. Bootstrap is conditional on frozen primary split readouts.
    bootstrap=[];probe_groups=[]
    rng=np.random.default_rng(1701)
    for e0,e1 in [(27,28),(28,29),(48,49),(49,50)]:
     with np.load(O/f'batches/background_095_e{e0:03d}.npz') as z:before={k:z[k] for k in z.files}
     with np.load(O/f'batches/background_095_e{e1:03d}.npz') as z:after={k:z[k] for k in z.files}
     for si,part in enumerate(S['splits']):
      ix=np.array(part['test_rows']);labels=np.array(S['labels'])[ix];colors=np.array(S['colors'])[ix]
      for ti,(target,y) in enumerate([('core',labels),('shortcut',colors)]):
       ok0=before[f'split{si}'][ti]==y;ok1=after[f'split{si}'][ti]==y
       gain=int((~ok0&ok1).sum());loss=int((ok0&~ok1).sum());n=len(y)
       if si==0:
        sampled=rng.multinomial(n,[gain/n,loss/n,1-(gain+loss)/n],size=10000)
        dif=(sampled[:,0]-sampled[:,1])/n
        lo,hi=np.quantile(dif,[.025,.975])
        bootstrap.append(dict(from_epoch=e0,to_epoch=e1,target=target,split_seed=part['seed'],n=n,gains=gain,losses=loss,
          observed_difference_pp=100*(gain-loss)/n,conditional_bootstrap_low_pp=100*lo,conditional_bootstrap_high_pp=100*hi,
          scope='fixed primary probe split/readouts; image resampling uncertainty only, not seed variance; no multiplicity correction'))
       for digit in range(10):
        for kind in ['aligned','conflicting']:
         mask=(labels==digit)&((labels==colors) if kind=='aligned' else (labels!=colors))
         probe_groups.append(dict(from_epoch=e0,to_epoch=e1,target=target,split_seed=part['seed'],digit=digit,kind=kind,n=int(mask.sum()),
           accuracy_before=float(ok0[mask].mean()),accuracy_after=float(ok1[mask].mean()),gains=int((~ok0&ok1&mask).sum()),losses=int((ok0&~ok1&mask).sum())))
    pd.DataFrame(bootstrap).to_csv(O/'probe_event_paired_bootstrap.csv',index=False)
    pd.DataFrame(probe_groups).to_csv(O/'probe_event_group_scores.csv',index=False)
    E=pd.read_csv(O/'event_metrics.csv');bg=E[(E['mode']=='background')&(E.train_corr==.95)]
    ep=bg.copy()
    for target in ['core','shortcut']:
     q=A[(A['mode']=='background')&(A.train_corr==.95)&(A.target==target)][['epoch','mean','min','max']].rename(columns={k:f'{target}_probe_{k}' for k in ['mean','min','max']})
     ep=ep.merge(q,on='epoch',validate='one_to_one')
    ep.to_csv(O/'anomaly_combined.csv',index=False)
else:
    ep=pd.read_csv(O/'anomaly_combined.csv')
C={.95:'#c44e52',.8:'#dd9222',.65:'#2878b5'}
def save(fig,name):fig.savefig(FIG/(name+'.png'),dpi=180);fig.savefig(FIG/(name+'.pdf'));plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
for i,mode in enumerate(['background','foreground']):
 for j,target in enumerate(['core','shortcut']):
  ax=axs[i,j]
  for corr in [.95,.8,.65]:
   q=A[(A['mode']==mode)&(A.train_corr==corr)&(A.target==target)]
   ax.plot(q.epoch,q['mean']*100,color=C[corr],label=f'corr={corr:.2f}')
   ax.fill_between(q.epoch,q['min']*100,q['max']*100,color=C[corr],alpha=.15)
  ax.set(title=f'{mode}: {target} linear decodability',xlabel='Epoch',ylabel='Held-out accuracy (%)',ylim=(0,101));ax.legend();ax.grid(alpha=.25)
fig.suptitle('Final pooled features / ridge alpha=1 / 3 fixed balanced partitions; bands=min-max, not CI')
save(fig,'01_probe_epoch_curves')
fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for ax,mode in zip(axs,['background','foreground']):
 for corr in [.95,.8,.65]:
  q=A[(A['mode']==mode)&(A.train_corr==corr)&(A.target=='core')&(A.epoch<=5)]
  ax.plot(q.epoch,q['mean']*100,'o-',color=C[corr],label=f'corr={corr:.2f}')
  ax.fill_between(q.epoch,q['min']*100,q['max']*100,color=C[corr],alpha=.15)
 ax.axhline(90,color='gray',ls=':');ax.axhline(95,color='gray',ls='--');ax.set(title=mode,xlabel='Epoch',ylabel='Core probe held-out accuracy (%)',ylim=(75,101));ax.legend();ax.grid(alpha=.25)
fig.suptitle('Early linear decodability; epoch 0 uses untrained model features')
save(fig,'02_early_core_decodability')
LE=[0,1,2,3,5,10,20,27,28,29,46,47,48,49,50]
LAYERS=['features.3','features.7','features.14','features.21','avgpool']
for target in ['core','shortcut']:
 fig,axs=plt.subplots(2,3,figsize=(16,7),layout='constrained')
 for i,mode in enumerate(['background','foreground']):
  for j,corr in enumerate([.95,.8,.65]):
   q=curves[(curves['mode']==mode)&(curves.train_corr==corr)&(curves.target==target)&curves.epoch.isin(LE)]
   mat=q.pivot(index='layer',columns='epoch',values='mean').reindex(index=LAYERS,columns=LE)*100
   im=axs[i,j].imshow(mat,vmin=10,vmax=100,aspect='auto',interpolation='nearest',cmap='viridis')
   axs[i,j].set(title=f'{mode} corr={corr:.2f}',xlabel='Observed epoch');axs[i,j].set_xticks(range(len(LE)),LE,rotation=60);axs[i,j].set_yticks(range(len(LAYERS)),LAYERS)
 fig.colorbar(im,ax=axs,label='Mean held-out accuracy (%)',shrink=.8);fig.suptitle(f'{target} layer readouts; feature.28 equals avgpool (duplicate omitted)')
 save(fig,'03_layers_'+target)
fig,axs=plt.subplots(3,2,figsize=(13,10),layout='constrained')
for j,window in enumerate([(25,30),(44,50)]):
 q=ep[ep.epoch.between(*window)]
 for col,label in [('val_acc','Val accuracy'),('val_wga','WGA'),('val_cf_all_acc','All-color CF accuracy')]:axs[0,j].plot(q.epoch,q[col]*100,'o-',label=label)
 for col,label in [('val_cf_flip_rate','Paired flip'),('val_cf_all_flip_rate','All-color flip')]:axs[1,j].plot(q.epoch,q[col]*100,'o-',label=label)
 for target in ['core','shortcut']:
  axs[2,j].plot(q.epoch,q[target+'_probe_mean']*100,'o-',label=target+' probe')
  axs[2,j].fill_between(q.epoch,q[target+'_probe_min']*100,q[target+'_probe_max']*100,alpha=.15)
 for ax in axs[:,j]:ax.axvline(28 if j==0 else 49,color='gray',ls=':');ax.set(xlabel='Epoch',ylabel='Percent (%)');ax.legend(fontsize=8);ax.grid(alpha=.25)
 axs[0,j].set_title('Background corr=.95')
fig.suptitle('Behavior setbacks vs preserved linear readouts; probe bands are partition min-max')
save(fig,'04_anomaly_windows')
G=pd.read_csv(O/'group_event_deltas.csv');fig,axs=plt.subplots(1,2,figsize=(12,6),layout='constrained')
for ax,e0,e1 in [(axs[0],27,28),(axs[1],48,49)]:
 q=G[(G['mode']=='background')&(G.train_corr==.95)&(G.group_level=='merged20')&(G.from_epoch==e0)&(G.to_epoch==e1)].sort_values('net_additional_errors')
 ax.barh(q.group,q.net_additional_errors,color=['#c44e52' if v>0 else '#2878b5' for v in q.net_additional_errors]);ax.axvline(0,color='black',lw=.7)
 ax.set(title=f'BG .95 e{e0} -> e{e1}',xlabel='Net additional errors (count)');ax.grid(axis='x',alpha=.25)
fig.suptitle('Group concentration: positive=net correct-count loss; individual prediction transitions unavailable')
save(fig,'05_group_error_changes')
fig,axs=plt.subplots(2,3,figsize=(15,7.5),layout='constrained')
for i,mode in enumerate(['background','foreground']):
 for j,corr in enumerate([.95,.8,.65]):
  b=behavior[(behavior['mode']==mode)&(behavior.train_corr==corr)];p=A[(A['mode']==mode)&(A.train_corr==corr)&(A.target=='core')]
  axs[i,j].plot(b.epoch,b.val_cf_all_flip_rate*100,label='All-color flip');axs[i,j].plot(b.epoch,b.val_cf_all_acc*100,label='All-color accuracy');axs[i,j].plot(p.epoch,p['mean']*100,label='Core probe mean')
  axs[i,j].set(title=f'{mode} corr={corr:.2f}',xlabel='Epoch',ylabel='Percent (%)',ylim=(0,102));axs[i,j].legend(fontsize=7);axs[i,j].grid(alpha=.25)
fig.suptitle('Representation decodability and decision robustness are different observables')
save(fig,'06_probe_vs_behavior')
status=json.loads((O/'progress.json').read_text());status.update(stage='summaries and figures complete',status='awaiting report');(O/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
print(pd.DataFrame(summary)[['mode','train_corr','core_90_first_epoch','core_95_first_epoch','shortcut_95_first_epoch','behavior95_first_epoch','core_e0_mean','core_e50_mean','shortcut_e50_mean']].to_string(index=False))
if not FIGURES_ONLY: print(pd.DataFrame(bootstrap).to_string(index=False))
