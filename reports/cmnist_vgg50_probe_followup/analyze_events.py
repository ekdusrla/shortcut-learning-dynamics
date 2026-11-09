"""Only new trajectory/event summaries; do not rerun earlier archive/behavior analysis."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
O=Path(__file__).resolve().parent;ROOT=O.parents[1];OLD=ROOT/'reports/cmnist_vgg50_analysis_review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((OLD/'source_manifest.json').read_text())
for item in manifest:assert sha(ROOT/item['path'])==item['sha256']
old=pd.read_csv(OLD/'epoch_metrics.csv');events=[];groups=[];traj=[];source=[]
for mode in ['background','foreground']:
 for corr in [.95,.8,.65]:
  p=ROOT/f'runs/vgg11_{"bg" if mode=="background" else "fg"}_corr{round(corr*100):03d}_seed42_e50_img224_all'
  d=pd.read_csv(p/'history.csv');cached=old[(old['mode']==mode)&(old.train_corr==corr)]
  assert d.epoch.tolist()==cached.epoch.tolist()
  for c in d:assert np.allclose(d[c],cached[c],equal_nan=True),c
  for f in ['config.json','history.csv','test_metrics.json']:
   source.append({'path':str((p/f).relative_to(ROOT)),'sha256':sha(p/f)})
  for e in [25,26,27,28,29,30,44,45,46,47,48,49,50]:
   r=d[d.epoch==e].iloc[0]
   events.append({'mode':mode,'train_corr':corr,'epoch':e,**{c:float(r[c]) for c in ['train_acc','val_acc','val_aligned_acc','val_conflicting_acc','val_wga','val_cf_flip_rate','val_cf_acc','val_cf_acc_drop','val_cf_all_flip_rate','val_cf_all_acc','val_cf_all_color_rate']}})
  trained=d[d.epoch>=1];flip=trained.val_cf_all_flip_rate.to_numpy();acc=trained.val_cf_all_acc.to_numpy()
  traj.append(dict(mode=mode,train_corr=corr,flip_e1=float(flip[0]),flip_e50=float(flip[-1]),
   flip_increase_steps=int((np.diff(flip)>0).sum()),largest_flip_increase_pp=float(100*np.diff(flip).max()),
   simultaneous_flip_increase_and_CF_acc_decrease_steps=int(((np.diff(flip)>0)&(np.diff(acc)<0)).sum()),
   spearman_epoch_flip=float(spearmanr(trained.epoch,flip).statistic),
   late_OLS_flip_slope_pp_per_epoch=float(np.polyfit(trained.loc[trained.epoch>=26,'epoch'],100*trained.loc[trained.epoch>=26,'val_cf_all_flip_rate'],1)[0])))
  for e0,e1 in [(27,28),(28,29),(48,49),(49,50)]:
   r0=d[d.epoch==e0].iloc[0];r1=d[d.epoch==e1].iloc[0]
   total_loss=int(round(5000*(r0.val_acc-r1.val_acc)))
   for merge in [False,True]:
    cells=[(f'g{y}{c}',y,'aligned' if y==c else 'conflicting') for y in range(10) for c in range(10)] if not merge else [(f'g{y}{y}',y,'aligned') for y in range(10)]+[(f'g{y}other',y,'conflicting') for y in range(10)]
    correct_delta=0
    for key,y,kind in cells:
     n=int(r0[f'val_{key}_n']);assert n==int(r1[f'val_{key}_n'])
     c0=int(round(n*r0[f'val_{key}_acc']));c1=int(round(n*r1[f'val_{key}_acc']))
     lost=c0-c1;correct_delta+=lost
     groups.append(dict(mode=mode,train_corr=corr,from_epoch=e0,to_epoch=e1,group_level='merged20' if merge else 'cell100',group=key,digit=y,kind=kind,n=n,correct_before=c0,correct_after=c1,
       acc_before=float(r0[f'val_{key}_acc']),acc_after=float(r1[f'val_{key}_acc']),net_additional_errors=lost,total_net_additional_errors=total_loss,
       fraction_of_net_change=lost/total_loss if total_loss else None))
    assert correct_delta==total_loss
pd.DataFrame(events).to_csv(O/'event_metrics.csv',index=False);pd.DataFrame(groups).to_csv(O/'group_event_deltas.csv',index=False)
pd.DataFrame(traj).to_csv(O/'trajectory_nonmonotonicity.csv',index=False)
review={'success':True,'raw_history_matches_previous_derived_table':True,'original_manifest_hashes_match':True,
 'C2I':'Previous bounds are algebraically valid for common-denominator paired CF; not identifiable in general for 10 classes; all-color accuracy uses different denominator.',
 'H1':'Long-run descriptive support, not monotone or a seed-generalized law.',
 'H2':'Existing criterion measures robustness, not the onset of core representation learning; direct formulation remains untested by behavior alone.',
 'generalization_gap':'Correctly described as online train minus shifted held-out accuracy; not an IID fixed-model gap.',
 'source_files':source,'prior_report_sources':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [OLD/'analysis_report.md',OLD/'summary_table.csv',OLD/'missing_artifacts.md']+list((OLD/'figures').glob('*'))]}
(O/'review_validation.json').write_text(json.dumps(review,indent=2)+'\n')
print('Saved prior-analysis validation, event windows, group net-error decomposition and nonmonotonicity summaries.')
g=pd.DataFrame(groups);b=g[(g['mode']=='background')&(g.train_corr==.95)&(g.from_epoch==27)&(g.to_epoch==28)&(g.group_level=='merged20')]
print(b.sort_values('net_additional_errors',ascending=False).head(6).to_string(index=False))
