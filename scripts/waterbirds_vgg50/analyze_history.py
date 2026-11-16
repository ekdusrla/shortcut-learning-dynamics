"""CPU-only post-hoc CSV analysis. Never loads checkpoints/features or runs inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

REPO=Path(__file__).resolve().parents[2]
CODES=('095','080','065')
COLORS={'095':'#C44E52','080':'#DD8452','065':'#4C72B0'}
GROUP_NAMES={'g00':'Landbird / land','g01':'Landbird / water','g10':'Waterbird / land','g11':'Waterbird / water'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive(frame,prefix='val_'):
    d=frame.copy();p=prefix
    g={k:d[p+k+'_acc'] for k in GROUP_NAMES};n={k:d[p+k+'_n'] for k in GROUP_NAMES}
    total=sum(n.values());weighted=sum(g[k]*n[k] for k in g)/total
    if not np.allclose(weighted,d[p+'acc'],atol=1e-12,rtol=0):raise ValueError('Weighted accuracy inconsistent with group CSV')
    if not np.allclose(pd.concat(g.values(),axis=1).min(axis=1),d[p+'wga'],atol=1e-12,rtol=0):raise ValueError('WGA inconsistent')
    d[p+'n']=total
    for y in (0,1):
        keys=[f'g{y}{s}' for s in (0,1)]
        d[p+f'class{y}_recall']=sum(g[k]*n[k] for k in keys)/sum(n[k] for k in keys)
    d[p+'balanced_acc']=(d[p+'class0_recall']+d[p+'class1_recall'])/2
    d[p+'macro_group_acc']=sum(g.values())/4
    d[p+'balanced_aligned_acc']=(g['g00']+g['g11'])/2
    d[p+'balanced_conflicting_acc']=(g['g01']+g['g10'])/2
    d[p+'aligned_conflicting_gap']=d[p+'aligned_acc']-d[p+'conflicting_acc']
    d[p+'balanced_alignment_gap']=d[p+'balanced_aligned_acc']-d[p+'balanced_conflicting_acc']
    if p+'cf_original_correct_n' in d:
        if not np.allclose(d[p+'cf_original_correct_n'],d[p+'acc']*total,atol=1e-9,rtol=0):raise ValueError('C2I denominator inconsistent')
        d[p+'c2i_unconditional']=d[p+'acc']*d[p+'cf_correct_to_incorrect_rate']
        d[p+'i2c_unconditional']=d[p+'cf_acc']-d[p+'acc']+d[p+'c2i_unconditional']
        if not np.allclose(d[p+'c2i_unconditional']+d[p+'i2c_unconditional'],d[p+'cf_flip_rate'],atol=1e-12,rtol=0):
            raise ValueError('Binary flip/accuracy identities inconsistent')
    d[p+'worst_group']=pd.concat(g,axis=1).apply(lambda r:'|'.join(k for k,v in r.items() if abs(v-r.min())<1e-12),axis=1)
    return d


def intervals(epochs,mask):
    result=[];start=last=None
    for epoch,hit in zip(epochs,mask):
        if hit:
            if start is None:start=int(epoch)
            elif int(epoch)!=last+1:result.append((start,last));start=int(epoch)
            last=int(epoch)
        elif start is not None:result.append((start,last));start=last=None
    if start is not None:result.append((start,last))
    return result


def inventory_run(run,code):
    expected={f'epoch_{e:03d}' for e in range(51)}
    checkpoints=sorted((run/'checkpoints').glob('epoch_*.pt'));features=sorted((run/'features').glob('epoch_*.npz'))
    if {p.stem for p in checkpoints}!=expected or {p.stem for p in features}!=expected:raise ValueError(f'Incomplete epoch files: {run}')
    if any(p.stat().st_size<=0 for p in checkpoints+features):raise ValueError('Empty artifact')
    history=run/'history.csv';frame=pd.read_csv(history)
    if frame.epoch.tolist()!=list(range(51)) or len(history.read_text().splitlines())!=52:raise ValueError('History epochs/lines invalid')
    if not (run/'TRAIN_SUCCESS').is_file():raise ValueError('TRAIN_SUCCESS missing')
    cfg=json.loads((run/'config.json').read_text());completion=json.loads((run/'completion_integrity.json').read_text())
    if not completion.get('verified') or completion['checkpoints']!=51 or completion['manifest_sha256']!=cfg['manifest_sha256']:raise ValueError('Completion metadata invalid')
    for p in checkpoints:
        r=json.loads(Path(str(p)+'.integrity.json').read_text())
        if r['size_bytes']!=p.stat().st_size or Path(r['path']).resolve()!=p.resolve() or r['manifest_sha256']!=cfg['manifest_sha256']:
            raise ValueError('Checkpoint sidecar size/path/manifest mismatch')
        epoch=json.loads((run/'epoch_metrics'/f'{p.stem}.json').read_text())
        if epoch['checkpoint']!=r:raise ValueError('Epoch/sidecar linkage differs')
    for k,v in dict(model='vgg11_bn',pretrained=False,epochs=50,seed=42,image_size=224,batch_size=128,lr=.01,
                    momentum=.9,weight_decay=.0001,scheduler='none',amp=True,train_corr=int(code)/100,val_corr=.5,test_corr=.5).items():
        if cfg.get(k)!=v:raise ValueError(f'Unexpected config {k}')
    if not np.allclose(frame.lr,.01):raise ValueError('LR changed')
    frame['condition']='A_corr'+code;frame['corr']=int(code)/100;frame['run']=str(run.relative_to(REPO))
    derived=derive(frame)
    # Cross-check validation CSV against existing epoch JSON, without new inference.
    for e in range(51):
        logged=json.loads((run/'epoch_metrics'/f'epoch_{e:03d}.json').read_text())['splits']['val']
        for key in ('acc','wga','aligned_acc','conflicting_acc','cf_flip_rate','cf_correct_to_incorrect_rate'):
            if not math.isclose(float(derived.loc[e,'val_'+key]),logged[key],abs_tol=1e-12,rel_tol=0):raise ValueError('CSV/epoch JSON mismatch')
    record=dict(condition='A_corr'+code,checkpoints=len(checkpoints),features=len(features),history_data_rows=len(frame),history_lines=52,
        epochs='0..50',train_success=True,completion_verified_record=True,sidecar_linkage='PASS',manifest_sha256=cfg['manifest_sha256'],
        checkpoint_verification_scope='Names/counts/nonzero sizes/sidecars/completion record; binary SHA and tensors not re-read')
    return derived,record,cfg


def tables(data,out):
    final=data[data.epoch==50].copy()
    columns=['condition','corr','train_acc','val_acc','val_balanced_acc','val_macro_group_acc','val_class0_recall','val_class1_recall',
        'val_aligned_acc','val_conflicting_acc','val_balanced_conflicting_acc','val_wga','val_cf_flip_rate','val_cf_correct_to_incorrect_rate',
        'val_c2i_unconditional','val_cf_acc','val_cf_acc_drop','val_aligned_conflicting_gap']+[f'val_{g}_acc' for g in GROUP_NAMES]
    final[columns].to_csv(out/'final_comparison.csv',index=False)
    degenerate=(((data.val_class0_recall==1)&(data.val_class1_recall==0))|((data.val_class0_recall==0)&(data.val_class1_recall==1)))&(data.val_cf_flip_rate==0)
    data.loc[degenerate,['condition','epoch','val_acc','val_balanced_acc','val_class0_recall','val_class1_recall','val_cf_flip_rate','val_wga']].to_csv(out/'degenerate_epoch_flags.csv',index=False)
    phase_rows=[]
    for condition,d in data.groupby('condition',sort=False):
        for phase,lo,hi in [('baseline',0,0),('early',1,10),('middle',11,30),('late',31,50),('last10',41,50)]:
            block=d[d.epoch.between(lo,hi)]
            for metric in ['train_acc','val_acc','val_balanced_acc','val_class1_recall','val_wga','val_conflicting_acc',
                'val_balanced_conflicting_acc','val_cf_flip_rate','val_cf_correct_to_incorrect_rate','val_balanced_alignment_gap']:
                values=block[metric].dropna()
                statistics={k:np.nan for k in ('mean','median','min','max','std')} if values.empty else dict(mean=values.mean(),median=values.median(),min=values.min(),max=values.max(),std=values.std(ddof=0))
                phase_rows.append(dict(condition=condition,phase=phase,start=lo,end=hi,metric=metric,**statistics))
    pd.DataFrame(phase_rows).to_csv(out/'phase_summary.csv',index=False)
    peaks=[];rolling=[];runs=[];proxies=[];trends=[];spikes=[]
    for condition,d in data.groupby('condition',sort=False):
        d=d.set_index('epoch');active=d.loc[1:];e=int(active.val_wga.idxmax());peak=active.loc[e]
        future=active.loc[e+1:min(50,e+3),'val_wga']
        peaks.append(dict(condition=condition,peak_epoch=e,peak_wga=peak.val_wga,next_epoch_wga=d.loc[e+1,'val_wga'] if e<50 else np.nan,
            next3_mean=future.mean(),final_wga=d.loc[50,'val_wga'],last10_mean=d.loc[41:,'val_wga'].mean()))
        for window in (3,5,10):
            mean=active.val_wga.rolling(window).mean();floor=active.val_wga.rolling(window).min()
            for measure,series in [('mean',mean),('floor',floor)]:
                end=int(series.idxmax());block=active.loc[end-window+1:end,'val_wga']
                rolling.append(dict(condition=condition,window=window,criterion=measure,start=end-window+1,end=end,
                    score=series.loc[end],mean=block.mean(),min=block.min(),max=block.max()))
        for threshold in (.2,.3,.4,.5):
            for start,end in intervals(active.index,active.val_wga>=threshold):
                runs.append(dict(condition=condition,threshold=threshold,start=start,end=end,length=end-start+1,sustained5=end-start+1>=5))
        for threshold in (.2,.3):
            for duration in (3,5,10):
                hits=intervals(active.index,(active.val_wga>=threshold)&(active.val_balanced_conflicting_acc>=.5))
                valid=[(a,b) for a,b in hits if b-a+1>=duration]
                proxies.append(dict(condition=condition,wga_threshold=threshold,balanced_conflict_threshold=.5,duration=duration,
                    first_start=valid[0][0] if valid else np.nan,first_confirmation=valid[0][0]+duration-1 if valid else np.nan,
                    status='observed' if valid else 'not_observed_through_50'))
        for e in range(4,48):
            value=active.loc[e,'val_wga'];previous=active.loc[e-3:e-1,'val_wga'].median();following=active.loc[e+1:e+3,'val_wga'].mean()
            if value-previous>=.1 and value-following>=.1:
                spikes.append(dict(condition=condition,epoch=e,wga=value,previous3_median=previous,next3_mean=following))
        for metric in ('val_cf_flip_rate','val_cf_correct_to_incorrect_rate','val_wga','val_balanced_conflicting_acc'):
            for start in (1,11):
                series=active.loc[start:,metric]
                trends.append(dict(condition=condition,metric=metric,start=start,end=50,
                    spearman_descriptive=pd.Series(series.index,index=series.index).rank().corr(series.rank()),
                    linear_slope_per_epoch=np.polyfit(series.index,series.to_numpy(),1)[0]))
    for name,rows in [('wga_peaks',peaks),('wga_rolling_windows',rolling),('wga_threshold_runs',runs),
                      ('core_proxy_thresholds',proxies),('transient_wga_spikes',spikes),('descriptive_trends',trends)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    return final,peaks,rolling,proxies


def plots(data,out):
    figures=out/'figures';figures.mkdir()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120})
    def series(ax,key,title,baseline=None):
        for code in CODES:
            d=data[data.condition=='A_corr'+code]
            ax.plot(d.epoch,d[key],color=COLORS[code],label=f'corr={int(code)/100:.2f}',lw=1.6)
        if baseline is not None:ax.axhline(baseline,color='gray',ls=':',lw=1,label='Reference baseline')
        ax.set(title=title,xlabel='Epoch',xlim=(0,50),ylim=(0,1.02));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2)
    def save(fig,name):
        fig.tight_layout();fig.savefig(figures/(name+'.png'),dpi=200,bbox_inches='tight');fig.savefig(figures/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    series(axes[0],'train_acc','Online train accuracy (epoch 0 undefined)')
    d=data.iloc[0];prior=(d.val_g00_n+d.val_g01_n)/d.val_n
    series(axes[1],'val_acc','Validation accuracy',prior);axes[0].legend();axes[1].legend(fontsize=8)
    save(fig,'accuracy')
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,key,title in zip(axes.flat,['val_aligned_acc','val_conflicting_acc','val_wga','val_cf_flip_rate','val_cf_correct_to_incorrect_rate','val_aligned_conflicting_gap'],
                           ['Aligned accuracy','Conflicting accuracy','Worst-group accuracy','Counterfactual flip rate','Correct-to-incorrect / originally correct','Aligned minus conflicting accuracy']):
        series(ax,key,title)
    axes.flat[0].legend();save(fig,'shortcut_metrics')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,g in zip(axes.flat,GROUP_NAMES):series(ax,'val_'+g+'_acc',GROUP_NAMES[g])
    axes.flat[0].legend();save(fig,'group_accuracy')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,key,title in zip(axes.flat,['val_balanced_acc','val_class0_recall','val_class1_recall','val_balanced_conflicting_acc'],
                           ['Class-balanced accuracy','Landbird recall','Waterbird recall','Class-balanced conflicting accuracy']):
        series(ax,key,title,.5)
    axes.flat[0].legend(fontsize=8);save(fig,'class_imbalance')
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    for ax,code in zip(axes,CODES):
        d=data[data.condition=='A_corr'+code].set_index('epoch');active=d.loc[1:]
        ax.plot(d.index,d.val_wga,color=COLORS[code],alpha=.5,lw=1,label='Raw WGA')
        ax.plot(active.index,active.val_wga.rolling(5).mean(),color='black',label='Trailing 5-epoch mean')
        ax.plot(active.index,active.val_wga.rolling(5).min(),color=COLORS[code],ls='--',label='Trailing 5-epoch floor')
        e=int(active.val_wga.idxmax());ax.scatter([e],[active.loc[e,'val_wga']],marker='*',s=90,color='black')
        ax.set(title=f'corr={int(code)/100:.2f}',xlabel='Epoch',xlim=(0,50),ylim=(0,.65));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);save(fig,'wga_stability')


def analyze(run_root,output):
    root=Path(run_root).resolve();out=Path(output).resolve()
    if any(out.is_relative_to(root/f'A_corr{c}_seed42_e50_contentv3') for c in CODES):raise ValueError('Analysis output must not be inside training runs')
    if out.exists():raise FileExistsError(out)
    inputs={};preservation={};frames=[];checks=[];configs=[];tests=[]
    for code in CODES:
        run=root/f'A_corr{code}_seed42_e50_contentv3'
        for p in run.rglob('*'):
            if p.is_file():
                stat=p.stat();preservation[str(p)]=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
                if p.suffix in ('.csv','.json'):
                    inputs[str(p.relative_to(REPO))]=sha(p)
        frame,check,cfg=inventory_run(run,code);frames.append(frame);checks.append(check);configs.append(cfg)
        # Existing held-out final log only; no new test inference/epoch selection.
        final_test=json.loads((run/'epoch_metrics/epoch_050.json').read_text())['splits']['test']
        t=derive(pd.DataFrame([{'test_'+k:v for k,v in final_test.items()}]),prefix='test_')
        t.insert(0,'condition','A_corr'+code);tests.append(t)
    if len({c['manifest_sha256'] for c in configs})!=1:raise ValueError('Runs use different manifests')
    data=pd.concat(frames,ignore_index=True)
    if len(data)!=153:raise ValueError('Expected 153 epoch rows')
    for col in ['val_g00_n','val_g01_n','val_g10_n','val_g11_n']:
        if data[col].nunique()!=1:raise ValueError('Evaluation group counts changed')
    if any(not np.allclose(f.loc[0,[k for k in f if k.startswith('val_') and f[k].dtype.kind in 'fi']].astype(float),
                           frames[0].loc[0,[k for k in f if k.startswith('val_') and f[k].dtype.kind in 'fi']].astype(float),equal_nan=True) for f in frames[1:]):
        raise ValueError('Epoch0 differs between conditions')
    out.mkdir(parents=True)
    data.to_csv(out/'merged_history.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'run_verification.csv',index=False)
    pd.concat(tests,ignore_index=True).to_csv(out/'final_test_from_existing_logs.csv',index=False)
    tables(data,out);plots(data,out)
    (out/'provenance.json').write_text(json.dumps(dict(input_sha256=inputs,run_verification=checks,
        scope='CSV analysis plus existing final test/verification JSON; no checkpoint/feature contents loaded',
        phase_definition=dict(baseline=[0,0],early=[1,10],middle=[11,30],late=[31,50],last10=[41,50]),
        descriptive_rules=dict(sustain='Consecutive epochs, no smoothing for threshold runs',windows=[3,5,10],
            thresholds=[.2,.3,.4,.5],spike='At least .10 above previous3 median and next3 mean; epochs4..47'),
        hypothesis_scope='Single seed42, post-hoc descriptive analysis; no iid p-values or model/epoch selection'),indent=2))
    changes=[]
    for name,record in preservation.items():
        p=Path(name)
        if not p.exists() or p.stat().st_size!=record['size'] or p.stat().st_mtime_ns!=record['mtime_ns']:changes.append(name)
    for name,expected in inputs.items():
        if sha(REPO/name)!=expected:changes.append(name)
    (out/'source_preservation.json').write_text(json.dumps(dict(checked_files=len(preservation),changed=changes,
        checkpoint_feature_policy='Stat size+mtime only; content SHA not reread',csv_json_policy='SHA256 before/after'),indent=2))
    if changes:raise RuntimeError('Source artifacts changed during analysis')
    print(json.dumps(dict(output=str(out),rows=len(data),runs=checks),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=REPO/'runs/waterbirds_vgg50')
    p.add_argument('--output',type=Path,default=REPO/'reports/waterbirds_vgg50/analysis')
    args=p.parse_args();analyze(args.runs,args.output)

if __name__=='__main__':main()
