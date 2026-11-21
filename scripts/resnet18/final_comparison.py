"""Integrate verified experiment outputs without model inference."""
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT/'runs'
BASE = RUNS/'resnet18_comparison'
OUT = ROOT/'reports/final_integrated_comparison'
ARCHS = ['vgg11_bn','resnet18']
KEY = ['dataset','condition','architecture','epoch']
LOG_JOBS = {'colored_mnist':221986, 'waterbirds':221861}
EARLY = (1,5)
MID = (6,25)
LATE = (26,50)
METRICS = ['acc','wga','flip_rate','c2i_all','c2i_given_correct','i2c_rate']


def read(path):
    return json.loads(path.read_text())


def source_dirs(dataset):
    yield BASE/dataset/'analysis_comparison_e000_e050', 'common'
    if dataset=='waterbirds':
        for p in sorted((BASE/dataset).glob('A_corr*/full/resnet18_e051_e100')):
            yield p,'resnet_late'


def tagged(frame,dataset,scope):
    return frame.assign(dataset=dataset,comparison_scope=scope)


def load():
    tables={k:[] for k in ('accuracy','counterfactual','probes','cam_integrated','timing')}
    sources={}
    for dataset in LOG_JOBS:
        for path,scope in source_dirs(dataset):
            names={k:k for k in tables if k!='timing'}
            names['timing']='transition_candidates' if dataset=='colored_mnist' else 'core_timing'
            for name,file in names.items():
                source=path/f'{file}.csv'
                sources[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
                t=pd.read_csv(source)
                if name=='timing' and dataset=='colored_mnist':
                    t=t.assign(duration=3)
                if name=='timing' and dataset=='waterbirds':
                    # Waterbirds first_epoch in duration=3 rows is the start of the sustained run.
                    single=t[t.duration.eq(1)].drop(columns=['confirmation_epoch','at_E0','last_epoch','duration'])
                    join=['condition','architecture','metric','layer','threshold','direction']
                    t=t.merge(single.rename(columns={'first_epoch':'single_first_epoch'}),on=join,validate='many_to_one')
                    t=t.rename(columns={'first_epoch':'sustained_start','single_first_epoch':'first_epoch'})
                if name=='counterfactual' and dataset=='waterbirds':
                    t=t.rename(columns={'c2i':'c2i_count','i2c':'i2c_count','flips':'prediction_changes',
                                        'c2i_unconditional':'c2i_all','i2c_unconditional':'i2c_rate'})
                    t=t.assign(intervention='opposite_background',wrong_to_wrong_change=0)
                if name=='accuracy' and dataset=='waterbirds':
                    t=t.assign(scope='frozen')
                    online=t[t.online_train_accuracy.notna()].copy()
                    online=online.assign(scope='online',acc=online.online_train_accuracy,wga=np.nan)
                    t=pd.concat([t,online],ignore_index=True)
                tables[name].append(tagged(t,dataset,scope))
    return {k:pd.concat(v,ignore_index=True) for k,v in tables.items()},sources


def settings():
    rows,training=[],[]
    for dataset,job in LOG_JOBS.items():
        if dataset=='colored_mnist':
            names=[f'vgg11_{p}_corr{c}_seed42_e50_img224_all' for p in ('bg','fg') for c in ('095','080','065')]
        else:
            names=[f'A_corr{c}_seed42_e50_contentv3' for c in ('095','080','065')]
        for i,name in enumerate(names):
            rn=BASE/dataset/name/'full'
            identity=read(rn/'identity.json')
            spec=identity['spec']
            for arch,run in zip(ARCHS,[Path(spec['vgg_run']),rn]):
                cfg=read(run/'config.json')
                assert cfg['model']==arch
                for field in ('seed','image_size','train_corr','val_corr','test_corr','augment','pretrained',
                              'optimizer','scheduler','batch_size','amp','lr','momentum','weight_decay'):
                    assert cfg[field]==spec['config'][field]
                expected = 100 if dataset=='waterbirds' and arch=='resnet18' else 50
                h=pd.read_csv(run/'history.csv')
                assert list(h.epoch)==list(range(expected+1)) and (run/'TRAIN_SUCCESS').is_file()
                pointer=read(run/'latest.json')
                checkpoint=run/f'checkpoints/epoch_{expected:03d}.pt'
                assert pointer['epoch']==expected and Path(pointer['checkpoint']).resolve()==checkpoint.resolve()
                assert checkpoint.stat().st_size>0
                if 'manifest' in cfg:
                    manifest=Path(cfg['manifest'])
                    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==cfg['manifest_sha256']
                    assert cfg['manifest_sha256']==spec['config']['manifest_sha256']
                cond=name if dataset=='colored_mnist' else name.split('_seed')[0]
                extra={}
                if arch=='resnet18':
                    extra['training_job']=f'{job}_{i}'
                    log=ROOT/f'logs/resnet18_{job}_{i}.out'
                    elapsed=dict((int(e),float(s)) for e,s in re.findall(r'epoch=(\d+) total_seconds=([\d.]+)',log.read_text()))
                    assert set(elapsed)==set(range(expected+1))
                    extra['summed_logged_epoch_seconds']=sum(elapsed.values())
                    # Do not infer optimizer steps from batch sizes or GradScaler counters.
                    h['logged_epoch_seconds']=h.epoch.map(elapsed)
                elif dataset=='waterbirds':
                    extra['training_job']=str(220287+i)
                    elapsed={e:read(run/f'epoch_metrics/epoch_{e:03d}.json')['epoch_seconds']
                             for e in range(expected+1)}
                    extra['summed_logged_epoch_seconds']=sum(elapsed.values())
                    h['logged_epoch_seconds']=h.epoch.map(elapsed)
                else:
                    extra['training_job']=f'153567_{i}'
                sizes=(55000,5000,10000) if dataset=='colored_mnist' else (4795,1199,5794)
                rows.append(dict({**cfg,**extra},condition=cond,train_n=sizes[0],val_n=sizes[1],test_n=sizes[2],
                                 common_end_epoch=50,final_checkpoint=str(checkpoint.relative_to(ROOT)),
                                 final_checkpoint_bytes=checkpoint.stat().st_size))
                training.append(tagged(h.reindex(columns=['epoch','train_loss','train_acc','lr','logged_epoch_seconds'])
                    .assign(condition=cond,architecture=arch),dataset,'full_training_log'))
    return pd.DataFrame(rows),pd.concat(training,ignore_index=True)


def audit(t):
    a,c,p,cam=(t[k] for k in ('accuracy','counterfactual','probes','cam_integrated'))
    for table,keys in [(a,['split','scope']),(c,['split','intervention']),
                       (p,['layer','target','protocol','seed']),(cam,['protocol','side'])]:
        assert not table.duplicated(KEY+keys).any()
        assert table.query("comparison_scope=='common'").epoch.between(0,50).all()
        late=table.query("comparison_scope=='resnet_late'")
        assert late.epoch.between(51,100).all() and set(late.architecture)=={'resnet18'}
    counts=a.query("split=='val' and scope=='frozen'").groupby(['dataset','comparison_scope','architecture']).size()
    assert counts.to_dict()=={('colored_mnist','common','resnet18'):306,('colored_mnist','common','vgg11_bn'):306,
                              ('waterbirds','common','resnet18'):153,('waterbirds','common','vgg11_bn'):153,
                              ('waterbirds','resnet_late','resnet18'):150}
    cf=c[['prediction_changes','c2i_count','i2c_count','wrong_to_wrong_change']]
    assert (cf.prediction_changes==cf.c2i_count+cf.i2c_count+cf.wrong_to_wrong_change).all()
    assert np.allclose(c.prediction_changes/c.n,c.flip_rate) and np.allclose(c.c2i_count/c.n,c.c2i_all)
    original_correct=c.original_correct.combine_first(c.original_correct_n)
    assert original_correct.notna().all()
    expected_conditional=(c.c2i_count/original_correct.where(original_correct.gt(0)))
    assert np.allclose(expected_conditional,c.c2i_given_correct,equal_nan=True)
    assert np.allclose(c.i2c_count/c.n,c.i2c_rate)
    assert np.allclose(p.decoded_correlation,2*p.accuracy-1)
    assert len(cam.query("dataset=='colored_mnist'"))==336 and len(cam.query("dataset=='waterbirds'"))==144
    for dataset,n,m in [('colored_mnist',306,42),('waterbirds',303,21)]:
        v=read(BASE/dataset/'analysis_comparison_e000_e050/validation.json')
        if dataset=='colored_mnist':
            assert v['frozen_validation']['verified']==n and v['frozen_evaluation_missing']==0
            assert v['cam']['completed_bundles']==m
        else:
            assert v['checkpoint_crc_pass']==n and v['gradcam']['verified_batches']==m
            assert not v['final_analysis']['required_missing']
    for frame,keys,columns in [
        (a.query("scope=='frozen'"),['split'],['n']),
        (c,['split','intervention'],['n'])]:
        common=frame.query("comparison_scope=='common'")
        indices=['dataset','condition','epoch',*keys]
        lhs=common[common.architecture.eq('resnet18')].set_index(indices)[columns]
        rhs=common[common.architecture.eq('vgg11_bn')].set_index(indices)[columns]
        matched=lhs.index.intersection(rhs.index)
        assert len(matched)>0 and np.array_equal(lhs.loc[matched].to_numpy(),rhs.loc[matched].to_numpy())


def behavior(t):
    a=t['accuracy'].query("split=='val' and scope=='frozen'")
    c=t['counterfactual'].query("split=='val' and intervention in ['all9','opposite_background']")
    b=a.merge(c[KEY+['intervention','flip_rate','c2i_all','c2i_given_correct','i2c_rate']],on=KEY,validate='one_to_one')
    assert len(b)==len(a)
    return b


def excursions(series, higher_better):
    # Deterioration relative to the best earlier epoch and first recovery to that level.
    favorable=series if higher_better else -series
    drop=favorable.cummax()-favorable
    low=int(drop.idxmax())
    peak=int(favorable.loc[:low].idxmax())
    recovery=favorable.loc[low+1:]
    rec=recovery[recovery.ge(favorable.loc[peak]-1e-12)]
    return (peak,low,float(drop.loc[low])*100,int(rec.index[0]) if len(rec) else None)


def h1(b, probes):
    rows,phases,late_rows=[],[],[]
    primary=['wga','flip_rate','c2i_all','c2i_given_correct']
    for (dataset,cond,arch), frame in b.groupby(KEY[:3]):
        d=frame.set_index('epoch').sort_index()
        meta=dict(dataset=dataset,condition=cond,architecture=arch,intervention=d.intervention.iloc[0])
        old=d.loc[1:50]
        for start,end in [EARLY,MID,LATE,(46,50),(51,75),(76,100)]:
            block=d.loc[start:end]
            if not len(block):
                continue
            phases.append(dict(**meta,start=start,end=end,**block[METRICS].mean().to_dict()))
        before=old.loc[1:5,METRICS].mean()
        after=old.loc[26:50,METRICS].mean()
        delta=100*(after-before)
        oriented=np.array([delta.wga,-delta.flip_rate,-delta.c2i_all,-delta.c2i_given_correct])
        # Read all signs for the operational judgement; report magnitudes separately.
        verdict='지지' if (oriented>0).all() else '기각' if (oriented<0).all() else '부분 지지' if (oriented>0).any() else '판단 보류'
        entry=dict(**meta,verdict=verdict,**{f'delta_{k}_pp':delta[k] for k in METRICS})
        alt=100*(old.loc[31:50,METRICS].mean()-old.loc[1:10,METRICS].mean())
        entry.update({f'alternative_delta_{k}_pp':alt[k] for k in METRICS})
        for k in primary:
            entry['spearman_'+k]=old.index.to_series().rank().corr(old[k].rank()) if old[k].nunique()>1 else np.nan
            peak,low,gap,rec=excursions(old[k],k=='wga')
            entry.update({f'{k}_peak_epoch':peak,f'{k}_deterioration_epoch':low,
                          f'{k}_deterioration_pp':gap,f'{k}_recovery_epoch':rec})
        rows.append(entry)
        if 100 not in d.index:
            continue
        for k in METRICS:
            peak,low,gap,rec=excursions(d.loc[51:,k],k in ('acc','wga','i2c_rate'))
            late_rows.append(dict(**meta,metric=k,e50=d.loc[50,k],e100=d.loc[100,k],
                mean_e26_50=d.loc[26:50,k].mean(),mean_e51_100=d.loc[51:100,k].mean(),
                mean_e51_75=d.loc[51:75,k].mean(),mean_e76_100=d.loc[76:100,k].mean(),
                peak_epoch=peak,deterioration_epoch=low,deterioration_pp=gap,recovery_epoch=rec))
        pool=probes.query("dataset=='waterbirds' and condition==@cond and architecture==@arch and layer=='avgpool'")
        for target,q in pool.groupby('target'):
            x=q.set_index('epoch').accuracy.sort_index()
            late_rows.append(dict(**meta,metric=f'probe_{target}',e50=x.loc[50],e100=x.loc[100],
                mean_e26_50=x.loc[26:50].mean(),mean_e51_100=x.loc[51:100].mean(),
                mean_e51_75=x.loc[51:75].mean(),mean_e76_100=x.loc[76:100].mean()))
    return pd.DataFrame(rows),pd.DataFrame(phases),pd.DataFrame(late_rows)


def plot(b):
    for dataset in LOG_JOBS:
        small=b[b.dataset.eq(dataset)&b.epoch.le(50)]
        conds=sorted(small.condition.unique())
        fig,axes=plt.subplots(2 if len(conds)==6 else 1,3,figsize=(13,6 if len(conds)==6 else 3.3),squeeze=False)
        for ax,(cond,g) in zip(axes.flat,small.groupby('condition')):
            for arch,curve in g.groupby('architecture'):
                curve=curve.sort_values('epoch')
                for metric,style in [('wga','-'),('flip_rate','--'),('c2i_given_correct',':')]:
                    ax.plot(curve.epoch,100*curve[metric],style,label=f'{arch} {metric}',alpha=.8)
            ax.set(title=cond.split('_seed')[0],xlabel='Epoch',ylabel='Validation (%)',ylim=(0,100))
        axes.flat[0].legend(fontsize=6,ncol=2)
        fig.tight_layout()
        fig.savefig(OUT/f'{dataset}_architectures.png',dpi=145)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4),sharey=True)
    for ax,(dataset,q) in zip(axes,b[b.epoch.le(50)].groupby('dataset')):
        for (cond,arch),curve in q.groupby(['condition','architecture']):
            curve=curve.set_index('epoch').sort_index()
            baseline=curve.loc[1:5,'flip_rate'].mean()
            ax.plot(curve.loc[1:].index,100*(curve.loc[1:,'flip_rate']-baseline),
                    label=f'{cond.split("_seed")[0]} {arch}',alpha=.75)
        ax.axhline(0,color='black',lw=.6)
        ax.set(title=dataset,xlabel='Epoch',ylabel='Within-condition flip change from E1–E5 (pp)')
        ax.legend(fontsize=5,ncol=2)
    fig.tight_layout()
    fig.savefig(OUT/'dataset_relative_flip_change.png',dpi=145)
    plt.close(fig)
    q=b.query("dataset=='waterbirds' and architecture=='resnet18'")
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,(cond,x) in zip(axes,q.groupby('condition')):
        x=x.sort_values('epoch')
        for k in ('wga','flip_rate','c2i_given_correct'):
            ax.plot(x.epoch,100*x[k],label=k)
        ax.axvline(50,color='black',lw=.7)
        ax.set(title=cond,xlabel='Epoch',ylabel='Validation (%)')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT/'waterbirds_resnet_e000_e100.png',dpi=145)
    plt.close(fig)


def main():
    t,sources=load()
    audit(t)
    config,training=settings()
    b=behavior(t)
    verdicts,phases,late=h1(b,t['probes'])
    OUT.mkdir(exist_ok=False)
    for name,table in dict(**t,settings=config,training=training,h1_conditions=verdicts,
                           phase_means=phases,waterbirds_resnet_late=late).items():
        name='h2_timing' if name=='timing' else name
        table.to_csv(OUT/f'{name}.csv',index=False)
    for name,cols,extra in [('accuracy',['acc','wga','balanced_acc'],['split','scope']),
                            ('counterfactual',['flip_rate','c2i_all','c2i_given_correct','i2c_rate'],['split','intervention']),
                            ('probes',['accuracy','decoded_correlation'],['stage','target','protocol','fit_n','score_n'])]:
        shared=t[name].query("comparison_scope=='common'")
        if name=='probes':
            # Repeated probe splits are averaged descriptively, independent of model seeds.
            shared=shared.dropna(subset=['stage']).groupby(KEY+extra,as_index=False)[cols].mean()
        index=['dataset','condition','epoch',*extra]
        l=shared[shared.architecture.eq('resnet18')].set_index(index)[cols]
        r=shared[shared.architecture.eq('vgg11_bn')].set_index(index)[cols]
        points=l.dropna(how='all').index.intersection(r.dropna(how='all').index)
        l,r=l.loc[points],r.loc[points]
        pd.concat([l.add_prefix('resnet18_'),r.add_prefix('vgg11_bn_'),
                   (l-r).add_prefix('delta_')],axis=1).reset_index().to_csv(OUT/f'{name}_matched.csv',index=False)
    rows=[]
    for keys,curve in t['counterfactual'].query("split=='val'").groupby(
            ['dataset','condition','architecture','intervention']):
        curve=curve.set_index('epoch').sort_index()
        for lo,hi in [EARLY,MID,LATE,(51,75),(76,100)]:
            q=curve.loc[lo:hi]
            if len(q):
                rows.append(dict(zip(['dataset','condition','architecture','intervention'],keys),
                    start=lo,end=hi,**q[['flip_rate','c2i_all','c2i_given_correct','i2c_rate']].mean().to_dict()))
    pd.DataFrame(rows).to_csv(OUT/'cf_phase_means.csv',index=False)
    plot(b)
    (OUT/'validation.json').write_text(json.dumps(dict(status='PASS',rows={k:len(v) for k,v in t.items()},
        configs=len(config),common_epochs=[0,50],waterbirds_resnet_only=[51,100],source_sha256=sources,
        verification=['Unique observation keys','Matched sample denominators','CF count and rate identities',
                      'Probe decoded transformation','Training records, final pointer and manifest consistency',
                      'Frozen/CAM coverage from prior audit JSON'],
        warnings=['Single training seed','Missing VGG per-epoch frozen Train/Test and CMNIST Test CF',
                  'Optimizer step counts not independently logged','Stage matches are approximate',
                  'WB independent Test probe not run']),ensure_ascii=False,indent=2))
    print(OUT,flush=True)


if __name__=='__main__':
    main()
