"""Offline probes on existing frozen Validation snapshots; never loads VGG."""
import argparse,json,hashlib,sys,time,socket
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from evaluate import DecodedCorrelation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
CODES=('095','080','065')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def split_content(meta,frame,ids,seed=42):
    rows=frame.set_index('img_id').loc[ids];parent=list(range(len(rows)));tokens={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(rows.itertuples()):
        paths=['CUB_200_2011/images/'+row.img_filename,'places365/data_256/'+row.land_file,'places365/data_256/'+row.water_file]
        for h in {meta['assets'][p] for p in paths}:
            if h in tokens:parent[find(i)]=find(tokens[h])
            tokens[h]=i
    components=np.array([find(i) for i in range(len(rows))]);groups=np.unique(components);rng=np.random.default_rng(seed);rng.shuffle(groups)
    sets=[]
    for part in (groups[:len(groups)//2],groups[len(groups)//2:]):
        pools=[np.flatnonzero(np.isin(components,part)&(rows.y.to_numpy()==y)&(rows.place.to_numpy()==s)) for y in (0,1) for s in (0,1)]
        n=min(map(len,pools))
        if n<2:raise ValueError('Insufficient independent content components for balanced split')
        sets.append(np.sort(np.concatenate([rng.permutation(p)[:n] for p in pools])))
    assert not set(components[sets[0]])&set(components[sets[1]])
    return sets,rows,components

def probe(x,y,fit,score,seed=42):
    torch.manual_seed(seed);m=DecodedCorrelation(x.shape[1],2)
    m.fit(torch.from_numpy(x[fit]),torch.from_numpy(y[fit]),epochs=100,batch_size=256,lr=.001,weight_decay=.0001)
    result=m.evaluate(torch.from_numpy(x[score]),torch.from_numpy(y[score]),256)
    with torch.no_grad():pred=m(torch.from_numpy(x[score])).argmax(1).numpy()
    result['balanced_accuracy']=float(np.mean([np.mean(pred[y[score]==k]==k) for k in (0,1)]))
    return result

def transitions(d):
    result=[]
    for (c,layer),b in d[d.target=='core'].groupby(['condition','layer']):
        b=b.sort_values('epoch')
        for threshold in (.55,.6,.7,.8):
            for duration in (1,3,5,10):
                hits=b.accuracy.to_numpy()>=threshold;start=None
                for i in range(len(b)-duration+1):
                    if hits[i:i+duration].all():start=int(b.epoch.iloc[i]);break
                result.append(dict(condition=c,layer=layer,threshold=threshold,duration=duration,first_epoch=start,confirmation_epoch=None if start is None else start+duration-1,status='not_observed_through_50' if start is None else 'observed',post_hoc=True))
    return pd.DataFrame(result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);out.mkdir(parents=True);(out/'figures').mkdir();(out/'gradcam').mkdir()
    manifest=ROOT/'reports/waterbirds_vgg50/dataset_moana_u2_content_v3/protocol.json';meta=json.loads(manifest.read_text());before={};audit=[];reference=None;records=[];controls=[];behavior=[];started=time.time()
    for code in CODES:
        run=ROOT/f'runs/waterbirds_vgg50/A_corr{code}_seed42_e50_contentv3'
        assert (run/'TRAIN_SUCCESS').exists()
        cfg=json.loads((run/'config.json').read_text());assert cfg['manifest_sha256']==sha(manifest)
        framepath=manifest.parent/f'A_corr{code}.csv';assert sha(framepath)==meta['files'][framepath.name]['sha256']
        frame=pd.read_csv(framepath);val=frame[frame.split=='val'];base=np.load(run/'features/epoch_000.npz')
        identity=[base[k].copy() for k in ('indices','labels','spurious','layers')]
        if reference is None:
            reference=identity;sets,rows,components=split_content(meta,val,identity[0]);fit,score=sets
            assignment=pd.DataFrame(dict(img_id=identity[0],y=identity[1],place=identity[2],component=components,assignment=['fit' if i in fit else 'score' if i in score else 'excluded_for_balance' for i in range(128)]));assignment.to_csv(out/'probe_split.csv',index=False)
        assert all(np.array_equal(x,y) for x,y in zip(reference,identity))
        assert np.array_equal(val.set_index('img_id').loc[identity[0]].y,identity[1]) and np.array_equal(val.set_index('img_id').loc[identity[0]].place,identity[2])
        for epoch in range(51):
            path=run/f'features/epoch_{epoch:03d}.npz';before[str(path.relative_to(ROOT))]=sha(path)
            f=np.load(path);assert all(np.array_equal(f[k],v) for k,v in zip(('indices','labels','spurious','layers'),reference))
            ck=run/f'checkpoints/epoch_{epoch:03d}.pt';side=json.loads(Path(str(ck)+'.integrity.json').read_text());assert ck.stat().st_size==side['size_bytes']
            ep=run/f'epoch_metrics/epoch_{epoch:03d}.json';logged=json.loads(ep.read_text());assert all(s in logged['splits'] for s in ('train','val','test'))
            for split,metrics in logged['splits'].items():behavior.append(dict(condition='A_corr'+code,epoch=epoch,split=split,**metrics))
            for i,layer in enumerate(f['layers']):
                x=f[f'f{i}'];assert x.shape[0]==128 and np.isfinite(x).all()
                for target,key in [('core','labels'),('spurious','spurious')]:
                    result=probe(x,f[key],fit,score)
                    records.append(dict(condition='A_corr'+code,epoch=epoch,layer=str(layer),target=target,fit_n=len(fit),score_n=len(score),seed=42,evaluation_split='validation_content_holdout',**result))
                    if epoch in (0,50) and str(layer)=='avgpool':
                        shuffled=f[key].copy();shuffled[fit]=np.random.default_rng(42).permutation(shuffled[fit]);r=probe(x,shuffled,fit,score);controls.append(dict(condition='A_corr'+code,epoch=epoch,target=target,**r))
            audit.append(dict(condition=code,epoch=epoch,feature_sha256=before[str(path.relative_to(ROOT))],checkpoint_bytes=ck.stat().st_size,checkpoint_scope='existing sidecar size; binary SHA not reread'))
        print('completed probes',code,flush=True)
    d=pd.DataFrame(records);d.to_csv(out/'probe_summary.csv',index=False);pd.DataFrame(controls).to_csv(out/'permutation_controls.csv',index=False);pd.DataFrame(behavior).to_csv(out/'behavior_all_splits.csv',index=False)
    t=transitions(d);t.to_csv(out/'core_transition_analysis.csv',index=False)
    history=pd.read_csv(ROOT/'reports/waterbirds_vgg50/analysis/merged_history.csv');integrated=d.merge(history,on=['condition','epoch'],validate='many_to_one');integrated.to_csv(out/'integrated_metrics.csv',index=False)
    phases=[];relations=[]
    for (c,l,target),b in integrated.groupby(['condition','layer','target']):
        for phase,lo,hi in [('early',1,10),('middle',11,30),('late',31,50)]:
            subset=b[b.epoch.between(lo,hi)];phases.append(dict(condition=c,layer=l,target=target,phase=phase,probe_acc=subset.accuracy.mean(),decoded=subset.decoded_correlation.mean(),wga=subset.val_wga.mean(),cf_flip=subset.val_cf_flip_rate.mean(),c2i=subset.val_cf_correct_to_incorrect_rate.mean()))
        for metric in ('val_wga','val_cf_flip_rate','val_cf_correct_to_incorrect_rate'):
            for lo in (1,11):
                bb=b[b.epoch>=lo];relations.append(dict(condition=c,layer=l,target=target,start_epoch=lo,behavior_metric=metric,spearman_descriptive=bb.accuracy.rank().corr(bb[metric].rank()),delta_spearman_descriptive=bb.accuracy.diff().rank().corr(bb[metric].diff().rank())))
    pd.DataFrame(phases).to_csv(out/'phase_summary.csv',index=False);pd.DataFrame(relations).to_csv(out/'descriptive_relationships.csv',index=False)
    for target in ('core','spurious'):
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        for ax,layer in zip(axes.flat,reference[3]):
            for c,b in d[(d.target==target)&(d.layer==layer)].groupby('condition'):ax.plot(b.epoch,b.decoded_correlation,label=c)
            ax.set(title=str(layer),xlabel='Epoch',ylabel='2 x accuracy - 1',ylim=(-1,1));ax.axhline(0,color='gray',ls=':');ax.legend(fontsize=7)
        fig.suptitle(target+' decoded correlation | Validation content holdout | seed42');fig.tight_layout();fig.savefig(out/f'figures/decoded_{target}.png',dpi=160);plt.close(fig)
    primary=integrated[integrated.layer=='avgpool']
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for ax,(c,b) in zip(axes,primary.groupby('condition')):
        core=b[b.target=='core'];sp=b[b.target=='spurious'];ax.plot(core.epoch,core.accuracy,label='Core probe Acc');ax.plot(sp.epoch,sp.accuracy,label='Spurious probe Acc');ax.plot(core.epoch,core.val_wga,label='Val WGA');ax.plot(core.epoch,core.val_cf_flip_rate,label='Val CF Flip');ax.set(title=c+' | avgpool',xlabel='Epoch',ylim=(0,1));ax.legend(fontsize=7)
    fig.suptitle('Validation content holdout probes vs full Validation behavior');fig.tight_layout();fig.savefig(out/'figures/probes_behavior.png',dpi=160);plt.close(fig)
    candidates={c:[0,1,10,25,50] for c in ('A_corr'+x for x in CODES)}
    for r in t[(t.layer=='avgpool')&(t.threshold==.7)&(t.duration==3)].itertuples():
        if pd.notna(r.first_epoch):candidates[r.condition]=sorted(set(candidates[r.condition]+[max(0,int(r.first_epoch)-1),int(r.confirmation_epoch)]))
    (out/'gradcam/epoch_plan.json').write_text(json.dumps({'epochs':candidates,'selection':'fixed 0/1/10/25/50 plus post-hoc avgpool Acc>=.70 sustained3 start-1/confirmation; candidate only, not confirmed Core learning','sample_ids':assignment.groupby(['y','place']).head(2).img_id.tolist()},indent=2))
    assert all(sha(ROOT/k)==v for k,v in before.items())
    validation=dict(success=True,host=socket.gethostname(),features=153,checkpoints=153,probe_rows=len(d),layers=reference[3].tolist(),fit_n=len(fit),score_n=len(score),component_disjoint=True,balanced_four_groups=True,identity_fixed_across_epochs_conditions=True,feature_sha_preserved=True,epochs_test_metrics_present=153,gradcam_status='PENDING: source images absent on current node',independent_review_status='WATERBIRDS_INDEPENDENT_REVIEW_KO.md not found in current workspace',seconds=time.time()-started,peak_rss_scope='use /usr/bin/time -v for process RSS')
    (out/'validation_results.json').write_text(json.dumps(validation,indent=2));(out/'input_integrity.json').write_text(json.dumps(dict(features=before,manifest_sha256=sha(manifest),artifacts=audit),indent=2))
    (out/'protocol.json').write_text(json.dumps(dict(probe='reused evaluate.DecodedCorrelation Linear(D,2), AdamW,100epochs,batch256,lr.001,wd.0001,seed42,no feature scaling',primary_layer='avgpool',secondary_layers=reference[3].tolist(),split='Validation internally split by connected original bird/land/water encoded-content hashes; then each half downsampled to equal4groups',difference='Not original run.py analyze val-fit/test-score. No Test feature exists. Cannot compare as same protocol.',thresholds='post-hoc Acc .55/.60/.70/.80; durations1/3/5/10; baseline0 included and reported explicitly'),indent=2))
if __name__=='__main__':main()
