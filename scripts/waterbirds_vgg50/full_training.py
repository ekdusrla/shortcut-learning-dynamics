"""50-epoch Condition A support: read-only preflight, capacity and integrity checks."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import shortcut_experiment as exp
from scripts.waterbirds_vgg50.protocol import digest,load,CORRS
from scripts.waterbirds_vgg50.pilot_epochs import inspect_manifest

REPO=Path(__file__).resolve().parents[2]


def configuration(manifest,corr):
    if corr not in CORRS:raise ValueError('Unknown Condition A correlation')
    manifest=Path(manifest).resolve()
    if manifest.is_dir():manifest=manifest/'protocol.json'
    meta=json.loads(manifest.read_text())
    return dict(dataset='waterbirds',condition='A',model='vgg11_bn',pretrained=False,epochs=50,
        seed=42,image_size=224,train_corr=corr,val_corr=.5,test_corr=.5,augment=False,
        optimizer='sgd',lr=.01,momentum=.9,weight_decay=.0001,scheduler='none',batch_size=128,
        num_workers=4,amp=True,save_features=True,features_per_group=32,cf_chunk_size=128,
        manifest=str(manifest),manifest_sha256=digest(manifest),waterbirds_root=meta['root'],
        evaluation_protocol='single_pass_cf_v1',train_eval_shuffle=False)


def config_sha(cfg):return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()


def verify_pilot(path,manifest_sha):
    path=Path(path).resolve()
    if path.is_dir():path=path/'summary.json'
    p=json.loads(path.read_text())
    if not isinstance(p.get('config'),dict):raise ValueError('Full-epoch pilot summary with config required')
    cfg=p['config']
    required=dict(model='vgg11_bn',pretrained=False,lr=.01,momentum=.9,weight_decay=.0001,
        batch_size=128,image_size=224,seed=42,amp=True,num_workers=4,scheduler='none',train_corr=.95,val_corr=.5,test_corr=.5)
    if not p.get('success') or p.get('corr')!=.95 or p.get('manifest_sha256')!=manifest_sha or any(cfg.get(k)!=v for k,v in required.items()):
        raise ValueError('Successful A095 pilot with matching manifest/hyperparameters required')
    if not (path.parent/'PILOT_SUCCESS').is_file():raise ValueError('Pilot completion marker missing')
    records=p['epochs']
    if len(records)<2 or [r['epoch'] for r in records]!=list(range(len(records))):raise ValueError('Pilot epoch sequence invalid')
    for row in records:
        checkpoint=(path.parent/'checkpoints'/f'epoch_{row["epoch"]:03d}.pt').resolve()
        r=row['checkpoint']
        if Path(r['path']).resolve()!=checkpoint or not r['verified'] or checkpoint.stat().st_size!=r['size_bytes'] or digest(checkpoint)!=r['sha256']:
            raise ValueError('Pilot checkpoint integrity mismatch')
    latest=exp.load_checkpoint(records[-1]['checkpoint']['path'],'cpu')
    if latest['epoch']!=records[-1]['epoch'] or latest['config']!=cfg or latest['model']['classifier.weight'].shape[0]!=2:
        raise ValueError('Pilot checkpoint state/config mismatch')
    del latest
    return p


def disk_plan(pilot,root):
    # Measured pilot serialization size; one epoch0 and 50 momentum checkpoints.
    ck0=pilot['epochs'][0]['checkpoint']['size_bytes']
    trained=max(r['checkpoint']['size_bytes'] for r in pilot['epochs'] if r['epoch']>0)
    # Uncompressed 128 samples x sum(64,128,256,512,512,512) x FP32 + labels/NPZ overhead.
    feature_upper=128*1984*4+128*3*8+64*1024
    per_condition=ck0+50*trained+51*feature_upper+16*1024**2
    peak_three=3*(per_condition+trained)
    required=math.ceil((peak_three*1.25+1024**3)/1024**3)*1024**3
    parent=Path(root).resolve()
    while not parent.exists():parent=parent.parent
    free=shutil.disk_usage(parent).free
    return dict(checkpoint_epoch0_bytes=ck0,checkpoint_trained_bytes=trained,checkpoint_count_per_condition=51,
        checkpoint_bytes_per_condition=ck0+50*trained,feature_upper_bytes_per_epoch=feature_upper,
        planned_bytes_per_condition=per_condition,three_condition_peak_bytes=peak_three,
        recommended_free_bytes=required,filesystem_free_bytes=free,enough_space=free>=required,
        scope='Filesystem availability, not a separately verified user quota')


def runtime_estimate(pilot):
    phases=pilot['phases'];n=pilot['sample_counts'];epochs=[r for r in pilot['epochs'] if r['epoch']>0]
    trained=sum(phases[f'epoch_{r["epoch"]:03d}_train']['seconds'] for r in epochs)/len(epochs)
    # Old pilot val metrics used two passes; the new observer halves that work.
    factor=1 if pilot['config'].get('evaluation_protocol')=='single_pass_cf_v1' else .5
    val=sum(phases[f'epoch_{r["epoch"]:03d}_val']['seconds'] for r in epochs)/len(epochs)*factor
    features=sum(phases[f'epoch_{r["epoch"]:03d}_features']['seconds'] for r in epochs)/len(epochs)
    checkpoint=sum(phases[f'epoch_{r["epoch"]:03d}_checkpoint']['seconds'] for r in epochs)/len(epochs)
    test=val*n['test']/n['val'];train_eval=val*n['train']/n['val']/2
    epoch=trained+val+test+train_eval+features+checkpoint
    total=50*epoch+val+test+train_eval+features+checkpoint+phases['source_validation']['seconds']
    return dict(estimated_epoch_seconds=epoch,estimated_50epoch_seconds=total,
        assumptions='Single-pass equivalent metrics; pilot timing version accounted; sample-count scaling; no contention measured',
        pilot_mean_epoch_seconds=sum(r['epoch_seconds'] for r in epochs)/len(epochs))


def preflight(manifest,pilot_path,output_root=REPO/'runs/waterbirds_vgg50'):
    meta=inspect_manifest(manifest);pilot=verify_pilot(pilot_path,meta['manifest_sha256'])
    disk=disk_plan(pilot,output_root)
    if not disk['enough_space']:raise ValueError('Insufficient space for all three 50-epoch conditions')
    return dict(manifest=meta,pilot_path=str(Path(pilot_path).resolve()),pilot_corr=.95,
        pilot_reused_for_conditions=list(CORRS),disk=disk,runtime=runtime_estimate(pilot),
        memory_profile_scope='Only A095 GPU pilot measured; all conditions share batch/image/model and paired population')


def checkpoint_record(path,cfg,epoch):
    saved=exp.load_checkpoint(path,'cpu')
    if saved['epoch']!=epoch or saved['config']!=cfg or saved['checkpoint_boundary']!='after_feature_snapshot':
        raise ValueError('Checkpoint epoch/config/RNG boundary mismatch')
    if saved['model']['classifier.weight'].shape[0]!=2:raise ValueError('Binary head mismatch')
    if epoch and not saved['optimizer']['state']:raise ValueError('Optimizer momentum state missing')
    del saved
    return dict(epoch=epoch,path=str(Path(path).resolve()),sha256=digest(path),size_bytes=Path(path).stat().st_size,
        config_sha256=config_sha(cfg),manifest_sha256=cfg['manifest_sha256'],verified=True)


def save_verified(path,epoch,model,optimizer,scheduler,scaler,cfg,gen):
    from scripts.waterbirds_vgg50.run import save_epoch_checkpoint
    from scripts.waterbirds_vgg50.download import atomic_json
    path=Path(path);sidecar=Path(str(path)+'.integrity.json')
    if path.exists() or sidecar.exists():raise FileExistsError(f'Checkpoint already exists; preserved: {path}')
    save_epoch_checkpoint(path,epoch,model,optimizer,scheduler,scaler,cfg,gen)
    record=checkpoint_record(path,cfg,epoch)
    atomic_json(sidecar,record)
    atomic_json(path.parent.parent/'latest.json',dict(epoch=epoch,checkpoint=record['path'],sha256=record['sha256']))
    return record


def resume_state(out,cfg):
    out=Path(out);pointer=json.loads((out/'latest.json').read_text());epoch=int(pointer['epoch'])
    path=(out/'checkpoints'/f'epoch_{epoch:03d}.pt').resolve()
    if Path(pointer['checkpoint']).resolve()!=path:raise ValueError('Latest path escapes selected run')
    sidecar=json.loads(Path(str(path)+'.integrity.json').read_text())
    if pointer.get('sha256')!=sidecar['sha256'] or digest(path)!=sidecar['sha256'] or sidecar['manifest_sha256']!=cfg['manifest_sha256']:
        raise ValueError('Resume checkpoint SHA/manifest mismatch')
    if checkpoint_record(path,cfg,epoch)!=sidecar:raise ValueError('Checkpoint sidecar differs')
    metrics_path=out/'epoch_metrics'/f'epoch_{epoch:03d}.json'
    if metrics_path.exists():
        row=json.loads(metrics_path.read_text())
        if row.get('epoch')!=epoch:raise ValueError('Epoch metrics boundary differs')
        if row.get('checkpoint') is None:
            from scripts.waterbirds_vgg50.download import atomic_json
            row['checkpoint']=sidecar;row['recovered_checkpoint_commit']=True
            atomic_json(metrics_path,row)
    saved=exp.load_checkpoint(path,'cpu')
    return saved


@contextmanager
def run_lock(out):
    import fcntl
    with (Path(out)/'.training.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another process is training in this run')
        yield


def fit(model,datasets,cfg,out,device,saved=None,end_epoch=None):
    from scripts.waterbirds_vgg50.run import metrics
    from scripts.waterbirds_vgg50.download import atomic_json
    out=Path(out);gen=torch.Generator().manual_seed(819)
    optimizer=torch.optim.SGD(model.parameters(),lr=cfg['lr'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer,factor=1.,total_iters=1)
    amp=exp.is_amp_enabled(device,cfg);scaler=torch.amp.GradScaler('cuda',enabled=amp)
    train_loader=exp.create_loader(datasets['train'],cfg['batch_size'],True,cfg['num_workers'],gen,device)
    eval_loaders={s:exp.create_loader(d,cfg['batch_size'],False,cfg['num_workers'],
        torch.Generator().manual_seed(2000+i),device) for i,(s,d) in enumerate(datasets.items())}
    start=0
    if saved is not None:
        exp.reconcile_history(out/'history.csv',saved['epoch'])
        start=exp.restore_training_state({**saved,'config':{**cfg,'save_features':False}},model,optimizer,scheduler,scaler,gen,device)
        del saved
    end=cfg['epochs'] if end_epoch is None else end_epoch # CPU tests bound the helper; CLI always 50.
    rows=[]
    for epoch in range(start,end+1):
        begin=time.perf_counter()
        stats=exp.train_one_epoch(model,train_loader,optimizer,scaler,device,amp) if epoch else {}
        vals={s:metrics(model,loader,device,amp) for s,loader in eval_loaders.items()}
        if epoch:scheduler.step()
        exp.append_history(out/'history.csv',epoch,stats,vals['val'],optimizer.param_groups[0]['lr'])
        exp.save_feature_snapshot(model,datasets['val'],'vgg11_bn',device,cfg['batch_size'],cfg['features_per_group'],
            out/'features'/f'epoch_{epoch:03d}.npz',cfg['num_workers'],42)
        (out/'epoch_metrics').mkdir(exist_ok=True)
        pending=dict(epoch=epoch,epoch_seconds=time.perf_counter()-begin,online_train=stats,splits=vals,checkpoint=None)
        atomic_json(out/'epoch_metrics'/f'epoch_{epoch:03d}.json',pending)
        record=save_verified(out/'checkpoints'/f'epoch_{epoch:03d}.pt',epoch,model,optimizer,scheduler,scaler,cfg,gen)
        if device.type=='cuda':torch.cuda.synchronize()
        row=dict(epoch=epoch,epoch_seconds=time.perf_counter()-begin,online_train=stats,splits=vals,checkpoint=record)
        (out/'epoch_metrics').mkdir(exist_ok=True)
        atomic_json(out/'epoch_metrics'/f'epoch_{epoch:03d}.json',row)
        rows.append(row);print(json.dumps(row),flush=True)
    return rows


def train(args):
    import os,socket
    from scripts.waterbirds_vgg50.download import atomic_json
    out=Path(args.output).resolve();allowed=(REPO/'runs/waterbirds_vgg50').resolve()
    if out==allowed or not out.is_relative_to(allowed):raise ValueError('Separate Waterbirds training subtree required')
    check=preflight(args.manifest,args.pilot,allowed);cfg=configuration(check['manifest']['manifest'],args.corr)
    if socket.gethostname().split('.')[0]!='moana-u2' or not os.environ.get('SLURM_JOB_ID') or args.device!='cuda':
        raise ValueError('Full training requires an allocated moana-u2 CUDA session')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('One allocated GPU required')
    if args.resume:
        if not out.is_dir():raise FileNotFoundError(out)
    else:out.mkdir(parents=True,exist_ok=False)
    with run_lock(out):
        saved=resume_state(out,cfg) if args.resume else None
        if args.resume and json.loads((out/'config.json').read_text())!=cfg:raise ValueError('Run config differs')
        atomic_json(out/'preflight.json',check)
        if not args.resume:atomic_json(out/'config.json',cfg)
        datasets=load(cfg['manifest'],args.corr,require_local=True)
        exp.seed_everything(42);model=exp.build_model(cfg).to('cuda')
        rows=fit(model,datasets,cfg,out,torch.device('cuda'),saved=saved)
        pointer=json.loads((out/'latest.json').read_text())
        if pointer['epoch']==50:
            verification=verify_run(out,cfg)
            atomic_json(out/'completion_integrity.json',verification)
            (out/'TRAIN_SUCCESS').touch()


def verify_run(out,cfg,upto=None):
    out=Path(out);end=cfg['epochs'] if upto is None else upto;records=[]
    for epoch in range(end+1):
        path=out/'checkpoints'/f'epoch_{epoch:03d}.pt'
        record=json.loads(Path(str(path)+'.integrity.json').read_text())
        if record['sha256']!=digest(path) or record['size_bytes']!=path.stat().st_size or record['config_sha256']!=config_sha(cfg) or record['manifest_sha256']!=cfg['manifest_sha256']:
            raise ValueError(f'Checkpoint integrity mismatch: {epoch}')
        row=json.loads((out/'epoch_metrics'/f'epoch_{epoch:03d}.json').read_text())
        if row.get('checkpoint')!=record or not (out/'features'/f'epoch_{epoch:03d}.npz').exists():raise ValueError('Incomplete epoch artifacts')
        records.append(record)
    return dict(verified=True,checkpoints=len(records),manifest_sha256=cfg['manifest_sha256'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--pilot',required=True);p.add_argument('--report',type=Path)
    args=p.parse_args();result=preflight(args.manifest,args.pilot)
    if args.report:
        with args.report.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
