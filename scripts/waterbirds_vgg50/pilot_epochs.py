"""A_corr095 full-epoch GPU pilot, isolated outputs and unchanged 50-epoch protocol.

CLI never permits CPU training. CPU tests use the injected execute() helpers.
"""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
import data
import shortcut_experiment as exp
from scripts.waterbirds_vgg50.resources import tree_rss
from scripts.waterbirds_vgg50.protocol import digest, load, validate_frame, CORRS
from scripts.waterbirds_vgg50.run import metrics, save_epoch_checkpoint

REPO=Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST=REPO/'reports/waterbirds_vgg50/dataset_moana_u2_content_v3/protocol.json'


def manifest_path(path):
    p=Path(path).resolve();return p/'protocol.json' if p.is_dir() else p


def inspect_manifest(path):
    path=manifest_path(path);m=json.loads(path.read_text())
    if m.get('version')!=3 or m.get('condition')!='A' or m.get('val_corr')!=.5 or m.get('test_corr')!=.5:
        raise ValueError('Pilot requires Condition A content v3 val/test=.50')
    if digest(path.parent/'split_audit.json')!=m['split_audit_sha256']:
        raise ValueError('Split audit changed')
    audit=json.loads((path.parent/'split_audit.json').read_text())
    if audit['final']['cross_split_hashes']!=0:raise ValueError('Manifest reports leakage')
    frames={};stats={}
    for corr in CORRS:
        name=f'A_corr{round(corr*100):03d}.csv';p=path.parent/name
        if digest(p)!=m['files'][name]['sha256']:raise ValueError('Manifest CSV changed')
        frame=pd.read_csv(p);frames[corr]=frame
        stats[name]=validate_frame(frame,m['assets'],corr)
        if stats[name]!=m['files'][name]['splits']:raise ValueError('Recorded group/correlation stats differ')
    for corr in CORRS[1:]:
        pd.testing.assert_frame_equal(frames[corr].drop(columns='place'),frames[.95].drop(columns='place'))
        pd.testing.assert_frame_equal(frames[corr][frames[corr].split!='train'],frames[.95][frames[.95].split!='train'])
    return dict(manifest=str(path),manifest_sha256=digest(path),csv_sha256={n:r['sha256'] for n,r in m['files'].items()},
        root=m['root'],data_node=m.get('data_node'),condition='A',corr=.95,paired_comparison='PASS',
        cross_split_content_leakage=0,stats=stats,scope='Metadata/CSV/audit verification; original image hashes are checked on moana-u2 at execution')


def config(preflight,epochs):
    if epochs not in (1,2):raise ValueError('Pilot epochs must be 1 or 2')
    return dict(dataset='waterbirds',condition='A',model='vgg11_bn',pretrained=False,
        epochs=50,pilot_epochs=epochs,seed=42,image_size=224,train_corr=.95,val_corr=.5,test_corr=.5,
        augment=False,optimizer='sgd',lr=.01,momentum=.9,weight_decay=.0001,scheduler='none',
        batch_size=128,num_workers=4,amp=True,save_features=True,features_per_group=32,
        cf_chunk_size=128,manifest=preflight['manifest'],manifest_sha256=preflight['manifest_sha256'],
        waterbirds_root=preflight['root'],pilot=True,evaluation_protocol='single_pass_cf_v1')


def verify_routes(datasets,manifest):
    frame=pd.read_csv(manifest_path(manifest).parent/'A_corr095.csv');examples=[]
    for split,dataset in datasets.items():
        rows=frame[frame.split==split].reset_index(drop=True)
        if dataset.files!=rows.img_filename.tolist():raise ValueError('Bird route differs')
        np.testing.assert_array_equal(dataset.ids,rows.img_id.to_numpy())
        np.testing.assert_array_equal(dataset.labels,rows.y.to_numpy())
        np.testing.assert_array_equal(dataset.spurious,rows.place.to_numpy())
        np.testing.assert_array_equal(dataset.scenes,rows[['land_file','water_file']].to_numpy())
        # The train path never reconstructs/randomizes backgrounds from correlation.
        if split=='train':
            for y,s in dataset.groups:
                i=int(np.flatnonzero((dataset.labels==y)&(dataset.spurious==s))[0])
                examples.append(dict(split=split,img_id=int(dataset.ids[i]),label=y,place=s,
                    factual_scene=dataset.scenes[i,s],opposite_scene=dataset.scenes[i,1-s]))
    return dict(all_manifest_routes_match=True,examples=examples)


class MemoryMonitor:
    def __init__(self):
        self.stop=threading.Event();self.lock=threading.Lock();self.phase='startup'
        self.parent_peak=0;self.tree_peak=0;self.by_phase={}
    def sample(self):
        try:
            parent=next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
            tree=tree_rss()
        except (OSError,StopIteration):return
        with self.lock:
            self.parent_peak=max(self.parent_peak,parent);self.tree_peak=max(self.tree_peak,tree)
            self.by_phase[self.phase]=max(self.by_phase.get(self.phase,0),tree)
    def start(self):
        self.sample()
        def run():
            while not self.stop.wait(.2):self.sample()
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start()
    def finish(self):self.sample();self.stop.set();self.thread.join()


def execute(cfg,output,dataset_factory,device,preflight,model_factory=exp.build_model):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    monitor=MemoryMonitor();monitor.start();start=time.perf_counter()
    summary=dict(success=False,host=socket.gethostname(),device=str(device),config=cfg,
        manifest_sha256=cfg['manifest_sha256'],corr=.95,pilot_kind='full_epochs',preflight=preflight,phases={},epochs=[],cuda_peak_allocated_bytes=0,cuda_peak_reserved_bytes=0)
    def phase(name,fn):
        monitor.phase=name;monitor.sample()
        if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        before=time.perf_counter()
        try:return fn()
        finally:
            if device.type=='cuda':torch.cuda.synchronize()
            monitor.sample()
            allocated=torch.cuda.max_memory_allocated() if device.type=='cuda' else None
            reserved=torch.cuda.max_memory_reserved() if device.type=='cuda' else None
            summary['phases'][name]=dict(seconds=time.perf_counter()-before,gpu_peak_allocated_bytes=allocated,
                gpu_peak_reserved_bytes=reserved,cpu_tree_rss_peak_bytes=monitor.by_phase.get(name,0))
            if allocated is not None:
                summary['cuda_peak_allocated_bytes']=max(summary['cuda_peak_allocated_bytes'],allocated)
                summary['cuda_peak_reserved_bytes']=max(summary['cuda_peak_reserved_bytes'],reserved)
    try:
        (out/'config.json').write_text(json.dumps(cfg,indent=2))
        (out/'manifest_preflight.json').write_text(json.dumps(preflight,indent=2))
        exp.seed_everything(42);gen=torch.Generator().manual_seed(819)
        datasets=phase('source_validation',dataset_factory)
        route=verify_routes(datasets,cfg['manifest']);(out/'manifest_routes.json').write_text(json.dumps(route,indent=2))
        loaders={s:exp.create_loader(datasets[s],cfg['batch_size'],s=='train',cfg['num_workers'],gen if s=='train' else None,device) for s in ('train','val')}
        summary['loader_options']=dict(workers=cfg['num_workers'],**exp.loader_memory_options(cfg['num_workers']))
        summary['sample_counts']={s:len(d) for s,d in datasets.items()}
        model=phase('model_initialization',lambda:model_factory(cfg).to(device))
        optimizer=torch.optim.SGD(model.parameters(),lr=cfg['lr'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer,factor=1.,total_iters=1)
        amp=exp.is_amp_enabled(device,cfg);scaler=torch.amp.GradScaler('cuda',enabled=amp)
        if device.type=='cuda':
            p=torch.cuda.get_device_properties(device)
            summary['gpu']=dict(name=p.name,total_memory_bytes=p.total_memory,torch=torch.__version__,cuda=torch.version.cuda,amp=amp)
        for epoch in range(cfg['pilot_epochs']+1):
            epoch_start=time.perf_counter()
            online=phase(f'epoch_{epoch:03d}_train',lambda:exp.train_one_epoch(model,loaders['train'],optimizer,scaler,device,amp)) if epoch else {}
            val=phase(f'epoch_{epoch:03d}_val',lambda:metrics(model,loaders['val'],device,amp))
            if not math.isfinite(val['acc']) or not math.isfinite(val['loss']):raise ValueError('Nonfinite validation metrics')
            if epoch:scheduler.step()
            phase(f'epoch_{epoch:03d}_features',lambda:exp.save_feature_snapshot(model,datasets['val'],'vgg11_bn',device,
                cfg['batch_size'],cfg['features_per_group'],out/'features'/f'epoch_{epoch:03d}.npz',cfg['num_workers'],42))
            path=out/'checkpoints'/f'epoch_{epoch:03d}.pt'
            def checkpoint():
                save_epoch_checkpoint(path,epoch,model,optimizer,scheduler,scaler,cfg,gen)
                saved=exp.load_checkpoint(path,'cpu')
                if saved['epoch']!=epoch or saved['config']!=cfg or saved['checkpoint_boundary']!='after_feature_snapshot':
                    raise ValueError('Checkpoint verification failed')
                if epoch and not saved['optimizer']['state']:raise ValueError('Optimizer state missing')
                del saved
            phase(f'epoch_{epoch:03d}_checkpoint',checkpoint)
            exp.append_history(out/'history.csv',epoch,online,val,optimizer.param_groups[0]['lr'])
            checkpoint_record=dict(path=str(path.resolve()),verified=True,size_bytes=path.stat().st_size,sha256=digest(path))
            row=dict(epoch=epoch,epoch_seconds=time.perf_counter()-epoch_start,online_train=online,val=val,checkpoint=checkpoint_record)
            summary['epochs'].append(row)
            (out/f'epoch_{epoch:03d}.json').write_text(json.dumps(row,indent=2))
            print(json.dumps(row),flush=True)
        summary['success']=True
        (out/'PILOT_SUCCESS').write_text('Full-epoch pilot only; not completion of 50-epoch training.\n')
    except Exception as error:
        summary['error']=traceback.format_exc();summary['oom']=isinstance(error,torch.cuda.OutOfMemoryError)
        raise
    finally:
        monitor.finish();summary.update(total_seconds=time.perf_counter()-start,
            cpu_parent_rss_peak_bytes=monitor.parent_peak,cpu_tree_rss_peak_bytes=monitor.tree_peak,
            rss_scope='Sampled every .2 seconds; shared RSS may be double-counted; abrupt cgroup kill may prevent final JSON')
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument('--epochs',type=int,choices=(1,2),default=2)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--preflight-only',action='store_true',help='Metadata/CSV check only; no original images/GPU/training')
    args=parser.parse_args();preflight=inspect_manifest(args.manifest)
    if args.preflight_only:print(json.dumps(preflight,indent=2));return
    if not args.output:parser.error('--output required')
    allowed=(REPO/'runs/waterbirds_vgg50_pilots').resolve();out=args.output.resolve()
    if out==allowed or not out.is_relative_to(allowed):raise ValueError('Output must be a new Waterbirds pilot subtree')
    if socket.gethostname().split('.')[0]!='moana-u2' or preflight['data_node']!='moana-u2':raise ValueError('Use moana-u2 node-local data')
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Allocated SLURM GPU session required')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one allocated GPU required')
    cfg=config(preflight,args.epochs)
    execute(cfg,out,lambda:load(cfg['manifest'],.95,require_local=True),torch.device('cuda'),preflight)

if __name__=='__main__':main()
