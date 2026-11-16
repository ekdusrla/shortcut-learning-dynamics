"""Bounded manual pilot: two batches + resume + features; never submit SLURM."""
import argparse
import json
import sys
import tempfile
import threading
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import shortcut_experiment as exp
from scripts.waterbirds_vgg50.resources import tree_rss
from scripts.waterbirds_vgg50.protocol import load, digest
from scripts.waterbirds_vgg50.run import metrics

class Limited:
    def __init__(self,loader): self.loader=loader; self.dataset=loader.dataset
    def __iter__(self):
        for i,batch in enumerate(self.loader):
            if i>=2: break
            yield batch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True)
    p.add_argument('--corr',type=float,choices=[.95,.8,.65],required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); result=dict(success=False,cpu_tree_rss_peak_bytes=0,manifest_sha256=digest(a.manifest),corr=a.corr)
    stop=threading.Event()
    def monitor():
        while not stop.is_set():
            result['cpu_tree_rss_peak_bytes']=max(result['cpu_tree_rss_peak_bytes'],tree_rss()); stop.wait(.2)
    with Path(a.output).open('x') as report:
        t=threading.Thread(target=monitor,daemon=True); t.start()
        try:
            if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('One allocated GPU required')
            result['gpu']=torch.cuda.get_device_name(0)
            device=torch.device('cuda'); exp.seed_everything(42)
            ds=load(a.manifest,a.corr,require_local=True)
            cfg=dict(dataset='waterbirds',model='vgg11_bn',pretrained=False,amp=True,save_features=False)
            model=exp.build_model(cfg).to(device); gen=torch.Generator().manual_seed(819)
            opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=1e-4)
            sched=torch.optim.lr_scheduler.ConstantLR(opt,factor=1.,total_iters=1)
            scaler=torch.amp.GradScaler('cuda')
            def stage(name,fn):
                torch.cuda.reset_peak_memory_stats(); value=fn(); torch.cuda.synchronize()
                result[name]=dict(value=value,allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
            stage('train_two_batches',lambda:exp.train_one_epoch(model,Limited(exp.create_loader(ds['train'],128,True,4,gen)),opt,scaler,device,True))
            stage('validation_two_batches',lambda:metrics(model,Limited(exp.create_loader(ds['val'],128,False,4)),device,True))
            with tempfile.TemporaryDirectory() as td:
                path=Path(td)/'checkpoints/epoch_001.pt'
                exp.save_checkpoint(path,1,model,opt,sched,scaler,cfg,gen)
                saved=exp.load_checkpoint(path,'cpu')
                assert exp.restore_training_state(saved,model,opt,sched,scaler,gen,device)==2
                del saved
                stage('fp32_features',lambda:exp.save_feature_snapshot(model,ds['val'],'vgg11_bn',device,128,32,Path(td)/'features.npz',4,42))
            result['success']=True
        except Exception:
            result['error']=traceback.format_exc(); raise
        finally:
            stop.set(); t.join(); report.write(json.dumps(result,indent=2)); report.flush()
if __name__=='__main__': main()
