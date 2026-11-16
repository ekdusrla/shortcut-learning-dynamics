"""Waterbirds-only entrypoint; no monkey patches to shared modules."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import shortcut_experiment as exp
from scripts.waterbirds_vgg50.protocol import prepare, load, digest, CORRS, EVAL_CORR

def save_epoch_checkpoint(path, epoch, model, optimizer, scheduler, scaler, cfg, gen):
    import os
    import random
    import numpy as np
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    rng=np.random.get_state()
    state=dict(epoch=epoch,config=cfg,model=model.state_dict(),optimizer=optimizer.state_dict(),
               scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
               torch_rng=torch.get_rng_state(),loader_rng=gen.get_state(),
               python_rng=random.getstate(),
               numpy_rng=dict(kind=rng[0],keys=rng[1].tolist(),pos=rng[2],has_gauss=rng[3],cached_gaussian=rng[4]),
               checkpoint_boundary='after_feature_snapshot',
               cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    tmp=path.with_suffix('.tmp'); torch.save(state,tmp); os.replace(tmp,path)
    pointer=path.parent.parent/'latest.json'; tmp=pointer.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(epoch=epoch,checkpoint=str(path.resolve())),indent=2)); os.replace(tmp,pointer)


@torch.inference_mode()
def metrics(model, loader, device, amp):
    """Observe evaluate's original/CF predictions; no duplicate forwards/load passes.

    Waterbirds only. evaluate emits the original batch followed by CF chunks.
    The observer checks this contract and never alters logits or model weights.
    """
    state={};correct=flips=0
    class TrackingLoader:
        dataset=loader.dataset
        def __iter__(self):
            for batch in loader:
                if 'cf_all_pixel_values' in batch:raise ValueError('Binary Waterbirds metrics only')
                state.clear();state.update(original=True,labels=batch['labels'],cf='cf_pixel_values' in batch,offset=0)
                yield batch
                if state['cf'] and state['offset']!=len(batch['labels']):raise ValueError('Incomplete counterfactual observer')
    def observe(_module,_inputs,logits):
        nonlocal correct,flips
        prediction=logits.argmax(1)
        if state['original']:
            state['original']=False
            if state['cf']:
                state['y']=state['labels'].to(logits.device);state['ok']=prediction==state['y']
                correct+=int(state['ok'].sum())
        else:
            if not state['cf']:raise ValueError('Unexpected extra evaluation forward')
            first=state['offset'];last=first+len(prediction)
            flips+=int((state['ok'][first:last] & (prediction!=state['y'][first:last])).sum())
            state['offset']=last
    handle=model.register_forward_hook(observe)
    try:result=exp.evaluate(model,TrackingLoader(),device,amp,128)
    finally:handle.remove()
    result['cf_correct_to_incorrect_rate']=flips/correct if correct else float('nan')
    result['cf_original_correct_n']=correct
    return result


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--root',required=True); a.add_argument('--output',required=True)
    a=sub.add_parser('train'); a.add_argument('--manifest',required=True); a.add_argument('--corr',type=float,choices=CORRS,required=True)
    a.add_argument('--pilot',required=True); a.add_argument('--output',required=True); a.add_argument('--resume',action='store_true'); a.add_argument('--device',default='cuda')
    a=sub.add_parser('analyze'); a.add_argument('--run',required=True); a.add_argument('--epoch',type=int,required=True)
    a.add_argument('--output',required=True); a.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.action=='prepare':
        from scripts.waterbirds_vgg50.storage import local_root
        local_root(args.root)
        prepare(args.root,args.output,expected_birds=11788,expected_scenes_per_category=5000); return
    if args.action=='analyze':
        analyze(args); return
    from scripts.waterbirds_vgg50.full_training import train
    train(args)


def analyze(args):
    from evaluate import DecodedCorrelation
    from gradcam import GradCAM, unnormalize_image, save_visualization
    out=Path(args.output).resolve()
    allowed=Path(__file__).resolve().parents[2]/'runs/waterbirds_vgg50'
    if not out.is_relative_to(allowed): raise ValueError('Analysis output outside Waterbirds subtree')
    ck=Path(args.run)/'checkpoints'/f'epoch_{args.epoch:03d}.pt'
    saved=exp.load_checkpoint(ck,'cpu'); cfg=saved['config']
    if digest(cfg['manifest'])!=cfg['manifest_sha256']: raise ValueError('Protocol changed')
    ds=load(cfg['manifest'],cfg['train_corr'],require_local=True); device=exp.get_device(args.device)
    out.mkdir(parents=True,exist_ok=False)
    exp.seed_everything(42); model=exp.build_model(cfg).to(device); model.load_state_dict(saved['model']); model.eval()
    del saved
    # Fit on independent, jointly balanced validation groups; score on held-out test.
    # This deliberately avoids the CMNIST ten-color crossed protocol.
    for split in ('val','test'):
        per_group=min(32, min(int(((ds[split].labels==y)&(ds[split].spurious==s)).sum()) for y in (0,1) for s in (0,1)))
        exp.save_feature_snapshot(model,ds[split],'vgg11_bn',device,128,per_group,out/f'{split}.npz',4,42)
    import numpy as np
    f={s:np.load(out/f'{s}.npz') for s in ('val','test')}
    rows=[]
    for i,layer in enumerate(f['val']['layers']):
        for target in ('labels','spurious'):
            exp.seed_everything(42)
            probe=DecodedCorrelation(f['val'][f'f{i}'].shape[1],2).to(device)
            probe.fit(torch.from_numpy(f['val'][f'f{i}']),torch.from_numpy(f['val'][target]),epochs=100,batch_size=256,lr=.001,weight_decay=.0001)
            score=probe.evaluate(torch.from_numpy(f['test'][f'f{i}']),torch.from_numpy(f['test'][target]),256)
            probe.save(out/f'probe_{i}_{target}.pt')
            rows.append(dict(epoch=args.epoch,layer=str(layer),target=target,classes=2,chance=.5,**score))
    (out/'probes.json').write_text(json.dumps(rows,indent=2))
    sample=ds['val'][0]
    with GradCAM(model,'features.18') as cam:
        logits, heat=cam(sample['pixel_values'][None].to(device),class_id=int(sample['labels']))
    save_visualization(unnormalize_image(sample['pixel_values']),heat[0].numpy(),out/'gradcam.png',title=f'Waterbirds A epoch {args.epoch}, y={sample["labels"]}')
    (out/'protocol.json').write_text(json.dumps(dict(checkpoint=str(ck),checkpoint_sha256=digest(ck),config=cfg,probe_fit='balanced validation <=32/group',probe_score='balanced test <=32/group',cam_index=int(sample['index'])),indent=2))

if __name__=='__main__': main()
