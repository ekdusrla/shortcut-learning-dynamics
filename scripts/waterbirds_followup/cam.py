"""Manual, single-GPU CAM inference on preserved checkpoints; never trains VGG."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from PIL import Image
import shortcut_experiment as exp
from gradcam import GradCAM,unnormalize_image,save_visualization
from scripts.waterbirds_vgg50.protocol import load,digest
ROOT=Path(__file__).resolve().parents[2]
def region_metrics(heat,mask):
    h=np.asarray(heat,dtype=float);m=np.asarray(mask,dtype=float)
    if h.shape!=m.shape or not np.isfinite(h).all() or not np.isfinite(m).all() or m.min()<0 or m.max()>1:raise ValueError('Mask/CAM alignment or range invalid')
    area=float(m.mean());total=float(h.sum())
    if not 0<area<1:raise ValueError('Degenerate mask')
    return dict(mask_area_fraction=area,zero_cam=total<=1e-12,bird_mass_fraction=None if total<=1e-12 else float((h*m).sum()/total),bird_area_normalized_enrichment=None if total<=1e-12 else float((h*m).sum()/total/area),bird_mean=float((h*m).sum()/m.sum()),background_mean=float((h*(1-m)).sum()/(1-m).sum()))
def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--code',choices=('095','080','065'),required=True);a=p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Requires one allocated visible GPU; use manual SLURM submission')
    torch.set_num_threads(4);run=ROOT/f'runs/waterbirds_vgg50/A_corr{a.code}_seed42_e50_contentv3';cfg=json.loads((run/'config.json').read_text());manifest=Path(cfg['manifest']);assert digest(manifest)==cfg['manifest_sha256']
    datasets=load(manifest,cfg['train_corr'],require_local=True);ds=datasets['val'];meta=json.loads(manifest.read_text());plan=json.loads((a.analysis/'gradcam/epoch_plan.json').read_text());out=a.analysis/f'gradcam/A_corr{a.code}'
    if out.exists():raise FileExistsError(out)
    # Validate immutable images and matching sample IDs across conditions before allocating a model.
    indices={int(k):i for i,k in enumerate(ds.ids)};ids=plan['sample_ids'];assert all(k in indices for k in ids)
    out.mkdir(parents=True);rows=[];device=torch.device('cuda');started=time.time();torch.cuda.reset_peak_memory_stats()
    import resource
    for epoch in plan['epochs']['A_corr'+a.code]:
        ck=run/f'checkpoints/epoch_{epoch:03d}.pt';side=json.loads(Path(str(ck)+'.integrity.json').read_text());assert digest(ck)==side['sha256']
        saved=exp.load_checkpoint(ck,'cpu')
        with torch.device('meta'):model=exp.build_model(saved['config'])
        model.to_empty(device=device);model.load_state_dict(saved['model'],strict=True);model.eval();del saved
        with GradCAM(model,'features.18') as cam:
            for img_id in ids:
                i=indices[img_id];sample=ds[i];mask_path=ds.cub_dir/'segmentations'/ds.files[i].replace('.jpg','.png')
                soft=np.asarray(Image.open(mask_path).convert('RGB'),dtype=np.float32).mean(2)/255
                mask=np.asarray(Image.fromarray(soft,mode='F').resize((224,224),Image.Resampling.BILINEAR))
                predictions={}
                for kind,key in [('factual','pixel_values'),('cf','cf_pixel_values')]:
                    logits,heat=cam(sample[key][None].to(device),class_id=int(sample['labels']));prediction=int(logits.argmax(1));predictions[kind]=prediction;h=heat[0].numpy()
                    stem=f'e{epoch:03d}_id{img_id}_{kind}';np.save(out/(stem+'.npy'),h)
                    save_visualization(unnormalize_image(sample[key]),h,out/(stem+'.png'),title=f'A{a.code} E{epoch} id{img_id} {kind}; true-target={sample["labels"]}, pred={prediction}')
                    rows.append(dict(condition='A_corr'+a.code,epoch=epoch,img_id=img_id,y=int(sample['labels']),place=int(sample['spurious']) if kind=='factual' else 1-int(sample['spurious']),original_place=int(sample['spurious']),kind=kind,correct=prediction==int(sample['labels']),prediction=prediction,aligned=int(sample['labels'])==(int(sample['spurious']) if kind=='factual' else 1-int(sample['spurious'])),layer='features.18',target='true_class_fixed_factual_and_cf',**region_metrics(h,mask)))
                for row in rows[-2:]:row['pair_prediction_flip']=predictions['factual']!=predictions['cf'];row['pair_c2i']=predictions['factual']==int(sample['labels']) and predictions['cf']!=int(sample['labels'])
        del model;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out/'region_metrics.csv',index=False)
    (out/'validation.json').write_text(json.dumps(dict(success=True,seconds=time.time()-started,gpu_peak_allocated=torch.cuda.max_memory_allocated(),gpu_peak_reserved=torch.cuda.max_memory_reserved(),parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,normalization='Existing GradCAM per-map min/max [0,1], same display scale; absolute amplitude NOT comparable. Zero maps flagged; soft mask bilinear224 matches no-augmentation image Resize geometry.',sampling='Fixed two samples per group, same IDs across all conditions; coverage of correct/incorrect must be checked from results, not guaranteed.',epochs=plan['epochs']['A_corr'+a.code]),indent=2));(out/'CAM_SUCCESS').touch()
if __name__=='__main__':main()
