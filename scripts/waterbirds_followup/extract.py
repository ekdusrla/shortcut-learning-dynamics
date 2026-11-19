"""Optional manual GPU extraction of missing Test snapshots, reusing saved Val features."""
import argparse,json,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import shortcut_experiment as exp
from scripts.waterbirds_vgg50.protocol import load,digest
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--code',choices=('095','080','065'),required=True);p.add_argument('--epochs',default='0:50');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Requires allocated GPU')
 if a.output.exists():raise FileExistsError(a.output)
 epochs=list(range(51)) if a.epochs=='0:50' else [int(x) for x in a.epochs.split(',')]
 if not epochs or len(set(epochs))!=len(epochs) or any(x<0 or x>50 for x in epochs):raise ValueError('Invalid epochs')
 run=ROOT/f'runs/waterbirds_vgg50/A_corr{a.code}_seed42_e50_contentv3';cfg=json.loads((run/'config.json').read_text());assert digest(cfg['manifest'])==cfg['manifest_sha256']
 ds=load(cfg['manifest'],cfg['train_corr'],require_local=True);a.output.mkdir(parents=True);torch.set_num_threads(4)
 for e in epochs:
  ck=run/f'checkpoints/epoch_{e:03d}.pt';side=json.loads(Path(str(ck)+'.integrity.json').read_text());assert digest(ck)==side['sha256'];saved=exp.load_checkpoint(ck,'cpu')
  with torch.device('meta'):model=exp.build_model(saved['config'])
  model.to_empty(device='cuda');model.load_state_dict(saved['model'],strict=True);model.eval();del saved
  shutil.copyfile(run/f'features/epoch_{e:03d}.npz',a.output/f'val_epoch_{e:03d}.npz')
  exp.seed_everything(42);exp.save_feature_snapshot(model,ds['test'],'vgg11_bn',torch.device('cuda'),16,32,a.output/f'test_epoch_{e:03d}.npz',0,42)
  del model;torch.cuda.empty_cache()
 (a.output/'EXTRACTION_SUCCESS').write_text(json.dumps(dict(epochs=epochs,manifest_sha256=cfg['manifest_sha256'],protocol='fixed jointly balanced val/test 32/group, FP32 inference, batch16, seed42; existing val reused'),indent=2))
if __name__=='__main__':main()
