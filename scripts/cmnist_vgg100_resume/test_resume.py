"""Offline CPU tests: real checkpoint restore and tiny synthetic continuation only."""
import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import shortcut_experiment as exp
RESULTS=[]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tiny(cfg):return nn.Linear(2,2)
def make_saved(parent):
 cfg=dict(dataset='colored_mnist',model='vgg11_bn',epochs=50,seed=42,image_size=224,
          train_corr=.95,val_corr=.1,test_corr=.1,mnist_val_size=5000,color_mode='background',
          color_pairing='all',waterbirds_root='data/raw',pretrained=False,augment=False,
          lr=.01,momentum=.9,scheduler='none',optimizer='sgd',weight_decay=.0001,
          batch_size=128,num_workers=4,features_per_group=32,amp=True,save_features=False)
 m=tiny(cfg); opt=torch.optim.SGD(m.parameters(),lr=.01,momentum=.9,weight_decay=.0001)
 sch=torch.optim.lr_scheduler.ConstantLR(opt,factor=1.,total_iters=1)
 m(torch.ones(1,2)).sum().backward();opt.step();sch.step()
 gen=torch.Generator().manual_seed(819)
 exp.save_checkpoint(parent/'checkpoints/epoch_050.pt',50,m,opt,sch,torch.amp.GradScaler('cpu'),cfg,gen)
 (parent/'config.json').write_text(json.dumps(cfg))
 for e in range(51):exp.append_history(parent/'history.csv',e,{'acc':1.,'loss':0.},{'acc':.95},.01)
 return exp.load_checkpoint(parent/'checkpoints/epoch_050.pt','cpu')
class ResumeTests(unittest.TestCase):
 def test_real_checkpoints(self):
  for tag in ['bg','fg']:
   with self.subTest(tag=tag):
    p=ROOT/f'runs/vgg11_{tag}_corr095_seed42_e50_img224_all'; cp=p/'checkpoints/epoch_050.pt'
    paths=[cp,p/'config.json',p/'history.csv',p/'test_metrics.json',p/'latest.json']
    before={str(x):digest(x) for x in paths}
    saved=exp.load_checkpoint(cp,'cpu');cfg=saved['config']
    self.assertEqual(cfg,json.loads((p/'config.json').read_text()))
    self.assertEqual(saved['epoch'],50);self.assertEqual(cfg['scheduler'],'none')
    self.assertNotIn('python_rng',saved);self.assertNotIn('numpy_rng',saved)
    # Meta construction avoids allocating torchvision's discarded full VGG head.
    with torch.device('meta'): model=exp.build_model(cfg)
    model.to_empty(device='cpu')
    opt=torch.optim.SGD(model.parameters(),lr=cfg['lr'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    sch=torch.optim.lr_scheduler.ConstantLR(opt,factor=1.,total_iters=1)
    scaler=torch.amp.GradScaler('cpu');gen=torch.Generator()
    start=exp.restore_training_state(saved,model,opt,sch,scaler,gen,torch.device('cpu'))
    self.assertEqual(start,51);self.assertEqual(scaler.state_dict(),saved['scaler'])
    for k,v in model.state_dict().items():
     self.assertTrue(torch.equal(v,saved['model'][k]),k);self.assertTrue(torch.isfinite(v).all())
    self.assertEqual(len(opt.state),34)
    for k,v in opt.state_dict()['state'].items():
     self.assertTrue(torch.equal(v['momentum_buffer'],saved['optimizer']['state'][k]['momentum_buffer']))
     self.assertTrue(torch.isfinite(v['momentum_buffer']).all())
    self.assertEqual(sch.state_dict(),saved['scheduler'])
    self.assertTrue(torch.equal(torch.get_rng_state(),saved['torch_rng']))
    self.assertTrue(torch.equal(gen.get_state(),saved['loader_rng']))
    # Recreate the exact epoch-51 shuffled index sequence from checkpoint generator.
    ref=torch.Generator();ref.set_state(saved['loader_rng'])
    ds=torch.utils.data.TensorDataset(torch.arange(55000))
    loader=exp.create_loader(ds,128,True,0,gen)
    ref_loader=exp.create_loader(ds,128,True,0,ref)
    self.assertTrue(torch.equal(torch.cat([b[0] for b in loader]),torch.cat([b[0] for b in ref_loader])))
    self.assertEqual(len(saved['cuda_rng']),1);self.assertEqual(saved['cuda_rng'][0].dtype,torch.uint8)
    with tempfile.TemporaryDirectory() as tmp:
     out=Path(tmp)/'extension';new=exp.extension_config(saved,100,out,cp)
     exp.prepare_extension(out,cp,saved,100)
     self.assertEqual(new['epochs'],100)
     self.assertEqual((out/'history.csv').read_bytes(),(p/'history.csv').read_bytes())
     self.assertFalse((out/'checkpoints').exists())
    # Scheduler steps only, no model forward/backward or optimizer training.
    with patch.object(opt,'_opt_called',True,create=True):
     for e in range(51,101):
      self.assertEqual(opt.param_groups[0]['lr'],.01);sch.step()
    self.assertEqual(sch.last_epoch,100)
    self.assertEqual(before,{str(x):digest(x) for x in paths})
    RESULTS.append(dict(mode=cfg['color_mode'],checkpoint=str(cp),epoch=50,start_epoch=start,
       model_tensors=len(saved['model']),momentum_buffers=len(opt.state),scheduler=saved['scheduler'],
       scaler=saved['scaler'],cuda_rng_present=True,cuda_rng_restored_on_gpu=False,
       python_numpy_rng_present=False,lr_51_to_100=.01,original_sha256=before))
    del saved,model,opt,sch,scaler,gen
 def test_extension_guards(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'parent';saved=make_saved(p);cp=p/'checkpoints/epoch_050.pt'
   for out in [p,p/'child',p.parent]:
    with self.assertRaises(ValueError):exp.extension_config(saved,100,out,cp)
   out=Path(tmp)/'used';out.mkdir();(out/'sentinel').touch()
   with self.assertRaises(FileExistsError):exp.extension_config(saved,100,out,cp)
   for target in [49,50]:
    with self.assertRaises(ValueError):exp.extension_config(saved,target,Path(tmp)/'fresh',cp)
   saved['config']['scheduler']='cosine'
   with self.assertRaises(ValueError):exp.extension_config(saved,100,Path(tmp)/'fresh',cp)
 def test_color_pairing_and_split_reconstruction(self):
  import types
  from PIL import Image
  class FakeMNIST:
   def __init__(self,n):self.labels=np.arange(n,dtype=np.int64)%10
   def __len__(self):return len(self.labels)
   def __getitem__(self,key):
    if isinstance(key,str):return self.labels
    return {'label':int(self.labels[key]),'image':Image.fromarray(np.full((28,28),127,dtype=np.uint8))}
  raw={'train':FakeMNIST(60000),'test':FakeMNIST(10000)}
  for tag in ['bg','fg']:
   cfg=json.loads((ROOT/f'runs/vgg11_{tag}_corr095_seed42_e50_img224_all/config.json').read_text())
   with patch.dict(sys.modules,{'datasets':types.SimpleNamespace(load_dataset=lambda *a,**k:raw)}):
    first=exp.build_datasets(cfg)
    random.random();np.random.rand();torch.rand(3)
    second=exp.build_datasets(cfg)
   self.assertEqual(len(first['train']),55000)
   for split in first:
    for attr in ['indices','labels','spurious','cf_spurious']:
     self.assertTrue(np.array_equal(getattr(first[split],attr),getattr(second[split],attr)))
   # Exercise actual coloring/CF path on one local synthetic image, no HF access.
   for key,value in first['val'][0].items():
    self.assertTrue(torch.equal(value,second['val'][0][key]) if torch.is_tensor(value) else value==second['val'][0][key])

 def test_rng_roundtrip(self):
  with tempfile.TemporaryDirectory() as tmp:
   saved=make_saved(Path(tmp))
   expected=(random.random(),np.random.rand(),torch.rand(3))
   m=tiny({});o=torch.optim.SGD(m.parameters(),lr=.01,momentum=.9);sch=torch.optim.lr_scheduler.ConstantLR(o,factor=1.)
   exp.restore_training_state(saved,m,o,sch,torch.amp.GradScaler('cpu'),torch.Generator(),torch.device('cpu'))
   self.assertEqual(random.random(),expected[0]);self.assertEqual(np.random.rand(),expected[1]);self.assertTrue(torch.equal(torch.rand(3),expected[2]))
 def test_integrated_epoch_range_and_preservation(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'parent';make_saved(p);out=Path(tmp)/'extended';before=(p/'history.csv').read_bytes()
   calls=[]
   def step(*a):calls.append('train');return dict(acc=1.,loss=0.)
   args=argparse.Namespace(output=str(out),device='cpu',resume=str(p/'checkpoints/epoch_050.pt'),extend_to_epoch=100,epochs=100,cf_chunk_size=128)
   with patch.object(exp,'build_datasets',return_value={k:[] for k in ['train','val','test']}),patch.object(exp,'create_loader',return_value=[]),patch.object(exp,'build_model',side_effect=tiny),patch.object(exp,'train_one_epoch',side_effect=step),patch.object(exp,'evaluate',return_value=dict(acc=.96,wga=.90,aligned_acc=1.,conflicting_acc=.95,cf_flip_rate=.04,cf_acc=.99)):
    exp.run_training(args)
   self.assertEqual(len(calls),50)
   with (out/'history.csv').open() as f:epochs=[int(r['epoch']) for r in csv.DictReader(f)]
   self.assertEqual(epochs,list(range(101)))
   self.assertEqual(sorted(x.name for x in (out/'checkpoints').glob('*.pt')),[f'epoch_{e:03d}.pt' for e in range(51,101)])
   self.assertEqual((p/'history.csv').read_bytes(),before)
   self.assertEqual((out/'parent_history.csv').read_bytes(),before)
   self.assertEqual(json.loads((out/'test_metrics.json').read_text())['epoch'],100)
 def test_cli_and_task_mapping(self):
  import os
  for i,mode in [(0,'background'),(1,'foreground')]:
   result=subprocess.run(['bash',str(Path(__file__).parent/'task.sh'),'--dry-run'],cwd=ROOT,env={**os.environ,'SLURM_ARRAY_TASK_ID':str(i)},capture_output=True,text=True,check=True)
   self.assertIn('--color-mode '+mode,result.stdout);self.assertIn('--extend-to-epoch 100',result.stdout)
   import shlex
   argv=shlex.split(result.stdout.split('Command: ',1)[1])[3:]
   # argv starts at train; parse the exact shell-generated invocation without running it.
   with patch.object(sys,'argv',['shortcut_experiment.py']+argv),patch.object(exp,'run_training') as mocked:
    exp.main();args=mocked.call_args.args[0]
    self.assertEqual(args.extend_to_epoch,100);self.assertEqual(args.cf_chunk_size,128);self.assertEqual(args.color_mode,mode)
  with patch.object(sys,'argv',['shortcut_experiment.py','train','--dataset','colored_mnist','--model','vgg11_bn','--output','/tmp/unused','--extend-to-epoch','100']),patch.object(exp,'run_training') as mocked:
   with self.assertRaises(SystemExit):exp.main()
   mocked.assert_not_called()
  bad=subprocess.run(['bash',str(Path(__file__).parent/'task.sh'),'--dry-run'],env={**os.environ,'SLURM_ARRAY_TASK_ID':'2'},capture_output=True)
  self.assertNotEqual(bad.returncode,0)
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ResumeTests))
 (Path(__file__).parent/'cpu_validation.json').write_text(json.dumps({'success':result.wasSuccessful(),'tests_run':result.testsRun,'real_checkpoints':RESULTS,'scope':'CPU only; no real model inference/training or CUDA restore'},indent=2))
 sys.exit(0 if result.wasSuccessful() else 1)
