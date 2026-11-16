import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
import shortcut_experiment as exp
from scripts.waterbirds_vgg50 import full_training as full
from scripts.waterbirds_vgg50.run import metrics
from scripts.waterbirds_vgg50.test_pilot_epochs import TinyModel
from scripts.waterbirds_vgg50.test_waterbirds import fixture
from scripts.waterbirds_vgg50.protocol import prepare,load


class Tests(unittest.TestCase):
    def test_single_pass_metrics_equivalence_and_chunk_order(self):
        class D(torch.utils.data.Dataset):
            num_classes=2;groups=[(0,0),(0,1),(1,0),(1,1)]
            def __len__(self):return 135
            def __getitem__(self,i):
                return dict(pixel_values=torch.tensor([float(i%2)]),cf_pixel_values=torch.tensor([float(1-i%2)]),labels=(i//2)%2,spurious=i%2,index=i)
        class M(nn.Module):
            def __init__(self):super().__init__();self.calls=0
            def forward(self,x):self.calls+=1;return torch.cat([1-x,x],1)*10
        model=M();loader=exp.create_loader(D(),129,False,0)
        reference=exp.evaluate(model,loader,torch.device('cpu'))
        model.calls=0;actual=metrics(model,loader,torch.device('cpu'),False)
        self.assertEqual(model.calls,5) # 2 factual forwards + CF chunks 128/1/6.
        for key,value in reference.items():
            if isinstance(value,float) and math.isnan(value):self.assertTrue(math.isnan(actual[key]))
            else:self.assertEqual(actual[key],value)
        self.assertEqual(actual['cf_correct_to_incorrect_rate'],1.)
        self.assertEqual(len(model._forward_hooks),0)

    def test_fit_resume_integrity_and_separate_eval_shuffle(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'data';root.mkdir();fixture(root);manifest=Path(td)/'manifest';prepare(root,manifest)
            ds=load(manifest/'protocol.json',.95)
            class Small(torch.utils.data.Dataset):
                num_classes=2;groups=[(0,0),(0,1),(1,0),(1,1)]
                def __init__(self,d):
                    self.source=d;self.indices=exp.balanced_indices(d,1,42)
                    self.labels=d.labels[self.indices];self.spurious=d.spurious[self.indices]
                def __len__(self):return len(self.indices)
                def __getitem__(self,i):return self.source[self.indices[i]]
            ds={s:Small(d) for s,d in ds.items()}
            cfg=full.configuration(manifest/'protocol.json',.95);cfg.update(batch_size=2,num_workers=0,features_per_group=1)
            self.assertEqual(cfg['epochs'],50)
            complete=Path(td)/'complete';complete.mkdir();partial=Path(td)/'partial';partial.mkdir()
            exp.seed_everything(42);model=TinyModel()
            with contextlib.redirect_stdout(io.StringIO()):rows=full.fit(model,ds,cfg,complete,torch.device('cpu'),end_epoch=2)
            initial=exp.load_checkpoint(complete/'checkpoints/epoch_000.pt','cpu')
            self.assertTrue(torch.equal(initial['loader_rng'],torch.Generator().manual_seed(819).get_state()))
            self.assertEqual(len(rows),3)
            self.assertEqual(full.verify_run(complete,cfg,upto=2)['checkpoints'],3)
            self.assertEqual(set(rows[-1]['splits']),{'train','val','test'})
            exp.seed_everything(42);other=TinyModel()
            with contextlib.redirect_stdout(io.StringIO()):full.fit(other,ds,cfg,partial,torch.device('cpu'),end_epoch=1)
            saved=full.resume_state(partial,cfg)
            with contextlib.redirect_stdout(io.StringIO()):full.fit(other,ds,cfg,partial,torch.device('cpu'),saved=saved,end_epoch=2)
            for key,value in model.state_dict().items():torch.testing.assert_close(value,other.state_dict()[key],rtol=0,atol=0)
            metrics_path=partial/'epoch_metrics/epoch_002.json'
            pending=json.loads(metrics_path.read_text());pending['checkpoint']=None;metrics_path.write_text(json.dumps(pending))
            full.resume_state(partial,cfg)
            self.assertTrue(json.loads(metrics_path.read_text())['recovered_checkpoint_commit'])
            last=partial/'checkpoints/epoch_002.pt';before=last.read_bytes()
            with self.assertRaises(FileExistsError):
                full.save_verified(last,2,other,None,None,None,cfg,None)
            self.assertEqual(last.read_bytes(),before)
            last.write_bytes(before[:-1]+bytes([before[-1]^1]))
            with self.assertRaisesRegex(ValueError,'SHA'):full.resume_state(partial,cfg)

    def test_capacity_and_condition_configs(self):
        pilot=dict(epochs=[dict(epoch=0,checkpoint=dict(size_bytes=36969761)),dict(epoch=1,checkpoint=dict(size_bytes=73887871))])
        plan=full.disk_plan(pilot,Path('/tmp'))
        self.assertEqual(plan['checkpoint_bytes_per_condition'],36969761+50*73887871)
        self.assertEqual(plan['checkpoint_count_per_condition'],51)
        with patch('scripts.waterbirds_vgg50.full_training.shutil.disk_usage',return_value=type('Disk',(),{'free':1})()):
            self.assertFalse(full.disk_plan(pilot,Path('/tmp'))['enough_space'])
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'protocol.json';p.write_text(json.dumps({'root':'/local_datasets/USER/waterbirds_v1'}))
            for corr in (.95,.8,.65):
                cfg=full.configuration(p,corr)
                for k,v in dict(epochs=50,train_corr=corr,val_corr=.5,test_corr=.5,model='vgg11_bn',pretrained=False,lr=.01,
                    momentum=.9,weight_decay=.0001,batch_size=128,image_size=224,seed=42,amp=True,num_workers=4,scheduler='none').items():self.assertEqual(cfg[k],v)

if __name__=='__main__':unittest.main()
