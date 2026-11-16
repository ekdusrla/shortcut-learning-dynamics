import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from unittest.mock import patch
import data
import shortcut_experiment as exp
from scripts.waterbirds_vgg50.protocol import prepare, load, digest
from scripts.waterbirds_vgg50.run import metrics, save_epoch_checkpoint


def fixture(root):
    cub=root/'CUB_200_2011'; cub.mkdir()
    names=[]; splits=[]
    for i in range(1,161):
        species='001.Albatross' if i%2 else '002.Sparrow'
        name=f'{species}/{i}.jpg'; names.append(f'{i} {name}\n'); splits.append(f'{i} {int(i<=120)}\n')
        for kind,ext in [('images','.jpg'),('segmentations','.png')]:
            p=cub/kind/Path(name).with_suffix(ext); p.parent.mkdir(parents=True,exist_ok=True)
            a=np.zeros((32,32,3),dtype=np.uint8)
            if kind=='images':
                a[:]=np.random.default_rng(i).integers(0,256,a.shape,dtype=np.uint8)
            else: a[8:24,8:24]=255
            Image.fromarray(a).save(p)
    (cub/'images.txt').write_text(''.join(names)); (cub/'train_test_split.txt').write_text(''.join(splits))
    for s,categories in enumerate(data.TARGET_PLACES):
        for category in categories:
            for i in range(20):
                p=root/'places365'/data.PLACES_PREFIX/category[0]/category/f'{i}.jpg'
                p.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(np.random.default_rng(i+1000*s+100*len(category)).integers(0,256,(32,32,3),dtype=np.uint8)).save(p)

class Tests(unittest.TestCase):
    def test_protocol_pair_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'raw'; root.mkdir(); fixture(root)
            a=Path(td)/'a'; b=Path(td)/'b'
            prepare(root,a); prepare(root,b)
            self.assertEqual((a/'protocol.json').read_bytes(),(b/'protocol.json').read_bytes())
            datasets=[load(a/'protocol.json',c) for c in (.95,.8,.65)]
            self.assertEqual([len(d['train']) for d in datasets],[96]*3)
            for split in ('val','test'):
                for d in datasets[1:]:
                    np.testing.assert_array_equal(d[split].spurious,datasets[0][split].spurious)
            d=datasets[0]['test']; item=d[0]
            self.assertEqual(tuple(item['pixel_values'].shape),(3,224,224))
            # The fully opaque central bird is exactly unchanged by background intervention.
            torch.testing.assert_close(item['pixel_values'][:,80:140,80:140],item['cf_pixel_values'][:,80:140,80:140],rtol=0,atol=0)
            b0=next(iter(exp.create_loader(d,2,False,0)))
            b4=next(iter(exp.create_loader(d,2,False,4)))
            for key in b0: torch.testing.assert_close(b0[key],b4[key],rtol=0,atol=0)
            source=next((root/'CUB_200_2011/images').rglob('*.jpg')); source.write_bytes(b'changed')
            with self.assertRaises(ValueError): load(a/'protocol.json',.95)
            with self.assertRaises(FileExistsError): prepare(root,a)

    def test_metrics_denominators(self):
        class D(torch.utils.data.Dataset):
            num_classes=2; groups=[(0,0),(0,1),(1,0),(1,1)]
            def __len__(self): return 4
            def __getitem__(self,i):
                return dict(pixel_values=torch.tensor([float([0,1,1,1][i])]),cf_pixel_values=torch.tensor([float([1,0,0,1][i])]),labels=i//2,spurious=i%2,index=i)
        class M(nn.Module):
            def forward(self,x): return torch.cat([1-x,x],1)*10
        result=metrics(M(),exp.create_loader(D(),3,False,0),torch.device('cpu'),False)
        self.assertEqual(result['acc'],.75)
        self.assertEqual(result['cf_flip_rate'],.75)
        self.assertAlmostEqual(result['cf_correct_to_incorrect_rate'],2/3)
        self.assertEqual(result['wga'],0.)

    def test_vgg_backward_resume_cam_probe(self):
        # Actual 224px compact VGG forward/backward, bounded batch of two.
        from torchvision import models
        from gradcam import GradCAM
        from evaluate import DecodedCorrelation
        cfg=dict(dataset='waterbirds',model='vgg11_bn',pretrained=False,save_features=False)
        # Avoid standard 500MB temporary classifier during construction in CPU smoke test.
        model=exp.build_model(cfg)
        self.assertEqual(model.classifier.out_features,2)
        opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.ConstantLR(opt,factor=1,total_iters=1)
        scaler=torch.amp.GradScaler('cuda',enabled=False); gen=torch.Generator().manual_seed(819)
        x=torch.randn(2,3,224,224); y=torch.tensor([0,1])
        logits=model(x); self.assertEqual(tuple(logits.shape),(2,2))
        nn.functional.cross_entropy(logits,y).backward(); opt.step(); sched.step(); model.eval()
        expected=model(x).detach().clone()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'checkpoints/epoch_001.pt'
            save_epoch_checkpoint(path,1,model,opt,sched,scaler,cfg,gen)
            saved=exp.load_checkpoint(path,'cpu')
            self.assertEqual(saved['checkpoint_boundary'],'after_feature_snapshot')
            with torch.no_grad(): model.classifier.weight.zero_()
            self.assertEqual(exp.restore_training_state(saved,model,opt,sched,scaler,gen,torch.device('cpu')),2)
            torch.testing.assert_close(model(x),expected,rtol=0,atol=0)
        with GradCAM(model,'features.18') as cam:
            scores,heat=cam(x[:1],0)
            self.assertTrue(torch.isfinite(heat).all()); self.assertEqual(tuple(heat.shape),(1,224,224))
        probe=DecodedCorrelation(2,2)
        probe.fit(torch.eye(2),y,epochs=1,batch_size=2)
        self.assertIn('accuracy',probe.evaluate(torch.eye(2),y,2))

if __name__=='__main__': unittest.main()
