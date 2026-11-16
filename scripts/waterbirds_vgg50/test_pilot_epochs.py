import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
from torch import nn
import data
import shortcut_experiment as exp
from scripts.waterbirds_vgg50.protocol import prepare,load,digest
from scripts.waterbirds_vgg50.test_waterbirds import fixture
from scripts.waterbirds_vgg50.pilot_epochs import inspect_manifest,config,verify_routes,execute


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__();self.features=nn.Sequential(nn.Conv2d(3,4,1),nn.ReLU(),nn.MaxPool2d(2))
        self.avgpool=nn.AdaptiveAvgPool2d(1);self.classifier=nn.Linear(4,2)
    def forward(self,x):return self.classifier(self.avgpool(self.features(x)).flatten(1))


class Tests(unittest.TestCase):
    def setup_fixture(self,td):
        root=Path(td)/'raw';root.mkdir();fixture(root)
        manifest=Path(td)/'manifest';prepare(root,manifest)
        return root,manifest/'protocol.json'

    def test_fixed_config_and_manifest_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            root,manifest=self.setup_fixture(td);before=digest(manifest)
            summary=inspect_manifest(manifest)
            self.assertEqual(summary['paired_comparison'],'PASS');self.assertEqual(summary['cross_split_content_leakage'],0)
            cfg=config(summary,2)
            for k,v in dict(model='vgg11_bn',pretrained=False,lr=.01,momentum=.9,weight_decay=.0001,
                batch_size=128,image_size=224,seed=42,epochs=50,pilot_epochs=2,val_corr=.5,test_corr=.5,train_corr=.95,scheduler='none',amp=True,num_workers=4).items():self.assertEqual(cfg[k],v)
            self.assertEqual(exp.loader_memory_options(4),dict(pin_memory=False,persistent_workers=False,prefetch_factor=1))
            with self.assertRaises(ValueError):config(summary,50)
            self.assertEqual(digest(manifest),before)
            csv=manifest.parent/'A_corr095.csv';csv.write_bytes(csv.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'CSV changed'):inspect_manifest(manifest)

    def test_manifest_routes_and_real_composited_pixels(self):
        with tempfile.TemporaryDirectory() as td:
            root,manifest=self.setup_fixture(td);datasets=load(manifest,.95)
            result=verify_routes(datasets,manifest);self.assertTrue(result['all_manifest_routes_match'])
            self.assertEqual({(r['label'],r['place']) for r in result['examples']},{(0,0),(0,1),(1,0),(1,1)})
            dataset=datasets['val']
            for s in (0,1):
                i=int(np.flatnonzero(dataset.spurious==s)[0]);sample=dataset[i]
                rel=dataset.files[i]
                with Image.open(root/'CUB_200_2011/images'/rel) as im:rgb=np.asarray(im.convert('RGB'))
                with Image.open(root/'CUB_200_2011/segmentations'/Path(rel).with_suffix('.png')) as im:mask=np.asarray(im.convert('RGB'))/255
                bird=Image.fromarray(np.around(rgb*mask).astype(np.uint8))
                # Expected scene is independently read from the manifest-selected route.
                for place,key in ((s,'pixel_values'),(1-s,'cf_pixel_values')):
                    with Image.open(root/'places365/data_256'/dataset.scenes[i,place]) as im:
                        expected=data.to_tensor(dataset,data.combine_and_mask(im.convert('RGB'),mask,bird),i)
                    torch.testing.assert_close(sample[key],expected,rtol=0,atol=0)
            datasets['train'].spurious[0]=1-datasets['train'].spurious[0]
            with self.assertRaises(AssertionError):verify_routes(datasets,manifest)

    def test_two_epoch_cpu_orchestration_checkpoints_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root,manifest=self.setup_fixture(td);preflight=inspect_manifest(manifest);cfg=config(preflight,2)
            # CPU-only bounded orchestration smoke; actual VGG224 backward is in test_waterbirds.
            cfg.update(batch_size=2,num_workers=0,features_per_group=1)
            datasets=load(manifest,.95)
            routes=verify_routes(datasets,manifest)
            from unittest.mock import patch
            class Small(torch.utils.data.Dataset):
                num_classes=2;groups=[(0,0),(0,1),(1,0),(1,1)]
                def __init__(self,source):
                    self.source=source;self.indices=[int(np.flatnonzero((source.labels==y)&(source.spurious==s))[0]) for y,s in self.groups]
                    self.labels=source.labels[self.indices];self.spurious=source.spurious[self.indices]
                def __len__(self):return len(self.indices)
                def __getitem__(self,i):return self.source[self.indices[i]]
            small={s:Small(d) for s,d in datasets.items()}
            out=Path(td)/'pilot'
            with patch('scripts.waterbirds_vgg50.pilot_epochs.verify_routes',return_value=routes),contextlib.redirect_stdout(io.StringIO()):
                result=execute(cfg,out,lambda:small,torch.device('cpu'),preflight,model_factory=lambda c:TinyModel())
            self.assertTrue(result['success']);self.assertEqual(len(result['epochs']),3)
            self.assertEqual(result['sample_counts']['train'],4)
            self.assertTrue((out/'PILOT_SUCCESS').exists());self.assertFalse((out/'TRAIN_SUCCESS').exists())
            for epoch in (0,1,2):
                checkpoint=out/'checkpoints'/f'epoch_{epoch:03d}.pt';saved=exp.load_checkpoint(checkpoint,'cpu')
                self.assertEqual(saved['epoch'],epoch);self.assertEqual(saved['config']['pilot_epochs'],2)
                self.assertEqual(saved['checkpoint_boundary'],'after_feature_snapshot')
                self.assertTrue(result['epochs'][epoch]['checkpoint']['verified'])
            clone=TinyModel();clone.load_state_dict(saved['model'])
            self.assertEqual(saved['config']['epochs'],50)
            self.assertIn('epoch_002_train',result['phases']);self.assertGreater(result['cpu_parent_rss_peak_bytes'],0)
            saved_summary=(out/'summary.json').read_bytes()
            with self.assertRaises(FileExistsError):execute(cfg,out,lambda:small,torch.device('cpu'),preflight,model_factory=lambda c:TinyModel())
            self.assertEqual((out/'summary.json').read_bytes(),saved_summary)

if __name__=='__main__':unittest.main()
