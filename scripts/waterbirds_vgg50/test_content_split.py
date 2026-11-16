import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from scripts.waterbirds_vgg50.test_waterbirds import fixture
from scripts.waterbirds_vgg50.protocol import prepare,load,digest,CORRS,validate_frame
from scripts.waterbirds_vgg50.content_split import inventory,legacy_scenes,leakage_report,counts


def snapshot(root):return {str(p.relative_to(root)):(p.stat().st_mtime_ns,digest(p)) for p in root.rglob('*') if p.is_file()}


class Tests(unittest.TestCase):
    def make(self,td):
        root=Path(td)/'raw';root.mkdir();fixture(root);return root
    def test_actual_scene_alias_leak_is_repaired_without_touching_sources(self):
        with tempfile.TemporaryDirectory() as td:
            root=self.make(td);base,assets,_,_=inventory(root);old=legacy_scenes(root,base,42)
            a=old[old.split=='train'].iloc[0].land_file;b=old[old.split=='test'].iloc[0].land_file
            scenes=root/'places365/data_256';(scenes/b).write_bytes((scenes/a).read_bytes())
            before=snapshot(root);base,assets,_,_=inventory(root)
            prior=leakage_report(legacy_scenes(root,base,42),assets)
            self.assertGreater(prior['by_domain']['Places']['cross_split_hashes'],0)
            self.assertEqual(prior['by_domain']['CUB']['cross_split_hashes'],0)
            original_counts=counts(base);out=Path(td)/'prepared';meta=prepare(root,out)
            audit=json.loads((out/'split_audit.json').read_text())
            self.assertEqual(audit['moved_bird_ids'],[]);self.assertEqual(meta['final_counts'],original_counts)
            self.assertEqual(audit['final']['cross_split_hashes'],0)
            self.assertEqual(snapshot(root),before)
            reference=None
            for corr in CORRS:
                f=pd.read_csv(out/f'A_corr{round(corr*100):03d}.csv');validate_frame(f,meta['assets'],corr)
                self.assertEqual(set(f.img_id),set(base.img_id));self.assertEqual(counts(f),original_counts)
                if reference is not None:
                    pd.testing.assert_frame_equal(f.drop(columns='place'),reference.drop(columns='place'))
                    pd.testing.assert_frame_equal(f[f.split!='train'],reference[reference.split!='train'])
                reference=f
                self.assertEqual(len(load(out/'protocol.json',corr)['train']),96)
            # Restore a genuinely leaky manifest assignment: the strict checker still fails.
            f=reference.copy();shared=f.iloc[0].land_file
            f.loc[f.split=='test','land_file']=shared
            with self.assertRaisesRegex(ValueError,'leakage'):validate_frame(f,meta['assets'],.65)

    def test_cub_train_val_duplicate_keeps_class_quotas(self):
        with tempfile.TemporaryDirectory() as td:
            root=self.make(td);base,_,_,_=inventory(root)
            train=base[(base.split=='train')&(base.y==1)].iloc[0]
            val=base[(base.split=='val')&(base.y==1)].iloc[0]
            images=root/'CUB_200_2011/images';(images/val.img_filename).write_bytes((images/train.img_filename).read_bytes())
            before=snapshot(root);out=Path(td)/'prepared';meta=prepare(root,out)
            frame=pd.read_csv(out/'A_corr095.csv')
            self.assertEqual(counts(frame),counts(base))
            self.assertEqual(len(set(frame[frame.img_id.isin([train.img_id,val.img_id])].split)),1)
            self.assertEqual(snapshot(root),before)

    def test_cub_official_train_test_duplicates_are_test_only(self):
        with tempfile.TemporaryDirectory() as td:
            root=self.make(td);base,_,_,_=inventory(root)
            train=base[(base.split=='train')&(base.y==1)].iloc[0]
            test=base[(base.split=='test')&(base.y==1)].iloc[0]
            images=root/'CUB_200_2011/images';(images/train.img_filename).write_bytes((images/test.img_filename).read_bytes())
            before=snapshot(root);out=Path(td)/'prepared';prepare(root,out)
            frame=pd.read_csv(out/'A_corr095.csv')
            self.assertTrue((frame[frame.img_id.isin([train.img_id,test.img_id])].split=='test').all())
            self.assertEqual(len(frame[frame.split=='test']),41)
            self.assertEqual(len(frame[frame.split=='val']),24)
            self.assertEqual(len(frame[frame.split=='train']),95)
            self.assertEqual(len(frame),160);self.assertEqual(snapshot(root),before)

    def test_conflicting_bird_labels_fail_and_sources_are_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root=self.make(td);images=root/'CUB_200_2011/images'
            (images/'002.Sparrow/2.jpg').write_bytes((images/'001.Albatross/1.jpg').read_bytes())
            before=snapshot(root);out=Path(td)/'prepared'
            with self.assertRaisesRegex(ValueError,'conflicting labels'):prepare(root,out)
            self.assertFalse(out.exists());self.assertEqual(snapshot(root),before)

    def test_duplicate_background_kinds_and_cross_domain_pin(self):
        with tempfile.TemporaryDirectory() as td:
            root=self.make(td)
            land=next((root/'places365/data_256/b/bamboo_forest').glob('*.jpg'))
            water=next((root/'places365/data_256/o/ocean').glob('*.jpg'))
            water.write_bytes(land.read_bytes()) # Same content under both categorical labels.
            bird=root/'CUB_200_2011/images/001.Albatross/121.jpg'
            other=next(p for p in (root/'places365/data_256/f/forest/broadleaf').glob('*.jpg'))
            other.write_bytes(bird.read_bytes()) # Exact RGB reuse in the two source domains.
            before=snapshot(root);out=Path(td)/'prepared';meta=prepare(root,out)
            audit=json.loads((out/'split_audit.json').read_text())
            self.assertEqual(audit['scene_group_split'][digest(bird)],'test')
            self.assertEqual(audit['final']['cross_split_hashes'],0)
            for corr in CORRS:
                frame=pd.read_csv(out/f'A_corr{round(corr*100):03d}.csv')
                validate_frame(frame,meta['assets'],corr)
            self.assertEqual(snapshot(root),before)

    def test_unused_train_candidate_is_distinguished_from_runtime_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            root=self.make(td);base,assets,_,_=inventory(root)
            old=legacy_scenes(root,base,42)
            train=old[old.split=='train'].iloc[[0]];test=old[old.split=='test'].iloc[[0]]
            frame=pd.concat([train,test]).reset_index(drop=True)
            frame.loc[0,'place']=0;frame.loc[1,'place']=1
            frame.loc[0,'water_file']=frame.loc[1,'water_file']
            self.assertGreater(leakage_report(frame,assets)['cross_split_hashes'],0)
            self.assertEqual(leakage_report(frame,assets,'factual')['cross_split_hashes'],0)
            self.assertEqual(leakage_report(frame,assets,'runtime')['cross_split_hashes'],0)

    def test_within_split_cycles_and_correlation_reuse_are_not_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            root=self.make(td);out=Path(td)/'one';other=Path(td)/'two'
            prepare(root,out);prepare(root,other)
            self.assertEqual((out/'protocol.json').read_bytes(),(other/'protocol.json').read_bytes())
            f=pd.read_csv(out/'A_corr095.csv');train=f[f.split=='train']
            self.assertLess(train.land_file.nunique(),len(train)) # Intentional scene cycling.
            audit=json.loads((out/'split_audit.json').read_text())
            self.assertEqual(audit['final']['cross_split_hashes'],0)
            self.assertEqual(audit['original_counts'],audit['final_counts'])

if __name__=='__main__':unittest.main()
