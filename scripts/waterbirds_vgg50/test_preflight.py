import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from scripts.waterbirds_vgg50.check_data import inspect_root, probe_range
from scripts.waterbirds_vgg50.protocol import prepare, load, EVAL_CORR, CORRS, digest
from scripts.waterbirds_vgg50.test_waterbirds import fixture

class Response(io.BytesIO):
    def __init__(self, body=b'x'*512, status=206, header='bytes 0-511/10000', **headers):
        super().__init__(body)
        self.status=status
        self.headers={'Content-Range':header, **headers}
        self.read_called=False
    def read(self, n=-1):
        self.read_called=True
        assert 0<=n<=513
        return super().read(n)

class Tests(unittest.TestCase):
    def test_range_contract(self):
        def opener(req, timeout):
            self.assertEqual(req.get_header('Range'),'bytes=0-511')
            return Response()
        self.assertEqual(probe_range('https://example.invalid/archive',opener)['probe_bytes'],512)
        for response in (Response(status=200),Response(header='bytes 512-1023/10000'),
                         Response(**{'Content-Encoding':'gzip'}),Response(**{'Content-Length':'99999'})):
            with self.assertRaises(ValueError):
                probe_range('https://example.invalid/archive',lambda *a,**k:response)
            self.assertFalse(response.read_called)
        for body in (b'x'*511,b'x'*513):
            with self.assertRaises(ValueError):
                probe_range('https://example.invalid/archive',lambda *a,**k:Response(body))

    def test_missing_and_partial_data_not_ready(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(inspect_root(td)['ready'])
            root=Path(td); fixture(root)
            marker=root/'places365/data_256/b/bamboo_forest/.complete'
            marker.write_text('5000')
            before={str(p):p.stat().st_mtime_ns for p in root.rglob('*')}
            result=inspect_root(root)
            self.assertFalse(result['ready'])
            self.assertEqual(result['categories']['b/bamboo_forest']['jpeg_count'],20)
            self.assertEqual(before,{str(p):p.stat().st_mtime_ns for p in root.rglob('*')})

    def test_legacy_downloader_failure_modes(self):
        import data
        import tarfile
        # Reproduce ignored Range using only an in-memory tar header, no network.
        response=io.BytesIO(tarfile.TarInfo('data_256/b/bamboo_forest/0001.jpg').tobuf())
        response.status=200
        with patch('data.urllib.request.urlopen',return_value=response):
            offset,name=data._tar_header_at('https://example.invalid/archive',512)
        self.assertEqual(offset,512)  # Actual 200 response begins at zero: legacy code fails to reject it.
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)/'data_256/b/bamboo_forest';folder.mkdir(parents=True)
            (folder/'.complete').write_text('5000')
            with patch('data.urllib.request.urlopen') as opener:
                data.fetch_places_category(td,'bamboo_forest')
            opener.assert_not_called()  # Legacy marker skips even an empty category.
            self.assertEqual(list(folder.glob('*.jpg')),[])

    def test_eval050_legacy_compatibility_and_leakage(self):
        self.assertEqual(EVAL_CORR,.5)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'raw';root.mkdir();fixture(root)
            out=Path(td)/'manifest'; meta=prepare(root,out)
            self.assertEqual(meta['val_corr'],.5);self.assertEqual(meta['test_corr'],.5)
            reference=None
            for corr in CORRS:
                ds=load(out/'protocol.json',corr)
                frame=pd.read_csv(out/f'A_corr{round(corr*100):03d}.csv')
                for split in ('val','test'):
                    f=frame[frame.split==split]
                    for y in (0,1):
                        sub=f[f.y==y]
                        self.assertLessEqual(abs((sub.place==y).sum()-.5*len(sub)),.5)
                    if split=='test': self.assertEqual(float((f.y==f.place).mean()),.5)
                if reference is not None:
                    pd.testing.assert_frame_equal(frame.drop(columns='place'),reference.drop(columns='place'))
                    pd.testing.assert_frame_equal(frame[frame.split!='train'],reference[reference.split!='train'])
                reference=frame
                for first,second in (('train','val'),('train','test'),('val','test')):
                    a=frame[frame.split==first];b=frame[frame.split==second]
                    self.assertTrue(set(a.img_id).isdisjoint(b.img_id))
                    a_scenes=set(a.land_file)|set(a.water_file)
                    b_scenes=set(b.land_file)|set(b.water_file)
                    self.assertTrue(a_scenes.isdisjoint(b_scenes))
            # Original v1 .50 manifests remain usable without rewriting their contents.
            legacy=dict(meta,version=1);legacy.pop('val_corr');legacy.pop('test_corr')
            path=out/'legacy.json';path.write_text(json.dumps(legacy));before=digest(path)
            load(path,.95);self.assertEqual(before,digest(path))
            with patch('scripts.waterbirds_vgg50.protocol.EVAL_CORR',.10):
                prepare(root,Path(td)/'cancelled010')
            with self.assertRaisesRegex(ValueError,'0.50'):
                load(Path(td)/'cancelled010/protocol.json',.95)
            # Duplicate scene aliases are grouped before splitting; actual leakage still rejected.
            a=reference[reference.split=='train'].iloc[0].land_file
            b=reference[reference.split=='test'].iloc[0].land_file
            scenes=root/'places365/data_256'
            (scenes/b).write_bytes((scenes/a).read_bytes())
            repaired=Path(td)/'grouped';prepare(root,repaired)
            audit=json.loads((repaired/'split_audit.json').read_text())
            self.assertTrue(any(len(rows)>1 for rows in audit['scene_duplicates'].values()))
            self.assertEqual(audit['final']['cross_split_hashes'],0)
            load(repaired/'protocol.json',.95)

if __name__=='__main__': unittest.main()
