import unittest
import numpy as np
import pandas as pd
from scripts.waterbirds_followup.cam import region_metrics
from scripts.waterbirds_followup.probes import transitions,split_content
from evaluate import DecodedCorrelation
import torch
class Tests(unittest.TestCase):
 def test_regions(self):
  mask=np.array([[1,0],[1,0]],float)
  self.assertEqual(region_metrics(np.ones((2,2)),mask)['bird_area_normalized_enrichment'],1)
  self.assertTrue(region_metrics(np.zeros((2,2)),mask)['zero_cam'])
  self.assertIsNone(region_metrics(np.zeros((2,2)),mask)['bird_mass_fraction'])
  with self.assertRaises(ValueError):region_metrics(np.ones((3,3)),mask)
 def test_censoring_and_baseline(self):
  d=pd.DataFrame(dict(condition=['a']*51,layer=['avgpool']*51,target=['core']*51,epoch=range(51),accuracy=[.8]+[.5]*50))
  t=transitions(d)
  self.assertEqual(t[(t.threshold==.8)&(t.duration==1)].first_epoch.iloc[0],0)
  self.assertTrue(t[t.duration==3].first_epoch.isna().all())
 def test_decoded_definition(self):
  m=DecodedCorrelation(1,2)
  with torch.no_grad():
   m.linear.weight.copy_(torch.tensor([[-1.],[1.]]));m.linear.bias.zero_()
  result=m.evaluate(torch.tensor([[-1.],[1.]]),torch.tensor([0,1]))
  self.assertEqual(result['accuracy'],1);self.assertEqual(result['decoded_correlation'],1)
  result=m.evaluate(torch.tensor([[-1.],[1.]]),torch.tensor([1,0]))
  self.assertEqual(result['decoded_correlation'],-1)
 def test_content_split(self):
  rows=[];assets={}
  for i in range(80):
   bird=f'b{i}.jpg';land=f'l{i//2}.jpg';water=f'w{i//2}.jpg'
   assets['CUB_200_2011/images/'+bird]='bird'+str(i)
   assets['places365/data_256/'+land]='land'+str(i//2)
   assets['places365/data_256/'+water]='water'+str(i//2)
   rows.append(dict(img_id=i,img_filename=bird,land_file=land,water_file=water,y=(i//2)%2,place=i%2))
  (fit,score),frame,components=split_content({'assets':assets},pd.DataFrame(rows),np.arange(80))
  self.assertFalse(set(components[fit])&set(components[score]))
  for part in (fit,score):
   self.assertEqual(len(set(frame.iloc[part].groupby(['y','place']).size())),1)
if __name__=='__main__':unittest.main()
