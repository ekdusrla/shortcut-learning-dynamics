"""Check original hashes, output consistency, and C2I identifiability bounds by exhaustive toy predictions."""
import ast
import hashlib
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
O=Path(__file__).resolve().parent;ROOT=O.parents[1]
module=ast.parse((O/'analyze_metrics.py').read_text())
function=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='c2i_bounds')
ns={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),'<c2i_bounds>','exec'),ns)
# Three classes already allow wrong->different wrong; enumerate all paired predictions.
cases=0;groups=0
for n in range(1,5):
 possible=defaultdict(set)
 for pairs in itertools.product(list(itertools.product(range(3),repeat=2)),repeat=n):
  A=sum(x==0 for x,y in pairs);B=sum(y==0 for x,y in pairs);F=sum(x!=y for x,y in pairs)
  C=sum(x==0 and y!=0 for x,y in pairs)
  possible[(A,B,F)].add(C);cases+=1
 for (A,B,F),choices in possible.items():
  low,high,*_=ns['c2i_bounds'](A/n,B/n,F/n,n)
  assert low==min(choices) and high==max(choices),(n,A,B,F,choices,low,high)
  groups+=1
for entry in json.loads((O/'source_manifest.json').read_text()):
 assert hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()==entry['sha256']
s=pd.read_csv(O/'summary_table.csv');d=pd.read_csv(O/'epoch_metrics.csv')
assert len(s)==6 and len(d)==306
for _,r in s.iterrows():
 block=d[(d['mode']==r['mode'])&(d.train_corr==r.train_corr)]
 assert block.epoch.tolist()==list(range(51))
 assert np.allclose(block.online_train_minus_val_pp,100*(block.train_acc-block.val_acc),equal_nan=True)
 assert (block.c2i_lower_count<=block.c2i_upper_count).all()
 assert np.allclose(block.c2i_lower_count,block.c2i_lower_count.round())
 assert np.allclose(block.c2i_upper_count,block.c2i_upper_count.round())
for f in (O/'figures').glob('*.png'):
 with Image.open(f) as im:im.verify()
assert len(list((O/'figures').glob('*.png')))==7
assert len(list((O/'figures').glob('*.pdf')))==7
for link in re.findall(r'\]\(([^)]+)\)',(O/'analysis_report.md').read_text()):assert (O/link).exists(),link
result={'success':True,'original_small_source_hashes_unchanged':True,'experiments':6,'history_rows':306,
        'png_figures':7,'pdf_figures':7,'c2i_bounds_exhaustive_toy_prediction_cases':cases,'distinct_aggregate_cases':groups,
        'scope':'Only saved metrics and CPU archive/feature inspection; no inference, GPU or new experiments'}
(O/'analysis_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
