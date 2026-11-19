"""CPU-only collection after all three manually submitted CAM jobs succeed."""
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def main():
 p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 frames=[];coverage=[]
 for code in ('095','080','065'):
  root=a.analysis/f'gradcam/A_corr{code}';assert (root/'CAM_SUCCESS').exists()
  v=json.loads((root/'validation.json').read_text());assert v['success']
  d=pd.read_csv(root/'region_metrics.csv');assert set(d.kind)=={'factual','cf'}
  for r in d.itertuples():
   heat=np.load(root/f'e{r.epoch:03d}_id{r.img_id}_{r.kind}.npy');assert heat.shape==(224,224) and np.isfinite(heat).all()
  coverage.extend(dict(condition='A_corr'+code,epoch=e,kind=k,correct_n=int(b.correct.sum()),incorrect_n=int((~b.correct).sum()),aligned_n=int(b.aligned.sum()),conflicting_n=int((~b.aligned).sum()),zero_maps=int(b.zero_cam.sum())) for (e,k),b in d.groupby(['epoch','kind']))
  frames.append(d)
 all_data=pd.concat(frames);assert all_data.groupby(['condition','epoch']).img_id.nunique().eq(8).all()
 assert all_data.groupby('condition').img_id.apply(lambda x:tuple(sorted(set(x)))).nunique()==1
 a.output.mkdir(parents=True);all_data.to_csv(a.output/'region_metrics.csv',index=False);pd.DataFrame(coverage).to_csv(a.output/'coverage.csv',index=False)
 summary=all_data.groupby(['condition','epoch','kind'])[['bird_mass_fraction','bird_area_normalized_enrichment','bird_mean','background_mean']].mean().reset_index();summary.to_csv(a.output/'summary.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,key in zip(axes,('bird_mass_fraction','bird_area_normalized_enrichment')):
  for (c,k),b in summary.groupby(['condition','kind']):ax.plot(b.epoch,b[key],label=f'{c} {k}',marker='o')
  ax.set(xlabel='Epoch (fixed/candidate points)',ylabel=key,title='Fixed 8 Validation images, true-class CAM');ax.legend(fontsize=6)
 fig.tight_layout();fig.savefig(a.output/'cam_region_summary.png',dpi=160);plt.close(fig)
 # Same IDs, shared fixed epochs. Per-map normalized displays, not raw activation amplitudes.
 for c in ('095','080','065'):
  root=a.analysis/f'gradcam/A_corr{c}';d=all_data[all_data.condition=='A_corr'+c]
  for img_id in sorted(d.img_id.unique()):
   epochs=sorted(d.epoch.unique());fig,axes=plt.subplots(len(epochs),2,figsize=(12,2.6*len(epochs)),squeeze=False)
   for row,e in enumerate(epochs):
    for col,kind in enumerate(('factual','cf')):
     axes[row,col].imshow(Image.open(root/f'e{e:03d}_id{img_id}_{kind}.png'));axes[row,col].axis('off');axes[row,col].set_title(f'A{c} E{e} id{img_id} {kind}')
   fig.tight_layout();fig.savefig(a.output/f'comparison_A{c}_id{img_id}.png',dpi=100);plt.close(fig)
 (a.output/'cam_integration.md').write_text('''# Grad-CAM 후속 결과 수집

실제 CAM_SUCCESS 3개를 확인하고 결과를 수집했다. `coverage.csv`에서 각 Epoch의 correct/incorrect와 aligned/conflicting, zero-map 수를 먼저 확인한다. 고정8개 균형 Validation 사례 분석이며 전체1199개의 평균 주목이라고 일반화하지 않는다.

![영역별 요약](cam_region_summary.png)

`comparison_*.png`는 동일 ID의 시간 비교다. 정답 target, features.18, 원본/CF, map별0~1 정규화이므로 절대 활성 크기 변화는 해석하지 않는다. mask mass 및 면적 보정은 상대적 위치 요약이며 인과적 사용 증거가 아니다. CAM이 새 영역에 모여도 Core 활용 시점을 확정하지 않는다. 행동/Probe와 함께 검토하며 이번 결과만으로 H1/H2의 확정을 자동 생성하지 않았다. 기존 CPU 보고서를 덮어쓰지 않았다.
''')
 (a.output/'validation_results.json').write_text(json.dumps(dict(success=True,conditions=3,same_ids=True,rows=len(all_data),h1_h2_scope='descriptive CAM supplements only; no causal or transition confirmation'),indent=2))
if __name__=='__main__':main()
