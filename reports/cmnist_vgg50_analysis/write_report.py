"""Build Korean report from validated CSV/JSON; no original writes."""
from pathlib import Path
import json
import pandas as pd
O=Path(__file__).resolve().parent
s=pd.read_csv(O/'summary_table.csv'); audits=json.loads((O/'artifact_audit.json').read_text())
def pct(v):return f'{100*v:.2f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def ep(v):return '미도달' if pd.isna(v) else str(int(v))
text='''# Colored MNIST Shortcut Learning Dynamics 분석

분석 대상은 SLURM Array 153567의 6개 실행이다. 원본을 읽기 전용으로 사용했고 새 학습, 재추론, GPU 작업, sbatch/srun은 실행하지 않았다. 아래 수치는 실제 저장 Metrics와 그로부터 계산한 기술 통계다. 정확도·비율은 %, 차이는 percentage point(pp)로 표기한다.

## 1. 실험 구성 및 결과 파일 검증

` scripts/cmnist_vgg50/task.sh`와 `README.ko.md`에서 출발하여 지정된 6개 run과 `logs/cmnist_153567_{0..5}.{out,err}`만 확인했다. 공통 config는 VGG11_BN, Colored MNIST, seed 42, 50 epoch, 224px, batch 128, SGD(lr .01, momentum .9, weight decay .0001), scheduler none, AMP true, workers 4, pairing all, validation 5000, val/test corr .10, pretrained false, augment false, features/group 32다. Color mode와 train correlation 이외의 **저장된 모든 config 항목이 동일**하며 로그의 `[config]`와 config.json도 일치한다.

CF chunk 128은 task.sh의 명시적 인자다. config.json에는 해당 항목이 기록되지 않아 당시 실제 적용값을 독립 검증할 수 없다. 현재 코드의 기본값도 128이지만 실행 당시 소스 snapshot은 없어 전체 코드 동일성까지 소급 확정할 수 없다. 로그에는 torch 2.5.1+cu121, torchvision 0.20.1+cu121, RTX A2000 12GB가 기록되어 있다.

'''
text+=table(['Task','Mode','Train corr','History','Checkpoint','Feature','최종 Test'],[[a['task'],a['mode'],a['corr'],'0–50 (51행)','0–50 (51개)','0–50 (51개)', 'epoch 50 / 10000개'] for a in audits])
text+='''

6개 모두 TRAIN_SUCCESS, FINAL TEST 및 정상 종료 지점의 Run directory 로그가 있다. stderr는 HF Hub 비인증 다운로드 경고뿐이다. history의 Epoch 중복·누락 없음, validation 표본수 5000 고정, 그룹 count 고정, 주요 지표 유한값·범위, CF drop=원본 accuracy−CF accuracy, 20개 그룹 기반 WGA 재계산을 검사했다. Test는 10000개이며 epoch 50이다. latest.json도 모두 epoch 50을 가리킨다.

306개 checkpoint는 ZIP 구조·비어 있지 않은 구성원·파일 위치와 pickle 명령어에서 읽은 Epoch 메타데이터를 확인했다. 모델을 역직렬화하거나 tensor를 메모리에 적재하지 않았다. 따라서 모든 tensor payload의 CRC/수치 유한성이나 재추론 가능성까지 검증한 것은 아니다. 306개 Feature NPZ는 모든 배열을 파일별로 순차 읽어 압축 데이터 읽기 성공, 3200개 고유 sample, 100개 label×color 그룹당 32개, 6개 레이어 및 feature 유한성을 검증했다. 3200개는 merged WGA 20그룹이 아니라 원래 100개 색상 그룹 기준이다.

SLURM accounting은 sacct의 sandbox 제한 이후 재시도에서도 DB 연결 timeout으로 조회하지 못했다. 따라서 COMPLETED/ExitCode를 독립 확인했다는 주장은 하지 않는다. **실제 산출물과 종료 로그는 6개 학습·최종 평가 완료를 뒷받침하며, 저장된 Metrics 분석에는 누락이 없다.** 자세한 검증은 artifact_audit.json, 작은 원본 파일의 SHA-256은 source_manifest.json에 있다.

기존 analyze.py의 history 곡선 정의를 재사용하되 해당 함수가 원본 run에 그림을 쓰므로 직접 호출하지 않고 별도 스크립트로 옮겼다. evaluate.py는 분류/linear probe 재추론용이라 실행하지 않았다. 원본과 기존 학습 코드는 수정하지 않았다.

## 2. 지표 정의와 해석 기준

- Train accuracy는 각 학습 batch의 갱신 중 예측을 집계한 online 값이다. Validation은 해당 Epoch 모델 평가이므로 동일 조건 정확도로 직접 비교하지 않는다. epoch 0 Train은 원래 NaN이다.
- Validation은 epoch 0–50, Train은 1–50, Test는 **epoch 50만** 존재한다. Test 곡선을 생성하거나 보간하지 않고 마지막 점만 그렸다.
- Aligned는 y=color, conflicting은 y≠color다. pairing all의 WGA는 digit별 aligned/나머지 색상을 합친 **20개 그룹**의 최소 정확도이며, 100개 개별 셀 최소값이 아니다.
- Paired CF는 aligned↔해당 샘플의 alternate 색상 교환이다. val/test는 약 90% conflicting이므로 교환 후 대부분 aligned가 된다. CF accuracy가 높거나 drop이 음수라는 사실만으로 색상 의존성이 작다고 판단할 수 없다.
- All-color flip은 원래 색상 외 9가지 개입 전체에서 원래 예측과 달라진 비율이다. All-color accuracy와 color rate는 이 중 **개입 색상≠정답**인 부분만 집계한다. color rate는 예측=개입 색상 비율이다. All-color accuracy의 분모는 paired CF나 원본 accuracy와 다르므로 저장되지 않은 all-color accuracy drop을 같은 지표처럼 만들지 않았다.
- Flip 감소는 같은 오답을 유지해도 발생한다. conflicting/All-color accuracy 증가, color rate 감소를 함께 확인했다. 특히 무학습 epoch 0의 낮은 flip은 Core 학습 증거가 아니다.
- corr .10은 90%가 비정렬된 환경이며 학습보다 aligned가 현저히 적다. 다만 **10-class pairing all에서는 독립 색상의 chance aligned 확률도 .10**이다. 비정렬 색상이 9가지에 분산되므로 이진 문제의 결정론적 역상관과 같지 않다. 이 환경을 “항상 반대 label을 가리키는 색상”으로 해석해서는 안 된다.

## 3. 6개 실험의 핵심 결과

최종 Epoch 50 결과다. 상세 Test/Validation 지표와 초기·중기·후기 평균은 summary_table.csv에 있다.

'''
text+=table(['Mode','corr','Train','Val','Test','Val conflicting','Val WGA','Val CF flip','Val all flip','Val all acc','Val color rate'],[[r['mode'],r.train_corr]+[pct(r[k]) for k in ['train_e50_acc','val_e50_acc','test_e50_acc','val_e50_conflicting_acc','val_e50_wga','val_e50_cf_flip_rate','val_e50_cf_all_flip_rate','val_e50_cf_all_acc','val_e50_cf_all_color_rate']] for _,r in s.iterrows()])
text+='''

## 4. Epoch별 Shortcut Learning Dynamics — H1

초기(1–5), 중기(6–25), 후기(26–50)를 공통 구간으로 정의했다. epoch 0은 무학습 baseline이므로 학습 후 감소의 기준에 섞지 않는다. 아래는 각 구간의 all-color flip 평균이다.

'''
text+=table(['Mode','corr','초기 mean','중기 mean','후기 mean','e1→e50 all flip','e1→e50 conflicting acc','Flip 증가 step/49','과거 최저 대비 최대 반등(pp)'],[[r['mode'],r.train_corr]+[pct(r[f'{w}_cf_all_flip_rate_mean']) for w in ['early_1_5','middle_6_25','late_26_50']]+[pct(r.val_e1_cf_all_flip_rate)+'→'+pct(r.val_e50_cf_all_flip_rate),pct(r.val_e1_conflicting_acc)+'→'+pct(r.val_e50_conflicting_acc),int(r.cf_all_flip_rate_increasing_steps),f'{r.cf_all_flip_rate_max_rebound_pp:.2f}'] for _,r in s.iterrows()])
text+='''

**관찰:** 6개 모두 초기→중기→후기의 평균 flip이 낮아지고, epoch 1→50에서 conflicting 및 all-color accuracy가 높아진다. .95에서 초기 변화가 가장 크고 중후기에도 개선이 이어진다. Train은 모두 100%에 이르지만 validation/test에는 correlation별 격차가 남는다. .65/.80은 초기부터 높은 강건 정확도를 보여 .95보다 개선 여지가 작다.

**반대 관찰:** epoch 0→1에서는 all-color flip이 증가하는 조건이 있으며 foreground는 세 조건 모두 증가한다. 이후에도 모든 조건에서 flip 증가 step이 발생한다. 예를 들어 foreground .65는 e3→e5에서 all-color flip 2.01→3.17%, conflicting accuracy 97.06→95.97%로 동반 악화된다. background .95의 e30→e40도 all-color flip 4.83→5.22%, all-color accuracy 95.99→95.64%로 반등한다. 일시 변동을 지속적인 후기 악화로 확대 해석하지 않는다.

**판정:** H1은 “학습이 진행되며 전반적인 색상 의존성이 감소한다”는 약한 형태로 지지된다. “매 Epoch 지속적/단조 감소” 또는 “무학습 시점부터 감소”라는 강한 형태는 지지되지 않는다. 최종에도 flip/color-following이 0이 아니므로 완전한 불변성을 주장할 수 없다.

## 5. Correlation 비교 및 Core 학습 Proxy — H2

Core representation 학습 시점은 직접 관측하지 않았다. 행동 기반 proxy를 모든 실행에 동일하게 적용했다: **conflicting accuracy≥95%, all-color conflicting CF accuracy≥95%, all-color flip≤5%를 연속 3 Epoch 만족하는 구간의 첫 Epoch**. epoch≥1만 허용하며 미래 2개 관측으로 지속 여부를 확인하는 회고적 기준이다. 성능과 불변성을 함께 요구하며, threshold 90%(flip≤10%), 98%(flip≤2%)도 같은 규칙으로 민감도를 확인했다. 이는 사전에 사용할 수 있는 명시적인 공통 기준이지만 이번 결과를 본 뒤 설정했으므로 preregistered 기준은 아니다. 95% 미도달은 Core를 전혀 배우지 않았다는 뜻이 아니다.

'''
text+=table(['Mode','corr','90% proxy','95% proxy','98% proxy','e50 paired CF acc','e50 CF drop(pp)'],[[r['mode'],r.train_corr,ep(r.proxy_90_first_epoch),ep(r.proxy_95_first_epoch),ep(r.proxy_98_first_epoch),pct(r.val_e50_cf_acc),pct(r.val_e50_cf_acc_drop)] for _,r in s.iterrows()])
text+='''

**관찰:** 두 mode 모두 .95의 강건성 개선 기준 도달이 .80/.65보다 늦다. 특히 .95에서는 높은 초기 Train 정확도와 낮은 conflicting/all-color 정확도, 높은 color rate가 공존한다. 이는 초기 색상 의존과 이후 강건 행동 개선의 증거다. 단순히 Train이 높아졌기 때문이라는 판정이 아니다. threshold에 따라 세부 도달 시점과 약한 상관 두 조건의 차이는 달라진다.

최종 all-color flip은 두 mode 모두 .95>.80>.65이고 all-color accuracy는 반대 순서다. Test 정확도 역시 두 mode에서 .65>.80>.95다. 후반 곡선은 모두 고성능 영역으로 접근하지만 완전히 수렴하지 않으며 특히 .95의 잔여 민감도가 더 크다. 동일 validation 표본을 사용한 기술적 비교로 해석한다.

**판정:** H2는 본 단일 seed의 두 mode에서 강한 상관(.95)이 강건 행동 개선을 지연시킨다는 proxy 수준으로 지지된다. 실제 Core representation 형성 시점, 변화의 인과 메커니즘, 일반적인 correlation–delay 법칙을 증명한 것은 아니다.

## 6. Background / Foreground 비교

아래는 foreground−background의 최종 차이다. 정확도 양수는 foreground 우세, flip 양수는 foreground 민감도가 높음을 뜻한다.

'''
diff=[]
for c in [.95,.80,.65]:
 b=s[(s['mode']=='background')&(s.train_corr==c)].iloc[0]; f=s[(s['mode']=='foreground')&(s.train_corr==c)].iloc[0]
 diff.append([c]+[f'{100*(f[k]-b[k]):+.2f}' for k in ['test_e50_acc','val_e50_conflicting_acc','val_e50_cf_all_acc','val_e50_cf_all_flip_rate']]+[ep(b.proxy_95_first_epoch),ep(f.proxy_95_first_epoch)])
text+=table(['corr','ΔTest(pp)','ΔVal conflict(pp)','ΔVal all acc(pp)','ΔVal all flip(pp)','BG 95% epoch','FG 95% epoch'],diff)
text+='''

同じ .95에서 foreground는 e1 all-color flip 82.25%로 background 91.61%보다 낮고 conflicting accuracy는 23.65%로 background 12.64%보다 높다. 그러나 e50에서는 foreground의 all-color flip이 더 높고 Test는 더 낮다. 따라서 “한 위치의 shortcut이 언제나 더 강하다”는 결론은 부적절하다. .65의 최종 차이는 작으며 seed 반복 없이 위치 효과의 통계적 유의성을 말할 수 없다. 두 mode 모두 H1의 전반적 감소와 H2의 강한 상관 지연 패턴은 관찰된다.

## 7. 시각화

그림은 관측값만 연결했고 누락 Epoch를 보간하지 않았다. 이 실행에는 validation Epoch 누락이 없다. epoch 0 Train NaN은 빈 값으로 유지했다. Test는 e50 단일 점이다. 축은 백분율, drop 축은 pp다. 확대 범위가 서로 다른 panel의 기울기를 시각적 크기만으로 비교하지 않는다.

1. [Train / Validation / Test](figures/01_accuracy.png)
2. [Aligned / Conflicting / WGA](figures/02_groups.png)
3. [Paired / All-color Flip](figures/03_flip_rates.png)
4. [CF Accuracy / Drop / All-color Accuracy](figures/04_counterfactual.png)
5. [Correlation별 색상 민감도](figures/05_correlation_dependence.png)
6. [Background vs Foreground](figures/06_background_vs_foreground.png)

각 PNG와 동일한 PDF를 제공한다. mode별 세 correlation을 같은 축에 표시했고 mode 비교에서는 같은 correlation을 같은 축에 표시했다.

## 8. 결과 해석의 한계와 추가 분석

seed 42 한 번씩이므로 seed 간 분산, 신뢰구간, 유의성 또는 모집단 일반화를 제시하지 않는다. Epoch를 독립 반복실험으로 간주하지 않는다. Validation의 반복 관측, 동일 digit의 여러 개입도 독립 표본이 아니다. threshold 도달 Epoch는 epoch 해상도, 3회 지속 조건, threshold에 의존하며 학습 시작 순간을 식별하지 않는다. 측정 간격이 한 Epoch라 첫 Epoch 내의 shortcut 형성/회복 순서는 알 수 없다.

색상 개입은 digit 내용은 유지하지만 색 영역의 시각적 대비 등도 바꿀 수 있다. all-color robustness는 Core 활용에 부합하는 행동 증거이나 내부 표상의 유일한 인과 설명은 아니다. 큰 특징 snapshot이 존재한다고 Core 또는 색상 정보 사용이 검증된 것은 아니다. Probe의 정보 복호화 가능성과 실제 분류기의 정보 사용도 구별해야 한다. pretrained/augmentation은 false이며 결과는 본 데이터·모델·학습 설정으로 제한된다.

Grad-CAM, linear probe 결과 및 중간 Test는 저장되지 않았다. 후속 작업·비용·비GPU 대안은 [missing_artifacts.md](missing_artifacts.md)에 정리했다. 현재 소스와 실행 시점 소스 일치 여부, checkpoint tensor 전체 무결성, SLURM accounting은 별도 한계다.

## 9. 최종 결론

H1은 장기 추세로 지지되며 단조 감소로는 지지되지 않는다. .95의 all-color flip은 background 91.61→4.82%, foreground 82.25→5.77%(e1→50)로 감소하고 all-color accuracy는 각각 12.86→96.03%, 23.41→95.33%로 증가한다. Flip 감소 단독이 아닌 강건 정확도와 색상 추종 감소가 함께 관찰된다.

H2는 동일 proxy 기준에서 강한 correlation의 지연으로 지지된다. 정확한 95% 도달 Epoch는 위 표를 참조한다. 최종 Test는 background .95/.80/.65가 96.71/98.66/99.08%, foreground는 95.93/98.55/99.01%이며 강한 상관의 성능 차이가 남는다. 두 mode의 일관된 correlation 패턴을 확인했으나 위치별 절대 우열과 통계적 일반화는 확정할 수 없다.

## 재현 및 파일

프로젝트 루트에서 다음 CPU 명령으로 검증·그림·표를 생성하고 보고서를 재생성한다.

```bash
MPLCONFIGDIR=/tmp/cmnist-analysis-mpl /data/dada0228/conda_envs/shortcut-xai/bin/python reports/cmnist_vgg50_analysis/analyze_metrics.py
/data/dada0228/conda_envs/shortcut-xai/bin/python reports/cmnist_vgg50_analysis/write_report.py
```

summary_table.csv는 비율을 0–1 단위로 저장하고 `_pp`는 percentage point 단위다. proxy 도달 미관측은 빈 칸이다. epoch_metrics.csv는 원본 history를 mode/correlation과 함께 합친 파생 테이블이다. source_manifest.json은 주요 원본 Metrics/config/log의 크기와 SHA-256을 저장한다. 모든 생성 파일은 이 디렉터리에만 저장한다.
'''
text=text.replace('同じ','동일한').replace('` scripts/','`scripts/')
extra='\n\n최종 Test의 강건성 지표도 아래에 제시한다. epoch 50, % 단위이며 drop만 pp다.\n\n'
extra+=table(['Mode','corr','Aligned','Conflict','WGA','CF flip','CF acc','CF drop','All flip','All acc','Color rate'],[[r['mode'],r.train_corr]+[pct(r['test_e50_'+m]) for m in ['aligned_acc','conflicting_acc','wga','cf_flip_rate','cf_acc','cf_acc_drop','cf_all_flip_rate','cf_all_acc','cf_all_color_rate']] for _,r in s.iterrows()])+'\n\n'
text=text.replace('## 4. Epoch별',extra+'## 4. Epoch별')
dynamics='\n\nAligned/conflicting 격차와 paired CF, WGA의 초기→최종 변화도 함께 확인했다. 아래는 epoch 1→50이며 % 단위다.\n\n'
dynamics+=table(['Mode','corr','Aligned','WGA','Paired flip','Paired CF acc','CF drop(pp)','All-color color rate'],[[r['mode'],r.train_corr]+[pct(r['val_e1_'+m])+'→'+pct(r['val_e50_'+m]) for m in ['aligned_acc','wga','cf_flip_rate','cf_acc','cf_acc_drop','cf_all_color_rate']] for _,r in s.iterrows()])+'\n\n'
text=text.replace('## 5. Correlation 비교',dynamics+'## 5. Correlation 비교')

text=text.replace('정확한 95% 도달 Epoch는 위 표를 참조한다.','95% proxy 도달은 background .65/.80/.95에서 e2/e3/e46, foreground에서 e2/e3/50 Epoch 내 미도달이다.')
(O/'analysis_report.md').write_text(text)
