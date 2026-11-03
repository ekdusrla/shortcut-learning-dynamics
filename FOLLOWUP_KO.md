# Epoch별 shortcut dynamics 분석 실행 가이드

기존 0.95 학습 모델을 재학습하지 않고 저장된 checkpoint에서 분석한다. 모든 명령은 저장소 루트에서 실행한다. 기존 `reports/epoch100`, 원래 checkpoint·history·feature, 앞선 분석 결과는 보존한다.

## 파일 목록과 검증 범위

| 파일 | 역할 |
|---|---|
| `followup_common.py` | 고정 probe manifest 생성, 원본/분할 검증, 공통 checkpoint 로딩·로그 |
| `audit_followup.py` | 실제 checkpoint·기존 feature NPZ 구조와 split 조사 |
| `probe_dynamics.py` | Epoch × Layer × Model × Target별 feature 추출과 probe 학습·평가 |
| `gradcam_dynamics.py` | 고정 정답 target의 동일 원본 counterfactual Grad-CAM, grid와 지표 |
| `prepare_correlations.py` | 두 모델의 0.80/0.65 실험 설정·Bash/PowerShell 명령 생성만 수행 |
| `integrate_dynamics.py` | 기존 history와 새 probe/CAM 결과 통합, CSV·그래프·한국어 보고서 |
| `validate_followup.py` | 저장한 probe 예측의 NumPy 독립 계산 및 결과 artifact 검증 |
| `test_followup.py` | 오프라인 데이터 균형·누수·가중치 고정·출력 보존·지표·실험 설정 테스트 |
| `configs/followup_corr/` | 실제 준비한 seed42의 네 실험 설정과 실행 명령. 학습 미실행 |
| `reports/followup_v1/` | 실제 epoch100의 새 독립 평가와 기존 history 통합 결과 |

실제 CPU 검증: 모델별 train/test 각각 3,200개 입력, ResNet 5개/VGG 6개 stage, probe 22행, CAM 20쌍. 오프라인 테스트 7개와 저장 결과 검증을 통과했다. GPU 실행 경로는 구현했지만 로컬 CUDA가 없어 실제 GPU 검증은 하지 않았다. 나머지 9개 epoch × 2개 모델, 서버 원본 snapshot split 대조, 추가 correlation 학습 및 H2 검증은 미실행이다.

## 이번 구현의 기준

- 분석 epoch: **0, 1, 3, 5, 10, 20, 30, 50, 70, 100**.
- 기본 probe 레이어는 기존 snapshot의 **모든 stage**: ResNet의 layer1–4/avgpool, VGG의 다섯 MaxPool 출력/avgpool. `--all-spatial-layers`를 추가하면 모든 Conv/BatchNorm/ReLU/pool 모듈도 포함한다. 모든 feature는 spatial mean pooling 후 probe에 넣는다.
- Grad-CAM 레이어는 ResNet `layer4`, VGG `features.25`. 같은 원본의 정답 숫자를 모든 epoch/모델에서 target으로 고정한다.
- 새 output 디렉터리는 반드시 없어야 한다. 기존 output을 지정하면 실패한다. 중간 실행 실패 폴더도 삭제하지 않고 다른 이름으로 재시도한다.
- 요청 epoch가 없으면 기본 동작은 실패다. `--allow-missing`을 명시한 경우에만 가용 checkpoint로 부분 분석하며 누락 목록과 `partial` 상태를 저장한다.
- CPU/GPU 공통 코드이며 `--device auto/cpu/cuda/mps`, feature batch size, workers, CPU threads를 제어할 수 있다.

## 코드 기준과 기존 NPZ 판정

작업 시작 브랜치 `feat/epoch100-analysis`는 구형 2-class 기준이었다. 로컬 `main`의 검증된 10-class `data.py`, `shortcut_experiment.py`, `evaluate.py`, `gradcam.py`, `download_waterbirds.sh`를 새 `feat/followup-dynamics` 브랜치에 반영했다. 이 파일들에 새로운 모델 구조나 학습 규칙을 도입한 것은 아니다. GitHub의 후속 PR에서는 최신 10-class 기준과 비교한다.

학습 코드의 `save_feature_snapshot`은 `ds['val']`의 그룹별 고정 표본을 저장한다. `next` pairing의 20개 그룹에 32개씩이면 640행이다. NPZ 자체에는 원본 split 정보가 없으며, `layers/labels/spurious/indices/f0...`만으로 독립적인 probe 학습/평가 구성이 확보되지 않는다. 실제 서버 파일은 아래 audit의 `--verify-datasets`로 split별 index/label/color를 대조한다. 기존 640행 파일을 무작위로 행 분할하여 새 독립 실험처럼 사용하지 않는다.

`color-pairing=next/random`은 숫자마다 두 색상만 허용하므로 correlation 0.1도 독립적이지 않다. `all`에서는 P(s=y)=0.1, P(s≠y의 각 색)=0.9/9=0.1로 생성 확률상 독립적이다. 유한 표본의 정확한 균형까지 보장하지는 않는다.

새 protocol은 모델 validation에서 숫자별 원본 32개를 probe 학습용으로, 공식 test에서 숫자별 원본 32개를 평가용으로 선택한다. **각 원본을 10색 모두로 칠해 split마다 3,200개 입력, 각 숫자×색상 조합마다 32개**를 만든다. 원본 source/index를 기준으로 분리하며 픽셀 SHA256도 기록한다. 같은 원본을 다른 색으로 칠한 입력을 학습·평가 양쪽에 넣지 않는다. 모델의 기존 학습 데이터와 색상 배정은 바꾸지 않는다.

`--per-digit`은 늘릴 수 있지만 held-out validation의 해당 숫자 수보다 많으면 실패한다. 생성된 protocol JSON을 보관하고 모든 epoch/모델에서 재사용한다. 다른 모델 seed의 validation 분할은 달라질 수 있으므로 seed별 reference run으로 별도 protocol을 만들고 별도로 분석한다. 코드가 probe 학습 원본과 해당 모델의 training split이 겹치면 실행을 거부한다.

## 서버: 기존 checkpoint 분석

서버의 Python 환경에 `requirements.txt` 의존성이 설치되어 있어야 한다. 실제 run 이름이 다르면 경로만 바꾼다. 아래 명령은 Bash 기준이며 PowerShell에서도 줄 연결 없이 한 줄씩 실행할 수 있다.

```bash
cd /home/dada0228/shortcut-learning-dynamics

python -m unittest test_followup -v

python audit_followup.py --runs runs/resnet18_corr095_seed42_e100 runs/vgg11_corr095_seed42_e100 --output outputs/followup_v1/audit --verify-datasets

python followup_common.py --reference-run runs/resnet18_corr095_seed42_e100 --output outputs/followup_v1/protocol.json --per-digit 32 --seed 42

python probe_dynamics.py --runs runs/resnet18_corr095_seed42_e100 runs/vgg11_corr095_seed42_e100 --manifest outputs/followup_v1/protocol.json --output outputs/followup_v1/probes --device auto --batch-size 128 --workers 4

python gradcam_dynamics.py --runs runs/resnet18_corr095_seed42_e100 runs/vgg11_corr095_seed42_e100 --manifest outputs/followup_v1/protocol.json --output outputs/followup_v1/gradcam --device auto --samples-per-digit 1

python integrate_dynamics.py --runs runs/resnet18_corr095_seed42_e100 runs/vgg11_corr095_seed42_e100 --probe-results outputs/followup_v1/probes --gradcam-results outputs/followup_v1/gradcam --output outputs/followup_v1/integrated
```

최초 다운로드 없이 기존 Hugging Face 캐시만 쓸 때는 audit/prepare/probe/Grad-CAM 명령에 `--offline`을 붙인다. 외부 캐시의 lock 파일 접근 권한은 필요하다. `--epochs 0 1 3 ...`으로 분석 epoch를 바꿀 수 있다. 기본값은 위의 10개다.

기본 checkpoint 경로는 `<run>/checkpoints/epoch_XXX.pt`이며 `<run>/epoch_XXX.pt`, `<run>/*_epoch_XXX.pt`도 찾는다. 같은 epoch의 후보가 여러 개이면 임의 선택하지 않고 실패한다. checkpoint 내부 epoch·10-class head·이미지 크기·color mode 및 원본 분할도 확인한다.

큰 실험을 작업 단위로 나누려면 `--epochs 0 1 3` 등으로 별도 output 디렉터리를 만들고 통합할 때 `--probe-results job1 job2`, `--gradcam-results job1 job2`로 넘긴다. 겹치는 run/epoch/layer/target 행, 다른 protocol, 다른 probe hyperparameter를 섞으면 실패한다. 이 방식으로 완료된 job을 재실행하지 않고 나머지 epoch를 처리할 수 있다.

## 지표와 해석

- Core accuracy: 숫자 10-class probe 정확도.
- Shortcut accuracy: 색상 10-class probe 정확도.
- `chance_adjusted_accuracy = (accuracy − 0.1)/0.9`: 균등 추측 0, 완전 정답 1, chance 미만은 음수. **Pearson correlation이 아니다.**
- `legacy_decoded_correlation = 2×accuracy−1`: 기존 결과 호환용이며 10-class chance는 −0.8이다.
- Probe는 `evaluate.DecodedCorrelation`을 재사용한다. 모델은 freeze/eval, probe만 AdamW로 100 epochs 학습한다. LR 0.001, weight decay 0.0001, batch 256, seed 42. 모든 layer/target/epoch마다 probe RNG를 재설정한다. 원래 feature scale을 유지하며, 미학습 정보와 probe 미수렴을 같은 의미로 취급하지 않는다.
- 기본 독립 probe에는 `next` 학습에서 해당 숫자와 함께 보지 못했던 8개 색상 조합이 포함된다. 기존 2색 기반 probe·paired flip rate와 평가 분포가 다르다. 이전 epoch100 결과와 새 결과를 같은 추세선으로 연결하지 않는다.
- Grad-CAM의 aligned 색상은 y, counterfactual 색상은 (y+1)%10이다. 같은 원본·같은 target을 사용해 heatmap 및 예측 변화를 비교한다. 원래 palette와 painting/normalization을 재사용한다.
- `cam_mean_absolute_difference`: 표본별 정규화한 두 CAM의 mean absolute difference.
- `cam_cosine_similarity`: sum(A×B)/(||A||₂||B||₂). 0 CAM이면 빈 값.
- `foreground_mass`: 원본 grayscale를 입력 크기로 bilinear resize해 intensity>0.1인 mask 안 CAM 합 / 전체 CAM 합. 0 CAM이면 빈 값. mask 면적 비율도 제공한다.
- Foreground 모드에서 숫자와 색상은 같은 픽셀에 있다. foreground mass로 둘의 의존도를 분리할 수 없다. 최종층의 낮은 spatial resolution과 표본별 정규화도 고려해야 한다. Grad-CAM은 직접적인 인과적 증거가 아니다.
- 숫자별 1개 원본으로 얻은 CAM pair flip은 전체 test flip rate의 추정치로 쓰지 않는다. Probe 3,200개 입력도 320개 원본의 색상 반복이므로 3,200개의 독립적 원본으로 오인하면 안 된다.

## 0.80 / 0.65 학습 준비 — 자동 실행하지 않음

아래 명령은 설정과 실행 명령을 **생성만** 한다. 기존 0.95 모델을 재학습하지 않는다.

```bash
python prepare_correlations.py --reference-runs runs/resnet18_corr095_seed42_e100 runs/vgg11_corr095_seed42_e100 --output configs/followup_server --run-root runs/followup_corr --seeds 42
```

두 모델 × 두 correlation의 JSON, `commands.sh`, `commands.ps1`, 전체 plan JSON이 생성된다. 기존 설정을 복사해 train correlation/seed/output만 바꾸며 100 epochs, SGD, LR 0.01, momentum 0.9, weight decay 1e-4, batch 128, **scheduler none**, 64×64, foreground/next, val/test 0.50을 그대로 유지한다. 출력 경로에 기존 결과가 있으면 준비 단계와 학습 runner가 거부한다. `--seeds 42 43 44`로 이후 seed를 준비할 수 있다.

직접 실행하는 한 가지 예시(이번 작업에서는 실행하지 않았다):

```bash
python shortcut_experiment.py train --dataset colored_mnist --model resnet18 --epochs 100 --seed 42 --image-size 64 --train-corr 0.80 --val-corr 0.50 --test-corr 0.50 --mnist-val-size 5000 --color-mode foreground --color-pairing next --lr 0.01 --momentum 0.9 --scheduler none --weight-decay 0.0001 --batch-size 128 --num-workers 4 --features-per-group 32 --output runs/followup_corr/resnet18_corr080_seed42_e100
```

H2는 이 추가 실험 결과를 확보한 뒤 같은 protocol과 사전 정의한 관측 기준으로 비교한다. `observed_core_thresholds.csv`의 기본 기준은 chance-adjusted score 0.9에 처음 도달한 **관측 checkpoint**이다. sparse epoch 또는 epoch100 하나만으로 실제 학습 시작 시점을 단정하지 않는다. seed별 불확실성은 추가 seed 결과 확보 후 분석한다.

## 저장 구조

```text
outputs/followup_v1/
  protocol.json
  audit/{audit.json,execution.log,invocation.json}
  probes/
    probe_results.csv                 # run × epoch × layer × target
    checkpoint_inventory.json         # 가용/누락 목록
    protocol.json / invocation.json / execution.log / status.json
    <run>/epoch_XXX/
      train_features.npz / test_features.npz
      probe_layer_XXX_core.pt / probe_layer_XXX_shortcut.pt
      metadata.json
  gradcam/
    gradcam_metrics.csv                # run × epoch × 동일 원본 pair
    checkpoint_inventory.json / protocol.json / invocation.json / execution.log / status.json
    <run>/epoch_XXX/
      yX_index_XXXXX_{aligned,conflicting}_{input,heatmap,overlay}.png
      yX_index_XXXXX.npz / yX_index_XXXXX.json / checkpoint.json
      aligned_conflicting_grid.png
    comparison_grids/                  # 동일 원본의 모델/epoch 비교
  integrated/
    history_combined.csv / probe_combined.csv / gradcam_combined.csv
    probe_history_joined.csv / gradcam_history_joined.csv
    observed_core_thresholds.csv / gradcam_epoch_summary.csv
    *_accuracy_epochs.png / *_model_comparison.png / *_layer_heatmaps.png
    val_*.png / decodability_vs_flip_rate.png / gradcam_pair_change.png
    coverage.json / report_ko.md / execution.log / invocation.json / status.json
```

통합 도구는 누락 epoch를 heatmap에서 회색으로 표시한다. 실제 checkpoint가 없는 probe/CAM 수치를 채우지 않는다. 로그의 전체 학습 dynamics와 실제 확보된 checkpoint의 분석 범위를 `coverage.json` 및 한국어 보고서에 구분한다.

저장된 probe의 예측을 NumPy 행렬 연산으로 독립 대조하고, 조합 균형·원본 분리·동일 CAM 표본/target·heatmap 범위를 확인하려면 다음 명령을 사용한다.

```bash
python validate_followup.py --probe-results outputs/followup_v1/probes --gradcam-results outputs/followup_v1/gradcam --output outputs/followup_v1/validation.json
```

`--all-spatial-layers` 사용 시 같은 ReLU 모듈이 한 forward에서 여러 번 실행되면 기존 collector 규칙에 따라 마지막 호출의 pooled output을 저장한다. 기본 snapshot stage는 각 forward에 한 번 실행되는 모듈이다.

## 로컬 실행 범위

로컬에 있는 대상 checkpoint는 두 모델의 epoch100뿐이다. 0/1/3/5/10/20/30/50/70 checkpoint와 원래 640행 feature NPZ는 서버에서 필요하다. 로컬 실행에서는 `--allow-missing`으로 epoch100의 새 독립 probe와 counterfactual CAM만 계산했다. 별도 2-epoch pilot checkpoint를 100-epoch 실험의 누락 checkpoint 대신 사용하지 않았다.

로컬 산출물은 저장소 상위 `shortcut_results/followup_v1/`에 있다. `protocol.json`, `probes_v2/`, `gradcam/`, `audit/`, `integrated_v2/`를 확인한다. 초기 실패 실행 폴더도 보존했다. 실행한 feature는 각 split 3,200행이며 ResNet 5개, VGG 6개 stage다. 학습은 linear probe에 한하며 기존 모델 학습과 추가 correlation 학습을 실행하지 않았다.

기존 결과 보존 테스트, data balance/원본 누수 방지, frozen model 불변성, zero-CAM 처리, 누락/중복 checkpoint 거부 및 학습 설정 보존은 `python -m unittest test_followup -v`로 검증한다. 최종 통합 보고서와 그래프는 `reports/followup_v1/`에 있다. 대용량 feature·checkpoint·probe 가중치와 실행 캐시는 Git 업로드 대상에서 제외한다.
