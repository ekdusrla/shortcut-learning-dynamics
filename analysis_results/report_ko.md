# ResNet18 · VGG11_bn Shortcut Learning Dynamics

## 핵심 결론

Seed 42의 이번 실험에서 **VGG11_bn은 처음에는 색상 의존성이 더 크지만 빠르게 개선되어 최종적으로 더 낮은 Flip Rate와 더 높은 Validation/Test 성능을 보인다.** 두 모델 모두 학습 초기 이후 의존성이 줄어들지만 단조 감소하지 않는다. ResNet18은 초기 개선 후 정체하며 후반 Flip Rate가 미세하게 증가한다.

실제 자료는 `shortcut_results/runs/`의 CSV/JSON이다. 원본 8개 파일의 크기와 SHA256은 `file_inventory.csv`에 기록했다. 원본을 수정하지 않았다. 서버 소스와 로그는 제공되지 않아 서버 코드가 로컬 코드와 같은지는 확정할 수 없다. 계산 방식 설명은 현재 로컬 구현 기준이다.

전체 그래프: [graphs.html](graphs.html). SVG 15개를 개별 파일로도 저장했다. 재실행 명령: `powershell -NoProfile -ExecutionPolicy Bypass -File analysis_results/analyze_results.ps1`.

## 실제 파일과 정합성

두 run에는 각각 다음 네 파일만 있다. Checkpoint, Feature NPZ, 학습 로그는 이번 복사본에 없다. `save_features=true` 설정만으로 특징 파일이 포함되었다고 가정하지 않았다.

| 파일 | 역할·구조 |
|---|---|
| config.json | 모델, 데이터, optimizer, seed 등 설정 객체 |
| history.csv | **Epoch별 지표 원자료**, E0~100의 101개 행. epoch/lr/train_loss/train_acc/val_loss/val_acc/val_aligned_acc/val_conflicting_acc/val_g{y}{s}_acc 및 _n/val_wga/val_cf_acc/val_cf_flip_rate/val_cf_acc_drop |
| latest.json | epoch=100과 서버 checkpoint 경로. checkpoint 자체의 존재·무결성은 확인 불가 |
| test_metrics.json | {epoch:100, metrics:{...}}. 최종 test 평가만 저장 |

두 config의 차이는 model 하나이다. Colored MNIST, correlation 0.95/0.50/0.50, seed 42, SGD, LR 0.01, momentum 0.9, weight decay 0.0001, batch 128, scheduler none, image size 64, epochs 100이 모두 요청과 일치한다. Foreground/next pairing, pretrained=false, augment=false, AMP=true, workers=4도 동일하다. 실제 LR도 전 구간 0.01이다.

두 history 모두 Epoch 누락·중복이 없다. 모든 202개 행에서 20개 그룹 표본수 합은 5,000이며 aligned 2,528개, conflict 2,472개이다. 그룹별 수치로 재계산한 전체/Aligned/Conflict 정확도와 WGA는 CSV와 오차 1e-10 이내로 일치한다. CF drop도 val_acc−val_cf_acc와 일치한다. 두 run의 val/test 그룹 표본수는 같으나 indices가 없어 개별 예제 동일성까지 직접 검증하지는 못했다. 상세 결과는 `validation_checks.csv`에 있다.

## 지표 계산 검토

주 함수는 `shortcut_experiment.py:evaluate`, 기록은 `append_history`이다.

- Train Accuracy는 train 모드에서 가중치가 갱신되는 동안 minibatch 예측을 누적한 정확도이다. Epoch 종료 고정 모델을 eval 모드에서 측정하는 Validation Accuracy와 조건이 다르다. E0 Train Accuracy는 NaN이며 추정하지 않았다.
- WGA는 데이터 구성상 가능한 (숫자, 색상) 20개 그룹의 최소 정확도이다. 100개 조합 전체가 아니다. 가능한 그룹 하나라도 비면 NaN이다.
- Aligned/Conflict는 y==s/y!=s 표본의 pooled accuracy이다.
- Flip Rate는 동일 숫자의 색상만 main↔alternate로 바꾸어 예측 클래스가 달라지는 비율이다. CF Accuracy는 변경 후 원래 숫자의 정답률이다. CF drop=원본−CF 정확도이며 음수도 가능하다.
- `evaluate.py`는 별도 linear probe용 스크립트이며 현재 build_datasets가 제공하지 않는 probe split을 요구해 그대로 실행할 수 없다. 출력 2×accuracy−1은 Pearson correlation이 아니고 10-class 균등 추측 기준은 −0.8이다.
- `analyze.py`는 원본 run 내부에 그래프를 쓰므로 이번에는 실행하지 않았다. 독립 Epoch PCA 좌표는 직접 비교하면 안 된다. 기존 CF drop 그래프의 y축 하한 −0.05는 음수가 큰 경우 잘릴 수 있다.
- `gradcam.py`는 공간 activation에 대한 class gradient를 평균해 채널 가중치를 만들고 ReLU·보간·이미지별 정규화를 적용한다. 인과적 의존성의 정량 검정은 아니다. --target-class가 0/1로 제한되어 숫자 2~9를 명시 선택할 수 없다. 기본 예측 클래스 선택은 가능하다.
- VGG는 global average pooling과 linear head를 사용하는 VGG11_bn 변형이다. 원래 4096-unit classifier를 쓰는 VGG 전체에 결론을 일반화하지 않는다.

## 핵심 수치 비교

정확도/Flip은 %, 차이와 gap/drop은 percentage point(pp) 단위이다.

| 지표 | ResNet E1 | VGG E1 | ResNet E100 | VGG E100 | 최종 VGG−ResNet (pp) |
|---|---:|---:|---:|---:|---:|
| Train Accuracy | 98.4073 | 98.5727 | 100.0000 | 100.0000 | 0.0000 |
| Validation Accuracy | 97.5600 | 96.3200 | 98.7800 | 99.3200 | +0.5400 |
| Validation WGA | 78.9916 | 56.7227 | 91.1765 | 96.0870 | +4.9105 |
| Aligned Accuracy | 99.8813 | 99.9604 | 100.0000 | 100.0000 | 0.0000 |
| Conflict Accuracy | 95.1861 | 92.5971 | 97.5324 | 98.6246 | +1.0922 |
| Flip Rate ↓ | 4.6400 | 7.2200 | 2.3600 | 1.4200 | −0.9400 |
| CF Accuracy | 97.7600 | 96.4400 | 98.8600 | 99.2600 | +0.4000 |
| Aligned−Conflict gap ↓ | 4.6952 | 7.3634 | 2.4676 | 1.3754 | −1.0922 |
| CF Accuracy Drop | −0.2000 | −0.1200 | −0.0800 | +0.0600 | +0.1400 |

| 최종 Test 지표 | ResNet18 | VGG11_bn | VGG−ResNet (pp) |
|---|---:|---:|---:|
| Accuracy | 98.9700 | 99.3200 | +0.3500 |
| WGA | 94.7955 | 96.0967 | +1.3011 |
| Aligned Accuracy | 100.0000 | 99.9797 | −0.0203 |
| Conflict Accuracy | 97.9728 | 98.6814 | +0.7085 |
| Flip Rate ↓ | 1.8800 | 1.2700 | −0.6100 |
| CF Accuracy | 99.1400 | 99.4100 | +0.2700 |
| CF Accuracy Drop | −0.1700 | −0.0900 | +0.0800 |

CF drop이 거의 0이어도 변경 전후 오류가 상쇄될 수 있어 Flip Rate는 0이 아닐 수 있다. 이 실험에서도 CF drop만으로 의존성 차이를 평가하기는 어렵다. Test는 최종값만 있으며 Epoch별 test 곡선은 생성하지 않았다.

## Epoch별 변화 해석

![Flip Rate](compare_val_cf_flip_rate.svg)

![WGA](compare_val_wga.svg)

![Conflict Accuracy](compare_val_conflicting_acc.svg)

**E0:** ResNet/VGG validation accuracy는 10.46%/10.06%, WGA와 Flip은 모두 0%이다. 일부 숫자 그룹만 100%, 나머지 0%인 기록은 상수 클래스 예측과 부합한다. 낮은 Flip이 core 학습을 의미하지 않는 사례이다.

**E1~4:** E1부터 conflict 정확도가 95.19%/92.60%로 높아 이미 상당한 숫자 구별 능력이 있다. 동시에 aligned와의 차이 및 색상 개입 민감도가 남는다. 두 모델 모두 E2에서 처음 Flip이 감소한다: ResNet 4.64→3.24%, VGG 7.22→3.04%. VGG의 Flip이 E2부터 더 낮다. 이것은 최초 관측 감소이며 통계적으로 검증된 변화점은 아니다. Epoch 내부 기록이 없어 첫 Epoch의 학습 순서는 판별 불가하다.

ResNet은 E1→E4에서 Flip 4.64→2.22%로 감소하나 E5에 2.44%로 반등한다. VGG는 E1→E3에서 7.22→2.08%로 감소한 뒤 E4에 2.60%로 반등한다. VGG WGA도 E3 94.94%→E4 87.83%로 일시 하락한다.

| Flip 선형 기울기 (pp/Epoch) | ResNet18 | VGG11_bn |
|---|---:|---:|
| E1~10 | −0.138788 | −0.391636 |
| E11~30 | −0.004135 | −0.010992 |
| E31~60 | −0.002492 | −0.003199 |
| E61~100 | +0.001049 | −0.001315 |

각 기울기는 해당 구간 모든 원자료의 최소제곱 추세이며 유의성 검정은 아니다. VGG의 초기 개선이 빠르지만 출발 의존성도 높으므로 기울기만으로 구조의 본질적 우월성을 주장하지 않는다.

탐색적 최초 도달시간은 Flip≤3%가 두 모델 E3, Flip≤2%는 VGG E5이고 ResNet 미도달, Flip≤1.5%는 VGG E9이고 ResNet 미도달이다. 이는 단일 Epoch 통과 기준이며 지속적 달성이나 사전등록 지표가 아니다.

**중후반:** E1→E100 Flip은 ResNet −2.28pp(상대 약 49.1%), VGG −5.80pp(약 80.3%) 감소한다. Conflict 정확도는 +2.3463pp/+6.0275pp, CF 정확도는 +1.10pp/+2.82pp 개선된다. 그러나 ResNet WGA는 E2 92.61%가 최종 91.18%보다 높다. 오래 학습할수록 모든 robustness 지표가 좋아지지는 않는다.

| E91~100 평균 | ResNet18 | VGG11_bn |
|---|---:|---:|
| Validation Accuracy (%) | 98.8080 | 99.3020 |
| WGA (%) | 91.6807 | 96.0870 |
| Conflict Accuracy (%) | 97.6092 | 98.5882 |
| Flip Rate (%) | 2.3560 | 1.4100 |
| CF Accuracy (%) | 98.8260 | 99.2880 |

후반 평균도 최종 Epoch와 같은 모델 순서이다. ResNet의 후반 Flip 기울기는 미세한 양수이나 강한 shortcut 재발로 단정할 정도는 아니다. VGG도 후기 개선은 작아 plateau에 가까워진다. E1~100 사이 Flip 증가/감소 횟수는 ResNet 48/47회, VGG 43/48회로 단조 감소가 아니다. 이는 기술적 요약이며 Epoch들을 독립 표본으로 취급하지 않는다.

최종 validation 최약 그룹은 ResNet 숫자8·색9(91.1765%, n=238), VGG 숫자2·색3(96.0870%, n=230)이다. Test에서는 두 모델 모두 숫자2·색3이 최약 그룹(94.7955%/96.0967%, n=538)이다.

## 연구 가설 및 추가 실험

**H1은 제한적으로 지지된다.** 학습된 초기 E1 대비 E100에서 Flip과 aligned−conflict gap이 감소하고 CF/conflict 정확도가 개선된다. 하지만 E0→E1에서 Flip이 증가하고 이후에도 반등·정체가 있다. “학습 초기 이후 전반적 감소”는 결과와 일치하나 “Epoch 증가에 따라 항상 감소”는 지지하지 않는다. Seed 하나이므로 통계적 일반화나 구조 간 유의한 차이를 주장할 수 없다.

**H2는 검증할 수 없다.** 0.95/0.80/0.65 × 두 모델 × 동일 복수 seed를 실행하고 split, pairing, 색칠 영역, 전처리, optimizer, budget을 고정한다. 도달시간은 conflict/CF 정확도·WGA·Flip의 사전 정의 기준을 5 Epoch 연속 만족하는 최초 시점 등으로 정하고 threshold 민감도를 보고한다. Seed를 반복 단위로 paired 비교 및 불확실성 평가를 수행한다. 현재 Flip≤2%만 쓰면 ResNet은 100 Epoch 내 미도달이므로 우측 검열로 처리해야 하며 임의의 도달시간으로 대체하면 안 된다. 미도달이 많으면 budget 연장/검열을 고려한 분석이 필요하다. AUC와 후반 평균도 함께 비교한다.

Feature NPZ가 없어 표현 변화 분석은 수행하지 않았다. 추가 제공 시 indices/labels/spurious를 대조한 뒤 같은 표본의 레이어별 centered linear CKA, 공통 PCA 투영, 독립 분할 linear probe 분석이 가능하다. 서로 독립 적합한 PCA 축의 이동을 표현 변화량으로 쓰지 않는다. NPZ에 paired recolor feature가 없으면 counterfactual feature distance도 계산할 수 없다. Decodability는 classifier의 인과적 사용을 증명하지 않는다.

Correlation 0.50에서도 숫자당 두 색상만 배정되므로 숫자·색상이 독립이 아니다. 색상만으로 후보 숫자를 좁힐 수 있다. 다음 실험으로 다중 seed 반복, grayscale 대조군, color-only baseline, 숫자와 독립인 10색 배정, 100개 숫자×색상 조합 평가를 권한다. 현재 결론은 main↔alternate 개입 범위로 제한한다.

## 저장한 결과물

- `graphs.html`: 15개 SVG 그래프 모음.
- 모델별 `*_accuracy.svg`: E0~100 정확도/WGA/group/CF 곡선. `*_accuracy_zoom.svg`: E1~100 확대본. `*_shortcut.svg`: Flip/gap/drop.
- `compare_*.svg`: 지표별 두 모델 비교. CF drop 음수도 표시.
- `ResNet18_epochs.csv`, `VGG11_bn_epochs.csv`: 101개 Epoch 원지표와 계산한 gap, 단위 0~1. 미측정은 NaN.
- `summary.csv`: 초기/최종/후반 평균/범위/증감 횟수, 단위 0~1.
- `segment_slopes.csv`: 구간별 기울기, pp/Epoch.
- `file_inventory.csv`, `validation_checks.csv`: 원본 파일 목록·hash와 검증.
- `analyze_results.ps1`: 재현용 코드. Python 실행환경이 없어 외부 패키지 없이 PowerShell로 계산하고 SVG를 생성했다.
