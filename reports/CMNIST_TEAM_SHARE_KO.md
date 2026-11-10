# Colored MNIST × VGG11_BN: 팀 공유

**학습 초기에 커진 색상 의존성은 Epoch50까지 대체로 감소했지만, 매 Epoch 줄어들지는 않았다. Correlation .95에서 Epoch100까지 연장해도 강건성은 개선되지 않았다.** 색상과 숫자의 상관이 강할수록 강건한 예측에 도달하는 시점은 늦었지만, 숫자 형태를 실제로 배우기 시작한 시점이 늦어졌다고 단정할 수는 없다.

## 1. 무엇을 확인하려 했는가?

숫자 형태가 정답을 결정하지만 색상도 정답과 강하게 연결된 데이터에서, 모델이 학습 시간에 따라 색상 Shortcut에 얼마나 의존하는지 확인했다.

- **H1:** 학습이 진행되면 초기 Shortcut 의존성이 약해지는가?
- **H2:** 숫자–색상 상관이 강할수록 Core(숫자 형태) 학습이 늦어지는가?

## 2. 실험 설정과 지표

| 항목 | 설정 |
|---|---|
| 데이터 | Colored MNIST, Train 55,000 / Validation 5,000 / Test 10,000 |
| 색상 상관 | 숫자별 지정 색상을 가질 확률 .95 / .80 / .65; 나머지는 다른 9색에서 선택 |
| 평가 분포 | Validation/Test 상관 .10: 숫자와 10색이 독립인 기준 |
| Background(BG) | 흰 숫자 + 밝기를 0.5배로 줄인 색상 배경 |
| Foreground(FG) | 검정 배경 + 색상 숫자 |
| 모델·학습 | 사전학습 없는 VGG11_BN, 512→10 선형 head, 224px, Seed42, SGD, LR .01 고정, batch128, 증강 없음 |
| 학습 범위 | **6조건 모두 Epoch50**, 이후 **BG/FG .95만 Epoch100까지 resume** |

같은 숫자의 색상만 바꿔 예측 변화를 측정했다. 아래 표의 비율은 모두 %이며, 별도 표시가 없으면 Validation 결과다.

| 지표 | 읽는 방법 |
|---|---|
| Accuracy | 원본 숫자의 정답률 |
| Counterfactual Flip Rate | 색상을 바꿨을 때 예측이 달라지는 비율. 오답→정답도 포함 |
| All-color C2I | 원본에서 맞았으나 색상 변경 후 틀리는 비율. **본문은 원본×9개 대체색 전체 쌍을 분모로 사용** |
| Linear Probe | 고정된 내부 Feature에서 숫자·색상을 선형 분류기로 얼마나 해독할 수 있는가 |
| Grad-CAM | 정답 클래스와 관련된 공간 영역을 살펴보는 보조 설명 |

분모는 구분해야 한다. 원본 수를 `N`, 원본 정답 수를 `A`라 하면 paired C2I는 `C/N` 또는 `C/A`, All-color C2I는 `C9/(9N)` 또는 `C9/(9A)`다. 본문의 All-color C2I는 **`C9/(9N)`**이며, 이미지별로 한 색상이라도 실패하는 비율과 다르다. 기존 All-color Accuracy는 개입 색상≠정답인 쌍만 집계하므로 전체 9색 지표와 분모가 다르다.

## 3. 핵심 결과

### ① Epoch50: 상관이 강할수록 색상 의존성이 많이 남았다

| 조건 | Val Accuracy | Test Accuracy | All-color Flip | All-color C2I |
|---|---:|---:|---:|---:|
| BG .95 | 96.40 | 96.71 | 4.82 | 2.0467 |
| BG .80 | 98.54 | 98.66 | 1.36 | 0.5667 |
| BG .65 | 98.88 | 99.08 | 0.87 | 0.3533 |
| FG .95 | 95.56 | 95.93 | 5.77 | 2.3800 |
| FG .80 | 98.34 | 98.55 | 1.63 | 0.6311 |
| FG .65 | 98.86 | 99.01 | 0.94 | 0.4333 |

두 mode 모두 .95에서 정확도가 낮고 색상 민감도가 높았다. Epoch50에서는 FG의 All-color C2I가 BG보다 높지만, 지표와 시점에 따라 순위가 바뀌므로 FG가 항상 불리하다고 일반화할 수는 없다.

[조건별 그래프](metric_final_analysis_406/figures/06_correlation_overview.png) · [수치 CSV](metric_final_analysis_406/summary_table.csv)

### ② Epoch가 늘면 대체로 감소했지만, 중간 반등이 있었다

여섯 조건 모두 Epoch1→50의 Flip과 C2I가 감소했다. 예를 들어 .95의 All-color C2I는 BG **15.1600→2.0467%**, FG **17.4244→2.3800%**로 줄었다. 그러나 인접 49전환 중 C2I가 증가한 경우도 조건별 **18–23회** 있었다. 무학습 Epoch0→1에는 오히려 색상 민감도가 크게 증가했다.

대표적인 예외는 **BG .95 Epoch28**이다. 숫자5·색상 불일치 그룹의 정확도가 Epoch27→28→29에서 **98.98→77.41→97.97%**로 급락했다가 회복했다. 이 구간에도 Core Probe는 약97.7–97.8%로 높았다. 내부 정보가 해독 가능해도 실제 예측은 실패할 수 있으며, 급락 원인은 아직 확정되지 않았다.

[C2I 학습 곡선](metric_final_analysis_406/figures/05_c2i_all9.png) · [Epoch28 분석 그림](metric_final_analysis_406/figures/08_epoch28_anomaly.png)

### ③ .95의 Epoch100 연장은 강건성을 개선하지 못했다

| 조건 | Val Accuracy E50→100 | All-color Flip E50→100 | All-color C2I E50→100 |
|---|---:|---:|---:|
| BG .95 | 96.40→95.56 | 4.82→6.32 | 2.0467→2.6800 |
| FG .95 | 95.56→95.22 | 5.77→6.52 | 2.3800→2.6556 |

마지막 5 Epoch 평균으로 비교해도 C2I 증가가 남았다. 따라서 이 설정에서는 더 오래 학습하는 것만으로 색상 의존성이 계속 줄어들지 않았다. LR·SGD·BN 등이 원인 후보일 수 있지만, 이를 분리한 실험은 하지 않았다.

[Epoch50 이후 그래프](metric_final_analysis_406/figures/07_resume_late_dynamics.png)

### ④ Probe와 Grad-CAM은 실제 사용의 직접 증거가 아니다

Epoch50의 Core Probe 정확도는 **97.21–98.21%**, 색상 Probe는 **99.92–100%**였다. 색상 정보가 거의 완벽히 해독되는 상태에서도 원모델의 색상 민감도는 조건마다 달랐다. 무학습 모델에서도 Core Probe가 BG86.38%, FG80.83%였으므로, 높은 Probe 정확도를 Core 학습 시작으로 볼 수 없다.

강건 행동 기준에 처음 도달한 Epoch는 다음과 같다.

| 학습 상관 | BG | FG |
|---|---|---|
| .65 | 2 | 2 |
| .80 | 3 | 3 |
| .95 | 46, 이후 재이탈 | Epoch100까지 미충족 |

기준은 색상 불일치 원본 정확도≥95%, 기존 All-color Accuracy≥95%, All-color Flip≤5%를 **3 Epoch 연속** 충족한 구간의 시작이다. Core Probe의 주 기준90%는 모든 조건이 Epoch1에 도달했다. 보조 기준95%에서만 .95가 늦었다(BG2, FG3; 나머지1).

Grad-CAM도 숫자 획 주변에 나타나면서 예측은 색상에 따라 바뀌는 사례가 있었다. **Probe는 정보의 해독 가능성, CAM은 공간 설명이며, 원모델이 그 정보를 실제로 사용한다는 인과적 증거는 아니다.**

[Probe 그래프](metric_final_analysis_406/figures/09_probe_decodability.png) · [Probe 상세](cmnist_vgg50_probe_followup/analysis_report.md) · [CAM 비교](metric_final_analysis_406/figures/11_cam_phases_bg.png)

## 4. 가설에 대한 현재 판단

| 가설 | 판단 |
|---|---|
| H1 | **초기 이후의 전반적 감소는 지지.** 중간 반등과 Epoch100의 악화가 있어 단조 감소는 지지하지 않음. |
| H2 | **강한 상관에서 강건 행동 도달이 늦다는 점은 관측.** 실제 Core 학습 시작이 늦어지는지는 미확정. |

## 5. 한계와 다음 논의

모델 Seed42 하나의 결과다. Correlation 비교는 공통 Epoch50까지이며, .80/.65의 Epoch100 결과는 없다. Probe도 Epoch50까지만 있어 후기 행동 악화를 표현 변화에 직접 연결할 수 없다. BG/FG는 색상 위치 외에 밝기·대비도 다르고, resume은 중단 없는 100 Epoch 학습과 완전히 동일하다고 보장할 수 없다.

다음 논의는 세 가지에 집중할 수 있다.

1. **재현성:** 여러 모델 seed에서도 correlation 효과와 후기 재상승이 유지되는가?
2. **정보와 사용의 분리:** Epoch51–100 Probe 및 초기 step별 색상·형태 개입으로 실제 Core 사용을 어떻게 측정할 것인가?
3. **이상 현상의 원인:** Epoch28 실제 실패 사례와 LR·BN 등의 영향을 어떤 대조 실험으로 분리할 것인가?

세부 설정·검증은 [최종 분석 보고서](metric_final_analysis_406/analysis_report.md), [무결성 보고서](metric_final_analysis_406/integrity_report.md), [기존 Epoch50 보고서](cmnist_vgg50_analysis_review/analysis_report.md)를 참조한다. C2I 수치는 과거 집계값의 추정 범위가 아니라 후속 개별 예측 분석의 확정값을 사용했다.
