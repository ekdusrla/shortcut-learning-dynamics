# Shortcut Learning Dynamics · 최종 통합 분석

## A. 비교 범위와 데이터 무결성 · PASS

분석 단위는 같은 데이터셋·조건·split·Epoch·평가 정의를 공유하는 백본 쌍이다. 데이터셋별 원본 검증 JSON과 통합 CSV, 훈련 config, History, 최종 체크포인트 포인터 및 크기, Waterbirds 매니페스트 해시를 대조했다. 기존 전체 체크포인트·CAM 검증 기록을 재사용했다. 체크포인트 전수 CRC나 GPU 추론은 이번 통합에서 반복하지 않았다.

| 데이터셋 | 조건 / 백본 | 직접 비교 | 별도 범위 | 반사실 |
|---|---:|---|---|---|
| Colored MNIST | BG/FG × .95/.80/.65 = 6 | E0–E50 | 일부 VGG E51–E100 제외 | 같은 숫자의 색을 교체. Paired 및 All9 별도 |
| Waterbirds A | .95/.80/.65 = 3 | E0–E50 | ResNet18 E51–E100 별도 | 같은 새·마스크에 반대 유형 배경 합성 |

두 백본은 seed 42, 224×224, batch 128, SGD(lr .01, momentum .9, weight decay 1e-4), 고정 LR, AMP, pretrained=False, augmentation=False를 공유한다. Train/Val/Test는 Colored MNIST 55,000/5,000/10,000개, Waterbirds 4,795/1,199/5,794개다. 목표 Val/Test corr은 각각 .10/.10과 .50/.50이다. 공통 frozen Val은 Colored MNIST 612행, Waterbirds 306행이며 후기 Waterbirds ResNet18 Val 150행은 별도 scope다. 짝지어진 정확도·전이 비교에서는 표본 수·유효 쌍 분모를 대조했다.

Colored MNIST의 ResNet18 frozen Train/Val/Test는 전 Epoch가 있다. 기존 VGG의 전 Epoch frozen Train/Test와 Test 반사실 전이는 공통 자료에 없으므로 해당 비교는 제외했다. VGG Train의 온라인 정확도와 ResNet18 frozen Train 정확도를 직접 비교하지 않았다. Waterbirds의 CF는 기존 집계에서 이진 전이를 복원했으므로 표본별 CF 전이 비교 범위는 아니다. 파일의 `comparison_scope`와 `intervention`을 유지했다.

[실험 설정](settings.csv) · [무결성 및 소스 해시](validation.json) · [Accuracy/WGA](accuracy.csv) · [짝지어진 성능 차이](accuracy_matched.csv) · [반사실 전이](counterfactual.csv) · [짝지어진 CF 차이](counterfactual_matched.csv)

## B. H1 · 부분 지지, WARN

조건별 1차 기술 판정은 E1–E5 평균 대비 E26–E50에서 WGA 증가, Flip 및 두 C2I 비율 감소가 모두 나타나면 지지, 모두 반대면 기각, 개선과 악화가 섞이면 부분 지지로 두었다. 순위 추세와 초기 기준 민감도를 별도 확인했다. 아래는 **후기−초기 변화(%p)**, 각 셀은 WGA / Flip / 조건부 C2I 순서다. Colored MNIST는 All9, Waterbirds는 반대 배경 개입이다.

| 조건 | VGG11_BN | ResNet18 |
|---|---:|---:|
| Colored MNIST BG .95 | +44.52 / −34.38 / −22.26 | +29.31 / −20.07 / −10.79 |
| Colored MNIST BG .80 | +10.78 / −4.21 / −1.88 | +5.02 / −2.40 / −1.17 |
| Colored MNIST BG .65 | +9.25 / −2.83 / −1.29 | +7.76 / −1.40 / −.51 |
| Colored MNIST FG .95 | +43.03 / −30.44 / −18.46 | +24.50 / −17.98 / −10.50 |
| Colored MNIST FG .80 | +9.10 / −4.93 / −2.46 | +7.09 / −2.76 / −1.38 |
| Colored MNIST FG .65 | +8.12 / −2.31 / −1.10 | +5.90 / −1.47 / −.69 |
| Waterbirds A .95 | +2.45 / −7.16 / −11.87 | +.22 / −2.71 / −7.14 |
| Waterbirds A .80 | +14.01 / −6.01 / −8.73 | +3.59 / −2.15 / −10.47 |
| Waterbirds A .65 | +20.25 / −10.24 / −6.79 | −6.46 / −12.81 / −14.00 |

**Colored MNIST · 지지.** 12개 모델×조건의 1차 행동 지표가 모두 개선 방향이었다. 일시적 악화는 있었다. Background .95 VGG는 E27→28에 All9 Flip이 3.08%p 상승해 E30에 이전 수준을 회복했다. E50 avgpool 색 Probe 정확도는 ResNet18 97.75–100%, VGG 99.88–100%였다. 행동상 민감도 감소와 색 정보의 표현 내 잔존이 함께 관측됐다.

**Waterbirds · 부분 지지.** 1차 구간 기준 6개 중 5개가 동시 개선을 보였지만, ResNet18 .80의 평균 Flip은 초기→중기→후기 **38.28→29.25→36.13%**, .95는 **69.57→59.88→66.86%**로 반등했다. E1–E50 Flip–Epoch 순위상관도 각각 **+.233, +.145**였다. 기존 구간 E1–E10→E31–E50을 쓰면 이 두 조건의 평균 Flip은 각각 **+7.14, +6.06%p** 상승한다. .65 ResNet18은 1차 구간에서 Flip·C2I가 줄어든 가운데 WGA는 **17.26→10.80%**로 하락했다. I2C도 **16.73→9.43%**로 줄어 Flip 감소에는 정답으로의 전이 감소도 포함됐다. VGG .65 E31의 단일 클래스 예측은 낮은 Flip이 강건함을 뜻하지 않는 추가 반례다.

CAM은 고정 표본과 타깃별 공간 지표로 사용했다. Colored MNIST .95 ResNet18 원본 예측 클래스 CAM의 Shortcut 질량은 E0→E50 Background **70.9→66.2%**, Foreground **26.7→29.1%**였으나 두 조건 모두 행동 Flip이 낮아졌다. Foreground의 숫자와 색 영역은 겹친다. Waterbirds의 새 영역 질량도 마스크 면적과 타깃에 의존한다. I2C·순위상관·악화폭과 회복 시점은 표에 병기했다. 시간점 간 독립성이나 p값 검정은 전제하지 않았다.

[조건별 H1과 민감도·회복](h1_conditions.csv) · [초기·중기·후기 지표](phase_means.csv) · [Paired/All9 별도 구간 비교](cf_phase_means.csv) · [CAM 통합 자료](cam_integrated.csv)

## C. H2 · 원래 가설 판단 보류, 행동 해석 부분 지지 · WARN

표의 시점은 **3 Epoch 연속 기준의 시작**이며, 확인은 2 Epoch 뒤다. 미도달을 종료 Epoch로 대체하지 않았다. H2 CSV는 단일 최초 충족, 지속 시작·확인, 이용 가능한 후속 유지·민감도 기록을 보존한다.

| 조건과 운영 기준 | VGG .95/.80/.65 | ResNet18 .95/.80/.65 |
|---|---|---|
| Colored MNIST BG 숫자 Probe ≥95% | E2/E1/E1 | E2/E1/E1 |
| Colored MNIST FG 숫자 Probe ≥95% | E3/E1/E1 | E3/E1/E1 |
| Colored MNIST BG 역사적 강건 행동 | E46/E3/E2 | 미도달/E4/E2 |
| Colored MNIST FG 역사적 강건 행동 | 미도달/E3/E2 | 미도달/E2/E1 |
| Waterbirds avgpool Core Probe ≥60%, E0–E50 | E46/E20/E7 | 미도달/미도달/미도달 |
| Waterbirds avgpool Core Probe ≥55%, E0–E50 | E30/E15/E3 | 미도달/미도달/E14 |
| Waterbirds 공동 강건 행동, E0–E50 | 미도달/미도달/E16 | 미도달/미도달/미도달 |

Colored MNIST 역사적 Probe는 숫자 10분류의 세 split 최솟값을 기준으로 한다. 90%에서는 모두 E1이고, 별도 이진 Core의 55–70% 기준에서는 양 모델 모두 E0부터 충족했다. 역사적 행동 기준은 Conflicting Acc≥95%, conflict-only All-color Acc≥95%, All9 Flip≤5%의 동시 충족이다. 이 조건의 순서는 해독 시점과 일치하지 않는다.

Waterbirds 공동 행동 기준은 WGA≥20%와 균형 Conflicting Accuracy≥50%다. .65 VGG는 5 Epoch 지속 기준에서 E36, 10 Epoch 기준에서는 미도달이다. ResNet18 .65는 한 Epoch만 요구하면 E8에 도달하지만 3회 지속에는 E100까지 미도달했다. avgpool Core 60%도 세 조건 모두 E100까지 미도달했다. .65의 탐색적 E14는 55% 기준 결과다. E17에서 Core Probe 66.07%와 WGA 10.95%가 함께 관측됐다.

**해석.** VGG Waterbirds의 Probe 순서와 두 Colored MNIST 백본의 엄격한 행동 기준은 높은 corr의 지연 방향과 부합한다. 그러나 기준을 바꾸면 순서가 사라지거나 미도달이 늘어난다. 디코딩 가능성이나 행동 기준은 모델이 Core를 실제 예측에 쓰기 시작한 시점을 측정하지 못한다.

[H2 최초 도달·지속·민감도 표](h2_timing.csv) · [Probe 전체](probes.csv) · [동일 프로토콜·구조 stage 비교](probes_matched.csv)

## D. 아키텍처 비교 · PASS, 일반화 WARN

Colored MNIST E50 Val에서 Background .65의 WGA는 ResNet18 **98.16%**, VGG **97.11%**, All9 Flip은 **0.73/0.87%**였다. Foreground .95에서는 순위가 바뀌어 WGA **90.56/92.13%**, Flip **7.16/5.77%**, 조건부 C2I **3.12/2.49%**였다. Background .65도 ResNet18의 조건부 C2I **0.45%**가 VGG **0.36%**보다 높아 Flip 순위가 그대로 유지되지 않았다.

Waterbirds E50 Val에서 ResNet18/VGG WGA는 .95 **5.84/5.84%**, .80 **11.68/17.52%**, .65 **11.68/29.93%**였다. .65의 Flip은 **20.18/16.51%**, 조건부 C2I **13.84/11.16%**, avgpool Core Probe **57.14/71.43%**였다. VGG는 .80/.65에서 WGA와 반사실 결과가 유리했다. 반면 .95에서는 E50 WGA가 같고 두 모델 모두 강건 행동 지속 기준에 미도달했다. 공통 CAM도 입력은 같지만 백본의 대상 레이어 해상도·수용 영역은 다르다.

학습 진행 정도를 확인하기 위해 온라인 Train Accuracy/Loss를 별도 집계했다. E1 Waterbirds Train Accuracy 범위는 ResNet18 **71.72–80.54%**, VGG **75.37–84.96%**였다. E50에는 두 데이터셋의 양 백본 모두 온라인 Train Accuracy가 100%였다. Waterbirds E50 Train Loss는 ResNet18 **.000041–.000053**, VGG **.000239–.000279**인데 .80/.65의 Val WGA는 VGG가 높았다. 이는 같은 Epoch 또는 낮은 Train Loss가 같은 held-out 강건성을 보장하지 않음을 보여 준다.

로그의 epoch 구간 합은 Waterbirds E0–E50에서 백본별 3조건 평균 ResNet18 **2.46h**, VGG **2.51h**였다. Colored MNIST ResNet18 E0–E50은 **2.16–2.20h**였다. SLURM 회계의 Colored MNIST E50 경과는 ResNet18 2:09:37–2:12:26, VGG 4:22:56–4:29:48이었다. Waterbirds는 VGG E50 2:30:24–2:31:39, ResNet18 E100 4:53:09–4:55:57이다. SLURM 경과와 runner 기록은 계산·평가·입출력을 함께 포함하며 순수 optimizer 시간과 동일한 측정이 아니다. 실제 optimizer update 횟수는 독립 로그로 확인하지 못해 추정하지 않았다.

[훈련 로그 시계열](training.csv) · [SLURM 학습 기록](training_slurm.csv) · [Colored MNIST 백본 그래프](colored_mnist_architectures.png) · [Waterbirds 백본 그래프](waterbirds_architectures.png)

## E. 데이터셋 비교 · WARN

Colored MNIST에서는 두 백본과 두 색 위치 모두 후기 행동 개선이 나타났다. Waterbirds에서는 높은 corr에서 반등과 구간 선택에 따른 결론 변화가 있었고, .65 ResNet18은 감소한 배경 민감도와 하락한 WGA가 공존했다. 두 데이터셋 모두 높은 corr에서 엄격한 행동 기준의 미도달이 나타났으나, 어떤 Probe 임곗값에서 지연을 볼 수 있는지는 백본과 데이터셋에 따라 달랐다. 이미지 구성, 클래스 불균형, 반사실 종류 및 학습 시계열이 다르므로 이 차이를 데이터셋 효과 또는 백본과 데이터셋의 인과적 상호작용으로 분리할 수 없다. 그래프는 각 조건 자체의 초기 평균 대비 변화만 표현한다.

[데이터셋별 상대 Flip 변화 그래프](dataset_relative_flip_change.png)

## F. Waterbirds ResNet18 E51–E100 · PASS, 해석 WARN

Val 평균 **E26–E50 → E51–E100**, 단위 %다. VGG와의 동일 학습 길이 비교에는 포함하지 않았다.

| Corr | WGA | Flip | 조건부 C2I |
|---|---:|---:|---:|
| .95 | 5.31 → 5.12 | 66.86 → 66.41 | 56.50 → 55.96 |
| .80 | 11.39 → 10.89 | 36.13 → 36.15 | 25.24 → 25.29 |
| .65 | 10.80 → 11.30 | 19.81 → 19.66 | 13.74 → 13.69 |

뚜렷한 추가 강건성 향상은 관측되지 않았다. .80은 후기 최저 Flip E56에서 E63까지 **+5.00%p** 반등 후 E64에 회복했다. 같은 조건의 조건부 C2I는 E56→E81 **+4.04%p** 악화했고 E100까지 E56 수준으로 회복하지 않았다. .65의 E50→E100 Core Probe는 **57.14→58.93%**, Spurious는 **76.79→76.79%**였다. .95의 Core/Spurious는 E50과 E100에 **50.00/92.86%**로 같았다. 예측 클래스 CAM의 새 영역 평균 질량은 세 조건 모두 E75→E100에 원본·CF 각각 0.5%p 이내로 움직였다.

[후기 지표, 악화·회복 및 Probe CSV](waterbirds_resnet_late.csv) · [Waterbirds ResNet18 E0–E100 그래프](waterbirds_resnet_e000_e100.png)

## G. Waterbirds VGG E100 추가 학습 · WARN, 목적에 따라 선택

공통 E0–E50은 관측된 백본 차이, H1의 예외 및 H2의 Probe/행동 불일치를 판단하기에 충분하다. ResNet18 E51–E100에서는 반등과 정체가 있었지만 60% Core 지속 기준이나 공동 행동 기준의 새로운 도달은 없었다. 이 결과만으로 VGG가 E50 이후 같은 형태를 보일지는 알 수 없다.

VGG Val E41–E50의 WGA 범위는 .95 **4.38–10.95%**, .80 **8.76–24.82%**, .65 **17.52–36.50%**였다. 같은 구간 Flip은 **61.22–70.31%**, **27.11–40.20%**, **12.93–21.27%**였다. 세 조건이 안정적으로 수렴했다고 판정할 근거가 부족하다. E100 동등 비교를 통해 후기 순위가 달라질 가능성은 열려 있다.

현재 E0–E50 결론을 위해 VGG E100은 **필수가 아니다**. 연구 질문이 후기 백본×학습 길이 상호작용이면 **중간 이상의 가치**가 있다. 그 경우 VGG E50을 같은 optimizer·스케줄과 데이터 조건으로 E100까지 잇고, 연장 전 원본 E50 결과를 보존하는 설계를 검토할 수 있다. E51–E100 평균·최종 10개 Epoch·지속 기준을 사전 명시하고, 백본·corr별 추가 seed 반복에서 재현성을 점검한다. 단일 E100 확장만으로 seed 변동성을 추정할 수는 없다.

## H. 최종 결론 및 한계 · PASS / WARN

통합 산출물과 범위 검증은 **PASS**다. H1은 전체적으로 **부분 지지**, H2의 실제 Core 학습 시점은 **판단 보류**, 행동 기준의 지연 해석은 **부분 지지**다. 단일 seed, Epoch 의존성, 기존 VGG 평가 누락, 근사적 layer 대응 및 CAM의 공간 집계 한계가 남는다. 검증 실패는 없었다. 기존 모델·체크포인트·평가 원본은 읽기 전용으로 사용했고, GPU 작업·SLURM 제출·Git commit/push는 수행하지 않았다.
