> GitHub 공유본: 집계 지표·대표 그림·재현 소스만 포함합니다. 원본 checkpoint/Feature, 전체 archive 감사 및 per-Epoch 예측·로그는 별도 보관합니다. 원본 자료가 필요한 검증과 per-sample 재계산은 이 공유본만으로 실행할 수 없습니다.

# Colored MNIST × VGG11_BN: 선형 정보와 비단조적 Shortcut 행동의 구분

대상은 기존 Epoch 50 실험 6개(background/foreground × corr .95/.80/.65), model seed 42다. 기존 review 보고서와 원본 결과를 보존했다. 이번에는 **저장 Feature의 작은 CPU Linear Probe**, 기존 집계 지표의 그룹 분해, late checkpoint의 head/BN state 읽기만 수행했다. 원본 모델의 추가 학습·추론, GPU, SLURM은 실행하지 않았다.

## 0. 중단 상태 확인 및 재개 결과

이전 Conversation interrupted 직후 후속 분석 디렉터리는 비어 있었다. 완료되어 있던 것은 기존 코드/데이터 구조 검토, 대표 Feature 확인, 동일 snapshot 내 grayscale 중복 없음 확인, 작은 Ridge 비용 점검이었다. **저장된 Probe 결과는 없었으므로 다시 실행할 기존 Probe 계산도 없었다.** 이전 review 분석 자체는 재실행하지 않고 기존 CSV/원본 History를 재사용했다. 확인한 프로세스에 후속 Probe 작업은 없었고, 이전 중단의 시스템 원인을 보여 주는 계산 오류 로그는 발견되지 않았다.

재개 후 split JSON 저장 시 NumPy int64 직렬화 오류가 한 번 발생했다. 이 오류는 이번 재개의 저장 코드 오류이며 과거 Conversation interrupted의 원인으로 단정하지 않는다. 수정 후 실행이 완료됐으며 `split_serialization_error.json`에 오류를 보존했다. `status.md`는 재개 당시 조사, `execution.log`는 단계별 로그, `progress.json`은 현재 상태를 기록한다. `batches/`에 run×Epoch별 JSON/예측 NPZ를 원자적으로 저장하며 재실행 시 완료분을 건너뛴다.

## 1. 기존 분석 검증 결과

원본 history/config/test/log의 SHA-256은 review 시점 기록과 일치한다. 원본 History의 모든 열이 기존 epoch_metrics.csv와 수치적으로 일치함을 확인했고, 새 분석에서는 이미 완료된 archive 전체 검증과 기존 그림을 다시 생성하지 않았다.

- WGA는 100개 색상 cell의 최소가 아니라 digit×{aligned,conflicting} 20그룹의 최소다. 기존 해석은 타당하다.
- Paired flip은 correct→incorrect, incorrect→correct, incorrect→다른 incorrect를 모두 포함한다. 단일 flip을 C2I로 해석하지 않은 것은 타당하다. 기존 C2I 정수 범위 공식도 같은 분모의 paired CF 집계에 대해 타당하다.
- All-color flip은 원래 색상 외 9개 개입 전체, All-color accuracy/color rate는 개입 색상≠label인 부분만 분모에 포함한다. 두 분모를 혼용하지 않은 것은 타당하다.
- Val/Test corr=.10은 10-class pairing all에서 독립 색상과 같은 aligned chance다. 90% conflicting을 이진 결정론적 역상관처럼 해석하면 안 된다.
- Train online−Val gap은 모델 평가 시점과 데이터 분포가 다르므로 기술 통계다. 기존 보고서가 IID generalization gap과 구분한 것은 타당하다.
- H1은 e1→50의 장기 행동 변화로만 지지되며 단조 감소로는 지지되지 않는다. H2의 기존 95% 기준은 **강건 행동 도달**이다. 이를 Core 학습 시작으로 부를 근거는 없으며, “H2 지지”라는 표현은 이 Proxy에만 제한해야 한다.

## 2. 기존 Probe 코드 검토와 이번 CPU 프로토콜

### 기존 코드의 의미

`probe_dynamics.py`는 checkpoint에서 모델을 로드하고 `encode_layers()`로 **새 이미지 forward**를 수행한다. `followup_common.py`의 CrossedColors는 각 원본 digit을 10색 모두로 재색칠한다. Probe train은 모델 Validation 원본에서 선택하고 probe test는 공식 MNIST Test에서 선택하여 두 source-image 집합을 분리한다. read_manifest는 원본 index 및 grayscale pixel 중복을 검사한다. Label은 core=숫자 class 0–9, shortcut=색상 index 0–9다. Model training index와의 중복도 check_config에서 거부한다. 이 프로토콜은 색상과 digit을 정확히 독립화한다.

하지만 새 인코딩이 필요하므로 기존 probe_dynamics.py의 main은 실행하지 않았다. 기존 evaluate.py는 Validation에서 probe를 학습하고 Test에서 평가하지만, 원본 상관 분포를 그대로 사용할 수 있어 label/color confounding에 주의해야 한다. 이번에는 해당 main도 실행하지 않았다. 기존 코드의 `probe_scores`와 `evaluate.DecodedCorrelation`은 CPU 민감도 분석에 재사용했다.

### 저장 Feature로 가능한 대안

snapshot은 모델 Validation의 3200개 원본을 label×color 100개 cell당 32개 선택한 것이다. **각 원본은 단일 색상**이며 CrossedColors의 동일 digit 10색 인코딩과 같은 데이터는 아니다. 3200개 source index가 고유하고 모든 Epoch/조건에서 ids/label/color가 정확히 일치함을 확인했다. 현재 고정 공식 캐시 기준 동일 grayscale pixels가 snapshot 내부 또는 원본 모델 Train에 중복되는지도 검사했고 **제외된 샘플은 0개**였다.

각 cell에서 24개를 probe train, 8개를 held-out 평가로 나눴다: **2400/800**. Train/test가 각각 정확한 10×10 joint balance라서 한 target에서 다른 target을 추측하는 분포상 shortcut을 줄였다. Split seed 1701/1702/1703의 3개 고정 분할을 모든 model/Epoch/layer/target에 재사용한다. 분할별 source images 및 pixels는 겹치지 않는다. 서로 다른 분할끼리는 겹칠 수 있어 이를 독립 model seed 반복으로 취급하지 않는다.

데이터 누수 방지: Feature 표준화(mean/std)는 probe train 2400개에서만 적합한다. Held-out labels는 fit/전처리/하이퍼파라미터 선택에 쓰지 않았다. 원본 모델 학습 데이터와 source index가 겹치지 않는다. 다만 pool 자체가 모델 Validation이었고 Epoch 곡선·가설도 사후 분석이므로 외부 독립 Test의 확증 실험은 아니다. pixel 중복 검사는 현재 새로 준비한 공식 raw revision에 근거하며, 과거 raw image 전체 동일성의 한계는 남아 있다.

### Probe 모델과 계산 범위

주 분석은 train-only z-score 후 **Ridge(alpha=1) one-hot 회귀 → 10-class argmax**다. Core/shortcut 두 독립 linear readout을 20개 RHS로 함께 계산했지만 target 간 weight를 공유하거나 서로를 입력으로 주지 않는다. intercept는 target centering으로 처리했다. 이산적인 선형 해독 가능성의 측정이며 낮은 값이 선형 분리 불가능의 수학적 증명은 아니다.

- 주 곡선: features.3(64차원), avgpool(512차원)의 Epoch 0–50 전체.
- 추가 레이어: features.7/.14/.21을 사전 고정한 15 Epoch에서 비교.
- features.28과 avgpool은 **306개 파일 모두에서 값이 완전히 같아** 독립된 레이어 증거로 중복 계산하지 않았다.
- 민감도: 사전에 고른 20개 snapshot의 주 분할에서 Ridge alpha=.1/10, 기존 DecodedCorrelation(AdamW lr=.001, wd=.0001, batch256, 100 epochs)을 train-only 표준화 입력으로 비교했다. 이는 encoder 재학습이 아니다.
- 총 5412개 Probe 결과 행: 주 Ridge 5292 + alpha 민감도 80 + softmax 40. CPU 1 thread로 수행했고 checkpoint/원본 Feature에 쓰지 않았다.
- 구현 검증: Ridge 예측이 sklearn Ridge와 일치했다. Train labels를 섞은 24개 음성 대조는 평균 9.46%, 범위 8.25–11.13%로 chance 10% 근처였다.

Chance는 두 target 모두 10%다. 기존 2×accuracy−1은 chance에서 −.8이므로 실제 상관계수나 chance-zero 지표로 부르지 않는다. CSV에는 accuracy 및 (accuracy−.1)/.9를 함께 기록했다.

## 3. Epoch별 Core/Shortcut 정보 변화

최종 pooled layer의 3개 고정 분할 평균이다. Shortcut 해독성이 높다는 것은 분류기가 색상을 실제 사용한다는 증명이 아니다.

| Mode | corr | Core e0 | Core e1 | Core e2 | Core e50 | Shortcut e50 |
|---|---|---|---|---|---|---|
| background | 0.95 | 86.38 | 92.08 | 95.75 | 97.79 | 100.00 |
| background | 0.8 | 86.38 | 96.38 | 96.92 | 98.04 | 100.00 |
| background | 0.65 | 86.38 | 97.12 | 97.46 | 98.21 | 100.00 |
| foreground | 0.95 | 80.83 | 91.79 | 95.46 | 97.21 | 100.00 |
| foreground | 0.8 | 80.83 | 96.25 | 97.00 | 98.08 | 100.00 |
| foreground | 0.65 | 80.83 | 96.42 | 97.25 | 97.88 | 99.92 |

무학습 e0에서 이미 Core가 BG 86.38%, FG 80.83%로 해독 가능하다. 입력 digit 정보가 random representation에도 남아 있고 새 readout이 그것을 학습할 수 있다는 뜻이다. 따라서 높은 Core Probe를 본 순간을 원본 모델이 Core를 처음 학습한 시점으로 지정할 수 없다. e0에서 같은 mode의 세 correlation 결과가 같은 것도 같은 초기 모델/동일 snapshot 배정과 일치한다.

학습 후 Core 해독성은 초기에 높아지고 최종에 약 97–98% 수준이다. 동시에 Shortcut은 대부분 100%에 가깝게 계속 해독 가능하다. **행동적 색상 민감도의 감소는 표현에서 색상 정보가 사라지는 것과 동일하지 않다.** 하위/상위 레이어 차이는 그림과 probe_curves.csv로 제공한다. 차원수·pooling·정규화·readout 제약도 다르므로 레이어 정확도를 순수한 인과적 정보량으로 해석하지 않는다.

### 결과를 보기 전에 고정한 관측 기준

protocol.json에 다음을 사전 고정했다: avgpool에서 세 분할 **모두** held-out accuracy≥90%(보조95%)를 연속 3 Epoch 만족하는 최초 구간 시작. e0도 포함한다. 이 규칙은 이번 Probe 점수를 보기 전에 정했지만 연구 전체가 사전등록된 것은 아니다. 명칭은 **선형 해독성 기준 도달**이며 Core 학습 시작의 직접 추정이 아니다.

| Mode | corr | Core90% Epoch | Core95% Epoch | Shortcut95% Epoch | 행동95% Epoch |
|---|---|---|---|---|---|
| background | 0.95 | 1 | 2 | 0 | 46 |
| background | 0.8 | 1 | 1 | 0 | 3 |
| background | 0.65 | 1 | 1 | 0 | 2 |
| foreground | 0.95 | 1 | 3 | 0 | 미도달 |
| foreground | 0.8 | 1 | 1 | 0 | 3 |
| foreground | 0.65 | 1 | 1 | 0 | 2 |

90% Core 기준은 모든 조건에서 e1로 같고, 95%에서는 .95가 BG e2/FG e3으로 약한 correlation의 e1보다 늦다. 그러나 기존 행동 기준의 BG e46/FG 미도달과 차이가 매우 크다. Threshold를 바꾸면 결론의 강도도 달라진다. Shortcut95%는 e0부터 만족해 최초 생성 시점이 관측 창의 왼쪽에서 잘린다. 임의의 Core 학습 시작 Epoch를 새로 부여하지 않는다.

## 4. 후기 이상 Epoch 분석: 행동, 그룹, Probe, checkpoint state

### Background .95 e28 및 e49

같은 원본 Metrics와 같은 고정 Probe 분할을 사용한 비교다. Probe 평균은 3개 분할, 행동은 전체 Validation 5000개다.

| Epoch | Val acc | WGA | All flip | Core Probe mean | Core Probe min–max | Shortcut Probe mean |
|---|---|---|---|---|---|---|
| 27 | 96.28 | 92.86 | 4.97 | 98.00 | 97.62–98.38 | 100.00 |
| 28 | 93.68 | 77.41 | 8.05 | 97.75 | 97.38–98.12 | 100.00 |
| 29 | 96.06 | 91.47 | 5.20 | 97.71 | 97.50–98.00 | 100.00 |
| 46 | 96.56 | 93.78 | 4.56 | 97.62 | 97.25–98.00 | 100.00 |
| 47 | 96.56 | 93.55 | 4.64 | 97.42 | 97.25–97.50 | 100.00 |
| 48 | 96.40 | 91.71 | 4.74 | 97.79 | 97.50–98.00 | 100.00 |
| 49 | 95.78 | 88.99 | 5.78 | 97.67 | 97.12–98.12 | 100.00 |
| 50 | 96.40 | 93.09 | 4.82 | 97.79 | 97.25–98.25 | 100.00 |

e27→28의 WGA는 92.86→77.41%, All-color flip은 4.97→8.05%다. e29에는 WGA 91.47%, Val 96.06%로 회복한다. 전체 정답 수는 e27→28에 130개 순감소했다. digit 5 conflicting 그룹(394개)의 정답은 **390→305**, 즉 85개 순감소로 전체 순감소의 **65.4%**를 차지한다. e29에는 386개 정답으로 회복했다. 감소가 이 그룹에 집중됐지만 digit7/1/0 conflicting도 함께 악화됐다.

| 그룹 | n | 정답 전→후 | 정확도 전→후 | 정답 순감소 | 전체 순감소 대비 |
|---|---|---|---|---|---|
| g5other | 394 | 390→305 | 98.98→77.41 | 85 | 65.38% |
| g7other | 475 | 451→425 | 94.95→89.47 | 26 | 20.00% |
| g1other | 503 | 499→477 | 99.20→94.83 | 22 | 16.92% |
| g0other | 439 | 430→411 | 97.95→93.62 | 19 | 14.62% |
| g4other | 451 | 433→425 | 96.01→94.24 | 8 | 6.15% |

이는 집계 정답 수의 순차이이며 같은 85개 샘플이 모두 correct→incorrect로 바뀌었다는 뜻은 아니다. 다른 그룹의 개선이 상쇄하므로 기여율 합이 100%를 넘을 수 있다. 원본 개별 예측은 없어 동일 샘플의 분류 전환/마진은 확인할 수 없다.

e48→49는 전체 정답이 31개 순감소하며 digit8 conflicting 22개, digit7 conflicting 15개, digit0 conflicting 11개 순감소가 크다. e28처럼 digit5에 집중된 사건은 아니다. e50에서 총 정답은 다시 31개 늘고 All-color flip도 4.82%로 복귀한다. e46–48 기준 충족을 영구적인 전환으로 해석할 수 없다.

### 행동 악화와 선형 정보 손실은 같은가?

Core Probe 평균은 e27→28에 **98.00→97.75%**, e48→49에 **97.79→97.67%**다. Primary 분할의 digit5 conflicting held-out 72개는 e27/e28 모두 Core readout이 정답이었다. 이는 해당 표본에서 digit 정보를 읽을 수 있음을 보이며 전체 394개 또는 원래 classifier의 판단을 대신하지 않는다.

아래 bootstrap은 같은 primary 분할의 held-out image를 재표집한 10000회 조건부 구간이다. Probe fit/모델 seed는 고정이다. 사전 정한 사건 비교만 제시하며 다중 비교 보정이나 모델 seed 불확실성을 포함하지 않는다.

| 변화 | Target | 개선/악화 표본 | Probe 차이(pp) | 조건부 bootstrap 95% 구간(pp) |
|---|---|---|---|---|
| e27→e28 | core | 3/5 | -0.250 | [-1.000, +0.375] |
| e28→e29 | core | 3/4 | -0.125 | [-0.750, +0.500] |
| e48→e49 | core | 4/2 | +0.250 | [-0.375, +0.875] |
| e49→e50 | core | 2/4 | -0.250 | [-0.875, +0.375] |

대폭적인 선형 해독성 붕괴는 관찰되지 않았다. 이 작은 구간이 정보의 완전한 동일성이나 원인의 부재를 증명하지는 않는다. Shortcut은 모든 사건에서 주 readout이 100%였다. 800개 held-out에서 관측 오답이 없다는 사실을 모집단 오류율 0으로 확대하지 않는다. 원본 모델의 Feature “사용”은 새 readout이 달라진 decision boundary로 읽는 것과 다르다.

### CPU checkpoint 상태 진단

BG .95의 e25–50 checkpoint 26개를 순차 CPU 로드해 head/BN state만 비교했다. 모델 forward나 state 교체는 하지 않았다. e27→28의 classifier weight 상대 L2 변화는 **0.417%**, 후기 25개 전환 중 가장 크다. BN running variance도 8개 BN 중 5개에서 이 전환의 상대 변화가 가장 컸다. 반면 digit5 head row 자체는 상대 변화 순위 5/25라 특정 head row 폭주로 단정할 수 없다. e48→49 classifier 변화는 **0.164%, 순위17/25**로 큰 jump가 아니다.

이 결과는 e28에서 head와 정규화 통계 변화가 동반됐다는 **추가 관찰**이다. BN이 원인인지, weight/head가 원인인지, 샘플 margin이나 상호작용 때문인지는 state 교환/고정 비교 없이는 분리되지 않는다. 단순 checkpoint norm으로 shortcut 사용의 인과를 증명하지 않는다. e49의 threshold 재이탈도 Core 정보 소실로 설명할 직접 증거는 없다.

## 5. H1/H2의 더 엄밀한 평가

### H1: Epoch 증가와 Shortcut 의존성 감소

- **직접 관찰에 가까운 행동 증거:** 같은 이미지의 색상 개입에 대한 prediction sensitivity는 e1→50에서 크게 감소한다. CF 정확도 증가·color-following 감소와 함께 해석해야 한다. 이것이 개입으로 측정한 색상 민감도의 증거다.
- **비단조적 반례:** 49개 인접 Epoch 전환 중 flip 증가가 18–23회이고, 동시에 CF accuracy가 내려가는 전환도 13–20회다. 따라서 엄격한 단조 감소는 기각되는 기술적 진술이다. 통계적 유의성 검정을 했다는 뜻은 아니다.
- **표현 측면의 반례/구분:** Shortcut Probe는 e0부터 높고 후기에도 거의 100%다. 색상 정보 삭제로 설명하는 형태의 H1은 지지되지 않는다. 행동과 representation decodability가 분리된다.
- **간접 통계:** 아래 Spearman은 시간과 관측 flip의 서술적 관계다. Epoch를 독립 표본으로 가정하는 p-value를 계산하지 않았다.

| Mode | corr | e1→50 All flip | 증가 step/49 | Flip↑ 및 CF acc↓ step | Epoch–flip Spearman | 후기 slope(pp/Epoch) |
|---|---|---|---|---|---|---|
| background | 0.95 | 91.61→4.82 | 21 | 19 | -0.829 | -0.0278 |
| background | 0.8 | 12.42→1.36 | 23 | 16 | -0.915 | -0.0097 |
| background | 0.65 | 8.42→0.87 | 18 | 13 | -0.836 | -0.0004 |
| foreground | 0.95 | 82.25→5.77 | 20 | 20 | -0.805 | -0.0084 |
| foreground | 0.8 | 16.54→1.63 | 20 | 14 | -0.790 | -0.0022 |
| foreground | 0.65 | 6.56→0.94 | 22 | 13 | -0.787 | -0.0007 |

**H1 판정:** 이 실험의 장기적인 행동적 색상 의존성 감소는 지지된다. 매 Epoch 감소, e0부터 감소, 색상 정보가 표현에서 사라짐, 다른 seed로의 법칙은 지지되지 않거나 판단할 수 없다.

### H2: Correlation이 강하면 Core 학습 시작이 늦어지는가?

- **직접 확인한 것:** 정해진 Linear readout에서 digit label 정보가 초기에 해독 가능하고 .95에서도 e1–3이면 높은 정확도가 나온다. 동일 색상/label joint balance와 고정 평가 분할에서 측정했다.
- **약한 Proxy 지지:** 95% 해독성 기준은 .95가 1–2 Epoch 늦다. 그러나 90% 기준은 모든 조건이 e1이다. 이는 수준 기준 도달에 대한 threshold-sensitive 관찰이다.
- **강한 행동 Proxy 지지:** .95의 원래 classifier가 색상 개입에 강건해지는 시점은 크게 지연된다. BG e46/FG 50내 미도달이다.
- **반례/예외:** 무학습 e0에도 digit이 80–86%로 해독 가능하다. “Core 정보가 없었다가 e46에 처음 생겼다”는 해석은 이번 Probe와 맞지 않는다. 높은 상관에서도 정보를 읽을 수 있는 시점과 classifier가 그 정보를 안정적으로 활용하는 시점은 다르다.
- **최종 판정:** 원문 H2의 **실제 Core 학습 시작 지연은 확정할 수 없다**. 이를 강건 행동 지연으로 바꾸면 지지된다. Linear decodability의 높은 수준 도달에는 약한 threshold 의존적 지지가 있지만, 실제 학습 시작 Epoch는 지정하지 않는다.

## 6. 통계적·방법론적 한계

1. Main-model seed는 42 한 개다. 세 Probe 분할은 같은 모델/데이터의 반복 partition이고, Epoch도 독립 반복 실험이 아니다. 모집단 수준 유의성이나 seed 간 신뢰구간을 제시하지 않는다.
2. Balanced validation snapshot의 within-pool 일반화다. 공식 Test로의 외부 일반화, 모든 색상을 같은 digit에 교차한 인과적 분리, 다른 MNIST instance로의 안정성은 별도다.
3. Pooling/차원/regularization/standardization 및 Ridge와 softmax solver가 결과에 영향을 준다. 저장된 다른 layer 및 별도 solver 민감도 결과를 함께 제공했지만 모든 하이퍼파라미터를 탐색해 최적 성능을 주장하지 않았다.
4. Main-model 행동 평가와 Probe 평가는 표본 수/구성, classifier, dtype 경로가 다르다. Feature snapshot은 FP32, 원래 Validation은 AMP 설정이며 각 Probe는 새 head다. 작은 숫자 차이나 불일치를 한 원인으로 단정하지 않는다.
5. 과거 raw revision/image hashes 부재 때문에 현재 raw 캐시 기반 pixel 중복 검사가 과거 입력의 byte 동일성까지 소급 증명하지 않는다. 원본 index/label/color 일치는 확인했다.
6. 원본 per-sample logits/예측이 없어 main-model C2I, 특정 이미지의 margin, probe와 main classifier 오류의 일치를 직접 분석할 수 없다. 기존 집계 C2I 범위는 그대로 유효하다.
7. Bootstrap CI는 고정 readout과 800개 평가 표본에 조건부다. 0을 포함한다고 동등성이 증명되는 것은 아니고, 100% Shortcut에서 퇴화한 경험적 구간은 진짜 오류율 0을 뜻하지 않는다.

## 7. 추가 검증의 우선순위와 자원 계획 — 실행하지 않음

**우선순위 1: 재인코딩 없는 CPU 검증의 반복 가능성.** 이번 저장 결과와 scripts로 재현 가능하다. 주 fit은 1 CPU/BLAS1에서 약 233초였고 실제 snapshot은 한 번에 하나씩 읽었다. peak RSS는 측정하지 않았으므로 실측값을 만들지 않는다. 같은 규모 재실행은 CPU1–4, RAM 4GiB를 보수적으로 요청할 수 있다. 이미지 backbone 학습은 없다. 완료된 batches가 있으면 fit을 건너뛴다.

**우선순위 2: e28 원인 분리용 소규모 GPU 추론.** e27/e28의 backbone/head 및 BN running state를 메모리에서 교차 고정한 4개 조건부터 비교한다. 원본 checkpoint는 불변이다. 동일 Validation의 per-sample prediction/logit/margin을 저장하고 paired 이미지 단위로 비교해야 한다. 기본 원본+paired CF는 조건당 10000 image forward, 4조건 40000개다. All-color까지 하면 조건당 55000개, 4조건 220000개다. 관계가 확인된 뒤 BN 개별 ablation을 확장한다. 합성 state의 out-of-distribution 효과가 있어 그것만으로 인과를 단정하지 않는다.

**우선순위 3: 기존 probe_dynamics.py의 외부 Test crossed-color protocol.** mode별 manifest를 prepare하고 같은 manifest를 해당 mode의 3 correlation에 사용한다. per_digit32면 probe train/test 각각 320 originals×10색=3200 inputs다. 선택 Epoch [0,1,2,3,27,28,29,46,47,48,49,50]의 12개×6조건이면 encoding은 72×6400=460800 image forwards, 모든 e0–50이면 306×6400=1958400개다. 반응별 layer GAP feature를 저장한 뒤 CPU readout을 돌리면 GPU에서 Probe100 epochs를 반복할 필요를 줄일 수 있다. read_manifest의 원본/pixel 중복 검사 및 check_config를 유지해야 한다. 기존 script의 main은 재인코딩을 수행하므로 이번에는 실행하지 않았다.

자원 요청은 기존 task.sh와 같은 비율인 **1 GPU당 CPU4, RAM16GiB**를 기준으로 하며, 2 GPU를 동시에 요청한다면 CPU8/RAM32GiB가 필요하다. GPU마다 독립 Task를 권장한다. batch64, workers≤4, thread4로 시작하고 실제 노드 메모리를 pilot에서 측정한다. VGG11_BN의 이전 AMP 합성 메모리 기록은 있었지만 이 GPU 후속 경로의 실측 peak나 시간은 없으므로 고정 시간을 주장하지 않는다. 유효 처리량 q(images/s)를 pilot으로 측정한 뒤 encoding GPU-hours=입력수/q/3600으로 산정하고 I/O·queue·readout은 별도다. 예컨대 q=100이라는 가정에서 460800 inputs는 1.28 GPU-hours이며 **실측 추정값이 아니다**. 현재 클러스터 account/QOS/partition 및 GPU당 정책을 제출 직전에 재확인해야 한다.

**우선순위 4: 여러 model seed 및 더 촘촘한 초기 관측.** Core 학습 시작 지연을 실제로 묻는다면 목적과 측정 기준을 사전 정의하고 seed 반복, e0→e1 사이 step-level frozen evaluation, representation/사용 개입을 함께 설계한다. 새 모델 학습은 이번 범위 밖이다. 현재 결과로 이 비용을 대신 검증했다고 주장하지 않는다.

## 8. 완료·미완료 목록 및 재개 방법

완료: 기존 분석 검증, 누수/중복 감사, 프로토콜 사전 고정, 306개 snapshot의 CPU Probe, 고정 3분할 및 solver/alpha/label-permutation 검증, e28/e49 그룹 분해와 조건부 paired 비교, late checkpoint head/BN 상태 비교, H1/H2 재평가, 표/그림/단계별 파일 저장.

미완료 또는 현재 자료로 불가: 외부 Test crossed-color Probe, 원본 모델의 per-sample transition/margin, BN/head 교차 추론에 의한 원인 분리, 여러 main-model seed, 실제 Core 학습 시작 시점. 모두 이유와 자원 계획을 위에 제시했으며 GPU/새 모델 학습/SLURM을 실행하지 않았다. 이는 허용된 기존 자료 분석의 누락이 아니라 추가 자료/실험이 필요한 항목이다.

재개/재현 명령(기존 결과를 건너뛰며 원본은 읽기 전용):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/dada0228/conda_envs/shortcut-xai/bin/python reports/cmnist_vgg50_probe_followup/run_probes.py
/data/dada0228/conda_envs/shortcut-xai/bin/python reports/cmnist_vgg50_probe_followup/analyze_events.py
CUDA_VISIBLE_DEVICES='' /data/dada0228/conda_envs/shortcut-xai/bin/python reports/cmnist_vgg50_probe_followup/checkpoint_drift.py
/data/dada0228/conda_envs/shortcut-xai/bin/python reports/cmnist_vgg50_probe_followup/summarize.py
/data/dada0228/conda_envs/shortcut-xai/bin/python reports/cmnist_vgg50_probe_followup/write_report.py
```

- protocol.json/split_manifest.json: 결과 전 고정한 방법과 모든 split row/source identity.
- batches/: run×Epoch별 완료 결과/primary avgpool held-out 예측, 중단 후 재사용 단위.
- probe_results.csv/probe_curves.csv/probe_summary.csv/decodability_times.csv: 전체 readout, 3분할 통계, 핵심 비교, 사전 기준 도달.
- sensitivity_results.csv/permutation_controls.csv/probe_sanity_checks.json: solver·alpha 및 label shuffle 검증.
- event_metrics.csv/group_event_deltas.csv/anomaly_combined.csv/probe_event_group_scores.csv: 행동과 Probe의 사건 전후 비교.
- probe_event_paired_bootstrap.csv: primary 고정 분할의 조건부 image bootstrap.
- checkpoint_drift.csv/checkpoint_sources.json: late head/BN state의 기술적 변화와 원본 provenance.
- trajectory_nonmonotonicity.csv/review_validation.json/feature_source_manifest.json: 기존 분석 검토·비단조성·입력 파일 provenance.
- validate_results.py/validation_results.json: 원본 Feature306개·late checkpoint26개 및 이전 보고서/지표의 SHA 보존, 표/그림, 완료306개 batch를 fit 없이 재개하는 검사.
- completion.json: 완료/추가 자료가 필요한 항목의 최종 목록.
- environment.json: CPU 실행 라이브러리 버전 및 관련 원본 코드 hash.

그림은 PNG/PDF를 모두 제공하며 보간하지 않았다. band는 3개 partition min–max이며 CI가 아니다.

1. [Epoch별 Core/Shortcut Probe](figures/01_probe_epoch_curves.png)
2. [초기 Core 해독성과 사전 기준](figures/02_early_core_decodability.png)
3. [Core 레이어 비교](figures/03_layers_core.png), [Shortcut 레이어 비교](figures/03_layers_shortcut.png)
4. [이상 Epoch 전후 행동과 Probe](figures/04_anomaly_windows.png)
5. [그룹별 정답 순감소](figures/05_group_error_changes.png)
6. [선형 해독성과 행동 강건성](figures/06_probe_vs_behavior.png)
