> GitHub 공유본: 집계 지표·대표 그림·재현 소스만 포함합니다. 원본 checkpoint/Feature, 전체 archive 감사 및 per-Epoch 예측·로그는 별도 보관합니다. 원본 자료가 필요한 검증과 per-sample 재계산은 이 공유본만으로 실행할 수 없습니다.

# Colored MNIST VGG11_BN: Epoch 50 Shortcut Learning Dynamics 재분석

분석 범위는 Job 153567의 **Epoch 50 실험 6개**다. Epoch 100 Resume 및 그 실패 로그는 섞지 않았다. 기존 보고서·분석 코드는 보존했고, 기존 분석 구현을 이 디렉터리에 재사용해 원본 Metrics에서 다시 계산했다. 이번 요청의 Generalization Gap과 Correct-to-Incorrect 식별 가능성을 추가했다. GPU, 추가 추론/실험, srun, sbatch는 실행하지 않았다. 비율 표시는 %, 차이는 pp이며 CSV의 비율은 0–1이다.

## 1. 실험 완료 현황 및 설정 검증

먼저 task.sh의 명시적 경로에서 6개 run을 선택했다. 각 run의 config/history/test/latest/success, Job 153567의 두 로그, epoch 0–50 checkpoint/feature만 읽었다. 프로젝트 전체나 다른 run은 탐색하지 않았다.

| Task | Mode | Train corr | History Epoch | Checkpoint | Feature | Test |
|---|---|---|---|---|---|---|
| 0 | background | 0.95 | 0–50, 51행 | 0–50, 51개 | 0–50, 51개 | e50 / 10000개 |
| 1 | background | 0.8 | 0–50, 51행 | 0–50, 51개 | 0–50, 51개 | e50 / 10000개 |
| 2 | background | 0.65 | 0–50, 51행 | 0–50, 51개 | 0–50, 51개 | e50 / 10000개 |
| 3 | foreground | 0.95 | 0–50, 51행 | 0–50, 51개 | 0–50, 51개 | e50 / 10000개 |
| 4 | foreground | 0.8 | 0–50, 51행 | 0–50, 51개 | 0–50, 51개 | e50 / 10000개 |
| 5 | foreground | 0.65 | 0–50, 51행 | 0–50, 51개 | 0–50, 51개 | e50 / 10000개 |

공통 실제 config: dataset=colored_mnist, model=vgg11_bn, seed=42, epochs=50, image_size=224, batch_size=128, num_workers=4, SGD(lr=.01, momentum=.9, weight_decay=.0001), scheduler=none, amp=true, pretrained=false, augment=false, color_pairing=all, mnist_val_size=5000, val/test corr=.10, features_per_group=32. Mode/train corr 외의 모든 저장 config 항목이 같다. 각 로그의 config도 해당 config.json과 정확히 일치한다. CF chunk=128은 task.sh 인자이며 원본 config에는 기록되지 않아 실행값을 독립 확인한 것으로 표현하지 않는다.

6개 모두 TRAIN_SUCCESS, FINAL TEST, 마지막 Run directory 로그가 존재한다. stderr는 HF Hub 비인증 경고뿐이며 traceback/OOM은 없다. History Epoch 중복/누락 없음, 주요 값 유한성·범위, Validation 5000개 및 group counts 고정, Test 10000개, latest epoch=50, CF drop=acc−CF acc, 20개 merged group으로 계산한 WGA를 검증했다. 원본 소규모 Metrics/config/log SHA-256은 이전 분석 기록과도 일치한다.

체크포인트 306개는 파일 존재·ZIP 구조·Epoch 메타데이터를 확인했다. tensor 전체를 로드하거나 재추론하지 않았으므로 모든 tensor payload의 CRC/수치 유효성까지 재검증한 것은 아니다. Feature NPZ 306개는 파일별 순차 읽기로 3200개 고유 index, label×color 100그룹당 32개, 6개 레이어 배열 shape 및 유한값을 검사했다. 원본 파일은 쓰지 않았다.

기존 분석 자료: Feature는 존재하지만 이 6개 run에 evaluation_e050, probes, gradcam_comparison은 없다. Training online accuracy/loss와 Epoch별 Validation 및 CF는 분석 가능하다. **Epoch별 Test와 직접 저장한 Correct-to-Incorrect는 없다.** Checkpoint가 존재하는 것과 해당 평가가 이미 수행된 것은 구별한다. SLURM accounting은 이번에 조회하지 않았으며 완료 판단은 실제 산출물과 종료 로그 기준이다.

## 2. 조건별 주요 지표 요약

Epoch 50 결과다. WGA/CF는 Validation 값이며 전체 Test 지표는 CSV에 포함했다.

| Mode | corr | Train | Val | Test | Aligned | Conflict | WGA | Paired flip | All flip | All CF acc | Color rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| background | 0.95 | 100.00 | 96.40 | 96.71 | 100.00 | 96.00 | 93.09 | 3.96 | 4.82 | 96.03 | 2.68 |
| background | 0.8 | 100.00 | 98.54 | 98.66 | 100.00 | 98.38 | 96.67 | 1.44 | 1.36 | 98.53 | 0.62 |
| background | 0.65 | 100.00 | 98.88 | 99.08 | 100.00 | 98.75 | 97.11 | 1.04 | 0.87 | 98.89 | 0.38 |
| foreground | 0.95 | 100.00 | 95.56 | 95.93 | 100.00 | 95.06 | 92.13 | 4.82 | 5.77 | 95.33 | 3.11 |
| foreground | 0.8 | 100.00 | 98.34 | 98.55 | 100.00 | 98.15 | 96.77 | 1.68 | 1.63 | 98.42 | 0.70 |
| foreground | 0.65 | 100.00 | 98.86 | 99.01 | 99.80 | 98.75 | 96.54 | 1.04 | 0.94 | 98.76 | 0.43 |

Train은 모든 조건에서 100%지만 .95는 최종에도 높은 색상 민감도와 상대적으로 낮은 held-out 정확도를 보인다. 반대로 paired CF accuracy는 모든 조건에서 매우 높아, 그것만으로 조건을 구분하면 의존성 차이를 과소평가한다.

Generalization Gap은 요청에 맞춰 **저장 online Train−Validation** 차이를 계산했다. Train은 학습 batch별 갱신 중 예측, Val은 Epoch 종료 모델의 평가이며 Train/Val의 색상 분포도 다르다. 따라서 이는 분포 이동 아래의 기술적 차이이며, 동일 모델·동일 분포에서 평가한 통상적 IID generalization gap 또는 순수 과적합의 추정량이 아니다. Test gap은 e50에서만 계산했고 e0 Train NaN은 그대로 유지했다.

| Mode | corr | Train−Val e1(pp) | Train−Val e50(pp) | Train−Test e50(pp) |
|---|---|---|---|---|
| background | 0.95 | 73.42 | 3.60 | 3.29 |
| background | 0.8 | 1.85 | 1.46 | 1.34 |
| background | 0.65 | 0.10 | 1.12 | 0.92 |
| foreground | 0.95 | 63.55 | 4.44 | 4.07 |
| foreground | 0.8 | 3.31 | 1.66 | 1.45 |
| foreground | 0.65 | -5.07 | 1.14 | 0.99 |

## 3. Epoch별 학습 Dynamics

초기 1–5, 중기 6–25, 후기 26–50을 공통 구간으로 사용한다. 아래는 All-color flip 구간 평균이다. 무학습 e0은 학습 후 감소 기준에서 제외했다.

| Mode | corr | 초기 mean | 중기 mean | 후기 mean | Conflict e1→e50 | All flip 증가 step/49 | 최대 Conflict 개선 Epoch/pp |
|---|---|---|---|---|---|---|---|
| background | 0.95 | 39.62 | 7.43 | 5.23 | 12.64→96.00 | 21 | e1→e2 / +51.28 |
| background | 0.8 | 5.72 | 2.06 | 1.51 | 87.07→98.38 | 23 | e1→e2 / +7.03 |
| background | 0.65 | 3.70 | 1.35 | 0.87 | 87.99→98.75 | 18 | e1→e2 / +7.99 |
| foreground | 0.95 | 36.51 | 8.29 | 6.08 | 23.65→95.06 | 20 | e1→e2 / +45.98 |
| foreground | 0.8 | 6.63 | 2.25 | 1.70 | 84.78→98.15 | 20 | e1→e2 / +9.79 |
| foreground | 0.65 | 3.34 | 1.48 | 1.03 | 92.10→98.75 | 22 | e1→e2 / +4.34 |

**관찰:** 6개 모두 초기→중기→후기의 평균 All-color flip이 감소한다. e1→50의 conflicting/All-color accuracy도 높아진다. 특히 .95는 첫 Epoch에 높은 Train과 낮은 conflicting accuracy가 공존하고 e1→2에서 큰 개선을 보인다. .65/.80은 e1부터 비교적 높은 강건 정확도를 보이며 개선 여지가 작다. 초기부터 Core를 언제 처음 배웠는지는 Epoch 내 평가가 없어 알 수 없다.

후기 안정성은 아래와 같다. 범위와 표준편차는 e26–50의 기술 통계이며 독립 seed 간 불확실성이나 신뢰구간이 아니다.

| Mode | corr | Val 범위(pp) | WGA 범위(pp) | All flip 범위(pp) | All flip SD(pp) | 95% 기준 e50 충족 | 첫 3회 구간 이후 재이탈 Epoch |
|---|---|---|---|---|---|---|---|
| background | 0.95 | 2.96 | 16.37 | 3.49 | 0.66 | 예 | 49 |
| background | 0.8 | 0.48 | 1.38 | 0.52 | 0.11 | 예 | 없음 |
| background | 0.65 | 0.46 | 1.70 | 0.31 | 0.07 | 예 | 없음 |
| foreground | 0.95 | 0.86 | 3.82 | 1.44 | 0.38 | 아니오 | 해당 없음 |
| foreground | 0.8 | 0.44 | 1.46 | 0.47 | 0.09 | 예 | 없음 |
| foreground | 0.65 | 0.44 | 1.61 | 0.23 | 0.06 | 예 | 없음 |

Background .95의 후기 예외는 e28이다. e27→28→29의 WGA는 92.86→77.41→91.47%, Val accuracy는 96.28→93.68→96.06%, All-color flip은 4.97→8.05→5.20%다. e28 최저 그룹은 digit 5의 conflicting 그룹(394개)이며 77.41%다. 그룹 count가 고정되고 WGA 재계산도 일치하므로 누락 Epoch나 빈 그룹으로 생긴 값으로 보이지 않는다. 원인은 별도 분석 없이 확정하지 않는다.

또한 Background .95는 e46–48에 95% 기준을 연속 충족한 후 e49 All-color flip=5.78%로 재이탈하고 e50=4.82%로 복귀했다. 최초 충족 e46을 안정적인 영구 전환으로 해석하지 않는다.

평균 개선은 매 Epoch 개선과 다르다. 모든 조건에 flip 증가 step이 있고 e0→1부터도 감소하지 않는 조건이 있다. Foreground .65는 e3→5에서 All-color flip 2.01→3.17%, conflicting accuracy 97.06→95.97%로 동반 악화된다. Background .95의 e30→40은 All-color flip 4.83→5.22%, All-color accuracy 95.99→95.64%다. 따라서 후기 곡선을 완전한 plateau 또는 단조 감소라고 표현하지 않는다. 전체 주요 지표의 구간 평균/최소/최대는 stage_summary.csv에 있다.

## 4. Correlation별 비교

두 mode 모두 e50의 All-color flip/color rate는 .95>.80>.65이고 강건 정확도·Test 정확도는 대체로 반대 순서다. 후반 고성능 영역으로 접근하지만 차이가 없어지지는 않는다. .95와 .65의 최종 Test 차이는 Background 2.37pp, Foreground 3.08pp다.

Core 학습 시점은 직접 측정하지 않았다. 기존 분석과 같은 행동 proxy를 적용했다: **conflicting accuracy≥95%, All-color conflicting CF accuracy≥95%, All-color flip≤5%를 연속 3 Epoch 만족하는 구간의 첫 Epoch**. 세 관측 후 회고적으로 확인하는 기준이다. 이번 수치를 보기 전 preregister된 규칙은 아니며 threshold 선택에 의존한다. threshold 90/98%(flip≤10/2%) 민감도를 함께 보고한다.

| Mode | corr | 90% proxy | 95% proxy | 98% proxy |
|---|---|---|---|---|
| background | 0.95 | 11 | 46 | 미도달 |
| background | 0.8 | 2 | 3 | 21 |
| background | 0.65 | 2 | 2 | 18 |
| foreground | 0.95 | 11 | 미도달 | 미도달 |
| foreground | 0.8 | 2 | 3 | 28 |
| foreground | 0.65 | 1 | 2 | 15 |

.95가 .80/.65보다 늦게 강건 행동에 도달하는 것은 두 mode와 세 threshold에서 관찰된다. 단 .95의 일부 기준은 50 Epoch 내 미도달이므로 “정확히 언제”가 아니라 “관측 구간 내 미도달”이다. 이는 Core를 전혀 배우지 않았다는 뜻이 아니다. 최초 강건 기준 도달과 Core 정보의 최초 형성, 기준 충족 후 영구 유지도 각각 다른 질문이다.

## 5. Color Mode 비교

최종 차이는 Foreground−Background로 표시했다. Accuracy 양수는 Foreground 우세, flip 양수는 Foreground 민감도가 더 높음을 뜻한다.

| corr | ΔTest(pp) | ΔAll CF acc(pp) | ΔAll flip(pp) | BG 95% Epoch | FG 95% Epoch |
|---|---|---|---|---|---|
| 0.95 | -0.78 | -0.70 | +0.95 | 46 | 미도달 |
| 0.8 | -0.11 | -0.11 | +0.27 | 3 | 3 |
| 0.65 | -0.07 | -0.13 | +0.08 | 2 | 2 |

.95에서 e1 Foreground는 Background보다 all-color flip이 낮다(82.25 vs 91.61%)가 e50에서는 더 높다(5.77 vs 4.82%). Foreground .65는 초기 90% 기준을 더 빨리 만족하지만, 최종 Test는 Background 99.08%, Foreground 99.01%로 차이가 0.07pp에 불과하다. shortcut 위치의 절대적 우열이 학습 전 구간에서 유지된다고 결론 내리지 않는다. 두 mode의 장기 감소와 강한 상관 지연 패턴은 공통이다.

## 6. Counterfactual 기반 Shortcut 의존성 및 지표 정의

현재 evaluate 구현과 데이터 생성 코드를 대조했다.

- Aligned/conflicting: label=color / label≠color. WGA는 100개 개별 cell 최소가 아니라 digit×{aligned,other}로 합친 **20개 그룹 최소**다.
- Paired flip `F/N`: 원본과 paired recolor의 예측 class가 달라진 비율. correct→incorrect뿐 아니라 incorrect→correct, incorrect→다른 incorrect도 포함한다.
- Paired CF acc `B/N`: 같은 샘플을 main↔alternate로 바꾼 후 정답률. CF drop은 원본 acc−CF acc다.
- All-color flip: 원래 색상 외 9색 모두에서 원본 예측과 비교한 변경 비율.
- All-color acc와 color rate: 9색 개입 중 **개입 색상≠label**인 항목만 분모에 포함한다. 전자는 예측=label, 후자는 예측=개입 색상이다. 단일 CF/원본과 분모가 달라 이 셋으로 paired와 같은 전환율 공식을 쓰지 않는다.

Val/Test는 약 90% conflicting이므로 paired CF는 대부분 aligned로 바뀐다. .95 e50에서도 paired CF acc가 BG 99.52%, FG 99.50%이고 drop이 −3.12/−3.94pp인 것은 색상 의존성이 완전히 없다는 증거가 아니다. All-color acc 96.03/95.33%, flip 4.82/5.77%, color rate 2.68/3.11%를 같이 보면 잔여 민감도가 드러난다. 원본 corr=.10은 10-class pairing all에서 독립 색상의 aligned chance와 같다. 90% 비정렬이라는 사실을 이진 문제의 결정론적 역상관과 혼동하지 않는다.

### Correct-to-Incorrect는 직접 관측되지 않았다

원본 정답 수 A, paired CF 정답 수 B, 전체 예측 변경 수 F, 표본 수 N만 저장되어 있다. C=correct→incorrect, U=incorrect→correct, W=incorrect→다른 incorrect라 두면 B−A=U−C, F=C+U+W다. 다중 class에서는 W가 있어 **C를 일반적으로 유일하게 정할 수 없다**. 따라서 CF drop이나 전체 flip을 C2I로 이름 바꾸지 않았고 관측 C2I 곡선도 만들지 않았다.

다만 저장 집계값과 모순되지 않는 정수 범위는 정확히 계산할 수 있다:

```text
max(0, A−B, F+A−N) ≤ C ≤ min(A, N−B, floor((F+A−B)/2))
```

이는 추정값·모델 재추론·신뢰구간이 아니라 집계값의 논리적 제약이다. 조건부 C2I는 C/A, 전체 표본 기준은 C/N로 분모도 구분해야 한다. 아래는 **Validation e50의 가능한 범위**이며 실제 발생 수는 모른다.

| Mode | corr | 가능한 C2I 개수 / 5000 | 전체 표본 기준 범위(%) | 원본 정답 조건부 범위(%) |
|---|---|---|---|---|
| background | 0.95 | 18–21 | 0.36–0.42 | 0.37–0.44 |
| background | 0.8 | 0–8 | 0.00–0.16 | 0.00–0.16 |
| background | 0.65 | 0–5 | 0.00–0.10 | 0.00–0.10 |
| foreground | 0.95 | 19–22 | 0.38–0.44 | 0.40–0.46 |
| foreground | 0.8 | 1–8 | 0.02–0.16 | 0.02–0.16 |
| foreground | 0.65 | 0–6 | 0.00–0.12 | 0.00–0.12 |

Epoch별 범위와 e50 Test 범위는 CSV에 있다. 범위가 좁아도 직접 저장된 pair prediction이 있다는 뜻은 아니다. 정확한 전환 분해에는 별도 per-sample 원본/CF 예측이 필요하다. 단일 flip 감소만으로 shortcut 감소를 선언하지 않고 conflicting/All-color 정확도 증가 및 color rate 감소를 동시에 해석했다.

## 7. H1/H2 검증 결과

**H1 — 장기 감소는 지지, 단조 감소는 지지되지 않음.**

- 지지 관찰: 6조건의 초기→중기→후기 평균 flip 감소, e1→50 강건 정확도 증가 및 color rate 감소. .95 all flip은 BG 91.61→4.82%, FG 82.25→5.77%다.
- 반대 관찰: 모든 조건에서 중간 flip 반등이 있고 무학습 e0부터 지속 감소하지 않는다. 낮은 e0 flip은 같은 무의미한 예측 유지로도 발생할 수 있다.
- 판단 불가: Core representation의 실제 정보 내용, 완전한 색상 불변성, unseen seed/모델/데이터로의 일반화. 색상 변화는 시각적 대비도 바꿀 수 있어 모든 오류 변화를 의식적인 shortcut 사용과 동일시하지 않는다.

**H2 — 강한 상관의 강건 행동 지연은 지지, Core 최초 학습 시점으로는 확정 불가.**

- 지지 관찰: 동일 proxy의 95% 기준은 BG .65/.80/.95에서 e2/e3/e46, FG에서 e2/e3/미도달이다. 90/98% threshold에서도 .95가 더 늦거나 관측 기간 내 미도달이다.
- 예상과 다른/제한 관찰: 모든 조건은 e1부터 일부 digit 성능을 보이며, 95% 도달을 Core 학습 시작으로 부를 수 없다. 약한 상관 두 조건의 차이는 threshold에 따라 같아지기도 하고, mode별 절대 우열도 바뀐다. 엄밀한 Core 지연 가설에 대한 직접적인 반증·확증 자료는 없다.
- 판단 불가: 내부 표상 형성 시점, 인과적 학습 메커니즘, seed 분산. 단일 seed이며 Epoch를 독립 실험 반복으로 취급하지 않았다. p-value나 통계적 유의성을 주장하지 않는다.

## 8. 예상과 다른 결과 및 가능한 원인

**관찰과 가능한 설명을 구별한다.** 아래 설명은 추가 분석 없이 원인으로 확정하지 않는다.

- 초기 e0 flip보다 e1 flip이 큰 조건: 무학습 모델의 같은 class 반복 예측에서, 학습 후 입력/색상에 반응하는 모델로 바뀐 결과일 수 있다. e0의 작은 flip을 강건 학습으로 해석하지 않는다.
- Paired CF accuracy가 원본보다 높음: 데이터 생성상 대부분 conflicting을 aligned로 바꾸는 개입 방향이 이를 설명할 수 있다. All-color conflicting 개입이 더 엄격한 보완 지표다.
- .95에서 Train은 높지만 강건 성능은 낮음: 높은 정렬 비율과 적은 conflicting 학습 예제가 색상 기반 해법을 유리하게 만들었을 수 있다. 데이터 비율과 shortcut 사용의 인과 경로를 분리한 추가 실험은 하지 않았다.
- 후기 반등/위치별 우열 교차: SGD minibatch와 BN state, validation 표본 구성, 색 대비/표현 난이도 등이 후보지만 하나를 확정할 증거는 없다. 단순 accuracy 변동을 Core 학습 퇴행의 직접 증거로 표현하지 않는다.

## 9. 시각화·누락 자료·후속 제안

같은 지표는 mode별 세 correlation을 같은 축에 표시했다. Test는 e50 점만, e0 Train은 NaN으로 유지했고 누락값을 보간하지 않았다. 모든 그림에 대응하는 PDF를 제공한다. 서로 다른 패널의 자동 축 범위에 주의한다.

1. [Epoch vs Train/Validation/Test Accuracy](figures/01_accuracy.png)
2. [Aligned/Conflicting/Worst-group Accuracy](figures/02_groups.png)
3. [Paired/All-color Flip Rate](figures/03_flip_rates.png)
4. [CF Accuracy/Drop/All-color Accuracy](figures/04_counterfactual.png)
5. [Correlation별 Shortcut 민감도](figures/05_correlation_dependence.png)
6. [Background vs Foreground](figures/06_background_vs_foreground.png)
7. [Online Train−Held-out 기술적 Gap](figures/07_descriptive_gap.png)

우선순위 제안(이번에는 실행하지 않음):

1. 정확한 C2I/U/W 분해가 필요하면 고정된 선택 Epoch의 checkpoint를 하나씩 읽어 동일 Validation 샘플의 원본/paired CF 예측을 별도 디렉터리에 저장한다. 두 개입×5000개로 checkpoint당 10000 image forwards다. 기존 평가에 9색까지 포함하면 55000개다. 정의와 분모를 사전 고정한다.
2. 기존 Feature NPZ로 CPU PCA/label·color probe를 설계할 수 있다. 같은 sample indices 및 고정 held-out partition, layer/정규화/solver를 먼저 정한다. 이번 작업에서는 probe 학습도 수행하지 않았다. 색상 decodability와 실제 분류기의 색상 사용은 다르다.
3. Probe/CAM이 필요하면 명시적 실행 범위를 정한 뒤 별도 작업으로 수행한다. 기존 evaluate.py/gradcam.py는 원본 run에 쓰는 기본값을 피하고 새 reports 하위 output을 명시해야 한다. 단일 sample CAM은 대표성이 부족하다.
4. Epoch별 Test는 누락이지만 Test를 반복해 선택에 쓰면 평가 누출이 발생한다. 필요성을 먼저 정하고 validation 중심의 가설 검증을 우선한다.
5. 여러 seed 및 필요시 상관/위치 factorial 반복으로 분산을 확인해야 한다. 새 학습은 이번 요청 범위 밖이며 직접 실행하지 않았다. 50 Epoch 이후의 강건성 유지는 이번 6조건 데이터만으로 판단하지 않는다.

원본 파일/이전 보고서 보존, 분석 코드 재사용, 추론 없는 재검증을 완료했다. 실행 로그와 artifact_audit/source_manifest 및 파생 CSV로 재현 가능하다. 주요 한계는 단일 seed, 집계만 저장된 예측, Test의 단일 시점, Core 직접 지표 부재다.

## 재현 및 산출물

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1  /data/dada0228/conda_envs/shortcut-xai/bin/python reports/cmnist_vgg50_analysis_review/analyze_metrics.py
/data/dada0228/conda_envs/shortcut-xai/bin/python reports/cmnist_vgg50_analysis_review/write_report.py
```

- summary_table.csv: 6조건 e1/e50, Test, 구간 평균, proxy, 기술적 gap, C2I 범위.
- epoch_metrics.csv: 원본 History와 mode/corr, Epoch별 gap·C2I 범위. count 열은 계산 정수, rate 열은 0–1, gap은 pp다.
- stage_summary.csv: 초기/중기/후기 평균·범위.
- proxy_thresholds.csv: 90/95/98% 공통 proxy 민감도.
- artifact_audit.json/source_manifest.json: 파일·설정·자료 검증 및 원본 hash.
- supplement_method.json: C2I 범위와 gap의 수학적 정의.
- validate_analysis.py/analysis_validation.json: 원본 hash, CSV/그림/링크 및 C2I 범위 공식을 검증. 3-class toy 예측 7380개를 전수 확인하여 집계별 가능한 C2I 최솟값/최댓값과 일치했다. 이 toy 검증은 새로운 연구 실험이 아니다.
- missing_artifacts.md: 추가 평가가 필요한 자료 목록.
