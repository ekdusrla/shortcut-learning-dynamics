# Shortcut Learning Dynamics 핵심 결과

[전체 보고서](REPORT.ko.md) · [검증 기록](VALIDATION.ko.md) · [실험 목록](EXPERIMENT_INVENTORY.csv)

두 데이터셋과 두 백본에서 Fixed 학습 18개를 E0–E50로 비교했다. 완결된 로컬 기록에는 추가로 VGG 연장 2개와 Dynamic 2개가 있고, 별도의 초기 64px E100 2개는 기존 자료로 검토했다.

| 연구 가설 | 결론 | 주요 근거 |
| --- | --- | --- |
| H1 · 후기 shortcut 의존성 감소 | 부분 지지 | Colored MNIST의 12개 조건은 초기 대비 후기 All9 Flip·C2I가 감소하고 WGA가 상승했다. Waterbirds ResNet18에 Flip 반등 또는 WGA 하락이 있었다. 두 VGG 0.95 연장에서도 E50→E100 WGA가 낮아졌다. |
| H2-a · 해독 가능성 출현 지연 | 모델·기준 의존 | Colored MNIST 숫자 probe ≥90%의 3-epoch 시작은 모두 E1이다. ≥95%에서 일부 지연이 있다. Waterbirds VGG의 core probe ≥60%는 0.65/0.80/0.95에서 E7/E20/E46이지만 ResNet18은 E50까지 세 조건 모두 미도달이다. |
| H2-b · 강건 행동 및 실제 활용 출현 지연 | 일부 조건 지지 | BG VGG의 90% Target·10% Flip 공동 유지 시작은 E2/E2/E11이다. 같은 세 run의 paired 강건 행동 유지 시작은 E2/E3/E20이다. 같은 데이터셋의 역사적 All9 행동 기준과 도달값은 다르다. |

### 최종 E50 · 공통 FP32 Validation

| 조건 | Accuracy | WGA | Spurious Flip | Core Target | C2I 전체 쌍 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 0.65 | 98.88% | 97.11% | 1.04% | 98.24% | 0.10% |
| Fixed 0.80 | 98.54% | 96.67% | 1.44% | 97.54% | 0.16% |
| Fixed 0.95 | 96.40% | 93.09% | 3.96% | 94.32% | 0.42% |
| Stepwise | 99.02% | 97.53% | 0.90% | 98.16% | 0.14% |
| Gradual | 98.84% | 97.33% | 0.98% | 97.78% | 0.10% |

Dynamic은 Fixed 0.95보다 최종 WGA·Core Target이 높고 Flip이 낮았다. Fixed 0.65보다 WGA는 +0.42/+0.22%p, Core Target은 −0.08/−0.46%p였다. Stepwise E34에서는 WGA가 95.39→79.11%로 떨어졌고, Core Target·Flip·C2I와 함께 E36에 직전 수준으로 회복했다. Core Target 상승과 Flip 감소가 동시에 나타난 전환은 8개 중 3개다. Gradual과 Stepwise의 Core 기준 최초 충족(E7/E11)은 모두 각자의 첫 전환 전이었다.

후속 검증의 우선순위는 동일 하드웨어에서 여러 seed 반복, 평균 상관을 맞춘 비교, 다른 백본·색상과 Waterbirds의 품질 검증된 Core 교체 순이다. 현재 데이터로 schedule 자체의 인과 효과나 통계적 유의성을 주장하지 않는다.

[Epoch별 그래프](figures/epoch_curves.png) · [전환별 표](tables/paired_transitions.csv) · [두 C2I 분모와 Core Flip을 포함한 E50 표](tables/paired_e50.csv)
