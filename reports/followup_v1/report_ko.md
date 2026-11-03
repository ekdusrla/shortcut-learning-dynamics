# Shortcut Learning Dynamics 후속 분석

원래 모델은 재학습하지 않았다. 기존 학습 로그는 전체 epoch를 사용하며, probe와 Grad-CAM은 실제 체크포인트가 있는 epoch만 분석한다. 누락 epoch를 보간하거나 결과를 생성하지 않는다.

## 데이터 및 지표

- 학습 코드상 기존 feature snapshot은 validation에서 각 (숫자, 색상) 그룹당 최대 32개씩 고정 추출한 pooled feature다. next pairing의 20개 그룹 × 32 = 640개이며 독립적 숫자/색상 probe용 데이터가 아니다. 실제 원본 NPZ가 없는 경우 이 분할 판단은 코드에 근거한 것이며 audit_followup.py --verify-datasets로 서버 파일을 대조할 수 있다.
- 기존 코드의 next/random은 숫자마다 두 색상만 허용한다. correlation=0.1로도 독립적이지 않다. all pairing의 0.1은 생성 확률상 독립적이지만 표본상 완전 균형은 보장하지 않는다.
- 새 protocol은 validation 원본으로 probe를 학습하고 공식 test 원본으로 평가한다. 각 숫자의 같은 수 원본을 10색 모두로 칠해 모든 숫자×색상 셀의 수를 정확히 맞춘다. 재색칠된 같은 원본이 probe 학습/평가 양쪽으로 나뉘지 않는다.
- chance-adjusted accuracy = (accuracy − 0.1) / 0.9. 기존 호환 컬럼 legacy_decoded_correlation = 2 × accuracy − 1을 함께 보존한다. 둘 다 Pearson correlation이 아니다.
- probe는 원래 feature scale을 그대로 사용한 AdamW linear classifier다. 해독 성능은 representation과 probe 최적화 조건에 관한 결과이며 인과적 의존도는 아니다.

## 학습 로그와 H1

- resnet18_corr095_seed42_e100: epoch 1 flip rate 0.0464 → epoch 100 0.0236. 학습된 epoch 사이 증가 구간은 48개다. 초기→최종 감소와 단조 감소를 구분해야 하며, epoch 0의 낮은 flip rate는 무작위/상수 예측 때문에 생길 수 있다.
- vgg11_corr095_seed42_e100: epoch 1 flip rate 0.0722 → epoch 100 0.0142. 학습된 epoch 사이 증가 구간은 43개다. 초기→최종 감소와 단조 감소를 구분해야 하며, epoch 0의 낮은 flip rate는 무작위/상수 예측 때문에 생길 수 있다.

## Layer별 probe 결과

- resnet18_corr095_seed42_e100 / avgpool: 관측 epoch [100]; 최종 관측 숫자 accuracy 0.8103, 색상 accuracy 0.9994.
- resnet18_corr095_seed42_e100 / layer1: 관측 epoch [100]; 최종 관측 숫자 accuracy 0.3006, 색상 accuracy 0.9862.
- resnet18_corr095_seed42_e100 / layer2: 관측 epoch [100]; 최종 관측 숫자 accuracy 0.4928, 색상 accuracy 0.9972.
- resnet18_corr095_seed42_e100 / layer3: 관측 epoch [100]; 최종 관측 숫자 accuracy 0.7169, 색상 accuracy 1.0000.
- resnet18_corr095_seed42_e100 / layer4: 관측 epoch [100]; 최종 관측 숫자 accuracy 0.8103, 색상 accuracy 0.9994.
- vgg11_corr095_seed42_e100 / avgpool: 관측 epoch [100]; 최종 관측 숫자 accuracy 0.8712, 색상 accuracy 1.0000.
- vgg11_corr095_seed42_e100 / features.14: 관측 epoch [100]; 최종 관측 숫자 accuracy 0.5650, 색상 accuracy 0.9978.
- vgg11_corr095_seed42_e100 / features.21: 관측 epoch [100]; 최종 관측 숫자 accuracy 0.8284, 색상 accuracy 0.9997.
- vgg11_corr095_seed42_e100 / features.28: 관측 epoch [100]; 최종 관측 숫자 accuracy 0.8712, 색상 accuracy 1.0000.
- vgg11_corr095_seed42_e100 / features.3: 관측 epoch [100]; 최종 관측 숫자 accuracy 0.2028, 색상 accuracy 0.9287.
- vgg11_corr095_seed42_e100 / features.7: 관측 epoch [100]; 최종 관측 숫자 accuracy 0.2656, 색상 accuracy 0.9712.

관측된 checkpoint에서 threshold에 처음 도달한 epoch는 실제 학습 시작 시점 추정치가 아니다. epoch 100만 있는 경우에는 언제 해독 성능이 증가했는지 판단할 수 없다. 단일 seed의 작은 원본 표본을 여러 색상으로 반복했으므로 각 재색칠 입력을 독립적인 원본 관측치로 취급하지 않는다.

## Grad-CAM의 정량화와 해석

- 두 모델/모든 epoch에서 같은 공식 test 원본에 정답 숫자를 고정 target으로 사용한다. aligned 색상 y와 conflicting 색상 (y+1) mod 10을 비교하며 숫자 모양은 동일하다.
- cam_mean_absolute_difference = 평균 |A−B|. A와 B는 입력 해상도로 보간 후 각자 0–1 정규화한 heatmap이다.
- cam_cosine_similarity = sum(A×B)/(||A||₂||B||₂). 0 heatmap이면 정의 불가로 빈 값 처리한다.
- foreground_mass = sum(CAM × mask)/sum(CAM). mask는 원본 회색조를 입력 크기로 bilinear resize한 뒤 밝기 > 0.1인 영역이다. 비교 기준 면적 foreground_area_fraction도 저장한다. 0 heatmap이면 빈 값 처리한다.
- paired prediction flip과 정답 확률을 함께 기록한다. 이 소수 고정 표본의 flip은 전체 test flip rate와 다르다.
- foreground 색상과 숫자 모양이 같은 픽셀에 있으므로 foreground mass로 색상/모양 의존도를 분리할 수 없다. 최종층의 낮은 공간 해상도와 표본별 정규화도 고려해야 한다. Grad-CAM 수치 및 그림은 직접적인 인과적 증거가 아니다.

## 통합 질문에 대한 현재 판단

- H1: 위 학습 로그에서 초기 학습 epoch 대비 최종 paired flip rate와 중간 변동을 확인할 수 있다. probe의 정보 보존과 예측 민감도 감소는 별개다.
- Core/shortcut decodability가 언제 변했는지는 여러 epoch의 checkpoint probe가 필요하다. 색상 정보가 최종 feature에서 해독된다는 사실을 shortcut 의존성 증가로 해석하지 않는다.
- 두 모델 차이는 같은 protocol/epoch/layer와 학습 조건에서 비교해야 한다. 서로 다른 아키텍처의 중간 stage는 구조적으로 완전히 대응하지 않는다.
- H2: correlation 0.80/0.65 결과가 확보된 뒤 같은 protocol로 검증한다. 현재 0.95 결과만으로 강한 correlation의 학습 지연을 결론 내리지 않는다.

## 실행 범위

```json
[
  {
    "run_id": "resnet18_corr095_seed42_e100",
    "probe_epochs": [
      100
    ],
    "gradcam_epochs": [
      100
    ],
    "missing_probe_epochs": [
      0,
      1,
      3,
      5,
      10,
      20,
      30,
      50,
      70
    ],
    "missing_gradcam_epochs": [
      0,
      1,
      3,
      5,
      10,
      20,
      30,
      50,
      70
    ]
  },
  {
    "run_id": "vgg11_corr095_seed42_e100",
    "probe_epochs": [
      100
    ],
    "gradcam_epochs": [
      100
    ],
    "missing_probe_epochs": [
      0,
      1,
      3,
      5,
      10,
      20,
      30,
      50,
      70
    ],
    "missing_gradcam_epochs": [
      0,
      1,
      3,
      5,
      10,
      20,
      30,
      50,
      70
    ]
  }
]
```

기존 epoch100 분석은 상관된 val/test 원본 분포에서 수행한 별도 protocol이므로 이번 독립 probe 수치와 같은 곡선에 직접 연결하지 않는다. 새 probe의 10색 중 8색은 next pairing 학습에서 해당 숫자에 관측되지 않았던 조합이다. 원래 paired flip rate와 새 probe의 평가 분포는 다르므로 두 지표의 직접적인 수치 대응을 가정하지 않는다. 레이어 차원·feature scale·probe 수렴 정도도 해독 성능 차이에 영향을 줄 수 있다.
