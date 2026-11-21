# Shortcut Learning Dynamics · 최종 통합 비교

Colored MNIST와 Waterbirds에서 VGG11_BN과 ResNet18의 Shortcut Learning Dynamics를 비교한 결과다. 두 백본의 직접 비교는 E0–E50을 사용한다. Waterbirds ResNet18의 E51–E100은 별도 분석이다.

- [한국어 통합 보고서](analysis_report.md)
- [H1 조건별 분석](h1_conditions.csv) · [H2 전환 시점](h2_timing.csv) · [검증 기록](validation.json)
- [Accuracy/WGA](accuracy_matched.csv) · [Flip/C2I/I2C](counterfactual_matched.csv) · [Probe](probes_matched.csv)
- [Colored MNIST 그래프](colored_mnist_architectures.png) · [Waterbirds 그래프](waterbirds_architectures.png) · [Waterbirds ResNet18 후기 분석](waterbirds_resnet_late.csv)

## 분석 코드

[통합 스크립트](../../scripts/resnet18/final_comparison.py)는 기존 실험 산출물의 CSV·JSON과 run metadata를 읽어 표와 그래프를 생성한다. 위 결과 파일은 GitHub에서 바로 확인할 수 있다. 스크립트를 실행하려면 원본 실험 디렉터리와 NumPy, pandas, Matplotlib가 필요하다. 실험 메타데이터에 기록된 로컬 경로가 유효해야 하며, 스크립트는 출력 디렉터리가 이미 존재하면 덮어쓰기 전에 중단한다.

## PR Description

### 변경 내용

- Colored MNIST와 Waterbirds의 VGG11_BN / ResNet18 결과를 공통 E0–E50에서 통합했다.
- Waterbirds ResNet18 E51–E100은 별도 CSV와 그래프로 정리했다.
- 설정, Accuracy/WGA, Flip·C2I·I2C, Probe, CAM과 H1/H2 결과 및 검증 기록을 추가했다.

### 주요 결과

- H1은 부분 지지다. Colored MNIST에서는 장기적인 행동 강건성 개선이 관찰됐다. Waterbirds에서는 Flip 반등과 WGA 악화가 관찰됐다.
- H2의 실제 Core 학습 시점은 판단을 보류했다. 높은 상관에서 강건 행동 기준 도달이 늦어지는 경향은 일부 조건에서 관찰됐다.
- 단일 seed의 기술적 비교다. Probe와 Grad-CAM은 각각 해독 가능성과 공간 분포를 보여준다.

### 검증

- 공통 Epoch 범위, 비교 쌍의 분모, 반사실 전이 산식 및 Probe 변환식을 검증했다.
- 기존 체크포인트와 원본 평가 자료를 읽기 전용으로 사용했다. 통합은 CPU에서 수행했다.
