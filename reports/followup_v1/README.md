# 후속 분석 — 로컬 실행 결과

기존 모델은 재학습하지 않았다. 로컬의 두 epoch100 checkpoint를 사용했으며 다른 요청 epoch의 결과는 만들지 않았다. 전체 서버 실행 명령과 파일 설명은 [실행 가이드](../../FOLLOWUP_KO.md)에 있다.

## 독립 probe의 최종 feature 결과

숫자별 32개 원본 × 10가지 색상으로 학습/평가 각각 3,200개 입력을 구성했다. 학습 원본은 모델 validation에서, 평가 원본은 공식 test에서 선택했다. 같은 원본의 재색칠 입력은 한쪽 split에만 포함했다. 원래 2색 pairing 평가와 다른 조건이므로 앞선 epoch100 결과와 수치를 직접 연결하지 않는다.

| 모델 | 숫자 accuracy | 숫자 chance-adjusted | 색상 accuracy | 색상 chance-adjusted |
|---|---:|---:|---:|---:|
| ResNet18 | 81.03% | 0.7892 | 99.94% | 0.9993 |
| VGG11-BN | 87.12% | 0.8569 | 100.00% | 1.0000 |

Chance-adjusted accuracy = (accuracy − 0.1)/0.9이며 Pearson correlation이 아니다. 기존 2×accuracy−1은 별도 CSV 컬럼에 보존했다. 색상 정보의 높은 해독 가능성이 예측의 색상 의존도를 직접 뜻하지는 않는다.

## 결과 보기

- [한국어 통합 분석 보고서](report_ko.md)
- [전체 stage probe 결과](probe_combined.csv) · [기존 history와 probe 연결](probe_history_joined.csv)
- [Grad-CAM 지표](gradcam_combined.csv) · [기존 history와 Grad-CAM 연결](gradcam_history_joined.csv)
- [ResNet18 layer heatmap](resnet18_corr095_seed42_e100_layer_heatmaps.png) · [VGG11 layer heatmap](vgg11_corr095_seed42_e100_layer_heatmaps.png)
- [숫자 probe 그래프](core_accuracy_epochs.png) · [색상 probe 그래프](shortcut_accuracy_epochs.png)
- [숫자 최종 feature 모델 비교](core_model_comparison.png) · [색상 최종 feature 모델 비교](shortcut_model_comparison.png)
- [Validation flip rate 전체 추이](val_cf_flip_rate.png) · [해독 성능과 flip rate](decodability_vs_flip_rate.png)
- [ResNet18 동일 원본 재색칠 Grad-CAM](cam_grids/resnet18_corr095_seed42_e100_epoch100.png)
- [VGG11 동일 원본 재색칠 Grad-CAM](cam_grids/vgg11_corr095_seed42_e100_epoch100.png)
- [고정 표본 protocol](protocol.json) · [가용/누락 epoch](coverage.json) · [실제 파일 조사](audit.json) · [저장 결과 검증 PASS](validation.json)

CAM은 정답 숫자를 고정 target으로 사용했다. aligned 색상 y와 conflicting 색상 (y+1)%10에서 숫자 픽셀은 동일하다. 정량 지표의 계산식과 한계는 통합 보고서에 정리했다.

기존 validation paired flip rate는 ResNet18 epoch1 0.0464→epoch100 0.0236, VGG11 0.0722→0.0142로 감소했지만 단조 감소는 아니다. 새 probe는 epoch100만 있어 해독 성능의 시간적 증가/감소 시점은 아직 알 수 없다. Correlation 0.80/0.65는 설정만 준비했으며 H2 검증은 추가 학습 결과가 필요하다.

검증: 오프라인 테스트 7개 통과, 저장된 probe 22개의 예측을 NumPy로 독립 대조, CAM 20쌍의 범위·파일·동일 원본/target 확인, 숫자×색상 조합 균형 및 원본 분리 확인. 대용량 feature/probe 가중치는 저장소 상위 `shortcut_results/followup_v1/`에 보존했다.
