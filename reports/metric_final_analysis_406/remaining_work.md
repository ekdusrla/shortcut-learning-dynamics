# 남은 평가와 후속 연구

이번 406개 Validation C2I 및 계획된96시점 Grad-CAM 평가는 완료됐다. 아래는 현재 완료 범위와 구분해야 할 항목이다. 이번에는 실행하지 않았다.

1. **다중 Seed 재현:** 현재는 Seed42 하나다. 일반화와 학습 Dynamics의 변동성을 검증하려면 별도 학습 승인과 GPU 자원 계획이 필요하다. 기존 반복 Epoch/Probe split은 대체물이 아니다.
2. **Probe51–100:** 저장 Feature를 같은 분할·표준화·Ridge 프로토콜로 분석하면 추가 encoder 추론은 피할 수 있다. 다만 새 readout 학습이므로 별도 승인 후 수행한다. CPU1, RAM4–8GiB, 한 snapshot씩 처리하는 방식을 권장한다.
3. **Epoch28 실제 실패 원본의 CAM:** 기존 전환 CSV에서20원본을 선정하면27/28/29 ×20 ×원본/CF=120 gradient-image 입력이다. GPU1/CPU4/RAM16GiB가 권장되며 새 모델 계산 승인이 필요하다. 현재 고정 digit5 사례는 실패 샘플이 아니다.
4. **Test C2I:** 이번406개 곡선은 Validation이다. 저장 Test Accuracy/Flip 끝점은 재사용했지만 Test 전체 C2I를 새로 평가한 것은 아니다. Validation과 구분해 계획해야 한다.
5. **인과적 Feature 사용:** 색상·형태 분리 개입, grayscale/shape ablation, palette 외 평가가 필요하다. 지원 분포 변화와 새 추론 비용을 고려해야 한다.

원본100 실험, 이번 .95 Resume, weak correlation50 Epoch는 서로 다른 범위다. 새로운 학습·추론·SLURM 제출은 각각 사용자 승인 및 직접 제출 절차를 따른다.
