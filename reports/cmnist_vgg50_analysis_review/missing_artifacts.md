# 누락 및 추가 실행이 필요한 항목

| 항목 | 현재 상태 | 가능한 범위/후속 방법 |
|---|---|---|
| Epoch별 Test | e50만 존재 | 중간 Test 곡선 생성 불가. 필요성을 사전 결정 후 별도 재추론; 이번 미실행 |
| Train aligned/conflicting/WGA/CF | 저장 안 됨 | Train online accuracy/loss만 사용. 필요하면 종료 모델을 학습셋에 별도 평가 |
| Paired correct→incorrect | 직접 저장 안 됨 | 집계로 정확한 논리적 하한/상한 계산 가능. 점 추정/관측 곡선은 생성하지 않음 |
| Per-sample predictions/logits | 없음 | 정확한 전환표/paired bootstrap에는 별도 예측 저장 필요 |
| Probe / Grad-CAM | 6개 대상 run에 없음 | Feature NPZ는 전 Epoch 존재. 분석 설계 후 별도 CPU probe 또는 CAM 작업, 이번 미실행 |
| Core 학습 시작 시점 | 직접 관측 안 됨 | 강건 행동 proxy만 제공. 표현 정보 형성/분류기 실제 사용과 구별 |
| Seed 반복 | seed42만 | 통계적 일반화, 신뢰구간, seed 간 변동 평가 불가 |
| Epoch 0 Train | 무학습 baseline NaN | 손상 아님. 보간하지 않음 |
| Checkpoint tensor 전체 무결성 | ZIP/epoch metadata 검증 | 전체 tensor CRC/finite/재추론은 이번 범위 밖 |
| 소스/환경/CF chunk의 실행 당시 완전한 provenance | 부분 기록 | config/log/task 참조. 당시 source snapshot이나 chunk config가 없어 완전 소급 확정 불가 |

GPU/새 학습/SLURM 제출은 실행하지 않았다. 기존 결과와 보고서를 변경하지 않았다.
