# 회의용 요약 — Colored MNIST 후속406 Checkpoint

6개 Task 모두 COMPLETED/0:0이다. 406개 Checkpoint, Metric812행, CAM1,920개를 검증했고 저장 지표2,436개 비교가 모두 일치했다. 새 학습·추론·코드 수정은 하지 않았다.

## 발표할 핵심 결과

1. **H1은 단조 감소로 지지되지 않는다.** Epoch1→50에는 여섯 조건 모두 C2I가 줄지만, All9 C2I 증가는49전환 중18–23회 발생한다. 초기0→1에는 색상 민감도가 크게 증가한다.
2. **.95의100 Epoch 연장은 강건성을 더 개선하지 않았다.** All9 C/(9N)은 BG2.0467→2.6800%, FG2.3800→2.6556%다. 마지막5 Epoch 평균에서도 악화가 남는다.
3. **H2에서 관찰한 것은 강건 행동 도달 지연이다.** .65는Epoch2, .80은3, BG .95는46, FG .95는100까지 미충족이다. Core95% Probe 기준은 훨씬 빠르고 실제 학습 시작을 뜻하지 않는다.
4. **Epoch28은 국소적·일시적 실패다.** BG .95의 숫자5 conflicting Accuracy가98.98→77.41→97.97%로 변한다. 새 오류193개 중160개는 다음 Epoch에 회복했다. 원인은 미확정이다.
5. Probe는 정보의 해독 가능성, CAM은 공간 설명이다. 실제 Feature 사용의 인과적 증거로 합치지 않는다. Seed42 단일 실험과 Resume RNG 한계도 명시한다.

## 공통 Epoch50 결과(%)

| 조건 | Val Acc | Conflicting | WGA | Paired Flip | All9 Flip | Paired C/N | All9 C/(9N) | All9 C/(9A) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BG .95 | 96.40 | 96.00 | 93.09 | 3.96 | 4.82 | 0.4200 | 2.0467 | 2.1231 |
| BG .80 | 98.54 | 98.38 | 96.67 | 1.44 | 1.36 | 0.1600 | 0.5667 | 0.5751 |
| BG .65 | 98.88 | 98.75 | 97.11 | 1.04 | 0.87 | 0.1000 | 0.3533 | 0.3573 |
| FG .95 | 95.56 | 95.06 | 92.13 | 4.82 | 5.77 | 0.4400 | 2.3800 | 2.4906 |
| FG .80 | 98.34 | 98.15 | 96.77 | 1.68 | 1.63 | 0.1600 | 0.6311 | 0.6418 |
| FG .65 | 98.86 | 98.75 | 96.54 | 1.04 | 0.94 | 0.1200 | 0.4333 | 0.4383 |

## 회의에 사용할 그림

- [전체 RQ](figures/06_correlation_overview.png): Correlation별 Accuracy/Flip/All9 C2I
- [50 이후](figures/07_resume_late_dynamics.png): 추가 학습과 비단조 변동
- [Epoch28](figures/08_epoch28_anomaly.png): WGA·C2I·집중 그룹
- [Probe](figures/09_probe_decodability.png): 정보와 행동의 시점 차이
- [BG CAM](figures/11_cam_phases_bg.png), [FG CAM](figures/11_cam_phases_fg.png): 동일 원본 비교

권장 표현: “초기 획득 이후 Shortcut 민감도가 낮아지지만 단조적이지 않으며 강한 상관에서는 후기 재상승이 관찰됐다.”

피할 표현: “100 Epoch이면 Shortcut이 사라진다”, “Probe/CAM이 Core 학습 시작을 증명한다.”

[전체 보고서](analysis_report.md) · [검증 보고서](integrity_report.md) · [핵심 지표 CSV](summary_table.csv)
