# Corr=.95 Epoch50→100 Resume 확장: Noise vs Real Signal 검증

`cmnist_vgg100_resume`(다연님 쪽)에서 corr=.95를 epoch 50→100으로 resume 연장했을 때
숏컷(색상) 의존도가 다시 증가한 것으로 보였던 현상에 대해, 같은 체크포인트에서
독립적으로 2회 재현(resume_repeat)하고 resume 없이 처음부터 100epoch 돌린 대조군
(continuous)을 추가로 돌려서 이게 재현 가능한 진짜 신호인지, resume 특유의 계산
흔들림(noise)인지 검증했다.

## 방법

- 원본: `vgg11_{bg,fg}_corr095_seed42_e100_from050_img224_all_retry191154` (다연님 계정, 기존 결과)
- resume_repeat: 동일 epoch50 체크포인트에서 독립 2회 재현 (hd0070sn 계정, 새로 실행)
- continuous: resume 없이 seed42로 epoch 0→100 처음부터 학습 (hd0070sn 계정, 새로 실행)
- 46-50epoch(early) 대 96-100epoch(late) 구간 평균의 차이(delta)를 비교
- 판정 기준: |원본 delta| / 독립재현 표준편차 ≥ 3배면 real signal, ≤1.5배면 noise

## 결과

| 조건 | 지표 | 원본 delta | 재현 표준편차 | 비율 | 판정 |
|---|---|---:|---:|---:|---|
| BG | All9 Flip | +0.0076 | 0.0112 | 0.68x | Noise |
| BG | WGA | -0.0012 | 0.0530 | 0.02x | Noise |
| FG | All9 Flip | +0.0051 | 0.0004 | 12.95x | **Real signal** |
| FG | WGA | -0.0086 | (n=2, 거의 동일) | 원본이 재현값 밖 | 원본이 재현보다 큰 하락 |

continuous(resume 없는 대조군)도 BG/FG 모두 같은 방향으로 움직여, 이 변화가
**resume 메커니즘 자체의 결함이 아님**을 확인했다.

## 결론

corr=.95의 epoch50→100 "숏컷 의존도 재증가"는 색상 위치에 따라 다르게 해석해야 한다.
BG는 통계적으로 noise와 구분되지 않지만, FG는 독립 재현 2회가 거의 동일한 크기로
재현되어 재현 가능한 실제 학습 동역학으로 판단한다. 두 조건 모두 resume 여부와
무관하게 같은 방향으로 움직이므로, 이는 resume의 RNG 복원 문제가 아니라 학습 자체의
특성이다.

## 한계

- 독립 재현은 조건당 2회로 표본이 작다(특히 FG WGA는 표준편차 추정이 불안정함).
- seed는 42 하나만 사용했다(다른 seed에서 재현되는지는 추가 검증 필요).
- 재현/대조군 실행은 hd0070sn 계정에서 별도로 수행했으며, 원본과 동일한 conda
  환경(다연님의 `shortcut-xai`)과 MNIST 캐시를 읽기 전용으로 공유했다.

## 재현 방법

```bash
python analyze_resume_noise_followup.py \
  --csv <원본 history.csv>:original \
  --csv <재현1 history.csv>:resume_repeat \
  --csv <재현2 history.csv>:resume_repeat \
  --csv <continuous history.csv>:continuous \
  --metric val_cf_all_flip_rate --metric val_wga \
  --early-start 46 --early-end 50 --late-start 96 --late-end 100 \
  --out-dir <출력폴더>
```

실행 스크립트: `scripts/cmnist_vgg100_resume_followup/task_rep.sh`,
`scripts/cmnist_vgg100_resume_followup/task_continuous.sh`
