# 406개 평가 완료·무결성 검증

**판정: 완료, 검증 통과.** 기존 결과의 CPU 산술과 표·그림 생성만 수행했다. 새 추론·학습·GPU 실행·SLURM 제출·취소 및 프로젝트 코드 수정은 없었다.

## SLURM과 완료 범위

| Task | 조건 | Epoch | 개수 | State / ExitCode | 시간 |
|---|---|---|---:|---|---|
| 0 | BG .95 | 0–100 | 101 | COMPLETED / 0:0 | 02:16:52 |
| 1 | BG .80 | 0–50 | 51 | COMPLETED / 0:0 | 01:12:05 |
| 2 | BG .65 | 0–50 | 51 | COMPLETED / 0:0 | 01:11:44 |
| 3 | FG .95 | 0–100 | 101 | COMPLETED / 0:0 | 02:17:12 |
| 4 | FG .80 | 0–50 | 51 | COMPLETED / 0:0 | 01:12:19 |
| 5 | FG .65 | 0–50 | 51 | COMPLETED / 0:0 | 01:12:18 |


모든 Task는2026-11-10 01:38:30 KST에 시작했다. 마지막 종료는03:55:42로 실제 wall span은2시간17분12초, Task elapsed 합은9시간22분30초다. 이는 앞서 제안한2GPU 시간 추정이 아닌 실제 accounting이다.

## 수행한 검사

- 예상 Epoch 목록, 실제 디렉터리, 조건별 완료 marker 일치: **406개, 누락·중복 없음**.
- Paired/All9 각406행, 총812행. 20개 digit×alignment 그룹×2개입×406=16,240그룹 행.
- 코드·manifest signature, GPU full-file Checkpoint SHA256 audit406개, 준비 manifest의 metadata/anchor SHA, 조건별 audit 해시가 각 Epoch marker와 일치.
- 현재 Checkpoint ZIP metadata406개도 대조. 마스터 대량 I/O를 피하기 위해 약28GiB의 tensor 파일을 이번에 전량 재해시하지는 않았음. GPU audit 기록과 현재 metadata 검사를 현재 전체 tensor 재해시와 구분함.
- 저장 Metric6개×406=**2,436개** 대조에서 최대 절대 차이**0**.
- 개별 예측의 고유 Validation 인덱스5,000개와 N×9 구조를 확인. 전체 및 그룹 C2I를 다시 계산하고 F=C2I+I2C+오답→다른 오답 항등식 확인.
- 현재 코드 signature로 신규 Pilot Gate6개 재검증. 기존 V1 결과20파일 및 당시 코드 snapshot 무결성도 유지.
- CAM96시점×20=**1,920개**. 224×224, 유한값·0–1, PNG 정상, 정답 Target/features.25/FP32, 고정 원본10개 확인. 모든 필수 산출물 SHA 일치.

## NaN과 오류의 구분

OOM, Traceback, 비정상 종료가 없다. 전체 Metric의 필수 수치는 유한값이다.

6개 zero-CAM은 FG 각 correlation의 Epoch0, 숫자2/index46044의 원본·CF다. 해당 mass와 cosine은 정의되지 않아 결측을 유지했다. 이를0% 중요도나 손상 파일로 취급하지 않았다. 유효 mass는1,914개이며 통계 CSV에 유효 n을 남겼다.

그룹 C/A는 A=0인208행에서 정의되지 않는다. Paired 행의 All9 전용 열 결측도 정상 스키마다. 전체406개 C2I 비율은 모두 정의된다. 무학습 Epoch0의 Train online Accuracy와 중간 Epoch Test는 미저장으로 구분했다.

## 자원

PyTorch peak allocated 최대1.845GiB, reserved 최대2.484GiB다. 부모+Worker RSS 관측 최대11.750GiB, SLURM MaxRSS 최대5.531GiB다. 측정 정의가 다르며 RSS 합에는 공유 페이지 중복과 짧은 peak 누락 가능성이 있다. GPU 값은 전체 장치 모니터링이 아닌 allocator 계측이다.

[자원 CSV](resource_usage.csv) · [SLURM CSV](slurm_jobs.csv) · [Checkpoint별 검사](checkpoint_validation.csv) · [검증 JSON](integrity_validation.json)

원본 평가 폴더의 상위 validation_results.json은 prepare 상태로 남아 있다. 기존 파일을 보존하기 위해 collector로 덮어쓰지 않았다. 이번 검증 JSON과 실제 조건별 완료 marker가 현재 완료 근거다. 코드 수정은 필요하지 않았다.
