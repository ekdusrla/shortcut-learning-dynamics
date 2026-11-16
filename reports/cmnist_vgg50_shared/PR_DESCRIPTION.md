## 변경 내용

Colored MNIST × VGG11_BN의 Epoch 50 실험 6개(background/foreground × train corr .95/.80/.65)를 팀원이 확인하고 재현할 수 있도록 실행 코드와 집계 결과를 공유합니다.

- 실제 config 6개와 원본 History/Test Metrics의 byte-identical 복사본 18개, SHA-256 기록
- 성능·Counterfactual·Generalization Gap·C2I 범위 분석, PNG/PDF 그래프
- 저장 Feature의 CPU Linear Probe 결과와 Epoch 28/49 그룹별 이상 현상 분석
- 공식 MNIST revision 고정·오프라인 캐시 검증 및 isolated Resume 도구
- `EXPERIMENTS.ko.md`, 의존성, CSV-only 그림 재생성 경로

`origin/dev`에서 생성한 `feat/cmnist-vgg11-results` 브랜치이며 PR 대상은 `dev`입니다. dev에는 초기 이진 CMNIST 구현이 남아 있어, 실제 실험에서 사용한 10-class 데이터 생성·평가 및 관련 source를 기존 main 작업본에서 함께 포함했습니다. 특히 데이터/평가 공통 인터페이스 변경을 검토해 주세요.

## 결과 요약

- H1: 장기적인 행동적 색상 민감도 감소는 관찰되지만 단조 감소는 아닙니다.
- H2: 높은 상관에서 강건 행동이 늦게 나타나지만, Core 학습의 실제 시작 시점으로 확정할 수 없습니다. .95의 Core95% 선형 해독성은 BG e2/FG e3, 행동95% 기준은 BG e46/FG 미도달입니다.
- BG .95 e28의 전체 정답 순감소 130개 중 digit5 conflicting 그룹이 85개(65.4%)를 차지합니다. Probe 정보 붕괴나 BN/head의 단일 원인으로 단정하지 않았습니다.
- 모델 seed는 42 한 개이며, Probe의 반복 분할을 독립 모델 반복으로 취급하지 않습니다.

## 검증

- [x] Counterfactual CPU 테스트 2개 통과
- [x] DataLoader CPU 테스트 2개 통과
- [x] checkpoint/cache/per-Epoch prediction 없이 집계 CSV만 둔 임시 clone에서 표·그림·보고서 재생성 통과
- [x] 18개 metric 복사본이 원본과 동일함을 확인
- [x] 게시 파일 크기·SHA 및 토큰 패턴 검사 통과
- [x] 원본 체크포인트·Feature·실험 결과 보존
- [x] dataset/checkpoint/cache/대량 중간 예측·로그 제외
- [x] GPU·새 모델 학습·SLURM 제출 미실행

## 확인 방법

```bash
python -m pip install -r requirements-analysis.txt
python reports/cmnist_vgg50_probe_followup/summarize.py --figures-only
python reports/cmnist_vgg50_probe_followup/write_report.py
```

전체 Probe 재적합과 원본 hash 재검증은 별도 보관한 Feature/Checkpoint가 필요합니다. GitHub에는 집계 지표와 대표 그래프만 포함했습니다. Epoch 100 Resume 준비 자료와 Epoch 50 완료 결과를 구분했습니다.

이 PR은 검토용이며 merge하지 않습니다.
