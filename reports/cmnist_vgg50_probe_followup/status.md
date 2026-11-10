# 작업 재개 상태

## 중단 직전 확인된 내용

- 기존 review 보고서/CSV/누락 항목/그림 목록과 probe_dynamics.py, followup_common.py, evaluate.py, Feature 저장 코드를 검토했다.
- snapshot은 Epoch당 3200개, label×color 100그룹당 32개이며 6개 pooled 레이어다.
- 대표 파일에서 features.28과 avgpool 값이 동일함을 확인했다.
- 3200개 원본 index가 고유하고 현재 공식 MNIST 캐시 기준 snapshot 안의 grayscale pixel도 모두 고유함을 확인했다.
- 작은 CPU 계산에서 joint Ridge 1회가 약 .067초였다. 이는 실제 Probe 결과가 아닌 계산 비용 점검이다.

## 재개 시점의 실제 파일·프로세스

- 이 분석 디렉터리는 비어 있었다. 저장된 Probe 결과나 부분 결과는 없었다.
- 중단 전에 생성된 것은 빈 분석 디렉터리뿐이며 원본/기존 보고서는 변경하지 않았다.
- 확인한 실행 프로세스에 후속 Probe 작업은 없었다.
- 대화의 Conversation interrupted 외 계산 오류나 crash를 보여 주는 Probe 로그는 발견되지 않았다. 중단의 시스템 원인을 단정할 수 없다.

## 재개 정책

이전의 Epoch별 행동 분석은 재실행하지 않고 원본 History 및 기존 검증 자료를 재사용한다.
새 CPU Probe 분석만 실행하며, 프로토콜/분할을 결과 전에 고정한다. run×epoch별 결과를 원자적으로 저장하고 재실행 시 완료된 결과를 건너뛴다.
GPU, SLURM, 원본 모델 추가 학습/추론은 수행하지 않는다. 작은 CPU linear readout 학습만 이전 요청에 따라 분석으로 수행한다.

현재 단계별 상태는 progress.json과 execution.log에 기록한다. 최종 완료·미완료 사항은 analysis_report.md에 정리한다.

## 최종 상태

허용된 기존 자료 CPU 분석을 완료했다. 306개 snapshot/5412개 Probe 행, 표/그림7종, 그룹·checkpoint 진단 및 H1/H2 재평가가 저장됐다. validation_results.json의 보존·재개 검사도 통과했다. 추가 자료가 필요한 항목은 completion.json과 analysis_report.md에 남겼으며 GPU/SLURM/원본 모델 학습은 실행하지 않았다.
