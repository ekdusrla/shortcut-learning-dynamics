# Colored MNIST × VGG11_BN: 팀원용 실험 및 결과 안내

이 저장소에는 **Epoch 50 완료 실험 6개**의 실행·분석 코드, 작은 원본 Metrics 복사본, 보고서·표·그림을 공유합니다. GPU 실행 없이 결과를 읽고 표/그림을 재생성할 수 있습니다. Checkpoint와 원본 Feature는 별도 대용량 산출물이며 GitHub에 포함하지 않습니다.

## 먼저 볼 자료

| 목적 | 자료 |
|---|---|
| 최신 결론: 선형 정보와 실제 Shortcut 행동의 구분 | [후속 Probe 보고서](reports/cmnist_vgg50_probe_followup/analysis_report.md) |
| 6조건의 성능·CF·비단조적 학습 Dynamics | [Epoch 50 분석 보고서](reports/cmnist_vgg50_analysis_review/analysis_report.md) |
| 빠른 숫자 비교 | [성능 요약 CSV](reports/cmnist_vgg50_analysis_review/summary_table.csv), [Probe 요약 CSV](reports/cmnist_vgg50_probe_followup/probe_summary.csv) |
| Epoch별 지표 | [학습/CF CSV](reports/cmnist_vgg50_analysis_review/epoch_metrics.csv), [Probe CSV](reports/cmnist_vgg50_probe_followup/probe_results.csv) |
| 이상 Epoch 분석 | [행동·Probe 결합표](reports/cmnist_vgg50_probe_followup/anomaly_combined.csv), [그룹 순변화](reports/cmnist_vgg50_probe_followup/group_event_deltas.csv) |
| 그림 | [학습 Dynamics](reports/cmnist_vgg50_analysis_review/figures/), [Probe/그룹 변화](reports/cmnist_vgg50_probe_followup/figures/) — PNG/PDF |
| 실제 실험 설정·원본 지표 | [6개 run의 config/history/test_metrics 복사본](reports/cmnist_vgg50_shared/source_metrics/), [원본 SHA-256](reports/cmnist_vgg50_shared/metric_sources.json) |
| 누락 및 남은 작업 | [완료/미완료 목록](reports/cmnist_vgg50_probe_followup/completion.json), [누락 지표](reports/cmnist_vgg50_analysis_review/missing_artifacts.md) |
| 공유 파일 목록·크기·해시 | [공유 manifest](reports/cmnist_vgg50_shared/publish_manifest.json) |

## 실험 조건

공통: VGG11_BN(compact 512→10 head), seed42, 50 Epoch, image224, batch128, SGD lr=.01/momentum=.9/weight_decay=.0001, scheduler none, AMP, workers4, color_pairing=all, val/test corr=.10, Validation5000. ImageNet pretrained/augmentation은 사용하지 않았습니다.

| Mode | Train corr | 최종 Test acc | 최종 Val All-color flip | 95% 강건 행동 기준 최초 Epoch |
|---|---:|---:|---:|---:|
| Background | .95 | 96.71% | 4.82% | 46 |
| Background | .80 | 98.66% | 1.36% | 3 |
| Background | .65 | 99.08% | 0.87% | 2 |
| Foreground | .95 | 95.93% | 5.77% | 50 Epoch 내 미도달 |
| Foreground | .80 | 98.55% | 1.63% | 3 |
| Foreground | .65 | 99.01% | 0.94% | 2 |

강건 행동 기준은 conflicting accuracy≥95%, All-color conflicting CF accuracy≥95%, All-color flip≤5%를 3 Epoch 연속 만족하는 구간 시작입니다. Core 학습 시작의 직접 관측값이 아닙니다.

## 현재 결론과 해석 제한

- **H1:** 장기적으로 색상 개입에 대한 예측 민감도는 감소하지만 단조 감소는 아닙니다. Background .95는 e28 WGA77.41%로 일시 악화했고, e46–48 기준 충족 뒤 e49에도 재이탈했습니다.
- **H2:** 높은 correlation의 강건 행동 지연은 관찰됩니다. 그러나 Core95% 선형 해독성은 .95에서도 BG e2/FG e3에 도달했고, 90% 기준은 모든 조건이 e1입니다. 실제 Core 학습 시작 Epoch는 식별할 수 없습니다.
- 무학습 e0에도 digit은 약 80–86% 선형 해독 가능하며, 색상은 후기에도 거의 100% 읽힙니다. Probe 정보 존재와 원래 classifier의 정보 사용은 다릅니다.
- 원본 모델 seed는 하나입니다. Probe 3개 partition이나 Epoch를 독립 모델 반복으로 간주하지 않습니다.
- Test는 e50 한 번만 저장했습니다. 정확한 원본 C2I는 저장되지 않았으며 논리적 하한/상한만 계산했습니다.
- pairing all의 corr=.10은 10-class 독립 색상의 aligned chance와 같습니다. 이진 결정론적 역상관으로 해석하지 않습니다.
- 원본 raw revision/image hashes가 기록되지 않아 새 공식 MNIST 캐시와 과거 전체 pixel 동일성은 확정하지 못합니다.

## 저장 구조 및 범위

```text
scripts/cmnist_vgg50/            실제 Epoch 50 array 실행·메모리 검증·환경 기록
scripts/cmnist_vgg100_resume/    Epoch 100 Resume/캐시 준비 도구와 CPU 검증
reports/cmnist_vgg50_analysis/   최초 분석의 재현 코드/원본 hash만 공유
reports/cmnist_vgg50_analysis_review/  성능·CF·C2I 범위·gap 재검증
reports/cmnist_vgg50_probe_followup/   저장 Feature CPU Probe·이상 Epoch·가설 재검증
reports/cmnist_mnist_cache/      공식 revision 캐시 준비·오프라인 검증 provenance
reports/cmnist_vgg50_shared/source_metrics/<run>/
    config.json / history.csv / test_metrics.json
```

`source_metrics`는 각 원본 파일과 byte-identical한 **복사본**이며 원본 run에 쓰지 않습니다. 원본은 `runs/vgg11_{bg,fg}_corr{095,080,065}_seed42_e50_img224_all`에 있습니다. 빈 TRAIN_SUCCESS나 checkpoint/Feature를 복사본에 만들어 완료를 가장하지 않습니다.

**대량의 중간 결과 `batches/`는 GitHub에서 제외했습니다.** 집계 CSV와 이미 계산한 사건 비교·그래프만 공유합니다. 작은 파일이어도 수백 개의 per-Epoch 예측 파일과 실행 로그는 별도 스토리지에 보관합니다. 일부 provenance는 원래 클러스터의 절대 경로/mtime/hash를 보존합니다. 팀원 PC에서는 그 경로가 존재한다고 가정하지 마세요.

**Epoch 100은 별도 범위:** Resume 구현/실행 준비 자료는 포함하지만, 본 결과 표는 Epoch 50의 6조건만 다룹니다. Job191154 실패 기록이나 준비 상태를 Epoch100 완료 결과로 혼합하지 않았습니다.

## GPU 없이 결과 확인·재생성

Python3.11 환경에서:

```bash
python -m pip install -r requirements-analysis.txt
python reports/cmnist_vgg50_probe_followup/summarize.py --figures-only
python reports/cmnist_vgg50_probe_followup/write_report.py
```

이 두 스크립트는 공유된 집계 CSV로 표·그림·보고서를 재생성합니다. per-sample bootstrap 재계산은 별도 보관한 예측 파일이 필요합니다. 원본 checkpoint/Feature와 GPU는 필요하지 않습니다. 산출물은 해당 reports 디렉터리에 다시 저장되므로 별도 clone에서 실행하세요. 공개된 기존 결과는 바로 읽을 수 있으며 계산을 다시 할 필요는 없습니다.

`requirements-analysis.txt`는 관측된 분석 환경입니다. 실제 Feature Probe 재적합·학습 코드에는 `requirements.txt`의 torch/torchvision도 필요합니다. GPU CUDA 조합은 [당시 환경 기록](scripts/cmnist_vgg50/environment.ko.md)과 [설치 목록](scripts/cmnist_vgg50/installed-packages.txt)을 참고하세요. 당시 검토 문서의 미검증/후속 완료 사항은 최신 보고서와 함께 읽어야 합니다.

### 원본 Feature/Checkpoint가 있어야 가능한 작업

- `run_probes.py`: 원본 snapshot에서 Probe를 다시 적합. 기존 `batches`가 있으면 완료분을 재사용합니다.
- `checkpoint_drift.py`: 원본 checkpoint head/BN state 읽기.
- `validate_results.py` 또는 이전 `analyze_metrics.py`: 원본 Feature/checkpoint/로그까지 hash/구조 재검증.
- `probe_dynamics.py`: 새 crossed-color feature 인코딩이 포함되므로 CPU 결과 열람용 명령이 아닙니다.

이 작업들은 GitHub만 clone했다고 원본 대용량 자료까지 생기는 것이 아닙니다. 팀 공유 스토리지에서 별도로 받은 뒤 기존 manifest의 hash로 확인하세요. 대용량 파일의 업로드 위치/접근권한은 별도 관리합니다.

## 실제 실행 및 MNIST 캐시

실행 스크립트는 원래 클러스터 경로·Python·노드·account/QOS가 명시된 기록입니다. 새로운 환경에서 그대로 제출하지 마세요. 범용 CLI 예시는 root README에 있고, 6개 실제 config 및 task.sh가 이번 조건의 기준입니다. GPU당 CPU4/RAM16GiB의 기존 요청을 참고하되 현재 정책은 제출 직전 확인합니다.

공식 MNIST 캐시는 GitHub에서 제외합니다. 새 공유 경로에 준비할 경우 `mnist_cache.py prepare`가 고정 revision을 다운로드·검증하며 기존 cache는 덮어쓰지 않습니다. 사용한 revision은 `77f3279092a1c1579b2250db8eafed0ad422088c`입니다. 상세는 [캐시 준비 보고서](reports/cmnist_mnist_cache/dataset_preparation_report.ko.md)를 참고하세요.

이 공유 정리에서는 새로운 모델 학습·GPU·SLURM 작업을 실행하지 않습니다.

## 공유 범위와 PR

CONTRIBUTING.md에 맞춰 `origin/dev`에서 `feat/cmnist-vgg11-results`를 생성했고 PR 대상은 `dev`입니다. `main`이나 `dev`에 직접 push/merge하지 않습니다. `Add:`, `Docs:` 형식으로 실행 코드와 결과 문서를 구분합니다.

`dev`에는 초기 이진 CMNIST 코드가 남아 있어, 이번 10-class 실험 재현에 필요한 실행 코드는 기존 main 작업본에서 함께 가져왔습니다. 데이터·평가 인터페이스 변경은 PR에서 검토해야 합니다. 다른 main의 과거 실험 산출물이나 원격 브랜치 자체를 병합하지 않았습니다.

공유 manifest는 이 PR에 새로 포함하는 파일의 크기/hash와 제외 범위를 기록합니다. 로그·per-Epoch 예측·Checkpoint·Feature·cache를 제외하고 집계 결과와 재현 소스만 포함했습니다.
