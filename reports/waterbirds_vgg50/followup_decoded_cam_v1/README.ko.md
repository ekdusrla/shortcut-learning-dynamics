# Waterbirds Decoded/CAM 후속 분석 v1

[분석 보고서](analysis_report.md) · [Probe 결과](probe_summary.csv) · [전환 민감도](core_transition_analysis.csv) · [검증](validation_results.json)

## 완료 및 프로토콜

기존 VGG 학습 결과를 보존하고 저장된 Validation Feature만 사용했다. 기존 evaluate.DecodedCorrelation을 재사용한 CPU Probe이며 새로운 VGG 학습이 아니다. **원래 val-fit/test-score와 달리 Validation 내부 content holdout**이다. fit56/score56, 각4그룹14개, 모든 조건/시점/Layer에서 동일 source-content-disjoint 분할을 사용했다. 단일 seed42·작은 holdout이므로 독립 Test 일반화 또는 내부 정보의 인과적 사용으로 해석하지 않는다.

주 분석 avgpool + 보조5Layer, Core/Spurious 각각 Linear(D,2), AdamW100epochs,batch256,lr.001,wd.0001. Decode=2×Accuracy−1. `.55/.60/.70/.80` 및1/3/5/10회 지속 기준은 사후 분석이다. baseline0 도달과 미도달(NR)을 구분한다.

- `probe_summary.csv`: 1836개 Probe, Accuracy/BA/decoded/Layer/seed/분할.
- `probe_split.csv`: 동일 ID·component·분할. fit와score가 새·양쪽 배경 SHA를 공유하지 않는다.
- `behavior_all_splits.csv`: 기존 Epoch JSON의 frozen Train/Val/Test 459행. online Train은 기존 History에만 있다.
- `integrated_metrics.csv`, `phase_summary.csv`, `descriptive_relationships.csv`: 기존 행동 지표와 통합·초중후기·기술적 상관.
- `probe_behavior_disagreements.csv`: Core/Spurious 동시 복원, 높은Core와 낮은WGA/높은Flip의 사후 사례.
- `permutation_controls.csv`: E0/E50 avgpool fit label permutation 1회. 정식 null 분포가 아니다.
- `input_integrity.json`: 입력 Feature SHA와 Checkpoint 크기 감사. 큰 Checkpoint SHA 전체 재검증은 하지 않았다.
- `gradcam/epoch_plan.json`: 비교 계획만 존재하며 실제 CAM은 미실행이다.

## CPU 재현

프로젝트 루트에서 `requirements.txt` 환경을 사용한다. 기존 output은 재사용/덮어쓰기하지 않는다.

```bash
export PYTHON=/data/dada0228/conda_envs/shortcut-xai/bin/python
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/wb-followup-mpl
"$PYTHON" scripts/waterbirds_followup/probes.py --output reports/waterbirds_vgg50/followup_decoded_cam_v2
"$PYTHON" scripts/waterbirds_followup/report.py --analysis reports/waterbirds_vgg50/followup_decoded_cam_v2
"$PYTHON" -m unittest scripts.waterbirds_followup.test_followup -v
```

## 미완료: GPU 추론 계획

현재 moana-r5에는 원본 `/local_datasets/dada0228/waterbirds_v1`이 없어 CAM과 Test Feature 추출은 실행하지 않았다. 기존 Manifest가 지정한 moana-u2에서 실행해야 한다. 아래 명령은 사용자 실행용이며 이번에 제출하지 않았다. 자원 상태/QoS는 변할 수 있으므로 확인 없이 동시 수를 고정하지 않는다.

```bash
sinfo -N -n moana-u2 -o '%N %t %c %m %G'
scontrol show node moana-u2
squeue -u "$USER"
sacctmgr show qos format=Name,MaxTRESPU,MaxJobsPU,MaxSubmitPU -P
# 해당 노드에서:
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
free -h
df -h /local_datasets "$PWD"
```

GPU1/CPU4/RAM16G/2시간 템플릿을 준비했다. 과거 학습 Pilot의 GPU allocated5.60GiB/reserved5.84GiB, CPU tree RSS6.24GiB는 batch128 학습 수치이며 새 CAM 측정값이 아니다. CAM은 한 이미지씩 FP32, 모델 한 개/한 Epoch씩 로드하고 DataLoader worker를 사용하지 않는다. 추가 Test 추출은 batch16/workers0·128개/Epoch로 메모리를 제한한다. 먼저 한 조건·소수 Epoch Pilot으로 peak/RSS/time을 확인해야 한다. 신규 작업의 사용량·소요 시간은 아직 실측하지 않았다.

동시 가능한 수는 최대3과 GPU 잔여·CPU잔여/4·RAM잔여/16GiB·QoS 잔여의 최솟값이다. 템플릿에는 `%2` 같은 고정 제한이 없다. SSD16GiB/NFS cache4GiB downloader 정책은 변경하지 않았고 새 데이터 다운로드는 없다. 결과는 프로젝트 reports에 기록하며 원본 SSD에 추가하지 않는다. 추출128×1984×4 bytes의 비압축 Feature는 파일당 약0.97MiB, Test153개 약148MiB이고 Val 복사본까지 약297MiB+압축 메타 여유가 필요하다. CAM heatmap은 224²×4=200704bytes/장, 계획 장수에 따라 PNG·비교 그림 용량이 추가된다. `gradcam/epoch_plan.json`으로 장수를 확인하고 최소1GiB 이상의 추가 여유를 확보한다.

```bash
# 원본 데이터 노드에서 프로젝트 루트로 이동한 뒤 실제 학교 계정/QoS/Partition을 설정한다.
export WB_PROJECT_ROOT="$PWD"
export WB_ANALYSIS="$PWD/reports/waterbirds_vgg50/followup_decoded_cam_v1"
export WB_DATA_NODE=moana-u2
export PYTHON_BIN=/data/dada0228/conda_envs/shortcut-xai/bin/python
export WB_PARTITION='<partition>' WB_ACCOUNT='<account>' WB_QOS='<qos>'
mkdir -p logs
# 위에서 CUDA_VISIBLE_DEVICES를 비웠다면 제출 전에 해제해 SLURM이 GPU를 배정하도록 한다.
unset CUDA_VISIBLE_DEVICES

# CAM: 먼저 1조건만 제출해 측정한 후 나머지 실행 여부 판단
sbatch --array=0 --nodelist="$WB_DATA_NODE" --partition="$WB_PARTITION" \
  --account="$WB_ACCOUNT" --qos="$WB_QOS" scripts/waterbirds_followup/cam.slurm
# 성공 후 080/065를 실행. 이미 성공한095 출력은 다시 실행하지 않는다.
# sbatch --array=1-2 --nodelist="$WB_DATA_NODE" --partition="$WB_PARTITION" \
#   --account="$WB_ACCOUNT" --qos="$WB_QOS" scripts/waterbirds_followup/cam.slurm
```

CAM worker는 현재 고정 E0/1/10/25/50 및 후보 시점을 한 조건씩 처리한다. 더 작은 Pilot이 필요하면 `epoch_plan.json`을 새 analysis 디렉터리에 복사해 소수 Epoch 계획으로 실행하고, 전체 결과는 별도 경로에 둔다. 완료된 디렉터리를 덮어쓰지 않는다. 실패한 부분 출력에도 CAM_SUCCESS가 없으면 완료로 취급하지 않는다. 재시도는 실패 출력을 보존한 새 경로에서 수행한다.

세 CAM_SUCCESS가 생긴 다음 CPU에서 실제 결과를 수집한다.

```bash
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" scripts/waterbirds_followup/collect_cam.py \
 --analysis "$WB_ANALYSIS" --output "$WB_ANALYSIS/cam_integration_v1"
```

독립 Test Feature가 필요하면 별도 mode로 추출한다. 새로운 VGG 학습이 아니다.

```bash
export WB_MODE=extract
export WB_EXTRACT_EPOCHS=0
export WB_EXTRACT_OUTPUT="$WB_ANALYSIS/extraction_pilot"
# 먼저 E0 단일조건 메모리/시간 측정
sbatch --array=0 --nodelist="$WB_DATA_NODE" --partition="$WB_PARTITION" \
 --account="$WB_ACCOUNT" --qos="$WB_QOS" scripts/waterbirds_followup/cam.slurm
# 위 Pilot 완료 및 자원 확인 후 새 경로에서 전체 추출
# export WB_EXTRACT_EPOCHS=0:50 WB_EXTRACT_OUTPUT="$WB_ANALYSIS/independent_features"
# sbatch --array=0-2 --nodelist="$WB_DATA_NODE" --partition="$WB_PARTITION" \
#  --account="$WB_ACCOUNT" --qos="$WB_QOS" scripts/waterbirds_followup/cam.slurm
# 3조건 전체 추출이 완료된 후 원래 val→test 프로토콜의 Probe를 CPU로 계산:
# CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" scripts/waterbirds_followup/independent_probe.py \
#  --features "$WB_ANALYSIS/independent_features" --output "$WB_ANALYSIS/independent_test_probes_v1"
```

Test Feature 추출/독립 Probe 결과는 현재 Validation holdout 결과와 별도로 보관·해석한다. 같은 지표로 섞지 않는다. 실제 CAM/독립 Test가 확보되면 새 버전 보고서에서 H1/H2를 재평가하며, 현재 보고서는 실행하지 않은 증거를 채워 넣지 않는다. Commit/Push/Pull 및 SLURM 제출은 이번 작업에서 실행하지 않았다.
