> MNIST 캐시 준비 및 두 Task CPU 오프라인 검사 완료: [데이터셋 준비·검증 보고서](../../reports/cmnist_mnist_cache/dataset_preparation_report.ko.md).
> 공유 경로는 프로젝트의 `.cache/huggingface`이며, 재제출 전 `task.sh --check-data`를 확인한다.
> 아래의 과거 준비 기록 중 cache 미검증 사항은 새 보고서의 검증 결과로 보완된다.

# Corr .95 Epoch 50 → 100 Resume 준비

GPU 학습·모델 재추론·SLURM 제출 없이 준비 및 CPU 검증만 완료했다.
프로젝트 루트: `/data/dada0228/shortcut-learning-dynamics`.
기존 `scripts/cmnist_vgg50/task.sh`와 원본 run은 변경하지 않았다.

## 1. 대상과 새 출력

아래 경로는 모두 프로젝트 루트 기준이다.

| Array Task | 조건 | 기존 run | 원본 체크포인트 |
|---|---|---|---|
| 0 | background / .95 | `runs/vgg11_bg_corr095_seed42_e50_img224_all` | 해당 run의 `checkpoints/epoch_050.pt` |
| 1 | foreground / .95 | `runs/vgg11_fg_corr095_seed42_e50_img224_all` | 해당 run의 `checkpoints/epoch_050.pt` |

새 출력은 다음과 같다. 아직 실제 run 디렉터리는 생성하지 않았다.

- Task 0: `runs/cmnist_vgg100_resume/vgg11_bg_corr095_seed42_e100_from050_img224_all`
- Task 1: `runs/cmnist_vgg100_resume/vgg11_fg_corr095_seed42_e100_from050_img224_all`
- SLURM 로그: `runs/cmnist_vgg100_resume/logs/cmnist_%A_%a.{out,err}`

새 run의 `history.csv`는 원본 0–50을 복사하고 51–100을 추가한다.
`parent_history.csv`에는 원본 History의 변경 없는 복사본을 따로 보관한다.
새 checkpoint/feature는 **51–100만** 생성하며 0–50 자료는 원본에서 참조한다.
`resume_provenance.json`은 원본 경로·config·checkpoint/history SHA-256·목표 Epoch·CF chunk를 기록한다.
최종 Test는 새 run의 `test_metrics.json`에 **epoch 100**으로 저장한다.
원본 epoch 50 Test 결과는 원본에 그대로 남는다. 최신 포인터도 새 run에만 저장한다.

## 2. 실제 Epoch 50 체크포인트에서 확인한 상태

두 파일 모두 `torch.load(map_location='cpu', weights_only=True)`로 순차 로드했다.

- `epoch=50`, config는 각 원본 config.json과 동일.
- `model`: 현재 compact-head VGG11_BN에 strict 로드 성공. 모든 tensor 유한값 확인.
- `optimizer`: SGD, LR .01, momentum .9, weight decay .0001, dampening 0, nesterov false.
  34개 파라미터에 대한 Momentum buffer가 있고 복원 후 값이 원본과 정확히 일치한다.
- `scheduler`: `ConstantLR(factor=1.0, total_iters=1)`, `last_epoch=50`, `_step_count=51`, `_last_lr=[.01]`.
- `scaler`: AMP GradScaler state 존재. CPU용 GradScaler 객체에 동일 state 로드/비교 성공.
  BG scale=2097152, growth tracker=1763; FG scale=262144, growth tracker=1163.
- `torch_rng`: CPU RNG byte tensor(5056개).
- `loader_rng`: 전용 shuffle Generator RNG byte tensor(5056개).
- `cuda_rng`: 1개 GPU RNG byte tensor(16개). **존재/형식 확인만 수행; GPU 복원은 미실행.**
- Python `random` 및 NumPy global RNG는 **원본에 없음**.

모델·optimizer·scheduler·scaler 및 저장된 RNG를 이어받는 stateful Resume은 가능하다.
다만 완전한 비트 단위 Resume 또는 과거에 처음부터 100 Epoch를 실행했을 경우와의 일치를 보장하지 않는다.

## 3. Scheduler와 재현성 한계

기존 설정은 **scheduler=none, LR=.01 고정**이다. 인터페이스를 위해 ConstantLR 객체를 사용한다.
원본 scheduler state를 복원하고 51–100에서도 LR=.01을 유지한다. CPU에서 50번 scheduler step 후
`last_epoch=100`, 모든 Epoch의 LR=.01을 확인했다. 추가 warmup·감쇠·scheduler 초기화를 하지 않는다.

이번 변경은 `--resume <checkpoint> --extend-to-epoch 100`을 명시했을 때에만 목표 Epoch를 확장한다.
기존 `--resume`만의 동작(원본 총 Epoch 유지)은 유지한다. `--epochs 100`만 주면 원래처럼 저장된 총 Epoch를 쓴다.
Cosine의 연장 요청은 명시적으로 거부한다. 따라서 `T_max=50`을 100으로 바꾸거나,
Cosine의 50 이후 LR 반등 정책을 암묵적으로 선택하지 않는다. 본 실험에는 Cosine이 적용되지 않는다.

데이터/배치 순서:

- config를 checkpoint에서 복원하므로 split, seed=42, color mode, correlation, pairing all, batch=128,
  workers=4, image=224, AMP=true, val/test=.10, augment=false를 유지한다.
- `data.py`의 split/색상 배정은 seed가 고정된 로컬 `np.random.default_rng`로 재생성한다.
  global NumPy/Python RNG의 누락과 분리되어 있다. 재현성 확인을 위해 해당 함수 부분만 추가 확인했다.
- 학습 shuffle은 전용 torch Generator를 사용하고 원본 `loader_rng`를 복원한다.
  CPU 테스트에서 저장 Generator 상태로 생성한 다음 Epoch의 55000개 shuffled index가 일치한다.
- 실제 HF 원본 데이터 revision/hash는 체크포인트에 없다. 실제 cache 내용의 역사적 동일성까지 검증한 것은 아니다.
  실행 스크립트는 기존 cache만 쓰도록 HF offline 모드로 실행하며 cache 누락 시 실패한다.
- Python/NumPy global RNG가 원본에 없어 seed 42에서의 fallback 상태를 사용하며 로그에 경고한다.
  이후 저장 checkpoint부터는 두 RNG state도 안전한 builtin 값으로 저장하고 복원한다.
- **기존 checkpoint는 Feature snapshot 실행 직전에 저장**된다. Feature DataLoader는 전용 train Generator가 아닌
  global torch RNG에서 base seed를 하나 소비한다. 이 후속 소비를 Resume에서 재실행하지 않는다.
  따라서 저장된 RNG 자체는 복원하지만 연속 실행에서 다음 Epoch에 도달한 global RNG와는 차이가 날 수 있다.
  현재 CMNIST 색칠/변환은 결정적이고 compact VGG head에는 dropout이 없으며 train shuffle RNG는 별도다.
  그래도 향후 랜덤 연산 변경, worker 내 랜덤 변환, CUDA nondeterministic kernel까지 동일하다고 주장하지 않는다.
- cuDNN 결정적 알고리즘을 새로 강제하지 않는다. PyTorch/CUDA/cuDNN·장치·데이터 버전이 달라지면 결과가 달라질 수 있다.

검증 한계의 대안은 이번 결과를 “Epoch 50 상태에서 고정 LR로 연장한 실험”으로 해석하는 것이다.
과거의 누락 RNG나 실행 직후 상태를 현재 복구할 방법은 없다. 완전한 경로 비교가 필요하면 향후 RNG 저장 경계를
정리하고 동일 환경에서 uninterrupted/resume 대조 실험을 별도로 설계해야 한다. 이번에는 추가 GPU 실험을 하지 않았다.

## 4. 구현 및 안전 장치

수정: `shortcut_experiment.py`의 checkpoint RNG 저장, 명시적 연장 CLI, 별도 경로 검증,
History 복사/provenance, 공통 복원 함수와 복원 로그. Validation/Counterfactual 계산 및 학습 step은 변경하지 않았다.

추가: 본 디렉터리의 `task.sh`, `test_resume.py`, `README.ko.md`, `cpu_validation.json`,
`cpu_validation.log`, `metrics_regression.log`.

- `--extend-to-epoch`는 `--resume` 필수, 저장 총 Epoch보다 큰 값만 허용.
- 원본 run 자체·그 하위·상위 디렉터리로의 출력 거부. 기존 내용이 있는 새 출력도 거부.
- source history가 정확히 0–checkpoint Epoch인지, config.json과 checkpoint config가 같은지 확인 후 복사.
- 원본 checkpoint를 CPU로 로드하고 모델을 장치에 올린 후 optimizer/scaler/RNG를 복원한다.
  복원 완료 후 추가 CPU state 복사본을 해제한다. 두 실험 checkpoint를 동시에 로드하지 않는다.
- 로그에 checkpoint, source/target Epoch, 실제 config, 시작 Epoch 51, 복원 항목, Momentum state 수,
  scheduler/LR, scaler 값, RNG 제한을 출력한다.
- Task 스크립트는 `mkdir`로 출력 경로를 원자적으로 확보한다. 같은 출력으로 재제출하면 실패하므로 덮어쓰지 않는다.
- 이미 시작한 새 run의 중단 복구가 필요하면 새 run 안에서 기존 `--resume latest` 기능을 사용해야 한다.
  이때 `--extend-to-epoch`를 다시 쓰지 않는다. 자동 재제출/복구 스크립트는 이번 범위에 포함하지 않았다.

## 5. CPU 사전 검증

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cmnist-resume-mpl \
 /data/dada0228/conda_envs/shortcut-xai/bin/python scripts/cmnist_vgg100_resume/test_resume.py
bash -n scripts/cmnist_vgg100_resume/task.sh
PYTHONPYCACHEPREFIX=/tmp/cmnist-resume-pycache \
 /data/dada0228/conda_envs/shortcut-xai/bin/python -m py_compile \
 shortcut_experiment.py scripts/cmnist_vgg100_resume/test_resume.py
```

6개 테스트 통과:

1. 두 실제 체크포인트의 strict 모델 호환, Momentum 값, Scheduler/GradScaler, CPU/loader RNG 복원,
   scheduler 51–100 LR 고정, 새 출력의 History 복사 및 원본 주요 파일 SHA-256 보존.
2. 원본/하위/상위/기존 출력, 비증가 target, Cosine 연장 거부.
3. 로컬 synthetic MNIST로 동일 split·label·색상·paired/all-color 개입 재생성 확인.
   네트워크 다운로드나 실제 MNIST 전체 이미지 로딩은 하지 않았다.
4. 새 checkpoint의 Python/NumPy/PyTorch RNG 저장·복원 roundtrip.
5. 임시 디렉터리·작은 모델·모의 학습/평가로 실제 run_training 제어 흐름을 51–100까지 실행.
   50번 학습 호출, History 0–100, 새 checkpoint 51–100만 생성, 최종 Test epoch 100을 확인.
   이 테스트의 수치는 합성 fixture이며 연구 결과가 아니다.
6. dry-run에서 두 Task 매핑과 실제 생성 명령의 CLI parsing, 잘못된 Task/Resume 인자 거부.

추가로 기존 `test_cf_chunking`, `test_loader_memory` 총 4개 테스트 통과.
Multiprocessing 회귀 테스트는 sandbox의 IPC socket 제한으로 첫 시도가 중단되었고,
GPU를 숨긴 CPU 실행으로 sandbox 밖에서 재실행해 통과했다. 로그는 metrics_regression.log에 있다.
Python 문법 및 bash -n 검증도 통과했다.
CUDA RNG 실제 적용, 실제 GPU optimizer step/AMP 연산, 전체 HF cache 접근·GPU 메모리·처리시간은 검증하지 않았다.

## 6. 메모리·시간·저장 비용 추정

리소스 요청은 기존과 동일한 1 GPU / 4 CPU / RAM 16GiB / 최대 6일이다. Array는 `0-1%2`다.
동일 moana-u2의 RTX A2000을 요구하며 2 Task의 실제 동시 시작 여부는 스케줄러 자원에 달려 있다.
기존 task.sh의 6일 요청을 그대로 사용했으며 현재 클러스터 정책을 이번 작업에서 재조회하지 않았다.

- 기존 README의 합성 측정: AMP train allocated 5.60 / reserved 7.82GiB.
  이번 GPU 실측이 아니며 실제 peak 보장값이 아니다. Epoch 연장 자체는 activation 크기를 늘리지 않는다.
- CPU 메모리: 기존 loader 설정(prefetch=1, pin=false, persistent=false) 유지.
  9색 CF CPU tensor와 worker 임시 복사를 포함한 기존 구조 추정은 약 7–10GiB 이상에 프로세스 오버헤드가 추가된다.
  16GiB 내 성공을 새로 실측하지 않았다. 원본 50 Epoch 완료는 같은 설정의 실행 가능성을 뒷받침하는 경험적 자료다.
- 원본 config.json→test_metrics.json mtime 간격은 BG 약 4.49시간, FG 약 4.45시간이다.
  SLURM accounting 실측이 아닌 **파일 시각 기반 전체 실행 근사치**다. 비슷한 처리량이면 추가 50 Epoch도
  Task당 약 4.5시간, 합계 약 9 GPU-hours 규모다. 2 GPU 동시 실행이면 경과 시간은 약 4.5시간 수준이나,
  I/O 경합·다른 작업·GPU 상태·queue 대기는 반영하지 않았다. 초기 평가 생략 등으로 차이가 날 수 있다.
- source epoch 50 크기는 checkpoint 약 73.9MB, Feature 약 22.3MB다.
  2실험×50회 추가 저장 시 약 9.6GB(약 9GiB) 디스크가 필요하다. 원본 전체를 복제하지 않는다.
  실제 압축률 변동/임시 저장 여유를 포함해 최소 12GiB 이상의 추가 여유를 권장한다.

## 7. 사용자가 제출 직전에 확인할 사항

먼저 CPU dry-run으로 경로/명령만 확인할 수 있다. 이 명령은 디렉터리 생성이나 CUDA 검사를 하지 않는다.

```bash
SLURM_ARRAY_TASK_ID=0 bash scripts/cmnist_vgg100_resume/task.sh --dry-run
SLURM_ARRAY_TASK_ID=1 bash scripts/cmnist_vgg100_resume/task.sh --dry-run
```

원본 TRAIN_SUCCESS와 checkpoint 존재, 새 출력 두 경로 미존재, Python 경로 및 torch 2.5.1+cu121/
torchvision 0.20.1+cu121 환경, 기존 HF MNIST cache 접근, 디스크 여유, node/GPU·account/QOS·6일 time limit
유효성을 확인한다. HF offline cache 누락 시 새 다운로드를 시도하지 않고 종료한다.
SLURM 로그 부모 경로는 **제출 전에** 있어야 한다. 아래 명령은 안내용이며 이번 작업에서는 실행하지 않았다.

```bash
cd /data/dada0228/shortcut-learning-dynamics
mkdir -p runs/cmnist_vgg100_resume/logs
sbatch scripts/cmnist_vgg100_resume/task.sh
```

이번 스크립트는 학습 중 Epoch별 Validation/CF 및 Feature snapshot, Epoch 100 최종 Test까지 수행한다.
별도 evaluate.py probe나 Grad-CAM은 제출하지 않는다. 제출 후 로그에서 source epoch 50,
`Continuing from epoch 51 through 100`, 복원 상태, LR=.01을 확인한다.

완료 후에는 원본 0–50과 새 51–100을 합친 History로 동일 강건 기준
(conflicting≥.95, all-color accuracy≥.95, all-color flip≤.05, 연속 3 Epoch)을 적용해야 한다.
Epoch 49–51 또는 50–52처럼 Resume 경계를 가로지르는 구간도 포함한다.
최초 충족 이후에는 충족 비율, 연속 유지 길이, 재이탈 Epoch도 함께 보고한다.
