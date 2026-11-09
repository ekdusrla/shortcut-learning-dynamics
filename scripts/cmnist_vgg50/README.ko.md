# Counterfactual chunking 적용 업데이트

기존 아래 검토 기록의 1152장 동시 forward 문제를 수정했다.
shortcut_experiment.py train/eval/stress는 --cf-chunk-size (기본 128)를 지원한다.
task.sh도 128을 명시한다. 학습 batch 128, image 224, features-per-group 32와
나머지 실험 조건은 변경하지 않았다. 단일 대체 이미지와 9색 대체 이미지 모두
CPU에서 순서대로 나눈 뒤 청크만 GPU로 전송한다. 전체 9색 GPU 전송은 제거했다.
작은 예측 tensor만 이어붙여 기존 (sample,color) 순서와 모든 집계식을 유지한다.

CPU test_cf_chunking.py 2개 테스트 통과: 원래 전체 forward와 logits 근접성,
argmax 및 전체 evaluate 지표 일치, 비정수 청크 경계, 마지막 작은 batch,
전체 이미지 수, eval/no-grad, 청크 상한, 잘못된 크기 거부를 검사한다.
evaluate.py는 별도의 FP32 classifier/probe 경로로 cf_all을 forward하지 않으므로
수정하지 않았다. Grad-CAM의 단일 샘플 경로도 변경하지 않았다.

GPU 추가 확인: 실제 epoch-0 validation과 feature snapshot, DataLoader RAM,
AMP 청크 추론의 peak allocated/reserved 및 전체/청크 예측 동등성을 확인해야 한다.
128은 제공된 일반 AMP/FP32 추론 메모리 결과를 근거로 선택했으나 실제 경로의
GPU 메모리는 이번에 측정하지 않았다. eval 모드에서도 GPU kernel의 batch별
수치 차이로 경계 logits의 argmax가 달라질 가능성은 있다.
CPU RAM의 9색 tensor/prefetch 비용은 그대로다. 실제 GPU 학습/테스트는 미실행.

CPU 검증 명령:
```bash
OMP_NUM_THREADS=1 /data/dada0228/conda_envs/shortcut-xai/bin/python -m unittest test_cf_chunking -v
```

---
아래는 chunking 적용 이전 검토 기록이다. 제출 명령은 그대로 사용할 수 있다.

# 최종 제출 검토

## 판정
스크립트와 CLI 설정은 일치한다. 다만 실제 validation의 9색 counterfactual
배치가 기존 memory_test.py에서 검증되지 않아, 메모리 측면의 제출 준비 완료 판정은 보류한다.
실제 sbatch 제출, 학습, 추가 GPU 테스트는 이번 검토에서 수행하지 않았다.

## 확인된 환경 (사용자 제공 검증 결과)
moana-u2, RTX A2000 12GB, driver 535.247.01, torch 2.5.1+cu121,
CUDA 동작 및 VGG11_BN features.18 확인 완료.
AMP train allocated 5.60 / reserved 7.82 GiB;
AMP validation allocated 1.88 GiB;
FP32 evaluation allocated 3.22 / reserved 10.88 GiB; 합성 테스트 OOM 없음.
이 결과는 아래 실제 validation 경로의 성공을 보장하지 않는다.

## 정적 검증
- task.sh → shortcut_experiment.py train → run_training(args).
- --no-amp를 지정하지 않으므로 config amp=True. cuda에서 GradScaler와 FP16 autocast 활성화.
- batch 128, image 224, features-per-group 32, epoch 50, seed 42,
  SGD lr .01/momentum .9/weight-decay .0001, scheduler none 유지.
- colored_mnist, val/test corr .10, mnist-val-size 5000, pairing all 유지.
- array 0-5%6, task당 gpu:normal:1/CPU 4/RAM 16G/6일, moana-u2 고정.
  DDP/torchrun 없이 프로세스 1개. CUDA_VISIBLE_DEVICES는 SLURM 설정 유지.
- index 0/1/2: background .95/.80/.65; 3/4/5: foreground .95/.80/.65.
- runs/vgg11_{bg,fg}_corr{095,080,065}_seed42_e50_img224_all:
  여섯 경로는 서로 다르며 검토 시 모두 미존재. mkdir로 기존 경로 덮어쓰기 방지.
- logs/cmnist_%A_%a.{out,err}: array job ID와 task ID로 구분.
  train/eval/gradcam 제출은 각각 다른 job ID이므로 충돌하지 않는다.
- epoch 0과 매 epoch checkpoints/epoch_NNN.pt 저장.
- Evaluation: 각 run의 checkpoints/epoch_050.pt, evaluation_e050/metrics.csv 및 probes.
- Grad-CAM: checkpoints/epoch_002.pt, epoch_025.pt, epoch_050.pt.
  존재하는 파일만 처리. val index 0, target class 0, features.18 고정.
- 데이터 설정을 평가/CAM에도 명시하고 절대 checkpoint 경로 사용.
- TRAIN_SUCCESS는 전체 학습 명령 성공 후 생성. aftercorr로 같은 index 성공에 의존.

## 기존 메모리 테스트와 실제 실행의 차이
1. train step의 모델·AMP·optimizer·입력 크기는 합성 테스트와 일치한다.
2. data.build_datasets는 val/test에 counterfactual=True를 설정한다.
   shortcut_experiment.evaluate는 cf_all_pixel_values (128,9,3,224,224)를
   GPU로 옮겨 flatten(0,1) 후 1152장으로 한 번에 forward한다.
   epoch 0 validation부터 실행되므로 첫 학습 step 이전 OOM 가능성이 있다.
   AMP 128장 validation 결과를 단순 적용할 수 없고, 1152장 첫 Conv 출력만
   FP16 약 6.89GiB이며 입력/다른 activation/workspace가 추가된다.
3. save_feature_snapshot은 AMP 없이 FP32, batch 128로 실행되며 hook이
   pooled feature를 CPU에 저장한다. 매 epoch 수행하며 optimizer 상태도 남는다.
4. evaluate.py는 FP32 분류 및 linear probe encoding을 추가 수행한다.
   데이터 loader가 val/test의 9색 텐서도 CPU에서 만들므로 CPU RAM·prefetch
   비용도 합성 테스트보다 크다. num-workers 4/RAM 16G의 실제 여유는 미검증.
5. Grad-CAM은 단일 샘플이지만 gradient와 hook을 사용한다.
6. reserved 메모리는 allocator 캐시를 포함한다. 10.88GiB reserved 자체가
   10.88GiB의 살아 있는 tensor를 의미하지 않지만 여유가 넉넉하다는 근거도 아니다.

학습 조건을 바꾸지 않고 진행하려면 먼저 위 실제 validation 경로를 검증해야 한다.
OOM이면 사용자 승인 후 counterfactual 추론을 chunking하는 구현을 검토할 수 있다.
이번에는 모델 코드/하이퍼파라미터를 변경하지 않았다.

## 제출 명령 (추가 메모리 검증 후 사용; 현재 실행하지 않음)
```bash
cd /data/dada0228/shortcut-learning-dynamics
export PYTHON_BIN=/data/dada0228/conda_envs/shortcut-xai/bin/python
mkdir -p logs
train_id=$(sbatch --parsable scripts/cmnist_vgg50/task.sh train)
train_id=${train_id%%;*}
echo "Training array: $train_id"
sbatch --dependency="aftercorr:$train_id" scripts/cmnist_vgg50/task.sh eval
sbatch --dependency="aftercorr:$train_id" scripts/cmnist_vgg50/task.sh gradcam
```
학습을 먼저 제출하고, 반환된 ID에 Evaluation/Grad-CAM을 연결한다.
후속 두 단계는 서로 독립이며 GPU 자원이 부족하면 큐에서 대기한다.
기존 환경 구성 기록은 environment.ko.md, 실제 설치 목록은 installed-packages.txt 참고.

## 실제 GPU validation 검증 (스크립트 준비만 완료, 미실행)

validate_gpu.py는 실제 데이터/모델/create_loader/evaluate를 사용하여 val 5000개를
batch 128, image 224, pairing all, AMP, cf chunk 128, workers 4로 검사한다.
9색 이미지 45000개를 처리하며 validation metrics와 CUDA peak allocated/reserved를 기록한다.
이후 72개 대체 이미지만 전체 forward와 chunk 17/128로 비교한다.
1152장 전체 forward는 실행하지 않는다. 전체/청크 logits 허용오차는
atol=.005, rtol=.005이며 argmax는 완전 일치를 요구한다.
원본 epoch-0 FP32 feature snapshot 경로도 features-per-group 32로 검사한다.
이 snapshot은 임시 디렉터리에 생성 후 제거하고 기존 run에는 쓰지 않는다.
학습/optimizer step/checkpoint 저장은 수행하지 않는다.

GPU 할당 세션을 사용자가 직접 시작하고 실행:
```bash
cd /data/dada0228/shortcut-learning-dynamics
srun --partition=batch_ce_ugrad --account=ugrad_ce --qos=qos_dada0228_2026_2 --nodelist=moana-u2 --gres=gpu:normal:1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash
```
할당된 셸에서:
```bash
cd /data/dada0228/shortcut-learning-dynamics
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/cmnist-validation-mpl-${SLURM_JOB_ID}"
/data/dada0228/conda_envs/shortcut-xai/bin/python scripts/cmnist_vgg50/validate_gpu.py --color-mode background --train-corr 0.95 --output validation_gpu_bg095.json
```
foreground도 확인하려면 별도로 --color-mode foreground와 새로운 --output을 사용한다.
각 조건 확인 시 train-corr .80/.65도 지정할 수 있다. 기존 보고서가 있으면 중단한다.

성공 기준: 종료 코드 0, JSON success=true, validation samples=5000,
all_counterfactual_images=45000, comparison prediction_mismatches=0 및 logits_close=true,
feature 추출 완료, CUDA/CPU OOM 및 worker 오류 없음.
CUDA peak는 각 단계별 bytes로 기록한다. reserved는 allocator cache도 포함한다.
CPU는 /proc을 0.2초 간격으로 읽어 부모+worker RSS 합의 관측 최대치를 기록한다.
공유 페이지 중복과 짧은 피크 누락 가능성이 있어 정확한 cgroup peak는 아니다.
할당 셸에서 /usr/bin/time -v로 실행하면 부모 RSS도 추가 기록할 수 있다.
프로세스가 cgroup OOM으로 강제 종료되면 JSON 기록도 불완전할 수 있으므로
SLURM MaxRSS와 종료 상태도 확인한다.

## Job 153558 CPU OOM 대응
cgroup 16GiB 한계 도달 및 oom_kill은 사용자 측정으로 확인됨.
정적 원인 분석: MNIST는 원본 HF 데이터에서 __getitem__ 시점에만 색칠/224 변환한다.
전체 224 RGB float 캐시나 45000개 대체 이미지 동시 보관은 없다.
하지만 val batch마다 원본+1대체+9대체=11종 float32가 생성된다.
128*11*3*224*224*4 = 847773696 bytes, 약 .790GiB.
기존 workers 4/prefetch 2는 최대 8개 배치 선행 처리(약 6.32GiB)를 허용하고,
worker 내 개별 sample 목록과 collate 결과가 겹치는 복사, shared-memory 전송,
pin thread 복사, 현재 batch, HF/프로세스 메모리가 더해진다.
persistent worker는 단계 사이에도 살아 있어 val/feature/test loader 풀이 겹칠 수 있다.
정확히 어느 할당이 kill을 유발했는지는 프로파일 없이 확정할 수 없지만,
이 구조가 cgroup 측정과 일치하는 주요 원인이다. 모델 생성 시 표준 VGG head의
일시적 CPU 할당도 있으나 전체 이미지 사전 캐싱은 아니다.

loader_memory_options 공통 함수로 prefetch_factor=1, pin_memory=False,
persistent_workers=False 적용. train/val/test, feature snapshot, evaluate.py,
validate_gpu.py(create_loader 경유)에 적용된다. workers 4와 GPU chunk 128 유지.
GPU 전송 overlap/worker 재시작 성능은 달라질 수 있지만 데이터/지표 정의는 동일하다.

예상 host tensor 규모: 선행 batch 약 3.16GiB + 현재 batch .79GiB +
4 worker의 collate 일시 복사 최대 약 3.16GiB 등으로 대략 7~10GiB 이상에
HF/프로세스/allocator 오버헤드가 더해질 수 있다. 정확한 peak나 16GiB 성공은
재실측 전 보장하지 않는다. shared pages의 RSS 합은 cgroup 사용량과 다르다.
CPU 테스트는 두 색상 모드에서 이전/수정 loader의 모든 tensor bitwise 동일성을
검사하며 기존 chunking 전체 평가 지표 동등성 테스트도 함께 실행한다.

재실행: 위 srun의 CPU 4/RAM 16G/GPU 1 설정을 유지하고 새 결과 파일 사용:
```bash
/data/dada0228/conda_envs/shortcut-xai/bin/python scripts/cmnist_vgg50/validate_gpu.py --color-mode background --train-corr 0.95 --output validation_gpu_bg095_hostmem_v2.json
```
이번 수정에서 GPU 테스트와 SLURM 제출은 수행하지 않았다.
