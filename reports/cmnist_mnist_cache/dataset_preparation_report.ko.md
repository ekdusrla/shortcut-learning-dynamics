# MNIST 공유 캐시 준비 및 오프라인 검증

공식 `ylecun/mnist`의 고정 revision을 새 공유 캐시에 준비했고, 현재 CPU 노드 moana-r5에서 실제 오프라인 로딩과 변경하지 않은 `data.py`의 전체 데이터셋 구성 경로를 검증했다. GPU 학습/재추론, srun, sbatch는 실행하지 않았다. 원본 학습 코드, 체크포인트, Metrics, Feature 및 Job 191154 실패 출력은 변경하지 않았다.

## 1. 데이터 로딩·분할·색상 적용의 정확한 정의

현재 `data.py:469`는 `load_dataset("ylecun/mnist")`를 호출한다. 코드에는 revision, cache_dir, 별도 데이터 경로가 없으며 HF 환경변수를 따른다. 기존 source와 동일한 호출을 유지했다.

공식 Train 60000개와 Test 10000개를 아래처럼 사용한다.

```python
rng = np.random.default_rng(42)
indices = rng.permutation(60000)
val_ids = indices[:5000]
train_ids = indices[5000:]
test_ids = np.arange(10000)
```

따라서 최종 Train/Validation/Test 크기는 **55000/5000/10000**이다. Label별 stratified split이 아닌 전체 순열 분할이다. Test 순서를 섞지 않는다. raw label 및 image 행 순서가 바뀌면 같은 seed라도 다른 실험이 되므로 이번 검증은 행 순서까지 포함해 hash를 기록했다.

색상 배정은 split별 로컬 `np.random.default_rng`를 사용한다.

- Train seed: 42+100=142, corr=.95.
- Validation seed: 42+200=242, corr=.10.
- Test seed: 42+300=342, corr=.10.
- 먼저 `rng.random(n)<corr`로 aligned 여부를 생성한다. 이어 `rng.integers(1,10,n)`으로 nonzero offset을 생성하여 alternate=(label+offset)%10으로 정한다.
- pairing all에서는 aligned이면 color=label, 아니면 alternate다. 모든 샘플이 label 이외의 9색 중 하나를 alternate로 가질 수 있다. 생성 후 색상은 Epoch마다 다시 추첨하지 않는다.
- Paired CF는 main↔alternate 교환이며, All-color CF는 원래 색상을 제외한 9색을 색상 번호 오름차순으로 적용한다. Train은 CF를 만들지 않고 Val/Test에서만 생성한다.
- Foreground: grayscale alpha×색상 + 나머지×검정. Background: alpha×흰색 + 나머지×0.5×색상.
- uint8 RGB로 round/clip 후 224×224 Resize, ToTensor, ImageNet mean/std Normalize를 적용한다. 이번 config의 augment=false를 유지한다.

모드별 차이는 색상을 칠하는 위치뿐이다. Split indices, label, spurious color, paired CF color 배정은 Background/Foreground에서 완전히 일치한다. 학습 shuffle은 별도 DataLoader Generator의 checkpoint state를 사용하는 기존 Resume 경로를 유지한다.

## 2. 체크포인트 Config 및 Git 대조

두 원본 Epoch 50 checkpoint를 각각 CPU/weights_only=True로 로드하여 config.json과 동일함을 확인했다. seed 42, image 224, val size 5000, train corr .95, val/test corr .10, pairing all, 각 color mode, augment=false를 검사했다. Dataset 설정을 바꾸거나 checkpoint를 다시 저장하지 않았다.

현재 Git HEAD는 `5c7de670803994d1bcd853ed0f962dc6ce1ad0bd`(2026-11-04)다. `data.py`의 마지막 변경은 `8ee02ea5c1eb87de25c22962b8307ad118fd5286`(2026-11-03, all-color CF 추가)이며, 현재 파일의 Git blob은 HEAD:data.py와 정확히 같다. 이보다 앞선 `fb80f63`에서 pairing all 기능이 추가되어 있다. 확인한 코드 변경은 기존 Epoch 50 실행일인 11월 7일보다 이르다.

다만 원본 checkpoint에 실행 당시 Git commit/source snapshot과 raw dataset revision/fingerprint가 없다. 따라서 “현재 tracked 데이터 코드 및 저장 config가 일치한다”는 확인과 “실행 당시 모든 source/data bytes가 동일하다”는 증명을 구별한다. 현재 이미 수정되어 있던 shortcut_experiment.py/evaluate.py는 이번 작업에서 추가 변경하지 않았다.

## 3. 준비한 캐시와 provenance

공유 HF 루트:

`/data/dada0228/shortcut-learning-dynamics/.cache/huggingface`

환경변수는 두 Task에서 동일하다.

```text
HF_HOME=/data/dada0228/shortcut-learning-dynamics/.cache/huggingface
HF_DATASETS_CACHE=$HF_HOME/datasets
HF_HUB_CACHE=$HF_HOME/hub
HF_MODULES_CACHE=$HF_HOME/modules
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

공식 repo: `ylecun/mnist`.
고정 revision: **77f3279092a1c1579b2250db8eafed0ad422088c**.
실제 Arrow 경로:

```text
datasets/ylecun___mnist/mnist/0.0.0/77f3279092a1c1579b2250db8eafed0ad422088c/mnist-train.arrow
datasets/ylecun___mnist/mnist/0.0.0/77f3279092a1c1579b2250db8eafed0ad422088c/mnist-test.arrow
```

| Raw split | 샘플 수 | HF fingerprint |
|---|---:|---|
| train | 60000 | `061e2b9748088db0` |
| test | 10000 | `c3819adbdf4fc121` |

Validation은 공식 split이 아니므로 별도 HF fingerprint를 임의 생성하지 않았다. 파생 Train/Val/Test의 indices/labels/colors SHA-256은 offline_validation.json에 있다.

`mnist_manifest.json`은 준비 시각, repo/revision, 라이브러리 버전, raw split 수/fingerprint/schema/class counts, Arrow SHA-256, 행 순서가 포함된 label SHA-256, 전체 28×28 grayscale pixel SHA-256을 기록한다. Label hash는 little-endian int64 배열, pixel hash는 split 내 원래 순서대로 각 이미지의 uint8 grayscale bytes를 이어 계산했다. HF fingerprint는 provenance 보조값이며 파일/픽셀 무결성 hash와 구별한다.

데이터 준비 크기는 현재 약 38MiB다. 실제 Arrow 합은 약 20.6MB이며 다운로드 및 metadata가 함께 남아 있다. 다른 데이터셋이나 토큰 파일은 복사하지 않았다.

## 4. 안전한 준비 절차

`scripts/cmnist_vgg100_resume/mnist_cache.py`가 절차를 구현한다.

1. 이미 존재하는 목적지는 거부한다. 새 staging cache를 같은 공유 저장소 안에 만든다.
2. 공식 repo에서 지정 revision을 조회하고 immutable commit SHA로 고정한다. implicit token과 Xet 사용은 끄며 라이브러리 버전은 바꾸지 않는다.
3. 그 SHA를 명시하여 `load_dataset`으로 내려받는다. torchvision MNIST, 다른 mirror, 다른 포맷의 수동 재구성으로 바꾸지 않는다.
4. raw sample 수/schema/label 범위, 전체 label/pixel hash, Arrow hash 및 fingerprint를 기록한다.
5. 새 CPU subprocess에서 HF 두 Offline 옵션을 켜고 socket 연결도 차단해 검증한다. 다운로드 프로세스의 메모리나 연결에 의존하지 않는다.
6. 성공한 staging을 최종 경로에 게시하고 **최종 경로에서 다시** CPU 오프라인 검증한다. 실패 시 자동 덮어쓰기/삭제/학습 제출을 하지 않는다.

현재 캐시는 이미 준비되었다. 아래 prepare 명령은 절차 재현용이며 같은 경로에서 다시 실행하면 덮어쓰기를 거부한다.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
 /data/dada0228/conda_envs/shortcut-xai/bin/python \
 scripts/cmnist_vgg100_resume/mnist_cache.py prepare \
 --revision 77f3279092a1c1579b2250db8eafed0ad422088c
```

향후 별도 캐시가 필요하면 비어 있는 다른 `--cache-root`를 지정하고, 인터넷 가능한 CPU 환경에서 같은 라이브러리 버전으로 준비한다. GPU Task는 다운로드하지 않는다. 패키지 업그레이드나 원본 데이터 변환도 이번 절차에 포함하지 않았다.

## 5. 실제 검증 결과

CPU 전용 검증이 통과했다.

- HF Offline 옵션과 socket 차단 상태에서 `load_dataset("ylecun/mnist")` 성공. 실제 socket 연결 시도는 0회이며 CUDA는 초기화되지 않았다.
- 고정 revision 온라인 준비 시와 오프라인 로딩 시 raw split fingerprint, 전체 label/pixel hash, schema 일치.
- 변경하지 않은 `data.build_datasets(cfg)`를 실제 데이터와 두 원본 config로 호출하여 성공.
- Train/Val/Test 55000/5000/10000 및 전체 indices, labels, 색상 배정이 독립 계산식과 일치.
- Background와 Foreground에서 동일한 index/label/color 배열 및 그룹별 count를 사용함을 확인.
- 기존 History의 Validation 100개 label×color cell count 및 기존 최종 Test의 100개 cell count가 두 실험 모두 정확히 일치.
- 기존 Epoch 50 Feature의 샘플 원본 index에 대응하는 **3200개 label/color metadata가 각 mode에서 모두 일치**. Feature tensor를 재계산하거나 모델을 추론하지 않았다.
- split별 첫/마지막 실제 샘플의 224px tensor를 독립 색칠식과 비교. Paired/9색 CF 이미지와 순서도 확인. 모든 70000개의 224px CF를 생성한 것은 아니다.
- 원본 config/history/test metrics/checkpoint/feature 파일 SHA-256이 검증 전후 동일.
- 수정 task.sh의 Task 0/1 `--check-data` 모두 최종 캐시에서 PASS.
- bash -n 및 Python 문법 검증 통과.

실현된 상관은 Bernoulli 추출 결과이므로 설정값과 정확히 같지는 않다.

| Derived split | 샘플 수 | Aligned 수 | 실현 aligned 비율 |
|---|---:|---:|---:|
| train | 55000 | 52240 | 94.9818% |
| val | 5000 | 505 | 10.1000% |
| test | 10000 | 1019 | 10.1900% |

현재 검증은 moana-r5에서 했다. moana-u2에서의 공유 mount/권한 및 CPU preflight는 이번 실행에서 직접 확인하지 못했다.

## 6. 데이터 동일성의 한계

새로 준비한 데이터는 **지정 공식 revision 안에서** 행 순서·label·pixel을 hash로 고정했다. Config/Git 대조와 과거 그룹별 count/3200개 저장 metadata 비교도 통과했다. 이는 과거 설정 및 데이터 순서와의 호환성을 뒷받침한다.

그러나 과거 raw revision, 원본 이미지 hash, 전체 raw fingerprint는 저장되지 않았다. 같은 label 순서와 group counts를 갖는 서로 다른 이미지도 논리적으로 가능하므로, 과거 70000개 이미지가 byte/pixel 단위로 완전히 같다는 주장은 하지 않는다. 새 revision을 “과거 revision을 복구한 값”이라고 보고하지 않는다. Epoch 50 이후 결과는 이 provenance를 명시한 연장 실험으로 해석해야 한다. RNG/비결정적 GPU 연산 관련 Resume 한계도 이전 README의 설명대로 남아 있다.

## 7. 변경 파일 및 재제출 절차

- 추가: `scripts/cmnist_vgg100_resume/mnist_cache.py` — 안전한 준비/CPU 오프라인 검증.
- 수정: `scripts/cmnist_vgg100_resume/task.sh` — 공유 modules 경로 및 manifest/검증 결과, data.py SHA, Arrow SHA, 선택된 캐시 파일, fingerprint를 확인한 뒤 학습 진입.
- 수정: `scripts/cmnist_vgg100_resume/README.ko.md` — 본 보고서와 캐시 검사 절차 연결.
- 생성: `.cache/huggingface/`의 공식 데이터 캐시, mnist_manifest.json, offline_validation.json.
- 생성: 본 보고서 디렉터리의 JSON 스냅샷 및 CPU 검증 로그. 캐시의 live JSON과 구분되는 이번 검증 시점의 기록이다.

현재 task.sh는 manifest와 offline_validation.json을 필수로 요구한다. 파일이 없는 임의 캐시를 조용히 사용하지 않는다. 새 upstream revision을 같은 캐시에 추가하지 않는 전용 캐시로 유지한다. data.py의 호출 자체는 revision이 없지만 Task의 선택 경로·fingerprint·hash 검사가 다른 캐시를 선택하면 실패하도록 보호한다.

재제출 전 아래는 **GPU를 사용하지 않는 검사**다.

```bash
cd /data/dada0228/shortcut-learning-dynamics
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
 /data/dada0228/conda_envs/shortcut-xai/bin/python \
 scripts/cmnist_vgg100_resume/mnist_cache.py verify
SLURM_ARRAY_TASK_ID=0 bash scripts/cmnist_vgg100_resume/task.sh --check-data
SLURM_ARRAY_TASK_ID=1 bash scripts/cmnist_vgg100_resume/task.sh --check-data
```

가능하면 사용자가 접근 가능한 moana-u2 셸에서도 두 --check-data를 실행하여 동일한 공유 경로와 PASS를 확인한다. 추가 GPU 할당 명령은 필요하지 않다. 이번 작업에서는 SSH 접근/SLURM 명령을 추가 실행하지 않았다.

Job 191154의 실패 출력은 그대로 보존했다. 재제출 출력은 두 mode 모두 `..._e100_from050_img224_all_retry191154`이며, 기존 출력이 있으면 거부한다. 원본 Epoch 50 checkpoint에서 다시 시작하며 실패 디렉터리를 --resume latest 대상으로 쓰지 않는다.

위 검사 PASS, 두 retry 출력 경로 미존재, 로그 부모 디렉터리 및 자원/account/QOS 유효성을 확인한 뒤 사용자가 직접 제출한다. 다음 명령은 **안내용이며 실행하지 않았다**.

```bash
mkdir -p runs/cmnist_vgg100_resume/logs
sbatch scripts/cmnist_vgg100_resume/task.sh
```

GPU Task 로그에는 pin된 revision과 fingerprint, 캐시 경로 및 `[offline MNIST] PASS`가 먼저 출력되고, 이어 기존 Resume 로그에 Epoch 51–100과 LR=.01이 기록되어야 한다. Offline fallback의 `Using the latest cached version ... (offline mode is enabled)` 문구 자체는 정상이며 마지막 PASS 여부를 확인한다.
