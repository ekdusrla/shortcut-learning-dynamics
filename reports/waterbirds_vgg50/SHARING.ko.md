# Waterbirds Condition A 공유·재현 안내

[최종 결과와 H1/H2 결론](WATERBIRDS_FINAL_RESULTS_KO.md)을 먼저 읽고, [상세 분석](analysis/README.ko.md)과 [최종 비교표](analysis/final_comparison.ko.md)를 참고한다. 세 조건은 VGG11_BN, pretrained=False, 50 Epoch, seed42, SGD(0.01/0.9/1e-4), batch128, AMP, 224×224, scheduler=none으로 학습했다. Validation/Test correlation은 0.50이다.

## 공개 파일과 원본의 관계

- `../../scripts/waterbirds_vgg50/`: downloader·무결성/누수 검사·데이터 구성·학습/Resume·평가·Pilot·SLURM·CPU 테스트·분석 코드.
- [shared/source_metrics](shared/source_metrics/): 세 조건의 원본과 바이트 단위로 같은 `history.csv`, 경로를 정규화한 config/completion/epoch_000~050 JSON, TRAIN_SUCCESS. 경량 Epoch JSON에는 기존 frozen Train/Val/Test 지표와 체크포인트 무결성 기록이 포함된다. 체크포인트 자체는 없다.
- [shared/dataset](shared/dataset/): 세 Manifest CSV, 원본 이미지 해시/그룹 수/누수 감사 JSON. 이미지는 포함하지 않는다.
- [analysis](analysis/): 통합 CSV·요약 CSV·보고서·PNG/SVG. `tests_analysis.txt`는 기존 CPU 테스트 결과의 텍스트 복사본이다.
- [export_manifest.json](shared/export_manifest.json): 각 공유본의 크기, 공개본 SHA256과 원본 SHA256, 변환 정책. 원본 결과는 수정하지 않았다.

공유 JSON의 `PROJECT_ROOT`, `/local_datasets/USER`는 자리표시자다. 원본의 NaN은 표준 JSON `null`로 변환했으며, 수치가 존재하는 지표는 변경하지 않았다. Dataset JSON에서는 노드 이름을 제외했다. **JSON 안의 기존 config/manifest/checkpoint SHA는 원본 실행의 식별자**이며 경로를 정규화한 공개 JSON의 SHA가 아니다. 공개본 바이트 무결성은 export_manifest의 `published_sha256`로 검사한다. 원본 대비 변경 추적은 `source_sha256`로 구분한다.

기존 보고서·provenance·CSV의 `runs/...` 표기는 원본 출처다. GitHub에서는 같은 basename의 `shared/source_metrics/`로 대응한다. `analysis/provenance.json`은 당시 원본 입력 해시 기록으로, 원본 전체가 GitHub에 있다는 의미가 아니다. 완료 검증은 기존 기록을 공유하며 공개본으로 큰 모델 파일의 무결성을 새로 검증할 수는 없다.

## 기존 CSV만으로 결과 확인·그래프 재현

프로젝트 루트에서 Python 3.10 이상 및 `requirements.txt`의 numpy/pandas/matplotlib를 사용한다. 아래 경로는 torch·원본 이미지·Checkpoint·Feature·GPU·네트워크를 필요로 하지 않는다.

```bash
python3 scripts/waterbirds_vgg50/reproduce_shared.py
# 새 경로에만 표와 그래프 생성
MPLCONFIGDIR=/tmp/waterbirds-mpl python3 scripts/waterbirds_vgg50/reproduce_shared.py \
  --output /tmp/waterbirds-analysis-rerun
```

검사는 공유본 170개 해시, Epoch 0~50, 공통 설정, CSV/기존 Epoch JSON 지표, 통합 CSV의 일치를 확인한다. 원본 모델 추론을 재실행하는 검사가 아니다. PNG/SVG 렌더링 바이트는 matplotlib/font 환경에 따라 달라질 수 있다.

## 데이터 준비와 학습 재현

Waterbirds 원본은 배포하지 않는다. CUB 이미지·마스크 각 11,788개와 Places365 네 카테고리 각 5,000개가 필요하다. 이 실험은 기존 프로젝트의 CUB+Places 합성 재구성이며 공식 Waterbirds composite와 동일한 split이라는 뜻은 아니다. Class prior는 보존되고 조건 간 원본 pairing이 유지된다.

아래는 재현을 위한 사용자 실행 예시다. 게시 준비 과정에서는 다운로드나 학습을 실행하지 않았다. 데이터는 해당 노드의 `/local_datasets`에 두고, SSD 16GiB·NFS Range cache 4GiB·Places category별 1GiB 한도를 유지한다. downloader는 검증 완료 원본을 재사용하고 실패 시 정상 게시본을 보존한다. 압축파일은 검증·게시 후 삭제한다.

```bash
export PYTHON=python3
export WB_ROOT="/local_datasets/$USER/waterbirds_v1"
# 정책 및 공간을 먼저 점검. 실제 다운로드는 사용자가 별도로 실행한다.
bash download_waterbirds.sh --help
bash download_waterbirds.sh --root "$WB_ROOT" --check-only
# bash download_waterbirds.sh --root "$WB_ROOT" --transport curl

# 공유용 Dataset JSON을 그대로 학습에 사용하지 않는다.
# 원본이 준비된 노드에서 새 runtime Manifest를 생성하고 correlation·누수·pairing을 검증한다.
python3 scripts/waterbirds_vgg50/run.py prepare --root "$WB_ROOT" \
  --output reports/waterbirds_vgg50/dataset_local
export WB_MANIFEST="$PWD/reports/waterbirds_vgg50/dataset_local/protocol.json"
```

공유 Manifest CSV 및 감사 기록은 기존 실험의 원본 배정을 확인하기 위한 자료다. 재생성 후 표본 수·그룹 수·correlation과 콘텐츠 split/pairing을 기존 감사 기록과 비교한다. 절대경로/노드가 달라지면 runtime Manifest SHA가 달라지므로 기존 config의 원본 SHA를 새 환경의 값으로 오인하지 않는다.

SLURM 템플릿은 GPU1/CPU4/RAM16G, 본 학습 2시간을 유지한다. 개인 account/QoS/partition은 포함하지 않으며 제출 시 지정한다. 데이터는 노드별 로컬 저장소이므로 `WB_DATA_NODE`와 `--nodelist`를 동일하게 지정해야 한다. 자원에 따른 스케줄링 외 동시 실행 cap은 없다.

```bash
# 학교 정책에 맞는 실제 값으로 설정한다. 값은 Git에 저장하지 않는다.
export WB_DATA_NODE='<dataset-node>'
export WB_PARTITION='<partition>'
export WB_ACCOUNT='<account>'
export WB_QOS='<qos>'
export PYTHON_BIN=python3
mkdir -p logs
# 먼저 성공한 2-Epoch Pilot이 필요하다.
# sbatch --nodelist="$WB_DATA_NODE" --partition="$WB_PARTITION" \
#   --account="$WB_ACCOUNT" --qos="$WB_QOS" scripts/waterbirds_vgg50/pilot_epochs.sh
# export WB_PILOT='<successful-pilot-directory>/summary.json'
# sbatch --array=0-2 --nodelist="$WB_DATA_NODE" --partition="$WB_PARTITION" \
#   --account="$WB_ACCOUNT" --qos="$WB_QOS" scripts/waterbirds_vgg50/task.sh train
```

기존 결과와 같은 출력 디렉터리가 있으면 새 학습은 거부된다. 재현은 새 clone 또는 독립 출력 경로에서 수행한다. Checkpoint/Feature 없이 기존 학습을 Resume할 수는 없다.

## CPU 회귀 검사

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/waterbirds-tests-mpl \
python3 -m unittest discover -s scripts/waterbirds_vgg50 -p 'test_*.py' -v
```

실제 외부 대용량 전송과 GPU 학습은 테스트하지 않는다. HTTP 회귀는 fixture/로컬 서버로 검사한다.
