# 전용 GPU 환경

환경 경로: /data/dada0228/conda_envs/shortcut-xai
Python 3.11, torch 2.5.1+cu121, torchvision 0.20.1+cu121을 선택했다.
공식 조합: https://pytorch.org/get-started/previous-versions/
CUDA 12.1 minor compatibility의 Linux 최소 드라이버는 525.60.13이다.
https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/index.html
실제 moana-u2 드라이버는 SSH host key 검증 실패로 미확인이다.
현재 로그인 노드 nvidia-smi는 드라이버에 연결하지 못한다. 따라서 GPU 성공 판정은 보류한다.

의존성: datasets 3.6.0, Pillow 11.1.0, NumPy 1.26.4, pandas 2.2.3,
scikit-learn 1.6.1, matplotlib 3.10.1. requirements.txt의 하한을 충족한다.
프로젝트의 torch.amp.GradScaler('cuda'), weights_only 로드 및 Python 타입 문법을 지원한다.
전체 실제 설치 버전은 installed-packages.txt에 기록한다.

## 검증 및 수동 메모리 테스트
관리자가 제공한 호스트 키로 SSH 신뢰 설정을 해결하고, GPU 할당 세션에서 실행한다.
무할당 상태의 GPU를 SSH에서 임의 점유하지 않는다. 아래 명령은 자동 실행하지 않았다.

```bash
cd /data/dada0228/shortcut-learning-dynamics
/data/dada0228/conda_envs/shortcut-xai/bin/python scripts/cmnist_vgg50/check_cuda.py
# 아래는 별도 GPU 메모리 테스트이며 아직 실행하지 않음
/data/dada0228/conda_envs/shortcut-xai/bin/python scripts/cmnist_vgg50/memory_test.py
```

GPU 할당 요청 예시(사용자가 직접 실행):
```bash
srun --partition=batch_ce_ugrad --account=ugrad_ce --qos=qos_dada0228_2026_2 --nodelist=moana-u2 --gres=gpu:normal:1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:10:00 --pty bash
```

메모리 테스트는 프로젝트 원본 VGG11_BN/compact head, batch 128, 224,
seed 42, SGD lr .01/momentum .9/weight_decay .0001, 기본 AMP로 합성 데이터
3 step만 실행하고 저장하지 않는다. train AMP, validation AMP, evaluate.py의
FP32 추론을 각각 검사한다. 실제 데이터·counterfactual feature 추출·Grad-CAM을
모두 재현하는 것은 아니므로 성공해도 전체 실행의 OOM 부재를 보장하지 않는다.

12GB에서도 OOM 위험은 남는다. 첫 Conv 출력만 FP32 약 1.53GiB,
FP16 약 .77GiB이며 역전파용 activation, BN, gradient, cuDNN workspace가 추가된다.
compact head로 표준 VGG의 거대한 classifier 메모리는 줄어 있지만 입력 해상도의
activation 비용은 그대로다. 특히 evaluate.py는 AMP 없이 실행한다.
실측 전에는 batch 128 안전을 확정할 수 없다. batch 등 실험 조건은 변경하지 않았다.
OOM이면 측정 결과를 바탕으로 별도 승인된 activation checkpointing 등을 검토한다.

SLURM: moana-u2 고정, gpu:normal:1, CPU 4, RAM 16G, 6일, array 0-5%6.
6개 동시 실행은 GPU 6개/CPU 24개/RAM 96GiB를 요청하며 노드의 구성 자원
GPU 8개/CPU 48개/RAM 122174MiB 이내다. 실제 여유 자원에 따라 대기한다.
기존 결과와 다른 Conda 환경은 수정하지 않았다.

최종 설치 검증: Python 3.11.18, torch 2.5.1+cu121, torchvision 0.20.1+cu121,
NumPy 1.26.4. PyArrow는 NumPy 호환성을 위해 19.0.1로 고정했다.
프로젝트 train/evaluate/gradcam과 모든 직접 의존성 import 성공,
NumPy→Torch 변환 성공, VGG 모델 CPU 생성 및 features.18 확인 성공,
pip check 통과. 로그인 노드 CUDA available=False; GPU 검증 완료를 의미하지 않는다.
설치 중 NumPy 충돌 파일은 전용 환경 내부 numpy-install-conflict-backup에 보존했다.
