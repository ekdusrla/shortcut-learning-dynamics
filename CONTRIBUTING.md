# Git 협업 및 개발 규칙

## 1. 프로젝트 협업 원칙

본 프로젝트는 Shortcut Learning이 학습 과정에서 어떻게 변화하는지 분석하는 것을 목표로 한다.

- 모든 코드는 GitHub 저장소를 통해 공유한다.
- 개인 작업은 별도 브랜치에서 진행한다.
- `main` 브랜치에 직접 커밋하지 않는다.
- 기능 구현 및 수정 사항은 Pull Request(PR)를 통해 `dev` 브랜치에 통합한다.
- 실험 결과의 재현성을 위해 데이터 설정, 모델 설정, 하이퍼파라미터 및 랜덤 시드를 기록한다.

## 2. 브랜치 관리 규칙

| 브랜치 | 용도 |
|---|---|
| `main` | 안정화된 최종 코드 관리 |
| `dev` | 팀원들의 작업 내용을 통합하는 개발 브랜치 |
| `feat/*` | 새로운 기능 구현 |
| `fix/*` | 버그 수정 |
| `chore/*` | 환경 설정, 문서 및 기타 관리 작업 |

### 브랜치 이름 예시

- `feat/data-generation`: Colored MNIST 데이터 생성
- `feat/training`: VGG 및 ResNet 학습 코드
- `feat/evaluation`: Decoded Core Correlation 및 Flip Rate 구현
- `feat/stress-test`: Gaussian Noise, Blur, Crop 구현
- `feat/visualization`: 실험 결과 시각화
- `fix/checkpoint-resume`: 체크포인트 저장 및 재개 오류 수정
- `chore/update-readme`: README 수정

모든 기능 및 수정 브랜치는 `dev`에서 생성한다. 검증이 완료된 코드는 PR을 통해 `dev`에 병합한다. 최종적으로 안정화된 버전만 `main`에 반영한다.

## 3. 기본 작업 절차

### 3.1. 작업 시작

최신 `dev` 브랜치를 가져온 뒤 작업 브랜치를 생성한다.

```bash
git switch dev
git pull origin dev
git switch -c feat/작업이름
```

### 3.2. 작업 및 커밋

변경 사항을 확인한 후 관련 파일만 커밋한다.

```bash
git status
git add 수정한_파일
git commit -m "Add: 작업 내용"
```

### 3.3. 원격 저장소에 업로드

```bash
git push -u origin feat/작업이름
```

### 3.4. Pull Request 생성

GitHub에서 작업 브랜치를 `dev` 대상으로 PR을 생성한다.

- PR 제목과 설명에 변경 사항을 명확하게 작성한다.
- 가능하면 실행 방법과 테스트 결과를 함께 작성한다.
- 충돌 및 주요 오류를 확인한 후 병합한다.
- 다른 팀원의 작업을 덮어쓰거나 검토 없이 `main`에 직접 반영하지 않는다.

## 4. 커밋 메시지 규칙

커밋 메시지는 변경 목적을 알 수 있도록 다음 형식을 사용한다.

`유형: 변경 내용`

| 유형 | 사용 목적 | 예시 |
|---|---|---|
| `Add` | 새로운 기능 추가 | `Add: Colored MNIST generation` |
| `Fix` | 버그 수정 | `Fix: checkpoint resume error` |
| `Update` | 기존 기능 및 설정 변경 | `Update: training hyperparameters` |
| `Refactor` | 기능 변경 없는 코드 구조 개선 | `Refactor: evaluation functions` |
| `Test` | 테스트 코드 및 검증 추가 | `Test: flip rate evaluation` |
| `Docs` | 문서 수정 | `Docs: update experiment guide` |
| `Chore` | 환경 및 기타 관리 작업 | `Chore: update requirements` |

하나의 커밋에는 가능한 한 하나의 논리적인 변경 사항만 포함한다.

## 5. 코드 작성 및 통합 규칙

- 기능별로 파일과 모듈을 분리한다.
- 공통 기능은 중복 구현하지 않고 재사용한다.
- 함수와 변수 이름은 역할을 명확하게 드러내도록 작성한다.
- 주요 함수에는 필요한 경우 입력값, 반환값 및 동작을 설명하는 주석이나 docstring을 작성한다.
- 공통 함수의 입력 및 반환 형식을 변경할 때는 관련 코드에 미치는 영향을 확인한다.
- 새로운 라이브러리를 추가하거나 버전을 변경할 때는 `requirements.txt`를 갱신한다.
- 다른 팀원의 코드와 통합하기 전에 최소한 기본 실행 여부를 확인한다.

## 6. 실험 및 재현성 관리 규칙

본 프로젝트는 학습 epoch에 따른 Shortcut 의존성 변화를 분석하므로 실험 조건을 일관되게 관리한다.

각 실험에서 다음 정보를 기록한다.

- 모델 종류: VGG 또는 ResNet
- 데이터셋 및 Shortcut 상관관계 설정
- Optimizer, Learning Rate, Momentum, Batch Size, Weight Decay
- 최대 Epoch 및 Scheduler 설정
- Random Seed
- 평가 지표 및 평가 시점
- 실험 결과와 실행 환경

실험 설정은 가능하면 `configs/` 아래의 설정 파일로 관리한다.

예시:

```text
configs/
├── colored_mnist_vgg.yaml
└── colored_mnist_resnet.yaml
```

모델, 데이터 조건 및 Seed를 구분하여 결과를 저장한다. 실험 설정을 변경한 경우 이전 결과를 덮어쓰지 말고 서로 다른 설정으로 구분한다.

## 7. 파일 및 데이터 관리 규칙

GitHub에는 코드 실행과 재현에 필요한 파일을 올리고, 대용량 데이터와 학습 산출물은 기본적으로 저장소에 포함하지 않는다.

### GitHub에 포함할 파일

- Python 소스 코드
- `README.md`, `CONTRIBUTING.md`
- `requirements.txt`
- 실험 설정 파일
- 테스트 코드
- 필요한 경우 작은 크기의 예시 데이터 또는 샘플

### 기본적으로 GitHub에 포함하지 않을 파일

- 원본 데이터셋 및 대용량 전처리 데이터
- 학습 체크포인트 (`.pt`, `.pth`, `.ckpt`)
- 대량의 실험 결과 및 로그
- 가상환경 및 캐시 파일
- API 키, 비밀번호, 토큰 등 인증 정보

데이터와 체크포인트를 저장소에서 제외하더라도, 데이터를 생성하거나 다운로드하는 코드와 체크포인트를 불러오는 코드는 공유한다. 재현에 필요한 설정 및 데이터 준비 방법은 문서화한다.

## 8. Pull Request 체크리스트

PR을 생성하기 전에 다음 항목을 확인한다.

- [ ] 변경 목적과 내용이 명확하다.
- [ ] 기본 실행 및 관련 테스트를 확인했다.
- [ ] 다른 기능에 영향을 주는 변경 사항을 확인했다.
- [ ] 실험 설정 및 의존성 변경 사항을 반영했다.
- [ ] 데이터, 체크포인트, 비밀 정보 등 불필요한 파일이 포함되지 않았다.
- [ ] 필요한 문서와 실행 방법을 업데이트했다.

## 9. 기본 원칙

1. `main`에 직접 작업하지 않는다.
2. 모든 기능 개발과 수정은 별도 브랜치에서 진행한다.
3. PR을 통해 변경 사항을 검토하고 `dev`에 통합한다.
4. 실험 설정과 결과를 명확히 기록하여 재현성을 확보한다.
5. 공통 코드 및 인터페이스를 변경할 때는 팀원들과 먼저 협의한다.