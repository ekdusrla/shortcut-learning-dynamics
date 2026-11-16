# Waterbirds GitHub 업로드 준비 검증

현재 브랜치는 `main`, 기준 Commit은 `5c7de67`, remote는 `origin` (`https://github.com/ekdusrla/shortcut-learning-dynamics.git`)이다. Commit/Push는 실행하지 않았다.

## 선택 목록과 용량

[파일별 크기·SHA256 목록](upload_inventory.csv)은 공개 payload 233개를 모두 열거한다. 이 문서와 목록 파일까지 합쳐 **235개, 15,827,177 bytes (약 15.094 MiB)**다. Git 객체 압축 전 파일 크기 합계이며 실제 전송량은 다를 수 있다. 목록의 README·공통 코드 해시는 working tree 전체가 아니라 선별된 Staging 버전을 기준으로 한다.

- Waterbirds Python/SLURM 코드 및 회귀 테스트, downloader wrapper.
- 공통 `shortcut_experiment.py` 중 Waterbirds가 사용하는 loader 메모리 제한, CF 분할 평가, Feature loader, checkpoint RNG 기록, state 복원 함수만 선택.
- 원본과 동일한 history.csv 3개, 공개용 config/completion/epoch JSON(51개×3조건), 완료 marker, 소규모 데이터 Manifest/감사 기록과 출처 해시.
- 최종 보고서·상세 분석·모든 요약 CSV·그래프 PNG/SVG 10개·[공유 및 재현 안내](SHARING.ko.md).
- 루트 README의 Waterbirds 링크, 캐시/배열/압축파일 제외 규칙.

## 제외 및 보존

Checkpoint `.pt/.pth/.ckpt`, Feature `.npz/.npy`, 원본 이미지/데이터, 임시 cache/압축파일, 전체 Conda 환경, 불필요한 로그는 포함하지 않는다. 원본 `runs/`는 계속 ignore하며 경량 공유본만 별도 경로에 복사했다. 과거 준비 단계의 서버 자원 조회·preservation·preflight·내부 경로 포함 보고서는 제외했다. 개인 account/QoS/partition과 경로는 공개 SLURM/JSON에서 자리표시자 또는 외부 설정으로 바꿨다. 원본과 공개본 SHA 차이는 [공유 안내](SHARING.ko.md)에서 설명한다.

기존 Colored MNIST 보고서·스크립트·설정·`evaluate.py` 변경, root README의 기존 Colored MNIST 추가분, 공통 코드의 Colored MNIST Epoch 확장 CLI는 Staging에 포함하지 않는다. 기존 working tree 내용은 보존했다. 따라서 README와 공통 코드가 staged/unstaged 양쪽에 나타나는 것은 의도된 부분 Staging이다. 이후 `git add .`, `git add -A`, `git commit -a`, 또는 해당 파일 전체 재추가를 하면 이 분리가 사라질 수 있다.

## 검증

`.gitattributes`는 원본 history CSV의 CRLF와 Matplotlib SVG path의 공백을 그대로 보존하도록 해당 파일에만 whitespace 규칙을 지정한다. 원본 그래프·지표 바이트를 바꾸지 않으며 Python·문서 등 다른 파일의 공백 검사는 유지한다.

- 선택한 파일만 HEAD에 더한 별도 복사본에서 **CPU 회귀 47개 PASS**. GPU 및 실원본 학습/평가는 실행하지 않았다. Sandbox에서 IPC/로컬 HTTP 소켓이 차단되어, 해당 테스트만 제한 밖에서 재실행했다.
- 누락된 Colored MNIST RSS sampler 의존성은 동일 함수의 Waterbirds 내부 복사로 분리했다. Colored MNIST 파일은 변경하지 않았다.
- 공개본 170개 SHA, 설정·153 Epoch 행·기존 Epoch JSON·통합 CSV 일치 PASS. 기존 표/그림 5종의 CPU 재생성도 확인했다.
- Python 문법 27개 및 Bash 문법 PASS. Markdown 링크/이미지 48개 검사 PASS(이 문서의 링크는 추가로 검사).
- 10MiB 초과 파일 0개. 바이너리 가중치/배열/압축파일 0개. 토큰·private key·인증정보 URL·개인 경로 패턴 검사 0건. `.invalid` URL 거부 테스트의 가짜 `user:password` 문자열은 실제 인증정보가 아니므로 검사에서 구분했다. 패턴 검사는 모든 가능한 비밀 검출을 보장하지 않는다.
- 원본 run 633개 size/mtime 전후 변경 0건. 원본 history/JSON과 Dataset 파일의 SHA도 export 출처 기록과 일치한다.
- Staging 후 `git diff --check`와 `git diff --cached --check` 및 선택 목록의 정확한 일치를 확인한다. Staging 검증은 Commit 없이 수행한다.

## 사용자가 최종 확인 후 실행할 명령

```bash
cd /path/to/shortcut-learning-dynamics
git diff --cached --stat
git diff --cached --check
git diff --cached --name-only
# main과 분리된 새 브랜치에 현재 Staging 내용만 Commit
# 같은 이름의 브랜치가 이미 있다면 다른 새 이름을 지정한다.
git switch -c share/waterbirds-condition-a
git commit -m "Add Waterbirds shortcut learning experiments and analysis"
git push -u origin share/waterbirds-condition-a
```

Push는 일반 Push이며 Force 옵션을 사용하지 않는다. 다른 미커밋 작업은 working tree에 남는다. Staging 이후 파일을 바꾸었다면 Commit 전에 diff와 목록을 다시 확인한다.
