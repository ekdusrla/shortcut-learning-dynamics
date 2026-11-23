# 최종 공유본 검증 기록

[최종 보고서](REPORT.ko.md) · [핵심 결과](RESULT_SUMMARY.ko.md) · [공유 후보와 크기](PUBLICATION_FILES.csv)

## 검증한 자료

| 검증 범위 | 결과 |
| --- | --- |
| 실행 설정과 상태 | 로컬 config 34개와 별도 초기 E100 기록 2개를 인벤토리에 기록했다. 로컬에는 기본 Fixed 18개, 완료 연장 2개, Dynamic 2개, 파일럿 10개, 중단된 연장 시도 2개가 있었다. 분석·평가의 재실행은 추가 학습 run으로 세지 않았다. |
| 기본 Fixed | 기존 통합 설정 18개를 로컬 config와 대조하고, checkpoint pointer의 epoch와 최종 파일 크기를 확인했다. 18개 기본 조건의 공통 Val은 918개 model-epoch다. Waterbirds ResNet18만의 후기 150개를 더한 Val 1,068개와, 개입 수준의 Val 1,680행을 분리했다. |
| 체크포인트 파일 | 완료 로컬 학습 기록 22개의 마지막 checkpoint ZIP CRC를 직접 검사했다. Colored MNIST BG VGG의 통일된 개입 평가 입력 255개는 현재 SHA-256, 기존 Fixed 감사 또는 Dynamic 제출 전 manifest, 평가 기록의 해시가 모두 일치했다. 22개 마지막 파일 CRC 범위와 255개 SHA-256 범위는 일부 중첩된다. |
| 통일된 개입 | 다섯 조건 각각 E0–E50 51행과 매 epoch 5,000쌍을 확인했다. 총 1,275,000쌍 행의 ID·donor·레이블·색상 해시, FP32 프로토콜, 원래 색 유지, 코어·색 Flip, 원본/반사실 정확도, Target·Switched-to-Target, WGA 및 두 C2I 비율을 원본 샘플과 대조했다. |
| 반사실 분모 | 기존 Val CSV에서 명시된 Flip 건수/유효 쌍, C2I 건수/유효 쌍을 재계산했다. 통일 paired 평가에서 원본 정답 분모도 개별 예측에서 재계산했다. All9 원본 정답 쌍은 원본 이미지 정답 수의 9배다. Waterbirds의 C2I는 이진 집계에 기반한 값으로, 예측별 검증은 통일 Colored MNIST 자료에 해당한다. |
| 추세와 시점 | E1–E5와 E26–E50의 WGA·Flip·두 C2I를 18개 조건에서 독립 계산했다. 90/95% 숫자 probe의 split 최솟값, Waterbirds 55/60%, 역사적 All9 행동과 후속 paired/Core 공동 기준을 원본과 대조했다. Dynamic 여덟 전환의 실현 상관, 3개 동시 개선, Stepwise E34→E36 복귀를 저장 자료로 확인했다. |
| 표·그래프 | 고정 곡선 세 장은 공유용 Val CSV와 같은 정렬된 epoch·분모로 그렸다. 후기·CAM·Dynamic 재사용 PNG는 원본과 동일한 바이트를 복사했다. E50 표는 공유용 원시 CSV에서 생성했다. VGG E100 끝점은 원본 history·C2I·Test 파일을 키로 결합했다. |
| 기술적 공유 검사 | GitHub 상대 이미지와 링크가 신규 공유본 또는 현재 추적 파일을 가리키는지 확인했다. 새 공유 후보는 Markdown/CSV/PNG뿐이다. git diff --check를 통과했고 신규 텍스트의 줄 끝 공백도 검사했다. |

## 그대로 둔 차이와 결측

- **역사적 All9 대 후속 paired.** VGG BG 0.95의 강건 행동 3-epoch 시작 E46과 E20은 Flip 분모가 다르다. 두 기준의 conflicting accuracy와 9색 conflict-only accuracy 항목은 동일하다.
- **Probe 기준.** 역사적 숫자 probe는 세 split 최솟값, paired 5조건 비교는 평균이다. BG에서는 90/95% 도달값이 일치한다. FG와 Waterbirds는 주 보고서에 명시한 프로토콜에서만 비교했다. 레거시 E100의 AdamW 전체 split 및 crossed-color 결과는 별도다.
- **WGA와 C2I.** Frozen Val 그룹 정확도와 online Train을 혼용하지 않았다. C2I 전체와 원본 정답 분모, Paired와 All9를 독립 열로 유지했다. 신규 Dynamic C2I는 신규 FP32 추론 결과다.
- **없는 결과.** Waterbirds 실데이터 Core-only, Waterbirds 독립 Test probe, Waterbirds VGG E100, Colored MNIST 초기 E0–E70 probe·CAM, 주 VGG의 전 epoch frozen Train/Test와 Test 전이 비교는 확보되지 않았다. 일부 CAM epoch와 원본 checkpoint 무결성의 전수 검사는 이전 보고서를 참조한다. 본 보고서에서 관측되지 않은 epoch를 보간하지 않았다.
- **해석 한계.** seed 42의 각 run을 반복 학습으로 세지 않았다. 단일 그룹의 최솟값이나 공간 CAM 질량에서 활용 메커니즘을 추론하지 않았다. 평균 상관과 GPU가 다른 Dynamic 설정에서 schedule 단독 효과를 주장하지 않는다.

## 파일 보존과 공유 범위

기존 reports/runs는 경로·크기·수정시간 목록을 작업 전후 대조했다. 결과 표가 사용하는 로컬 소스의 읽기 시점 SHA-256은 [소스 목록](tables/source_hashes.csv)에 기록했다. 이전부터 있던 README·코드·결과 변경은 유지했다. 작업 브랜치는 analysis/final-shortcut-comparison이고 작업 시작 HEAD는 865fd32였다. README의 기존 변경과 공유 보고서의 diff를 섞지 않기 위해 README는 수정하지 않았다.

공유 후보는 reports/final 전체다. 목록 CSV는 자기 자신을 제외한 파일의 경로·바이트 수·SHA-256을 열거하며, 목록까지 포함하면 30개 파일이다. 원본 체크포인트·feature·로컬 로그는 추가하지 않는다. 외부 링크는 기존 README에서 인용한 group_DRO만 사용했고, 문서에서 클릭 가능한 저장소 상대 링크는 현재 Git 추적 파일 또는 새 공유 파일로 제한했다. 인벤토리 속 원본 로컬 경로는 provenance이며 공유 바이너리가 아니다. GPU 실행, Git staging, commit, push를 하지 않았다.
