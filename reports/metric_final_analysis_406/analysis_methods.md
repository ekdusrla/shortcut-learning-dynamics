# 분석 방법과 재현 범위

프로젝트 코드는 수정하지 않았다. 원본 collector도 실행하지 않아 기존 준비 상태 JSON을 덮어쓰지 않았다. 임시 CPU 분석 절차로 별도 산출물을 생성했다.

1. 기존 check_pilot_gate/validate_marker/epoch_identity와 metadata_sha를 읽기 전용으로 재사용했다. GPU full-file audit406개, live ZIP metadata406개, 산출물 SHA와 실제 성공 marker를 확인했다.
2. 기존 transitions로 원본·CF 예측을 재집계하고 그룹별 분모·횟수를 확인했다. 모델을 로드하거나 추론하지 않았다.
3. 해당 source history에서0–50/51–100을 연결했다. .80/.65는50에서 끝나며 Test는 저장된 끝점만 사용했다.
4. Epoch0은 무학습 기준으로 따로 해석한다.1–50의 인접 증가 횟수와 rank 상관은 기술 통계다. 초기1–5/중간6–25/후기26–50 및46–50/96–100 평균은 기술 요약이다. 누락 Epoch는 보간하지 않았다.
5. 행동 기준은 Conflicting Accuracy≥.95, 기존 conflict-only All-color Accuracy≥.95, All9 Flip≤.05의3연속 충족이다. 최초 통과와 유지·재이탈을 구분했다.
6. 기존 Probe 결과와 protocol만 재사용하고 D=2a−1을 명시했다. 새 readout 학습은 없다.3split 범위는 신뢰구간이 아니다.
7. CAM map·PNG·메타데이터로 유효성, 고정 원본/Target/Layer, zero flag를 확인했다. Foreground excess는 유효 map별 mass−mask 면적을 먼저 계산한 값이다.
8. 대표 montage는 기존 PNG의 이미지 영역을 재배치했다. 숫자0과5를 선택했고,5의 고정 원본은 실제 오류 사례가 아니라는 점을 명시했다. 재정규화·새 gradient 계산은 없다.
9. 입력 해시는 source_manifest.json, 검사 결과는 integrity_validation.json에 저장했다. 기존 코드·Checkpoint·Feature·결과 파일에는 쓰지 않았다.

CAM Confidence는 예측 클래스의 최대 softmax 확률이며 top2 margin은 상위 두 확률의 차이다. 잘못 예측한 경우 정답 Target의 확률과 같지 않다.
