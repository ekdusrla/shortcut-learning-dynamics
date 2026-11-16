# Grad-CAM 대표 사례와 해석

이번 후속 평가의 계획된96시점, 고정 Validation 원본10개×원본/CF=1,920개 map을 분석했다. 과거 원본100 실험의 CAM은 포함하지 않았다. 정답 숫자 Target, features.25, FP32로 일관되며 주 평가의 AMP 예측과 구분한다.

## 이미지에서 확인한 관측

저장 grid와 새 montage를 직접 확인했다. 초기 BG .95 Epoch1에서는 색상 변경으로 예측이 정답으로 바뀌는 사례가 많다. 숫자0은 원본 p9→CF p0, 숫자5는 p2→p5다. 그런데 이때도 정답 Target CAM은 digit 주변을 강하게 표시한다. **형태 주변 CAM이 보인다는 사실은 원분류기의 실제 형태 사용을 증명하지 않는다.** 대상 클래스는 실제 예측 클래스가 아닌 고정된 정답이다.

후기 map은 대체로 획·윤곽 주변에 분포하지만 배경과 digit 내부 공간까지 넓게 퍼진다. 색상 쌍 간 차이도 남는다. Epoch50/100 고정10쌍은 모두 원본·CF 정답이지만 전체 Validation C2I는0이 아니다. 표본 중9개는 conflicting,1개는 aligned여서 대부분의 paired CF가 정답 색상으로 바뀐다. 전체 오류 집단이나 All9 민감도를 대표하지 않는다.

[BG 단계별 사례](figures/11_cam_phases_bg.png), [FG 사례](figures/11_cam_phases_fg.png)는 숫자0/5의 동일 원본을0,1,3,50,51,100에 비교한 기존 PNG 재배치다. 새 forward/backward는 없다. 원래 grid의 예측·Confidence는 아래 링크와 개별 CSV에서 확인할 수 있다.

## 공간 Proxy

mask는 원본 grayscale alpha를224px로 resize한 뒤 alpha>.1로 정의했다. 고정10개의 면적 평균은19.0561%다. CAM mass는 정규화 map의 양의 합 중 mask 안의 비율이며, background mass는 보완 영역이다. 이를 인과적 Core/Shortcut 기여도로 부르지 않는다. Foreground에서는 형태와 색상이 같은 픽셀에 있다. Mask threshold, Layer,14×14→224 보간에도 민감하다.

| 조건 | Epoch1 cosine | Epoch50 | Epoch100 |
|---|---:|---:|---:|
| BG .95 | .8155 | .9699 | .9547 |
| FG .95 | .8376 | .9667 | .9570 |

위 cosine은 각 Epoch 안에서 원본–CF를 비교한 값이며 Epoch 간 map cosine이 아니다. 후기 쌍 유사성은 초기보다 높지만50→100에서는 낮아진다. 행동 악화와 방향이 맞는 보조 관찰이며 인과적 증거는 아니다. BG .95의 원본 foreground mass는 Epoch1 41.09%,50 41.54%,100 40.48%로 단조 증가하지 않는다. 같은 기간 행동은 크게 변한다. FG .95는26.12→35.26→35.10%다. Mode 간 명도·합성 차이가 있어 이를 Core 사용량의 순위로 볼 수 없다.

6개 zero-map은 FG Epoch0 숫자2의 원본·CF다. ReLU와 독립 min-max 정규화에 의해0이 될 수 있으며 정보 부재를 증명하지 않는다. Mass/cosine 결측을 제외한 유효 n을 보존했다. Foreground excess 그림은 각 유효 map에서 mass−mask 면적을 먼저 계산한 뒤 평균했다. Band나 seed 일반화 통계가 아니다.

## Epoch28 사례의 한계

[고정 숫자5 비교](figures/12_cam_anomaly_digit5.png)의 index35515는27/28/29 모두 정답이다. 원본 Confidence만99.916→96.853→99.781%로 변하고 형태 주변 CAM의 차이는 작다. 이 그림으로 그룹 Accuracy 급락의 원인을 설명할 수 없다.

같은 숫자5 conflicting 그룹은98.98→77.41→97.97%였다. 실제 실패 원본이 현재 고정 표본에 포함되지 않는다는 한계를 보여준다. 추가 원인 분석은 오류 전환 CSV에서 실제 실패 원본을 선택해 CAM을 계산해야 하며, 이번에는 수행하지 않았다.

## 근거 파일

- [BG .95 Epoch1 원본 grid](examples/cam_bg095_epoch001.png)
- [BG .95 Epoch28 원본 grid](examples/cam_bg095_epoch028.png)
- [FG .95 Epoch100 원본 grid](examples/cam_fg095_epoch100.png)
- [공간 지표 그림](figures/10_gradcam_spatial_proxy.png), [시점 통계](gradcam_epoch_summary.csv), [개별 메타데이터](gradcam_metadata.csv), [Zero-CAM 목록](zero_cam_cases.csv)
