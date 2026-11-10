# 공유 범위와 재현성

이번 변경은 완료된 분석 결과의 공유다. 학습·평가 코드를 변경하거나 재실행하지 않으며, main/dev를 병합하지 않는다.

## 설정 구분

분석 대상은10-class Colored MNIST, VGG11_BN,224px,all pairing,Train corr .95/.80/.65,Val/Test .1,Seed42다. .95는Epoch50부터Resume해100까지 연장했다. 현재dev의 기존README에 있는2-class 예제는 이 실험의 재현 명령이 아니다. 본 PR은 그 학습 코드를 업데이트하지 않는다.

보고서의 코드 해시와 평가 검증은 실제 서버에서 사용한 실행본에 대한 기록이다. PR 기준 브랜치의 실행본과 동일하다고 주장하지 않는다. 전체 재실행에는 해당 코드 revision/서버의Checkpoint·캐시·Feature·예측이 필요하다. Clone만으로 평가를 재실행할 수 있다는 의미가 아니다.

## 포함 자료

한국어 보고서, 회의용 요약, 집계 CSV/검증 JSON,13종 PNG/PDF,대표CAM grid3개를 포함한다. anomaly_sample_transitions.csv는 선정된 오류 원본의 인덱스·예측 변화이며 전체 예측 덤프가 아니다. gradcam_metadata.csv는 고정10개 샘플의 선정 시점 메타데이터다.

원본 데이터셋,Checkpoint(.pt/.pth/.ckpt),Feature/개별 예측 NPZ,캐시,원본 SLURM 로그,인증정보는 포함하지 않는다. source_manifest.json은 서버에서 검증한 원본 산출물의 해시 목록이며 해당 원본 전체가 GitHub에 있다는 뜻이 아니다.

## 공유 복사본의 변경

서버의 원본 분석 결과는 수정하지 않았다. 공유 복사본에서만 서버 절대 프로젝트 경로를{PROJECT_ROOT}/로 표시하고,대표CAM 링크를examples/에 포함한 파일로 변경했다. 수치·지표 정의는 변경하지 않았다. publication_manifest.json에는 게시 파일과 원본 분석 파일의 SHA256을 별도로 기록한다.

이 자료의완료 검증은 실제 평가 종료 후의 검사이며 GitHub CI 결과가 아니다. 새로운 학습·추론·GPU/SLURM 작업은 수행하지 않았다. 단일Seed와Resume RNG한계는 전체 보고서에 명시했다.
