# Shortcut Learning Dynamics

Colored MNIST × VGG11_BN의 Epoch 50 실험 6개, 학습 Dynamics와 CPU Linear Probe 결과를 공유합니다.

- [팀원용 실험·결과 안내](EXPERIMENTS.ko.md)
- [학습 Dynamics 보고서](reports/cmnist_vgg50_analysis_review/analysis_report.md)
- [Core/Shortcut Probe 후속 보고서](reports/cmnist_vgg50_probe_followup/analysis_report.md)
- [실제 실험 설정 6개](configs/cmnist_vgg50/)
- [작은 원본 지표 복사본](reports/cmnist_vgg50_shared/source_metrics/)
- [협업 및 PR 규칙](CONTRIBUTING.md)

## 결과 확인과 그림 재생성 (GPU 불필요)

Python 3.11 환경:

```bash
python -m pip install -r requirements-analysis.txt
python reports/cmnist_vgg50_probe_followup/summarize.py --figures-only
python reports/cmnist_vgg50_probe_followup/write_report.py
```

## 실험 실행 코드

모델 실행은 `shortcut_experiment.py`, 데이터 생성은 `data.py`, 별도 평가는 `evaluate.py`가 담당합니다. 전체 실행 의존성은 `requirements.txt`입니다. 실제 실행 인자는 [Epoch50 task](scripts/cmnist_vgg50/task.sh)에, Resume/공식 MNIST cache 준비는 [Resume 안내](scripts/cmnist_vgg100_resume/README.ko.md)에 기록했습니다. 클러스터 경로·자원 정책을 확인한 뒤 별도로 실행해야 합니다.

이 공유 변경에서는 모델 학습·GPU·SLURM을 실행하지 않았습니다. 체크포인트, 원본 Feature, 다운로드 데이터, 대량 중간 산출물·로그·인증정보는 포함하지 않습니다.
