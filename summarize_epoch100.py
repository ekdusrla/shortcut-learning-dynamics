"""Validate and summarize the artifacts from extract_epoch100.py."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'shortcut_results' / 'analysis_epoch100'
rows = list(csv.DictReader((OUT / 'decoded_correlation.csv').open(encoding='utf-8')))
assert len(rows) == 2
metas = []
checks = []
for row in rows:
    run = 'resnet18_corr095_seed42_e100' if row['model'] == 'resnet18' else 'vgg11_corr095_seed42_e100'
    dest = OUT / run
    meta = json.loads((dest / 'metadata.json').read_text())
    metas.append(meta)
    for split, n in [('val', 5000), ('test', 10000)]:
        with np.load(dest / f'{split}_features.npz') as data:
            assert data['features'].shape == (n, 512)
            assert np.isfinite(data['features']).all()
            assert len(np.unique(data['indices'])) == n
    assert len(list((dest / 'gradcam').glob('*.png'))) == 20
    assert len(list((dest / 'gradcam').glob('*.json'))) == 20
    maps = list((dest / 'gradcam').glob('*.npy'))
    assert len(maps) == 20
    for path in maps:
        cam = np.load(path)
        assert cam.shape == (64, 64) and np.isfinite(cam).all()
        assert cam.min() >= 0 and cam.max() <= 1
    original = json.loads((ROOT / 'shortcut_results' / 'runs' / run / 'test_metrics.json').read_text())['metrics']['acc']
    assert abs(original - float(row['test_classifier_accuracy'])) < 1e-6
    for target in ['core', 'spurious']:
        assert abs(float(row[f'{target}_decoded_correlation']) - (2 * float(row[f'{target}_accuracy']) - 1)) < 1e-12
    checks.append(dict(model=row['model'], features='PASS', gradcam='PASS', original_test_accuracy='PASS',
                       zero_heatmaps=sum(float(np.load(p).max()) == 0 for p in maps)))
assert [x['original_index'] for x in metas[0]['gradcam_samples']] == [x['original_index'] for x in metas[1]['gradcam_samples']]
report = ['# 100 epoch Grad-CAM 및 decoded correlation', '',
          '제공된 두 epoch 100 체크포인트에서 업데이트된 `gradcam.py`와 `evaluate.py`의 클래스를 사용해 추출했다.', '',
          '| 모델 | 숫자 probe 정확도 | 숫자 decoded correlation | 색상 probe 정확도 | 색상 decoded correlation | 원래 분류기 test 정확도 |',
          '|---|---:|---:|---:|---:|---:|']
for r in rows:
    report.append(f"| {r['model']} | {float(r['core_accuracy']):.2%} | {float(r['core_decoded_correlation']):.4f} | {float(r['spurious_accuracy']):.2%} | {float(r['spurious_decoded_correlation']):.4f} | {float(r['test_classifier_accuracy']):.2%} |")
report += ['', '## 방법', '',
           '- 최종 분류기 직전 512차원 feature를 validation 5,000개와 test 10,000개 전체에서 다시 추출했다.',
           '- validation으로 숫자(core) 및 색상(spurious) 10-class linear probe를 각각 학습하고 test로 평가했다. 원래 모델 가중치는 고정했다.',
           '- 업데이트 코드 기본 설정: seed 42, probe 100 epochs, AdamW, LR 0.001, weight decay 0.0001, batch 256. 학습 feature는 원래 scale을 유지했다.',
           '- decoded correlation은 코드 정의에 따라 2 × accuracy − 1이다. Pearson 상관계수가 아니며 10-class 균등 추측 기준값은 −0.8이다.',
           '- Grad-CAM은 ResNet18 layer4와 VGG11 features.25에 적용했다. 숫자 0–9마다 색상 일치/불일치 그룹의 첫 test 표본을 선택했으며 두 모델에서 같은 20개 입력을 사용했다.',
           '- Grad-CAM target은 각 입력의 예측 class이다. 각 그룹의 서로 다른 원본 표본을 선택했으므로 일치/불일치 열을 같은 숫자의 재색칠 전후 비교로 해석하면 안 된다.',
           '- 각 heatmap은 입력 크기 64×64로 보간하고 표본별 0–1 정규화했다. 최종층 공간 해상도가 낮아 넓게 번지는 heatmap이며 표본 간 절대 중요도 비교에는 적합하지 않다.', '',
           '## 결과 파일', '',
           '- `decoded_correlation.csv`: 두 모델 수치 비교.',
           '- 각 모델의 `gradcam_overview.png`: 20개 입력과 overlay 모음.',
           '- 각 모델의 `gradcam/`: 20개 3-panel PNG, float32 heatmap NPY, 예측과 표본 정보를 담은 JSON.',
           '- 각 모델의 `val_features.npz`, `test_features.npz`: features, labels, spurious, 원본 indices.',
           '- 각 모델의 `core_probe.pt`, `spurious_probe.pt`: 학습된 probe 가중치.',
           '- 각 모델의 `metadata.json`: checkpoint SHA256, 원래 config, probe 설정, 표본 목록.', '',
           '## 해석 및 검증', '',
           '숫자와 색상 probe 정확도는 각 정보가 최종 feature에서 선형적으로 해독되는 정도다. 높은 색상 해독 성능만으로 원래 분류기가 색상에 얼마나 인과적으로 의존하는지 결론 내릴 수는 없다. 본 결과는 seed 42의 두 체크포인트 및 probe 학습 1회에 대한 측정이다.', '',
           '두 모델의 원래 test 정확도가 제공된 test_metrics.json과 일치했다. feature shape와 유한값, heatmap 범위와 파일 수, 두 모델의 동일 표본 사용, decoded correlation 계산식을 검증했다.', '',
           '재실행: 프로젝트의 가상환경 Python으로 `extract_epoch100.py` 실행 후 `summarize_epoch100.py` 실행. 원본 MNIST 캐시가 필요하다.']
(OUT / 'report_ko.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
sources = {}
for name in ['gradcam.py', 'evaluate.py', 'data.py', 'shortcut_experiment.py', 'extract_epoch100.py', 'summarize_epoch100.py']:
    sources[name] = hashlib.sha256((ROOT / 'shortcut_learning_experiment' / name).read_bytes()).hexdigest()
(OUT / 'validation.json').write_text(json.dumps(dict(checks=checks, matching_gradcam_samples=True, source_sha256=sources), indent=2))
print('\n'.join(report[:9]))
print('Validation:', checks)
