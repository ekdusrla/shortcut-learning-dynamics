"""Join history, probe, and Grad-CAM artifacts without inventing missing epochs."""
import argparse
import csv
import json
import logging
from pathlib import Path

from followup_common import EPOCHS, start_output, write_json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_tables(paths, filename):
    tables = []
    for path in paths:
        file = path / filename if path.is_dir() else path
        tables.append(pd.read_csv(file))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def consistent_protocol(frame, name):
    if not frame.empty and frame['protocol_sha256'].nunique() != 1:
        raise ValueError(f'{name} mixes different data protocols')


def plot_lines(frame, value, grouping, output, title, ylabel):
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, group in frame.groupby(grouping, dropna=False):
        group = group.sort_values('epoch')
        key = key if isinstance(key, tuple) else (key,)
        ax.plot(group['epoch'], group[value], marker='o', markersize=3,
                label=' / '.join(map(str, key)))
    ax.set(xlabel='Epoch (observed checkpoints only)', ylabel=ylabel, title=title)
    ax.set_xlim(min(0, frame['epoch'].min()) - 2, max(100, frame['epoch'].max()) + 2)
    ax.grid(alpha=.25)
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--probe-results', type=Path, nargs='*', default=[])
    p.add_argument('--gradcam-results', type=Path, nargs='*', default=[])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--core-threshold', type=float, default=.9,
                   help='Predeclared descriptive chance-adjusted score threshold; not a learned onset estimate')
    args = p.parse_args()
    if not 0 < args.core_threshold < 1:
        p.error('core-threshold must be in (0,1)')
    out = start_output(args.output, args)
    histories, configs = [], {}
    for run in args.runs:
        cfg = json.loads((run / 'config.json').read_text())
        if run.name in configs:
            raise ValueError('Duplicate run names')
        configs[run.name] = cfg
        history = pd.read_csv(run / 'history.csv')
        if history['epoch'].duplicated().any():
            raise ValueError('Duplicate history epochs')
        history['run_id'], history['model'], history['train_corr'], history['model_seed'] = run.name, cfg['model'], cfg['train_corr'], cfg['seed']
        histories.append(history)
    history = pd.concat(histories, ignore_index=True)
    probe = read_tables(args.probe_results, 'probe_results.csv')
    cam = read_tables(args.gradcam_results, 'gradcam_metrics.csv')
    consistent_protocol(probe, 'Probe results')
    consistent_protocol(cam, 'Grad-CAM results')
    if not probe.empty and not cam.empty and probe['protocol_sha256'].iloc[0] != cam['protocol_sha256'].iloc[0]:
        raise ValueError('Probe and Grad-CAM protocols differ')
    if not probe.empty:
        keys = ['run_id', 'epoch', 'layer', 'target']
        if probe.duplicated(keys).any():
            raise ValueError('Overlapping probe jobs; supply nonoverlapping result inputs')
        for name in ['probe_seed', 'probe_epochs', 'probe_lr', 'probe_batch_size', 'probe_weight_decay', 'feature_scaling']:
            if probe[name].nunique() != 1:
                raise ValueError(f'Inconsistent probe hyperparameters: {name}')
    if not cam.empty and cam.duplicated(['run_id', 'epoch', 'original_index']).any():
        raise ValueError('Overlapping Grad-CAM jobs')
    for frame in [probe, cam]:
        if not frame.empty:
            for run_id in frame['run_id'].unique():
                if run_id not in configs:
                    raise ValueError(f'No corresponding history for {run_id}')
                group = frame[frame.run_id == run_id]
                cfg = configs[run_id]
                if not (group['train_corr'] == cfg['train_corr']).all() or not (group['model'] == cfg['model']).all() or not (group['model_seed'] == cfg['seed']).all():
                    raise ValueError('Result config does not match selected history')
    history.to_csv(out / 'history_combined.csv', index=False)
    history_columns = ['run_id', 'epoch', 'val_acc', 'val_cf_flip_rate', 'val_conflicting_acc', 'val_aligned_acc']
    if not probe.empty:
        probe.to_csv(out / 'probe_combined.csv', index=False)
        joined = probe.merge(history[history_columns], on=['run_id', 'epoch'], how='left', validate='many_to_one')
        joined.to_csv(out / 'probe_history_joined.csv', index=False)
    if not cam.empty:
        cam.to_csv(out / 'gradcam_combined.csv', index=False)
        cam.merge(history[history_columns], on=['run_id', 'epoch'], how='left', validate='many_to_one').to_csv(out / 'gradcam_history_joined.csv', index=False)
    for column in ['val_acc', 'val_cf_flip_rate', 'val_conflicting_acc']:
        plot_lines(history, column, ['run_id'], out / f'{column}.png', column, column)
    report = ['# Shortcut Learning Dynamics 후속 분석', '',
              '원래 모델은 재학습하지 않았다. 기존 학습 로그는 전체 epoch를 사용하며, probe와 Grad-CAM은 실제 체크포인트가 있는 epoch만 분석한다. 누락 epoch를 보간하거나 결과를 생성하지 않는다.', '',
              '## 데이터 및 지표', '',
              '- 학습 코드상 기존 feature snapshot은 validation에서 각 (숫자, 색상) 그룹당 최대 32개씩 고정 추출한 pooled feature다. next pairing의 20개 그룹 × 32 = 640개이며 독립적 숫자/색상 probe용 데이터가 아니다. 실제 원본 NPZ가 없는 경우 이 분할 판단은 코드에 근거한 것이며 audit_followup.py --verify-datasets로 서버 파일을 대조할 수 있다.',
              '- 기존 코드의 next/random은 숫자마다 두 색상만 허용한다. correlation=0.1로도 독립적이지 않다. all pairing의 0.1은 생성 확률상 독립적이지만 표본상 완전 균형은 보장하지 않는다.',
              '- 새 protocol은 validation 원본으로 probe를 학습하고 공식 test 원본으로 평가한다. 각 숫자의 같은 수 원본을 10색 모두로 칠해 모든 숫자×색상 셀의 수를 정확히 맞춘다. 재색칠된 같은 원본이 probe 학습/평가 양쪽으로 나뉘지 않는다.',
              '- chance-adjusted accuracy = (accuracy − 0.1) / 0.9. 기존 호환 컬럼 legacy_decoded_correlation = 2 × accuracy − 1을 함께 보존한다. 둘 다 Pearson correlation이 아니다.',
              '- probe는 원래 feature scale을 그대로 사용한 AdamW linear classifier다. 해독 성능은 representation과 probe 최적화 조건에 관한 결과이며 인과적 의존도는 아니다.', '',
              '## 학습 로그와 H1', '']
    coverage, summaries = [], []
    for run_id, cfg in configs.items():
        h = history[history.run_id == run_id].sort_values('epoch')
        first_trained = h[h.epoch > 0].iloc[0]
        last = h.iloc[-1]
        increases = int((h[h.epoch > 0]['val_cf_flip_rate'].diff() > 1e-8).sum())
        report.append(f'- {run_id}: epoch {int(first_trained.epoch)} flip rate {first_trained.val_cf_flip_rate:.4f} → epoch {int(last.epoch)} {last.val_cf_flip_rate:.4f}. 학습된 epoch 사이 증가 구간은 {increases}개다. 초기→최종 감소와 단조 감소를 구분해야 하며, epoch 0의 낮은 flip rate는 무작위/상수 예측 때문에 생길 수 있다.')
        available = sorted(probe[probe.run_id == run_id].epoch.unique().tolist()) if not probe.empty else []
        cam_available = sorted(cam[cam.run_id == run_id].epoch.unique().tolist()) if not cam.empty else []
        coverage.append(dict(run_id=run_id, probe_epochs=available, gradcam_epochs=cam_available,
                             missing_probe_epochs=[e for e in EPOCHS if e not in available],
                             missing_gradcam_epochs=[e for e in EPOCHS if e not in cam_available]))
    report += ['', '## Layer별 probe 결과', '']
    if not probe.empty:
        for target in ['core', 'shortcut']:
            subset = probe[probe.target == target]
            plot_lines(subset, 'accuracy', ['run_id', 'layer'], out / f'{target}_accuracy_epochs.png', f'{target} probe accuracy', 'Accuracy')
            plot_lines(subset[subset.layer == 'avgpool'], 'chance_adjusted_accuracy', ['run_id'],
                       out / f'{target}_model_comparison.png', f'{target}: final feature model comparison', 'Chance-adjusted accuracy')
        for run_id, group in probe.groupby('run_id'):
            layers = list(dict.fromkeys(group['layer']))
            epochs = sorted(set(EPOCHS) | set(group.epoch))
            fig, axes = plt.subplots(1, 2, figsize=(14, max(4, .35 * len(layers))))
            cmap = plt.get_cmap('viridis').copy()
            cmap.set_bad('#dddddd')
            for ax, target in zip(axes, ['core', 'shortcut']):
                grid = group[group.target == target].pivot(index='layer', columns='epoch', values='chance_adjusted_accuracy').reindex(index=layers, columns=epochs)
                im = ax.imshow(np.ma.masked_invalid(grid.to_numpy()), aspect='auto', vmin=-1/9, vmax=1, cmap=cmap)
                ax.set_xticks(range(len(epochs)), labels=epochs)
                ax.set_yticks(range(len(layers)), labels=layers)
                ax.set(xlabel='Epoch; gray = missing', title=f'{run_id}: {target}')
                fig.colorbar(im, ax=ax, label='Chance-adjusted accuracy')
            fig.tight_layout()
            fig.savefig(out / f'{run_id}_layer_heatmaps.png', dpi=160)
            plt.close(fig)
            for layer, layer_group in group.groupby('layer'):
                core = layer_group[layer_group.target == 'core'].sort_values('epoch')
                shortcut = layer_group[layer_group.target == 'shortcut'].sort_values('epoch')
                reached = core[core.chance_adjusted_accuracy >= args.core_threshold]
                summaries.append(dict(run_id=run_id, layer=layer, observed_epochs=','.join(map(str, core.epoch)),
                                      threshold=args.core_threshold,
                                      first_observed_core_threshold_epoch=int(reached.iloc[0].epoch) if len(reached) else None))
                report.append(f'- {run_id} / {layer}: 관측 epoch {core.epoch.tolist()}; 최종 관측 숫자 accuracy {core.iloc[-1].accuracy:.4f}, 색상 accuracy {shortcut.iloc[-1].accuracy:.4f}.')
        pd.DataFrame(summaries).to_csv(out / 'observed_core_thresholds.csv', index=False)
        report += ['', '관측된 checkpoint에서 threshold에 처음 도달한 epoch는 실제 학습 시작 시점 추정치가 아니다. epoch 100만 있는 경우에는 언제 해독 성능이 증가했는지 판단할 수 없다. 단일 seed의 작은 원본 표본을 여러 색상으로 반복했으므로 각 재색칠 입력을 독립적인 원본 관측치로 취급하지 않는다.']
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, target in zip(axes, ['core', 'shortcut']):
            subset = joined[(joined.target == target) & (joined.layer == 'avgpool')]
            for run_id, g in subset.groupby('run_id'):
                ax.scatter(g['val_cf_flip_rate'], g['chance_adjusted_accuracy'], label=run_id)
                for _, r in g.iterrows():
                    ax.annotate(str(r['epoch']), (r.val_cf_flip_rate, r.chance_adjusted_accuracy), fontsize=7)
            ax.set(xlabel='Validation paired flip rate', ylabel=f'{target} adjusted accuracy')
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / 'decodability_vs_flip_rate.png', dpi=160)
        plt.close(fig)
    report += ['', '## Grad-CAM의 정량화와 해석', '',
               '- 두 모델/모든 epoch에서 같은 공식 test 원본에 정답 숫자를 고정 target으로 사용한다. aligned 색상 y와 conflicting 색상 (y+1) mod 10을 비교하며 숫자 모양은 동일하다.',
               '- cam_mean_absolute_difference = 평균 |A−B|. A와 B는 입력 해상도로 보간 후 각자 0–1 정규화한 heatmap이다.',
               '- cam_cosine_similarity = sum(A×B)/(||A||₂||B||₂). 0 heatmap이면 정의 불가로 빈 값 처리한다.',
               '- foreground_mass = sum(CAM × mask)/sum(CAM). mask는 원본 회색조를 입력 크기로 bilinear resize한 뒤 밝기 > 0.1인 영역이다. 비교 기준 면적 foreground_area_fraction도 저장한다. 0 heatmap이면 빈 값 처리한다.',
               '- paired prediction flip과 정답 확률을 함께 기록한다. 이 소수 고정 표본의 flip은 전체 test flip rate와 다르다.',
               '- foreground 색상과 숫자 모양이 같은 픽셀에 있으므로 foreground mass로 색상/모양 의존도를 분리할 수 없다. 최종층의 낮은 공간 해상도와 표본별 정규화도 고려해야 한다. Grad-CAM 수치 및 그림은 직접적인 인과적 증거가 아니다.']
    if not cam.empty:
        aggregate = cam.groupby(['run_id', 'model', 'epoch'], as_index=False).agg(
            pair_count=('original_index', 'count'), cam_mean_absolute_difference=('cam_mean_absolute_difference', 'mean'),
            cam_cosine_similarity=('cam_cosine_similarity', 'mean'), pair_flip_rate=('pair_prediction_flip', 'mean'),
            aligned_foreground_mass=('aligned_foreground_mass', 'mean'), conflicting_foreground_mass=('conflicting_foreground_mass', 'mean'))
        aggregate.to_csv(out / 'gradcam_epoch_summary.csv', index=False)
        plot_lines(aggregate, 'cam_mean_absolute_difference', ['run_id'], out / 'gradcam_pair_change.png', 'Counterfactual normalized CAM difference', 'Mean absolute difference')
    report += ['', '## 통합 질문에 대한 현재 판단', '',
               '- H1: 위 학습 로그에서 초기 학습 epoch 대비 최종 paired flip rate와 중간 변동을 확인할 수 있다. probe의 정보 보존과 예측 민감도 감소는 별개다.',
               '- Core/shortcut decodability가 언제 변했는지는 여러 epoch의 checkpoint probe가 필요하다. 색상 정보가 최종 feature에서 해독된다는 사실을 shortcut 의존성 증가로 해석하지 않는다.',
               '- 두 모델 차이는 같은 protocol/epoch/layer와 학습 조건에서 비교해야 한다. 서로 다른 아키텍처의 중간 stage는 구조적으로 완전히 대응하지 않는다.',
               '- H2: correlation 0.80/0.65 결과가 확보된 뒤 같은 protocol로 검증한다. 현재 0.95 결과만으로 강한 correlation의 학습 지연을 결론 내리지 않는다.', '',
               '## 실행 범위', '', '```json', json.dumps(coverage, ensure_ascii=False, indent=2), '```', '',
               '기존 epoch100 분석은 상관된 val/test 원본 분포에서 수행한 별도 protocol이므로 이번 독립 probe 수치와 같은 곡선에 직접 연결하지 않는다. 새 probe의 10색 중 8색은 next pairing 학습에서 해당 숫자에 관측되지 않았던 조합이다. 원래 paired flip rate와 새 probe의 평가 분포는 다르므로 두 지표의 직접적인 수치 대응을 가정하지 않는다. 레이어 차원·feature scale·probe 수렴 정도도 해독 성능 차이에 영향을 줄 수 있다.']
    (out / 'report_ko.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    write_json(out / 'coverage.json', coverage)
    write_json(out / 'status.json', dict(status='complete', probe_rows=len(probe), gradcam_pairs=len(cam)))
    logging.info('Integrated report saved: %s', out)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.exception('Integration failed')
        raise
