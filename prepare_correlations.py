"""Write configs and shell commands ONLY; never launch training."""
import argparse
import json
from pathlib import Path
import shlex

from followup_common import start_output, write_json


def training_command(cfg, output):
    command = ['python', 'shortcut_experiment.py', 'train']
    for key in ['dataset', 'model', 'epochs', 'seed', 'image_size', 'train_corr', 'val_corr',
                'test_corr', 'mnist_val_size', 'color_mode', 'color_pairing', 'lr', 'momentum',
                'scheduler', 'weight_decay', 'batch_size', 'num_workers', 'features_per_group']:
        command += ['--' + key.replace('_', '-'), str(cfg[key])]
    for key, flag in [('pretrained', '--pretrained'), ('augment', '--augment')]:
        if cfg.get(key):
            command.append(flag)
    if not cfg.get('amp', True):
        command.append('--no-amp')
    if not cfg.get('save_features', True):
        command.append('--no-features')
    return command + ['--output', Path(output).as_posix()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True, help='New config/command directory')
    p.add_argument('--run-root', type=Path, default=Path('runs/followup_corr'))
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--correlations', type=float, nargs='+', default=[.80, .65])
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**32 for s in args.seeds):
        p.error('seeds must be unique uint32 values')
    if not args.correlations or any(c not in [.80, .65] for c in args.correlations):
        p.error('This follow-up prepares only correlation 0.80 and 0.65')
    out = start_output(args.output, args)
    commands, plans, planned_paths = [], [], set()
    for run in args.reference_runs:
        cfg = json.loads((run / 'config.json').read_text())
        if cfg['dataset'] != 'colored_mnist' or cfg['epochs'] != 100 or cfg['train_corr'] != .95:
            raise ValueError('Reference must be the completed 100-epoch corr=0.95 Colored MNIST run')
        if cfg['optimizer'] != 'sgd':
            raise ValueError('Unsupported reference optimizer')
        for corr in args.correlations:
            for seed in args.seeds:
                planned = {**cfg, 'train_corr': corr, 'seed': seed}
                name = f'{cfg["model"]}_corr{round(corr * 100):03d}_seed{seed}_e100'
                run_output = args.run_root / name
                if run_output.exists() or run_output in planned_paths:
                    raise FileExistsError(f'Training target exists or is duplicated: {run_output}')
                planned_paths.add(run_output)
                write_json(out / f'{name}.json', planned)
                command = training_command(planned, run_output)
                commands.append(command)
                plans.append(dict(reference_run=str(run), config=planned, output=str(run_output), command=command))
    (out / 'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n\n' +
                                      '\n'.join(shlex.join(c) for c in commands) + '\n', encoding='utf-8')
    def ps_quote(s):
        return "'" + s.replace("'", "''") + "'"
    (out / 'commands.ps1').write_text("$ErrorActionPreference = 'Stop'\n" +
        '\n'.join('& ' + ' '.join(ps_quote(s) for s in c) +
                  '\nif ($LASTEXITCODE -ne 0) { throw "Training failed; stopped remaining jobs" }' for c in commands) + '\n', encoding='utf-8')
    write_json(out / 'plan.json', dict(training_executed=False, experiments=plans,
                                      note='Commands preserve every reference training option except train_corr/seed/output. Run manually from repository root.'))
    print(f'Prepared {len(plans)} training commands. NO training was launched: {out}')


if __name__ == '__main__':
    main()
