"""Manual GPU validation only: no training, checkpoints, or run output changes."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import shortcut_experiment as exp


def tree_rss():
    """Sample Linux parent + descendant RSS; shared pages may be counted twice."""
    processes = {}
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            fields = {}
            for line in (directory / 'status').read_text().splitlines():
                if line.startswith(('PPid:', 'VmRSS:')):
                    key, value = line.split(':', 1)
                    fields[key] = int(value.split()[0])
            processes[int(directory.name)] = (fields.get('PPid', -1), fields.get('VmRSS', 0))
        except (OSError, ValueError):
            continue
    selected = {os.getpid()}
    while True:
        expanded = selected | {pid for pid, (parent, _) in processes.items() if parent in selected}
        if expanded == selected:
            break
        selected = expanded
    return sum(processes.get(pid, (0, 0))[1] for pid in selected) * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--color-mode', choices=['background', 'foreground'], default='background')
    parser.add_argument('--train-corr', type=float, choices=[.95, .80, .65], default=.95)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Exclusive creation protects previous validation reports.
    with args.output.open('x') as report:
        result = {'success': False, 'host': socket.gethostname(), 'cpu_tree_rss_peak_bytes': 0}
        stop = threading.Event()
        def sample():
            while not stop.is_set():
                result['cpu_tree_rss_peak_bytes'] = max(result['cpu_tree_rss_peak_bytes'], tree_rss())
                stop.wait(.2)
        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
        try:
            assert result['host'].split('.')[0] == 'moana-u2'
            assert torch.cuda.is_available() and torch.cuda.device_count() == 1
            device = torch.device('cuda')
            properties = torch.cuda.get_device_properties(0)
            assert 'A2000' in properties.name
            result.update(torch=torch.__version__, cuda=torch.version.cuda,
                          gpu=properties.name, vram_bytes=properties.total_memory)
            cfg = dict(dataset='colored_mnist', model='vgg11_bn', pretrained=False,
                       seed=42, image_size=224, train_corr=args.train_corr, val_corr=.10,
                       test_corr=.10, mnist_val_size=5000, color_mode=args.color_mode,
                       color_pairing='all', amp=True, batch_size=128, num_workers=4,
                       features_per_group=32)
            result['config'] = cfg
            exp.seed_everything(42)
            datasets = exp.build_datasets(cfg)
            model = exp.build_model(cfg).to(device).eval()
            loader = exp.create_loader(datasets['val'], 128, False, 4, device=device)
            # Exact epoch-0 metric path, including all 5000 validation samples,
            # single-counterfactual and all nine alternative colors.
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            metrics = exp.evaluate(model, loader, device, amp=True, cf_chunk_size=128)
            torch.cuda.synchronize()
            result['validation'] = dict(metrics=metrics, seconds=time.monotonic()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                samples=len(datasets['val']), all_counterfactual_images=len(datasets['val'])*9)
            print(json.dumps(result['validation']), flush=True)
            # Safe comparison: only 8 source images x 9 colors = 72 images.
            # Never forward the full 1152-image counterfactual batch.
            batch = next(iter(loader))
            images = batch['cf_all_pixel_values'][:8].flatten(0, 1)
            assert len(images) == 72 and len(images) <= 128
            with torch.inference_mode():
                with exp.autocast_context(device, True):
                    full_logits = model(images.to(device)).float().cpu()
                    small_logits = torch.cat([
                        model(chunk.to(device)).float().cpu() for chunk in images.split(17)])
                actual = exp.counterfactual_predictions(model, images, device, True, 17).cpu()
            production = exp.counterfactual_predictions(model, images, device, True, 128).cpu()
            reference = full_logits.argmax(1)
            assert torch.equal(reference, production)
            result['comparison'] = dict(images=len(images), comparison_chunk_size=17,
                production_chunk_size=128,
                max_abs_logit_difference=(full_logits-small_logits).abs().max().item(),
                prediction_mismatches=(reference != actual).sum().item(),
                logits_close=torch.allclose(full_logits, small_logits, atol=.005, rtol=.005))
            # Tolerances assess AMP kernel roundoff, not a change to evaluation metrics.
            torch.testing.assert_close(full_logits, small_logits, atol=.005, rtol=.005)
            assert torch.equal(reference, actual), 'AMP argmax mismatch; inspect near-tied logits'
            # Epoch 0 also performs FP32 feature extraction: exercise it without
            # writing snapshots by using a temporary directory removed on exit.
            import tempfile
            with tempfile.TemporaryDirectory(prefix='cmnist-gpu-check-') as temporary:
                torch.cuda.reset_peak_memory_stats()
                exp.save_feature_snapshot(model, datasets['val'], 'vgg11_bn', device,
                    128, 32, Path(temporary)/'epoch_000.npz', 4, 42)
                torch.cuda.synchronize()
                result['features'] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                         peak_reserved_bytes=torch.cuda.max_memory_reserved())
            result['success'] = True
        except Exception:
            result['error'] = traceback.format_exc()
            if torch.cuda.is_available():
                result['cuda_memory_summary'] = torch.cuda.memory_summary()
            raise
        finally:
            stop.set()
            monitor.join()
            json.dump(result, report, indent=2)
            report.write('\n')
            print('Report:', args.output, 'success:', result['success'], flush=True)

if __name__ == '__main__':
    main()
