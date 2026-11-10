"""Fixed true-class Grad-CAM on same-original aligned/counterfactual pairs."""
import csv
import logging

from followup_common import (CrossedColors, common_args, inventory, load_model, load_raw,
                             read_manifest, setup, sha256, start_output, write_json)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from gradcam import GradCAM, default_layer, unnormalize_image


def compare_maps(aligned, conflicting, mask):
    a, b = aligned.astype(float), conflicting.astype(float)
    def mass(cam):
        return float(cam[mask].sum() / cam.sum()) if cam.sum() > 0 else None
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dict(cam_mean_absolute_difference=float(np.abs(a - b).mean()),
                cam_cosine_similarity=float((a * b).sum() / norm) if norm > 0 else None,
                aligned_foreground_mass=mass(a), conflicting_foreground_mass=mass(b),
                foreground_area_fraction=float(mask.mean()),
                aligned_zero_cam=bool(a.max() == 0), conflicting_zero_cam=bool(b.max() == 0))


def pair_grid(items, output, title):
    fig, axes = plt.subplots(len(items), 4, figsize=(12, 2.8 * len(items)), squeeze=False)
    for axes_row, (label, a_img, a_cam, b_img, b_cam) in zip(axes, items):
        values = [a_img, np.clip(.55 * a_img + .45 * plt.get_cmap('jet')(a_cam)[..., :3], 0, 1),
                  b_img, np.clip(.55 * b_img + .45 * plt.get_cmap('jet')(b_cam)[..., :3], 0, 1)]
        for ax, pixels, kind in zip(axes_row, values, ['Aligned input', 'Aligned CAM', 'CF input', 'CF CAM']):
            ax.imshow(pixels)
            ax.axis('off')
            ax.set_title(f'{label}\n{kind}', fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main():
    parser = common_args(__doc__)
    parser.add_argument('--samples-per-digit', type=int, default=1)
    args = parser.parse_args()
    if args.samples_per_digit < 1:
        parser.error('samples-per-digit must be positive')
    device = setup(args)
    out = start_output(args.output, args)
    try:
        records = inventory(args, out)
        manifest = read_manifest(args.manifest)
        if args.samples_per_digit > manifest['per_digit']:
            raise ValueError('Requested more Grad-CAM originals than protocol contains')
        write_json(out / 'protocol.json', manifest)
        protocol_sha = sha256(args.manifest)
        raw = load_raw()
        dataset = CrossedColors(raw, manifest, 'test')
        selected = [i for i, r in enumerate(dataset.records)
                    if i % manifest['per_digit'] < args.samples_per_digit]
        rows, model_groups, epoch_groups = [], {}, {}
        for record in records:
            model, cfg, checkpoint = load_model(record, manifest, device)
            layer = default_layer(model, cfg['model'])
            checkpoint_sha = sha256(checkpoint)
            dest = out / record['run_id'] / f'epoch_{record["epoch"]:03d}'
            dest.mkdir(parents=True)
            gallery = []
            logging.info('Grad-CAM %s epoch %d layer %s', record['run_id'], record['epoch'], layer)
            with GradCAM(model, layer) as gradcam:
                for original_pos in selected:
                    sample = dataset.records[original_pos]
                    y = sample['label']
                    # Fixed intervention y -> (y+1)%10, identical across all models and epochs.
                    colors = [y, (y + 1) % 10]
                    tensors = torch.stack([dataset[original_pos * 10 + s]['pixel_values'] for s in colors])
                    logits, maps = gradcam(tensors.to(device), class_id=y)
                    images = [unnormalize_image(t) for t in tensors]
                    cams = maps.numpy()
                    if not np.isfinite(cams).all():
                        raise ValueError('Non-finite Grad-CAM')
                    gray = raw['test'][sample['index']]['image'].convert('L')
                    mask = np.asarray(gray.resize((manifest['config']['image_size'],) * 2,
                                                 Image.Resampling.BILINEAR)) / 255 > .1
                    stem = f'y{y}_index_{sample["index"]:05d}'
                    probs = logits.softmax(1).numpy()
                    metadata = dict(run_id=record['run_id'], model=cfg['model'], epoch=record['epoch'],
                                    original_index=sample['index'], target_class=y, aligned_color=colors[0],
                                    conflicting_color=colors[1], layer=layer,
                                    aligned_prediction=int(logits[0].argmax()), conflicting_prediction=int(logits[1].argmax()),
                                    aligned_target_probability=float(probs[0, y]), conflicting_target_probability=float(probs[1, y]),
                                    pair_prediction_flip=bool(logits[0].argmax() != logits[1].argmax()),
                                    train_corr=cfg['train_corr'], model_seed=cfg['seed'],
                                    protocol_sha256=protocol_sha, checkpoint_sha256=checkpoint_sha,
                                    **compare_maps(cams[0], cams[1], mask))
                    rows.append(metadata)
                    write_json(dest / f'{stem}.json', metadata)
                    np.savez_compressed(dest / f'{stem}.npz', aligned_cam=cams[0], conflicting_cam=cams[1],
                                        foreground_mask=mask, probabilities=probs)
                    for kind, image, cam in zip(['aligned', 'conflicting'], images, cams):
                        plt.imsave(dest / f'{stem}_{kind}_heatmap.png', cam, cmap='jet', vmin=0, vmax=1)
                        overlay = np.clip(.55 * image + .45 * plt.get_cmap('jet')(cam)[..., :3], 0, 1)
                        plt.imsave(dest / f'{stem}_{kind}_overlay.png', overlay)
                        plt.imsave(dest / f'{stem}_{kind}_input.png', image)
                    item = (f'y={y} id={sample["index"]}', images[0], cams[0], images[1], cams[1])
                    gallery.append(item)
                    key = (record['run_id'], sample['index'])
                    epoch_groups.setdefault(key, []).append((record['epoch'], item))
                    model_label = f'{cfg["model"]}\ncorr={cfg["train_corr"]} seed={cfg["seed"]}'
                    model_groups.setdefault((record['epoch'], sample['index']), []).append((model_label, item))
            pair_grid(gallery, dest / 'aligned_conflicting_grid.png',
                      f'{record["run_id"]} epoch {record["epoch"]}, {layer}; target=true digit')
            write_json(dest / 'checkpoint.json', dict(checkpoint=str(checkpoint), config=cfg,
                                                       checkpoint_sha256=checkpoint_sha))
            with (out / 'gradcam_metrics.csv').open('w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            del model
        grid_dir = out / 'comparison_grids'
        grid_dir.mkdir()
        for (run, index), entries in epoch_groups.items():
            items = [(f'epoch {epoch}\n{item[0]}', *item[1:]) for epoch, item in sorted(entries)]
            pair_grid(items, grid_dir / f'{run}_index_{index}_epochs.png', 'Same original, same target class across epochs')
        for (epoch, index), entries in model_groups.items():
            items = [(f'{run}\n{item[0]}', *item[1:]) for run, item in entries]
            pair_grid(items, grid_dir / f'epoch_{epoch:03d}_index_{index}_models.png', 'Same original and recolor across models')
        write_json(out / 'status.json', dict(status='partial' if len(records) < len(args.runs) * len(args.epochs) else 'complete',
                                             pairs=len(rows), available_checkpoints=len(records)))
    except Exception as error:
        logging.exception('Grad-CAM analysis failed')
        write_json(out / 'status.json', dict(status='failed', error=str(error)))
        raise


if __name__ == '__main__':
    main()
