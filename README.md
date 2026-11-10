# Shortcut learning dynamics: VGG/ResNet × Colored MNIST/Waterbirds

PyTorch + Hugging Face Datasets experiment. Requirements: Python 3.10+, GPU recommended for Waterbirds.

## 최신 분석 결과: Colored MNIST 후속406 Checkpoint

[회의용 요약](reports/metric_final_analysis_406/meeting_summary.md) · [전체 한국어 보고서](reports/metric_final_analysis_406/analysis_report.md) · [비교 표와 그래프](reports/metric_final_analysis_406/README.md)

10-class Colored MNIST × VGG11_BN의 C2I·Probe·Grad-CAM 분석입니다. Corr .95는Resume100 Epoch, .80/.65는50 Epoch입니다. 아래의 기존2-class 실행 예제와 별개이며, 이번 변경은 학습 코드 업데이트를 포함하지 않습니다. [공유 범위와 재현성](reports/metric_final_analysis_406/PUBLICATION_NOTES.md)을 확인해 주세요.

## Install

```bash
pip install -r requirements.txt
```

Use a PyTorch+TorchVision version combination appropriate for your system/CUDA. The runner uses `torch.amp.GradScaler` and `torch.load(weights_only=True)`.

## Four experiments

```bash
# Colored MNIST: target=0 for digits 0-4; target=1 for digits 5-9
# shortcut=red(0)/green(1); train P(shortcut==target)=0.95
python shortcut_experiment.py train --dataset colored_mnist --model resnet18 --epochs 20 --output runs/cm_resnet
python shortcut_experiment.py train --dataset colored_mnist --model vgg11_bn --epochs 20 --output runs/cm_vgg

# Waterbirds: target=landbird(0)/waterbird(1); shortcut=land(0)/water(1)
# pretrained weights are highly recommended for this smaller dataset
python shortcut_experiment.py train --dataset waterbirds --model resnet18 --pretrained --epochs 20 --batch-size 32 --output runs/wb_resnet
python shortcut_experiment.py train --dataset waterbirds --model vgg11_bn --pretrained --epochs 20 --batch-size 32 --output runs/wb_vgg
```

Default image size: 64 for Colored MNIST and 128 for Waterbirds (can override with `--image-size 224` for Waterbirds). Defaults: SGD(momentum=0.9), LR 0.01, cosine scheduler, seed 42, training from scratch unless `--pretrained`. ImageNet normalization is used for both datasets. VGG uses **VGG-11 with BatchNorm + global average pooling + a 2-way linear head** (not the original 4096-unit classifier); ResNet uses ResNet-18 with a 2-way linear head. Run multiple seeds for scientific conclusions. For close architecture comparison, fix the same dataset split, preprocessing, seed list, training budget, and pretrained condition.

### Resume from any epoch (optimizer + scheduler + AMP state included)

```bash
python shortcut_experiment.py train --dataset colored_mnist --model resnet18 --output runs/cm_resnet --resume latest
# or --resume runs/cm_resnet/checkpoints/epoch_005.pt
```

**Important:** Resume restores the saved configuration, including total epochs. Keep the same `--output` run directory and avoid starting a fresh run into a directory containing old history. For an *earlier* checkpoint when newer checkpoints exist, first copy/fork the run into a clean directory and remove newer state there; the runner intentionally blocks overwriting newer saved epochs. It also rejects starting a fresh run into an existing history directory. Checkpoints are saved at the *end* of each epoch and at epoch 0 (untrained). On resume, latest.json points to the last committed checkpoint. A history row left by an interrupted checkpoint write is automatically discarded on resume.

### Evaluate a checkpoint on test

```bash
python shortcut_experiment.py eval --run runs/cm_resnet --checkpoint runs/cm_resnet/checkpoints/epoch_005.pt --split test
python shortcut_experiment.py eval --run runs/wb_resnet --checkpoint latest --split test
```

### Hidden-state features

```bash
python shortcut_experiment.py features --run runs/cm_resnet --checkpoint latest --split test --features-per-group 64 --output runs/cm_resnet/test_features.npz
python analyze.py --run runs/cm_resnet --epoch 5 --layer layer3
python analyze.py --run runs/cm_vgg --epoch 5 --layer features.14
```

Model hooks collect an average-pooled vector for each stage, with shape `[N, channels]` rather than storing huge spatial activation maps. `np.load('...npz')` keys: `layers` (module names), `labels`, `spurious`, `indices`, and `f0`, `f1`, ..., each with shape `[N, channels]`. Stage names for ResNet: `layer1, layer2, layer3, layer4, avgpool`; VGG names are generated from `MaxPool2d` stage indices + `avgpool`. To collect unpooled `[N,C,H,W]` maps, edit `FeatureHookCollector.hook` to omit `x.mean(dim=(2,3))`, and ensure adequate RAM/storage.

## Artifacts

```
runs/<name>/
  config.json
  latest.json
  history.csv                # train + validation by epoch, includes 0
  test_metrics.json          # final epoch only
  checkpoints/epoch_000.pt   # baseline untrained
  checkpoints/epoch_001.pt   # model, optimizer, schedule, RNG states
  ...
  features/epoch_000.npz     # fixed balanced val subset for hidden-state trends
  features/epoch_001.npz
  ...
  learning_curves.png        # created by analyze.py
  hidden_pca_epoch_005.png   # created by analyze.py --epoch 5
```

## Metrics and interpretation

* `val_acc`: standard classification accuracy.
* `val_g00_acc`, `val_g01_acc`, `val_g10_acc`, `val_g11_acc`: accuracy conditioned on target **y** and spurious attribute **s**. `val_gXX_n`: counts.
* `val_wga`: minimum accuracy across **all four groups**; NaN if a group has zero examples.
* `val_aligned_acc`, `val_conflicting_acc`: accuracy when `y==s` vs `y!=s`. A growing gap suggests shortcut dependence, not definitive proof by itself.
* Colored MNIST only: `val_cf_flip_rate` is fraction of samples for which the prediction changes when **the same digit** is recolored red ↔ green. `val_cf_acc` accuracy after the swap; `val_cf_acc_drop` original accuracy minus swapped accuracy (the drop may be negative). **Flip rate is the most direct intervention-based reliance metric here.** Waterbirds has no naive analogous intervention: it would need a genuine same-bird alternate-background pair to avoid confounds.
* Feature snapshot and PCA: visualization of what features are *represented*. Decodability/separation is **not evidence** by itself that the classifier uses a feature causally.

## Dataset protocol and caution

* `ylecun/mnist`: split the official train into 55,000 train and 5,000 validation via a fixed seed; use the official 10,000 test images. The color assignment is *fixed per example* across epochs, not resampled per access. `--train-corr .95`, `--val-corr .50`, `--test-corr .10` are independent Bernoulli assignments, so actual sample proportions may differ slightly. Test is deliberately *anti-correlated* (distribution shift). This is **one binary Colored MNIST construction**, not a claim to reproduce every paper's Colored MNIST protocol.
* `grodino/waterbirds`: Hugging Face train/validation/test splits, label = landbird(0)/waterbird(1), place = land(0)/water(1), so Waterbirds `P(place==label)` is mostly high in train and more balanced in validation/test. The runner uses the dataset's actual metadata labels and group splits rather than resampling background correlations.
* To avoid selecting a model on the test set, validation is evaluated each epoch and the test set **once at the final epoch**. You can separately load earlier checkpoints to evaluate them for a retrospective analysis, but do not tune hyperparameters using those test outcomes.
* For Waterbirds, foreground/background are composite natural images. Worst-group accuracy can be noisy for small groups. Cross-seed statistics/uncertainty intervals strengthen conclusions.
* The pipeline does not do exact bitwise determinism across GPU/kernel or data-loader changes. RNG states are stored, but exact restart reproducibility can require matching hardware/library versions and `num_workers`, `--augment`, deterministic kernels.
* HF dataset download and pretrained weight download require internet on first use. Respect third-party dataset and weight licensing for redistribution.

### Raw spatial activation maps (one sample, not pooled)

```bash
python inspect_hidden.py --run runs/cm_resnet --checkpoint runs/cm_resnet/checkpoints/epoch_005.pt --split val --sample-index 0 --layer layer3 --channels 16
python inspect_hidden.py --run runs/cm_vgg --checkpoint latest --layer features.14 --channels 16
python inspect_hidden.py --run runs/cm_vgg --list-layers
```

This generates a full, **unpooled** activation tensor in `raw_epoch_....npz` (key `hidden`, shape `[C,H,W]`) plus a grid image `raw_epoch_....png`. Use this for the actual middle feature maps; the snapshots in `features/` are spatially pooled to permit compact per-epoch storage.
