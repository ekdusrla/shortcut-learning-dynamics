"""Checkpoint correctness verification for shortcut_experiment.py.

Checks the three things assigned under "model checkpoint 구현 (정환)":
  1. Random-seed reproducibility of model initialization.
  2. Model weights round-trip unchanged through save_checkpoint -> load_checkpoint.
  3. Bit-exact equivalence between:
       (a) training straight through epoch 1 -> epoch 2, no interruption
       (b) training epoch 1, saving a checkpoint, reloading everything into
           brand-new model/optimizer/scheduler/scaler/generator objects
           (simulating a fresh process), then continuing to epoch 2
     If (a) and (b) end up with identical weights, resume is truly lossless.

Needs no internet and no real dataset -- uses a tiny synthetic dataset that
matches the {pixel_values, labels, spurious, index} contract the real
ColoredMNIST/Waterbirds datasets use. Run this in the team's actual
environment (Colab / school GPU), where torch + torchvision are already
installed, next to a checkout of the repo:

    python verify_checkpoint.py --code-dir /path/to/shortcut_learning_experiment
    python verify_checkpoint.py --code-dir /path/to/repo --model vgg11_bn
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TinySyntheticDataset(Dataset):
    """Matches the real datasets' __getitem__ contract without needing internet."""

    def __init__(self, n, image_size, seed):
        rng = np.random.default_rng(seed)
        self.images = rng.random((n, 3, image_size, image_size)).astype("float32")
        self.labels = rng.integers(0, 2, size=n)
        self.spurious = rng.integers(0, 2, size=n)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "pixel_values": torch.from_numpy(self.images[i]),
            "labels": int(self.labels[i]),
            "spurious": int(self.spurious[i]),
            "index": int(i),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code-dir", required=True,
                        help="directory containing shortcut_experiment.py")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "vgg11_bn"])
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--n", type=int, default=32, help="synthetic sample count")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.code_dir).resolve()))
    import shortcut_experiment as se  # noqa: E402

    device = torch.device("cpu")
    cfg = {"model": args.model, "pretrained": False, "lr": 0.01, "weight_decay": 1e-4,
           "epochs": 2, "seed": args.seed, "batch_size": 8, "num_workers": 0, "amp": False}

    results = {}

    # ---- 1. seed reproducibility of model init ----
    se.seed_everything(args.seed)
    model_a = se.build_model(cfg)
    se.seed_everything(args.seed)
    model_b = se.build_model(cfg)
    results["same_seed_init_identical"] = all(
        torch.equal(pa, pb) for pa, pb in
        zip(model_a.state_dict().values(), model_b.state_dict().values()))
    print(f"[1] Same-seed model init identical: {results['same_seed_init_identical']}")

    # ---- setup for checkpoint round-trip ----
    ds = TinySyntheticDataset(args.n, args.image_size, seed=0)

    def fresh_training_state():
        se.seed_everything(args.seed)
        model = se.build_model(cfg).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9,
                                    weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        gen = torch.Generator().manual_seed(cfg["seed"] + 777)
        loader = se.create_loader(ds, cfg["batch_size"], True, cfg["num_workers"], gen, device)
        return model, optimizer, scheduler, scaler, gen, loader

    # (a) continuous: epoch 1 then epoch 2, no checkpoint involved
    model, optimizer, scheduler, scaler, gen, loader = fresh_training_state()
    se.train_one_epoch(model, loader, optimizer, scaler, device, amp=False)
    scheduler.step()
    se.train_one_epoch(model, loader, optimizer, scaler, device, amp=False)
    continuous_weights = {k: v.clone() for k, v in model.state_dict().items()}

    # (b) checkpoint-interrupted: epoch 1, save, reload into brand-new objects, epoch 2
    model2, optimizer2, scheduler2, scaler2, gen2, loader2 = fresh_training_state()
    se.train_one_epoch(model2, loader2, optimizer2, scaler2, device, amp=False)
    scheduler2.step()
    ckpt_path = Path(args.code_dir) / "_verify_tmp" / "epoch_001.pt"
    se.save_checkpoint(ckpt_path, 1, model2, optimizer2, scheduler2, scaler2, cfg, gen2)

    # Simulate a brand-new process: fresh objects, nothing carried over except the file.
    saved = se.load_checkpoint(ckpt_path, device)
    model3 = se.build_model(cfg).to(device)
    optimizer3 = torch.optim.SGD(model3.parameters(), lr=cfg["lr"], momentum=0.9,
                                 weight_decay=cfg["weight_decay"])
    scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=cfg["epochs"])
    scaler3 = torch.amp.GradScaler("cuda", enabled=False)
    gen3 = torch.Generator().manual_seed(999999)  # deliberately WRONG seed before restore
    model3.load_state_dict(saved["model"])
    optimizer3.load_state_dict(saved["optimizer"])
    scheduler3.load_state_dict(saved["scheduler"])
    scaler3.load_state_dict(saved["scaler"])
    torch.set_rng_state(saved["torch_rng"].cpu())
    gen3.set_state(saved["loader_rng"].cpu())
    loader3 = se.create_loader(ds, cfg["batch_size"], True, cfg["num_workers"], gen3, device)

    results["weights_after_load_match"] = all(
        torch.equal(a, b) for a, b in zip(model2.state_dict().values(), model3.state_dict().values()))
    print(f"[2] Model weights identical right after save -> load: {results['weights_after_load_match']}")

    se.train_one_epoch(model3, loader3, optimizer3, scaler3, device, amp=False)
    resumed_weights = model3.state_dict()

    results["resumed_matches_continuous"] = all(
        torch.equal(continuous_weights[k], resumed_weights[k]) for k in continuous_weights)
    print(f"[3] Resumed epoch-2 weights == continuously-trained epoch-2 weights: "
          f"{results['resumed_matches_continuous']}")
    if not results["resumed_matches_continuous"]:
        max_diff = max((continuous_weights[k] - resumed_weights[k]).abs().max().item()
                       for k in continuous_weights)
        print(f"    Max abs diff: {max_diff}")

    shutil.rmtree(Path(args.code_dir) / "_verify_tmp", ignore_errors=True)

    print()
    if all(results.values()):
        print("All checks passed: seeding is reproducible and checkpoint "
              "save/resume is bit-exact lossless (CPU, single-process DataLoader).")
    else:
        print("Some checks FAILED -- see output above.")
    print("\nNote: this run used num_workers=0 and CPU, so it does not exercise "
          "torch.cuda RNG state or multi-worker DataLoader seeding. On the real "
          "GPU run, re-check resume with --num-workers matching the real config, "
          "since the README already flags multi-worker/GPU exact-determinism as "
          "not guaranteed bit-for-bit.")


if __name__ == "__main__":
    main()
