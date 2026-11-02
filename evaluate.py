"""Evaluate a trained model and its core/spurious linear probes.

Requires shortcut_experiment.py with build_model/build_datasets.
Datasets must provide train, val, test, probe; each batch must have
pixel_values, labels (core), and spurious.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DecodedCorrelation(nn.Module):
    """A linear probe for one target: core OR spurious labels.

    Input is a frozen model's last hidden state, not an image.
    Use different instances for core and spurious classification.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        if input_dim < 1 or output_dim < 2:
            raise ValueError("Invalid probe input/output dimension")
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CrossEntropyLoss expects logits, not softmax probabilities.
        return self.linear(x)

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        if epochs < 1 or batch_size < 1 or lr <= 0:
            raise ValueError("Invalid probe training hyperparameters")
        if weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        if features.ndim != 2:
            raise ValueError("Expected features with shape [N, D]")
        if features.size(1) != self.linear.in_features:
            raise ValueError("Hidden dimension mismatch")
        if len(features) == 0 or len(features) != len(labels):
            raise ValueError("Features and labels must match")

        labels = labels.long().reshape(-1).cpu()
        classes = torch.unique(labels)
        if not torch.equal(
            classes,
            torch.arange(self.linear.out_features),
        ):
            raise ValueError(
                "Probe labels must cover every class 0..N-1"
            )

        # Store cached hidden states on CPU; train just the probe.
        dataset = TensorDataset(features.detach().float().cpu(),
                                labels)
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True)
        device = next(self.parameters()).device
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        self.train()
        for _ in range(epochs):
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self(x), y)
                loss.backward()
                optimizer.step()
        self.eval()

    @torch.no_grad()
    def evaluate(
        self, features: torch.Tensor, labels: torch.Tensor,
        batch_size: int = 256,
    ) -> dict:
        if len(features) == 0 or len(features) != len(labels):
            raise ValueError("Features and labels must match")
        device = next(self.parameters()).device
        self.eval()
        correct = 0
        for start in range(0, len(labels), batch_size):
            x = features[start:start + batch_size].to(device)
            y = labels[start:start + batch_size].to(device)
            if y.min().item() < 0 or (
                y.max().item() >= self.linear.out_features
            ):
                raise ValueError("Unknown class in test labels")
            pred = self(x).argmax(dim=1)
            correct += (pred == y).sum().item()

        accuracy = correct / len(labels)
        # Paper's signed-label agreement: P(correct)-P(incorrect).
        # For N classes, random-guess baseline is 2/N - 1.
        return {
            "accuracy": accuracy,
            "decoded_correlation": 2 * accuracy - 1,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "input_dim": self.linear.in_features,
            "output_dim": self.linear.out_features,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(
        cls, path: str | Path, device: str = "cpu"
    ) -> "DecodedCorrelation":
        state = torch.load(
            path, map_location="cpu", weights_only=True
        )
        probe = cls(state["input_dim"], state["output_dim"])
        probe.load_state_dict(state["state_dict"])
        return probe.to(device).eval()

    @staticmethod
    @torch.no_grad()
    def encode(
        model: nn.Module, loader: DataLoader,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict]:
        """Extract final-classifier inputs and both label arrays."""
        model.eval()
        final = getattr(model, "fc", None)
        if not isinstance(final, nn.Linear):
            final = getattr(model, "classifier", None)
        if not isinstance(final, nn.Linear):
            raise TypeError("Expected final fc/classifier nn.Linear")

        features = []
        targets = {"labels": [], "spurious": []}
        captured = []

        def hook(_module, inputs):
            captured.append(inputs[0].detach().float().cpu())

        handle = final.register_forward_pre_hook(hook)
        try:
            for batch in loader:
                captured.clear()
                model(batch["pixel_values"].to(device))
                if len(captured) != 1:
                    raise RuntimeError(
                        "Classifier hook was not called once"
                    )
                features.append(captured[0])
                for key in targets:
                    targets[key].append(
                        batch[key].long().flatten().cpu()
                    )
        finally:
            handle.remove()
        if not features:
            raise ValueError("Empty feature dataset")
        return torch.cat(features), {
            key: torch.cat(values)
            for key, values in targets.items()
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run", type=Path, default=None,
                        help="Run directory for --checkpoint latest")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--target", default="both",
                        choices=["core", "spurious", "both"])
    parser.add_argument("--probe-epochs", "--max-iter", type=int,
                        dest="probe_epochs", default=100)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float,
                        default=1e-4)
    parser.add_argument("--core-classes", type=int, default=None)
    parser.add_argument("--spurious-classes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV output path")
    parser.add_argument("--probe-dir", type=Path, default=None,
                        help="Directory for saved linear probes")
    parser.add_argument("--load-probes", action="store_true",
                        help="Reuse saved probes instead of training")
    args = parser.parse_args()

    if args.batch_size < 1 or args.probe_batch_size < 1:
        parser.error("Batch sizes must be positive")
    if args.workers < 0:
        parser.error("workers must be nonnegative")
    if args.probe_epochs < 1:
        parser.error("probe-epochs must be positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    checkpoint = Path(args.checkpoint)
    if args.checkpoint == "latest":
        if args.run is None:
            parser.error("--run is required with --checkpoint latest")
        pointer = json.loads(
            (args.run / "latest.json").read_text()
        )
        epoch = int(pointer["epoch"])
        checkpoint = (args.run / "checkpoints"
                      / f"epoch_{epoch:03d}.pt")

    from shortcut_experiment import build_datasets, build_model

    saved = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    config = saved["config"]
    epoch = int(saved["epoch"])

    # We restore every trained weight; no download is needed.
    model = build_model({**config, "pretrained": False})
    # The training project may use a 2-class or 10-class head.
    head_name = "fc" if isinstance(
        getattr(model, "fc", None), nn.Linear
    ) else "classifier"
    head = getattr(model, head_name)
    n_outputs = saved["model"][f"{head_name}.weight"].size(0)
    if head.out_features != n_outputs:
        setattr(model, head_name, nn.Linear(
            head.in_features, n_outputs
        ))
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval().requires_grad_(False)

    # build_datasets() must provide all four existing splits.
    datasets = build_datasets(config)
    loaders = {
        name: DataLoader(
            datasets[name], batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers,
        )
        for name in ("train", "val", "test", "probe")
    }

    # Original classifier accuracy on train/val/test.
    accuracies = {}
    with torch.no_grad():
        for name in ("train", "val", "test"):
            correct = total = 0
            for batch in loaders[name]:
                x = batch["pixel_values"].to(device)
                y = batch["labels"].to(device).long()
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
            if not total:
                raise ValueError(f"Empty {name} dataset")
            accuracies[name] = correct / total

    # The probe learns from 'probe' only and is tested on 'test'.
    probe_x, probe_y = DecodedCorrelation.encode(
        model, loaders["probe"], device
    )
    test_x, test_y = DecodedCorrelation.encode(
        model, loaders["test"], device
    )

    out = args.output or (
        checkpoint.parent.parent / "evaluation.csv"
    )
    probe_dir = args.probe_dir or (
        checkpoint.parent.parent / "probes"
    )
    row = {
        "epoch": epoch,
        "checkpoint": str(checkpoint.resolve()),
        "decoded_core_correlation": "",
        "decoded_spurious_correlation": "",
        "core_probe_accuracy": "",
        "spurious_probe_accuracy": "",
        "train_accuracy": accuracies["train"],
        "validation_accuracy": accuracies["val"],
        "test_accuracy": accuracies["test"],
    }

    names = ("core", "spurious") if args.target == "both" \
        else (args.target,)
    for name in names:
        key = "labels" if name == "core" else "spurious"
        count = (args.core_classes if name == "core"
                 else args.spurious_classes)
        if count is None:
            count = int(probe_y[key].max().item()) + 1
        probe_path = probe_dir / f"epoch_{epoch:03d}_{name}.pt"

        if args.load_probes:
            probe = DecodedCorrelation.load(
                probe_path, str(device)
            )
            if probe.linear.out_features != count:
                raise ValueError("Saved probe class count differs")
        else:
            probe = DecodedCorrelation(
                probe_x.shape[1], count
            ).to(device)
            probe.fit(
                probe_x, probe_y[key],
                epochs=args.probe_epochs,
                batch_size=args.probe_batch_size,
                lr=args.probe_lr,
                weight_decay=args.probe_weight_decay,
            )
            probe.save(probe_path)

        result = probe.evaluate(
            test_x, test_y[key], args.probe_batch_size
        )
        row[f"decoded_{name}_correlation"] = (
            result["decoded_correlation"]
        )
        row[f"{name}_probe_accuracy"] = result["accuracy"]
        print(name, result, "saved:", probe_path)

    # One row per checkpoint; reruns replace their previous row.
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out.exists():
        with out.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        for old in rows:
            if old["checkpoint"] == row["checkpoint"]:
                # Keep other probe metrics if --target is selective.
                for field in row:
                    if row[field] == "":
                        row[field] = old[field]
        rows = [
            r for r in rows
            if r["checkpoint"] != row["checkpoint"]
        ]
    rows.append(row)
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)

    print("Accuracy:", accuracies)
    print("CSV:", out)
