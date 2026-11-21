"""Re-run the team's Waterbirds Condition A experiment on a single local PC.

Reuses the team code in scripts/waterbirds_vgg50/ unchanged (download, manifest, dataset,
training loop, metrics, checkpoint integrity). Only the server-specific guards are bypassed:
  * data lives under <repo>/data/waterbirds_rerun50/ instead of /local_datasets/<user>/
  * the archive cache is a local folder instead of network storage
  * no SLURM job / moana-u2 hostname check, no 2-epoch pilot gate (it only estimated time/disk)
The training configuration comes from full_training.configuration(), so hyperparameters are
the team's: VGG11_BN (no pretraining), SGD lr 0.01 constant, momentum 0.9, wd 1e-4, batch 128,
224px, AMP, seed 42, 50 epochs, no augmentation.

Usage (from the repo root):
  python3 reports/waterbirds_rerun50/scripts/local_pipeline.py download
  python3 reports/waterbirds_rerun50/scripts/local_pipeline.py prepare
  python3 reports/waterbirds_rerun50/scripts/local_pipeline.py compare
  python3 reports/waterbirds_rerun50/scripts/local_pipeline.py train --corr 0.95 [--seed 42] [--resume]

--seed (default 42 = the team run) changes only the two training random sources the team code fixes:
model init (seed_everything) and the train-loader shuffle generator (team: 819 = 42 + 777, here seed + 777).
With --seed 42 both are exactly the team values. The validation images saved as feature snapshots stay
the team's fixed subset (seed 42), so probes compare the same 128 images across seeds.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import scripts.waterbirds_vgg50.storage as storage  # noqa: E402

DATA = REPO / "data" / "waterbirds_rerun50"
ROOT = DATA / "waterbirds_v1"          # CUB + Places images (git-ignored)
CACHE = DATA / "archive_cache"         # downloaded archives / range cache (git-ignored)
MANIFEST = DATA / "manifest"           # regenerated protocol.json + A_corr*.csv
RUNS = REPO / "runs" / "waterbirds_rerun50"
SHARED = REPO / "reports" / "waterbirds_vgg50" / "shared" / "dataset"

_local_root = storage.local_root
storage.local_root = lambda root, base=None: _local_root(root, DATA)


class LocalPolicy(storage.StoragePolicy):
    """Same byte budgets as the team policy, without the node-local SSD / NFS cache checks."""
    def __init__(self, root, cache):
        self.root = storage.local_root(root)
        self.cache = Path(cache).absolute()


def cmd_download(_):
    from scripts.waterbirds_vgg50.download import prepare
    from scripts.waterbirds_vgg50.curl_transport import CurlOpener
    ROOT.mkdir(parents=True, exist_ok=True)
    prepare(ROOT, opener=CurlOpener(), policy=LocalPolicy(ROOT, CACHE))
    print("download done:", ROOT)


def cmd_prepare(_):
    from scripts.waterbirds_vgg50.protocol import prepare
    prepare(ROOT, MANIFEST, expected_birds=11788, expected_scenes_per_category=5000)
    print("manifest written:", MANIFEST)


def cmd_compare(_):
    """Row-by-row comparison of the regenerated manifests with the team's shared ones."""
    import pandas as pd
    ok = True
    for name in ("A_corr095.csv", "A_corr080.csv", "A_corr065.csv"):
        mine = pd.read_csv(MANIFEST / name).sort_values("img_id").reset_index(drop=True)
        team = pd.read_csv(SHARED / name).sort_values("img_id").reset_index(drop=True)
        same_shape = mine.shape == team.shape
        diff = {c: int((mine[c] != team[c]).sum()) for c in team.columns} if same_shape and list(mine.columns) == list(team.columns) else None
        print(name, "rows", len(mine), "vs", len(team), "| differing cells per column:", diff)
        ok &= bool(diff) and not any(diff.values())
    mine = json.loads((MANIFEST / "protocol.json").read_text())
    team = json.loads((SHARED / "protocol.json").read_text())
    for key in ("final_counts", "source_counts", "original_counts"):
        print(key, "same" if mine.get(key) == team.get(key) else f"DIFFERENT\n mine={mine.get(key)}\n team={team.get(key)}")
        ok &= mine.get(key) == team.get(key)
    print("IDENTICAL to team manifest" if ok else "MANIFEST DIFFERS from team")


def cmd_train(args):
    import torch
    import shortcut_experiment as exp
    from scripts.waterbirds_vgg50.download import atomic_json
    from scripts.waterbirds_vgg50.full_training import configuration, fit, resume_state, run_lock, verify_run
    from scripts.waterbirds_vgg50.protocol import load
    cfg = configuration(MANIFEST / "protocol.json", args.corr)
    cfg["seed"] = args.seed
    out = RUNS / f"A_corr{round(args.corr * 100):03d}_seed{args.seed}_e50_contentv3"
    _loader = exp.create_loader

    def create_loader(ds, batch_size, shuffle, workers, gen=None, device=None):
        # fit() builds the train-loader generator as manual_seed(819); reseed it to seed + 777 (819 for seed 42).
        if shuffle and gen is not None: gen.manual_seed(args.seed + 777)
        return _loader(ds, batch_size, shuffle, workers, gen, device)
    exp.create_loader = create_loader
    if args.resume:
        if not out.is_dir(): raise FileNotFoundError(out)
    else:
        out.mkdir(parents=True, exist_ok=False)
    with run_lock(out):
        saved = resume_state(out, cfg) if args.resume else None
        if args.resume and json.loads((out / "config.json").read_text()) != cfg: raise ValueError("Run config differs")
        if not args.resume: atomic_json(out / "config.json", cfg)
        datasets = load(cfg["manifest"], args.corr, require_local=True)
        exp.seed_everything(args.seed); model = exp.build_model(cfg).to("cuda")
        fit(model, datasets, cfg, out, torch.device("cuda"), saved=saved)
        if json.loads((out / "latest.json").read_text())["epoch"] == 50:
            atomic_json(out / "completion_integrity.json", verify_run(out, cfg))
            (out / "TRAIN_SUCCESS").touch()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("download", "prepare", "compare"): sub.add_parser(name)
    t = sub.add_parser("train"); t.add_argument("--corr", type=float, choices=(0.95, 0.8, 0.65), required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    {"download": cmd_download, "prepare": cmd_prepare, "compare": cmd_compare, "train": cmd_train}[a.cmd](a)
