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
  python3 reports/waterbirds_rerun50/scripts/local_pipeline.py train --corr 0.95 [--seed 42] [--data cubfull] [--pretrained] [--lr 0.01] [--scheduler cosine] [--scene-resample] [--resume]

--seed (default 42 = the team run) changes only the two training random sources the team code fixes:
model init (seed_everything) and the train-loader shuffle generator (team: 819 = 42 + 777, here seed + 777).
With --seed 42 both are exactly the team values. The validation images saved as feature snapshots stay
the team's fixed subset (seed 42), so probes compare the same 128 images across seeds.

Options beyond the team run (all off by default; each is recorded in config.json and appended to the run folder name):
--data cubfull: train on data/waterbirds_rerun50/manifest_cubfull (cubfull_manifest.py): 9,390 train birds instead of
  4,795, same val, a 1,199-bird test. The manifest path and hash are recorded in config.json. *_cubfull folders.
--pretrained: ImageNet VGG11_BN features (torchvision VGG11_BN_Weights.DEFAULT) + a new 2-class head; all layers are trained.
  *_pretrained folders.
--lr X: initial learning rate (team 0.01). *_lr{X} folders.
--scheduler cosine (default none = team, lr constant): CosineAnnealingLR(T_max = 50) from the initial lr, stepped once
  per epoch. *_cosine folders.
--scene-resample (option B): the training loader pastes each bird on a new scene every epoch, drawn from the train scene
  pool of its place kind (land/water), keyed on (seed, epoch, img_id). Places, so the correlation, and the data order are
  unchanged; evaluation (val, test, train split) uses the manifest scenes. Recorded as scene_resample. *_multiscene folders.
--cache-eval / --prefetch N / --workers N: speed only (evaluation images composited once and reused; more batches built
  ahead; more loader processes). Training order, samples and every metric are unchanged.
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


def scene_pool():
    """Train-pool Places scenes per place kind {0: land, 1: water}, one per content hash (as in cubfull_manifest.py)."""
    import data
    meta = json.loads((MANIFEST / "protocol.json").read_text()); audit = json.loads((MANIFEST / "split_audit.json").read_text())
    kinds = {0: ("b/bamboo_forest/", "f/forest/broadleaf/"), 1: ("o/ocean/", "l/lake/natural/")}
    prefix = "places365/" + data.PLACES_PREFIX + "/"
    train_hash = {h for h, s in audit["scene_group_split"].items() if s == "train"}
    pool, seen = {0: [], 1: []}, set()
    for path, h in sorted(meta["assets"].items()):
        if not path.startswith(prefix) or h not in train_hash or h in seen: continue
        rel = path[len(prefix):]
        for kind, pre in kinds.items():
            if rel.startswith(pre): pool[kind].append(rel); seen.add(h)
    return pool


class CachedEval:
    """Evaluation split whose images are composited once (bit-identical to compositing every epoch).

    Stores the 224x224 uint8 image right after the dataset's Resize (and the counterfactual's, if any); ToTensor and
    Normalize run per item exactly as in the dataset transform. Only for loaders without shuffling or augmentation.
    """
    def __init__(self, ds, workers):
        import numpy as np
        import torch
        from torchvision import transforms
        self.ds = ds
        self.post = transforms.Compose(ds.transform.transforms[1:])  # ToTensor, Normalize
        assert isinstance(ds.transform.transforms[0], transforms.Resize) and len(ds.transform.transforms) == 3 and ds.stress is None
        raw = __import__("copy").copy(ds); raw.transform = transforms.Compose([ds.transform.transforms[0], np.asarray])
        loader = torch.utils.data.DataLoader(raw, batch_size=64, num_workers=workers, collate_fn=lambda b: b)
        items = [x for batch in loader for x in batch]
        self.img = np.stack([np.asarray(x["pixel_values"]) for x in items])
        self.cf = np.stack([np.asarray(x["cf_pixel_values"]) for x in items]) if ds.counterfactual else None
        self.meta = [{k: v for k, v in x.items() if not k.endswith("pixel_values")} for x in items]
        for name in ("labels", "spurious", "ids", "counterfactual"): setattr(self, name, getattr(ds, name))

    def __len__(self):
        return len(self.meta)

    def __getattr__(self, name):  # anything else (groups, num_classes, ...) from the wrapped dataset
        if name == "ds": raise AttributeError(name)
        return getattr(self.ds, name)

    def __getitem__(self, i):
        out = dict(self.meta[i], pixel_values=self.post(self.img[i]))
        if self.cf is not None: out["cf_pixel_values"] = self.post(self.cf[i])
        return out


_CACHE = {}


def cached(ds, workers):
    if id(ds) not in _CACHE: _CACHE[id(ds)] = CachedEval(ds, workers)
    return _CACHE[id(ds)]


def cmd_train(args):
    import torch
    import shortcut_experiment as exp
    from scripts.waterbirds_vgg50.download import atomic_json
    from scripts.waterbirds_vgg50.full_training import configuration, fit, resume_state, run_lock, verify_run
    from scripts.waterbirds_vgg50.protocol import load
    manifest = MANIFEST if args.data == "team" else DATA / f"manifest_{args.data}"
    cfg = configuration(manifest / "protocol.json", args.corr)
    cfg["seed"] = args.seed
    out = RUNS / f"A_corr{round(args.corr * 100):03d}_seed{args.seed}_e{cfg['epochs']}_contentv3"
    if args.data != "team": out = out.with_name(out.name + f"_{args.data}")
    if args.pretrained:
        # ImageNet VGG11_BN features (torchvision VGG11_BN_Weights.DEFAULT); the 2-class head is still new.
        cfg["pretrained"] = True; out = out.with_name(out.name + "_pretrained")
    if args.lr is not None:
        cfg["lr"] = args.lr; out = out.with_name(out.name + f"_lr{args.lr:g}")
    if args.scheduler == "cosine":
        # fit() hard-codes ConstantLR; swap in the cosine schedule of shortcut_experiment (T_max = epochs, stepped per epoch).
        cfg["scheduler"] = "cosine"
        out = out.with_name(out.name + "_cosine")
        torch.optim.lr_scheduler.ConstantLR = lambda optimizer, **_: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    if args.scene_resample:
        import copy
        import numpy as np
        import data
        cfg["scene_resample"] = "per_epoch_train_pool"
        out = out.with_name(out.name + "_multiscene")
        pool = scene_pool()
        cfg["scene_pool_sizes"] = [len(pool[0]), len(pool[1])]

        class Resampled(data.Waterbirds):
            # Each epoch every training bird gets a new scene of its own place kind (land or water) from the train
            # scene pool; the place (so the correlation) is unchanged. The draw depends only on (seed, epoch, img_id),
            # so it is reproducible and resume-safe. self.scenes is this copy's own array (workers each hold a copy).
            def __getitem__(self, i):
                s = int(self.spurious[i])
                r = np.random.default_rng([args.seed, self.epoch, int(self.ids[i])])
                self.scenes[i, s] = pool[s][r.integers(len(pool[s]))]
                return super().__getitem__(i)

        _train_one_epoch = exp.train_one_epoch

        def train_one_epoch(model, loader, *a, **k):
            # epoch being trained = last saved checkpoint + 1 (epoch_000 is the untrained model), also after --resume
            done = [int(p.stem.split("_")[1]) for p in (out / "checkpoints").glob("epoch_*.pt")]
            loader.dataset.epoch = max(done) + 1
            return _train_one_epoch(model, loader, *a, **k)
        exp.train_one_epoch = train_one_epoch
    _loader = exp.create_loader

    def create_loader(ds, batch_size, shuffle, workers, gen=None, device=None):
        # fit() builds the train-loader generator as manual_seed(819); reseed it to seed + 777 (819 for seed 42).
        if shuffle and gen is not None: gen.manual_seed(args.seed + 777)
        if shuffle and args.scene_resample:
            # Only the shuffled training loader resamples scenes; the train split is evaluated on its manifest scenes.
            ds = copy.copy(ds); ds.__class__ = Resampled; ds.scenes = ds.scenes.copy(); ds.epoch = 0
        # --workers only changes how many processes composite images; the order comes from `gen` in the main
        # process and there is no augmentation, so samples are identical. cfg (num_workers=4) is left as is.
        if not shuffle and args.cache_eval:
            # Evaluation images (and their counterfactuals) are the same every epoch: composite them once.
            ds = cached(ds, args.workers or workers)
        if args.prefetch and (args.workers or workers) > 0:
            # --prefetch only lets each worker build more batches ahead (team: 1); order and samples are unchanged.
            from torch.utils.data import DataLoader
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=args.workers or workers, generator=gen,
                              pin_memory=False, persistent_workers=False, prefetch_factor=args.prefetch)
        return _loader(ds, batch_size, shuffle, args.workers or workers, gen, device)
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
        if json.loads((out / "latest.json").read_text())["epoch"] == cfg["epochs"]:
            atomic_json(out / "completion_integrity.json", verify_run(out, cfg))
            (out / "TRAIN_SUCCESS").touch()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("download", "prepare", "compare"): sub.add_parser(name)
    t = sub.add_parser("train"); t.add_argument("--corr", type=float, choices=(0.95, 0.8, 0.65), required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--data", choices=("team", "cubfull"), default="team",
                   help="cubfull = manifest_cubfull from cubfull_manifest.py (CUB test birds moved into train)")
    t.add_argument("--scene-resample", action="store_true",
                   help="option B: every epoch each training bird gets a new scene of its place kind from the train scene pool")
    t.add_argument("--scheduler", choices=("none", "cosine"), default="none", help="none = team (lr 0.01 constant)")
    t.add_argument("--cache-eval", action="store_true", help="composite the evaluation splits once instead of every epoch (same images)")
    t.add_argument("--prefetch", type=int, default=0, help="batches built ahead per loader process (0 = team value, 1)")
    t.add_argument("--workers", type=int, default=0, help="data-loader processes (0 = team value from cfg, 4)")
    t.add_argument("--pretrained", action="store_true", help="start from ImageNet VGG11_BN weights (team: from scratch)")
    t.add_argument("--lr", type=float, default=None, help="initial lr (default: team 0.01); added to the folder name")
    t.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    {"download": cmd_download, "prepare": cmd_prepare, "compare": cmd_compare, "train": cmd_train}[a.cmd](a)
