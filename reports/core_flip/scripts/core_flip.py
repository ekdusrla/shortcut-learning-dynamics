"""Core flip vs spurious flip on every saved checkpoint (evaluation only, no training).

For each validation image i (original prediction p):
  * spurious swap: same core, other shortcut -- the team's counterfactual
      Waterbirds: same bird on its scene of the other place; Colored MNIST: same digit pixels in the paired other color.
  * core swap: same shortcut, other core
      Waterbirds: a bird of the other class (random partner j) pasted on image i's own scene;
      Colored MNIST: a digit of another class (random partner j) painted in image i's color.
Per epoch: spurious_flip = P(pred changes under the spurious swap), core_flip = P(pred changes under the core swap),
core_follow = P(pred on the core swap == partner's class), and the 4-way split of images into
core-only / spurious-only / both / neither (which swap changes the prediction).
spurious_flip on the full val set must equal history.csv's val_cf_flip_rate (consistency check).

Usage (from the repo root):
  python3 reports/core_flip/scripts/core_flip.py --dataset waterbirds      # 9 runs (seed 42/43/44 x corr)
  python3 reports/core_flip/scripts/core_flip.py --dataset cmnist          # 6 runs (BG/FG x corr)
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "reports" / "core_flip"
sys.path.insert(0, str(REPO))
import shortcut_experiment as se  # noqa: E402

dev = torch.device("cuda")
PARTNER_SEED = 0
CORRS = [("095", 0.95), ("080", 0.80), ("065", 0.65)]


def partners(labels, seed=PARTNER_SEED):
    """For each i a random j with labels[j] != labels[i] (fixed seed, same for every run of a dataset)."""
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(np.flatnonzero(labels != y)) for y in labels])


class Triples(Dataset):
    """(original, spurious swap, core swap) tensors for validation image k."""
    def __init__(self, make, n):
        self.make, self.n = make, n

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        return torch.stack(self.make(k)).half()


def build(make, n, workers=6):
    dl = DataLoader(Triples(make, n), batch_size=64, num_workers=workers)
    return torch.cat(list(dl)).to(dev)  # (n, 3, C, H, W) fp16 on GPU (autocast casts inputs to fp16 anyway)


# ---------------------------------------------------------------- Waterbirds
WB_DATA = REPO / "data" / "waterbirds_rerun50"
WB_RUNS = REPO / "runs" / "waterbirds_rerun50"


def waterbirds_val(corr):
    import scripts.waterbirds_vgg50.storage as storage
    from scripts.waterbirds_vgg50.protocol import load
    _local_root = storage.local_root
    storage.local_root = lambda root, base=None: _local_root(root, WB_DATA)  # local data root, as in local_pipeline.py
    return load(WB_DATA / "manifest" / "protocol.json", corr, verify=False)["val"]


def waterbirds_jobs():
    for ctag, corr in CORRS:
        ds = waterbirds_val(corr); y, s = ds.labels, ds.spurious; j = partners(y)

        def make(k, ds=ds, y=y, s=s, j=j):
            x, bird = ds.composite(k, int(s[k]))
            xs = ds.composite(k, 1 - int(s[k]), bird)[0]
            other = ds.composite(int(j[k]), int(s[j[k]]))[1]         # partner's bird (segmentation + pixels)
            xc = ds.composite(k, int(s[k]), other)[0]                  # on image k's own scene
            return x, xs, xc
        meta = dict(labels=y, spurious=s, partner=j, partner_label=y[j], aligned=(y == s))
        runs = []
        for seed in (42, 43, 44):
            d = WB_RUNS / f"A_corr{ctag}_seed{seed}_e50_contentv3"
            runs.append((dict(dataset="waterbirds", mode="", corr=ctag, seed=seed), d, None))
        yield make, len(y), meta, runs


# ---------------------------------------------------------------- Colored MNIST
CM_RUNS = REPO / "runs" / "rerun50"


def cmnist_jobs():
    for tag in ("bg", "fg"):
        for ctag, _ in CORRS:
            d = CM_RUNS / f"vgg11_{tag}_corr{ctag}"; cfg = json.loads((d / "config.json").read_text())
            ds = se.build_datasets(cfg)["val"]; ds.counterfactual = False
            y, s = ds.labels, ds.spurious; j = partners(y)

            def make(k, ds=ds, s=s, j=j):
                img = ds.base[int(ds.indices[k])]["image"]; other = ds.base[int(ds.indices[j[k]])]["image"]
                return ds.paint(img, int(s[k]), k), ds.paint(img, int(ds.cf_spurious[k]), k), ds.paint(other, int(s[k]), k)
            meta = dict(labels=y, spurious=s, partner=j, partner_label=y[j], aligned=(y == s))
            yield make, len(y), meta, [(dict(dataset="cmnist", mode=tag, corr=ctag, seed=42), d, cfg)]


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(model, X, bs=128):
    out = []
    for a in range(0, len(X), bs):
        x = X[a:a + bs].flatten(0, 1)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x).argmax(1).view(-1, 3).cpu())
    return torch.cat(out).numpy()  # (n, 3): original, spurious swap, core swap


def summarize(P, meta, sel):
    p, ps, pc = P[sel, 0], P[sel, 1], P[sel, 2]
    s_chg, c_chg = ps != p, pc != p
    return dict(n=int(sel.sum()), acc=float((p == meta["labels"][sel]).mean()),
                spurious_flip=float(s_chg.mean()), core_flip=float(c_chg.mean()),
                core_follow=float((pc == meta["partner_label"][sel]).mean()),
                core_only=float((c_chg & ~s_chg).mean()), spurious_only=float((s_chg & ~c_chg).mean()),
                both=float((c_chg & s_chg).mean()), neither=float((~c_chg & ~s_chg).mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=("waterbirds", "cmnist"), required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    jobs = waterbirds_jobs() if a.dataset == "waterbirds" else cmnist_jobs()
    for make, n, meta, runs in jobs:
        t0 = time.time(); X = build(make, n); print(f"built {n} triples in {time.time() - t0:.0f}s", flush=True)
        subsets = {"all": np.ones(n, bool), "aligned": meta["aligned"], "conflicting": ~meta["aligned"]}
        for key, d, cfg in runs:
            hist = pd.read_csv(d / "history.csv").set_index("epoch")
            t0 = time.time()
            for e in range(51):
                saved = se.load_checkpoint(d / "checkpoints" / f"epoch_{e:03d}.pt", dev)
                m = se.build_model({**(cfg or saved["config"]), "pretrained": False}).to(dev)
                m.load_state_dict(saved["model"]); m.eval()
                P = predict(m, X); del m, saved
                for name, sel in subsets.items():
                    rows.append(dict(**key, epoch=e, subset=name, **summarize(P, meta, sel)))
                r = rows[-3]  # "all"
                checks.append(dict(**key, epoch=e, acc=r["acc"], hist_acc=hist.val_acc[e],
                                   spurious_flip=r["spurious_flip"], hist_flip=hist.val_cf_flip_rate[e]))
            print(key, f"51 epochs in {time.time() - t0:.0f}s", flush=True)
        del X; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(OUT / f"core_flip_{a.dataset}.csv", index=False)
    c = pd.DataFrame(checks); c.to_csv(OUT / f"check_{a.dataset}.csv", index=False)
    print("max |acc - history|", (c.acc - c.hist_acc).abs().max(), " max |spurious_flip - history|", (c.spurious_flip - c.hist_flip).abs().max())


if __name__ == "__main__":
    main()
