"""Checkpoint-based Waterbirds figures (GPU): background-swap grids, Grad-CAM on the bird mask, stress test.

Usage: [WB_SEED=43] python figs_checkpoints.py [--n 300] [--parts swapgrid,cam,stress,stressgrid]
  --n: test images per bird x background group (balanced subset, fixed seed)."""
import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from common import CORRS, CORR_COLOR, GROUPS, INK, MUTED, OUT, REPO, RUNS, out_csv, run_name, save, style

sys.path.insert(0, str(REPO)); sys.path.insert(0, str(OUT / "scripts"))
import local_pipeline as lp  # noqa: E402  (applies the local data-root patch)
import shortcut_experiment as se  # noqa: E402
from data import Stress  # noqa: E402
from scripts.waterbirds_vgg50.protocol import load  # noqa: E402

dev = torch.device("cuda")
STRESS_EPOCH = 50
GRID_EPOCHS = [0, 1, 2, 5, 10, 25, 50]
CAM_EPOCHS = [1, 2, 5, 10, 25, 50]
CLS = {0: "land", 1: "water"}
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]; STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def to_image(x):
    return (x.cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def test_set(ctag):
    corr = {c: v for c, v in CORRS}[ctag]
    return load(lp.MANIFEST / "protocol.json", corr, verify=False)["test"]


def subset(ds, n, seed=0):
    rng = np.random.default_rng(seed); idx = []
    for y in (0, 1):
        for s in (0, 1):
            g = np.flatnonzero((ds.labels == y) & (ds.spurious == s))
            idx += list(rng.permutation(g)[:n])
    return np.array(sorted(idx))


class Pairs(Dataset):
    """(original, background-swapped, y, place, bird mask at 224) for chosen test indices."""
    def __init__(self, ds, idx, stress=None):
        self.ds, self.idx, self.stress = ds, idx, stress

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k]); self.ds.stress = self.stress
        y, s = int(self.ds.labels[i]), int(self.ds.spurious[i])
        x, bird = self.ds.composite(i, s); xs = self.ds.composite(i, 1 - s, bird)[0]
        m = torch.from_numpy(np.asarray(bird[0], np.float32)[..., 0])[None, None]
        m = F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        return x, xs, y, s, m


def models_for(ctag, epochs):
    d = RUNS / run_name(ctag); ms = {}
    for e in epochs:
        p = d / "checkpoints" / f"epoch_{e:03d}.pt"
        if not p.exists(): continue
        saved = se.load_checkpoint(p, dev); m = se.build_model(saved["config"]).to(dev)
        m.load_state_dict(saved["model"]); ms[e] = m.eval()
    return ms


@torch.no_grad()
def predict(ms, ds, idx, stress=None):
    dl = DataLoader(Pairs(ds, idx, stress), batch_size=64, num_workers=4)
    P = {e: ([], []) for e in ms}; Y, S = [], []
    for x, xs, y, s, _ in dl:
        x, xs = x.to(dev, non_blocking=True), xs.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            for e, m in ms.items():
                P[e][0].append(m(x).argmax(1).cpu()); P[e][1].append(m(xs).argmax(1).cpu())
        Y.append(y); S.append(s)
    ds.stress = None
    return {e: (torch.cat(a).numpy(), torch.cat(b).numpy()) for e, (a, b) in P.items()}, torch.cat(Y).numpy(), torch.cat(S).numpy()


def summarize(p, ps, y, s):
    groups = {f"g{a}{b}": ((y == a) & (s == b)) for a in (0, 1) for b in (0, 1)}
    gacc = {g: float((p[m] == y[m]).mean()) for g, m in groups.items()}
    correct = p == y
    return dict(acc=float(correct.mean()), balanced_acc=float(np.mean(list(gacc.values()))), wga=min(gacc.values()),
                flip=float((p != ps).mean()), c2i=float((correct & (ps != y)).sum() / max(correct.sum(), 1)),
                pred_waterbird=float((p == 1).mean()), **{f"{g}_acc": v for g, v in gacc.items()})


def pick_birds(ds, per_group=1, seed=3):
    rng = np.random.default_rng(seed); out = []
    for g in ("g10", "g01", "g11", "g00"):  # conflicting groups first
        y, s = int(g[1]), int(g[2])
        out += list(rng.permutation(np.flatnonzero((ds.labels == y) & (ds.spurious == s)))[:per_group])
    return out


def mark(ax, ok):
    col = "#1a7f37" if ok else "#cf222e"
    for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(0.6 if ok else 2.2)
    return col


def part_swapgrid(ctags):
    for ctag in ctags:
        ms = models_for(ctag, GRID_EPOCHS)
        if not ms: continue
        ds = test_set(ctag); idx = pick_birds(ds)
        pairs = [Pairs(ds, idx)[k] for k in range(len(idx))]
        fig, axes = plt.subplots(len(ms), 2 * len(idx), figsize=(1.45 * 2 * len(idx), 1.55 * len(ms)))
        for r, (e, m) in enumerate(ms.items()):
            with torch.no_grad():
                X = torch.stack([t for x, xs, *_ in pairs for t in (x, xs)]).to(dev); p = m(X).argmax(1).cpu().numpy()
            for k, (x, xs, y, s, _) in enumerate(pairs):
                for j, (img, bg) in enumerate(((x, s), (xs, 1 - s))):
                    ax = axes[r, 2 * k + j]; ax.imshow(to_image(img)); ax.set_xticks([]); ax.set_yticks([])
                    pr = int(p[2 * k + j]); col = mark(ax, pr == y)
                    ax.set_title(f"{CLS[pr]}bird", fontsize=7, color=col, pad=1)
                    if r == 0: ax.text(0.5, 1.32, f"{CLS[y]}bird on {CLS[bg]}" + ("\n(original)" if j == 0 else "\n(swapped)"),
                                       transform=ax.transAxes, ha="center", fontsize=7, color=INK)
            axes[r, 0].set_ylabel(f"epoch {e}", fontsize=8, color=INK, rotation=0, ha="right", va="center")
        fig.suptitle(f"Waterbirds corr {int(ctag) / 100:.2f}: the same bird on its own and on the opposite background, prediction per epoch\n"
                     "green = correct bird class, red = wrong", fontsize=9, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0.04, 0, 1, 0.955)); save(fig, f"05_swap_grid_{ctag}")
        del ms; torch.cuda.empty_cache()


def gradcam(model, x, layer="features.25"):
    acts = {}
    h = dict(model.named_modules())[layer].register_forward_hook(lambda mod, i, o: acts.update(a=o))
    with torch.enable_grad():
        logits = model(x); p = logits.argmax(1)
        g = torch.autograd.grad(logits.gather(1, p[:, None]).sum(), acts["a"])[0]
    h.remove()
    cam = (g.mean((2, 3), keepdim=True) * acts["a"].detach()).sum(1, keepdim=True).clamp(min=0)
    return p.cpu().numpy(), F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0].cpu().numpy()


def part_cam(ctags, n):
    rows = []
    for ctag in ctags:
        ms = models_for(ctag, CAM_EPOCHS)
        if not ms: continue
        ds = test_set(ctag)
        idx = [i for i in subset(ds, n) if ds.labels[i] != ds.spurious[i]]  # conflicting only
        items = [Pairs(ds, [i])[0] for i in idx]
        X = torch.stack([t[0] for t in items]); Y = np.array([t[2] for t in items]); M = torch.stack([t[4] for t in items]).numpy()
        bird = M > 0.5; keep = {}
        for e, m in ms.items():
            P, A = [], []
            for j in range(0, len(X), 50):
                p, a = gradcam(m, X[j:j + 50].to(dev)); P.append(p); A.append(a)
            P, A = np.concatenate(P), np.concatenate(A); t = A.sum((1, 2))
            share = np.where(t > 0, (A * bird).sum((1, 2)) / np.maximum(t, 1e-12), np.nan)
            for lab, sel in (("correct", P == Y), ("wrong (background's class)", P != Y)):
                if sel.sum() >= 5:
                    rows.append(dict(corr=ctag, epoch=e, predicted=lab, n=int(sel.sum()), bird_share=float(np.nanmean(share[sel])),
                                     bird_area=float(bird[sel].mean())))
            keep[e] = (P[:6], A[:6])
        fig, axes = plt.subplots(6, len(keep) + 1, figsize=(1.75 * (len(keep) + 1), 1.8 * 6))
        for r in range(6):
            img = to_image(X[r]); axes[r, 0].imshow(img)
            axes[r, 0].set_title(f"{CLS[Y[r]]}bird on {CLS[1 - Y[r]]}", fontsize=7, color=INK)
            gimg = img.mean(-1, keepdims=True).repeat(3, -1)
            for k, (e, (p, a)) in enumerate(keep.items(), 1):
                an = (a[r] - a[r].min()) / max(a[r].max() - a[r].min(), 1e-12)
                axes[r, k].imshow(np.clip(0.5 * gimg + 0.5 * plt.cm.jet(an)[..., :3], 0, 1))
                sh = (a[r] * bird[r]).sum() / max(a[r].sum(), 1e-12)
                col = "#1a7f37" if p[r] == Y[r] else "#cf222e"
                axes[r, k].set_title(f"ep{e}: {CLS[p[r]]}, on bird {sh * 100:.0f}%", fontsize=6.5, color=col)
        for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Waterbirds corr {int(ctag) / 100:.2f}: Grad-CAM (predicted class, VGG features.25) on background-conflicting test birds\n"
                     "title: green = correct, red = wrong; % = share of the map inside the bird mask", fontsize=8, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, f"06_gradcam_grid_{ctag}")
        del ms; torch.cuda.empty_cache()
    df = pd.DataFrame(rows); df.to_csv(out_csv("gradcam_bird_share.csv"), index=False)
    plot_cam_share(df)


def plot_cam_share(df, min_n=20):
    if df.empty: return
    df = df.copy(); df["corr"] = df["corr"].astype(str).str.zfill(3)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for ctag, corr in CORRS:
        for lab, ls in (("correct", "-"), ("wrong (background's class)", "--")):
            g = df[(df["corr"] == ctag) & (df.predicted == lab)]; g = g[g.n >= min_n]
            if len(g): ax.plot(g.epoch, g.bird_share * 100, color=CORR_COLOR[ctag], ls=ls, marker="o", ms=3, label=f"corr {corr:.2f}, {lab}")
    area = df.bird_area.mean(); ax.axhline(area * 100, color=MUTED, lw=1, ls=":"); ax.text(1.0, area * 100 + 0.4, "bird area (share of the image)", fontsize=7, color=MUTED, va="bottom")
    ax.set_xscale("log"); ax.set_xlim(0.85, 60); ax.set_xticks(CAM_EPOCHS); ax.set_xticklabels(CAM_EPOCHS); ax.minorticks_off()
    style(ax, "Grad-CAM share inside the bird mask (conflicting test birds)", "epoch", "% of the map")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(); save(fig, "07_gradcam_bird_share")


SETTINGS = [("clean", None, 0)] + [(k, k, v) for k in ("gaussian_noise", "gaussian_blur", "crop") for v in (1, 3, 5)]


def part_stress(ctags, n):
    rows = []
    for ctag in ctags:
        ms = models_for(ctag, [STRESS_EPOCH])
        if not ms: continue
        ds = test_set(ctag); idx = subset(ds, n)
        for name, kind, sev in SETTINGS:
            st = None if kind is None else Stress(kind, sev, 42)
            P, Y, S = predict(ms, ds, idx, st); p, ps = P[STRESS_EPOCH]
            rows.append(dict(corr=ctag, stress=name, severity=sev, n=len(idx), **summarize(p, ps, Y, S)))
            print(ctag, name, sev, {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
        del ms; torch.cuda.empty_cache()
    df = pd.DataFrame(rows); df.to_csv(out_csv("stress_epoch50.csv"), index=False)
    if df.empty: return
    df["corr"] = df["corr"].astype(str).str.zfill(3)
    show = [("balanced_acc", "balanced accuracy (%)"), ("wga", "worst-group acc (%)"), ("flip", "background-swap flip (%)")]
    fig, axes = plt.subplots(len(show), 3, figsize=(13, 8), sharex=True)
    for k, kind in enumerate(("gaussian_noise", "gaussian_blur", "crop")):
        for r, (m, lab) in enumerate(show):
            ax = axes[r, k]
            for ctag, corr in CORRS:
                g = df[df["corr"] == ctag]
                if g.empty: continue
                g = pd.concat([g[g.stress == "clean"], g[g.stress == kind]]).sort_values("severity")
                ax.plot(g.severity, g[m] * 100, color=CORR_COLOR[ctag], marker="o", ms=3, label=f"corr {corr:.2f}")
            style(ax, kind.replace("_", " ") if r == 0 else None, "severity (0 = clean)" if r == len(show) - 1 else None, lab if k == 0 else None)
            ax.set_xticks([0, 1, 3, 5])
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Waterbirds stress test at epoch 50 ({n} test images per bird × background group; same corruption on original and swapped image)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "08_stress_epoch50")


def part_stressgrid(ctags):
    for ctag in ctags:
        ms = models_for(ctag, [STRESS_EPOCH])
        if not ms: continue
        m = ms[STRESS_EPOCH]; ds = test_set(ctag); idx = pick_birds(ds)[:3]
        fig, axes = plt.subplots(len(SETTINGS), 2 * len(idx), figsize=(1.45 * 2 * len(idx), 1.4 * len(SETTINGS)))
        for r, (name, kind, sev) in enumerate(SETTINGS):
            st = None if kind is None else Stress(kind, sev, 42)
            pairs = [Pairs(ds, idx, st)[k] for k in range(len(idx))]; ds.stress = None
            with torch.no_grad():
                X = torch.stack([t for x, xs, *_ in pairs for t in (x, xs)]).to(dev); p = m(X).argmax(1).cpu().numpy()
            for k, (x, xs, y, s, _) in enumerate(pairs):
                for j, (img, bg) in enumerate(((x, s), (xs, 1 - s))):
                    ax = axes[r, 2 * k + j]; ax.imshow(to_image(img)); ax.set_xticks([]); ax.set_yticks([])
                    pr = int(p[2 * k + j]); col = mark(ax, pr == y); ax.set_title(f"{CLS[pr]}bird", fontsize=7, color=col, pad=1)
                    if r == 0: ax.text(0.5, 1.32, f"{CLS[y]}bird on {CLS[bg]}", transform=ax.transAxes, ha="center", fontsize=7, color=INK)
            label = "clean" if st is None else f"{kind.replace('gaussian_', '')} {sev}\n({st.describe(224)})"
            axes[r, 0].set_ylabel(label, fontsize=7, color=INK, rotation=0, ha="right", va="center")
        fig.suptitle(f"Waterbirds corr {int(ctag) / 100:.2f}, epoch {STRESS_EPOCH}: three test birds (original | swapped background) under each corruption\n"
                     "green = correct bird class, red = wrong", fontsize=9, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0.07, 0, 1, 0.965)); save(fig, f"09_stress_examples_{ctag}")
        del ms; torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--corrs", default="095,080,065")
    ap.add_argument("--parts", default="swapgrid,cam,stress,stressgrid")
    a = ap.parse_args(); ctags = a.corrs.split(","); parts = a.parts.split(",")
    if "swapgrid" in parts: part_swapgrid([c for c in ctags if c in ("095", "065")])
    if "cam" in parts: part_cam(ctags, a.n)
    if "stress" in parts: part_stress(ctags, a.n)
    if "stressgrid" in parts: part_stressgrid([c for c in ctags if c in ("095", "065")])
    if "camplot" in parts: plot_cam_share(pd.read_csv(OUT / "gradcam_bird_share.csv"))
