"""Figures that need the saved checkpoints (GPU).

08 digit x color accuracy through training (per run), 09 snapshot overview of all runs,
10 recolor grids (same digit in all 10 colors, prediction per epoch), 11 stress test at epoch 50,
12/13 Grad-CAM on Background runs (color lives in the background, so the map can separate it).
Usage: python figs_checkpoints.py [--n 1000] [--runs bg095,fg095,...] [--parts heat,grid,stress,cam,camplot,stressgrid]"""
import argparse
import json
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from common import CORRS, CORR_COLOR, INK, MODES, MUTED, OUT, REPO, run_dir, save, style

sys.path.insert(0, str(REPO))
import shortcut_experiment as se  # noqa: E402
from data import MNIST_COLORS, Stress, to_image  # noqa: E402

dev = torch.device("cuda")
STRESS_EPOCH = 50
CAM_EPOCHS = [1, 2, 3, 5, 10, 50]
HEAT_EPOCHS = [0, 1, 2, 3, 5, 10, 25, 50]
NAMES = {"bg": "BG", "fg": "FG"}


class Recolor(Dataset):
    """Test digit i in all 10 colors (same pixels), in the run's color mode."""
    def __init__(self, ds, n, stress=None):
        self.ds, self.n, self.stress = ds, n, stress

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        self.ds.stress = self.stress
        img = self.ds.base[int(self.ds.indices[i])]["image"]
        return torch.stack([self.ds.paint(img, c, i) for c in range(10)]), int(self.ds.labels[i]), int(self.ds.spurious[i])


def models_for(tag, ctag, epochs):
    d = run_dir(tag, ctag); cfg = json.loads((d / "config.json").read_text()); ms = {}
    for e in epochs:
        p = d / "checkpoints" / f"epoch_{e:03d}.pt"
        if not p.exists(): continue
        m = se.build_model({**cfg, "pretrained": False}).to(dev)
        m.load_state_dict(se.load_checkpoint(p, dev)["model"]); ms[e] = m.eval()
    return cfg, ms


@torch.no_grad()
def predict_all(cfg, ms, n, stress=None):
    """Return {epoch: preds (n, 10 colors)}, labels (n,), original colors (n,)."""
    ds = se.build_datasets(cfg)["test"]; ds.counterfactual = False
    dl = DataLoader(Recolor(ds, n, stress), batch_size=25, num_workers=4)
    P = {e: [] for e in ms}; Y, S = [], []
    for x, y, s in dl:
        x = x.flatten(0, 1).to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            for e, m in ms.items():
                P[e].append(m(x).argmax(1).view(-1, 10).cpu())
        Y.append(y); S.append(s)
    return {e: torch.cat(v).numpy() for e, v in P.items()}, torch.cat(Y).numpy(), torch.cat(S).numpy()


def heat(ax, M, title, ylab=True, annotate=True):
    ax.imshow(M * 100, cmap="Blues", vmin=0, vmax=100)
    if annotate:
        for d in range(10):
            for c in range(10):
                ax.text(c, d, f"{M[d, c] * 100:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if M[d, c] > 0.6 else INK)
    ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.tick_params(labelsize=6, colors=MUTED, length=0)
    if not ylab: ax.set_yticklabels([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title, fontsize=8, color=INK)


def acc_matrix(P, Y):
    M = np.zeros((10, 10))
    for d in range(10): M[d] = (P[Y == d] == d).mean(0)
    return M


def part_heat(runs, n):
    rows, snap = [], {}
    for tag, ctag in runs:
        cfg, ms = models_for(tag, ctag, HEAT_EPOCHS)
        if not ms: continue
        P, Y, _ = predict_all(cfg, ms, n)
        eps = sorted(P)
        fig, axes = plt.subplots(1, len(eps), figsize=(2.3 * len(eps), 2.9))
        for k, e in enumerate(eps):
            M = acc_matrix(P[e], Y); off = ~np.eye(10, dtype=bool)
            color = np.mean([(P[e][Y == d][:, c] == c).mean() for d in range(10) for c in range(10) if c != d])
            heat(axes[k], M, f"epoch {e}\nunseen-color acc {M[off].mean() * 100:.0f}%, picks color {color * 100:.0f}%", k == 0)
            rows.append(dict(mode=tag, corr=ctag, epoch=e, diag_acc=np.diag(M).mean(), offdiag_acc=M[off].mean(), picks_color=color))
            snap[(tag, ctag, e)] = M
        axes[0].set_ylabel("true digit", fontsize=7, color=MUTED)
        for ax in axes: ax.set_xlabel("color painted", fontsize=7, color=MUTED)
        fig.suptitle(f"{NAMES[tag]} corr 0.{ctag[1:]}: test accuracy (%) of each digit painted in each color ({n} digits x 10 colors; "
                     "diagonal = the digit's main color)", fontsize=9, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.9)); save(fig, f"08_digit_color_{tag}{ctag}")
        del ms; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(OUT / "digit_color_summary.csv", index=False)
    eps = [1, 3, 10, 50]; order = [(t, c) for t, _, _ in MODES for c, _ in CORRS if (t, c, 1) in snap]
    if not order: return
    fig, axes = plt.subplots(len(order), len(eps), figsize=(2.2 * len(eps), 2.2 * len(order)))
    axes = np.atleast_2d(axes)
    for r, (t, c) in enumerate(order):
        for k, e in enumerate(eps):
            ax = axes[r, k]
            if (t, c, e) not in snap: ax.axis("off"); continue
            heat(ax, snap[(t, c, e)], f"epoch {e}" if r == 0 else "", k == 0, annotate=False)
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(f"{NAMES[t]} 0.{c[1:]}", fontsize=9, color=INK, rotation=0, ha="right", va="center")
    fig.suptitle("Digit (rows) x painted color (cols) test accuracy: dark = correct.\nA bright off-diagonal means the color, not the shape, decides",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0.04, 0, 1, 0.93)); save(fig, "09_digit_color_overview")


def part_grid(runs, picks=(3, 7)):
    eps = [0, 1, 2, 3, 5, 10, 50]
    for tag, ctag in runs:
        cfg, ms = models_for(tag, ctag, eps)
        if not ms: continue
        ds = se.build_datasets(cfg)["test"]; ds.counterfactual = False
        idx = [int(np.flatnonzero(ds.labels == d)[0]) for d in picks]
        fig, axes = plt.subplots(len(idx) * len(ms), 10, figsize=(11, 1.15 * len(idx) * len(ms)))
        r = 0
        for i, d in zip(idx, picks):
            img = ds.base[int(ds.indices[i])]["image"]
            X = torch.stack([ds.paint(img, c, i) for c in range(10)])
            for e, m in ms.items():
                with torch.no_grad(): p = m(X.to(dev)).argmax(1).cpu().numpy()
                for c in range(10):
                    ax = axes[r, c]; ax.imshow(to_image(X[c])); ax.set_xticks([]); ax.set_yticks([])
                    state = "ok" if p[c] == d else ("color" if p[c] == c else "other")
                    col = {"ok": "#1a7f37", "color": "#cf222e", "other": MUTED}[state]
                    ax.set_title(f"{p[c]}", fontsize=7, color=col, pad=1)
                    for s in ax.spines.values(): s.set_edgecolor(col); s.set_linewidth(2.2 if state != "ok" else 0.6)
                axes[r, 0].set_ylabel(f"digit {d}\nepoch {e}", fontsize=7, color=INK, rotation=0, ha="right", va="center")
                r += 1
        fig.suptitle(f"{NAMES[tag]} corr 0.{ctag[1:]}: one test digit painted in all 10 colors, prediction above each image\n"
                     "green = correct, red = the painted color's class, gray = another digit", fontsize=9, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0.06, 0, 1, 0.97)); save(fig, f"10_recolor_grid_{tag}{ctag}")
        del ms; torch.cuda.empty_cache()


def part_stress(runs, n):
    settings = [("clean", None, 0)] + [(k, k, s) for k in ("gaussian_noise", "gaussian_blur", "crop") for s in (1, 3, 5)]
    rows = []
    for tag, ctag in runs:
        cfg, ms = models_for(tag, ctag, [STRESS_EPOCH])
        if not ms: continue
        for name, kind, sev in settings:
            st = None if kind is None else Stress(kind, sev, cfg["seed"])
            P, Y, S = predict_all(cfg, ms, n, st); p = P[STRESS_EPOCH]
            orig = p[np.arange(len(Y)), S]
            flip9 = np.mean([(p[i, c] != orig[i]) for i in range(len(Y)) for c in range(10) if c != S[i]])
            rows.append(dict(mode=tag, corr=ctag, stress=name, severity=sev, acc=float((orig == Y).mean()), flip9=float(flip9)))
            print(tag, ctag, name, sev, rows[-1]["acc"], rows[-1]["flip9"], flush=True)
        del ms; torch.cuda.empty_cache()
    df = pd.DataFrame(rows); df.to_csv(OUT / "stress_epoch50.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), sharex=True)
    for k, kind in enumerate(("gaussian_noise", "gaussian_blur", "crop")):
        for r, (m, lab) in enumerate((("acc", "test accuracy (%)"), ("flip9", "9-color flip (%)"))):
            ax = axes[r, k]
            for tag, _, _ in MODES:
                for ctag, corr in CORRS:
                    g = df[(df["mode"] == tag) & (df["corr"] == ctag)]
                    if g.empty: continue
                    g = pd.concat([g[g.stress == "clean"], g[g.stress == kind]]).sort_values("severity")
                    ax.plot(g.severity, g[m] * 100, color=CORR_COLOR[ctag], ls="-" if tag == "bg" else "--", marker="o", ms=3,
                            label=f"{NAMES[tag]} {corr:.2f}")
            style(ax, kind.replace("_", " ") if r == 0 else None, "severity (0 = clean)" if r == 1 else None, lab if k == 0 else None)
            ax.set_xticks([0, 1, 3, 5])
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(f"Stress test at epoch 50 ({n} test digits; corruption applied after painting, same for every color)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "11_stress_epoch50")


def gradcam(model, x, layer="features.25"):
    acts = {}
    h = dict(model.named_modules())[layer].register_forward_hook(lambda mod, i, o: acts.update(a=o))
    with torch.enable_grad():
        logits = model(x); p = logits.argmax(1)
        g = torch.autograd.grad(logits.gather(1, p[:, None]).sum(), acts["a"])[0]
    h.remove()
    cam = (g.mean((2, 3), keepdim=True) * acts["a"].detach()).sum(1, keepdim=True).clamp(min=0)
    return p.cpu().numpy(), F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0].cpu().numpy()


def part_cam(runs, n):
    eps = CAM_EPOCHS; rows = []
    for tag, ctag in runs:
        cfg, ms = models_for(tag, ctag, eps)
        if not ms: continue
        ds = se.build_datasets(cfg)["test"]; ds.counterfactual = False
        idx = np.flatnonzero(ds.labels != ds.spurious)[:n]
        X = torch.stack([ds[int(i)]["pixel_values"] for i in idx]); Y, S = ds.labels[idx], ds.spurious[idx]
        gray = torch.stack([torch.from_numpy(np.asarray(ds.base[int(ds.indices[i])]["image"], np.float32) / 255) for i in idx])
        digit = F.interpolate(gray[:, None], size=X.shape[-2:], mode="bilinear", align_corners=False)[:, 0].numpy() > 0.1
        keep = {}
        for e, m in ms.items():
            P, A = [], []
            for j in range(0, len(X), 50):
                p, a = gradcam(m, X[j:j + 50].to(dev)); P.append(p); A.append(a)
            P, A = np.concatenate(P), np.concatenate(A); t = A.sum((1, 2))
            share = np.where(t > 0, (A * digit).sum((1, 2)) / np.maximum(t, 1e-12), np.nan)
            for lab, sel in (("correct digit", P == Y), ("color's class", P == S)):
                if sel.sum() >= 5:
                    rows.append(dict(mode=tag, corr=ctag, epoch=e, predicted=lab, n=int(sel.sum()), digit_share=float(np.nanmean(share[sel])),
                                     digit_area=float(digit[sel].mean())))
            keep[e] = (P[:6], A[:6])
        # grid: 6 conflicting samples x epochs
        fig, axes = plt.subplots(6, len(keep) + 1, figsize=(1.75 * (len(keep) + 1), 1.8 * 6))
        for r in range(6):
            img = to_image(X[r]); axes[r, 0].imshow(img); axes[r, 0].set_title(f"digit {Y[r]}, color->{S[r]}", fontsize=7, color=INK)
            g = img.mean(-1, keepdims=True).repeat(3, -1)
            for k, (e, (p, a)) in enumerate(keep.items(), 1):
                an = (a[r] - a[r].min()) / max(a[r].max() - a[r].min(), 1e-12)
                axes[r, k].imshow(np.clip(0.5 * g + 0.5 * plt.cm.jet(an)[..., :3], 0, 1))
                sh = (a[r] * digit[r]).sum() / max(a[r].sum(), 1e-12)
                col = "#1a7f37" if p[r] == Y[r] else ("#cf222e" if p[r] == S[r] else MUTED)
                axes[r, k].set_title(f"ep{e}: pred {p[r]}, on digit {sh * 100:.0f}%", fontsize=6.5, color=col)
        for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{NAMES[tag]} corr 0.{ctag[1:]}: Grad-CAM (predicted class, VGG features.25) on color-conflicting test digits\n"
                     "title: green = correct, red = predicted the color's class; % = share of the map on the digit strokes",
                     fontsize=8, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, f"12_gradcam_grid_{tag}{ctag}")
        del ms; torch.cuda.empty_cache()
    df = pd.DataFrame(rows); df.to_csv(OUT / "gradcam_digit_share.csv", index=False)
    plot_cam_share(df, eps)



def part_stressgrid(runs, digit=7):
    """Example images for the stress test: one test digit in all 10 colors under every corruption, with epoch-50 predictions."""
    settings = [("clean", None, 0)] + [(k, k, v) for k in ("gaussian_noise", "gaussian_blur", "crop") for v in (1, 3, 5)]
    for tag, ctag in runs:
        cfg, ms = models_for(tag, ctag, [STRESS_EPOCH])
        if not ms: continue
        m = ms[STRESS_EPOCH]
        ds = se.build_datasets(cfg)["test"]; ds.counterfactual = False
        i = int(np.flatnonzero(ds.labels == digit)[0]); img = ds.base[int(ds.indices[i])]["image"]
        fig, axes = plt.subplots(len(settings), 10, figsize=(11, 1.2 * len(settings)))
        for r, (name, kind, sev) in enumerate(settings):
            st = None if kind is None else Stress(kind, sev, cfg["seed"])
            ds.stress = st
            X = torch.stack([ds.paint(img, c, i) for c in range(10)])
            with torch.no_grad(): p = m(X.to(dev)).argmax(1).cpu().numpy()
            for c in range(10):
                ax = axes[r, c]; ax.imshow(to_image(X[c])); ax.set_xticks([]); ax.set_yticks([])
                state = "ok" if p[c] == digit else ("color" if p[c] == c else "other")
                col = {"ok": "#1a7f37", "color": "#cf222e", "other": MUTED}[state]
                ax.set_title(f"{p[c]}", fontsize=7, color=col, pad=1)
                for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(2.2 if state != "ok" else 0.6)
            label = "clean" if st is None else f"{kind.replace('gaussian_', '')} {sev}\n({st.describe(cfg['image_size'])})"
            axes[r, 0].set_ylabel(label, fontsize=7, color=INK, rotation=0, ha="right", va="center")
        ds.stress = None
        fig.suptitle(f"{NAMES[tag]} corr 0.{ctag[1:]}, epoch {STRESS_EPOCH}: test digit {digit} in all 10 colors under each corruption; "
                     "prediction above each image\ngreen = correct, red = the painted color's class, gray = another digit",
                     fontsize=9, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0.07, 0, 1, 0.96)); save(fig, f"14_stress_examples_{tag}{ctag}")
        del ms; torch.cuda.empty_cache()


def plot_cam_share(df, eps, min_n=20):
    """Figure 13 from gradcam_digit_share.csv; points with fewer than min_n images are dropped."""
    if df.empty: return
    df = df.copy(); df["corr"] = df["corr"].astype(str).str.zfill(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for k, (tag, _, mname) in enumerate(MODES):
        ax = axes[k]
        for ctag, corr in CORRS:
            for lab, ls in (("correct digit", "-"), ("color's class", "--")):
                g = df[(df["mode"] == tag) & (df["corr"] == ctag) & (df.predicted == lab)]
                g = g[g.n >= min_n]
                if len(g): ax.plot(g.epoch, g.digit_share * 100, color=CORR_COLOR[ctag], ls=ls, marker="o", ms=3,
                                   label=f"corr {corr:.2f}, predicted {lab}")
        area = df[df["mode"] == tag].digit_area.mean()
        if not np.isnan(area): ax.axhline(area * 100, color=MUTED, lw=1, ls=":"); ax.text(50, area * 100, " digit area", fontsize=7, color=MUTED, va="center")
        ax.set_xscale("symlog", linthresh=3); ax.set_xticks(eps); ax.set_xticklabels(eps)
        style(ax, f"{mname}: Grad-CAM share on the digit strokes", "epoch", "% of the map" if k == 0 else None)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Where Grad-CAM points, split by what the model predicted (BG: color is in the background; FG: color is on the strokes)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save(fig, "13_gradcam_digit_share")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--runs", default="bg095,fg095,bg080,fg080,bg065,fg065")
    ap.add_argument("--parts", default="heat,grid,stress,cam")
    a = ap.parse_args()
    runs = [(r[:2], r[2:]) for r in a.runs.split(",")]
    parts = a.parts.split(",")
    if "heat" in parts: part_heat(runs, a.n)
    if "grid" in parts: part_grid([r for r in runs if r[1] in ("095", "065")])
    if "stress" in parts: part_stress(runs, a.n)
    if "cam" in parts: part_cam(runs, min(a.n, 300))
    if "stressgrid" in parts: part_stressgrid(runs)
    if "camplot" in parts: plot_cam_share(pd.read_csv(OUT / "gradcam_digit_share.csv"), CAM_EPOCHS)
