"""Example predictions, Grad-CAM and stress test for the pretrained + all CUB + B runs and the original runs (GPU).

Reuses reports/waterbirds_rerun50/scripts/figs_checkpoints.py (same Grad-CAM layer, stress settings and plots), with one
necessary change: the test images. cubfull moved 4,595 of the 5,794 team test birds into train, so the figures use only
the 1,199-bird cubfull test split. It is a subset of the team test with the team's places and scenes, and neither the
original nor the cubfull runs trained on any of it, so both settings are evaluated on identical images.
Balanced subset: 137 images per bird x background group (all waterbirds, random landbirds, seed 0).
Example birds are random picks (fixed seed) among birds of typical size (mask area within the middle 50% of this test
split), the same for both settings; none is chosen by its prediction. The numbers use all 548 images, any bird size.
The Grad-CAM grids show 3 waterbirds on land and 3 landbirds on water.
Writes figures/05-09 (*_original = original runs, *_cosine = cosine runs), gradcam_bird_share*.csv and stress_epoch50*.csv.
Usage (from the repo root): python3 reports/waterbirds_cubfull_pretrained/scripts/examples.py [--parts swapgrid,cam,stress,stressgrid]
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "reports" / "waterbirds_cubfull_pretrained"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "reports" / "waterbirds_rerun50" / "scripts"))
import figs_checkpoints as fc  # noqa: E402

CTAGS = ["095", "080", "065"]
N = 137  # per group: the number of test waterbirds on each background
SETTINGS = [("", "_cubfull_pretrained_lr0.01_multiscene", "pretrained + all CUB + B"), ("_original", "", "original, from scratch"),
            ("_cosine", "_cubfull_pretrained_lr0.01_cosine_multiscene", "pretrained + all CUB + B, cosine")]
_sets, _subset, _build = {}, fc.subset, fc.se.build_model


def test_set(ctag):
    if ctag not in _sets:
        _sets[ctag] = fc.load(fc.lp.DATA / "manifest_cubfull" / "protocol.json", dict(fc.CORRS)[ctag], verify=False)["test"]
    return _sets[ctag]


def typical(ds):
    """Test birds whose mask area (share of the image) is within the middle 50% of the split."""
    if "typical" not in _sets:
        dl = DataLoader(Areas(ds), batch_size=64, num_workers=4); A = np.concatenate([b.numpy() for b in dl])
        lo, hi = np.percentile(A, [25, 75]); _sets["typical"] = (A >= lo) & (A <= hi)
        print(f"bird area: median {np.median(A):.3f}, example birds within [{lo:.3f}, {hi:.3f}]", flush=True)
    return _sets["typical"]  # same birds in the test split of every corr


class Areas(Dataset):
    def __init__(self, ds): self.pairs = fc.Pairs(ds, np.arange(len(ds)))
    def __len__(self): return len(self.pairs)
    def __getitem__(self, k): return float((self.pairs[k][4] > 0.5).float().mean())


def pick_birds(ds, per_group=1, seed=3):
    rng = np.random.default_rng(seed); ok = typical(ds)
    return [i for y, s in ((1, 0), (0, 1), (1, 1), (0, 0))
            for i in rng.permutation(np.flatnonzero((ds.labels == y) & (ds.spurious == s) & ok))[:per_group]]


def cam_subset(ds, n, seed=0):
    """Same balanced subset, reordered so the six grid rows are 3 random waterbirds on land, then 3 landbirds on water."""
    idx = _subset(ds, n, seed); rng = np.random.default_rng(1); ok = typical(ds)[idx]
    first = [i for y in (1, 0) for i in rng.permutation(idx[(ds.labels[idx] == y) & (ds.spurious[idx] == 1 - y) & ok])[:3]]
    return np.array(first + [i for i in idx if i not in set(first)])


def use(tag, suffix, label):
    fc.test_set = test_set; fc.pick_birds = pick_birds
    fc.se.build_model = lambda cfg: _build({**cfg, "pretrained": False})  # weights come from the checkpoint
    fc.run_name = lambda ctag, seed=None: f"A_corr{ctag}_seed42_e50_contentv3{suffix}"
    fc.out_csv = lambda name: OUT / name.replace(".csv", f"{tag}.csv")

    def save(fig, name, force=False):
        (OUT / "figures").mkdir(parents=True, exist_ok=True)
        t = fig._suptitle
        if t is not None: t.set_text(f"[{label}] " + t.get_text())
        else: fig.axes[0].set_title(f"[{label}] " + fig.axes[0].get_title(loc="left"), fontsize=8, loc="left")
        fig.savefig(OUT / "figures" / f"{name}{tag}.png", dpi=130, bbox_inches="tight"); plt.close(fig)
        print("saved", f"{name}{tag}.png", flush=True)
    fc.save = save


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", default="swapgrid,cam,stress,stressgrid")
    ap.add_argument("--settings", default="final,original,cosine")
    a = ap.parse_args(); parts = a.parts.split(",")
    for tag, suffix, label in [x for x in SETTINGS if (x[0].strip("_") or "final") in a.settings.split(",")]:
        use(tag, suffix, label)
        if "swapgrid" in parts: fc.part_swapgrid(CTAGS)
        if "cam" in parts:
            fc.subset = cam_subset; fc.part_cam(CTAGS, N); fc.subset = _subset
        if "stress" in parts: fc.part_stress(CTAGS, N)
        if "stressgrid" in parts: fc.part_stressgrid(CTAGS)


if __name__ == "__main__":
    main()
