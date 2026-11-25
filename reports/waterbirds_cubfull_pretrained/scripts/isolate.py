"""Bird alone vs scene alone, every epoch of a Waterbirds run (evaluation only, GPU).

For each validation image (1,199, the team val split, the same for cubfull runs):
  bird only   the same bird pasted on a plain gray image (124, 116, 104 = ImageNet mean colour, ~0 after normalisation)
  scene only  the same scene with no bird (cropped and resized as when the bird is pasted)
Per epoch 1-50:
  bird_only_bal   = balanced accuracy of the class prediction on the bird alone (chance 50%)
  scene_place_acc = share of scenes alone predicted as their place's class (land -> landbird, water -> waterbird)
Rows for (suffix, seed) are replaced in reports/waterbirds_cubfull_pretrained/isolate.csv.
Usage (from the repo root):
  python3 reports/waterbirds_cubfull_pretrained/scripts/isolate.py                      # final setting
  python3 reports/waterbirds_cubfull_pretrained/scripts/isolate.py --suffix ""          # original team runs
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "reports" / "core_flip" / "scripts"))
import core_flip as cf  # noqa: E402  (waterbirds_val, build, run folders)
import shortcut_experiment as se  # noqa: E402
from data import combine_and_mask, crop_and_resize, to_tensor  # noqa: E402

OUT = REPO / "reports" / "waterbirds_cubfull_pretrained" / "isolate.csv"
GRAY = (124, 116, 104)
FINAL = "_cubfull_pretrained_lr0.01_multiscene"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default=FINAL, help="run-folder suffix after A_corr*_seed*_e50_contentv3")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rows = []
    for ctag, corr in cf.CORRS:
        ds = cf.waterbirds_val(corr); y, s = ds.labels, ds.spurious

        def make(k):
            x, (seg, bird) = ds.composite(k, int(s[k]))
            scene = Image.open(ds.places_dir / ds.scenes[k, int(s[k])]).convert("RGB")
            bird_only = combine_and_mask(Image.new("RGB", scene.size, GRAY), seg, bird)
            return x, to_tensor(ds, bird_only, k), to_tensor(ds, crop_and_resize(scene, bird), k)
        X = cf.build(make, len(y))
        run = cf.WB_RUNS / f"A_corr{ctag}_seed{a.seed}_e50_contentv3{a.suffix}"
        for e in range(1, 51):
            saved = se.load_checkpoint(run / "checkpoints" / f"epoch_{e:03d}.pt", cf.dev)
            m = se.build_model({**saved["config"], "pretrained": False}).to(cf.dev); m.load_state_dict(saved["model"]); m.eval()
            P = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                for i in range(0, len(X), 64): P.append(m(X[i:i + 64].flatten(0, 1)).argmax(1).view(-1, 3).cpu())
            P = torch.cat(P).numpy()
            rows.append(dict(corr=ctag, seed=a.seed, suffix=a.suffix, epoch=e,
                             bird_only_bal=np.mean([(P[y == c, 1] == c).mean() for c in (0, 1)]),
                             scene_place_acc=(P[:, 2] == s).mean()))
        print(run.name, "done", flush=True)
        del X; torch.cuda.empty_cache()
    new = pd.DataFrame(rows)
    if OUT.exists():
        old = pd.read_csv(OUT, dtype={"corr": str, "suffix": str}, keep_default_na=False)
        new = pd.concat([old[~((old.suffix == a.suffix) & (old.seed == a.seed))], new])
    new.to_csv(OUT, index=False); print("wrote", OUT)


if __name__ == "__main__":
    main()
