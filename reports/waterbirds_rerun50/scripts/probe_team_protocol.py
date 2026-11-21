"""Usage: python3 probe_team_protocol.py  (CPU, ~10 min; writes seeds/probe_team_protocol.csv)

Apply the team's probe protocol (scripts/waterbirds_followup/probes.py: content-disjoint 56/56 split,
DecodedCorrelation linear probe) to the rerun seed 42/43/44 last-conv features."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from scripts.waterbirds_followup.probes import split_content, probe
torch.set_num_threads(4)
man = ROOT / "data/waterbirds_rerun50/manifest"; meta = json.loads((man / "protocol.json").read_text())
rows = []
for seed in (42, 43, 44):
    for code in ("095", "080", "065"):
        run = ROOT / f"runs/waterbirds_rerun50/A_corr{code}_seed{seed}_e50_contentv3"
        frame = pd.read_csv(man / f"A_corr{code}.csv"); val = frame[frame.split == "val"]
        base = np.load(run / "features/epoch_000.npz"); ids = base["indices"]
        (fit, score), _, _ = split_content(meta, val, ids)
        li = list(base["layers"]).index("features.28")
        for ep in range(51):
            f = np.load(run / f"features/epoch_{ep:03d}.npz"); assert np.array_equal(f["indices"], ids)
            x = f[f"f{li}"].astype(np.float32)
            for target, key in (("core", "labels"), ("spurious", "spurious")):
                r = probe(x, f[key].astype(np.int64), fit, score)
                rows.append(dict(seed=seed, corr=code, epoch=ep, target=target, acc=r["accuracy"], fit_n=len(fit), score_n=len(score)))
        print(seed, code, "fit/score", len(fit), len(score), flush=True)
d = pd.DataFrame(rows); d.to_csv(ROOT / "reports/waterbirds_rerun50/seeds/probe_team_protocol.csv", index=False)
print((d[d.epoch >= 31].pivot_table(index=["target", "corr"], columns="seed", values="acc") * 100).round(1))
print((d[d.epoch == 50].pivot_table(index=["target", "corr"], columns="seed", values="acc") * 100).round(1))
