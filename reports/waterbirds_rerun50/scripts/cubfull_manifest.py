"""Waterbirds Condition A manifest with ~2x training birds: most of the CUB test split moves into train.

Starts from the team manifest (data/waterbirds_rerun50/manifest, protocol v3) and writes data/waterbirds_rerun50/manifest_cubfull:
  * val: unchanged (same 1,199 birds, places and scenes), so validation metrics, probes and flips stay comparable.
  * test: 1,199 CUB-test birds with the val group counts (landbird 463 land / 462 water, waterbird 137 / 137),
          keeping their team places and scenes.
  * train: the team's 4,795 train rows unchanged + the other 4,595 CUB-test birds (9,390 in all, 22.5% waterbird).
          Places of the added birds are drawn per class so the whole train split meets corr exactly
          (|aligned - corr * n| <= 0.5, the team check). Their scenes come from the train scene pool
          (scene_group_split == 'train' in split_audit.json): first the train-pool scenes no train bird uses yet,
          then cycling over the whole train pool ("within-split cycling allowed", the team scene policy).
The team loader validates the result (no cross-split CUB/Places content, exact correlations) before writing.
Usage (from the repo root): python3 reports/waterbirds_rerun50/scripts/cubfull_manifest.py
"""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.waterbirds_vgg50.protocol import CORRS, digest, validate_frame  # noqa: E402
import data  # noqa: E402

SRC = REPO / "data" / "waterbirds_rerun50" / "manifest"
DST = REPO / "data" / "waterbirds_rerun50" / "manifest_cubfull"
SEED = 42
KINDS = {0: ("b/bamboo_forest/", "f/forest/broadleaf/"), 1: ("o/ocean/", "l/lake/natural/")}


def main():
    meta = json.loads((SRC / "protocol.json").read_text()); audit = json.loads((SRC / "split_audit.json").read_text())
    frames = {c: pd.read_csv(SRC / f"A_corr{round(c * 100):03d}.csv") for c in CORRS}
    base = frames[CORRS[0]]
    for f in frames.values():  # val/test places and all scenes are the same in every correlation file
        assert (f[["img_id", "split", "land_file", "water_file"]].values == base[["img_id", "split", "land_file", "water_file"]].values).all()
        assert (f.loc[f.split != "train", "place"].values == base.loc[base.split != "train", "place"].values).all()
    rng = np.random.default_rng(SEED)

    # test: val group counts, drawn from the CUB test birds with their team places
    val = base[base.split == "val"]; test = base[base.split == "test"]; keep = []
    for (y, s), n in val.groupby(["y", "place"]).size().items():
        ids = test[(test.y == y) & (test.place == s)].img_id.to_numpy()
        keep += list(rng.choice(ids, size=n, replace=False))
    keep = set(int(i) for i in keep)
    moved = base[(base.split == "test") & ~base.img_id.isin(keep)].img_id.to_numpy()

    # scenes for the added birds from the train scene pool (one representative per content hash)
    prefix = "places365/" + data.PLACES_PREFIX + "/"
    train_hash = {h for h, s in audit["scene_group_split"].items() if s == "train"}
    pool, seen = {0: [], 1: []}, set()
    for path, h in sorted(meta["assets"].items()):
        if not path.startswith(prefix) or h not in train_hash or h in seen: continue
        rel = path[len(prefix):]
        for kind, pre in KINDS.items():
            if rel.startswith(pre): pool[kind].append(rel); seen.add(h)
    used = {0: set(base.loc[base.split == "train", "land_file"]), 1: set(base.loc[base.split == "train", "water_file"])}
    scenes = {}
    for kind in (0, 1):
        unused = [p for p in pool[kind] if p not in used[kind]]
        order = list(rng.permutation(unused)) + list(rng.permutation(pool[kind]))
        while len(order) < len(moved): order += list(rng.permutation(pool[kind]))
        scenes[kind] = order[:len(moved)]
    print("train scene pool (land, water):", len(pool[0]), len(pool[1]), "| not yet used by train birds:",
          sum(p not in used[0] for p in pool[0]), sum(p not in used[1] for p in pool[1]))

    DST.mkdir(parents=True, exist_ok=False)
    out_meta = copy.deepcopy(meta); out_meta["files"] = {}
    out_meta["condition_name"] = meta.get("condition_name", "") + " | cubfull: CUB test birds moved into train (val unchanged)"
    out_meta["derived_from"] = dict(manifest=str(SRC / "protocol.json"), sha256=digest(SRC / "protocol.json"), seed=SEED,
                                    script="reports/waterbirds_rerun50/scripts/cubfull_manifest.py")
    for c, f in frames.items():
        f = f.copy(); f = f.set_index("img_id")
        f.loc[moved, "split"] = "train"
        f.loc[moved, "land_file"] = scenes[0]; f.loc[moved, "water_file"] = scenes[1]
        r = np.random.default_rng(SEED + round(c * 100))
        for y in (0, 1):
            old = f[(f.split == "train") & (f.y == y) & ~f.index.isin(moved)]
            new = f.loc[moved][f.loc[moved].y == y].index.to_numpy()
            target = round(c * (len(old) + len(new))) - int((old.place == y).sum())
            aligned = set(r.choice(new, size=target, replace=False).tolist())
            f.loc[new, "place"] = [y if i in aligned else 1 - y for i in new]
        f = f.reset_index()[base.columns]
        reports = validate_frame(f, meta["assets"], c)
        name = f"A_corr{round(c * 100):03d}.csv"; f.to_csv(DST / name, index=False)
        out_meta["files"][name] = dict(sha256=digest(DST / name), splits=reports)
        print(name, {s: (r["n"], r["groups"]) for s, r in reports.items()})
    out_meta["final_counts"] = {s: dict(n=int((base.img_id.isin(keep) if s == "test" else (base.split == s) | (base.img_id.isin(moved) & (s == "train"))).sum()))
                                for s in ("train", "val", "test")}
    (DST / "protocol.json").write_text(json.dumps(out_meta, indent=2))
    print("written", DST)


if __name__ == "__main__":
    main()
