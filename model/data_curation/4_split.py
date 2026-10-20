"""Step 4 for curation: applies review decisions, splits by duplicate cluster, reports, and exports.

Rules:
  - duplicate copies (keep == False) and label-conflict images are removed
  - your review decisions (drop / relabel) from review_candidates.csv are applied
  - if a holdout source is set, every cluster touching it is removed from train/val/test, and the
    holdout source becomes `test_external` (a genuinely independent test set)
  - the rest is split train/val/test with StratifiedGroupKFold: classes stay balanced,
    and a duplicate cluster never straddles two splits

Outputs: data/curated/curated.csv (the final metadata table), and with --export an ImageFolder tree.

Usage:
  python 04_split.py
  python 04_split.py --holdout roboflow_b --export
"""
import argparse
import os
import shutil

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config as C


def apply_review(df):
    path = C.WORK / "review_candidates.csv"
    if not path.exists():
        print("No review_candidates.csv found: skipping review decisions.")
        return df
    rev = pd.read_csv(path)
    rev["action"] = rev["action"].fillna("").astype(str).str.strip().str.lower()
    rev = rev[rev.action != ""]
    drop, relabel = set(), {}
    for r in rev.itertuples():
        if r.action == "drop":
            drop.add(r.path)
        elif r.action.startswith("relabel:"):
            new = r.action.split(":", 1)[1].strip()
            if new not in C.SHARED_LABELS:
                raise SystemExit(f"Bad relabel target '{new}' for {r.path}. Use one of {C.SHARED_LABELS}")
            relabel[r.path] = new
        elif r.action != "keep":
            print(f"  ignoring unknown action '{r.action}' for {r.path}")
    df = df[~df.path.isin(drop)].copy()
    df["relabeled"] = df.path.isin(relabel)
    df["mapped_label"] = [relabel.get(p, l) for p, l in zip(df.path, df.mapped_label)]
    print(f"Review decisions applied: {len(drop)} dropped, {len(relabel)} relabeled, "
          f"{int((rev.action == 'keep').sum())} confirmed.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=C.HOLDOUT_SOURCE, help="source to keep as external test set")
    ap.add_argument("--export", action="store_true", help="copy images into data/curated/<split>/<class>/")
    ap.add_argument("--link", action="store_true", help="with --export: symlink instead of copy")
    args = ap.parse_args()

    full = pd.read_csv(C.WORK / "manifest_clustered.csv")
    df = full[full.keep & ~full.label_conflict].copy()
    print(f"{len(full)} images -> {len(df)} after duplicates and conflicts removed")
    df = apply_review(df)

    ext = pd.DataFrame()
    if args.holdout:
        if args.holdout not in set(full.source):
            raise SystemExit(f"Holdout source '{args.holdout}' not found. Sources: {sorted(set(full.source))}")
        touched = set(full.loc[full.source == args.holdout, "cluster_id"])
        ext = df[df.source == args.holdout].copy()
        ext["split"] = "test_external"
        contaminated = df[df.cluster_id.isin(touched) & (df.source != args.holdout)]
        print(f"Holdout '{args.holdout}': {len(ext)} images; removed {len(contaminated)} images from other "
              f"sources that are near-duplicates of it")
        df = df[~df.cluster_id.isin(touched)].copy()

    clusters_per_class = df.groupby("mapped_label").cluster_id.nunique()
    n_splits = max(3, min(10, int(clusters_per_class.min())))
    if clusters_per_class.min() < 10:
        print(f"WARNING: smallest class has only {int(clusters_per_class.min())} clusters; "
              f"using {n_splits} folds, so val/test are ~{100 // n_splits}% each and may be noisy.")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=C.SEED)
    folds = [te for _, te in cv.split(df, df.mapped_label, df.cluster_id)]
    df["split"] = "train"
    df.iloc[folds[0], df.columns.get_loc("split")] = "test"
    df.iloc[folds[1], df.columns.get_loc("split")] = "val"

    out = pd.concat([df, ext]) if len(ext) else df
    C.OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(C.OUT / "curated.csv", index=False)

    print("\nImages per split x class:")
    print(pd.crosstab(out.mapped_label, out.split, margins=True))
    print("\nImages per class x source (check no class comes mostly from one source):")
    xs = pd.crosstab(out.mapped_label, out.source, margins=True)
    print(xs)
    share = pd.crosstab(out.mapped_label, out.source, normalize="index")
    for cls, row in share.iterrows():
        if row.max() > 0.7:
            print(f"WARNING: {cls}: {row.max():.0%} of images come from '{row.idxmax()}'. "
                  f"The model may learn that source's look instead of the condition.")

    if args.export:
        for r in out.itertuples():
            dest = C.OUT / r.split / r.mapped_label
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / f"{r.source}_{r.sha256[:8]}_{r.path.split('/')[-1]}"
            if args.link:
                if not target.exists():
                    target.symlink_to(os.path.abspath(r.path))
            else:
                shutil.copy2(r.path, target)
        print(f"\nExported to {C.OUT}/<split>/<class>/")
    print(f"Final table: {C.OUT / 'curated.csv'}")


if __name__ == "__main__":
    main()
