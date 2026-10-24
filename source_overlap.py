"""After step 2, this creates a per-source overlap table, to choose the external holdout source.

Choosing a holdout is a trade-off. In step 4 the holdout source becomes an independent test set, and
every image from OTHER sources that sits in a duplicate cluster with it is removed from training.
A heavily overlapping source is therefore expensive to hold out, and a source with one or two classes
makes a poor test set. This table shows both costs.

Usage:
  python source_overlap.py
"""
import pandas as pd

import config as C

def main():
    df = pd.read_csv(C.WORK / "manifest_clustered.csv")
    kept = df[df.keep & ~df.label_conflict]
    by_source = {s: set(kept[kept.source == s].cluster_id) for s in df.source.unique()}

    rows = []
    for s in sorted(df.source.unique()):
            k = kept[kept.source == s]
            cost = int(kept[kept.source != s].cluster_id.isin(by_source[s]).sum())
            rows.append({
                "source": s,
                "raw": int((df.source == s).sum()),
                "dup_copies_removed": int(((df.source == s) & ~df.keep).sum()),
                "label_conflicts": int(((df.source == s) & df.label_conflict).sum()),
                "kept": len(k),
                "classes": k.mapped_label.nunique(),
                "pct_overlapping_other_sources": round(100 * (k.cluster_n_sources > 1).mean(), 1) if len(k) else 0.0,
                "train_images_lost_if_holdout": cost,
            })
    table = pd.DataFrame(rows).set_index("source")
    

    



