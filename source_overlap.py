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

    



