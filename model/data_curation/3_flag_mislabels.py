"""Step 3 of curation: flag images whose label looks wrong, so you can review them by hand.

Idea: train a simple classifier on the image embeddings using cross-validation, where an
image is NEVER predicted by a model that saw its own duplicate cluster. Images the model
confidently assigns to a different class than their label are the likeliest mislabels.

Outputs (in data/work):
  review_candidates.csv   every kept image, most suspicious first, with an empty `action` column
  review_sheets/<class>.png   contact sheets of the most suspicious images per class

How to review: open the sheets, then fill the `action` column in review_candidates.csv for rows
you checked:   drop   |   relabel:<shared label>   |   keep
Step 4 applies whatever you filled in. Rows you leave empty are left alone.
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

import config as C


def contact_sheet(rows, out, cols=6, size=192):
    n = len(rows)
    nrows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, nrows * (size + 14)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, row in enumerate(rows.itertuples()):
        x, y = (k % cols) * size, (k // cols) * (size + 14)
        try:
            with Image.open(row.path) as im:
                im = im.convert("RGB")
                im.thumbnail((size, size))
                sheet.paste(im, (x, y + 14))
        except Exception:
            pass
        draw.text((x + 2, y + 1), f"#{row.review_id} ->{row.predicted} {row.p_given:.2f}", fill="black")
    sheet.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", type=int, default=48, help="images per class on each contact sheet")
    args = ap.parse_args()

    df = pd.read_csv(C.WORK / "manifest_clustered.csv")
    E = np.load(C.WORK / "embeddings.npy")
    mask = (df.keep & ~df.label_conflict).to_numpy()
    sub = df[mask].reset_index(drop=True)
    X = E[mask]

    classes = sorted(sub.mapped_label.unique())
    y = sub.mapped_label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    groups = sub.cluster_id.to_numpy()

    oof = np.zeros((len(sub), len(classes)))
    n_splits = max(3, min(5, int(pd.Series(groups).groupby(y).nunique().min())))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=C.SEED)
    for tr, te in cv.split(X, y, groups):
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        oof[np.ix_(te, clf.classes_)] = clf.predict_proba(X[te])

    sub["predicted"] = [classes[i] for i in oof.argmax(1)]
    sub["p_given"] = oof[np.arange(len(sub)), y]
    sub["p_pred"] = oof.max(1)
    sub["suspicion"] = 1 - sub.p_given
    sub["flagged"] = ((sub.predicted != sub.mapped_label) & (sub.p_pred >= 0.7)) | (sub.p_given < 0.1)

    try:  # optional second opinion
        from cleanlab.filter import find_label_issues
        issues = find_label_issues(labels=y, pred_probs=oof)
        sub["cleanlab_issue"] = issues
    except ImportError:
        sub["cleanlab_issue"] = np.nan

    sub = sub.sort_values("suspicion", ascending=False).reset_index(drop=True)
    sub["review_id"] = sub.index
    sub["action"] = ""
    cols = ["review_id", "path", "source", "original_label", "mapped_label", "predicted",
            "p_given", "p_pred", "flagged", "cleanlab_issue", "action"]
    sub[cols].to_csv(C.WORK / "review_candidates.csv", index=False)

    sheets = C.WORK / "review_sheets"
    sheets.mkdir(exist_ok=True)
    for c in classes:
        contact_sheet(sub[sub.mapped_label == c].head(args.sheet), sheets / f"{c}.png")

    print(f"\n{len(sub)} images scored ({n_splits}-fold, grouped by cluster).")
    print(f"Flagged as likely mislabeled: {int(sub.flagged.sum())}")
    print(sub[sub.flagged].groupby(["source", "mapped_label"]).size().to_string() or "none")
    print(f"\nReview: {C.WORK / 'review_sheets'}  then fill `action` in {C.WORK / 'review_candidates.csv'}")


if __name__ == "__main__":
    main()
