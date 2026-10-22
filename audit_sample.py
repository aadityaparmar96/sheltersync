#Blind Label Audit
import argparse
import shutil

import pandas as pd
from PIL import Image, ImageDraw

import config as C

CURATED = C.OUT / "curated.csv"
BACKUP = C.OUT / "curated_before_audit.csv"
PRIVATE = C.OUT / "audit_sample.csv"
REVIEWER = C.OUT / "audit_for_reviewer.csv"
SHEETS = C.OUT / "audit_sheets"
EXTRA = {"unclear", "not_dog"}

def make_sheet(rows, out, cols=6, size=200):
    nrows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, nrows * (size + 16)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, r in enumerate(rows.itertuples()):
        x, y = (k % cols) * size, (k // cols) * (size + 16)
        try:
            with Image.open(r.path) as im:
                im = im.convert("RGB")
                im.thumbnail((size, size))
                sheet.paste(im, (x, y + 16))
        except Exception:
            pass
        draw.text((x + 3, y + 2), f"#{r.audit_id}", fill="black")
    sheet.save(out)


def cmd_make(args):
    df = pd.read_csv(CURATED)
    pool = df[df.split.isin(args.splits)]
    if pool.empty:
        raise SystemExit(f"No rows in splits {args.splits}. Run step 4 first.")
    parts = [g.sample(min(len(g), args.n_per_class), random_state=C.SEED) for _, g in pool.groupby("mapped_label")]
    s = pd.concat(parts).sample(frac=1, random_state=C.SEED).reset_index(drop=True)  # shuffled: blind
    s["audit_id"] = range(len(s))
    s = s.rename(columns={"mapped_label": "current_label"})
    s[["audit_id", "path", "source", "split", "current_label"]].to_csv(PRIVATE, index=False)
    pd.DataFrame({"audit_id": s.audit_id, "vet_label": "", "vet_note": ""}).to_csv(REVIEWER, index=False)
    SHEETS.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(s), args.per_sheet):
        make_sheet(s.iloc[i:i + args.per_sheet], SHEETS / f"sheet_{i // args.per_sheet + 1:02d}.png")
    print(f"Sampled {len(s)} images ({s.current_label.value_counts().to_dict()})")
    print(f"Give the reviewer: {SHEETS}/ and {REVIEWER}. Keep {PRIVATE} to yourself.")
    print(f"Allowed vet_label values: {C.SHARED_LABELS + sorted(EXTRA)}")


def cmd_apply(args):
    sample = pd.read_csv(PRIVATE)
    rev = pd.read_csv(REVIEWER).fillna("")
    rev["vet_label"] = rev.vet_label.astype(str).str.strip().str.lower()
    bad = rev[(rev.vet_label != "") & ~rev.vet_label.isin(set(C.SHARED_LABELS) | EXTRA)]
    if len(bad):
        print(f"Ignoring {len(bad)} rows with invalid vet_label values: {sorted(set(bad.vet_label))}")
    rev = rev[rev.vet_label.isin(set(C.SHARED_LABELS) | EXTRA)]
    m = sample.merge(rev, on="audit_id", how="inner")
    if m.empty:
        raise SystemExit("No reviewed rows yet. Fill vet_label in audit_for_reviewer.csv first.")

    labelable = m[m.vet_label.isin(C.SHARED_LABELS)].copy()
    labelable["agree"] = labelable.vet_label == labelable.current_label
    print(f"\n{len(m)} images reviewed ({len(labelable)} given a class, "
          f"{int((m.vet_label == 'unclear').sum())} unclear, {int((m.vet_label == 'not_dog').sum())} not a dog)")
    print(f"Label agreement with the vet: {labelable.agree.mean():.1%}  (n={len(labelable)})")
    print("\nAgreement per original class:")
    print(labelable.groupby("current_label").agree.agg(["mean", "size"]).round(3).to_string())
    print("\nAgreement per source:")
    print(labelable.groupby("source").agree.agg(["mean", "size"]).round(3).to_string())
    print("\nConfusion (rows = dataset label, columns = vet label):")
    print(pd.crosstab(labelable.current_label, labelable.vet_label).to_string())

    if not BACKUP.exists():
        shutil.copy2(CURATED, BACKUP)
    base = pd.read_csv(BACKUP)
    base["label_audited"] = False
    decision = m.set_index("path").vet_label
    changed = dropped = 0
    keep_rows = []
    for r in base.itertuples():
        v = decision.get(r.path)
        if v is None:
            keep_rows.append(True)
        elif v in EXTRA:
            dropped += 1
            keep_rows.append(False)
        else:
            if v != r.mapped_label:
                changed += 1
            base.at[r.Index, "mapped_label"] = v
            base.at[r.Index, "label_audited"] = True
            keep_rows.append(True)
    base[keep_rows].to_csv(CURATED, index=False)
    print(f"\ncurated.csv updated: {changed} labels corrected, {dropped} images removed. "
          f"Original saved as {BACKUP.name}.")
    print("Caveat: a single reviewer. Report the sample size and say there is no inter-rater agreement.")




