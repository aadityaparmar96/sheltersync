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


