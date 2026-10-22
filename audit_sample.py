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



