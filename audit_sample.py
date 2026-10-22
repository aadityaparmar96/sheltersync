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


