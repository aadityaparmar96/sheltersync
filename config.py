"""Settings for the curation pipeline."""
import os
import re
from pathlib import Path

DATA = Path(os.environ.get("CURATION_DATA", "data"))
RAW = DATA / "raw"      # one subfolder per source dataset
WORK = DATA / "work"    # intermediate files (manifest, embeddings, review lists)
OUT = DATA / "curated"  # final curated dataset

SOURCE_PRIORITY = [] #left empty for now

LICENSES = {} # left empty for now

HOLDOUT_SOURCE = None #A set can be exclusively kept as a holdout test set if required


SPLIT_DIRS = {"train", "val", "valid", "validation", "test"} #for each of the datasets


