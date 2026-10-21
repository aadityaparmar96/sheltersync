"""Settings for the curation pipeline."""
import os
import re
from pathlib import Path

DATA = Path(os.environ.get("CURATION_DATA", "data"))
RAW = DATA / "raw"      # one subfolder per source dataset
WORK = DATA / "work"    # intermediate files (manifest, embeddings, review lists)
OUT = DATA / "curated"  # final curated dataset


