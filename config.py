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

SHARED_LABELS = ["fungal", "allergic", "mange", "bacterial", "healthy"] #more to add later

LABEL_MAP = {
    "fungal_infections": "fungal",
    "allergic_dermatitis": "allergic",
    "demodicosis_mange": "mange",
    "bacterial_dermatosis": "bacterial",
    "healthy_control": "healthy",
    "ringworm": "fungal",  # dermatophyte (fungal) infection. 
    
    "healthy": "healthy",
    "mange": "mange",
    "hotspot": None,  # acute moist dermatitis: no clean shared class
    "malassezia_dermatitis": "fungal",  # yeast infection
    # Extra names found in the Roboflow sets. 
    "demodicosis": "mange",  # demodectic mange
    "scabies": "mange",  # sarcoptic mange
    "fine": "healthy",  # F-pet_v6 name for healthy
    "hypersensitivity": "allergic",
    "hypersensitivity_allergic_dermatosis": "allergic",
    "flea_allergy": "allergic",  
    "dermatitis": None,  
}

# image filters
MIN_SIDE = 128
MAX_ASPECT = 3.0

# duplicate handling
DINO_MODEL = "facebook/dinov2-small"
CLUSTER_PHASH_HAMMING = 6   # images this close are grouped (and always split together)
CLUSTER_COSINE = 0.93
DUP_PHASH_HAMMING = 2       # images this close are the *same photo*: only one is kept
DUP_COSINE = 0.985

SEED = 0

def norm_label(name: str) -> str:
    """'1_Fungal_Infections' -> 'fungal_infections'."""
    s = re.sub(r"^\d+[_\-\s]*", "", name.lower())
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def sources() -> dict:
    """Ordered {source_name: folder} for every sub-folder of RAW."""
    found = [p.name for p in sorted(RAW.iterdir()) if p.is_dir()] if RAW.exists() else []
    ordered = [s for s in SOURCE_PRIORITY if s in found] + [s for s in found if s not in SOURCE_PRIORITY]
    return {s: RAW / s for s in ordered}


def license_of(source: str) -> str:
    return LICENSES.get(source, "UNKNOWN - CHECK")

