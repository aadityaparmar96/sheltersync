"""
F-pet_v6 is a segmentation project, its classes are per-region masks, not per-image folders, so the
"Folder Structure" export does not exist for it. This script derives ONE label per image from its masks
and copies the image into data/raw/<out-name>/<label>/. It also records how much of each image the
lesion covers (data/work/<out-name>_lesion_stats.csv), which can serve later as a simple severity proxy.

Exports from Roboflow as COCO (segmentation), unzip, and point --src at the folder that contains
train/ valid/ test/, each holding an _annotations.coco.json.

Labelling rules (printed in the report so you can defend them in a write-up):
  - area per class is summed over the image's annotations (the COCO `area` field)
  - classes named in --healthy-names ("fine", "healthy") count as healthy skin, not disease
  - label = the disease class with the largest total area
  - if a second disease class covers >= --ambiguous-ratio of the largest one, the image is ambiguous: skipped
  - no disease annotation but a healthy region -> that healthy label
  - no annotations at all, or annotations without an area -> skipped

Usage:
  python 01b_import_coco_segmentation.py --src path/to/fpet_coco_export --out-name rf_fpet_v6
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unzipped Roboflow COCO export (has train/valid/test folders)")
    ap.add_argument("--out-name", default="rf_fpet_v6", help="folder to create under data/raw/")
    ap.add_argument("--healthy-names", nargs="+", default=["fine", "healthy"])
    ap.add_argument("--ambiguous-ratio", type=float, default=0.5)
    args = ap.parse_args()

    healthy_set = {C.norm_label(n) for n in args.healthy_names}
    src, out = Path(args.src), C.RAW / args.out_name
    ann_files = sorted(src.rglob("_annotations.coco.json"))
    if not ann_files:
        raise SystemExit(f"No _annotations.coco.json under {src}. Export the project from Roboflow in COCO format.")

    skipped, written, stats = Counter(), Counter(), []
    for jf in ann_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        areas = {}
        for a in data.get("annotations", []):
            name = cats.get(a["category_id"])
            if name is None:
                continue
            per = areas.setdefault(a["image_id"], {})
            per[name] = per.get(name, 0.0) + float(a.get("area") or 0.0)

        for im in data.get("images", []):
            f = jf.parent / im["file_name"]
            if not f.exists():
                skipped["image file missing"] += 1
                continue
            per = areas.get(im["id"], {})
            disease = sorted(((v, k) for k, v in per.items() if C.norm_label(k) not in healthy_set), reverse=True)
            healthy = sorted(((v, k) for k, v in per.items() if C.norm_label(k) in healthy_set), reverse=True)
            if disease:
                top_area, label = disease[0]
                if top_area <= 0:
                    skipped["annotation has no area"] += 1
                    continue
                if len(disease) > 1 and disease[1][0] >= args.ambiguous_ratio * top_area:
                    skipped["ambiguous (two disease classes of similar size)"] += 1
                    continue
            elif healthy:
                label = healthy[0][1]
            else:
                skipped["no annotations"] += 1
                continue

            dest_dir = out / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{jf.parent.name}_{f.name}"
            shutil.copy2(f, dest)
            written[label] += 1
            img_area = float(im.get("width") or 0) * float(im.get("height") or 0)
            lesion = sum(v for v, _ in disease)
            stats.append(dict(path=dest.as_posix(), label=label, n_disease_classes=len(disease),
                              lesion_frac=min(lesion / img_area, 1.0) if img_area else float("nan")))

    C.WORK.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stats).to_csv(C.WORK / f"{args.out_name}_lesion_stats.csv", index=False)
    print(f"\nWrote {sum(written.values())} images to {out}")
    for k, v in sorted(written.items()):
        print(f"  {k}: {v}")
    if skipped:
        print("\nSkipped:")
        for k, v in skipped.most_common():
            print(f"  {k}: {v}")
    print("\nNext: add any new class names printed above to LABEL_MAP in config.py, then run 01_build_manifest.py.")
    print("Label names are the ORIGINAL category names; LABEL_MAP maps them to the shared classes.")


if __name__ == "__main__":
    main()
