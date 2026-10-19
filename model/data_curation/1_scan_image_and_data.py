"""Step 1: scan every source, validate images, map labels, write the metadata table.

Outputs (in data/work):
  manifest.csv   one row per usable image: path, source, license, labels, size, sha256
  rejected.csv   images dropped and why (unreadable, too small, label dropped on purpose...)
  unmapped.csv   class folders not in LABEL_MAP, with counts. Decide on these, then re-run.
"""
import hashlib

import pandas as pd
from PIL import Image

import config as C

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(x, **_):
        return x

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_from_path(path, root):
    """Nearest parent folder that is not a train/val/test folder."""
    for part in reversed(path.relative_to(root).parent.parts):
        if part.lower() not in C.SPLIT_DIRS:
            return part
    return None


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    bad = {v for v in C.LABEL_MAP.values() if v is not None and v not in C.SHARED_LABELS}
    if bad:
        raise SystemExit(f"LABEL_MAP maps to labels not in SHARED_LABELS: {sorted(bad)}")

    srcs = C.sources()
    if not srcs:
        raise SystemExit(f"No source folders found under {C.RAW}. Put one folder per dataset there.")
    C.WORK.mkdir(parents=True, exist_ok=True)

    rows, rejected, unmapped = [], [], {}
    for name, root in srcs.items():
        files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMG_EXT]
        for p in tqdm(files, desc=name):
            raw_label = label_from_path(p, root)
            if raw_label is None:
                rejected.append((p.as_posix(), name, "no class folder"))
                continue
            key = C.norm_label(raw_label)
            if key not in C.LABEL_MAP:
                unmapped[(name, raw_label)] = unmapped.get((name, raw_label), 0) + 1
                continue
            mapped = C.LABEL_MAP[key]
            if mapped is None:
                rejected.append((p.as_posix(), name, f"label '{raw_label}' dropped by LABEL_MAP"))
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    w, h = im.size
            except Exception as e:  # corrupt / truncated / not really an image
                rejected.append((p.as_posix(), name, f"unreadable ({type(e).__name__})"))
                continue
            if min(w, h) < C.MIN_SIDE:
                rejected.append((p.as_posix(), name, f"too small ({w}x{h})"))
                continue
            if max(w, h) / min(w, h) > C.MAX_ASPECT:
                rejected.append((p.as_posix(), name, f"extreme aspect ratio ({w}x{h})"))
                continue
            rows.append(dict(
                path=p.as_posix(), source=name, license=C.license_of(name),
                original_label=raw_label, mapped_label=mapped,
                width=w, height=h, sha256=sha256_of(p), n_bytes=p.stat().st_size,
            ))

    manifest = pd.DataFrame(rows)
    manifest.to_csv(C.WORK / "manifest.csv", index=False)
    pd.DataFrame(rejected, columns=["path", "source", "reason"]).to_csv(C.WORK / "rejected.csv", index=False)
    pd.DataFrame(
        [(s, l, n) for (s, l), n in sorted(unmapped.items())], columns=["source", "folder", "n_images"]
    ).to_csv(C.WORK / "unmapped.csv", index=False)

    print(f"\nKept {len(manifest)} images, rejected {len(rejected)}.")
    if len(manifest):
        print("\nImages per class x source:")
        print(pd.crosstab(manifest.mapped_label, manifest.source, margins=True))
    if rejected:
        print("\nRejection reasons:")
        print(pd.DataFrame(rejected, columns=["path", "source", "reason"]).reason
              .str.replace(r"\(.*\)|'.*'", "", regex=True).value_counts().to_string())
    if unmapped:
        print("\nUNMAPPED class folders (add them to LABEL_MAP in config.py, mapped or None, then re-run):")
        for (s, l), n in sorted(unmapped.items()):
            print(f"  {s}/{l}: {n} images")


if __name__ == "__main__":
    main()
